#!/usr/bin/env python3
"""
Step 5: Per-Token KSG MI Analysis with SPI-Based Token Ordering

Uses the KSG estimator (same as timer_mi_ksg_pca.py) to compute per-token
mutual information for Timer representations on synthetic concepts.

Pipeline:
  1. Load per-token hidden states and patch embeddings from Step 2
  2. For each concept, layer, and token position:
       - I(H, Y): KSG(pca(H_token), pca(Y_param))  — prediction power
       - I(X, H): KSG(pca(X_patch), pca(H_token))  — compression
  3. Compute SPI = I(H,Y) / (I(X,H) + bias) per token
  4. Order tokens by SPI, split into high/low groups
  5. Train linear probes on high-SPI vs low-SPI groups
  6. Compare probe quality (MSE, R^2) between groups

The KSG estimator and SPI bias follow timer_mi_ksg_pca.py exactly.

Usage:
    # Run both between-layer (top_k) and within-layer (top 25% / bottom 25%) analyses:
    python probe/ts_concept_ksg_mi_analysis.py \
        --rep_dir ./results/synthetic/representations/ \
        --output_dir ./results/synthetic/ksg_mi_token/ \
        --top_k 4 --probe_epochs 200 --probe_lr 1e-3 \
        --pca_dim 32 --k_neighbors 5 --spi_bias_factor 0.15
"""

from __future__ import annotations

import argparse
import os
import sys
import gc
import json
import multiprocessing as mp
import concurrent.futures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.spatial as ss
import scipy.special as sp
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# KSG MI Estimator — GPU (torch) + CPU (numpy+scipy) fallback
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_ksg_neighbor_counts(
    x: torch.Tensor, y: torch.Tensor, epsilon: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated L-inf neighbor count for KSG.
    x, y: [N, D] on GPU
    epsilon: [N] radii (L-inf balls)
    Returns: nx, ny: [N] neighbor counts (excluding self)
    """
    # L-inf distance: dist[i,j] = max_d |x[i,d] - x[j,d]|
    diff_x = x.unsqueeze(1) - x.unsqueeze(0)            # [N, N, D]
    dist_x = diff_x.abs().max(dim=2).values              # [N, N]
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)
    dist_y = diff_y.abs().max(dim=2).values

    # Count neighbors within epsilon (row-wise broadcast)
    nx = (dist_x <= epsilon.unsqueeze(1)).sum(dim=1) - 1  # [N], exclude self
    ny = (dist_y <= epsilon.unsqueeze(1)).sum(dim=1) - 1
    return nx.clamp_(min=0), ny.clamp_(min=0)


def compute_mi_ksg(
    x: np.ndarray, y: np.ndarray, k: int = 5,
    device: str = "cpu"
) -> float:
    """
    KSG (Kraskov, Stoegbauer, Grassberger) Estimator for I(X;Y).
    x: [N, D_x], y: [N, D_y]
    Returns MI in bits.

    device: "cuda" to use GPU acceleration, "cpu" for NumPy fallback.
    GPU path uses torch.cdist for L-inf k-NN and fully vectorized neighbor counts.
    """
    N = x.shape[0]
    if N <= k + 1:
        return 0.0

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    xy = np.concatenate((x, y), axis=1)

    # ── GPU path ───────────────────────────────────────────────────────────
    if device == "cuda":
        try:
            x_t = torch.from_numpy(x.astype(np.float32)).cuda()
            y_t = torch.from_numpy(y.astype(np.float32)).cuda()
            xy_t = torch.from_numpy(xy.astype(np.float32)).cuda()

            # k-NN on xy in L-inf metric → per-point epsilon
            dist_xy = torch.cdist(xy_t, xy_t, p=float("inf"))      # [N, N]
            eps, _ = dist_xy.topk(k + 1, largest=False, sorted=True)  # [N, k+1]
            epsilon = eps[:, k]                                       # [N]
            eps_strict = (epsilon - 1e-7).clamp_(min=0)

            # Vectorized neighbor counts on GPU
            nx, ny = _gpu_ksg_neighbor_counts(x_t, y_t, eps_strict, k)

            # digamma on GPU (torch.special.digamma)
            # digamma(k) and digamma(N) are scalars — compute on CPU
            psi_k = float(torch.special.digamma(k))
            psi_N = float(torch.special.digamma(N))
            mean_psi = float(
                torch.mean(torch.special.digamma(nx.float() + 1) +
                          torch.special.digamma(ny.float() + 1))
            )
            mi = psi_k - mean_psi + psi_N
            return max(0.0, mi / np.log(2))

        except Exception as e:
            # Fall back to CPU if GPU fails (e.g. OOM, no CUDA)
            pass

    # ── CPU fallback (vectorized NumPy + cKDTree) ───────────────────────────
    tree_xy = ss.cKDTree(xy)
    dist_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    epsilon = dist_xy[:, k]
    eps_strict = np.maximum(epsilon - 1e-10, 0)

    # Vectorized neighbor count (no Python loop)
    diff_x = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_x = np.max(np.abs(diff_x), axis=2)
    diff_y = y[:, np.newaxis, :] - y[np.newaxis, :, :]
    dist_y = np.max(np.abs(diff_y), axis=2)
    nx = np.sum(dist_x <= eps_strict[:, np.newaxis], axis=1) - 1
    ny = np.sum(dist_y <= eps_strict[:, np.newaxis], axis=1) - 1
    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    psi_k = sp.digamma(k)
    psi_N = sp.digamma(N)
    mean_psi = np.mean(sp.digamma(nx + 1) + sp.digamma(ny + 1))
    mi = psi_k - mean_psi + psi_N
    return max(0.0, mi / np.log(2))


# ─────────────────────────────────────────────────────────────────────────────
# Linear Probe
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from probe.ts_concept_synthetic_dataset import TSConceptGenerator, CONCEPT_PARAM_SPEC


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(d_model, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    z: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Train linear probe, return val MSE and R^2.
    Uses 80/20 split within the provided data.
    """
    N = z.shape[0]
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    z_np = z.numpy()
    y_np = y.numpy()

    z_tr, z_vl, y_tr, y_vl = train_test_split(
        z_np, y_np, train_size=0.8, random_state=seed, shuffle=True
    )
    z_tr = torch.from_numpy(z_tr).float()
    z_vl = torch.from_numpy(z_vl).float()
    y_tr = torch.from_numpy(y_tr).float()
    y_vl = torch.from_numpy(y_vl).float()

    in_dim = z_tr.shape[1]
    out_dim = y_tr.shape[1]
    probe = LinearProbe(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    crit = nn.MSELoss()

    n = z_tr.shape[0]
    bs = min(256, n)
    for _ in range(n_epochs):
        probe.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            bi = idx[i:i+bs]
            opt.zero_grad()
            loss = crit(probe(z_tr[bi].to(device)), y_tr[bi].to(device))
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(z_vl.to(device))
        mse = crit(pred, y_vl.to(device)).item()
        ss_res = ((pred.cpu() - y_vl) ** 2).sum().item()
        ss_tot = ((y_vl - y_vl.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return mse, r2


# ─────────────────────────────────────────────────────────────────────────────
# Dual-Stream KSG MI Analysis: X-stream (patch) vs Y-stream (param embedding)
#
# For each decoder layer l:
#   H_X^0 = patch embedding (X)                    — [N, N_patches, D]
#   H_Y^0 = ParamEmbed(Y) = Linear(Y)            — [N, 1, D]
#   H_X^l = DecoderLayer_l(H_X^{l-1})            — [N, N_patches, D]
#   H_Y^l = DecoderLayer_l(H_Y^{l-1})            — [N, 1, D]
#
# MI metrics per layer (after pooling over patches):
#   I(XH_l)  = I(pool(H_X^l), pool(H_Y^l))       — X/Y stream alignment
#   I(HY_l)  = I(pool(H_Y^l), Y)                  — Y-stream prediction power
#   SPI_l    = I(HY_l) / (I(XH_l) + bias)         — selectivity
#
# Linear probe trains on pool(H_Y^l) → Y.
# ─────────────────────────────────────────────────────────────────────────────

class ParamEmbedding(nn.Module):
    """Project raw param vector [N, 4] → [N, D] via learned linear layer."""
    def __init__(self, param_dim: int = 4, d_model: int = 1024):
        super().__init__()
        self.proj = nn.Linear(param_dim, d_model, bias=True)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: [N, param_dim] raw normalized params
        Returns:
            [N, 1, d_model] — single token representing the param query
        """
        h = self.proj(Y)                       # [N, D]
        return h.unsqueeze(1)                  # [N, 1, D]


def _compute_token_ksg_metrics(
    E_X: np.ndarray,
    H_X: np.ndarray,
    H_Y: np.ndarray,
    pca_dim: int,
    k_neighbors: int,
    ksg_device: str = "cpu",
) -> tuple[float, float]:
    """
    Per-token KSG MI: compute I(E_X, H_X) and I(H_X, H_Y) for one token position.

    Args:
        E_X: [N, D] — patch embedding at this token position
        H_X: [N, D] — decoder hidden state at layer l, token pi
        H_Y: [N, D] — decoder hidden state for Y-stream at layer l (token 0)
        pca_dim, k_neighbors

    Returns:
        mi_ex: I(E_X, H_X) in bits
        mi_hy: I(H_X, H_Y) in bits
    """
    N = E_X.shape[0]

    # PCA on E_X and H_X
    n_comp_ex = min(pca_dim, N, E_X.shape[1])
    pca_ex = PCA(n_components=n_comp_ex, random_state=42)
    E_X_red = pca_ex.fit_transform(E_X)

    n_comp_hx = min(pca_dim, N, H_X.shape[1])
    pca_hx = PCA(n_components=n_comp_hx, random_state=42)
    H_X_red = pca_hx.fit_transform(H_X)

    # PCA on H_Y
    n_comp_y = min(pca_dim, N, H_Y.shape[1])
    pca_y = PCA(n_components=n_comp_y, random_state=42)
    H_Y_red = pca_y.fit_transform(H_Y)

    # I(E_X, H_X) — input compression
    mi_ex = compute_mi_ksg(E_X_red, H_X_red, k=k_neighbors, device=ksg_device)

    # I(H_X, H_Y) — X/Y stream alignment
    mi_hy = compute_mi_ksg(H_X_red, H_Y_red, k=k_neighbors, device=ksg_device)

    return float(mi_ex), float(mi_hy)


def run_dual_stream_ksg_analysis(
    layer_tokens: list[torch.Tensor],
    patch_tokens: torch.Tensor,
    params: torch.Tensor,
    concept_idx: torch.Tensor,
    concepts: list[str],
    n_samples_per_concept: int,
    d_model: int,
    n_layers: int,
    output_dir: str,
    decoder_layers: nn.ModuleList,
    pca_dim: int = 32,
    k_neighbors: int = 5,
    n_samples_ksg: int = 500,
    spi_bias_factor: float = 0.15,
    probe_epochs: int = 200,
    probe_lr: float = 1e-3,
    device: torch.device = None,
    seed: int = 42,
    percentile: float = 20.0,
):
    """
    Dual-stream per-token KSG MI analysis.

    For each layer l, each token position pi, and each (concept, param_dim):
      E_X[pi]  = patch embedding at token pi                      — [N, D]
      H_X^l[pi] = decoder hidden state at layer l, token pi       — [N, D]
      H_Y^l     = Y-stream at layer l, token 0                    — [N, D]

    Per-token metrics:
      I(E_X, H_X^l[pi]) — how much the layer preserves input info
      I(H_X^l[pi], H_Y^l) — X/Y stream alignment at this token
      SPI[pi] = I(H_X^l[pi], H_Y^l) / (I(E_X, H_X^l[pi]) + bias)

    Per-layer, per (concept, dim):
      Sort tokens by SPI, split top/bottom percentile groups.
      Train linear probe on each group → compare MSE/R².
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    N, N_patches, D = layer_tokens[0].shape
    print(f"  [DualStream] N={N}, N_patches={N_patches}, D={D}")
    print(f"  [DualStream] Decoder layers: {n_layers}, PCA dim: {pca_dim}")
    print(f"  [DualStream] Token percentile: top {percentile}%, bottom {percentile}%")

    layer_tokens_np = [lt.numpy() for lt in layer_tokens]
    patch_tokens_np = patch_tokens.numpy()
    params_np = params.numpy()
    concept_idx_np = concept_idx.numpy()

    # E_X per token position: [N, N_patches, D] → [N_patches, N, D]
    E_X_per_token = patch_tokens_np.transpose(1, 0, 2)   # [N_patches, N, D]

    # Single shared ParamEmbed: Linear(4, D), frozen random init
    param_embed = ParamEmbedding(param_dim=4, d_model=d_model).to(device)
    param_embed.eval()
    for p in param_embed.parameters():
        p.requires_grad = False

    results = {}

    for layer_idx in range(n_layers):
        print(f"  [DualStream] Layer {layer_idx}/{n_layers-1} ...", flush=True)

        # ── 1. Run Y stream through all layers up to layer_idx ───────────────
        #    H_Y^l = decoder_layers[l](...)[0][:, 0, :] at layer l
        # We need H_Y^l at THIS layer only (token 0).
        # Y stream: starts from ParamEmbed(Y) [N, 1, D], passes through all layers.
        # We pre-compute H_Y^l for all N samples (batched) per layer.
        Y_all = torch.from_numpy(params_np).float().to(device)       # [N, 4]
        H_Y_init_all = param_embed(Y_all)                            # [N, 1, D]
        h_y = H_Y_init_all
        for li in range(layer_idx + 1):
            h_y, _, _ = decoder_layers[li](h_y, attn_mask=None)
        H_Y_l_np = h_y[:, 0, :].detach().cpu().numpy()              # [N, D]

        # H_X^l: pre-computed per-layer tokens [N, N_patches, D]
        H_X_l_np = layer_tokens_np[layer_idx]                        # [N, N_patches, D]

        for ci, concept in enumerate(concepts):
            mask = concept_idx_np == ci
            n_c = mask.sum()
            if n_c < 20:
                continue

            active_dims, _ = CONCEPT_PARAM_SPEC[concept]
            Y_concept_np = params_np[mask]                            # [n_c, 4]

            for dim_idx in active_dims:
                Y_target_np = Y_concept_np[:, dim_idx]                # [n_c]

                # H_Y for this concept at this layer: H_Y_l_np indexed by mask
                H_Y_c = H_Y_l_np[mask]                                # [n_c, D]

                # Compute per-token MI
                top_n = max(1, int(np.ceil(N_patches * percentile / 100.0)))

                token_mi_ex = np.zeros(N_patches)   # I(E_X, H_X) per token
                token_mi_hy = np.zeros(N_patches)  # I(H_X, H_Y) per token
                token_spi   = np.zeros(N_patches)

                rng = np.random.default_rng(seed + layer_idx * 1000 + ci * 100 + dim_idx)

                for pi in range(N_patches):
                    # All three arrays are indexed by concept-local [0, n_c-1]
                    # E_X_per_token[pi] contains all N samples → apply mask
                    e_x = E_X_per_token[pi][mask][:]     # [n_c, D]
                    h_x = H_X_l_np[mask][:, pi, :]       # [n_c, D]
                    h_y_tok = H_Y_c                       # [n_c, D]

                    # Subsample for KSG
                    if n_samples_ksg is not None and n_c > n_samples_ksg:
                        idx = rng.choice(n_c, n_samples_ksg, replace=False)
                        idx = np.sort(idx)
                        e_x = e_x[idx].copy()
                        h_x = h_x[idx].copy()
                        h_y_tok = h_y_tok[idx].copy()

                    ksg_dev = device.type if hasattr(device, 'type') else str(device)
                    mi_ex, mi_hy = _compute_token_ksg_metrics(
                        e_x, h_x, h_y_tok,
                        pca_dim=pca_dim,
                        k_neighbors=k_neighbors,
                        ksg_device=ksg_dev,
                    )
                    token_mi_ex[pi] = mi_ex
                    token_mi_hy[pi] = mi_hy

                # Sort by I(H_X, H_Y) descending, split top/bottom percentile
                sorted_pos = np.argsort(token_mi_hy)[::-1]
                top_pos = sorted_pos[:top_n].copy()
                bot_pos = sorted_pos[-top_n:].copy()

                # Collect H_X at top/bottom SPI positions for probing
                # H_X_top[pi] = H_X_l_np[mask][:, pi, :] averaged over pi in top_pos
                H_X_top = H_X_l_np[mask][:, top_pos, :].mean(axis=1)  # [n_c, D]
                H_X_bot = H_X_l_np[mask][:, bot_pos, :].mean(axis=1)  # [n_c, D]

                Y_cpu = torch.from_numpy(Y_target_np).float()
                H_X_top_t = torch.from_numpy(H_X_top).float()
                H_X_bot_t = torch.from_numpy(H_X_bot).float()

                mse_top, r2_top = train_probe(H_X_top_t, Y_cpu, probe_epochs, probe_lr, device, seed)
                mse_bot, r2_bot = train_probe(H_X_bot_t, Y_cpu, probe_epochs, probe_lr, device, seed)
                delta_mse = mse_bot - mse_top

                results[(layer_idx, concept, dim_idx)] = {
                    "mi_ex_curve": token_mi_ex.tolist(),     # per-token I(E_X, H_X)
                    "mi_hy_curve": token_mi_hy.tolist(),     # per-token I(H_X, H_Y)
                    "mse_top":  mse_top,
                    "mse_bot":  mse_bot,
                    "r2_top":   r2_top,
                    "r2_bot":   r2_bot,
                    "delta_mse": delta_mse,
                    "top_pos":  top_pos.tolist(),
                    "bot_pos":  bot_pos.tolist(),
                    "percentile": percentile,
                }

        gc.collect()
        torch.cuda.empty_cache()

    return results


def compute_token_ksg_mi(
    H_token: np.ndarray,   # [N, D]
    Y_param: np.ndarray,   # [N, P]
    X_patch: np.ndarray,   # [N, D_patch]
    pca_dim: int,
    k_neighbors: int,
    n_samples_ksg: int = None,
    rng: np.random.Generator = None,
    ksg_device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute per-token MI metrics using KSG estimator (with PCA).

    For one token position, given:
      H_token: [N, D] hidden state for this token across samples
      Y_param: [N, P] concept parameter values (already sliced per concept/dim)
      X_patch: [N, D] patch embedding for this token

    Returns:
      mi_hy: I(H, Y) in bits (scalar)
      mi_xh: I(X, H) in bits (scalar)
      spi_bias: the bias value used (= 0.15 * mean(ixh_overall))
    """
    N = H_token.shape[0]

    if rng is None:
        rng = np.random.default_rng(42)

    if n_samples_ksg is not None and n_samples_ksg < N:
        idx = rng.choice(N, n_samples_ksg, replace=False)
        idx = np.sort(idx)
        H_token = H_token[idx]
        Y_param = Y_param[idx]
        X_patch = X_patch[idx]
        N = n_samples_ksg

    # PCA on H_token
    n_comp_h = min(pca_dim, N, H_token.shape[1])
    pca_h = PCA(n_components=n_comp_h, random_state=42)
    H_reduced = pca_h.fit_transform(H_token)

    # PCA on Y_param
    n_comp_y = min(pca_dim, N, Y_param.shape[1])
    pca_y = PCA(n_components=n_comp_y, random_state=42)
    Y_reduced = pca_y.fit_transform(Y_param)

    # PCA on X_patch
    n_comp_x = min(pca_dim, N, X_patch.shape[1])
    pca_x = PCA(n_components=n_comp_x, random_state=42)
    X_reduced = pca_x.fit_transform(X_patch)

    # I(H, Y)
    mi_hy = compute_mi_ksg(H_reduced, Y_reduced, k=k_neighbors, device=ksg_device)

    # I(X, H)
    mi_xh = compute_mi_ksg(X_reduced, H_reduced, k=k_neighbors, device=ksg_device)

    return mi_hy, mi_xh


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing: worker for KSG per-task (layer, concept, dim_idx)
# ─────────────────────────────────────────────────────────────────────────────

def _ksg_task_worker(args_tuple):
    """
    Top-level function for multiprocessing / threading.
    Each worker computes KSG MI for all 5 token positions of one task.

    Optimizations applied:
      - Subsample ONCE before the patch loop (not per-patch)
      - Y_param PCA ONCE before the patch loop (not per-patch)
      - H_token PCA and X_patch PCA per patch (data differs per token)
      - GPU-accelerated KSG when ksg_device="cuda" (via torch.cdist)

    Returns: (key, mi_hy_vals, mi_xh_vals) or None on skip.
    """
    (layer_idx, ci, dim_idx, H_concept, X_concept,
     Y_concept, max_samples, pca_dim, k_neighbors,
     n_samples_ksg, task_seed, ksg_device) = args_tuple

    N_patches = H_concept.shape[1]
    mi_hy_vals = np.zeros(N_patches)
    mi_xh_vals = np.zeros(N_patches)
    rng = np.random.default_rng(task_seed)

    n_c = H_concept.shape[0]
    if n_c < 20:
        return None

    # ── Subsample ONCE before the patch loop ───────────────────────────────────
    if max_samples is not None and n_c > max_samples:
        idx = rng.choice(n_c, max_samples, replace=False)
        idx = np.sort(idx)
        H_concept = H_concept[idx]
        X_concept = X_concept[idx]
        Y_concept = Y_concept[idx]
        n_c = max_samples

    if n_samples_ksg is not None and n_c > n_samples_ksg:
        idx = rng.choice(n_c, n_samples_ksg, replace=False)
        idx = np.sort(idx)
        H_concept = H_concept[idx]
        X_concept = X_concept[idx]
        Y_concept = Y_concept[idx]
        n_c = n_samples_ksg

    # ── Y_param PCA ONCE (same for all patches) ───────────────────────────────
    n_comp_y = min(pca_dim, n_c, Y_concept.shape[1])
    pca_y = PCA(n_components=n_comp_y, random_state=42)
    Y_reduced = pca_y.fit_transform(Y_concept)

    # ── Loop over patches (only H and X differ per token) ───────────────────
    for pi in range(N_patches):
        H_tok = H_concept[:, pi, :]
        X_tok = X_concept[:, pi, :]

        # PCA on H_token
        n_comp_h = min(pca_dim, n_c, H_tok.shape[1])
        pca_h = PCA(n_components=n_comp_h, random_state=42)
        H_reduced = pca_h.fit_transform(H_tok)

        # PCA on X_patch
        n_comp_x = min(pca_dim, n_c, X_tok.shape[1])
        pca_x = PCA(n_components=n_comp_x, random_state=42)
        X_reduced = pca_x.fit_transform(X_tok)

        # I(H, Y) and I(X, H) — GPU when ksg_device="cuda"
        mi_hy = compute_mi_ksg(H_reduced, Y_reduced, k=k_neighbors, device=ksg_device)
        mi_xh = compute_mi_ksg(X_reduced, H_reduced, k=k_neighbors, device=ksg_device)
        mi_hy_vals[pi] = mi_hy
        mi_xh_vals[pi] = mi_xh

    return ((layer_idx, ci, dim_idx), mi_hy_vals, mi_xh_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Main: compute SPI-based token ordering and probe analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_ksg_spi_token_analysis(
    layer_tokens: list[torch.Tensor],   # list of [N, N_patches, D]
    patch_tokens: torch.Tensor,         # [N, N_patches, D_patch]
    params: torch.Tensor,                # [N, 4]
    concept_idx: torch.Tensor,           # [N]
    labels: list[dict],
    concepts: list[str],
    n_samples_per_concept: int,
    d_model: int,
    n_layers: int,
    output_dir: str,
    top_k: int = 4,
    max_samples: int = None,
    probe_epochs: int = 200,
    probe_lr: float = 1e-3,
    pca_dim: int = 32,
    k_neighbors: int = 5,
    n_samples_ksg: int = 500,
    spi_bias_factor: float = 0.15,
    device: torch.device = None,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Compute per-token KSG-MI, rank by SPI, split into high/low groups,
    and compare linear probe quality.
    Parallelised across (layer, concept, dim_idx) tasks via multiprocessing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    N, N_patches, D = layer_tokens[0].shape
    _, _, D_patch = patch_tokens.shape
    print(f"  N={N}, N_patches={N_patches}, D={D}, D_patch={D_patch}")

    # Convert all to numpy once
    layer_tokens_np = [lt.numpy() for lt in layer_tokens]
    patch_tokens_np = patch_tokens.numpy()
    params_np = params.numpy()
    concept_idx_np = concept_idx.numpy()

    # ── Build task list: (layer_idx, ci, dim_idx, H_concept, X_concept, Y_concept, ...) ──
    task_list = []
    for layer_idx in range(n_layers):
        for ci, concept in enumerate(concepts):
            mask = concept_idx_np == ci
            n_c = mask.sum()
            if n_c < 20:
                continue
            active_dims, param_names = CONCEPT_PARAM_SPEC[concept]
            H_concept = layer_tokens_np[layer_idx][mask]        # [n_c, N_patches, D]
            X_concept = patch_tokens_np[mask]                   # [n_c, N_patches, D_patch]
            Y_concept = params_np[mask]                         # [n_c, 4]
            for dim_idx in active_dims:
                task_list.append((
                    layer_idx, ci, dim_idx,
                    H_concept,
                    X_concept,
                    Y_concept[:, dim_idx:dim_idx+1],
                    max_samples, pca_dim, k_neighbors, n_samples_ksg,
                    seed + layer_idx * 1000 + ci * 100 + dim_idx,
                    device.type if hasattr(device, 'type') else str(device),
                ))

    n_tasks = len(task_list)
    print(f"  [{n_layers} layers × {len(concepts)} concepts] = {n_tasks} tasks")
    if num_workers == 0:
        num_workers = min(mp.cpu_count(), n_tasks)
    print(f"  Parallelising KSG across {num_workers} CPU workers ...")

    # ── Run KSG in parallel ────────────────────────────────────────────────────
    task_counter = [0]

    def _cb(_):
        task_counter[0] += 1
        done = task_counter[0]
        if done % max(1, n_tasks // 10) == 0 or done == n_tasks:
            print(f"    KSG [{done}/{n_tasks} ({100*done/n_tasks:.0f}%)]  ", flush=True)

    results_ksg = {}

    # ThreadPoolExecutor: threads share GPU context, unlike multiprocessing.
    # GIL is released during GPU ops (torch.cuda kernels), enabling parallelism.
    # For GPU: use min(num_workers, 4) threads to avoid oversubscription.
    # For CPU: num_workers threads for full CPU parallelism.
    if device.type == "cuda":
        ksg_workers = min(num_workers, 4)
    else:
        ksg_workers = num_workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=ksg_workers) as pool:
        futures = {pool.submit(_ksg_task_worker, task): task for task in task_list}
        for done_f in concurrent.futures.as_completed(futures):
            res = done_f.result()
            if res is not None:
                key, mi_hy_vals, mi_xh_vals = res
                results_ksg[key] = (mi_hy_vals, mi_xh_vals)
            _cb(None)

    gc.collect()
    torch.cuda.empty_cache()

    # ── Assemble SPI curves, train probes on GPU ─────────────────────────────────
    print("  Training linear probes on GPU ...", flush=True)
    results = {}
    mi_matrix = {}

    for (layer_idx, ci, dim_idx), (mi_hy_vals, mi_xh_vals) in results_ksg.items():
        layer = layer_idx
        concept = concepts[ci]
        H_layer = layer_tokens_np[layer]
        mask = concept_idx_np == ci

        # SPI
        spi_bias = spi_bias_factor * mi_xh_vals.mean()
        spi_vals = mi_hy_vals / (mi_xh_vals + spi_bias)

        # Token selection
        sorted_pos = np.argsort(spi_vals)[::-1]
        high_pos = sorted_pos[:top_k].copy()
        low_pos = sorted_pos[-top_k:].copy()

        H_high = H_layer[mask][:, high_pos, :].mean(axis=1)
        H_low  = H_layer[mask][:, low_pos,  :].mean(axis=1)
        Y_target = params_np[mask, dim_idx]

        # Probes on GPU
        mse_h, r2_h = train_probe(
            torch.from_numpy(H_high).float(),
            torch.from_numpy(Y_target).float(),
            probe_epochs, probe_lr, device, seed,
        )
        mse_l, r2_l = train_probe(
            torch.from_numpy(H_low).float(),
            torch.from_numpy(Y_target).float(),
            probe_epochs, probe_lr, device, seed,
        )

        delta_mse = mse_l - mse_h

        mi_matrix[(layer, concept, dim_idx)] = {
            "mi_hy": mi_hy_vals.copy(),
            "mi_xh": mi_xh_vals.copy(),
        }

        results[(layer, concept, dim_idx)] = {
            "mi_hy_curve": mi_hy_vals.tolist(),
            "mi_xh_curve": mi_xh_vals.tolist(),
            "spi_curve": spi_vals.tolist(),
            "spi_bias": float(spi_bias),
            "mse_high": mse_h,
            "mse_low": mse_l,
            "r2_high": r2_h,
            "r2_low": r2_l,
            "delta_mse": delta_mse,
            "high_pos": high_pos.tolist(),
            "low_pos": low_pos.tolist(),
            "top_k": top_k,
        }

        if len(results) % 20 == 0:
            print(f"    Probes [{len(results)}/{len(results_ksg)}]", flush=True)

    print(f"  Between-layer done: {len(results)} tasks.", flush=True)
    return results, mi_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Within-layer percentile analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_ksg_within_layer_analysis(
    layer_tokens: list[torch.Tensor],
    patch_tokens: torch.Tensor,
    params: torch.Tensor,
    concept_idx: torch.Tensor,
    labels: list[dict],
    concepts: list[str],
    n_samples_per_concept: int,
    d_model: int,
    n_layers: int,
    output_dir: str,
    percentile: float = 10.0,
    max_samples: int = None,
    probe_epochs: int = 200,
    probe_lr: float = 1e-3,
    pca_dim: int = 32,
    k_neighbors: int = 5,
    n_samples_ksg: int = 500,
    spi_bias_factor: float = 0.15,
    device: torch.device = None,
    seed: int = 42,
):
    """
    Within-layer analysis: for each layer, compute per-token SPI, split into
    top-N% and bottom-N% groups, and compare linear probe quality.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    N, N_patches, D = layer_tokens[0].shape
    _, _, D_patch = patch_tokens.shape

    layer_tokens_np = [lt.numpy() for lt in layer_tokens]
    patch_tokens_np = patch_tokens.numpy()
    params_np = params.numpy()
    concept_idx_np = concept_idx.numpy()

    top_n = max(1, int(np.ceil(N_patches * percentile / 100.0)))

    results = {}

    for layer in range(n_layers):
        print(f"  Layer {layer}/{n_layers-1} ...", flush=True)
        H_layer = layer_tokens_np[layer]

        for ci, concept in enumerate(concepts):
            mask = concept_idx_np == ci
            n_c = mask.sum()
            if n_c < 20:
                continue

            active_dims, param_names = CONCEPT_PARAM_SPEC[concept]
            param_dim_names = [s.strip() for s in param_names.split(",")]

            for dim_idx in active_dims:
                Y_param = params_np[mask, dim_idx].reshape(-1, 1)

                # Compute SPI per token
                spi_vals = np.zeros(N_patches)
                ksg_dev = device.type if hasattr(device, 'type') else str(device)
                for pi in range(N_patches):
                    H_tok = H_layer[mask, pi, :]
                    X_pat = patch_tokens_np[mask, pi, :]

                    if max_samples is not None and n_c > max_samples:
                        idx = rng.choice(n_c, max_samples, replace=False)
                        idx = np.sort(idx)
                        spi_vals[pi], _ = compute_token_ksg_mi(
                            H_tok[idx], Y_param[idx], X_pat[idx],
                            pca_dim=pca_dim, k_neighbors=k_neighbors,
                            n_samples_ksg=n_samples_ksg, rng=rng,
                            ksg_device=ksg_dev,
                        )
                    else:
                        spi_vals[pi], mi_xh_tok = compute_token_ksg_mi(
                            H_tok, Y_param, X_pat,
                            pca_dim=pca_dim, k_neighbors=k_neighbors,
                            n_samples_ksg=n_samples_ksg, rng=rng,
                            ksg_device=ksg_dev,
                        )

                # Recompute with bias for ranking
                # We need I(X,H) to compute SPI, so redo without storing
                spi_bias = 0.0  # bias only affects magnitude, not ordering
                sorted_pos = np.argsort(spi_vals)[::-1]
                top_pos = sorted_pos[:top_n].copy()
                bot_pos = sorted_pos[-top_n:].copy()

                H_top = H_layer[mask][:, top_pos, :].mean(axis=1)
                H_bot = H_layer[mask][:, bot_pos, :].mean(axis=1)
                Y_target = params_np[mask, dim_idx]

                mse_t, r2_t = train_probe(
                    torch.from_numpy(H_top).float(),
                    torch.from_numpy(Y_target).float(),
                    probe_epochs, probe_lr, device, seed,
                )
                mse_b, r2_b = train_probe(
                    torch.from_numpy(H_bot).float(),
                    torch.from_numpy(Y_target).float(),
                    probe_epochs, probe_lr, device, seed,
                )

                delta_mse = mse_b - mse_t
                better = "TOP" if delta_mse > 0 else "BOT"

                param_name = param_dim_names[dim_idx] if dim_idx < len(param_dim_names) else f"dim{dim_idx}"

                results[(layer, concept, dim_idx)] = {
                    "spi_top": float(spi_vals[top_pos].mean()),
                    "spi_bot": float(spi_vals[bot_pos].mean()),
                    "mse_top": mse_t,
                    "mse_bot": mse_b,
                    "r2_top": r2_t,
                    "r2_bot": r2_b,
                    "delta_mse": delta_mse,
                    "top_n": top_n,
                    "percentile": percentile,
                }

        gc.collect()
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def _agg(results, key, layer, concept, active_dims):
    vals = [results.get((layer, concept, d), {}).get(key, np.nan) for d in range(4)]
    return float(np.nanmean([vals[d] for d in active_dims if not np.isnan(vals[d])]))


def plot_ksg_spi_summary(results, concepts, n_layers, output_dir, top_k):
    """
    2x3 summary figure for between-layer (top_k) analysis.

    Row 0: MSE per layer — [High bar | Low bar | Delta bar]
    Row 1: R^2  per layer — [High bar | Low bar | Delta bar]
    Each bar group = one layer, all concepts aggregated.
    Delta = Low - High  (positive = High-SPI wins, shown in blue).
    """
    layers = list(range(n_layers))
    x = np.arange(n_layers)
    w = 0.28

    # Aggregate all concepts per layer
    mse_h = [np.nanmean([v["mse_high"] for k, v in results.items() if k[0] == l])
             for l in layers]
    mse_l = [np.nanmean([v["mse_low"]  for k, v in results.items() if k[0] == l])
             for l in layers]
    delta_mse = [np.nanmean([v["delta_mse"] for k, v in results.items() if k[0] == l])
                 for l in layers]

    r2_h  = [np.nanmean([v["r2_high"]  for k, v in results.items() if k[0] == l])
             for l in layers]
    r2_l  = [np.nanmean([v["r2_low"]   for k, v in results.items() if k[0] == l])
             for l in layers]
    delta_r2 = [np.nanmean([v["r2_high"] - v["r2_low"] for k, v in results.items() if k[0] == l])
                for l in layers]

    fig, axes = plt.subplots(2, 3, figsize=(3 * n_layers + 2, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    # ── Row 0: MSE ──────────────────────────────────────────────────────────
    # [0,0] High-SPI MSE
    ax = axes[0, 0]
    bars = ax.bar(x, mse_h, w, color="steelblue", alpha=0.85)
    for xi, v in enumerate(mse_h):
        ax.text(xi, v + v * 0.01, f"{v:.4f}", ha="center", va="bottom",
                fontsize=7.5, color="steelblue")
    ax.set_title("High-SPI MSE", fontsize=10, fontweight="bold")
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylim(bottom=0); ax.grid(True, alpha=0.25, axis="y")

    # [0,1] Low-SPI MSE
    ax = axes[0, 1]
    bars = ax.bar(x, mse_l, w, color="coral", alpha=0.85)
    for xi, v in enumerate(mse_l):
        ax.text(xi, v + v * 0.01, f"{v:.4f}", ha="center", va="bottom",
                fontsize=7.5, color="coral")
    ax.set_title("Low-SPI MSE", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylim(bottom=0); ax.grid(True, alpha=0.25, axis="y")

    # [0,2] Delta MSE
    ax = axes[0, 2]
    colors_dm = ["steelblue" if v > 0 else "coral" for v in delta_mse]
    bars = ax.bar(x, delta_mse, w, color=colors_dm, alpha=0.85)
    ax.axhline(0, color="black", lw=1.2)
    for xi, v in enumerate(delta_mse):
        ax.text(xi, v + 0.0001 * np.sign(v) if v != 0 else 0.0001,
                f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5, color=colors_dm[xi])
    ax.set_title(r"$\Delta$MSE (Low $-$ High)", fontsize=10, fontweight="bold")
    ax.set_ylabel("ΔMSE", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    # ── Row 1: R^2 ─────────────────────────────────────────────────────────
    # [1,0] High-SPI R²
    ax = axes[1, 0]
    bars = ax.bar(x, r2_h, w, color="steelblue", alpha=0.85)
    for xi, v in enumerate(r2_h):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="steelblue")
    ax.set_title("High-SPI R²", fontsize=10, fontweight="bold")
    ax.set_xlabel("Layer"); ax.set_ylabel("R²", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylim(0, 1.08); ax.grid(True, alpha=0.25, axis="y")

    # [1,1] Low-SPI R²
    ax = axes[1, 1]
    bars = ax.bar(x, r2_l, w, color="coral", alpha=0.85)
    for xi, v in enumerate(r2_l):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="coral")
    ax.set_title("Low-SPI R²", fontsize=10, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylim(0, 1.08); ax.grid(True, alpha=0.25, axis="y")

    # [1,2] Delta R²
    ax = axes[1, 2]
    colors_dr = ["steelblue" if v > 0 else "coral" for v in delta_r2]
    bars = ax.bar(x, delta_r2, w, color=colors_dr, alpha=0.85)
    ax.axhline(0, color="black", lw=1.2)
    for xi, v in enumerate(delta_r2):
        ax.text(xi, v + 0.01 * np.sign(v) if abs(v) > 0.005 else 0.01,
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5, color=colors_dr[xi])
    ax.set_title(r"$\Delta$R² (High $-$ Low)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Layer"); ax.set_ylabel("ΔR²", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    # Shared legend at top
    handles = [
        mpatches.Patch(color="steelblue", alpha=0.85, label="High-SPI tokens"),
        mpatches.Patch(color="coral", alpha=0.85, label="Low-SPI tokens"),
        mpatches.Patch(color="steelblue", alpha=0.85, label=r"Δ > 0: High wins"),
        mpatches.Patch(color="coral", alpha=0.85, label=r"Δ < 0: Low wins"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(
        f"Between-Layer: High-SPI vs Low-SPI (top_k={top_k}, all {len(concepts)} concepts aggregated)\n"
        f"SPI = I(H,Y) / (I(X,H) + 0.15·mean(I(X,H))) | KSG Estimator",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, "ksg_spi_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_ksg_spi_profile(results, concepts, n_layers, output_dir):
    """SPI profile: per-layer R² line + heatmap + per-concept bars."""
    n_concepts = len(concepts)
    layers = list(range(n_layers))
    x = np.arange(n_layers)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1.4])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]

    # ── Panel [0]: Per-layer R² line ──────────────────────────────────────
    ax = axes[0]
    r2_h_all, r2_l_all = [], []
    for li, layer in enumerate(layers):
        vals_h = [v["r2_high"] for k, v in results.items() if k[0] == layer]
        vals_l = [v["r2_low"]  for k, v in results.items() if k[0] == layer]
        r2_h_all.append(np.nanmean(vals_h))
        r2_l_all.append(np.nanmean(vals_l))

    ax.plot(x, r2_h_all, "o-", color="steelblue", lw=2, ms=6, label="High-SPI tokens")
    ax.plot(x, r2_l_all, "s--", color="coral",    lw=2, ms=6, label="Low-SPI tokens")
    ax.fill_between(x, r2_h_all, r2_l_all,
                    where=[a > b for a, b in zip(r2_h_all, r2_l_all)],
                    alpha=0.15, color="steelblue", label="High-SPI wins")
    ax.fill_between(x, r2_h_all, r2_l_all,
                    where=[a <= b for a, b in zip(r2_h_all, r2_l_all)],
                    alpha=0.15, color="coral", label="Low-SPI wins")
    ax.set_xlabel("Layer"); ax.set_ylabel("Avg R²")
    ax.set_title("Per-Layer R²: High-SPI vs Low-SPI (KSG, avg over concepts)")
    ax.set_xticks(x); ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)

    # ── Panel [1]: R² diff heatmap ────────────────────────────────────────
    ax = axes[1]
    diff_matrix = np.full((n_layers, n_concepts), np.nan)
    for li, layer in enumerate(layers):
        for ci, concept in enumerate(concepts):
            vals_h = [results.get((layer, concept, d), {}).get("r2_high", np.nan) for d in range(4)]
            vals_l = [results.get((layer, concept, d), {}).get("r2_low",  np.nan) for d in range(4)]
            active_dims = CONCEPT_PARAM_SPEC[concept][0]
            vh = np.nanmean([vals_h[d] for d in active_dims])
            vl = np.nanmean([vals_l[d] for d in active_dims])
            diff_matrix[li, ci] = vh - vl

    v_abs = max(0.05, np.nanmax(np.abs(diff_matrix)))
    im = ax.imshow(diff_matrix, aspect="auto", cmap="RdBu_r", vmin=-v_abs, vmax=v_abs)
    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels([c[:16] for c in concepts], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
    ax.set_ylabel("Layer")
    ax.set_title("R² Difference: (High - Low)\nBlue = High-SPI better, Red = Low-SPI better", fontsize=10)
    plt.colorbar(im, ax=ax, label="ΔR²")
    for li in range(n_layers):
        for ci in range(n_concepts):
            val = diff_matrix[li, ci]
            if np.isnan(val):
                continue
            ax.text(ci, li, f"{val:.3f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(val) > v_abs * 0.55 else "black")

    # ── Panel [2]: Per-concept R² bars ───────────────────────────────────
    ax_conc = axes[2]
    ax_conc.remove()

    n_cols_c = min(3, n_concepts)
    n_rows_c = (n_concepts + n_cols_c - 1) // n_cols_c

    inset_axes = []
    for ci, concept in enumerate(concepts):
        row = ci // n_cols_c
        col = ci % n_cols_c
        box = [col / n_cols_c, 1.0 - (row + 1) / n_rows_c,
               1.0 / n_cols_c - 0.02, 1.0 / n_rows_c - 0.03]
        ax_ins = fig.add_axes(box)
        inset_axes.append(ax_ins)

        active_dims = CONCEPT_PARAM_SPEC[concept][0]
        for li, layer in enumerate(layers):
            vals_h = [results.get((layer, concept, d), {}).get("r2_high", np.nan) for d in range(4)]
            vals_l = [results.get((layer, concept, d), {}).get("r2_low",  np.nan) for d in range(4)]
            r2h = np.nanmean([vals_h[d] for d in active_dims])
            r2l = np.nanmean([vals_l[d] for d in active_dims])
            bw = 0.35
            ax_ins.bar(li - bw/2, r2h, bw, color="steelblue", alpha=0.85)
            ax_ins.bar(li + bw/2, r2l, bw, color="coral",    alpha=0.85)

        ax_ins.set_title(concept[:20], fontsize=9, fontweight="bold")
        ax_ins.set_xticks(range(n_layers))
        ax_ins.set_xticklabels([f"L{l}" for l in range(n_layers)], fontsize=7)
        ax_ins.tick_params(axis="y", labelsize=7)
        ax_ins.grid(True, alpha=0.2, axis="y")
        ax_ins.set_ylim(0, 1.05)

    fig.text(0.5, 0.28, "Per-Concept R²: blue = High-SPI tokens, coral = Low-SPI tokens (KSG)",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    plt.suptitle(f"KSG-Based SPI Profile: Per-Layer, Per-Concept (SPI ordering)\n"
                 f"SPI = I(H,Y) / (I(X,H) + 0.15*mean(I(X,H)))",
                 fontsize=12, fontweight="bold")
    out_path = os.path.join(output_dir, "ksg_spi_profile.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_ksg_within_layer(results, concepts, n_layers, output_dir, percentile):
    """
    Within-layer percentile analysis plots.

    Figure 1 (summary): 3 panels across layers:
      [0] Per-layer avg MSE: Top vs Bottom 25% bars
      [1] Per-layer avg R²: Top vs Bottom 25% bars
      [2] Win-rate bar per layer

    Figure 2 (per-concept grid): rows=concepts, cols=layers.
    Each cell: grouped bar — Top25% MSE (blue) | Bottom25% MSE (coral)
              then   Top25% R² (blue, hatched) | Bottom25% R² (coral, hatched)
    So the top/bottom comparison is immediately visible per concept and per layer.
    """
    layers = list(range(n_layers))
    n_concepts = len(concepts)
    x = np.arange(n_layers)
    w = 0.35

    # Aggregate per-layer averages
    mse_t_all, mse_b_all = [], []
    r2_t_all, r2_b_all = [], []
    win_rates = []

    for li, layer in enumerate(layers):
        vt = [v["mse_top"] for k, v in results.items() if k[0] == layer]
        vb = [v["mse_bot"] for k, v in results.items() if k[0] == layer]
        mse_t_all.append(np.nanmean(vt))
        mse_b_all.append(np.nanmean(vb))
        rt = [v["r2_top"] for k, v in results.items() if k[0] == layer]
        rb = [v["r2_bot"] for k, v in results.items() if k[0] == layer]
        r2_t_all.append(np.nanmean(rt))
        r2_b_all.append(np.nanmean(rb))
        wins = sum(1 for k, v in results.items() if k[0] == layer and v["delta_mse"] > 0)
        total = sum(1 for k, v in results.items() if k[0] == layer)
        win_rates.append(wins / max(total, 1))

    # ── Figure 1: Summary ─────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(3 * n_layers + 2, 5))
    fig1.subplots_adjust(wspace=0.35)

    # [0] MSE bars
    ax = axes1[0]
    bars_t = ax.bar(x - w/2, mse_t_all, w, label=f"Top {percentile:.0f}% SPI", color="steelblue", alpha=0.85)
    bars_b = ax.bar(x + w/2, mse_b_all, w, label=f"Bottom {percentile:.0f}% SPI", color="coral", alpha=0.85)
    for xi in range(len(x)):
        lo = max(mse_t_all[xi], mse_b_all[xi])
        b = "T" if mse_t_all[xi] < mse_b_all[xi] else "B"
        ax.text(xi, lo + lo * 0.01, b, ha="center", va="bottom",
                color="steelblue" if b == "T" else "coral", fontsize=9, fontweight="bold")
    ax.set_xlabel("Layer"); ax.set_ylabel("MSE (mean)")
    ax.set_title(f"Avg MSE: Top vs Bottom {percentile:.0f}% SPI (KSG)", fontsize=10)
    ax.set_xticks(x); ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y"); ax.set_ylim(bottom=0)

    # [1] R² bars
    ax = axes1[1]
    bars_t = ax.bar(x - w/2, r2_t_all, w, label=f"Top {percentile:.0f}%", color="steelblue", alpha=0.85)
    bars_b = ax.bar(x + w/2, r2_b_all, w, label=f"Bottom {percentile:.0f}%", color="coral", alpha=0.85)
    for xi in range(len(x)):
        hi = max(r2_t_all[xi], r2_b_all[xi])
        b = "T" if r2_t_all[xi] > r2_b_all[xi] else "B"
        ax.text(xi, hi + 0.01, b, ha="center", va="bottom",
                color="steelblue" if b == "T" else "coral", fontsize=9, fontweight="bold")
    ax.set_xlabel("Layer"); ax.set_ylabel("R² (mean)")
    ax.set_title(f"Avg R²: Top vs Bottom {percentile:.0f}% SPI (KSG)", fontsize=10)
    ax.set_xticks(x); ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y"); ax.set_ylim(0, 1.05)

    # [2] Win rate
    ax = axes1[2]
    colors_wr = ["steelblue" if r > 0.5 else "coral" for r in win_rates]
    bars = ax.bar(x, win_rates, color=colors_wr, alpha=0.85)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, label="random")
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wr:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Layer"); ax.set_ylabel("Win Rate")
    ax.set_title(f"Win Rate (>0.5 = Top {percentile:.0f}% better)", fontsize=10)
    ax.set_xticks(x); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.25, axis="y"); ax.legend(fontsize=8)

    fig1.suptitle(f"Within-Layer Summary: Top vs Bottom {percentile:.0f}% SPI Tokens (KSG)\n"
                   f"SPI = I(H,Y) / (I(X,H) + 0.15*mean(I(X,H)))",
                   fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path1 = os.path.join(output_dir, f"ksg_within_layer_summary_{percentile:.0f}pct.png")
    fig1.savefig(out_path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path1}")

    # ── Figure 2: Per-concept subplot grid ────────────────────────────────────
    # rows=concepts, cols=layers; each cell = top25% MSE | bottom25% MSE bars
    #                                        top25% R²  | bottom25% R²  bars
    from matplotlib.gridspec import GridSpec
    fig_h = max(3.0 * n_concepts + 1.5, 6)
    fig2 = plt.figure(figsize=(3 * n_layers + 2, fig_h))
    gs = GridSpec(n_concepts, n_layers, figure=fig2,
                  hspace=0.6, wspace=0.3,
                  left=0.06, right=0.98, top=0.93, bottom=0.06)

    cmap = plt.cm.tab10.colors

    for ci, concept in enumerate(concepts):
        active_dims = CONCEPT_PARAM_SPEC[concept][0]

        for li, layer in enumerate(layers):
            ax = fig2.add_subplot(gs[ci, li])

            # Collect per-dim values for this concept/layer, then average
            mse_t_vals = [_agg(results, "mse_top", layer, concept, active_dims)]
            mse_b_vals = [_agg(results, "mse_bot", layer, concept, active_dims)]
            r2_t_vals  = [_agg(results, "r2_top",  layer, concept, active_dims)]
            r2_b_vals  = [_agg(results, "r2_bot",  layer, concept, active_dims)]

            mt, mb = np.nanmean(mse_t_vals), np.nanmean(mse_b_vals)
            rt, rb = np.nanmean(r2_t_vals), np.nanmean(r2_b_vals)

            bw = 0.35
            # MSE bars (upper half)
            ax.bar(0 - bw/2, mt, bw, color="steelblue", alpha=0.85)
            ax.bar(1 + bw/2, mb, bw, color="coral", alpha=0.85)
            # R² bars (lower half, hatched)
            ax.bar(0 - bw/2, rt, bw, color="steelblue", alpha=0.45, hatch="//")
            ax.bar(1 + bw/2, rb, bw, color="coral",    alpha=0.45, hatch="\\\\")
            ax.axvline(0.5, color="gray", lw=0.8, ls="--")

            # Winner labels
            mse_win = "T" if mt < mb else "B"
            r2_win  = "T" if rt > rb else "B"
            y_top = max(mt, mb, 0.001)
            ax.text(0, y_top + y_top * 0.02, mse_win, ha="center", va="bottom",
                    color="steelblue" if mse_win == "T" else "coral",
                    fontsize=8, fontweight="bold")
            ax.text(1, max(rt, rb) + 0.02, r2_win, ha="center", va="bottom",
                    color="steelblue" if r2_win == "T" else "coral",
                    fontsize=8, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["T", "B"], fontsize=8)
            ax.set_xlim(-0.7, 1.7)
            ax.grid(True, alpha=0.2, axis="y")
            ax.tick_params(axis="y", labelsize=7)

            if ci == 0:
                ax.set_title(f"Layer {layer}", fontsize=9, fontweight="bold")
            if li == 0:
                ax.set_ylabel(f"{concept[:14]}\nMSE  |  R²", fontsize=8, fontweight="bold")
            if ci == n_concepts - 1:
                ax.set_xlabel("Group", fontsize=8)

            # Cell border
            for spine in ax.spines.values():
                spine.set_edgecolor("lightgray")
                spine.set_linewidth(0.8)

    # Shared legend at bottom
    handles = [
        mpatches.Patch(color="steelblue", alpha=0.85, label=f"Top {percentile:.0f}% SPI — MSE"),
        mpatches.Patch(color="coral",    alpha=0.85, label=f"Bottom {percentile:.0f}% SPI — MSE"),
        mpatches.Patch(color="steelblue", alpha=0.45, hatch="//", label=f"Top {percentile:.0f}% SPI — R²"),
        mpatches.Patch(color="coral",    alpha=0.45, hatch="\\\\", label=f"Bottom {percentile:.0f}% SPI — R²"),
    ]
    fig2.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
                bbox_to_anchor=(0.5, 0.01))

    fig2.suptitle(
        f"Per-Concept: Top vs Bottom {percentile:.0f}% SPI Tokens — MSE & R² by Layer (KSG)\n"
        f"Solid bars = MSE | Hatched bars = R² | T = Top {percentile:.0f}% | B = Bottom {percentile:.0f}%",
        fontsize=12, fontweight="bold",
    )
    out_path2 = os.path.join(output_dir, f"ksg_within_layer_concept_grid_{percentile:.0f}pct.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Per-Token KSG MI Analysis with SPI-Based Token Ordering"
    )
    parser.add_argument("--rep_dir", type=str, required=False,
                        default="./results/synthetic/representations/",
                        help="Directory with layer_token_representations.pt from Step 2")
    parser.add_argument("--output_dir", type=str,
                        default="./results/synthetic/ksg_mi_token/",
                        help="Output directory")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of top/bottom SPI token positions (between-layer analysis)")
    parser.add_argument("--percentile", type=float, default=25.0,
                        help="Percentile for within-layer analysis: top/bottom N%% tokens (default: 25.0)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per concept for KSG computation (default: all)")
    parser.add_argument("--max_samples_ksg", type=int, default=500,
                        help="Subsample N for KSG computation (default: 500)")
    parser.add_argument("--probe_epochs", type=int, default=200,
                        help="Epochs for linear probe training")
    parser.add_argument("--probe_lr", type=float, default=1e-3,
                        help="Learning rate for probe")
    parser.add_argument("--pca_dim", type=int, default=32,
                        help="PCA dimension for KSG (default: 32)")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="K-nearest neighbors for KSG (default: 5)")
    parser.add_argument("--spi_bias_factor", type=float, default=0.15,
                        help="SPI denominator bias factor (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Specific layers to analyze (default: all)")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of parallel processes for KSG computation. "
                             "0=auto-detect (all available CPUs). "
                             "Set to match number of GPUs for best throughput.")
    parser.add_argument("--dual_stream", action="store_true",
                        help="Run dual-stream analysis: X-stream vs Y-stream through decoder layers. "
                             "Requires --ckpt_path to load decoder weights.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to Timer checkpoint (required for dual_stream mode).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("Step 5: Per-Token KSG MI Analysis (SPI-Based Ordering)")
    print("=" * 70)
    print(f"  rep_dir          : {args.rep_dir}")
    print(f"  output_dir       : {args.output_dir}")
    print(f"  top_k            : {args.top_k}")
    print(f"  percentile       : {args.percentile}")
    print(f"  pca_dim          : {args.pca_dim}")
    print(f"  k_neighbors      : {args.k_neighbors}")
    print(f"  spi_bias_factor  : {args.spi_bias_factor}")
    print(f"  max_samples      : {args.max_samples}")
    print(f"  max_samples_ksg  : {args.max_samples_ksg}")
    print(f"  probe_epochs     : {args.probe_epochs}")
    print(f"  probe_lr         : {args.probe_lr}")
    print(f"  device           : {device}")
    print("=" * 70)

    # ── Load token representations ─────────────────────────────────────────────
    print("\n[0] Loading token representations...")
    token_path = os.path.join(args.rep_dir, "layer_token_representations.pt")
    if not os.path.exists(token_path):
        print(f"ERROR: Token representations not found at {token_path}")
        print("Please re-run Step 2 with --save_token_reps flag.")
        return

    ckpt = torch.load(token_path, map_location="cpu", weights_only=False)
    layer_tokens = ckpt["layer_tokens"]
    params = ckpt["params"]
    concept_idx = ckpt["concept_idx"]
    labels = ckpt["labels"]
    concepts = ckpt["concepts"]
    n_samples_per_concept = ckpt["n_samples_per_concept"]
    d_model = ckpt["d_model"]
    n_layers_total = ckpt["n_layers"]

    # Load patch embeddings: prefer dedicated patch_tokens key, fall back to layer 0 hidden
    if "patch_tokens" in ckpt:
        patch_tokens = ckpt["patch_tokens"]
        print(f"  Loaded patch_tokens from checkpoint: {patch_tokens.shape}")
    else:
        print("  Warning: patch_tokens not found in checkpoint.")
        print("  Using layer 0 hidden state as proxy for patch embeddings.")
        patch_tokens = layer_tokens[0] if isinstance(layer_tokens[0], torch.Tensor) else layer_tokens[0]
        print(f"  Proxy patch_tokens shape: {patch_tokens.shape}")

    print(f"  Loaded: {n_layers_total} layers, shape: {layer_tokens[0].shape}")
    print(f"  Concepts: {concepts}")

    if args.layers is not None:
        layer_tokens = [layer_tokens[l] for l in args.layers if l < n_layers_total]
        n_layers = len(layer_tokens)
    else:
        n_layers = n_layers_total

    # ── Dual-Stream Analysis ───────────────────────────────────────────────────
    if args.dual_stream:
        if args.ckpt_path is None:
            print("ERROR: --ckpt_path required for dual_stream mode.")
            return
        print(f"\n[Dual-Stream] Loading Timer decoder from {args.ckpt_path} ...")
        import argparse as _argparse
        ns = _argparse.Namespace(
            task_name="forecast",
            is_training=0, is_finetuning=0, train_test=0,
            use_multi_gpu=False, d_layers=1, target="OT",
            checkpoints="./checkpoints/", inverse=False,
            use_amp=False, use_weight_decay=0, weight_decay=0.01,
            loss="MSE", lradj="type1", train_epochs=0, patience=3,
            learning_rate=1e-4, itr=1, finetune_epochs=0,
            output_attention=False, distil=True,
            model_id="timer_probe", model="Timer",
            output_len_list=None, mask_rate=0.25,
            data_type="custom", decay_fac=0.75,
            cos_warm_up_steps=100, cos_max_decay_steps=60000,
            cos_max_decay_epoch=10, cos_max=1e-4, cos_min=2e-6,
            patch_len=getattr(args, "patch_len", 96),
            stride=getattr(args, "stride", 96),
            d_model=d_model,
            d_ff=getattr(args, "d_ff", d_model * 2),
            n_heads=getattr(args, "n_heads", 8),
            dropout=0.1, activation="gelu",
            e_layers=n_layers,
            factor=3,
            ckpt_path=args.ckpt_path,
            is_injection_test=False,
            enable_refinement=False,
            use_align_loss=False,
            align_loss_layers=list(range(n_layers)),
            truncate_guide=True,
        )
        for k, v in vars(_argparse.Namespace()).items():
            if not hasattr(ns, k):
                setattr(ns, k, v)

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
        from Timer import Model as TimerModel
        timer_model = TimerModel(ns).to(device)
        ckpt_sd = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        # Strip prefixes the same way Timer.py does
        if "state_dict" in ckpt_sd:
            sd = {k.replace("model.", ""): v for k, v in ckpt_sd["state_dict"].items()}
        else:
            sd = ckpt_sd
        for prefix in ["module.", "model.", "backbone."]:
            sd = {k.replace(prefix, ""): v for k, v in sd.items()}
        sd = {k.replace("enc_embedding.", "patch_embedding."): v for k, v in sd.items()}
        sd = {k.replace("decoder.", ""): v for k, v in sd.items()}
        timer_model.backbone.load_state_dict(sd, strict=False)
        timer_model.eval()
        decoder_layers = timer_model.backbone.decoder.attn_layers

        print(f"  [Dual-Stream] Decoder layers: {len(decoder_layers)}")

        results_ds = run_dual_stream_ksg_analysis(
            layer_tokens=layer_tokens,
            patch_tokens=patch_tokens,
            params=params,
            concept_idx=concept_idx,
            concepts=concepts,
            n_samples_per_concept=n_samples_per_concept,
            d_model=d_model,
            n_layers=n_layers,
            output_dir=args.output_dir,
            decoder_layers=decoder_layers,
            pca_dim=args.pca_dim,
            k_neighbors=args.k_neighbors,
            n_samples_ksg=args.max_samples_ksg,
            spi_bias_factor=args.spi_bias_factor,
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr,
            device=device,
            seed=args.seed,
        )

        # Save dual-stream results
        ds_results_path = os.path.join(args.output_dir, "ksg_dual_stream_results.pt")
        torch.save({
            "results": results_ds,
            "concepts": concepts,
            "n_layers": n_layers,
            "config": vars(args),
        }, ds_results_path)
        print(f"[Dual-Stream] Saved: {ds_results_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY: Dual-Stream KSG MI (X-stream vs Y-stream)")
        print("=" * 70)
        for layer in range(n_layers):
            vals_r2 = [v["r2"] for k, v in results_ds.items() if k[0] == layer]
            vals_spi = [v["spi"] for k, v in results_ds.items() if k[0] == layer]
            vals_mse = [v["mse"] for k, v in results_ds.items() if k[0] == layer]
            if vals_r2:
                print(f"  L{layer:2d}  avg_R2={np.nanmean(vals_r2):.4f}  "
                      f"avg_SPI={np.nanmean(vals_spi):.4f}  avg_MSE={np.nanmean(vals_mse):.4f}")

        print("=" * 70)
        return

    # ── Mode dispatch: always run both analyses ─────────────────────────────────
    # 1. Between-layer: top_k / bottom_k tokens
    print(f"\n[Between-Layer] top_k={args.top_k}")
    results, mi_matrix = run_ksg_spi_token_analysis(
        layer_tokens=layer_tokens,
        patch_tokens=patch_tokens,
        params=params,
        concept_idx=concept_idx,
        labels=labels,
        concepts=concepts,
        n_samples_per_concept=n_samples_per_concept,
        d_model=d_model,
        n_layers=n_layers,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_samples=args.max_samples,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        pca_dim=args.pca_dim,
        k_neighbors=args.k_neighbors,
        n_samples_ksg=args.max_samples_ksg,
        spi_bias_factor=args.spi_bias_factor,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    results_path = os.path.join(args.output_dir, "ksg_spi_results.pt")
    torch.save({
        "results": results,
        "concepts": concepts,
        "n_layers": n_layers,
        "top_k": args.top_k,
        "config": vars(args),
    }, results_path)
    print(f"Saved: {results_path}")

    print("=" * 70)
    print("SUMMARY: High-SPI vs Low-SPI Token Probe (KSG, between-layer)")
    print("=" * 70)
    wins_high = sum(1 for v in results.values() if v["delta_mse"] > 0)
    total = len(results)
    print(f"  High-SPI WIN: {wins_high}/{total} ({100*wins_high/total:.1f}%)")
    print(f"  Low-SPI WIN:  {total - wins_high}/{total} ({100*(total-wins_high)/total:.1f}%)")

    # 2. Within-layer: top 25% / bottom 25% tokens
    within_pct = 25.0
    print(f"\n[Within-Layer] percentile={within_pct:.0f}%")
    results_wl = run_ksg_within_layer_analysis(
        layer_tokens=layer_tokens,
        patch_tokens=patch_tokens,
        params=params,
        concept_idx=concept_idx,
        labels=labels,
        concepts=concepts,
        n_samples_per_concept=n_samples_per_concept,
        d_model=d_model,
        n_layers=n_layers,
        output_dir=args.output_dir,
        percentile=within_pct,
        max_samples=args.max_samples,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        pca_dim=args.pca_dim,
        k_neighbors=args.k_neighbors,
        n_samples_ksg=args.max_samples_ksg,
        spi_bias_factor=args.spi_bias_factor,
        device=device,
        seed=args.seed,
    )

    out_path = os.path.join(args.output_dir, f"ksg_within_layer_{within_pct:.0f}pct.pt")
    torch.save({"results": results_wl, "percentile": within_pct, "config": vars(args)}, out_path)
    print(f"Saved: {out_path}")

    print(f"\nSUMMARY: Within-layer Top vs Bottom {within_pct:.0f}% SPI (KSG)")
    wins_top = sum(1 for v in results_wl.values() if v["delta_mse"] > 0)
    total_wl = len(results_wl)
    print(f"  Top-{within_pct:.0f}% WIN:  {wins_top}/{total_wl} ({100*wins_top/total_wl:.1f}%)")
    print(f"  Bottom-{within_pct:.0f}% WIN: {total_wl - wins_top}/{total_wl} ({100*(total_wl-wins_top)/total_wl:.1f}%)")

    # ── Plotting ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n[Plotting...]")
        # Between-layer plots
        plot_ksg_spi_summary(results, concepts, n_layers, args.output_dir, args.top_k)
        plot_ksg_spi_profile(results, concepts, n_layers, args.output_dir)
        # Within-layer plots
        plot_ksg_within_layer(results_wl, concepts, n_layers, args.output_dir, within_pct)

    print(f"\n[Done] Step 5 complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
