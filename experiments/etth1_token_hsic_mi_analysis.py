#!/usr/bin/env python3
"""
Per-token HSIC (MI surrogate) for Timer on ETTh1, per decoder layer.

Reference: "Information Peaks in Transformer Representations" (DR paper)
Method: HSIC(X, Y) measures the Hilbert-Schmidt Independence Criterion, a
        non-parametric estimator of mutual information between:
          - X = h_x^{(l)}[t] : representation of token t at decoder layer l
          - Y = h_y^{(l)}[t] : representation of ground-truth future window at
                                the SAME layer l when the GT window is re-fed
                                through the decoder (patched, embedded, then
                                passed through the same l transformer layers).

Per-layer, per-token HSIC curve:
  For each token index t (patch position) and decoder layer l:
    HSIC_{l,t} = HSIC( h_x^{(l)}[:, t, :],  h_y^{(l)}[:, t, :] )
  Aggregated across all samples in the batch → produces a "curve" [HSIC_0 … HSIC_{N-1}]
  The curve peaks identify tokens that carry the most predictive information
  about the future at that particular layer.

Usage (single GPU):
  python experiments/etth1_token_hsic_mi_analysis.py \
    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
    --root_path ./datasets/ --data ETTh1 --data_path ETTh1.csv

Usage (multi-GPU, recommended):
  torchrun --nnodes=1 --nproc_per_node=5 \
    experiments/etth1_token_hsic_mi_analysis.py \
    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
    --root_path ./datasets/ --data ETTh1 --data_path ETTh1.csv \
    --use_multi_gpu --batch_size 64 --out_dir ./results/token_hsic_etth1/

Output:
  results/token_hsic_etth1/global_token_hsic_peaks_etth1_token.json — per-layer, per-token HSIC curves
  results/token_hsic_etth1/plots/token_hsic_per_layer.png          — bar chart per layer
  results/token_hsic_etth1/plots/token_hsic_overlay.png           — overlay all layers
"""

import argparse
import json
import os
import sys
from typing import Optional, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model as TimerModel
from utils.masking import TriangularCausalMask
from utils.hsic.hsic_core import compute_stable_rank


# ─── HSIC core (copied from utils/hsic/hsic_core.py to keep this script self-contained) ──

def median_sq_bandwidth(X: torch.Tensor) -> float:
    """Median heuristic: median of upper-triangle pairwise squared Euclidean distances."""
    n = X.shape[0]
    if n < 2:
        return 1.0
    d = torch.cdist(X, X, p=2.0)
    triu = torch.triu_indices(n, n, offset=1, device=X.device)
    sq = d[triu[0], triu[1]] ** 2
    med = torch.median(sq)
    return float(med.clamp(min=1e-12).item())


def hsic_with_separate_sigmas(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x_sq: float,
    sigma_y_sq: float,
) -> float:
    """
    Biased HSIC with separate Gaussian RBF bandwidths for X and Y.

    HSIC(X, Y) = (1/(n-1)^2) * tr(H K H @ H L H)
    """
    n = X.shape[0]
    if n < 2:
        return 0.0
    d2_x = torch.cdist(X, X, p=2.0) ** 2
    d2_y = torch.cdist(Y, Y, p=2.0) ** 2
    K = torch.exp(-d2_x / (2.0 * sigma_x_sq))
    L = torch.exp(-d2_y / (2.0 * sigma_y_sq))
    H = torch.eye(n, device=X.device, dtype=X.dtype) - (1.0 / n)
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float((torch.trace(Kc @ Lc) / ((n - 1) ** 2)).item())


def zscore_cols(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Column-wise z-score normalization."""
    m = t.mean(dim=0, keepdim=True)
    s = t.std(dim=0, keepdim=True).clamp(min=eps)
    return (t - m) / s


# ─── Stable-rank bandwidth normalisation ─────────────────────────────────────

def extract_stable_ranks(model: torch.nn.Module) -> Dict[int, float]:
    """
    Extract the stable rank sr_l for each decoder layer, where:
        M = W_Q @ W_K^T / sqrt(d_head)
        sr_l = ||M||_F^2 / ||M||_2^2

    W_Q / W_K are the query/key projection matrices of the attention layer's
    AttentionLayer submodule. d_head = d_model // n_heads.

    Returns:
        dict[l -> sr_l (float)]
    """
    core = _unwrap_timer(model)
    d_model = core.decoder.attn_layers[0].attention.out_projection.in_features
    n_heads = core.decoder.attn_layers[0].attention.n_heads
    d_head = d_model // n_heads
    stable_ranks = {}
    for l, attn_layer in enumerate(core.decoder.attn_layers):
        inner = attn_layer.attention          # AttentionLayer
        w_q = inner.query_projection.weight  # [D, D]
        w_k = inner.key_projection.weight    # [D, D]
        stable_ranks[l] = compute_stable_rank(w_q, w_k, d_head)
    return stable_ranks


def build_layer_sigmas(
    h_x_per_layer: Dict[int, torch.Tensor],
    stable_ranks: Dict[int, float],
    sigma_base: float,
) -> Dict[int, float]:
    """
    Adjust per-layer Gaussian-kernel bandwidth via stable-rank normalisation.

    sigma_l = sigma_base * sqrt(sr_l / sr_ref)

    The reference sr_ref is layer 1 (the first transformer layer after embedding),
    which serves as a natural baseline for the model's initial representational
    capacity. Using the stable rank as a proxy for the layer's "effective
    dimensionality" corrects for the ~sqrt(d) distance scaling across transformer
    layers, giving the HSIC kernel a consistent scale regardless of which layer
    is being examined.

    Args:
        h_x_per_layer:  dict[l -> Tensor [B, N, D]] (used only for shape info)
        stable_ranks:   dict[l -> sr_l]
        sigma_base:     baseline bandwidth (median heuristic from layer 0/embedding)

    Returns:
        dict[l -> sigma_l_sq (float)]
    """
    # Use layer 1 as the reference (sr_ref), not layer 0
    sr_ref = stable_ranks.get(1, 1.0)
    if sr_ref < 1.0:
        sr_ref = 1.0
    sigma_sq_dict = {}
    for l in sorted(h_x_per_layer.keys()):
        sr_l = stable_ranks.get(l, 1.0)
        if sr_l < 1.0:
            sr_l = 1.0
        sigma_l = sigma_base * np.sqrt(sr_l / sr_ref)
        sigma_sq_dict[l] = sigma_l * sigma_l
    return sigma_sq_dict


# ─── Model unwrapping ─────────────────────────────────────────────────────────

def _unwrap_timer(model):
    if hasattr(model, "module"):
        return model.module
    return model


# ─── Per-token HSIC computation ──────────────────────────────────────────────

def compute_hsic_per_token_batch(
    h_x_per_layer: Dict[int, torch.Tensor],
    h_y_per_layer: Dict[int, torch.Tensor],
    device: torch.device,
    sigma_sq_per_layer: Optional[Dict[int, float]] = None,
) -> Dict[int, np.ndarray]:
    """
    Compute per-token HSIC for each decoder layer across all samples in the
    current batch. This is the core HSIC calculation referenced by the DR paper.

    For each layer l and each token t:
        HSIC_{l,t} = HSIC( h_x^{(l)}[:, t, :],  h_y^{(l)}[:, t, :] )

    where:
      - h_x^{(l)}[:, t, :]  : hidden state of token t at layer l (from input x)
      - h_y^{(l)}[:, t, :]  : hidden state of token t at layer l (from GT future
                               y re-fed through the decoder up to layer l)

    Both tensors are z-scored column-wise before computing HSIC.

    Bandwidth selection:
      - If sigma_sq_per_layer is provided, use stable-rank-adjusted bandwidths:
          sigma_l = sigma_base * sqrt(sr_l / sr_ref)
        where sr_l is the stable rank of W_Q @ W_K^T / sqrt(d_head) for layer l.
      - Otherwise, fall back to the median heuristic (original behaviour).

    Args:
        h_x_per_layer: dict[layer_idx -> Tensor of shape [B, N, D]]
        h_y_per_layer: dict[layer_idx -> Tensor of shape [B, N, D]]
        sigma_sq_per_layer: optional dict[layer_idx -> sigma_sq] for stable-rank
                            bandwidth normalisation. When None, reverts to median.

    Returns:
        dict[layer_idx -> np.ndarray of shape [N]]  (one HSIC value per token)
    """
    results = {}
    for layer_idx in sorted(h_x_per_layer.keys()):
        hx = h_x_per_layer[layer_idx].to(device)
        hy = h_y_per_layer[layer_idx].to(device)
        B, N, D = hx.shape

        if sigma_sq_per_layer is not None and layer_idx in sigma_sq_per_layer:
            # Stable-rank adjusted: same sigma for both X and Y kernels
            sigma_sq = sigma_sq_per_layer[layer_idx]
            sigma_x_sq = sigma_sq
            sigma_y_sq = sigma_sq
        else:
            # Median heuristic (original): sigma is the median of squared pairwise
            # distances over all token representations in this layer.
            hx_z = zscore_cols(hx.reshape(B * N, D))
            hy_z = zscore_cols(hy.reshape(B * N, D))
            sigma_x_sq = median_sq_bandwidth(hx_z)
            sigma_y_sq = median_sq_bandwidth(hy_z)

        # Z-score token representations (needed for median path; no-op if using
        # stable-rank because sigma is data-independent anyway).
        hx_z = zscore_cols(hx.reshape(B * N, D))
        hy_z = zscore_cols(hy.reshape(B * N, D))

        hsic_vals = []
        for t in range(N):
            X_t = hx_z[t::N]
            Y_t = hy_z[t::N]
            score = hsic_with_separate_sigmas(X_t, Y_t, sigma_x_sq, sigma_y_sq)
            hsic_vals.append(score)
        results[layer_idx] = np.asarray(hsic_vals, dtype=np.float64)

    return results


# ─── Hidden-state extraction ──────────────────────────────────────────────────

def extract_per_layer_hidden_states(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    n_vars: int,
    n_layers: int,
    device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Extract per-layer hidden states for both input (h_x) and GT future (h_y).

    h_x^{(l)} : input x passes through patch embedding then layers 0..l
    h_y^{(l)} : GT full batch_y (aligned to seq_len) re-fed through the SAME
                patch embedding + decoder layers 0..l

    batch_y is sliced to seq_len so that N_y == N_x (patch count matches),
    ensuring 1-to-1 token correspondence for per-token HSIC computation.

    Returns:
        h_x_per_layer: dict[l -> Tensor [B, N, D]]
        h_y_per_layer: dict[l -> Tensor [B, N, D]]
    """
    core = _unwrap_timer(model)
    B = batch_x.shape[0]
    seq_len = batch_x.shape[1]
    x = batch_x.to(device).float()
    # Slice y to match seq_len for same patch count N_y == N_x
    y = batch_y[:, -seq_len:, :].to(device).float()

    # Normalize (Non-stationary Transformer)
    means = x.mean(1, keepdim=True).detach()
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x_norm = (x - means) / stdev
    y_norm = (y - means) / stdev  # same stats for y

    # Patch + embed both streams
    x2 = x_norm.permute(0, 2, 1)  # [B, M, T]
    y2 = y_norm.permute(0, 2, 1)  # [B, M, T]

    dec_in_x, _ = core.enc_embedding(x2)   # [B*M, N, D]
    dec_in_y, _ = core.enc_embedding(y2)   # [B*M, N, D]

    BM, N, D = dec_in_x.shape
    mask = TriangularCausalMask(BM, N, device=device)

    # Mean-pool over variates → [B, N, D]
    def pool_x(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    def pool_y(z):
        # Same n_vars, N, D for y
        return z.view(B, n_vars, N, D).mean(dim=1)

    h_x_per_layer = {}
    h_y_per_layer = {}

    h_x = dec_in_x
    h_y = dec_in_y

    for i, layer_module in enumerate(core.decoder.attn_layers):
        h_x, _ = layer_module(h_x, attn_mask=mask)
        h_y, _ = layer_module(h_y, attn_mask=mask)

        h_x_per_layer[i] = pool_x(h_x).detach().cpu()
        h_y_per_layer[i] = pool_y(h_y).detach().cpu()

    return h_x_per_layer, h_y_per_layer


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_token_hsic_per_layer(
    token_hsic_per_layer: dict[int, np.ndarray],
    n_layers: int,
    output_path: str,
    patch_len: int,
    seq_len: int,
):
    """Bar chart of per-token HSIC values, one subplot per decoder layer."""
    N = next(iter(token_hsic_per_layer.values())).shape[0]
    fig, axes = plt.subplots(
        2, n_layers // 2, figsize=(4 * n_layers, 7), squeeze=False
    )
    axes = axes.flatten()

    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        hsic_curve = token_hsic_per_layer.get(layer_idx, np.zeros(N))
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, N))
        bars = ax.bar(range(N), hsic_curve, color=colors, edgecolor="navy", linewidth=0.5)
        ax.set_xlabel("Token (Patch) Index", fontsize=9)
        ax.set_ylabel("HSIC", fontsize=9)
        ax.set_title(f"Decoder Layer {layer_idx}\nPer-token HSIC", fontsize=10)
        ax.set_xticks(range(N))
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate max
        max_t = int(np.argmax(hsic_curve))
        ax.annotate(f"max=t{max_t}\n({hsic_curve[max_t]:.4f})",
                    xy=(max_t, hsic_curve[max_t]),
                    xytext=(max_t + 0.5, hsic_curve[max_t] * 1.02),
                    fontsize=7, color="crimson")

    plt.suptitle(
        f"Per-token HSIC (MI) per Decoder Layer | seq_len={seq_len}, patch_len={patch_len}",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def plot_token_hsic_overlay(
    token_hsic_per_layer: dict[int, np.ndarray],
    n_layers: int,
    output_path: str,
):
    """Overlay all layers' HSIC curves on one plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.tab10
    for layer_idx in range(n_layers):
        hsic_curve = token_hsic_per_layer.get(layer_idx)
        if hsic_curve is None:
            continue
        ax.plot(range(len(hsic_curve)), hsic_curve, "o-",
                color=cmap(layer_idx % 10), linewidth=1.8, markersize=5,
                label=f"Layer {layer_idx}")

    ax.set_xlabel("Token (Patch) Index", fontsize=11)
    ax.set_ylabel("HSIC (MI Surrogate)", fontsize=11)
    ax.set_title("Per-token HSIC Overlay: All Decoder Layers", fontsize=13)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def plot_token_hsic_heatmap(
    token_hsic_per_layer: dict[int, np.ndarray],
    n_layers: int,
    output_path: str,
):
    """Heatmap: rows=layers, cols=tokens, color=HSIC."""
    N = next(iter(token_hsic_per_layer.values())).shape[0]
    matrix = np.stack([token_hsic_per_layer[l] for l in range(n_layers)], axis=0)  # [L, N]

    fig, ax = plt.subplots(figsize=(max(6, N * 0.8), max(4, n_layers * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Token (Patch) Index", fontsize=11)
    ax.set_ylabel("Decoder Layer", fontsize=11)
    ax.set_xticks(range(N))
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_title("Per-token HSIC Heatmap: Layers × Tokens", fontsize=13)
    plt.colorbar(im, ax=ax, label="HSIC")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def plot_high_low_token_comparison(
    token_hsic_per_layer: dict[int, np.ndarray],
    n_layers: int,
    output_path: str,
):
    """Bar chart: mean HSIC of high-MI vs low-MI tokens per layer."""
    mean_high = []
    mean_low = []
    std_high = []
    std_low = []

    for layer_idx in range(n_layers):
        vals = token_hsic_per_layer.get(layer_idx, np.zeros(1))
        if len(vals) < 4:
            mean_high.append(np.nan)
            mean_low.append(np.nan)
            std_high.append(0)
            std_low.append(0)
            continue

        q3 = np.percentile(vals, 75)
        q1 = np.percentile(vals, 25)
        high_mask = vals >= q3
        low_mask = vals <= q1

        mean_high.append(np.mean(vals[high_mask]) if high_mask.any() else np.nan)
        mean_low.append(np.mean(vals[low_mask]) if low_mask.any() else np.nan)
        std_high.append(np.std(vals[high_mask]) if high_mask.any() else 0)
        std_low.append(np.std(vals[low_mask]) if low_mask.any() else 0)

    x = np.arange(n_layers)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.9), 5))
    ax.bar(x - width / 2, mean_high, width, yerr=std_high,
           label="High HSIC (Q3+)", color="crimson", alpha=0.8, capsize=3)
    ax.bar(x + width / 2, mean_low, width, yerr=std_low,
           label="Low HSIC (Q1-)", color="steelblue", alpha=0.8, capsize=3)
    ax.set_xlabel("Decoder Layer", fontsize=11)
    ax.set_ylabel("Mean HSIC", fontsize=11)
    ax.set_title("High-MI vs Low-MI Token Mean HSIC per Layer", fontsize=13)
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


# ─── Namespace builder ────────────────────────────────────────────────────────

def build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    for k, v in {
        "task_name": "forecast",
        "is_training": 0,
        "is_finetuning": 0,
        "train_test": 0,
        "use_multi_gpu": False,
        "d_layers": 1,
        "d_model": 1024,
        "d_ff": 2048,
        "e_layers": 8,
        "n_heads": 8,
        "dropout": 0.1,
        "factor": 1,
        "activation": "gelu",
        "target": "OT",
        "checkpoints": args.root_path,
        "inverse": False,
        "use_amp": False,
        "use_weight_decay": 0,
        "weight_decay": 0.01,
        "loss": "MSE",
        "lradj": "type1",
        "train_epochs": 0,
        "patience": 3,
        "learning_rate": 1e-4,
        "itr": 1,
        "finetune_epochs": 0,
        "output_attention": False,
        "distil": True,
        "model_id": "token_hsic_analysis",
        "model": "Timer",
        "output_len_list": None,
        "mask_rate": 0.25,
        "data_type": "custom",
        "decay_fac": 0.75,
        "cos_warm_up_steps": 100,
        "cos_max_decay_steps": 60000,
        "cos_max_decay_epoch": 10,
        "cos_max": 1e-4,
        "cos_min": 2e-6,
        "embed": "timeF",
        "freq": "h",
        "stride": args.patch_len,
        "features": "M",
    }.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-token HSIC (MI) analysis per decoder layer for Timer on ETTh1"
    )
    parser.add_argument("--ckpt_path", type=str,
                        default="./checkpoints/Timer_forecast_1.0.ckpt",
                        help="Path to Timer checkpoint")
    parser.add_argument("--root_path", type=str, default="./datasets/",
                        help="Root path to dataset directory")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv",
                        help="Dataset CSV filename")
    parser.add_argument("--data", type=str, default="ETTh1",
                        help="Dataset name (ETTh1, ETTm1, etc.)")
    parser.add_argument("--features", type=str, default="M",
                        choices=["M", "S", "MS"],
                        help="Feature type: M=multivariate, S=single, MS=multivariate-single")
    parser.add_argument("--seq_len", type=int, default=672,
                        help="Input sequence length")
    parser.add_argument("--label_len", type=int, default=576,
                        help="Decoder label length")
    parser.add_argument("--pred_len", type=int, default=96,
                        help="Prediction length")
    parser.add_argument("--output_len", type=int, default=96,
                        help="Output sequence length")
    parser.add_argument("--patch_len", type=int, default=96,
                        help="Patch length (stride)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for HSIC computation")
    parser.add_argument("--e_layers", type=int, default=8,
                        help="Number of encoder/decoder layers (same as num_layers for Timer)")
    parser.add_argument("--factor", type=int, default=3,
                        help="Attention factor (for FullAttention)")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="DataLoader num_workers")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Number of decoder layers in Timer")
    parser.add_argument("--d_model", type=int, default=1024,
                        help="Model dimension d_model")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward dimension d_ff")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--activation", type=str, default="gelu",
                        help="FFN activation function")
    parser.add_argument("--embed", type=str, default="timeF",
                        help="Time embedding type")
    parser.add_argument("--freq", type=str, default="h",
                        help="Time frequency (h=hourly, t=minutely)")
    parser.add_argument("--use_ims", action="store_true",
                        help="Use incremental multi-scale dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default="./results/token_hsic_etth1/",
                        help="Output directory")
    parser.add_argument("--model_id", type=str, default="etth1",
                        help="Model identifier for output files")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_multi_gpu", action="store_true",
                        help="Use multi-GPU with torchrun")
    parser.add_argument("--use_stable_rank_sigma", action="store_true",
                        help="Adjust HSIC kernel bandwidth via stable-rank normalisation "
                             "across layers (sigma_l = sigma_base * sqrt(sr_l / sr_ref), ref=layer-1)")
    args = parser.parse_args()

    # ── DDP setup ──────────────────────────────────────────────────────────────
    rank = 0
    local_rank = 0
    world_size = 1
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Multi-GPU requires torchrun")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        print(f"[Token HSIC] rank {rank}/{world_size}", flush=True)

    device = torch.device(f"cuda:{local_rank}" if args.use_multi_gpu else args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_multi_gpu:
        dist.barrier()

    # ── 1. Build model ─────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n[1] Loading Timer model from: {args.ckpt_path}")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    ns = build_namespace(args)
    ns.ckpt_path = args.ckpt_path
    ns.d_model = args.d_model
    ns.d_ff = args.d_ff
    ns.e_layers = args.e_layers
    ns.factor = args.factor
    ns.n_heads = args.n_heads
    ns.dropout = args.dropout
    ns.use_multi_gpu = bool(args.use_multi_gpu)

    model = TimerModel(ns).to(device)
    model.eval()

    if args.use_multi_gpu:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    core = _unwrap_timer(model)
    n_layers = len(core.decoder.attn_layers)
    N = args.seq_len // args.patch_len

    if rank == 0:
        print(f"  Model on device={device}, num_layers={n_layers}, N={N} patches, "
              f"d_model={args.d_model}")

    # ── 2. Stable-rank bandwidth schedule ─────────────────────────────────────
    stable_ranks = extract_stable_ranks(model)
    if rank == 0:
        print("\n[2] Stable ranks per layer:")
        for l, sr in stable_ranks.items():
            print(f"  Layer {l}: sr = {sr:.2f}")

    sigma_sq_per_layer = None
    if args.use_stable_rank_sigma:
        # sigma_base = median squared distance from layer 0 (computed on first batch)
        _sigmas_computed = False

    # ── 3. Load data ───────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n[3] Loading {args.data} data (batch_size={args.batch_size})...")

    ns.batch_size = args.batch_size
    _, train_loader = data_provider(ns, flag="train")
    _, val_loader = data_provider(ns, flag="val")

    # Determine actual n_vars from first batch
    for _, (_, _, _, _) in enumerate(train_loader):
        pass  # just to trigger loader init
    # n_vars is extracted from enc_embedding output dynamically

    # ── 4. Collect hidden states and compute per-token HSIC ───────────────────
    if rank == 0:
        print(f"\n[4] Extracting hidden states and computing per-token HSIC...")

    # Accumulate: per-layer, per-token HSIC values across all batches
    # hsic_accum[l, t] = running sum of HSIC values
    # count_accum[l]   = number of batches processed
    hsic_accum = {l: np.zeros(N, dtype=np.float64) for l in range(n_layers)}
    count_accum = {l: 0 for l in range(n_layers)}

    loaders = [("train", train_loader), ("val", val_loader)]
    batch_count = 0

    for split_name, loader in loaders:
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            # Use first batch to determine actual n_vars and sigma_base (stable-rank path)
            if batch_count == 0:
                with torch.no_grad():
                    tmp = batch_x[:1].float().to(device)
                    tmp2 = tmp.permute(0, 2, 1)
                    _, n_vars_actual = core.enc_embedding(tmp2)
                    n_vars = int(n_vars_actual)
                    if rank == 0:
                        print(f"  Actual n_vars detected: {n_vars}")

                # ── Compute sigma_base on first batch (stable-rank path) ──────────
                if args.use_stable_rank_sigma:
                    # Extract one batch of hidden states (just to get per-layer reps)
                    h_x_tmp, h_y_tmp = extract_per_layer_hidden_states(
                        model, batch_x, batch_y, n_vars=n_vars,
                        n_layers=n_layers, device=device,
                    )
                    # sigma_base = median squared distance from layer 0 (z-scored)
                    hx0 = h_x_tmp[0].to(device)
                    hy0 = h_y_tmp[0].to(device)
                    B0, N0, D0 = hx0.shape
                    hx0_z = zscore_cols(hx0.reshape(B0 * N0, D0))
                    hy0_z = zscore_cols(hy0.reshape(B0 * N0, D0))
                    sigma_base = float(np.sqrt(median_sq_bandwidth(hx0_z)))
                    # Also use hy0 for a symmetric estimate and average
                    sigma_base_y = float(np.sqrt(median_sq_bandwidth(hy0_z)))
                    sigma_base = (sigma_base + sigma_base_y) / 2.0
                    if rank == 0:
                        print(f"  sigma_base (stable-rank) = {sigma_base:.4f}")

                    sigma_sq_per_layer = build_layer_sigmas(h_x_tmp, stable_ranks, sigma_base)
                    if rank == 0:
                        print("  Per-layer sigma (stable-rank adjusted):")
                        for l, sq in sigma_sq_per_layer.items():
                            print(f"    Layer {l}: σ² = {sq:.4f}  (sr = {stable_ranks.get(l, 0):.2f})")

            # Extract per-layer hidden states for h_x and h_y
            h_x_per_layer, h_y_per_layer = extract_per_layer_hidden_states(
                model, batch_x, batch_y,
                n_vars=n_vars,
                n_layers=n_layers,
                device=device,
            )

            # Compute per-token HSIC for this batch
            batch_hsic = compute_hsic_per_token_batch(
                h_x_per_layer, h_y_per_layer, device,
                sigma_sq_per_layer=sigma_sq_per_layer,
            )

            # Accumulate (average across batches)
            for layer_idx in range(n_layers):
                if layer_idx in batch_hsic:
                    hsic_accum[layer_idx] += batch_hsic[layer_idx]
                    count_accum[layer_idx] += 1

            batch_count += 1

            if batch_count >= 50:
                break

        if batch_count >= 50:
            break

    if args.use_multi_gpu:
        dist.barrier()

    # Average HSIC across all batches
    token_hsic_per_layer = {}
    for layer_idx in range(n_layers):
        if count_accum[layer_idx] > 0:
            token_hsic_per_layer[layer_idx] = hsic_accum[layer_idx] / count_accum[layer_idx]
        else:
            token_hsic_per_layer[layer_idx] = np.zeros(N, dtype=np.float64)

    if rank == 0:
        print(f"\n  Computed per-token HSIC across {batch_count} batches")

        # ── 4. Build high/low MI patch sets per layer ─────────────────────────
        layers_dict = {}
        for layer_idx in range(n_layers):
            hsic_curve = token_hsic_per_layer[layer_idx]
            q3 = float(np.percentile(hsic_curve, 75))
            q1 = float(np.percentile(hsic_curve, 25))
            high_mi_patches = [int(i) for i, v in enumerate(hsic_curve) if v >= q3]
            low_mi_patches = [int(i) for i, v in enumerate(hsic_curve) if v <= q1]

            layers_dict[str(layer_idx)] = {
                "hsic_curve": hsic_curve.tolist(),
                "q3_threshold": q3,
                "q1_threshold": q1,
                "high_mi_patches": high_mi_patches,
                "low_mi_patches": low_mi_patches,
                "high_mi_ratio": len(high_mi_patches) / N,
                "hsic_mean": float(hsic_curve.mean()),
                "hsic_std": float(hsic_curve.std()),
                "hsic_max": float(hsic_curve.max()),
                "hsic_min": float(hsic_curve.min()),
            }

        result_json = {
            "model_id": args.model_id,
            "N": N,
            "patch_len": args.patch_len,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "n_vars": n_vars,
            "num_layers": n_layers,
            "num_batches": batch_count,
            "use_stable_rank_sigma": args.use_stable_rank_sigma,
            "stable_ranks": {str(l): stable_ranks.get(l) for l in range(n_layers)},
            "sigma_base": float(sigma_base) if args.use_stable_rank_sigma else None,
            "sigma_sq_per_layer": {str(l): sigma_sq_per_layer.get(l) if sigma_sq_per_layer else None
                                   for l in range(n_layers)},
            "layers": layers_dict,
        }

        json_path = os.path.join(args.out_dir, f"global_token_hsic_peaks_{args.model_id}.json")
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"\n[5] JSON saved: {json_path}")

        # ── 6. Generate plots ─────────────────────────────────────────────────
        print(f"\n[6] Generating plots...")
        plot_dir = os.path.join(args.out_dir, "plots")

        plot_token_hsic_per_layer(
            token_hsic_per_layer, n_layers,
            os.path.join(plot_dir, "token_hsic_per_layer.png"),
            patch_len=args.patch_len, seq_len=args.seq_len,
        )
        plot_token_hsic_overlay(
            token_hsic_per_layer, n_layers,
            os.path.join(plot_dir, "token_hsic_overlay.png"),
        )
        plot_token_hsic_heatmap(
            token_hsic_per_layer, n_layers,
            os.path.join(plot_dir, "token_hsic_heatmap.png"),
        )
        plot_high_low_token_comparison(
            token_hsic_per_layer, n_layers,
            os.path.join(plot_dir, "high_low_token_comparison.png"),
        )

        # ── 7. Summary table ─────────────────────────────────────────────────
        print("\n" + "=" * 90)
        if args.use_stable_rank_sigma:
            print(f"{'Layer':>6} | {'SR':>7} | {'Sigma':>8} | {'HSIC Mean':>10} | {'HSIC Std':>10} | "
                  f"{'HSIC Max':>10} | {'#High':>7} | {'#Low':>7} | {'Top Token':>10}")
            print("-" * 90)
            for layer_idx in range(n_layers):
                ld = layers_dict[str(layer_idx)]
                sr = stable_ranks.get(layer_idx, 0)
                sq = sigma_sq_per_layer.get(layer_idx, 0) if sigma_sq_per_layer else 0
                top_t = int(np.argmax(token_hsic_per_layer[layer_idx]))
                print(f"{layer_idx:>6} | {sr:>7.2f} | {sq:>8.4f} | "
                      f"{ld['hsic_mean']:>10.6f} | {ld['hsic_std']:>10.6f} | "
                      f"{ld['hsic_max']:>10.6f} | {len(ld['high_mi_patches']):>7} | "
                      f"{len(ld['low_mi_patches']):>7} | {top_t:>10}")
        else:
            print(f"{'Layer':>6} | {'HSIC Mean':>10} | {'HSIC Std':>10} | "
                  f"{'HSIC Max':>10} | {'#High':>7} | {'#Low':>7} | "
                  f"{'Top Token':>10}")
            print("-" * 90)
            for layer_idx in range(n_layers):
                ld = layers_dict[str(layer_idx)]
                top_t = int(np.argmax(token_hsic_per_layer[layer_idx]))
                print(f"{layer_idx:>6} | {ld['hsic_mean']:>10.6f} | {ld['hsic_std']:>10.6f} | "
                      f"{ld['hsic_max']:>10.6f} | {len(ld['high_mi_patches']):>7} | "
                      f"{len(ld['low_mi_patches']):>7} | {top_t:>10}")
        print("=" * 90)

        print(f"\n[Done] Results saved to: {args.out_dir}")

    if args.use_multi_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
