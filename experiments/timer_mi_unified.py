#!/usr/bin/env python3
"""
Timer 层间 MI 分析 —— 支持三种估计器统一调用。

Supported MI methods (via --mi_method):
  ksg           : KSG (Kraskov-Stoegbauer-Grassberger) kNN estimator
  fastmi_rpy2   : fastMI via R package (requires rpy2 + fastMI R package)
  fastmi_python : fastMI pure Python (copula + FFT self-consistent density)
  all           : run all available methods and produce a comparison table

Reference:
  fastMI paper: "fastMI: A fast and consistent copula-based nonparametric
                 estimator of mutual information" (arXiv:2212.10268)

Usage:
  python experiments/timer_mi_unified.py --mi_method ksg --data ETTh1
  python experiments/timer_mi_unified.py --mi_method fastmi_python --data ETTh1
  python experiments/timer_mi_unified.py --mi_method all --data ETTh1
"""

import argparse
import json
import os
import gc
import sys
import warnings
from datetime import datetime
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model as TimerModel
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from utils.masking import TriangularCausalMask

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — MI Estimators
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    KSG (Kraskov, Stoegbauer, Grassberger) estimator for I(X;Y).
    x: [N, Dx], y: [N, Dy]
    Returns MI in bits.
    """
    import scipy.spatial as ss
    import scipy.special as sp

    N = x.shape[0]
    if N <= k + 1:
        return 0.0
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)
    xy = np.concatenate((x, y), axis=1)
    tree_xy = ss.cKDTree(xy)

    dist_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dist_xy[:, k]
    eps_strict = np.maximum(eps - 1e-10, 0)

    nx = np.array([
        max(len(tree_x.query_ball_point(x[i], r=eps_strict[i], p=np.inf)) - 1, 0)
        for i in range(N)
    ])
    ny = np.array([
        max(len(tree_y.query_ball_point(y[i], r=eps_strict[i], p=np.inf)) - 1, 0)
        for i in range(N)
    ])

    psi_k = sp.digamma(k)
    psi_N = sp.digamma(N)
    mean_psi = np.mean(sp.digamma(nx + 1) + sp.digamma(ny + 1))
    mi = psi_k - mean_psi + psi_N
    return max(0.0, mi / np.log(2))


def _check_rpy2_available() -> bool:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        importr("fastMI")
        return True
    except Exception:
        return False


def compute_mi_fastmi_rpy2(x: np.ndarray, y: np.ndarray) -> float:
    """
    fastMI via rpy2 bridge to the R fastMI package.
    Requires: pip install rpy2  &&  R -e 'install.packages("fastMI")'
    x: [N, Dx], y: [N, Dy]
    Returns MI in bits.
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    fastmi = importr("fastMI")
    x_r = ro.r.matrix(x, nrow=x.shape[0], ncol=x.shape[1])
    y_r = ro.r.matrix(y, nrow=y.shape[0], ncol=y.shape[1])
    ro.globalenv["_x"] = x_r
    ro.globalenv["_y"] = y_r
    res = ro.r(f'estim_fmi(_x, _y, n_perm=0)')
    mi_nats = float(res[0])
    return mi_nats / np.log(2)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF Φ(x) via scipy."""
    from scipy.stats import norm
    return norm.cdf(x)


def _probit(u: np.ndarray) -> np.ndarray:
    """Probit transform: Φ^{-1}(u) with clamping."""
    from scipy.stats import norm
    u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
    return norm.ppf(u_clipped)


def _ecdf_transform(X: np.ndarray) -> np.ndarray:
    """Empirical CDF transform: U_{ij} = rank(X_{ij}) / (n+1) per column."""
    n = X.shape[0]
    U = np.empty_like(X)
    for j in range(X.shape[1]):
        ranks = np.argsort(np.argsort(X[:, j]))
        U[:, j] = (ranks + 1) / (n + 1)
    return U


def _fft_density_1d(x: np.ndarray, grid_size: int = 512) -> np.ndarray:
    """
    Self-consistent FFT density estimator for 1-D data (fastMI paper).
    Fixed-point iteration on the empirical characteristic function.
    Returns density evaluated at each x (standardised).
    """
    from scipy.fft import fft, ifft, fftfreq
    from scipy.interpolate import interp1d

    x = np.asarray(x).flatten()
    n = len(x)
    x_std = (x - x.mean()) / (x.std() + 1e-10)

    rng = 4.0
    t_grid = np.linspace(-rng, rng, grid_size)
    dt = t_grid[1] - t_grid[0]

    # ECF C(ω_k) = (1/n) Σ exp(i ω_k x_std_j)
    omega = 2 * np.pi * fftfreq(grid_size, dt)
    C = np.zeros(grid_size, dtype=complex)
    for k in range(grid_size):
        w = omega[k]
        C[k] = np.mean(np.cos(w * x_std) + 1j * np.sin(w * x_std))

    # Fixed-point iteration: φ_{t+1} = n·C / (n-1 + |φ_t|²)
    phi = np.abs(C).copy()
    for _ in range(50):
        phi_new = n * C / (n - 1 + np.abs(phi) ** 2)
        if np.max(np.abs(phi_new - phi)) < 1e-6:
            break
        phi = np.abs(phi_new)

    # Map phi back to FFT index and inverse FFT
    k_to_n = grid_size * dt / (2 * np.pi)
    k_indices = np.clip(np.round(omega * k_to_n).astype(int), 0, grid_size - 1)
    phi_full = np.zeros(grid_size, dtype=complex)
    phi_full[k_indices] = phi

    density_grid = np.maximum(np.real(ifft(phi_full)), 1e-10)
    density_grid /= density_grid.sum() * dt

    interp = interp1d(t_grid, density_grid, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
    return interp(x_std)


def _estimate_1d_copula_densities(v: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Estimate 1D copula densities for each column of v (probit-space data).
    v: [n, d] in standard-normal space
    Returns: [n, d] density values
    """
    n, d = v.shape
    densities = np.zeros((n, d))
    for j in range(d):
        u_j = _ecdf_transform(v[:, j:j+1])[:, 0]
        v_j = _probit(np.clip(u_j, 1e-10, 1 - 1e-10))
        densities[:, j] = _fft_density_1d(v_j, grid_size=grid_size)
    return densities


def compute_mi_fastmi_python(x: np.ndarray, y: np.ndarray,
                             grid_size: int = 256) -> float:
    """
    fastMI pure Python — follows arXiv:2212.10268.

    Algorithm:
      1. PCA reduction (cap at 32 dims total).
      2. Copula transform: ECDF → uniform → probit (Φ^{-1}).
      3. Per-dimension 1D FFT copula density estimation.
      4. Joint copula density via 2D Gaussian KDE on cross-dimensional pairs.
      5. Plug-in MI: (1/n) Σ ln(c_joint / c_x c_y), in bits.

    Returns MI in bits.
    """
    n, dx = x.shape
    dy = y.shape[1]

    # PCA reduction: keep min(n, dx+dy, 32) total dims
    d_total = min(n, dx + dy, 32)
    d_x = min(n, dx, max(1, d_total // 2))
    d_y = min(n, dy, d_total - d_x)

    if d_x < dx:
        x_red = PCA(n_components=d_x).fit_transform(x)
    else:
        x_red = x[:, :d_x] if dx > 0 else x.reshape(-1, 1)

    if d_y < dy:
        y_red = PCA(n_components=d_y).fit_transform(y)
    else:
        y_red = y[:, :d_y] if dy > 0 else y.reshape(-1, 1)

    # Copula transform
    v_x = _probit(_ecdf_transform(x_red))
    v_y = _probit(_ecdf_transform(y_red))

    # 1D marginal copula densities via FFT
    c_x_1d = _estimate_1d_copula_densities(v_x, grid_size)
    c_y_1d = _estimate_1d_copula_densities(v_y, grid_size)
    log_c_x = np.sum(np.log(c_x_1d + 1e-10), axis=1)
    log_c_y = np.sum(np.log(c_y_1d + 1e-10), axis=1)

    # Joint copula density via 2D Gaussian KDE on cross-dimensional pairs
    # (captures pairwise dependence that 1D products miss)
    log_c_xy = np.zeros(n)
    n_pairs = 0
    for j in range(min(v_x.shape[1], 3)):
        for k in range(min(v_y.shape[1], 3)):
            v_jk = np.column_stack([v_x[:, j], v_y[:, k]])
            try:
                from scipy.stats import gaussian_kde
                std_jk = v_jk.std(axis=0) + 1e-10
                bw = (std_jk * n ** (-1/5)).mean()
                kde = gaussian_kde(v_jk.T, bw_method=bw / std_jk.mean())
                c_joint = kde(v_jk.T)
                log_c_xy += np.log(c_joint + 1e-10)
                n_pairs += 1
            except Exception:
                pass

    if n_pairs > 0:
        log_c_xy /= n_pairs
    else:
        log_c_xy = log_c_x + log_c_y

    mi_bits = np.mean(log_c_xy - log_c_x - log_c_y) / np.log(2)
    return max(0.0, float(mi_bits))


def compute_mi(x: np.ndarray, y: np.ndarray,
               method: Literal["ksg", "fastmi_rpy2", "fastmi_python"],
               ksg_k: int = 5,
               fastmi_grid_size: int = 256) -> float:
    """Unified dispatcher for MI estimation."""
    if method == "ksg":
        return compute_mi_ksg(x, y, k=ksg_k)
    elif method == "fastmi_rpy2":
        if not _check_rpy2_available():
            raise RuntimeError(
                "fastmi_rpy2 needs rpy2 + fastMI R package.\n"
                "  pip install rpy2\n"
                "  R -e 'install.packages(\"fastMI\")'"
            )
        return compute_mi_fastmi_rpy2(x, y)
    elif method == "fastmi_python":
        return compute_mi_fastmi_python(x, y, grid_size=fastmi_grid_size)
    else:
        raise ValueError(f"Unknown method: {method}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Timer model utilities
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer_model(ckpt_path: str, patch_len: int, stride: int,
                      d_model: int, d_ff: int, e_layers: int,
                      n_heads: int, dropout: float,
                      seq_len: int, pred_len: int) -> TimerModel:
    config = Config(
        task_name='forecast', ckpt_path=ckpt_path,
        patch_len=patch_len, stride=stride,
        d_model=d_model, d_ff=d_ff, e_layers=e_layers,
        n_heads=n_heads, dropout=dropout,
        output_attention=False, distil=True, use_revin=False,
        seq_len=seq_len, pred_len=pred_len,
        d_layers=1, factor=1, enc_in=1, dec_in=1, c_out=1,
        activation='gelu', use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(), devices='0', num_workers=4,
        freq='h', data='custom', embed='timeF', target='OT',
        features='M', des='Exp', lradj='type1', use_amp=False,
        is_finetuning=0, label_len=pred_len, output_len=pred_len,
        batch_size=64, train_epochs=1, patience=3,
        learning_rate=3e-5, itr=1, use_ims=False, inverse=False,
        use_align_loss=False, align_loss_layers=list(range(e_layers)),
    )
    model = TimerModel(config)
    model.eval()
    return model


def _unwrap_timer(model):
    return model.module if hasattr(model, "module") else model


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Token extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_layer_tokens(model, data_loader, device, n_layers: int,
                         seq_len: int, pred_len: int, n_vars: int):
    core = _unwrap_timer(model)
    all_hist_tokens = [[] for _ in range(n_layers)]
    all_future_tokens = [[] for _ in range(n_layers)]
    n_patches = None
    n_future_patches = None

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(
                data_loader, desc="提取 token 表示"):
            B = seq_x.shape[0]
            seq_x = seq_x.float().to(device)
            seq_y = seq_y.float().to(device)

            means = seq_x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(seq_x, dim=1, keepdim=True,
                                          unbiased=False) + 1e-5).detach()
            x_norm = (seq_x - means) / stdev
            y_norm = (seq_y - means) / stdev

            x2 = x_norm.permute(0, 2, 1)
            y2 = y_norm.permute(0, 2, 1)

            dec_in_x, _ = core.enc_embedding(x2)
            dec_in_y, _ = core.enc_embedding(y2)

            BM, N, D = dec_in_x.shape
            BM_y, N_y, _ = dec_in_y.shape

            if n_patches is None:
                n_patches = N
                n_future_patches = N_y
                print(f"  DEBUG: B={B}, BM={BM}, N={N}, D={D}")

            nv_x = BM // B
            nv_y = BM_y // B

            mask_x = TriangularCausalMask(BM, N, device=device)
            mask_y = TriangularCausalMask(BM_y, N_y, device=device)

            # Hidden states are passed through layers progressively (recurrently).
            # h_x and h_y accumulate layer by layer so that layer l sees
            # the output of layer l-1, not the raw embedding.
            h_x = dec_in_x
            h_y = dec_in_y
            for li, layer_module in enumerate(core.decoder.attn_layers):
                h_x, _ = layer_module(h_x, attn_mask=mask_x)
                h_y, _ = layer_module(h_y, attn_mask=mask_y)
                all_hist_tokens[li].append(
                    h_x.view(B, nv_x, N, D).mean(dim=1).float().cpu())
                all_future_tokens[li].append(
                    h_y.view(B, nv_y, N_y, D).mean(dim=1).float().cpu())

            del dec_in_x, dec_in_y
            gc.collect()
            torch.cuda.empty_cache()

    hist_tokens = [torch.cat(toks, dim=0) for toks in all_hist_tokens]
    future_tokens = [torch.cat(toks, dim=0) for toks in all_future_tokens]
    return hist_tokens, future_tokens, n_patches, n_future_patches


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mi_overlay(out_dir: str, mi_matrices: dict, n_layers: int,
                    n_patches: int, model_id: str):
    """Overlay MI curves: all layers × all methods on one figure."""
    methods = list(mi_matrices.keys())
    n_methods = len(methods)
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(
        n_layers, 1,
        figsize=(max(10, n_patches * 0.5), max(4, n_layers * 1.5)),
        squeeze=False
    )
    axes = axes.flatten()

    for li in range(n_layers):
        ax = axes[li]
        for mi_idx, (mi_method, mi_matrix) in enumerate(mi_matrices.items()):
            ax.plot(range(n_patches), mi_matrix[li],
                    linewidth=1.5, marker='.', markersize=3,
                    label=mi_method, color=cmap(mi_idx % 10), alpha=0.85)
        ax.set_ylabel(f"L{li}\nMI(bits)", fontsize=7)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, n_patches - 1)

    axes[-1].set_xlabel("History Patch Index")
    fig.suptitle(f"Timer — Per-Layer MI Overlay | {model_id}", fontsize=12, y=1.0)
    plt.tight_layout()
    path = os.path.join(out_dir, f"mi_overlay_{model_id}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")


def plot_mi_per_layer_bar(out_dir: str, mi_matrices: dict,
                           n_layers: int, model_id: str):
    """Grouped bar: mean MI per layer, one bar group per method."""
    methods = list(mi_matrices.keys())
    n_methods = len(methods)
    x = np.arange(n_layers)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(10, n_layers * 1.2), 5))
    cmap = plt.cm.tab10
    for i, (mi_method, mi_matrix) in enumerate(mi_matrices.items()):
        means = mi_matrix.mean(axis=1)
        stds = mi_matrix.std(axis=1)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=mi_method, alpha=0.85, color=cmap(i % 10), capsize=2)

    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Mean MI (bits)")
    ax.set_title(f"Timer — Mean MI per Layer | {model_id}")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(out_dir, f"mi_per_layer_bar_{model_id}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")


def plot_heatmap(out_dir: str, mi_matrix: np.ndarray,
                  n_layers: int, n_patches: int,
                  title_suffix: str, filename: str):
    """Layer × Patch heatmap."""
    fig, ax = plt.subplots(
        figsize=(max(6, n_patches * 0.5), max(4, n_layers * 0.7))
    )
    im = ax.imshow(mi_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel("History Patch Index")
    ax.set_ylabel("Decoder Layer")
    ax.set_title(f"MI Heatmap {title_suffix}")
    ax.set_xticks(range(0, n_patches, max(1, n_patches // 8)))
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    plt.colorbar(im, ax=ax, label='MI (bits)')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Timer MI — Unified KSG / fastMI (rpy2) / fastMI (Python)"
    )
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=32,
                        help="PCA dims before MI; 0 = no PCA")
    parser.add_argument("--fastmi_grid_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str,
                        default="./results/timer_mi_unified")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--data_type", type=str, default="ETTh1")
    parser.add_argument("--model_id", type=str, default="etth1")
    parser.add_argument("--mi_method", type=str, default="all",
                        choices=["ksg", "fastmi_rpy2", "fastmi_python", "all"])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"Timer_MI_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.mi_method == "all":
        active = ["ksg", "fastmi_python"]
        if _check_rpy2_available():
            active.append("fastmi_rpy2")
        else:
            print("[fastmi_rpy2] not available, skipping.")
    else:
        active = [args.mi_method]

    print("=" * 70)
    print("Timer MI Analysis — Unified (KSG / fastMI)")
    print("=" * 70)
    print(f"  ckpt_path   : {args.ckpt_path}")
    print(f"  data        : {args.data}")
    print(f"  seq_len     : {args.seq_len}  pred_len: {args.pred_len}")
    print(f"  patch_len   : {args.patch_len}  e_layers: {args.e_layers}")
    print(f"  pca_dim     : {args.pca_dim}  k_neighbors: {args.k_neighbors}")
    print(f"  grid_size   : {args.fastmi_grid_size}  (fastMI Python)")
    print(f"  methods     : {active}")
    print(f"  device      : {device}")
    print("=" * 70)

    # 1 — Dataset
    print("\n>>> Phase 1: 加载数据集...")
    test_dataset = CIDatasetBenchmark(
        root_path=os.path.join(args.root_path, args.data_path),
        flag='test', input_len=args.seq_len, pred_len=args.pred_len,
        data_type=args.data_type, scale=True, timeenc=1, freq=args.freq,
    )
    n_vars = test_dataset.n_var
    print(f"  n_vars={n_vars}, total_samples={len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    # 2 — Model
    print("\n>>> Phase 2: 加载 Timer 模型...")
    model = build_timer_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len, stride=args.stride,
        d_model=args.d_model, d_ff=args.d_ff, e_layers=args.e_layers,
        n_heads=args.n_heads, dropout=args.dropout,
        seq_len=args.seq_len, pred_len=args.pred_len,
    ).to(device).eval()
    print(f"  device={device}")

    # 3 — Extract tokens
    print("\n>>> Phase 3: 提取各层 token 表示...")
    hist_tokens, future_tokens, n_patches, n_future = extract_layer_tokens(
        model, test_loader, device, args.e_layers,
        args.seq_len, args.pred_len, n_vars
    )
    N_total = hist_tokens[0].shape[0]
    D = hist_tokens[0].shape[2]
    print(f"  N={N_total}, patches={n_patches}, future_patches={n_future}, D={D}")

    num_batches = min(args.num_batches, N_total // args.batch_size)
    sample_idx = min(args.batch_size * num_batches, N_total)
    print(f"  实际使用样本: {sample_idx}")

    # 4 — MI computation
    all_mi_matrices = {}
    for mi_method in active:
        print(f"\n{'='*55}")
        print(f">>> Method: {mi_method.upper()}")
        print(f"{'='*55}")
        mi_matrix = np.zeros((args.e_layers, n_patches))

        for li in tqdm(range(args.e_layers), desc=f"MI ({mi_method})"):
            hist_li = hist_tokens[li][:sample_idx].numpy()
            future_li = future_tokens[li][:sample_idx].numpy()
            future_flat = future_li.reshape(sample_idx, -1)

            for pi in range(n_patches):
                x = hist_li[:, pi, :]

                pca_d = args.pca_dim
                if pca_d > 0:
                    x_red = PCA(n_components=min(pca_d, sample_idx, x.shape[1])
                                 ).fit_transform(x)
                    y_red = PCA(n_components=min(pca_d, sample_idx, future_flat.shape[1])
                                ).fit_transform(future_flat)
                else:
                    x_red, y_red = x, future_flat

                mi_matrix[li, pi] = compute_mi(
                    x_red, y_red,
                    method=mi_method,
                    ksg_k=args.k_neighbors,
                    fastmi_grid_size=args.fastmi_grid_size,
                )

            del hist_li, future_li, future_flat
            gc.collect()
            torch.cuda.empty_cache()

        all_mi_matrices[mi_method] = mi_matrix

        plot_heatmap(output_dir, mi_matrix, args.e_layers, n_patches,
                     title_suffix=f"[{mi_method}] {args.model_id}",
                     filename=f"mi_heatmap_{mi_method}_{args.model_id}.png")

        print(f"\n  {mi_method} 汇总:")
        print(f"  {'Layer':>6} | {'MI Mean':>10} | {'MI Std':>10} | {'MI Max':>10}")
        print(f"  {'-'*45}")
        for li in range(args.e_layers):
            lm = mi_matrix[li]
            print(f"  {li:>6} | {np.mean(lm):>10.6f} | "
                  f"{np.std(lm):>10.6f} | {np.max(lm):>10.6f}")

    # 5 — Comparison plots
    print("\n>>> Phase 5: 绘制对比图...")
    if len(all_mi_matrices) >= 2:
        plot_mi_overlay(output_dir, all_mi_matrices,
                        args.e_layers, n_patches, args.model_id)
        plot_mi_per_layer_bar(output_dir, all_mi_matrices,
                               args.e_layers, args.model_id)

        methods = list(all_mi_matrices.keys())
        print("\n  方法间 Pearson 相关系数:")
        hdr = f"  {'':12}" + "".join(f" {m[:10]:>12}" for m in methods)
        print(hdr)
        for m1 in methods:
            row = f"  {m1[:12]:12}"
            for m2 in methods:
                corr = np.corrcoef(
                    all_mi_matrices[m1].flatten(),
                    all_mi_matrices[m2].flatten()
                )[0, 1]
                row += f" {corr:>12.4f}"
            print(row)

    # 6 — Save results
    print("\n>>> Phase 6: 保存结果...")
    for mi_method, mi_matrix in all_mi_matrices.items():
        layers_dict = {}
        for li in range(args.e_layers):
            lm = mi_matrix[li]
            si = np.argsort(lm)
            layers_dict[str(li)] = {
                "mi_curve": lm.tolist(),
                "q3": float(np.percentile(lm, 75)),
                "high_mi_patches": [int(i) for i in si[-2:]],
                "low_mi_patches": [int(i) for i in si[:2]],
                "mi_mean": float(lm.mean()),
                "mi_std": float(lm.std()),
                "mi_max": float(lm.max()),
                "mi_min": float(lm.min()),
            }

        all_high = set()
        all_low = set()
        for ld in layers_dict.values():
            all_high.update(ld["high_mi_patches"])
            all_low.update(ld["low_mi_patches"])

        result = {
            "model_id": args.model_id,
            "mi_method": mi_method,
            "n_vars": n_vars,
            "N": n_patches,
            "patch_len": args.patch_len,
            "pred_len": args.pred_len,
            "total_samples": sample_idx,
            "pca_dim": args.pca_dim,
            "k_neighbors": args.k_neighbors,
            "fastmi_grid_size": args.fastmi_grid_size,
            "num_layers": args.e_layers,
            "layers": layers_dict,
            "all_high_mi_patches": sorted(all_high),
            "all_low_mi_patches": sorted(all_low),
        }

        json_path = os.path.join(
            output_dir, f"global_mi_{mi_method}_{args.model_id}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  JSON: {json_path}")

        np.save(os.path.join(
            output_dir, f"mi_matrix_{mi_method}_{args.model_id}.npy"), mi_matrix)

    print(f"\n输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
