#!/usr/bin/env python3
"""
Layer-wise HSIC between history-patch hidden states h_x^l and a **prediction-side code** built from
the true future y (same normalization as Timer.forecast), before Timer's Linear proj head.

Intended setup (matches “历史序列拼上预测序列再喂进架构”):
  - Prefer **one forward** on concat(history, future) in time, so patch indices and positional
    embeddings are those of the **joint** sequence. We then split tokens into pure-history vs
    pure-future patches (no token may straddle the history|future time boundary).
  - If that boundary cuts through a patch (mixed patch), we **fallback** to two encodes
    (history-only, future-only) so shapes still match; positions on the future branch then restart
    at 0 (documented in sample summary).
  - For **each** history patch t: MI proxy = HSIC(h_x^l[:, t, :], z_y^l) where z_y^l is the
    **concatenation** of all future-patch token vectors along the feature dimension (no mean over
    patches). If n_y_future=1, z_y^l is just that single patch embedding.

MI proxy: HSIC over batch×vars at each (l, t), or per-sample HSIC/cos over vars.

Methodology aligned with arXiv:2506.02867 (MI via HSIC proxy):
  - Gaussian kernels K_X, K_Y on sample rows.
  - Centering H = I - (1/n) 11^T.
  - HSIC(X, Y) = (1/(n-1)^2) * tr(K_X H K_Y H).

Output layout:
  {run_dir}/aggregate/  — mean/std over batches, summary, overlay plots
  {run_dir}/samples/sample_XXXXXX/ — layer_XX_mi.png (x=input patch index t, y=MI),
    layer_patch_mi_heatmap.png, mi_all_layers_patches_summary.png (all layers on one axes),
    mi_proxy_matrix.npy.
    Default first 100 samples (--max_samples 100); use 0 for all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model as TimerModel


def _centering_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """H = I - (1/n) 11^T."""
    ones = torch.ones(n, n, device=device, dtype=dtype)
    eye = torch.eye(n, device=device, dtype=dtype)
    return eye - ones / n


def _median_sigma(samples: torch.Tensor) -> torch.Tensor:
    """Median heuristic for Gaussian kernel bandwidth (pdist on rows)."""
    n = samples.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=samples.device, dtype=samples.dtype)
    d = torch.pdist(samples)
    med = d.median()
    return med.clamp(min=torch.tensor(1e-8, device=samples.device, dtype=samples.dtype))


def hsic_gaussian_xy(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    HSIC with independent bandwidths for X and Y.
    x, y: [n, d_x], [n, d_y]; must have same n.
    """
    n = x.shape[0]
    if n < 4:
        return torch.tensor(float("nan"), device=x.device, dtype=torch.float32)

    xd = x.double()
    yd = y.double()
    sigma_x = _median_sigma(xd)
    sigma_y = _median_sigma(yd)

    dist_x = torch.cdist(xd, xd) ** 2
    dist_y = torch.cdist(yd, yd) ** 2
    kx = torch.exp(-dist_x / (2.0 * sigma_x * sigma_x))
    ky = torch.exp(-dist_y / (2.0 * sigma_y * sigma_y))

    h = _centering_matrix(n, xd.device, xd.dtype)
    val = torch.trace(kx @ h @ ky @ h) / ((n - 1) ** 2)
    return val.float()


def iqr_peak_mask(m: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Definition-style outliers: O = { t : m_t > Q3 + 1.5 * IQR }.
    Returns (boolean mask length N, threshold).
    """
    m_full = np.asarray(m, dtype=np.float64).ravel()
    if m_full.size == 0:
        return np.zeros(0, dtype=bool), float("nan")
    finite = m_full[np.isfinite(m_full)]
    if finite.size < 2:
        return np.zeros(m_full.size, dtype=bool), float("nan")
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = q3 - q1
    thresh = q3 + 1.5 * iqr
    mask = np.zeros(m_full.size, dtype=bool)
    mask[:] = np.where(np.isfinite(m_full), m_full > thresh, False)
    return mask, float(thresh)


def _num_patches(seq_len: int, patch_len: int, stride: int, pad_right: int = 0) -> int:
    lp = seq_len + pad_right
    if lp < patch_len:
        return 0
    return (lp - patch_len) // stride + 1


def _extract_y_future(batch_y: torch.Tensor, use_ims: bool, label_len: int, pred_len: int) -> torch.Tensor:
    """Future window [B, pred_len, M] in scaled data space."""
    if use_ims:
        return batch_y[:, label_len : label_len + pred_len, :].contiguous()
    return batch_y.contiguous()


def _normalize_like_timer(x_enc: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Apply same per-batch mean/std as Timer.forecast (from x_enc)."""
    means = x_enc.mean(1, keepdim=True)
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return (y - means) / stdev


def joint_pure_history_future_patch_indices(
    L_hist: int,
    L_tot: int,
    patch_len: int,
    stride: int,
    pad_right: int,
) -> tuple[list[int], list[int], bool]:
    """
    On a length-L_tot series (already concat of history|future), after right-pad pad_right,
    classify each patch index j as pure history, pure future, or mixed w.r.t. boundary L_hist.

    Pure history: patch covers only timesteps [0, L_hist).
    Pure future: patch covers only [L_hist, +inf) within data.
    """
    Lp = L_tot + pad_right
    if Lp < patch_len or L_hist < 0 or L_hist > L_tot:
        return [], [], True
    n_tot = (Lp - patch_len) // stride + 1
    hist_idx: list[int] = []
    fut_idx: list[int] = []
    has_mixed = False
    for j in range(n_tot):
        start = j * stride
        end = j * stride + patch_len
        if end <= L_hist:
            hist_idx.append(j)
        elif start >= L_hist:
            fut_idx.append(j)
        else:
            has_mixed = True
    return hist_idx, fut_idx, has_mixed


def prediction_side_code(hy_tokens: torch.Tensor) -> torch.Tensor:
    """
    Future / prediction segment: concatenate all patch token vectors along the feature axis.
    hy_tokens: [B*M, n_y, d] -> [B*M, n_y * d]. No averaging over patches.
    """
    return hy_tokens.reshape(hy_tokens.shape[0], -1)


def register_encoder_layer_hooks(model: TimerModel, storage: list) -> list:
    """Hooks on each EncoderLayer output tensor (before final Encoder norm)."""
    handles = []

    def _hook(_module, _inp, out):
        storage.append(out[0].detach())

    for layer in model.decoder.attn_layers:
        handles.append(layer.register_forward_hook(_hook))
    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


def compute_hsic_matrix_per_input_patch(
    hx: list[torch.Tensor],
    hy: list[torch.Tensor],
    n_layers: int,
    n_x: int,
) -> np.ndarray:
    """HSIC [L, n_x]: column t is HSIC(h_x[:, t, :], prediction_side_code(h_y tokens))."""
    hsic_batch = np.full((n_layers, n_x), np.nan, dtype=np.float64)
    for li in range(n_layers):
        hx_li = hx[li]
        hy_li = hy[li]
        hy_z = prediction_side_code(hy_li)
        for tx in range(n_x):
            if tx >= hx_li.shape[1]:
                continue
            xt = hx_li[:, tx, :]
            val = hsic_gaussian_xy(xt, hy_z)
            hsic_batch[li, tx] = float(val.item())
    return hsic_batch


def compute_per_sample_hsic_and_cos_per_input_patch(
    hx: list[torch.Tensor],
    hy: list[torch.Tensor],
    n_layers: int,
    n_x: int,
    b: int,
    n_vars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per input patch t: HSIC across variable rows (n_vars>=4) else nan; cosine vs concat future code.
    """
    hsic_ps = np.full((n_layers, n_x), np.nan, dtype=np.float64)
    cos_ps = np.full((n_layers, n_x), np.nan, dtype=np.float64)

    for li in range(n_layers):
        hx_li = hx[li]
        hy_li = hy[li]
        r0, r1 = b * n_vars, (b + 1) * n_vars
        hy_z = prediction_side_code(hy_li)
        yt = hy_z[r0:r1]
        for tx in range(n_x):
            if tx >= hx_li.shape[1]:
                continue
            xt = hx_li[r0:r1, tx, :]
            if n_vars >= 4:
                val = hsic_gaussian_xy(xt, yt)
                hsic_ps[li, tx] = float(val.item())
            vx = xt.mean(dim=0)
            vy = yt.mean(dim=0)
            c = F.cosine_similarity(vx.unsqueeze(0), vy.unsqueeze(0), dim=-1)
            cos_ps[li, tx] = float(c.item())
    return hsic_ps, cos_ps


def style_patch_axis(ax, n_patches: int) -> None:
    """Avoid degenerate auto x-scale (e.g. single point -> [-0.04, 0.04])."""
    if n_patches <= 0:
        return
    ax.set_xlim(-0.5, (n_patches - 1) + 0.5)
    if n_patches <= 48:
        ax.set_xticks(np.arange(n_patches))


def combined_mi_matrix(hsic_ps: np.ndarray, cos_ps: np.ndarray) -> np.ndarray:
    """Per cell: HSIC if finite, else cosine (same as single-sample MI curve)."""
    return np.where(np.isfinite(hsic_ps), hsic_ps, cos_ps)


def plot_sample_mi_per_layer_and_heatmap(
    sdir: str,
    mi_mat: np.ndarray,
    hsic_ps: np.ndarray,
    n_layers: int,
    n_patches: int,
    global_sample_index: int,
    n_vars: int,
) -> None:
    """
    For one sample: one PNG per layer (x=patch id, y=MI), heatmap [layer × patch], and one overlay
    figure with every layer's MI-vs-patch curve on the same axes (mi_all_layers_patches_summary.png).
    """
    patch_axis = np.arange(n_patches, dtype=np.float64)

    for li in range(n_layers):
        y = mi_mat[li]
        row_hsic = hsic_ps[li]
        layer_mode = "HSIC" if np.any(np.isfinite(row_hsic)) else "cosine"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(patch_axis, y, marker="o", markersize=6, linewidth=1.5, color="C0")
        ax.set_xlabel("Input patch index t (vs concat of future-patch tokens as z_y)")
        ax.set_ylabel("MI proxy")
        ax.set_title(f"Sample {global_sample_index} | Layer {li} ({layer_mode})")
        style_patch_axis(ax, n_patches)
        ax.grid(True, alpha=0.3)
        mask, thr = iqr_peak_mask(y)
        if np.any(mask) and np.isfinite(thr):
            ax.axhline(thr, color="red", linestyle="--", linewidth=1, alpha=0.85)
            ax.scatter(patch_axis[mask], y[mask], color="red", zorder=5, s=45, label="IQR peak")
            ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(sdir, f"layer_{li:02d}_mi.png"), dpi=150)
        plt.close(fig)

    fig_hm, ax_hm = plt.subplots(
        figsize=(max(6.0, 0.45 * n_patches + 2), max(3.5, 0.42 * n_layers + 1.5))
    )
    im = ax_hm.imshow(mi_mat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax_hm.set_xlabel("Input patch index t")
    ax_hm.set_ylabel("Layer")
    ax_hm.set_xticks(np.arange(n_patches))
    ax_hm.set_yticks(np.arange(n_layers))
    ax_hm.set_yticklabels([f"L{j}" for j in range(n_layers)])
    cbar = fig_hm.colorbar(im, ax=ax_hm)
    cbar.set_label("MI proxy (HSIC or cos if HSIC n/a)")
    ax_hm.set_title(
        f"Sample {global_sample_index}: layer × input-patch MI (n_vars={n_vars})",
        fontsize=11,
    )
    fig_hm.tight_layout()
    fig_hm.savefig(os.path.join(sdir, "layer_patch_mi_heatmap.png"), dpi=150)
    plt.close(fig_hm)

    # One summary figure: every layer's MI vs patch index on the same axes.
    fig_all, ax_all = plt.subplots(figsize=(max(8.0, 0.38 * n_patches + 3.0), 5.0))
    for li in range(n_layers):
        row_hsic = hsic_ps[li]
        linestyle = "-" if np.any(np.isfinite(row_hsic)) else ":"
        ax_all.plot(
            patch_axis,
            mi_mat[li],
            label=f"L{li}",
            alpha=0.88,
            linestyle=linestyle,
            marker="o" if n_patches <= 24 else None,
            markersize=4,
        )
    ax_all.set_xlabel("Input patch index t")
    ax_all.set_ylabel("MI proxy")
    ax_all.set_title(
        f"Sample {global_sample_index}: all layers × all patches (n_vars={n_vars})"
    )
    style_patch_axis(ax_all, n_patches)
    ax_all.grid(True, alpha=0.3)
    ax_all.legend(loc="best", ncol=2, fontsize=8)
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(sdir, "mi_all_layers_patches_summary.png"), dpi=150)
    plt.close(fig_all)


def plot_layer_curves(
    out_path: str,
    n_patches: int,
    n_layers: int,
    primary: np.ndarray,
    primary_label: str,
    std: np.ndarray | None,
    title_prefix: str,
    secondary: np.ndarray | None = None,
    secondary_label: str | None = None,
    overlay: bool = False,
) -> None:
    """primary, std, secondary: [L, N]."""
    patch_axis = np.arange(n_patches, dtype=np.float64)
    if overlay:
        fig, ax = plt.subplots(figsize=(10, 5))
        for li in range(n_layers):
            ax.plot(
                patch_axis,
                primary[li],
                label=f"L{li}",
                alpha=0.85,
                marker="o" if n_patches <= 24 else None,
                markersize=4,
            )
        ax.set_xlabel("Patch index t")
        ax.set_ylabel(primary_label)
        ax.set_title(f"All layers: {title_prefix}")
        ax.legend(loc="best", ncol=2, fontsize=7)
        style_patch_axis(ax, n_patches)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    for li in range(n_layers):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            patch_axis,
            primary[li],
            label=primary_label,
            color="C0",
            marker="o" if n_patches <= 24 else None,
            markersize=4,
        )
        if std is not None and np.any(np.isfinite(std[li])):
            ax.fill_between(
                patch_axis,
                primary[li] - std[li],
                primary[li] + std[li],
                alpha=0.25,
                color="C0",
            )
        if secondary is not None and np.any(np.isfinite(secondary[li])):
            ax.plot(
                patch_axis,
                secondary[li],
                label=secondary_label or "cos align",
                color="C1",
                marker="s" if n_patches <= 24 else None,
                markersize=3,
                alpha=0.85,
            )
        mask, thr = iqr_peak_mask(primary[li])
        if np.any(mask) and np.isfinite(thr):
            ax.axhline(thr, color="red", linestyle="--", linewidth=1, label=f"IQR thr={thr:.4g}")
            ax.scatter(patch_axis[mask], np.asarray(primary[li])[mask], color="red", zorder=5, label="peaks")
        ax.set_xlabel("Patch index t")
        ax.set_ylabel(primary_label)
        ax.set_title(f"Layer {li}: {title_prefix}")
        ax.legend(loc="best", fontsize=8)
        style_patch_axis(ax, n_patches)
        fig.tight_layout()
        fig.savefig(out_path.replace("layer_XX", f"layer_{li:02d}"), dpi=150)
        plt.close(fig)


def build_args_ns(p: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="forecast",
        ckpt_path=p.ckpt_path,
        patch_len=p.patch_len,
        seq_len=p.seq_len,
        label_len=p.label_len,
        pred_len=p.pred_len,
        output_len=p.output_len,
        d_model=p.d_model,
        d_ff=p.d_ff,
        e_layers=p.e_layers,
        n_heads=p.n_heads,
        dropout=p.dropout,
        factor=p.factor,
        activation=p.activation,
        output_attention=False,
        features=p.features,
        root_path=p.root_path,
        data_path=p.data_path,
        data=p.data,
        embed=p.embed,
        freq=p.freq,
        stride=p.stride,
        subset_rand_ratio=p.subset_rand_ratio,
        use_ims=p.use_ims,
        batch_size=p.batch_size,
        num_workers=p.num_workers,
        use_multi_gpu=False,
        use_gpu=torch.cuda.is_available() and not getattr(p, "cpu", False),
        gpu=p.gpu,
        inverse=False,
        periodic_embedding_branch=getattr(p, "periodic_embedding_branch", 0),
        periodic_emb_bank_dim=getattr(p, "periodic_emb_bank_dim", 0),
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Layer-wise HSIC (MI proxy) for Timer on test set")
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--root_path", type=str, default="./datasets/ETT-small/")
    ap.add_argument("--data_path", type=str, default="ETTh1.csv")
    ap.add_argument("--data", type=str, default="ETTh1")
    ap.add_argument("--features", type=str, default="M")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--output_len", type=int, default=96)
    ap.add_argument("--use_ims", action="store_true")
    ap.add_argument("--patch_len", type=int, default=96)
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--e_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--embed", type=str, default="timeF")
    ap.add_argument("--freq", type=str, default="h")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--subset_rand_ratio", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for HSIC (n = B * M_eff)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=100)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max per-sample folders with plots (default 100). Use 0 for all test samples.",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="Run on CPU even if CUDA is available")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/mi_hsic_timer",
        help="Base directory; a subfolder with run id and timestamp is created",
    )
    ap.add_argument("--run_id", type=int, default=1, help="Run index embedded in output folder name")
    ap.add_argument("--periodic_embedding_branch", type=int, default=0)
    ap.add_argument("--periodic_emb_bank_dim", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    cfg = build_args_ns(args)
    _, loader = data_provider(cfg, flag="test")

    try:
        model = TimerModel(cfg).float().to(device)
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch" in msg and "patch_embedding" in msg:
            raise RuntimeError(
                "Checkpoint shape mismatch: almost always --patch_len (and proj output dim) must match "
                "the checkpoint (e.g. Timer_forecast_1.0.ckpt uses patch_len=96, not 1). "
                "Align --seq_len/--label_len/--pred_len/--output_len/--use_ims with training. "
                "See scripts/mi_hsic_layerwise_timer.sh defaults."
            ) from e
        raise
    model.eval()

    pad_r = model.enc_embedding.padding_patch_layer.padding
    pad_right = pad_r[-1] if isinstance(pad_r, tuple) and len(pad_r) >= 2 else 0
    patch_len = model.enc_embedding.patch_len
    stride = model.enc_embedding.stride

    n_layers = len(model.decoder.attn_layers)
    pred_horizon = args.output_len if args.use_ims else args.pred_len

    sum_hsic = None
    sum_sq = None
    n_batches_used = 0
    n_patches_ref = None

    samples_root = None
    aggregate_dir = None
    global_sample_idx = 0
    max_samples = int(args.max_samples) if args.max_samples > 0 else None
    # Each entry [n_layers, n_patches] — used for aggregate per-layer sample summary plots.
    recorded_sample_mi_mats: list[np.ndarray] = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        args.output_dir,
        f"{args.data}_run{args.run_id:03d}_{ts}",
    )
    aggregate_dir = os.path.join(run_dir, "aggregate")
    samples_root = os.path.join(run_dir, "samples")
    os.makedirs(aggregate_dir, exist_ok=True)
    os.makedirs(samples_root, exist_ok=True)

    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break

        batch_x, batch_y, batch_x_mark, _batch_y_mark = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        b, seq_len, _m = batch_x.shape
        y_future = _extract_y_future(batch_y, args.use_ims, args.label_len, pred_horizon)
        L_fut = y_future.shape[1]
        L_tot = seq_len + L_fut

        n_x_standalone = _num_patches(seq_len, patch_len, stride, pad_right)
        n_y_standalone = _num_patches(L_fut, patch_len, stride, pad_right)
        if n_x_standalone <= 0 or n_y_standalone <= 0:
            continue

        # Golden y: only observed future timesteps (e.g. 96), same norm as Timer.forecast.
        y_norm = _normalize_like_timer(batch_x, y_future)

        hx: list[torch.Tensor] = []
        hy: list[torch.Tensor] = []
        forward_mode = "split_encode"
        n_x = n_x_standalone
        n_y_tokens = n_y_standalone

        with torch.no_grad():
            means = batch_x.mean(1, keepdim=True)
            stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_n = (batch_x - means) / stdev
            x_perm = x_n.permute(0, 2, 1)

            hist_idx, fut_idx, has_mixed = joint_pure_history_future_patch_indices(
                seq_len, L_tot, patch_len, stride, pad_right
            )

            if (
                not has_mixed
                and hist_idx
                and fut_idx
            ):
                xy_n = torch.cat([x_n, y_norm], dim=1)
                xy_perm = xy_n.permute(0, 2, 1)
                h_all: list[torch.Tensor] = []
                dec_xy, n_vars = model.enc_embedding(xy_perm)
                if dec_xy.shape[0] < 4:
                    continue
                handles_j = register_encoder_layer_hooks(model, h_all)
                _ = model.decoder(dec_xy)[0]
                remove_hooks(handles_j)
                if len(h_all) != n_layers:
                    continue
                hi = torch.tensor(hist_idx, device=h_all[0].device, dtype=torch.long)
                fi = torch.tensor(fut_idx, device=h_all[0].device, dtype=torch.long)
                hx = [h.index_select(1, hi) for h in h_all]
                hy = [h.index_select(1, fi) for h in h_all]
                forward_mode = "joint_concat"
                n_x = len(hist_idx)
                n_y_tokens = len(fut_idx)
            else:
                y_perm = y_norm.permute(0, 2, 1)
                dec_in, n_vars = model.enc_embedding(x_perm)
                dec_y, n_vars_y = model.enc_embedding(y_perm)
                if dec_y.shape[0] != dec_in.shape[0] or n_vars_y != n_vars:
                    continue
                if dec_in.shape[0] < 4:
                    continue

                handles_x = register_encoder_layer_hooks(model, hx)
                _ = model.decoder(dec_in)[0]
                remove_hooks(handles_x)

                handles_y = register_encoder_layer_hooks(model, hy)
                _ = model.decoder(dec_y)[0]
                remove_hooks(handles_y)

                if has_mixed:
                    forward_mode = "split_encode_mixed_boundary_fallback"
                n_x = n_x_standalone
                n_y_tokens = n_y_standalone

        if len(hx) != n_layers or len(hy) != n_layers:
            continue

        n_hx = hx[0].shape[1]
        n_hy = hy[0].shape[1]
        if n_x > n_hx or n_y_tokens > n_hy:
            continue

        hsic_batch = compute_hsic_matrix_per_input_patch(hx, hy, n_layers, n_x)
        if np.all(np.isnan(hsic_batch)):
            continue

        if sum_hsic is None:
            sum_hsic = np.nan_to_num(hsic_batch, nan=0.0)
            sum_sq = np.nan_to_num(hsic_batch ** 2, nan=0.0)
            n_patches_ref = n_x
        elif n_x == n_patches_ref:
            sum_hsic += np.nan_to_num(hsic_batch, nan=0.0)
            sum_sq += np.nan_to_num(hsic_batch ** 2, nan=0.0)
        else:
            continue

        n_batches_used += 1

        # Per-sample directories (each batch element -> one folder)
        for sb in range(b):
            if max_samples is not None and global_sample_idx >= max_samples:
                break
            hsic_ps, cos_ps = compute_per_sample_hsic_and_cos_per_input_patch(
                hx, hy, n_layers, n_x, sb, n_vars
            )
            sdir = os.path.join(samples_root, f"sample_{global_sample_idx:06d}")
            os.makedirs(sdir, exist_ok=True)
            np.save(os.path.join(sdir, "hsic_per_patch.npy"), hsic_ps)
            np.save(os.path.join(sdir, "cos_align_per_patch.npy"), cos_ps)

            sample_summary = {
                "global_sample_index": global_sample_idx,
                "batch_index": bi,
                "in_batch_index": sb,
                "forward_mode": forward_mode,
                "n_input_patches": int(n_x),
                "n_future_patch_tokens": int(n_y_tokens),
                "prediction_side_z_y": "concatenate future patch token dims (no mean); HSIC(h_x[t], z_y)",
                "n_vars": int(n_vars),
                "hsic_valid": bool(n_vars >= 4),
                "note": "y = true future; joint_concat = one encode on cat(history,future) with clean patch boundary.",
            }
            peaks_layers = []
            for li in range(n_layers):
                row = hsic_ps[li]
                mask, thr = iqr_peak_mask(row)
                idx = np.where(mask)[0].tolist()
                strengths = row[mask].tolist() if len(idx) else []
                peaks_layers.append(
                    {
                        "layer": li,
                        "iqr_threshold": thr,
                        "peak_indices": idx,
                        "peak_strength_max": float(np.max(strengths)) if strengths else None,
                    }
                )
            sample_summary["per_layer_peaks_hsic"] = peaks_layers
            with open(os.path.join(sdir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(sample_summary, f, indent=2)

            mi_mat = combined_mi_matrix(hsic_ps, cos_ps)
            np.save(os.path.join(sdir, "mi_proxy_matrix.npy"), mi_mat)
            plot_sample_mi_per_layer_and_heatmap(
                sdir, mi_mat, hsic_ps, n_layers, n_x, global_sample_idx, n_vars
            )
            recorded_sample_mi_mats.append(mi_mat.copy())
            global_sample_idx += 1

    if n_batches_used == 0 or sum_hsic is None:
        raise RuntimeError("No valid batches; check shapes, patch_len vs seq_len/pred_len, or batch_size.")

    mean_hsic = sum_hsic / n_batches_used
    var = sum_sq / n_batches_used - mean_hsic ** 2
    std_hsic = np.sqrt(np.maximum(var, 0.0))

    np.save(os.path.join(aggregate_dir, "hsic_mean.npy"), mean_hsic)
    np.save(os.path.join(aggregate_dir, "hsic_std.npy"), std_hsic)

    peaks_report = []
    for li in range(n_layers):
        row = mean_hsic[li]
        mask, thr = iqr_peak_mask(row)
        idx = np.where(mask)[0].tolist()
        strengths = row[mask].tolist() if len(idx) else []
        peaks_report.append(
            {
                "layer": li,
                "mi_proxy_mean_over_patches": float(np.mean(row)),
                "mi_proxy_std_over_patches": float(np.std(row)),
                "iqr_threshold": thr,
                "peak_indices": idx,
                "peak_strength_max": float(np.max(strengths)) if strengths else None,
                "peak_strength_mean": float(np.mean(strengths)) if strengths else None,
            }
        )

    meta = {
        "arxiv_ref": "2506.02867",
        "hsic_formula": "tr(Kx H Ky H) / (n-1)^2",
        "kernel": "gaussian_median_heuristic",
        "n_batches": n_batches_used,
        "n_samples_written": global_sample_idx,
        "n_layers": n_layers,
        "n_input_patches_mi": int(n_patches_ref),
        "pairing": "For each history patch t: HSIC(h_x^l[t], z_y^l) with z_y = concat of future patch tokens (no mean)",
        "batch_size": args.batch_size,
        "patch_len": patch_len,
        "stride": stride,
        "seq_len": args.seq_len,
        "pred_horizon": int(pred_horizon),
        "use_ims": bool(args.use_ims),
        "run_id": args.run_id,
        "ckpt_path": args.ckpt_path,
        "aggregate_dir": aggregate_dir,
        "samples_dir": samples_root,
    }

    with open(os.path.join(aggregate_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "per_layer": peaks_report}, f, indent=2)

    npr = int(n_patches_ref)
    patch_axis = np.arange(npr, dtype=np.float64)

    for li in range(n_layers):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            patch_axis,
            mean_hsic[li],
            label="HSIC mean",
            color="C0",
            marker="o" if npr <= 24 else None,
            markersize=4,
        )
        ax.fill_between(
            patch_axis,
            mean_hsic[li] - std_hsic[li],
            mean_hsic[li] + std_hsic[li],
            alpha=0.25,
            color="C0",
        )
        mask, thr = iqr_peak_mask(mean_hsic[li])
        if np.any(mask) and np.isfinite(thr):
            ax.axhline(thr, color="red", linestyle="--", linewidth=1, label=f"IQR thr={thr:.4g}")
            ax.scatter(patch_axis[mask], mean_hsic[li][mask], color="red", zorder=5, label="peaks")
        ax.set_xlabel("Input patch index t")
        ax.set_ylabel("HSIC (MI proxy)")
        ax.set_title(f"Layer {li}: HSIC(h_x^l[t], z_y^l) aggregate (z_y = concat future tokens)")
        ax.legend(loc="best", fontsize=8)
        style_patch_axis(ax, npr)
        fig.tight_layout()
        fig.savefig(os.path.join(aggregate_dir, f"layer_{li:02d}_hsic.png"), dpi=150)
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for li in range(n_layers):
        ax2.plot(
            patch_axis,
            mean_hsic[li],
            label=f"L{li}",
            alpha=0.85,
            marker="o" if npr <= 24 else None,
            markersize=4,
        )
    ax2.set_xlabel("Input patch index t")
    ax2.set_ylabel("HSIC (MI proxy)")
    ax2.set_title("All layers: mean HSIC vs input patch t (Timer, aggregate)")
    ax2.legend(loc="best", ncol=2, fontsize=7)
    style_patch_axis(ax2, npr)
    fig2.tight_layout()
    fig2.savefig(os.path.join(aggregate_dir, "all_layers_hsic_overlay.png"), dpi=150)
    plt.close(fig2)

    # Per-layer summary: all recorded samples on one figure (MI vs patch id), + sample×patch heatmap.
    if recorded_sample_mi_mats:
        n_s = len(recorded_sample_mi_mats)
        npr_s = recorded_sample_mi_mats[0].shape[1]
        patch_ax_s = np.arange(npr_s, dtype=np.float64)
        # Avoid unreadable legends when many samples are saved.
        legend_max_samples = 24
        use_sample_legend = n_s <= legend_max_samples
        for li in range(n_layers):
            fig_s, ax_s = plt.subplots(figsize=(9, 4.8))
            for k, mm in enumerate(recorded_sample_mi_mats):
                if mm.shape[1] != npr_s:
                    continue
                ax_s.plot(
                    patch_ax_s,
                    mm[li],
                    alpha=0.82 if n_s <= 40 else max(0.12, 0.85 - 0.006 * n_s),
                    label=(f"sample_{k:02d}" if use_sample_legend else None),
                    marker="o" if npr_s <= 16 and n_s <= 20 else None,
                    markersize=3,
                )
            n_mean = min(npr_s, mean_hsic.shape[1])
            if n_mean > 0:
                ax_s.plot(
                    np.arange(n_mean, dtype=np.float64),
                    mean_hsic[li, :n_mean],
                    color="black",
                    linewidth=2.2,
                    linestyle="--",
                    label="batch mean HSIC",
                    zorder=10,
                )
            ax_s.set_xlabel("Input patch index t")
            ax_s.set_ylabel("MI proxy")
            title_extra = f"{n_s} samples" + ("" if use_sample_legend else f", legend off (>{legend_max_samples})")
            ax_s.set_title(
                f"Layer {li}: per-sample MI vs t; dashed=batch mean | {title_extra}"
            )
            style_patch_axis(ax_s, npr_s)
            ax_s.grid(True, alpha=0.3)
            if use_sample_legend or n_mean > 0:
                ax_s.legend(ncol=2, fontsize=7, loc="best")
            fig_s.tight_layout()
            fig_s.savefig(
                os.path.join(aggregate_dir, f"layer_{li:02d}_mi_all_samples.png"),
                dpi=150,
            )
            plt.close(fig_s)

            stack = np.stack([m[li] for m in recorded_sample_mi_mats if m.shape[1] == npr_s], axis=0)
            if stack.size > 0:
                fig_h, ax_h = plt.subplots(
                    figsize=(max(6.0, 0.35 * npr_s + 2), max(3.0, 0.35 * stack.shape[0] + 1.5))
                )
                imh = ax_h.imshow(stack, aspect="auto", cmap="viridis", interpolation="nearest")
                ax_h.set_xlabel("Input patch index t")
                ax_h.set_ylabel("Sample index")
                ax_h.set_xticks(np.arange(npr_s))
                ax_h.set_yticks(np.arange(stack.shape[0]))
                ax_h.set_yticklabels([f"{j}" for j in range(stack.shape[0])])
                fig_h.colorbar(imh, ax=ax_h, label="MI proxy")
                ax_h.set_title(f"Layer {li}: MI proxy (sample × input patch t)")
                fig_h.tight_layout()
                fig_h.savefig(
                    os.path.join(aggregate_dir, f"layer_{li:02d}_mi_sample_x_patch.png"),
                    dpi=150,
                )
                plt.close(fig_h)

    with open(os.path.join(run_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "aggregate/ (population-level, over all batches that passed filters):\n"
            "  hsic_mean.npy, hsic_std.npy, summary.json — batch-mean/std HSIC per (layer, input patch t).\n"
            "  layer_LL_hsic.png — for layer L: mean HSIC vs t, shaded band = batch std; red = IQR peak line.\n"
            "  all_layers_hsic_overlay.png — every layer's batch-mean HSIC vs t on one axes.\n"
            "  layer_LL_mi_all_samples.png — thin curves = each saved sample's MI vs t; black dashed = batch mean.\n"
            "  layer_LL_mi_sample_x_patch.png — heatmap rows=samples, cols=t, for that layer.\n"
            "samples/sample_XXXXXX/: layer_LL_mi.png, layer_patch_mi_heatmap.png, "
            "mi_all_layers_patches_summary.png, npy.\n"
            "Prefer joint encode on cat(history,true future); else split encode if patch crosses boundary.\n"
            "z_y = concat of future patch token vectors (no mean); HSIC(h_x[t], z_y) per layer.\n"
            "Default --max_samples=100; use --max_samples 0 for every test sample.\n"
        )

    print(f"Wrote aggregate -> {aggregate_dir}")
    print(f"Wrote {global_sample_idx} samples under {samples_root}")


if __name__ == "__main__":
    main()
