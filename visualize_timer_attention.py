#!/usr/bin/env python3
"""
Extract self-attention from Timer (patch tokens) and save per-layer, per-head heatmaps.

Run from repo root:
  python visualize_timer_attention.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \\
    --root_path ./datasets/ETT-small/ --data_path ETTh1.csv --data ETTh1

Or: ./visualize_timer_attention.sh

Use synthetic sine (no CSV): add --synthetic_sin; match --seq_len/--label_len/--pred_len/--patch_len/--sin_n_vars to your checkpoint.

Each run writes only under output_dir/<run_id>/ (a dedicated subfolder). Auto run_id includes microseconds; repeated
--run_id reuses name only if that folder is missing, otherwise gets run_id_002, run_id_003, ... See runs_index.txt.

Attention tensor shape per layer: [B_eff, n_heads, N, N] where B_eff = batch * n_vars after patching.
Rows = query token index, columns = key token index (attention weight).

Real data: flag=test with the same root_path/data_path/data/seq_len/label_len/pred_len/output_len/use_ims as run.py.
Optional --test_metrics_json points to JSON from exp.test (set env FORECAST_TEST_METRICS_JSON in ETTh12.sh) to print run.py MSE/MAE on figures only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch

# Repo root on sys.path (this file lives at repository root)
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize, PowerNorm

from data_provider.data_factory import data_provider
from models.Timer import Model as TimerModel

# Cividis: blue→yellow. Default linear_quantile = quantile vmax + PowerNorm (low weights → brighter stripes).
DEFAULT_ATTN_CMAP = "cividis"


def _norm_for_attn(
    attn_array: np.ndarray,
    gamma: float,
    vmax_quantile: float,
    discrete_levels: int,
    scale_mode: str,
):
    """
    Color normalization for attention heatmaps.

    linear01: Map [0, 1] linearly (softmax weights). Matches common paper figures; diagonal peaks pop.
    linear_quantile: vmax = quantile(positive, vmax_quantile), then PowerNorm(gamma) on [0,vmax] to
        exaggerate low weights (stronger stripes than plain linear).
    compressed: same vmax rule as linear_quantile but slightly different small-sample vmax handling.

    discrete_levels: optional BoundaryNorm; linear01 uses evenly spaced [0,1] bin edges;
    linear_quantile/compressed use power-spaced or even bin edges on [0, vmax_plot].
    """
    if scale_mode == "linear01":
        vmax_plot = 1.0
        if discrete_levels >= 2:
            boundaries = np.linspace(0.0, 1.0, int(discrete_levels) + 1)
            boundaries = boundaries.copy()
            boundaries[-1] = 1.0 + 1e-9
            norm = BoundaryNorm(boundaries, 256, clip=True)
        else:
            norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
        return norm, vmax_plot

    if scale_mode == "linear_quantile":
        attn_array = np.asarray(attn_array, dtype=np.float64)
        attn_array = np.nan_to_num(attn_array, nan=0.0, posinf=0.0, neginf=0.0)
        flat = attn_array.ravel()
        pos = flat[flat > 1e-12]
        q = float(np.clip(vmax_quantile, 0.5, 1.0))
        if pos.size >= 4:
            hi = float(np.quantile(pos, q))
            vmax_plot = max(hi, 1e-10)
        else:
            vmax_plot = max(float(flat.max()), 1e-10)
        vmax_plot = float(min(vmax_plot, 1.0))
        g = max(float(gamma), 0.06)
        if discrete_levels >= 2:
            t = np.linspace(0.0, 1.0, int(discrete_levels) + 1)
            boundaries = vmax_plot * np.power(t, 1.0 / g)
            boundaries = np.unique(np.clip(boundaries, 0.0, vmax_plot))
            if boundaries.size < 3:
                norm = PowerNorm(gamma=g, vmin=0.0, vmax=vmax_plot)
            else:
                boundaries = boundaries.copy()
                boundaries[-1] = boundaries[-1] * (1.0 + 1e-6)
                norm = BoundaryNorm(boundaries, 256, clip=True)
        else:
            norm = PowerNorm(gamma=g, vmin=0.0, vmax=vmax_plot)
        return norm, vmax_plot

    # --- compressed ---
    attn_array = np.asarray(attn_array, dtype=np.float64)
    attn_array = np.nan_to_num(attn_array, nan=0.0, posinf=0.0, neginf=0.0)
    flat = attn_array.ravel()
    pos = flat[flat > 1e-10]
    q = float(np.clip(vmax_quantile, 0.5, 1.0))
    if pos.size >= 16:
        hi = float(np.quantile(pos, q))
        vmax = float(np.clip(float(pos.max()), 1e-8, 1.0))
    else:
        vmax = float(np.clip(float(flat.max()), 1e-8, 1.0))
        hi = float(np.quantile(flat, q))
    vmax_plot = max(min(vmax, hi), 1e-8) if hi > 0 else vmax

    g = max(float(gamma), 0.06)
    if discrete_levels >= 2:
        t = np.linspace(0.0, 1.0, int(discrete_levels) + 1)
        boundaries = vmax_plot * np.power(t, 1.0 / g)
        boundaries = np.unique(np.clip(boundaries, 0.0, vmax_plot))
        if boundaries.size < 3:
            norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax_plot)
        else:
            boundaries = boundaries.copy()
            boundaries[-1] = boundaries[-1] * (1.0 + 1e-6)
            norm = BoundaryNorm(boundaries, 256, clip=True)
    else:
        norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax_plot)
    return norm, vmax_plot


def _allocate_run_subdirectory(base_output_dir: str, preferred_run_id: str) -> tuple[str, str]:
    """
    Reserve a unique leaf directory under base_output_dir.
    If preferred_run_id is unused, use it; else use preferred_run_id_002, _003, ...
    """
    candidate = os.path.join(base_output_dir, preferred_run_id)
    if not os.path.exists(candidate):
        return candidate, preferred_run_id
    n = 2
    while n < 100000:
        rid = f"{preferred_run_id}_{n:03d}"
        candidate = os.path.join(base_output_dir, rid)
        if not os.path.exists(candidate):
            return candidate, rid
        n += 1
    raise RuntimeError("Could not find a free run subdirectory name")


def _load_test_metrics_line(json_path: str, output_len: int) -> str | None:
    """
    Read MSE/MAE written by exp_forecast.test when FORECAST_TEST_METRICS_JSON is set.
    Picks the record whose output_len matches args.output_len, else the last entry.
    """
    if not json_path or not str(json_path).strip():
        return None
    path = os.path.abspath(os.path.expanduser(json_path.strip()))
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        return None
    chosen = None
    for r in records:
        if int(r.get("output_len", -1)) == int(output_len):
            chosen = r
            break
    if chosen is None and records:
        chosen = records[-1]
    if not chosen:
        return None
    return (
        f"Test MSE={float(chosen['mse']):.6f}  MAE={float(chosen['mae']):.6f}  "
        f"(output_len={int(chosen['output_len'])}, from run.py test)"
    )


def _integer_axis_ticks(ax, n_tokens: int, *, labelsize: float | None = None) -> None:
    """Tick every index 0 .. n_tokens-1 (step 1) for reading stripe alignment."""
    ticks = np.arange(n_tokens)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=labelsize)


def parse_args():
    p = argparse.ArgumentParser(description="Timer attention heatmaps (all layers, all heads)")
    p.set_defaults(use_ims=True)
    p.add_argument("--ckpt_path", type=str, required=True, help="Timer checkpoint (.ckpt or .pth)")
    p.add_argument("--root_path", type=str, default="./datasets/ETT-small/", help="Data directory")
    p.add_argument("--data_path", type=str, default="ETTh1.csv", help="CSV file name under root_path")
    p.add_argument("--data", type=str, default="ETTh1", help="Dataset type tag (ETTh1, etc.)")
    p.add_argument("--task_name", type=str, default="forecast")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--label_len", type=int, default=576)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--output_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=96)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--subset_rand_ratio", type=float, default=1.0)
    p.add_argument(
        "--no_use_ims",
        action="store_false",
        dest="use_ims",
        help="Disable IMS layout (use CIDatasetBenchmark instead of CIAutoRegressionDatasetBenchmark)",
    )
    p.add_argument("--output_dir", type=str, default="./attention_maps_timer", help="Base dir; each run uses a subfolder unless --flat_output")
    p.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Experiment label for this run (folder name / file prefix). Default: run_YYYYMMDD_HHMMSS",
    )
    p.add_argument(
        "--flat_output",
        action="store_true",
        help="Write into output_dir directly with {run_id}_ prefixed filenames (no run subfolder)",
    )
    p.add_argument(
        "--reuse_exact_run_id",
        action="store_true",
        help="If the run subfolder already exists, write there anyway (overwrites). Default is to pick a new suffix folder.",
    )
    p.add_argument("--batch_index", type=int, default=0, help="Which batch from the test loader")
    p.add_argument("--batch_eff_index", type=int, default=0, help="Index along B_eff after patch embed (usually 0)")
    p.add_argument(
        "--gamma",
        type=float,
        default=0.10,
        help="PowerNorm gamma for linear_quantile/compressed (smaller => more stretch at low weights / sharper stripes)",
    )
    p.add_argument(
        "--vmax_quantile",
        type=float,
        default=0.68,
        help="linear_quantile & compressed: lower => smaller vmax => stronger contrast (try 0.55–0.75)",
    )
    p.add_argument(
        "--attn_scale",
        type=str,
        choices=("linear01", "linear_quantile", "compressed"),
        default="linear_quantile",
        help="linear_quantile: quantile vmax + PowerNorm (default, loud stripes). linear01: [0,1]. compressed: alt vmax rule",
    )
    p.add_argument(
        "--attn_levels",
        type=int,
        default=0,
        help="If >=2, discrete bands; linear01 uses even bins on [0,1]; compressed uses power-spaced bins; 0=continuous",
    )
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--attn_cmap",
        type=str,
        default=DEFAULT_ATTN_CMAP,
        help="Matplotlib colormap (default cividis; try viridis / inferno / turbo)",
    )
    p.add_argument(
        "--synthetic_sin",
        action="store_true",
        help="Do not load CSV: build batch_x / batch_y as multivariate sinusoids (IMS shapes).",
    )
    p.add_argument(
        "--sin_n_vars",
        type=int,
        default=1,
        help="Channels M in [B, L, M]; must match the trained model (often 1 for benchmark-style batches).",
    )
    p.add_argument("--sin_batch_size", type=int, default=1, help="Batch size B for synthetic input")
    p.add_argument(
        "--sin_base_period",
        type=float,
        default=1.0,
        help="Scales normalized time in sin() (higher => more cycles across seq_len).",
    )
    p.add_argument(
        "--sin_noise_std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise on top of sine (0 = pure).",
    )
    p.add_argument(
        "--sin_seed",
        type=int,
        default=None,
        help="torch.manual_seed when sin_noise_std > 0",
    )
    p.add_argument(
        "--test_metrics_json",
        type=str,
        default="",
        help="JSON from exp test (env FORECAST_TEST_METRICS_JSON); show those MSE/MAE on figures only",
    )
    p.add_argument(
        "--periodic_embedding_branch",
        type=int,
        default=0,
        help="1 = match training with hour/day embedding residual before proj (needs matching ckpt for proj/embed)",
    )
    p.add_argument(
        "--periodic_emb_bank_dim",
        type=int,
        default=0,
        help="Must match training when periodic branch used bank_dim→d_model Linear",
    )
    return p.parse_args()


def build_config_ns(args: argparse.Namespace):
    """Namespace with all fields TimerModel and data_provider expect."""
    from types import SimpleNamespace

    ns = SimpleNamespace(
        task_name=args.task_name,
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        output_len=args.output_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        factor=args.factor,
        activation=args.activation,
        output_attention=True,
        features=args.features,
        root_path=args.root_path,
        data_path=args.data_path,
        data=args.data,
        embed=args.embed,
        freq=args.freq,
        stride=args.stride,
        subset_rand_ratio=args.subset_rand_ratio,
        use_ims=args.use_ims,
        batch_size=1,
        num_workers=0,
        use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(),
        gpu=args.gpu,
        inverse=False,
        periodic_embedding_branch=getattr(args, "periodic_embedding_branch", 0),
        periodic_emb_bank_dim=getattr(args, "periodic_emb_bank_dim", 0),
    )
    return ns


def build_synthetic_sin_batch(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    IMS-style tensors: batch_x [B, seq_len, M], batch_y [B, label_len+pred_len, M].
    Time marks feed periodic_embedding_branch when enabled (hour/day indices per patch center).
    """
    B = args.sin_batch_size
    Lx = args.seq_len
    Ly = args.label_len + args.pred_len
    M = args.sin_n_vars

    t_x = torch.arange(Lx, dtype=torch.float32, device=device).view(1, Lx, 1)
    t_x_norm = t_x * (args.sin_base_period / max(Lx - 1, 1)) * (2 * math.pi)
    freqs = torch.linspace(1.0, 1.0 + 0.5 * max(M - 1, 0), M, device=device).view(1, 1, M)
    phase = torch.linspace(0.0, 0.5 * math.pi, M, device=device).view(1, 1, M)
    batch_x = torch.sin(t_x_norm * freqs + phase).expand(B, -1, -1).contiguous()

    t_y = torch.arange(Ly, dtype=torch.float32, device=device).view(1, Ly, 1)
    t_y_norm = t_y * (args.sin_base_period / max(Ly - 1, 1)) * (2 * math.pi)
    batch_y = torch.sin(t_y_norm * freqs + phase + 0.2 * math.pi).expand(B, -1, -1).contiguous()

    if args.sin_noise_std > 0:
        if args.sin_seed is not None:
            torch.manual_seed(args.sin_seed)
        batch_x = batch_x + args.sin_noise_std * torch.randn_like(batch_x)
        batch_y = batch_y + args.sin_noise_std * torch.randn_like(batch_y)

    mark_dim = 4
    batch_x_mark = torch.zeros(B, Lx, mark_dim, dtype=torch.float32, device=device)
    batch_y_mark = torch.zeros(B, Ly, mark_dim, dtype=torch.float32, device=device)
    return batch_x, batch_y, batch_x_mark, batch_y_mark


def plot_head_map(
    attn_2d: np.ndarray,
    out_path: str,
    title: str,
    gamma: float,
    dpi: int,
    cmap: str,
    vmax_quantile: float,
    attn_levels: int,
    attn_scale: str,
    metrics_line: str | None = None,
):
    """
    attn_2d: [N, N] non-negative weights. See _norm_for_attn for scaling / discrete bands.
    """
    attn_2d = np.asarray(attn_2d, dtype=np.float64)
    attn_2d = np.nan_to_num(attn_2d, nan=0.0, posinf=0.0, neginf=0.0)
    norm, vmax_plot = _norm_for_attn(attn_2d, gamma, vmax_quantile, attn_levels, attn_scale)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi)
    im = ax.imshow(
        attn_2d,
        cmap=cmap,
        aspect="equal",
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight", rotation=270, labelpad=16)
    cbar.ax.tick_params(labelsize=8)
    if attn_scale == "linear01":
        ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    else:
        ticks = np.linspace(0.0, vmax_plot, num=6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.4f}" for t in ticks])

    ax.set_xlabel("Key token (patch index)")
    ax.set_ylabel("Query token (patch index)")
    ax.set_title(title)
    n = attn_2d.shape[0]
    if n <= 32:
        _integer_axis_ticks(ax, n, labelsize=9)
    else:
        step = max(1, n // 16)
        ax.set_xticks(np.arange(0, attn_2d.shape[1], step))
        ax.set_yticks(np.arange(0, n, step))

    if metrics_line:
        fig.text(
            0.5,
            0.01,
            metrics_line,
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "wheat", "alpha": 0.9},
        )
        fig.subplots_adjust(bottom=0.12)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_grid(
    attn_heads: list[np.ndarray],
    out_path: str,
    n_heads: int,
    layer_idx: int,
    gamma: float,
    dpi: int,
    cmap: str,
    vmax_quantile: float,
    attn_levels: int,
    attn_scale: str,
    metrics_line: str | None = None,
):
    """One figure: all heads in this layer as a row x col grid."""
    ncols = min(4, n_heads)
    nrows = int(np.ceil(n_heads / ncols))
    # Shared vmax across heads in this layer for comparable scale
    stack = np.stack([np.nan_to_num(h, nan=0.0) for h in attn_heads], axis=0)
    norm, vmax_plot = _norm_for_attn(stack, gamma, vmax_quantile, attn_levels, attn_scale)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), dpi=dpi, squeeze=False)
    for h in range(n_heads):
        r, c = divmod(h, ncols)
        ax = axes[r][c]
        im = ax.imshow(
            attn_heads[h],
            cmap=cmap,
            aspect="equal",
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        nt = attn_heads[h].shape[0]
        if nt <= 32:
            _integer_axis_ticks(ax, nt, labelsize=7)
    # Hide empty axes
    for h in range(n_heads, nrows * ncols):
        r, c = divmod(h, ncols)
        axes[r][c].axis("off")

    if attn_scale == "linear01":
        supt = f"Layer {layer_idx} — shared color scale 0 .. 1 (linear)"
    else:
        supt = f"Layer {layer_idx} — shared color scale 0 .. {vmax_plot:.4f} ({attn_scale})"
    if metrics_line:
        supt = supt + "\n" + metrics_line
    fig.suptitle(supt, fontsize=11)
    fig.subplots_adjust(right=0.88, top=0.88 if metrics_line else 0.92)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        label="Attention weight",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_layers_all_heads_summary(
    attns: list[torch.Tensor],
    b_idx: int,
    out_path: str,
    gamma: float,
    dpi: int,
    cmap: str,
    vmax_quantile: float,
    attn_levels: int,
    attn_scale: str,
    run_id: str,
    metrics_line: str | None = None,
) -> None:
    """
    Single figure: rows = layers, cols = heads. Shared color scale for direct comparison.
    Integer ticks 0..N-1 on every cell (N<=32); with 8 patches you get labels 0-7.
    """
    # [L, H, N, N]
    stack = torch.stack([layer[b_idx].detach().float().cpu() for layer in attns], dim=0).numpy()
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    n_layers, n_heads, n_tok, n_tok_k = stack.shape
    assert n_tok == n_tok_k

    norm, vmax_plot = _norm_for_attn(stack, gamma, vmax_quantile, attn_levels, attn_scale)

    # Larger cells + constrained_layout so the full 8×8 grid is not clipped vs. manual axes.
    cell = 2.55
    fig_w = max(cell * n_heads + 1.2, 12.0)
    fig_h = max(cell * n_layers + 1.4, 10.0)
    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        squeeze=False,
        constrained_layout=True,
    )

    mappable = None
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li, hi]
            mappable = ax.imshow(
                stack[li, hi],
                cmap=cmap,
                aspect="equal",
                norm=norm,
                origin="upper",
                interpolation="nearest",
            )
            if n_tok <= 32:
                _integer_axis_ticks(ax, n_tok, labelsize=5.5)
            if li == 0:
                ax.set_title(f"H{hi}", fontsize=9, fontweight="semibold")
            if hi == 0:
                ax.set_ylabel(f"L{li}", fontsize=9, fontweight="semibold")
            if li == n_layers - 1:
                ax.set_xlabel("Key", fontsize=7)
            ax.tick_params(length=2, pad=1)

    bands = f" | {attn_levels} bands" if attn_levels >= 2 else ""
    if attn_scale == "linear01":
        scale_part = f"scale 0–1 linear{bands}"
    elif attn_scale == "linear_quantile":
        scale_part = f"stripe boost vmax≈{vmax_plot:.4f} q={vmax_quantile} γ={gamma}{bands}"
    else:
        scale_part = f"scale max≈{vmax_plot:.4f} compressed{bands}"
    title_main = f"[{run_id}]  All layers × all heads  |  {scale_part}  |  Query × Key (patch tokens)"
    if metrics_line:
        title_main = title_main + "\n" + metrics_line
    fig.suptitle(title_main, fontsize=10)
    assert mappable is not None
    fig.colorbar(mappable, ax=axes, shrink=0.72, location="right", label="Attention", pad=0.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    base_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    preferred_run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    if args.flat_output:
        run_dir = base_output_dir
        run_id = preferred_run_id
        prefix = f"{run_id}_"
    elif args.reuse_exact_run_id:
        run_id = preferred_run_id
        run_dir = os.path.join(base_output_dir, run_id)
        prefix = ""
    else:
        run_dir, run_id = _allocate_run_subdirectory(base_output_dir, preferred_run_id)
        prefix = ""

    os.makedirs(run_dir, exist_ok=True)

    cfg = build_config_ns(args)

    if cfg.use_gpu:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    if args.synthetic_sin:
        batch_x, batch_y, batch_x_mark, batch_y_mark = build_synthetic_sin_batch(args, device)
        print(
            f"Synthetic sin (torch.sin, no CSV): batch_x={tuple(batch_x.shape)} "
            f"batch_y={tuple(batch_y.shape)}"
        )
        if args.sin_noise_std > 0:
            print(f"  + Gaussian noise std={args.sin_noise_std} (seed={args.sin_seed})")
    else:
        _, data_loader = data_provider(cfg, flag="test")
        it = iter(data_loader)
        for _ in range(args.batch_index + 1):
            batch = next(it)
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : cfg.label_len, :], dec_inp], dim=1).float().to(device)

    model = TimerModel(cfg).float().to(device)
    model.eval()

    with torch.no_grad():
        _dec_out, attns = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    if not isinstance(attns, list) or len(attns) == 0:
        raise RuntimeError("No attention returned; ensure output_attention=True and checkpoint loads correctly.")

    metrics_line = _load_test_metrics_line(args.test_metrics_json, args.output_len)
    if metrics_line:
        print(metrics_line)

    n_layers = len(attns)
    b_idx = args.batch_eff_index
    meta_path = os.path.join(run_dir, f"{prefix}run_meta.json")
    cmap = args.attn_cmap
    vq = args.vmax_quantile

    meta = {
        "run_id": run_id,
        "run_id_requested": args.run_id,
        "run_id_preferred": preferred_run_id,
        "run_dir": run_dir,
        "data_source": "synthetic_sin" if args.synthetic_sin else "dataset",
        "flat_output": args.flat_output,
        "reuse_exact_run_id": args.reuse_exact_run_id,
        "ckpt_path": args.ckpt_path,
        "n_layers": n_layers,
        "gamma": args.gamma,
        "vmax_quantile": vq,
        "attn_levels": args.attn_levels,
        "attn_scale": args.attn_scale,
        "attn_cmap": cmap,
        "test_metrics_json": args.test_metrics_json or None,
    }
    for k, v in vars(args).items():
        if k not in meta:
            try:
                json.dumps(v)
                meta[f"arg_{k}"] = v
            except TypeError:
                meta[f"arg_{k}"] = str(v)

    for li, attn_layer in enumerate(attns):
        # [B, H, L, S]
        a = attn_layer[b_idx].detach().float().cpu().numpy()
        n_heads = a.shape[0]
        heads_np = [a[hi] for hi in range(n_heads)]

        layer_dir = os.path.join(run_dir, f"{prefix}layer_{li:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        for hi, mat in enumerate(heads_np):
            title = f"[{run_id}] Layer {li} Head {hi} (query→key)"
            out_png = os.path.join(layer_dir, f"head_{hi:02d}.png")
            plot_head_map(
                mat,
                out_png,
                title,
                args.gamma,
                args.dpi,
                cmap,
                vq,
                args.attn_levels,
                args.attn_scale,
                metrics_line,
            )

        grid_path = os.path.join(run_dir, f"{prefix}layer_{li:02d}_all_heads.png")
        plot_layer_grid(
            heads_np,
            grid_path,
            n_heads,
            li,
            args.gamma,
            args.dpi,
            cmap,
            vq,
            args.attn_levels,
            args.attn_scale,
            metrics_line,
        )

        meta[f"layer_{li}_attn_shape"] = list(attn_layer.shape)

    summary_name = f"{prefix}all_layers_all_heads_summary.png"
    summary_path = os.path.join(run_dir, summary_name)
    plot_all_layers_all_heads_summary(
        attns,
        b_idx,
        summary_path,
        args.gamma,
        args.dpi,
        cmap,
        vq,
        args.attn_levels,
        args.attn_scale,
        run_id,
        metrics_line,
    )

    meta["summary_png"] = summary_name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    index_path = os.path.join(base_output_dir, "runs_index.txt")
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(f"{run_id}\t{os.path.abspath(summary_path)}\t{datetime.now().isoformat()}\n")

    print(f"run_id={run_id}")
    if not args.flat_output and run_id != preferred_run_id:
        print(f"(requested id '{preferred_run_id}' was already taken; using unique folder name above)")
    print(f"Saved under: {run_dir}")
    print(f"Summary (all layers × heads): {os.path.abspath(summary_path)}")
    print(f"Run index appended: {index_path}")


if __name__ == "__main__":
    main()
