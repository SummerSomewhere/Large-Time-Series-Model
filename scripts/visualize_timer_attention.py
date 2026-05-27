#!/usr/bin/env python3
"""
Extract self-attention from Timer (patch tokens) and save per-layer, per-head heatmaps.

Run from repo root:
  python scripts/visualize_timer_attention.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \\
    --root_path ./datasets/ETT-small/ --data_path ETTh1.csv --data ETTh1

Attention tensor shape per layer: [B_eff, n_heads, N, N] where B_eff = batch * n_vars after patching.
Rows = query token index, columns = key token index (attention weight).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Repo root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm

from data_provider.data_factory import data_provider
from models.Timer import Model as TimerModel


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
    p.add_argument("--output_dir", type=str, default="./attention_maps_timer")
    p.add_argument("--batch_index", type=int, default=0, help="Which batch from the test loader")
    p.add_argument("--batch_eff_index", type=int, default=0, help="Index along B_eff after patch embed (usually 0)")
    p.add_argument(
        "--gamma",
        type=float,
        default=0.45,
        help="PowerNorm gamma (<1 stretches small weights for visible separation)",
    )
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--gpu", type=int, default=0)
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
    )
    return ns


def plot_head_map(
    attn_2d: np.ndarray,
    out_path: str,
    title: str,
    gamma: float,
    dpi: int,
):
    """
    attn_2d: [N, N] non-negative weights. Uses PowerNorm to spread small/large gaps on the color scale.
    """
    attn_2d = np.asarray(attn_2d, dtype=np.float64)
    attn_2d = np.nan_to_num(attn_2d, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.clip(attn_2d.max(), 1e-8, 1.0))
    # High percentile cap avoids a single spike washing out the rest of the scale
    hi = float(np.quantile(attn_2d, 0.995))
    vmax_plot = max(min(vmax, hi), 1e-8) if hi > 0 else vmax
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax_plot)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi)
    im = ax.imshow(
        attn_2d,
        cmap="turbo",
        aspect="equal",
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight", rotation=270, labelpad=16)
    cbar.ax.tick_params(labelsize=8)
    # Finer ticks on colorbar
    ticks = np.linspace(0.0, vmax_plot, num=6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.4f}" for t in ticks])

    ax.set_xlabel("Key token (patch index)")
    ax.set_ylabel("Query token (patch index)")
    ax.set_title(title)
    ax.set_xticks(np.arange(attn_2d.shape[1]))
    ax.set_yticks(np.arange(attn_2d.shape[0]))
    # Avoid overcrowding: show every k-th tick label for long sequences
    n = attn_2d.shape[0]
    if n > 24:
        step = max(1, n // 12)
        ax.set_xticks(np.arange(0, n, step))
        ax.set_yticks(np.arange(0, n, step))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_grid(
    attn_heads: list[np.ndarray],
    out_path: str,
    n_heads: int,
    layer_idx: int,
    gamma: float,
    dpi: int,
):
    """One figure: all heads in this layer as a row x col grid."""
    ncols = min(4, n_heads)
    nrows = int(np.ceil(n_heads / ncols))
    # Shared vmax across heads in this layer for comparable scale
    stack = np.stack([np.nan_to_num(h, nan=0.0) for h in attn_heads], axis=0)
    vmax = float(np.clip(stack.max(), 1e-8, 1.0))
    hi = float(np.quantile(stack, 0.995))
    vmax_plot = max(min(vmax, hi), 1e-8) if hi > 0 else vmax
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax_plot)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), dpi=dpi, squeeze=False)
    for h in range(n_heads):
        r, c = divmod(h, ncols)
        ax = axes[r][c]
        im = ax.imshow(
            attn_heads[h],
            cmap="turbo",
            aspect="equal",
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
    # Hide empty axes
    for h in range(n_heads, nrows * ncols):
        r, c = divmod(h, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"Layer {layer_idx} — shared color scale (0 .. {vmax_plot:.4f})", fontsize=12)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"),
        cax=cbar_ax,
        label="Attention weight",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cfg = build_config_ns(args)

    os.makedirs(args.output_dir, exist_ok=True)

    if cfg.use_gpu:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

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
        dec_out, attns = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    if not isinstance(attns, list) or len(attns) == 0:
        raise RuntimeError("No attention returned; ensure output_attention=True and checkpoint loads correctly.")

    n_layers = len(attns)
    b_idx = args.batch_eff_index
    meta_path = os.path.join(args.output_dir, "run_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt={args.ckpt_path}\n")
        f.write(f"n_layers={n_layers}\n")
        f.write(f"dec_out_shape={tuple(dec_out.shape)}\n")

    for li, attn_layer in enumerate(attns):
        # [B, H, L, S]
        a = attn_layer[b_idx].detach().float().cpu().numpy()
        n_heads = a.shape[0]
        heads_np = [a[hi] for hi in range(n_heads)]

        layer_dir = os.path.join(args.output_dir, f"layer_{li:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        for hi, mat in enumerate(heads_np):
            title = f"Timer Layer {li} Head {hi} (query→key attention)"
            out_png = os.path.join(layer_dir, f"head_{hi:02d}.png")
            plot_head_map(mat, out_png, title, args.gamma, args.dpi)

        grid_path = os.path.join(args.output_dir, f"layer_{li:02d}_all_heads.png")
        plot_layer_grid(heads_np, grid_path, n_heads, li, args.gamma, args.dpi)

        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(f"layer_{li} attn shape (full)={tuple(attn_layer.shape)}\n")

    print(f"Saved attention maps under: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
