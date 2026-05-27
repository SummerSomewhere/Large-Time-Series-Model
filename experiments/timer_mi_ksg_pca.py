#!/usr/bin/env python3
"""
Timer 层间 MI 分析 (基于 KSG Estimator & PCA)。
输出 JSON 格式与 global_mi_peaks_{model_id}.json 一致，
包含 hsic_curve, sigma_y_sq, high/low_mi_patches, all_high/low_mi_patches 等字段。

Usage:
  python experiments/timer_mi_ksg_pca.py \
    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
    --root_path ./datasets/ --data ETTh1 --data_path ETTh1.csv \
    --out_dir ./results/timer_mi_ksg_pca/ --model_id etth1
"""

import argparse
import json
import os
import gc
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as ss
import scipy.special as sp
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from utils.masking import TriangularCausalMask


# ─── KSG MI Estimator ─────────────────────────────────────────────────────────

def compute_mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    KSG (Kraskov, Stoegbauer, Grassberger) Estimator for I(X;Y).
    x: [N, D_x], y: [N, D_y]
    Returns MI in bits.
    """
    N = x.shape[0]
    if N <= k + 1:
        return 0.0

    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)

    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)
    xy = np.concatenate((x, y), axis=1)
    tree_xy = ss.cKDTree(xy)

    dist_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    epsilon = dist_xy[:, k]
    eps_strict = np.maximum(epsilon - 1e-10, 0)

    nx = np.array([len(tree_x.query_ball_point(x[i], r=eps_strict[i], p=np.inf)) - 1 for i in range(N)])
    ny = np.array([len(tree_y.query_ball_point(y[i], r=eps_strict[i], p=np.inf)) - 1 for i in range(N)])
    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    psi_k = sp.digamma(k)
    psi_N = sp.digamma(N)
    mean_psi = np.mean(sp.digamma(nx + 1) + sp.digamma(ny + 1))
    mi = psi_k - mean_psi + psi_N
    return max(0.0, mi / np.log(2))


# ─── Config & Model ───────────────────────────────────────────────────────────

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer_model(ckpt_path: str, patch_len: int, stride: int,
                       d_model: int, d_ff: int, e_layers: int,
                       n_heads: int, dropout: float,
                       seq_len: int, pred_len: int) -> Model:
    config = Config(
        task_name='forecast',
        ckpt_path=ckpt_path,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        n_heads=n_heads,
        dropout=dropout,
        output_attention=False,
        distil=True,
        use_revin=False,
        seq_len=seq_len,
        pred_len=pred_len,
        d_layers=1,
        factor=1,
        enc_in=1,
        dec_in=1,
        c_out=1,
        activation='gelu',
        use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(),
        devices='0',
        num_workers=4,
        freq='h',
        data='custom',
        embed='timeF',
        target='OT',
        features='M',
        des='Exp',
        lradj='type1',
        use_amp=False,
        is_finetuning=0,
        label_len=pred_len,
        output_len=pred_len,
        batch_size=64,
        train_epochs=1,
        patience=3,
        learning_rate=3e-5,
        itr=1,
        use_ims=False,
        inverse=False,
        use_align_loss=False,
        align_loss_layers=list(range(e_layers)),
    )
    model = Model(config)
    model.eval()
    return model


# ─── Token Extraction ─────────────────────────────────────────────────────────

def _unwrap_timer(model):
    if hasattr(model, "module"):
        return model.module
    return model


def extract_layer_tokens(model, data_loader, device, n_layers: int,
                           pred_len: int, n_vars: int, patch_len: int):
    """
    Extract per-layer hidden states for history (x) and GT future (y).
    Both x and y go through the SAME pathway:
        enc_embedding (patch embedding) -> decoder layers 0..l

    Also extracts patch embedding output (dec_in_x) before Transformer layers:
        x_patch_tokens: [N_total, n_patches, D] — patch embedding output

    Returns:
        hist_tokens:  list of [N_total, n_patches, D]
        future_tokens: list of [N_total, n_future_patches, D]
        x_patch_tokens: [N_total, n_patches, D]  — patch embedding output (dec_in_x), dim=D
        n_patches: int
        n_future_patches: int
    """
    core = _unwrap_timer(model)

    all_hist_tokens = [[] for _ in range(n_layers)]
    all_future_tokens = [[] for _ in range(n_layers)]
    all_x_patch_tokens = []

    n_patches = None
    n_future_patches = None

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(data_loader, desc="提取 token 表示"):
            B = seq_x.shape[0]

            seq_x = seq_x.float().to(device)
            seq_y = seq_y.float().to(device)

            # Normalize (same as Timer.forecast)
            means = seq_x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = (seq_x - means) / stdev
            y_norm = (seq_y - means) / stdev

            # Patch embedding: [B, T, M] -> [B*M, N, D]
            x2 = x_norm.permute(0, 2, 1)
            y2 = y_norm.permute(0, 2, 1)

            dec_in_x, n_vars_x = core.enc_embedding(x2)
            dec_in_y, n_vars_y = core.enc_embedding(y2)

            # ── X: patch embedding output (dec_in_x, before Transformer layers) ──
            BM, N, D = dec_in_x.shape
            BM_y, N_y, _ = dec_in_y.shape

            if n_patches is None:
                n_patches = N
                n_future_patches = N_y
                print(f"  DEBUG: B={B}, BM={BM}, N={N}, D={D}, BM/B={BM/B:.2f}")
                print(f"  DEBUG: returned n_vars_x={n_vars_x}")
                print(f"  检测到 n_patches={n_patches}, n_future_patches={n_future_patches}")

            # Verify BM is divisible by B
            assert BM % B == 0, f"BM={BM} not divisible by B={B}"
            derived_n_vars_x = BM // B
            assert BM_y % B == 0, f"BM_y={BM_y} not divisible by B={B}"
            derived_n_vars_y = BM_y // B

            x_patch_emb = dec_in_x.view(B, derived_n_vars_x, N, D).mean(dim=1).float().cpu()
            all_x_patch_tokens.append(x_patch_emb)

            # x and y may have different sequence lengths -> different patch counts
            # -> use separate masks for each branch
            mask_x = TriangularCausalMask(BM, N, device=device)
            mask_y = TriangularCausalMask(BM_y, N_y, device=device)

            # Iterate through decoder layers, collecting per-layer outputs.
            # Hidden states are passed through layers progressively (recurrently):
            # h_x and h_y accumulate layer by layer so that layer l sees
            # the output of layer l-1, not the raw embedding.
            h_x = dec_in_x
            h_y = dec_in_y
            for li, layer_module in enumerate(core.decoder.attn_layers):
                h_x, _, _ = layer_module(h_x, attn_mask=mask_x)
                h_y, _, _ = layer_module(h_y, attn_mask=mask_y)
                all_hist_tokens[li].append(h_x.view(B, derived_n_vars_x, N, D).mean(dim=1).float().cpu())
                all_future_tokens[li].append(h_y.view(B, derived_n_vars_y, N_y, D).mean(dim=1).float().cpu())

            del dec_in_x, dec_in_y
            gc.collect()
            torch.cuda.empty_cache()

    hist_tokens = [torch.cat(toks, dim=0) for toks in all_hist_tokens]
    future_tokens = [torch.cat(toks, dim=0) for toks in all_future_tokens]
    x_patch_tokens = torch.cat(all_x_patch_tokens, dim=0)  # [N_total, n_patches, D]

    return hist_tokens, future_tokens, x_patch_tokens, n_patches, n_future_patches


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_mi_results(out_dir: str, mi_hy: np.ndarray, mi_xh: np.ndarray,
                    n_layers: int, n_patches: int,
                    stable_ranks: np.ndarray = None):
    """
    Plot I(H,Y), I(X,H) and their difference I(H,Y)-I(X,H).
    mi_hy: [n_layers, n_patches] — I(H, Y)
    mi_xh: [n_layers, n_patches] — I(X, H)
    """
    patch_axis = np.arange(n_patches)
    cmap = plt.get_cmap("turbo")
    colors = [cmap(li / max(1, n_layers - 1)) for li in range(n_layers)]

    mi_diff = mi_hy - mi_xh

    # SPI bias: 15% of mean I(X,H) across all layers and all tokens
    spi_bias = 0.15 * mi_xh.mean()
    print(f"  SPI bias (15% of mean I(X,H)): {spi_bias:.6f}")

    # ── Combined: I(H,Y) | I(X,H) | I(H,Y)-I(X,H) | SPI ──────────────────────
    # 2x2 layout: each cell = line plot (left 70%) + per-layer mean bar chart (right 20%)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    titles = ["I(H, Y) — Layer vs Patch (top 25% marked)",
              "I(X, H) — Layer vs Patch",
              "I(H, Y) - I(X, H) — Information Gain per Layer",
              f"SPI — Semantic Purity Index: I(H,Y)/(I(X,H)+bias), bias={spi_bias:.4f}"]
    ydata = [mi_hy, mi_xh, mi_diff, mi_hy / (mi_xh + spi_bias)]
    ylabels = ["MI (bits)", "MI (bits)", "MI Difference (bits)", "SPI"]
    hlines = [None, None, 0.0, None]

    for ax, data, title, ylabel, hline in zip(axes, ydata, titles, ylabels, hlines):
        for li in range(n_layers):
            ax.plot(patch_axis, data[li], alpha=0.85, color=colors[li],
                    linewidth=1.5, marker=".", markersize=3, label=f"L{li}")
            if title.startswith("SPI") or title.startswith("I(H, Y)"):
                threshold = np.percentile(data[li], 75)
                top_mask = data[li] >= threshold
                top_patches = np.where(top_mask)[0]
                ax.scatter(top_patches, data[li][top_mask], s=40,
                           color=colors[li], marker="^", edgecolors="black",
                           linewidths=0.5, zorder=4)
        if hline is not None:
            ax.axhline(y=hline, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, ncol=min(8, n_layers), loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("History Patch Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_triple_overlay.png"), dpi=150)
    plt.close()

    # ── 2x2 grid: line plot (left) + per-layer mean bar chart (right) ──────────
    # Each subplot: [line_axes | bar_axes], separated by a divider
    fig, axes = plt.subplots(4, 2, figsize=(16, 18),
                               gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08})
    titles_grid = [
        "I(H, Y) per Patch with Per-Layer Mean",
        "Mean I(H, Y)",
        "I(X, H) per Patch with Per-Layer Mean",
        "Mean I(X, H)",
        "I(H, Y) - I(X, H) per Patch with Per-Layer Mean",
        "Mean Info. Gain",
        f"SPI per Patch with Per-Layer Mean  (bias={spi_bias:.4f})",
        "Mean SPI",
    ]
    ydata_grid = [mi_hy, mi_xh, mi_diff, mi_hy / (mi_xh + spi_bias)]
    ylabels_grid = ["MI (bits)", "MI (bits)", "MI Diff (bits)", "SPI"]

    for row in range(4):
        ax_line = axes[row, 0]
        ax_bar = axes[row, 1]
        data = ydata_grid[row]
        ylabel = ylabels_grid[row]

        for li in range(n_layers):
            ax_line.plot(patch_axis, data[li], alpha=0.85, color=colors[li],
                         linewidth=1.5, marker=".", markersize=3, label=f"L{li}")
            # Mark top 25% patches with triangle
            threshold = np.percentile(data[li], 75)
            top_mask = data[li] >= threshold
            top_patches = np.where(top_mask)[0]
            if len(top_patches) > 0:
                ax_line.scatter(top_patches, data[li][top_mask], s=35,
                                color=colors[li], marker="^", edgecolors="black",
                                linewidths=0.5, zorder=4)
        ax_line.set_ylabel(ylabel, fontsize=9)
        ax_line.set_title(titles_grid[row * 2], fontsize=9)
        ax_line.legend(fontsize=6, ncol=min(8, n_layers), loc="best")
        ax_line.grid(True, alpha=0.3)

        mean_vals = data.mean(axis=1)  # [n_layers]
        x_pos = np.arange(n_layers)
        bars = ax_bar.bar(x_pos, mean_vals, color=colors, edgecolor="white",
                          linewidth=0.5, width=0.7)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)
        ax_bar.set_ylabel("Mean", fontsize=8)
        ax_bar.set_title(titles_grid[row * 2 + 1], fontsize=9)
        ax_bar.grid(True, alpha=0.3, axis="y")
        for bar, mv in zip(bars, mean_vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{mv:.3f}", ha="center", va="bottom", fontsize=6.5,
                        color="black")

        if row == 0:
            ax_bar.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)

    axes[-1, 0].set_xlabel("History Patch Index", fontsize=10)
    axes[-1, 1].set_xlabel("Layer", fontsize=9)
    fig.suptitle("MI Analysis: Line Plot (per patch) + Per-Layer Mean Bar Chart", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(out_dir, "mi_triple_with_mean_bar.png"), dpi=150)
    plt.close()

    # Also keep individual files for convenience
    fig, ax = plt.subplots(figsize=(10, 5))
    for li in range(n_layers):
        ax.plot(patch_axis, mi_hy[li], alpha=0.85, color=colors[li],
                linewidth=1.5, marker=".", markersize=3, label=f"L{li}")
        threshold = np.percentile(mi_hy[li], 75)
        top_mask = mi_hy[li] >= threshold
        top_patches = np.where(top_mask)[0]
        ax.scatter(top_patches, mi_hy[li][top_mask], s=40,
                   color=colors[li], marker="^", edgecolors="black",
                   linewidths=0.5, zorder=4)
    ax.set_xlabel("History Patch Index"); ax.set_ylabel("MI (bits)")
    ax.set_title("I(H, Y) — Layer vs Patch (top 25% marked)")
    ax.legend(fontsize=7, ncol=min(8, n_layers), loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_hy_overlay.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for li in range(n_layers):
        ax.plot(patch_axis, mi_xh[li], alpha=0.85, color=colors[li],
                linewidth=1.5, marker=".", markersize=3, label=f"L{li}")
    ax.set_xlabel("History Patch Index"); ax.set_ylabel("MI (bits)")
    ax.set_title("I(X, H) — Layer vs Patch")
    ax.legend(fontsize=7, ncol=min(8, n_layers), loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_xh_overlay.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for li in range(n_layers):
        ax.plot(patch_axis, mi_diff[li], alpha=0.85, color=colors[li],
                linewidth=1.5, marker=".", markersize=3, label=f"L{li}")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("History Patch Index"); ax.set_ylabel("MI Difference (bits)")
    ax.set_title("I(H, Y) - I(X, H) — Information Gain per Layer")
    ax.legend(fontsize=7, ncol=min(8, n_layers), loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_diff_overlay.png"), dpi=150)
    plt.close()

    # ── Subplot 4: Combined — I(X,H) (dashed) + I(H,Y) (solid) per layer ───────
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=False)
    axes = axes.flatten()
    for li in range(n_layers):
        ax = axes[li]
        ax.plot(patch_axis, mi_xh[li], alpha=0.7, color=colors[li],
                linewidth=1.2, linestyle="--", label="I(X,H)")
        ax.plot(patch_axis, mi_hy[li], alpha=0.7, color=colors[li],
                linewidth=1.2, linestyle="-", label="I(H,Y)")
        ax.fill_between(patch_axis, mi_xh[li], mi_hy[li],
                        alpha=0.15, color=colors[li])
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"Layer {li}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if li == 0:
            ax.legend(fontsize=6, loc="best")
    fig.text(0.5, 0.01, "History Patch Index", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "MI (bits)", va="center", rotation="vertical", fontsize=10)
    fig.suptitle("I(X,H) (dashed) vs I(H,Y) (solid) — shaded = I(H,Y)-I(X,H)", fontsize=11)
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "mi_combined_overlay.png"), dpi=150)
    plt.close()

    # ── Heatmap: I(H,Y) - I(X,H) ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mi_diff, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("History Patch Index")
    ax.set_ylabel("Layer")
    ax.set_title("I(H, Y) - I(X, H) Heatmap (Information Gain)")
    ax.set_yticks(range(n_layers))
    ax.set_xticks(range(0, n_patches, max(1, n_patches // 8)))
    plt.colorbar(im, ax=ax, label="MI Diff (bits)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_diff_heatmap.png"), dpi=150)
    plt.close()

    # ── Mean I(X,H) and I(H,Y) per layer ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_layers)
    width = 0.35
    bars1 = ax.bar(x - width/2, mi_hy.mean(axis=1), width, label="I(H,Y)", color="steelblue")
    bars2 = ax.bar(x + width/2, mi_xh.mean(axis=1), width, label="I(X,H)", color="coral")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean MI (bits)")
    ax.set_title("Mean MI per Layer: I(H,Y) vs I(X,H)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_mean_per_layer.png"), dpi=150)
    plt.close()

    # ── Information Plane: I(X,H) vs I(H,Y) ───────────────────────────────────
    mean_ixh = mi_xh.mean(axis=1)   # [n_layers]
    mean_ihy = mi_hy.mean(axis=1)   # [n_layers]
    cmap = plt.get_cmap("turbo")
    colors_plane = [cmap(li / max(1, n_layers - 1)) for li in range(n_layers)]

    fig, ax = plt.subplots(figsize=(9, 7))

    for li in range(n_layers):
        ax.scatter(mean_ixh[li], mean_ihy[li], s=120, color=colors_plane[li],
                    zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(f"L{li}", (mean_ixh[li], mean_ihy[li]),
                    textcoords="offset points", xytext=(7, 3),
                    fontsize=8, color=colors_plane[li], fontweight="bold")

    for li in range(n_layers - 1):
        ax.annotate("",
            xy=(mean_ixh[li + 1], mean_ihy[li + 1]),
            xytext=(mean_ixh[li], mean_ihy[li]),
            arrowprops=dict(arrowstyle="->", color=colors_plane[li + 1],
                            lw=1.5, alpha=0.7))

    ax.plot(mean_ixh, mean_ihy, "--", color="gray", linewidth=1.0,
            alpha=0.5, zorder=1)

    best_layer = np.argmax(mean_ihy - mean_ixh)
    ax.axhline(y=mean_ihy[best_layer], color="gray", linestyle=":",
               linewidth=0.8, alpha=0.5)
    ax.axvline(x=mean_ixh[best_layer], color="gray", linestyle=":",
               linewidth=0.8, alpha=0.5)
    ax.scatter(mean_ixh[best_layer], mean_ihy[best_layer], s=300,
               facecolors="none", edgecolors="gold", linewidths=2.5,
               zorder=4, label=f"Best (L{best_layer})")

    ax.set_xlabel("I(X; H) — Compression (bits)", fontsize=11)
    ax.set_ylabel("I(H; Y) — Prediction Power (bits)", fontsize=11)
    ax.set_title("Information Plane: Model Flow through Layers\n"
                 "(← Compression  |  ↑ Prediction  |  ★ = Optimal Balance)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    x_pad = (mean_ixh.max() - mean_ixh.min()) * 0.08
    y_pad = (mean_ihy.max() - mean_ihy.min()) * 0.08
    ax.set_xlim(mean_ixh.min() - x_pad, mean_ixh.max() + x_pad)
    ax.set_ylim(mean_ihy.min() - y_pad, mean_ihy.max() + y_pad)

    arrow_x0, arrow_y0 = mean_ixh[0], mean_ihy[0]
    arrow_x1, arrow_y1 = mean_ixh[-1], mean_ihy[-1]
    ax.annotate("Input", (arrow_x0, arrow_y0), xytext=(-35, -15),
                textcoords="offset points", fontsize=8, color="gray")
    ax.annotate("Output", (arrow_x1, arrow_y1), xytext=(5, 8),
                textcoords="offset points", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_information_plane.png"), dpi=150)
    plt.close()

    # ── Stable Rank per layer ──────────────────────────────────────────────────
    if stable_ranks is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Stable rank bar chart
        ax = axes[0]
        x = np.arange(n_layers)
        bars = ax.bar(x, stable_ranks, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Stable Rank  sr(H) = ||H||_F^2 / ||H||_2^2", fontsize=9)
        ax.set_title("Stable Rank per Layer\n(lower = narrower bottleneck)", fontsize=10)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, sr in zip(bars, stable_ranks):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{sr:.1f}", ha="center", va="bottom", fontsize=8)

        # Right: Stable Rank vs I(X,H) — check if compression is structural
        ax = axes[1]
        mean_ixh = mi_xh.mean(axis=1)
        ax.scatter(mean_ixh, stable_ranks, s=120, color=colors, edgecolors="white",
                   linewidths=0.8, zorder=3)
        for li in range(n_layers):
            ax.annotate(f"L{li}", (mean_ixh[li], stable_ranks[li]),
                        textcoords="offset points", xytext=(7, 3),
                        fontsize=8, color=colors[li], fontweight="bold")
        ax.set_xlabel("I(X; H) — Compression (bits)", fontsize=10)
        ax.set_ylabel("Stable Rank", fontsize=10)
        ax.set_title("Stable Rank vs Compression\n"
                     "(if left-down together → structural bottleneck)", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "stable_rank_analysis.png"), dpi=150)
        plt.close()

    return mi_diff


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Timer 层间 MI 分析 (KSG + PCA)")
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/Timer_forecast_1.0.ckpt")
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
    parser.add_argument("--sample_ratio", type=float, default=0.05,
                        help="Fraction of test set to sample for MI computation (default: 0.05 = 1/20)")
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./results/timer_mi_ksg_pca")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--data_type", type=str, default="ETTh1")
    parser.add_argument("--model_id", type=str, default="etth1")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"Timer_MI_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Timer 层间 MI 分析 (KSG Estimator + PCA)")
    print("=" * 70)
    print(f"  ckpt_path  : {args.ckpt_path}")
    print(f"  data_path  : {args.data_path}")
    print(f"  seq_len    : {args.seq_len}")
    print(f"  pred_len   : {args.pred_len}")
    print(f"  patch_len  : {args.patch_len}")
    print(f"  stride     : {args.stride}")
    print(f"  e_layers   : {args.e_layers}")
    print(f"  pca_dim    : {args.pca_dim}")
    print(f"  device     : {device}")
    print(f"  model_id   : {args.model_id}")
    print("=" * 70)

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print("\n>>> Phase 1: 加载数据集...")
    test_dataset = CIDatasetBenchmark(
        root_path=os.path.join(args.root_path, args.data_path),
        flag='test',
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = test_dataset.n_var
    print(f"  变量数: {n_vars}")
    total_samples = len(test_dataset)
    print(f"  测试集总样本数: {total_samples}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    # ── 2. Load Timer model ───────────────────────────────────────────────────
    print("\n>>> Phase 2: 加载 Timer 模型...")
    model = build_timer_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    model = model.to(device)
    model.eval()
    print(f"  模型加载完成，设备: {device}")

    # ── 3. Extract layer representations ─────────────────────────────────────
    print("\n>>> Phase 3: 提取各层 token 表示...")
    hist_tokens, future_tokens, x_patch_tokens, n_patches, n_future_patches = extract_layer_tokens(
        model, test_loader, device,
        args.e_layers, args.pred_len, n_vars,
        patch_len=args.patch_len,
    )
    N_total = hist_tokens[0].shape[0]
    D = hist_tokens[0].shape[2]
    print(f"  样本数 N={N_total}, 历史 patches={n_patches}, "
          f"未来 patches={n_future_patches}, D={D}")
    print(f"  Raw patch tokens: {x_patch_tokens.shape} (should be [N, n_patches, D])")

    # Randomly sample 1/20 of test set for faster MI computation
    sample_ratio = args.sample_ratio
    n_sampled = max(int(N_total * sample_ratio), 1)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(N_total, size=n_sampled, replace=False)
    sample_indices = np.sort(sample_indices)


    print(f"  随机采样 {n_sampled}/{N_total} 样本 (ratio={sample_ratio})")

    # ── 4. Compute MI per layer per patch ────────────────────────────────────
    print("\n>>> Phase 4: 计算 MI (KSG Estimator + PCA)...")
    mi_hy_matrix = np.zeros((args.e_layers, n_patches))
    mi_xh_matrix = np.zeros((args.e_layers, n_patches))
    sigma_y_sq_agg = np.zeros(args.e_layers)

    # Raw patch tokens for I(X,H)
    x_patch_sampled = x_patch_tokens[sample_indices].numpy()  # [n_sampled, n_patches, patch_len]
    assert x_patch_sampled.shape[1] == n_patches, \
        f"x_patch patches={x_patch_sampled.shape[1]} != n_patches={n_patches}"

    for li in tqdm(range(args.e_layers), desc="计算 MI (按层)"):
        hist_li = hist_tokens[li][sample_indices].numpy()
        future_li = future_tokens[li][sample_indices].numpy()

        # sigma_y_sq from future tokens (all patches × all dims)
        sigma_y_sq_agg[li] = float(np.var(future_li))

        # PCA on all future patches (shared across patches of this layer)
        future_flat = future_li.reshape(n_sampled, -1)
        n_comp_y = min(args.pca_dim, n_sampled, future_flat.shape[1])
        pca_y = PCA(n_components=n_comp_y)
        future_reduced = pca_y.fit_transform(future_flat)

        for pi in tqdm(range(n_patches), desc=f"Layer {li} patches", leave=False):
            # ── I(H, Y): hidden state patch vs future ──────────────────────────────
            x_hidden = hist_li[:, pi, :]

            n_comp_x = min(args.pca_dim, n_sampled, x_hidden.shape[1])
            pca_x = PCA(n_components=n_comp_x)
            x_hidden_reduced = pca_x.fit_transform(x_hidden)

            mi_hy_val = compute_mi_ksg(x_hidden_reduced, future_reduced, k=args.k_neighbors)
            mi_hy_matrix[li, pi] = mi_hy_val

            # ── I(X, H): patch embedding vs hidden state ──────────────────────────
            x_emb = x_patch_sampled[:, pi, :]  # [n_sampled, D]

            n_comp_xemb = min(args.pca_dim, n_sampled, x_emb.shape[1])
            pca_xemb = PCA(n_components=n_comp_xemb)
            x_emb_reduced = pca_xemb.fit_transform(x_emb)

            mi_xh_val = compute_mi_ksg(x_emb_reduced, x_hidden_reduced, k=args.k_neighbors)
            mi_xh_matrix[li, pi] = mi_xh_val

        del hist_li, future_li, future_flat
        gc.collect()
        torch.cuda.empty_cache()

    # ── 5. Build and save results JSON ────────────────────────────────────────
    print("\n>>> Phase 5: 保存结果...")

    layers_dict = {}
    for li in range(args.e_layers):
        layer_mi = mi_hy_matrix[li]
        q3 = float(np.percentile(layer_mi, 75))
        sorted_idx = np.argsort(layer_mi)
        high_mi_patches = [int(sorted_idx[-1]), int(sorted_idx[-2])]
        low_mi_patches = [int(sorted_idx[0]), int(sorted_idx[1])]

        layers_dict[str(li)] = {
            "hsic_curve": layer_mi.tolist(),
            "q3_threshold": q3,
            "sigma_y_sq": sigma_y_sq_agg[li],
            "high_mi_patches": high_mi_patches,
            "low_mi_patches": low_mi_patches,
            "high_mi_ratio": len(high_mi_patches) / len(layer_mi),
            "mi_xh_curve": mi_xh_matrix[li].tolist(),
        }

    all_high = set()
    all_low = set()
    for layer_data in layers_dict.values():
        all_high.update(layer_data["high_mi_patches"])
        all_low.update(layer_data["low_mi_patches"])

    result_json = {
        "model_id": args.model_id,
        "n_vars": n_vars,
        "N": n_patches,
        "patch_len": args.patch_len,
        "stride": args.patch_len,
        "pred_len": args.pred_len,
        "total_samples": n_sampled,
        "layers": layers_dict,
        "all_high_mi_patches": sorted(list(all_high)),
        "all_low_mi_patches": sorted(list(all_low)),
        "num_layers": args.e_layers,
    }

    json_path = os.path.join(output_dir, f"global_mi_peaks_{args.model_id}.json")
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  JSON 已保存: {json_path}")

    np.save(os.path.join(output_dir, "mi_hy_matrix.npy"), mi_hy_matrix)
    np.save(os.path.join(output_dir, "mi_xh_matrix.npy"), mi_xh_matrix)

    # ── Compute Stable Rank per layer (use sampled indices for speed) ─────────────
    print(f"\n>>> Computing Stable Rank per layer (n={n_sampled} samples)...")
    stable_ranks = []
    for li in range(args.e_layers):
        H = hist_tokens[li][sample_indices].numpy()  # [n_sampled, n_patches, D]
        H_flat = H.reshape(H.shape[0], -1)           # [n_sampled, n_patches*D]
        fro_sq = np.sum(H_flat ** 2)                  # ||H||_F^2
        power_iter = H_flat @ H_flat.T               # [n_sampled, n_sampled]
        eig_top = np.linalg.eigvalsh(power_iter)
        spec_sq = np.max(eig_top)                     # ||H||_2^2
        sr = fro_sq / (spec_sq + 1e-12)
        stable_ranks.append(sr)
        print(f"  Layer {li}: sr={sr:.2f}, Fro^2={fro_sq:.4f}, Spec^2={spec_sq:.4f}")
    stable_ranks = np.array(stable_ranks)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    print("\n>>> Phase 6: 绘图...")
    plot_mi_results(output_dir, mi_hy_matrix, mi_xh_matrix, args.e_layers, n_patches,
                    stable_ranks)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    mi_diff = mi_hy_matrix - mi_xh_matrix
    print("\n" + "=" * 70)
    print("MI 结果汇总 (KSG + PCA)")
    print("=" * 70)
    print(f"{'Layer':>6} | {'I(H,Y) Mean':>12} | {'I(X,H) Mean':>12} | {'Diff Mean':>12}")
    print("-" * 52)
    for li in range(args.e_layers):
        print(f"{li:>6} | {np.mean(mi_hy_matrix[li]):>12.6f} | "
              f"{np.mean(mi_xh_matrix[li]):>12.6f} | {np.mean(mi_diff[li]):>12.6f}")

    print(f"\n输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
