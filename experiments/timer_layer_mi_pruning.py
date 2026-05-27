#!/usr/bin/env python3
"""
Timer 基于层间 MI 变化率的 Transformer 层剪枝实验。

核心思路（方法一：相对变化率 Δ_rel）：
  - 计算每层平均 I(H_l, Y) = mean(hsic_curve[l])，即该层对预测目标的信息量
  - 相邻层变化率 Δ_rel[l] = |I_l - I_{l-1}| / I_{l-1}
  - 变化率 < threshold → 两层对预测的贡献几乎相同 → 中间层冗余 → 剪掉
  - threshold 由数据驱动确定：取 Δ_rel 的某个分位数

三步流程：
  [诊断]   计算各层平均 MI 及相邻变化率，可视化诊断
  [阈值]   用百分位数自动确定阈值（方法一）
  [剪枝]   跳过冗余层，对比推理速度 + MSE

对比实验设计：
  - Baseline（无剪枝）
  - 方法一：Δ_rel 变化率引导（数据驱动阈值：P10~P80）

Usage:
    python experiments/timer_layer_mi_pruning.py \
        --mi_result_dir ./results/timer_mi_ksg_pca/Timer_MI_20260525_120000 \
        --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 \
        --seq_len 672 --pred_len 96 --label_len 48 \
        --batch_size 64 --max_samples 1000 \
        --percentiles 10,20,30,40,50,60,70,80,90,95 \
        --gpu 0 --out_dir ./results/timer_layer_mi_pruning/ --model_id etth1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_loader_benchmark import CIDatasetBenchmark


# ============================================================================
# Nature Figure Style
# ============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "orange":         "#E07B39",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}


def add_panel_label(ax, label, x=-0.08, y=1.06, fontsize=10,
                    fontweight="bold", color="black"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha="left", va="bottom")


def finalize_figure(fig, out_path, dpi=300, pad=1.2):
    from pathlib import Path
    fig.tight_layout(pad=pad)
    base = Path(out_path)
    os.makedirs(base.parent, exist_ok=True)
    base = base.with_suffix("")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {base}.{{svg,pdf,png}}")


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(trues: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    t = trues.reshape(-1)
    p = preds.reshape(-1)
    mse = float(np.mean((t - p) ** 2))
    mae = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((t - p) / (np.abs(t) + 1e-8))) * 100)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ============================================================================
# Config & Model
# ============================================================================

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer_model(ckpt_path: str, patch_len: int, stride: int,
                      d_model: int, d_ff: int, e_layers: int,
                      n_heads: int, dropout: float,
                      seq_len: int, pred_len: int):
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
    from models.Timer import Model
    model = Model(config)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ============================================================================
# Timer Forward (with layer skip)
# ============================================================================

def _forecast_forward(
    model,
    seq_x: torch.Tensor,
    device: torch.device,
    skip_layers: set[int],
    pred_len: int,
) -> torch.Tensor:
    """
    Timer 前向传播，支持选择性跳过 encoder 层。

    完整复制 Timer.forecast 内部逻辑：
      1. Non-stationary 归一化
      2. PatchEmbedding（unfold + linear projection）
      3. 可选跳过指定 encoder block
      4. Linear projection → 预测输出

    Args:
        model:       Timer 模型
        seq_x:       [B, seq_len, M] 原始输入
        device:      torch device
        skip_layers: 要跳过的层索引集合
        pred_len:    预测长度

    Returns:
        dec_out: [B, pred_len, M] 预测结果
    """
    core = _unwrap(model)
    B, L, M = seq_x.shape

    means = seq_x.mean(1, keepdim=True).detach()
    stdev = torch.sqrt(
        torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    x_norm = (seq_x - means) / stdev

    x_perm = x_norm.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x_perm)
    BM, N, D = dec_in.shape

    from utils.masking import TriangularCausalMask
    causal_mask = TriangularCausalMask(BM, N, device=device)

    hidden = dec_in
    num_layers = len(core.decoder.attn_layers)

    with torch.no_grad():
        for li in range(num_layers):
            if li in skip_layers:
                continue
            hidden, _, _ = core.decoder.attn_layers[li](
                hidden, attn_mask=causal_mask)

        if core.decoder.norm is not None:
            hidden = core.decoder.norm(hidden)

    dec_out = core.proj(hidden)

    dec_out = dec_out.view(B, n_vars, N, core.patch_len)
    dec_out = dec_out.mean(dim=1)
    dec_out = dec_out.reshape(B, n_vars, -1).transpose(1, 2)
    dec_out = dec_out[:, -pred_len:, :]
    dec_out = dec_out * stdev + means

    return dec_out


# ============================================================================
# 推理函数
# ============================================================================

def forecast_baseline(
    model,
    data_loader,
    device: torch.device,
    pred_len: int,
    max_samples: int = 0,
    desc: str = "Baseline",
) -> tuple[np.ndarray, float]:
    """
    Timer Baseline 推理（无剪枝）。
    返回 (preds, avg_time_per_sample_ms)。
    preds shape: [N, pred_len, n_vars]
    """
    core = _unwrap(model)
    preds_list: list[np.ndarray] = []
    times_list: list[float] = []
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y, *_ in tqdm(data_loader, desc=desc, leave=False):
            B = batch_x.shape[0]
            total_samples += B

            seq_x = batch_x.float().to(device)

            start = time.time()
            dec_out = _forecast_forward(
                model, seq_x, device,
                skip_layers=set(), pred_len=pred_len)
            elapsed = time.time() - start
            times_list.append(elapsed / B)

            preds_list.append(dec_out.cpu().numpy())

            if max_samples > 0 and total_samples >= max_samples:
                break

    preds = np.concatenate(preds_list, axis=0)
    avg_time = float(np.mean(times_list)) if times_list else 0.0
    return preds, avg_time


def forecast_pruned(
    model,
    data_loader,
    skip_layers: set[int],
    device: torch.device,
    pred_len: int,
    max_samples: int = 0,
    desc: str = "Pruned",
) -> tuple[np.ndarray, float]:
    """
    Timer 剪枝推理（跳过指定层）。
    返回 (preds, avg_time_per_sample_ms)。
    """
    core = _unwrap(model)
    preds_list: list[np.ndarray] = []
    times_list: list[float] = []
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y, *_ in tqdm(data_loader, desc=desc, leave=False):
            B = batch_x.shape[0]
            total_samples += B

            seq_x = batch_x.float().to(device)

            start = time.time()
            dec_out = _forecast_forward(
                model, seq_x, device,
                skip_layers=skip_layers, pred_len=pred_len)
            elapsed = time.time() - start
            times_list.append(elapsed / B)

            preds_list.append(dec_out.cpu().numpy())

            if max_samples > 0 and total_samples >= max_samples:
                break

    preds = np.concatenate(preds_list, axis=0)
    avg_time = float(np.mean(times_list)) if times_list else 0.0
    return preds, avg_time


# ============================================================================
# MI 剪枝分析
# ============================================================================

def compute_layer_mi_stats(layers_dict: dict, n_layers: int) -> dict:
    """
    计算每层平均 I(H_l, Y) 及相关统计量。
    基于现有 MI 结果（hsic_curve）的 per-patch 均值。
    """
    layer_mean: list[float] = []
    layer_std: list[float] = []
    layer_min: list[float] = []
    layer_max: list[float] = []

    for li in range(n_layers):
        curve = layers_dict[str(li)]['hsic_curve']
        arr = np.array(curve, dtype=np.float64)
        layer_mean.append(float(np.nanmean(arr)))
        layer_std.append(float(np.nanstd(arr)))
        layer_min.append(float(np.nanmin(arr)))
        layer_max.append(float(np.nanmax(arr)))

    layer_mean = np.array(layer_mean, dtype=np.float64)

    delta_abs = np.abs(np.diff(layer_mean))
    delta_rel = delta_abs / (layer_mean[:-1] + 1e-10)

    return {
        "layer_mean": layer_mean,
        "layer_std": np.array(layer_std),
        "layer_min": np.array(layer_min),
        "layer_max": np.array(layer_max),
        "delta_abs": delta_abs,
        "delta_rel": delta_rel,
    }


def find_prune_candidates_by_percentile(
    stats: dict, percentile: float,
) -> tuple[list[int], dict]:
    """
    方法一（数据驱动）：基于相对变化率分位数确定阈值，找出可剪层。

    逻辑：
      1. Δ_rel < threshold（变化小）→ 该层相对前一层信息量几乎不变
      2. 不剪首尾层（L0 和最后一层）
      3. 只剪中间层：变化小的那一层本身（l 是 l-1 和 l+1 之间的中间层）

    Returns:
        skip_layers: 要跳过的层索引列表
        info: 诊断信息字典
    """
    delta_rel = stats["delta_rel"]
    n = len(stats["layer_mean"])
    threshold = float(np.percentile(delta_rel, percentile))

    skip_layers: list[int] = []
    for l in range(1, n - 1):
        if delta_rel[l - 1] < threshold:
            skip_layers.append(l)

    info = {
        "method": "percentile",
        "percentile": percentile,
        "threshold": threshold,
        "delta_rel": delta_rel.tolist(),
    }
    return skip_layers, info


# ============================================================================
# 绘图函数
# ============================================================================

def plot_mi_diagnostic(
    layers_dict: dict,
    stats: dict,
    n_layers: int,
    output_dir: str,
):
    """
    Nature Figure 1: MI 诊断图。
    展示每层平均 I(H,Y)、变化率 Δ、阈值线。
    """
    layer_mean = stats["layer_mean"]
    layer_std = stats["layer_std"]
    delta_abs = stats["delta_abs"]
    delta_rel = stats["delta_rel"]

    percentiles_to_show = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    thresholds_pct = {p: float(np.percentile(delta_rel, p)) for p in percentiles_to_show}
    default_th = thresholds_pct[50]  # P50 as the reference threshold marker in the bar chart

    mi_curves = [layers_dict[str(li)]['hsic_curve'] for li in range(n_layers)]
    mi_matrix = np.array(mi_curves)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.4)

    # a: 热力图
    ax = axes[0, 0]
    vmin = np.nanmin(mi_matrix)
    vmax = np.nanmax(mi_matrix)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
    im = ax.imshow(mi_matrix, aspect="auto", cmap="RdYlBu_r", norm=norm)
    ax.set_xlabel("Patch index", fontsize=7)
    ax.set_ylabel("Layer", fontsize=7)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=6)
    n_patches = mi_matrix.shape[1]
    ax.set_xticks([0, n_patches // 2, n_patches - 1])
    ax.set_xticklabels(["0", f"{n_patches//2}", f"{n_patches-1}"], fontsize=6)
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("I(H_l, Y)", fontsize=6)
    ax.set_title("a  Per-layer I(H, Y) heatmap", fontsize=7.5, pad=3)
    add_panel_label(ax, "a", y=1.08)

    # b: MI 曲线 + Δ_rel 柱状
    ax = axes[0, 1]
    ax2 = ax.twinx()
    x = np.arange(n_layers)

    bar_colors = [
        PALETTE["red_strong"] if d >= default_th else PALETTE["neutral_light"]
        for d in delta_rel
    ]
    ax.bar(x[1:], delta_rel * 100, width=0.6,
           color=bar_colors, edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axhline(default_th * 100, color=PALETTE["red_strong"], ls="--",
               lw=1.0, label=f"thr={default_th*100:.1f}%")

    ax2.plot(x, layer_mean, "o-", color=PALETTE["blue_main"],
             lw=1.5, ms=3.5, zorder=5, label=r"$\overline{I}$(H, Y)")
    ax2.fill_between(x, layer_mean - layer_std,
                      layer_mean + layer_std, color=PALETTE["blue_main"], alpha=0.15)

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel(r"$\Delta_{\mathrm{rel}}$ (%)", fontsize=7, color=PALETTE["red_strong"])
    ax2.set_ylabel(r"$\overline{I}$(H, Y)", fontsize=7, color=PALETTE["blue_main"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{li}" for li in x], fontsize=6)
    ax.tick_params(labelsize=6, colors=PALETTE["red_strong"])
    ax2.tick_params(labelsize=6, colors=PALETTE["blue_main"])
    ax.set_title("b  MI and layer-wise change rate", fontsize=7.5, pad=3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, handlelength=1.5)
    add_panel_label(ax, "b", y=1.08)

    # c: Δ_rel 分布直方图
    ax = axes[1, 0]
    p95_val = float(np.percentile(delta_rel, 95)) * 1.1
    bins = np.linspace(0, p95_val, 20)
    ax.hist(delta_rel * 100, bins=bins, color=PALETTE["blue_main"],
            edgecolor="white", linewidth=0.5, alpha=0.8)
    for p in percentiles_to_show:
        t = thresholds_pct[p]
        ax.axvline(t * 100, ls="--", lw=0.8, color=PALETTE["neutral_mid"],
                   label=f"P{p}={t*100:.2f}%")
    ax.set_xlabel(r"$\Delta_{\mathrm{rel}}$ (%)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    ax.legend(fontsize=5, handlelength=1.5)
    ax.set_title(r"c  Distribution of $\Delta_{\mathrm{rel}}$", fontsize=7.5, pad=3)
    add_panel_label(ax, "c", y=1.08)

    # d: 各百分位剪层数
    ax = axes[1, 1]
    prune_counts = {}
    for p in percentiles_to_show:
        skips, _ = find_prune_candidates_by_percentile(stats, p)
        prune_counts[p] = skips

    pcts = list(prune_counts.keys())
    counts = [len(prune_counts[p]) for p in pcts]
    bar_colors2 = [PALETTE["blue_main"] if c > 0 else PALETTE["neutral_light"] for c in counts]
    ax.bar([str(p) for p in pcts], counts, color=bar_colors2,
           edgecolor="white", linewidth=0.5)
    for i, (p, c) in enumerate(zip(pcts, counts)):
        skips = prune_counts[p]
        ax.text(i, c + 0.05, f"L{skips}" if skips else "-", ha="center",
                va="bottom", fontsize=5, color=PALETTE["neutral_dark"])

    ax.set_xlabel("Percentile (%)", fontsize=7)
    ax.set_ylabel("Number of layers to prune", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    ax.set_title("d  Layers pruned at each percentile", fontsize=7.5, pad=3)
    add_panel_label(ax, "d", y=1.08)

    fig.suptitle("Fig. 1  Timer MI-based layer pruning diagnostic", fontsize=8, fontweight="bold")
    finalize_figure(fig, os.path.join(output_dir, "fig1_mi_diagnostic"), dpi=300, pad=0.8)


def plot_pruning_results(
    baseline_metrics: dict,
    baseline_time: float,
    results_by_method: dict,
    output_dir: str,
):
    """
    Nature Figure 2: 剪枝效果对比图（多方法汇总）。
    """
    methods = sorted(results_by_method.keys())
    n_pruned_all = [results_by_method[m]["n_pruned"] for m in methods]
    mses = [results_by_method[m]["metrics"]["MSE"] for m in methods]
    maes = [results_by_method[m]["metrics"]["MAE"] for m in methods]
    times = [results_by_method[m]["time"] for m in methods]
    speedups = [baseline_time / t if t > 0 else 1.0 for t in times]
    deltas_mse = [(m - baseline_metrics["MSE"]) / baseline_metrics["MSE"] * 100
                   for m in mses]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.4)

    # a: MSE vs 剪层数
    ax = axes[0, 0]
    ax.plot(n_pruned_all, mses, "o-", color=PALETTE["blue_main"],
            lw=1.5, ms=4, zorder=5)
    ax.axhline(baseline_metrics["MSE"], color=PALETTE["neutral_mid"],
               ls=":", lw=1.2, label=f"Baseline={baseline_metrics['MSE']:.4f}")
    for i, (n, m) in enumerate(zip(n_pruned_all, methods)):
        short_label = m if len(m) <= 12 else m[:12] + "..."
        ax.annotate(short_label, (n, mses[i]), textcoords="offset points",
                    xytext=(4, 4), fontsize=5, color=PALETTE["neutral_dark"])

    ax.set_xlabel("Number of layers pruned", fontsize=7)
    ax.set_ylabel("MSE", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(fontsize=6, handlelength=1.5)
    ax.set_title("a  MSE vs. number of layers pruned", fontsize=7.5, pad=3)
    add_panel_label(ax, "a", y=1.08)

    # b: Speedup vs ΔMSE (Pareto)
    ax = axes[0, 1]
    scatter = ax.scatter(
        speedups, deltas_mse,
        c=n_pruned_all, cmap="plasma",
        s=50, zorder=5, edgecolors="white", linewidth=0.5,
    )
    ax.axhline(0, color=PALETTE["neutral_dark"], ls="--", lw=0.8)
    ax.axvline(1.0, color=PALETTE["neutral_mid"], ls=":", lw=0.8)

    for i, m in enumerate(methods):
        short_label = m if len(m) <= 12 else m[:12] + "..."
        ax.annotate(short_label, (speedups[i], deltas_mse[i]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=5, color=PALETTE["neutral_dark"])

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.9, pad=0.02)
    cbar.ax.set_ylabel("Layers pruned", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_xlabel("Speedup (x)", fontsize=7)
    ax.set_ylabel(r"$\Delta$MSE (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_title(r"b  Pareto frontier: Speedup vs. $\Delta$MSE", fontsize=7.5, pad=3)
    add_panel_label(ax, "b", y=1.08)

    # c: Speedup 条形图
    ax = axes[1, 0]
    bar_colors3 = [
        PALETTE["blue_main"] if s > 1.0 else PALETTE["neutral_light"]
        for s in speedups
    ]
    ax.bar(range(len(methods)), speedups, color=bar_colors3,
           edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color=PALETTE["neutral_dark"], ls="--", lw=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m[:8] for m in methods], fontsize=5, rotation=30, ha="right")
    ax.set_xlabel("Method", fontsize=7)
    ax.set_ylabel("Speedup (x)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)

    for i, s in enumerate(speedups):
        ax.text(i, s + 0.01, f"{s:.2f}x", ha="center", va="bottom",
                fontsize=5, color=PALETTE["neutral_dark"])

    ax.set_title("c  Inference speedup per method", fontsize=7.5, pad=3)
    add_panel_label(ax, "c", y=1.08)

    # d: MSE + MAE 双轴对比
    ax = axes[1, 1]
    x_pos = range(len(methods))
    ax2 = ax.twinx()
    ax.bar(x_pos, mses, width=0.4, label="MSE",
           color=PALETTE["blue_main"], edgecolor="white", linewidth=0.5, alpha=0.8)
    ax2.bar([i + 0.4 for i in x_pos], maes, width=0.4, label="MAE",
             color=PALETTE["red_strong"], edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axhline(baseline_metrics["MSE"], color=PALETTE["blue_main"],
               ls=":", lw=1.0, alpha=0.6)
    ax2.axhline(baseline_metrics["MAE"], color=PALETTE["red_strong"],
                ls=":", lw=1.0, alpha=0.6)

    ax.set_xticks([i + 0.2 for i in x_pos])
    ax.set_xticklabels([m[:8] for m in methods], fontsize=5, rotation=30, ha="right")
    ax.set_xlabel("Method", fontsize=7)
    ax.set_ylabel("MSE", fontsize=7, color=PALETTE["blue_main"])
    ax2.set_ylabel("MAE", fontsize=7, color=PALETTE["red_strong"])
    ax.tick_params(labelsize=6, colors=PALETTE["blue_main"])
    ax2.tick_params(labelsize=6, colors=PALETTE["red_strong"])
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, handlelength=1.5)
    ax.set_title("d  MSE & MAE comparison", fontsize=7.5, pad=3)
    add_panel_label(ax, "d", y=1.08)

    fig.suptitle("Fig. 2  Timer layer pruning: speed and accuracy", fontsize=8, fontweight="bold")
    finalize_figure(fig, os.path.join(output_dir, "fig2_pruning_effects"), dpi=300, pad=0.8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Timer 层间 MI 变化率引导的 Transformer 层剪枝实验"
    )
    # Data
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--data_type", type=str, default="ETTh1",
                        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最大测试样本数 (0=全部)")
    # MI result
    parser.add_argument("--mi_result_dir", type=str,
                        default="./results/timer_mi_ksg_pca/Timer_MI_20260514_073311")
    parser.add_argument("--model_id", type=str, default="etth1")
    # Thresholds
    parser.add_argument("--percentiles", type=str, default="10,20,30,40,50,60,70,80",
                        help="逗号分隔的分位数列表 (方法一)")
    # Model
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Runtime
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./results/timer_layer_mi_pruning/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"Timer_LayerPrune_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    percentiles = [float(p) for p in args.percentiles.split(",")]

    print("=" * 70)
    print("  Timer MI 变化率引导的 Transformer 层剪枝实验")
    print("  方法一：相对变化率 Δ_rel = |I_l - I_{l-1}| / I_{l-1}（百分位数阈值 P10~P80）")
    print("=" * 70)
    print(f"  mi_result_dir : {args.mi_result_dir}")
    print(f"  data          : {args.data}")
    print(f"  data_type     : {args.data_type}")
    print(f"  seq_len       : {args.seq_len}")
    print(f"  pred_len      : {args.pred_len}")
    print(f"  patch_len     : {args.patch_len}")
    print(f"  percentiles   : {percentiles}")
    print(f"  device        : {device}")
    print(f"  output_dir    : {output_dir}")
    print("=" * 70)

    # ── Phase 0: 加载 MI 结果 ───────────────────────────────────────────────
    print("\n>>> Phase 0: 加载 MI 结果...")
    mi_json_path = os.path.join(args.mi_result_dir, f"global_mi_peaks_{args.model_id}.json")
    if not os.path.exists(mi_json_path):
        raise FileNotFoundError(
            f"MI result JSON not found: {mi_json_path}\n"
            f"Please run timer_mi_ksg_pca.py first to generate MI scores."
        )

    with open(mi_json_path) as f:
        mi_summary = json.load(f)

    n_layers = mi_summary['num_layers']
    layers_dict = mi_summary['layers']
    print(f"  MI 结果: {n_layers} 层")

    # ── Phase 1: 计算层 MI 统计量 ─────────────────────────────────────────
    print("\n>>> Phase 1: 计算层 MI 统计量...")
    stats = compute_layer_mi_stats(layers_dict, n_layers)

    layer_mean = stats["layer_mean"]
    delta_rel = stats["delta_rel"]

    print(f"  每层平均 I(H,Y): {[f'{v:.4f}' for v in layer_mean]}")
    print(f"  Δ_rel 范围: [{delta_rel.min():.4f}, {delta_rel.max():.4f}]")

    print("\n  方法一（百分位数阈值）:")
    for p in percentiles:
        skips, info = find_prune_candidates_by_percentile(stats, p)
        print(f"    P{int(p):2d}: threshold={info['threshold']:.4f}  →  skip {skips}")

    # ── Phase 2: 加载数据集 ────────────────────────────────────────────────
    print("\n>>> Phase 2: 加载数据集...")

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
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"  变量数: {n_vars}, 测试集: {len(test_dataset)}")

    # 收集 ground truth
    all_y_raw = []
    with torch.no_grad():
        for bx, by, *_ in test_loader:
            all_y_raw.append(by.numpy())
    all_y_raw = np.concatenate(all_y_raw, axis=0)
    print(f"  真值形状: {all_y_raw.shape}")

    # ── Phase 3: 加载 Timer 模型 ──────────────────────────────────────────
    print("\n>>> Phase 3: 加载 Timer 模型...")

    model = build_timer_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        stride=args.patch_len,
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

    detected_layers = len(_unwrap(model).decoder.attn_layers)
    patch_size = _unwrap(model).patch_len
    print(f"  模型: Timer (ckpt={args.ckpt_path})")
    print(f"  patch_len={patch_size}, d_model={args.d_model}, "
          f"num_layers={detected_layers}")
    print(f"  设备: {device}")

    # ── Phase 4: Baseline 推理 ─────────────────────────────────────────────
    print("\n>>> Phase 4: Baseline 推理 (无剪枝)...")

    baseline_preds, baseline_time = forecast_baseline(
        model=model,
        data_loader=test_loader,
        device=device,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        desc="Baseline inference",
    )

    min_n = min(all_y_raw.shape[0], baseline_preds.shape[0])
    preds_ot = baseline_preds[:min_n, :, 0]
    trues_ot = all_y_raw[:min_n, :, 0]
    baseline_metrics = compute_metrics(trues_ot, preds_ot)
    print(f"  Baseline  MSE={baseline_metrics['MSE']:.6f}, "
          f"MAE={baseline_metrics['MAE']:.6f}, "
          f"Time={baseline_time*1000:.2f}ms/sample")

    # ── Phase 5: 各方法剪枝实验 ─────────────────────────────────────────
    print("\n>>> Phase 5: 剪枝实验...")
    results_by_method: dict = {}

    # Baseline
    results_by_method["Baseline"] = {
        "metrics": dict(baseline_metrics),
        "time": baseline_time,
        "skip_layers": [],
        "n_pruned": 0,
        "method": "baseline",
    }

    # 方法一：百分位数阈值
    for p in percentiles:
        method_name = f"P{int(p)}"
        print(f"\n  === {method_name} (Percentile P{int(p):2d}) ===")
        skip_layers, info = find_prune_candidates_by_percentile(stats, p)
        skip_set = set(skip_layers)
        n_pruned = len(skip_set)

        if n_pruned == 0:
            print(f"    无可剪层，跳过")
            results_by_method[method_name] = {
                "metrics": dict(baseline_metrics),
                "time": baseline_time,
                "skip_layers": skip_layers,
                "n_pruned": 0,
                "info": info,
            }
            continue

        print(f"    阈值 Δ_rel < {info['threshold']:.4f}")
        print(f"    跳过层: {skip_layers}")
        print(f"    保留层: {[li for li in range(n_layers) if li not in skip_set]}")

        pruned_preds, pruned_time = forecast_pruned(
            model=model,
            data_loader=test_loader,
            skip_layers=skip_set,
            device=device,
            pred_len=args.pred_len,
            max_samples=args.max_samples,
            desc=f"{method_name} pruned inference",
        )

        min_p = min(trues_ot.shape[0], pruned_preds.shape[0])
        pruned_preds_ot = pruned_preds[:min_p, :, 0]
        pruned_trues_ot = trues_ot[:min_p]
        pruned_metrics = compute_metrics(pruned_trues_ot, pruned_preds_ot)

        speedup = baseline_time / pruned_time if pruned_time > 0 else 1.0
        delta_mse_pct = (pruned_metrics["MSE"] - baseline_metrics["MSE"]) \
            / baseline_metrics["MSE"] * 100
        delta_mae_pct = (pruned_metrics["MAE"] - baseline_metrics["MAE"]) \
            / baseline_metrics["MAE"] * 100

        print(f"    MSE={pruned_metrics['MSE']:.6f}, "
              f"MAE={pruned_metrics['MAE']:.6f}, "
              f"ΔMSE={delta_mse_pct:+.2f}%, "
              f"ΔMAE={delta_mae_pct:+.2f}%, "
              f"Time={pruned_time*1000:.2f}ms/sample, "
              f"Speedup={speedup:.3f}x")

        results_by_method[method_name] = {
            "metrics": pruned_metrics,
            "time": pruned_time,
            "skip_layers": skip_layers,
            "n_pruned": n_pruned,
            "info": info,
        }

    # ── Phase 6: 绘图 ────────────────────────────────────────────────────
    print("\n>>> Phase 6: 绘图...")

    plot_mi_diagnostic(
        layers_dict=layers_dict,
        stats=stats,
        n_layers=n_layers,
        output_dir=output_dir,
    )

    plot_pruning_results(
        baseline_metrics=baseline_metrics,
        baseline_time=baseline_time,
        results_by_method=results_by_method,
        output_dir=output_dir,
    )

    # ── Phase 7: 保存结果 ────────────────────────────────────────────────
    print("\n>>> Phase 7: 保存结果...")

    results_json = {
        "model": "Timer",
        "ckpt_path": args.ckpt_path,
        "num_layers": n_layers,
        "num_patches": int(args.seq_len // args.patch_len),
        "patch_len": args.patch_len,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "percentiles": percentiles,
        "baseline": {
            "metrics": baseline_metrics,
            "time_per_sample_ms": baseline_time * 1000,
        },
        "methods": {
            method: {
                "n_pruned": results_by_method[method]["n_pruned"],
                "skip_layers": results_by_method[method]["skip_layers"],
                "metrics": results_by_method[method]["metrics"],
                "time_per_sample_ms": results_by_method[method]["time"] * 1000,
                "speedup": baseline_time / results_by_method[method]["time"]
                    if results_by_method[method]["time"] > 0 else 1.0,
                "delta_mse_pct": (
                    results_by_method[method]["metrics"]["MSE"] - baseline_metrics["MSE"]
                ) / baseline_metrics["MSE"] * 100,
            }
            for method in results_by_method
        },
        "stats": {
            "layer_mean": layer_mean.tolist(),
            "delta_rel": delta_rel.tolist(),
        },
    }

    out_json = os.path.join(output_dir, "pruning_results.json")
    with open(out_json, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"  结果已保存: {out_json}")

    # ── 打印汇总表 ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  结果汇总")
    print(f"{'='*80}")
    print(f"  [Baseline] MSE={baseline_metrics['MSE']:.6f}, "
          f"MAE={baseline_metrics['MAE']:.6f}, "
          f"Time={baseline_time*1000:.2f}ms/sample")

    sorted_methods = sorted(results_by_method.keys(),
                            key=lambda m: results_by_method[m]["n_pruned"])
    for method in sorted_methods:
        r = results_by_method[method]
        speedup = baseline_time / r["time"] if r["time"] > 0 else 1.0
        delta_mse = (r["metrics"]["MSE"] - baseline_metrics["MSE"]) \
            / baseline_metrics["MSE"] * 100
        delta_mae = (r["metrics"]["MAE"] - baseline_metrics["MAE"]) \
            / baseline_metrics["MAE"] * 100
        print(f"  [{method:>12}] skip={str(r['skip_layers']):>15s}, "
              f"MSE={r['metrics']['MSE']:.6f}, "
              f"MAE={r['metrics']['MAE']:.6f}, "
              f"ΔMSE={delta_mse:+.2f}%, "
              f"ΔMAE={delta_mae:+.2f}%, "
              f"Speedup={speedup:.3f}x")
    print(f"\n  输出目录: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
