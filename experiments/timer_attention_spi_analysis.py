#!/usr/bin/env python3
"""
Timer Attention Hub 验证实验 — 多层版
=====================================
对所有 Transformer 层执行 I(h_x, h_y) vs 入度注意力相关性分析。

从 timer_mi_ksg_pca.py 生成的 JSON 文件读取 MI 曲线，与注意力入度对齐做相关性分析。

实验逻辑：
  1. 加载预训练 Timer，通过 attention hook 提取所有层的自注意力权重 [B, H, S, S]。
  2. 对注意力头求均值，再沿 Query 维对列求和，计算 in-degree：
       in_degree[b, j] = (1/H) * Σ_i Attn[b, h, i, j]
  3. 从 JSON 读取每层的 I(h_x, h_y) 曲线 [n_patches]。
  4. 将 IHY 曲线 tile 到每个样本每个变量，与 in_degree 逐元素对齐。
  5. 计算 Pearson/Spearman 相关系数，绘制散点图。

Usage:
  python experiments/timer_attention_spi_analysis.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv \
      --seq_len 672 --pred_len 96 --patch_len 96 \
      --e_layers 8 --gpu 0 \
      --mi_result_dir ./outputs/timer_mi_ksg_pca/run_YYYYMMDD_HHMMSS \
      --out_dir ./outputs/timer_attention_spi
"""

import argparse
import gc
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import EncoderLayer


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Config & Model Builder
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(ckpt_path: str, patch_len: int, stride: int,
                d_model: int, d_ff: int, e_layers: int,
                n_heads: int, dropout: float,
                seq_len: int, pred_len: int) -> Model:
    cfg = Config(
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
    model = Model(cfg)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MI 曲线加载（从 JSON）
# ═══════════════════════════════════════════════════════════════════════════════

def load_mi_curve(
    mi_result_dir: str,
    target_layer: int,
    metric: str = "IHY",
) -> dict:
    """
    从 timer_mi_ksg_pca.py 生成的 global_mi_peaks_*.json 中读取指定层的 MI 曲线。

    Args:
        mi_result_dir: timer_mi_ksg_pca 输出目录（包含 global_mi_peaks_*.json）
        target_layer: 目标层 index
        metric: "IHY" → I(H,Y) / "IXH" → I(X,H)

    Returns:
        dict with keys:
          "score_per_patch" : [n_patches] 一维数组
          "n_patches"       : int
          "mi_hy_curve"     : [n_patches] I(H,Y) 曲线
          "mi_xh_curve"     : [n_patches] I(X,H) 曲线
          "metric_label"    : str, e.g. r"$I(H, Y)$"
    """
    pattern = os.path.join(mi_result_dir, "global_mi_peaks_*.json")
    matched = glob.glob(pattern)
    if not matched:
        raise FileNotFoundError(
            f"未找到 MI 结果文件: {pattern}\n"
            f"请检查 --mi_result_dir 参数。"
        )
    json_path = matched[0]
    print(f"  [MI Loader] 读取 MI 结果: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    layers_dict = data.get("layers", {})
    layer_key = str(target_layer)
    if layer_key not in layers_dict:
        raise ValueError(
            f"MI JSON 中未找到 layer={target_layer}，"
            f"可用层: {list(layers_dict.keys())}"
        )

    layer_info = layers_dict[layer_key]
    mi_hy = np.array(layer_info["hsic_curve"], dtype=np.float64)
    mi_xh = np.array(layer_info["mi_xh_curve"], dtype=np.float64)

    if metric == "IHY":
        score = mi_hy
        metric_label = r"$I(H, Y)$"
    elif metric == "IXH":
        score = mi_xh
        metric_label = r"$I(X, H)$"
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'IHY' or 'IXH'.")

    print(
        f"  [MI Loader] layer={target_layer}, n_patches={len(score)}, "
        f"metric={metric_label}"
    )

    return {
        "score_per_patch": score,
        "n_patches": len(score),
        "mi_hy_curve": mi_hy,
        "mi_xh_curve": mi_xh,
        "metric_label": metric_label,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: In-degree Attention Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_in_degree_attention(attn_tensor: torch.Tensor) -> torch.Tensor:
    """
    从原始注意力权重计算每个 Token 的入度注意力权重。

    语义说明（以 Timer FullAttention 格式为准）：
      attn_tensor[b, h, i, j] = softmax_j( Q[b,h,i,:] · K[b,h,:,j] / √d )
                               = 位置 i 的 Query 对位置 j 的 Key 的注意力权重
                               = 信息从 j（source / Key）流向 i（target / Query）

    因此，位置 j 的入度（in-degree）= Σ_i attn_tensor[:, :, i, j]
    即所有 Query 位置 i 对 j 的注意力权重之和，反映 j 被"汇聚"了多少信息。

    步骤：
      1. 对 H 维度求均值：attn_mean[b, i, j]        —— [B, N, N]
      2. 对 Query 维度（dim=1，axis=i）求列和：
         in_degree[b, j] = Σ_i attn_mean[b, i, j]    —— [B, N]

    Args:
        attn_tensor: [B, H, N, N] 原始注意力权重（已 detach）

    Returns:
        in_degree: [B, N] 每个样本每个 Token 位置的入度注意力权重
    """
    attn_mean = attn_tensor.mean(dim=1)   # [B, N, N] — heads averaging
    in_degree = attn_mean.sum(dim=1)       # [B, N] — column sum (in-degree)
    return in_degree


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: 统计分析
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correlations(
    score_flat: np.ndarray,
    in_degree_flat: np.ndarray,
) -> dict:
    """计算 Pearson 和 Spearman 相关系数。"""
    mask = np.isfinite(score_flat) & np.isfinite(in_degree_flat)
    x = score_flat[mask].astype(np.float64)
    y = in_degree_flat[mask].astype(np.float64)

    if len(x) < 3:
        raise ValueError(f"有效数据点不足（{len(x)} < 3）")

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_samples": int(len(x)),
        "n_nan_excluded": int(len(score_flat) - len(x)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: 绘图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_attention_hub_scatter(
    score_flat: np.ndarray,
    in_degree_flat: np.ndarray,
    correlations: dict,
    target_layer: int,
    n_patches: int,
    out_dir: str,
    metric_label: str = r"$I(H, Y)$",
) -> str:
    """绘制 MI Score vs. In-degree Attention 的散点图 + 趋势拟合线。"""
    mask = np.isfinite(score_flat) & np.isfinite(in_degree_flat)
    x = score_flat[mask]
    y = in_degree_flat[mask]

    slope, intercept, r_val, p_val, std_err = __import__("scipy.stats", fromlist=["linregress"]).linregress(x, y)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("#f8f9fa")

    norm = plt.Normalize(vmin=x.min(), vmax=x.max())
    cmap = plt.cm.get_cmap("viridis")
    colors = cmap(norm(x))

    scatter = ax.scatter(x, y, c=colors, alpha=0.4, s=12,
                        edgecolors="none", rasterized=True)

    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color="#d62728", linewidth=2.5,
            label=f"Trend: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_val**2:.4f}")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(f"{metric_label} (color-coded)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    r_p = correlations["pearson_r"]
    r_s = correlations["spearman_r"]
    n = correlations["n_samples"]

    stat_text = (
        f"Pearson  r = {r_p:+.4f}  (p = {correlations['pearson_p']:.2e})\n"
        f"Spearman ρ = {r_s:+.4f}  (p = {correlations['spearman_p']:.2e})\n"
        f"N tokens = {n:,} (excl. {correlations['n_nan_excluded']:,} NaN)"
    )
    ax.text(0.97, 0.97, stat_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.85), family="monospace")

    ax.set_xlabel(f"{metric_label} Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("In-degree Attention Weight (Total Received Attention)",
                  fontsize=12, fontweight="bold")
    ax.set_title(
        f"{metric_label} vs. In-degree Attention Weight per Token\n"
        f"Target Layer = {target_layer}  |  N_patches = {n_patches}",
        fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.25, color="gray", linestyle="--")
    ax.tick_params(labelsize=10)

    out_path = os.path.join(out_dir, f"attn_hub_spi_vs_indegree_L{target_layer}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  散点图已保存: {out_path}")
    return out_path


def plot_multi_layer_comparison(
    layer_results: dict,
    out_dir: str,
) -> str:
    """多层相关系数对比柱状图。"""
    layers = sorted(layer_results.keys())
    pearson_rs = [layer_results[l]["pearson_r"] for l in layers]
    spearman_rs = [layer_results[l]["spearman_r"] for l in layers]
    n_samples = [layer_results[l]["n_samples"] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, pearson_rs, width, label="Pearson r",
                   color="steelblue", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, spearman_rs, width, label="Spearman ρ",
                   color="coral", edgecolor="black", alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{li}" for li in layers], fontsize=10)
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation Coefficient", fontsize=11, fontweight="bold")
    ax.set_title("SPI vs. In-degree Attention — Correlation by Layer",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(-1.05, 1.05)

    for bar, v in zip(bars1, pearson_rs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 if v >= 0 else v - 0.05,
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, fontweight="bold")
    for bar, v in zip(bars2, spearman_rs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 if v >= 0 else v - 0.05,
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, fontweight="bold")

    ax2 = axes[1]
    ax2.bar(x, n_samples, color="#2ca02c", edgecolor="black", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{li}" for li in layers], fontsize=10)
    ax2.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax2.set_ylabel("N Tokens (valid)", fontsize=11, fontweight="bold")
    ax2.set_title("Number of Valid Token Data Points per Layer",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.25, axis="y")

    for xi, ni in zip(x, n_samples):
        ax2.text(xi, ni + max(n_samples) * 0.01, f"{ni:,}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Attention Hub Hypothesis — Multi-Layer Correlation Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "attn_hub_multi_layer_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  多层对比图已保存: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  Timer Attention Hub — Multi-Layer Analysis")
    print("  Metric: I(h_x, h_y) vs In-degree Attention")
    print("=" * 70)
    print(f"  ckpt_path     : {args.ckpt_path}")
    print(f"  data_path     : {args.data_path}")
    print(f"  seq_len       : {args.seq_len}")
    print(f"  pred_len      : {args.pred_len}")
    print(f"  patch_len     : {args.patch_len}")
    print(f"  e_layers      : {args.e_layers}")
    print(f"  mi_result_dir : {args.mi_result_dir}")
    print(f"  mi_metric     : {args.mi_metric}")
    print(f"  device        : {device}")
    print("=" * 70)

    # ── 1. Load MI curves from JSON ──────────────────────────────────────────
    print("\n>>> [1/5] 从 JSON 加载 MI 曲线...")
    score_per_patch_dict = {}
    mi_info = {}

    for li in range(args.e_layers):
        try:
            info = load_mi_curve(args.mi_result_dir, li, metric=args.mi_metric)
            score_per_patch_dict[li] = info["score_per_patch"]
            mi_info[li] = info
        except (FileNotFoundError, ValueError) as e:
            print(f"  [WARN] Layer {li}: {e}")
            import sys
            sys.exit(1)

    n_patches = len(list(score_per_patch_dict.values())[0])
    for li, score in score_per_patch_dict.items():
        if len(score) != n_patches:
            raise ValueError(f"Layer {li} MI 长度 ({len(score)}) != n_patches ({n_patches})")
    print(f"  所有层 n_patches = {n_patches}")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    print("\n>>> [2/5] 加载数据集...")
    test_dataset = CIDatasetBenchmark(
        root_path=os.path.join(args.root_path, args.data_path),
        flag="test",
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = test_dataset.n_var
    total_samples = len(test_dataset)
    print(f"  变量数: {n_vars}, 测试集总样本数: {total_samples}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # ── 3. Model ───────────────────────────────────────────────────────────────
    print("\n>>> [3/5] 加载 Timer 模型...")
    model = build_timer(
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
    print(f"  模型加载完成，设备: {device}")

    core = _unwrap(model)
    n_layers = args.e_layers

    # ── 4. Register attention hooks ────────────────────────────────────────────
    print(f"\n>>> [4/5] 注册所有 {n_layers} 层的注意力钩子...")

    collected_attns = [[] for _ in range(n_layers)]

    def _make_hooked_forward(layer_idx, orig_fwd):
        def hooked_fwd(self, queries, keys, values, attn_mask,
                       tau=None, delta=None, _layer_idx=layer_idx, _orig_fwd=orig_fwd):
            V, A, logits = _orig_fwd(queries, keys, values, attn_mask, tau=tau, delta=delta)
            if A is not None:
                collected_attns[_layer_idx].append(A.detach().clone())
            return V, A, logits
        return hooked_fwd

    for li, layer_module in enumerate(core.decoder.attn_layers):
        inner = layer_module.attention
        full_attn = inner.inner_attention
        if not isinstance(full_attn, FullAttention):
            print(f"  警告: Layer {li} 不是 FullAttention，跳过")
            continue
        original_fwd = full_attn.forward
        full_attn._orig_forward = original_fwd
        full_attn.forward = _make_hooked_forward(li, original_fwd).__get__(
            full_attn, FullAttention)
        print(f"  已为 Layer {li} 注册注意力钩子")

    # ── 5. Forward pass: collect in-degree attention ──────────────────────────
    print("\n>>> [5/5] 前向传播，收集注意力入度...")

    n_patches_actual = None
    batch_count = 0
    all_x_emb = []

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(
                test_loader, desc="收集注意力"):
            if args.n_batches is not None and batch_count >= args.n_batches:
                break
            batch_count += 1

            B = seq_x.shape[0]
            sx = seq_x.float().to(device)
            sy = seq_y.float().to(device)

            means = sx.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            xn = (sx - means) / stdev
            yn = (sy - means) / stdev

            x2 = xn.permute(0, 2, 1)
            y2 = yn.permute(0, 2, 1)

            dec_in_x, _ = core.enc_embedding(x2)
            dec_in_y, _ = core.enc_embedding(y2)
            BM, N, D = dec_in_x.shape

            if n_patches_actual is None:
                n_patches_actual = N
                print(f"  n_patches={n_patches_actual}, D={D}, n_vars={n_vars}")

            # Mean-pool patch embeddings across variables (consistent with Timer forward)
            x_emb_view = dec_in_x.view(B, BM // B, N, D).mean(dim=1).float().cpu()
            all_x_emb.append(x_emb_view)

            # Forward through all layers (triggers hook captures)
            hx = dec_in_x
            hy = dec_in_y
            for layer_module in core.decoder.attn_layers:
                hx, _, _ = layer_module(hx, attn_mask=None)
                hy, _, _ = layer_module(hy, attn_mask=None)

            del dec_in_x, dec_in_y, hx, hy
            gc.collect()
            torch.cuda.empty_cache()

    # Restore original forwards
    for layer_module in core.decoder.attn_layers:
        inner = layer_module.attention
        full_attn = inner.inner_attention
        if isinstance(full_attn, FullAttention) and hasattr(full_attn, "_orig_forward"):
            full_attn.forward = full_attn._orig_forward
            delattr(full_attn, "_orig_forward")

    N_total = sum(b.shape[0] for b in all_x_emb)
    print(f"\n  收集完成: {batch_count} batches, N_total={N_total}")

    # ── 6. Compute in-degree per layer ───────────────────────────────────────
    print("\n>>> [6/6] 计算各层入度注意力...")

    in_deg_per_layer = {}
    for li in range(n_layers):
        if len(collected_attns[li]) == 0:
            print(f"  Layer {li}: 无注意力矩阵，跳过")
            in_deg_per_layer[li] = np.zeros((N_total, n_patches_actual))
            continue

        in_deg_list = []
        for attn_batch in collected_attns[li]:
            in_deg = compute_in_degree_attention(attn_batch)   # [B, S]
            B_a, S_a = in_deg.shape
            if S_a == 0:
                continue
            if S_a > n_patches_actual:
                in_deg = in_deg[:, :n_patches_actual]
            elif S_a < n_patches_actual:
                pad = torch.zeros(B_a, n_patches_actual - S_a, device=in_deg.device)
                in_deg = torch.cat([in_deg, pad], dim=1)
            in_deg_list.append(in_deg.cpu())
        if not in_deg_list:
            in_deg_per_layer[li] = np.zeros((N_total, n_patches_actual))
            continue
        in_deg_cat = torch.cat(in_deg_list, dim=0).numpy()
        in_deg_per_layer[li] = in_deg_cat
        print(f"  Layer {li}: in_degree shape={in_deg_cat.shape}, mean={in_deg_cat.mean():.4f}")

    # Detect M (variables per sample) from in_degree shape
    vars_per_sample = in_deg_per_layer[0].shape[0] // N_total
    print(f"\n  Detected vars_per_sample={vars_per_sample}")
    assert N_total * vars_per_sample == in_deg_per_layer[0].shape[0]

    # Align n_patches between attention and MI JSON
    n_p = min(n_patches_actual, n_patches)
    if n_p != n_patches_actual or n_p != n_patches:
        print(f"  对齐 n_patches: attention={n_patches_actual}, MI={n_patches} -> n_patches={n_p}")

    # Flatten in_degree
    in_deg_flat_per_layer = {}
    for li in range(n_layers):
        idg = in_deg_per_layer[li]
        in_deg_flat_per_layer[li] = idg[:, :n_p].flatten()

    # Tile IHY curve to match in_degree shape
    ihy_flat_per_layer = {}
    for li in range(n_layers):
        score_patch = score_per_patch_dict[li][:n_p]
        ihy_tiled = np.tile(score_patch, (N_total * vars_per_sample, 1))
        ihy_flat_per_layer[li] = ihy_tiled.flatten()
        print(f"  Layer {li}: IHY tiled shape={ihy_tiled.shape}, "
              f"in_degree shape={in_deg_flat_per_layer[li].shape}")

    # ── 7. Correlation analysis + plotting ───────────────────────────────────
    print("\n>>> [7/7] 各层相关性分析...")

    all_results = []
    all_scatter_paths = []

    for li in range(n_layers):
        score_flat = ihy_flat_per_layer[li]
        in_deg_flat = in_deg_flat_per_layer[li]
        metric_label = mi_info.get(li, {}).get("metric_label", r"$I(H, Y)$")

        correlations = compute_correlations(score_flat, in_deg_flat)

        print(f"\n{'=' * 60}")
        print(f"  Layer {li} — {metric_label} vs In-degree Attention")
        print(f"{'=' * 60}")
        print(f"  N tokens (valid)  : {correlations['n_samples']:,}")
        print(f"  Pearson  r        : {correlations['pearson_r']:+.4f}"
              f"  (p = {correlations['pearson_p']:.2e})")
        print(f"  Spearman ρ        : {correlations['spearman_r']:+.4f}"
              f"  (p = {correlations['spearman_p']:.2e})")

        scatter_path = plot_attention_hub_scatter(
            score_flat=score_flat,
            in_degree_flat=in_deg_flat,
            correlations=correlations,
            target_layer=li,
            n_patches=n_p,
            out_dir=args.out_dir,
            metric_label=metric_label,
        )
        all_scatter_paths.append(scatter_path)

        all_results.append({
            "layer": li,
            "n_patches": n_p,
            "n_valid": correlations["n_samples"],
            "pearson_r": correlations["pearson_r"],
            "pearson_p": correlations["pearson_p"],
            "spearman_r": correlations["spearman_r"],
            "spearman_p": correlations["spearman_p"],
        })

    # Multi-layer comparison
    if len(all_results) > 1:
        layer_results_dict = {r["layer"]: r for r in all_results}
        plot_multi_layer_comparison(layer_results_dict, args.out_dir)

    # ── 8. Save results JSON ──────────────────────────────────────────────────
    results_json = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "model": "Timer",
        "dataset": args.data_type,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "n_vars": n_vars,
        "patch_len": args.patch_len,
        "n_patches": n_p,
        "n_total_samples": N_total,
        "vars_per_sample": vars_per_sample,
        "mi_result_dir": args.mi_result_dir,
        "mi_metric": args.mi_metric,
        "layer_correlations": {
            str(r["layer"]): {
                "pearson_r": r["pearson_r"],
                "pearman_r": r["spearman_r"],
            } for r in all_results
        },
    }
    json_path = os.path.join(args.out_dir, "attention_hub_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  JSON: {json_path}")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Attention Hub Analysis — 最终结果汇总")
    print("=" * 80)
    print(f"{'Layer':>6} | {'Pearson r':>12} | {'Spearman ρ':>12} | {'N Tokens':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['layer']:>6} | {r['pearson_r']:>+12.4f} | "
              f"{r['spearman_r']:>+12.4f} | {r['n_valid']:>10,}")
    print("=" * 80)
    print(f"\n输出目录: {args.out_dir}")
    print("假设验证:")
    print("  - 若 r > 0：高 SPI Token 收到更多注意力 → 支持「注意力汇聚中心」假设")
    print("  - 若 r ≈ 0：SPI 与注意力无关")
    print("  - 若 r < 0：高 SPI Token 收到更少注意力 → 存在信息旁路机制")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Timer Attention Hub: I(h_x, h_y) vs In-degree Attention — Multi-Layer"
    )
    p.add_argument("--ckpt_path",    type=str,
                   default="checkpoints/Timer_forecast_1.0.ckpt")
    p.add_argument("--root_path",    type=str, default="./datasets/")
    p.add_argument("--data_path",    type=str, default="ETTh1.csv")
    p.add_argument("--data_type",    type=str, default="ETTh1")
    p.add_argument("--seq_len",      type=int, default=672)
    p.add_argument("--pred_len",     type=int, default=96)
    p.add_argument("--patch_len",    type=int, default=96)
    p.add_argument("--stride",       type=int, default=96)
    p.add_argument("--d_model",      type=int, default=1024)
    p.add_argument("--d_ff",         type=int, default=2048)
    p.add_argument("--e_layers",     type=int, default=8)
    p.add_argument("--n_heads",      type=int, default=8)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--n_batches",    type=int, default=None,
                   help="最大 batch 数（None=全部）")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--freq",        type=str, default="h")
    p.add_argument("--model_id",    type=str, default="etth1")
    p.add_argument("--out_dir",     type=str,
                   default="./outputs/timer_attention_spi")
    p.add_argument("--mi_result_dir", type=str,
                   default="./outputs/timer_mi_ksg_pca",
                   help="timer_mi_ksg_pca.py 输出目录（包含 global_mi_peaks_*.json）")
    p.add_argument("--mi_metric",   type=str, default="IHY",
                   choices=["IHY", "IXH"],
                   help="IHY=I(H,Y)（默认），IXH=I(X,H)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()
                   if not callable(v) and not str(k).startswith("_")}, f, indent=2)

    run_pipeline(args)
