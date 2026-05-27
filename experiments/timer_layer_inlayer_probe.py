#!/usr/bin/env python3
"""
Timer 层间 + 层内 Probe 分析（含时序分解模块）

模仿 MOMENT 的 layer_inlayer_probe.py，为 Timer 定制。

[模块一：时序分解（STL Probe）]
  - 利用 STL 分解将原始序列拆解为 趋势（Trend）、周期（Seasonal）和 残差（Residual）
  - 对每个分量分别执行层间 Probe + 层内 Probe
  - 证明模型在不同层对不同时序特征的敏感度不同

[层间 Probe (Layer-wise)]
  - 对每层所有 patch 做 mean pooling → [N, D] 表示
  - 训练线性 probe 预测目标分量
  - 比较各层 probe 质量 (MSE / R²)

[层内 Probe (In-layer)]
  - 读取 timer_mi_ksg_pca.py 生成的 JSON（包含每层的 hsic_curve，即 I(H,Y)）
  - 按 I(H,Y) 分位数划分高/低 MI patch 组
  - 对每组做 mean pooling → 训练线性 probe

Usage:
  python experiments/timer_layer_inlayer_probe.py \
      --mi_result_dir ./outputs/timer_mi_ksg_pca/Timer_MI_20260516_114023/ \
      --root_path ./datasets/ --data_path ETTh1.csv \
      --seq_len 672 --pred_len 96 --patch_len 96 \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --stl_period 24 \
      --out_dir ./outputs/timer_layer_inlayer_probe/
"""

import argparse
import gc
import json
import os
import sys
import datetime
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from utils.masking import TriangularCausalMask


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
# SECTION 2: Token Extraction (from timer_mi_ksg_pca.py)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_layer_tokens(model, data_loader, device, n_layers: int):
    """
    提取每层每个 token 的 hidden states，与 timer_mi_ksg_pca.py 保持一致。

    Returns:
        hist_tokens:  list of [N, n_patches, D]  per layer
        future_tokens: list of [N, n_future_patches, D] per layer
        x_patch_tokens: [N, n_patches, D] — patch embedding output (dec_in_x)
        n_patches: int
        n_future_patches: int
    """
    core = _unwrap(model)
    all_hist = [[] for _ in range(n_layers)]
    all_future = [[] for _ in range(n_layers)]
    all_x_patch = []
    n_patches = None
    n_future_patches = None

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(
                data_loader, desc="提取 token 表示"):
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
            BM_y, N_y, _ = dec_in_y.shape

            if n_patches is None:
                n_patches = N
                n_future_patches = N_y
                print(f"  n_patches={n_patches}, n_future_patches={n_future_patches}, D={D}")

            derived_n_vars = BM // B
            x_patch_emb = dec_in_x.view(B, derived_n_vars, N, D).mean(dim=1).float().cpu()
            all_x_patch.append(x_patch_emb)

            mask_x = TriangularCausalMask(BM, N, device=device)
            mask_y = TriangularCausalMask(BM_y, N_y, device=device)

            hx = dec_in_x
            hy = dec_in_y
            for li, layer_module in enumerate(core.decoder.attn_layers):
                hx, _, _ = layer_module(hx, attn_mask=mask_x)
                hy, _, _ = layer_module(hy, attn_mask=mask_y)
                derived_n_vars_y = BM_y // B
                all_hist[li].append(
                    hx.view(B, derived_n_vars, N, D).mean(dim=1).float().cpu())
                all_future[li].append(
                    hy.view(B, derived_n_vars_y, N_y, D).mean(dim=1).float().cpu())

            del dec_in_x, dec_in_y, hx, hy
            gc.collect()
            torch.cuda.empty_cache()

    hist_tokens = [torch.cat(toks, dim=0) for toks in all_hist]
    future_tokens = [torch.cat(toks, dim=0) for toks in all_future]
    x_patch_tokens = torch.cat(all_x_patch, dim=0)
    return hist_tokens, future_tokens, x_patch_tokens, n_patches, n_future_patches


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MI Curve Loading (from timer_mi_ksg_pca.py JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def load_mi_from_json(mi_result_dir: str, n_layers: int) -> dict:
    """从 timer_mi_ksg_pca.py 的输出 JSON 读取每层的 MI 曲线。"""
    pattern = os.path.join(mi_result_dir, "global_mi_peaks_*.json")
    matched = glob.glob(pattern)
    if not matched:
        raise FileNotFoundError(f"未找到 MI 结果文件: {pattern}")
    json_path = matched[0]
    print(f"  [MI Loader] 读取: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    layers_dict = data.get("layers", {})
    mi_curves = {}
    for li in range(n_layers):
        key = str(li)
        if key not in layers_dict:
            raise ValueError(f"JSON 中未找到 layer={li}，可用键: {list(layers_dict.keys())}")
        mi_curves[li] = np.array(layers_dict[key]["hsic_curve"], dtype=np.float64)
    return mi_curves, data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: STL Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def stl_decompose_series(series: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对连续时间序列做 STL 分解（Seasonal-Trend Decomposition using LOESS）。

    参数:
        series: [T,] 一维连续时间序列
        period: 季节性周期长度
            - ETTh1 (小时数据, daily=24, weekly=168)
            - ETTm1 (分钟数据, 15min=96, hourly=96, daily=96*24)
    返回:
        trend, seasonal, residual 均为 [T,]
    """
    from statsmodels.tsa.seasonal import STL

    series = np.asarray(series).flatten()
    if series.ndim != 1:
        raise ValueError(f"STL 输入必须是一维数组，得到 shape={series.shape}")

    if len(series) < 2 * period:
        raise ValueError(
            f"序列长度 {len(series)} < 2*period={2*period}，无法做 STL 分解"
        )

    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid


def decompose_full_dataset(
    test_dataset: CIDatasetBenchmark,
    target_var: int = 0,
    stl_period: int = 24,
) -> dict:
    """
    从 CIDatasetBenchmark 逐样本提取 seq_y（已 scaling），
    对每个样本的 target window 做 STL 分解，返回对齐的 4 分量 targets。

    严格与 test_dataset.__getitem__ 的样本顺序对齐：
      index → c_begin = index // n_timepoint,  s_begin = index % n_timepoint

    返回:
        dict: {
            "original": [N, pred_len],
            "trend":    [N, pred_len],
            "seasonal": [N, pred_len],
            "residual": [N, pred_len],
            "n_total":  int,
        }
    """
    n_total = len(test_dataset)
    pred_len = test_dataset.pred_len

    targets = {
        "original": np.zeros((n_total, pred_len), dtype=np.float64),
        "trend":    np.zeros((n_total, pred_len), dtype=np.float64),
        "seasonal": np.zeros((n_total, pred_len), dtype=np.float64),
        "residual": np.zeros((n_total, pred_len), dtype=np.float64),
    }

    for i in range(n_total):
        _, seq_y, _, _ = test_dataset[i]
        seq_y = np.asarray(seq_y, dtype=np.float64)[:, 0]

        trend_i, seasonal_i, residual_i = stl_decompose_series(seq_y, stl_period)

        targets["original"][i] = seq_y
        targets["trend"][i]    = trend_i
        targets["seasonal"][i] = seasonal_i
        targets["residual"][i] = residual_i

        if (i + 1) % 5000 == 0:
            print(f"  [STL] 分解进度: {i+1}/{n_total}")

    trend_all    = targets["trend"].flatten()
    seasonal_all = targets["seasonal"].flatten()
    residual_all = targets["residual"].flatten()

    print(f"  [STL] 分解完成: {n_total} 样本, period={stl_period}")
    print(f"  [STL] Trend    range: [{trend_all.min():.4f}, {trend_all.max():.4f}]")
    print(f"  [STL] Seasonal range: [{seasonal_all.min():.4f}, {seasonal_all.max():.4f}]")
    print(f"  [STL] Residual range: [{residual_all.min():.4f}, {residual_all.max():.4f}]")

    targets["n_total"] = n_total
    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Linear Probe
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)

    def forward(self, x):
        return self.linear(x)


def train_probe(z_train, y_train, z_val, y_val, n_epochs: int, lr: float, device: torch.device):
    """训练线性 probe，返回 (val_mse, val_r2, val_pred, train_curve)。"""
    in_dim = z_train.shape[1]
    out_dim = y_train.shape[1]

    probe = LinearProbe(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    crit = nn.MSELoss()

    n = z_train.shape[0]
    bs = min(256, n)
    train_curve = []

    for ep in range(n_epochs):
        probe.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            bi = idx[i:i + bs]
            opt.zero_grad()
            loss = crit(probe(z_train[bi].to(device)), y_train[bi].to(device))
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            pred = probe(z_val.to(device))
            val_mse = crit(pred, y_val.to(device)).item()
        train_curve.append(val_mse)

    probe.eval()
    with torch.no_grad():
        final_pred = probe(z_val.to(device))
        final_mse = crit(final_pred, y_val.to(device)).item()
        ss_res = ((final_pred.cpu() - y_val) ** 2).sum().item()
        ss_tot = ((y_val - y_val.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return final_mse, r2, final_pred.detach().cpu(), train_curve


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layer_wise_results(layer_mse, layer_r2, layer_indices, output_dir,
                           suffix="", title_prefix=""):
    """层间 Probe 结果：MSE 和 R² 曲线。"""
    title_prefix = title_prefix or "Layer-wise"
    suffix = f"_{suffix}" if suffix else ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.plot(layer_indices, [layer_mse[l] for l in layer_indices],
            'o-', color='steelblue', lw=2, ms=6)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title(f"{title_prefix}: MSE vs Layer Depth", fontsize=13)
    ax.set_xticks(layer_indices)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layer_indices, [layer_r2[l] for l in layer_indices],
            'o-', color='darkorange', lw=2, ms=6)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"{title_prefix}: R² vs Layer Depth", fontsize=13)
    ax.set_xticks(layer_indices)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"probe_layer_wise{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


def plot_inlayer_probe_results(inlayer_results, layer_indices, output_dir,
                               suffix="", title_prefix=""):
    """层内 Probe：高 MI vs 低 MI token 的 MSE/R² 对比柱状图。"""
    title_prefix = title_prefix or "In-Layer"
    suffix = f"_{suffix}" if suffix else ""
    n_layers = len(layer_indices)
    x = np.arange(n_layers)
    w = 0.35

    mse_h = [inlayer_results[l]['mse_high'] for l in layer_indices]
    mse_l = [inlayer_results[l]['mse_low'] for l in layer_indices]
    r2_h = [inlayer_results[l]['r2_high'] for l in layer_indices]
    r2_l = [inlayer_results[l]['r2_low'] for l in layer_indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    ax = axes[0]
    bars_h = ax.bar(x - w / 2, mse_h, w, label="High-MI Tokens", color="steelblue", alpha=0.85)
    bars_l = ax.bar(x + w / 2, mse_l, w, label="Low-MI Tokens", color="coral", alpha=0.85)
    for xi in range(n_layers):
        better = "H" if mse_h[xi] < mse_l[xi] else "L"
        color = "steelblue" if better == "H" else "coral"
        lo = max(mse_h[xi], mse_l[xi])
        ax.text(xi, lo + lo * 0.01, better, ha="center", va="bottom",
                color=color, fontweight="bold", fontsize=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title(f"{title_prefix}: MSE (High-MI vs Low-MI)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layer_indices], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")

    ax = axes[1]
    bars_h = ax.bar(x - w / 2, r2_h, w, label="High-MI Tokens", color="steelblue", alpha=0.85)
    bars_l = ax.bar(x + w / 2, r2_l, w, label="Low-MI Tokens", color="coral", alpha=0.85)
    for xi in range(n_layers):
        better = "H" if r2_h[xi] > r2_l[xi] else "L"
        color = "steelblue" if better == "H" else "coral"
        hi = max(r2_h[xi], r2_l[xi])
        ax.text(xi, hi + 0.01, better, ha="center", va="bottom",
                color=color, fontweight="bold", fontsize=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"{title_prefix}: R² (High-MI vs Low-MI)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layer_indices], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, f"probe_inlayer_comparison{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


def plot_combined_summary(layer_mse, layer_r2, inlayer_results, layer_indices,
                         output_dir, suffix="", title_prefix="",
                         dataset_name="ETTh1"):
    """组合图：层间 MSE/R² + 层内 ΔMSE + win-rate + ΔR²。"""
    title_prefix = title_prefix or "Timer Probe"
    suffix = f"_{suffix}" if suffix else ""
    n_layers = len(layer_indices)
    x = np.arange(n_layers)

    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(layer_indices, [layer_mse[l] for l in layer_indices],
            'o-', color='steelblue', lw=2, label='MSE')
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11, color='steelblue')
    ax.set_title(f"{title_prefix}: Layer-wise MSE", fontsize=12)
    ax.set_xticks(layer_indices)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(layer_indices, [layer_r2[l] for l in layer_indices],
             's-', color='darkorange', lw=2, label='R²')
    ax2.set_ylabel("R²", fontsize=11, color='darkorange')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    delta_mse = [(inlayer_results[l]['mse_low'] - inlayer_results[l]['mse_high'])
                 for l in layer_indices]
    colors = ['steelblue' if d > 0 else 'coral' for d in delta_mse]
    ax.bar(x, delta_mse, color=colors, alpha=0.85)
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("ΔMSE (Low - High)", fontsize=11)
    ax.set_title(f"{title_prefix}: ΔMSE (Positive = High-MI Better)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layer_indices], fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    ax = fig.add_subplot(gs[1, 0])
    win_rates = []
    for l in layer_indices:
        wr = 1.0 if inlayer_results[l]['mse_high'] < inlayer_results[l]['mse_low'] else 0.0
        win_rates.append(wr)
    colors_wr = ['steelblue' if r > 0.5 else 'coral' for r in win_rates]
    bars = ax.bar(x, win_rates, color=colors_wr, alpha=0.85)
    ax.axhline(0.5, color='black', ls='--', lw=1.2, label='Random baseline')
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("High-MI Win (1=Yes)", fontsize=11)
    ax.set_title(f"{title_prefix}: High-MI Token Wins per Layer", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layer_indices], fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.25, axis="y")
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{'H' if wr > 0.5 else 'L'}", ha="center", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    delta_r2 = [(inlayer_results[l]['r2_high'] - inlayer_results[l]['r2_low'])
                for l in layer_indices]
    colors = ['steelblue' if d > 0 else 'coral' for d in delta_r2]
    ax.bar(x, delta_r2, color=colors, alpha=0.85)
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("ΔR² (High - Low)", fontsize=11)
    ax.set_title(f"{title_prefix}: ΔR² (Positive = High-MI Better)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layer_indices], fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle(f"{title_prefix} ({dataset_name})", fontsize=14, fontweight="bold")
    path = os.path.join(output_dir, f"probe_combined_summary{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


def plot_stl_cross_component(
    all_component_results: dict,
    component_names: list,
    layer_indices: list,
    output_dir: str,
):
    """
    跨分量对比：绘制所有分量的 layer-wise MSE / R² 在同一张图上，
    以及 ΔMSE 热力图。
    """
    components = [c for c in ["original", "trend", "seasonal", "residual"]
                 if c in all_component_results]
    colors = {
        "original": "#4C78A8",
        "trend":    "#F58518",
        "seasonal": "#54A24B",
        "residual": "#E45756",
    }
    labels = {
        "original": "Original",
        "trend":    "Trend",
        "seasonal": "Seasonal",
        "residual": "Residual",
    }

    # ── 1. Layer-wise MSE 对比 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    ax = axes[0]
    for comp in components:
        res = all_component_results[comp]
        lw_mse = [res["layer_wise"]["mse"].get(l, float("nan")) for l in layer_indices]
        ax.plot(layer_indices, lw_mse, 'o-', color=colors[comp],
                lw=2, ms=6, label=labels[comp], alpha=0.9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Layer-wise Probe MSE: Cross-Component Comparison", fontsize=13)
    ax.set_xticks(layer_indices)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for comp in components:
        res = all_component_results[comp]
        lw_r2 = [res["layer_wise"]["r2"].get(l, float("nan")) for l in layer_indices]
        ax.plot(layer_indices, lw_r2, 's-', color=colors[comp],
                lw=2, ms=6, label=labels[comp], alpha=0.9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Layer-wise Probe R²: Cross-Component Comparison", fontsize=13)
    ax.set_xticks(layer_indices)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "probe_stl_cross_component.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

    # ── 2. ΔMSE 热力图：layers × components ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    n_layers = len(layer_indices)
    n_comps = len(components)

    delta_matrix = np.zeros((n_comps, n_layers))
    for ci, comp in enumerate(components):
        res = all_component_results[comp]
        for li_idx, li in enumerate(layer_indices):
            inl = res["in_layer"].get(str(li), {})
            delta_matrix[ci, li_idx] = inl.get("delta_mse", 0.0)

    im = ax.imshow(delta_matrix, cmap="RdBu", aspect="auto",
                   vmin=-np.abs(delta_matrix).max(), vmax=np.abs(delta_matrix).max())
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{li}" for li in layer_indices], fontsize=10)
    ax.set_yticks(range(n_comps))
    ax.set_yticklabels([labels[c] for c in components], fontsize=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Component", fontsize=12)
    ax.set_title("ΔMSE Heatmap (High-MI − Low-MI)\nPositive=High-MI Wins", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for ci in range(n_comps):
        for li_idx in range(n_layers):
            v = delta_matrix[ci, li_idx]
            fc = "white" if abs(v) > np.abs(delta_matrix).max() * 0.6 else "black"
            ax.text(li_idx, ci, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, color=fc, fontweight="bold")

    path = os.path.join(output_dir, "probe_stl_delta_mse_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

    # ── 3. 分量胜率对比（High-MI 胜率 per component per layer）──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    ax = axes[0]
    n_groups = n_comps
    width = 0.8 / n_layers
    x_base = np.arange(n_layers)
    for ci, comp in enumerate(components):
        res = all_component_results[comp]
        vals = []
        for li in layer_indices:
            inl = res["in_layer"].get(str(li), {})
            vals.append(1.0 if inl.get("delta_mse", 0) > 0 else 0.0)
        offset = (ci - n_groups / 2 + 0.5) * width
        bars = ax.bar(x_base + offset, vals, width * 0.9,
                      label=labels[comp], color=colors[comp], alpha=0.85)
        for xi, v in enumerate(vals):
            ax.text(x_base[xi] + offset, v + 0.02, "H" if v > 0 else "L",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("High-MI Win (1=Yes)", fontsize=11)
    ax.set_title("In-Layer High-MI Win per Layer & Component", fontsize=12)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"L{li}" for li in layer_indices], fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.axhline(0.5, color='black', ls='--', lw=1)
    ax.legend(fontsize=9, ncol=n_comps)
    ax.grid(True, alpha=0.25, axis="y")

    # 胜率汇总条形图
    ax = axes[1]
    win_rates = []
    for comp in components:
        res = all_component_results[comp]
        wins = sum(
            1.0 if res["in_layer"].get(str(li), {}).get("delta_mse", 0) > 0
            else 0.0
            for li in layer_indices
        )
        win_rates.append(wins / n_layers)
    bar_colors = [colors[c] for c in components]
    bars = ax.bar(range(n_comps), win_rates, color=bar_colors, alpha=0.85)
    ax.axhline(0.5, color='black', ls='--', lw=1, label='Random baseline')
    ax.set_xticks(range(n_comps))
    ax.set_xticklabels([labels[c] for c in components], fontsize=11)
    ax.set_ylabel("High-MI Win Rate", fontsize=11)
    ax.set_title("In-Layer: High-MI Win Rate by Component", fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.25, axis="y")
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, wr + 0.03,
                f"{wr:.1%}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "probe_stl_winrate_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_component_probe(
    component_name: str,
    targets_flat: np.ndarray,
    hist_tokens: list,
    mi_curves: dict,
    n_patches_mi: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    targets_t: torch.Tensor,
    args,
    output_dir: str,
) -> dict:
    """对单个分量（original/trend/seasonal/residual）跑完整 probe 流程。"""
    suffix = component_name
    title_prefix = component_name.capitalize()
    if component_name == "original":
        title_prefix = "Original"
    elif component_name == "seasonal":
        title_prefix = "Seasonal"

    print(f"\n{'=' * 70}")
    print(f"  Component: {title_prefix}")
    print(f"{'=' * 70}")

    N = targets_flat.shape[0]
    D = hist_tokens[0].shape[2]
    layer_indices = list(range(args.e_layers))

    # ── Layer-wise Probe ──────────────────────────────────────────────────────
    layer_mse = {}
    layer_r2 = {}

    print(f"\n  [Layer-wise Probe]")
    for li in range(args.e_layers):
        z = hist_tokens[li].mean(dim=1)
        z_train, z_val = z[train_idx], z[val_idx]
        y_train, y_val = targets_t[train_idx], targets_t[val_idx]

        mse, r2, _, _ = train_probe(
            z_train, y_train, z_val, y_val,
            args.probe_epochs, args.probe_lr, args.device)
        layer_mse[li] = mse
        layer_r2[li] = r2
        print(f"    Layer {li:02d}: MSE={mse:.6f}, R²={r2:.4f}")

    # ── In-layer Probe ────────────────────────────────────────────────────────
    q_hi = int(n_patches_mi * 0.15)
    inlayer_results = {}

    print(f"\n  [In-layer Probe] (High: top 15%, Low: bottom 15%)")
    for li in range(args.e_layers):
        mi_curve = mi_curves[li]
        sorted_idx = np.argsort(mi_curve)
        high_mi_patch_idx = sorted_idx[-q_hi:].tolist()
        low_mi_patch_idx = sorted_idx[:-q_hi].tolist()

        tok = hist_tokens[li]
        high_rep = tok[:, high_mi_patch_idx, :].mean(dim=1)
        low_rep = tok[:, low_mi_patch_idx, :].mean(dim=1)

        high_train = high_rep[train_idx]
        high_val = high_rep[val_idx]
        low_train = low_rep[train_idx]
        low_val = low_rep[val_idx]
        y_train, y_val = targets_t[train_idx], targets_t[val_idx]

        mse_h, r2_h, _, _ = train_probe(
            high_train, y_train, high_val, y_val,
            args.probe_epochs, args.probe_lr, args.device)
        mse_l, r2_l, _, _ = train_probe(
            low_train, y_train, low_val, y_val,
            args.probe_epochs, args.probe_lr, args.device)

        delta_mse = mse_l - mse_h
        delta_r2 = r2_h - r2_l

        inlayer_results[li] = {
            'mse_high': mse_h, 'mse_low': mse_l,
            'r2_high': r2_h, 'r2_low': r2_l,
            'delta_mse': delta_mse, 'delta_r2': delta_r2,
            'high_mi_patch_idx': high_mi_patch_idx,
            'low_mi_patch_idx': low_mi_patch_idx,
        }

        better = "HIGH" if delta_mse > 0 else "LOW"
        print(f"    Layer {li:02d}: H MSE={mse_h:.6f} R²={r2_h:.4f} | "
              f"L MSE={mse_l:.6f} R²={r2_l:.4f} | "
              f"ΔMSE={delta_mse:+.6f} ({better})")

    # ── Plotting ─────────────────────────────────────────────────────────────
    plot_layer_wise_results(
        layer_mse, layer_r2, layer_indices, output_dir,
        suffix=suffix, title_prefix=title_prefix)
    plot_inlayer_probe_results(
        inlayer_results, layer_indices, output_dir,
        suffix=suffix, title_prefix=title_prefix)
    plot_combined_summary(
        layer_mse, layer_r2, inlayer_results, layer_indices, output_dir,
        suffix=suffix, title_prefix=title_prefix, dataset_name=args.data_type)

    results = {
        "component": component_name,
        "layer_wise": {"mse": layer_mse, "r2": layer_r2},
        "in_layer": inlayer_results,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  [Summary] {title_prefix} Layer-wise:")
    print(f"  {'Layer':>6} | {'MSE':>12} | {'R²':>10}")
    print("  " + "-" * 35)
    for li in layer_indices:
        print(f"  {li:>6} | {layer_mse[li]:>12.6f} | {layer_r2[li]:>10.4f}")

    high_wins = sum(1 for r in inlayer_results.values() if r['delta_mse'] > 0)
    print(f"\n  [{title_prefix}] High-MI 胜率: {high_wins}/{args.e_layers} "
          f"({high_wins / args.e_layers * 100:.0f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Timer 层间 + 层内 Probe 分析（含时序分解 STL）")
    parser.add_argument("--mi_result_dir", type=str,
                        default="./outputs/timer_mi_ksg_pca",
                        help="timer_mi_ksg_pca.py 输出目录")
    parser.add_argument("--root_path",   type=str, default="./datasets/")
    parser.add_argument("--data_path",   type=str, default="ETTh1.csv")
    parser.add_argument("--data_type",   type=str, default="ETTh1")
    parser.add_argument("--seq_len",     type=int, default=672)
    parser.add_argument("--pred_len",    type=int, default=96)
    parser.add_argument("--patch_len",   type=int, default=96)
    parser.add_argument("--stride",      type=int, default=96)
    parser.add_argument("--d_model",     type=int, default=1024)
    parser.add_argument("--d_ff",        type=int, default=2048)
    parser.add_argument("--e_layers",    type=int, default=8)
    parser.add_argument("--n_heads",     type=int, default=8)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--ckpt_path",   type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_lr",    type=float, default=1e-3)
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--freq",        type=str, default="h")
    parser.add_argument("--model_id",    type=str, default="etth1")
    parser.add_argument("--out_dir",     type=str,
                        default="./outputs/timer_layer_inlayer_probe")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--stl_period",  type=int, default=24,
                        help="STL 季节性周期长度（小时数据 daily=24）")
    parser.add_argument("--stl_target_var", type=int, default=0,
                        help="STL 分解的目标变量索引（默认 0，第 1 列）")
    parser.add_argument("--components",  type=str, nargs="+",
                        default=["original", "trend", "seasonal", "residual"],
                        choices=["original", "trend", "seasonal", "residual"],
                        help="要分析的分量（默认全部 4 个）")
    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Timer 层间 + 层内 Probe 分析（含 STL 时序分解）")
    print("=" * 70)
    print(f"  mi_result_dir  : {args.mi_result_dir}")
    print(f"  ckpt_path      : {args.ckpt_path}")
    print(f"  data_path      : {args.data_path}")
    print(f"  seq_len        : {args.seq_len}")
    print(f"  pred_len       : {args.pred_len}")
    print(f"  patch_len      : {args.patch_len}")
    print(f"  e_layers       : {args.e_layers}")
    print(f"  stl_period     : {args.stl_period}")
    print(f"  components     : {args.components}")
    print(f"  probe_epochs   : {args.probe_epochs}")
    print(f"  device         : {args.device}")
    print(f"  out_dir        : {output_dir}")
    print("=" * 70)

    # ── 1. Load MI curves from JSON ───────────────────────────────────────────
    print("\n>>> Phase 1: 从 JSON 加载 MI 曲线...")
    mi_curves, mi_json_data = load_mi_from_json(args.mi_result_dir, args.e_layers)
    n_patches_mi = len(mi_curves[0])
    print(f"  MI 曲线: {args.e_layers} 层, 每层 {n_patches_mi} patches")

    # ── 2. Load dataset ────────────────────────────────────────────────────────
    print("\n>>> Phase 2: 加载数据集...")
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

    # ── 3. Load Timer model ────────────────────────────────────────────────────
    print("\n>>> Phase 3: 加载 Timer 模型...")
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
    model = model.to(args.device)
    model.eval()
    print(f"  模型加载完成，设备: {args.device}")

    # ── 4. Extract layer tokens ───────────────────────────────────────────────
    print("\n>>> Phase 4: 提取测试集每层 token 表示...")
    hist_tokens, future_tokens, x_patch_tokens, n_patches, n_future_patches = \
        extract_layer_tokens(model, test_loader, args.device, args.e_layers)

    N = hist_tokens[0].shape[0]
    D = hist_tokens[0].shape[2]
    print(f"  提取完成: N={N}, n_patches={n_patches}, D={D}, n_layers={len(hist_tokens)}")

    # ── 5. STL Decomposition ─────────────────────────────────────────────────
    print(f"\n>>> Phase 5: STL 时序分解 (period={args.stl_period})...")
    targets_decomposed = decompose_full_dataset(
        test_dataset=test_dataset,
        target_var=args.stl_target_var,
        stl_period=args.stl_period,
    )

    n_stl = targets_decomposed["n_total"]
    if n_stl != N:
        print(f"  [WARN] STL 样本数 ({n_stl}) != 注意力提取样本数 ({N}), "
              f"以较小值 {min(n_stl, N)} 为准")
        n_final = min(n_stl, N)
    else:
        n_final = N

    # train/val split（固定 seed，对所有分量一致）
    all_idx = np.arange(n_final)
    train_idx, val_idx = train_test_split(
        all_idx, train_size=0.8, random_state=args.seed)
    print(f"  Probe 训练集: {len(train_idx)}, 验证集: {len(val_idx)}")

    # ── 6. Probe for each component ─────────────────────────────────────────
    all_component_results = {}

    for comp in args.components:
        if comp not in targets_decomposed:
            print(f"  [WARN] 分量 '{comp}' 不存在，跳过")
            continue

        targets_flat = targets_decomposed[comp][:n_final]
        targets_t = torch.from_numpy(targets_flat).float()

        results = run_component_probe(
            component_name=comp,
            targets_flat=targets_flat,
            hist_tokens=hist_tokens[:n_final],
            mi_curves=mi_curves,
            n_patches_mi=n_patches_mi,
            train_idx=train_idx,
            val_idx=val_idx,
            targets_t=targets_t,
            args=args,
            output_dir=output_dir,
        )
        all_component_results[comp] = results

    # ── 7. Cross-component comparison plots ───────────────────────────────────
    if len(all_component_results) >= 2:
        print("\n>>> Phase 7: 跨分量对比绘图...")
        layer_indices = list(range(args.e_layers))
        plot_stl_cross_component(
            all_component_results, args.components, layer_indices, output_dir)

    # ── 8. Save all results ─────────────────────────────────────────────────
    print(f"\n>>> Phase 8: 保存结果...")
    final_results = {
        "components": {k: {
            "layer_wise": {
                "mse": {str(kk): vv for kk, vv in v["layer_wise"]["mse"].items()},
                "r2":  {str(kk): vv for kk, vv in v["layer_wise"]["r2"].items()},
            },
            "in_layer": {
                str(kk): {
                    "mse_high": vv["mse_high"],
                    "mse_low":  vv["mse_low"],
                    "r2_high":  vv["r2_high"],
                    "r2_low":   vv["r2_low"],
                    "delta_mse": vv["delta_mse"],
                    "delta_r2":  vv["delta_r2"],
                } for kk, vv in v["in_layer"].items()
            },
        } for k, v in all_component_results.items()},
        "config": {
            "n_layers": args.e_layers,
            "n_patches": n_patches,
            "d_model": int(D),
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "data_path": args.data_path,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "mi_result_dir": args.mi_result_dir,
            "stl_period": args.stl_period,
            "stl_target_var": args.stl_target_var,
            "components": args.components,
        }
    }

    results_path = os.path.join(output_dir, "probe_results.pt")
    torch.save(final_results, results_path)
    print(f"  保存: {results_path}")

    # JSON 摘要
    summary_path = os.path.join(output_dir, "probe_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "components": {k: {
                "best_layer_mse": min(v["layer_wise"]["mse"].items(), key=lambda x: x[1]),
                "best_layer_r2":  max(v["layer_wise"]["r2"].items(), key=lambda x: x[1]),
                "high_mi_win_rate": sum(
                    1 for r in v["in_layer"].values() if r["delta_mse"] > 0
                ) / max(1, len(v["in_layer"])),
            } for k, v in all_component_results.items()},
            "config": final_results["config"],
        }, f, indent=2)
    print(f"  保存: {summary_path}")

    print("\n" + "=" * 70)
    print("  完成！")
    print(f"  结果目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
