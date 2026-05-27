#!/usr/bin/env python3
"""
ETTh1 + Timer: 平均能量分布对比 (Mean PSD Profiles)

实验目标：
    高 MI Patch 的 PSD 曲线是否更"平坦"（含更多高频细节/噪声）？
    低 MI Patch 的 PSD 曲线是否在低频处有尖锐峰（代表稳定的周期性）？

操作步骤：
    1. 提取所有 patch 的原始时间序列（patch_len 长度）
    2. 计算每个 patch 与未来真值的 HSIC MI 分数
    3. 将 patch 分为高 MI 组（>Q3，top 25%）和低 MI 组（<Q1，bottom 25%）
    4. 用 Welch 方法计算每个 patch 的 PSD
    5. 对原始 PSD 直接累加求平均（不做归一化）
    6. 分别对两组求平均，绘制对比图

用法（单卡）：
    python experiments/etth1_psd_analysis.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1

用法（多卡，必须用 torchrun）：
    torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_psd_analysis.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 \
      --use_multi_gpu
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from scipy.signal import welch as scipy_welch
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.masking import TriangularCausalMask

MI_DECODER_LAYER_CAP = 8


# ─── 加载 etth1_mi_hsic_peaks 输出的 JSON 文件 ───────────────────────────────

def load_hsic_mi_curve(hsic_file: str, layer_idx: int = -1) -> tuple[np.ndarray, float, float, int]:
    """
    加载 etth1_mi_hsic_peaks.py 输出的全局 MI JSON 文件。

    返回: (mi_curve, q3_threshold, q1_threshold, N) — MI 曲线、Q3 阈值、Q1 阈值、patch 数
    """
    import json
    with open(hsic_file, "r") as f:
        data = json.load(f)

    num_layers = data.get("num_layers", 0)
    layers = data.get("layers", {})

    # 确定层索引
    if layer_idx < 0:
        actual_layer = num_layers + layer_idx  # -1 → 最后一层
    else:
        actual_layer = layer_idx

    layer_key = str(actual_layer)
    if layer_key not in layers:
        available = list(layers.keys())
        raise ValueError(f"Layer {layer_key} not found. Available layers: {available}")

    layer_data = layers[layer_key]
    mi_curve = np.array(layer_data["hsic_curve"], dtype=np.float64)
    q3_threshold = layer_data["q3_threshold"]
    valid_mi = mi_curve[np.isfinite(mi_curve)]
    q1_threshold = float(np.percentile(valid_mi, 25))

    return mi_curve, q3_threshold, q1_threshold, len(mi_curve)


# ─── HSIC / MI 相关函数（用于无 JSON 时的回退计算） ──────────────────────────

def _median_sq_bandwidth(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    d = torch.cdist(X, X, p=2.0)
    triu = torch.triu_indices(n, n, offset=1, device=X.device)
    sq = d[triu[0], triu[1]] ** 2
    med = torch.median(sq)
    return med.clamp(min=1e-12)


def rbf_kernel(X: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(X, X, p=2.0) ** 2
    return torch.exp(-d2 / (2.0 * sigma_sq))


def hsic_unbiased_gaussian(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    if n < 4:
        return torch.tensor(float("nan"), device=X.device, dtype=X.dtype)
    sx = _median_sq_bandwidth(X)
    sy = _median_sq_bandwidth(Y)
    K = rbf_kernel(X, sx)
    Lm = rbf_kernel(Y, sy)
    H = torch.eye(n, device=X.device, dtype=X.dtype) - (1.0 / n)
    Kc = H @ K @ H
    Lc = H @ Lm @ H
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)


def zscore_cols(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = t.mean(dim=0, keepdim=True)
    s = t.std(dim=0, keepdim=True).clamp(min=eps)
    return (t - m) / s


def mi_sequence_hsic(Hbm: torch.Tensor, Hy: torch.Tensor) -> np.ndarray:
    B, N_x, D = Hbm.shape
    B2, _, _ = Hy.shape
    assert B == B2, f"Batch size mismatch: Hbm={Hbm.shape}, Hy={Hy.shape}"
    hy_single = Hy[:, 0, :].unsqueeze(1)
    Hy_expanded = hy_single.expand(-1, N_x, -1)
    out = []
    for p in range(N_x):
        X = zscore_cols(Hbm[:, p, :].contiguous())
        Y = zscore_cols(Hy_expanded[:, p, :].contiguous())
        out.append(hsic_unbiased_gaussian(X, Y).detach().float().cpu().item())
    return np.asarray(out, dtype=np.float64)


def forward_collect_layers(model, x_enc):
    core = model.module if hasattr(model, "module") else model
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers = []
    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _, _ = o3(h, attn_mask=mask)
        layers.append(pool(h.detach()))
    return layers, int(n_vars), int(N)


def forward_y_collect_layers(model, y):
    core = model.module if hasattr(model, "module") else model
    B, L, M = y.shape
    means = y.mean(1, keepdim=True).detach()
    x = y - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers = []
    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _, _ = o3(h, attn_mask=mask)
        layers.append(pool(h.detach()))
    return layers


def build_namespace(args):
    ns = argparse.Namespace(**vars(args))
    for k, v in {
        "task_name": "forecast",
        "is_training": 0,
        "is_finetuning": 0,
        "train_test": 0,
        "use_multi_gpu": False,
        "d_layers": 1,
        "target": "OT",
        "checkpoints": "./checkpoints/",
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
        "model_id": "psd_analysis",
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
        "dropout": 0.1,
        "activation": "gelu",
        "embed": "timeF",
        "freq": "h",
    }.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


# ─── PSD 计算 ──────────────────────────────────────────────────────────────────

def welch_psd(x: np.ndarray, fs: float = 1.0, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    用 scipy.signal.welch 计算功率谱密度（归一化为概率分布）。

    x: 1D 时间序列，shape [T]
    fs: 采样频率（归一化为 1.0）
    nperseg: 每段长度（默认取 min(256, len(x) // 4)）

    返回: (freqs, psd) — PSD 归一化为概率分布（sum(psd) * df ≈ 1.0）
    """
    T = len(x)
    if nperseg is None or nperseg <= 0:
        seg = min(256, T)
    else:
        seg = min(nperseg, T)
    # nfft 固定为 nperseg，保证输出长度一致
    freqs, psd = scipy_welch(x, fs=fs, nperseg=seg, noverlap=seg // 2, nfft=seg)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total = np.sum(psd) * df
    if total > 0:
        psd = psd / total
    return freqs.astype(np.float64), psd.astype(np.float64)


def normalize_psd(freqs: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将 PSD 归一化为概率分布：
        psd_norm = psd / sum(psd * df)
    返回 (freqs, psd_norm)，使得 sum(psd_norm) ≈ 1.0
    """
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total = np.sum(psd) * df
    if total > 0:
        psd = psd / total
    return freqs, psd


def extract_patches_from_batch(
    batch_x: torch.Tensor,
    patch_len: int,
    stride: int,
    mean: torch.Tensor,
    stdev: torch.Tensor,
) -> np.ndarray:
    """
    从 batch_x 中提取所有 patch 的原始时间序列（反归一化后）。

    batch_x: [B, L, M]  归一化后的输入
    patch_len, stride: patch 参数
    mean: [B, 1, M]  用于反归一化
    stdev: [B, 1, M]  用于反归一化

    返回: npatches [B*N_patches, patch_len, M] — 所有 batch、所有 patch 的多变量时间序列
    """
    B, L, M = batch_x.shape
    N = (L - patch_len) // stride + 1

    # 反归一化：还原原始尺度
    x_raw = batch_x * stdev + mean  # [B, L, M]

    patches = []
    for p in range(N):
        start = p * stride
        end = start + patch_len
        seg = x_raw[:, start:end, :]  # [B, patch_len, M]
        patches.append(seg)
    # [N, B, patch_len, M] → [B*N, patch_len, M]
    patches = torch.stack(patches, dim=0)  # [N, B, patch_len, M]
    patches = patches.permute(1, 0, 2, 3)   # [B, N, patch_len, M]
    patches = patches.reshape(B * N, patch_len, M)
    return patches.cpu().numpy()


# ─── 绘图函数 ──────────────────────────────────────────────────────────────────

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.signal import find_peaks
from datetime import datetime
import textwrap


def find_top_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    low_freq_max: float = 0.1,
    n_peaks: int = 3,
    prominence: float | None = None,
) -> list[tuple[float, float, float]]:
    """
    在低频区间 [0, low_freq_max] 内找 PSD 的 top-n 峰值。

    返回: list of (freq, psd_val, period)  按 PSD 降序排列
    """
    mask = freqs <= low_freq_max
    f_sub = freqs[mask]
    p_sub = psd[mask]

    if len(f_sub) < 3:
        return []

    prom = prominence if prominence is not None else (np.max(p_sub) - np.min(p_sub)) * 0.05
    peaks, properties = find_peaks(p_sub, prominence=prom)

    if len(peaks) == 0:
        return []

    peak_psds = p_sub[peaks]
    top_idx = np.argsort(peak_psds)[::-1][:n_peaks]

    result = []
    for idx in top_idx:
        f = float(f_sub[peaks[idx]])
        p = float(p_sub[peaks[idx]])
        period = 1.0 / f if f > 1e-8 else float("inf")
        result.append((f, p, period))
    return result

def plot_mean_psd_comparison(
    out_path: str,
    freqs: np.ndarray,
    psd_high: np.ndarray,
    psd_low: np.ndarray,
    psd_all: np.ndarray,
    label_high: str,
    label_low: str,
    label_all: str,
    n_high: int,
    n_low: int,
    n_total: int,
    freq_hours_per_step: float = 1.0,
    peaks_high: list[tuple[float, float, float]] | None = None,
    peaks_low: list[tuple[float, float, float]] | None = None,
    peaks_all: list[tuple[float, float, float]] | None = None,
) -> None:
    """
    绘制高 MI / 低 MI / 全局平均 PSD 对比图（对数坐标 + 线性坐标双面板）。

    freqs: 频率轴，单位为 cycles/hour（由 fs_hz = 1/freq_hours_per_step 换算得到）
    freq_hours_per_step: 每个 time-step 对应多少真实小时
    """

    peak_colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.grid(True, alpha=0.3, which="major")
        ax.grid(True, alpha=0.15, which="minor")

    def _annotate_peaks(ax, peaks, label_prefix, color):
        for i, (f, p, T) in enumerate(peaks):
            c = peak_colors[i % len(peak_colors)]
            ax.axvline(f, color=c, ls=":", lw=1.2, alpha=0.7)
            ax.annotate(
                f"{label_prefix}#{i+1}\nf={f:.3f}\nT={T:.1f}",
                xy=(f, p),
                xytext=(f + 0.008, p * (1.5 if i % 2 == 0 else 0.6)),
                fontsize=7,
                color=c,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.7),
            )

    # ── 左图：对数坐标 ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(freqs, psd_high, lw=2, color="#e41a1c", label=f"{label_high} (n={n_high})")
    ax.semilogy(freqs, psd_low,  lw=2, color="#377eb8", label=f"{label_low}  (n={n_low})")
    ax.semilogy(freqs, psd_all,  lw=1.5, color="grey", ls="--", label=f"{label_all} (n={n_total})")

    # 动态周期参考线（日周期 24、周周期 168）
    for period_h in [24, 168]:
        f_ref = 1.0 / period_h
        mask_ref = freqs <= f_ref
        if not np.any(mask_ref):
            continue
        ax.axvline(f_ref, color="purple", ls="--", lw=1.2,
                   label=f"T={period_h} (f={f_ref:.4f})")

    if peaks_high:
        _annotate_peaks(ax, peaks_high, "H", "#e41a1c")
    if peaks_low:
        _annotate_peaks(ax, peaks_low, "L", "#377eb8")

    ax.set_xlabel("Frequency (cycles / time-step)", fontsize=11)
    ax.set_ylabel("Normalized PSD (log scale)", fontsize=11)
    ax.set_title("Mean PSD Profile — Log Scale", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(freqs[0], freqs[-1])

    # ── 右图：线性坐标，聚焦低频区域 ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(freqs, psd_high, lw=2, color="#e41a1c", label=f"{label_high} (n={n_high})")
    ax.plot(freqs, psd_low,  lw=2, color="#377eb8", label=f"{label_low}  (n={n_low})")
    ax.plot(freqs, psd_all,  lw=1.5, color="grey", ls="--", label=f"{label_all} (n={n_total})")

    for period_h in [24, 168]:
        f_ref = 1.0 / period_h
        mask_ref = freqs <= f_ref
        if not np.any(mask_ref):
            continue
        ax.axvline(f_ref, color="purple", ls="--", lw=1.2,
                   label=f"T={period_h}h (f={f_ref:.4f})")

    if peaks_high:
        _annotate_peaks(ax, peaks_high, "H", "#e41a1c")
    if peaks_low:
        _annotate_peaks(ax, peaks_low, "L", "#377eb8")

    ax.set_xlabel("Frequency (cycles / time-step)", fontsize=11)
    ax.set_ylabel("Normalized PSD (linear scale)", fontsize=11)
    ax.set_title("Mean PSD Profile — Linear Scale (Low-Freq Detail)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(freqs[0], min(freqs[-1], 0.12))
    ax.set_ylim(bottom=0.0)

    fig.suptitle(
        f"High-MI vs Low-MI Patch: Mean PSD Comparison",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psd_ratio(
    out_path: str,
    freqs: np.ndarray,
    ratio: np.ndarray,
    label_high: str,
    label_low: str,
    freq_hours_per_step: float = 1.0,
    peaks_ratio: list[tuple[float, float, float]] | None = None,
) -> None:
    """绘制高/低 MI PSD 比值曲线，并标注动态峰值。"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    valid = ratio > 0
    ax.semilogy(freqs[valid], ratio[valid], lw=1.5, color="#4daf4a")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="ratio = 1 (equal)")
    ax.fill_between(freqs[valid], 1.0, ratio[valid], where=(ratio[valid] > 1),
                    alpha=0.15, color="#e41a1c", label="high-MI more energy")
    ax.fill_between(freqs[valid], 1.0, ratio[valid], where=(ratio[valid] < 1),
                    alpha=0.15, color="#377eb8", label="low-MI more energy")

    # 周期参考线（归一化频率：日周期 T=24，周周期 T=168）
    for period_h in [24, 168]:
        f_ref = 1.0 / period_h
        mask_ref = freqs[valid] <= f_ref
        if not np.any(mask_ref):
            continue
        ax.axvline(f_ref, color="purple", ls="--", lw=1.2,
                   label=f"T={period_h} (f={f_ref:.4f})")

    # 标注动态峰值
    if peaks_ratio:
        peak_colors = ["#ff7f0e", "#2ca02c", "#d62728"]
        for i, (f, p, T) in enumerate(peaks_ratio):
            c = peak_colors[i % len(peak_colors)]
            ax.axvline(f, color=c, ls=":", lw=1.2, alpha=0.8)
            ax.annotate(
                f"R#{i+1}  f={f:.3f}  T={T:.1f}",
                xy=(f, p),
                xytext=(f + 0.008, p * (1.8 if i % 2 == 0 else 0.5)),
                fontsize=7.5,
                color=c,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.8),
            )

    ax.set_xlabel("Normalized Frequency (cycles / time-step)", fontsize=11)
    ax.set_ylabel("PSD Ratio (high-MI / low-MI)", fontsize=11)
    ax.set_title(f"PSD Ratio: {label_high} / {label_low} (cycles / time-step)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(freqs[valid][0], min(freqs[valid][-1], 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patch_mi_histogram(
    out_path: str,
    mi_flat: np.ndarray,
    q3: float,
) -> None:
    """绘制所有 patch MI 分数的直方图，标注 Q3 分位线。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(mi_flat, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(q3, color="red", ls="--", lw=2, label=f"Q3 = {q3:.4f}")
    ax.set_xlabel("HSIC MI Score (per patch)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Patch MI Scores (All Test Batches)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--root_path", type=str, default="./datasets/")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--data", type=str, default="ETTh1")
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
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--subset_rand_ratio", type=float, default=1.0)
    p.add_argument("--use_ims", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", type=str, default="./results/psd_etth1")
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test loader")
    p.add_argument("--nperseg", type=int, default=0, help="Welch nperseg; 0=auto")
    p.add_argument("--freq_hours_per_step", type=float, default=1.0,
                  help="每个 time-step 对应多少真实小时。ETTh1=1.0, ETTm1=0.25")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_multi_gpu", action="store_true")
    p.add_argument("--hsic_file", type=str, default=None,
                  help="etth1_mi_hsic_peaks 输出的 JSON 文件路径。若指定则直接加载全局 MI 曲线进行分组；否则在运行时计算。")
    p.add_argument("--hsic_layer", type=int, default=-1,
                  help="使用 JSON 中哪一层的 MI 曲线（-1=最后一层）")
    args = p.parse_args()

    rank = 0
    local_rank = 0
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "Multi-GPU requires torchrun. Example:\n"
                "  torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_psd_analysis.py \\\n"
                "    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt --use_multi_gpu ..."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("use_multi_gpu requires CUDA.")
        n_visible = torch.cuda.device_count()
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > n_visible:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} but only {n_visible} CUDA device(s) visible. "
                f"Use torchrun --nproc_per_node={n_visible}."
            )
        if local_rank >= n_visible:
            raise RuntimeError(f"LOCAL_RANK={local_rank} >= visible devices={n_visible}.")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        ws = dist.get_world_size()
        if ws > 1:
            print(f"[psd] rank {rank}/{ws}: shard {rank} of test set via DistributedSampler.",
                  flush=True)

    ns = build_namespace(args)
    ns.use_multi_gpu = bool(args.use_multi_gpu)
    ns.use_gpu = bool(torch.cuda.is_available())

    # ── 自动生成带时间戳的输出目录 ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(args.out_dir.rstrip("/")) or "./results",
                           f"{os.path.basename(args.out_dir)}_{timestamp}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if args.use_multi_gpu:
        dist.barrier()

    device = torch.device(f"cuda:{local_rank}") if args.use_multi_gpu else torch.device(args.device)

    _, loader = data_provider(ns, flag="test")
    model = Model(ns).to(device)
    model.eval()
    if args.use_multi_gpu:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    core = model.module if hasattr(model, "module") else model
    patch_len = int(ns.patch_len)
    stride = int(core.backbone.patch_embedding.stride)

    # ── 加载全局 HSIC MI 曲线（从 JSON 文件）───────────────────────────────
    global_mi_curve = None
    global_q3 = None
    global_q1 = None
    if args.hsic_file:
        print(f"[PSD] Loading global MI curve from: {args.hsic_file}")
        global_mi_curve, global_q3, global_q1, N = load_hsic_mi_curve(args.hsic_file, args.hsic_layer)
        print(f"[PSD] Loaded MI curve: layer={args.hsic_layer}, N={N}, Q3={global_q3:.6f}, Q1={global_q1:.6f}")
    else:
        print("[PSD] Warning: --hsic_file not specified. Using runtime batch-level MI (legacy mode).")

    # ── PSD 参数 ──────────────────────────────────────────────────────────────
    nperseg = args.nperseg if args.nperseg > 0 else min(256, patch_len // 4)
    # 采样频率固定为 1.0（每个 time-step 为 1 个单位），横轴为归一化频率
    fs_hz = 1.0

    # ── 收集统计量（各 rank 独立收集，最后 all-reduce） ──────────────────────────
    sum_psd_high = None   # 高 MI 组 PSD 之和
    sum_psd_low = None    # 低 MI 组 PSD 之和
    sum_psd_all = None    # 全局 PSD 之和
    n_high_total = 0      # 高 MI 组 patch 总数
    n_low_total = 0       # 低 MI 组 patch 总数
    n_all_total = 0       # 全局 patch 总数
    sum_mi = None         # MI 值求和（用于算全局均值）
    n_mi_batches = 0
    freqs_global = None

    for batch_idx, (batch_x, batch_y, _, _) in enumerate(loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        B, Lx, C = batch_x.shape

        if B < 4:
            continue

        if args.use_ims:
            y_future = batch_y[:, ns.label_len: ns.label_len + ns.pred_len, :]
        else:
            y_future = batch_y[:, -ns.pred_len:, :]

        with torch.no_grad():
            layer_h_x, nvars, N = forward_collect_layers(model, batch_x)
            layer_h_y = forward_y_collect_layers(model, y_future)

        # ── 使用全局 HSIC MI 曲线（从 JSON 加载）────────────
        if global_mi_curve is not None:
            mi_curve = global_mi_curve
            q3 = global_q3
            q1 = global_q1
        else:
            mi_curve = mi_sequence_hsic(layer_h_x[-1].cpu(), layer_h_y[-1].cpu())
            valid_mi = mi_curve[np.isfinite(mi_curve)]
            if valid_mi.size == 0:
                continue
            q3 = float(np.percentile(valid_mi, 75))
            q1 = float(np.percentile(valid_mi, 25))

        # 归一化参数（用于反归一化 patch 时间序列）
        means = batch_x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(batch_x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        # 提取所有 patch 时间序列 [B*N, patch_len, M]
        patches_raw = extract_patches_from_batch(batch_x, patch_len, stride, means, stdev)

        # 对所有变量求平均，得到 [B*N, patch_len] 的标量时间序列
        if patches_raw.shape[2] > 1:
            patches_ts = patches_raw.mean(axis=2)  # [B*N, patch_len]
        else:
            patches_ts = patches_raw[:, :, 0]

        # ── 全局系综平均（Ensemble Averaging） ───────────────────────────────────
        # 对每个 patch：计算 PSD → 直接累加到对应组的 total_psd
        # 不再做 batch-level 归一化，最后除以该组总 patch 数
        this_freqs = None
        total_psd_high = None
        total_psd_low  = None
        total_psd_all  = None

        for pi in range(N):
            for bi in range(B):
                idx = bi * N + pi
                ts = patches_ts[idx]
                freqs_i, psd_i = welch_psd(ts, fs=fs_hz, nperseg=nperseg)

                if this_freqs is None:
                    this_freqs = freqs_i
                    n_freq = len(freqs_i)
                    total_psd_high = np.zeros(n_freq)
                    total_psd_low  = np.zeros(n_freq)
                    total_psd_all  = np.zeros(n_freq)

                total_psd_all += psd_i
                mi_val = mi_curve[pi]
                if np.isfinite(mi_val) and mi_val > q3:
                    total_psd_high += psd_i
                elif np.isfinite(mi_val) and mi_val < q1:
                    total_psd_low += psd_i

        if this_freqs is None:
            continue

        n_high = 0
        n_low  = 0
        n_all  = 0
        for pi in range(N):
            for bi in range(B):
                mi_val = mi_curve[pi]
                if np.isfinite(mi_val) and mi_val > q3:
                    n_high += 1
                elif np.isfinite(mi_val) and mi_val < q1:
                    n_low  += 1
                n_all += 1

        if sum_psd_high is None:
            n_freq = len(this_freqs)
            sum_psd_high = np.zeros(n_freq)
            sum_psd_low = np.zeros(n_freq)
            sum_psd_all = np.zeros(n_freq)
            sum_mi = np.zeros(N)
            freqs_global = this_freqs

        sum_psd_high += total_psd_high
        sum_psd_low  += total_psd_low
        sum_psd_all  += total_psd_all
        sum_mi += mi_curve
        n_high_total += n_high
        n_low_total  += n_low
        n_all_total  += n_all
        n_mi_batches += 1

        if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
            break

    # ── 多卡 all-reduce ──────────────────────────────────────────────────────
    if args.use_multi_gpu:
        shape_t = torch.tensor([len(freqs_global) if freqs_global is not None else 0],
                               device=device, dtype=torch.long)
        dist.all_reduce(shape_t, op=dist.ReduceOp.MAX)

        n_f = int(shape_t[0].item())
        if sum_psd_high is not None:
            sum_psd_high = np.zeros(n_f) if n_f > len(sum_psd_high) else sum_psd_high
            sum_psd_low  = np.zeros(n_f) if n_f > len(sum_psd_low)  else sum_psd_low
            sum_psd_all  = np.zeros(n_f) if n_f > len(sum_psd_all)  else sum_psd_all

        def to_t(x):
            return torch.from_numpy(x).to(device=device, dtype=torch.float64)

        t_h = to_t(sum_psd_high) if sum_psd_high is not None else torch.zeros(n_f, device=device)
        t_l = to_t(sum_psd_low)  if sum_psd_low  is not None else torch.zeros(n_f, device=device)
        t_a = to_t(sum_psd_all) if sum_psd_all  is not None else torch.zeros(n_f, device=device)
        t_nh = torch.tensor([float(n_high_total)], device=device)
        t_nl = torch.tensor([float(n_low_total)],  device=device)
        t_na = torch.tensor([float(n_all_total)],  device=device)

        dist.all_reduce(t_h, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_l, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_a, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_nh, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_nl, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_na, op=dist.ReduceOp.SUM)

        sum_psd_high = t_h.cpu().numpy()
        sum_psd_low  = t_l.cpu().numpy()
        sum_psd_all  = t_a.cpu().numpy()
        n_high_total = int(t_nh.item())
        n_low_total  = int(t_nl.item())
        n_all_total  = int(t_na.item())

        if sum_mi is not None:
            t_mi = to_t(sum_mi)
            t_nmb = torch.tensor([float(n_mi_batches)], device=device)
            dist.all_reduce(t_mi, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_nmb, op=dist.ReduceOp.SUM)
            sum_mi = t_mi.cpu().numpy()
            n_mi_batches = int(t_nmb.item())

        dist.barrier()

    # ── 绘图（只在 rank 0） ───────────────────────────────────────────────────
    if rank == 0 and sum_psd_high is not None and n_high_total > 0 and n_low_total > 0:
        psd_high_avg = sum_psd_high / n_high_total
        psd_low_avg  = sum_psd_low  / n_low_total
        psd_all_avg  = sum_psd_all  / n_all_total

        # 平均 MI 曲线（用于直方图）
        avg_mi = sum_mi / n_mi_batches if n_mi_batches > 0 else np.array([])
        q3_avg = float(np.percentile(avg_mi, 75)) if avg_mi.size > 0 else 0.0
        q1_avg = float(np.percentile(avg_mi, 25)) if avg_mi.size > 0 else 0.0

        # ── 峰值检测（低频区间 ≤ 0.1 cycles/hour ≈ period ≥ 10h）───────────────
        peaks_high = find_top_peaks(freqs_global, psd_high_avg, low_freq_max=0.1, n_peaks=3)
        peaks_low  = find_top_peaks(freqs_global, psd_low_avg,  low_freq_max=0.1, n_peaks=3)
        peaks_all  = find_top_peaks(freqs_global, psd_all_avg,  low_freq_max=0.1, n_peaks=3)

        ratio = np.zeros_like(psd_high_avg)
        nonzero = psd_low_avg > 0
        ratio[nonzero] = psd_high_avg[nonzero] / psd_low_avg[nonzero]
        peaks_ratio = find_top_peaks(freqs_global, ratio, low_freq_max=0.1, n_peaks=3)

        # 图1：主对比图
        plot_mean_psd_comparison(
            out_path=os.path.join(out_dir, "psd_high_vs_low_mi.png"),
            freqs=freqs_global,
            psd_high=psd_high_avg,
            psd_low=psd_low_avg,
            psd_all=psd_all_avg,
            label_high="High-MI (Q3+)",
            label_low="Low-MI (<Q1, bottom 25%)",
            label_all="All Patches",
            n_high=n_high_total,
            n_low=n_low_total,
            n_total=n_all_total,
            freq_hours_per_step=args.freq_hours_per_step,
            peaks_high=peaks_high,
            peaks_low=peaks_low,
            peaks_all=peaks_all,
        )

        # 图2：比值图
        plot_psd_ratio(
            out_path=os.path.join(out_dir, "psd_ratio_high_over_low.png"),
            freqs=freqs_global,
            ratio=ratio,
            label_high="High-MI",
            label_low="Low-MI",
            freq_hours_per_step=args.freq_hours_per_step,
            peaks_ratio=peaks_ratio,
        )

        # 图3：MI 直方图
        if avg_mi.size > 0:
            plot_patch_mi_histogram(
                out_path=os.path.join(out_dir, "mi_score_histogram.png"),
                mi_flat=avg_mi,
                q3=q3_avg,
            )

        # 打印统计摘要
        print("\n=== PSD Analysis Summary ===")
        print(f"Total patches:    {n_all_total}  (high={n_high_total}, low={n_low_total})")
        print(f"Total batches:   {n_mi_batches}")
        print(f"PSD freq points: {len(freqs_global)}")
        print(f"Output directory: {out_dir}")

        # 打印各频段能量占比（低频 0~0.1，高频 0.1~0.5）
        eps = 1e-12
        lo_mask = freqs_global <= 0.1
        hi_mask = (freqs_global > 0.1) & (freqs_global <= 0.5)
        df = freqs_global[1] - freqs_global[0] if len(freqs_global) > 1 else 1.0

        for name, psd_avg, n_p in [
            ("High-MI", psd_high_avg, n_high_total),
            ("Low-MI",  psd_low_avg,  n_low_total),
            ("All",     psd_all_avg,  n_all_total),
        ]:
            lo_e = float(np.sum(psd_avg[lo_mask]) * df)
            hi_e = float(np.sum(psd_avg[hi_mask]) * df)
            print(f"  {name:8s}: Low-freq (≤0.1) energy = {lo_e:.4f}  |  High-freq (>0.1) energy = {hi_e:.4f}  |  ratio hi/lo = {hi_e/(lo_e+eps):.4f}")

        # 打印动态峰值（频率 + 周期）
        print(f"\n  [Peak Detection] Top-3 peaks in low-freq band (≤0.1 c/step, period ≥10 steps):")
        for name, peaks in [("High-MI", peaks_high), ("Low-MI", peaks_low), ("All", peaks_all)]:
            if peaks:
                entries = " | ".join([f"f={f:.4f} c/step (T={T:.1f} steps)" for f, _, T in peaks])
                print(f"    {name:8s}: {entries}")
            else:
                print(f"    {name:8s}: (no peaks found)")

        if peaks_ratio:
            print(f"  [Peak Detection] Ratio peaks: " + " | ".join(
                [f"f={f:.4f} c/step (T={T:.1f} steps)" for f, _, T in peaks_ratio]
            ))

        print(f"\nFigures saved to {out_dir}/")

    if args.use_multi_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
