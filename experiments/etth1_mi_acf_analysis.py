#!/usr/bin/env python3
"""
ETTh1 + Timer: 平均自相关函数对比 (Mean ACF Profiles)

实验目标：
    高 MI Patch 的 ACF 曲线是否更"快速衰减"（信息更局部化）？
    低 MI Patch 的 ACF 曲线是否衰减更慢（存在长程周期性依赖）？

操作步骤：
    1. 提取所有 patch 的原始时间序列（patch_len 长度）
    2. 加载 HSIC MI 分数（由 etth1_mi_hsic_peaks.py 生成）
    3. 将 patch 分为高 MI 组（>Q3）和低 MI 组（<=Q3）
    4. 对每个 patch 计算从 lag=0 到 lag=L//2 的归一化 ACF
    5. 同组内所有样本在相同 lag 处求平均
    6. 绘制高/低 MI 组的平均 ACF 对比图，含置信区间

数学定义（归一化 ACF）：
    rho(k) = sum_{t=1}^{n-k} (x_t - mu)(x_{t+k} - mu) / sum_{t=1}^{n} (x_t - mu)^2
    其中 mu = mean(x)，k 为滞后阶数，范围 0 <= k <= L//2

用法（单卡）：
    python experiments/etth1_mi_acf_analysis.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTm1.csv --data ETTm1

用法（多卡，必须用 torchrun）：
    torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_acf_analysis.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTm1.csv --data ETTm1 \
      --use_multi_gpu
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

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
from models.Timer import Model
from utils.masking import TriangularCausalMask


MI_DECODER_LAYER_CAP = 8


# ─── 加载 HSIC MI JSON ────────────────────────────────────────────────────────

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

    if layer_idx < 0:
        actual_layer = num_layers + layer_idx
    else:
        actual_layer = layer_idx

    layer_key = str(actual_layer)
    if layer_key not in layers:
        available = list(layers.keys())
        raise ValueError(f"Layer {layer_key} not found. Available: {available}")

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
        "model_id": "acf_analysis",
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


# ─── ACF 计算 ─────────────────────────────────────────────────────────────────

def normalized_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    计算归一化自相关函数（ACF），从 lag=0 到 lag=max_lag。

    数学定义：
        rho(k) = sum_{t=1}^{n-k} (x_t - mu)(x_{t+k} - mu) / sum_{t=1}^{n} (x_t - mu)^2

    参数:
        x: 1D 时间序列，shape [T]
        max_lag: 最大滞后阶数（通常取 T//2）

    返回: ACF 数组，shape [max_lag + 1]，从 rho(0)=1 开始
    """
    n = len(x)
    max_lag = min(max_lag, n - 1)
    if max_lag < 0:
        return np.array([1.0])

    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    numerator_all = x - mu

    # 分母：序列方差（lag=0 时为方差）
    denom = np.sum(numerator_all ** 2)
    if denom < 1e-12:
        # 常数序列，ACF 全部为 0（除 lag=0 外）
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        return acf

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0  # lag=0 时恒为 1

    for k in range(1, max_lag + 1):
        cov = np.sum(numerator_all[:-k] * numerator_all[k:])
        acf[k] = cov / denom

    return acf


def raw_acf_cov(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, float]:
    """
    计算未归一化的自协方差（原始协方差值），从 lag=0 到 lag=max_lag。

    返回: (cov_array, denom) 其中 cov_array[0] = denom（自身方差）
    """
    n = len(x)
    max_lag = min(max_lag, n - 1)
    if max_lag < 0:
        return np.array([1.0]), 1.0

    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    numerator_all = x - mu
    denom = np.sum(numerator_all ** 2)

    cov = np.zeros(max_lag + 1)
    cov[0] = denom
    for k in range(1, max_lag + 1):
        cov[k] = np.sum(numerator_all[:-k] * numerator_all[k:])

    return cov, denom


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
    返回: [B*N_patches, patch_len, M]
    """
    B, L, M = batch_x.shape
    N = (L - patch_len) // stride + 1

    x_raw = batch_x * stdev + mean
    patches = []
    for p in range(N):
        start = p * stride
        end = start + patch_len
        seg = x_raw[:, start:end, :]
        patches.append(seg)
    patches = torch.stack(patches, dim=0)
    patches = patches.permute(1, 0, 2, 3)
    patches = patches.reshape(B * N, patch_len, M)
    return patches.cpu().numpy()


# ─── 绘图函数 ─────────────────────────────────────────────────────────────────

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.signal import find_peaks


def find_acf_peaks(
    lags: np.ndarray,
    acf: np.ndarray,
    min_lag: int = 3,
    n_peaks: int = 3,
    prominence: float | None = None,
) -> list[tuple[int, float, float]]:
    """
    在 lag >= min_lag 的区间找 ACF 的峰值。

    返回: list of (lag, acf_val, prominence)  按 ACF 降序排列
    """
    if len(acf) <= min_lag:
        return []

    search = acf[min_lag:]
    search_lags = lags[min_lag:]

    if len(search) < 3:
        return []

    prom = prominence if prominence is not None else (np.max(search) - np.min(search)) * 0.05
    peaks, properties = find_peaks(search, prominence=prom)

    if len(peaks) == 0:
        return []

    peak_vals = search[peaks]
    top_idx = np.argsort(peak_vals)[::-1][:n_peaks]

    result = []
    for idx in top_idx:
        lag = int(search_lags[peaks[idx]])
        val = float(peak_vals[idx])
        prom_val = float(properties["prominences"][idx])
        result.append((lag, val, prom_val))
    return result


def plot_mean_acf_comparison(
    out_path: str,
    lags: np.ndarray,
    acf_high: np.ndarray,
    acf_low: np.ndarray,
    acf_all: np.ndarray,
    ci_high: np.ndarray,
    ci_low: np.ndarray,
    n_high: int,
    n_low: int,
    n_total: int,
    confidence_level: float = 0.95,
    peaks_high: list | None = None,
    peaks_low: list | None = None,
) -> None:
    """
    绘制高 MI / 低 MI / 全局平均 ACF 对比图，含 95% 置信区间。
    """
    z_score = 1.96  # 95% 置信区间
    n_ref = min(n_high, n_low, n_total)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ── 左图：完整 ACF（lag=0 到 L//2）───────────────────────────────────────
    ax = axes[0]
    lag_full = np.arange(len(acf_high))

    ax.fill_between(lag_full, -z_score / np.sqrt(n_ref), z_score / np.sqrt(n_ref),
                    alpha=0.12, color="grey", label=f"{confidence_level*100:.0f}% CI")
    ax.plot(lag_full, acf_high, lw=2, color="#e41a1c", label=f"High-MI (>Q3, top 25%) (n={n_high})")
    ax.plot(lag_full, acf_low,  lw=2, color="#377eb8", label=f"Low-MI (<Q1, bottom 25%) (n={n_low})")
    ax.plot(lag_full, acf_all,  lw=1.5, color="grey", ls="--", label=f"All Patches (n={n_total})")

    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(z_score / np.sqrt(n_ref), color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(-z_score / np.sqrt(n_ref), color="grey", ls=":", lw=0.8, alpha=0.6)

    ax.set_xlabel("Lag (time-steps)", fontsize=11)
    ax.set_ylabel("Normalized ACF  ρ(k)", fontsize=11)
    ax.set_title("Mean ACF — Full Range", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim([0, len(lag_full) - 1])
    ax.set_ylim([-0.3, 1.05])

    # 标注峰值
    if peaks_high:
        for lag, val, _ in peaks_high:
            ax.axvline(lag, color="#e41a1c", ls=":", lw=1.0, alpha=0.6)
            ax.annotate(f"H@lag{lag}\nρ={val:.3f}", xy=(lag, val),
                        xytext=(lag + 2, val * 0.85),
                        fontsize=7, color="#e41a1c",
                        arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=0.8))
    if peaks_low:
        for lag, val, _ in peaks_low:
            ax.axvline(lag, color="#377eb8", ls=":", lw=1.0, alpha=0.6)
            ax.annotate(f"L@lag{lag}\nρ={val:.3f}", xy=(lag, val),
                        xytext=(lag + 2, val * 0.7),
                        fontsize=7, color="#377eb8",
                        arrowprops=dict(arrowstyle="->", color="#377eb8", lw=0.8))

    # ── 右图：聚焦低 lag 区间（lag 0 ~ 78）────────────────────────────────────
    ax = axes[1]
    lag_zoom_end = min(78, len(acf_high) - 1)
    lag_zoom = np.arange(lag_zoom_end + 1)

    ax.fill_between(lag_zoom, -z_score / np.sqrt(n_ref), z_score / np.sqrt(n_ref),
                    alpha=0.12, color="grey", label=f"{confidence_level*100:.0f}% CI")
    ax.plot(lag_zoom, acf_high[:lag_zoom_end + 1], lw=2, color="#e41a1c",
            label=f"High-MI (n={n_high})")
    ax.plot(lag_zoom, acf_low[:lag_zoom_end + 1],  lw=2, color="#377eb8",
            label=f"Low-MI (n={n_low})")
    ax.plot(lag_zoom, acf_all[:lag_zoom_end + 1], lw=1.5, color="grey", ls="--",
            label=f"All (n={n_total})")

    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(z_score / np.sqrt(n_ref), color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(-z_score / np.sqrt(n_ref), color="grey", ls=":", lw=0.8, alpha=0.6)

    ax.set_xlabel("Lag (time-steps)", fontsize=11)
    ax.set_ylabel("Normalized ACF  ρ(k)", fontsize=11)
    ax.set_title(f"Mean ACF — Zoomed (lag 0 ~ {lag_zoom_end})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim([0, lag_zoom_end])
    ax.set_ylim([-0.3, 1.05])

    fig.suptitle("High-MI vs Low-MI Patch: Mean Autocorrelation Function (ACF) Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acf_difference(
    out_path: str,
    lags: np.ndarray,
    diff: np.ndarray,
    ci_diff: float,
    peaks_diff: list | None = None,
) -> None:
    """
    绘制高 MI - 低 MI 的 ACF 差值曲线，含 95% 置信区间。
    差值 > 0：High-MI 在该 lag 处自相关更强
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    valid_lags = np.arange(len(diff))
    ax.fill_between(valid_lags, -ci_diff, ci_diff, alpha=0.12, color="grey",
                    label=f"95% CI (noise level)")
    ax.plot(valid_lags, diff, lw=1.5, color="#4daf4a", label="High-MI − Low-MI")
    ax.axhline(0, color="red", ls="--", lw=1, label="zero (equal)")

    ax.set_xlabel("Lag (time-steps)", fontsize=11)
    ax.set_ylabel("Δ ACF  (High-MI − Low-MI)", fontsize=11)
    ax.set_title("ACF Difference: High-MI minus Low-MI", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, len(valid_lags) - 1])

    if peaks_diff:
        for lag, val, _ in peaks_diff:
            color = "#ff7f0e" if val > 0 else "#2ca02c"
            ax.axvline(lag, color=color, ls=":", lw=1.0, alpha=0.7)
            ax.annotate(f"@lag{lag}\nΔ={val:.3f}", xy=(lag, val),
                        xytext=(lag + 1.5, val * (1.3 if val > 0 else 0.5)),
                        fontsize=7.5, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decay_rate_comparison(
    out_path: str,
    lags: np.ndarray,
    acf_high: np.ndarray,
    acf_low: np.ndarray,
    n_high: int,
    n_low: int,
    half_life_high: float | None,
    half_life_low: float | None,
) -> None:
    """
    绘制 ACF 绝对值的对数衰减曲线，对比高/低 MI 组的衰减速度。
    半衰期越小 → 衰减越快 → 信息越局部化。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左图：ACF 绝对值的对数坐标
    ax = axes[0]
    abs_high = np.abs(acf_high)
    abs_low  = np.abs(acf_low)

    ax.semilogy(lags, abs_high, lw=2, color="#e41a1c",
                label=f"High-MI (n={n_high})")
    ax.semilogy(lags, abs_low,  lw=2, color="#377eb8",
                label=f"Low-MI (n={n_low})")

    if half_life_high is not None:
        ax.axvline(half_life_high, color="#e41a1c", ls="--", lw=1.2, alpha=0.7,
                   label=f"High-MI half-life={half_life_high:.1f}")
    if half_life_low is not None:
        ax.axvline(half_life_low, color="#377eb8", ls="--", lw=1.2, alpha=0.7,
                   label=f"Low-MI half-life={half_life_low:.1f}")

    ax.set_xlabel("Lag (time-steps)", fontsize=11)
    ax.set_ylabel("|ACF| (log scale)", fontsize=11)
    ax.set_title("ACF Decay — Log Scale", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim([0, len(lags) - 1])

    # 右图：累积能量（从 lag=0 开始累积到当前 lag 的 ACF^2）
    ax = axes[1]
    cum_high = np.cumsum(acf_high ** 2)
    cum_low  = np.cumsum(acf_low ** 2)
    total_high = cum_high[-1] if cum_high[-1] > 0 else 1.0
    total_low  = cum_low[-1] if cum_low[-1] > 0 else 1.0

    cum_high_norm = cum_high / total_high
    cum_low_norm  = cum_low  / total_low

    ax.plot(lags, cum_high_norm, lw=2, color="#e41a1c",
            label=f"High-MI (n={n_high})")
    ax.plot(lags, cum_low_norm,  lw=2, color="#377eb8",
            label=f"Low-MI (n={n_low})")
    ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.6,
               label="50% energy")

    ax.set_xlabel("Lag (time-steps)", fontsize=11)
    ax.set_ylabel("Cumulative Energy Fraction", fontsize=11)
    ax.set_title("Cumulative ACF² Energy (PACF-style)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, len(lags) - 1])
    ax.set_ylim([0, 1.05])

    fig.suptitle("ACF Decay Rate Comparison: High-MI vs Low-MI",
                 fontsize=13, fontweight="bold")
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


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def half_lag(acf: np.ndarray, threshold: float = 0.5) -> float | None:
    """计算 ACF 首次跌破 threshold（绝对值）时的 lag，作为半衰期指标。"""
    abs_acf = np.abs(acf)
    idx = np.where(abs_acf < threshold)[0]
    if len(idx) == 0:
        return None
    return float(idx[0])


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
    p.add_argument("--out_dir", type=str, default="./results/acf_analysis")
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test loader")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_multi_gpu", action="store_true")
    p.add_argument("--hsic_file", type=str, default=None,
                  help="etth1_mi_hsic_peaks 输出的 JSON 文件路径。")
    p.add_argument("--hsic_layer", type=int, default=-1,
                  help="使用 JSON 中哪一层的 MI 曲线（-1=最后一层）")
    args = p.parse_args()

    rank = 0
    local_rank = 0
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Multi-GPU requires torchrun.")
        if not torch.cuda.is_available():
            raise RuntimeError("use_multi_gpu requires CUDA.")
        n_visible = torch.cuda.device_count()
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > n_visible:
            raise RuntimeError(f"WORLD_SIZE={world_size} but only {n_visible} CUDA device(s).")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        if world_size > 1:
            print(f"[acf] rank {rank}/{world_size}: starting...", flush=True)

    ns = build_namespace(args)
    ns.use_multi_gpu = bool(args.use_multi_gpu)
    ns.use_gpu = bool(torch.cuda.is_available())

    # 自动生成带时间戳的输出目录
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

    # 加载全局 HSIC MI 曲线
    global_mi_curve = None
    global_q3 = None
    if args.hsic_file:
        print(f"[ACF] Loading global MI curve from: {args.hsic_file}")
        global_mi_curve, global_q3, global_q1, N = load_hsic_mi_curve(args.hsic_file, args.hsic_layer)
        print(f"[ACF] Loaded: layer={args.hsic_layer}, N={N}, Q3={global_q3:.6f}, Q1={global_q1:.6f}")
    else:
        print("[ACF] Warning: --hsic_file not specified. Using runtime batch-level MI (legacy).")

    max_lag = patch_len // 2

    # ── 收集统计量 ──────────────────────────────────────────────────────────────
    sum_acf_high = None
    sum_acf_low = None
    sum_acf_all = None
    n_high_total = 0
    n_low_total = 0
    n_all_total = 0
    sum_denom_high = 0.0
    sum_denom_low  = 0.0
    sum_denom_all  = 0.0
    lags_global = np.arange(max_lag + 1)

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

        # 使用全局 HSIC MI 曲线
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

        # 反归一化并提取 patch 时间序列
        means = batch_x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(batch_x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        patches_raw = extract_patches_from_batch(batch_x, patch_len, stride, means, stdev)

        # 多变量求平均 → [B*N, patch_len] 标量时间序列
        if patches_raw.shape[2] > 1:
            patches_ts = patches_raw.mean(axis=2)
        else:
            patches_ts = patches_raw[:, :, 0]

        for pi in range(N):
            for bi in range(B):
                idx = bi * N + pi
                ts = patches_ts[idx]
                cov_i, denom_i = raw_acf_cov(ts, max_lag)

                if sum_acf_all is None:
                    n_lag = len(cov_i)
                    sum_acf_high = np.zeros(n_lag)
                    sum_acf_low  = np.zeros(n_lag)
                    sum_acf_all  = np.zeros(n_lag)
                    sum_denom_high = 0.0
                    sum_denom_low  = 0.0
                    sum_denom_all  = 0.0

                sum_acf_all += cov_i
                sum_denom_all += denom_i
                mi_val = mi_curve[pi]
                if np.isfinite(mi_val) and mi_val > q3:
                    sum_acf_high += cov_i
                    sum_denom_high += denom_i
                    n_high_total += 1
                elif np.isfinite(mi_val) and mi_val < q1:
                    sum_acf_low += cov_i
                    sum_denom_low += denom_i
                    n_low_total += 1
                n_all_total += 1

        if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
            break

    # ── 多卡 all-reduce ──────────────────────────────────────────────────────
    if args.use_multi_gpu:
        def to_t(x):
            return torch.from_numpy(x).to(device=device, dtype=torch.float64)

        t_h = to_t(sum_acf_high) if sum_acf_high is not None else torch.zeros(max_lag + 1, device=device)
        t_l = to_t(sum_acf_low)  if sum_acf_low  is not None else torch.zeros(max_lag + 1, device=device)
        t_a = to_t(sum_acf_all)  if sum_acf_all  is not None else torch.zeros(max_lag + 1, device=device)
        t_nh = torch.tensor([float(n_high_total)], device=device)
        t_nl = torch.tensor([float(n_low_total)],  device=device)
        t_na = torch.tensor([float(n_all_total)],  device=device)
        t_dh = torch.tensor([sum_denom_high], device=device)
        t_dl = torch.tensor([sum_denom_low],  device=device)
        t_da = torch.tensor([sum_denom_all],  device=device)

        dist.all_reduce(t_h, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_l, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_a, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_nh, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_nl, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_na, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_dh, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_dl, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_da, op=dist.ReduceOp.SUM)

        sum_acf_high = t_h.cpu().numpy()
        sum_acf_low  = t_l.cpu().numpy()
        sum_acf_all  = t_a.cpu().numpy()
        n_high_total = int(t_nh.item())
        n_low_total  = int(t_nl.item())
        n_all_total  = int(t_na.item())
        sum_denom_high = t_dh.item()
        sum_denom_low  = t_dl.item()
        sum_denom_all  = t_da.item()

        dist.barrier()

    # ── 绘图（只在 rank 0） ───────────────────────────────────────────────────
    if rank == 0 and sum_acf_high is not None and n_high_total > 0 and n_low_total > 0:
        denom_high = sum_denom_high if sum_denom_high > 0 else 1.0
        denom_low  = sum_denom_low  if sum_denom_low  > 0 else 1.0
        denom_all  = sum_denom_all  if sum_denom_all  > 0 else 1.0

        acf_high_avg = sum_acf_high / denom_high
        acf_low_avg  = sum_acf_low  / denom_low
        acf_all_avg  = sum_acf_all  / denom_all

        lags = np.arange(len(acf_high_avg))
        z_score = 1.96
        n_ref = min(n_high_total, n_low_total, n_all_total)
        ci = z_score / np.sqrt(n_ref)

        # ── 图1：平均 ACF 对比 ────────────────────────────────────────────────
        peaks_high = find_acf_peaks(lags, acf_high_avg, min_lag=3, n_peaks=3)
        peaks_low  = find_acf_peaks(lags, acf_low_avg,  min_lag=3, n_peaks=3)

        plot_mean_acf_comparison(
            out_path=os.path.join(out_dir, "acf_high_vs_low_mi.png"),
            lags=lags,
            acf_high=acf_high_avg,
            acf_low=acf_low_avg,
            acf_all=acf_all_avg,
            ci_high=np.full_like(acf_high_avg, ci),
            ci_low=np.full_like(acf_low_avg, ci),
            n_high=n_high_total,
            n_low=n_low_total,
            n_total=n_all_total,
            confidence_level=0.95,
            peaks_high=peaks_high,
            peaks_low=peaks_low,
        )

        # ── 图2：ACF 差值曲线 ─────────────────────────────────────────────────
        diff = acf_high_avg - acf_low_avg
        peaks_diff = find_acf_peaks(lags, diff, min_lag=1, n_peaks=3)
        plot_acf_difference(
            out_path=os.path.join(out_dir, "acf_difference_high_minus_low.png"),
            lags=lags,
            diff=diff,
            ci_diff=ci * np.sqrt(2),
            peaks_diff=peaks_diff,
        )

        # ── 图3：衰减速度对比 ─────────────────────────────────────────────────
        hl_high = half_lag(acf_high_avg, threshold=0.5)
        hl_low  = half_lag(acf_low_avg,  threshold=0.5)
        plot_decay_rate_comparison(
            out_path=os.path.join(out_dir, "acf_decay_rate_comparison.png"),
            lags=lags,
            acf_high=acf_high_avg,
            acf_low=acf_low_avg,
            n_high=n_high_total,
            n_low=n_low_total,
            half_life_high=hl_high,
            half_life_low=hl_low,
        )

        # ── 图4：MI 直方图 ────────────────────────────────────────────────────
        if global_mi_curve is not None:
            plot_patch_mi_histogram(
                out_path=os.path.join(out_dir, "mi_score_histogram.png"),
                mi_flat=global_mi_curve,
                q3=global_q3,
            )

        # ── 打印统计摘要 ──────────────────────────────────────────────────────
        print("\n=== ACF Analysis Summary ===")
        print(f"Total patches:    {n_all_total}  (high={n_high_total}, low={n_low_total})")
        print(f"Max lag:         {max_lag}")
        print(f"95% CI half-width: ±{ci:.4f}")
        print(f"Output directory: {out_dir}")

        print(f"\n  [Half-life] lag at which |ACF| first < 0.5:")
        print(f"    High-MI: {'{:.1f}'.format(hl_high) if hl_high else 'N/A (> max_lag)'}")
        print(f"    Low-MI:  {'{:.1f}'.format(hl_low)  if hl_low  else 'N/A (> max_lag)'}")

        print(f"\n  [Lag=1 ACF] Early dependence strength:")
        print(f"    High-MI ρ(1) = {acf_high_avg[1]:.4f}")
        print(f"    Low-MI  ρ(1) = {acf_low_avg[1]:.4f}")
        print(f"    Δ = {diff[1]:+.4f}  ({'High-MI stronger' if diff[1] > 0 else 'Low-MI stronger'})")

        print(f"\n  [Peak Detection] Significant ACF peaks (lag >= 3):")
        for name, peaks in [("High-MI", peaks_high), ("Low-MI", peaks_low)]:
            if peaks:
                entries = " | ".join([f"lag={lag}  ρ={val:.3f}" for lag, val, _ in peaks])
                print(f"    {name:8s}: {entries}")
            else:
                print(f"    {name:8s}: (no significant peaks)")

        if peaks_diff:
            print(f"  [Difference Peaks]: " + " | ".join(
                [f"lag={lag}  Δ={val:+.3f}" for lag, val, _ in peaks_diff]
            ))

        # 低频能量比（lag>10 的 ACF 绝对值之和）
        lo_mask = lags > 10
        lo_e_high = float(np.sum(np.abs(acf_high_avg[lo_mask])))
        lo_e_low  = float(np.sum(np.abs(acf_low_avg[lo_mask])))
        print(f"\n  [Long-range dependence] Sum(|ACF|) for lag > 10:")
        print(f"    High-MI: {lo_e_high:.4f}")
        print(f"    Low-MI:  {lo_e_low:.4f}")
        print(f"    Ratio high/low: {lo_e_high / (lo_e_low + 1e-12):.3f}")

        print(f"\nFigures saved to {out_dir}/")

    if args.use_multi_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
