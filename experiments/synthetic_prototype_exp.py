#!/usr/bin/env python3
"""
合成数据原型注入实验：源函数 → 常量序列 泛化研究

功能：
1. 源函数（提取原型）: 低频正弦波 y = sin(ω_low t)
2. 目标函数（接收注入）: 常量序列 y = 1
3. 用高/低 MI 原型分别引导常量序列预测，对比泛化效果

MI 计算逻辑（与 etth1_mi_hsic_peaks.py 一致）：
    - 使用 HSIC 作为互信息代理
    - Q3 分组：MI > Q3 为高 MI 组，MI <= Q3 为低 MI 组
    - 从每组中随机采样多个 patch 的原型，取平均

用法：
    python experiments/synthetic_prototype_exp.py
"""

import argparse
import os
import sys
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from utils.masking import TriangularCausalMask

# 与 etth1_mi_hsic_peaks.py 一致
MI_DECODER_LAYER_CAP = 8


# ============================================================
# HSIC (Mutual Information Proxy) 计算函数（与 etth1_mi_hsic_peaks.py 完全一致）
# ============================================================

def _median_sq_bandwidth(X: torch.Tensor) -> torch.Tensor:
    """Median heuristic: median pairwise squared Euclidean distance."""
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
    """
    Unbiased HSIC with Gaussian RBF kernels.
    Bandwidth sigma^2 = median pairwise squared Euclidean distance.
    """
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
    """
    全局 HSIC：对所有样本一次性计算每个 patch 的依赖分数。
    与 etth1_mi_hsic_peaks.py 的 mi_sequence_hsic 完全一致。

    Hbm: [total_samples, N, D] patch 表示
    Hy: [total_samples, N, D] 未来窗口表示
    Returns: [N] 每个 patch 位置的 HSIC 分数
    """
    total_samples, N_x, D = Hbm.shape
    _, N_y, _ = Hy.shape

    hy_single = Hy[:, 0, :].unsqueeze(1)  # [total_samples, 1, D]
    Hy_expanded = hy_single.expand(-1, N_x, -1)  # [total_samples, N_x, D]

    out = []
    for p in range(N_x):
        X = zscore_cols(Hbm[:, p, :].contiguous())
        Y = zscore_cols(Hy_expanded[:, p, :].contiguous())
        out.append(hsic_unbiased_gaussian(X, Y).detach().float().cpu().item())
    return np.asarray(out, dtype=np.float64)


def q3_and_high_patches(m: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Third quartile of curve m; patch indices where MI is strictly above Q3.
    与 etth1_mi_hsic_peaks.py 的 q3_and_high_patches 完全一致。
    """
    if m.size == 0 or not np.all(np.isfinite(m)):
        return float("nan"), np.array([], dtype=int)
    q3 = float(np.percentile(m, 75))
    return q3, np.where(m > q3)[0]


def top_k_patches(m: np.ndarray, k_high: Optional[int] = None, k_low: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    按 MI 分数排序，取 top-k 最高的和最低的 patch。

    Args:
        m: MI 分数数组 [N]
        k_high: 高 MI 组取前 k_high 个（默认 None = 使用 Q3 分组）
        k_low: 低 MI 组取前 k_low 个（默认 None = 使用 Q3 分组）

    Returns:
        high_patches: 高 MI patch 索引数组
        low_patches: 低 MI patch 索引数组
        threshold: 分界阈值（最高 k_high 和最低 k_low 之间的分界值）
    """
    if m.size == 0 or not np.all(np.isfinite(m)):
        return np.array([], dtype=int), np.array([], dtype=int), float("nan")

    # 排序
    sorted_idx = np.argsort(m)[::-1]  # 降序排列

    n = len(m)
    k_h = min(k_high, n) if k_high is not None else max(1, n // 4)  # 默认取前 25%
    k_l = min(k_low, n) if k_low is not None else max(1, n // 4)    # 默认取后 25%

    high_patches = sorted_idx[:k_h]  # 最高 k_h 个
    low_patches = sorted_idx[-k_l:]  # 最低 k_l 个

    # 分界阈值 = low_patches 中最高的分数
    threshold = float(m[low_patches].max()) if len(low_patches) > 0 else float("nan")

    return high_patches, low_patches, threshold


# ============================================================
# Decoder 层特征提取（与 etth1_mi_hsic_peaks.py 完全一致）
# ============================================================

def _unwrap_timer(model: torch.nn.Module):
    return model.module if hasattr(model, "module") else model


def forward_collect_layers(model: torch.nn.Module, x_enc: torch.Tensor):
    """
    x_enc: [B, L, M] in Timer convention (before permute).
    Returns up to MI_DECODER_LAYER_CAP tensors [B, N, D]: output after each of the first K decoder
    attention blocks only (K = min(MI_DECODER_LAYER_CAP, len(attn_layers))).
    """
    core = _unwrap_timer(model)
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev

    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape
    assert BM == B * n_vars

    def pool(z: torch.Tensor) -> torch.Tensor:
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers: list[torch.Tensor] = []

    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _, _ = o3(h, attn_mask=mask)
        layers.append(pool(h.detach()))
    return layers, int(n_vars), int(N)


def forward_y_collect_layers(model: torch.nn.Module, y: torch.Tensor):
    """
    将真值 y 也通过 decoder 提取隐表示。
    y: [B, L, M] 原始真值
    返回每一层的 h_y: list[Tensor], 每个 shape [B, N, D]
    """
    core = _unwrap_timer(model)
    B, L, M = y.shape
    means = y.mean(1, keepdim=True).detach()
    x = y - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev

    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    def pool(z: torch.Tensor) -> torch.Tensor:
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers: list[torch.Tensor] = []

    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _, _ = o3(h, attn_mask=mask)
        layers.append(pool(h.detach()))
    return layers


class LinearDataset(Dataset):
    """常数值数据集 y = 1 - 用于原型提取（源函数，导数为0，无变化）"""
    def __init__(self, n_samples=1000, seq_len=96):
        self.n_samples = n_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = np.linspace(0, 1, self.seq_len)
        y = np.ones(self.seq_len)  # 常数值 y = 1，shape [seq_len]
        x_mark = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        return torch.FloatTensor(y).unsqueeze(-1), torch.FloatTensor(x_mark)


class TriangularDataset(Dataset):
    """三角波数据集 - 用于原型提取（源函数，低频、导数变化极慢）"""
    def __init__(self, n_samples=1000, seq_len=96, period=1.0, amplitude=1.0):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.period = period
        self.amplitude = amplitude

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = np.linspace(0, 1, self.seq_len)
        y = self.amplitude * np.abs(2 * (t / self.period - np.floor(t / self.period + 0.5)))  # 三角波，shape [seq_len]
        x_mark = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        return torch.FloatTensor(y).unsqueeze(-1), torch.FloatTensor(x_mark)


class ConstantDataset(Dataset):
    """常数值数据集 y = 1 - 用于测试泛化（目标函数，预测常量序列）"""
    def __init__(self, n_samples=500, seq_len=96):
        self.n_samples = n_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        y = np.ones(self.seq_len)  # 常数值 y = 1，shape [seq_len]
        x_mark = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        t_for_plot = np.linspace(0, 1, self.seq_len)
        return torch.FloatTensor(y).unsqueeze(-1), torch.FloatTensor(x_mark), torch.FloatTensor(t_for_plot), 1.0, 0.0


class SinDataset(Dataset):
    """低频正弦波数据集 - 用于原型提取（源函数）"""
    def __init__(self, n_samples=1000, seq_len=96, omega=2 * np.pi):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.omega = omega

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = np.linspace(0, 1, self.seq_len)
        y = np.sin(self.omega * t)  # 低频正弦波
        x_mark = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        return torch.FloatTensor(y).unsqueeze(-1), torch.FloatTensor(x_mark)


class HighFreqSinDataset(Dataset):
    """高频正弦波数据集 y = sin(ω_high t) - 用于测试泛化（目标函数）"""
    def __init__(self, n_samples=500, seq_len=96, omega_high=None):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.omega_high = omega_high if omega_high is not None else (2 * np.pi * 10)  # 高频 omega

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = np.linspace(0, 1, self.seq_len)
        y = np.sin(self.omega_high * t)  # 高频正弦波
        x_mark = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        t_for_plot = np.linspace(0, 1, self.seq_len)
        return torch.FloatTensor(y).unsqueeze(-1), torch.FloatTensor(x_mark), torch.FloatTensor(t_for_plot), 1.0, 0.0


class SgnSinDataset(Dataset):
    """符号函数数据集 y = sgn(sin(ωt)) - 用于测试泛化"""
    def __init__(self, n_samples=500, seq_len=96, omega=None):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.omega = omega if omega is not None else (2 * np.pi * np.random.randint(1, 5))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = np.linspace(0, 1, self.seq_len)
        y = np.sign(np.sin(self.omega * t))  # 固定 y = sgn(sin(ωt))
        x_mark = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        # 返回用于绘图的时间轴
        t_for_plot = np.linspace(0, 1, self.seq_len)
        return torch.FloatTensor(y).unsqueeze(-1), torch.FloatTensor(x_mark), torch.FloatTensor(t_for_plot), 1.0, 0.0


def extract_prototypes_from_sin(
    model,
    source_loader,
    device,
    seq_len,
    pred_len,
    layer_idx=7,
    use_q3=True,
    k_high=None,
    k_low=None,
):
    """
    从源函数数据集提取高/低 MI 原型（基于 HSIC 计算）

    支持两种分组模式：
        - use_q3=True: 使用 Q3 分组（与 etth1_mi_hsic_peaks.py 一致）
        - use_q3=False: 使用 top-k 分组（取 MI 最高的 k_high 个和最低的 k_low 个）

    Args:
        model: Timer 模型
        source_loader: 源函数数据加载器
        device: 设备
        seq_len: 序列长度
        pred_len: 预测长度
        layer_idx: 提取的层索引 (默认7=最后一层)
        use_q3: 是否使用 Q3 分组（默认 True）
        k_high: top-k 分组时高 MI 组 patch 数
        k_low: top-k 分组时低 MI 组 patch 数

    Returns:
        h_proto_high: 高 MI 组原型 [1, 1, D]
        h_proto_low: 低 MI 组原型 [1, 1, D]
        mi_curve: 每个 patch 的 HSIC 分数 [N]
        threshold: Q3 阈值或 top-k 分界阈值
        high_patches: 高 MI patch 索引
        low_patches: 低 MI patch 索引
    """
    model.eval()

    # 收集所有样本的 patch 表示和未来窗口表示
    h_x_list = []
    h_y_list = []

    with torch.no_grad():
        for batch_y, batch_x_mark in source_loader:
            batch_y = batch_y.float().to(device)
            B = batch_y.shape[0]

            # Normalization
            means = batch_y.mean(1, keepdim=True).detach()
            x = batch_y - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = x / stdev

            # Patch embedding
            x_perm = x_norm.permute(0, 2, 1)  # [B, 1, seq_len]
            dec_in, n_vars = model.enc_embedding(x_perm)  # [B, N, D]
            B_dim, N, D = dec_in.shape

            # 获取指定层输出
            mask = None
            h = dec_in
            for i, attn_layer in enumerate(model.decoder.attn_layers):
                if i >= layer_idx:
                    break
                h, _, _ = attn_layer(h, attn_mask=mask, tau=None, delta=None)

            # Reshape: [B, N, D] -> [B, n_vars, N, D] -> [B, n_vars, N, D].mean(dim=1)
            h_pooled = h.view(B, n_vars, N, D).mean(dim=1)

            # 未来窗口：取最后 pred_len 个时间步作为未来
            y_future = batch_y[:, -pred_len:, :]
            y_norm = y_future  # 已归一化

            # 未来窗口也通过 patch embedding
            y_perm = y_norm.permute(0, 2, 1)  # [B, 1, pred_len]
            dec_in_y, _ = model.enc_embedding(y_perm)  # [B, N_y, D]
            h_y_pooled = dec_in_y.view(B, n_vars, -1, D).mean(dim=1)  # [B, N_y, D]

            h_x_list.append(h_pooled.detach())
            h_y_list.append(h_y_pooled.detach())

    # 合并所有 batch
    h_x_all = torch.cat(h_x_list, dim=0)  # [total_samples, N, D]
    h_y_all = torch.cat(h_y_list, dim=0)  # [total_samples, N, D]
    total_samples = h_x_all.shape[0]

    # 计算每个 patch 的 HSIC（全数据集一次性计算）
    mi_curve = mi_sequence_hsic(h_x_all, h_y_all)  # [N]

    # 分组
    if use_q3:
        # Q3 分组（与 etth1_mi_hsic_peaks.py 一致）
        threshold, high_patches = q3_and_high_patches(mi_curve)
        low_patches = np.array([p for p in range(N) if p not in high_patches], dtype=int)
        group_desc = f"Q3={threshold:.6f}"
    else:
        # top-k 分组
        high_patches, low_patches, threshold = top_k_patches(mi_curve, k_high=k_high, k_low=k_low)
        group_desc = f"Top-{k_high}/{k_low}"

    print(f"[HSIC] MI curve: {mi_curve}")
    print(f"[HSIC] Grouping: {group_desc}")
    print(f"[HSIC] High-MI patches (n={len(high_patches)}): {high_patches.tolist()}")
    print(f"[HSIC] Low-MI patches (n={len(low_patches)}): {low_patches.tolist()}")

    # 高 MI 组原型：对所有样本、高 MI patch 取平均
    if len(high_patches) > 0:
        h_proto_high = h_x_all[:, high_patches.copy(), :].mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
    else:
        h_proto_high = h_x_all.mean(dim=(0, 1), keepdim=True)

    # 低 MI 组原型：对所有样本、低 MI patch 取平均
    if len(low_patches) > 0:
        h_proto_low = h_x_all[:, low_patches.copy(), :].mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
    else:
        h_proto_low = h_x_all.mean(dim=(0, 1), keepdim=True)

    print(f"[HSIC] High-MI prototype shape: {h_proto_high.shape}")
    print(f"[HSIC] Low-MI prototype shape: {h_proto_low.shape}")

    return h_proto_high, h_proto_low, mi_curve, threshold, high_patches, low_patches


def plot_3way_prediction_comparison(
    times, baselines, high_proto_preds, low_proto_preds, trues,
    a_values, b_values, save_path, n_show=12
):
    """绘制 Baseline vs High-MI Proto vs Low-MI Proto 的预测对比图"""
    n_show = min(n_show, len(times))
    n_cols = 4
    n_rows = n_show

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
    fig.suptitle("Baseline / High-MI Proto / Low-MI Proto: Constant (y=1) Prediction", fontsize=14, fontweight="bold")

    # 生成预测区间的时间轴 (从输入序列末尾开始)
    full_seq_len = len(times[0])
    pred_len = len(baselines[0])
    t_pred = np.linspace(0, 1, pred_len)  # 预测段归一化时间

    for i in range(n_show):
        for col in range(n_cols):
            ax = axes[i, col]

            if col == 0:
                # Baseline
                ax.plot(t_pred, baselines[i], 'b-', alpha=0.8, linewidth=1.5, label='Baseline')
                ax.plot(t_pred, trues[i], 'k--', alpha=0.4, linewidth=1)
            elif col == 1:
                # High-MI Proto
                ax.plot(t_pred, high_proto_preds[i], 'r-', alpha=0.8, linewidth=1.5, label='High-MI')
                ax.plot(t_pred, trues[i], 'k--', alpha=0.4, linewidth=1)
            elif col == 2:
                # Low-MI Proto
                ax.plot(t_pred, low_proto_preds[i], 'g-', alpha=0.8, linewidth=1.5, label='Low-MI')
                ax.plot(t_pred, trues[i], 'k--', alpha=0.4, linewidth=1)
            else:
                # All together
                ax.plot(t_pred, baselines[i], 'b-', alpha=0.7, linewidth=1, label='Baseline')
                ax.plot(t_pred, high_proto_preds[i], 'r-', alpha=0.7, linewidth=1, label='High-MI')
                ax.plot(t_pred, low_proto_preds[i], 'g-', alpha=0.7, linewidth=1, label='Low-MI')
                ax.plot(t_pred, trues[i], 'k--', alpha=0.5, linewidth=1.2, label='True')

            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3)

            # 标签
            if i == 0:
                titles = ['Baseline', 'High-MI Proto', 'Low-MI Proto', 'All']
                ax.set_title(titles[col], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"#{i+1}", fontsize=8)

    # 添加图例到最后一列
    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_3way_scatter_comparison(
    baselines, high_proto_preds, low_proto_preds, trues,
    save_path
):
    """绘制三组对比的散点图和分布"""
    base_mses = np.array([((b - t) ** 2).mean() for b, t in zip(baselines, trues)])
    high_mses = np.array([((p - t) ** 2).mean() for p, t in zip(high_proto_preds, trues)])
    low_mses = np.array([((p - t) ** 2).mean() for p, t in zip(low_proto_preds, trues)])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 上排：散点图对比
    max_val = max(base_mses.max(), high_mses.max(), low_mses.max())

    # Baseline vs High-MI
    ax = axes[0, 0]
    ax.scatter(base_mses, high_mses, alpha=0.5, s=15, c='red', label='High-MI')
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax.set_xlabel("Baseline MSE")
    ax.set_ylabel("High-MI MSE")
    ax.set_title("Baseline vs High-MI Proto")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val * 1.1])
    ax.set_ylim([0, max_val * 1.1])

    # Baseline vs Low-MI
    ax = axes[0, 1]
    ax.scatter(base_mses, low_mses, alpha=0.5, s=15, c='green', label='Low-MI')
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax.set_xlabel("Baseline MSE")
    ax.set_ylabel("Low-MI MSE")
    ax.set_title("Baseline vs Low-MI Proto")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val * 1.1])
    ax.set_ylim([0, max_val * 1.1])

    # High-MI vs Low-MI
    ax = axes[0, 2]
    ax.scatter(high_mses, low_mses, alpha=0.5, s=15, c='purple', label='Low-MI')
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax.set_xlabel("High-MI MSE")
    ax.set_ylabel("Low-MI MSE")
    ax.set_title("High-MI vs Low-MI Proto")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val * 1.1])
    ax.set_ylim([0, max_val * 1.1])

    # 下排：MSE 分布直方图
    bins = np.linspace(0, max_val * 1.1, 40)

    ax = axes[1, 0]
    ax.hist(base_mses, bins=bins, alpha=0.5, label='Baseline', color='blue')
    ax.hist(high_mses, bins=bins, alpha=0.5, label='High-MI', color='red')
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("MSE Distribution: Baseline vs High-MI")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(base_mses, bins=bins, alpha=0.5, label='Baseline', color='blue')
    ax.hist(low_mses, bins=bins, alpha=0.5, label='Low-MI', color='green')
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("MSE Distribution: Baseline vs Low-MI")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 改善率分布
    ax = axes[1, 2]
    imp_high = (base_mses - high_mses) / (base_mses + 1e-8) * 100
    imp_low = (base_mses - low_mses) / (base_mses + 1e-8) * 100
    ax.hist(imp_high, bins=40, alpha=0.5, label='High-MI', color='red')
    ax.hist(imp_low, bins=40, alpha=0.5, label='Low-MI', color='green')
    ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Improvement over Baseline (%)")
    ax.set_ylabel("Count")
    ax.set_title("Improvement Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_3way_aggregated_comparison(all_results, save_path):
    """绘制三组聚合统计对比图"""
    metrics = ["MSE", "MAE"]
    categories = list(all_results.keys())
    n_cats = len(categories)

    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5))
    if n_cats == 1:
        axes = [axes]

    colors = ['#3498db', '#e74c3c', '#2ecc71'][:n_cats]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [all_results[cat].get(metric, 0) for cat in categories]

        bars = ax.bar(categories, values, color=colors[:n_cats], alpha=0.8, edgecolor='black')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 添加改善百分比标注
        baseline_val = values[0]
        for i, val in enumerate(values[1:], 1):
            delta = val - baseline_val
            pct = delta / baseline_val * 100
            arrow = "↓" if delta < 0 else "↑"
            ax.text(bar.get_x() + bar.get_width()/2, val / 2,
                   f"{arrow}{abs(pct):.1f}%", ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def metric(y_true, y_pred):
    """计算评估指标"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"MSE": float(mse), "MAE": float(mae)}


def run_prediction(model, loader, args, device):
    """运行预测并收集结果"""
    preds = []
    trues = []
    times_list = []
    a_list = []
    b_list = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            # 兼容 2 元素 (source dataset) 和 5 元素 (target dataset)
            if len(batch) == 5:
                batch_y, batch_x_mark, batch_t, batch_a, batch_b = batch
            else:
                batch_y, batch_x_mark = batch
                batch_t = torch.linspace(0, 1, args.pred_len).repeat(batch_y.size(0), 1)
                batch_a = torch.zeros(batch_y.size(0))
                batch_b = torch.zeros(batch_y.size(0))
            batch_y = batch_y.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :args.seq_len - args.pred_len, :], dec_inp], dim=1)

            outputs = model(batch_y, batch_x_mark, dec_inp, batch_x_mark)
            pred = outputs[:, -args.pred_len:, :].cpu().numpy()
            true = batch_y[:, -args.pred_len:, :].cpu().numpy()

            for j in range(len(pred)):
                preds.append(pred[j, :, 0])
                trues.append(true[j, :, 0])
                times_list.append(batch_t[j].numpy() if isinstance(batch_t[j], torch.Tensor) else batch_t[j])
                a_list.append(batch_a[j].item() if hasattr(batch_a[j], 'item') else batch_a[j])
                b_list.append(batch_b[j].item() if hasattr(batch_b[j], 'item') else batch_b[j])

    return preds, trues, times_list, a_list, b_list


def main():
    p = argparse.ArgumentParser(description="合成数据原型注入实验 (三组对比)")
    p.add_argument("--ckpt_path", type=str, default="checkpoints/Timer_forecast_1.0.ckpt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sin_samples", type=int, default=500, help="源函数样本数（用于原型提取）")
    p.add_argument("--source_type", type=str, default="sin", choices=["linear", "triangular", "sin"], help="源函数类型: linear(常量y=1), triangular(三角波), sin(正弦波)")
    p.add_argument("--test_samples", type=int, default=500, help="测试样本数")
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_dir", type=str, default="./results/synthetic_exp")
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--patch_len", type=int, default=96)
    p.add_argument("--stride", type=int, default=96)
    p.add_argument("--extract_layer", type=int, default=7, help="提取原型的层索引 (0-7)")
    p.add_argument("--top_k", type=int, default=0, help="top-k 分组：取 MI 最高的 top-k 个和最低的 top-k 个；0=使用 Q3 分组")
    p.add_argument("--k_low", type=int, default=0, help="低 MI 组 patch 数（默认等于 --top_k）")
    args = p.parse_args()

    # top_k 兼容：k_low 默认等于 top_k
    if args.k_low == 0:
        args.k_low = args.top_k

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 计算 patch 数
    n_patches = args.seq_len // args.patch_len

    # ── 1. 准备数据集 ──────────────────────────────────────────────
    print("\n[1] 准备数据集...")
    
    # 源函数数据集（用于原型提取）
    if args.source_type == "linear":
        source_dataset = LinearDataset(n_samples=args.sin_samples, seq_len=args.seq_len)
        source_name = "Constant (y=1)"
    elif args.source_type == "triangular":
        source_dataset = TriangularDataset(n_samples=args.sin_samples, seq_len=args.seq_len)
        source_name = "Triangular Wave"
    else:  # sin
        source_dataset = SinDataset(n_samples=args.sin_samples, seq_len=args.seq_len)
        source_name = "Sine Wave"
    
    print("="*70)
    print(f"合成数据原型注入实验: {source_name} → Constant y=1 (Baseline / High-MI / Low-MI)")
    print("="*70)
    
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 目标函数数据集（常量 y=1）
    target_dataset = ConstantDataset(n_samples=args.test_samples, seq_len=args.seq_len)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  源函数 ({source_name}) 样本数: {len(source_dataset)}")
    print(f"  目标函数 (Constant y=1) 样本数: {len(target_dataset)}")
    print(f"  序列长度: {args.seq_len}, 预测长度: {args.pred_len}, Patch数: {n_patches}")

    # ── 2. 加载模型 ───────────────────────────────────────────────
    print("\n[2] 加载 Timer 模型...")
    from argparse import Namespace
    config = Namespace(
        task_name='forecast',
        is_training=0,
        is_finetuning=0,
        train_test=0,
        d_layers=1,
        target='OT',
        checkpoints='./checkpoints/',
        inverse=False,
        use_amp=False,
        use_weight_decay=0,
        weight_decay=0.01,
        loss='MSE',
        lradj='type1',
        train_epochs=0,
        patience=3,
        learning_rate=1e-4,
        itr=1,
        finetune_epochs=0,
        output_attention=False,
        distil=True,
        model_id='synthetic_exp',
        model='Timer',
        output_len_list=None,
        mask_rate=0.25,
        data_type='custom',
        decay_fac=0.75,
        cos_warm_up_steps=100,
        cos_max_decay_steps=60000,
        cos_max_decay_epoch=10,
        cos_max=1e-4,
        cos_min=2e-6,
        dropout=args.dropout,
        activation=args.activation,
        embed='timeF',
        freq='h',
        features='M',
        stride=args.stride,
        seq_len=args.seq_len,
        label_len=args.seq_len - args.pred_len,
        pred_len=args.pred_len,
        output_len=args.pred_len,
        patch_len=args.patch_len,
        d_model=args.d_model,
        d_ff=args.d_model * 2,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        factor=args.factor,
        use_prototype=False,
        prototype_path=None,
        prototype_scale=1.0,
        ckpt_path=args.ckpt_path,
    )

    model = Model(config).to(device)
    model.eval()
    print(f"  模型加载完成 (device: {device})")

    # ── 3. 提取高/低 MI 原型（支持 top-k 或 Q3 分组）─────────────────────
    use_q3 = (args.top_k == 0)  # top_k=0 表示使用 Q3 分组
    group_mode = "Q3 分组" if use_q3 else f"Top-{args.top_k} 分组"
    print(f"\n[3] 从 sin 数据集提取原型 (Layer {args.extract_layer}, {group_mode})...")

    if use_q3:
        # 使用 Q3 分组
        (
            h_proto_high, h_proto_low, mi_curve, q3, high_patches, low_patches
        ) = extract_prototypes_from_sin(
            model, source_loader, device, args.seq_len, args.pred_len,
            layer_idx=args.extract_layer, use_q3=True
        )
        threshold = q3
    else:
        # 使用 top-k 分组
        (
            h_proto_high, h_proto_low, mi_curve, threshold, high_patches, low_patches
        ) = extract_prototypes_from_sin(
            model, source_loader, device, args.seq_len, args.pred_len,
            layer_idx=args.extract_layer, use_q3=False,
            k_high=args.top_k, k_low=args.k_low
        )
        q3 = threshold  # 保持变量名兼容

    print(f"  High-MI 原型形状: {h_proto_high.shape}")
    print(f"  Low-MI 原型形状: {h_proto_low.shape}")
    print(f"  MI curve: {mi_curve}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  High-MI patches ({len(high_patches)}): {high_patches.tolist()}")
    print(f"  Low-MI patches ({len(low_patches)}): {low_patches.tolist()}")

    # 保存原型
    high_proto_path = os.path.join(args.output_dir, "sin_prototype_high.pt")
    low_proto_path = os.path.join(args.output_dir, "sin_prototype_low.pt")
    torch.save({
        'h_proto': h_proto_high,
        'source': 'sin',
        'layer': args.extract_layer,
        'mi': 'high',
        'mi_curve': mi_curve,
        'q3': q3,
        'high_patches': high_patches,
        'low_patches': low_patches,
    }, high_proto_path)
    torch.save({
        'h_proto': h_proto_low,
        'source': 'sin',
        'layer': args.extract_layer,
        'mi': 'low',
        'mi_curve': mi_curve,
        'q3': q3,
        'high_patches': high_patches,
        'low_patches': low_patches,
    }, low_proto_path)
    print(f"  High-MI 原型已保存: {high_proto_path}")
    print(f"  Low-MI 原型已保存: {low_proto_path}")

    # ── 4. 加载带原型的模型 ────────────────────────────────────────
    print("\n[4] 加载带原型的模型...")

    # High-MI 模型
    config_high = Namespace(**vars(config))
    config_high.use_prototype = True
    config_high.prototype_path = high_proto_path
    model_high = Model(config_high).to(device)
    model_high.eval()
    print("  High-MI 原型注入模型加载完成")

    # Low-MI 模型
    config_low = Namespace(**vars(config))
    config_low.use_prototype = True
    config_low.prototype_path = low_proto_path
    model_low = Model(config_low).to(device)
    model_low.eval()
    print("  Low-MI 原型注入模型加载完成")

    # ── 5. Baseline 预测 ─────────────────────────────────────────
    print("\n[5] Baseline 预测 (无原型)...")
    baseline_preds, baseline_trues, times_list, a_list, b_list = run_prediction(model, target_loader, args, device)
    baseline_results = metric(
        np.array([np.concatenate(baseline_trues)]),
        np.array([np.concatenate(baseline_preds)])
    )
    print(f"  Baseline MSE: {baseline_results['MSE']:.6f}, MAE: {baseline_results['MAE']:.6f}")

    # ── 6. High-MI 原型引导预测 ───────────────────────────────────
    print("\n[6] High-MI 原型引导预测...")
    high_proto_preds, _, _, _, _ = run_prediction(model_high, target_loader, args, device)
    high_proto_results = metric(
        np.array([np.concatenate(baseline_trues)]),
        np.array([np.concatenate(high_proto_preds)])
    )
    print(f"  High-MI Proto MSE: {high_proto_results['MSE']:.6f}, MAE: {high_proto_results['MAE']:.6f}")

    # ── 7. Low-MI 原型引导预测 ────��───────────────────────────────
    print("\n[7] Low-MI 原型引导预测...")
    low_proto_preds, _, _, _, _ = run_prediction(model_low, target_loader, args, device)
    low_proto_results = metric(
        np.array([np.concatenate(baseline_trues)]),
        np.array([np.concatenate(low_proto_preds)])
    )
    print(f"  Low-MI Proto MSE: {low_proto_results['MSE']:.6f}, MAE: {low_proto_results['MAE']:.6f}")

    # ── 8. 绘制对比图 ─────────────────────────────────────────────
    print("\n[8] 绘制对比图...")

    all_results = {
        "Baseline": baseline_results,
        "High-MI Proto": high_proto_results,
        "Low-MI Proto": low_proto_results
    }

    # 图1: 三组预测曲线对比
    plot_3way_prediction_comparison(
        times_list, baseline_preds, high_proto_preds, low_proto_preds, baseline_trues,
        a_list, b_list,
        save_path=os.path.join(args.output_dir, "prediction_comparison_3way.png"),
        n_show=12
    )

    # 图2: 散点和分布对比
    plot_3way_scatter_comparison(
        baseline_preds, high_proto_preds, low_proto_preds, baseline_trues,
        save_path=os.path.join(args.output_dir, "metric_comparison_3way.png")
    )

    # 图3: 聚合统计
    plot_3way_aggregated_comparison(
        all_results,
        save_path=os.path.join(args.output_dir, "aggregate_comparison_3way.png")
    )

    # ── 9. 保存结果 ───────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "results_3way.txt")
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("合成数据原型注入实验: 源函数 → 常量序列 泛化\n")
        f.write("Baseline / High-MI Proto / Low-MI Proto 三组对比\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"源函数样本数: {args.sin_samples}\n")
        f.write(f"测试样本数: {args.test_samples}\n")
        f.write(f"序列长度: {args.seq_len}\n")
        f.write(f"预测长度: {args.pred_len}\n")
        f.write(f"原型提取层: Layer {args.extract_layer}\n")
        f.write(f"Q3 阈值: {q3:.6f}\n")
        f.write(f"高 MI patches ({len(high_patches)}): {high_patches.tolist()}\n")
        f.write(f"低 MI patches ({len(low_patches)}): {low_patches.tolist()}\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<20} {'MSE':>12} {'MAE':>12} {'vs Baseline':>15}\n")
        f.write("-" * 70 + "\n")

        methods = [
            ("Baseline", baseline_results),
            ("High-MI Proto", high_proto_results),
            ("Low-MI Proto", low_proto_results)
        ]

        for name, res in methods:
            delta = res['MSE'] - baseline_results['MSE']
            pct = delta / baseline_results['MSE'] * 100
            arrow = "↓" if delta < 0 else "↑"
            f.write(f"{name:<20} {res['MSE']:>12.6f} {res['MAE']:>12.6f} {arrow}{abs(pct):>6.2f}%\n")

        f.write("-" * 70 + "\n\n")

        # 分析结论
        f.write("分析结论:\n")
        best = min(methods, key=lambda x: x[1]['MSE'])
        f.write(f"  最佳方法: {best[0]} (MSE = {best[1]['MSE']:.6f})\n\n")

        high_delta = high_proto_results['MSE'] - baseline_results['MSE']
        low_delta = low_proto_results['MSE'] - baseline_results['MSE']

        f.write(f"  High-MI Proto vs Baseline: {high_delta:+.6f} ({high_delta/baseline_results['MSE']*100:+.2f}%)\n")
        f.write(f"  Low-MI Proto vs Baseline:  {low_delta:+.6f} ({low_delta/baseline_results['MSE']*100:+.2f}%)\n\n")

        if high_delta < 0 and low_delta < 0:
            f.write("  两种原型都有助于预测\n")
        elif high_delta > 0 and low_delta > 0:
            f.write("  两种原型都无法改善预测 (源函数与目标函数泛化差距过大)\n")
        elif high_delta > low_delta:
            f.write("  Low-MI 原型效果更好，可能更具泛化性\n")
        else:
            f.write("  High-MI 原型效果更好\n")

    # ── 10. 保存 MI 曲线图 ─────────────────────────────────────────
    mi_plot_path = os.path.join(args.output_dir, "mi_curve_q3.png")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    patches_idx = np.arange(len(mi_curve))
    ax.plot(patches_idx, mi_curve, "o-", ms=4, lw=1.5, label="HSIC MI Score")
    ax.axhline(q3, color="red", ls="--", lw=2, label=f"Q3 = {q3:.6f}")
    
    # 高 MI patch 用红色标记
    if len(high_patches) > 0:
        ax.scatter(high_patches, mi_curve[high_patches], 
                  color="red", s=80, zorder=5, marker="^", label=f"High-MI (>{q3:.4f})")
    # 低 MI patch 用蓝色标记
    if len(low_patches) > 0:
        ax.scatter(low_patches, mi_curve[low_patches], 
                  color="blue", s=50, zorder=4, marker="v", label=f"Low-MI (≤{q3:.4f})")
    
    ax.set_xlabel("Patch Index", fontsize=11)
    ax.set_ylabel("HSIC MI Score", fontsize=11)
    ax.set_title(f"MI Curve with Q3 Threshold (Layer {args.extract_layer})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(mi_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] MI curve saved: {mi_plot_path}")

    print(f"\n结果已保存: {results_path}")
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
