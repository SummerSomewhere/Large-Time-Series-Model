#!/usr/bin/env python3
"""
频域分析脚本：对比高 MI Patch vs 低 MI Patch 的频率特性差异

功能：
1. Patch 筛选 - 根据 MI 值分组为 Top 5% 和 Bottom 95%
2. PSD 计算 - 归一化后的功率谱密度对比
3. 差分分析 - 绘制差异谱并标注峰值
4. 时域可视化 - 随机抽取样本对比

使用方法:
    python scripts/analysis/psd_analysis.py --mi_scores path/to/mi.npy --data path/to/data.csv

或者在代码中直接调用:
    from psd_analysis import analyze_mi_patches
    analyze_mi_patches(mi_scores, data, patch_size, stride)
"""

import argparse
import os
import numpy as np
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq


def normalize_psd(psd: np.ndarray) -> np.ndarray:
    """
    对 PSD 进行归一化，消除量级差异

    归一化方法: PSD / sum(PSD)，使所有 PSD 面积为 1（概率密度形式）
    这样可以只比较频率成分分布，而不受总能量影响
    """
    psd_sum = np.sum(psd)
    if psd_sum > 0:
        return psd / psd_sum
    return psd


def compute_psd_welch(
    patch_data: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Welch 方法计算功率谱密度

    参数:
        patch_data: 单个 Patch 的时域数据，shape [L] 或 [L, C]
        fs: 采样频率（默认 1.0 表示归一化频率）
        nperseg: 每个段的样本数（默认取数据长度的 1/4）
        noverlap: 段之间重叠的样本数（默认取 nperseg 的 1/2）

    返回:
        freqs: 频率数组（单位由 fs 决定）
        psd: 功率谱密度数组
    """
    # 处理多通道：取平均
    if patch_data.ndim > 1:
        patch_data = np.mean(patch_data, axis=1)

    L = len(patch_data)
    if nperseg is None:
        nperseg = max(8, L // 4)
    if noverlap is None:
        noverlap = nperseg // 2

    # 使用 Hanning 窗减少频谱泄露
    freqs, psd = signal.welch(
        patch_data,
        fs=fs,
        window='hann',  # Hanning 窗
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant'  # 去除直流分量
    )

    return freqs, psd


def compute_psd_fft(
    patch_data: np.ndarray,
    fs: float = 1.0,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 FFT 直接计算功率谱密度（适合短数据）

    参数:
        patch_data: 单个 Patch 的时域数据
        fs: 采样频率
        window: 窗函数类型

    返回:
        freqs: 正频率轴（0 到 fs/2）
        psd: 归一化后的功率谱密度
    """
    # 处理多通道
    if patch_data.ndim > 1:
        patch_data = np.mean(patch_data, axis=1)

    L = len(patch_data)

    # 应用窗函数
    if window == 'hann':
        window_vals = np.hanning(L)
    elif window == 'hamming':
        window_vals = np.hamming(L)
    elif window == 'blackman':
        window_vals = np.blackman(L)
    else:
        window_vals = np.ones(L)

    # 加窗
    windowed = patch_data * window_vals

    # FFT
    fft_vals = rfft(windowed)

    # 功率谱（取模平方）
    psd = np.abs(fft_vals) ** 2

    # 归一化（消除窗函数能量损失）
    window_energy = np.sum(window_vals ** 2)
    psd = psd / window_energy * 2 / L  # 单边谱归一化

    # 频率轴: 从 0 到 fs/2，共 L/2+1 个点
    freqs = rfftfreq(L, d=1.0/fs)

    return freqs, psd


def select_patches_by_mi(
    mi_scores: np.ndarray,
    top_percent: float = 5.0,
    bottom_percent: float = 95.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 MI 值筛选 Patch

    参数:
        mi_scores: MI 值数组，shape [N] 或 [N, M]（M 是 variates）
        top_percent: 高 MI 组的百分比（默认 5%）
        bottom_percent: 保留底部百分比（默认 95%，即排除 top 5%）

    返回:
        top_indices: 高 MI Patch 的索引
        bottom_indices: 低 MI Patch 的索引
    """
    if mi_scores.ndim > 1:
        # 多 variates 情况：取平均或最大值
        mi_mean = mi_scores.mean(axis=1) if mi_scores.shape[1] > 1 else mi_scores.flatten()
    else:
        mi_mean = mi_scores.flatten()

    N = len(mi_mean)
    n_top = max(1, int(N * top_percent / 100))

    # 排序找索引
    sorted_indices = np.argsort(mi_mean)[::-1]  # 降序
    top_indices = sorted_indices[:n_top]
    bottom_indices = sorted_indices[n_top:]

    return top_indices, bottom_indices


def patch_indices_to_time_indices(
    patch_indices: np.ndarray,
    patch_size: int,
    stride: int
) -> List[slice]:
    """
    将 Patch 索引转换为原始数据的时间索引（slice 对象列表）

    例如：patch_size=24, stride=12, patch_idx=3
    -> 对应原始数据索引: 3*12 到 3*12+24 = [36:60]
    """
    time_slices = []
    for p_idx in patch_indices:
        start = p_idx * stride
        end = start + patch_size
        time_slices.append(slice(start, end))
    return time_slices


def compute_mean_psd(
    data: np.ndarray,
    time_slices: List[slice],
    fs: float = 1.0,
    use_welch: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算多个 Patch 的平均 PSD

    参数:
        data: 原始时序数据，shape [L, C]
        time_slices: Patch 对应的时间切片列表
        fs: 采样频率
        use_welch: 是否使用 Welch 方法（True 更稳定）
        normalize: 是否归一化 PSD

    返回:
        mean_psd: 平均 PSD（归一化后）
        freqs: 频率轴
    """
    psd_list = []

    for sl in time_slices:
        patch_data = data[sl]  # [patch_size, C] 或 [patch_size]

        if use_welch:
            freqs, psd = compute_psd_welch(patch_data, fs=fs)
        else:
            freqs, psd = compute_psd_fft(patch_data, fs=fs)

        if normalize:
            psd = normalize_psd(psd)

        psd_list.append(psd)

    # 计算平均 PSD
    mean_psd = np.mean(psd_list, axis=0)

    return mean_psd, freqs


def detect_spectral_peaks(
    delta_psd: np.ndarray,
    freqs: np.ndarray,
    threshold_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    检测差分谱中的显著峰值

    参数:
        delta_psd: 差分 PSD
        freqs: 频率轴
        threshold_ratio: 相对最大值的阈值比例

    返回:
        peak_freqs: 峰值频率
        peak_values: 峰值处的差分值
    """
    threshold = np.max(delta_psd) * threshold_ratio
    peaks_mask = (delta_psd > threshold) & (delta_psd > 0)

    # 找局部极大值
    peaks_mask &= (np.diff(delta_psd, append=0) > 0) & (np.diff(delta_psd, prepend=0) < 0)

    peak_freqs = freqs[peaks_mask]
    peak_values = delta_psd[peaks_mask]

    return peak_freqs, peak_values


def plot_psd_comparison(
    freqs: np.ndarray,
    mean_psd_high: np.ndarray,
    mean_psd_low: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    子图 1: 同时绘制高 MI 和低 MI Patch 的平均 PSD 曲线

    频率轴说明:
    - x 轴为归一化频率（0 到 0.5，对应 0 到 Nyquist 频率）
    - 或根据 fs 显示实际频率值（如 0.1 Hz）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 PSD 曲线
    ax.plot(freqs, mean_psd_high, 'r-', linewidth=2, label='Top 5% 高 MI Patch', alpha=0.8)
    ax.plot(freqs, mean_psd_low, 'b-', linewidth=2, label='Bottom 95% 低 MI Patch', alpha=0.8)

    # 填充区域增强对比
    ax.fill_between(freqs, mean_psd_high, alpha=0.3, color='red')
    ax.fill_between(freqs, mean_psd_low, alpha=0.3, color='blue')

    ax.set_xlabel('归一化频率 (×π rad/sample)', fontsize=11)
    ax.set_ylabel('归一化功率谱密度', fontsize=11)
    ax.set_title('图 1: 高 MI vs 低 MI Patch 平均 PSD 对比\n(频率集中点 = 规律信号，白噪声 = 平坦谱)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, freqs[-1]])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_psd_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path.replace('.png', '_psd_comparison.png')}")

    return fig


def plot_delta_spectrum(
    freqs: np.ndarray,
    delta_psd: np.ndarray,
    peak_freqs: np.ndarray,
    peak_values: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    子图 2: 绘制差分谱并标注峰值

    正值区域（红色）= 高 MI patch 能量更多的频率
    负值区域（蓝色）= 低 MI patch 能量更多的频率
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制差分谱
    ax.plot(freqs, delta_psd, 'k-', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # 填充正负区域
    ax.fill_between(freqs, delta_psd, 0, where=(delta_psd > 0), color='red', alpha=0.5, label='高 MI 更多')
    ax.fill_between(freqs, delta_psd, 0, where=(delta_psd < 0), color='blue', alpha=0.5, label='低 MI 更多')

    # 标注峰值
    if len(peak_freqs) > 0:
        ax.scatter(peak_freqs, peak_values, c='red', s=80, zorder=5, marker='^')
        for pf, pv in zip(peak_freqs, peak_values):
            ax.annotate(
                f'{pf:.3f}',
                xy=(pf, pv),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color='red'
            )

    ax.set_xlabel('归一化频率 (×π rad/sample)', fontsize=11)
    ax.set_ylabel('ΔPSD (高 MI - 低 MI)', fontsize=11)
    ax.set_title('图 2: 差分功率谱\n(红色峰值 = 高 MI patch 的主导频率)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, freqs[-1]])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_delta_spectrum.png'), dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path.replace('.png', '_delta_spectrum.png')}")

    return fig


def plot_time_domain_comparison(
    data: np.ndarray,
    time_slice_high: slice,
    time_slice_low: slice,
    fs: float = 1.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    子图 3: 随机抽取高/低 MI Patch 的时域对比
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 时间轴
    t_high = np.arange(data[time_slice_high].shape[0]) / fs
    t_low = np.arange(data[time_slice_low].shape[0]) / fs

    # 高 MI Patch
    patch_high = data[time_slice_high]
    if patch_high.ndim > 1:
        patch_high = patch_high.mean(axis=1)
    axes[0].plot(t_high, patch_high, 'r-', linewidth=1.5, label='Top 5% 高 MI Patch')
    axes[0].set_ylabel('幅值', fontsize=11)
    axes[0].set_title('图 3: 时域信号对比', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 低 MI Patch
    patch_low = data[time_slice_low]
    if patch_low.ndim > 1:
        patch_low = patch_low.mean(axis=1)
    axes[1].plot(t_low, patch_low, 'b-', linewidth=1.5, label='Bottom 95% 低 MI Patch')
    axes[1].set_xlabel('时间 (samples)', fontsize=11)
    axes[1].set_ylabel('幅值', fontsize=11)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_time_domain.png'), dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path.replace('.png', '_time_domain.png')}")

    return fig


def analyze_mi_patches(
    mi_scores: np.ndarray,
    data: np.ndarray,
    patch_size: int = 24,
    stride: int = 12,
    fs: float = 1.0,
    top_percent: float = 5.0,
    output_dir: str = "experiments/output/psd_analysis",
    random_seed: int = 42
) -> dict:
    """
    主分析函数：对比高 MI Patch vs 低 MI Patch 的频域特性

    参数:
        mi_scores: MI 值数组，shape [N] 或 [N, M]
        data: 原始时序数据，shape [L, C]
        patch_size: Patch 长度
        stride: Patch 滑动步长
        fs: 采样频率
        top_percent: 高 MI 组的百分比
        output_dir: 输出目录

    返回:
        results: 包含分析结果的字典
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("频域分析: 高 MI Patch vs 低 MI Patch")
    print(f"{'='*60}")
    print(f"数据形状: {data.shape}, Patch数: {len(mi_scores)}")
    print(f"Patch大小: {patch_size}, 步长: {stride}")
    print(f"高 MI 组: Top {top_percent}%, 低 MI 组: Bottom {100-top_percent}%")

    # 1. Patch 筛选
    top_indices, bottom_indices = select_patches_by_mi(mi_scores, top_percent=top_percent)
    print(f"\n高 MI Patch 数: {len(top_indices)}")
    print(f"低 MI Patch 数: {len(bottom_indices)}")

    # 转换为时间切片
    top_slices = patch_indices_to_time_indices(top_indices, patch_size, stride)
    bottom_slices = patch_indices_to_time_indices(bottom_indices, patch_size, stride)

    # 2. PSD 计算
    print("\n计算 PSD...")
    mean_psd_high, freqs = compute_mean_psd(data, top_slices, fs=fs, normalize=True)
    mean_psd_low, _ = compute_mean_psd(data, bottom_slices, fs=fs, normalize=True)

    # 3. 差分分析
    delta_psd = mean_psd_high - mean_psd_low
    peak_freqs, peak_values = detect_spectral_peaks(delta_psd, freqs, threshold_ratio=0.3)

    print(f"\n差分谱峰值检测结果:")
    print(f"  峰值数量: {len(peak_freqs)}")
    for pf, pv in zip(peak_freqs[:5], peak_values[:5]):
        print(f"    频率 {pf:.4f}: ΔPSD = {pv:.6f}")

    # 4. 随机选择用于时域对比的 Patch
    sample_high_idx = np.random.choice(len(top_slices))
    sample_low_idx = np.random.choice(len(bottom_slices))

    # 5. 可视化
    print("\n生成可视化...")
    save_path = os.path.join(output_dir, "psd_analysis.png")

    fig = plt.figure(figsize=(16, 12))

    # 子图 1: PSD 对比
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(freqs, mean_psd_high, 'r-', linewidth=2, label='Top 5% 高 MI', alpha=0.8)
    ax1.plot(freqs, mean_psd_low, 'b-', linewidth=2, label='Bottom 95% 低 MI', alpha=0.8)
    ax1.fill_between(freqs, mean_psd_high, alpha=0.3, color='red')
    ax1.fill_between(freqs, mean_psd_low, alpha=0.3, color='blue')
    ax1.set_xlabel('归一化频率')
    ax1.set_ylabel('归一化 PSD')
    ax1.set_title('图1: 平均 PSD 对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图 2: 差分谱
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(freqs, delta_psd, 'k-', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.fill_between(freqs, delta_psd, 0, where=(delta_psd > 0), color='red', alpha=0.5)
    ax2.fill_between(freqs, delta_psd, 0, where=(delta_psd < 0), color='blue', alpha=0.5)
    if len(peak_freqs) > 0:
        ax2.scatter(peak_freqs[:5], peak_values[:5], c='red', s=80, zorder=5, marker='^')
    ax2.set_xlabel('归一化频率')
    ax2.set_ylabel('ΔPSD')
    ax2.set_title('图2: 差分功率谱')
    ax2.grid(True, alpha=0.3)

    # 子图 3: 高 MI 时域
    ax3 = fig.add_subplot(2, 2, 3)
    patch_high = data[top_slices[sample_high_idx]]
    if patch_high.ndim > 1:
        patch_high = patch_high.mean(axis=1)
    ax3.plot(patch_high, 'r-', linewidth=1.2)
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('幅值')
    ax3.set_title(f'图3: 高 MI Patch 时域 (idx={top_indices[sample_high_idx]})')
    ax3.grid(True, alpha=0.3)

    # 子图 4: 低 MI 时域
    ax4 = fig.add_subplot(2, 2, 4)
    patch_low = data[bottom_slices[sample_low_idx]]
    if patch_low.ndim > 1:
        patch_low = patch_low.mean(axis=1)
    ax4.plot(patch_low, 'b-', linewidth=1.2)
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('幅值')
    ax4.set_title(f'图4: 低 MI Patch 时域 (idx={bottom_indices[sample_low_idx]})')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n综合图片已保存: {save_path}")

    # 汇总统计
    results = {
        'top_indices': top_indices,
        'bottom_indices': bottom_indices,
        'freqs': freqs,
        'mean_psd_high': mean_psd_high,
        'mean_psd_low': mean_psd_low,
        'delta_psd': delta_psd,
        'peak_freqs': peak_freqs,
        'peak_values': peak_values,
        'spectral_entropy_high': -np.sum(mean_psd_high * np.log(mean_psd_high + 1e-10)),
        'spectral_entropy_low': -np.sum(mean_psd_low * np.log(mean_psd_low + 1e-10)),
    }

    # 打印汇总
    print(f"\n{'='*60}")
    print("分析汇总")
    print(f"{'='*60}")
    print(f"频谱熵 (高 MI): {results['spectral_entropy_high']:.4f}")
    print(f"频谱熵 (低 MI): {results['spectral_entropy_low']:.4f}")
    print(f"结论: {'高MI patch更规律' if results['spectral_entropy_high'] < results['spectral_entropy_low'] else '低MI patch更规律'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="高 MI Patch 频域分析")
    parser.add_argument("--mi_scores", type=str, required=True, help="MI 值 .npy 文件路径")
    parser.add_argument("--data", type=str, required=True, help="原始数据 .npy 文件路径")
    parser.add_argument("--patch_size", type=int, default=24, help="Patch 长度")
    parser.add_argument("--stride", type=int, default=12, help="滑动步长")
    parser.add_argument("--fs", type=float, default=1.0, help="采样频率")
    parser.add_argument("--top_percent", type=float, default=5.0, help="高 MI 组百分比")
    parser.add_argument("--output_dir", type=str, default="experiments/output/psd_analysis", help="输出目录")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 加载数据
    print(f"加载 MI 值: {args.mi_scores}")
    mi_scores = np.load(args.mi_scores)
    print(f"加载数据: {args.data}")
    data = np.load(args.data)

    # 分析
    results = analyze_mi_patches(
        mi_scores=mi_scores,
        data=data,
        patch_size=args.patch_size,
        stride=args.stride,
        fs=args.fs,
        top_percent=args.top_percent,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
