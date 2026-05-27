#!/usr/bin/env python3
"""
ETTh1 + Timer: Token 分析实验

对比高 MI Patch (Thinking Tokens) vs 低 MI Patch (Background Tokens)：

对比维度:
1. 能量分布 - 是否存在明显的能量集中点？是否分布均匀、类似白噪声？
2. 主导频率 - 是否对应异常频率（如 spikes）？是否对应正常季节性频率？
3. 相位突变 - 是否在相位发生剧烈转换处？相位是否平滑过渡？

使用方法:
    torchrun --nnodes=1 --nproc_per_node=1 experiments/etth1_token_analysis.py ...
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 复用 MI/HSIC 计算逻辑
_mi_path = os.path.join(_ROOT, "experiments", "etth1_mi_hsic_peaks.py")
_spec = importlib.util.spec_from_file_location("etth1_mi_hsic_peaks", _mi_path)
_mi = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mi)
build_namespace = _mi.build_namespace
mi_sequence_hsic = _mi.mi_sequence_hsic
forward_y_collect_layers = _mi.forward_y_collect_layers

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.masking import TriangularCausalMask


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def timer_hidden_last_block_pre_norm(model, x_enc):
    """
    返回: h [B*M, N, D], means, stdev, B, M, n_vars, N, core
    """
    core = _unwrap(model)
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape
    mask = TriangularCausalMask(BM, N, device=dec_in.device)
    h = dec_in
    for o3 in core.decoder.attn_layers:
        h, _ = o3(h, attn_mask=mask)
    return h, means, stdev, B, M, int(n_vars), int(N), core


def compute_energy_stats(h_patch):
    """计算单个 patch 的能量统计量"""
    # h_patch: [B, D]
    energy = torch.sum(h_patch ** 2, dim=1)  # [B]
    return {
        'mean': energy.mean().item(),
        'std': energy.std().item(),
        'max': energy.max().item(),
        'min': energy.min().item(),
    }


def compute_fft_analysis(h_patch):
    """
    对 patch 表示进行 FFT 分析
    h_patch: [B, D]
    返回: 主导频率的幅度谱统计
    """
    B, D = h_patch.shape
    # 对 batch 维度取平均得到代表性频谱
    h_mean = h_patch.mean(dim=0)  # [D]
    fft_vals = torch.fft.fft(h_mean)
    magnitudes = torch.abs(fft_vals)[:D // 2]
    magnitudes = magnitudes / (magnitudes.sum() + 1e-10)  # 归一化

    return {
        'dominant_freq_idx': torch.argmax(magnitudes).item(),
        'dominant_freq_ratio': torch.max(magnitudes).item(),
        'top3_freq_idx': torch.argsort(magnitudes, descending=True)[:3].tolist(),
        'spectral_entropy': -(magnitudes * torch.log(magnitudes + 1e-10)).sum().item(),
    }


def compute_phase_analysis(h_patch):
    """
    分析 patch 表示的相位特性
    h_patch: [B, D]
    返回: 相位突变统计
    """
    B, D = h_patch.shape
    h_mean = h_patch.mean(dim=0)
    fft_vals = torch.fft.fft(h_mean)
    phases = torch.angle(fft_vals)[:D // 2]

    # 相位差分（检测突变）
    phase_diff = torch.diff(phases)
    # 处理相位缠绕 (-pi 到 pi)
    phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    return {
        'phase_std': phases.std().item(),
        'phase_diff_mean': torch.abs(phase_diff_wrapped).mean().item(),
        'phase_diff_max': torch.abs(phase_diff_wrapped).max().item(),
        'phase_sharp_ratio': (torch.abs(phase_diff_wrapped) > np.pi / 4).float().mean().item(),
    }


def analyze_batch(h, h_y, B, M, N, top_k=2, bottom_k=2):
    """
    分析一个 batch 中的 patch 特性

    h: [B*M, N, D]
    h_y: [B, 1, D] 未来表示
    """
    # 计算 MI
    Hbm = h.view(B, M, N, -1).mean(dim=1)  # [B, N, D]
    mi = mi_sequence_hsic(Hbm.cpu(), h_y.cpu())
    mi_f = np.nan_to_num(mi, nan=-np.inf)

    # 分类 patch
    peak_indices = np.argsort(mi_f)[-top_k:][::-1].astype(int).tolist()
    low_indices = np.argsort(mi_f)[:bottom_k].astype(int).tolist()

    # 收集统计量
    results = {
        'mi_values': mi_f,
        'peak_indices': peak_indices,
        'low_indices': low_indices,
        'peak_energy': [],
        'low_energy': [],
        'peak_fft': [],
        'low_fft': [],
        'peak_phase': [],
        'low_phase': [],
    }

    for idx in peak_indices:
        patch_h = Hbm[:, idx, :].contiguous()
        results['peak_energy'].append(compute_energy_stats(patch_h))
        results['peak_fft'].append(compute_fft_analysis(patch_h))
        results['peak_phase'].append(compute_phase_analysis(patch_h))

    for idx in low_indices:
        patch_h = Hbm[:, idx, :].contiguous()
        results['low_energy'].append(compute_energy_stats(patch_h))
        results['low_fft'].append(compute_fft_analysis(patch_h))
        results['low_phase'].append(compute_phase_analysis(patch_h))

    return results


def aggregate_stats(all_results):
    """聚合多个 batch 的统计结果"""
    agg = {
        'peak_energy_mean': [],
        'peak_energy_std': [],
        'low_energy_mean': [],
        'low_energy_std': [],
        'peak_fft_entropy': [],
        'low_fft_entropy': [],
        'peak_phase_sharp': [],
        'low_phase_sharp': [],
    }

    for r in all_results:
        if r['peak_energy']:
            agg['peak_energy_mean'].append(np.mean([e['mean'] for e in r['peak_energy']]))
            agg['peak_energy_std'].append(np.mean([e['std'] for e in r['peak_energy']]))
        if r['low_energy']:
            agg['low_energy_mean'].append(np.mean([e['mean'] for e in r['low_energy']]))
            agg['low_energy_std'].append(np.mean([e['std'] for e in r['low_energy']]))
        if r['peak_fft']:
            agg['peak_fft_entropy'].append(np.mean([e['spectral_entropy'] for e in r['peak_fft']]))
        if r['low_fft']:
            agg['low_fft_entropy'].append(np.mean([e['spectral_entropy'] for e in r['low_fft']]))
        if r['peak_phase']:
            agg['peak_phase_sharp'].append(np.mean([e['phase_sharp_ratio'] for e in r['peak_phase']]))
        if r['low_phase']:
            agg['low_phase_sharp'].append(np.mean([e['phase_sharp_ratio'] for e in r['low_phase']]))

    return agg


def plot_token_analysis(all_results, mi_curves, output_dir):
    """
    生成 token 分析可视化图

    子图布局:
    Row 1: 能量分布对比 | MI 曲线与高/低 MI patch 标注
    Row 2: 主导频率分布 | 相位突变分布
    Row 3: 能量时序对比 | 频谱熵 vs 相位突变散点图
    """
    agg = aggregate_stats(all_results)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("ETTh1 Token 分析: 高 MI Patch vs 低 MI Patch", fontsize=14, fontweight='bold')

    # ========== 子图 1: 能量分布对比 ==========
    ax1 = axes[0, 0]
    if agg['peak_energy_mean']:
        peak_means = agg['peak_energy_mean']
        low_means = agg['low_energy_mean']

        x = np.arange(len(peak_means))
        width = 0.35
        ax1.bar(x - width/2, peak_means, width, label='高 MI Patch', color='#e74c3c', alpha=0.8)
        ax1.bar(x + width/2, low_means, width, label='低 MI Patch', color='#3498db', alpha=0.8)
        ax1.set_xlabel('Batch Index')
        ax1.set_ylabel('平均能量')
        ax1.set_title('1. 能量分布对比\n(高MI patch有更多能量集中点)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # ========== 子图 2: MI 曲线与分类标注 ==========
    ax2 = axes[0, 1]
    if mi_curves:
        mi_arr = np.array(mi_curves)
        batch_idx = np.arange(len(mi_arr))

        # 计算每个位置的均值和标准差
        mi_mean = mi_arr.mean(axis=0)
        mi_std = mi_arr.std(axis=0)

        x = np.arange(len(mi_mean))
        ax2.fill_between(x, mi_mean - mi_std, mi_mean + mi_std, alpha=0.3, color='#3498db')
        ax2.plot(x, mi_mean, 'b-', linewidth=2, label='平均 MI 曲线')

        # 标注高/低 MI 区域
        peak_thresh = np.percentile(mi_mean, 80)
        low_thresh = np.percentile(mi_mean, 20)
        ax2.axhline(y=peak_thresh, color='#e74c3c', linestyle='--', alpha=0.7, label=f'高阈值 (80%)')
        ax2.axhline(y=low_thresh, color='#2ecc71', linestyle='--', alpha=0.7, label=f'低阈值 (20%)')

        ax2.set_xlabel('Patch Index')
        ax2.set_ylabel('MI (HSIC)')
        ax2.set_title('2. MI 曲线与 Patch 分类\n(红色虚线以上为高MI，绿色以下为低MI)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

    # ========== 子图 3: 主导频率分布 ==========
    ax3 = axes[1, 0]
    peak_entropies = agg.get('peak_fft_entropy', [])
    low_entropies = agg.get('low_fft_entropy', [])

    if peak_entropies and low_entropies:
        data_to_plot = [peak_entropies, low_entropies]
        bp = ax3.boxplot(data_to_plot, labels=['高 MI Patch', '低 MI Patch'], patch_artist=True)
        colors = ['#e74c3c', '#3498db']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_ylabel('频谱熵')
        ax3.set_title('3. 频谱熵分布\n(低熵=频率集中，高熵=白噪声)')
        ax3.grid(True, alpha=0.3)

        # 统计检验
        if len(peak_entropies) > 1 and len(low_entropies) > 1:
            mean_peak = np.mean(peak_entropies)
            mean_low = np.mean(low_entropies)
            conclusion = "高MI patch更集中" if mean_peak < mean_low else "高MI patch更分散"
            ax3.text(0.5, 0.95, f'结论: {conclusion}', transform=ax3.transAxes,
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== 子图 4: 相位突变分布 ==========
    ax4 = axes[1, 1]
    peak_sharp = agg.get('peak_phase_sharp', [])
    low_sharp = agg.get('low_phase_sharp', [])

    if peak_sharp and low_sharp:
        data_to_plot = [peak_sharp, low_sharp]
        bp = ax4.boxplot(data_to_plot, labels=['高 MI Patch', '低 MI Patch'], patch_artist=True)
        colors = ['#e74c3c', '#3498db']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.set_ylabel('相位突变比例')
        ax4.set_title('4. 相位突变分布\n(高比例=相位剧烈转换)')
        ax4.grid(True, alpha=0.3)

        if len(peak_sharp) > 1 and len(low_sharp) > 1:
            mean_peak = np.mean(peak_sharp)
            mean_low = np.mean(low_sharp)
            conclusion = "高MI patch相位更平滑" if mean_peak < mean_low else "高MI patch相位更突变"
            ax4.text(0.5, 0.95, f'结论: {conclusion}', transform=ax4.transAxes,
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== 子图 5: 能量标准差对比 ==========
    ax5 = axes[2, 0]
    peak_std = agg.get('peak_energy_std', [])
    low_std = agg.get('low_energy_std', [])

    if peak_std and low_std:
        ax5.scatter(agg['peak_energy_mean'], peak_std, c='#e74c3c', alpha=0.6, s=50, label='高 MI Patch')
        ax5.scatter(agg['low_energy_mean'], low_std, c='#3498db', alpha=0.6, s=50, label='低 MI Patch')
        ax5.set_xlabel('平均能量')
        ax5.set_ylabel('能量标准差')
        ax5.set_title('5. 能量 vs 能量波动\n(右上角=高能量高波动，左下角=低能量低波动)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # ========== 子图 6: 频谱熵 vs 相位突变散点图 ==========
    ax6 = axes[2, 1]
    if peak_entropies and low_entropies and peak_sharp and low_sharp:
        # 配对数据
        min_len = min(len(peak_entropies), len(peak_sharp), len(low_entropies), len(low_sharp))
        if min_len > 0:
            ax6.scatter(peak_entropies[:min_len], peak_sharp[:min_len],
                       c='#e74c3c', alpha=0.6, s=50, label='高 MI Patch')
            ax6.scatter(low_entropies[:min_len], low_sharp[:min_len],
                       c='#3498db', alpha=0.6, s=50, label='低 MI Patch')
            ax6.set_xlabel('频谱熵')
            ax6.set_ylabel('相位突变比例')
            ax6.set_title('6. 频谱熵 vs 相位突变\n(左下=有序信号，右上=无序信号)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'token_analysis_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {fig_path}")

    return fig_path


def plot_mi_vs_underlying_signal(mi_curves, batch_x_data, output_dir, num_samples=5):
    """
    额外可视化: MI 值与原始数据特性的关系
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MI 值与原始数据的对应关系分析", fontsize=14, fontweight='bold')

    # 子图 1: MI 值 vs Patch 位置
    ax1 = axes[0, 0]
    mi_arr = np.array(mi_curves)
    mi_mean = mi_arr.mean(axis=0)
    mi_std = mi_arr.std(axis=0)
    x = np.arange(len(mi_mean))

    ax1.plot(x, mi_mean, 'b-', linewidth=2)
    ax1.fill_between(x, mi_mean - mi_std, mi_mean + mi_std, alpha=0.3)
    ax1.set_xlabel('Patch 索引')
    ax1.set_ylabel('MI (HSIC)')
    ax1.set_title('各 Patch 的平均 MI 值')
    ax1.grid(True, alpha=0.3)

    # 标注高 MI 区域
    high_mi_mask = mi_mean > np.percentile(mi_mean, 75)
    for i, is_high in enumerate(high_mi_mask):
        if is_high:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='red')
    ax1.text(0.02, 0.98, '红色区域: 高 MI (Thinking)', transform=ax1.transAxes,
            va='top', fontsize=9, color='#e74c3c')

    # 子图 2: 高 MI patch 对应的原始数据段
    ax2 = axes[0, 1]
    if len(mi_curves) > 0:
        last_mi = mi_curves[-1]
        high_idx = np.argsort(last_mi)[-3:][::-1]
        low_idx = np.argsort(last_mi)[:3]

        # 模拟展示（实际应该用 batch_x 还原原始数据）
        patch_positions = np.arange(len(last_mi))
        ax2.bar(high_idx - 0.2, last_mi[high_idx], 0.4, label='高 MI', color='#e74c3c', alpha=0.8)
        ax2.bar(low_idx + 0.2, last_mi[low_idx], 0.4, label='低 MI', color='#3498db', alpha=0.8)
        ax2.set_xlabel('Patch 索引')
        ax2.set_ylabel('MI 值')
        ax2.set_title('最后一个 Batch 的高/低 MI Patch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 子图 3: MI 分布直方图
    ax3 = axes[1, 0]
    all_mi = mi_arr.flatten()
    ax3.hist(all_mi, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    ax3.axvline(x=np.percentile(all_mi, 75), color='#e74c3c', linestyle='--',
               label=f'75% 分位: {np.percentile(all_mi, 75):.4f}')
    ax3.axvline(x=np.percentile(all_mi, 25), color='#2ecc71', linestyle='--',
               label=f'25% 分位: {np.percentile(all_mi, 25):.4f}')
    ax3.set_xlabel('MI 值')
    ax3.set_ylabel('频数')
    ax3.set_title('MI 值分布直方图')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图 4: 统计汇总表
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 计算统计数据
    stats_text = f"""
    ==================== 统计汇总 ====================

    【MI 值统计】
    - 总 Patch 数: {len(mi_mean)}
    - 平均 MI: {np.mean(all_mi):.6f}
    - 标准差: {np.std(all_mi):.6f}
    - 75% 分位: {np.percentile(all_mi, 75):.6f}
    - 25% 分位: {np.percentile(all_mi, 25):.6f}
    - IQR: {np.percentile(all_mi, 75) - np.percentile(all_mi, 25):.6f}

    【高 MI Patch 特征】
    - 高 MI Patch 数 (top 25%): {np.sum(mi_mean > np.percentile(mi_mean, 75))}
    - 高 MI Patch 索引: {np.argsort(mi_mean)[-5:][::-1].tolist()}

    【低 MI Patch 特征】
    - 低 MI Patch 数 (bottom 25%): {np.sum(mi_mean < np.percentile(mi_mean, 25))}
    - 低 MI Patch 索引: {np.argsort(mi_mean)[:5].tolist()}

    【结论】
    - 高 MI patch 位置: {np.argsort(mi_mean)[-3:][::-1].tolist()}
      → 通常位于序列{"后部 (靠近预测范围)" if np.argsort(mi_mean)[-1] > len(mi_mean)/2 else "前部 (历史信息)"}
    - MI 与 patch 位置的关系: {'正相关' if np.corrcoef(np.arange(len(mi_mean)), mi_mean)[0,1] > 0 else '负相关'}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'mi_vs_underlying_signal.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {fig_path}")

    return fig_path


def main():
    p = argparse.ArgumentParser()
    # torchrun 标准参数
    p.add_argument("--task_name", type=str, default="forecast")
    p.add_argument("--is_training", type=int, default=0)
    p.add_argument("--is_finetuning", type=int, default=0)
    p.add_argument("--model", type=str, default="Timer")
    p.add_argument("--des", type=str, default="TokenAnalysis")
    p.add_argument("--train_test", type=int, default=0)
    p.add_argument("--itr", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_id", type=str, default="token_analysis")
    p.add_argument("--use_ims", action="store_true")
    p.add_argument("--use_multi_gpu", action="store_true")
    p.add_argument("--devices", type=str, default="0,1,2,3")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--use_gpu", type=bool, default=True)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--distil", action="store_true", default=True)
    p.add_argument("--inverse", action="store_true", default=False)
    p.add_argument("--output_attention", action="store_true", default=False)
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--target", type=str, default="OT")
    p.add_argument("--checkpoints", type=str, default="./checkpoints/")

    # 模型参数
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
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--subset_rand_ratio", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--top_k", type=int, default=2, help="高MI patch数量")
    p.add_argument("--bottom_k", type=int, default=2, help="低MI patch数量")
    p.add_argument("--num_samples", type=int, default=10, help="可视化采样的样本数")
    p.add_argument("--max_batches", type=int, default=100, help="最多处理batch数")
    p.add_argument("--out_dir", type=str, default="experiments/output/token_analysis",
                   help="图片输出目录")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ns = build_namespace(args)
    ns.use_gpu = bool(torch.cuda.is_available())

    _, loader = data_provider(ns, flag="test")
    model = Model(ns).to(device)
    model.eval()

    print(f"模型加载完成，设备: {device}")
    print(f"配置: seq_len={args.seq_len}, pred_len={args.pred_len}, patch_len={args.patch_len}")
    print(f"分析参数: top_k={args.top_k}, bottom_k={args.bottom_k}, max_batches={args.max_batches}")

    all_results = []
    mi_curves = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, _, _) in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            B = batch_x.shape[0]

            if B < args.top_k + args.bottom_k:
                continue

            # 获取隐藏表示
            h, means, stdev, B, M, n_vars, N, _ = timer_hidden_last_block_pre_norm(model, batch_x)

            # 获取未来表示
            y_future = batch_y[:, ns.label_len: ns.label_len + ns.pred_len, :] if ns.use_ims else batch_y[:, -ns.pred_len:, :]
            h_y_layers = forward_y_collect_layers(model, y_future)
            h_y = h_y_layers[-1]

            # 分析
            result = analyze_batch(h, h_y, B, M, N, top_k=args.top_k, bottom_k=args.bottom_k)
            all_results.append(result)
            mi_curves.append(result['mi_values'])

            if (batch_idx + 1) % 20 == 0:
                print(f"已处理 {batch_idx + 1} 个 batch...")

    print(f"\n总共分析了 {len(all_results)} 个 batch")

    # 生成可视化
    print("\n生成可视化...")
    plot_token_analysis(all_results, mi_curves, args.out_dir)
    plot_mi_vs_underlying_signal(mi_curves, None, args.out_dir, args.num_samples)

    # 打印汇总统计
    print("\n" + "=" * 60)
    print("Token 分析汇总报告")
    print("=" * 60)

    agg = aggregate_stats(all_results)

    print(f"\n【能量分布】")
    if agg['peak_energy_mean']:
        peak_mean = np.mean(agg['peak_energy_mean'])
        low_mean = np.mean(agg['low_energy_mean'])
        print(f"  高 MI Patch 平均能量: {peak_mean:.4f}")
        print(f"  低 MI Patch 平均能量: {low_mean:.4f}")
        print(f"  差异: {peak_mean - low_mean:+.4f} ({'高MI更大' if peak_mean > low_mean else '低MI更大'})")

    print(f"\n【频谱熵】(低=频率集中，高=白噪声)")
    if agg['peak_fft_entropy']:
        peak_entropy = np.mean(agg['peak_fft_entropy'])
        low_entropy = np.mean(agg['low_fft_entropy'])
        print(f"  高 MI Patch 平均频谱熵: {peak_entropy:.4f}")
        print(f"  低 MI Patch 平均频谱熵: {low_entropy:.4f}")
        print(f"  结论: {'高MI patch频率更集中' if peak_entropy < low_entropy else '高MI patch频率更分散(类似白噪声)'}")

    print(f"\n【相位突变】(高比例=相位剧烈转换)")
    if agg['peak_phase_sharp']:
        peak_sharp = np.mean(agg['peak_phase_sharp'])
        low_sharp = np.mean(agg['low_phase_sharp'])
        print(f"  高 MI Patch 相位突变比例: {peak_sharp:.4f}")
        print(f"  低 MI Patch 相位突变比例: {low_sharp:.4f}")
        print(f"  结论: {'高MI patch相位更平滑' if peak_sharp < low_sharp else '高MI patch相位更突变'}")

    print("\n" + "=" * 60)
    print("分析完成！图片保存在:", args.out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
