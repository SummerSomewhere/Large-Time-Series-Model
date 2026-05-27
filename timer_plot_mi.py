#!/usr/bin/env python3
"""
绘制 Timer 模型每个样本、每层、每个 patch 的 MI 图

生成图片：
- results/mi/timer/{data}_gtmodel={gt_model}_testmodel={model}_sample{{id}}_layer{{layer}}.png
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def plot_mi_per_sample_layer(mi_dict, save_dir, dataset):
    """
    为每个样本的每层绘制 MI 图

    Args:
        mi_dict: MI 结果字典
        save_dir: 图片保存目录
        dataset: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)

    num_samples = len(mi_dict)
    if num_samples == 0:
        print("No MI data found!")
        return

    # 获取层列表
    first_sample = mi_dict[0]
    layers = sorted(first_sample['reps'].keys())
    num_layers = len(layers)

    print(f"Found {num_samples} samples, {num_layers} layers")

    for sample_id in range(num_samples):
        sample_data = mi_dict[sample_id]
        total_patches = sample_data.get('total_tokens', -1)

        for layer_idx, layer in enumerate(layers):
            mi_values = sample_data['reps'][layer]

            # 转为 numpy
            if isinstance(mi_values, torch.Tensor):
                mi_values = mi_values.cpu().numpy()
            else:
                mi_values = np.array(mi_values)

            # 创建图片
            fig, ax = plt.subplots(figsize=(12, 4))

            patch_indices = np.arange(len(mi_values))

            # 绘制 MI 曲线
            ax.plot(patch_indices, mi_values, 'b-', linewidth=1.5, label='MI (HSIC)')
            ax.fill_between(patch_indices, mi_values, alpha=0.3)

            # 标注峰值
            peak_idx = np.argmax(mi_values)
            peak_val = mi_values[peak_idx]
            ax.scatter([peak_idx], [peak_val], color='red', s=100, zorder=5)
            ax.annotate(f'Peak: {peak_val:.4f}\n(Patch {peak_idx})',
                       xy=(peak_idx, peak_val),
                       xytext=(peak_idx + 2, peak_val),
                       fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='red'))

            ax.set_xlabel('Patch Index', fontsize=11)
            ax.set_ylabel('MI (HSIC)', fontsize=11)
            ax.set_title(f'{dataset} - Sample {sample_id}, Layer {layer}', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            # 保存图片
            save_path = os.path.join(save_dir, f'{dataset}_sample{sample_id}_layer{layer}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved: {save_path}")

        print(f"Sample {sample_id} completed ({total_patches} patches)")


def plot_mi_heatmap(mi_dict, save_dir, dataset):
    """
    绘制所有样本所有层的 MI 热力图

    Args:
        mi_dict: MI 结果字典
        save_dir: 图片保存目录
        dataset: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)

    num_samples = len(mi_dict)
    if num_samples == 0:
        return

    first_sample = mi_dict[0]
    layers = sorted(first_sample['reps'].keys())
    num_layers = len(layers)

    # 获取最大 patch 数
    max_patches = max(sample_data['reps'][layers[0]].shape[0] 
                      if isinstance(sample_data['reps'][layers[0]], torch.Tensor) 
                      else len(sample_data['reps'][layers[0]]) 
                      for sample_data in mi_dict.values())

    # 构建 MI 矩阵 [num_layers, num_samples * max_patches]
    mi_matrix = np.zeros((num_layers, num_samples * max_patches))

    for layer_idx, layer in enumerate(layers):
        for sample_id in range(num_samples):
            mi_values = mi_dict[sample_id]['reps'][layer]
            if isinstance(mi_values, torch.Tensor):
                mi_values = mi_values.cpu().numpy()
            else:
                mi_values = np.array(mi_values)

            start_idx = sample_id * max_patches
            end_idx = start_idx + len(mi_values)
            mi_matrix[layer_idx, start_idx:end_idx] = mi_values

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(mi_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Sample * Patch Index', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title(f'{dataset} - MI Heatmap (All Samples, All Layers)', fontsize=12)

    # 设置 y 轴标签
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'Layer {l}' for l in layers])

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MI (HSIC)', fontsize=10)

    # 添加样本分隔线
    for i in range(1, num_samples):
        ax.axvline(x=i * max_patches - 0.5, color='white', linewidth=1, linestyle='--')

    save_path = os.path.join(save_dir, f'{dataset}_mi_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved heatmap: {save_path}")


def plot_mi_summary(mi_dict, save_dir, dataset):
    """
    绘制汇总统计图

    Args:
        mi_dict: MI 结果字典
        save_dir: 图片保存目录
        dataset: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)

    num_samples = len(mi_dict)
    if num_samples == 0:
        return

    first_sample = mi_dict[0]
    layers = sorted(first_sample['reps'].keys())
    num_layers = len(layers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 每层平均 MI
    ax1 = axes[0, 0]
    layer_means = []
    layer_stds = []
    for layer in layers:
        all_mi = []
        for sid in range(num_samples):
            mi_vals = mi_dict[sid]['reps'][layer]
            if isinstance(mi_vals, torch.Tensor):
                mi_vals = mi_vals.cpu().numpy()
            all_mi.extend(mi_vals)
        layer_means.append(np.mean(all_mi))
        layer_stds.append(np.std(all_mi))

    ax1.bar(range(num_layers), layer_means, yerr=layer_stds, capsize=3, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average MI (HSIC)')
    ax1.set_title('Average MI per Layer')
    ax1.set_xticks(range(num_layers))
    ax1.set_xticklabels([f'L{l}' for l in layers])
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 每层峰值 MI
    ax2 = axes[0, 1]
    layer_peaks = []
    for layer in layers:
        all_mi = []
        for sid in range(num_samples):
            mi_vals = mi_dict[sid]['reps'][layer]
            if isinstance(mi_vals, torch.Tensor):
                mi_vals = mi_vals.cpu().numpy()
            all_mi.extend(mi_vals)
        layer_peaks.append(np.max(all_mi))

    ax2.bar(range(num_layers), layer_peaks, color='coral', alpha=0.7)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Max MI (HSIC)')
    ax2.set_title('Max MI per Layer')
    ax2.set_xticks(range(num_layers))
    ax2.set_xticklabels([f'L{l}' for l in layers])
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 峰值位置分布
    ax3 = axes[1, 0]
    peak_positions = []
    for sid in range(num_samples):
        for layer in layers:
            mi_vals = mi_dict[sid]['reps'][layer]
            if isinstance(mi_vals, torch.Tensor):
                mi_vals = mi_vals.cpu().numpy()
            peak_pos = np.argmax(mi_vals)
            peak_positions.append(peak_pos)

    ax3.hist(peak_positions, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Peak Position (Patch Index)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Peak Positions')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 每个样本的平均 MI 热力图
    ax4 = axes[1, 1]
    sample_layer_matrix = np.zeros((num_samples, num_layers))
    for sid in range(num_samples):
        for layer_idx, layer in enumerate(layers):
            mi_vals = mi_dict[sid]['reps'][layer]
            if isinstance(mi_vals, torch.Tensor):
                mi_vals = mi_vals.cpu().numpy()
            sample_layer_matrix[sid, layer_idx] = np.mean(mi_vals)

    im = ax4.imshow(sample_layer_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Sample ID')
    ax4.set_title('Average MI per Sample and Layer')
    ax4.set_xticks(range(num_layers))
    ax4.set_xticklabels([f'L{l}' for l in layers])
    plt.colorbar(im, ax=ax4, label='Avg MI')

    plt.suptitle(f'{dataset} - MI Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{dataset}_mi_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved summary: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制 Timer 模型的 MI 图")

    parser.add_argument("--data", type=str, default="ETTh1", help="数据集名称")
    parser.add_argument("--model_id", type=str, default="timer_mi", help="模型标识")
    parser.add_argument("--mi_dir", type=str, default="timer_mi_results", help="MI 结果目录")
    parser.add_argument("--save_dir", type=str, default="timer_mi_results/figures", help="图片保存目录")
    parser.add_argument("--plot_type", type=str, default="all",
                       choices=["all", "sample_layer", "heatmap", "summary"],
                       help="图片类型: all=全部, sample_layer=每个样本每层, heatmap=热力图, summary=汇总")

    args = parser.parse_args()

    # 加载 MI 结果
    mi_file = os.path.join(args.mi_dir, f'{args.data}_gtmodel={args.model_id}_testmodel={args.model_id}.pth')

    if not os.path.exists(mi_file):
        print(f"MI file not found: {mi_file}")
        print("Please run timer_calculate_mi.py first!")
        return

    print(f"Loading MI results from {mi_file}...")
    mi_dict = torch.load(mi_file)

    # 生成图片
    if args.plot_type in ["all", "sample_layer"]:
        print("\nGenerating sample-layer MI plots...")
        plot_mi_per_sample_layer(mi_dict, args.save_dir, args.data)

    if args.plot_type in ["all", "heatmap"]:
        print("\nGenerating heatmap...")
        plot_mi_heatmap(mi_dict, args.save_dir, args.data)

    if args.plot_type in ["all", "summary"]:
        print("\nGenerating summary plots...")
        plot_mi_summary(mi_dict, args.save_dir, args.data)

    print(f"\nAll figures saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
