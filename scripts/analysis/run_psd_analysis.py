#!/usr/bin/env python3
"""
PSD 分析脚本 - 运行入口

功能：对比高 MI Patch vs 低 MI Patch 的频域特性差异

使用方法:
    bash scripts/analysis/run_psd_analysis.sh
    或
    python scripts/analysis/run_psd_analysis.py
"""

import os
import sys

# 确保项目根目录在路径中
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import numpy as np
import argparse

from scripts.analysis.psd_analysis import analyze_mi_patches


def generate_demo_data(
    n_patches: int = 100,
    patch_size: int = 24,
    stride: int = 12,
    n_channels: int = 7,
    n_timesteps: int = 960
) -> tuple:
    """
    生成演示用的模拟数据

    模拟场景：
    - 时序数据包含多个频率成分
    - 高 MI Patch 包含更强的周期信号
    - 低 MI Patch 更接近白噪声
    """
    print("\n" + "="*60)
    print("生成模拟数据...")
    print("="*60)

    # 计算总时间步长
    total_len = n_patches * stride + patch_size
    print(f"数据总长度: {total_len}")

    # 生成基础信号（多频率成分）
    t = np.arange(total_len)
    base_signal = np.zeros((total_len, n_channels))

    # 频率1: 低频趋势（所有通道共享）
    base_signal += 0.3 * np.sin(2 * np.pi * t / 96)  # 周期 96

    # 频率2: 中频周期
    base_signal += 0.2 * np.sin(2 * np.pi * t / 24)   # 周期 24

    # 随机噪声
    np.random.seed(42)
    noise = np.random.randn(total_len, n_channels) * 0.1
    base_signal += noise

    # 创建 Patch 索引的 MI 值
    # 前 20 个 Patch 高 MI（有明显周期）
    # 后 80 个 Patch 低 MI（噪声为主）
    mi_scores = np.zeros(n_patches)
    mi_scores[:20] = np.random.uniform(0.6, 0.9, size=20)
    mi_scores[20:] = np.random.uniform(0.1, 0.4, size=n_patches - 20)
    np.random.shuffle(mi_scores)  # 打乱顺序模拟真实情况

    print(f"生成 MI 分数: shape = {mi_scores.shape}")
    print(f"  Top 5% 阈值: {np.percentile(mi_scores, 95):.4f}")
    print(f"  高 MI 均值: {mi_scores[mi_scores > np.percentile(mi_scores, 95)].mean():.4f}")
    print(f"  低 MI 均值: {mi_scores[mi_scores <= np.percentile(mi_scores, 95)].mean():.4f}")

    return mi_scores, base_signal


def save_demo_data(mi_scores: np.ndarray, data: np.ndarray, output_dir: str):
    """保存模拟数据到文件"""
    os.makedirs(output_dir, exist_ok=True)

    mi_path = os.path.join(output_dir, "demo_mi_scores.npy")
    data_path = os.path.join(output_dir, "demo_data.npy")

    np.save(mi_path, mi_scores)
    np.save(data_path, data)

    print(f"\n模拟数据已保存:")
    print(f"  MI 分数: {mi_path}")
    print(f"  时序数据: {data_path}")

    return mi_path, data_path


def main():
    parser = argparse.ArgumentParser(description="运行 PSD 频域分析")
    parser.add_argument("--n_patches", type=int, default=100, help="Patch 数量")
    parser.add_argument("--patch_size", type=int, default=24, help="Patch 长度")
    parser.add_argument("--stride", type=int, default=12, help="滑动步长")
    parser.add_argument("--top_percent", type=float, default=5.0, help="高 MI 组百分比")
    parser.add_argument("--output_dir", type=str, default="experiments/output/psd_analysis", help="输出目录")
    parser.add_argument("--fs", type=float, default=1.0, help="采样频率")
    parser.add_argument("--skip_demo", action="store_true", help="跳过演示数据生成")
    parser.add_argument("--mi_path", type=str, help="使用已有的 MI 分数文件")
    parser.add_argument("--data_path", type=str, help="使用已有的数据文件")
    args = parser.parse_args()

    if args.skip_demo:
        # 使用真实数据
        if not args.mi_path or not args.data_path:
            print("错误: 使用 --skip_demo 时必须指定 --mi_path 和 --data_path")
            sys.exit(1)
        mi_path = args.mi_path
        data_path = args.data_path
        mi_scores = np.load(mi_path)
        data = np.load(data_path)
    else:
        # 生成演示数据
        mi_scores, data = generate_demo_data(
            n_patches=args.n_patches,
            patch_size=args.patch_size,
            stride=args.stride
        )
        mi_path, data_path = save_demo_data(mi_scores, data, "experiments/output/psd_analysis")

    # 执行分析
    results = analyze_mi_patches(
        mi_scores=mi_scores,
        data=data,
        patch_size=args.patch_size,
        stride=args.stride,
        fs=args.fs,
        top_percent=args.top_percent,
        output_dir=args.output_dir
    )

    print("\n" + "="*60)
    print("分析完成!")
    print(f"结果保存在: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()