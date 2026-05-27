#!/usr/bin/env python3
"""
Timer MI (HSIC) 计算

模仿 MI-Peaks/src/cal_mi.py，为时间序列模型 Timer 定制。

核心功能：
1. 加载输入序列和真值序列的激活
2. 对每个样本的每层，计算各 patch 与真值之间的 HSIC
3. 使用 batch 维度进行 HSIC 估计（需要 B >= 4）
4. 支持多 GPU 分布式计算

数据格式：
- acts[sample_id]['reps'][layer_idx]: torch.Tensor [N, D] - 输入序列各 patch 表示
- gt_acts[sample_id][layer_idx]: torch.Tensor [N_gt, D] - 真值序列各 patch 表示

MI 计算逻辑（与 etth1_mi_hsic_peaks.py 一致）：
- 对每个 patch position p：
  - X = h_x[:, p, :]   shape [B, D] - 该位置所有样本的表示
  - Y = h_y[:, 0, :]   shape [B, D] - 真值第一个 patch
  - 计算 HSIC(X, Y)
- 返回每个 patch 位置的 MI 值 [N]
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# 添加项目路径
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from timer_mi_estimators import hsic_unbiased, zscore, estimate_mi_hsic


def mi_sequence_hsic(
    Hbm: torch.Tensor,
    Hy: torch.Tensor,
) -> np.ndarray:
    """
    计算每个 patch 位置与真值之间的 HSIC
    
    与 etth1_mi_hsic_peaks.py 中的实现一致：
    - 对每个 patch index p：
      - X = Hbm[:, p, :]  shape [B, D] - 该位置所有样本的表示
      - Y = Hy[:, 0, :]   shape [B, D] - 真值第一个 patch（扩展到 N_x 个位置）
    - 使用 batch 维度计算 HSIC
    
    Args:
        Hbm: [B, N_x, D] 输入序列各 patch 的 hidden states
        Hy:  [B, N_y, D] 真值序列各 patch 的 hidden states
    
    Returns:
        np.ndarray: [N_x] 每个 patch 位置的 HSIC 值
    """
    B, N_x, D = Hbm.shape
    B2, N_y, _ = Hy.shape
    assert B == B2, f"Batch size mismatch: Hbm={Hbm.shape}, Hy={Hy.shape}"

    # 真值只有 1 个 patch（或取第一个 patch），扩展到 N_x 个位置
    hy_single = Hy[:, 0, :].unsqueeze(1)  # [B, 1, D]
    Hy_expanded = hy_single.expand(-1, N_x, -1)  # [B, N_x, D]

    out = []
    for p in range(N_x):
        X = zscore(Hbm[:, p, :].contiguous())
        Y = zscore(Hy_expanded[:, p, :].contiguous())
        hsic_val = hsic_unbiased(X, Y).detach().float().cpu().item()
        out.append(hsic_val)
    
    return np.asarray(out, dtype=np.float64)


def calculate_mi(
    acts: dict,
    gt_acts: dict,
    layers: list = None,
    num_samples: int = -1,
    save_dir: str = 'results/mi/',
    min_batch_size: int = 4,
    args=None,
) -> dict:
    """
    计算 MI 的主函数
    
    遍历每个样本，计算每个层每个 patch 的 MI 值
    
    Args:
        acts: 输入序列激活 {sample_id: {'reps': {layer: [N, D]}}}
        gt_acts: 真值激活 {sample_id: {layer: [N_gt, D]}}
        layers: 要计算的层索引列表，None 表示全部
        num_samples: 样本数量，-1 表示全部
        save_dir: 结果保存目录
        min_batch_size: 最小 batch 大小（用于 HSIC 估计）
        args: 命令行参数
    
    Returns:
        final_mi_dict: {
            sample_id: {
                'reps': {layer: [N] MI values per patch},
                'total_patches': int
            }
        }
    """
    num_samples = len(acts) if num_samples < 0 else min(num_samples, len(acts))
    
    # 确定层索引
    if layers is None or len(layers) == 0:
        layers = list(acts[0]['reps'].keys())
    layers = sorted(layers)
    
    # 尝试加载已计算的结果（断点续算）
    save_file = _get_save_file(args, save_dir)
    try:
        final_mi_dict = torch.load(save_file)
        print(f"Loaded existing results from {save_file}")
    except:
        final_mi_dict = None
    
    if final_mi_dict is None:
        final_mi_dict = {
            sid: {
                'reps': {layer: [] for layer in layers},
                'total_patches': -1
            }
            for sid in range(num_samples)
        }
    
    # 按 batch 计算 MI（使用相邻样本构建 batch）
    batch_size = min_batch_size
    n_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="计算 MI batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        if current_batch_size < min_batch_size:
            # 样本不足，跳过或合并到上一个 batch
            continue
        
        for layer in layers:
            # 收集该 batch 该层的所有激活
            Hbm_batch = []  # [B, N, D]
            Hy_batch = []   # [B, N_gt, D]
            
            for sid in range(start_idx, end_idx):
                if layer in acts[sid]['reps'] and layer in gt_acts[sid]:
                    Hbm_batch.append(acts[sid]['reps'][layer])  # [N, D]
                    Hy_batch.append(gt_acts[sid][layer])        # [N_gt, D]
            
            if len(Hbm_batch) < min_batch_size:
                continue
            
            # Stack 成 batch
            Hbm = torch.stack(Hbm_batch, dim=0)  # [B, N, D]
            Hy = torch.stack(Hy_batch, dim=0)     # [B, N_gt, D]
            
            # 计算 MI 序列
            mi_seq = mi_sequence_hsic(Hbm, Hy)  # [N]
            
            # 分配到各个样本
            for i, sid in enumerate(range(start_idx, end_idx)):
                if sid in final_mi_dict:
                    final_mi_dict[sid]['reps'][layer] = mi_seq
                    if final_mi_dict[sid]['total_patches'] < 0:
                        final_mi_dict[sid]['total_patches'] = len(mi_seq)
        
        # 定期保存
        torch.save(final_mi_dict, save_file)
    
    return final_mi_dict


def _get_save_file(args, save_dir):
    """生成保存文件名"""
    os.makedirs(save_dir, exist_ok=True)
    
    if args is not None:
        dataset = getattr(args, 'dataset', args.data)
        model_tag = getattr(args, 'model_tag', 'Timer')
        gt_model_tag = getattr(args, 'gt_model_tag', 'Timer')
        return os.path.join(save_dir, f"{dataset}_gt={gt_model_tag}_test={model_tag}.pth")
    else:
        return os.path.join(save_dir, "timer_mi_results.pth")


def load_reps(dataset_name: str, model_tag: str, is_gt: bool = False, 
              data_dir: str = "MI-Peaks/acts/timer") -> dict:
    """
    加载激活数据
    
    Args:
        dataset_name: 数据集名称
        model_tag: 模型标识
        is_gt: 是否是真值激活
        data_dir: 数据目录
    
    Returns:
        acts: 激活字典
    """
    if is_gt:
        file_path = os.path.join(data_dir, "gt", f"{dataset_name}_{model_tag}.pth")
    else:
        file_path = os.path.join(data_dir, "reasoning_evolve", f"{dataset_name}_{model_tag}.pth")
    
    print(f"Loading activations from {file_path}...")
    return torch.load(file_path)


def main():
    parser = argparse.ArgumentParser(description="计算 Timer 模型的 MI (HSIC)")
    
    # 数据参数
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="ETTh1", help="数据集名称")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data_dir", type=str, default="MI-Peaks/acts/timer",
                       help="激活数据目录")
    
    # 模型参数
    parser.add_argument("--model_tag", type=str, default="Timer", 
                       help="输入序列模型标识")
    parser.add_argument("--gt_model_tag", type=str, default="Timer",
                       help="真值模型标识（通常相同）")
    
    # MI 计算参数
    parser.add_argument("--layers", nargs='+', type=int, default=None,
                       help="要计算的层索引，None 表示全部")
    parser.add_argument("--sample_num", type=int, default=-1,
                       help="样本数量，-1 表示全部")
    parser.add_argument("--min_batch_size", type=int, default=4,
                       help="最小 batch 大小（HSIC 需要 B >= 4）")
    
    # 输出参数
    parser.add_argument("--save_dir", type=str, default="results/mi/timer",
                       help="结果保存目录")
    
    args = parser.parse_args()
    
    # 加载数据
    acts = load_reps(args.data, args.model_tag, is_gt=False, data_dir=args.data_dir)
    gt_acts = load_reps(args.data, args.gt_model_tag, is_gt=True, data_dir=args.data_dir)
    
    # 解析层索引
    if args.layers is not None:
        layers = [int(l) for l in args.layers]
    else:
        layers = None
    
    # 计算 MI
    final_mi_dict = calculate_mi(
        acts=acts,
        gt_acts=gt_acts,
        layers=layers,
        num_samples=args.sample_num,
        save_dir=args.save_dir,
        min_batch_size=args.min_batch_size,
        args=args,
    )
    
    # 保存结果
    save_file = _get_save_file(args, args.save_dir)
    torch.save(final_mi_dict, save_file)
    print(f"Results saved to {save_file}")
    
    # 打印统计信息
    _print_summary(final_mi_dict)


def _print_summary(mi_dict: dict):
    """打印 MI 结果摘要"""
    if not mi_dict:
        return
    
    # 收集所有层的 MI 值
    all_layers = list(list(mi_dict.values())[0]['reps'].keys())
    
    print("\n" + "="*60)
    print("MI (HSIC) 结果摘要")
    print("="*60)
    
    for layer in sorted(all_layers):
        all_mi = []
        for sid, data in mi_dict.items():
            if layer in data['reps'] and len(data['reps'][layer]) > 0:
                all_mi.extend(data['reps'][layer])
        
        if all_mi:
            all_mi = np.array(all_mi)
            print(f"\nLayer {layer}:")
            print(f"  平均 MI: {np.mean(all_mi):.6f}")
            print(f"  最大 MI: {np.max(all_mi):.6f}")
            print(f"  最小 MI: {np.min(all_mi):.6f}")
            print(f"  标准差:  {np.std(all_mi):.6f}")
            
            # 找到 MI 峰值位置
            peak_idx = np.argmax(all_mi)
            print(f"  峰值位置: {peak_idx}")
    
    print("="*60)


if __name__ == '__main__':
    main()
