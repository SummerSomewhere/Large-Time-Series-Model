#!/usr/bin/env python3
"""
Timer MI 完整流程

整合激活提取、MI计算和可视化，中间数据保存在临时目录，完成后清理。

流程：
1. 加载模型，提取输入序列激活
2. 提取真值序列激活
3. 计算 MI (HSIC)
4. 绘制并保存图片到主目录
"""

import os
import sys
import argparse
import shutil
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_factory import data_provider
from utils.masking import TriangularCausalMask
from timer_mi_estimators import estimate_mi_hsic


def preprocess_input(x: torch.Tensor) -> torch.Tensor:
    """Timer 标准的输入预处理：z-score 标准化"""
    means = x.mean(dim=1, keepdim=True)
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return x / stdev


def collect_input_activations(model, batch_x, patch_len, stride, collect_layers=None, pool_over_vars=True):
    """收集输入序列的激活"""
    device = batch_x.device
    B, L, M = batch_x.shape
    
    x = preprocess_input(batch_x)
    x2 = x.permute(0, 2, 1)
    
    core = model.module if hasattr(model, "module") else model
    
    dec_in, n_vars = core.patch_embedding(x2)
    
    if len(dec_in.shape) == 3:
        BM, N, D = dec_in.shape
        dec_in = dec_in.view(B, M, N, D)
    
    _, n_vars_real, N, D = dec_in.shape
    
    if pool_over_vars:
        dec_in_pooled = dec_in.mean(dim=1)
    else:
        dec_in_pooled = dec_in
    
    BM = B * n_vars_real
    mask = TriangularCausalMask(BM, N, device=device)
    
    layers_outputs = []
    
    with torch.no_grad():
        if hasattr(core.decoder, 'attn_layers'):
            h = dec_in.view(B * n_vars_real, N, D)
            for i, layer in enumerate(core.decoder.attn_layers):
                if collect_layers is not None and i not in collect_layers:
                    continue
                if pool_over_vars:
                    h_in = h.view(B, n_vars_real, N, D).mean(dim=1)
                else:
                    h_in = h.view(B, n_vars_real, N, D)
                h_flat = h_in.view(B * (1 if pool_over_vars else n_vars_real), -1, D)
                h_out, _ = layer(h_flat, attn_mask=mask)
                layers_outputs.append(h_out)
                if pool_over_vars:
                    h = h_out.view(B, 1, N, D).expand(-1, n_vars_real, -1, -1).reshape(B * n_vars_real, N, D)
                else:
                    h = h_out
        else:
            h = dec_in_pooled
            for i, layer in enumerate(core.decoder.layers):
                if collect_layers is not None and i not in collect_layers:
                    continue
                h, _ = layer(h, h, attn_mask=mask)
                layers_outputs.append(h)
    
    layer_indices = list(range(len(layers_outputs))) if collect_layers is None else collect_layers
    
    acts = {}
    for idx, layer_idx in enumerate(layer_indices):
        h = layers_outputs[idx]
        if len(h.shape) == 3:
            acts[layer_idx] = h[0].cpu()
        elif len(h.shape) == 4:
            acts[layer_idx] = h[0].cpu()
    
    return acts


def collect_gt_activations(model, batch_y, patch_len, stride, collect_layers=None, 
                          pred_len=96, label_len=576, use_ims=False, pool_over_vars=True):
    """收集真值序列的激活"""
    device = batch_y.device
    B, L_y, M = batch_y.shape
    
    y = preprocess_input(batch_y)
    
    if use_ims:
        y_future = y[:, label_len:label_len + pred_len, :]
    else:
        y_future = y[:, -pred_len:, :]
    
    y2 = y_future.permute(0, 2, 1)
    
    core = model.module if hasattr(model, "module") else model
    
    dec_in, n_vars = core.patch_embedding(y2)
    
    if len(dec_in.shape) == 3:
        BM, N, D = dec_in.shape
        dec_in = dec_in.view(B, M, N, D)
    
    _, n_vars_real, N, D = dec_in.shape
    
    if pool_over_vars:
        dec_in_pooled = dec_in.mean(dim=1)
    else:
        dec_in_pooled = dec_in
    
    BM = B * n_vars_real
    mask = TriangularCausalMask(BM, N, device=device)
    
    layers_outputs = []
    
    with torch.no_grad():
        if hasattr(core.decoder, 'attn_layers'):
            h = dec_in.view(B * n_vars_real, N, D)
            for i, layer in enumerate(core.decoder.attn_layers):
                if collect_layers is not None and i not in collect_layers:
                    continue
                if pool_over_vars:
                    h_in = h.view(B, n_vars_real, N, D).mean(dim=1)
                else:
                    h_in = h.view(B, n_vars_real, N, D)
                h_flat = h_in.view(B * (1 if pool_over_vars else n_vars_real), -1, D)
                h_out, _ = layer(h_flat, attn_mask=mask)
                layers_outputs.append(h_out)
                if pool_over_vars:
                    h = h_out.view(B, 1, N, D).expand(-1, n_vars_real, -1, -1).reshape(B * n_vars_real, N, D)
                else:
                    h = h_out
        else:
            h = dec_in_pooled
            for i, layer in enumerate(core.decoder.layers):
                if collect_layers is not None and i not in collect_layers:
                    continue
                h, _ = layer(h, h, attn_mask=mask)
                layers_outputs.append(h)
    
    layer_indices = list(range(len(layers_outputs))) if collect_layers is None else collect_layers
    
    acts = {}
    for idx, layer_idx in enumerate(layer_indices):
        h = layers_outputs[idx]
        if len(h.shape) == 3:
            acts[layer_idx] = h[0].cpu()
        elif len(h.shape) == 4:
            acts[layer_idx] = h[0].cpu()
    
    return acts


def patch_centers(num_patches, patch_len, stride):
    """计算每个 patch 的中心位置"""
    centers = np.arange(num_patches) * stride + patch_len / 2
    return centers


def plot_mi_heatmap(mi_per_sample, save_path, patch_centers, num_layers):
    """绘制 MI 热力图"""
    num_samples = len(mi_per_sample)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    
    for layer_idx in range(min(num_layers, 8)):
        ax = axes[layer_idx]
        
        mi_values = []
        for sid in range(num_samples):
            if layer_idx in mi_per_sample[sid]['reps']:
                mi_values.append(mi_per_sample[sid]['reps'][layer_idx].numpy())
        
        if mi_values:
            mi_matrix = np.stack(mi_values)
            mean_mi = mi_matrix.mean(axis=0)
            std_mi = mi_matrix.std(axis=0)
            
            x = patch_centers[:len(mean_mi)]
            ax.plot(x, mean_mi, 'b-', linewidth=2, label='Mean MI')
            ax.fill_between(x, mean_mi - std_mi, mean_mi + std_mi, alpha=0.3)
            ax.set_xlabel('Patch Center Position')
            ax.set_ylabel('MI (HSIC)')
            ax.set_title(f'Layer {layer_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved MI heatmap to {save_path}")


def plot_mi_per_sample(mi_per_sample, save_dir, patch_centers, sample_indices=None):
    """为每个样本绘制 MI 图"""
    os.makedirs(save_dir, exist_ok=True)
    
    if sample_indices is None:
        sample_indices = list(mi_per_sample.keys())[:10]  # 默认最多10个样本
    
    for sid in sample_indices:
        if sid not in mi_per_sample:
            continue
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes = axes.flatten()
        
        for layer_idx in range(min(8, len(mi_per_sample[sid]['reps']))):
            ax = axes[layer_idx]
            
            if layer_idx in mi_per_sample[sid]['reps']:
                mi_values = mi_per_sample[sid]['reps'][layer_idx].numpy()
                x = patch_centers[:len(mi_values)]
                
                ax.plot(x, mi_values, 'b-', linewidth=1.5)
                ax.set_xlabel('Position')
                ax.set_ylabel('MI (HSIC)')
                ax.set_title(f'Layer {layer_idx}')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sample {sid} - MI per Patch')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{sid:04d}_mi.png'), dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Saved sample MI plots to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Timer MI 完整流程")
    
    # 数据参数
    parser.add_argument("--root_path", type=str, default="./datasets/", help="数据集根目录")
    parser.add_argument("--data", type=str, default="ETTh1", help="数据集名称")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="数据文件名")
    parser.add_argument("--features", type=str, default="M", help="特征类型")
    
    # 模型参数
    parser.add_argument("--seq_len", type=int, default=672, help="输入序列长度")
    parser.add_argument("--label_len", type=int, default=576, help="标签序列长度")
    parser.add_argument("--pred_len", type=int, default=96, help="预测序列长度")
    parser.add_argument("--patch_len", type=int, default=96, help="patch 长度")
    parser.add_argument("--stride", type=int, default=96, help="patch 步长")
    parser.add_argument("--d_model", type=int, default=1024, help="模型维度")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈维度")
    parser.add_argument("--e_layers", type=int, default=8, help="编码器层数")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    
    # 其他参数
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--sample_num", type=int, default=-1, help="样本数量，-1 表示全部")
    parser.add_argument("--plot_num", type=int, default=100, help="绘图样本数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_ims", action='store_true', default=False, help="是��使用 IMS 模式")
    
    # 输出参数
    parser.add_argument("--save_dir", type=str, default="results/mi/timer", help="结果保存目录")
    parser.add_argument("--keep_temp", action='store_true', default=False, help="保留中间结果")
    parser.add_argument("--output_fig", type=str, default="mi_analysis.png", help="输出图片文件名")
    
    args = parser.parse_args()
    
    temp_dir = os.path.join(args.save_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("=" * 60)
    print("Timer MI 完整流程")
    print("=" * 60)
    print(f"数据集: {args.data}")
    print(f"样本数量: {args.sample_num}")
    print(f"临时目录: {temp_dir}")
    print()
    
    # ── 1. 加载模型 ──────────────────────────────────────────────
    print("[1/4] 加载模型...")
    
    class Config:
        pass
    
    configs = Config()
    configs.task_name = 'forecast'
    configs.is_training = 0
    configs.is_finetuning = 0
    configs.train_test = 0
    configs.use_multi_gpu = False
    configs.d_layers = 1
    configs.target = 'OT'
    configs.checkpoints = './checkpoints/'
    configs.inverse = False
    configs.use_amp = False
    configs.model_id = 'timer_mi'
    configs.model = 'Timer'
    configs.ckpt_path = ''
    configs.output_attention = False
    configs.dropout = 0.1
    configs.activation = 'gelu'
    configs.embed = 'timeF'
    configs.freq = 'h'
    configs.distil = True
    configs.model_id = 'timer_mi'
    configs.model = 'Timer'
    
    # 命令行参数覆盖
    for arg in ['root_path', 'data_path', 'data', 'seq_len', 'label_len', 'pred_len',
                'patch_len', 'stride', 'd_model', 'd_ff', 'e_layers', 'n_heads',
                'features', 'batch_size']:
        if hasattr(args, arg):
            setattr(configs, arg, getattr(args, arg))
    
    device = torch.device(args.device)
    model = Model(configs).to(device)
    model.eval()
    
    # 加载 checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Loaded checkpoint from {args.ckpt_path}")
    
    # 加载数据
    _, loader = data_provider(configs, flag='test')
    
    # ── 2. 提取激活 ──────────────────────────────────────────────
    print("\n[2/4] 提取激活...")
    
    all_input_acts = {}
    all_gt_acts = {}
    sample_id = 0
    max_samples = args.sample_num
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(loader, desc="提取激活"):
        if sample_id >= max_samples:
            break
        
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        
        # 提取输入激活
        input_acts = collect_input_activations(
            model, batch_x, configs.patch_len, configs.stride, pool_over_vars=True
        )
        
        # 提取真值激活
        gt_acts = collect_gt_activations(
            model, batch_y, configs.patch_len, configs.stride,
            pred_len=configs.pred_len, label_len=configs.label_len,
            use_ims=args.use_ims, pool_over_vars=True
        )
        
        # 保存
        all_input_acts[sample_id] = {'reps': input_acts}
        all_gt_acts[sample_id] = gt_acts
        
        sample_id += 1
    
    print(f"提取了 {len(all_input_acts)} 个样本的激活")
    
    # ── 3. 计算 MI ──────────────────────────────────────────────
    print("\n[3/4] 计算 MI (HSIC)...")
    
    final_mi_dict = {}
    
    for sid in tqdm(range(len(all_input_acts)), desc="计算 MI"):
        final_mi_dict[sid] = {'reps': {}}
        
        for layer_idx in all_input_acts[sid]['reps'].keys():
            num_patches = all_input_acts[sid]['reps'][layer_idx].shape[0]
            mi_values = torch.zeros(num_patches)
            
            for i in range(num_patches):
                mi_values[i] = estimate_mi_hsic(
                    all_input_acts[sid]['reps'][layer_idx][i],
                    all_gt_acts[sid][layer_idx][0]
                )
            
            final_mi_dict[sid]['reps'][layer_idx] = mi_values
    
    # ── 4. 绘制图片 ──────────────────────────────────────────────
    print("\n[4/4] 绘制图片...")
    
    num_patches = all_input_acts[0]['reps'][0].shape[0]
    centers = patch_centers(num_patches, configs.patch_len, configs.stride)
    num_layers = len(all_input_acts[0]['reps'])
    
    # 主热力图
    output_fig = os.path.join(args.save_dir, args.output_fig)
    plot_mi_heatmap(final_mi_dict, output_fig, centers, num_layers)
    
    # 每个样本的图
    sample_fig_dir = os.path.join(args.save_dir, "sample_mi_figures")
    plot_mi_per_sample(final_mi_dict, sample_fig_dir, centers, 
                      sample_indices=list(range(min(args.plot_num, len(final_mi_dict)))))
    
    # ── 清理临时文件 ──────────────────────────────────────────────
    if not args.keep_temp:
        print("\n清理临时文件...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("已清理临时目录")
    
    print("\n" + "=" * 60)
    print("完成！")
    print(f"主图片: {output_fig}")
    print(f"样本图片: {sample_fig_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
