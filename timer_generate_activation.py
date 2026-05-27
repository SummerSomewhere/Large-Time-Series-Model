#!/usr/bin/env python3
"""
Timer 输入序列激活提取

模仿 MI-Peaks/src/generate_activation.py，为时间序列模型 Timer 定制。

功能：
- 提取 Timer Encoder 各层的 hidden states
- 按 patch 维度组织输出
- 支持指定层索引

数据格式：
{
    sample_id: {
        'reps': {
            layer_idx: {
                'hiddens': torch.Tensor [num_patches, d_model]  # 各 patch 的表示
            }
        },
        'patch_emb': torch.Tensor [num_patches, patch_len],  # patch 原始数据（可选）
        'token_ids': None  # 时间序列没有 token 概念，设为 None
    }
}
"""

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm

# 添加项目路径
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_factory import data_provider
from utils.masking import TriangularCausalMask


class Hook:
    """Hook 类：捕获每一层 Decoder/Encoder 的输出"""
    
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_inputs, module_outputs):
        """
        捕获模块输出并 detach 到 CPU
        module_outputs: [B, N, D] 或 [B, L, D]
        """
        if isinstance(module_outputs, tuple):
            hidden = module_outputs[0]
        else:
            hidden = module_outputs
        
        # 取第一个样本，detach 并移到 CPU
        h = hidden[0].detach().cpu()
        self.outputs.append(h)


class ActivationCollector:
    """Timer 模型激活收集器"""
    
    def __init__(self, model, patch_len, stride, collect_layers=None):
        """
        Args:
            model: Timer 模型
            patch_len: patch 长度
            stride: patch 步长
            collect_layers: 要收集的层索引列表，None 表示全部
        """
        self.model = model
        self.patch_len = patch_len
        self.stride = stride
        self.collect_layers = collect_layers
        self.hooks = []
        self.handles = []
        
        # 获取 decoder 层
        core = model.module if hasattr(model, "module") else model
        
        # 根据模型类型获取 attn_layers
        if hasattr(core, 'decoder'):
            if hasattr(core.decoder, 'attn_layers'):
                # RefinementEncoder, InjectionEncoder, Encoder
                self.attn_layers = core.decoder.attn_layers
            elif hasattr(core.decoder, 'layers'):
                # Decoder (带 cross attention)
                self.attn_layers = core.decoder.layers
            else:
                raise ValueError(f"Unknown decoder type: {type(core.decoder)}")
        else:
            raise ValueError("Model does not have 'decoder' attribute")
    
    def register_hooks(self):
        """注册所有层的 forward hooks"""
        self.hooks = []
        self.handles = []
        
        num_layers = len(self.attn_layers)
        
        for i, layer in enumerate(self.attn_layers):
            if self.collect_layers is not None and i not in self.collect_layers:
                continue
            
            hook = Hook()
            handle = layer.register_forward_hook(hook)
            self.hooks.append(hook)
            self.handles.append(handle)
    
    def clear(self):
        """清空所有 hook 的缓存"""
        for hook in self.hooks:
            hook.outputs = []
    
    def remove_hooks(self):
        """移除所有注册的 hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.hooks = []
    
    def get_layer_indices(self):
        """获取实际收集的层索引"""
        if self.collect_layers is None:
            return list(range(len(self.attn_layers)))
        return self.collect_layers


def preprocess_input(x: torch.Tensor) -> torch.Tensor:
    """
    Timer 标准的输入预处理：z-score 标准化
    
    Args:
        x: [B, L, M] 原始输入
    
    Returns:
        标准化后的输入
    """
    means = x.mean(dim=1, keepdim=True)
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x = x / stdev
    return x


def collect_activations(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    patch_len: int,
    stride: int,
    collect_layers: list = None,
    use_pool: bool = True,
    pool_over_vars: bool = True,
) -> dict:
    """
    收集一个 batch 的激活
    
    Args:
        model: Timer 模型
        batch_x: [B, L, M] 输入序列
        patch_len: patch 长度
        stride: patch 步长
        collect_layers: 要收集的层索引
        use_pool: 是否对 n_vars 维度做平均池化
        pool_over_vars: 是否在 n_vars 上池化（True=跨变量平均，False=保留每个变量）
    
    Returns:
        acts: {
            'reps': {
                layer_idx: torch.Tensor [N, D] 或 [B, N, D] 或 [B, n_vars, N, D]
            },
            'patch_emb': torch.Tensor [N, patch_len]
        }
    """
    device = batch_x.device
    B, L, M = batch_x.shape
    
    # 预处理
    x = preprocess_input(batch_x)
    
    # 转换为 patch embedding 格式
    x2 = x.permute(0, 2, 1)  # [B, M, L]
    
    # 获取 model core
    core = model.module if hasattr(model, "module") else model
    
    # Patch embedding
    dec_in, n_vars = core.patch_embedding(x2)  # [B*M, N, D] 或 [B, n_vars, N, D]
    
    # Reshape: 如果输出是 3D [B*M, N, D]，重塑为 4D [B, M, N, D] 然后池化
    if len(dec_in.shape) == 3:
        BM, N, D = dec_in.shape
        assert BM == B * M, f"Batch size mismatch: {BM} != {B * M}"
        dec_in = dec_in.view(B, M, N, D)  # [B, M, N, D]
    
    _, n_vars_real, N, D = dec_in.shape
    
    # 是否跨变量池化
    if pool_over_vars:
        dec_in_pooled = dec_in.mean(dim=1)  # [B, N, D]
    else:
        dec_in_pooled = dec_in  # [B, n_vars, N, D]
    
    # 创建因果 mask
    BM = B * n_vars_real
    mask = TriangularCausalMask(BM, N, device=device)
    
    # 注册 hooks
    collector = ActivationCollector(model, patch_len, stride, collect_layers)
    collector.register_hooks()
    
    # Forward
    with torch.no_grad():
        if hasattr(core.decoder, 'attn_layers'):
            # Encoder-style decoder
            h = dec_in.view(B * n_vars_real, N, D)
            layers_outputs = []
            for i, layer in enumerate(core.decoder.attn_layers):
                if collect_layers is not None and i not in collect_layers:
                    continue
                if pool_over_vars:
                    h_in = h.view(B, n_vars_real, N, D).mean(dim=1)  # [B, N, D]
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
            # Decoder-style (with cross attention) - 简化处理
            h = dec_in_pooled
            layers_outputs = []
            for i, layer in enumerate(core.decoder.layers):
                if collect_layers is not None and i not in collect_layers:
                    continue
                h, _ = layer(h, h, attn_mask=mask)
                layers_outputs.append(h)
    
    # 收集结果
    layer_indices = collector.get_layer_indices()
    acts = {
        'reps': {},
        'patch_emb': None  # patch 原始数据（可选）
    }
    
    for idx, layer_idx in enumerate(layer_indices):
        h = layers_outputs[idx]
        # h: [B, N, D] 或 [B, n_vars, N, D]
        if pool_over_vars and len(h.shape) == 3:
            acts['reps'][layer_idx] = h  # [B, N, D]
        elif not pool_over_vars and len(h.shape) == 4:
            acts['reps'][layer_idx] = h  # [B, n_vars, N, D]
        else:
            acts['reps'][layer_idx] = h
    
    # 清理
    collector.remove_hooks()
    
    return acts


def load_model_and_data(args):
    """
    加载模型和数据
    
    Returns:
        model: Timer 模型
        loader: 数据加载器
        configs: 配置对象
    """
    from models.Timer import Model
    
    # 构建配置
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
    configs.use_weight_decay = 0
    configs.weight_decay = 0.01
    configs.loss = 'MSE'
    configs.lradj = 'type1'
    configs.train_epochs = 0
    configs.patience = 3
    configs.learning_rate = 1e-4
    configs.itr = 1
    configs.finetune_epochs = 0
    configs.output_attention = False
    configs.distil = True
    configs.model_id = 'timer_mi'
    configs.model = 'Timer'
    configs.output_len_list = None
    configs.mask_rate = 0.25
    configs.data_type = 'custom'
    configs.decay_fac = 0.75
    configs.cos_warm_up_steps = 100
    configs.cos_max_decay_steps = 60000
    configs.cos_max_decay_epoch = 10
    configs.cos_max = 1e-4
    configs.cos_min = 2e-6
    
    # 命令行参数覆盖
    for arg in ['root_path', 'data_path', 'data', 'seq_len', 'label_len', 'pred_len',
                'patch_len', 'd_model', 'd_ff', 'e_layers', 'd_layers', 'n_heads',
                'factor', 'dropout', 'activation', 'embed', 'freq', 'features',
                'stride', 'num_workers', 'batch_size']:
        if hasattr(args, arg):
            setattr(configs, arg, getattr(args, arg))
    
    # 设置 patch_len 和 stride
    if hasattr(args, 'patch_len'):
        configs.patch_len = args.patch_len
    if hasattr(args, 'stride') and args.stride > 0:
        configs.patch_len = args.stride  # Timer 中 stride = patch_len
    
    # 加载模型
    device = torch.device(args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    model.eval()
    
    # 加载数据
    _, loader = data_provider(configs, flag='test')
    
    return model, loader, configs


def main():
    parser = argparse.ArgumentParser(description="生成 Timer 模型的中间层激活")
    
    # 数据参数
    parser.add_argument("--root_path", type=str, default="./datasets/", help="数据集根目录")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="数据文件名")
    parser.add_argument("--data", type=str, default="ETTh1", help="数据集名称")
    parser.add_argument("--features", type=str, default="M", help="特征类型: M, S, M+S")
    
    # 模型参数
    parser.add_argument("--seq_len", type=int, default=672, help="输入序列长度")
    parser.add_argument("--label_len", type=int, default=576, help="标签序列长度")
    parser.add_argument("--pred_len", type=int, default=96, help="预测序列长度")
    parser.add_argument("--patch_len", type=int, default=96, help="patch 长度")
    parser.add_argument("--stride", type=int, default=96, help="patch 步长")
    parser.add_argument("--d_model", type=int, default=1024, help="模型维度")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈维度")
    parser.add_argument("--e_layers", type=int, default=8, help="编码器层数")
    parser.add_argument("--d_layers", type=int, default=1, help="解码器层数")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--factor", type=int, default=3, help="注意力 factor")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--activation", type=str, default="gelu", help="激活函数")
    
    # 激活收集参数
    parser.add_argument("--layers", nargs='+', type=int, default=None, 
                       help="要收集的层索引，None 表示全部")
    parser.add_argument("--pool_over_vars", action='store_true', default=True,
                       help="是否跨变量池化")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="MI-Peaks/acts/timer", 
                       help="激活保存目录")
    parser.add_argument("--sample_num", type=int, default=-1, 
                       help="样本数量，-1 表示全部")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型 checkpoint 路径")
    
    args = parser.parse_args()
    
    # 加载模型和数据
    model, loader, configs = load_model_and_data(args)
    
    # 加载 checkpoint
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Loaded checkpoint from {args.ckpt_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "reasoning_evolve"), exist_ok=True)
    
    # 解析层索引
    if args.layers is not None:
        layers = [int(l) for l in args.layers]
    else:
        layers = None
    
    # 收集激活
    all_acts = {}
    sample_id = 0
    
    # 计算样本数量限制
    max_samples = args.sample_num if args.sample_num > 0 else len(loader.dataset)
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(loader, desc="收集激活"):
        if sample_id >= max_samples:
            break
        
        batch_x = batch_x.float().to(device)
        
        # 收集激活
        acts = collect_activations(
            model=model,
            batch_x=batch_x,
            patch_len=configs.patch_len,
            stride=configs.stride,
            collect_layers=layers,
            pool_over_vars=args.pool_over_vars,
        )
        
        # 存储（只保留第一个样本，节省内存）
        for b in range(batch_x.shape[0]):
            if sample_id >= max_samples:
                break
            
            all_acts[sample_id] = {
                'reps': {},
                'token_ids': None,
            }
            
            for layer_idx, h in acts['reps'].items():
                # h: [B, N, D] 或 [B, n_vars, N, D]
                if len(h.shape) == 3:
                    # [B, N, D] -> 取第 b 个样本 -> [N, D]
                    all_acts[sample_id]['reps'][layer_idx] = h[b].cpu()
                elif len(h.shape) == 4:
                    # [B, n_vars, N, D] -> 取第 b 个样本 -> [n_vars, N, D]
                    all_acts[sample_id]['reps'][layer_idx] = h[b].cpu()
            
            sample_id += 1
    
    # 保存
    output_file = os.path.join(
        args.output_dir, 
        "reasoning_evolve", 
        f"{args.data}_{args.model_id}.pth"
    )
    torch.save(all_acts, output_file)
    print(f"Saved activations to {output_file}")
    print(f"Total samples: {len(all_acts)}")


if __name__ == '__main__':
    main()
