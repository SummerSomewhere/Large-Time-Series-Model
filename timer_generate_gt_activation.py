#!/usr/bin/env python3
"""
Timer 真值序列激活提取

模仿 MI-Peaks/src/generate_gt_activation.py，为时间序列模型 Timer 定制。

功能：
- 将真值 (ground truth) 未来序列通过 Timer Encoder
- 提取各层的 hidden states 作为"正确答案"的表示
- 用于与输入序列的激活计算 MI

数据格式：
{
    sample_id: {
        layer_idx: torch.Tensor [num_patches, d_model]  # 各 patch 的表示
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
        """
        if isinstance(module_outputs, tuple):
            hidden = module_outputs[0]
        else:
            hidden = module_outputs
        
        h = hidden[0].detach().cpu()
        self.outputs.append(h)


def preprocess_input(x: torch.Tensor) -> torch.Tensor:
    """
    Timer 标准的输入预处理：z-score 标准化
    """
    means = x.mean(dim=1, keepdim=True)
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x = x / stdev
    return x


def collect_gt_activations(
    model: torch.nn.Module,
    batch_y: torch.Tensor,
    patch_len: int,
    stride: int,
    collect_layers: list = None,
    pred_len: int = 96,
    label_len: int = 576,
    use_ims: bool = False,
    pool_over_vars: bool = True,
) -> dict:
    """
    收集真值序列的激活
    
    Args:
        model: Timer 模型
        batch_y: [B, L_y, M] 真值序列（含历史和未来）
        patch_len: patch 长度
        stride: patch 步长
        collect_layers: 要收集的层索引
        pred_len: 预测长度
        label_len: 标签长度
        use_ims: 是否使用 IMS 模式
        pool_over_vars: 是否跨变量池化
    
    Returns:
        acts: {
            layer_idx: torch.Tensor [N, D] 或 [n_vars, N, D]
        }
    """
    device = batch_y.device
    B, L_y, M = batch_y.shape
    
    # 预处理
    y = preprocess_input(batch_y)
    
    # 提取未来部分（Timer 的目标）
    if use_ims:
        # IMS 模式：未来从 label_len 开始
        y_future = y[:, label_len:label_len + pred_len, :]
    else:
        # 标准模式：未来从末尾取 pred_len
        y_future = y[:, -pred_len:, :]
    
    # 转换为 patch embedding 格式
    y2 = y_future.permute(0, 2, 1)  # [B, M, L_future]
    
    # 获取 model core
    core = model.module if hasattr(model, "module") else model
    
    # Patch embedding
    dec_in, n_vars = core.patch_embedding(y2)  # [B*M, N, D] 或类似
    
    # Reshape
    if len(dec_in.shape) == 3:
        BM, N, D = dec_in.shape
        dec_in = dec_in.view(B, M, N, D)  # [B, M, N, D]
    
    _, n_vars_real, N, D = dec_in.shape
    
    # 跨变量池化
    if pool_over_vars:
        dec_in_pooled = dec_in.mean(dim=1)  # [B, N, D]
    else:
        dec_in_pooled = dec_in  # [B, n_vars, N, D]
    
    # 创建因果 mask
    BM = B * n_vars_real
    mask = TriangularCausalMask(BM, N, device=device)
    
    # 收集各层激活
    layers_outputs = []
    
    with torch.no_grad():
        if hasattr(core.decoder, 'attn_layers'):
            # Encoder-style decoder
            h = dec_in.view(B * n_vars_real, N, D)
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
            # Decoder-style
            h = dec_in_pooled
            for i, layer in enumerate(core.decoder.layers):
                if collect_layers is not None and i not in collect_layers:
                    continue
                h, _ = layer(h, h, attn_mask=mask)
                layers_outputs.append(h)
    
    # 组织输出
    layer_indices = []
    if collect_layers is None:
        layer_indices = list(range(len(layers_outputs)))
    else:
        layer_indices = collect_layers
    
    acts = {}
    for idx, layer_idx in enumerate(layer_indices):
        h = layers_outputs[idx]
        # h: [B, N, D] 或 [B, n_vars, N, D]
        if len(h.shape) == 3:
            # 取第一个样本作为代表
            acts[layer_idx] = h[0].cpu()  # [N, D]
        elif len(h.shape) == 4:
            acts[layer_idx] = h[0].cpu()  # [n_vars, N, D]
    
    return acts


def load_model_and_data(args):
    """加载模型和数据"""
    from models.Timer import Model
    
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
    
    for arg in ['root_path', 'data_path', 'data', 'seq_len', 'label_len', 'pred_len',
                'patch_len', 'd_model', 'd_ff', 'e_layers', 'd_layers', 'n_heads',
                'factor', 'dropout', 'activation', 'embed', 'freq', 'features',
                'stride', 'num_workers', 'batch_size']:
        if hasattr(args, arg):
            setattr(configs, arg, getattr(args, arg))
    
    if hasattr(args, 'patch_len'):
        configs.patch_len = args.patch_len
    if hasattr(args, 'stride') and args.stride > 0:
        configs.patch_len = args.stride
    
    device = torch.device(args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    model.eval()
    
    _, loader = data_provider(configs, flag='test')
    
    return model, loader, configs


def main():
    parser = argparse.ArgumentParser(description="生成 Timer 真值序列的激活")
    
    # 数据参数
    parser.add_argument("--root_path", type=str, default="./datasets/", help="数据集根目录")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="数据文件名")
    parser.add_argument("--data", type=str, default="ETTh1", help="数据集名称")
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
    parser.add_argument("--use_ims", action='store_true', default=False,
                       help="是否使用 IMS 模式")
    
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
    os.makedirs(os.path.join(args.output_dir, "gt"), exist_ok=True)
    
    # 解析层索引
    if args.layers is not None:
        layers = [int(l) for l in args.layers]
    else:
        layers = None
    
    # 收集激活
    all_acts = {}
    sample_id = 0
    max_samples = args.sample_num if args.sample_num > 0 else len(loader.dataset)
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(loader, desc="收集真值激活"):
        if sample_id >= max_samples:
            break
        
        batch_y = batch_y.float().to(device)
        
        # 收集真值激活
        acts = collect_gt_activations(
            model=model,
            batch_y=batch_y,
            patch_len=configs.patch_len,
            stride=configs.stride,
            collect_layers=layers,
            pred_len=configs.pred_len,
            label_len=configs.label_len,
            use_ims=args.use_ims,
            pool_over_vars=args.pool_over_vars,
        )
        
        # 存储
        for b in range(batch_y.shape[0]):
            if sample_id >= max_samples:
                break
            
            all_acts[sample_id] = acts  # 所有层共享同一个 acts（因为是同一个真值）
            sample_id += 1
    
    # 保存
    output_file = os.path.join(
        args.output_dir, 
        "gt", 
        f"{args.data}_{args.model_id}.pth"
    )
    torch.save(all_acts, output_file)
    print(f"Saved GT activations to {output_file}")
    print(f"Total samples: {len(all_acts)}")


if __name__ == '__main__':
    main()
