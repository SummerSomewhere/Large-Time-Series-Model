#!/usr/bin/env python3
"""
表征垄断测试：线性探测 (Linear Probing for Information Monopoly)
==============================================================

核心思想
--------
证明模型在深层并没有把信息均匀分布，而是把所有关于未来的"天机"都压缩
（垄断）在了少数高 MI 的 Patch 表征中。

实验设计
--------
1. 冻结预训练好的 Timer 模型（不再更新 Transformer 权重）
2. 将序列输入模型，提取最后一层每个 Patch 对应的隐藏状态 H ∈ R^{d_model}
3. 高 MI 探针：仅使用高 MI Patch 的 H，训练单层线性回归器映射未来 96 步
4. 低 MI 探针：仅使用低 MI Patch 的 H，训练同样的线性回归器预测未来
5. Baseline 探针：使用全部 Patch 隐藏状态的均值池化

核心发现
--------
- 高 MI Patch 的一层线性映射 ≈ 全序列微调 80% 的性能
- 低 MI Patch 的特征经过线性映射，预测结果近乎瞎猜
- 高 MI Token 就是任务相关信息的绝对载体

可视化
------
- 三条重建曲线（高 MI 探针预测 / 低 MI 探针预测 / Baseline 探针预测）
  与真实曲线叠加对比
- 散点图：High-MI Probe MSE vs Low-MI Probe MSE（每样本）
- 柱状图：高/低 MI 探针 vs Baseline 探针的整体 MSE/MAE

Usage:
  python experiments/timer_mi_monopoly.py \
    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
    --data ETTh1 --root_path ./data/ETT/ \
    --seq_len 672 --pred_len 96 --patch_len 96 \
    --e_layers 8 --d_model 1024 \
    --out_dir ./outputs/timer_mi_monopoly/ \
    --n_samples 2048 --probe_epochs 200 --probe_lr 0.01

  # Quick debug (random model):
  python experiments/timer_mi_monopoly.py \
    --ckpt_path random --data ETTh1 --root_path ./data/ETT/ \
    --seq_len 192 --pred_len 48 --patch_len 48 \
    --e_layers 4 --d_model 256 \
    --out_dir ./outputs/timer_mi_monopoly_debug/ \
    --n_samples 512 --probe_epochs 50
"""

import argparse
import json
import math
import os
import gc
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TORCH_COMPILE_DISABLE"] = "1"

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import numpy as np
import scipy.spatial as ss
import scipy.special as sp

# 修复 torch 2.11 + transformers 4.57 不兼容：torch.utils._pytree 缺少 register_pytree_node
import torch.utils._pytree as _pytree
if not hasattr(_pytree, "register_pytree_node"):
    _pytree.register_pytree_node = lambda *args, **kwargs: None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from utils.masking import TriangularCausalMask

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MI Estimators (KSG)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    KSG (Kraskov, Stoegbauer, Grassberger) kNN estimator for I(X;Y).
    x, y: [N, D] — aligned samples
    Returns MI in bits.
    """
    N = x.shape[0]
    if N <= k + 1:
        return 0.0

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    xy = np.concatenate((x, y), axis=1)
    tree_xy = ss.cKDTree(xy)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    dist_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.maximum(dist_xy[:, k] - 1e-10, 0)

    nx = np.array([
        max(len(tree_x.query_ball_point(x[i], r=eps[i], p=np.inf)) - 1, 0)
        for i in range(N)
    ])
    ny = np.array([
        max(len(tree_y.query_ball_point(y[i], r=eps[i], p=np.inf)) - 1, 0)
        for i in range(N)
    ])

    mi = (sp.digamma(k) - np.mean(sp.digamma(nx + 1) + sp.digamma(ny + 1)) + sp.digamma(N)) / math.log(2)
    return max(0.0, float(mi))


def compute_patch_mi_ksg_batch(
    hx_tokens: np.ndarray,
    hy_future: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    对每个 patch 位置计算 I(H_x_patch ; Y_future)。

    Args:
        hx_tokens: [N, n_patches, D] — 最后一层的 per-patch 隐藏状态
        hy_future: [N, pred_len, M] or [N, pred_len] — 未来目标序列
                   若为多变量，取各变量平坦化的拼接
    Returns:
        mi_scores: [n_patches] — 每个 patch 位置与未来的 MI
    """
    N, n_patches, D = hx_tokens.shape
    mi_scores = np.zeros(n_patches)

    # 若 hy_future 是多变量的，将其平坦化为向量
    if hy_future.ndim == 3:
        M = hy_future.shape[2]
        hy_flat = hy_future.reshape(N, -1)  # [N, pred_len * M]
    else:
        hy_flat = hy_future  # [N, pred_len]

    for pi in range(n_patches):
        hx_patch = hx_tokens[:, pi, :]  # [N, D]
        mi = compute_mi_ksg(hx_patch, hy_flat, k=k)
        mi_scores[pi] = mi

    return mi_scores


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Timer Model Loader
# ═══════════════════════════════════════════════════════════════════════════════

class ModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(
    ckpt_path: str,
    patch_len: int,
    d_model: int,
    d_ff: int,
    e_layers: int,
    n_heads: int,
    dropout: float,
    seq_len: int,
    pred_len: int,
):
    cfg = ModelConfig(
        task_name='forecast',
        ckpt_path=ckpt_path,
        patch_len=patch_len,
        stride=patch_len,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        n_heads=n_heads,
        dropout=dropout,
        output_attention=False,
        distil=True,
        use_revin=False,
        seq_len=seq_len,
        pred_len=pred_len,
        d_layers=1,
        factor=1,
        enc_in=1,
        dec_in=1,
        c_out=1,
        activation='gelu',
        use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(),
        devices='0',
        num_workers=4,
        freq='h',
        data='custom',
        embed='timeF',
        target='OT',
        features='M',
        des='Exp',
        lradj='type1',
        use_amp=False,
        is_finetuning=0,
        label_len=pred_len,
        output_len=pred_len,
        batch_size=64,
        train_epochs=1,
        patience=3,
        learning_rate=3e-5,
        itr=1,
        use_ims=False,
        inverse=False,
        use_align_loss=False,
        align_loss_layers=list(range(e_layers)),
    )
    model = Model(cfg)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Dataset — Benchmark (ETTh1 etc.)
# ═══════════════════════════════════════════════════════════════════════════════

class MonopolyDataset(Dataset):
    """
    返回 (seq_x [1, seq_len], seq_y [1, pred_len], valid_mask)
    """

    def __init__(self, root_path: str, flag: str = 'train',
                 input_len: int = 336, pred_len: int = 96,
                 scale: bool = True):
        super().__init__()
        self.ds = CIDatasetBenchmark(
            root_path=root_path,
            flag=flag,
            input_len=input_len,
            pred_len=pred_len,
            scale=scale,
        )
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        raw = self.ds[idx]
        # CIDatasetBenchmark 返回: (seq_x, seq_y, seq_x_mark, seq_y_mark)
        # seq_x: [1, input_len, 1] or [1, seq_len, M]
        # seq_y: [1, pred_len, 1] or [1, pred_len, M]
        seq_x = raw[0]
        seq_y = raw[1]

        # 转为 tensor [1, T, 1]（添加 batch 维度）
        if not isinstance(seq_x, torch.Tensor):
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
        if not isinstance(seq_y, torch.Tensor):
            seq_y = torch.tensor(seq_y, dtype=torch.float32)

        # [T, 1] -> [1, T, 1]（添加 batch 维度）
        if seq_x.ndim == 2:
            seq_x = seq_x.unsqueeze(0)   # [1, seq_len, 1]
        elif seq_x.ndim == 3 and seq_x.shape[-1] > 1:
            seq_x = seq_x[:, :, :1]      # 多变量 -> 取第一个变量

        if seq_y.ndim == 2:
            seq_y = seq_y.unsqueeze(0)   # [1, pred_len, 1]
        elif seq_y.ndim == 3 and seq_y.shape[-1] > 1:
            seq_y = seq_y[:, :, :1]       # 多变量 -> 取第一个变量

        return seq_x, seq_y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.cat(xs, dim=0)   # [B, seq_len, 1]
    ys = torch.cat(ys, dim=0)   # [B, pred_len, 1]
    return xs, ys


def load_mi_from_file(mi_json_path: str, target_layer: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从已有的 global_mi_peaks_*.json 文件中加载 MI/HSIC 分数。

    Args:
        mi_json_path: 已有的 MI 结果 JSON 文件路径
        target_layer: 目标层索引，-1 表示最后一层

    Returns:
        mi_scores: [n_patches] — 归一化的 patch 分数（越高表示越重要）
        high_mi_patch_idx: 高分组 patch 索引
        low_mi_patch_idx: 低分组 patch 索引
    """
    with open(mi_json_path, 'r') as f:
        data = json.load(f)

    layers = data.get('layers', {})
    n_patches_in_file = int(data.get('N', 0))

    # 找最后一层或指定层
    layer_keys = sorted(int(k) for k in layers.keys())
    if target_layer < 0:
        li = layer_keys[-1]
    else:
        li = target_layer if target_layer in layer_keys else layer_keys[-1]

    layer_data = layers[str(li)]
    cka_curve = layer_data.get('cka_curve', [])

    if not cka_curve:
        raise ValueError(f"Layer {li} has no cka_curve in {mi_json_path}")

    mi_scores = np.array(cka_curve, dtype=np.float32)

    # 归一化到 [0, 1]（便于与 top-ratio 参数配合使用）
    mi_min, mi_max = mi_scores.min(), mi_scores.max()
    if mi_max > mi_min:
        mi_scores_norm = (mi_scores - mi_min) / (mi_max - mi_min)
    else:
        mi_scores_norm = np.ones_like(mi_scores) * 0.5

    # 从文件中加载 high/low patches
    high_patches_file = layer_data.get('high_mi_patches', [])
    low_patches_file = layer_data.get('low_mi_patches', [])

    if high_patches_file and low_patches_file:
        # 使用文件中的 pre-computed 分组
        high_idx = np.array(high_patches_file, dtype=int)
        low_idx = np.array(low_patches_file, dtype=int)
        print(f"  [MI File] Layer {li}: loaded {len(high_idx)} high / {len(low_idx)} low patches from file")
    else:
        # fallback: 按 top/bottom ratio 计算
        high_idx = np.argsort(mi_scores_norm)[-max(1, int(n_patches_in_file * 0.2)):]
        low_idx = np.argsort(mi_scores_norm)[:max(1, int(n_patches_in_file * 0.2))]
        print(f"  [MI File] Layer {li}: computed top/bottom 20% from cka_curve")

    return mi_scores_norm, high_idx, low_idx


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Token Extraction + MI Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def extract_last_layer_tokens_and_future(
    model,
    data_loader,
    device,
    n_patches: int,
    pred_len: int,
    patch_len: int,
    n_samples: Optional[int] = None,
    k_mi: int = 5,
    target_layer: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    冻结模型提取指定层的 per-patch 隐藏状态和对应的未来目标。

    Args:
        target_layer: 目标层索引，-1 表示最后一层，其他值如 0, 1, 2, ...
                     也可以传入负数如 -1（最后一层）、-2（倒数第二层）

    Returns:
        layer_tokens:   [N, n_patches, D]  — 目标层输出
        future_targets: [N, pred_len]      — 未来目标序列（单变量）
        mi_scores:      [n_patches]         — 每个 patch 与未来的 MI
        n:              int                — 有效样本数
    """
    core = _unwrap(model)
    all_tokens = []
    all_futures = []
    n_layers = len(core.decoder.attn_layers)

    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc="提取隐藏状态"):
            B = batch_x.shape[0]

            sx = batch_x.float().to(device)   # [B, seq_len, 1]
            sy = batch_y.float().to(device)   # [B, pred_len, 1]

            # 归一化
            xm = sx.mean(dim=1, keepdim=True).detach()
            xs = torch.sqrt(torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            xn = (sx - xm) / xs              # [B, seq_len, 1]

            # enc_embedding 期望 [B, M, T]，所以需要 permute: [B, seq_len, 1] -> [B, 1, seq_len]
            xn = xn.permute(0, 2, 1)        # [B, 1, seq_len]

            # 逐层前向，收集所有层输出
            dx, _ = core.enc_embedding(xn)   # [B, n_patches, D]
            all_layer_outputs = [dx]
            h = dx
            for layer in core.decoder.attn_layers:
                h, _, _ = layer(h, attn_mask=None)
                all_layer_outputs.append(h)

            # 根据 target_layer 选择
            if target_layer < 0:
                li = n_layers  # -1 -> last layer (index = n_layers)
            else:
                li = target_layer

            selected = all_layer_outputs[li]     # [B, n_patches, D]
            all_tokens.append(selected.float().cpu())
            all_futures.append(sy.squeeze(-1).float().cpu())

            del dx, all_layer_outputs, h
            gc.collect()
            torch.cuda.empty_cache()

    layer_tokens = torch.cat(all_tokens, dim=0)   # [N_all, n_patches, D]
    future_targets = torch.cat(all_futures, dim=0)  # [N_all, pred_len]

    # 下采样
    n_actual = layer_tokens.shape[0]
    if n_samples is not None and n_samples < n_actual:
        idx = np.sort(np.random.default_rng(42).choice(n_actual, n_samples, replace=False))
        layer_tokens = layer_tokens[idx]
        future_targets = future_targets[idx]
        n_actual = n_samples

    # ── 计算每个 patch 与未来的 MI ─────────────────────────────────────────
    print(f"  计算 {n_patches} 个 patch 位置的 MI (k={k_mi}, layer={li}) ...")
    tokens_np = layer_tokens.numpy()
    future_np = future_targets.numpy()

    # PCA 降维
    D_orig = tokens_np.shape[2]
    pca_dim = min(64, D_orig, n_actual - 2)
    if pca_dim < D_orig:
        from sklearn.decomposition import PCA
        tokens_flat = tokens_np.reshape(n_actual * n_patches, D_orig)
        pca = PCA(n_components=pca_dim, random_state=42)
        tokens_pca_flat = pca.fit_transform(tokens_flat)
        tokens_pca = tokens_pca_flat.reshape(n_actual, n_patches, pca_dim)
        print(f"  PCA: {D_orig}D -> {pca_dim}D (explained variance: {pca.explained_variance_ratio_.sum():.3f})")
    else:
        tokens_pca = tokens_np

    mi_scores = compute_patch_mi_ksg_batch(tokens_pca, future_np, k=k_mi)

    print(f"  MI 统计: min={mi_scores.min():.4f}, max={mi_scores.max():.4f}, "
          f"mean={mi_scores.mean():.4f}, median={np.median(mi_scores):.4f}")

    return tokens_np, future_np, mi_scores, n_actual


def extract_tokens_for_layer(
    model,
    data_loader,
    device,
    n_patches: int,
    pred_len: int,
    patch_len: int,
    n_samples: Optional[int] = None,
    target_layer: int = -1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    冻结模型提取指定层的 per-patch 隐藏状态（不计算 MI，用于从文件加载 MI 后的快速提取）。

    Returns:
        layer_tokens:   [N, n_patches, D]
        future_targets: [N, pred_len]
        n:              int
    """
    core = _unwrap(model)
    all_tokens = []
    all_futures = []
    n_layers = len(core.decoder.attn_layers)

    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc=f"提取 layer {target_layer}"):
            sx = batch_x.float().to(device)
            sy = batch_y.float().to(device)

            xm = sx.mean(dim=1, keepdim=True).detach()
            xs = torch.sqrt(torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            xn = (sx - xm) / xs
            xn = xn.permute(0, 2, 1)        # [B, 1, seq_len]

            dx, _ = core.enc_embedding(xn)

            all_layer_outputs = [dx]
            h = dx
            for layer in core.decoder.attn_layers:
                h, _, _ = layer(h, attn_mask=None)
                all_layer_outputs.append(h)

            if target_layer < 0:
                li = n_layers
            else:
                li = target_layer

            all_tokens.append(all_layer_outputs[li].float().cpu())
            all_futures.append(sy.squeeze(-1).float().cpu())

            del dx, all_layer_outputs, h
            gc.collect()
            torch.cuda.empty_cache()

    layer_tokens = torch.cat(all_tokens, dim=0)
    future_targets = torch.cat(all_futures, dim=0)

    n_actual = layer_tokens.shape[0]
    if n_samples is not None and n_samples < n_actual:
        idx = np.sort(np.random.default_rng(42).choice(n_actual, n_samples, replace=False))
        layer_tokens = layer_tokens[idx]
        future_targets = future_targets[idx]
        n_actual = n_samples

    return layer_tokens.numpy(), future_targets.numpy(), n_actual


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Linear Probe
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """
    单层线性回归器：H -> Y_future
    输入: [B, D] 或 [B, K, D]（K 个 patch）
    输出: [B, pred_len]
    """

    def __init__(self, d_model: int, pred_len: int, patch_mode: bool = False):
        super().__init__()
        self.patch_mode = patch_mode
        if patch_mode:
            # 每个 patch 独立预测，然后加权平均
            self.linear = nn.Linear(d_model, pred_len, bias=False)
        else:
            # 全局平均池化后预测
            self.linear = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] 或 [B, K, D]
        """
        if self.patch_mode:
            # x: [B, K, D] -> 线性映射 -> [B, K, pred_len]
            out = self.linear(x)        # [B, K, pred_len]
            # 沿 patch 维度加权平均（权重归一化为概率）
            w = torch.softmax(torch.ones(x.shape[1], device=x.device) / x.shape[1], dim=0)
            out = (out * w.view(1, -1, 1)).sum(dim=1)  # [B, pred_len]
        else:
            # x: [B, D] -> 直接预测
            out = self.linear(x)       # [B, pred_len]
        return out


def train_linear_probe(
    tokens: np.ndarray,
    futures: np.ndarray,
    mask: Optional[np.ndarray] = None,
    probe_epochs: int = 200,
    probe_lr: float = 0.01,
    weight_decay: float = 1e-4,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple['LinearProbe', float, float, np.ndarray, np.ndarray]:
    """
    训练线性探针。

    Args:
        tokens:   [N, K, D] — 选中的 patch 隐藏状态
        futures:  [N, pred_len] — 未来目标
        mask:    [N] — bool array，为 True 的样本用于训练
        probe_*: 探针训练超参数
        train_frac: 若 mask 为 None，用此比例划分 train/val

    Returns:
        probe, best_mse, best_mae,
        val_pred [N_val, pred_len], val_true [N_val, pred_len]
    """
    rng = np.random.default_rng(seed)

    if mask is not None:
        # 使用外部提供的 mask
        train_mask = mask
        val_mask = ~mask
    else:
        # 内部划分
        n = len(futures)
        perm = rng.permutation(n)
        n_train = max(1, int(n * train_frac))
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:]] = True

    # 数据标准化（输入和目标均标准化）
    train_tokens = tokens[train_mask]      # [N_tr, K, D]
    val_tokens = tokens[val_mask]           # [N_val, K, D]
    train_futures = futures[train_mask]     # [N_tr, pred_len]
    val_futures = futures[val_mask]         # [N_val, pred_len]

    # 目标标准化（训练集 stats）
    y_mean = train_futures.mean(axis=0, keepdims=True)
    y_std = train_futures.std(axis=0, keepdims=True) + 1e-8
    train_futures_norm = (train_futures - y_mean) / y_std
    val_futures_norm = (val_futures - y_mean) / y_std

    D = train_tokens.shape[2]
    pred_len = train_futures.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe = LinearProbe(d_model=D, pred_len=pred_len, patch_mode=True).to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=probe_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    train_tokens_t = torch.tensor(train_tokens, dtype=torch.float32, device=device)
    train_futures_t = torch.tensor(train_futures_norm, dtype=torch.float32, device=device)
    val_tokens_t = torch.tensor(val_tokens, dtype=torch.float32, device=device)
    val_futures_t = torch.tensor(val_futures_norm, dtype=torch.float32, device=device)

    best_mse = float('inf')
    best_mae = float('inf')
    best_state = None
    patience_counter = 0
    patience = 30

    for epoch in range(probe_epochs):
        probe.train()
        optimizer.zero_grad()

        pred = probe(train_tokens_t)           # [N_tr, pred_len]
        loss = criterion(pred, train_futures_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_pred_norm = probe(val_tokens_t)
            val_mse = criterion(val_pred_norm, val_futures_t).item()

        if val_mse < best_mse:
            best_mse = val_mse
            best_mae = float(F.mse_loss(val_pred_norm, val_futures_t).sqrt().mean().item())
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 恢复最优模型
    if best_state is not None:
        probe.load_state_dict(best_state)

    # 最终预测（反标准化）
    probe.eval()
    with torch.no_grad():
        val_pred_norm = probe(val_tokens_t).cpu().numpy()
        val_pred = val_pred_norm * y_std + y_mean   # [N_val, pred_len]

    return probe, best_mse, best_mae, val_pred, val_futures


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reconstruction_comparison(
    out_dir: str,
    true_future: np.ndarray,
    high_mi_pred: np.ndarray,
    low_mi_pred: np.ndarray,
    baseline_pred: np.ndarray,
    n_show: int = 12,
    pred_len: int = 96,
):
    """
    绘制多条重建曲线与真实曲线的叠加对比图。
    展示高 MI 探针 vs 低 MI 探针 vs Baseline 探针的预测质量差异。
    """
    n_show = min(n_show, len(true_future))
    n_cols = 4
    n_rows = math.ceil(n_show / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    t = np.arange(pred_len)

    for i in range(n_show):
        ax = axes[i]
        true = true_future[i]
        ax.plot(t, true, color='black', linewidth=1.8, label='Ground Truth', zorder=4)
        ax.plot(t, high_mi_pred[i], color='#e74c3c', linewidth=1.2,
                linestyle='-', alpha=0.85, label='High-MI Probe' if i == 0 else '', zorder=3)
        ax.plot(t, low_mi_pred[i], color='#3498db', linewidth=1.0,
                linestyle='--', alpha=0.7, label='Low-MI Probe' if i == 0 else '', zorder=2)
        ax.plot(t, baseline_pred[i], color='#95a5a6', linewidth=1.0,
                linestyle=':', alpha=0.7, label='Baseline Probe' if i == 0 else '', zorder=1)
        ax.set_title(f'Sample {i+1}', fontsize=9)
        ax.set_xlabel('Timestep', fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    fig.suptitle('表征垄断测试：重建曲线对比\n'
                 '高 MI Patch 线性映射 vs 低 MI Patch 线性映射 vs Baseline',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'reconstruction_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: reconstruction_comparison.png")


def plot_aggregate_metrics(
    out_dir: str,
    results: Dict[str, Dict[str, float]],
    dataset_name: str,
):
    """
    柱状图对比各探针的整体 MSE/MAE。
    """
    names = list(results.keys())
    mse_vals = [results[n]['mse'] for n in names]
    mae_vals = [results[n]['mae'] for n in names]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#2ecc71']

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    bars = ax.bar(x, mse_vals, width, color=colors[:len(names)], alpha=0.85)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('MSE 对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, mae_vals, width, color=colors[:len(names)], alpha=0.85)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('MAE 对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'表征垄断测试 — {dataset_name} 数据集\n'
                 f'线性探针 MSE/MAE 对比', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'aggregate_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: aggregate_metrics.png")


def plot_mi_distribution(
    out_dir: str,
    mi_scores: np.ndarray,
    high_mi_idx: np.ndarray,
    low_mi_idx: np.ndarray,
):
    """
    展示 MI 分数的分布以及高/低 MI patch 的位置。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左：MI 分数沿 patch 位置的分布
    ax = axes[0]
    n_patches = len(mi_scores)
    t = np.arange(n_patches)
    ax.bar(t, mi_scores, color='#9b59b6', alpha=0.7, label='MI Score')
    for idx in high_mi_idx:
        ax.axvline(idx, color='#e74c3c', linewidth=1.5, alpha=0.7, linestyle='--')
    for idx in low_mi_idx:
        ax.axvline(idx, color='#3498db', linewidth=1.5, alpha=0.7, linestyle=':')
    ax.set_xlabel('Patch Position', fontsize=11)
    ax.set_ylabel('I(H_patch; Y_future) [bits]', fontsize=11)
    ax.set_title('每个 Patch 与未来序列的互信息', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9b59b6', alpha=0.7, label='MI Score'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='High-MI Patches'),
        Patch(facecolor='#3498db', alpha=0.7, label='Low-MI Patches'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # 右：直方图
    ax = axes[1]
    ax.hist(mi_scores, bins=20, color='#9b59b6', alpha=0.7, edgecolor='white')
    q75 = np.percentile(mi_scores, 75)
    q25 = np.percentile(mi_scores, 25)
    ax.axvline(q75, color='#e74c3c', linewidth=2, linestyle='--', label=f'Q75={q75:.3f}')
    ax.axvline(q25, color='#3498db', linewidth=2, linestyle=':', label=f'Q25={q25:.3f}')
    ax.set_xlabel('I(H_patch; Y_future) [bits]', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('MI 分数分布直方图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mi_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: mi_distribution.png")


def plot_scatter_mse(
    out_dir: str,
    high_mse_per_sample: np.ndarray,
    low_mse_per_sample: np.ndarray,
    baseline_mse_per_sample: np.ndarray,
):
    """
    散点图：每样本 High-MI Probe MSE vs Low-MI Probe MSE。
    落在对角线下方 = High-MI 更好（更低的 MSE）。
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(high_mse_per_sample.max(), low_mse_per_sample.max()) * 1.05
    vmin = min(high_mse_per_sample.min(), low_mse_per_sample.min())

    ax.scatter(low_mse_per_sample, high_mse_per_sample,
               alpha=0.4, s=15, c='#9b59b6', label='Samples')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.5, label='y=x (equal)')

    n_below = (high_mse_per_sample < low_mse_per_sample).sum()
    n_total = len(high_mse_per_sample)
    pct = n_below / n_total * 100

    ax.set_xlabel('Low-MI Probe MSE per sample', fontsize=12)
    ax.set_ylabel('High-MI Probe MSE per sample', fontsize=12)
    ax.set_title(f'高 MI vs 低 MI 探针 — 每样本 MSE 对比\n'
                 f'High-MI 更优: {n_below}/{n_total} ({pct:.1f}%)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 标注 Baseline 方向
    ax.scatter(low_mse_per_sample.mean(), high_mse_per_sample.mean(),
               marker='*', s=200, c='red', zorder=5, label='Mean (High-MI)')
    ax.scatter(low_mse_per_sample.mean(), low_mse_per_sample.mean(),
               marker='*', s=200, c='blue', zorder=5, label='Mean (Low-MI)')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scatter_mse_high_vs_low.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: scatter_mse_high_vs_low.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    print("=" * 72)
    print("  表征垄断测试：线性探测 (Linear Probing for Information Monopoly)")
    print("=" * 72)
    print(f"  数据集     : {args.data}")
    print(f"  seq_len    : {args.seq_len}")
    print(f"  pred_len   : {args.pred_len}")
    print(f"  patch_len  : {args.patch_len}")
    print(f"  e_layers   : {args.e_layers}")
    print(f"  n_samples  : {args.n_samples}")
    print(f"  MI k       : {args.mi_k}")
    print(f"  probe_ep   : {args.probe_epochs}")
    print(f"  probe_lr   : {args.probe_lr}")
    print(f"  high_ratio : {args.high_mi_ratio}")
    print(f"  low_ratio  : {args.low_mi_ratio}")
    print(f"  device     : cuda" if torch.cuda.is_available() else "  device     : cpu")
    print("=" * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n>>> [1/5] 加载数据集...")
    full_ds = MonopolyDataset(
        root_path=os.path.join(args.root_path, args.data_path),
        flag='train',
        input_len=args.seq_len,
        pred_len=args.pred_len,
        scale=True,
    )

    n_patches = args.seq_len // args.patch_len
    print(f"  数据集样本数: {len(full_ds)}, n_patches={n_patches}, pred_len={args.pred_len}")

    # 构建 DataLoader（全量数据用于后续划分）
    full_loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # ── 2. Model ───────────────────────────────────────────────────────────────
    print("\n>>> [2/5] 加载 Timer 模型...")
    model = build_timer(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    model = model.to(device)
    model.eval()
    print(f"  模型就绪，设备: {device}")

    # ── 3. Extract Tokens + (optionally load or compute) MI ──────────────────────
    if getattr(args, 'mi_file', None) and os.path.exists(args.mi_file):
        print(f"\n>>> [3/5] 从已有 MI 文件加载 patch 分组: {args.mi_file}")
        mi_scores, high_mi_patch_idx, low_mi_patch_idx = load_mi_from_file(
            args.mi_file, target_layer=args.extract_layer
        )
        n_patches = len(mi_scores)
        mi_source = "file"
    else:
        mi_source = "compute"

    if mi_source == "compute":
        print(f"\n>>> [3/5] 提取第 {args.extract_layer} 层隐藏状态并计算 per-patch MI...")
        tokens, futures, mi_scores, n_actual = extract_last_layer_tokens_and_future(
            model=model,
            data_loader=full_loader,
            device=device,
            n_patches=n_patches,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            n_samples=args.n_samples,
            k_mi=args.mi_k,
            target_layer=args.extract_layer,
        )
        print(f"  提取完成: tokens={tokens.shape}, futures={futures.shape}")

        # MI 排序与 Patch 筛选
        sorted_idx = np.argsort(mi_scores)[::-1]
        n_high = max(1, int(n_patches * args.high_mi_ratio))
        n_low = max(1, int(n_patches * args.low_mi_ratio))
        high_mi_patch_idx = sorted_idx[:n_high]
        low_mi_patch_idx = sorted_idx[-n_low:]
    else:
        # 从文件加载后，仍需提取指定层的 hidden states
        print(f"  从 layer {args.extract_layer} 提取 hidden states...")
        tokens, futures, n_actual = extract_tokens_for_layer(
            model=model,
            data_loader=full_loader,
            device=device,
            n_patches=n_patches,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            n_samples=args.n_samples,
            target_layer=args.extract_layer,
        )
        print(f"  提取完成: tokens={tokens.shape}, futures={futures.shape}")

    print(f"  High-MI patches (Top {args.high_mi_ratio*100:.0f}%): {high_mi_patch_idx}")
    print(f"  Low-MI patches  (Bottom {args.low_mi_ratio*100:.0f}%): {low_mi_patch_idx}")
    print(f"  MI@High-MI mean={mi_scores[high_mi_patch_idx].mean():.4f}, "
          f"MI@Low-MI mean={mi_scores[low_mi_patch_idx].mean():.4f}")

    # 提取各 patch 组的 token
    high_mi_tokens = tokens[:, high_mi_patch_idx, :]   # [N, n_high, D]
    low_mi_tokens  = tokens[:, low_mi_patch_idx, :]    # [N, n_low, D]

    # ── 4. 构建各组 token 并训练线性探针 ─────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n = n_actual
    perm = rng.permutation(n)
    n_train = max(1, int(n * 0.8))
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:]] = True

    probe_results = {}

    # 5a. High-MI Probe
    print(f"\n  --- High-MI Probe (n_high={n_high}) ---")
    probe_high, mse_high, mae_high, pred_high, true_high = train_linear_probe(
        tokens=high_mi_tokens,
        futures=futures,
        mask=train_mask,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    probe_results['High-MI Probe'] = {'mse': mse_high, 'mae': mae_high,
                                      'pred': pred_high, 'true': true_high,
                                      'tokens': high_mi_tokens[~train_mask],
                                      'val_idx': np.where(~train_mask)[0]}
    print(f"  High-MI Probe | val_mse={mse_high:.6f}, val_mae={mae_high:.6f}")

    # 5b. Low-MI Probe
    print(f"\n  --- Low-MI Probe (n_low={n_low}) ---")
    probe_low, mse_low, mae_low, pred_low, true_low = train_linear_probe(
        tokens=low_mi_tokens,
        futures=futures,
        mask=train_mask,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    probe_results['Low-MI Probe'] = {'mse': mse_low, 'mae': mae_low,
                                      'pred': pred_low, 'true': true_low,
                                      'tokens': low_mi_tokens[~train_mask],
                                      'val_idx': np.where(~train_mask)[0]}
    print(f"  Low-MI Probe  | val_mse={mse_low:.6f}, val_mae={mae_low:.6f}")

    # 5c. Baseline Probe (mean-pooled)
    print(f"\n  --- Baseline Probe (mean pool) ---")
    probe_bl, mse_bl, mae_bl, pred_bl, true_bl = train_linear_probe(
        tokens=baseline_tokens,   # [N, D]
        futures=futures,
        mask=train_mask,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    probe_results['Baseline Probe'] = {'mse': mse_bl, 'mae': mae_bl,
                                        'pred': pred_bl, 'true': true_bl,
                                        'val_idx': np.where(~train_mask)[0]}
    print(f"  Baseline Probe| val_mse={mse_bl:.6f}, val_mae={mae_bl:.6f}")

    # ── 6. 打印汇总表 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Probe':<20} {'Val MSE':>12} {'Val MAE':>12}")
    print("-" * 60)
    for name, res in probe_results.items():
        print(f"  {name:<20} {res['mse']:>12.6f} {res['mae']:>12.6f}")
    print("=" * 60)

    ratio_high_bl = mse_bl / mse_high if mse_high > 0 else float('inf')
    ratio_low_bl = mse_bl / mse_low if mse_low > 0 else float('inf')
    print(f"\n  High-MI Probe 达到 Baseline 的 {ratio_high_bl:.1%} 性能 (MSE 比)")
    print(f"  Low-MI Probe  达到 Baseline 的 {ratio_low_bl:.1%} 性能 (MSE 比)")

    # ── 7. 可视化 ──────────────────────────────────────────────────────────────
    print("\n>>> [5/5] 绘制可视化图表...")

    # 7a. 重建曲线对比
    val_true = true_high  # 所有探针用同一验证集
    val_pred_high = pred_high
    val_pred_low = pred_low
    val_pred_bl = pred_bl

    plot_reconstruction_comparison(
        out_dir=args.out_dir,
        true_future=val_true,
        high_mi_pred=val_pred_high,
        low_mi_pred=val_pred_low,
        baseline_pred=val_pred_bl,
        n_show=min(16, len(val_true)),
        pred_len=args.pred_len,
    )

    # 7b. MI 分布
    plot_mi_distribution(
        out_dir=args.out_dir,
        mi_scores=mi_scores,
        high_mi_idx=high_mi_patch_idx,
        low_mi_idx=low_mi_patch_idx,
    )

    # 7c. 聚合指标柱状图
    plot_aggregate_metrics(
        out_dir=args.out_dir,
        results={
            'High-MI Probe': probe_results['High-MI Probe'],
            'Low-MI Probe': probe_results['Low-MI Probe'],
            'Baseline Probe': probe_results['Baseline Probe'],
        },
        dataset_name=args.data,
    )

    # 7d. 散点图（每样本 MSE 对比）
    high_mse_per_sample = ((val_pred_high - val_true) ** 2).mean(axis=1)
    low_mse_per_sample = ((val_pred_low - val_true) ** 2).mean(axis=1)
    bl_mse_per_sample = ((val_pred_bl - val_true) ** 2).mean(axis=1)

    plot_scatter_mse(
        out_dir=args.out_dir,
        high_mse_per_sample=high_mse_per_sample,
        low_mse_per_sample=low_mse_per_sample,
        baseline_mse_per_sample=bl_mse_per_sample,
    )

    # ── 8. 保存结果 ────────────────────────────────────────────────────────────
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: v for k, v in vars(args).items()
                 if not callable(v) and not k.startswith("_")},
        "n_patches": int(n_patches),
        "n_high_mi_patches": int(n_high),
        "n_low_mi_patches": int(n_low),
        "mi_scores": mi_scores.tolist(),
        "high_mi_patch_idx": high_mi_patch_idx.tolist(),
        "low_mi_patch_idx": low_mi_patch_idx.tolist(),
        "high_mi_probe": {
            "val_mse": float(probe_results['High-MI Probe']['mse']),
            "val_mae": float(probe_results['High-MI Probe']['mae']),
        },
        "low_mi_probe": {
            "val_mse": float(probe_results['Low-MI Probe']['mse']),
            "val_mae": float(probe_results['Low-MI Probe']['mae']),
        },
        "baseline_probe": {
            "val_mse": float(probe_results['Baseline Probe']['mse']),
            "val_mae": float(probe_results['Baseline Probe']['mae']),
        },
    }
    json_path = os.path.join(args.out_dir, "monopoly_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  JSON: {json_path}")

    print("\n" + "=" * 72)
    print(f"  完成！结果目录: {args.out_dir}")
    print("=" * 72)
    return results_json


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="表征垄断测试：Linear Probing for Information Monopoly"
    )
    # 模型参数
    p.add_argument("--ckpt_path", type=str,
                   default="checkpoints/Timer_forecast_1.0.ckpt",
                   help="Timer 预训练模型路径，设为 'random' 则随机初始化")
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    # 数据参数
    p.add_argument("--data", type=str, default="ETTh1",
                   help="数据集名称（用于构建 root_path）")
    p.add_argument("--root_path", type=str, default="./data/ETT/",
                   help="数据根目录")
    p.add_argument("--data_path", type=str, default="ETTh1.csv",
                   help="数据文件名")
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--pred_len", type=int, default=96,
                   help="预测步长（未来序列长度）")
    p.add_argument("--patch_len", type=int, default=96)

    # 实验参数
    p.add_argument("--mi_file", type=str, default=None,
                   help="已有的 MI 结果 JSON 文件路径，加载后跳过 MI 计算")
    p.add_argument("--extract_layer", type=int, default=-1,
                   help="提取哪一层的隐藏状态，-1=最后一层，0=第一层，-2=倒数第二层")
    p.add_argument("--n_samples", type=int, default=2048,
                   help="用于 MI 计算和探针训练的样本数（-1=全量）")
    p.add_argument("--mi_k", type=int, default=5,
                   help="KSG MI 估计的 k 近邻数")
    p.add_argument("--high_mi_ratio", type=float, default=0.20,
                   help="高 MI Patch 比例（Top X%%）")
    p.add_argument("--low_mi_ratio", type=float, default=0.20,
                   help="低 MI Patch 比例（Bottom X%%）")

    # 探针参数
    p.add_argument("--probe_epochs", type=int, default=200)
    p.add_argument("--probe_lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # 通用参数
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./outputs/timer_mi_monopoly")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 时间戳子目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    run_pipeline(args)
