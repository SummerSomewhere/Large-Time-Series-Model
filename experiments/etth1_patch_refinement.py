#!/usr/bin/env python3
"""
ETTh1 + Timer: 两阶段 Patch Refinement 对比实验

实验设计：
- 对照组（baseline）：标准 Timer 推理（无 refinement）
- 实验组（two-stage）：两阶段推理
  - 第一阶段：标准推理，收集指定层的隐藏状态
  - 第二阶段：指定层的指定 token/patch 使用第一阶段指定层的输出重新初始化

参数说明：
- --refine_layer: 要替换的层索引（支持负数，如 -1 表示最后一层，-2 表示倒数第二层）
- --refine_token_idx: 要替换的 token/patch 索引（逗号分隔，如 "5,6" 表示替换第5和第6个 token）
- --use_layer_output: 第一阶段提取哪一层的输出用于替换（负数表示从后往前数）

参考：
- RR_model.py: 使用 hook 机制提取和注入隐藏状态
- MI-Peaks: 使用 HSIC 计算 token 级别的 MI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model


# ── HSIC MI 估计器 ───────────────────────────────────────────────────────────

def distmat(X):
    """计算距离矩阵"""
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def sigma_estimation(X, Y):
    """从中位数距离估计 sigma"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def kernelmat(X, sigma, ktype='gaussian'):
    """核矩阵计算"""
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)
        else:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)
    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(Kx, H)
    return Kxc


def hsic_normalized_cca(x, y, sigma=50., ktype='gaussian'):
    """HSIC 归一化 CCA 估计"""
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma, ktype=ktype)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def estimate_mi_hsic(x, y, ktype='gaussian', sigma=50.):
    """估计互信息（HSIC 方法）"""
    return hsic_normalized_cca(x, y, ktype=ktype, sigma=sigma)


# ── 预计算 MI Peaks 文件加载 ─────────────────────────────────────────────────


def load_global_mi_peaks(file_path: str) -> dict:
    """从 JSON 文件加载预计算的全局 MI peaks 数据。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MI peaks file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_layer_patches_from_peaks(
    peaks_data: dict,
    layer_idx: int,
    mode: str = "high",
) -> list[int]:
    """
    从预计算的 peaks 数据中获取指定层的高/低 MI patch 索引列表。

    Args:
        peaks_data: load_global_mi_peaks 返回的字典
        layer_idx: 层索引（支持负数，会自动转换）
        mode: "high" 返回高 MI patch，"low" 返回低 MI patch

    Returns:
        patch 索引列表
    """
    num_layers = peaks_data["num_layers"]
    # 负数索引转换
    actual_layer = num_layers + layer_idx if layer_idx < 0 else layer_idx
    actual_layer = max(0, min(actual_layer, num_layers - 1))

    layer_key = str(actual_layer)
    if layer_key not in peaks_data["layers"]:
        raise ValueError(f"Layer {layer_key} not found in peaks file. Available: {list(peaks_data['layers'].keys())}")

    if mode == "high":
        return peaks_data["layers"][layer_key]["high_mi_patches"]
    elif mode == "low":
        return peaks_data["layers"][layer_key]["low_mi_patches"]
    else:
        raise ValueError(f"mode must be 'high' or 'low', got {mode}")


def get_global_patches_from_peaks(
    peaks_data: dict,
    mode: str = "high",
) -> list[int]:
    """
    从预计算的 peaks 数据中获取所有层的高/低 MI patch 索引（跨层并集）。

    Args:
        peaks_data: load_global_mi_peaks 返回的字典
        mode: "high" 返回所有层高 MI patch 的并集，"low" 返回所有层低 MI patch 的并集

    Returns:
        patch 索引列表（去重排序）
    """
    if mode == "high":
        return peaks_data.get("all_high_mi_patches", [])
    elif mode == "low":
        return peaks_data.get("all_low_mi_patches", [])
    else:
        raise ValueError(f"mode must be 'high' or 'low', got {mode}")


def build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """构建完整的配置 namespace"""
    ns = argparse.Namespace(**vars(args))
    defaults = {
        "task_name": "forecast",
        "is_training": 0,
        "is_finetuning": 0,
        "train_test": 0,
        "use_multi_gpu": False,
        "d_layers": 1,
        "target": "OT",
        "checkpoints": "./checkpoints/",
        "inverse": False,
        "use_amp": False,
        "use_weight_decay": 0,
        "weight_decay": 0.01,
        "loss": "MSE",
        "lradj": "type1",
        "train_epochs": 0,
        "patience": 3,
        "learning_rate": 1e-4,
        "itr": 1,
        "finetune_epochs": 0,
        "output_attention": False,
        "distil": True,
        "model_id": "refinement_exp",
        "model": "Timer",
        "output_len_list": None,
        "mask_rate": 0.25,
        "data_type": "custom",
        "decay_fac": 0.75,
        "cos_warm_up_steps": 100,
        "cos_max_decay_steps": 60000,
        "cos_max_decay_epoch": 10,
        "cos_max": 1e-4,
        "cos_min": 2e-6,
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def forward_collect_single_layer(model, x_enc, target_layer_idx):
    """
    提取 decoder 指定层的隐藏状态

    Args:
        model: Timer 模型
        x_enc: 输入序列 [B, L, M]
        target_layer_idx: 目标层索引（支持负数）

    Returns:
        hidden_states: 指定层的隐藏状态 [B, N, D]
        n_vars: 变量数量
        N: patch 数量
        all_layers: 所有层的隐藏状态列表 [N_layers, B, N, D]
    """
    core = model.module if hasattr(model, "module") else model
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    all_layers = []
    mask = None

    # 遍历所有层并收集
    num_layers = len(core.decoder.attn_layers)
    for i, o3 in enumerate(core.decoder.attn_layers):
        h, _, _ = o3(h, attn_mask=mask)
        all_layers.append(pool(h.detach()))

    # 处理负数索引
    if target_layer_idx < 0:
        actual_layer_idx = num_layers + target_layer_idx
    else:
        actual_layer_idx = target_layer_idx

    actual_layer_idx = max(0, min(actual_layer_idx, num_layers - 1))
    target_hidden = all_layers[actual_layer_idx]

    return target_hidden, n_vars, N, all_layers


# ── 真正的层间替换：Hook + Two-Pass 机制 ─────────────────────────────────────


def forward_two_pass_layer_replacement(
    model, x_enc, y_future, selected_indices, target_layer_idx, replace_layer_idx=None,
    use_layer_output=-2, sigma=50., static_patch_indices=None,
):
    """
    真正的两阶段层间替换前向传播

    核心逻辑：
    - Pass 1: 正常前向，在目标层记录该层的输入和输出（用于替换）
    - Pass 2: 在替换层的**输入**处执行 token 替换，然后该层用替换后的输入继续计算

    替换发生在层输入处，使得替换后的值会经过该层的 Attention + FFN 计算。

    Args:
        model: Timer 模型
        x_enc: 输入序列 [B, L, M]
        y_future: GT 未来序列 [B, pred_len, M]（未使用，保留接口兼容）
        selected_indices: 每个样本选择的 patch 索引 [B, K]（运行时计算模式）
        target_layer_idx: 提取隐藏状态的层（负数支持）
        replace_layer_idx: 执行替换的层（默认为 target_layer_idx）
        use_layer_output: 用于计算 MI 的层（未使用，保留接口兼容）
        sigma: HSIC sigma 参数（未使用，保留接口兼容）
        static_patch_indices: 静态全局 patch 索引列表（预计算模式），优先级高于 selected_indices

    Returns:
        dec_out: 预测输出
        selected_indices: 实际使用的索引
    """
    core = model.module if hasattr(model, "module") else model
    B, L, M = x_enc.shape

    # 标准化
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    num_layers = len(core.decoder.attn_layers)

    # 标准化层索引
    target_layer = num_layers + target_layer_idx if target_layer_idx < 0 else target_layer_idx
    replace_layer = num_layers + replace_layer_idx if replace_layer_idx < 0 else (replace_layer_idx if replace_layer_idx is not None else target_layer)

    # ── Pass 1: 正常前向，记录目标层的输入和输出 ───────────────────────────────
    pass1_layer_input = None   # 目标层的输入
    pass1_layer_output = None  # 目标层的输出

    h = dec_in
    mask = None

    for layer_idx, o3 in enumerate(core.decoder.attn_layers):
        if layer_idx == target_layer:
            # 记录该层的输入
            pass1_layer_input = h.detach().clone()
            # 经过该层
            h, _, _ = o3(h, attn_mask=mask)
            # 记录该层的输出
            pass1_layer_output = h.detach().clone()
        else:
            h, _, _ = o3(h, attn_mask=mask)

    # ── Pass 2: 在替换层的输入处执行 token 替换 ───────────────────────────────
    h = dec_in
    mask = None

    # 确定要替换的 patch 索引
    if static_patch_indices is not None and len(static_patch_indices) > 0:
        # 预计算模式：所有样本使用相同的全局 patch 索引
        C = BM // B
        # [K] -> [B*C, K]
        sel_expanded = torch.tensor(
            static_patch_indices, dtype=torch.long, device=dec_in.device
        ).unsqueeze(0).expand(BM, -1).clone()
        actual_selected = sel_expanded
    elif selected_indices is not None:
        # 运行时计算模式：每个样本有独立的 patch 索引
        C = BM // B
        sel_expanded = selected_indices.repeat_interleave(C, dim=0)  # [B*C, K]
        actual_selected = sel_expanded
    else:
        # 无替换，执行普通推理
        for layer_idx, o3 in enumerate(core.decoder.attn_layers):
            h, _, _ = o3(h, attn_mask=mask)
        dec_out = core.proj(h)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)
        dec_out = dec_out * stdev + means
        return dec_out, selected_indices

    for layer_idx, o3 in enumerate(core.decoder.attn_layers):
        if layer_idx == replace_layer and pass1_layer_output is not None:
            # 在该层输入处执行替换：用目标层输出的对应位置替换当前输入
            h_in = h
            h_modified = h_in.clone()

            # 在层输入处替换指定 patch token
            for b_c in range(BM):
                for k in range(actual_selected.shape[1]):
                    patch_idx = actual_selected[b_c, k].item()
                    if patch_idx < N:
                        # 用 Pass 1 目标层的输出替换当前层的输入
                        h_modified[b_c, patch_idx, :] = pass1_layer_output[b_c, patch_idx, :]

            # 该层使用替换后的输入继续计算（经过 Attention + FFN）
            h, _, _ = o3(h_modified, attn_mask=mask)
        else:
            h, _, _ = o3(h, attn_mask=mask)

    # Projection
    dec_out = core.proj(h)  # [B*M, N, L]
    dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

    # 反标准化
    dec_out = dec_out * stdev + means

    return dec_out, actual_selected if static_patch_indices is not None else selected_indices


def forward_with_layer_replacement(
    model, x_enc, refine_layer_idx, token_indices, replacement_hidden
):
    """
    第二阶段前向传播：替换指定层指定 token 的隐藏状态（废弃，仅作参考）
    """
    pass


def forward_with_hidden_replacement(
    model, x_enc, token_indices, replacement_hidden, n_vars, N
):
    """
    第二阶段前向传播：在 embedding 层面替换指定 token 的初始表示（废弃）
    """
    pass


def run_baseline_experiment(ns, device, rank=0):
    """运行基线实验（标准推理）"""
    ns = build_namespace(ns)
    model = Model(ns).to(device)
    model.eval()

    if getattr(ns, 'use_multi_gpu', False):
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    _, loader = data_provider(ns, flag="test")

    preds, trues = [], []
    inference_time = 0.0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            t_start = time.time()

            if ns.use_ims:
                dec_inp = batch_y[:, :ns.label_len, :]
                y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
            else:
                dec_inp = batch_y[:, :ns.label_len, :]
                y_future = batch_y[:, -ns.pred_len:, :]

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if ns.use_ims:
                pred = outputs[:, -ns.pred_len:, :]
            else:
                pred = outputs[:, -ns.pred_len:, :]

            inference_time += time.time() - t_start

            preds.append(pred.cpu().numpy())
            trues.append(y_future.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    return preds, trues, inference_time


def calculate_patch_mi_and_select_topk(
    model, patch_hidden, y_future, use_layer_output, topk=1, sigma=50.
):
    """
    计算每个样本每个 patch 与 GT 的 MI（真正 per-sample per-patch）

    核心逻辑（参考 MI-Peaks）：
    - 对于 batch 内每个样本的每个 patch，使用 batch 内所有样本的对应位置
      构成核矩阵来估计 HSIC-MI
    - 每个样本的每个 patch 都得到一个独立的 MI 值

    Args:
        model: Timer 模型
        patch_hidden: patch 隐藏状态 [B, N, D]
        y_future: GT 未来序列 [B, pred_len, M]
        use_layer_output: decoder 层索引（用于提取 GT 的隐藏状态）
        topk: 选择 top-k 个高 MI patch
        sigma: HSIC sigma 参数

    Returns:
        selected_indices: 每个样本选择的 patch 索引 [B, topk]
    """
    B, N, D = patch_hidden.shape

    # 将 GT 也送入 decoder，获取相同层的隐藏状态
    gt_hidden, _, _, _ = forward_collect_single_layer(model, y_future, use_layer_output)
    # gt_hidden: [B, N, D] -> 每个样本取平均得到 [B, D]

    mi_scores = torch.zeros(B, N, device=patch_hidden.device)

    # 对每个 patch 位置，用 batch 内所有样本构建核矩阵
    # 计算每个样本在该 patch 位置与 GT 的 MI
    for j in range(N):
        # patch_hidden[:, j, :]: [B, D] - batch 内所有样本在第 j 个 patch 位置的值
        # gt_hidden.mean(dim=1): [B, D] - batch 内所有样本的 GT 表示

        # 计算 batch 内每个样本的 MI
        patch_vecs = patch_hidden[:, j, :]  # [B, D]
        gt_vecs = gt_hidden.mean(dim=1)       # [B, D]

        # 用当前 batch 的所有样本构建核矩阵，计算 HSIC-MI
        # 这里得到的 mi 是一个标量（batch 内统计量）
        mi = estimate_mi_hsic(patch_vecs, gt_vecs, sigma=sigma)
        mi_scores[:, j] = mi  # 所有样本共享该 patch 位置的 MI 值

    # 对每个样本选择 top-k 高 MI 的 patch
    _, topk_idx = torch.topk(mi_scores, k=min(topk, N), dim=1)
    selected_indices = topk_idx

    return selected_indices


def calculate_patch_mi_per_sample(
    model, patch_hidden, y_future, use_layer_output, topk=1, sigma=50.
):
    """
    真正的 per-sample per-patch MI 计算

    对每个样本：
    1. 使用 batch 内所有样本的信息构建核矩阵
    2. 计算该样本每个 patch 与 GT 的 HSIC-MI
    3. 返回每个样本独立的 MI 分数 [B, N]

    实现方式：
    - 第 b 个样本的第 j 个 patch：与其他样本的对应位置共同构建核矩阵
    - 这样每个样本都能得到 N 个独立的 MI 值
    """
    B, N, D = patch_hidden.shape

    # 将 GT 也送入 decoder，获取相同层的隐藏状态
    gt_hidden, _, _, _ = forward_collect_single_layer(model, y_future, use_layer_output)

    # 每个样本的 GT 表示：取平均 [B, D]
    gt_vecs = gt_hidden.mean(dim=1)  # [B, D]

    # 使用 batch 内所有样本构建核矩阵
    # 对于每个 patch 位置 j，计算 batch 内每个样本与 GT 的 MI
    mi_scores = torch.zeros(B, N, device=patch_hidden.device)

    # 对每个 patch 位置
    for j in range(N):
        patch_j = patch_hidden[:, j, :]  # [B, D] - batch 内所有样本在位置 j

        # 构建联合核矩阵：用 batch 内所有样本的 patch 和 GT
        # X = patch_j [B, D]，Y = gt_vecs [B, D]
        # 计算 HSIC-MI（返回单个标量，是 batch 统计量）
        mi = estimate_mi_hsic(patch_j, gt_vecs, sigma=sigma)

        # 该 patch 位置所有样本共享同一个 MI 统计量
        mi_scores[:, j] = mi

    return mi_scores  # [B, N]


def calculate_patch_mi_per_sample_v2(
    model, patch_hidden, y_future, use_layer_output, topk=1, sigma=50.
):
    """
    Per-sample MI 的另一种实现：使用 leave-one-out 思想

    对第 b 个样本：
    - 用 batch 内其他样本 (B-1 个) 构建核矩阵
    - 计算第 b 个样本的 MI

    这样每个样本都会得到独立的 MI 值。
    """
    B, N, D = patch_hidden.shape

    # 获取 GT 隐藏状态
    gt_hidden, _, _, _ = forward_collect_single_layer(model, y_future, use_layer_output)
    gt_vecs = gt_hidden.mean(dim=1)  # [B, D]

    mi_scores = torch.zeros(B, N, device=patch_hidden.device)

    for b in range(B):
        # 第 b 个样本的 patch 和 GT
        patch_b = patch_hidden[b]  # [N, D]
        gt_b = gt_vecs[b]  # [D]

        # 用其他样本构建核矩阵
        other_patches = torch.cat([patch_hidden[:b], patch_hidden[b+1:]], dim=0)  # [B-1, N, D]
        other_gt = torch.cat([gt_vecs[:b], gt_vecs[b+1:]], dim=0)  # [B-1, D]

        # 对每个 patch 位置，计算第 b 个样本的 MI
        for j in range(N):
            # 合并：当前样本 + 其他样本
            X = torch.cat([patch_b[j:j+1], other_patches[:, j, :]], dim=0)  # [B, D]
            Y = torch.cat([gt_b.unsqueeze(0), other_gt], dim=0)  # [B, D]

            # 计算 HSIC-MI
            mi = estimate_mi_hsic(X, Y, sigma=sigma)
            mi_scores[b, j] = mi

    # 对每个样本选择 top-k 高 MI 的 patch
    _, topk_idx = torch.topk(mi_scores, k=min(topk, N), dim=1)

    return topk_idx  # [B, topk]


def calculate_patch_mi_and_select_bottomk(
    model, patch_hidden, y_future, use_layer_output, bottomk=1, sigma=50.
):
    """
    计算每个 patch 与 GT 的 MI，选择 MI 最低的 patch（对比实验用）

    原理：
        1. 输入 patch_hidden：输入序列经过 decoder 在指定层的隐藏状态 [B, N, D]
        2. GT y_future：真实未来序列送入同一 decoder 获取隐藏状态
        3. 计算 patch_hidden vs gt_hidden 的 HSIC-MI，选择 MI 最低的 patch

    Args:
        model: Timer 模型
        patch_hidden: patch 隐藏状态 [B, N, D]
        y_future: GT 未来序列 [B, pred_len, M]
        use_layer_output: decoder 层索引（用于提取 GT 的隐藏状态）
        bottomk: 选择 bottom-k 个低 MI patch
        sigma: HSIC sigma 参数

    Returns:
        selected_indices: 每个样本选择的 patch 索引 [B, bottomk]
    """
    B, N, D = patch_hidden.shape

    # 将 GT 也送入 decoder，获取相同层的隐藏状态
    gt_hidden, _, _, _ = forward_collect_single_layer(model, y_future, use_layer_output)

    selected_indices = torch.zeros(B, bottomk, dtype=torch.long, device=patch_hidden.device)

    # 对每个 patch，计算与 GT 的 MI
    mi_scores = torch.zeros(B, N, device=patch_hidden.device)

    for j in range(N):
        patch_vecs = patch_hidden[:, j, :]  # [B, D]
        gt_vecs = gt_hidden.mean(dim=1)       # [B, D]

        mi = estimate_mi_hsic(patch_vecs, gt_vecs, sigma=sigma)
        mi_scores[:, j] = mi  # 所有样本共享该 patch 位置的 MI 值

    # 对每个样本选择 bottom-k 低 MI 的 patch
    _, bottomk_idx = torch.topk(-mi_scores, k=min(bottomk, N), dim=1)
    selected_indices = bottomk_idx

    return selected_indices


def forward_with_hidden_replacement_v2(
    model, x_enc, batch_y_future, selected_indices, n_vars, N, sigma=50.,
    target_layer_idx=-2, replace_layer_idx=None, static_patch_indices=None
):
    """
    第二阶段前向传播：真正的层间替换

    使用 Pass 1 提取的隐藏状态，在 Pass 2 时于指定层内部替换指定 patch token，
    然后继续往后跑剩余层。

    Args:
        model: Timer 模型
        x_enc: 输入序列 [B, L, M]
        batch_y_future: GT 未来序列 [B, pred_len, M]
        selected_indices: 每个样本选择的 patch 索引 [B, topk]（运行时计算模式）
        n_vars: 变量数量
        N: patch 数量
        sigma: HSIC sigma 参数
        target_layer_idx: 提取隐藏状态的层（负数支持）
        replace_layer_idx: 执行替换的层（默认为 target_layer_idx）
        static_patch_indices: 静态全局 patch 索引列表（预计算模式）

    Returns:
        dec_out: 预测输出
        selected_indices: 实际使用的索引
    """
    return forward_two_pass_layer_replacement(
        model, x_enc, batch_y_future, selected_indices,
        target_layer_idx=target_layer_idx,
        replace_layer_idx=replace_layer_idx,
        sigma=sigma,
        static_patch_indices=static_patch_indices,
    )


def run_bottomk_experiment(ns, device, rank=0):
    """
    运行对比实验：选择 MI 最低的 patch 进行替换

    支持预计算 MI 模式：
    - 当指定 --mi_peaks_file 时，直接使用文件中记录的低 MI patch 索引
    - 不再运行时计算 MI，大幅加速推理
    """
    ns = build_namespace(ns)
    model = Model(ns).to(device)
    model.eval()

    if getattr(ns, 'use_multi_gpu', False):
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    _, loader = data_provider(ns, flag="test")

    use_layer_output = getattr(ns, 'use_layer_output', -2)
    bottomk_patches = getattr(ns, 'bottomk_patches', 1)
    sigma = getattr(ns, 'mi_sigma', 50.0)

    # 预计算 MI 模式
    mi_peaks_file = getattr(ns, 'mi_peaks_file', '') or ''
    # bottomk 实验天然选低 MI，固定用 low 模式
    select_mode = "low"
    peaks_data = None
    static_patch_indices = None

    if mi_peaks_file and os.path.exists(mi_peaks_file):
        peaks_data = load_global_mi_peaks(mi_peaks_file)
        if rank == 0:
            print(f"[Bottom-K] Loaded MI peaks from: {mi_peaks_file}")
            print(f"[Bottom-K]   model_id={peaks_data.get('model_id')}, N={peaks_data.get('N')}")

        # 从 peaks 文件中获取指定层的低 MI patch 索引
        static_patch_indices = get_layer_patches_from_peaks(peaks_data, use_layer_output, "low")

        if rank == 0:
            print(f"[Bottom-K]   Static patches (layer {use_layer_output}, mode=low): {static_patch_indices}")
            print(f"[Bottom-K]   Total static patches: {len(static_patch_indices)}")

    if rank == 0:
        print(f"[Bottom-K] Extract layer: {use_layer_output}")
        print(f"[Bottom-K] Select bottom-{bottomk_patches} from global low-MI patches")
        print(f"[Bottom-K] HSIC sigma: {sigma}")
        print(f"[Bottom-K] Precomputed MI mode: {peaks_data is not None}")

    stage1_time = 0.0

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            t_start = time.time()
            target_hidden, n_vars, N, all_layers = forward_collect_single_layer(
                model, batch_x, use_layer_output
            )
            stage1_time += time.time() - t_start

            if batch_idx >= 1:
                break

    if rank == 0:
        print(f"[Bottom-K] Stage 1 time: {stage1_time:.4f}s")
        print(f"[Bottom-K] n_vars={n_vars}, N={N}")

    # 验证 N 与 peaks 文件一致
    if peaks_data is not None:
        N_from_file = peaks_data.get("N", N)
        if N_from_file != N:
            if rank == 0:
                print(f"[Bottom-K] WARNING: peaks file N={N_from_file} != actual N={N}. Using actual N.")
            static_patch_indices = [p for p in static_patch_indices if p < N] if static_patch_indices else []

    preds_bottomk, trues = [], []
    stage2_time = 0.0
    all_selected_indices = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            t_start = time.time()

            if ns.use_ims:
                dec_inp = batch_y[:, :ns.label_len, :]
                y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
            else:
                dec_inp = batch_y[:, :ns.label_len, :]
                y_future = batch_y[:, -ns.pred_len:, :]

            B = batch_x.shape[0]

            if peaks_data is not None and len(static_patch_indices) > 0:
                # ── 预计算 MI 模式：直接使用全局 patch 索引 ──────────────────────
                if bottomk_patches > 0:
                    # 取前 bottomk_patches 个 patch
                    selected = static_patch_indices[:bottomk_patches]
                    selected_indices = torch.tensor(
                        [selected for _ in range(B)],
                        dtype=torch.long, device=device
                    )
                else:
                    selected_indices = None
                all_selected_indices.append(selected_indices.cpu() if selected_indices is not None else None)

                dec_out = forward_with_hidden_replacement_v2(
                    model, batch_x, y_future, selected_indices, n_vars, N, sigma,
                    target_layer_idx=use_layer_output,
                    replace_layer_idx=getattr(ns, 'replace_layer_idx', None),
                    static_patch_indices=selected if selected_indices is not None else None,
                )[0]
            elif bottomk_patches > 0:
                # ── 运行时计算 MI 模式 ─────────────────────────────────────────────
                batch_hidden, _, _, _ = forward_collect_single_layer(
                    model, batch_x, use_layer_output
                )

                selected_indices = calculate_patch_mi_and_select_bottomk(
                    model, batch_hidden, y_future, use_layer_output,
                    bottomk=bottomk_patches, sigma=sigma
                )
                all_selected_indices.append(selected_indices.cpu())

                dec_out = forward_with_hidden_replacement_v2(
                    model, batch_x, y_future, selected_indices, n_vars, N, sigma,
                    target_layer_idx=use_layer_output,
                    replace_layer_idx=getattr(ns, 'replace_layer_idx', None),
                )[0]
            else:
                dec_out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if ns.use_ims:
                pred = dec_out[:, -ns.pred_len:, :]
            else:
                pred = dec_out[:, -ns.pred_len:, :]

            stage2_time += time.time() - t_start

            preds_bottomk.append(pred.cpu().numpy())
            trues.append(y_future.cpu().numpy())

    preds_bottomk = np.concatenate(preds_bottomk, axis=0)
    trues = np.concatenate(trues, axis=0)

    total_time = stage1_time + stage2_time
    all_selected_indices = [x for x in all_selected_indices if x is not None]
    all_selected_indices = torch.cat(all_selected_indices, dim=0) if all_selected_indices else None

    return preds_bottomk, trues, total_time, stage1_time, stage2_time, all_selected_indices


def run_two_stage_experiment(ns, device, rank=0):
    """
    运行两阶段 refinement 实验

    核心逻辑：
    - 第一阶段：提取指定层的隐藏状态
    - 第二阶段：每个样本计算各patch与GT的MI，自动选择top-k高MI的patch进行替换
      Y送回decoder不需要复制batch维度，直接使用单样本向量

    支持预计算 MI 模式：
    - 当指定 --mi_peaks_file 时，直接使用文件中记录的全局 patch 索引
    - 不再运行时计算 MI，大幅加速推理
    """
    ns = build_namespace(ns)
    model = Model(ns).to(device)
    model.eval()

    if getattr(ns, 'use_multi_gpu', False):
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    _, loader = data_provider(ns, flag="test")

    # 解析参数
    use_layer_output = getattr(ns, 'use_layer_output', -2)
    topk_patches = getattr(ns, 'topk_patches', 1)
    sigma = getattr(ns, 'mi_sigma', 50.0)

    # 预计算 MI 模式
    mi_peaks_file = getattr(ns, 'mi_peaks_file', '') or ''
    select_mode = getattr(ns, 'select_mode', 'high')
    peaks_data = None
    static_patch_indices = None

    if mi_peaks_file and os.path.exists(mi_peaks_file):
        peaks_data = load_global_mi_peaks(mi_peaks_file)
        N_from_file = peaks_data.get("N", 0)
        if rank == 0:
            print(f"[Two-Stage] Loaded MI peaks from: {mi_peaks_file}")
            print(f"[Two-Stage]   model_id={peaks_data.get('model_id')}, N={N_from_file}")
            print(f"[Two-Stage]   select_mode={select_mode}")

        # 从 peaks 文件中获取指定层的 patch 索引
        if select_mode == "high":
            static_patch_indices = get_layer_patches_from_peaks(peaks_data, use_layer_output, "high")
        else:
            static_patch_indices = get_layer_patches_from_peaks(peaks_data, use_layer_output, "low")

        if rank == 0:
            print(f"[Two-Stage]   Static patches (layer {use_layer_output}, mode={select_mode}): {static_patch_indices}")
            print(f"[Two-Stage]   Total static patches: {len(static_patch_indices)}")

    if rank == 0:
        print(f"[Two-Stage] Extract layer: {use_layer_output}")
        print(f"[Two-Stage] Select top-{topk_patches} from global high-MI patches")
        print(f"[Two-Stage] HSIC sigma: {sigma}")
        print(f"[Two-Stage] Precomputed MI mode: {peaks_data is not None}")

    # ── 第一阶段：收集模型结构参数 ─────────────────────────────────────────────
    stage1_time = 0.0

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            t_start = time.time()

            target_hidden, n_vars, N, all_layers = forward_collect_single_layer(
                model, batch_x, use_layer_output
            )

            stage1_time += time.time() - t_start

            if batch_idx >= 1:
                break

    if rank == 0:
        print(f"[Two-Stage] Stage 1 time: {stage1_time:.4f}s")
        print(f"[Two-Stage] n_vars={n_vars}, N={N}")

    # 验证 N 与 peaks 文件一致
    if peaks_data is not None:
        N_from_file = peaks_data.get("N", N)
        if N_from_file != N:
            if rank == 0:
                print(f"[Two-Stage] WARNING: peaks file N={N_from_file} != actual N={N}. Using actual N.")
            static_patch_indices = [p for p in static_patch_indices if p < N] if static_patch_indices else []

    # ── 第二阶段：每个样本自动选择高MI patch 并替换 ─────────────────────────────
    preds_refined, trues = [], []
    stage2_time = 0.0
    all_selected_indices = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            t_start = time.time()

            if ns.use_ims:
                dec_inp = batch_y[:, :ns.label_len, :]
                y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
            else:
                dec_inp = batch_y[:, :ns.label_len, :]
                y_future = batch_y[:, -ns.pred_len:, :]

            B = batch_x.shape[0]

            if peaks_data is not None and len(static_patch_indices) > 0:
                # ── 预计算 MI 模式：直接使用全局 patch 索引 ──────────────────────
                if topk_patches > 0:
                    # 取前 topk_patches 个 patch
                    selected = static_patch_indices[:topk_patches]
                    selected_indices = torch.tensor(
                        [selected for _ in range(B)],
                        dtype=torch.long, device=device
                    )
                else:
                    selected_indices = None
                all_selected_indices.append(selected_indices.cpu() if selected_indices is not None else None)

                dec_out = forward_with_hidden_replacement_v2(
                    model, batch_x, y_future, selected_indices, n_vars, N, sigma,
                    target_layer_idx=use_layer_output,
                    replace_layer_idx=getattr(ns, 'replace_layer_idx', None),
                    static_patch_indices=selected if selected_indices is not None else None,
                )[0]
            elif topk_patches > 0:
                # ── 运行时计算 MI 模式 ─────────────────────────────────────────────
                batch_hidden, _, _, _ = forward_collect_single_layer(
                    model, batch_x, use_layer_output
                )

                selected_indices = calculate_patch_mi_and_select_topk(
                    model, batch_hidden, y_future, use_layer_output,
                    topk=topk_patches, sigma=sigma
                )
                all_selected_indices.append(selected_indices.cpu())

                dec_out = forward_with_hidden_replacement_v2(
                    model, batch_x, y_future, selected_indices, n_vars, N, sigma,
                    target_layer_idx=use_layer_output,
                    replace_layer_idx=getattr(ns, 'replace_layer_idx', None),
                )[0]
            else:
                # 无需替换，执行标准推理
                dec_out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if ns.use_ims:
                pred = dec_out[:, -ns.pred_len:, :]
            else:
                pred = dec_out[:, -ns.pred_len:, :]

            stage2_time += time.time() - t_start

            preds_refined.append(pred.cpu().numpy())
            trues.append(y_future.cpu().numpy())

    preds_refined = np.concatenate(preds_refined, axis=0)
    trues = np.concatenate(trues, axis=0)

    total_time = stage1_time + stage2_time

    # 汇总选中的 patch 索引
    all_selected_indices = [x for x in all_selected_indices if x is not None]
    all_selected_indices = torch.cat(all_selected_indices, dim=0) if all_selected_indices else None

    return preds_refined, trues, total_time, stage1_time, stage2_time, all_selected_indices


def metric(y_true, y_pred):
    """计算预测评估指标"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


def main():
    p = argparse.ArgumentParser(description="ETTh1 两阶段 Patch Refinement 对比实验")
    # 模型参数
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Timer 模型检查点路径")
    p.add_argument("--root_path", type=str, default="./datasets/")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--data", type=str, default="ETTh1")
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--label_len", type=int, default=576)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--output_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=24)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./results/refinement_exp/")
    p.add_argument(
        "--use_ims", action="store_true",
        help="使用 IMS（Iterative Masked Prediction）模式"
    )
    p.add_argument(
        "--subset_rand_ratio", type=float, default=1.0,
        help="训练子集采样比例"
    )

    # 多卡参数
    p.add_argument(
        "--use_multi_gpu", action="store_true",
        help="使用多 GPU (torchrun)"
    )

    # 实验控制参数
    p.add_argument(
        "--skip_baseline", action="store_true",
        help="跳过基线实验（仅运行 Two-Stage）"
    )
    p.add_argument(
        "--skip_two_stage", action="store_true",
        help="跳过 Two-Stage 实验（仅运行基线）"
    )

    # ── Refinement 核心参数 ─────────────────────────────────────────────────────
    p.add_argument(
        "--refine_layer", type=int, default=-1,
        help="要替换的层索引（负数表示从后数，如 -1 表示最后一层）"
    )
    p.add_argument(
        "--refine_token_idx", type=str, default="",
        help="要替换的 token/patch 索引（逗号分隔，如 '5,6' 表示替换第5和第6个 token）"
    )
    p.add_argument(
        "--use_layer_output", type=int, default=-2,
        help="第一阶段提取哪一层的输出用于替换（负数表示从后数，如 -2 表示倒数第二层）"
    )
    p.add_argument(
        "--replace_layer_idx", type=int, default=None,
        help="执行替换的层索引（默认为 use_layer_output，即同层替换；可设为不同层实现跨层替换）"
    )
    p.add_argument(
        "--topk_patches", type=int, default=1,
        help="每个样本选择多少个高MI的patch进行替换（默认1）"
    )
    p.add_argument(
        "--mi_sigma", type=float, default=50.0,
        help="HSIC MI估计的sigma参数（默认50.0）"
    )
    p.add_argument(
        "--bottomk_patches", type=int, default=0,
        help="每个样本选择多少个低MI的patch进行替换（对比实验，默认0表示不运行）"
    )
    p.add_argument(
        "--mi_peaks_file", type=str, default="",
        help="预计算的全局 MI peaks JSON 文件路径（由 etth1_mi_hsic_peaks.py 生成）。"
             "指定后直接使用文件中记录的全局 patch 索引，不再运行时计算 MI"
    )
    p.add_argument(
        "--select_mode", type=str, default="high",
        help="Patch 选择模式: 'high' 选择高MI patch，'low' 选择低MI patch（默认 high）"
    )

    args = p.parse_args()

    rank = 0
    local_rank = 0
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError("多 GPU 需要 torchrun 设置 WORLD_SIZE/RANK/LOCAL_RANK")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if args.use_multi_gpu else args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    if not args.skip_baseline:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"基线实验：标准 Timer 推理")
            print(f"{'='*60}\n")

        preds_baseline, trues_baseline, baseline_time = run_baseline_experiment(
            args, device, rank=rank
        )
        baseline_metrics = metric(trues_baseline, preds_baseline)
        baseline_metrics["inference_time"] = baseline_time
        baseline_metrics["avg_time_per_sample"] = baseline_time / len(preds_baseline)
        results["baseline"] = baseline_metrics

        if rank == 0:
            print(f"\n基线结果:")
            for k, v in baseline_metrics.items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    if not args.skip_two_stage:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"两阶段 Refinement 实验（高 MI Patch）")
            print(f"  提取层 (use_layer_output): {args.use_layer_output}")
            print(f"  替换层 (replace_layer_idx): {args.replace_layer_idx or args.use_layer_output}")
            print(f"  预计算MI文件 (mi_peaks_file): {args.mi_peaks_file or 'None'}")
            print(f"  Patch选择模式 (select_mode): {args.select_mode}（高MI patch）")
            print(f"  从高MI集合中取 topk={args.topk_patches} 个 patch 进行替换")
            print(f"{'='*60}\n")

        preds_two_stage, trues_two_stage, two_stage_time, s1_time, s2_time, selected_indices = run_two_stage_experiment(
            args, device, rank=rank
        )
        two_stage_metrics = metric(trues_two_stage, preds_two_stage)
        two_stage_metrics["inference_time"] = two_stage_time
        two_stage_metrics["stage1_time"] = s1_time
        two_stage_metrics["stage2_time"] = s2_time
        two_stage_metrics["avg_time_per_sample"] = two_stage_time / len(preds_two_stage)
        results["two_stage"] = two_stage_metrics

        # 统计选中的patch分布
        if selected_indices is not None:
            unique_patches = torch.unique(selected_indices)
            two_stage_metrics["unique_patches_selected"] = len(unique_patches)
            two_stage_metrics["selected_patch_distribution"] = torch.bincount(
                selected_indices.flatten().long()
            ).cpu().tolist()
            if rank == 0:
                print(f"  选中patch分布: {unique_patches.tolist()}")
                print(f"  不同patch数量: {len(unique_patches)}")

        if rank == 0:
            print(f"\n两阶段结果:")
            for k, v in two_stage_metrics.items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── 对比实验：选择低 MI 的 patch ─────────────────────────────────────────
    if args.bottomk_patches > 0:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Bottom-K 对比实验（低 MI Patch）")
            print(f"  提取层 (use_layer_output): {args.use_layer_output}")
            print(f"  预计算MI文件 (mi_peaks_file): {args.mi_peaks_file or 'None'}")
            print(f"  Patch选择模式：自动（bottomk 固定选低MI patch）")
            print(f"  从低MI集合中取 bottomk={args.bottomk_patches} 个 patch 进行替换")
            print(f"{'='*60}\n")

        preds_bottomk, trues_bottomk, bottomk_time, bk_s1, bk_s2, bk_indices = run_bottomk_experiment(
            args, device, rank=rank
        )
        bottomk_metrics = metric(trues_bottomk, preds_bottomk)
        bottomk_metrics["inference_time"] = bottomk_time
        bottomk_metrics["stage1_time"] = bk_s1
        bottomk_metrics["stage2_time"] = bk_s2
        bottomk_metrics["avg_time_per_sample"] = bottomk_time / len(preds_bottomk)
        results["bottomk"] = bottomk_metrics

        if bk_indices is not None:
            unique_patches = torch.unique(bk_indices)
            bottomk_metrics["unique_patches_selected"] = len(unique_patches)
            if rank == 0:
                print(f"  选中patch分布: {unique_patches.tolist()}")
                print(f"  不同patch数量: {len(unique_patches)}")

        if rank == 0:
            print(f"\nBottom-K 结果:")
            for k, v in bottomk_metrics.items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    if rank == 0 and len(results) >= 2:
        print("\n" + "="*60)
        print("对比分析:")
        print("="*60)
        for key in ["MSE", "MAE", "RMSE", "inference_time"]:
            baseline_val = results["baseline"].get(key, float('inf'))
            row_lines = []
            for exp_name in ["two_stage", "bottomk"]:
                if exp_name in results:
                    val = results[exp_name].get(key, float('inf'))
                    if baseline_val != float('inf') and val != float('inf'):
                        improvement = (baseline_val - val) / baseline_val * 100
                        row_lines.append(f"  {exp_name}: {val:.6f} ({improvement:+.2f}%)")
            if row_lines:
                print(f"  {key}:")
                print(f"    基线:      {baseline_val:.6f}")
                for line in row_lines:
                    print(line)

        # 保存结果
        output_file = os.path.join(args.output_dir, "comparison_results.txt")
        with open(output_file, 'w') as f:
            f.write("两阶段 Patch Refinement 对���实验结果\n")
            f.write("="*60 + "\n\n")
            f.write(f"参数配置:\n")
            f.write(f"  refine_layer: {args.refine_layer}\n")
            f.write(f"  refine_token_idx: {args.refine_token_idx}\n")
            f.write(f"  use_layer_output: {args.use_layer_output}\n\n")

            for exp_name, metrics in results.items():
                f.write(f"实验: {exp_name}\n")
                for k, v in metrics.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            if "baseline" in results:
                f.write("改进分析 (相对于 baseline):\n")
                for key in ["MSE", "MAE", "RMSE", "inference_time"]:
                    baseline_val = results["baseline"].get(key, float('inf'))
                    for exp_name in ["two_stage", "bottomk"]:
                        if exp_name in results:
                            val = results[exp_name].get(key, float('inf'))
                            if baseline_val != float('inf') and val != float('inf'):
                                improvement = (baseline_val - val) / baseline_val * 100
                                f.write(f"  {key} [{exp_name}]: {improvement:+.2f}%\n")

        print(f"\n结果已保存到: {output_file}")

    if args.use_multi_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
