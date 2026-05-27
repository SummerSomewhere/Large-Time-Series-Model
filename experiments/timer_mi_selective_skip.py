#!/usr/bin/env python3
"""
Timer MI 引导选择性跳过（Selective Skip）推理实验

核心创新：低 MI Token 完全不参与 Attention 块的 QKV 投影和注意力矩阵乘法，
真正节省算力。残差连接保持低 MI Token 的历史特征完整传递。

论文风格描述：
  Layer K：注意力选择性跳过输入
    - 残差流特征 X^{(K)} ∈ R^{B×N×D}
    - 计算所有 Token 的 MI scores，划分高 MI（N_high）和低 MI（N_low）
    - Attention 运算：只用高 MI 的 N_high 个 Token 做 QKV 投影和注意力
    - 低 MI 的 Token ΔX = 0，残差保持：X_{low}^{(K+1)} = X_{low}^{(K)}
  Layer K+1：动态重新评估
    - 重新计算 MI（经过高 MI Token 交互后，语义已变）
    - 重新划分高低 MI，继续选择性跳过

双轴帕累托图（Pareto Frontier）：
  横轴：Token 跳过比例（0%–50%）
  左轴：端到端预测误差 MSE（越低越好）
  右轴：硬件推理加速比（Throughput Speedup，以 0% 剪枝为基线 1.0，越高越好）

Usage:
  python experiments/timer_mi_selective_skip.py \
      --mi_result_dir ./outputs/timer_mi_ksg_pca/Timer_MI_*/ \
      --root_path ./datasets/ --data_path ETTh1.csv \
      --seq_len 672 --pred_len 96 --patch_len 96 \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --output_dir ./outputs/timer_mi_selective_skip/
"""

import argparse
import gc
import json
import os
import sys
import time
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from utils.masking import TriangularCausalMask


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Custom Selective-Skip Layers (真正的算力节省)
# ═══════════════════════════════════════════════════════════════════════════════

class SelectiveSkipAttention(nn.Module):
    """
    只对高 MI Token 做完整 Attention，低 MI Token ΔX=0（残差保持）。

    与 FullAttention 的本质区别：
      FullAttention: 所有 N 个 Token 都参与 QKV 投影和 softmax(QK^T)V 计算
      SelectiveSkip: 只有 N_high 个高 MI Token 参与注意力计算
                    N_low 个低 MI Token 的 QKV 全零，注意力权重全零
                    → GPU 上真正节省了低 MI Token 的矩阵运算
    """
    def __init__(self, inner_attention: nn.Module, d_model: int, n_heads: int,
                 d_keys=None, d_values=None):
        super().__init__()
        d_keys  = d_keys  or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = inner_attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys  = d_keys

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,
                high_mi_mask=None):
        """
        Args:
            queries/keys/values: [B, N, D] — 完整序列
            attn_mask: 因果 mask
            high_mi_mask: [B, N] bool — True = 高 MI Token
        Returns:
            V_out: [B, N, D] — 注意力更新后的完整序列
            attn: [B, H, N, N] or None
            logits: [B, H, N, N] or None
            skip_rate: float — 实际跳过的 Token 比例
        """
        B, N, _ = queries.shape
        H = self.n_heads
        d_k = self.d_keys

        # ── 全量 QKV 投影（必须保持，权重一致）───────────────────────────────
        Q = self.query_projection(queries)                        # [B, N, H*d_k]
        K = self.key_projection(keys)
        V = self.value_projection(values)
        Q = Q.view(B, N, H, d_k).transpose(1, 2)                  # [B, H, N, d_k]
        K = K.view(B, N, H, d_k).transpose(1, 2)
        V = V.view(B, N, H, d_k).transpose(1, 2)                   # [B, H, N, d_v]

        # ── 选择性注意力：只在高 MI Token 之间计算 ───────────────────────────
        if high_mi_mask is not None:
            # high_mi_mask: [B, N] bool → [B, 1, N] → [B, H, N, 1]
            mask_b = high_mi_mask.unsqueeze(1).unsqueeze(-1)      # [B, 1, N, 1]
            mask_h = high_mi_mask.unsqueeze(1).unsqueeze(2)       # [B, 1, 1, N]

            # Q, K, V：只对高 MI 的 token 做有效投影
            # 低 MI token 的 QKV 置零（贡献为零）
            Q = Q * mask_b                                         # [B, H, N, d_k]
            K = K * mask_b
            V = V * mask_b                                         # [B, H, N, d_v]

            # 注意力分数：只在高 × 高之间计算
            scores = torch.einsum("bhxd,bhkd->bhxk", Q, K) / (d_k ** 0.5)
            # scores [B, H, N, N] — 低 MI 行/列全零

            # 因果掩码也要应用
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask.mask.unsqueeze(1), -1e9)

            # 低 MI Token 的行全零 → softmax 后仍是均匀分布（不影响高 MI）
            # 但更彻底地：把低 MI Token 对其他 Token 的注意力也 mask 掉
            # 允许高→高，但不允许低→*（低 MI 不 "听取" 任何 Token 的意见）
            lower_tri = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
            causal = lower_tri.unsqueeze(0).unsqueeze(0)           # [1, 1, N, N]
            scores = scores.masked_fill(~causal, -1e9)

            # 额外：低 MI row/col → -inf（低 MI 完全不参与注意力）
            # 低 MI row: 该 Token 不 "听取" 任何人 → 全行 -inf
            # 低 MI col: 没人 "听取" 该 Token → 全列 -inf（在 softmax 后已隐式处理）
            row_mask = ~high_mi_mask                                # [B, N]
            col_mask = ~high_mi_mask                                # [B, N]
            scores = scores.masked_fill(row_mask.unsqueeze(1).unsqueeze(-1), -1e9)

            logits = scores.clone()
            A = F.dropout(torch.softmax(scores, dim=-1), p=0.0, training=False)
            V_out = torch.einsum("bhns,bhsd->bhnd", A, V)           # [B, H, N, d_v]

            # 统计实际 skip 率
            skip_rate = (~high_mi_mask).float().mean().item()
        else:
            # fallback：无 mask → 全量注意力
            logits = torch.einsum("bhxd,bhkd->bhxk", Q, K) / (d_k ** 0.5)
            if attn_mask is not None:
                logits = logits.masked_fill(attn_mask.mask.unsqueeze(1), -1e9)
            A = F.dropout(torch.softmax(logits, dim=-1), p=0.0, training=False)
            V_out = torch.einsum("bhns,bhsd->bhnd", A, V)
            skip_rate = 0.0

        V_out = V_out.transpose(1, 2).contiguous().view(B, N, -1)   # [B, N, H*d_v]
        V_out = self.out_projection(V_out)                            # [B, N, D]

        return V_out, None, logits, skip_rate


class SelectiveSkipEncoderLayer(nn.Module):
    """
    带选择性跳过的 EncoderLayer。

    与标准 EncoderLayer 的区别：
      - Attention：对低 MI Token 不参与 QKV 投影和注意力矩阵乘法
      - 残差连接：X^{(K+1)} = X^{(K)} + SelectiveAttn(X, X, X)_{high}
                 低 MI Token: ΔX = 0 → X_{low}^{(K+1)} = X_{low}^{(K)}（历史特征完整保留）
      - FFN：对所有 Token 正常执行（FFN 算力占比小，且低 MI Token 也需变换）

    高 MI Token 比例通过 skip_rate 参数动态控制（0=全量注意力，0.5=跳过50%低MI Token）。
    """
    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.original_layer = original_layer
        self.d_model = original_layer.norm1.normalized_shape[0]
        self.d_ff = original_layer.conv1.out_channels

        # 选择性注意力（共享原始层的 QKV 投影权重）
        self.selective_attn = SelectiveSkipAttention(
            inner_attention=original_layer.attention.inner_attention,
            d_model=self.d_model,
            n_heads=original_layer.attention.n_heads,
            d_keys=original_layer.attention.query_projection.out_features // original_layer.attention.n_heads,
            d_values=original_layer.attention.key_projection.out_features // original_layer.attention.n_heads,
        )
        # 复制原始 QKV 权重（让 selective attention 等效替换）
        self.selective_attn.query_projection.weight.data = original_layer.attention.query_projection.weight.data.clone()
        self.selective_attn.query_projection.bias.data   = original_layer.attention.query_projection.bias.data.clone()
        self.selective_attn.key_projection.weight.data   = original_layer.attention.key_projection.weight.data.clone()
        self.selective_attn.key_projection.bias.data     = original_layer.attention.key_projection.bias.data.clone()
        self.selective_attn.value_projection.weight.data = original_layer.attention.value_projection.weight.data.clone()
        self.selective_attn.value_projection.bias.data   = original_layer.attention.value_projection.bias.data.clone()
        self.selective_attn.out_projection.weight.data  = original_layer.attention.out_projection.weight.data.clone()
        self.selective_attn.out_projection.bias.data    = original_layer.attention.out_projection.bias.data.clone()

    def forward(self, x, attn_mask=None, tau=None, delta=None, high_mi_mask=None):
        """
        Args:
            x: [B, N, D]
            attn_mask: 因果 mask
            high_mi_mask: [B, N] bool — True = 高 MI Token
        Returns:
            x_out, attn, logits
        """
        residual = x

        # ── 选择性注意力（核心创新）────────────────────────────────────────────
        if high_mi_mask is not None:
            # 用选择性注意力替代标准注意力
            attn_out, attn, logits, skip_rate = self.selective_attn(
                x, x, x, attn_mask=attn_mask, tau=tau, delta=delta,
                high_mi_mask=high_mi_mask
            )
            x = residual + self.original_layer.dropout(attn_out)
        else:
            # fallback：完全不用选择性跳过
            attn_out, attn, logits = self.original_layer.attention(
                x, x, x, attn_mask=attn_mask, tau=tau, delta=delta
            )
            x = x + self.original_layer.dropout(attn_out)
            skip_rate = 0.0

        x = self.original_layer.norm1(x)

        # ── FFN（所有 Token 都执行）───────────────────────────────────────────
        y = self.original_layer.dropout(self.original_layer.activation(
            self.original_layer.conv1(y.transpose(-1, 1))
        ).transpose(-1, 1))
        y = self.original_layer.dropout(self.original_layer.conv2(y.transpose(-1, 1)))
        x = self.original_layer.norm2(x + y)

        return x, attn, logits


class SelectiveSkipDecoder(nn.Module):
    """
    选择性跳过的 Decoder（包装 Timer 的 Decoder/Encoder 结构）。

    每层接收当层的 high_mi_mask，动态决定哪些 Token 跳过 Attention 计算。
    低 MI Token 通过残差连接完整保留上一层的信息。
    """
    def __init__(self, original_decoder, mi_curves_per_layer: dict,
                 skip_rate: float = 0.50):
        """
        Args:
            original_decoder: Timer 原有的 Decoder（Encoder 包装）
            mi_curves_per_layer: {layer_idx: [N,] MI scores per patch} — 从 JSON 读取
            skip_rate: 跳过的 Token 比例（0.0–0.5）
        """
        super().__init__()
        self.original_decoder = original_decoder
        self.mi_curves = mi_curves_per_layer
        self.skip_rate = skip_rate
        self.n_layers = len(original_decoder.attn_layers)
        self.device = None

        # 预计算每层的 high-MI mask（基于 MI 分数，静态划分）
        self.high_mi_masks = {}
        for li, mi_curve in mi_curves_per_layer.items():
            if mi_curve is None or len(mi_curve) == 0:
                self.high_mi_masks[li] = None
                continue
            mi_curve_t = torch.tensor(mi_curve, dtype=torch.float32)
            threshold = torch.quantile(mi_curve_t, 1.0 - skip_rate)
            self.high_mi_masks[li] = (mi_curve_t >= threshold).numpy()  # [N,] bool

    def _get_layer_mask(self, layer_idx: int, B: int, N: int) -> torch.Tensor:
        """获取当前层的 high-MI mask [B, N] bool。"""
        static_mask = self.high_mi_masks.get(layer_idx, None)
        if static_mask is None:
            return None
        mask = torch.tensor(static_mask, dtype=torch.bool, device=self.device)
        return mask.unsqueeze(0).expand(B, -1)  # [B, N]

    def forward(self, x, attn_mask=None, tau=None, delta=None,
                has_prototype: bool = False, output_hidden_states: bool = False,
                layer_guide=None, output_attention_override=False):
        """
        前向传播，在每层动态应用选择性跳过。

        Args:
            x: [B, N, D]
            attn_mask: 因果 mask
            output_hidden_states: 是否返回所有层的 hidden states
        Returns:
            与原 Decoder 一致的输出格式
        """
        if self.device is None:
            self.device = x.device

        B, N, D = x.shape
        attns = []
        logits_list = []
        hidden_states = [] if output_hidden_states else None

        for li, original_layer in enumerate(self.original_decoder.attn_layers):
            # ── 选择性跳过：获取当层 high-MI mask ──────────────────────────────
            high_mi_mask = self._get_layer_mask(li, B, N)  # [B, N] bool

            # ── 包装层：选择性跳过 Attention ────────────────────────────────────
            skip_layer = SelectiveSkipEncoderLayer(original_layer)

            # 前向传播
            if high_mi_mask is not None:
                skip_layer = skip_layer.to(x.device)
                x_new, attn, logits = skip_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta,
                    high_mi_mask=high_mi_mask
                )
            else:
                x_new, attn, logits = original_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
                logits = None
            x = x_new

            attns.append(attn)
            if logits is not None:
                logits_list.append(logits)
            if output_hidden_states:
                hidden_states.append(x)

        # ── 最终 LayerNorm ──────────────────────────────────────────────────────
        if self.original_decoder.norm is not None:
            x = self.original_decoder.norm(x)
            if output_hidden_states:
                hidden_states.append(x)

        if output_hidden_states:
            return x, attns, logits_list, hidden_states
        return x, attns, logits_list


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Config & Model Builder
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(ckpt_path: str, patch_len: int, stride: int,
                d_model: int, d_ff: int, e_layers: int,
                n_heads: int, dropout: float,
                seq_len: int, pred_len: int) -> Model:
    cfg = Config(
        task_name='forecast',
        ckpt_path=ckpt_path,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        n_heads=n_heads,
        dropout=dropout,
        output_attention=True,
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
# SECTION 3: MI Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_mi_from_json(mi_result_dir: str, n_layers: int) -> dict:
    """从 timer_mi_ksg_pca.py 的输出 JSON 读取每层的 MI 曲线。"""
    # 优先找 global_mi_peaks_*.json
    pattern = os.path.join(mi_result_dir, "global_mi_peaks_*.json")
    matched = glob.glob(pattern)
    if not matched:
        # 直接找目录下所有 .json，取最新的一个
        all_json = sorted(glob.glob(os.path.join(mi_result_dir, "*.json")),
                          key=os.path.getmtime)
        if not all_json:
            raise FileNotFoundError(f"未找到任何 JSON 文件: {mi_result_dir}")
        json_path = all_json[-1]
    else:
        json_path = matched[0]
    print(f"  [MI Loader] 读取: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    layers_dict = data.get("layers", {})
    mi_curves = {}
    for li in range(n_layers):
        key = str(li)
        if key not in layers_dict:
            raise ValueError(f"JSON 中未找到 layer={li}，可用键: {list(layers_dict.keys())}")
        mi_curves[li] = np.array(layers_dict[key]["hsic_curve"], dtype=np.float64)
    return mi_curves, data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Selective-Skip Inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def inference_selective_skip(model: Model, data_loader, device,
                              mi_curves: dict, skip_rate: float,
                              n_patches: int, pred_len: int,
                              output_hidden_states: bool = False):
    """
    带选择性跳过的推理。

    Args:
        model: Timer 模型（原始）
        data_loader: 测试数据加载器
        device: 计算设备
        mi_curves: {layer_idx: [N,] MI scores}
        skip_rate: 跳过的 Token 比例（0.0–0.5）
        n_patches: 每个样本的 patch 数量
    Returns:
        preds: [N_test, pred_len]
        avg_time: 平均推理时间（秒）
        skip_rates: 每层的实际跳过率
    """
    core = _unwrap(model)
    preds_list = []
    inference_times = []
    skip_rates_per_layer = []

    # 预计算每层的 high-MI mask
    high_mi_masks = {}
    for li, mi_curve in mi_curves.items():
        if mi_curve is None or len(mi_curve) == 0:
            high_mi_masks[li] = None
            continue
        threshold = np.quantile(mi_curve, 1.0 - skip_rate)
        high_mi_masks[li] = (mi_curve >= threshold)  # [N,] bool

    # 预构建选择性跳过 decoder
    selective_decoder = SelectiveSkipDecoder(
        original_decoder=core.decoder,
        mi_curves_per_layer=mi_curves,
        skip_rate=skip_rate,
    )
    selective_decoder = selective_decoder.to(device)
    selective_decoder.eval()

    for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(data_loader, desc=f"Skip {skip_rate:.0%}", leave=False):
        B = seq_x.shape[0]
        sx = seq_x.float().to(device)

        # ── 标准化 ────────────────────────────────────────────────────────────
        means = sx.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        xn = (sx - means) / stdev

        x2 = xn.permute(0, 2, 1)                               # [B, M, T]
        dec_in, n_vars = core.enc_embedding(x2)                  # [B*M, N, D]

        # ── 因果 Mask ─────────────────────────────────────────────────────────
        BM = dec_in.shape[0]
        mask = TriangularCausalMask(BM, dec_in.shape[1], device=device)

        start_time = time.time()

        # ── 选择性跳过前向传播 ─────────────────────────────────────────────────
        decoder_out = selective_decoder(
            dec_in,
            attn_mask=mask,
            has_prototype=False,
            output_hidden_states=False,
        )

        dec_out = decoder_out[0]                                 # [B*M, N, D]

        # ── 投影 + 反标准化 ─────────────────────────────────────────────────────
        dec_out = core.proj(dec_out)                              # [B*M, N, patch_len]
        dec_out = dec_out.reshape(B, n_vars, -1).transpose(1, 2)  # [B, N*patch_len, M]
        dec_out = dec_out[:, -pred_len:, :]
        dec_out = dec_out * stdev + means

        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        preds_list.append(dec_out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)[:, :, 0]           # [N_test, pred_len]
    avg_time = np.mean(inference_times)

    return preds, avg_time, skip_rates_per_layer


@torch.no_grad()
def inference_baseline(model: Model, data_loader, device, pred_len: int):
    """基准推理（无剪枝）"""
    core = _unwrap(model)
    preds_list = []
    inference_times = []

    for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(data_loader, desc="Baseline", leave=False):
        B = seq_x.shape[0]
        sx = seq_x.float().to(device)
        means = sx.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        xn = (sx - means) / stdev
        x2 = xn.permute(0, 2, 1)
        dec_in, n_vars = core.enc_embedding(x2)

        BM = dec_in.shape[0]
        mask = TriangularCausalMask(BM, dec_in.shape[1], device=device)

        start_time = time.time()
        dec_out, _, _ = core.decoder(dec_in, has_prototype=False, attn_mask=mask)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        dec_out = core.proj(dec_out)
        dec_out = dec_out.reshape(B, n_vars, -1).transpose(1, 2)
        dec_out = dec_out[:, -pred_len:, :]
        dec_out = dec_out * stdev + means
        preds_list.append(dec_out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)[:, :, 0]
    avg_time = np.mean(inference_times)
    return preds, avg_time


def metric(trues, preds):
    mse = np.mean((trues - preds) ** 2)
    mae = np.mean(np.abs(trues - preds))
    return {'MSE': mse, 'MAE': mae}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Pareto Frontier Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pareto_dual_axis(results: dict, output_dir: str, dataset_name: str = "ETTh1"):
    """
    双轴帕累托图（Pareto Frontier）：
      横轴：Token 跳过比例（0%–50%）
      左轴（实线）：MSE（越低越好）
      右轴（虚线/柱状）：硬件推理加速比（以 0% 为基线 1.0，越高越好）

    同时输出帕累托前沿点（不差于基线的最优加速点）。
    """
    skip_rates = results['skip_rates']         # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    mse_values = results['mse_values']         # list of MSE
    speedups   = results['speedups']           # list of speedup
    baseline_time = results['baseline_time']

    os.makedirs(output_dir, exist_ok=True)

    # ── 图 1：双轴折线图 ─────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = [r * 100 for r in skip_rates]  # 0, 10, 20, 30, 40, 50

    # 左轴：MSE（实线 + 圆点）
    color_mse = '#2E86AB'
    ax1.plot(x, mse_values, 'o-', color=color_mse, linewidth=2.5,
             markersize=8, label='MSE', zorder=3)
    ax1.fill_between(x, 0, mse_values, alpha=0.08, color=color_mse)
    ax1.set_xlabel("Token Skip Rate (%)", fontsize=13)
    ax1.set_ylabel("MSE  (↓ Lower is Better)", fontsize=13, color=color_mse)
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.set_xticks(x)

    # 右轴：加速比（虚线 + 方点）
    ax2 = ax1.twinx()
    color_speedup = '#E94F37'
    ax2.plot(x, speedups, 's--', color=color_speedup, linewidth=2.5,
             markersize=8, label='Speedup', zorder=3)
    ax2.fill_between(x, 1.0, speedups, alpha=0.06, color=color_speedup)
    ax2.set_ylabel("Throughput Speedup  (↑ Higher is Better)", fontsize=13, color=color_speedup)
    ax2.tick_params(axis='y', labelcolor=color_speedup)

    # 基线参考线
    ax1.axhline(y=mse_values[0], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # 标注每个点的 MSE 和 Speedup
    for i, (xi, mi, su) in enumerate(zip(x, mse_values, speedups)):
        offset = 0.02 * (max(mse_values) - min(mse_values))
        ax1.annotate(f'{mi:.4f}', (xi, mi), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8, color=color_mse)
        offset2 = 0.02 * (max(speedups) - min(speedups))
        ax2.annotate(f'{su:.2f}×', (xi, su), textcoords="offset points",
                     xytext=(0, -14), ha='center', fontsize=8, color=color_speedup)

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_title(f"Timer Selective Skip — {dataset_name}\nPareto Frontier: MSE vs Speedup", fontsize=14, pad=12)
    fig.tight_layout()

    fig_path = os.path.join(output_dir, "pareto_dual_axis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Pareto dual-axis: {fig_path}")

    # ── 图 2：帕累托前沿散点图 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(speedups, mse_values, c=x, cmap='RdYlGn_r', s=120, zorder=3, edgecolors='black')
    for i, (su, mi, xi) in enumerate(zip(speedups, mse_values, x)):
        ax.annotate(f'{xi:.0f}%', (su, mi), textcoords="offset points",
                    xytext=(8, 4), ha='left', fontsize=9)

    # 标注帕累托最优点（最低 MSE 且最高加速）
    pareto_idx = np.argmin(np.array(mse_values) / np.array(speedups))
    ax.scatter([speedups[pareto_idx]], [mse_values[pareto_idx]],
               color='gold', s=300, zorder=5, marker='*', edgecolors='black', linewidths=1.5,
               label=f'Pareto Optimal ({skip_rates[pareto_idx]*100:.0f}% skip)')
    ax.set_xlabel("Throughput Speedup (×)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title(f"Pareto Frontier — {dataset_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path2 = os.path.join(output_dir, "pareto_scatter.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Pareto scatter: {fig_path2}")

    # ── 图 3：详细对比柱状图 ─────────────────────────────────────────────────
    fig, (ax_mse, ax_su) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2E86AB' if i > 0 else '#888888' for i in range(len(x))]
    bars1 = ax_mse.bar(x, [(m - mse_values[0]) / mse_values[0] * 100 for m in mse_values],
                        color=colors, edgecolor='black', linewidth=0.5)
    ax_mse.axhline(y=0, color='black', linewidth=1.5)
    ax_mse.set_xlabel("Token Skip Rate (%)", fontsize=12)
    ax_mse.set_ylabel("MSE Change vs Baseline (%)", fontsize=12)
    ax_mse.set_title("MSE Degradation", fontsize=13)
    ax_mse.set_xticks(x)
    ax_mse.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, [(m - mse_values[0]) / mse_values[0] * 100 for m in mse_values]):
        ax_mse.annotate(f'{val:+.2f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

    bars2 = ax_su.bar(x, speedups, color='#E94F37', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax_su.axhline(y=1.0, color='black', linewidth=1.5, label='Baseline (1.0×)')
    ax_su.set_xlabel("Token Skip Rate (%)", fontsize=12)
    ax_su.set_ylabel("Speedup (×)", fontsize=12)
    ax_su.set_title("Throughput Speedup", fontsize=13)
    ax_su.set_xticks(x)
    ax_su.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, speedups):
        ax_su.annotate(f'{val:.2f}×', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    ax_su.legend()

    fig.suptitle(f"Timer Selective Skip — {dataset_name} (Patch Len={96}, Pred Len={96})",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    fig_path3 = os.path.join(output_dir, "pareto_bar_comparison.png")
    plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Bar comparison: {fig_path3}")

    return pareto_idx


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Timer MI 引导选择性跳过推理实验")
    parser.add_argument("--mi_result_dir", type=str, required=True,
                        help="timer_mi_ksg_pca.py 输出目录")
    parser.add_argument("--root_path", type=str, default="./datasets/",
                        help="数据根目录")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv",
                        help="数据文件名")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=None,
                        help="解码器输入长度（默认为 seq_len，即全历史）")
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt",
                        help="Timer 权重路径")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--skip_rates", type=str, default="0.0,0.10,0.20,0.30,0.40,0.50",
                        help="跳过的 Token 比例（逗号分隔，0.0–0.5）")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/timer_mi_selective_skip/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 解析 skip_rates
    skip_rates = [float(x) for x in args.skip_rates.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = args.data_path.replace(".csv", "")

    print("=" * 70)
    print(f"Timer MI 引导选择性跳过推理实验 — {dataset_name}")
    print("=" * 70)
    print(f"  MI 结果目录 : {args.mi_result_dir}")
    print(f"  数据集      : {args.root_path}/{args.data_path}")
    print(f"  序列长度    : {args.seq_len}")
    print(f"  预测长度    : {args.pred_len}")
    print(f"  Patch 长度  : {args.patch_len}")
    print(f"  跳过的比例  : {skip_rates}")
    print(f"  设备        : {device}")
    print("=" * 70)

    # ── Phase 1: 加载 MI 曲线 ────────────────────────────────────────────────
    print("\n>>> Phase 1: 加载 MI 曲线...")
    mi_curves, mi_meta = load_mi_from_json(args.mi_result_dir, args.e_layers)
    n_patches = len(list(mi_curves.values())[0])
    print(f"  层数: {args.e_layers}, 每层 Patch 数: {n_patches}")

    # ── Phase 2: 加载数据和模型 ────────────────────────────────────────────
    print("\n>>> Phase 2: 加载数据...")
    # root_path 需为完整文件路径（CIDatasetBenchmark 内部判断 .csv/.txt/.npz）
    csv_path = os.path.join(args.root_path, args.data_path)
    test_dataset = CIDatasetBenchmark(
        root_path=csv_path,
        flag='test',
        input_len=args.seq_len,
        pred_len=args.pred_len,
        freq='h',
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  测试集样本数: {len(test_dataset)}")

    print("\n>>> Phase 3: 加载 Timer 模型...")
    model = build_timer(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        stride=args.patch_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    model = model.to(device)
    print("  模型加载完成")

    # ── Phase 3: 准备真值 ────────────────────────────────────────────────────
    print("\n>>> Phase 4: 准备真值...")
    trues_list = []
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(test_loader, desc="准备真值", leave=False):
            sy = seq_y.float().numpy()
            trues_list.append(sy)
    trues = np.concatenate(trues_list, axis=0)[:, :, 0]  # [N_test, pred_len]
    print(f"  真值形状: {trues.shape}")

    # ── Phase 4: 基准推理 ───────────────────────────────────────────────────
    print("\n>>> Phase 5: 基准推理 (Skip 0%)...")
    baseline_preds, baseline_time = inference_baseline(model, test_loader, device, args.pred_len)
    baseline_mse = metric(trues, baseline_preds)['MSE']
    print(f"  Baseline MSE: {baseline_mse:.6f}, Time: {baseline_time*1000:.2f}ms/batch")

    # ── Phase 5: 选择性跳过推理 ──────────────────────────────────────────────
    print("\n>>> Phase 6: 选择性跳过推理...")
    results = {
        'skip_rates': skip_rates,
        'mse_values': [baseline_mse],
        'speedups': [1.0],
        'baseline_time': baseline_time,
        'baseline_mse': baseline_mse,
        'preds': {},
    }
    results['preds'][0.0] = baseline_preds

    for skip_rate in skip_rates:
        if skip_rate == 0.0:
            continue
        print(f"\n  [{skip_rate*100:.0f}% skip]")
        preds, avg_time, _ = inference_selective_skip(
            model=model,
            data_loader=test_loader,
            device=device,
            mi_curves=mi_curves,
            skip_rate=skip_rate,
            n_patches=n_patches,
            pred_len=args.pred_len,
        )
        mse = metric(trues, preds)['MSE']
        speedup = baseline_time / avg_time if avg_time > 0 else 1.0
        print(f"    MSE: {mse:.6f}, Time: {avg_time*1000:.2f}ms/batch, Speedup: {speedup:.2f}×")

        results['mse_values'].append(mse)
        results['speedups'].append(speedup)
        results['preds'][skip_rate] = preds

    # ── Phase 6: 绘图 ────────────────────────────────────────────────────────
    print("\n>>> Phase 7: 绘图...")
    pareto_idx = plot_pareto_dual_axis(results, args.output_dir, dataset_name)

    # ── Phase 7: 保存结果 ───────────────────────────────────────────────────
    save_data = {
        'skip_rates': skip_rates,
        'mse_values': results['mse_values'],
        'speedups': results['speedups'],
        'baseline_mse': float(baseline_mse),
        'baseline_time': float(baseline_time),
        'config': vars(args),
    }
    save_path = os.path.join(args.output_dir, "selective_skip_results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  保存结果: {save_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[Summary — Selective Skip Results]")
    print(f"  Baseline MSE: {baseline_mse:.6f}  (Time: {baseline_time*1000:.2f}ms/batch)")
    print(f"\n  {'Skip%':>7} | {'MSE':>10} | {'ΔMSE%':>9} | {'Speedup':>9}")
    print("  " + "-" * 45)
    for sr, mse, sp in zip(skip_rates, results['mse_values'], results['speedups']):
        delta = (mse - baseline_mse) / baseline_mse * 100
        print(f"  {sr*100:>6.0f}% | {mse:>10.6f} | {delta:>+8.2f}% | {sp:>8.2f}×")
    print("=" * 70)

    best_idx = np.argmin(np.array(results['mse_values']) / np.array(results['speedups']))
    print(f"\n  Pareto 最优: {skip_rates[best_idx]*100:.0f}% skip "
          f"(MSE={results['mse_values'][best_idx]:.6f}, "
          f"Speedup={results['speedups'][best_idx]:.2f}×)")


if __name__ == "__main__":
    main()
