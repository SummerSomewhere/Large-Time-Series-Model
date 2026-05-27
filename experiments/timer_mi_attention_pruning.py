#!/usr/bin/env python3
"""
Timer MI引导Attention剪枝实验

核心方法：低MI Token的Attention分数置零
- 在指定层，对每个Token计算其对其他所有Token的attention分数
- 低MI Token行对应的attention分数强制为0，保留残差连接
- 与SelectiveSkip的本质区别：不做token物理删除，只修改注意力权重

论文风格描述（Layer K）：
  注意力分数掩码（Attention Score Masking）
    - 残差流特征 X^{(K)} ∈ R^{B×N×D}
    - 计算所有Token的MI scores，划分高MI和低MI
    - 对低MI Token的attention分数行置零：A_{low,:} = 0
    - 高MI Token正常参与注意力计算：A_{high,:} = softmax(QK^T / √d)_{high,:}
    - 通过残差连接，低MI Token保持历史特征：X_{low}^{(K+1)} = X_{low}^{(K)}

图：Nature风格 — MSE vs Drop Rate（Pareto Frontier）

Usage:
    python experiments/timer_mi_attention_pruning.py \
        --mi_result_dir ./outputs/timer_mi_ksg_pca/Timer_MI_*/ \
        --root_path ./datasets/ \
        --data_path ETTh1.csv \
        --seq_len 672 \
        --pred_len 96 \
        --patch_len 96 \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --output_dir ./outputs/timer_mi_attention_pruning/
"""

import argparse
import gc
import glob
import json
import os
import sys
import time

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
# SECTION 1: Attention Score Masking Layers
# ═══════════════════════════════════════════════════════════════════════════════

class MaskedAttentionScore(nn.Module):
    """
    注意力分数掩码机制（Attention Score Masking）。

    与标准FullAttention的区别：
      FullAttention: 所有Token的attention分数正常计算
      MaskedAttentionScore: 低MI Token的attention分数行置零，
                            只在softmax之前操作，不破坏softmax的数值稳定性
    """
    def __init__(self, inner_attention: nn.Module, d_model: int, n_heads: int,
                 d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = inner_attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys  = d_keys

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,
                low_mi_mask=None, mi_scores=None):
        """
        Args:
            queries/keys/values: [B, N, D] — 完整序列
            attn_mask: 因果mask
            low_mi_mask: [B, N] bool — True = 低MI Token（需置零）
            mi_scores: [B, N] or None — 原始MI分数（用于可视化）
        Returns:
            V_out: [B, N, D]
            attn: [B, H, N, N]
            logits: [B, H, N, N]
            masked_ratio: float
        """
        B, N, _ = queries.shape
        H = self.n_heads
        d_k = self.d_keys

        Q = self.query_projection(queries).view(B, N, H, -1).transpose(1, 2)
        K = self.key_projection(keys).view(B, N, H, -1).transpose(1, 2)
        V = self.value_projection(values).view(B, N, H, -1).transpose(1, 2)

        scores = torch.einsum("bhxd,bhkd->bhxk", Q, K) / (d_k ** 0.5)

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.mask.unsqueeze(1), -1e9)

        logits = scores.clone()

        if low_mi_mask is not None:
            row_mask = low_mi_mask.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(row_mask, -1e9)
            masked_ratio = (~low_mi_mask).float().mean().item()
        else:
            masked_ratio = 1.0

        A = F.dropout(torch.softmax(scores, dim=-1), p=0.0, training=False)
        V_out = torch.einsum("bhns,bhsd->bhnd", A, V)
        V_out = V_out.transpose(1, 2).contiguous().view(B, N, -1)
        V_out = self.out_projection(V_out)

        return V_out, A, logits, masked_ratio


class MaskedScoreEncoderLayer(nn.Module):
    """
    带注意力分数掩码的EncoderLayer。
    包装原始层，在forward时注入低MI掩码。
    """
    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.original_layer = original_layer
        self.d_model = original_layer.norm1.normalized_shape[0]
        self.d_ff = original_layer.conv1.out_channels
        self.n_heads = original_layer.attention.n_heads
        self.d_keys = original_layer.attention.query_projection.out_features // self.n_heads
        self.d_values = original_layer.attention.key_projection.out_features // self.n_heads

        self.masked_attn = MaskedAttentionScore(
            inner_attention=original_layer.attention.inner_attention,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_keys=self.d_keys,
            d_values=self.d_values,
        )
        self.masked_attn.query_projection.weight.data = original_layer.attention.query_projection.weight.data.clone()
        self.masked_attn.query_projection.bias.data   = original_layer.attention.query_projection.bias.data.clone()
        self.masked_attn.key_projection.weight.data   = original_layer.attention.key_projection.weight.data.clone()
        self.masked_attn.key_projection.bias.data     = original_layer.attention.key_projection.bias.data.clone()
        self.masked_attn.value_projection.weight.data = original_layer.attention.value_projection.weight.data.clone()
        self.masked_attn.value_projection.bias.data   = original_layer.attention.value_projection.bias.data.clone()
        self.masked_attn.out_projection.weight.data  = original_layer.attention.out_projection.weight.data.clone()
        self.masked_attn.out_projection.bias.data    = original_layer.attention.out_projection.bias.data.clone()
        self.masked_attn.mask_flag = True

    def forward(self, x, attn_mask=None, tau=None, delta=None,
                low_mi_mask=None, mi_scores=None):
        residual = x

        if low_mi_mask is not None:
            attn_out, attn, logits, masked_ratio = self.masked_attn(
                x, x, x, attn_mask=attn_mask, tau=tau, delta=delta,
                low_mi_mask=low_mi_mask, mi_scores=mi_scores
            )
            x = residual + self.original_layer.dropout(attn_out)
        else:
            attn_out, attn, logits = self.original_layer.attention(
                x, x, x, attn_mask=attn_mask, tau=tau, delta=delta
            )
            x = x + self.original_layer.dropout(attn_out)
            masked_ratio = 1.0

        x = self.original_layer.norm1(x)

        y = self.original_layer.dropout(
            self.original_layer.activation(
                self.original_layer.conv1(x.transpose(-1, 1))
            ).transpose(-1, 1))
        y = self.original_layer.dropout(self.original_layer.conv2(y.transpose(-1, 1)))
        x = self.original_layer.norm2(x + y)

        return x, attn, logits, masked_ratio


class MaskedScoreDecoder(nn.Module):
    """
    包装Timer的Decoder，在每层应用低MI的注意力分数置零。
    """
    def __init__(self, original_decoder, mi_curves_per_layer: dict,
                 drop_rate: float = 0.5):
        super().__init__()
        self.original_decoder = original_decoder
        self.mi_curves = mi_curves_per_layer
        self.drop_rate = drop_rate
        self.n_layers = len(original_decoder.attn_layers)
        self.device = None

        self.low_mi_masks = {}
        for li, mi_curve in mi_curves_per_layer.items():
            if mi_curve is None or len(mi_curve) == 0:
                self.low_mi_masks[li] = None
                continue
            mi_t = torch.tensor(mi_curve, dtype=torch.float32)
            threshold = torch.quantile(mi_t, drop_rate)
            self.low_mi_masks[li] = (mi_t < threshold).numpy()

    def _get_layer_mask(self, layer_idx: int, B: int, N: int) -> torch.Tensor:
        static_mask = self.low_mi_masks.get(layer_idx, None)
        if static_mask is None:
            return None
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = torch.tensor(static_mask, dtype=torch.bool, device=self.device)
        return mask.unsqueeze(0).expand(B, -1)

    def forward(self, x, attn_mask=None, tau=None, delta=None,
                has_prototype: bool = False, output_hidden_states: bool = False,
                layer_guide=None, output_attention_override=False):
        if self.device is None:
            self.device = x.device

        B, N, D = x.shape
        attns = []
        logits_list = []
        hidden_states = [] if output_hidden_states else None

        for li, original_layer in enumerate(self.original_decoder.attn_layers):
            low_mi_mask = self._get_layer_mask(li, B, N)

            masked_layer = MaskedScoreEncoderLayer(original_layer)

            if low_mi_mask is not None:
                masked_layer = masked_layer.to(x.device)
                x_new, attn, logits, masked_ratio = masked_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta,
                    low_mi_mask=low_mi_mask
                )
            else:
                x_new, attn, logits = original_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta
                )
            x = x_new

            attns.append(attn)
            if logits is not None:
                logits_list.append(logits)
            if output_hidden_states:
                hidden_states.append(x)

        if self.original_decoder.norm is not None:
            x = self.original_decoder.norm(x)
            if output_hidden_states:
                hidden_states.append(x)

        if output_hidden_states:
            return x, attns, logits_list if logits_list else None, hidden_states
        return x, attns, logits_list if logits_list else None


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

def load_mi_from_json(mi_result_dir: str, n_layers: int):
    pattern = os.path.join(mi_result_dir, "global_mi_peaks_*.json")
    matched = glob.glob(pattern)
    if not matched:
        all_json = sorted(glob.glob(os.path.join(mi_result_dir, "*.json")),
                          key=os.path.getmtime)
        if not all_json:
            raise FileNotFoundError(f"未找到任何JSON文件: {mi_result_dir}")
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
            raise ValueError(f"JSON中未找到layer={li}，可用键: {list(layers_dict.keys())}")
        mi_curves[li] = np.array(layers_dict[key]["hsic_curve"], dtype=np.float64)
    return mi_curves, data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Inference Functions
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def inference_masked_score(model: Model, data_loader, device,
                           mi_curves: dict, drop_rate: float,
                           pred_len: int, output_dir: str = None):
    """
    低MI Token的attention分数置零进行推理。
    """
    core = _unwrap(model)
    preds_list = []
    inference_times = []

    masked_decoder = MaskedScoreDecoder(
        original_decoder=core.decoder,
        mi_curves_per_layer=mi_curves,
        drop_rate=drop_rate,
    )
    masked_decoder = masked_decoder.to(device)
    masked_decoder.eval()

    for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(data_loader,
                                                       desc=f"Drop {drop_rate:.0%}", leave=False):
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

        decoder_out = masked_decoder(
            dec_in,
            attn_mask=mask,
            has_prototype=False,
            output_hidden_states=False,
        )

        dec_out = decoder_out[0] if isinstance(decoder_out, tuple) else decoder_out

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


@torch.no_grad()
def inference_baseline(model: Model, data_loader, device, pred_len: int):
    """基准推理（无剪枝）。"""
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
# SECTION 5: Nature-Style Figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_nature_style(results: dict, output_dir: str,
                      dataset_name: str = "ETTh1",
                      seq_len: int = 672,
                      pred_len: int = 96,
                      patch_len: int = 96):
    """
    Nature风格图：
      - 简洁的黑白配色
      - 清晰的中文/英文标注
      - 统一的字体和尺寸
    """
    os.makedirs(output_dir, exist_ok=True)

    drop_rates = results['drop_rates']
    mse_values = results['mse_values']
    mae_values = results['mae_values']
    speedups = results['speedups']
    baseline_mse = results['baseline_mse']
    baseline_time = results['baseline_time']

    x = [d * 100 for d in drop_rates]

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.linewidth': 1.0,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── 图a: MSE vs Drop Rate ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, mse_values, 'o-', color='#222222', linewidth=2.0,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5,
            markeredgecolor='#222222', zorder=3)
    ax.fill_between(x, 0, mse_values, alpha=0.06, color='#444444')
    ax.axhline(y=baseline_mse, color='#888888', linestyle='--',
               linewidth=1.2, label=f'Baseline ({baseline_mse:.4f})', zorder=2)
    ax.set_xlabel("Token Drop Rate (%)")
    ax.set_ylabel("MSE")
    ax.set_title("a  MSE vs Drop Rate", fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── 图b: MSE Change ─────────────────────────────────────────────────────
    ax = axes[1]
    mse_change = [(m - baseline_mse) / baseline_mse * 100 for m in mse_values]
    colors = ['#2E7D32' if c <= 5 else '#C62828' if c >= 10 else '#F57C00'
              for c in mse_change]
    bars = ax.bar(x, mse_change, color=colors, width=7, zorder=3,
                  edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1.2)
    ax.set_xlabel("Token Drop Rate (%)")
    ax.set_ylabel("MSE Change (%)")
    ax.set_title("b  MSE Change vs Baseline", fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25, linestyle=':', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, mse_change):
        ax.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=8, color='#333333')

    # ── 图c: Speedup ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.bar(x, speedups, color='#1565C0', width=7, zorder=3,
           edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.axhline(y=1.0, color='black', linewidth=1.2, label='Baseline (1.0×)')
    ax.set_xlabel("Token Drop Rate (%)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("c  Throughput Speedup", fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=':', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (xi, su) in enumerate(zip(x, speedups)):
        ax.annotate(f'{su:.2f}×',
                    xy=(xi, su),
                    ha='center', va='bottom',
                    fontsize=8, color='#333333')

    fig.suptitle(
        f"Timer MI-Guided Attention Pruning — {dataset_name}\n"
        f"(Seq={seq_len}, Pred={pred_len}, Patch={patch_len})",
        fontsize=13, y=1.02
    )
    fig.tight_layout()

    fig_path = os.path.join(output_dir, "attention_pruning_nature.png")
    fig_path_pdf = os.path.join(output_dir, "attention_pruning_nature.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [Plot] Nature figures: {fig_path}")

    # ── 图d: 帕累托散点图 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    scatter = ax.scatter(speedups, mse_values, c=x, cmap='RdYlGn_r',
                         s=100, zorder=3, edgecolors='black', linewidths=0.8)
    for i, (su, mi, xi) in enumerate(zip(speedups, mse_values, x)):
        ax.annotate(f'{xi:.0f}%', (su, mi),
                    textcoords="offset points", xytext=(8, 4),
                    ha='left', fontsize=9, color='#333333')

    pareto_idx = np.argmin(np.array(mse_values) / np.array(speedups))
    ax.scatter([speedups[pareto_idx]], [mse_values[pareto_idx]],
               color='#FFD700', s=300, zorder=5, marker='*',
               edgecolors='black', linewidths=1.5,
               label=f'Pareto Optimal ({drop_rates[pareto_idx]*100:.0f}% drop)')
    ax.set_xlabel("Throughput Speedup (×)")
    ax.set_ylabel("MSE")
    ax.set_title("Pareto Frontier: MSE vs Speedup", fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Drop Rate (%)", fontsize=9)
    fig.tight_layout()

    fig_path2 = os.path.join(output_dir, "attention_pruning_pareto.png")
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [Plot] Pareto figure: {fig_path2}")

    return pareto_idx


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Timer MI引导Attention剪枝实验 — 低MI Token的Attention分数置零")
    parser.add_argument("--mi_result_dir", type=str, required=True,
                        help="timer_mi_ksg_pca.py输出目录")
    parser.add_argument("--root_path", type=str, default="./datasets/",
                        help="数据根目录")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv",
                        help="数据文件名")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=None,
                        help="解码器输入长度（默认为seq_len）")
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt",
                        help="Timer权重路径")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop_rates", type=str, default="0.0,0.10,0.20,0.30,0.40,0.50",
                        help="丢弃的低MI Token比例（逗号分隔）")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/timer_mi_attention_pruning/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    drop_rates = [float(x) for x in args.drop_rates.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = args.data_path.replace(".csv", "")

    print("=" * 70)
    print(f"Timer MI引导Attention剪枝实验 — {dataset_name}")
    print("=" * 70)
    print(f"  MI结果目录 : {args.mi_result_dir}")
    print(f"  数据集    : {args.root_path}/{args.data_path}")
    print(f"  序列长度  : {args.seq_len}")
    print(f"  预测长度  : {args.pred_len}")
    print(f"  Patch长度 : {args.patch_len}")
    print(f"  丢弃比例  : {drop_rates}")
    print(f"  设备      : {device}")
    print("=" * 70)

    # Phase 1: 加载MI曲线
    print("\n>>> Phase 1: 加载MI曲线...")
    mi_curves, mi_meta = load_mi_from_json(args.mi_result_dir, args.e_layers)
    n_patches = len(list(mi_curves.values())[0])
    print(f"  层数: {args.e_layers}, 每层Patch数: {n_patches}")

    # Phase 2: 加载数据
    print("\n>>> Phase 2: 加载数据...")
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

    # Phase 3: 加载模型
    print("\n>>> Phase 3: 加载Timer模型...")
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

    # Phase 4: 准备真值
    print("\n>>> Phase 4: 准备真值...")
    trues_list = []
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(test_loader,
                                                          desc="准备真值", leave=False):
            sy = seq_y.float().numpy()
            trues_list.append(sy)
    trues = np.concatenate(trues_list, axis=0)[:, :, 0]
    print(f"  真值形状: {trues.shape}")

    # Phase 5: 基准推理
    print("\n>>> Phase 5: 基准推理 (Drop 0%)...")
    baseline_preds, baseline_time = inference_baseline(model, test_loader, device, args.pred_len)
    baseline_mse = metric(trues, baseline_preds)['MSE']
    baseline_mae = metric(trues, baseline_preds)['MAE']
    print(f"  Baseline MSE: {baseline_mse:.6f}, MAE: {baseline_mae:.6f}, "
          f"Time: {baseline_time*1000:.2f}ms/batch")

    # Phase 6: Attention剪枝推理
    print("\n>>> Phase 6: Attention剪枝推理...")
    results = {
        'drop_rates': drop_rates,
        'mse_values': [baseline_mse],
        'mae_values': [baseline_mae],
        'speedups': [1.0],
        'baseline_time': baseline_time,
        'baseline_mse': baseline_mse,
        'preds': {},
    }
    results['preds'][0.0] = baseline_preds

    for drop_rate in drop_rates:
        if drop_rate == 0.0:
            continue
        print(f"\n  [Drop {drop_rate*100:.0f}%]")
        preds, avg_time = inference_masked_score(
            model=model,
            data_loader=test_loader,
            device=device,
            mi_curves=mi_curves,
            drop_rate=drop_rate,
            pred_len=args.pred_len,
        )
        mse = metric(trues, preds)['MSE']
        mae = metric(trues, preds)['MAE']
        speedup = baseline_time / avg_time if avg_time > 0 else 1.0
        print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}, "
              f"Time: {avg_time*1000:.2f}ms/batch, Speedup: {speedup:.2f}×")

        results['mse_values'].append(mse)
        results['mae_values'].append(mae)
        results['speedups'].append(speedup)
        results['preds'][drop_rate] = preds

    # Phase 7: 绘图
    print("\n>>> Phase 7: 绘图...")
    pareto_idx = plot_nature_style(
        results, args.output_dir, dataset_name,
        seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len
    )

    # Phase 8: 保存结果
    save_data = {
        'drop_rates': drop_rates,
        'mse_values': results['mse_values'],
        'mae_values': results['mae_values'],
        'speedups': results['speedups'],
        'baseline_mse': float(baseline_mse),
        'baseline_mae': float(baseline_mae),
        'baseline_time': float(baseline_time),
        'config': vars(args),
    }
    save_path = os.path.join(args.output_dir, "attention_pruning_results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  保存结果: {save_path}")

    # Summary
    print("\n" + "=" * 70)
    print("[Summary — Attention Pruning Results]")
    print(f"  Baseline MSE: {baseline_mse:.6f}  (Time: {baseline_time*1000:.2f}ms/batch)")
    print(f"\n  {'Drop%':>7} | {'MSE':>10} | {'MAE':>10} | {'ΔMSE%':>9} | {'Speedup':>9}")
    print("  " + "-" * 58)
    for dr, mse, mae, sp in zip(drop_rates, results['mse_values'],
                                results['mae_values'], results['speedups']):
        delta = (mse - baseline_mse) / baseline_mse * 100
        print(f"  {dr*100:>6.0f}% | {mse:>10.6f} | {mae:>10.6f} | {delta:>+8.2f}% | {sp:>8.2f}×")
    print("=" * 70)

    best_idx = np.argmin(np.array(results['mse_values']) / np.array(results['speedups']))
    print(f"\n  Pareto最优: {drop_rates[best_idx]*100:.0f}% drop "
          f"(MSE={results['mse_values'][best_idx]:.6f}, "
          f"Speedup={results['speedups'][best_idx]:.2f}×)")


if __name__ == "__main__":
    main()
