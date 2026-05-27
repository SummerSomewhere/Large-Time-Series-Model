#!/usr/bin/env python3
"""
Timer MI-Guided Feature Recycling 实验。

核心思想（模仿 LLM 的 Recursive Representation / RR 实验）：
    在 Transformer 解码过程中，当生成到特定 token 位置时，
    用之前提取的高 MI patch 表征替换当前低 MI patch 的表征，
    从而验证"高 MI 表征是否可以被回收利用来提升预测精度"。

在时序预测场景下的适配：
    - "高 MI patch" = 与未来真值 Y 互信息高的历史 patch
        → I(H_patch, Y) 高的 patch 位置
    - "低 MI patch" = 与未来真值 Y 互信息低的历史 patch
        → I(H_patch, Y) 低的 patch 位置
    - Feature Recycling：在前向传播过程中，
        将低 MI patch 的 hidden state 替换为高 MI patch 的 hidden state，
        观察是否提升预测性能。

实验设计（对照实验）：
    [基线]    Timer 原版前向传播（无任何干预）
    [Recycle] 高 MI patch 的 hidden state → 替换低 MI patch 的 hidden state（按 patch 索引）
    [Random]  随机选择 patch 进行替换（排除高 MI patch 自身）
    [Inverse] 低 MI patch 的 hidden state → 替换高 MI patch 的 hidden state（反向验证）

Usage:
    python experiments/timer_mi_feature_recycling.py \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --root_path ./datasets/ --data ETTh1 --data_path ETTh1.csv \
        --out_dir ./results/timer_mi_feature_recycling/ \
        --mi_result_dir ./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108 \
        --model_id etth1

Dependencies:
    pip install torch numpy matplotlib tqdm scikit-learn scipy pandas
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark


# ============================================================================
# Nature Figure Style
# ============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary":  "#3775BA",
    "green_3":       "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "orange":          "#E07B39",
    "neutral_light":   "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}


def add_panel_label(ax, label, x=-0.08, y=1.06, fontsize=10,
                    fontweight="bold", color="black"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha="left", va="bottom")


def finalize_figure(fig, out_path, dpi=300, pad=1.2):
    from pathlib import Path
    fig.tight_layout(pad=pad)
    base = Path(out_path)
    os.makedirs(base.parent, exist_ok=True)
    base = base.with_suffix("")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {base}.{{svg,pdf,png}}")


# ============================================================================
# Config & Model Builder
# ============================================================================

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(ckpt_path: str, patch_len: int, stride: int,
                d_model: int, d_ff: int, e_layers: int,
                n_heads: int, dropout: float,
                seq_len: int, pred_len: int,
                enc_in: int = 1) -> Model:
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
        output_attention=False,
        distil=True,
        use_revin=False,
        seq_len=seq_len,
        pred_len=pred_len,
        d_layers=1,
        factor=1,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
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


# ============================================================================
# Hook-Based Feature Recycling Core
# ============================================================================

class FeatureRecycler:
    """
    通过 register_forward_hook 在指定 Transformer 层注入/替换 hidden states。

    用法：
        recycler = FeatureRecycler(model, ...)
        recycler.register_hooks()

        # 第一次前向传播：提取高 MI patch 表征
        high_mi_hidden = recycler.extract_hidden_states(inputs)

        # 第二次前向传播：将高 MI 表征注入到低 MI patch 位置
        recycler.set_replacement(high_mi_hidden, low_mi_patch_indices)
        outputs = recycler.forward_with_recycling(inputs)

        recycler.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        e_layers: int,
        n_patches: int,
        recycle_layer: int,
        high_mi_patches: list[int],
        low_mi_patches: list[int],
        blend_ratio: float = 1.0,
        seed: int = 42,
        device=None,
    ):
        self.model = model
        self.e_layers = e_layers
        self.n_patches = n_patches
        self.recycle_layer = recycle_layer
        self.high_mi_patches = high_mi_patches
        self.low_mi_patches = low_mi_patches
        self.blend_ratio = blend_ratio
        self.seed = seed
        self.device = device or next(model.parameters()).device

        self._hooks = []
        self._extracted_hidden = None   # 保存第一次前向的高 MI 表征
        self._replacement_cache = {}     # layer_idx -> tensor
        self._recycle_mode = False
        self._extract_mode = True
        self._current_layer_idx = 0

        self._find_layers()

    def _find_layers(self):
        """找到 Transformer decoder attention layers."""
        core = _unwrap(self.model)
        if hasattr(core, 'decoder') and hasattr(core.decoder, 'attn_layers'):
            self.layers = core.decoder.attn_layers
        elif hasattr(core, 'model') and hasattr(core.model, 'layers'):
            self.layers = core.model.layers
        else:
            raise ValueError(f"Unsupported Timer architecture: {type(core)}")

    def register_hooks(self):
        """在 recycle_layer 注册前向 hook 和后向 hook。"""
        self._remove_hooks()

        def forward_pre_hook(module, inputs):
            if not self._recycle_mode:
                return inputs
            # 在进入 recycle_layer 之前，注入替换后的 hidden states
            h = inputs[0]  # [BM, N, D]
            if self._extracted_hidden is None or self._recycle_mode is False:
                return inputs

            B = h.shape[0]   # BM (batch * n_vars)
            D = h.shape[2]

            # 从缓存中获取该层的替换表征
            replacement = self._replacement_cache.get(self._current_layer_idx)
            if replacement is None:
                return inputs

            # 在指定 patch 位置替换
            h_new = h.clone()
            for patch_idx in self.low_mi_patches:
                if patch_idx < self.n_patches:
                    # 用 blend_ratio 混合：new = ratio * high_mi + (1-ratio) * original
                    h_new[:, patch_idx, :] = (
                        self.blend_ratio * replacement[:, patch_idx, :] +
                        (1 - self.blend_ratio) * h[:, patch_idx, :]
                    )
                    # 如果是 recycle_layer，用 high_mi_hidden 中对应 patch 的表征
                    if self._current_layer_idx == self.recycle_layer:
                        # 替换
                        if replacement.shape[1] > patch_idx:
                            h_new[:, patch_idx, :] = replacement[:, patch_idx, :]

            return (h_new,) + inputs[1:]

        def forward_hook(module, inputs, outputs):
            # 提取指定层的高 MI patch 表征
            if not self._extract_mode:
                return outputs
            h = outputs[0] if isinstance(outputs, tuple) else outputs  # [BM, N, D]
            if self._extracted_hidden is None:
                self._extracted_hidden = h.detach().clone()
            return outputs

        hook_layer_idx = min(self.recycle_layer, len(self.layers) - 1)
        pre_hook = self.layers[hook_layer_idx].register_forward_pre_hook(forward_pre_hook)
        out_hook = self.layers[hook_layer_idx].register_forward_hook(forward_hook)
        self._hooks = [pre_hook, out_hook]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def set_replacement(self, replacement_tensor: torch.Tensor):
        """设置替换用的表征张量 [BM, N, D]"""
        self._replacement_cache[self.recycle_layer] = replacement_tensor

    def enable_recycle_mode(self):
        self._recycle_mode = True
        self._extract_mode = False

    def enable_extract_mode(self):
        self._recycle_mode = False
        self._extract_mode = True

    def reset(self):
        self._extracted_hidden = None
        self._replacement_cache = {}
        self._recycle_mode = False
        self._extract_mode = True


# ============================================================================
# Alternative: Direct Patch Replacement (No Hooks — More Reliable)
# ============================================================================

class DirectFeatureRecycler:
    """
    直接在前向传播过程中替换 patch 表征，不依赖 hook。
    更稳定，适用于需要多次干预的场景。

    策略：
        1. 正常前向传播，提取各层 hidden states
        2. 找到高 MI patch 在 recycle_layer 的表征 H_high
        3. 用 H_high 替换低 MI patch 位置的表征，组成新的 hidden sequence
        4. 跳过 recycle_layer，继续后续层的前向传播
    """

    def __init__(
        self,
        model: nn.Module,
        e_layers: int,
        n_patches: int,
        recycle_layer: int,
        high_mi_patches: list[int],
        low_mi_patches: list[int],
        blend_ratio: float = 1.0,
        seed: int = 42,
        device=None,
    ):
        self.model = model
        self.e_layers = e_layers
        self.n_patches = n_patches
        self.recycle_layer = recycle_layer
        self.high_mi_patches = high_mi_patches
        self.low_mi_patches = low_mi_patches
        self.blend_ratio = blend_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = device or next(model.parameters()).device
        self._find_layers()

    def _find_layers(self):
        core = _unwrap(self.model)
        if hasattr(core, 'decoder') and hasattr(core.decoder, 'attn_layers'):
            self.layers = core.decoder.attn_layers
        elif hasattr(core, 'model') and hasattr(core.model, 'layers'):
            self.layers = core.model.layers
        else:
            raise ValueError(f"Unsupported model architecture")

    def forward_with_recycling(
        self,
        dec_in: torch.Tensor,   # [BM, N, D] patch embedding
        mask,                   # causal mask
    ) -> torch.Tensor:
        """
        执行带特征回收的前向传播。

        流程：
            dec_in -> layer 0 -> ... -> layer (recycle_layer-1)
                   -> [用高 MI 表征替换低 MI patch]
                   -> layer (recycle_layer+1) -> ... -> layer (e_layers-1)
        """
        h = dec_in
        high_mi_hidden = None
        recycle_layer_reached = False

        for li, layer_module in enumerate(self.layers):
            if recycle_layer_reached:
                recycle_layer_reached = False
                continue
            h, _, _ = layer_module(h, attn_mask=mask)

            if li == self.recycle_layer:
                high_mi_hidden = h.detach().clone()
                h_replaced = h.clone()

                for patch_idx in self.low_mi_patches:
                    if patch_idx >= self.n_patches:
                        continue
                    # 随机选一个高 MI patch 作为来源
                    src_idx = self.rng.choice(self.high_mi_patches)
                    if src_idx >= self.n_patches:
                        src_idx = self.high_mi_patches[0] if self.high_mi_patches else patch_idx
                    # blend：高 MI 表征 + 低 MI 表征
                    h_replaced[:, patch_idx, :] = (
                        self.blend_ratio * h[:, src_idx, :].clone()
                        + (1 - self.blend_ratio) * h[:, patch_idx, :]
                    )
                h = h_replaced
                recycle_layer_reached = True

        return h, high_mi_hidden

    def forward_random_recycling(
        self,
        dec_in: torch.Tensor,
        mask,
        exclude_patches: list[int],
    ) -> torch.Tensor:
        """
        随机替换对照：随机选择 patch 进行替换（排除 exclude_patches）。
        替换后跳过 recycle_layer，继续其余层。
        """
        h = dec_in
        recycle_layer_reached = False
        for li, layer_module in enumerate(self.layers):
            if recycle_layer_reached:
                recycle_layer_reached = False
                continue
            h, _, _ = layer_module(h, attn_mask=mask)

            if li == self.recycle_layer:
                h_replaced = h.clone()
                available = [p for p in range(self.n_patches) if p not in exclude_patches]
                if len(available) >= len(self.low_mi_patches):
                    random_patches = list(
                        self.rng.choice(available, size=len(self.low_mi_patches), replace=False)
                    )
                else:
                    random_patches = available[:len(self.low_mi_patches)]

                for src_idx, dst_idx in zip(random_patches, self.low_mi_patches):
                    if dst_idx < self.n_patches and src_idx < self.n_patches:
                        h_replaced[:, dst_idx, :] = h[:, src_idx, :]
                h = h_replaced
                recycle_layer_reached = True

        return h, None

    def forward_inverse_recycling(
        self,
        dec_in: torch.Tensor,
        mask,
    ) -> torch.Tensor:
        """
        反向替换对照：将低 MI patch 表征替换到高 MI patch 位置。
        替换后跳过 recycle_layer，继续其余层。
        """
        h = dec_in
        recycle_layer_reached = False
        for li, layer_module in enumerate(self.layers):
            if recycle_layer_reached:
                recycle_layer_reached = False
                continue
            h, _, _ = layer_module(h, attn_mask=mask)

            if li == self.recycle_layer:
                h_replaced = h.clone()
                for patch_idx in self.high_mi_patches:
                    if patch_idx >= self.n_patches:
                        continue
                    if len(self.low_mi_patches) == 0:
                        continue
                    src_idx = self.low_mi_patches[0]
                    if src_idx < self.n_patches:
                        h_replaced[:, patch_idx, :] = (
                            self.blend_ratio * h[:, src_idx, :].clone()
                            + (1 - self.blend_ratio) * h[:, patch_idx, :]
                        )
                h = h_replaced
                recycle_layer_reached = True

        return h, None


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(
    model,
    test_loader,
    n_vars: int,
    patch_len: int,
    seq_len: int,
    pred_len: int,
    n_patches: int,
    device,
    args,
    mi_summary: dict,
):
    """
    运行特征回收实验的四种对照条件。
    返回各条件的预测误差。

    Baseline 使用 model.forecast()（Timer 官方单步评估），
    其他条件使用手动 decoder 前向传播 + 干预。
    """
    from utils.masking import TriangularCausalMask
    results = {}

    # 从 MI summary 中提取高/低 MI patch 信息
    e_layers = args.e_layers
    all_high = set()
    all_low = set()
    layer_mi_curves = {}

    for li in range(e_layers):
        layer_data = mi_summary['layers'].get(str(li), {})
        hsic = layer_data.get('hsic_curve', [])
        layer_mi_curves[li] = hsic
        if hsic:
            sorted_idx = np.argsort(hsic)
            all_high.update([int(sorted_idx[-1]), int(sorted_idx[-2])])
            all_low.update([int(sorted_idx[0]), int(sorted_idx[1])])

    # 全局高/低 MI patch（跨层均值）
    global_mi = np.mean(
        [layer_mi_curves[li] for li in range(e_layers) if layer_mi_curves.get(li)],
        axis=0
    )
    if len(global_mi) > 0:
        sorted_global = np.argsort(global_mi)
        global_high_patches = [int(sorted_global[-1]), int(sorted_global[-2]), int(sorted_global[-3])]
        global_low_patches = [int(sorted_global[0]), int(sorted_global[1]), int(sorted_global[2])]
    else:
        global_high_patches = list(all_high)[:3] if all_high else []
        global_low_patches = list(all_low)[:3] if all_low else []

    print(f"\n  [MI Analysis] 全局高 MI patches: {global_high_patches}")
    print(f"  [MI Analysis] 全局低 MI patches: {global_low_patches}")
    print(f"  [MI Analysis] 跨层高 MI patches: {sorted(all_high)}")
    print(f"  [MI Analysis] 跨层低 MI patches: {sorted(all_low)}")

    # 确定 recycle_layer（默认使用最后一层）
    recycle_layer = args.recycle_layer if args.recycle_layer >= 0 else e_layers - 1
    print(f"  [Recycling] 替换层: L{recycle_layer}")

    # 获取 decoder norm 层（用于手动前向传播）
    core = _unwrap(model)
    decoder_norm = core.decoder.norm if hasattr(core.decoder, 'norm') and core.decoder.norm is not None else None

    # 统计信息
    metrics = {
        "baseline": {"mse": [], "mae": []},
        "recycle":  {"mse": [], "mae": []},
        "random":   {"mse": [], "mae": []},
        "inverse":  {"mse": [], "mae": []},
    }

    total_batches = 0
    max_batches = args.max_eval_batches

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(test_loader, desc="评估"):
            B = seq_x.shape[0]
            total_batches += 1
            if max_batches > 0 and total_batches > max_batches:
                break

            sx = seq_x.float().to(device)
            sy = seq_y.float().to(device)
            sx_mark = seq_x_mark.float().to(device)
            sy_mark = seq_y_mark.float().to(device)

            # ── [1] Baseline：用 Timer 官方 forecast 方法（单步非自回归）───────────
            # 构建 dec_inp（label_len 前缀 + pred_len 全零）
            dec_inp = torch.zeros_like(sy[:, -pred_len:, :]).float()
            dec_inp = torch.cat([sy[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            pred_base = model(sx, sx_mark, dec_inp, sy_mark,
                             output_attention_override=False)
            if isinstance(pred_base, (tuple, list)):
                pred_base = pred_base[0]
            # 提取预测部分
            pred_base = pred_base[:, -pred_len:, :]
            if pred_base.shape[1] < pred_len:
                pad = torch.zeros(B, pred_len - pred_base.shape[1], n_vars, device=device)
                pred_base = torch.cat([pred_base, pad], dim=1)
            gt = sy[:, :pred_len, :]

            mse_base = torch.mean((pred_base - gt) ** 2).item()
            mae_base = torch.mean(torch.abs(pred_base - gt)).item()
            metrics["baseline"]["mse"].append(mse_base)
            metrics["baseline"]["mae"].append(mae_base)

            # ── 手动 decoder 前向传播所需数据 ─────────────────────────────────────
            # 标准化（与 Timer 原版 forecast 一致）
            means = sx.mean(dim=1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            stdev = torch.clamp_min(stdev, 1e-5)
            x_norm = (sx - means) / stdev

            x2 = x_norm.permute(0, 2, 1)  # [B, M, T]
            dec_in, n_vars_actual = core.enc_embedding(x2)  # [B*M, N, D]
            BM, N, D = dec_in.shape
            mask_base = TriangularCausalMask(BM, N, device=device)

            def _manual_forward(h, skip_layer=-1, blend_fn=None):
                """手动过 decoder 层，可选跳过某一层并注入干预函数。"""
                h_in = h
                for li, layer_module in enumerate(core.decoder.attn_layers):
                    if li == skip_layer and blend_fn is not None:
                        h, _, _ = layer_module(h, attn_mask=mask_base)
                        h = blend_fn(h)
                    else:
                        h, _, _ = layer_module(h, attn_mask=mask_base)
                if decoder_norm is not None:
                    h = decoder_norm(h)
                return h

            def _proj_and_denorm(h):
                """proj + reshape + denorm"""
                dec_out = core.proj(h)  # [BM, N, pred_len]
                dec_out = dec_out.reshape(B, n_vars, -1).transpose(1, 2)  # [B, pred_len, M]
                pred = dec_out[:, :pred_len, :]
                if pred.shape[1] < pred_len:
                    pad = torch.zeros(B, pred_len - pred.shape[1], n_vars, device=device)
                    pred = torch.cat([pred, pad], dim=1)
                return pred * stdev.expand(-1, pred_len, -1) + means.expand(-1, pred_len, -1)

            # ── [2] Feature Recycling ────────────────────────────────────────────
            rng_rr = np.random.default_rng(args.seed)

            def blend_recycle(h):
                h_replaced = h.clone()
                for patch_idx in global_low_patches:
                    if patch_idx >= n_patches:
                        continue
                    src_idx = rng_rr.choice(global_high_patches)
                    if src_idx >= n_patches:
                        src_idx = global_high_patches[0]
                    h_replaced[:, patch_idx, :] = (
                        args.blend_ratio * h[:, src_idx, :].clone()
                        + (1 - args.blend_ratio) * h[:, patch_idx, :]
                    )
                return h_replaced

            h_recycle = _manual_forward(dec_in, skip_layer=recycle_layer, blend_fn=blend_recycle)
            pred_recycle_denorm = _proj_and_denorm(h_recycle)
            mse_recycle = torch.mean((pred_recycle_denorm - gt) ** 2).item()
            mae_recycle = torch.mean(torch.abs(pred_recycle_denorm - gt)).item()
            metrics["recycle"]["mse"].append(mse_recycle)
            metrics["recycle"]["mae"].append(mae_recycle)

            # ── [3] Random Recycling ──────────────────────────────────────────────
            rng_rand = np.random.default_rng(args.seed + total_batches)

            def blend_random(h):
                h_replaced = h.clone()
                available = [p for p in range(n_patches) if p not in global_high_patches]
                n_sel = min(len(global_low_patches), len(available))
                if n_sel > 0:
                    random_src = list(rng_rand.choice(available, size=n_sel, replace=False))
                    for src_idx, dst_idx in zip(random_src, global_low_patches[:n_sel]):
                        h_replaced[:, dst_idx, :] = h[:, src_idx, :]
                return h_replaced

            h_random = _manual_forward(dec_in, skip_layer=recycle_layer, blend_fn=blend_random)
            pred_random_denorm = _proj_and_denorm(h_random)
            mse_random = torch.mean((pred_random_denorm - gt) ** 2).item()
            mae_random = torch.mean(torch.abs(pred_random_denorm - gt)).item()
            metrics["random"]["mse"].append(mse_random)
            metrics["random"]["mae"].append(mae_random)

            # ── [4] Inverse Recycling ────────────────────────────────────────────
            rng_inv = np.random.default_rng(args.seed)

            def blend_inverse(h):
                h_replaced = h.clone()
                for patch_idx in global_high_patches:
                    if patch_idx >= n_patches:
                        continue
                    if len(global_low_patches) == 0:
                        continue
                    src_idx = global_low_patches[0]
                    h_replaced[:, patch_idx, :] = (
                        args.blend_ratio * h[:, src_idx, :].clone()
                        + (1 - args.blend_ratio) * h[:, patch_idx, :]
                    )
                return h_replaced

            h_inverse = _manual_forward(dec_in, skip_layer=recycle_layer, blend_fn=blend_inverse)
            pred_inverse_denorm = _proj_and_denorm(h_inverse)
            mse_inverse = torch.mean((pred_inverse_denorm - gt) ** 2).item()
            mae_inverse = torch.mean(torch.abs(pred_inverse_denorm - gt)).item()
            metrics["inverse"]["mse"].append(mse_inverse)
            metrics["inverse"]["mae"].append(mae_inverse)

            del h_recycle, h_random, h_inverse
            gc.collect()
            torch.cuda.empty_cache()

    # 汇总
    summary = {}
    for cond in metrics:
        ms = metrics[cond]["mse"]
        ma = metrics[cond]["mae"]
        summary[cond] = {
            "mse_mean": float(np.mean(ms)),
            "mse_std": float(np.std(ms)),
            "mae_mean": float(np.mean(ma)),
            "mae_std": float(np.std(ma)),
            "n_batches": len(ms),
        }

    return summary, global_high_patches, global_low_patches, global_mi


# ============================================================================
# Per-Layer Recycling Sweep
# ============================================================================

def sweep_recycle_layers(
    model,
    test_loader,
    n_vars: int,
    patch_len: int,
    seq_len: int,
    pred_len: int,
    n_patches: int,
    device,
    args,
    mi_summary: dict,
):
    """
    逐层 sweep：测试在哪些层进行特征回收效果最好。
    """
    from utils.masking import TriangularCausalMask
    e_layers = args.e_layers

    # 获取全局高/低 MI patches
    all_high = set()
    all_low = set()
    layer_mi_curves = {}

    for li in range(e_layers):
        layer_data = mi_summary['layers'].get(str(li), {})
        hsic = layer_data.get('hsic_curve', [])
        layer_mi_curves[li] = hsic
        if hsic:
            sorted_idx = np.argsort(hsic)
            all_high.update([int(sorted_idx[-1]), int(sorted_idx[-2])])
            all_low.update([int(sorted_idx[0]), int(sorted_idx[1])])

    global_mi = np.mean(
        [layer_mi_curves[li] for li in range(e_layers) if layer_mi_curves.get(li)],
        axis=0
    )
    if len(global_mi) > 0:
        sorted_global = np.argsort(global_mi)
        global_high = [int(sorted_global[-1]), int(sorted_global[-2]), int(sorted_global[-3])]
        global_low = [int(sorted_global[0]), int(sorted_global[1]), int(sorted_global[2])]
    else:
        global_high = list(all_high)[:3] if all_high else []
        global_low = list(all_low)[:3] if all_low else []

    print(f"\n  [Sweep] 高 MI patches: {global_high}, 低 MI patches: {global_low}")

    sweep_results = {}
    for target_layer in range(e_layers):
        print(f"\n  [Sweep L{target_layer}]")

        recycler = DirectFeatureRecycler(
            model=model, e_layers=e_layers, n_patches=n_patches,
            recycle_layer=target_layer,
            high_mi_patches=global_high,
            low_mi_patches=global_low,
            blend_ratio=args.blend_ratio,
            seed=args.seed, device=device,
        )

        ms_list, ma_list = [], []
        total_batches = 0

        with torch.no_grad():
            for seq_x, seq_y, _, _ in tqdm(
                test_loader, desc=f"Sweep L{target_layer}", leave=False
            ):
                B = seq_x.shape[0]
                total_batches += 1
                if args.max_eval_batches > 0 and total_batches > args.max_eval_batches:
                    break

                sx = seq_x.float().to(device)
                sy = seq_y.float().to(device)
                means = sx.mean(dim=1, keepdim=True).detach()
                stdev = torch.sqrt(
                    torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5
                ).detach()
                stdev = torch.clamp_min(stdev, 1e-5)
                x_norm = (sx - means) / stdev
                x2 = x_norm.permute(0, 2, 1)

                core = _unwrap(model)
                dec_in, n_vars_actual = core.enc_embedding(x2)
                BM, N, D = dec_in.shape
                derived_n_vars = BM // B
                mask_base = TriangularCausalMask(BM, N, device=device)

                h_recycle, _ = recycler.forward_with_recycling(dec_in, mask_base)
                dec_out = core.proj(h_recycle)
                dec_out = dec_out.reshape(B, n_vars_actual, -1).transpose(1, 2)
                pred = dec_out[:, :pred_len, :]
                if pred.shape[1] < pred_len:
                    pad = torch.zeros(B, pred_len - pred.shape[1], n_vars_actual, device=device)
                    pred = torch.cat([pred, pad], dim=1)
                pred_denorm = pred * stdev.expand(-1, pred_len, -1) + means.expand(-1, pred_len, -1)
                gt = sy[:, :pred_len, :]

                ms_list.append(torch.mean((pred_denorm - gt) ** 2).item())
                ma_list.append(torch.mean(torch.abs(pred_denorm - gt)).item())

                del dec_in, h_recycle
                gc.collect()

        sweep_results[target_layer] = {
            "mse_mean": float(np.mean(ms_list)),
            "mse_std": float(np.std(ms_list)),
            "mae_mean": float(np.mean(ma_list)),
            "mae_std": float(np.std(ma_list)),
            "n_batches": len(ms_list),
        }
        print(f"    MSE: {sweep_results[target_layer]['mse_mean']:.6f} ± "
              f"{sweep_results[target_layer]['mse_std']:.6f}")

    return sweep_results


# ============================================================================
# Visualization
# ============================================================================

def plot_main_results(results: dict, output_dir: str):
    """主对比图：四种条件的 MSE/MAE 对比。"""
    conditions = ["baseline", "recycle", "random", "inverse"]
    labels_cn = ["Baseline (无干预)", "Recycle (高→低 MI)", "Random (随机替换)", "Inverse (低→高 MI)"]
    colors = [
        PALETTE["blue_main"],
        PALETTE["green_3"],
        PALETTE["orange"],
        PALETTE["red_strong"],
    ]

    mse_means = [results[c]["mse_mean"] for c in conditions]
    mse_stds  = [results[c]["mse_std"]  for c in conditions]
    mae_means = [results[c]["mae_mean"] for c in conditions]
    mae_stds  = [results[c]["mae_std"]  for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    # MSE
    ax = axes[0]
    x = np.arange(len(conditions))
    bars = ax.bar(x, mse_means, yerr=mse_stds, color=colors,
                  edgecolor="white", lw=0.5, capsize=3)
    for bar, val in zip(bars, mse_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels_cn], fontsize=6)
    ax.set_ylabel("MSE", fontsize=8)
    ax.set_title("MSE: Feature Recycling Comparison", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "a")

    # MAE
    ax = axes[1]
    bars = ax.bar(x, mae_means, yerr=mae_stds, color=colors,
                  edgecolor="white", lw=0.5, capsize=3)
    for bar, val in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels_cn], fontsize=6)
    ax.set_ylabel("MAE", fontsize=8)
    ax.set_title("MAE: Feature Recycling Comparison", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "b")

    fig.tight_layout(pad=1.0)
    finalize_figure(fig, os.path.join(output_dir, "fig_main_comparison"), dpi=300)


def plot_sweep_results(sweep_results: dict, baseline_mse: float, output_dir: str):
    """逐层 sweep 结果可视化。"""
    layers = sorted(sweep_results.keys())
    mse_vals = [sweep_results[l]["mse_mean"] for l in layers]
    mse_stds = [sweep_results[l]["mse_std"] for l in layers]
    mae_vals = [sweep_results[l]["mae_mean"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    # MSE vs Layer
    ax = axes[0]
    ax.errorbar(layers, mse_vals, yerr=mse_stds, color=PALETTE["blue_main"],
                 linewidth=1.5, marker="o", markersize=5, capsize=3,
                 label="Recycle per layer")
    ax.axhline(baseline_mse, color=PALETTE["red_strong"], linestyle="--",
               linewidth=1.5, label=f"Baseline ({baseline_mse:.4f})")
    ax.fill_between(layers,
                    [v - s for v, s in zip(mse_vals, mse_stds)],
                    [v + s for v, s in zip(mse_vals, mse_stds)],
                    alpha=0.15, color=PALETTE["blue_main"])
    ax.set_xlabel("Recycle Layer", fontsize=8)
    ax.set_ylabel("MSE", fontsize=8)
    ax.set_title("MSE vs Recycle Layer", fontsize=8)
    ax.set_xticks(layers)
    ax.legend(fontsize=6, handlelength=1.5)
    ax.grid(True, alpha=0.25, lw=0.5)
    add_panel_label(ax, "a")

    # Relative improvement vs baseline
    ax = axes[1]
    rel_improve = [(baseline_mse - v) / baseline_mse * 100 for v in mse_vals]
    bar_colors = [
        PALETTE["green_3"] if v > 0 else PALETTE["red_strong"]
        for v in rel_improve
    ]
    bars = ax.bar(layers, rel_improve, color=bar_colors, edgecolor="white", lw=0.5)
    ax.axhline(0, color=PALETTE["neutral_dark"], ls="--", lw=0.8)
    ax.set_xlabel("Recycle Layer", fontsize=8)
    ax.set_ylabel("MSE Improvement (%)", fontsize=8)
    ax.set_title("Relative MSE Improvement vs Baseline", fontsize=8)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    for bar, val in zip(bars, rel_improve):
        sign = "+" if val > 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 if val > 0 else bar.get_height() - 0.03,
                f"{sign}{val:.2f}%", ha="center", va="bottom" if val > 0 else "top",
                fontsize=6)
    add_panel_label(ax, "b")

    fig.tight_layout(pad=1.0)
    finalize_figure(fig, os.path.join(output_dir, "fig_layer_sweep"), dpi=300)


def plot_mi_with_patches(global_mi: np.ndarray, high_patches: list, low_patches: list,
                          output_dir: str):
    """MI 曲线 + 高/低 patch 标注。"""
    n_patches = len(global_mi)
    x = np.arange(n_patches)

    fig, ax = plt.subplots(figsize=(6, 2.5))

    ax.plot(x, global_mi, color=PALETTE["blue_main"], linewidth=1.5,
            marker="o", markersize=4, label="I(H, Y) per patch")
    ax.axhline(global_mi.mean(), color=PALETTE["neutral_mid"],
               linestyle=":", linewidth=1.0, label=f"Mean ({global_mi.mean():.3f})")

    if high_patches:
        ax.scatter(high_patches, global_mi[high_patches],
                   color=PALETTE["green_3"], s=60, marker="^",
                   edgecolors="black", linewidths=0.5, zorder=5,
                   label=f"High MI (patches {high_patches})")
    if low_patches:
        ax.scatter(low_patches, global_mi[low_patches],
                   color=PALETTE["red_strong"], s=60, marker="v",
                   edgecolors="black", linewidths=0.5, zorder=5,
                   label=f"Low MI (patches {low_patches})")

    ax.set_xlabel("Patch Index", fontsize=8)
    ax.set_ylabel("I(H, Y) (bits)", fontsize=8)
    ax.set_title("MI Curve with High/Low Patch Annotations", fontsize=8)
    ax.legend(fontsize=6, handlelength=1.5, loc="best")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xticks(x)

    fig.tight_layout(pad=1.0)
    finalize_figure(fig, os.path.join(output_dir, "fig_mi_annotation"), dpi=300)


def plot_summary_table(results: dict, sweep_results: dict,
                        global_high: list, global_low: list,
                        output_dir: str):
    """汇总热力图：各条件 × 各指标。"""
    conditions = ["baseline", "recycle", "random", "inverse"]
    labels = ["Baseline", "Recycle", "Random", "Inverse"]

    mse_vals = [results[c]["mse_mean"] for c in conditions]
    mae_vals = [results[c]["mae_mean"] for c in conditions]

    best_mse = min(mse_vals)
    best_mae = min(mae_vals)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    # MSE heatmap
    ax = axes[0]
    mse_2d = np.array(mse_vals).reshape(1, -1)
    im = ax.imshow(mse_2d, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title("MSE Comparison", fontsize=8)
    for i, (val, cond) in enumerate(zip(mse_vals, conditions)):
        color = "white" if val == best_mse else "black"
        ax.text(i, 0, f"{val:.4f}", ha="center", va="center",
                fontsize=7, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6)

    # MAE heatmap
    ax = axes[1]
    mae_2d = np.array(mae_vals).reshape(1, -1)
    im2 = ax.imshow(mae_2d, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title("MAE Comparison", fontsize=8)
    for i, val in enumerate(mae_vals):
        color = "white" if val == best_mae else "black"
        ax.text(i, 0, f"{val:.4f}", ha="center", va="center",
                fontsize=7, color=color, fontweight="bold")
    plt.colorbar(im2, ax=ax, shrink=0.6)

    fig.tight_layout(pad=1.0)
    finalize_figure(fig, os.path.join(output_dir, "fig_summary_heatmap"), dpi=300)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Timer MI-Guided Feature Recycling 实验"
    )
    # Data
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--data_type", type=str, default="ETTh1")
    parser.add_argument("--enc_in", type=int, default=None)
    # Model
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # MI result
    parser.add_argument("--mi_result_dir", type=str,
                        default="./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108",
                        help="包含 MI 结果 JSON 的目录")
    parser.add_argument("--model_id", type=str, default="etth1")
    # Recycling config
    parser.add_argument("--recycle_layer", type=int, default=-1,
                        help="进行特征替换的层 (-1=最后一层)")
    parser.add_argument("--blend_ratio", type=float, default=1.0,
                        help="替换时高 MI 表征的混合比例 (0-1)")
    parser.add_argument("--do_layer_sweep", action="store_true",
                        help="是否逐层 sweep")
    parser.add_argument("--max_eval_batches", type=int, default=100,
                        help="最大评估 batch 数 (0=全部)")
    # Runtime
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str,
                        default="./results/timer_mi_feature_recycling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_len", type=int, default=None,
                        help="decoder label_len 前缀长度 (默认=pred_len)")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"Timer_FeatureRecycle_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  Timer MI-Guided Feature Recycling 实验")
    print("=" * 70)

    # ── Load MI results ───────────────────────────────────────────────────────
    print("\n>>> Phase 1: 加载 MI 分析结果...")

    import glob as _glob
    mi_json_candidates = _glob.glob(
        os.path.join(args.mi_result_dir, "global_mi_peaks_*.json")
    )
    if not mi_json_candidates:
        # fallback: search subdirectories
        mi_json_candidates = _glob.glob(
            os.path.join(args.mi_result_dir, "**", "global_mi_peaks_*.json"),
            recursive=True
        )

    mi_json_path = None
    for mc in sorted(mi_json_candidates):
        if args.model_id in os.path.basename(mc):
            mi_json_path = mc
            break
    if not mi_json_path and mi_json_candidates:
        mi_json_path = sorted(mi_json_candidates)[-1]

    if mi_json_path is None:
        print(f"  ERROR: 未找到 MI 结果 JSON，搜索路径: {args.mi_result_dir}")
        print(f"  请先运行 timer_mi_ksg_pca.py 生成 MI 结果。")
        sys.exit(1)

    print(f"  使用 MI 结果: {mi_json_path}")
    with open(mi_json_path) as f:
        mi_summary = json.load(f)

    mi_n_layers = mi_summary['num_layers']
    mi_n_patches = mi_summary['N']
    mi_patch_len = mi_summary.get('patch_len', None)
    print(f"  MI: layers={mi_n_layers}, patches={mi_n_patches}, patch_len={mi_patch_len}")

    # 验证 patch 数一致
    expected_n_patches = args.seq_len // args.patch_len
    if mi_n_patches != expected_n_patches:
        print(f"  [WARN] MI JSON patches={mi_n_patches} != 计算值={expected_n_patches}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("\n>>> Phase 2: 加载数据集...")

    test_dataset = CIDatasetBenchmark(
        root_path=os.path.join(args.root_path, args.data_path),
        flag='test',
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = test_dataset.n_var
    if args.enc_in is not None:
        n_vars_actual = args.enc_in
    else:
        n_vars_actual = n_vars
    print(f"  变量数: {n_vars_actual}, 测试样本: {len(test_dataset)}")

    n_p = args.seq_len // args.patch_len
    print(f"  seq_len={args.seq_len}, patch_len={args.patch_len} -> n_patches={n_p}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    # ── Load Timer model ─────────────────────────────────────────────────────
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
        enc_in=n_vars_actual,
    )
    model = model.to(device)
    model.eval()
    print(f"  Timer 模型加载完成，设备: {device}")

    # ── Main experiment ──────────────────────────────────────────────────────
    print("\n>>> Phase 4: 特征回收主实验...")

    if args.label_len is None:
        args.label_len = args.pred_len

    results, global_high, global_low, global_mi = run_experiment(
        model=model,
        test_loader=test_loader,
        n_vars=n_vars_actual,
        patch_len=args.patch_len,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_patches=n_p,
        device=device,
        args=args,
        mi_summary=mi_summary,
    )

    print("\n  实验结果汇总:")
    print(f"  {'条件':>12} | {'MSE Mean':>12} | {'MSE Std':>10} | {'MAE Mean':>12} | {'MAE Std':>10}")
    print(f"  {'-'*65}")
    for cond in ["baseline", "recycle", "random", "inverse"]:
        r = results[cond]
        print(f"  {cond:>12} | {r['mse_mean']:>12.6f} | {r['mse_std']:>10.6f} | "
              f"{r['mae_mean']:>12.6f} | {r['mae_std']:>10.6f}")

    baseline_mse = results["baseline"]["mse_mean"]
    recycle_mse = results["recycle"]["mse_mean"]
    print(f"\n  Recycle vs Baseline:")
    print(f"    MSE 改善: {(baseline_mse - recycle_mse) / baseline_mse * 100:+.2f}% "
          f"({baseline_mse:.6f} -> {recycle_mse:.6f})")

    # ── Layer sweep ──────────────────────────────────────────────────────────
    sweep_results = {}
    if args.do_layer_sweep:
        print("\n>>> Phase 5: 逐层 Sweep...")
        sweep_results = sweep_recycle_layers(
            model=model,
            test_loader=test_loader,
            n_vars=n_vars_actual,
            patch_len=args.patch_len,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            n_patches=n_p,
            device=device,
            args=args,
            mi_summary=mi_summary,
        )

        print("\n  Sweep 结果:")
        print(f"  {'Layer':>6} | {'MSE Mean':>12} | {'MSE Std':>10} | {'vs Baseline':>12}")
        print(f"  {'-'*50}")
        for li in sorted(sweep_results.keys()):
            sr = sweep_results[li]
            delta = (baseline_mse - sr["mse_mean"]) / baseline_mse * 100
            print(f"  L{li:>5} | {sr['mse_mean']:>12.6f} | {sr['mse_std']:>10.6f} | "
                  f"{delta:>+11.2f}%")
        best_layer = min(sweep_results, key=lambda l: sweep_results[l]["mse_mean"])
        print(f"\n  最佳回收层: L{best_layer} (MSE={sweep_results[best_layer]['mse_mean']:.6f})")

    # ── Plotting ─────────────────────────────────────────────────────────────
    print("\n>>> Phase 6: 绘图...")

    plot_main_results(results, output_dir)
    plot_mi_with_patches(global_mi, global_high, global_low, output_dir)
    plot_summary_table(results, sweep_results, global_high, global_low, output_dir)

    if sweep_results:
        plot_sweep_results(sweep_results, baseline_mse, output_dir)

    # ── Save results ─────────────────────────────────────────────────────────
    print("\n>>> Phase 7: 保存结果...")

    def make_serializable(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if hasattr(obj, 'item'): return obj.item()
        return obj

    output_data = {
        "config": vars(args),
        "mi_summary_path": mi_json_path,
        "high_mi_patches": global_high,
        "low_mi_patches": global_low,
        "recycle_layer": args.recycle_layer if args.recycle_layer >= 0 else args.e_layers - 1,
        "results": json.loads(json.dumps(results, default=make_serializable)),
        "layer_sweep": json.loads(json.dumps(sweep_results, default=make_serializable)) if sweep_results else {},
    }

    results_path = os.path.join(output_dir, "feature_recycling_results.json")
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  结果已保存: {results_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    print(f"\n  [MI Patch Info]")
    print(f"    高 MI patches: {global_high}")
    print(f"    低 MI patches: {global_low}")
    print(f"\n  [预测性能]")
    print(f"    Baseline MSE: {baseline_mse:.6f}")
    print(f"    Recycle  MSE: {recycle_mse:.6f} "
          f"({(baseline_mse - recycle_mse) / baseline_mse * 100:+.2f}%)")
    print(f"    Random   MSE: {results['random']['mse_mean']:.6f}")
    print(f"    Inverse  MSE: {results['inverse']['mse_mean']:.6f}")
    if sweep_results:
        print(f"\n  [最佳回收层] L{best_layer} (MSE={sweep_results[best_layer]['mse_mean']:.6f})")
    print(f"\n  输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
