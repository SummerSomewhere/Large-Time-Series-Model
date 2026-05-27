import torch
import numpy as np
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class RefinementEncoder(nn.Module):
    """带有 Refinement 机制的 Encoder"""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None,
                 refine_patches=None, random_refine_patches=False):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

        self.refine_patches = refine_patches or []
        self.random_refine_patches = random_refine_patches

        if self.random_refine_patches and len(self.refine_patches) > 0:
            import random
            import time
            # 每次运行时使用不同的时间戳作为种子，打破与全局 fix_seed 的绑定
            random.seed(time.time_ns() % 100000)
            idx = random.randint(0, 5)
            self.selected_patches = [idx]
            print(f"[Refinement] 随机选择的 patch: {self.selected_patches}")
        else:
            self.selected_patches = self.refine_patches

    def forward(self, x, attn_mask=None, tau=None, delta=None, has_prototype: bool = False,
                output_hidden_states: bool = False, layer_guide: torch.Tensor = None,
                output_attention_override: bool = False):
        B, L, D = x.shape
        attns = []
        logits_list = []
        hidden_states = [] if output_hidden_states else None
        num_layers = len(self.attn_layers)
        second_last_layer_idx = num_layers - 8  # 倒数第三层

        for i, attn_layer in enumerate(self.attn_layers):
            x, attn, logits = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            if logits is not None:
                logits_list.append(logits)
            if output_hidden_states:
                hidden_states.append(x)

            # 在倒数第三层之后，对选中的 patch 多转一圈后直接替代
            if i == second_last_layer_idx and len(self.selected_patches) > 0:
                x = self._apply_refinement(x)

        if self.norm is not None:
            x = self.norm(x)
            if output_hidden_states:
                hidden_states.append(x)

        if output_hidden_states:
            return x, attns, logits_list, hidden_states
        return x, attns, logits_list

    def _apply_refinement(self, h):
        """
        在倒数第三层之后，对选中的 patch 多转一圈完整的层，
        直接替代。
        """
        B, L, D = h.shape
        h_refined = h.clone()

        # 获取最后一层（用于做额外的 refinement）
        last_layer = self.attn_layers[-1]

        # 对选中的 patch 进行 refinement
        for patch_idx in self.selected_patches:
            if 0 <= patch_idx < L:
                h_patch = h[:, patch_idx:patch_idx+1, :]
                h_patch_refined, _, _ = last_layer(h_patch, attn_mask=None)
                h_refined[:, patch_idx:patch_idx+1, :] = h_patch_refined

        return h_refined


class InjectionEncoder(nn.Module):
    """
    带有特征注入（Feature Injection）机制的 Encoder。

    对齐 MOMENT：每层 prepend 一个 guide token，attention 后立即截断，
    该 token 只在该层被 self-attention 消费，不继续往后传。
    （即：每层注入、每层消耗，与 MOMENT 行为一致。）

    参数:
        attn_layers: Transformer Encoder 层列表
        norm_layer: LayerNorm 层
    """
    def __init__(self, attn_layers, norm_layer=None, truncate_guide: bool = True,
                 hsic_mi_curve=None, hsic_q3=None, hsic_q1=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.truncate_guide = truncate_guide
        self.hsic_mi_curve = hsic_mi_curve
        self.hsic_q3 = hsic_q3
        self.hsic_q1 = hsic_q1

    def forward(self, x, attn_mask=None, tau=None, delta=None, has_prototype: bool = False,
                output_hidden_states: bool = False, layer_guide: torch.Tensor = None,
                output_attention_override: bool = False):
        """
        Args:
            x: [B, N, D] or [B, N+1, D] if has_prototype=True
            has_prototype: whether prototype was prepended to x
            output_hidden_states: if True, return all layer outputs as a list
            layer_guide: [B, num_layers, D] or [B*M, num_layers, D].
                         第 i 层（i>=0）注入 layer_guide[:, i] 作为额外 token。
                         即：layer_guide[:, 0, :] 引导第 0 层，layer_guide[:, 1, :] 引导第 1 层...
                         若 layer_guide 某列为全零，则该层不注入。

        truncate_guide=True（默认）：对齐 MOMENT，每层 prepend 后立即截断，
                                   guide token 只在该层被 self-attention 消费，不往下传。
        truncate_guide=False：原始 Timer 行为，guide token 注入后持续存在，
                             贯穿整个前向过程。

        Returns:
            x, attns, logits_list, [hidden_states]
        """
        B, L, D = x.shape
        attns = []
        logits_list = []
        hidden_states = [] if output_hidden_states else None

        injection_occurred = []
        for i, attn_layer in enumerate(self.attn_layers):
            # ── 逐层引导注入：prepend layer_guide[:, i] ──────────────────────
            if layer_guide is not None:
                guide_tok = layer_guide[:, i:i + 1, :]   # [B, 1, D]
                if guide_tok.abs().sum() > 1e-8:
                    x = torch.cat([guide_tok, x], dim=1)  # [B, N+1, D]
                    injection_occurred.append(True)
                else:
                    injection_occurred.append(False)
            else:
                injection_occurred.append(False)
            # ─────────────────────────────────────────────────────────────────

            x, attn, logits = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            if logits is not None:
                logits_list.append(logits)

            # truncate_guide=True：对齐 MOMENT，每层 prepend 后立即截断
            if self.truncate_guide and x.shape[1] > L:
                x = x[:, 1:, :]  # [B, N, D]

            if output_hidden_states:
                hidden_states.append(x)

        if self.norm is not None:
            x = self.norm(x)
            if output_hidden_states:
                hidden_states.append(x)

        # truncate_guide=False：末尾统一剥离（原始 Timer 行为）
        if not self.truncate_guide:
            n_guide_injected = sum(injection_occurred)
            if n_guide_injected > 0:
                x = x[:, n_guide_injected:, :]
                if hidden_states is not None:
                    hidden_states = [h[:, n_guide_injected:, :] if h.shape[1] > L else h
                                     for h in hidden_states]

        if output_hidden_states:
            return x, attns, logits_list, hidden_states
        return x, attns, logits_list


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        padding = 0

        self.output_attention = configs.output_attention

        self.enable_refinement = getattr(configs, 'enable_refinement', False)
        self.refine_patches = getattr(configs, 'refine_patches', [5, 6])
        self.random_refine_patches = getattr(configs, 'random_refine_patches', False)
        self.is_injection_test = getattr(configs, 'is_injection_test', False)
        self.injection_alpha = getattr(configs, 'injection_alpha', 0.5)
        self.truncate_guide = getattr(configs, 'truncate_guide', True)
        self.hsic_mi_curve = getattr(configs, 'hsic_mi_curve', None)
        self.hsic_q3 = getattr(configs, 'hsic_q3', None)
        self.hsic_q1 = getattr(configs, 'hsic_q1', None)

        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, self.dropout)

        self.use_align_loss = getattr(configs, 'use_align_loss', False)
        self.align_loss_layers = getattr(configs, 'align_loss_layers', [0, 1, 2, 3, 4, 5, 6, 7])

        if self.is_injection_test:
            self.decoder = InjectionEncoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=True), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                truncate_guide=self.truncate_guide,
                hsic_mi_curve=self.hsic_mi_curve,
                hsic_q3=self.hsic_q3,
                hsic_q1=self.hsic_q1,
            )
        elif self.enable_refinement:
            self.decoder = RefinementEncoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=True), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                refine_patches=self.refine_patches,
                random_refine_patches=self.random_refine_patches
            )
        else:
            self.decoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=True), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

        self.proj = nn.Linear(self.d_model, configs.patch_len, bias=True)