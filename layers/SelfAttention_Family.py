from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        num_patches: int | None = None,
        n_heads: int | None = None,
        mi_bias_patch_indices: tuple[int, ...] | None = None,
        mi_bias_init_val: float = 0.5,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # Optional [H, P, P] additive logits before softmax (physically informed key-patch prior).
        self.mi_bias: nn.Parameter | None
        if (
            num_patches is not None
            and n_heads is not None
            and int(num_patches) > 0
            and int(n_heads) > 0
        ):
            p, h = int(num_patches), int(n_heads)
            self.mi_bias = nn.Parameter(torch.zeros(h, p, p))
            idxs = mi_bias_patch_indices if mi_bias_patch_indices else (3, 6)
            with torch.no_grad():
                for j in idxs:
                    jj = int(j)
                    if 0 <= jj < p:
                        self.mi_bias[:, :, jj] = float(mi_bias_init_val)
        else:
            self.mi_bias = None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # Match common QK^T / sqrt(d) logits, then add learnable prior, then softmax (see below).
        attn_logits = scale * scores
        if self.mi_bias is not None and L == self.mi_bias.shape[1] and S == self.mi_bias.shape[2]:
            attn_logits = attn_logits + self.mi_bias.unsqueeze(0)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            attn_logits = attn_logits.masked_fill(attn_mask.mask, float("-inf"))

        A = self.dropout(torch.softmax(attn_logits, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
