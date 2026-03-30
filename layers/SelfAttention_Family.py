import math

import numpy as np
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
        diurnal_heads=None,
        diurnal_lambda=0.0,
        diurnal_period=24.0,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # Periodic bias from patch index distance: add (lambda/scale)*cos(2*pi*|i-j|/period) to raw scores
        # for selected heads before the causal mask (period is in patch-token units, e.g. 24 for hourly patch_len=1).
        self.diurnal_heads = frozenset(diurnal_heads) if diurnal_heads else frozenset()
        self.diurnal_lambda = float(diurnal_lambda)
        self.diurnal_period = float(diurnal_period)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, physical_timestamps=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.diurnal_lambda != 0.0 and self.diurnal_heads and self.diurnal_period > 0:
            idx = torch.arange(L, device=queries.device, dtype=scores.dtype)
            dh = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
            cos_bias = torch.cos((2.0 * math.pi / self.diurnal_period) * dh)
            adj = (self.diurnal_lambda / scale) * cos_bias
            for h in self.diurnal_heads:
                scores[:, h, :, :] = scores[:, h, :, :] + adj

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttentionLastLayerResonance(nn.Module):
    """
    Last-layer MHA with learnable per-head resonance bias on raw attention scores.

    Bias_ij^h = λ_h * cos(2π ω_h (T_i - T_j) + φ_h) using cos(A-B) = cos A cos B + sin A sin B with
    A_i = 2π ω T_i + φ/2, B_j = 2π ω T_j - φ/2, so O(L) trig evals per head; scores still L×L from einsum.

    When timestamps are non-decreasing along the sequence (causal self-attn), this matches
    cos(2π ω |T_i - T_j| + φ) on the lower triangle where T_i >= T_j.

    Raw scores get (λ_h / scale) * cos_term before the causal mask, matching FullAttention diurnal scaling.
    """

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        n_heads=8,
        resonance_head_mask=None,
        omega_init=None,
        lambda_init=0.1,
        phi_init=0.0,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads
        if resonance_head_mask is None:
            rmask = torch.ones(n_heads, dtype=torch.bool)
        else:
            rmask = torch.as_tensor(resonance_head_mask, dtype=torch.bool).flatten()
            if rmask.numel() != n_heads:
                raise ValueError(
                    f"resonance_head_mask length {rmask.numel()} != n_heads {n_heads}"
                )
        self.register_buffer("resonance_head_mask", rmask)
        o_init = 1.0 / 24.0 if omega_init is None else float(omega_init)
        self.res_omega = nn.Parameter(torch.full((n_heads,), o_init))
        self.res_lambda = nn.Parameter(torch.full((n_heads,), float(lambda_init)))
        self.res_phi = nn.Parameter(torch.full((n_heads,), float(phi_init)))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, physical_timestamps=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "Resonance attention expects self-attention (L == S)"
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if physical_timestamps is not None:
            T = physical_timestamps.to(device=scores.device, dtype=scores.dtype)
            if T.dim() != 2 or T.shape[0] != B or T.shape[1] != L:
                raise ValueError(
                    f"physical_timestamps expected [B, L]={B, L}, got {tuple(T.shape)}"
                )
            two_pi = 2.0 * math.pi
            T_e = T.unsqueeze(-1)
            w = self.res_omega.view(1, 1, -1)
            ph = self.res_phi.view(1, 1, -1)
            ang_q = two_pi * w * T_e + 0.5 * ph
            ang_k = two_pi * w * T_e - 0.5 * ph
            cq = torch.cos(ang_q)
            sq = torch.sin(ang_q)
            ck = torch.cos(ang_k)
            sk = torch.sin(ang_k)
            term = torch.einsum("blh,bsh->blsh", cq, ck) + torch.einsum("blh,bsh->blsh", sq, sk)
            term = term.permute(0, 3, 1, 2).contiguous()
            coef = (self.res_lambda / scale).view(1, H, 1, 1) * self.resonance_head_mask.view(
                1, H, 1, 1
            ).to(dtype=scores.dtype)
            scores = scores + coef * term

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
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

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, physical_timestamps=None):
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
            physical_timestamps=physical_timestamps,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn