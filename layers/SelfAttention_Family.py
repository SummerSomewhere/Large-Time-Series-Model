import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) ≈ y (y > 0)."""
    y = max(float(y), 1e-6)
    return float(np.log(np.expm1(y)))


class FullAttentionHarmonicGatedResonance(nn.Module):
    """
    Harmonic gated resonance (one specialist head): no hard additive bias on QK^T.

    Uses physical_timestamps T only (same units as ω; Timer uses hours at patch centers).
    Resonance_ij = sum_k λ_k cos(2πkω(T_i-T_j)+φ_k) via cos(A-B)=cosA cosB+sinA sinB with
    A_i=2πkωT_i+φ_k/2, B_j=2πkωT_j-φ_k/2 — O(K·L) trig, then einsum to [B,L,L] logits.
    On causal lower triangle with non-decreasing T, T_i-T_j equals |T_i-T_j| for i>=j.

    Specialist logits: S_final = S_orig ⊙ σ(RS + b); b≈7 keeps σ≈1 when RS≈0 and λ init small.
    """

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        n_heads=8,
        specialist_head_idx=0,
        n_harmonics=3,
        omega_init=None,
        lambda_init=1e-4,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads
        h = int(specialist_head_idx)
        if h < 0 or h >= n_heads:
            raise ValueError(f"specialist_head_idx {h} out of range for n_heads={n_heads}")
        self.specialist_head_idx = h
        self.n_harmonics = int(n_harmonics)
        K = self.n_harmonics
        if K < 1:
            raise ValueError("n_harmonics must be >= 1")

        o0 = 1.0 / 24.0 if omega_init is None else float(omega_init)
        self.har_omega_raw = nn.Parameter(
            torch.tensor(_inv_softplus(o0 - 1e-6), dtype=torch.float32)
        )
        lam0 = float(lambda_init)
        self.har_lambda = nn.Parameter(torch.full((K,), lam0, dtype=torch.float32))
        self.har_phi = nn.Parameter(torch.zeros(K, dtype=torch.float32))
        # When RS≈0, σ(RS+b)≈1 so scores unchanged; λ small keeps RS small at init.
        self.har_sigmoid_bias = nn.Parameter(torch.tensor(7.0, dtype=torch.float32))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, physical_timestamps=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "Harmonic gated resonance expects self-attention (L == S)"
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if physical_timestamps is not None:
            T = physical_timestamps.to(device=scores.device, dtype=scores.dtype)
            if T.dim() != 2 or T.shape[0] != B or T.shape[1] != L:
                raise ValueError(
                    f"physical_timestamps expected [B, L]={B, L}, got {tuple(T.shape)}"
                )
            omega = F.softplus(self.har_omega_raw) + 1e-6
            phi = self.har_phi.to(dtype=scores.dtype)
            lam = self.har_lambda.to(dtype=scores.dtype)

            rs = torch.zeros(B, L, L, device=scores.device, dtype=scores.dtype)
            two_pi = 2.0 * math.pi
            for k in range(1, self.n_harmonics + 1):
                pk = phi[k - 1]
                lk = lam[k - 1]
                ang_i = two_pi * float(k) * omega * T + 0.5 * pk
                ang_j = two_pi * float(k) * omega * T - 0.5 * pk
                ci, si = torch.cos(ang_i), torch.sin(ang_i)
                cj, sj = torch.cos(ang_j), torch.sin(ang_j)
                pair = torch.einsum("bi,bj->bij", ci, cj) + torch.einsum("bi,bj->bij", si, sj)
                rs = rs + lk * pair

            hstar = self.specialist_head_idx
            s_orig = scores[:, hstar, :, :]
            gate = torch.sigmoid(rs + self.har_sigmoid_bias.to(dtype=scores.dtype))
            new_h = (s_orig * gate).unsqueeze(1)
            if hstar == 0:
                scores = torch.cat([new_h, scores[:, 1:, :, :]], dim=1)
            elif hstar == H - 1:
                scores = torch.cat([scores[:, :hstar, :, :], new_h], dim=1)
            else:
                scores = torch.cat(
                    [scores[:, :hstar, :, :], new_h, scores[:, hstar + 1 :, :, :]],
                    dim=1,
                )

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores = scores.masked_fill(attn_mask.mask, float("-inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

    @torch.no_grad()
    def fft_warmstart_omega(self, series_raw: torch.Tensor, dt_hours: float) -> None:
        """
        Set base omega from dominant rFFT bin of batch-mean detrended series [B, T] (cycles per hour).
        """
        if series_raw is None or series_raw.numel() == 0:
            return
        x = series_raw.to(dtype=torch.float32)
        x = x - x.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(x, dim=-1)
        mag = spec.abs().mean(dim=0)
        if mag.numel() <= 1:
            return
        mag = mag.clone()
        mag[0] = 0
        k = int(mag.argmax().item())
        L = x.shape[-1]
        freqs = torch.fft.rfftfreq(L, d=float(dt_hours), device=x.device, dtype=x.dtype)
        w_est = abs(freqs[k].item())
        w_est = max(float(w_est), 1e-5)
        target = w_est - 1e-6
        target = max(target, 1e-6)
        self.har_omega_raw.fill_(math.log(math.expm1(target)))


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