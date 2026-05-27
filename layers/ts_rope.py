"""
Time-Scale Aware RoPE (TS-RoPE): dominant frequencies from rFFT on the raw series drive
per-dimension rotation rates (before Q/K projection). Top-k spectrum is mapped to head_dim/2 pairs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_ts_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary scaling using precomputed cos/sin.

    q, k: [B, N, H, E]
    cos, sin: [B, N, 1, E] — broadcast over heads
    """
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class FrequencyAwareRoPE(nn.Module):
    """
    FFT-based dominant frequencies -> per-head_dim rotation steps; build cos/sin for each patch index.

    Input x is expected **before** patch Linear projection: [B, L, D] (e.g. normalized multivariate series).
    rFFT along L; amplitude mean over D; top-K bins (excluding DC) per batch row -> frequencies f = k/L.
    Radians advanced per patch step along index p: p * (2*pi * f_j * patch_len) after mapping K freqs to E/2 dims.
    """

    def __init__(self, topk: int = 5):
        super().__init__()
        self.topk = int(topk)

    def compute_freqs_cis(
        self,
        x: torch.Tensor,
        seq_len: int,
        n_patches: int,
        patch_len: int,
        stride: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build cos/sin for all patch positions and all variates (broadcast batch).

        Args:
            x: [B, L, D] real values (e.g. normalized x_enc).
            seq_len: L (time length).
            n_patches: number of patches N (must match encoder token length).
            patch_len / stride: from patch embedding (phase advance per patch).
            head_dim: E per head (even).

        Returns:
            cos, sin each [B * D, N, 1, head_dim] to match queries [B*D, N, H, head_dim] after variate flatten.
        """
        if head_dim % 2 != 0:
            raise ValueError(f"TS-RoPE requires even head_dim, got {head_dim}")
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype
        k = min(self.topk, max(1, L // 2))  # rfft bins excluding DC at most

        # rFFT along time; spectrum [B, F, D], F = L//2+1
        spec = torch.fft.rfft(x, dim=1, norm=None)
        amp = spec.abs()  # [B, F, D]
        amp_mean = amp.mean(dim=2)  # [B, Freq]

        n_fft_bins = amp_mean.size(1)
        E2 = head_dim // 2
        if n_fft_bins <= 1 or L < 2:
            # Degenerate: small uniform rotation rate per patch (near identity)
            theta_step = torch.full(
                (B, E2), 2.0 * math.pi / max(1.0, float(n_patches * max(L, 1))), device=device, dtype=torch.float32
            )
        else:
            # Exclude DC (index 0) from top-k
            amp_no_dc = amp_mean[:, 1:].contiguous()  # [B, n_fft_bins-1]
            k_eff = min(k, amp_no_dc.size(1))
            # topk on spectrum (detach for stable training through discrete selection)
            _, top_idx = torch.topk(amp_no_dc.detach(), k=k_eff, dim=1)
            # normalized cyclic frequency in [0, 0.5): f = bin / L
            top_idx_f = top_idx.float() + 1.0  # map back to bin 1..F-1
            f_sel = top_idx_f / float(max(L, 1))  # [B, k_eff]

            # radians per patch index for each selected frequency
            phase_adv = 2.0 * math.pi * f_sel * float(patch_len)  # [B, k_eff] (stride == patch_len in Timer)

            # Map K_eff values to head_dim/2 via linear interpolation along the sorted frequency axis
            if k_eff == 1:
                theta_b = phase_adv.expand(-1, E2)
            else:
                # [B, k_eff] -> [B, 1, k_eff] -> interpolate to E2
                src = phase_adv.unsqueeze(1)  # [B, 1, k_eff]
                theta_b = F.interpolate(src, size=E2, mode="linear", align_corners=True).squeeze(1)

            # theta_b[b, j] = radians per increment of patch index for dim pair j
            theta_step = theta_b.to(dtype=torch.float32)

        # positions 0..N-1
        pos = torch.arange(n_patches, device=device, dtype=torch.float32).view(1, n_patches, 1)
        angles = pos * theta_step.unsqueeze(1)  # [B, N, E2]
        # duplicate for cos/sin on full E dims (same as vanilla RoPE stacking)
        emb = torch.cat([angles, angles], dim=-1)  # [B, N, head_dim]
        cos_b = emb.cos().unsqueeze(2)  # [B, N, 1, head_dim]
        sin_b = emb.sin().unsqueeze(2)

        # Same rotation for each variate channel: [B, D, N, 1, head_dim] -> [B*D, N, 1, head_dim]
        cos_bm = cos_b.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(B * D, n_patches, 1, head_dim)
        sin_bm = sin_b.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(B * D, n_patches, 1, head_dim)

        return cos_bm.to(dtype=dtype), sin_bm.to(dtype=dtype)
