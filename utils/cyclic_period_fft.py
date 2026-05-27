"""FFT-based dominant period (in patch units) for cyclic attention penalties."""

import torch


def dominant_period_patches_from_fft(
    x_blm: torch.Tensor,
    *,
    patch_len: int,
    min_p: int = 2,
    max_p: int = 256,
) -> int:
    """
    Pool |x| over batch and variates, rFFT along time, pick strongest non-DC bin.
    Map temporal period in samples to an integer period in patch indices: ~ period_samples / patch_len.
    """
    _b, L, _m = x_blm.shape
    L = int(L)
    pl = int(patch_len)
    if L < 4 or pl <= 0:
        return max(int(min_p), min(int(max_p), 2))
    s = x_blm.float().abs().mean(dim=(0, 2))
    mag = torch.fft.rfft(s, dim=-1).abs()
    mag[0] = 0
    if mag.numel() <= 1:
        return max(int(min_p), min(int(max_p), 2))
    k_star = int(mag.argmax().item())
    if k_star == 0:
        k_star = max(1, int(mag[1:].argmax().item()) + 1)
    period_samples = float(L) / float(k_star)
    P = int(round(period_samples / float(pl)))
    P = max(int(min_p), min(int(max_p), max(2, P)))
    return P
