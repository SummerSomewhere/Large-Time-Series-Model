"""
Map time feature marks (as produced by utils.timefeatures) back to calendar indices.

time_features() stacks rows in the same order as time_features_from_frequency_str(freq).
"""
from __future__ import annotations

import torch


def _hour_dow_columns_for_freq(freq: str) -> tuple[int | None, int]:
    """
    Return (hour_feature_row_index, dow_feature_row_index) in the vstack from time_features().
    None hour means no hour channel (e.g. daily); use 0 for hour embedding in that case.
    """
    f = str(freq).strip().lower()
    if f in ("h", "hour", "hourly"):
        return 0, 1
    if f.endswith("h") and f[:-1].isdigit():
        return 0, 1
    if "min" in f or f in ("t", "s"):
        return 1, 2
    if f in ("d", "b", "w", "daily", "day"):
        return None, 0
    return 0, 1


def marks_to_hour_dow_indices(x_mark: torch.Tensor, freq: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode normalized [-0.5, 0.5] time features to integer hour (0-23) and weekday (0-6).

    x_mark: [B, L, F] float, same layout as dataset time stamps.
    """
    hi, di = _hour_dow_columns_for_freq(freq)
    B, L, F = x_mark.shape
    device, dtype_long = x_mark.device, torch.long
    if hi is not None and hi < F:
        h = ((x_mark[..., hi] + 0.5) * 23.0).round().long().clamp(0, 23)
    else:
        h = torch.zeros((B, L), device=device, dtype=dtype_long)
    if di < F:
        d = ((x_mark[..., di] + 0.5) * 6.0).round().long().clamp(0, 6)
    else:
        d = torch.zeros((B, L), device=device, dtype=dtype_long)
    return h, d


def patch_center_timestep_indices(
    seq_len: int,
    n_patches: int,
    patch_len: int,
    stride: int,
    pad_right: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Integer timestep index into x_mark [B, L, *] for each patch center (aligned with PatchEmbedding).
    Clamps to [0, seq_len - 1] when center falls in padded tail without marks.
    """
    idx = []
    Lp = seq_len + pad_right
    for p in range(n_patches):
        start = p * stride
        center = start + (patch_len - 1) // 2
        center = max(0, min(center, seq_len - 1))
        idx.append(center)
    return torch.tensor(idx, device=device, dtype=torch.long)
