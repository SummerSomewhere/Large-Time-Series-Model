#!/usr/bin/env python3
"""
Compare STFT magnitude spectra between high-MI and low-MI patches (same window as MI analysis).

Uses per-patch HSIC/MI scores from --ckpt_path (Timer forward) or --hsic_npy (aggregate npy),
then splits patches: default --mi_group_mode symmetric (high >= p_high, low <= p_low, mid between),
or high_vs_rest (high >= p_high e.g. top ~25% at p_high=0.75; all other finite patches = low).

Static --hsic_npy must have last dim P equal to n_patches for the current seq_len/patch_len/stride
(P changes when patch_len changes; no silent truncate/pad).

Averaging: for each patch we take mean |STFT| over STFT time frames; then within high (or low)
group we **mean over patches**.

Per-layer mode (default):
  - HSIC must be shaped [n_layers, n_patches] (2D npy) or computed live as a full layer x patch
    matrix. A 1D npy is treated as a single layer (row).
  - For each encoder layer index, MI scores are hsic_mat[layer], patches are split by quantiles,
    and STFT curves are produced independently (no mixing across layers).
  - --layers controls which layers to plot: default "all", or e.g. "0,2,-1" (-1 = last layer).

Outputs (per layer, per time window t0):
  - {stem}_mi_stft_high_vs_low_t{t0}_layer{LL}.png  — overlay high vs low mean |STFT|
  - {stem}_mi_stft_diff_t{t0}_layer{LL}.png       — high minus low
  - {stem}_mi_stft_groups_t{t0}_layer{LL}.csv      — per-patch MI and group label
  - {stem}_mi_stft_meta_t{t0}.json                 — window metadata (includes layers_plotted)

Optional cross-window aggregation (--aggregate_windows 1, needs --num_samples >= 2):
  - {stem}_mi_stft_high_vs_low_aggW{W}_layer{LL}.png
  - {stem}_mi_stft_diff_aggW{W}_layer{LL}.png

Full-dataset mode (--use_full_dataset 1):
  - full_dataset_mode=exp_split (default): same sliding time grid as exp_forecast + use_ims
    (CIAutoRegressionDatasetBenchmark): all s_begin in [0, n_timepoint) within train/val/test,
    global t0 = border1 + s_begin (see --dataset_flag). Matches data_factory borders, not full-CSV hop.
  - full_dataset_mode=sliding: t0 = t_start, t_start + hop, ... while t0 + seq_len <= series length
    (hop defaults to seq_len when --window_hop <= 0).
  - Averages HSIC/MI across windows to one vector per layer, then applies quantile split once.
  - Averages mean |STFT| curves across all windows (same masks). Writes:
  - {stem}_dataset_avg_hsic_mean.npy  [L, P] mean MI matrix
  - {stem}_mi_stft_dataset_avg_W{W}_layer{LL}.png / _diff_...
  - {stem}_mi_stft_groups_dataset_avg_W{W}_layer{LL}.csv
  - {stem}_mi_stft_meta_dataset_avg.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXPERIMENTS = os.path.join(_ROOT, "experiments")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.signal import stft

import mi_patch_curvature_analysis as _mpc


def _patch_grid_str(args: argparse.Namespace) -> str:
    stride = args.patch_len if args.stride < 0 else args.stride
    return (
        f"seq_len={int(args.seq_len)}, patch_len={int(args.patch_len)}, stride={int(stride)}"
    )


def hsic_npy_num_patches(path: str) -> int:
    """P for 2D [L,P] or length for 1D array."""
    arr = np.load(path)
    if arr.ndim == 1:
        return int(arr.size)
    return int(arr.shape[1])


def build_exp_forecast_ims_window_starts(args: argparse.Namespace) -> tuple[list[int], str]:
    """
    Global encoder start indices t0 aligned with CIAutoRegressionDatasetBenchmark (exp_forecast + use_ims):
    one window per s_begin in 0 .. n_timepoint-1 within the chosen split; t0 = border1 + s_begin.
    """
    from data_provider.data_loader_benchmark import CIAutoRegressionDatasetBenchmark

    csv_abs = os.path.abspath(args.csv)
    flag = str(getattr(args, "dataset_flag", "test"))
    pred_len_ds = int(args.output_len if flag == "test" else args.pred_len)
    timeenc = 0 if args.embed != "timeF" else 1
    data_type = str(getattr(args, "data", "ETTh1"))
    input_len = int(args.seq_len)
    loader_stride = int(getattr(args, "loader_stride", 1))

    ds = CIAutoRegressionDatasetBenchmark(
        root_path=csv_abs,
        flag=flag,
        input_len=input_len,
        label_len=int(args.label_len),
        pred_len=pred_len_ds,
        data_type=data_type,
        scale=True,
        timeenc=timeenc,
        freq=str(getattr(args, "freq", "h")),
        stride=loader_stride,
        subset_rand_ratio=float(getattr(args, "subset_rand_ratio", 1.0)),
    )
    n_tp = int(ds.n_timepoint)
    border1 = _benchmark_border1_etth_ettm(data_type, input_len, flag)
    windows = [border1 + i for i in range(n_tp)]
    desc = (
        f"CIAutoRegressionDatasetBenchmark split={flag}, border1={border1}, "
        f"n_timepoint={n_tp}, pred_len_ds={pred_len_ds}"
    )
    return windows, desc


def _benchmark_border1_etth_ettm(data_type: str, input_len: int, flag: str) -> int:
    type_map = {"train": 0, "val": 1, "test": 2}
    st = type_map[flag]
    if data_type in ("ETTh", "ETTh1", "ETTh2"):
        border1s = [
            0,
            12 * 30 * 24 - input_len,
            12 * 30 * 24 + 4 * 30 * 24 - input_len,
        ]
    elif data_type in ("ETTm", "ETTm1", "ETTm2"):
        border1s = [
            0,
            12 * 30 * 24 * 4 - input_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - input_len,
        ]
    else:
        raise ValueError(
            "full_dataset_mode=exp_split requires --data in "
            "ETTh1/ETTh2/ETTm1/ETTm2 (same as exp_forecast benchmark). "
            f"Got {data_type!r} or use full_dataset_mode=sliding."
        )
    return int(border1s[st])


def resolve_compute_hsic(
    n_p: int,
    ckpt_path: str,
    hsic_path: str,
    grid: str,
) -> tuple[bool, str]:
    """
    Prefer static npy when its patch dim matches n_p; otherwise use live ckpt if available.

    Returns (compute_hsic, hsic_path_for_loader). When compute_hsic is True, loader ignores npy.
    """
    has_ckpt = bool(ckpt_path) and os.path.isfile(ckpt_path)
    has_hsic = bool(hsic_path) and os.path.isfile(hsic_path)
    if has_hsic:
        p_npy = hsic_npy_num_patches(hsic_path)
        if p_npy == int(n_p):
            return False, hsic_path
        if has_ckpt:
            print(
                f"WARNING: HSIC npy has P={p_npy} but current grid needs n_p={n_p} ({grid}); "
                "falling back to live HSIC from --ckpt_path.",
                flush=True,
            )
            return True, hsic_path
        raise ValueError(
            f"HSIC npy has P={p_npy} patches per layer, but n_patches={n_p} for current grid ({grid}). "
            "Regenerate hsic_mean.npy with matching patch_len/seq_len/stride, or pass a compatible "
            "--ckpt_path for live HSIC."
        )
    if has_ckpt:
        return True, ""
    raise ValueError(
        f"Provide --hsic_npy with P={n_p} or --ckpt_path for live HSIC ({grid})."
    )


def load_hsic_matrix_full(
    args: argparse.Namespace,
    t0: int,
    n_p: int,
    hsic_path: str,
    compute_hsic: bool,
) -> tuple[np.ndarray, str]:
    """
    Return HSIC matrix [n_layers, n_p] (same patch count as window) and a short description.
    """
    if compute_hsic:
        import torch
        import mi_hsic_layerwise_timer as mi_hsic
        from models.Timer import Model as TimerModel

        cfg = _mpc.build_timer_configs(args)
        device = torch.device(
            "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
        )
        model = TimerModel(cfg).float().to(device)
        try:
            int(args.target_col)
            tgt_name = "OT"
        except ValueError:
            tgt_name = str(args.target_col)
        pred_horizon = int(args.output_len if args.use_ims else args.pred_len)
        # Global CSV index (same as STFT windows), not Dataset_ETT_hour test-split index.
        batch_x, batch_y = _mpc.load_ims_batch_timer_global_index(
            args.csv,
            t0,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.timer_features,
            tgt_name,
            args.embed,
            pred_horizon=pred_horizon,
        )
        hsic_mat, fwd_mode = mi_hsic.compute_hsic_mi_matrix_timer_batch(
            model,
            device,
            batch_x,
            batch_y,
            use_ims=bool(args.use_ims),
            label_len=args.label_len,
            pred_horizon=pred_horizon,
        )
        mat = np.asarray(hsic_mat, dtype=np.float64)
        n_layers, pw = int(mat.shape[0]), int(mat.shape[1])
        if pw != n_p:
            raise ValueError(
                f"Computed HSIC shape ({n_layers}, {pw}) != expected patches {n_p}"
            )
        return mat, f"live HSIC ({fwd_mode}), L={n_layers}"
    assert hsic_path
    arr = np.load(hsic_path)
    grid = _patch_grid_str(args)
    if arr.ndim == 1:
        row = np.asarray(arr, dtype=np.float64).ravel()
        if int(row.size) != int(n_p):
            raise ValueError(
                f"HSIC npy 1D length {row.size} != n_patches={n_p} for current grid ({grid}). "
                "Per-layer patch count changes with patch_len/seq_len; use an npy computed with "
                "the same settings, or use --ckpt_path for live HSIC."
            )
        return row.reshape(1, -1), "HSIC npy 1D row"
    mat = np.asarray(arr, dtype=np.float64)
    n_layers, pw = int(mat.shape[0]), int(mat.shape[1])
    if pw != n_p:
        raise ValueError(
            f"HSIC npy has P={pw} patches per layer, but n_patches={n_p} for current grid ({grid}). "
            "Per-layer patch count changes with patch_len; regenerate hsic_mean.npy (or aggregate) "
            "with matching patch_len/seq_len/stride, or use --ckpt_path for live HSIC."
        )
    return mat, f"HSIC npy 2D [L={n_layers}, P={n_p}]"


def _parse_layers_arg(raw: str, n_layers: int) -> list[int] | None:
    """None = all layers; else sorted unique indices in [0, n_layers)."""
    s = (raw or "").strip()
    if not s or s.lower() == "all":
        return None
    out: list[int] = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if i < 0:
            i = n_layers + i
        if 0 <= i < n_layers:
            out.append(i)
    return sorted(set(out)) if out else None


def mi_group_masks(
    args: argparse.Namespace, mi_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    """
    Split patches by MI. high_quantile defines the high-MI cutoff on the empirical distribution
    (e.g. 0.75 -> threshold at 75th percentile; high MI are >= that value, ~top 25%).

    symmetric: high >= p_high quantile, low <= p_low quantile; remaining finite patches are "mid".
    high_vs_rest: high >= p_high quantile; every other finite patch is low (no mid group).
    """
    finite = mi_scores[np.isfinite(mi_scores)]
    hi_th = float(np.percentile(finite, 100.0 * float(args.high_quantile)))
    mode = getattr(args, "mi_group_mode", "symmetric") or "symmetric"
    if mode == "high_vs_rest":
        high_mask = np.isfinite(mi_scores) & (mi_scores >= hi_th)
        low_mask = np.isfinite(mi_scores) & (~high_mask)
        return high_mask, low_mask, hi_th, None
    lo_th = float(np.percentile(finite, 100.0 * float(args.low_quantile)))
    high_mask = np.isfinite(mi_scores) & (mi_scores >= hi_th)
    low_mask = np.isfinite(mi_scores) & (mi_scores <= lo_th)
    return high_mask, low_mask, hi_th, lo_th


def assign_mi_group_labels(
    n_p: int,
    high_mask: np.ndarray,
    low_mask: np.ndarray,
    mi_group_mode: str,
) -> np.ndarray:
    if mi_group_mode == "high_vs_rest":
        grp = np.full(n_p, "nan", dtype=object)
        grp[low_mask] = "low"
        grp[high_mask] = "high"
        return grp
    grp = np.full(n_p, "mid", dtype=object)
    grp[low_mask] = "low"
    grp[high_mask] = "high"
    return grp


def patch_stft_mean_magnitude(
    patch_1d: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    nfft: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(patch_1d, dtype=np.float64).ravel()
    if x.size < 8:
        return np.array([]), np.array([])
    nfft_use = int(nfft) if nfft and int(nfft) > 0 else min(nperseg * 2, max(nperseg, 32))
    f, _, Zxx = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=min(nperseg, x.size),
        noverlap=min(noverlap, min(nperseg, x.size) - 1),
        nfft=min(nfft_use, max(nperseg, x.size)),
        boundary="zeros",
        padded=True,
    )
    mag = np.mean(np.abs(Zxx), axis=1)
    return f, mag


def aggregate_group_spectra(
    patches: list[np.ndarray],
    mask: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    nfft: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    freqs_ref: np.ndarray | None = None
    mags: list[np.ndarray] = []
    for i, p in enumerate(patches):
        if not mask[i]:
            continue
        f, mag = patch_stft_mean_magnitude(p, fs, nperseg, noverlap, nfft)
        if f.size == 0:
            continue
        if freqs_ref is None:
            freqs_ref = f
        elif f.shape != freqs_ref.shape or not np.allclose(f, freqs_ref):
            mag = np.interp(freqs_ref, f, mag)
        mags.append(mag)
    if not mags or freqs_ref is None:
        return None, None
    return freqs_ref, np.mean(np.stack(mags, axis=0), axis=0)


def process_one_stft_window(
    args: argparse.Namespace,
    y_full: np.ndarray,
    t0: int,
    hsic_path: str,
    compute_hsic: bool,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    For each encoder layer, split patches by MI quantiles and plot mean STFT for high vs low.
    Returns list of (layer_idx, f_axis, mag_hi, mag_lo) for optional cross-window aggregation.
    """
    T = int(args.seq_len)
    stride = args.patch_len if args.stride < 0 else args.stride
    x_win = y_full[t0 : t0 + T]
    patches = _mpc.window_patches(x_win, args.patch_len, stride)
    n_p = len(patches)
    if n_p == 0:
        raise RuntimeError("No patches")

    hsic_mat, hsic_desc = load_hsic_matrix_full(args, t0, n_p, hsic_path, compute_hsic)
    n_layers = int(hsic_mat.shape[0])
    layer_indices = _parse_layers_arg(args.layers, n_layers)
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    nperseg = min(int(args.stft_nperseg), args.patch_len)
    noverlap = int(args.stft_noverlap) if int(args.stft_noverlap) > 0 else max(0, nperseg // 2)
    nfft = int(args.stft_nfft) if int(args.stft_nfft) > 0 else None

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv))[0]

    out_for_agg: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    meta_layers: list[dict] = []

    for li in layer_indices:
        if li < 0 or li >= n_layers:
            continue
        mi_scores = np.asarray(hsic_mat[li], dtype=np.float64).ravel()[:n_p]
        if mi_scores.size < n_p:
            tmp = np.full(n_p, np.nan)
            tmp[: mi_scores.size] = mi_scores
            mi_scores = tmp

        finite = mi_scores[np.isfinite(mi_scores)]
        if finite.size < 4:
            print(f"WARNING: layer {li}: insufficient finite MI values; skip.", flush=True)
            continue

        high_mask, low_mask, hi_th, lo_th = mi_group_masks(args, mi_scores)

        f_hi, mag_hi = aggregate_group_spectra(
            patches, high_mask, args.fs, nperseg, noverlap, nfft
        )
        f_lo, mag_lo = aggregate_group_spectra(
            patches, low_mask, args.fs, nperseg, noverlap, nfft
        )

        mi_label = f"{hsic_desc}  layer {li}/{n_layers - 1}"
        md: dict = {
            "layer": li,
            "mi_label": mi_label,
            "mi_group_mode": getattr(args, "mi_group_mode", "symmetric"),
            "high_quantile_threshold": hi_th,
            "n_high": int(high_mask.sum()),
            "n_low": int(low_mask.sum()),
        }
        if lo_th is not None:
            md["low_quantile_threshold"] = lo_th
        meta_layers.append(md)

        grp = assign_mi_group_labels(n_p, high_mask, low_mask, args.mi_group_mode)
        pd.DataFrame(
            {
                "layer": li,
                "patch_index": np.arange(n_p),
                "mi_score": mi_scores,
                "group": grp,
            }
        ).to_csv(
            os.path.join(args.out_dir, f"{base}_mi_stft_groups_t{t0}_layer{li:02d}.csv"),
            index=False,
        )

        if (
            f_hi is None
            or f_lo is None
            or int(high_mask.sum()) < args.min_per_group
            or int(low_mask.sum()) < args.min_per_group
        ):
            print(
                f"WARNING: layer {li}: STFT overlay skipped (high={int(high_mask.sum())}, "
                f"low={int(low_mask.sum())}).",
                flush=True,
            )
            continue

        if f_hi.shape != f_lo.shape or not np.allclose(f_hi, f_lo):
            mag_lo = np.interp(f_hi, f_lo, mag_lo)

        out_for_agg.append((li, f_hi.copy(), mag_hi.copy(), mag_lo.copy()))

        if args.mi_group_mode == "high_vs_rest":
            leg_hi = (
                f"high MI (≥ p{args.high_quantile*100:.0f}={hi_th:.4g}, top ~"
                f"{100.0 * (1.0 - float(args.high_quantile)):.0f}%)"
            )
            leg_lo = "low MI (all other finite patches)"
        else:
            leg_hi = f"high MI (≥ p{args.high_quantile*100:.0f}={hi_th:.4g})"
            leg_lo = f"low MI (≤ p{args.low_quantile*100:.0f}={lo_th:.4g})"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(f_hi, mag_hi, label=leg_hi, lw=2)
        ax.plot(f_hi, mag_lo, label=leg_lo, lw=2)
        ax.set_xlabel("Frequency (cycles / sample @ fs=1)")
        ax.set_ylabel("Mean |STFT| (mean over patches in group, then mean over STFT frames)")
        ax.set_title(f"{base}  t0={t0}  {mi_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = os.path.join(
            args.out_dir, f"{base}_mi_stft_high_vs_low_t{t0}_layer{li:02d}.png"
        )
        fig.savefig(p1, dpi=150)
        plt.close(fig)

        diff = mag_hi - mag_lo
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(f_hi, diff, color="C2", lw=2)
        ax2.axhline(0.0, color="k", lw=0.8)
        ax2.set_xlabel("Frequency (cycles / sample)")
        ax2.set_ylabel("Δ mean|STFT| (high − low)")
        ax2.set_title(f"Layer {li}: high-MI minus low-MI (t0={t0})")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        p2 = os.path.join(args.out_dir, f"{base}_mi_stft_diff_t{t0}_layer{li:02d}.png")
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)

        print("Wrote:", p1, p2, flush=True)

    meta = {
        "csv": os.path.abspath(args.csv),
        "t_start": t0,
        "seq_len": T,
        "patch_len": args.patch_len,
        "stride": stride,
        "mi_group_mode": getattr(args, "mi_group_mode", "symmetric"),
        "high_quantile": float(args.high_quantile),
        "low_quantile": float(args.low_quantile),
        "hsic_source": hsic_desc,
        "n_layers": n_layers,
        "layers_plotted": layer_indices,
        "layers_detail": meta_layers,
        "n_patches": n_p,
        "stft": {
            "fs": args.fs,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": nfft,
        },
    }
    with open(os.path.join(args.out_dir, f"{base}_mi_stft_meta_t{t0}.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    return out_for_agg


def aggregate_layers_across_windows(
    args: argparse.Namespace,
    base: str,
    by_layer: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    n_windows: int,
) -> None:
    """Mean (f, mag_hi, mag_lo) per layer across time windows; same frequency grid via interpolation."""
    for li, curves in by_layer.items():
        if len(curves) < 2:
            continue
        f0 = curves[0][0]
        mhis = []
        mlos = []
        for f_i, mh, ml in curves:
            if f_i.shape != f0.shape or not np.allclose(f_i, f0):
                mhis.append(np.interp(f0, f_i, mh))
                mlos.append(np.interp(f0, f_i, ml))
            else:
                mhis.append(mh)
                mlos.append(ml)
        mh_mean = np.mean(np.stack(mhis, axis=0), axis=0)
        ml_mean = np.mean(np.stack(mlos, axis=0), axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(f0, mh_mean, label="high MI (mean over windows)", lw=2)
        ax.plot(f0, ml_mean, label="low MI (mean over windows)", lw=2)
        ax.set_xlabel("Frequency (cycles / sample @ fs=1)")
        ax.set_ylabel("Mean |STFT| (patches → STFT, then mean over windows)")
        if getattr(args, "mi_group_mode", "symmetric") == "high_vs_rest":
            agg_title = (
                f"{base}  layer {li}: aggregated over {n_windows} window(s), "
                f"high_vs_rest, high≥p{args.high_quantile}"
            )
        else:
            agg_title = (
                f"{base}  layer {li}: aggregated over {n_windows} window(s), "
                f"q_high={args.high_quantile}, q_low={args.low_quantile}"
            )
        ax.set_title(agg_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = os.path.join(
            args.out_dir,
            f"{base}_mi_stft_high_vs_low_aggW{n_windows}_layer{li:02d}.png",
        )
        fig.savefig(p1, dpi=150)
        plt.close(fig)

        diff = mh_mean - ml_mean
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(f0, diff, color="C2", lw=2)
        ax2.axhline(0.0, color="k", lw=0.8)
        ax2.set_xlabel("Frequency (cycles / sample)")
        ax2.set_ylabel("Δ mean|STFT| (high − low)")
        ax2.set_title(f"Layer {li}: aggregated spectral difference ({n_windows} windows)")
        ax2.grid(True, alpha=0.3)
        p2 = os.path.join(
            args.out_dir,
            f"{base}_mi_stft_diff_aggW{n_windows}_layer{li:02d}.png",
        )
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        print("Wrote aggregated:", p1, p2, flush=True)


def compute_mean_hsic_matrix(
    args: argparse.Namespace,
    y_full: np.ndarray,
    windows: list[int],
    hsic_path: str,
    compute_hsic: bool,
) -> tuple[np.ndarray, str]:
    """
    Mean HSIC matrix [n_layers, n_p] over sliding windows. Static npy is loaded once (no averaging).
    """
    T = int(args.seq_len)
    stride = args.patch_len if args.stride < 0 else args.stride
    t0 = windows[0]
    x_win = y_full[t0 : t0 + T]
    patches = _mpc.window_patches(x_win, args.patch_len, stride)
    n_p = len(patches)
    if n_p == 0:
        raise RuntimeError("No patches in first window")

    if not compute_hsic:
        mat, desc = load_hsic_matrix_full(args, t0, n_p, hsic_path, compute_hsic)
        return np.asarray(mat, dtype=np.float64), desc + " (single npy; same MI all windows)"

    acc: np.ndarray | None = None
    desc0 = ""
    for t0 in windows:
        x_win = y_full[t0 : t0 + T]
        patches_w = _mpc.window_patches(x_win, args.patch_len, stride)
        if len(patches_w) != n_p:
            raise ValueError(
                f"Inconsistent patch count: t0={t0} has {len(patches_w)} patches, "
                f"expected {n_p} (fix patch_len/stride/seq_len)."
            )
        mat, desc0 = load_hsic_matrix_full(args, t0, n_p, hsic_path, compute_hsic)
        m = np.asarray(mat, dtype=np.float64)
        acc = m if acc is None else acc + m
    assert acc is not None
    wn = float(len(windows))
    mean_mat = acc / wn
    return mean_mat, f"{desc0} (mean MI over {len(windows)} windows)"


def process_dataset_averaged_stft(
    args: argparse.Namespace,
    y_full: np.ndarray,
    windows: list[int],
    hsic_mean: np.ndarray,
    hsic_desc: str,
) -> None:
    """
    One quantile split from dataset-mean MI per layer; STFT group means averaged across windows.
    """
    T = int(args.seq_len)
    stride = args.patch_len if args.stride < 0 else args.stride
    base = os.path.splitext(os.path.basename(args.csv))[0]
    W = len(windows)
    os.makedirs(args.out_dir, exist_ok=True)

    np.save(
        os.path.join(args.out_dir, f"{base}_dataset_avg_hsic_mean.npy"),
        np.asarray(hsic_mean, dtype=np.float64),
    )

    t00 = windows[0]
    patches0 = _mpc.window_patches(y_full[t00 : t00 + T], args.patch_len, stride)
    n_p = len(patches0)

    n_layers = int(hsic_mean.shape[0])
    layer_indices = _parse_layers_arg(args.layers, n_layers)
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    nperseg = min(int(args.stft_nperseg), args.patch_len)
    noverlap = int(args.stft_noverlap) if int(args.stft_noverlap) > 0 else max(0, nperseg // 2)
    nfft = int(args.stft_nfft) if int(args.stft_nfft) > 0 else None

    meta_layers: list[dict] = []

    for li in layer_indices:
        if li < 0 or li >= n_layers:
            continue
        mi_scores = np.asarray(hsic_mean[li], dtype=np.float64).ravel()[:n_p]
        if mi_scores.size < n_p:
            tmp = np.full(n_p, np.nan)
            tmp[: mi_scores.size] = mi_scores
            mi_scores = tmp

        finite = mi_scores[np.isfinite(mi_scores)]
        if finite.size < 4:
            print(f"WARNING: layer {li}: insufficient finite MI; skip.", flush=True)
            continue

        high_mask, low_mask, hi_th, lo_th = mi_group_masks(args, mi_scores)

        mi_label = f"{hsic_desc}  layer {li}/{n_layers - 1}"

        mhi_list: list[np.ndarray] = []
        mlo_list: list[np.ndarray] = []
        f_ref: np.ndarray | None = None

        for t0 in windows:
            x_win = y_full[t0 : t0 + T]
            patches = _mpc.window_patches(x_win, args.patch_len, stride)
            f_hi, mag_hi = aggregate_group_spectra(
                patches, high_mask, args.fs, nperseg, noverlap, nfft
            )
            f_lo, mag_lo = aggregate_group_spectra(
                patches, low_mask, args.fs, nperseg, noverlap, nfft
            )
            if f_hi is None or f_lo is None:
                continue
            if f_ref is None:
                f_ref = f_hi
            elif f_hi.shape != f_ref.shape or not np.allclose(f_hi, f_ref):
                mag_hi = np.interp(f_ref, f_hi, mag_hi)
            if f_lo.shape != f_ref.shape or not np.allclose(f_lo, f_ref):
                mag_lo = np.interp(f_ref, f_lo, mag_lo)
            mhi_list.append(mag_hi)
            mlo_list.append(mag_lo)

        md_ds: dict = {
            "layer": li,
            "mi_label": mi_label,
            "mi_group_mode": getattr(args, "mi_group_mode", "symmetric"),
            "high_quantile_threshold": hi_th,
            "n_high": int(high_mask.sum()),
            "n_low": int(low_mask.sum()),
            "n_windows_used": len(mhi_list),
        }
        if lo_th is not None:
            md_ds["low_quantile_threshold"] = lo_th
        meta_layers.append(md_ds)

        grp = assign_mi_group_labels(n_p, high_mask, low_mask, args.mi_group_mode)
        pd.DataFrame(
            {
                "layer": li,
                "patch_index": np.arange(n_p),
                "mi_score_mean_over_windows": mi_scores,
                "group": grp,
            }
        ).to_csv(
            os.path.join(
                args.out_dir,
                f"{base}_mi_stft_groups_dataset_avg_W{W}_layer{li:02d}.csv",
            ),
            index=False,
        )

        if (
            not mhi_list
            or int(high_mask.sum()) < args.min_per_group
            or int(low_mask.sum()) < args.min_per_group
        ):
            print(
                f"WARNING: layer {li}: dataset STFT skipped (windows with spectra={len(mhi_list)}, "
                f"high={int(high_mask.sum())}, low={int(low_mask.sum())}).",
                flush=True,
            )
            continue

        assert f_ref is not None
        mh_mean = np.mean(np.stack(mhi_list, axis=0), axis=0)
        ml_mean = np.mean(np.stack(mlo_list, axis=0), axis=0)

        if args.mi_group_mode == "high_vs_rest":
            leg_hi = (
                f"high MI (≥ p{args.high_quantile*100:.0f}={hi_th:.4g}, top ~"
                f"{100.0 * (1.0 - float(args.high_quantile)):.0f}%)"
            )
            leg_lo = "low MI (all other finite patches)"
        else:
            leg_hi = f"high MI (≥ p{args.high_quantile*100:.0f}={hi_th:.4g})"
            leg_lo = f"low MI (≤ p{args.low_quantile*100:.0f}={lo_th:.4g})"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(f_ref, mh_mean, label=leg_hi, lw=2)
        ax.plot(f_ref, ml_mean, label=leg_lo, lw=2)
        ax.set_xlabel("Frequency (cycles / sample @ fs=1)")
        ax.set_ylabel(
            "Mean |STFT| (mean over patches in group per window, then mean over windows)"
        )
        ax.set_title(
            f"{base}  dataset avg  W={W}  t0 in [{windows[0]}, …]  {mi_label}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p1 = os.path.join(
            args.out_dir,
            f"{base}_mi_stft_dataset_avg_W{W}_layer{li:02d}.png",
        )
        fig.savefig(p1, dpi=150)
        plt.close(fig)

        diff = mh_mean - ml_mean
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(f_ref, diff, color="C2", lw=2)
        ax2.axhline(0.0, color="k", lw=0.8)
        ax2.set_xlabel("Frequency (cycles / sample)")
        ax2.set_ylabel("Δ mean|STFT| (high − low)")
        ax2.set_title(f"Layer {li}: dataset-averaged spectral difference (W={W} windows)")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        p2 = os.path.join(
            args.out_dir,
            f"{base}_mi_stft_diff_dataset_avg_W{W}_layer{li:02d}.png",
        )
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        print("Wrote dataset-averaged:", p1, p2, flush=True)

    meta = {
        "mode": "dataset_average",
        "csv": os.path.abspath(args.csv),
        "windows": windows,
        "n_windows": W,
        "window_hop_used": int(windows[1] - windows[0]) if len(windows) >= 2 else None,
        "full_dataset_mode": getattr(args, "full_dataset_mode", "exp_split"),
        "dataset_flag": getattr(args, "dataset_flag", "test"),
        "benchmark_data": getattr(args, "data", "ETTh1"),
        "seq_len": T,
        "patch_len": args.patch_len,
        "stride": stride,
        "mi_group_mode": getattr(args, "mi_group_mode", "symmetric"),
        "high_quantile": float(args.high_quantile),
        "low_quantile": float(args.low_quantile),
        "hsic_source": hsic_desc,
        "n_layers": n_layers,
        "layers_plotted": layer_indices,
        "layers_detail": meta_layers,
        "n_patches": n_p,
        "stft": {
            "fs": args.fs,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": nfft,
        },
    }
    with open(
        os.path.join(args.out_dir, f"{base}_mi_stft_meta_dataset_avg.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(meta, fp, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "STFT comparison: high-MI vs low-MI patches. "
            "Default: one overlay + diff + CSV per encoder layer (--layers)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All layers, single window, HSIC from aggregate npy
  python experiments/mi_patch_stft_compare.py --csv data.csv --hsic_npy aggregate/hsic_mean.npy

  # Only first and last encoder layers
  python experiments/mi_patch_stft_compare.py --csv data.csv --hsic_npy hsic.npy --layers "0,-1"

  # Multiple sliding windows; also average STFT curves across windows (per layer)
  python experiments/mi_patch_stft_compare.py --csv data.csv --hsic_npy hsic.npy \\
    --num_samples 5 --window_hop 672 --aggregate_windows 1 --out_dir ./figures/mi_stft

  # All valid sliding windows: mean MI over windows, mean STFT over windows (needs --ckpt for varying MI)
  python experiments/mi_patch_stft_compare.py --csv data.csv --ckpt_path ckpt.ckpt \\
    --use_full_dataset 1 --window_hop 672 --cpu --out_dir ./figures/mi_stft_full
""",
    )
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--target_col", type=str, default="OT")
    ap.add_argument("--t_start", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=1, help="Windows: t_start + i*window_hop")
    ap.add_argument("--window_hop", type=int, default=-1, help="Default: seq_len")
    ap.add_argument("--seq_len", type=int, default=672)
    ap.add_argument("--patch_len", type=int, default=96)
    ap.add_argument("--stride", type=int, default=-1)
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--hsic_npy", type=str, default="")
    ap.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices to plot (default all), e.g. 0,1,7 or -1 for last layer only",
    )
    ap.add_argument(
        "--aggregate_windows",
        type=int,
        default=0,
        help="1 = after all time windows, save extra figures averaging STFT curves across windows (per layer)",
    )
    ap.add_argument(
        "--use_full_dataset",
        type=int,
        default=0,
        help="1 = aggregate over many windows; mean MI per layer, then one quantile split; "
        "mean group STFT over windows. Window list: --full_dataset_mode.",
    )
    ap.add_argument(
        "--full_dataset_mode",
        type=str,
        default="exp_split",
        choices=["exp_split", "sliding"],
        help="exp_split: same global t0 grid as exp_forecast use_ims (CIAutoRegressionDatasetBenchmark). "
        "sliding: t0=t_start, t_start+hop, ... on full CSV (--window_hop).",
    )
    ap.add_argument(
        "--dataset_flag",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Train/val/test split for full_dataset_mode=exp_split (matches data_provider flag).",
    )
    ap.add_argument(
        "--data",
        type=str,
        default="ETTh1",
        help="Benchmark data_type for exp_split (ETTh1, ETTh2, ETTm1, ...).",
    )
    ap.add_argument("--freq", type=str, default="h")
    ap.add_argument(
        "--loader_stride",
        type=int,
        default=1,
        help="Stride for CIAutoRegressionDatasetBenchmark (exp_forecast --stride); not STFT patch stride.",
    )
    ap.add_argument(
        "--subset_rand_ratio",
        type=float,
        default=1.0,
        help="Same as exp_forecast; affects train CIAutoRegressionDatasetBenchmark length.",
    )
    ap.add_argument(
        "--write_per_window",
        type=int,
        default=0,
        help="1 = with --use_full_dataset 1, also write per-t0 figures (slow, many files).",
    )
    ap.add_argument("--out_dir", type=str, default="./figures/mi_stft_compare")
    ap.add_argument(
        "--mi_group_mode",
        type=str,
        default="symmetric",
        choices=["symmetric", "high_vs_rest"],
        help="symmetric: high>=p_high and low<=p_low (mid otherwise). "
        "high_vs_rest: high>=p_high (default ~top 25%% when p_high=0.75); "
        "low = all other finite patches (no mid).",
    )
    ap.add_argument("--high_quantile", type=float, default=0.75)
    ap.add_argument("--low_quantile", type=float, default=0.25)
    ap.add_argument("--min_per_group", type=int, default=2)
    ap.add_argument("--fs", type=float, default=1.0)
    ap.add_argument("--stft_nperseg", type=int, default=64)
    ap.add_argument("--stft_noverlap", type=int, default=-1)
    ap.add_argument("--stft_nfft", type=int, default=0)
    ap.add_argument("--label_len", type=int, default=576)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--output_len", type=int, default=96)
    ap.add_argument("--use_ims", type=int, default=1)
    ap.add_argument("--timer_patch_len", type=int, default=96)
    ap.add_argument("--timer_stride", type=int, default=1)
    ap.add_argument("--timer_features", type=str, default="M")
    ap.add_argument("--embed", type=str, default="timeF")
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--e_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--periodic_embedding_branch", type=int, default=0)
    ap.add_argument("--periodic_emb_bank_dim", type=int, default=0)
    ap.add_argument("--geometric_hpe", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = (args.ckpt_path or "").strip()
    hsic_path = (args.hsic_npy or "").strip()
    if not ckpt_path and not hsic_path:
        print("ERROR: provide --ckpt_path or --hsic_npy", file=sys.stderr)
        sys.exit(2)
    if ckpt_path and not os.path.isfile(ckpt_path):
        print(f"ERROR: ckpt not found: {ckpt_path}", file=sys.stderr)
        sys.exit(2)
    if hsic_path and not os.path.isfile(hsic_path):
        print(f"ERROR: hsic_npy not found: {hsic_path}", file=sys.stderr)
        sys.exit(2)

    try:
        target: str | int = int(args.target_col)
    except ValueError:
        target = args.target_col

    _, y_full = _mpc.load_series_1d(args.csv, target)
    T = int(args.seq_len)
    t0_base = int(args.t_start)
    _stride = args.patch_len if args.stride < 0 else args.stride
    if t0_base + T > len(y_full):
        print(
            f"ERROR: need series length >= t_start + seq_len ({t0_base + T}), got {len(y_full)}",
            file=sys.stderr,
        )
        sys.exit(2)
    _n_p_grid = len(
        _mpc.window_patches(y_full[t0_base : t0_base + T], args.patch_len, _stride)
    )
    print(
        f">>> Patch grid: n_patches={_n_p_grid} ({_patch_grid_str(args)})",
        flush=True,
    )
    try:
        compute_hsic, hsic_path_use = resolve_compute_hsic(
            _n_p_grid, ckpt_path, hsic_path, _patch_grid_str(args)
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    use_full = int(args.use_full_dataset) == 1

    windows: list[int] = []
    hop = int(args.window_hop) if int(args.window_hop) > 0 else T

    if use_full:
        wsched = str(getattr(args, "full_dataset_mode", "exp_split"))
        if wsched == "exp_split":
            try:
                windows, win_desc = build_exp_forecast_ims_window_starts(args)
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(2)
            print(f">>> use_full_dataset mode=exp_split ({win_desc})", flush=True)
            if len(windows) > 400:
                print(
                    f"WARNING: {len(windows)} windows — live HSIC will be very slow. "
                    "Consider full_dataset_mode=sliding with fewer hops, or HSIC npy with matching P.",
                    flush=True,
                )
            print(
                f">>> {len(windows)} window(s), global t0 from {windows[0]} to {windows[-1]}",
                flush=True,
            )
        else:
            t0 = t0_base
            while t0 + T <= len(y_full):
                windows.append(t0)
                t0 += hop
            if not windows:
                raise ValueError("No valid windows (series shorter than seq_len?).")
            print(
                f">>> use_full_dataset mode=sliding: {len(windows)} window(s), hop={hop}, "
                f"t0 from {windows[0]} to {windows[-1]}",
                flush=True,
            )
            win_desc = f"sliding hop={hop} from t_start={t0_base}"

        hsic_mean, hsic_desc = compute_mean_hsic_matrix(
            args, y_full, windows, hsic_path_use, compute_hsic
        )
        hsic_desc = f"{hsic_desc} | {win_desc}"
        if not compute_hsic:
            print(
                "NOTE: --hsic_npy is fixed; MI does not vary across windows — "
                "dataset MI average equals a single load. Use --ckpt_path for per-window MI.",
                flush=True,
            )
        process_dataset_averaged_stft(args, y_full, windows, hsic_mean, hsic_desc)
        if int(args.write_per_window) != 1:
            return

    ns = max(1, int(args.num_samples))
    hop = int(args.window_hop) if int(args.window_hop) > 0 else T
    if not use_full:
        windows = []
        if ns == 1:
            windows = [t0_base]
        else:
            for i in range(ns):
                t0 = t0_base + i * hop
                if t0 + T > len(y_full):
                    break
                windows.append(t0)
        if not windows:
            raise ValueError("No valid windows.")
        if ns > 1 and len(windows) < ns:
            print(f"num_samples capped to {len(windows)}.", flush=True)

        if int(args.aggregate_windows) == 1 and len(windows) < 2:
            print(
                "NOTE: --aggregate_windows 1 only writes aggW* figures when there are "
                f"at least 2 valid windows; got {len(windows)}. Increase --num_samples or "
                "shorten --seq_len / series length.",
                flush=True,
            )

    by_layer: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    base_name = os.path.splitext(os.path.basename(args.csv))[0]

    for wi, t0 in enumerate(windows):
        print(f">>> STFT window {wi + 1}/{len(windows)}  t_start={t0}", flush=True)
        rows = process_one_stft_window(args, y_full, t0, hsic_path_use, compute_hsic)
        for li, f_ax, mh, ml in rows:
            by_layer[li].append((f_ax, mh, ml))

    if int(args.aggregate_windows) == 1 and len(windows) > 1 and by_layer:
        print(
            f">>> Aggregating STFT across {len(windows)} window(s) (per layer)...",
            flush=True,
        )
        aggregate_layers_across_windows(args, base_name, by_layer, len(windows))


if __name__ == "__main__":
    main()
