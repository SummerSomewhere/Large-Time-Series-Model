#!/usr/bin/env python3
"""
Step 1: Extract Timer decoder layer-wise per-patch features.

Collects the pooled hidden representation [B, N, D] after each decoder
attention block via a hook-based forward pass (same logic as etth1_mi_hsic_peaks.py).
Saves one .pt file containing per-layer per-patch features for all test samples.

Output:
    {out_dir}/{dataset}_{n_samples}.pt  — torch.save with:
        features:  dict[int -> Tensor [S, N, D]]  per-layer patch features
        labels:    dict[str -> np.ndarray [S]]     semantic label arrays
        d_model:   int
        n_layers:   int
        N:          int   (num patches)
        sample_indices: np.ndarray [S]

Usage (Mode A — load pre-extracted):
    python experiments/etth1_linear_probe_patch_compare.py \
        --features_pt_path ./results/layerwise_probe_features/ETTh1_20000.pt \
        --labels_json ...

Usage (Mode B — extract on the fly, no --features_pt_path):
    python experiments/etth1_linear_probe_patch_compare.py \
        --ckpt_path ... --root_path ... --data_path ...

This script provides Mode B's upstream extraction so that Mode A can run offline:
    python experiments/etth1_layerwise_probe_features.py \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --root_path ./datasets/ --data_path ETTh1.csv \
        --seq_len 672 --pred_len 96 --label_len 576 --patch_len 96 \
        --e_layers 8 --factor 3 --d_model 1024 --d_ff 2048 --n_heads 8 \
        --out_dir ./results/layerwise_probe_features/ \
        --n_samples 20000 --batch_size 64 --num_workers 6
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model as TimerModel
from data_provider.data_factory import data_provider
from utils.masking import TriangularCausalMask


MI_DECODER_LAYER_CAP = 8


# ── Model unwrapping ────────────────────────────────────────────────────────

def _unwrap_timer(model):
    if hasattr(model, "module"):
        return model.module
    return model


# ── Forward helpers (identical to etth1_mi_hsic_peaks.py) ───────────────────

def forward_collect_layers(model, x_enc):
    """
    x_enc: [B, L, M] (Timer convention).
    Returns: list of [B, N, D] tensors — one per decoder layer.
    """
    core = _unwrap_timer(model)
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev

    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)   # [BM, N, D]
    BM, N, D = dec_in.shape
    assert BM == B * n_vars

    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers_out = []
    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _ = o3(h, attn_mask=mask)
        layers_out.append(pool(h.detach()))   # [B, N, D]
    return layers_out, int(n_vars), int(N)


# ── Semantic label extraction via STL decomposition ──────────────────────────

def extract_stl_labels(raw_df: np.ndarray, n_samples: int, stl_period: int = 24) -> dict[str, np.ndarray]:
    """
    Apply STL decomposition to the first principal component of raw_df.
    Returns dict of [n_samples] arrays:
        trend, seasonal, residual, trend_slope, residual_energy
    """
    from scipy.signal import detrend as sp_detrend
    from scipy.stats import linregress

    n_vars = raw_df.shape[1]
    all_series = []

    # Build per-variate mean time series and average across variates
    for v in range(n_vars):
        series = raw_df[:, v]
        all_series.append(series)
    combined = np.mean(all_series, axis=0)   # [T,]

    # Use last n_samples of combined series
    if combined.shape[0] > n_samples:
        combined = combined[-n_samples:]
    series = combined[:n_samples]
    T = series.shape[0]
    period = min(stl_period, T // 3)
    period = max(period, 2)

    # Trend: 2-pass moving-average + linear interpolation
    window = max(period if period % 2 == 1 else period + 1, 3)
    kernel = np.ones(window) / window
    trend_raw = np.convolve(series, kernel, mode="same")
    half = window // 2
    trend_raw[:half] = np.nan
    trend_raw[-half:] = np.nan
    valid = ~np.isnan(trend_raw)
    if valid.sum() > 2:
        x_coord = np.arange(T)[valid]
        trend_interp = np.empty(T)
        trend_interp[valid] = trend_raw[valid]
        trend_interp[~valid] = np.interp(np.where(~valid)[0], x_coord, trend_raw[valid])
    else:
        trend_interp = np.full(T, np.nanmean(trend_raw))

    # Seasonal: period-averaged deviation from trend
    detrended = series - trend_interp
    n_full = T // period
    seasonal = np.zeros(T)
    if n_full > 0:
        cut = n_full * period
        reshaped = detrended[:cut].reshape(n_full, period)
        period_mean = reshaped.mean(axis=0)
        period_mean -= period_mean.mean()
        for i in range(n_full):
            seasonal[i * period:(i + 1) * period] = period_mean
        seasonal[n_full * period:] = period_mean[:T - n_full * period]

    # Residual
    residual = series - trend_interp - seasonal

    # Per-sample labels: windowed trend slope and residual energy
    win = max(period, 24)
    n_windows = max(T // win, 1)
    labels: dict[str, np.ndarray] = {}

    trend_slopes = np.full(n_windows, np.nan)
    residual_energies = np.full(n_windows, np.nan)
    for w in range(n_windows):
        start = w * win
        end = min((w + 1) * win, T)
        if end - start < 3:
            continue
        tseg = trend_interp[start:end]
        slope, _, _, _, _ = linregress(np.arange(end - start), tseg)
        trend_slopes[w] = slope
        residual_energies[w] = float(np.mean(residual[start:end] ** 2))

    # Resample labels back to per-sample using nearest interpolation
    window_centers = np.arange(n_windows) * win + win // 2
    window_centers = np.clip(window_centers, 0, T - 1)

    for lbl_name, lbl_vals, arr in [
        ("trend_slope", trend_slopes, np.zeros(T)),
        ("residual_energy", residual_energies, np.zeros(T)),
    ]:
        if len(window_centers) > 0 and not np.all(np.isnan(lbl_vals)):
            valid_mask = ~np.isnan(lbl_vals)
            if valid_mask.sum() > 0:
                arr[:] = np.interp(np.arange(T), window_centers[valid_mask], lbl_vals[valid_mask])
            else:
                arr[:] = 0.0
        else:
            arr[:] = 0.0
        labels[lbl_name] = arr

    # Global per-sample labels
    labels["trend_mean"] = trend_interp
    labels["seasonal_amplitude"] = np.abs(seasonal)
    labels["residual_std"] = np.abs(residual)

    # Also add per-variate statistics for richness
    for v in range(min(n_vars, 3)):   # first 3 variates only to keep label dim manageable
        v_series = raw_df[-n_samples:, v] if raw_df.shape[0] > n_samples else raw_df[:, v][:n_samples]
        labels[f"var{v}_mean"] = v_series
        labels[f"var{v}_std"] = (v_series - v_series.mean()) ** 2

    # Truncate to exactly n_samples
    for k in labels:
        labels[k] = labels[k][:n_samples]

    return labels


def build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    for k, v in {
        "task_name": "forecast",
        "is_training": 0,
        "is_finetuning": 0,
        "train_test": 0,
        "use_multi_gpu": False,
        "d_layers": 1,
        "target": "OT",
        "checkpoints": "./checkpoints/",
        "inverse": False,
        "use_amp": False,
        "use_weight_decay": 0,
        "weight_decay": 0.01,
        "loss": "MSE",
        "lradj": "type1",
        "train_epochs": 0,
        "patience": 3,
        "learning_rate": 1e-4,
        "itr": 1,
        "finetune_epochs": 0,
        "output_attention": False,
        "distil": True,
        "model_id": "probe_features",
        "model": "Timer",
        "output_len_list": None,
        "mask_rate": 0.25,
        "data_type": "custom",
        "decay_fac": 0.75,
        "cos_warm_up_steps": 100,
        "cos_max_decay_steps": 60000,
        "cos_max_decay_epoch": 10,
        "cos_max": 1e-4,
        "cos_min": 2e-6,
    }.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Timer layer-wise feature extraction for linear probing")
    # Model / data
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=576)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--subset_rand_ratio", type=float, default=1.0)
    parser.add_argument("--use_ims", action="store_true")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Extraction control
    parser.add_argument("--n_samples", type=int, default=20000,
                        help="Number of test samples to extract")
    parser.add_argument("--out_dir", type=str, default="./results/layerwise_probe_features/")
    parser.add_argument("--stl_period", type=int, default=24,
                        help="STL period for seasonal decomposition (24 for hourly ETTh1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("Timer Layer-wise Feature Extraction")
    print("=" * 60)
    print(f"  ckpt_path  : {args.ckpt_path}")
    print(f"  data       : {args.data}")
    print(f"  seq_len    : {args.seq_len}")
    print(f"  n_samples  : {args.n_samples}")
    print(f"  device     : {device}")
    print(f"  out_dir    : {args.out_dir}")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("\n[1] Loading Timer model...")
    ns = build_namespace(args)
    model = TimerModel(ns)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    core = _unwrap_timer(model)
    print(f"  Loaded. Decoder layers: {len(core.decoder.attn_layers)}")

    # ── 2. Load dataset ─────────────────────────────────────────────────────
    print("\n[2] Building test data loader...")
    ns.batch_size = args.batch_size
    _, loader = data_provider(ns, "test")
    n_batches = len(loader)
    print(f"  Test loader: {n_batches} batches x {args.batch_size} = ~{n_batches * args.batch_size} samples")

    # ── 3. Extract features ─────────────────────────────────────────────────
    print("\n[3] Extracting layer features...")
    per_layer: list[list[torch.Tensor]] = [ [] for _ in range(MI_DECODER_LAYER_CAP) ]
    n_extracted = 0
    max_samples = args.n_samples

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        if n_extracted >= max_samples:
            break
        B = batch_x.shape[0]
        keep = min(B, max_samples - n_extracted)
        x_batch = batch_x[:keep].float().to(device)

        with torch.no_grad():
            layers_out, n_vars, N = forward_collect_layers(model, x_batch)

        for li in range(len(layers_out)):
            per_layer[li].append(layers_out[li][:keep].cpu())

        n_extracted += keep
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{n_batches}: {n_extracted} samples extracted")
        if n_extracted >= max_samples:
            break

    # Concatenate
    features: dict[int, torch.Tensor] = {}
    for li in range(MI_DECODER_LAYER_CAP):
        if per_layer[li]:
            features[li] = torch.cat(per_layer[li], dim=0)   # [S, N, D]
    n_layers = len(features)
    d_model = next(iter(features.values())).shape[2]
    N = next(iter(features.values())).shape[1]
    total_extracted = next(iter(features.values())).shape[0]
    print(f"  Extracted: {n_layers} layers, {total_extracted} samples, {N} patches, d_model={d_model}")

    # ── 4. Extract semantic labels ───────────────────────────────────────────
    print(f"\n[4] Extracting semantic labels (STL period={args.stl_period})...")
    import pandas as pd
    csv_path = os.path.join(args.root_path, args.data_path)
    df_raw = pd.read_csv(csv_path)
    raw_vals = df_raw.values.astype(np.float64)   # [T, n_vars]
    labels = extract_stl_labels(raw_vals, total_extracted, stl_period=args.stl_period)
    print(f"  Labels: {list(labels.keys())}")
    for k, v in labels.items():
        print(f"    {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    # ── 5. Save ─────────────────────────────────────────────────────────────
    dataset_name = args.data if args.data != "custom" else os.path.splitext(args.data_path)[0]
    out_path = os.path.join(args.out_dir, f"{dataset_name}_{total_extracted}.pt")
    labels_path = os.path.join(args.out_dir, f"{dataset_name}_{total_extracted}_labels.json")

    torch.save({
        "features": features,
        "d_model": d_model,
        "n_layers": n_layers,
        "N": N,
        "n_samples": total_extracted,
        "config": vars(args),
    }, out_path)
    print(f"\n[5] Saved features: {out_path}")

    # Save labels as JSON
    labels_serializable = {k: v.tolist() if hasattr(v, "tolist") else list(v) for k, v in labels.items()}
    with open(labels_path, "w") as f:
        json.dump(labels_serializable, f, indent=2)
    print(f"  Saved labels: {labels_path}")

    print(f"\n[Done] Feature extraction complete.")
    print(f"  Run linear probing with --features_pt_path {out_path} --labels_json {labels_path}")


if __name__ == "__main__":
    main()
