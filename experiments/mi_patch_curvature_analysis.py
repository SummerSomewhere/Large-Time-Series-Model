#!/usr/bin/env python3
"""
Map high-MI (or HSIC) patches to raw time series and compare with per-patch second-difference
(curvature), matching Geometric-HPE: k = mean(|diff(x, n=2)|) inside each patch.

Example (live HSIC, same as mi_hsic_layerwise_timer; ETTh test split + --features M, M>=4):
  python experiments/mi_patch_curvature_analysis.py \\
    --csv ./datasets/ETTh1.csv --target_col OT --t_start 10000 \\
    --seq_len 672 --patch_len 96 \\
    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt --layer_row 0 \\
    --out_dir ./figures/mi_curvature_etth1

Example (precomputed npy from mi_hsic_layerwise_timer aggregate):
  python experiments/mi_patch_curvature_analysis.py ... --hsic_npy path/to/hsic_mean.npy

Per-patch MI/HSIC must be computed the same way as experiments/mi_hsic_layerwise_timer.py
(HSIC proxy): either pass --ckpt_path to run Timer forward here, or --hsic_npy from that script,
or scripts/analysis/run_mi_curvature_etth1_computed.sh.
With --ckpt_path, use ETT hourly CSV with standard ETTh borders and --features M (M>=4) so HSIC has n>=4.
MI must be computed: either --ckpt_path (Timer+HSIC in this script) or --hsic_npy from mi_hsic_layerwise_timer.

Multi-window (one process): --num_samples N --window_hop H (default H=seq_len) walks windows
t_start + i*H for i=0..N-1; each step runs HSIC/MI → map to patches → figures/CSV (see scripts/analysis/mi_patch_curvature_etth1.sh).
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


def load_series_1d(csv_path: str, target: str | int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if isinstance(target, int):
        # Numeric columns only (skip date column if first column is non-numeric)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            raise ValueError("No numeric columns in CSV")
        if target < 0 or target >= num_df.shape[1]:
            raise ValueError(f"target index {target} out of range for {num_df.shape[1]} numeric cols")
        y = num_df.iloc[:, target].values.astype(np.float64)
    else:
        if target not in df.columns:
            raise ValueError(f"Column {target!r} not in CSV. Columns: {list(df.columns)}")
        y = pd.to_numeric(df[target], errors="coerce").values.astype(np.float64)
    t = np.arange(len(y), dtype=np.float64)
    return t, y


def num_patches(seq_len: int, patch_len: int, stride: int, padding: int = 0) -> int:
    lp = seq_len + int(padding)
    if lp < patch_len:
        return 0
    return (lp - patch_len) // int(stride) + 1


def patch_curvature_mean(patch_1d: np.ndarray) -> float:
    """Mean abs second difference inside patch (same as Geometric-HPE)."""
    x = np.asarray(patch_1d, dtype=np.float64).ravel()
    if x.size < 3:
        return 0.0
    d2 = np.diff(x, n=2)
    return float(np.mean(np.abs(d2)))


def window_patches(x_win: np.ndarray, patch_len: int, stride: int) -> list[np.ndarray]:
    """Non-overlapping or strided patches covering [0, len(x_win))."""
    out = []
    p = 0
    while p + patch_len <= len(x_win):
        out.append(x_win[p : p + patch_len].copy())
        p += stride
    return out


def load_mi_scores_from_npy(
    n_patches: int,
    hsic_npy: str,
    layer_row: int,
) -> tuple[np.ndarray, str]:
    """
    Load HSIC/MI proxy from a saved npy (same format as mi_hsic_layerwise_timer aggregate).
    For 2D [L,P], layer_row uses Python indexing: -1 = last layer, etc.
    """
    arr = np.load(hsic_npy)
    if arr.ndim == 1:
        row = arr
        mi_label = "HSIC/MI (1D curve, from npy)"
    else:
        n_layers = int(arr.shape[0])
        li = int(layer_row)
        if li < 0:
            li = n_layers + li
        if li < 0 or li >= n_layers:
            raise ValueError(
                f"layer_row={layer_row} resolves to index {li}, but hsic_npy has "
                f"{n_layers} layer(s); use 0..{n_layers - 1} or negative indices (e.g. -1 = last)."
            )
        row = np.asarray(arr[li], dtype=np.float64).ravel()
        layer_note = "last layer" if li == n_layers - 1 else f"layer {li}"
        mi_label = f"HSIC/MI ({layer_note}, {li}/{n_layers - 1}, from npy)"
    if row.size < n_patches:
        pad = np.full(n_patches, np.nan, dtype=np.float64)
        pad[: row.size] = row
        row = pad
    else:
        row = row[:n_patches]
    return row, mi_label


def tukey_high_mask(scores: np.ndarray) -> np.ndarray:
    """Boolean mask: score > Q3 + 1.5*IQR (finite only)."""
    s = scores[np.isfinite(scores)]
    if s.size < 4:
        return np.zeros_like(scores, dtype=bool)
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    thresh = q3 + 1.5 * iqr
    return np.isfinite(scores) & (scores > thresh)


def _ett_hour_test_border1(seq_len: int) -> int:
    """First global time index of ETTh test split (Dataset_ETT_hour border1s[2])."""
    return 12 * 30 * 24 + 4 * 30 * 24 - seq_len


def load_ims_batch_timer(
    csv_path: str,
    t_start: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    features: str,
    target: str,
    embed: str,
):
    """One IMS sample aligned with Timer / mi_hsic_layerwise_timer (ETTh-style borders)."""
    import torch
    from data_provider.data_loader import Dataset_ETT_hour

    root = os.path.dirname(os.path.abspath(csv_path))
    name = os.path.basename(csv_path)
    timeenc = 0 if embed != "timeF" else 1
    ds = Dataset_ETT_hour(
        root_path=root,
        flag="test",
        size=[seq_len, label_len, pred_len],
        features=features,
        data_path=name,
        target=target,
        scale=True,
        timeenc=timeenc,
        freq="h",
    )
    border1 = _ett_hour_test_border1(seq_len)
    idx = int(t_start) - border1
    if idx < 0 or idx >= len(ds):
        raise ValueError(
            f"t_start={t_start} maps to dataset index {idx}; valid global range for test is "
            f"[{border1}, {border1 + len(ds) - 1}] (seq_len={seq_len})."
        )
    seq_x, seq_y, _, _ = ds[idx]
    batch_x = torch.from_numpy(seq_x).float().unsqueeze(0)
    batch_y = torch.from_numpy(seq_y).float().unsqueeze(0)
    return batch_x, batch_y


def load_ims_batch_timer_global_index(
    csv_path: str,
    t_start: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    features: str,
    target: str,
    embed: str,
    pred_horizon: int | None = None,
):
    """
    Same batch_x/batch_y semantics as Dataset_ETT_hour + IMS, but t_start is a **global** CSV row
    index (0 .. len(csv)-1), not restricted to the test split. Scaler is fit on the standard ETT
    train slice only (same as Dataset_ETT_hour). Use this when MI/STFT windows slide over the full
    series (e.g. mi_patch_stft_compare use_full_dataset).
    """
    import torch

    root = os.path.dirname(os.path.abspath(csv_path))
    name = os.path.basename(csv_path)
    df_raw = pd.read_csv(os.path.join(root, name))

    border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    if features == "M" or features == "MS":
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    elif features == "S":
        df_data = df_raw[[target]]
    else:
        raise ValueError(f"features={features!r} not supported for global IMS batch")

    scaler = StandardScaler()
    train_data = df_data[border1s[0] : border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)
    n = int(len(data))

    ph = int(pred_horizon) if pred_horizon is not None else int(pred_len)
    s_begin = int(t_start)
    s_end = s_begin + int(seq_len)
    r_begin = s_end - int(label_len)
    r_end = r_begin + int(label_len) + ph

    if s_begin < 0 or r_end > n:
        raise ValueError(
            f"t_start={t_start} needs indices [0, {n}) for window ending at {r_end} "
            f"(seq_len={seq_len}, label_len={label_len}, pred_horizon={ph})."
        )

    seq_x = data[s_begin:s_end]
    seq_y = data[r_begin:r_end]
    batch_x = torch.from_numpy(seq_x).float().unsqueeze(0)
    batch_y = torch.from_numpy(seq_y).float().unsqueeze(0)
    return batch_x, batch_y


def build_timer_configs(args: argparse.Namespace) -> SimpleNamespace:
    """Minimal configs namespace for models.Timer.Model (matches mi_hsic_layerwise_timer.build_args_ns)."""
    return SimpleNamespace(
        task_name="forecast",
        ckpt_path=args.ckpt_path,
        patch_len=args.timer_patch_len,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        output_len=args.output_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        factor=args.factor,
        activation=args.activation,
        output_attention=False,
        features=args.timer_features,
        root_path=".",
        data_path=".",
        data="ETTh1",
        embed=args.embed,
        freq="h",
        stride=args.timer_stride,
        subset_rand_ratio=1.0,
        use_ims=bool(args.use_ims),
        batch_size=1,
        num_workers=0,
        use_multi_gpu=False,
        use_gpu=not args.cpu,
        gpu=args.gpu,
        inverse=False,
        periodic_embedding_branch=int(args.periodic_embedding_branch),
        periodic_emb_bank_dim=int(args.periodic_emb_bank_dim),
        geometric_hpe=int(args.geometric_hpe),
    )


def process_one_window(
    args: argparse.Namespace,
    t_full: np.ndarray,
    y_full: np.ndarray,
    t0: int,
    compute_hsic: bool,
    hsic_path: str,
) -> None:
    """HSIC/MI for one window, map scores to patches, write figures + CSV (one pipeline step)."""
    T = int(args.seq_len)
    if t0 + T > len(y_full):
        raise ValueError(f"Window [{t0}, {t0+T}) exceeds series length {len(y_full)}")
    x_win = y_full[t0 : t0 + T]
    t_win = t_full[t0 : t0 + T]

    stride = args.patch_len if args.stride < 0 else args.stride

    patches = window_patches(x_win, args.patch_len, stride)
    n_p = len(patches)
    if n_p == 0:
        raise RuntimeError("No patches; check seq_len, patch_len, stride")

    k = np.array([patch_curvature_mean(p) for p in patches], dtype=np.float64)

    if compute_hsic:
        import torch
        from experiments.mi_hsic_layerwise_timer import compute_hsic_mi_matrix_timer_batch
        from models.Timer import Model as TimerModel

        cfg = build_timer_configs(args)
        device = torch.device(
            "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
        )
        model = TimerModel(cfg).float().to(device)
        try:
            int(args.target_col)
            tgt_name = "OT"
        except ValueError:
            tgt_name = str(args.target_col)
        batch_x, batch_y = load_ims_batch_timer(
            args.csv,
            t0,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.timer_features,
            tgt_name,
            args.embed,
        )
        pred_horizon = args.output_len if args.use_ims else args.pred_len
        hsic_mat, fwd_mode = compute_hsic_mi_matrix_timer_batch(
            model,
            device,
            batch_x,
            batch_y,
            use_ims=bool(args.use_ims),
            label_len=args.label_len,
            pred_horizon=pred_horizon,
        )
        n_layers = int(hsic_mat.shape[0])
        li = int(args.layer_row)
        if li < 0:
            li = n_layers + li
        if li < 0 or li >= n_layers:
            raise ValueError(f"layer_row={args.layer_row} invalid for n_layers={n_layers}")
        mi_scores = np.asarray(hsic_mat[li], dtype=np.float64).ravel()
        if mi_scores.size != n_p:
            raise ValueError(
                f"Computed HSIC length {mi_scores.size} != window patch count {n_p}: "
                "use the same seq_len/patch_len/stride as the Timer checkpoint."
            )
        mi_label = f"HSIC/MI computed ({fwd_mode}, layer {li}/{n_layers - 1})"
    else:
        assert hsic_path
        mi_scores, mi_label = load_mi_scores_from_npy(n_p, hsic_path, args.layer_row)

    # Expected N from formula (with padding on virtual length)
    n_expect = num_patches(T, args.patch_len, stride, args.padding)
    if n_expect != n_p:
        print(
            f"WARNING: patch count from unfold ({n_p}) != formula ({n_expect}); "
            f"using actual {n_p} patches.",
            flush=True,
        )

    mi_scores = np.asarray(mi_scores, dtype=np.float64).ravel()[:n_p]
    if mi_scores.size < n_p:
        tmp = np.full(n_p, np.nan)
        tmp[: mi_scores.size] = mi_scores
        mi_scores = tmp

    # Computed HSIC must cover every patch in this window (finite values).
    if compute_hsic or (hsic_path and os.path.isfile(hsic_path)):
        if mi_scores.size < n_p or not np.all(np.isfinite(mi_scores[:n_p])):
            print(
                "ERROR: MI/HSIC must be finite for every patch in this window (n_patches="
                f"{n_p}); got length {mi_scores.size}. Align seq_len/patch_len with Timer / "
                "mi_hsic_layerwise_timer.",
                file=sys.stderr,
            )
            sys.exit(2)

    high_mask = tukey_high_mask(mi_scores)
    if not np.any(high_mask) and np.any(mi_scores > 0):
        high_mask = mi_scores >= np.nanpercentile(mi_scores, 90)

    order = np.argsort(-np.nan_to_num(mi_scores, nan=-np.inf))
    top_idx = [int(i) for i in order[: max(1, args.top_k)] if np.isfinite(mi_scores[int(i)])]

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv))[0]

    # --- Figure 1: series + shaded high-MI patches
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_win, x_win, "k-", lw=0.8, label="series")
    for p in top_idx:
        a = t0 + p * stride
        b = a + args.patch_len
        ax.axvspan(a, b, color="C1", alpha=0.25, label="top-MI patch" if p == top_idx[0] else "")
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    ax.legend(by.values(), by.keys(), loc="upper right")
    ax.set_xlabel("time index (global)")
    ax.set_ylabel("value")
    ax.set_title(
        f"{base}: window [{t0}, {t0+T}), top-{args.top_k} MI patches shaded ({mi_label})"
    )
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, f"{base}_mi_shaded_t{t0}.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # --- Figure 2: scatter MI vs curvature (HSIC scores are continuous)
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = np.isfinite(mi_scores) & np.isfinite(k)
    ms = mi_scores[valid]
    kv = k[valid]
    sc = ax.scatter(ms, kv, c=ms, cmap="viridis", s=40, alpha=0.85)
    fig.colorbar(sc, ax=ax, label=mi_label)
    for p in top_idx:
        if valid[p]:
            ax.scatter([float(mi_scores[p])], [k[p]], s=120, facecolors="none", edgecolors="red", linewidths=2)
    ax.set_xlabel(mi_label)
    ax.set_ylabel("mean |Δ²x| (patch curvature)")
    rho, pval = spearmanr(mi_scores[valid], k[valid])
    ax.set_title(
        f"Spearman ρ = {rho:.3f} (p = {pval:.2e}), n = {int(valid.sum())}\ncolor = HSIC/MI (computed)"
    )
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, f"{base}_mi_vs_curvature_t{t0}.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # --- Figure 3: per-patch lines
    fig, ax1 = plt.subplots(figsize=(14, 4))
    px = np.arange(n_p)
    ax1.bar(px - 0.2, np.nan_to_num(mi_scores), width=0.4, label=mi_label, alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(px, k, "s-", color="C2", ms=4, label="curvature k")
    ax1.set_xlabel("patch index p")
    ax1.set_ylabel("MI / HSIC score")
    ax2.set_ylabel("k = mean|Δ²x|")
    ax1.set_title("Per-patch MI (bars) vs curvature (line)")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
    fig.tight_layout()
    p3 = os.path.join(args.out_dir, f"{base}_per_patch_bars_t{t0}.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # --- CSV summary
    csv_out = os.path.join(args.out_dir, f"{base}_patch_stats_t{t0}.csv")
    pd.DataFrame(
        {
            "patch_index": np.arange(n_p),
            "t_start_global": t0 + np.arange(n_p) * stride,
            "t_end_global": t0 + np.arange(n_p) * stride + args.patch_len,
            "mi_score": mi_scores,
            "curvature_k": k,
            "tukey_high_mi": high_mask.astype(int),
        }
    ).to_csv(csv_out, index=False)

    print("Wrote:", p1, p2, p3, csv_out, flush=True)
    print(f"Spearman(mi, k) = {rho:.4f}, p = {pval:.4e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="MI/HSIC vs patch curvature + plots")
    ap.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    ap.add_argument(
        "--target_col",
        type=str,
        default="OT",
        help="Column name (e.g. OT) or integer column index for univariate series",
    )
    ap.add_argument("--t_start", type=int, default=0, help="First window start index in full series")
    ap.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of windows: t_start + i*window_hop for i=0..num_samples-1 (capped by series length)",
    )
    ap.add_argument(
        "--window_hop",
        type=int,
        default=-1,
        help="Spacing between windows when num_samples>1; default: seq_len",
    )
    ap.add_argument("--seq_len", type=int, default=672)
    ap.add_argument("--patch_len", type=int, default=96)
    ap.add_argument("--stride", type=int, default=-1, help="Default: same as patch_len")
    ap.add_argument("--padding", type=int, default=0, help="Right pad length (ReplicationPad1d), usually 0")
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Timer checkpoint: compute HSIC/MI here (same code as mi_hsic_layerwise_timer). "
        "Requires ETT hourly CSV with ETTh test split and --features M with M>=4.",
    )
    ap.add_argument(
        "--hsic_npy",
        type=str,
        default="",
        help="Precomputed [L,P] HSIC from mi_hsic_layerwise_timer (skips --ckpt_path forward).",
    )
    ap.add_argument(
        "--layer_row",
        type=int,
        default=0,
        help="Layer index if hsic_npy is 2D (default 0 = first layer; -1 = last layer)",
    )
    ap.add_argument("--top_k", type=int, default=2, help="Shade top-K patches by MI in time plot (default 2)")
    ap.add_argument("--out_dir", type=str, default="./figures/mi_patch_curvature")
    # Timer / mi_hsic_layerwise_timer alignment (used when --ckpt_path is set)
    ap.add_argument("--label_len", type=int, default=576)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--output_len", type=int, default=96)
    ap.add_argument("--use_ims", type=int, default=1, help="1 = IMS batch_y layout (default 1)")
    ap.add_argument("--timer_patch_len", type=int, default=96, help="Must match checkpoint patch embedding")
    ap.add_argument("--timer_stride", type=int, default=1, help="Stride in backbone (often 1; patch stride is patch_len)")
    ap.add_argument("--timer_features", type=str, default="M", help="M = multivariate for HSIC n>=4")
    ap.add_argument("--embed", type=str, default="timeF")
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--e_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="Force CPU for Timer forward")
    ap.add_argument("--periodic_embedding_branch", type=int, default=0)
    ap.add_argument("--periodic_emb_bank_dim", type=int, default=0)
    ap.add_argument("--geometric_hpe", type=int, default=0, help="1 if checkpoint uses Geometric-HPE embedding")
    args = ap.parse_args()

    ckpt_path = (args.ckpt_path or "").strip()
    hsic_path = (args.hsic_npy or "").strip()
    compute_hsic = bool(ckpt_path)

    if not ckpt_path and not hsic_path:
        print(
            "ERROR: Per-patch MI/HSIC must be computed in this pipeline. Provide either:\n"
            "  --ckpt_path ...  (Timer forward + same HSIC as mi_hsic_layerwise_timer.py), or\n"
            "  --hsic_npy .../aggregate/hsic_mean.npy  (from experiments/mi_hsic_layerwise_timer.py).",
            file=sys.stderr,
        )
        sys.exit(2)
    if ckpt_path and not os.path.isfile(ckpt_path):
        print(f"ERROR: --ckpt_path is not a readable file: {ckpt_path}", file=sys.stderr)
        sys.exit(2)
    if hsic_path and not os.path.isfile(hsic_path):
        print(f"ERROR: --hsic_npy is not a readable file: {hsic_path}", file=sys.stderr)
        sys.exit(2)
    if compute_hsic and hsic_path:
        print(
            "NOTE: both --ckpt_path and --hsic_npy set; computing MI from --ckpt_path (ignoring npy).",
            flush=True,
        )

    try:
        target: str | int = int(args.target_col)
    except ValueError:
        target = args.target_col

    t_full, y_full = load_series_1d(args.csv, target)
    T = int(args.seq_len)
    t0_base = int(args.t_start)
    ns = max(1, int(args.num_samples))
    hop = int(args.window_hop) if int(args.window_hop) > 0 else T

    windows: list[int] = []
    if ns == 1:
        windows = [t0_base]
    else:
        for i in range(ns):
            t0 = t0_base + i * hop
            if t0 + T > len(y_full):
                break
            windows.append(t0)
    if not windows:
        raise ValueError(
            f"No valid windows (N={len(y_full)}, seq_len={T}, t_start={t0_base}, "
            f"num_samples={ns}, window_hop={hop})."
        )
    if ns > 1 and len(windows) < ns:
        print(
            f"num_samples={ns} capped to {len(windows)} (series length {len(y_full)}).",
            flush=True,
        )

    print(
        f"Pipeline: MI/HSIC → map to patches → figures/CSV for {len(windows)} window(s).",
        flush=True,
    )
    for wi, t0 in enumerate(windows):
        print(f">>> sample {wi + 1}/{len(windows)}  t_start={t0}", flush=True)
        process_one_window(args, t_full, y_full, t0, compute_hsic, hsic_path)


if __name__ == "__main__":
    main()
