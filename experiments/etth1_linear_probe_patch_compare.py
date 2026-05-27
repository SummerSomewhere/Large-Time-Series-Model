#!/usr/bin/env python3
"""
Timer Decoder Layer-wise Linear Probe + MI Patch Comparison.

Performs Ridge Regression probing to measure how well each decoder layer's
per-patch representations can predict semantic labels derived from STL
decomposition (trend, seasonal, residual, etc.). Then compares R² / MSE
between high-MI patches (identified via etth1_mi_hsic_peaks.py) and
low-MI patches, layer by layer.

Mode A — Offline features (recommended, fast):
    python experiments/etth1_linear_probe_patch_compare.py \
        --features_pt_path ./results/layerwise_probe_features/ETTh1_20000.pt \
        --labels_json ./results/layerwise_probe_features/ETTh1_20000_labels.json \
        --e_layers 8 --pred_len 96 --n_samples 0 \
        --out_dir ./results/layerwise_probe/

Mode B — Hook-based inference (slow, for initial run):
    python experiments/etth1_linear_probe_patch_compare.py \
        --ckpt_path ... --root_path ... --data_path ETTh1.csv \
        (all model/data params)

Outputs (all in --out_dir):
    figA_r2_per_label.png         — R² vs layer depth (one line per semantic label)
    figB_mse_per_label.png        — log(MSE) vs layer depth
    figC_r2_heatmap.png           — heatmap: layers × labels, colour = R²
    figD_high_low_mi_compare.png  — R² comparison: high-MI patches vs low-MI patches
    figE_patch_r2_heatmap.png     — per-patch R² for each layer (sampled)
    ridge_results.json             — full structured results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model as TimerModel
from data_provider.data_factory import data_provider
from utils.masking import TriangularCausalMask

MI_DECODER_LAYER_CAP = 8


# ── Model unwrapping ─────────────────────────────────────────────────────────

def _unwrap_timer(model):
    if hasattr(model, "module"):
        return model.module
    return model


# ── Forward helpers (identical to etth1_mi_hsic_peaks.py) ───────────────────

def forward_collect_layers(model, x_enc):
    core = _unwrap_timer(model)
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev

    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
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
        layers_out.append(pool(h.detach()))
    return layers_out, int(n_vars), int(N)


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


# ── Ridge linear probe ───────────────────────────────────────────────────────

def ridge_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """Fit Ridge regression and return test R² and MSE."""
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    mse = float(np.mean((y_pred - y_test) ** 2))
    r2 = float(r2_score(y_test, y_pred))
    return r2, mse


def probe_all(
    features_dict: dict[int, torch.Tensor],
    labels: dict[str, np.ndarray],
    label_keys: list[str],
    test_ratio: float,
    alpha: float,
    max_samples: int,
    seed: int,
) -> dict:
    """
    Run Ridge probe for each (layer, label) pair.

    features_dict: {layer_idx -> [S, N, D] Tensor}
    labels: {label_name -> [S] np.ndarray}
    Returns dict with per-layer per-label R² and MSE.
    """
    rng = np.random.RandomState(seed)

    results = {
        "layers": [],
        "labels": label_keys,
        "r2": {},   # label_name -> [n_layers]
        "mse": {},  # label_name -> [n_layers]
    }

    n_layers = len(features_dict)
    layer_indices = sorted(features_dict.keys())

    for li, layer_idx in enumerate(layer_indices):
        feat: torch.Tensor = features_dict[layer_idx]   # [S, N, D]
        S, N, D = feat.shape

        # Subsample if needed
        if max_samples > 0 and S > max_samples:
            idx = rng.permutation(S)[:max_samples]
            feat = feat[idx]
            labels_sliced = {k: v[idx] for k, v in labels.items()}
        else:
            labels_sliced = labels

        S_eff = feat.shape[0]
        n_train = int(S_eff * (1 - test_ratio))
        perm = rng.permutation(S_eff)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        # Flatten [S, N, D] -> [S, N*D]
        X_all = feat.view(S_eff, -1).numpy()  # [S, N*D]
        X_train, X_test = X_all[train_idx], X_all[test_idx]

        layer_r2 = {}
        layer_mse = {}
        for lbl_name in label_keys:
            y = labels_sliced[lbl_name]
            if y.shape[0] != S_eff:
                continue
            y_train, y_test = y[train_idx], y[test_idx]
            r2, mse = ridge_probe(X_train, y_train, X_test, y_test, alpha=alpha)
            layer_r2[lbl_name] = r2
            layer_mse[lbl_name] = mse

        results["layers"].append(layer_idx)
        for lbl_name, r2 in layer_r2.items():
            if lbl_name not in results["r2"]:
                results["r2"][lbl_name] = []
                results["mse"][lbl_name] = []
            results["r2"][lbl_name].append(r2)
            results["mse"][lbl_name].append(layer_mse[lbl_name])

        print(f"  Layer {layer_idx}: R² = {layer_r2}")

    return results


def probe_patch_wise(
    features_dict: dict[int, torch.Tensor],
    labels: dict[str, np.ndarray],
    label_keys: list[str],
    test_ratio: float,
    alpha: float,
    seed: int,
) -> dict:
    """
    Per-patch Ridge probe: for each layer, fit Ridge for each (patch, label) pair.
    Returns {layer_idx -> {label_name -> np.ndarray [N] of R² scores}}.
    """
    rng = np.random.RandomState(seed)
    layer_patch_r2 = {}

    for layer_idx, feat in features_dict.items():
        S, N, D = feat.shape
        n_train = int(S * (1 - test_ratio))
        perm = rng.permutation(S)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        feat_np = feat.numpy()   # [S, N, D]
        patch_r2: dict[str, np.ndarray] = {}

        for lbl_name in label_keys:
            y = labels[lbl_name]
            if y.shape[0] != S:
                continue
            r2_per_patch = np.full(N, np.nan)
            for p in range(N):
                X_train = feat_np[train_idx, p, :]   # [n_train, D]
                X_test = feat_np[test_idx, p, :]      # [n_test, D]
                y_train, y_test = y[train_idx], y[test_idx]
                r2_per_patch[p] = ridge_probe(X_train, y_train, X_test, y_test, alpha=alpha)[0]
            patch_r2[lbl_name] = r2_per_patch

        layer_patch_r2[layer_idx] = patch_r2
        print(f"  Layer {layer_idx}: patch-wise R² done ({N} patches)")

    return layer_patch_r2


# ── High-MI vs Low-MI comparison ──────────────────────────────────────────────

def probe_high_low_mi(
    features_dict: dict[int, torch.Tensor],
    labels: dict[str, np.ndarray],
    label_keys: list[str],
    mi_peaks_path: str,
    test_ratio: float,
    alpha: float,
    seed: int,
) -> dict:
    """
    Split patches into high-MI and low-MI groups per layer using MI peaks JSON,
    then run Ridge probe for each group separately.
    """
    if not os.path.exists(mi_peaks_path):
        print(f"  [WARN] MI peaks file not found: {mi_peaks_path}")
        return {}

    with open(mi_peaks_path) as f:
        mi_data = json.load(f)

    high_patches_per_layer = {}
    for layer_str, layer_info in mi_data.get("layers", {}).items():
        li = int(layer_str)
        peaks = layer_info.get("high_mi_patches", [])
        high_patches_per_layer[li] = set(peaks)

    rng = np.random.RandomState(seed)
    results = {
        "layers": [],
        "labels": label_keys,
        "high_r2": {},    # label -> [n_layers]
        "low_r2": {},     # label -> [n_layers]
    }

    for li, feat in features_dict.items():
        S, N, D = feat.shape
        high_set = high_patches_per_layer.get(li, set())
        high_patches = sorted(high_set)
        low_patches = [p for p in range(N) if p not in high_set]

        if not high_patches or not low_patches:
            print(f"  Layer {li}: skipping MI comparison (high={len(high_patches)}, low={len(low_patches)})")
            continue

        n_train = int(S * (1 - test_ratio))
        perm = rng.permutation(S)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        feat_np = feat.numpy()   # [S, N, D]

        def probe_patch_group(patch_list, group_name):
            X_all = feat_np[:, patch_list, :].reshape(S, len(patch_list) * D)
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            r2_dict = {}
            for lbl_name in label_keys:
                y = labels[lbl_name]
                if y.shape[0] != S:
                    continue
                y_train, y_test = y[train_idx], y[test_idx]
                r2_dict[lbl_name] = ridge_probe(X_train, y_train, X_test, y_test, alpha=alpha)[0]
            return r2_dict

        high_r2 = probe_patch_group(high_patches, "high")
        low_r2 = probe_patch_group(low_patches, "low")

        results["layers"].append(li)
        for lbl_name in label_keys:
            if lbl_name not in results["high_r2"]:
                results["high_r2"][lbl_name] = []
                results["low_r2"][lbl_name] = []
            results["high_r2"][lbl_name].append(high_r2.get(lbl_name, np.nan))
            results["low_r2"][lbl_name].append(low_r2.get(lbl_name, np.nan))

        print(f"  Layer {li}: high-MI R²={high_r2}, low-MI R²={low_r2}")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

LABEL_DISPLAY_NAMES = {
    "trend_mean": "Trend (mean level)",
    "seasonal_amplitude": "Seasonal amplitude",
    "residual_std": "Residual std",
    "trend_slope": "Trend slope",
    "residual_energy": "Residual energy",
    "var0_mean": "Var0 mean",
    "var0_std": "Var0 std",
    "var1_mean": "Var1 mean",
    "var1_std": "Var1 std",
}


def _display_name(name: str) -> str:
    return LABEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def plot_r2_per_label(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = results["layers"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results["labels"])))
    for i, lbl in enumerate(results["labels"]):
        r2 = results["r2"].get(lbl, [])
        if len(r2) == len(layers):
            ax.plot(layers, r2, "o-", label=_display_name(lbl), color=colors[i], linewidth=1.5, markersize=5)
    ax.set_xlabel("Decoder Layer", fontsize=11)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title("R² per Semantic Label vs. Layer Depth", fontsize=13)
    ax.set_xticks(layers)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_mse_per_label(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = results["layers"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results["labels"])))
    for i, lbl in enumerate(results["labels"]):
        mse = results["mse"].get(lbl, [])
        if len(mse) == len(layers):
            ax.plot(layers, mse, "s--", label=_display_name(lbl), color=colors[i], linewidth=1.2, markersize=4)
    ax.set_xlabel("Decoder Layer", fontsize=11)
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.set_title("log(MSE) per Semantic Label vs. Layer Depth", fontsize=13)
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_r2_heatmap(results: dict, out_path: str):
    labels_list = results["labels"]
    layers = results["layers"]
    n_layers = len(layers)
    n_labels = len(labels_list)
    matrix = np.zeros((n_layers, n_labels))
    for j, lbl in enumerate(labels_list):
        r2_vals = results["r2"].get(lbl, [])
        if len(r2_vals) == n_layers:
            matrix[:, j] = r2_vals
        else:
            matrix[:, j] = np.nan

    fig, ax = plt.subplots(figsize=(max(6, n_labels * 0.8), max(4, n_layers * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels([_display_name(l) for l in labels_list], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=9)
    ax.set_title("R² Heatmap: Layers × Semantic Labels", fontsize=12)
    plt.colorbar(im, ax=ax, label="R²")
    for i in range(n_layers):
        for j in range(n_labels):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.5 else "black")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_high_low_mi_compare(mi_results: dict, out_path: str):
    if not mi_results or "layers" not in mi_results or not mi_results["layers"]:
        print(f"  [SKIP] No MI comparison data for {out_path}")
        return
    layers = mi_results["layers"]
    labels_list = mi_results["labels"]
    n_layers = len(layers)

    n_labels = len(labels_list)
    n_cols = min(3, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 2))

    for idx, lbl in enumerate(labels_list):
        ax = axes[idx // n_cols][idx % n_cols]
        high_r2 = mi_results["high_r2"].get(lbl, [])
        low_r2 = mi_results["low_r2"].get(lbl, [])
        if len(high_r2) == n_layers and len(low_r2) == n_layers:
            x = np.arange(n_layers)
            ax.plot(x, high_r2, "o-", label="High-MI patches", color=colors[0], linewidth=1.5)
            ax.plot(x, low_r2, "s--", label="Low-MI patches", color=colors[1], linewidth=1.5)
            ax.fill_between(x, high_r2, low_r2, alpha=0.1, color="gray")
        ax.set_xlabel("Layer")
        ax.set_ylabel("R²")
        ax.set_title(_display_name(lbl), fontsize=10)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0)

    for idx in range(n_labels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("R²: High-MI Patches vs. Low-MI Patches", fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_patch_r2_heatmap(
    patch_r2: dict[int, dict[str, np.ndarray]],
    layers: list[int],
    label_keys: list[str],
    out_path: str,
    max_layers_show: int = 4,
):
    """
    Show per-patch R² for the first `max_layers_show` layers as a heatmap per label.
    """
    if not patch_r2:
        print(f"  [SKIP] No patch-wise R² data for {out_path}")
        return

    # Pick a representative label
    primary_label = label_keys[0] if label_keys else None
    if primary_label is None:
        return

    layers_to_show = layers[:max_layers_show]
    n_layers = len(layers_to_show)

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), squeeze=False)
    vmax = 0.0
    for li in layers_to_show:
        r2_vals = patch_r2.get(li, {}).get(primary_label)
        if r2_vals is not None:
            vmax = max(vmax, np.nanmax(r2_vals))

    for col, li in enumerate(layers_to_show):
        ax = axes[0][col]
        r2_vals = patch_r2.get(li, {}).get(primary_label)
        if r2_vals is not None:
            N = len(r2_vals)
            matrix = r2_vals.reshape(1, N)
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(vmax, 0.01))
            ax.set_yticks([0])
            ax.set_yticklabels([f"Layer {li}"])
            ax.set_xlabel("Patch index")
            ax.set_title(f"Layer {li}", fontsize=11)
            plt.colorbar(im, ax=ax, label="R²", orientation="vertical", shrink=0.8)
            for p in range(N):
                val = r2_vals[p]
                if not np.isnan(val):
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(p, 0, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)
        else:
            ax.axis("off")

    fig.suptitle(f"Per-Patch R² ({_display_name(primary_label)})", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Timer Linear Probe + MI Patch Comparison")
    # Feature source
    parser.add_argument("--features_pt_path", type=str, default="",
                        help="Path to pre-extracted .pt features (Mode A)")
    parser.add_argument("--labels_json", type=str, default="",
                        help="Path to labels JSON (Mode A)")
    # Model / data (Mode B — hook inference)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
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
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--subset_rand_ratio", type=float, default=1.0)
    parser.add_argument("--use_ims", action="store_true")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Probing
    parser.add_argument("--n_samples", type=int, default=0,
                        help="0 = all available samples")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    # MI peaks (optional — enables high/low MI comparison)
    parser.add_argument("--mi_peaks_path", type=str, default="./global_mi_peaks_etth1.json")
    # Output
    parser.add_argument("--out_dir", type=str, default="./results/layerwise_probe/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs for inference (Mode B)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("Timer Linear Probe + MI Patch Comparison")
    print("=" * 60)
    print(f"  features_pt_path : {args.features_pt_path or '(none — Mode B)'}")
    print(f"  n_samples        : {args.n_samples or 'all'}")
    print(f"  test_ratio       : {args.test_ratio}")
    print(f"  alpha (Ridge)    : {args.alpha}")
    print(f"  out_dir          : {args.out_dir}")
    print("=" * 60)

    # ── 1. Load features ─────────────────────────────────────────────────────
    t0 = time.time()
    if args.features_pt_path and os.path.exists(args.features_pt_path):
        print("\n[1] Loading pre-extracted features...")
        ckpt = torch.load(args.features_pt_path, map_location="cpu")
        features: dict[int, torch.Tensor] = ckpt["features"]
        d_model = ckpt.get("d_model", 1024)
        n_layers_loaded = ckpt.get("n_layers", len(features))
        N = ckpt.get("N", 0)
        n_samples_loaded = ckpt.get("n_samples", 0)
        print(f"  Loaded: {n_layers_loaded} layers, {n_samples_loaded} samples, "
              f"{N} patches, d_model={d_model}")
    elif args.ckpt_path:
        print("\n[1] Extracting features via hook (Mode B)...")
        ns = build_namespace(args)
        model = TimerModel(ns)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model.to(device)
        model.eval()

        ns.batch_size = args.batch_size
        _, loader = data_provider(ns, "test")
        per_layer: list[list[torch.Tensor]] = [[] for _ in range(MI_DECODER_LAYER_CAP)]
        n_extracted = 0
        max_samples = args.n_samples if args.n_samples > 0 else 10**9

        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if n_extracted >= max_samples:
                break
            B = batch_x.shape[0]
            keep = min(B, max_samples - n_extracted)
            x_batch = batch_x[:keep].float().to(device)
            with torch.no_grad():
                layers_out, _, N = forward_collect_layers(model, x_batch)
            for li in range(len(layers_out)):
                per_layer[li].append(layers_out[li][:keep].cpu())
            n_extracted += keep

        features = {li: torch.cat(per_layer[li], dim=0) for li in range(MI_DECODER_LAYER_CAP) if per_layer[li]}
        d_model = next(iter(features.values())).shape[2]
        print(f"  Extracted: {len(features)} layers, {n_extracted} samples, {N} patches")
    else:
        raise RuntimeError("Must provide --features_pt_path or --ckpt_path")
    print(f"  Feature loading: {time.time()-t0:.1f}s")

    # ── 2. Load labels ──────────────────────────────────────────────────────
    t1 = time.time()
    if args.labels_json and os.path.exists(args.labels_json):
        print("\n[2] Loading semantic labels from JSON...")
        with open(args.labels_json) as f:
            labels_raw = json.load(f)
        labels: dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64) for k, v in labels_raw.items()}
    else:
        print("\n[2] Generating synthetic STL labels from raw data...")
        import pandas as pd
        from scipy.signal import detrend
        from scipy.stats import linregress

        csv_path = os.path.join(args.root_path, args.data_path)
        df = pd.read_csv(csv_path)
        raw = df.values.astype(np.float64)
        n_total = next(iter(features.values())).shape[0]
        raw_subset = raw[-n_total:] if raw.shape[0] > n_total else raw
        if raw_subset.shape[0] < n_total:
            raw_subset = np.pad(raw_subset, ((0, n_total - raw_subset.shape[0]), (0, 0)), mode="edge")

        T, n_vars = raw_subset.shape
        combined = raw_subset.mean(axis=1)
        period = 24
        window = max(period if period % 2 == 1 else period + 1, 3)
        kernel = np.ones(window) / window
        trend_raw = np.convolve(combined, kernel, mode="same")
        half = window // 2
        trend_raw[:half] = np.nan
        trend_raw[-half:] = np.nan
        valid = ~np.isnan(trend_raw)
        if valid.sum() > 2:
            xc = np.arange(T)[valid]
            trend_interp = np.empty(T)
            trend_interp[valid] = trend_raw[valid]
            trend_interp[~valid] = np.interp(np.where(~valid)[0], xc, trend_raw[valid])
        else:
            trend_interp = np.full(T, np.nanmean(trend_raw))

        seasonal = np.zeros(T)
        n_full = T // period
        if n_full > 0:
            detrended = combined - trend_interp
            reshaped = detrended[:n_full * period].reshape(n_full, period)
            pm = reshaped.mean(axis=0)
            pm -= pm.mean()
            for i in range(n_full):
                seasonal[i * period:(i + 1) * period] = pm
            seasonal[n_full * period:] = pm[:T - n_full * period]

        residual = combined - trend_interp - seasonal
        labels = {
            "trend_mean": trend_interp[:n_total],
            "seasonal_amplitude": np.abs(seasonal[:n_total]),
            "residual_std": np.abs(residual[:n_total]),
        }
        win = max(period, 24)
        n_w = max(T // win, 1)
        slopes = np.full(n_w, 0.0)
        energies = np.full(n_w, 0.0)
        for w in range(n_w):
            s, e = w * win, min((w + 1) * win, T)
            if e - s < 3:
                continue
            _, _, r, _, _ = linregress(np.arange(e - s), trend_interp[s:e])
            slopes[w] = r
            energies[w] = float(np.mean(residual[s:e] ** 2))
        centers = np.clip(np.arange(n_w) * win + win // 2, 0, n_total - 1)
        labels["trend_slope"] = np.interp(np.arange(n_total), centers, slopes)
        labels["residual_energy"] = np.interp(np.arange(n_total), centers, energies)

        for v in range(min(n_vars, 2)):
            vs = raw_subset[-n_total:, v]
            labels[f"var{v}_mean"] = vs
            labels[f"var{v}_std"] = (vs - vs.mean()) ** 2

    n_feat_samples = next(iter(features.values())).shape[0]
    for k, v in labels.items():
        labels[k] = v[:n_feat_samples]
    label_keys = list(labels.keys())
    print(f"  Labels: {label_keys}")
    for k, v in labels.items():
        print(f"    {k}: {v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
    print(f"  Label loading: {time.time()-t1:.1f}s")

    # ── 3. Global Ridge probe ────────────────────────────────────────────────
    t2 = time.time()
    print("\n[3] Running global Ridge probe (all patches, all labels)...")
    max_s = args.n_samples if args.n_samples > 0 else n_feat_samples
    global_results = probe_all(features, labels, label_keys, args.test_ratio, args.alpha, max_s, args.seed)
    print(f"  Global probe: {time.time()-t2:.1f}s")

    # ── 4. Patch-wise Ridge probe ─────────────────────────────────────────────
    t3 = time.time()
    print("\n[4] Running patch-wise Ridge probe...")
    patch_r2 = probe_patch_wise(features, labels, label_keys, args.test_ratio, args.alpha, args.seed)
    print(f"  Patch-wise probe: {time.time()-t3:.1f}s")

    # ── 5. High-MI vs Low-MI comparison ───────────────────────────────────────
    t4 = time.time()
    print("\n[5] Running high-MI vs low-MI comparison...")
    mi_results = probe_high_low_mi(
        features, labels, label_keys,
        args.mi_peaks_path, args.test_ratio, args.alpha, args.seed
    )
    print(f"  MI comparison: {time.time()-t4:.1f}s")

    # ── 6. Save results ──────────────────────────────────────────────────────
    print("\n[6] Saving results...")
    results_out = {
        "global": global_results,
        "patch_r2": {str(li): {k: v.tolist() if hasattr(v, "tolist") else list(v)
                               for k, v in patch_dict.items()}
                     for li, patch_dict in patch_r2.items()},
        "mi_compare": mi_results,
        "config": vars(args),
    }
    results_path = os.path.join(args.out_dir, "ridge_results.json")
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    # ── 7. Plots ────────────────────────────────────────────────────────────
    print("\n[7] Generating plots...")
    plot_r2_per_label(global_results, os.path.join(args.out_dir, "figA_r2_per_label.png"))
    plot_mse_per_label(global_results, os.path.join(args.out_dir, "figB_mse_per_label.png"))
    plot_r2_heatmap(global_results, os.path.join(args.out_dir, "figC_r2_heatmap.png"))
    plot_high_low_mi_compare(mi_results, os.path.join(args.out_dir, "figD_high_low_mi_compare.png"))
    plot_patch_r2_heatmap(
        patch_r2, global_results["layers"], label_keys,
        os.path.join(args.out_dir, "figE_patch_r2_heatmap.png")
    )

    print(f"\n[Done] Total time: {time.time()-t0:.1f}s")
    print(f"  All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
