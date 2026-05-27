#!/usr/bin/env python3
"""
Per-token L2 norm analysis across Timer decoder layers, grouped by high/low MI patches.

Reads the pre-computed MI peaks file (global_mi_peaks_etth1.json) to identify
which patch indices are high-MI vs low-MI for each decoder layer. Then runs Timer
forward passes on a sample batch, computes per-token (per-patch) L2 norms for
each layer, and plots their distributions side-by-side.

Hypothesis: high-MI tokens may have smaller L2 norms (information-dense),
or larger norms (more informative variance), depending on the model's behaviour.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sp_stats

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model as TimerModel
from data_provider.data_factory import data_provider


# ── Model unwrapping (handles DDP) ─────────────────────────────────────────

def _unwrap_timer(model):
    """Strip DDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module
    return model


# ── L2 norm computation per layer ───────────────────────────────────────────

def compute_per_layer_norms(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_x_mark: torch.Tensor,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """
    Run a forward pass and collect per-token L2 norms for each decoder layer.

    Returns:
        dict[layer_idx -> np.ndarray of shape [B, N]]
    """
    model.eval()
    x = batch_x.to(device)
    x_mark = batch_x_mark.to(device)
    # enc_embedding expects [B, M, T]; data loader returns [B, T, M]
    x2 = x.permute(0, 2, 1).float()

    with torch.no_grad():
        dec_in, n_vars = _unwrap_timer(model).enc_embedding(x2)  # [B*M, N, D]
        BM, N, D = dec_in.shape
        B = x.shape[0]

    # Build causal mask (no padding in enc path)
    from utils.masking import TriangularCausalMask
    mask = TriangularCausalMask(BM, N, device=device)

    # Mean-pool over variates to get [B, N, D]
    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    norms_per_layer = {}
    core = _unwrap_timer(model)
    h = dec_in

    for i, layer_module in enumerate(core.decoder.attn_layers):
        h, _, _ = layer_module(h, attn_mask=mask)
        # h: [B*M, N, D]  -> pool -> [B, N, D]
        pooled = pool(h)
        # L2 norm over feature dim for each token
        l2 = pooled.detach().norm(dim=-1).cpu().numpy()  # [B, N]
        norms_per_layer[i] = l2

    return norms_per_layer


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_norm_distributions(
    norms_per_layer: dict[int, np.ndarray],
    mi_info: dict,
    n_layers: int,
    output_path: str,
    n_bins: int = 40,
):
    """
    For each layer, plot overlaid histograms (or KDE) of L2 norms for
    high-MI vs low-MI tokens.
    """
    num_layers = n_layers
    fig, axes = plt.subplots(
        2, num_layers // 2, figsize=(4 * num_layers, 8), squeeze=False
    )
    axes = axes.flatten()

    all_high_l2 = []
    all_low_l2 = []

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_key = str(layer_idx)
        layer_data = mi_info["layers"].get(layer_key, {})

        high_mi_patches = set(layer_data.get("high_mi_patches", []))
        low_mi_patches = set(layer_data.get("low_mi_patches", []))
        all_patch_indices = set(range(mi_info["N"]))

        # Fallback: if no explicit high/low, use q3 threshold
        if not high_mi_patches and "hsic_curve" in layer_data:
            hsic_curve = layer_data["hsic_curve"]
            q3 = layer_data.get("q3_threshold", np.percentile(hsic_curve, 75))
            high_mi_patches = {i for i, v in enumerate(hsic_curve) if v >= q3}
            sorted_idx = np.argsort(hsic_curve)
            low_mi_patches = {int(sorted_idx[0]), int(sorted_idx[1])}

        norms = norms_per_layer.get(layer_idx)
        if norms is None:
            ax.set_title(f"Layer {layer_idx}\n(no data)")
            continue

        # norms shape: [B, N]; flatten
        norms_flat = norms.flatten()  # [B*N]
        patch_indices = np.tile(np.arange(norms.shape[1]), norms.shape[0])  # [B*N]

        # Build per-token labels
        high_mask = np.isin(patch_indices, list(high_mi_patches))
        low_mask = np.isin(patch_indices, list(low_mi_patches))

        high_l2 = norms_flat[high_mask]
        low_l2 = norms_flat[low_mask]

        all_high_l2.append(high_l2)
        all_low_l2.append(low_l2)

        # Plot
        if len(high_l2) > 1:
            ax.hist(
                high_l2, bins=n_bins, alpha=0.6, label="High MI",
                color="crimson", density=True, edgecolor="none"
            )
        if len(low_l2) > 1:
            ax.hist(
                low_l2, bins=n_bins, alpha=0.6, label="Low MI",
                color="steelblue", density=True, edgecolor="none"
            )

        # Summary stats
        stats_text = ""
        if len(high_l2) > 0:
            stats_text += f"High: μ={np.mean(high_l2):.3f} σ={np.std(high_l2):.3f}\n"
        if len(low_l2) > 0:
            stats_text += f"Low:  μ={np.mean(low_l2):.3f} σ={np.std(low_l2):.3f}"
        ax.text(0.97, 0.97, stats_text.strip(),
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("L2 Norm", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.suptitle(
        "Per-token L2 Norm Distribution: High-MI vs Low-MI Patches",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def plot_norm_scatter(
    norms_per_layer: dict[int, np.ndarray],
    mi_info: dict,
    n_layers: int,
    output_path: str,
):
    """
    Scatter plot: each point is one token; x=patch index, y=L2 norm.
    Points colored red/blue by high/low MI.
    One subplot per layer.
    """
    num_layers = n_layers
    fig, axes = plt.subplots(
        2, num_layers // 2, figsize=(4 * num_layers, 8), squeeze=False
    )
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_key = str(layer_idx)
        layer_data = mi_info["layers"].get(layer_key, {})

        high_mi_patches = set(layer_data.get("high_mi_patches", []))
        low_mi_patches = set(layer_data.get("low_mi_patches", []))
        all_patch_indices = set(range(mi_info["N"]))

        if not high_mi_patches and "hsic_curve" in layer_data:
            hsic_curve = layer_data["hsic_curve"]
            q3 = layer_data.get("q3_threshold", np.percentile(hsic_curve, 75))
            high_mi_patches = {i for i, v in enumerate(hsic_curve) if v >= q3}
            sorted_idx = np.argsort(hsic_curve)
            low_mi_patches = {int(sorted_idx[0]), int(sorted_idx[1])}

        norms = norms_per_layer.get(layer_idx)
        if norms is None:
            ax.set_title(f"Layer {layer_idx}\n(no data)")
            continue

        B, N = norms.shape
        patch_ids = np.tile(np.arange(N), B)          # [B*N]
        norms_flat = norms.flatten()

        high_mask = np.isin(patch_ids, list(high_mi_patches))
        low_mask = np.isin(patch_ids, list(low_mi_patches))

        # Jitter patch ids slightly for readability
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, patch_ids.shape)

        ax.scatter(
            patch_ids[low_mask] + jitter[low_mask],
            norms_flat[low_mask],
            alpha=0.4, s=8, c="steelblue", label="Low MI"
        )
        ax.scatter(
            patch_ids[high_mask] + jitter[high_mask],
            norms_flat[high_mask],
            alpha=0.7, s=12, c="crimson", label="High MI"
        )

        ax.set_xlabel("Patch Index (token position)", fontsize=9)
        ax.set_ylabel("L2 Norm", fontsize=9)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(range(N))

    plt.suptitle(
        "Per-token L2 Norm vs Patch Index: High-MI vs Low-MI",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def plot_mean_norm_per_layer(
    norms_per_layer: dict[int, np.ndarray],
    mi_info: dict,
    n_layers: int,
    output_path: str,
):
    """
    Line plot: mean L2 norm per layer, separately for high-MI and low-MI tokens.
    """
    num_layers = n_layers
    layers = np.arange(num_layers)
    mean_high = []
    mean_low = []
    std_high = []
    std_low = []

    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        layer_data = mi_info["layers"].get(layer_key, {})
        high_mi_patches = set(layer_data.get("high_mi_patches", []))
        low_mi_patches = set(layer_data.get("low_mi_patches", []))
        all_patch_indices = set(range(mi_info["N"]))

        if not high_mi_patches and "hsic_curve" in layer_data:
            hsic_curve = layer_data["hsic_curve"]
            q3 = layer_data.get("q3_threshold", np.percentile(hsic_curve, 75))
            high_mi_patches = {i for i, v in enumerate(hsic_curve) if v >= q3}
            sorted_idx = np.argsort(hsic_curve)
            low_mi_patches = {int(sorted_idx[0]), int(sorted_idx[1])}

        norms = norms_per_layer.get(layer_idx)
        if norms is None:
            mean_high.append(np.nan)
            mean_low.append(np.nan)
            std_high.append(np.nan)
            std_low.append(np.nan)
            continue

        B, N = norms.shape
        patch_ids = np.tile(np.arange(N), B)
        high_mask = np.isin(patch_ids, list(high_mi_patches))
        low_mask = np.isin(patch_ids, list(low_mi_patches))

        high_vals = norms.flatten()[high_mask]
        low_vals = norms.flatten()[low_mask]

        mean_high.append(np.mean(high_vals) if len(high_vals) > 0 else np.nan)
        mean_low.append(np.mean(low_vals) if len(low_vals) > 0 else np.nan)
        std_high.append(np.std(high_vals) if len(high_vals) > 0 else 0)
        std_low.append(np.std(low_vals) if len(low_vals) > 0 else 0)

    mean_high = np.array(mean_high)
    mean_low = np.array(mean_low)
    std_high = np.array(std_high)
    std_low = np.array(std_low)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layers, mean_high, yerr=std_high, fmt="o-", color="crimson",
                linewidth=2, markersize=6, label="High MI", capsize=4)
    ax.errorbar(layers + 0.15, mean_low, yerr=std_low, fmt="s-", color="steelblue",
                linewidth=2, markersize=6, label="Low MI", capsize=4)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean L2 Norm", fontsize=11)
    ax.set_title("Mean Per-token L2 Norm per Layer: High-MI vs Low-MI", fontsize=12)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def plot_hsic_curve_with_norm_heatmap(
    norms_per_layer: dict[int, np.ndarray],
    mi_info: dict,
    n_layers: int,
    output_path: str,
):
    """
    Top row: HSIC (MI) curve per layer (from mi_info).
    Bottom row: mean L2 norm per patch index per layer.
    """
    num_layers = n_layers
    fig, axes = plt.subplots(2, num_layers, figsize=(4 * num_layers, 6), squeeze=False)

    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        layer_data = mi_info["layers"].get(layer_key, {})

        # Top: HSIC curve
        ax_top = axes[0, layer_idx]
        hsic_curve = layer_data.get("hsic_curve", [])
        if hsic_curve:
            ax_top.plot(hsic_curve, "o-", color="purple", linewidth=1.5, markersize=4)
            q3 = layer_data.get("q3_threshold", 0)
            ax_top.axhline(q3, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"Q3={q3:.3f}")
            ax_top.legend(fontsize=7)
        ax_top.set_title(f"Layer {layer_idx}\nHSIC", fontsize=10)
        ax_top.grid(True, alpha=0.3)

        # Bottom: mean L2 norm per patch index
        ax_bot = axes[1, layer_idx]
        norms = norms_per_layer.get(layer_idx)
        if norms is not None:
            mean_per_patch = np.mean(norms, axis=0)  # [N]
            ax_bot.bar(range(len(mean_per_patch)), mean_per_patch,
                       color="steelblue", alpha=0.7, edgecolor="navy")
            ax_bot.set_xticks(range(len(mean_per_patch)))
        ax_bot.set_title(f"Layer {layer_idx}\nMean L2 Norm", fontsize=10)
        ax_bot.set_xlabel("Patch Index", fontsize=8)
        ax_bot.grid(True, alpha=0.3, axis="y")

    plt.suptitle("HSIC (MI) Curve vs Mean L2 Norm per Patch", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


# ── Statistical test ─────────────────────────────────────────────────────────

def run_statistical_tests(
    norms_per_layer: dict[int, np.ndarray],
    mi_info: dict,
    n_layers: int,
):
    """Run Mann-Whitney U test between high-MI and low-MI L2 norms per layer."""
    print("\nStatistical Tests (Mann-Whitney U): High-MI vs Low-MI L2 Norms")
    print("-" * 65)
    print(f"{'Layer':>6} | {'n_high':>7} | {'n_low':>7} | "
          f"{'μ_high':>8} | {'μ_low':>8} | {'U-stat':>10} | {'p-value':>10} | {'Significant':>11}")
    print("-" * 65)

    for layer_idx in range(n_layers):
        layer_key = str(layer_idx)
        layer_data = mi_info["layers"].get(layer_key, {})
        high_mi_patches = set(layer_data.get("high_mi_patches", []))
        low_mi_patches = set(layer_data.get("low_mi_patches", []))
        all_patch_indices = set(range(mi_info["N"]))

        if not high_mi_patches and "hsic_curve" in layer_data:
            hsic_curve = layer_data["hsic_curve"]
            q3 = layer_data.get("q3_threshold", np.percentile(hsic_curve, 75))
            high_mi_patches = {i for i, v in enumerate(hsic_curve) if v >= q3}
            sorted_idx = np.argsort(hsic_curve)
            low_mi_patches = {int(sorted_idx[0]), int(sorted_idx[1])}

        norms = norms_per_layer.get(layer_idx)
        if norms is None:
            print(f"{layer_idx:>6} | {'N/A':>7} | {'N/A':>7} | "
                  f"{'N/A':>8} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>11}")
            continue

        B, N = norms.shape
        patch_ids = np.tile(np.arange(N), B)
        high_mask = np.isin(patch_ids, list(high_mi_patches))
        low_mask = np.isin(patch_ids, list(low_mi_patches))

        high_l2 = norms.flatten()[high_mask]
        low_l2 = norms.flatten()[low_mask]

        if len(high_l2) < 2 or len(low_l2) < 2:
            print(f"{layer_idx:>6} | {len(high_l2):>7} | {len(low_l2):>7} | "
                  f"{'N/A':>8} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>11}")
            continue

        stat, pval = sp_stats.mannwhitneyu(high_l2, low_l2, alternative="two-sided")
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"{layer_idx:>6} | {len(high_l2):>7} | {len(low_l2):>7} | "
              f"{np.mean(high_l2):>8.4f} | {np.mean(low_l2):>8.4f} | "
              f"{stat:>10.1f} | {pval:>10.2e} | {sig:>11}")
    print("-" * 65)


# ── Main ─────────────────────────────────────────────────────────────────────

def build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Minimal args namespace for Timer Model + data_provider."""
    ns = argparse.Namespace(**vars(args))
    for k, v in {
        "task_name": "forecast",
        "is_training": 0,
        "is_finetuning": 0,
        "train_test": 0,
        "use_multi_gpu": False,
        "d_layers": 1,
        "d_model": 1024,
        "d_ff": 2048,
        "e_layers": 8,
        "n_heads": 8,
        "dropout": 0.1,
        "factor": 1,
        "activation": "gelu",
        "target": "OT",
        "checkpoints": args.root_path,
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
        "model_id": "token_norm_mi",
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


def main():
    parser = argparse.ArgumentParser(
        description="Per-token L2 norm analysis grouped by high/low MI patches"
    )
    parser.add_argument("--mi_file", type=str,
                        default="./global_mi_peaks_etth1.json",
                        help="Path to pre-computed MI peaks JSON file")
    parser.add_argument("--model_path", type=str,
                        default="./checkpoints/Timer_forecast_1.0.ckpt",
                        help="Path to Timer checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="./results/token_norm_mi_analysis/",
                        help="Output directory for plots and data")
    parser.add_argument("--root_path", type=str, default="./datasets/",
                        help="Root path to dataset directory")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv",
                        help="Dataset CSV filename")
    parser.add_argument("--data", type=str, default="ETTh1",
                        help="Dataset name (ETTh1, ETTh2, etc.)")
    parser.add_argument("--features", type=str, default="M",
                        choices=["M", "S", "MS"],
                        help="Feature type: M=multivariate, S=single, MS=multivariate-single")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=576)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of samples to use for the analysis")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="DataLoader num_workers")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Number of decoder layers in Timer (8 for ETTh1)")
    parser.add_argument("--d_model", type=int, default=1024,
                        help="Model dimension d_model")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward dimension d_ff")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--activation", type=str, default="gelu",
                        help="FFN activation function")
    parser.add_argument("--embed", type=str, default="timeF",
                        help="Time embedding type")
    parser.add_argument("--freq", type=str, default="h",
                        help="Time frequency (h=hourly, t=minutely)")
    parser.add_argument("--use_ims", action="store_true",
                        help="Use incremental multi-scale dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── 1. Load MI peaks info ─────────────────────────────────────────────────
    print(f"[1] Loading MI peaks info from: {args.mi_file}")
    if not os.path.exists(args.mi_file):
        raise FileNotFoundError(
            f"MI peaks file not found: {args.mi_file}\n"
            "Run etth1_mi_hsic_peaks.py first to generate it."
        )
    with open(args.mi_file, "r") as f:
        mi_info = json.load(f)
    n_layers = mi_info.get("num_layers", args.num_layers)
    n_patches = mi_info.get("N", args.seq_len // args.patch_len)
    print(f"  num_layers={n_layers}, num_patches={n_patches}")
    print(f"  Per-layer high MI patches: ", end="")
    for lk, lv in mi_info.get("layers", {}).items():
        print(f"L{lk}={len(lv.get('high_mi_patches', []))}", end=" ")
    print()

    # ── 2. Load Timer model ────────────────────────────────────────────────────
    print(f"\n[2] Loading Timer model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model_path}\n"
            "Download the Timer checkpoint first."
        )

    # Build namespace config (same pattern as etth1_mi_hsic_peaks.py)
    ns = build_namespace(args)
    ns.ckpt_path = args.model_path
    # Ensure these are set (may be set by build_namespace but prefer CLI args)
    ns.d_model = args.d_model
    ns.d_ff = args.d_ff
    ns.e_layers = args.num_layers
    ns.n_heads = args.n_heads
    ns.dropout = args.dropout
    ns.stride = args.patch_len
    # data_factory.py uses args.data for data_type
    ns.data = args.data
    model = TimerModel(ns).to(device)
    model.eval()
    print(f"  Model loaded on {device}")

    # ── 3. Load data ─────────────────────────────────────────────────────────
    print(f"\n[3] Loading {args.data} data (batch_size={args.batch_size})...")
    ns.batch_size = args.batch_size
    _, train_loader = data_provider(ns, flag="train")
    _, val_loader = data_provider(ns, flag="val")

    # Collect norms from both train and val (to get enough samples)
    all_norms_per_layer = {i: [] for i in range(n_layers)}

    print("  Running forward passes to collect per-token norms...")
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if sum(len(v) for v in all_norms_per_layer.values()) > 0 and batch_idx >= 20:
                break

            norms = compute_per_layer_norms(
                model, batch_x, batch_x_mark, device
            )
            for layer_idx, l2 in norms.items():
                all_norms_per_layer[layer_idx].append(l2)

    # Concatenate across batches: shape [total_batches * B, N]
    for layer_idx in all_norms_per_layer:
        if all_norms_per_layer[layer_idx]:
            all_norms_per_layer[layer_idx] = np.concatenate(
                all_norms_per_layer[layer_idx], axis=0
            )
        else:
            all_norms_per_layer[layer_idx] = np.zeros((0, n_patches))

    total_samples = next(iter(all_norms_per_layer.values())).shape[0]
    print(f"  Total samples collected: {total_samples} x {n_patches} tokens/layer")

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print(f"\n[4] Generating plots...")

    # 4a. Overlaid histograms of L2 norm distributions
    plot_norm_distributions(
        all_norms_per_layer, mi_info, n_layers,
        os.path.join(args.output_dir, "norm_distribution.png")
    )

    # 4b. Scatter: L2 norm vs patch index (high vs low MI)
    plot_norm_scatter(
        all_norms_per_layer, mi_info, n_layers,
        os.path.join(args.output_dir, "norm_scatter.png")
    )

    # 4c. Mean L2 norm per layer (line plot)
    plot_mean_norm_per_layer(
        all_norms_per_layer, mi_info, n_layers,
        os.path.join(args.output_dir, "mean_norm_per_layer.png")
    )

    # 4d. HSIC curve + norm heatmap side by side
    plot_hsic_curve_with_norm_heatmap(
        all_norms_per_layer, mi_info, n_layers,
        os.path.join(args.output_dir, "hsic_vs_norm.png")
    )

    # ── 5. Statistical tests ─────────────────────────────────────────────────
    run_statistical_tests(all_norms_per_layer, mi_info, n_layers)

    # ── 6. Save raw data ─────────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "norm_data.npz")
    np.savez(
        save_path,
        **{f"layer_{i}": v for i, v in all_norms_per_layer.items()}
    )
    print(f"\n[6] Raw norm data saved: {save_path}")

    print(f"\n[Done] All results in: {args.output_dir}")


if __name__ == "__main__":
    main()

