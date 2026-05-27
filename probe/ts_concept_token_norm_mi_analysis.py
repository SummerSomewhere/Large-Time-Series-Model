#!/usr/bin/env python3
"""
Token-Level L2 Norm vs HSIC Analysis on Synthetic Concepts (Timer version).

Hypothesis: High-HSIC token positions encode more concentrated concept information
than low-HSIC positions, and the two groups should have different norm distributions.

This script is used for analyzing Timer representations on synthetic concept data.
For real-data (ETTh1) analysis, see experiments/etth1_token_norm_mi_analysis.py.

Pipeline:
  1. Load synthetic concept dataset (from Step 1)
  2. Extract per-layer per-token hidden states via Timer forward
  3. For each layer, each token position t:
       HSIC(hx_t, param) — how much the token encodes the concept parameter
  4. Per layer, split all token positions into:
       top-K  (high HSIC)  vs  bottom-K  (low HSIC)
  5. For each group, collect L2 norms of ALL sample tokens in high/low groups
     and compare distributions with:
       - Mann-Whitney U test
       - Cliff's Delta effect size
       - Descriptive statistics
  6. Visualize:
       - HSIC heatmap per layer
       - Norm comparison bar plots with significance
       - Per-layer detailed profiles

Usage:
    python probe/ts_concept_token_norm_mi_analysis.py \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --dataset_path ./results/synthetic/concepts_dataset.pt \
        --seq_len 512 --pred_len 96 \
        --output_dir ./results/synthetic/norm_mi_analysis/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy import stats as scipy_stats

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from probe.ts_concept_synthetic_dataset import extract_param_vector, CONCEPT_PARAM_SPEC
from utils.hsic.hsic_core import (
    hsic_with_separate_sigmas,
    median_sq_bandwidth,
)

MI_DECODER_LAYER_CAP = 8


# ─────────────────────────────────────────────────────────────────────────────
# Timer forward (same pattern as forward_collect_layers in etth1_mi_hsic_peaks.py)
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_timer(model):
    return model.module if hasattr(model, "module") else model


def _apply_revin(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
    return (x - mean) / stdev


def timer_manual_forward(
    model,
    x: torch.Tensor,
    num_layers: int = MI_DECODER_LAYER_CAP,
) -> list[torch.Tensor]:
    """
    Run Timer forward and collect hidden states after each decoder attention block.

    Args:
        model: Timer model
        x: [B, seq_len] raw time series
        num_layers: number of decoder blocks to run

    Returns:
        layer_hs: list of [B, N, D] per-layer hidden states
    """
    B = x.shape[0]
    device = x.device

    core = _unwrap_timer(model)
    means = x.mean(1, keepdim=True)
    x_norm = x - means
    stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_norm = x_norm / stdev
    # Timer convention: univariate [B, L] -> [B, 1, L]; multivariate [B, M, L] -> [B, M, L]
    x2 = x_norm.unsqueeze(1).float() if x_norm.dim() == 2 else x_norm.permute(0, 2, 1).float()
    dec_in, n_vars = core.enc_embedding(x2)  # [BM, N, D]
    BM, N, D = dec_in.shape

    def pool(z: torch.Tensor) -> torch.Tensor:
        return z.view(B, n_vars, N, D).mean(dim=1)

    from utils.masking import TriangularCausalMask
    mask = TriangularCausalMask(BM, N, device=device)

    hidden_states = dec_in
    layer_hs: list[torch.Tensor] = []

    with torch.no_grad():
        for i, o3 in enumerate(core.decoder.attn_layers):
            if i >= num_layers:
                break
            hidden_states, _, _ = o3(hidden_states, attn_mask=mask)
            rep = pool(hidden_states.float())
            layer_hs.append(rep)

    return layer_hs


# ─────────────────────────────────────────────────────────────────────────────
# Core: HSIC per token (concept parameter level)
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_hsic(
    token_reps: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    """
    Compute HSIC(token_t, param) for each token position t.

    Args:
        token_reps: [N, N_patches, D] per-token hidden states
        target:     [N] scalar values (one per sample)

    Returns:
        hsic_curve: [N_patches] HSIC per token position
    """
    N, N_patches, D = token_reps.shape
    sigma_x_sq = median_sq_bandwidth(token_reps.flatten(0, 1))
    sigma_y_sq = median_sq_bandwidth(target.reshape(-1, 1))

    if not (np.isfinite(sigma_x_sq) and sigma_x_sq > 1e-10):
        sigma_x_sq = 1.0
    if not (np.isfinite(sigma_y_sq) and sigma_y_sq > 1e-10):
        sigma_y_sq = 1.0

    curve = np.zeros(N_patches, dtype=np.float64)
    target_np = target.numpy().astype(np.float64)
    token_np = token_reps.numpy().astype(np.float64)

    for p in range(N_patches):
        X = token_np[:, p, :]
        Y = target_np.reshape(-1, 1)
        curve[p] = hsic_with_separate_sigmas(X, Y, sigma_x_sq, sigma_y_sq)
    return curve


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    model,
    X: torch.Tensor,
    params: torch.Tensor,
    concept_idx: torch.Tensor,
    concepts: list[str],
    n_samples_per_concept: int,
    top_k: int = 4,
    max_samples: int = None,
    seed: int = 42,
    device: torch.device = None,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    N_total = X.shape[0]
    if max_samples is not None and max_samples < N_total:
        selected = np.random.choice(N_total, max_samples, replace=False)
        X = X[selected]
        params = params[selected]
        concept_idx = concept_idx[selected]
        N = max_samples
    else:
        N = N_total

    num_layers = MI_DECODER_LAYER_CAP
    print(f"  Collecting hidden states: N={N}, num_layers={num_layers}")

    # Collect per-layer hidden states in batches
    layer_hs_list: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    batch_size = 128
    for i in range(0, N, batch_size):
        bx = X[i:i+batch_size].float().to(device)
        layer_hs = timer_manual_forward(model, bx, num_layers=num_layers)
        for li in range(num_layers):
            layer_hs_list[li].append(layer_hs[li].cpu())

    layer_tokens: list[torch.Tensor] = []
    for li in range(num_layers):
        layer_tokens.append(torch.cat(layer_hs_list[li], dim=0))

    N_actual, N_patches, D = layer_tokens[0].shape
    print(f"  Shape per layer: [{N_actual}, {N_patches}, {D}]")

    # Compute L2 norms per position per layer: [N, N_patches]
    norm_per_layer: list[np.ndarray] = []
    for li in range(num_layers):
        norms = layer_tokens[li].norm(dim=2).numpy()
        norm_per_layer.append(norms)

    results_rows = []
    hsic_per_layer: list[np.ndarray] = []

    print(f"\n  Top/bottom {top_k} tokens per layer")
    for li in range(num_layers):
        tok_rep = layer_tokens[li]
        hsic_per_concept: list[np.ndarray] = []
        norm_per_concept_h: list[np.ndarray] = []
        norm_per_concept_l: list[np.ndarray] = []

        for ci, concept in enumerate(concepts):
            mask = concept_idx == ci
            if mask.sum() < 20:
                continue
            concept_tok = tok_rep[mask]           # [n_c, N_patches, D]
            concept_params = params[mask]           # [n_c, 4]

            # Use the primary param for this concept
            active_dims, _ = CONCEPT_PARAM_SPEC[concept]
            target = concept_params[:, active_dims[0]]  # [n_c]

            hsic_curve = compute_token_hsic(concept_tok, target)  # [N_patches]
            hsic_per_concept.append(hsic_curve)

            # Top/bottom positions
            sorted_pos = np.argsort(hsic_curve)[::-1]
            top_pos = sorted_pos[:top_k]
            bot_pos = sorted_pos[-top_k:]

            high_norms = concept_tok[:, top_pos, :].norm(dim=2).numpy().flatten()
            low_norms = concept_tok[:, bot_pos, :].norm(dim=2).numpy().flatten()
            norm_per_concept_h.append(high_norms)
            norm_per_concept_l.append(low_norms)

        # Aggregate across concepts
        all_high_norms = np.concatenate(norm_per_concept_h)
        all_low_norms = np.concatenate(norm_per_concept_l)
        avg_hsic = np.mean(hsic_per_concept, axis=0)  # [N_patches]

        hsic_per_layer.append(avg_hsic)

        # Mann-Whitney U test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u_stat, p_value = scipy_stats.mannwhitneyu(all_high_norms, all_low_norms, alternative="two-sided")

        # Cliff's Delta
        n1, n2 = len(all_high_norms), len(all_low_norms)
        delta = 0.0
        for a in all_high_norms.flat:
            for b in all_low_norms.flat:
                if a > b:
                    delta += 1
                elif a < b:
                    delta -= 1
        delta /= (n1 * n2)

        abs_d = abs(delta)
        if abs_d < 0.147:
            effect = "negligible"
        elif abs_d < 0.33:
            effect = "small"
        elif abs_d < 0.474:
            effect = "medium"
        else:
            effect = "large"

        if p_value < 1e-10:
            signif = "***"
        elif p_value < 1e-5:
            signif = "**"
        elif p_value < 0.001:
            signif = "*"
        elif p_value < 0.01:
            signif = "."
        else:
            signif = ""

        mu_high = float(np.mean(all_high_norms))
        mu_low = float(np.mean(all_low_norms))
        ratio = mu_high / (mu_low + 1e-10)
        direction = "H>L" if mu_high > mu_low else "H<L"

        results_rows.append({
            "layer": li,
            "n_high": int(n1),
            "n_low": int(n2),
            "n_tokens": N_patches,
            "top_k": top_k,
            "mu_high": mu_high,
            "mu_low": mu_low,
            "med_high": float(np.median(all_high_norms)),
            "med_low": float(np.median(all_low_norms)),
            "std_high": float(np.std(all_high_norms)),
            "std_low": float(np.std(all_low_norms)),
            "u_stat": float(u_stat),
            "p_value": float(p_value),
            "cliffs_delta": float(delta),
            "effect_size": effect,
            "significant": signif,
            "direction": direction,
            "ratio": float(ratio),
        })

    return {
        "n_samples": N,
        "n_tokens": N_patches,
        "top_k": top_k,
        "layers": results_rows,
    }, hsic_per_layer, norm_per_layer


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(rows: list[dict]):
    header = (
        f"{'Layer':>5} | {'n_high':>7} | {'n_low':>7} | "
        f"{'mu_high':>10} | {'mu_low':>10} | {'ratio':>8} | "
        f"{'U-stat':>12} | {'p-value':>12} | {'Cliff-d':>8} | {'Effect':>10} | {'Sig'}"
    )
    print("\n" + "=" * len(header))
    print("Statistical Tests (Mann-Whitney U): High-HSIC vs Low-HSIC L2 Norms (Timer)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['layer']:>5} | "
            f"{r['n_high']:>7} | "
            f"{r['n_low']:>7} | "
            f"{r['mu_high']:>10.4f} | "
            f"{r['mu_low']:>10.4f} | "
            f"{r['ratio']:>8.4f} | "
            f"{r['u_stat']:>12.1f} | "
            f"{r['p_value']:>12.2e} | "
            f"{r['cliffs_delta']:>8.4f} | "
            f"{r['effect_size']:>10} | "
            f"{r['significant']}"
        )
    print("=" * len(header))
    print(f"\nCliff's Delta interpretation: |d|<0.147 negligible | d|<0.33 small | d|<0.474 medium | d|>=0.474 large")
    print(f"Significance: *** p<1e-10  ** p<1e-5  * p<0.001  . p<0.01")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(results: dict, output_dir: str):
    rows = results["layers"]
    n_layers = len(rows)
    x = np.arange(n_layers)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Mean norm bar chart
    ax = axes[0, 0]
    mu_h = [r["mu_high"] for r in rows]
    mu_l = [r["mu_low"] for r in rows]
    err_h = [r["std_high"] for r in rows]
    err_l = [r["std_low"] for r in rows]
    w = 0.35
    ax.bar(x - w/2, mu_h, w, yerr=err_h, label="High-HSIC tokens", color="steelblue", alpha=0.85, capsize=3)
    ax.bar(x + w/2, mu_l, w, yerr=err_l, label="Low-HSIC tokens", color="coral", alpha=0.85, capsize=3)
    for ri, r in enumerate(rows):
        color = "green" if r["direction"] == "H>L" else "red"
        ax.text(ri, max(r["mu_high"], r["mu_low"]) + r["std_high"] * 1.5,
                r["significant"], ha="center", va="bottom", color=color, fontsize=11, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Mean L2 Norm: High-HSIC vs Low-HSIC Tokens")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    # 2. Cliff's Delta per layer
    ax = axes[0, 1]
    deltas = [r["cliffs_delta"] for r in rows]
    colors_d = ["green" if d < 0 else "red" for d in deltas]
    bars = ax.bar(x, deltas, color=colors_d, alpha=0.85)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.axhline(-0.147, color="gray", ls=":", lw=0.8, label="negligible")
    ax.axhline(0.147, color="gray", ls=":", lw=0.8)
    ax.axhline(-0.474, color="gray", ls="--", lw=0.8, label="large")
    ax.axhline(0.474, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cliff's Delta")
    ax.set_title("Effect Size (Cliff's Delta): High-HSIC vs Low-HSIC\n(<0 = high-HSIC norms larger)")
    ax.set_xticks(x)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25, axis="y")
    for bar, r in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r['cliffs_delta']:.3f}\n({r['effect_size']})",
                ha="center", va="bottom", fontsize=7, color="dimgray")

    # 3. Norm ratio
    ax = axes[1, 0]
    ratios = [r["ratio"] for r in rows]
    colors3 = ["green" if r < 1.0 else "coral" for r in ratios]
    ax.bar(x, ratios, color=colors3, alpha=0.85)
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="equal")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Norm Ratio (high / low)")
    ax.set_title("Norm Ratio: High-HSIC / Low-HSIC per Layer\n(<1 = high-HSIC tokens have smaller norms)")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    for xi, (bar, r) in enumerate(zip(ax.patches, ratios)):
        color = "darkgreen" if r < 1.0 else "darkred"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r:.3f}", ha="center", va="bottom", fontsize=8, color=color)

    # 4. Log p-value
    ax = axes[1, 1]
    pvals = [r["p_value"] for r in rows]
    log_p = [-np.log10(max(p, 1e-300)) for p in pvals]
    ax.bar(x, log_p, color="purple", alpha=0.75)
    ax.axhline(-np.log10(0.05), color="orange", ls="--", lw=1.2, label="p=0.05")
    ax.axhline(-np.log10(0.001), color="red", ls="--", lw=1.2, label="p=0.001")
    ax.set_xlabel("Layer")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Statistical Significance (Mann-Whitney U)\nHigher = more significant")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle(
        f"Token Norm vs HSIC Analysis | Timer | Synthetic Concepts | "
        f"N={results['n_samples']} | T={results['n_tokens']} | "
        f"top/bottom {results['top_k']} tokens",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")


def plot_hsic_heatmap(hsic_per_layer: list, output_dir: str):
    data = np.stack(hsic_per_layer, axis=0)   # [L, T]
    L, T = data.shape
    fig, ax = plt.subplots(figsize=(max(8, T * 0.15), max(4, L * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(0, T, max(1, T // 10)))
    ax.set_xticklabels([str(t) for t in np.arange(0, T, max(1, T // 10))])
    ax.set_yticks(np.arange(L))
    ax.set_yticklabels([f"L{i}" for i in range(L)])
    ax.set_title("Avg HSIC(History Token, Concept Param) per Layer (Timer)")
    plt.colorbar(im, ax=ax, label="HSIC")
    plt.tight_layout()
    out = os.path.join(output_dir, "hsic_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")


def plot_per_layer_detail(
    hsic_per_layer: list[np.ndarray],
    norm_per_layer: list[np.ndarray],
    rows: list[dict],
    output_dir: str,
):
    n_layers = len(hsic_per_layer)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for li in range(n_layers):
        ax = axes[li]
        hsic = hsic_per_layer[li]
        norms = norm_per_layer[li]
        mean_norms = norms.mean(axis=0)
        std_norms = norms.std(axis=0)
        r = rows[li]

        sorted_pos = np.argsort(hsic)[::-1]
        top_pos = sorted_pos[:r["top_k"]]
        bot_pos = sorted_pos[-r["top_k"]:]

        ax.fill_between(np.arange(len(mean_norms)),
                        mean_norms - std_norms,
                        mean_norms + std_norms,
                        alpha=0.2, color="steelblue")
        ax.plot(np.arange(len(mean_norms)), mean_norms,
                color="steelblue", lw=1.5, label="mean norm")

        ax2 = ax.twinx()
        hsic_n = (hsic - hsic.min()) / (hsic.max() - hsic.min() + 1e-10)
        ax2.plot(np.arange(len(hsic)), hsic_n, "--", color="orange", lw=1.2, alpha=0.8, label="HSIC (norm)")
        ax2.set_yticks([])

        for p in top_pos:
            ax.axvline(p, color="blue", alpha=0.3, lw=1.0, zorder=0)
        for p in bot_pos:
            ax.axvline(p, color="red", alpha=0.3, lw=1.0, zorder=0)

        handles = [
            mpatches.Patch(color="blue", alpha=0.4, label=f"Top {r['top_k']} HSIC"),
            mpatches.Patch(color="red", alpha=0.4, label=f"Bot {r['top_k']} HSIC"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
        ax.set_xlabel("token position")
        ax.set_ylabel("L2 norm", color="steelblue")
        ax.set_title(f"L{li}: ratio={r['ratio']:.3f}, d={r['cliffs_delta']:.3f} ({r['effect_size']}) {r['significant']}")
        ax.grid(True, alpha=0.2)

    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Per-Layer Norm Profiles with High/Low HSIC Token Positions (Timer)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, "per_layer_detail.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")


def plot_top_bottom_distributions(
    rows: list[dict],
    output_dir: str,
):
    n_layers = len(rows)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), sharey=True)
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for li, r in enumerate(rows):
        ax = axes[li]
        parts = ax.violinplot([r["mu_high"], r["mu_low"]],
                              positions=[0.6, 1.4],
                              showmeans=True, showmedians=True)
        colors_v = ["steelblue", "coral"]
        for body, color in zip(parts["bodies"], colors_v):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        for elem in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
            if elem in parts:
                parts[elem].set_color("dimgray")
        ax.set_xticks([0.6, 1.4])
        ax.set_xticklabels(["High HSIC", "Low HSIC"])
        ax.set_title(f"L{li}\nd={r['cliffs_delta']:.3f} ({r['effect_size']})\n{r['significant']}")
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_ylabel("Mean L2 Norm" if li == 0 else "")

    plt.suptitle("High-HSIC vs Low-HSIC Norm Distribution (violin)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "dist_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Token-Level L2 Norm vs HSIC Analysis (Timer + Synthetic Concepts)"
    )
    p.add_argument("--ckpt_path", type=str,
                   default="checkpoints/Timer_forecast_1.0.ckpt",
                   help="Path to Timer checkpoint")
    p.add_argument("--dataset_path", type=str,
                   default="./results/synthetic/concepts_dataset.pt",
                   help="Path to synthetic concepts dataset")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--patch_len", type=int, default=96)
    p.add_argument("--stride", type=int, default=96)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--top_k", type=int, default=4,
                   help="Top/bottom K tokens by HSIC")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max samples (None = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str,
                   default="./results/synthetic/norm_mi_analysis/")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("Token-Level L2 Norm vs HSIC Analysis (Timer + Synthetic Concepts)")
    print("=" * 70)
    print(f"  ckpt_path   : {args.ckpt_path}")
    print(f"  dataset_path : {args.dataset_path}")
    print(f"  seq_len     : {args.seq_len}")
    print(f"  patch_len   : {args.patch_len}")
    print(f"  top_k       : {args.top_k}")
    print(f"  seed        : {args.seed}")
    print(f"  device      : {device}")
    print(f"  output_dir  : {args.output_dir}")
    print("=" * 70)

    # ── 1. Load synthetic dataset ────────────────────────────────────────────
    print("\n[1] Loading synthetic concept dataset...")
    ckpt_ds = torch.load(args.dataset_path, map_location="cpu")
    X = ckpt_ds["X"]                        # [N, seq_len]
    labels = ckpt_ds["labels"]
    concepts = ckpt_ds["concepts"]
    n_samples_per_concept = ckpt_ds["n_samples_per_concept"]
    seq_len_ds = ckpt_ds["seq_len"]
    print(f"  X shape: {X.shape}")
    print(f"  concepts: {concepts}")

    # Build params (CONCEPT_PARAM_SPEC already imported at top of file)
    param_vectors = [extract_param_vector(lbl) for lbl in labels]
    params = torch.from_numpy(np.stack(param_vectors, axis=0)).float()
    concept_map = {c: i for i, c in enumerate(concepts)}
    concept_idx = torch.tensor(
        [concept_map[lbl["concept"]] for lbl in labels], dtype=torch.long
    )

    # ── 2. Load Timer ──────────────────────────────────────────────────────
    print("\n[2] Loading Timer model...")
    import argparse as _argparse
    ns = _argparse.Namespace(
        task_name="forecast", is_training=0, is_finetuning=0, train_test=0,
        use_multi_gpu=False, d_layers=1, target="OT",
        checkpoints="./checkpoints/", inverse=False, use_amp=False,
        use_weight_decay=0, weight_decay=0.01, loss="MSE", lradj="type1",
        train_epochs=0, patience=3, learning_rate=1e-4, itr=1,
        finetune_epochs=0, output_attention=False, distil=True,
        model_id="timer_norm_mi", model="Timer",
        output_len_list=None, mask_rate=0.25, data_type="custom",
        decay_fac=0.75, cos_warm_up_steps=100, cos_max_decay_steps=60000,
        cos_max_decay_epoch=10, cos_max=1e-4, cos_min=2e-6,
        patch_len=args.patch_len, stride=args.stride,
        d_model=args.d_model, d_ff=args.d_ff, n_heads=args.n_heads,
        dropout=0.1, activation="gelu", e_layers=args.e_layers, factor=args.factor,
    )
    sys.path.insert(0, os.path.join(_ROOT, "models"))
    from Timer import Model as TimerModel
    model = TimerModel(ns).to(device)
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  Loaded checkpoint from {args.ckpt_path}")
    else:
        print(f"  Warning: checkpoint not found at {args.ckpt_path}, using random init")
    model.eval()
    num_layers = MI_DECODER_LAYER_CAP
    print(f"  Timer loaded, {num_layers} layers will be analyzed")

    # ── 3. Run analysis ─────────────────────────────────────────────────────
    print("\n[3] Running norm vs HSIC analysis...")
    results, hsic_per_layer, norm_per_layer = run_analysis(
        model=model,
        X=X,
        params=params,
        concept_idx=concept_idx,
        concepts=concepts,
        n_samples_per_concept=n_samples_per_concept,
        top_k=args.top_k,
        max_samples=args.max_samples,
        seed=args.seed,
        device=device,
    )

    # ── 4. Print table ──────────────────────────────────────────────────────
    _print_table(results["layers"])

    # ── 5. Save results ─────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {results_path}")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    print("\n[6] Generating plots...")
    plot_summary(results, args.output_dir)
    plot_hsic_heatmap(hsic_per_layer, args.output_dir)
    plot_per_layer_detail(hsic_per_layer, norm_per_layer, results["layers"], args.output_dir)
    plot_top_bottom_distributions(results["layers"], args.output_dir)

    # Save raw data
    norm_data_path = os.path.join(args.output_dir, "norm_data.npz")
    np.savez_compressed(
        norm_data_path,
        **{f"norm_L{li}": norm_per_layer[li] for li in range(num_layers)},
        **{f"hsic_L{li}": hsic_per_layer[li] for li in range(num_layers)},
    )
    print(f"[Data] Raw norm data saved: {norm_data_path}")

    print(f"\n[Done] All results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
