#!/usr/bin/env python3
"""
Step 4: Token-Level HSIC Analysis - High vs Low HSIC Token Probe Comparison

Hypothesis: High-HSIC token positions encode more concentrated concept information
than low-HSIC positions.

Pipeline:
  1. Load per-token (unpooled) hidden states from Step 2
  2. For each layer and each concept param:
       - Compute HSIC(token_pos, param) using utils/hsic/hsic_core.py
       - Sort token positions by HSIC
       - Split into top-K (high HSIC) and bottom-K (low HSIC) groups
       - Take mean vector over tokens in each group
       - Run linear probe on high-HSIC vs low-HSIC averaged vectors
  3. Compare probe quality (MSE, R^2) between high-HSIC and low-HSIC groups
  4. Visualize with grouped bar charts

Usage:
    python probe/ts_concept_mi_token_analysis.py \
        --rep_dir ./results/synthetic/representations/ \
        --output_dir ./results/synthetic/mi_token_analysis/ \
        --top_k 4 --probe_epochs 200 --probe_lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from probe.ts_concept_synthetic_dataset import TSConceptGenerator, CONCEPT_PARAM_SPEC
from utils.hsic.hsic_core import (
    hsic_with_separate_sigmas,
    median_sq_bandwidth,
)


# ─────────────────────────────────────────────────────────────────────────────
# MI / HSIC Estimation (using HSIC from utils/hsic/hsic_core.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_mi(
    token_reps: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10,
    n_samples: int = None,
) -> np.ndarray:
    """
    Estimate HSIC(token_pos, param) for each token position.
    HSIC is a consistent, non-parametric dependence measure that is equivalent
    to a biased estimator of MI under RBF kernels.

    For each patch position, computes:
        HSIC(hx_t, y) = (1/(n-1)^2) * tr(H K H @ H L H)
    where K_ij = exp(-||hx_t[i] - hx_t[j]||^2 / (2*sigma_x^2))
          L_ij = exp(-||y[i]    - y[j]||^2    / (2*sigma_y^2))
    and sigma values are set via median heuristic.

    Args:
        token_reps: [N, N_patches, D] per-token hidden states
        target:     [N] scalar values (one per sample)
        n_bins:     ignored (kept for API compatibility)
        n_samples:  subsample N for faster computation (None = all)

    Returns:
        hsic_scores: [N_patches] HSIC scores per token position
    """
    N, N_patches, D = token_reps.shape

    if n_samples is not None and n_samples < N:
        idx = np.random.choice(N, n_samples, replace=False)
        token_reps = token_reps[idx]
        target = target[idx]
        N = n_samples

    target_np = target.numpy().astype(np.float64)
    token_np = token_reps.numpy().astype(np.float64)

    token_np = token_np.astype(np.float32)
    target_np = target_np.astype(np.float32)

    hsic_scores = np.zeros(N_patches)

    for p in range(N_patches):
        h_tok = torch.from_numpy(token_np[:, p, :])  # [N, D]
        y_tok = torch.from_numpy(target_np).unsqueeze(1) if target_np.ndim == 1 else torch.from_numpy(target_np)  # [N, 1] or [N]

        # Ensure y is [N, 1]
        y_tok = y_tok.reshape(N, 1)

        # Median bandwidth for token representations and for target
        sigma_x_sq = median_sq_bandwidth(h_tok)
        sigma_y_sq = median_sq_bandwidth(y_tok)

        # Fallback if bandwidth computation fails
        if not (np.isfinite(sigma_x_sq) and sigma_x_sq > 1e-10):
            sigma_x_sq = 1.0
        if not (np.isfinite(sigma_y_sq) and sigma_y_sq > 1e-10):
            sigma_y_sq = 1.0

        hsic_scores[p] = hsic_with_separate_sigmas(
            h_tok, y_tok,
            sigma_x_sq=float(sigma_x_sq),
            sigma_y_sq=float(sigma_y_sq),
        )

    return hsic_scores


# ─────────────────────────────────────────────────────────────────────────────
# Linear Probe
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(d_model, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    z: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Train linear probe, return val MSE and R^2.
    Uses 80/20 split within the provided data.
    """
    N = z.shape[0]
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    z_np = z.numpy()
    y_np = y.numpy()

    z_tr, z_vl, y_tr, y_vl = train_test_split(
        z_np, y_np, train_size=0.8, random_state=seed, shuffle=True
    )
    z_tr = torch.from_numpy(z_tr).float()
    z_vl = torch.from_numpy(z_vl).float()
    y_tr = torch.from_numpy(y_tr).float()
    y_vl = torch.from_numpy(y_vl).float()

    in_dim = z_tr.shape[1]
    out_dim = y_tr.shape[1]
    probe = LinearProbe(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    crit = nn.MSELoss()

    n = z_tr.shape[0]
    bs = min(256, n)
    for _ in range(n_epochs):
        probe.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            bi = idx[i:i+bs]
            opt.zero_grad()
            loss = crit(probe(z_tr[bi].to(device)), y_tr[bi].to(device))
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(z_vl.to(device))
        mse = crit(pred, y_vl.to(device)).item()
        ss_res = ((pred.cpu() - y_vl) ** 2).sum().item()
        ss_tot = ((y_vl - y_vl.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return mse, r2


# CONCEPT_PARAM_SPEC is imported from ts_concept_synthetic_dataset.py


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_mi_token_analysis(
    layer_tokens: list[torch.Tensor],   # list of [N, N_patches, D]
    params: torch.Tensor,                # [N, 4]
    concept_idx: torch.Tensor,           # [N]
    labels: list[dict],
    concepts: list[str],
    n_samples_per_concept: int,
    d_model: int,
    n_layers: int,
    output_dir: str,
    top_k: int = 4,
    max_samples: int = None,
    probe_epochs: int = 200,
    probe_lr: float = 1e-3,
    device: torch.device = None,
    seed: int = 42,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    N, N_patches, D = layer_tokens[0].shape

    results = {}  # (layer, concept, param_idx) -> {high_mse, low_mse, high_r2, low_r2}

    print(f"{'Layer':>5} | {'Concept':<25} | {'Param':>10} | {'MI_h':>8} | {'MI_l':>8} | {'MSE_h':>10} | {'MSE_l':>10} | {'R2_h':>8} | {'R2_l':>8} | {'ΔMSE':>10} | {'Better'}")
    print("-" * 130)

    for layer in range(n_layers):
        tok_rep = layer_tokens[layer]  # [N, N_patches, D]

        for ci, concept in enumerate(concepts):
            mask = concept_idx == ci
            if isinstance(tok_rep, torch.Tensor):
                concept_tok = tok_rep[mask].contiguous()
            else:
                concept_tok = tok_rep[mask].copy()
            concept_params = params[mask]          # [n_c, 4]

            active_dims, _ = CONCEPT_PARAM_SPEC[concept]
            n_c = concept_tok.shape[0]

            if max_samples is not None and n_c > max_samples:
                idx = np.random.choice(n_c, max_samples, replace=False)
                concept_tok = concept_tok[idx]
                concept_params = concept_params[idx]
                n_c = max_samples

            if n_c < 20:
                continue

            for dim_idx in active_dims:
                target = concept_params[:, dim_idx]  # [n_c]

                # Compute HSIC per token position
                hsic_scores = compute_token_mi(concept_tok, target)

                # Sort token positions by HSIC
                sorted_positions = np.argsort(hsic_scores)[::-1]  # descending

                high_pos = sorted_positions[:top_k].copy()   # top-K highest HSIC
                low_pos = sorted_positions[-top_k:].copy()  # bottom-K lowest HSIC

                # High-HSIC: mean over high-HSIC token positions
                selected_h = concept_tok[:, high_pos, :]
                if isinstance(selected_h, np.ndarray):
                    high_rep = torch.from_numpy(selected_h.mean(axis=1))
                    low_rep = torch.from_numpy(concept_tok[:, low_pos, :].mean(axis=1))
                else:
                    high_rep = selected_h.mean(dim=1)
                    low_rep = concept_tok[:, low_pos, :].mean(dim=1)

                # Target (only the active dim)
                y = target

                # Train probe on high-HSIC and low-HSIC vectors
                mse_h, r2_h = train_probe(high_rep, y, probe_epochs, probe_lr, device, seed)
                mse_l, r2_l = train_probe(low_rep, y, probe_epochs, probe_lr, device, seed)

                delta_mse = mse_l - mse_h
                better = "HIGH" if delta_mse > 0 else "LOW "

                param_name = CONCEPT_PARAM_SPEC[concept][1].split(",")[dim_idx].strip()
                hsic_h = hsic_scores[high_pos].mean()
                hsic_l = hsic_scores[low_pos].mean()

                results[(layer, concept, dim_idx)] = {
                    "hsic_high": hsic_h,
                    "hsic_low": hsic_l,
                    "mse_high": mse_h,
                    "mse_low": mse_l,
                    "r2_high": r2_h,
                    "r2_low": r2_l,
                    "delta_mse": delta_mse,
                    "high_pos": high_pos,
                    "low_pos": low_pos,
                    "top_k": top_k,
                }

                print(f"{layer:>5} | {concept:<25} | {param_name:>10} | "
                      f"{hsic_h:>8.4f} | {hsic_l:>8.4f} | "
                      f"{mse_h:>10.6f} | {mse_l:>10.6f} | "
                      f"{r2_h:>8.4f} | {r2_l:>8.4f} | "
                      f"{delta_mse:>10.6f} | {better}")

    return results


def plot_results(
    results: dict,
    concepts: list[str],
    n_layers: int,
    output_dir: str,
    top_k: int,
):
    """
    Redesigned: 4-panel summary figure for High-MI vs Low-MI probe comparison.

    Panel layout:
      [0,0] Per-layer MSE bar:  avg MSE across all (concept, dim) pairs
      [0,1] Per-layer R²  bar:  avg R²  across all (concept, dim) pairs
      [1,0] ΔMSE heatmap:       layer × concept  (Low - High)
      [1,1] Win-rate bar:       fraction of (concept,dim) where High-MI wins per layer
    """
    layers = list(range(n_layers))
    n_concepts = len(concepts)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Panel [0,0]: Per-layer MSE comparison ────────────────────────────────
    ax = axes[0, 0]
    x = np.arange(n_layers)
    w = 0.35

    mse_h_all, mse_l_all = [], []
    for li, layer in enumerate(layers):
        vals_h = [v["mse_high"] for k, v in results.items() if k[0] == layer]
        vals_l = [v["mse_low"]  for k, v in results.items() if k[0] == layer]
        mse_h_all.append(np.nanmean(vals_h))
        mse_l_all.append(np.nanmean(vals_l))

    bars_h = ax.bar(x - w/2, mse_h_all, w, label="High-MI tokens", color="steelblue", alpha=0.85)
    bars_l = ax.bar(x + w/2, mse_l_all, w, label="Low-MI tokens",  color="coral",    alpha=0.85)

    for xi, (bh, bl) in enumerate(zip(bars_h, bars_l)):
        lo = max(bh.get_height(), bl.get_height())
        better = "H" if mse_h_all[xi] < mse_l_all[xi] else "L"
        color = "steelblue" if better == "H" else "coral"
        ax.text(xi, lo + lo * 0.01, better,
                ha="center", va="bottom", color=color, fontweight="bold", fontsize=10)

    ax.set_xlabel("Layer")
    ax.set_ylabel("MSE (mean across concepts & dims)")
    ax.set_title("Per-Layer MSE: High-MI vs Low-MI Tokens")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(bottom=0)

    # ── Panel [0,1]: Per-layer R² comparison ────────────────────────────────
    ax = axes[0, 1]

    r2_h_all, r2_l_all = [], []
    for li, layer in enumerate(layers):
        vals_h = [v["r2_high"] for k, v in results.items() if k[0] == layer]
        vals_l = [v["r2_low"]  for k, v in results.items() if k[0] == layer]
        r2_h_all.append(np.nanmean(vals_h))
        r2_l_all.append(np.nanmean(vals_l))

    bars_h = ax.bar(x - w/2, r2_h_all, w, label="High-MI tokens", color="steelblue", alpha=0.85)
    bars_l = ax.bar(x + w/2, r2_l_all, w, label="Low-MI tokens",  color="coral",    alpha=0.85)

    for xi, (bh, bl) in enumerate(zip(bars_h, bars_l)):
        hi = max(bh.get_height(), bl.get_height())
        better = "H" if r2_h_all[xi] > r2_l_all[xi] else "L"
        color = "steelblue" if better == "H" else "coral"
        ax.text(xi, hi + 0.01, better,
                ha="center", va="bottom", color=color, fontweight="bold", fontsize=10)

    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (mean across concepts & dims)")
    ax.set_title("Per-Layer R²: High-MI vs Low-MI Tokens")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel [1,0]: ΔMSE heatmap (Low - High) per layer × concept ─────────
    ax = axes[1, 0]

    delta_matrix = np.full((n_layers, n_concepts), np.nan)
    for li, layer in enumerate(layers):
        for ci, concept in enumerate(concepts):
            dms = [results.get((layer, concept, d), {}).get("delta_mse", np.nan)
                   for d in range(4)]
            active_dims = CONCEPT_PARAM_SPEC[concept][0]
            vals = [dms[d] for d in active_dims if not np.isnan(dms[d])]
            delta_matrix[li, ci] = np.nanmean(vals) if vals else np.nan

    v_abs = np.nanmax(np.abs(delta_matrix))
    if v_abs < 1e-6:
        v_abs = 0.5
    im = ax.imshow(delta_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-v_abs, vmax=v_abs)
    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels([c[:16] for c in concepts], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
    ax.set_ylabel("Layer")
    ax.set_title("ΔMSE Heatmap: (Low - High)\nBlue = High-MI tokens better", fontsize=10)
    plt.colorbar(im, ax=ax, label="ΔMSE")

    for li in range(n_layers):
        for ci in range(n_concepts):
            val = delta_matrix[li, ci]
            if np.isnan(val):
                continue
            ax.text(ci, li, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(val) > v_abs * 0.55 else "black")

    # ── Panel [1,1]: Win-rate bar per layer ──────────────────────────────────
    ax = axes[1, 1]

    win_rates = []
    for layer in layers:
        total = 0
        wins_high = 0
        for k, v in results.items():
            if k[0] == layer:
                total += 1
                if v["delta_mse"] > 0:   # High-MI has lower MSE
                    wins_high += 1
        win_rates.append(wins_high / max(total, 1))

    colors_wr = ["steelblue" if r > 0.5 else "coral" for r in win_rates]
    bars = ax.bar(x, win_rates, color=colors_wr, alpha=0.85)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, label="random baseline")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction where High-MI wins")
    ax.set_title("Win Rate: High-MI Tokens Have Lower MSE\n(>0.5 = High-MI generally better)")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, axis="y")

    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wr:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.legend(fontsize=9)

    plt.suptitle(
        f"High-MI vs Low-MI Token Probe (top_k={top_k}, {n_concepts} concepts)\n"
        f"n_layers={n_layers}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "mi_token_probe_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_mi_profile(
    results: dict,
    concepts: list[str],
    n_layers: int,
    output_dir: str,
):
    """
    Redesigned: 3 panels + per-concept detail.

    [0]  Per-layer R² line plot: High-MI vs Low-MI (all concepts averaged).
         Shows how R² evolves across layers and which group wins.

    [1]  R² heatmap: layer × concept, difference (High - Low).
         Reveals where High-MI wins per concept/layer.

    [2]  Per-concept R² bar subplot grid.
         One subplot per concept, each showing n_layers pairs of bars.
         Lets you read off which concept/layer combinations favor High-MI tokens.
    """
    n_concepts = len(concepts)
    n_cols_c = min(3, n_concepts)
    n_rows_c = (n_concepts + n_cols_c - 1) // n_cols_c

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 1.4])

    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]

    layers = list(range(n_layers))
    x = np.arange(n_layers)

    # ── Panel [0]: Per-layer R² line plot ───────────────────────────────────
    ax = axes[0]

    r2_h_all, r2_l_all = [], []
    for li, layer in enumerate(layers):
        vals_h = [v["r2_high"] for k, v in results.items() if k[0] == layer]
        vals_l = [v["r2_low"]  for k, v in results.items() if k[0] == layer]
        r2_h_all.append(np.nanmean(vals_h))
        r2_l_all.append(np.nanmean(vals_l))

    ax.plot(x, r2_h_all, "o-", color="steelblue", lw=2,  ms=6, label="High-MI tokens")
    ax.plot(x, r2_l_all, "s--", color="coral",    lw=2,  ms=6, label="Low-MI tokens")
    ax.fill_between(x, r2_h_all, r2_l_all,
                    where=[a > b for a, b in zip(r2_h_all, r2_l_all)],
                    alpha=0.15, color="steelblue", label="High-MI wins region")
    ax.fill_between(x, r2_h_all, r2_l_all,
                    where=[a <= b for a, b in zip(r2_h_all, r2_l_all)],
                    alpha=0.15, color="coral", label="Low-MI wins region")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg R²")
    ax.set_title("Per-Layer R²: High-MI vs Low-MI (avg over concepts)")
    ax.set_xticks(x)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=min(min(r2_h_all), min(r2_l_all)) - 0.05)

    # ── Panel [1]: R² diff heatmap per layer × concept ───────────────────────
    ax = axes[1]

    diff_matrix = np.full((n_layers, n_concepts), np.nan)
    for li, layer in enumerate(layers):
        for ci, concept in enumerate(concepts):
            vals_h = [results.get((layer, concept, d), {}).get("r2_high", np.nan) for d in range(4)]
            vals_l = [results.get((layer, concept, d), {}).get("r2_low",  np.nan) for d in range(4)]
            active_dims = CONCEPT_PARAM_SPEC[concept][0]
            vh = np.nanmean([vals_h[d] for d in active_dims])
            vl = np.nanmean([vals_l[d] for d in active_dims])
            diff_matrix[li, ci] = vh - vl

    v_abs = max(0.05, np.nanmax(np.abs(diff_matrix)))
    im = ax.imshow(diff_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-v_abs, vmax=v_abs)
    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels([c[:16] for c in concepts], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
    ax.set_ylabel("Layer")
    ax.set_title("R² Difference: (High - Low)\nBlue = High-MI better, Red = Low-MI better", fontsize=10)
    plt.colorbar(im, ax=ax, label="ΔR²")

    for li in range(n_layers):
        for ci in range(n_concepts):
            val = diff_matrix[li, ci]
            if np.isnan(val):
                continue
            ax.text(ci, li, f"{val:.3f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(val) > v_abs * 0.55 else "black")

    # ── Panel [2]: Per-concept R² bar subplots ──────────────────────────────
    ax_conc = axes[2]
    ax_conc.remove()   # remove from grid, we'll use inset axes

    inset_axes = []
    for ci, concept in enumerate(concepts):
        row = ci // n_cols_c
        col = ci % n_cols_c
        box = [
            col / n_cols_c,
            1.0 - (row + 1) / n_rows_c,
            1.0 / n_cols_c - 0.02,
            1.0 / n_rows_c - 0.03,
        ]
        ax_ins = fig.add_axes(box)
        inset_axes.append(ax_ins)

        for li, layer in enumerate(layers):
            vals_h = [results.get((layer, concept, d), {}).get("r2_high", np.nan) for d in range(4)]
            vals_l = [results.get((layer, concept, d), {}).get("r2_low",  np.nan) for d in range(4)]
            active_dims = CONCEPT_PARAM_SPEC[concept][0]
            r2h = np.nanmean([vals_h[d] for d in active_dims])
            r2l = np.nanmean([vals_l[d] for d in active_dims])
            bw = 0.35
            ax_ins.bar(li - bw/2, r2h, bw, color="steelblue", alpha=0.85)
            ax_ins.bar(li + bw/2, r2l, bw, color="coral",    alpha=0.85)

        ax_ins.set_title(concept[:20], fontsize=9, fontweight="bold")
        ax_ins.set_xticks(range(n_layers))
        ax_ins.set_xticklabels([f"L{l}" for l in range(n_layers)], fontsize=7)
        ax_ins.tick_params(axis="y", labelsize=7)
        ax_ins.grid(True, alpha=0.2, axis="y")
        ax_ins.set_ylim(0, 1.05)

    # Shared ylabel only on leftmost
    if n_concepts > 0:
        box0 = inset_axes[0].get_position()
        inset_axes[0].set_ylabel("R²", fontsize=8)

    # Legend for per-concept
    fig.text(0.5, 0.28, "Per-Concept R²:  blue = High-MI tokens,  coral = Low-MI tokens",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    plt.suptitle(
        f"Token-Level R² Analysis: High-MI vs Low-MI (per-layer, per-concept)",
        fontsize=13, fontweight="bold",
    )
    out_path = os.path.join(output_dir, "mi_token_profile.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def _agg(results: dict, key: str, layer: int, concept: str, active_dims: list) -> float:
    vals = [results.get((layer, concept, d), {}).get(key, np.nan) for d in range(4)]
    return float(np.nanmean([vals[d] for d in active_dims if not np.isnan(vals[d])]))


def plot_concept_mse_r2_comparison(
    results: dict,
    concepts: list[str],
    n_layers: int,
    output_dir: str,
    top_k: int,
):
    """
    Two-column figure: one concept per row, two columns (MSE | R²).

    Each row:
      - Left  : per-layer MSE bar pairs (High-MI blue / Low-MI coral)
      - Right : per-layer R²  bar pairs

    Spans all concepts × all layers. Easy to read off per-concept
    performance of the two token groups.
    """
    n_concepts = len(concepts)
    n_cols = 2

    fig, axes = plt.subplots(n_concepts, n_cols, figsize=(12, 4 * n_concepts),
                               squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    layers = list(range(n_layers))
    x = np.arange(n_layers)
    w = 0.35

    for ci, concept in enumerate(concepts):
        active_dims = CONCEPT_PARAM_SPEC[concept][0]

        for col, metric in enumerate(["mse", "r2"]):
            ax = axes[ci, col]
            key_h = f"{metric}_high"
            key_l = f"{metric}_low"

            vals_h = [_agg(results, key_h, li, concept, active_dims) for li in layers]
            vals_l = [_agg(results, key_l, li, concept, active_dims) for li in layers]

            bars_h = ax.bar(x - w/2, vals_h, w, label="High-MI", color="steelblue", alpha=0.85)
            bars_l = ax.bar(x + w/2, vals_l, w, label="Low-MI",  color="coral",    alpha=0.85)

            if metric == "mse":
                for xi, (bh, bl) in enumerate(zip(bars_h, bars_l)):
                    lo = max(bh.get_height(), bl.get_height())
                    better = "H" if vals_h[xi] < vals_l[xi] else "L"
                    color = "steelblue" if better == "H" else "coral"
                    ax.text(xi, lo + lo * 0.015, better,
                            ha="center", va="bottom", color=color,
                            fontweight="bold", fontsize=9)
                ax.set_ylabel("MSE", fontsize=9)
                ax.set_ylim(bottom=0)
            else:
                for xi, (bh, bl) in enumerate(zip(bars_h, bars_l)):
                    hi = max(bh.get_height(), bl.get_height())
                    better = "H" if vals_h[xi] > vals_l[xi] else "L"
                    color = "steelblue" if better == "H" else "coral"
                    ax.text(xi, hi + 0.015, better,
                            ha="center", va="bottom", color=color,
                            fontweight="bold", fontsize=9)
                ax.set_ylabel("R²", fontsize=9)
                ax.set_ylim(0, 1.05)

            ax.set_xticks(x)
            ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
            ax.grid(True, alpha=0.25, axis="y")
            ax.tick_params(axis="y", labelsize=8)

            if ci == 0:
                ax.set_title(f"{'MSE' if metric == 'mse' else 'R²'}", fontsize=11, fontweight="bold")
            if ci == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

        axes[ci, 0].set_ylabel(f"{concept}\nMSE", fontsize=9)

    fig.text(0.5, 0.01, "Layer", ha="center", fontsize=10)
    fig.suptitle(
        f"Per-Concept MSE & R²: High-MI vs Low-MI Tokens (top_k={top_k})\n"
        f"blue = High-MI tokens,  coral = Low-MI tokens  |  H/L = which group wins per layer",
        fontsize=12, fontweight="bold", y=1.0,
    )
    out_path = os.path.join(output_dir, "mi_token_concept_mse_r2.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_per_concept_layer_metrics(
    results: dict,
    concepts: list[str],
    n_layers: int,
    output_dir: str,
    top_k: int,
):
    """
    Per-concept panel grid: rows = concepts, columns = layers.
    Each cell contains a grouped bar (High-MI / Low-MI) for both MSE and R²
    stacked vertically, so you can see both metrics side-by-side at a glance.

    Additionally produces two summary rows at the bottom:
      - Average MSE per layer
      - Average R² per layer
    """
    n_concepts = len(concepts)
    n_cols = n_layers
    n_rows = n_concepts + 2   # +2 for summary rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows),
                               squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    layers = list(range(n_layers))
    x = [0, 1]   # two bars per cell
    w = 0.35
    bar_labels = ["High-MI", "Low-MI"]
    bar_colors = ["steelblue", "coral"]

    summary_mse_h = []
    summary_mse_l = []
    summary_r2_h = []
    summary_r2_l = []

    for ci, concept in enumerate(concepts):
        active_dims = CONCEPT_PARAM_SPEC[concept][0]

        mse_h_vals = [_agg(results, "mse_high", li, concept, active_dims) for li in layers]
        mse_l_vals = [_agg(results, "mse_low",  li, concept, active_dims) for li in layers]
        r2_h_vals  = [_agg(results, "r2_high",   li, concept, active_dims) for li in layers]
        r2_l_vals  = [_agg(results, "r2_low",    li, concept, active_dims) for li in layers]

        summary_mse_h.append(mse_h_vals)
        summary_mse_l.append(mse_l_vals)
        summary_r2_h.append(r2_h_vals)
        summary_r2_l.append(r2_l_vals)

        for li, layer in enumerate(layers):
            ax = axes[ci, li]

            # MSE bars (top half of cell)
            mh, ml = mse_h_vals[li], mse_l_vals[li]
            ax.bar(0 - w/2, mh, w, color="steelblue", alpha=0.85)
            ax.bar(1 + w/2, ml, w, color="coral",    alpha=0.85)

            # R² bars (bottom half of cell)
            rh, rl = r2_h_vals[li], r2_l_vals[li]
            ax.bar(0 - w/2, rh, w, color="steelblue", alpha=0.5, hatch="//")
            ax.bar(1 + w/2, rl, w, color="coral",    alpha=0.5, hatch="\\\\")

            ax.axvline(0.5, color="gray", lw=0.8, ls="--")

            # Determine winners
            mse_win = "H" if mh < ml else "L"
            r2_win  = "H" if rh > rl else "L"

            # Labels
            ax.text(0, max(mh, ml) + max(mh, ml) * 0.02, mse_win,
                    ha="center", va="bottom",
                    color="steelblue" if mse_win == "H" else "coral",
                    fontsize=8, fontweight="bold")
            ax.text(1, max(rh, rl) + 0.02, r2_win,
                    ha="center", va="bottom",
                    color="steelblue" if r2_win == "H" else "coral",
                    fontsize=8, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["H", "L"], fontsize=8)
            ax.set_xlim(-0.7, 1.7)
            ax.grid(True, alpha=0.2, axis="y")
            ax.tick_params(axis="y", labelsize=7)

            if li == 0:
                ax.set_ylabel(f"{concept[:14]}", fontsize=8, fontweight="bold")
            if ci == 0:
                ax.set_title(f"Layer {li}", fontsize=9, fontweight="bold")

    # ── Summary rows ──────────────────────────────────────────────────────────
    avg_mse_h = np.mean(summary_mse_h, axis=0)
    avg_mse_l = np.mean(summary_mse_l, axis=0)
    avg_r2_h  = np.mean(summary_r2_h,  axis=0)
    avg_r2_l  = np.mean(summary_r2_l,  axis=0)

    summary_row_mse = n_concepts
    summary_row_r2  = n_concepts + 1

    for li, layer in enumerate(layers):
        # MSE summary
        ax = axes[summary_row_mse, li]
        mh, ml = avg_mse_h[li], avg_mse_l[li]
        ax.bar(0 - w/2, mh, w, color="steelblue", alpha=0.85)
        ax.bar(1 + w/2, ml, w, color="coral",    alpha=0.85)
        mse_win = "H" if mh < ml else "L"
        ax.text(0, mh + mh * 0.02, mse_win,
                ha="center", va="bottom",
                color="steelblue" if mse_win == "H" else "coral",
                fontsize=9, fontweight="bold")
        ax.text(1, ml + ml * 0.02, mse_win,
                ha="center", va="bottom",
                color="steelblue" if mse_win == "H" else "coral",
                fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["H", "L"], fontsize=8)
        ax.set_xlim(-0.7, 1.7)
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(axis="y", labelsize=7)
        if li == 0:
            ax.set_ylabel("Avg MSE\n(all concepts)", fontsize=8)
        if ci == 0:
            ax.set_title(f"Layer {li}", fontsize=9, fontweight="bold")

        # R² summary
        ax = axes[summary_row_r2, li]
        rh, rl = avg_r2_h[li], avg_r2_l[li]
        ax.bar(0 - w/2, rh, w, color="steelblue", alpha=0.85)
        ax.bar(1 + w/2, rl, w, color="coral",    alpha=0.85)
        r2_win = "H" if rh > rl else "L"
        ax.text(0, rh + 0.02, r2_win,
                ha="center", va="bottom",
                color="steelblue" if r2_win == "H" else "coral",
                fontsize=9, fontweight="bold")
        ax.text(1, rl + 0.02, r2_win,
                ha="center", va="bottom",
                color="steelblue" if r2_win == "H" else "coral",
                fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["H", "L"], fontsize=8)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(axis="y", labelsize=7)
        if li == 0:
            ax.set_ylabel("Avg R²\n(all concepts)", fontsize=8)

    # Legends at the bottom-right corner
    handles = [mpatches.Patch(color=c, label=l, alpha=0.85)
               for c, l in zip(bar_colors, bar_labels)]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.02),
               fontsize=9, title="Token Group")

    fig.text(0.5, 0.01, "H = High-MI tokens,  L = Low-MI tokens  |  "
             "solid = MSE/R² value,  winner label = which group is better",
             ha="center", fontsize=9, style="italic")

    fig.suptitle(
        f"Per-Layer & Per-Concept MSE & R²: High-MI vs Low-MI Tokens (top_k={top_k})\n"
        f"Top {n_concepts} rows: per-concept; Bottom 2 rows: averaged across all concepts",
        fontsize=11, fontweight="bold", y=1.0,
    )

    out_path = os.path.join(output_dir, "mi_token_concept_layer_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_compression_analysis(
    results: dict,
    concepts: list[str],
    n_layers: int,
    output_dir: str,
    top_k: int,
):
    """
    Evidence that later layers compress information: the gap between High-MI
    and Low-MI token probes shrinks in deeper layers.

    Three panels:
      [0]  |ΔR²| per layer  — absolute R² gap; drops in late layers = compression.
      [1]  |ΔMSE| per layer — absolute MSE gap; shrinks in late layers = compression.
      [2]  Normalised gap (|Δ| / early-layer mean); shows relative decay.

    Interpretation: if the two token groups carry similar information about
    concept parameters in late layers, the model has lost positional
    disentanglement — consistent with a compression/compression stage.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.35)

    layers = list(range(n_layers))

    # ── Per-layer gap magnitudes ─────────────────────────────────────────────
    delta_r2  = []   # R²_high - R²_low
    delta_mse = []   # MSE_low - MSE_high  (>0 means High wins)
    gap_r2    = []   # |ΔR²|
    gap_mse   = []   # |ΔMSE|

    per_concept_r2 = {c: [] for c in concepts}
    per_concept_mse = {c: [] for c in concepts}

    for li, layer in enumerate(layers):
        all_dh = [v["r2_high"]  - v["r2_low"]  for k, v in results.items() if k[0] == layer]
        all_dm = [v["mse_low"]  - v["mse_high"] for k, v in results.items() if k[0] == layer]
        delta_r2.append(np.nanmean(all_dh))
        delta_mse.append(np.nanmean(all_dm))
        gap_r2.append(np.nanmean(np.abs(all_dh)))
        gap_mse.append(np.nanmean(np.abs(all_dm)))

        for ci, concept in enumerate(concepts):
            active_dims = CONCEPT_PARAM_SPEC[concept][0]
            vals_h = [results.get((layer, concept, d), {}).get("r2_high",  np.nan) for d in range(4)]
            vals_l = [results.get((layer, concept, d), {}).get("r2_low",   np.nan) for d in range(4)]
            mvals_h = [results.get((layer, concept, d), {}).get("mse_high", np.nan) for d in range(4)]
            mvals_l = [results.get((layer, concept, d), {}).get("mse_low",  np.nan) for d in range(4)]
            r2_diff = [vals_h[d] - vals_l[d] for d in active_dims if np.isfinite(vals_h[d]) and np.isfinite(vals_l[d])]
            mse_diff = [mvals_l[d] - mvals_h[d] for d in active_dims if np.isfinite(mvals_l[d]) and np.isfinite(mvals_h[d])]
            per_concept_r2[concept].append(np.nanmean(r2_diff) if r2_diff else np.nan)
            per_concept_mse[concept].append(np.nanmean(mse_diff) if mse_diff else np.nan)

    # ── Panel 0: |ΔR²| per layer ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(layers, gap_r2, "o-", color="steelblue", lw=2, ms=8, label="Gap = |R²_high − R²_low|")

    # Shade region above/below zero line
    ax.fill_between(layers, 0, gap_r2, alpha=0.15, color="steelblue")
    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("|ΔR²|  (High-MI − Low-MI)", fontsize=10)
    ax.set_title("|ΔR²| per Layer\nSmaller gap → compression / disentanglement collapse", fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    # Annotate early vs late gap
    early_gap = np.mean(gap_r2[:max(1, n_layers // 3)])
    late_gap  = np.mean(gap_r2[2 * n_layers // 3:])
    ax.text(0.05, 0.95,
            f"Early avg: {early_gap:.4f}\nLate avg:  {late_gap:.4f}\nRatio: {late_gap/max(early_gap,1e-8):.2f}x",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ── Panel 1: |ΔMSE| per layer ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(layers, gap_mse, "s-", color="coral", lw=2, ms=8, label="Gap = |MSE_low − MSE_high|")

    ax.fill_between(layers, 0, gap_mse, alpha=0.15, color="coral")
    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("|ΔMSE|  (Low − High)", fontsize=10)
    ax.set_title("|ΔMSE| per Layer\nSmaller gap → compression / disentanglement collapse", fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    early_gap_m = np.mean(gap_mse[:max(1, n_layers // 3)])
    late_gap_m  = np.mean(gap_mse[2 * n_layers // 3:])
    ax.text(0.05, 0.95,
            f"Early avg: {early_gap_m:.4f}\nLate avg:  {late_gap_m:.4f}\nRatio: {late_gap_m/max(early_gap_m,1e-8):.2f}x",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ── Panel 2: Per-concept ΔR² line plot ────────────────────────────────────
    ax = axes[2]

    cmap = plt.cm.tab10.colors
    for ci, concept in enumerate(concepts):
        vals = per_concept_r2[concept]
        ax.plot(layers, vals, "o-", color=cmap[ci % 10],
                lw=1.5, ms=5, label=concept[:18], alpha=0.8)

    ax.axhline(0, color="black", ls="--", lw=1.2, label="no gap")
    ax.fill_between(layers,
                    [0] * n_layers,
                    [min(np.nanmean([per_concept_r2[c][l] for c in concepts]) for l in layers)],
                    alpha=0.08, color="steelblue")

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("ΔR²  (High-MI − Low-MI)", fontsize=10)
    ax.set_title("Per-Concept ΔR² across Layers\nPositive = High-MI wins; →0 = compression", fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")

    fig.suptitle(
        f"Compression Analysis: High-MI vs Low-MI Gap by Layer (top_k={top_k})\n"
        f"If later layers compress, gap should shrink toward zero",
        fontsize=12, fontweight="bold",
    )

    out_path = os.path.join(output_dir, "mi_token_compression_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Within-Layer Analysis: top/bottom percentile token probe
# ─────────────────────────────────────────────────────────────────────────────

def run_within_layer_percentile_analysis(
    layer_tokens: list[torch.Tensor],
    params: torch.Tensor,
    concept_idx: torch.Tensor,
    labels: list[dict],
    concepts: list[str],
    n_samples_per_concept: int,
    d_model: int,
    n_layers: int,
    output_dir: str,
    percentile: float = 10.0,
    max_samples: int = None,
    probe_epochs: int = 200,
    probe_lr: float = 1e-3,
    device: torch.device = None,
    seed: int = 42,
):
    """
    Within-layer analysis: for each layer, split all tokens into top-N%% and
    bottom-N%% groups by HSIC score, then compare linear probe performance.

    This answers: within a single layer, do high-MI tokens systematically
    encode more concept information than low-MI tokens?
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    N, N_patches, D = layer_tokens[0].shape
    top_n = max(1, int(np.ceil(N_patches * percentile / 100.0)))

    results = {}
    print(f"\n{'Layer':>5} | {'Concept':<25} | {'Param':<12} | "
          f"{'HSIC_top':>9} | {'HSIC_bot':>9} | "
          f"{'MSE_top':>10} | {'MSE_bot':>10} | "
          f"{'R2_top':>8} | {'R2_bot':>8} | {'Better':>6}")
    print("-" * 140)

    for layer in range(n_layers):
        tok_rep = layer_tokens[layer]

        for ci, concept in enumerate(concepts):
            mask = concept_idx == ci
            concept_tok = tok_rep[mask].contiguous()
            concept_params = params[mask]

            active_dims, _ = CONCEPT_PARAM_SPEC[concept]
            n_c = concept_tok.shape[0]

            if max_samples is not None and n_c > max_samples:
                idx = np.random.choice(n_c, max_samples, replace=False)
                concept_tok = concept_tok[idx]
                concept_params = concept_params[idx]
                n_c = max_samples

            if n_c < 20:
                continue

            for dim_idx in active_dims:
                target = concept_params[:, dim_idx]

                hsic_scores = compute_token_mi(concept_tok, target)
                sorted_pos = np.argsort(hsic_scores)[::-1]

                top_pos = sorted_pos[:top_n].copy()
                bot_pos = sorted_pos[-top_n:].copy()

                top_rep = concept_tok[:, top_pos, :].contiguous().mean(dim=1)
                bot_rep = concept_tok[:, bot_pos, :].contiguous().mean(dim=1)

                mse_top, r2_top = train_probe(top_rep, target, probe_epochs, probe_lr, device, seed)
                mse_bot, r2_bot = train_probe(bot_rep, target, probe_epochs, probe_lr, device, seed)

                delta_mse = mse_bot - mse_top
                better = "TOP" if delta_mse > 0 else "BOT"

                param_name = CONCEPT_PARAM_SPEC[concept][1].split(",")[dim_idx].strip()
                hsic_t = hsic_scores[top_pos].mean()
                hsic_b = hsic_scores[bot_pos].mean()

                results[(layer, concept, dim_idx)] = {
                    "hsic_top": hsic_t,
                    "hsic_bot": hsic_b,
                    "mse_top": mse_top,
                    "mse_bot": mse_bot,
                    "r2_top": r2_top,
                    "r2_bot": r2_bot,
                    "delta_mse": delta_mse,
                    "top_n": top_n,
                    "bottom_n": top_n,
                    "percentile": percentile,
                }

                print(f"{layer:>5} | {concept:<25} | {param_name:<12} | "
                      f"{hsic_t:>9.4f} | {hsic_b:>9.4f} | "
                      f"{mse_top:>10.6f} | {mse_bot:>10.6f} | "
                      f"{r2_top:>8.4f} | {r2_bot:>8.4f} | {better:>6}")

    return results


def plot_within_layer_percentile(
    results: dict,
    concepts: list[str],
    n_layers: int,
    output_dir: str,
    percentile: float,
):
    """Plot within-layer percentile analysis: top/bottom N% token comparison."""
    n_concepts = len(concepts)
    layers = list(range(n_layers))
    x = np.arange(n_layers)
    w = 0.35

    mse_top_all, mse_bot_all = [], []
    r2_top_all, r2_bot_all = [], []
    win_rates = []

    for li, layer in enumerate(layers):
        vt = [v["mse_top"] for k, v in results.items() if k[0] == layer]
        vb = [v["mse_bot"] for k, v in results.items() if k[0] == layer]
        mse_top_all.append(np.nanmean(vt))
        mse_bot_all.append(np.nanmean(vb))
        rt = [v["r2_top"] for k, v in results.items() if k[0] == layer]
        rb = [v["r2_bot"] for k, v in results.items() if k[0] == layer]
        r2_top_all.append(np.nanmean(rt))
        r2_bot_all.append(np.nanmean(rb))

        wins = sum(1 for k, v in results.items() if k[0] == layer and v["delta_mse"] > 0)
        total = sum(1 for k, v in results.items() if k[0] == layer)
        win_rates.append(wins / max(total, 1))

    def agg(key, layer, concept):
        active_dims = CONCEPT_PARAM_SPEC[concept][0]
        vals = [results.get((layer, concept, d), {}).get(key, np.nan) for d in range(4)]
        return float(np.nanmean([vals[d] for d in active_dims if not np.isnan(vals[d])]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    ax = axes[0, 0]
    ax.bar(x - w/2, mse_top_all, w, label=f"Top {percentile:.0f}% tokens", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, mse_bot_all, w, label=f"Bottom {percentile:.0f}% tokens", color="coral", alpha=0.85)
    for xi in range(len(x)):
        lo = max(mse_top_all[xi], mse_bot_all[xi])
        b = "H" if mse_top_all[xi] < mse_bot_all[xi] else "B"
        ax.text(xi, lo + lo * 0.01, b, ha="center", va="bottom",
                color="steelblue" if b == "H" else "coral", fontsize=9, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("MSE (mean)")
    ax.set_title(f"Per-Layer MSE: Top {percentile:.0f}% vs Bottom {percentile:.0f}% Tokens by HSIC")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(bottom=0)

    ax = axes[0, 1]
    ax.bar(x - w/2, r2_top_all, w, label=f"Top {percentile:.0f}%", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, r2_bot_all, w, label=f"Bottom {percentile:.0f}%", color="coral", alpha=0.85)
    for xi in range(len(x)):
        hi = max(r2_top_all[xi], r2_bot_all[xi])
        b = "H" if r2_top_all[xi] > r2_bot_all[xi] else "B"
        ax.text(xi, hi + 0.01, b, ha="center", va="bottom",
                color="steelblue" if b == "H" else "coral", fontsize=9, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (mean)")
    ax.set_title(f"Per-Layer R²: Top {percentile:.0f}% vs Bottom {percentile:.0f}% Tokens by HSIC")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(0, 1.05)

    ax = axes[0, 2]
    colors_wr = ["steelblue" if r > 0.5 else "coral" for r in win_rates]
    bars = ax.bar(x, win_rates, color=colors_wr, alpha=0.85)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, label="random baseline")
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wr:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"Win Rate: Top-{percentile:.0f}% Tokens Have Lower MSE\n(>0.5 = Top group better)")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    cmap = plt.cm.tab10.colors
    for ci, concept in enumerate(concepts):
        r2t = [agg("r2_top", li, concept) for li in layers]
        ax.plot(layers, r2t, "o-", color=cmap[ci % 10], lw=1.5, ms=5,
                label=f"{concept[:18]}", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title(f"Per-Concept R² across Layers (Top-{percentile:.0f}% tokens)")
    ax.set_xticks(layers)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(x, r2_top_all, "o-", color="steelblue", lw=2, ms=7, label=f"Top {percentile:.0f}%")
    ax.plot(x, r2_bot_all, "s--", color="coral", lw=2, ms=7, label=f"Bottom {percentile:.0f}%")
    ax.fill_between(x, r2_top_all, r2_bot_all,
                    where=[a > b for a, b in zip(r2_top_all, r2_bot_all)],
                    alpha=0.12, color="steelblue", label="Top wins")
    ax.fill_between(x, r2_top_all, r2_bot_all,
                    where=[a <= b for a, b in zip(r2_top_all, r2_bot_all)],
                    alpha=0.12, color="coral", label="Bottom wins")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg R²")
    ax.set_title("Per-Layer R²: Top vs Bottom Tokens (avg over concepts)")
    ax.set_xticks(x)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 2]
    hsic_t_avg = [np.nanmean([v["hsic_top"] for k, v in results.items() if k[0] == li])
                  for li in layers]
    hsic_b_avg = [np.nanmean([v["hsic_bot"] for k, v in results.items() if k[0] == li])
                  for li in layers]

    if results:
        top_n = results[(0, concepts[0], list(CONCEPT_PARAM_SPEC[concepts[0]][0])[0])]["top_n"]
    else:
        top_n = 1

    ax.bar(x - w/2, hsic_t_avg, w, label=f"Top {percentile:.0f}%", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, hsic_b_avg, w, label=f"Bottom {percentile:.0f}%", color="coral", alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg HSIC")
    ax.set_title(f"Avg HSIC of Selected Tokens\n(top_n={top_n}, bot_n={top_n})")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle(
        f"Within-Layer Percentile: Top {percentile:.0f}% vs Bottom {percentile:.0f}% HSIC Tokens\n"
        f"top_n={top_n} | {n_concepts} concepts, {n_layers} layers",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(output_dir, f"within_layer_percentile_{percentile:.0f}pct.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Entropy: Matrix-based entropy per layer (Layer-by-Layer paper)
# ─────────────────────────────────────────────────────────────────────────────

def matrix_based_entropy(gram: np.ndarray, alpha: float = 1.0) -> float:
    """
    Matrix-based Rényi entropy (Giraldo et al., 2014 / Skean et al., 2025).

    S_alpha(Z) = (1 / (1 - alpha)) * log_2( sum_i (lambda_i / tr(K))^alpha )
    For alpha=1: Shannon entropy of eigenvalue distribution.

    Args:
        gram: Gram matrix K = Z Z^T, shape [N, N]
        alpha: Rényi order (default 1.0 = Shannon)
    Returns:
        entropy in bits
    """
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = eigvals[eigvals > 1e-12]
    tr_k = np.sum(eigvals)
    if tr_k < 1e-12:
        return 0.0
    p = eigvals / tr_k
    if abs(alpha - 1.0) < 1e-6:
        p_safe = np.where(p > 0, p, 1e-12)
        return float(-np.sum(p_safe * np.log2(p_safe)))
    else:
        return float((1.0 / (1.0 - alpha)) * np.log2(np.sum(p ** alpha)))


def compute_feature_entropy_per_layer(
    layer_tokens: list[torch.Tensor],
    concepts: list[str],
    concept_idx: torch.Tensor,
    n_layers: int,
    alpha: float = 1.0,
    max_samples_per_concept: int = 200,
    output_dir: str = ".",
) -> dict:
    """
    Compute matrix-based feature entropy per layer, per concept.
    Inspired by Skean et al., 2025 "Layer by Layer" paper.

    For each layer and concept:
        Z = tokens [n_c, N_patches, D] -> [n_c * N_patches, D]
        K = Z Z^T
        S_alpha(K) = matrix-based entropy
    """
    results = {
        "per_layer": {},
        "per_layer_concept": {},
        "alpha": alpha,
    }

    for li in range(n_layers):
        tok = layer_tokens[li].numpy()
        overall_ents = []

        per_concept_ent = {}
        for ci, concept in enumerate(concepts):
            mask = concept_idx == ci
            concept_tok = tok[mask]

            if max_samples_per_concept and concept_tok.shape[0] > max_samples_per_concept:
                idx = np.random.choice(concept_tok.shape[0], max_samples_per_concept, replace=False)
                concept_tok = concept_tok[idx]

            if concept_tok.shape[0] < 10:
                per_concept_ent[concept] = np.nan
                continue

            Z_flat = concept_tok.reshape(-1, concept_tok.shape[-1])
            if Z_flat.shape[0] > Z_flat.shape[1]:
                gram = Z_flat @ Z_flat.T
                ent = matrix_based_entropy(gram, alpha=alpha)
            else:
                Z_c = Z_flat - Z_flat.mean(axis=0, keepdims=True)
                cov = Z_c.T @ Z_c / (Z_c.shape[0] - 1)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = eigvals[eigvals > 1e-12]
                tr_cov = np.sum(eigvals)
                if tr_cov < 1e-12:
                    per_concept_ent[concept] = np.nan
                    continue
                p = eigvals / tr_cov
                if abs(alpha - 1.0) < 1e-6:
                    p_safe = np.where(p > 0, p, 1e-12)
                    ent = float(-np.sum(p_safe * np.log2(p_safe)))
                else:
                    ent = float((1.0 / (1.0 - alpha)) * np.log2(np.sum(p ** alpha)))

            per_concept_ent[concept] = ent
            overall_ents.append(ent)

        overall_ent = float(np.nanmean(overall_ents)) if overall_ents else np.nan
        results["per_layer"][li] = {"overall": overall_ent, "per_concept": per_concept_ent}
        for concept, ent in per_concept_ent.items():
            results["per_layer_concept"][(li, concept)] = ent

    return results


def plot_feature_entropy(
    entropy_results: dict,
    concepts: list[str],
    n_layers: int,
    output_dir: str,
    alpha: float = 1.0,
):
    """
    Plot feature entropy per layer, mimicking Layer-by-Layer paper style.
    Three panels:
      [0]  Line plot: overall + per-concept entropy
      [1]  Heatmap: layer × concept
      [2]  Stacked bar: per-layer entropy breakdown
    Plus one extra: normalized entropy (÷ D=1024).
    """
    per_layer = entropy_results["per_layer"]
    layers = list(range(n_layers))
    n_concepts = len(concepts)

    overall_ents = [per_layer[li]["overall"] for li in layers]
    per_concept_ents = {c: [] for c in concepts}
    for li in layers:
        for c in concepts:
            per_concept_ents[c].append(per_layer[li]["per_concept"].get(c, np.nan))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.35)
    cmap = plt.cm.tab10.colors

    ax = axes[0]
    ax.plot(layers, overall_ents, "o-", color="black", lw=2.5, ms=8,
            label="Overall (mean)", zorder=5)
    for ci, concept in enumerate(concepts):
        vals = per_concept_ents[concept]
        ax.plot(layers, vals, "s--", color=cmap[ci % 10], lw=1.2, ms=5,
                label=concept[:18], alpha=0.75)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Matrix Entropy S_{alpha}(Z) [bits]")
    ax.set_title("Feature Entropy per Layer\n(Matrix-based, per concept)")
    ax.set_xticks(layers)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    mat = np.array([[per_layer[li]["per_concept"].get(c, np.nan) for c in concepts]
                    for li in layers])
    v_abs = max(0.1, np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=v_abs)
    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels([c[:16] for c in concepts], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylabel("Layer")
    ax.set_title(f"Entropy Heatmap: Layer × Concept\n(S_{alpha} [bits])")
    plt.colorbar(im, ax=ax, label="Entropy [bits]")
    for li in range(n_layers):
        for ci in range(n_concepts):
            val = mat[li, ci]
            if np.isnan(val):
                continue
            ax.text(ci, li, f"{val:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white" if val > v_abs * 0.6 else "black")

    ax = axes[2]
    bottom = np.zeros(n_layers)
    for ci, concept in enumerate(concepts):
        vals_arr = np.array([v if not np.isnan(v) else 0 for v in per_concept_ents[concept]])
        ax.bar(layers, vals_arr, bottom=bottom, label=concept[:18],
               color=cmap[ci % 10], alpha=0.85)
        bottom += vals_arr
    ax.plot(layers, overall_ents, "o-", color="black", lw=2, ms=7,
            label="Overall mean", zorder=5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Feature Entropy [bits]")
    ax.set_title("Per-Layer Entropy Breakdown\n(stacked by concept)")
    ax.set_xticks(layers)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle(
        f"Matrix-Based Feature Entropy per Layer (alpha={alpha})\n"
        f"Timer on Synthetic Dataset | {n_concepts} concepts, {n_layers} layers",
        fontsize=12, fontweight="bold",
    )
    out_path = os.path.join(output_dir, "feature_entropy_per_layer.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    norm_ents = [e / 1024.0 for e in overall_ents]
    ax2.plot(layers, norm_ents, "o-", color="steelblue", lw=2, ms=8)
    ax2.fill_between(layers, 0, norm_ents, alpha=0.15, color="steelblue")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Normalized Entropy (÷ D=1024)")
    ax2.set_title("Normalized Feature Entropy per Layer\n(Higher = more diverse / less compressed)")
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.25)
    if n_layers > 1:
        dip_layer = int(np.argmin(overall_ents))
        ax2.annotate(f"Valley @ L{dip_layer}\n({overall_ents[dip_layer]:.1f} bits)",
                     xy=(dip_layer, norm_ents[dip_layer]),
                     xytext=(dip_layer + 1, norm_ents[dip_layer] + 0.05),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     fontsize=9, color="red")
    out_path2 = os.path.join(output_dir, "feature_entropy_normalized.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path2}")



def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Token-Level MI Analysis - High vs Low MI Token Probe"
    )
    parser.add_argument("--rep_dir", type=str, required=False,
                        default="./results/synthetic/representations/",
                        help="Directory with layer_token_representations.pt from Step 2")
    parser.add_argument("--output_dir", type=str, default="./results/synthetic/mi_token_analysis/",
                        help="Output directory")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Number of top/bottom token positions to select (per group)")
    parser.add_argument("--percentile", type=float, default=10.0,
                        help="Percentile for within-layer analysis: top/bottom N%% tokens (default: 10.0)")
    parser.add_argument("--within_layer", action="store_true",
                        help="Run within-layer percentile analysis (top vs bottom percentile tokens)")
    parser.add_argument("--feature_entropy", action="store_true",
                        help="Compute matrix-based feature entropy per layer (Layer-by-Layer paper)")
    parser.add_argument("--entropy_alpha", type=float, default=1.0,
                        help="Renyi entropy alpha (default: 1.0 = Shannon)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per concept (default: all)")
    parser.add_argument("--probe_epochs", type=int, default=200,
                        help="Epochs for linear probe training")
    parser.add_argument("--probe_lr", type=float, default=1e-3,
                        help="Learning rate for probe")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Specific layers to analyze (default: all)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("Step 4: Token-Level MI Analysis")
    print("=" * 70)
    print(f"  rep_dir       : {args.rep_dir}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  top_k        : {args.top_k}")
    print(f"  percentile   : {args.percentile}")
    print(f"  within_layer   : {args.within_layer}")
    print(f"  feature_entropy: {args.feature_entropy}")
    print(f"  entropy_alpha  : {args.entropy_alpha}")
    print(f"  probe_epochs : {args.probe_epochs}")
    print(f"  probe_lr     : {args.probe_lr}")
    print(f"  device       : {device}")
    print("=" * 70)

    # ── Load token representations (shared by all analysis modes) ───────────────
    print("\n[0] Loading token representations...")
    token_path = os.path.join(args.rep_dir, "layer_token_representations.pt")
    if not os.path.exists(token_path):
        print(f"\nERROR: Token representations not found at {token_path}")
        print("Please re-run Step 2 with --save_token_reps flag.")
        return

    ckpt = torch.load(token_path, map_location="cpu", weights_only=False)
    layer_tokens = ckpt["layer_tokens"]
    params = ckpt["params"]
    concept_idx = ckpt["concept_idx"]
    labels = ckpt["labels"]
    concepts = ckpt["concepts"]
    n_samples_per_concept = ckpt["n_samples_per_concept"]
    d_model = ckpt["d_model"]
    n_layers_total = ckpt["n_layers"]

    print(f"  Loaded: {n_layers_total} layers, shape: {layer_tokens[0].shape}")
    print(f"  Concepts: {concepts}")

    if args.layers is not None:
        layer_tokens = [layer_tokens[l] for l in args.layers if l < n_layers_total]
        n_layers = len(layer_tokens)
    else:
        n_layers = n_layers_total

    # ── Mode dispatch ───────────────────────────────────────────────────────────
    if args.within_layer:
        print(f"\n[Within-Layer] Percentile={args.percentile}%, epochs={args.probe_epochs}")
        results_wl = run_within_layer_percentile_analysis(
            layer_tokens=layer_tokens,
            params=params,
            concept_idx=concept_idx,
            labels=labels,
            concepts=concepts,
            n_samples_per_concept=n_samples_per_concept,
            d_model=d_model,
            n_layers=n_layers,
            output_dir=args.output_dir,
            percentile=args.percentile,
            max_samples=args.max_samples,
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr,
            device=device,
            seed=args.seed,
        )
        out_path = os.path.join(args.output_dir, f"within_layer_results_{args.percentile:.0f}pct.pt")
        torch.save({"results": results_wl, "percentile": args.percentile, "config": vars(args)}, out_path)
        print(f"Saved: {out_path}")

        if not args.no_plot:
            plot_within_layer_percentile(results_wl, concepts, n_layers, args.output_dir, args.percentile)

        print("\n[Done] Within-layer analysis complete.")

    elif args.feature_entropy:
        print(f"\n[Feature Entropy] alpha={args.entropy_alpha}")
        entropy_results = compute_feature_entropy_per_layer(
            layer_tokens=layer_tokens,
            concepts=concepts,
            concept_idx=concept_idx,
            n_layers=n_layers,
            alpha=args.entropy_alpha,
            max_samples_per_concept=200,
            output_dir=args.output_dir,
        )
        out_path = os.path.join(args.output_dir, "feature_entropy_results.pt")
        torch.save({"entropy_results": entropy_results, "config": vars(args)}, out_path)
        print(f"Saved: {out_path}")

        if not args.no_plot:
            plot_feature_entropy(entropy_results, concepts, n_layers, args.output_dir, alpha=args.entropy_alpha)

        print("\nPer-layer entropy summary:")
        for li in range(n_layers):
            ent = entropy_results["per_layer"][li]["overall"]
            print(f"  Layer {li:2d}: overall entropy = {ent:.4f} bits")

        print("\n[Done] Feature entropy analysis complete.")

    else:
        # Original full analysis (between-layer, high vs low HSIC token positions)
        print(f"\n[Between-Layer] top_k={args.top_k}")
        results = run_mi_token_analysis(
            layer_tokens=layer_tokens,
            params=params,
            concept_idx=concept_idx,
            labels=labels,
            concepts=concepts,
            n_samples_per_concept=n_samples_per_concept,
            d_model=d_model,
            n_layers=n_layers,
            output_dir=args.output_dir,
            top_k=args.top_k,
            max_samples=args.max_samples,
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr,
            device=device,
            seed=args.seed,
        )

        results_path = os.path.join(args.output_dir, "mi_token_results.pt")
        torch.save(
            {
                "results": results,
                "concepts": concepts,
                "n_layers": n_layers,
                "top_k": args.top_k,
                "config": vars(args),
            },
            results_path,
        )
        print(f"Saved: {results_path}")

        print("=" * 70)
        print("SUMMARY: High-HSIC vs Low-HSIC Token Probe")
        print("=" * 70)
        wins_high = sum(1 for v in results.values() if v["delta_mse"] > 0)
        total = len(results)
        print(f"  High-HSIC WIN: {wins_high}/{total} ({100*wins_high/total:.1f}%)")
        print(f"  Low-HSIC WIN:  {total - wins_high}/{total} ({100*(total-wins_high)/total:.1f}%)")

        if not args.no_plot:
            print("\n[Plotting...]")
            plot_results(results, concepts, n_layers, args.output_dir, args.top_k)
            plot_mi_profile(results, concepts, n_layers, args.output_dir)
            plot_concept_mse_r2_comparison(results, concepts, n_layers, args.output_dir, args.top_k)
            plot_per_concept_layer_metrics(results, concepts, n_layers, args.output_dir, args.top_k)
            plot_compression_analysis(results, concepts, n_layers, args.output_dir, args.top_k)

        print(f"\n[Done] Step 4 complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
