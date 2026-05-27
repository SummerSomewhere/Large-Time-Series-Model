#!/usr/bin/env python3
"""
Timer MI-Guided Hidden-State Token Pruning Experiment

Pruning method: zero hidden states for low-MI tokens
- For the target layer(s), set the hidden states of low-MI tokens to the
  zero vector, so their information is completely removed from the residual
  stream before they flow into subsequent layers
- This is fundamentally different from attention masking (which only removes
  cross-token communication) — hidden-state pruning also cuts the residual
  shortcut path that carries token identity through the transformer
- Sequence length is unchanged — soft mask, not token deletion

Experiments:
  - Mode 1: per-layer pruning (one layer at a time)
  - Mode 2: all-layer pruning (all layers masked simultaneously)

Usage:
    python experiments/timer_mi_hidden_pruning_experiment.py \
        --mi_dir ./timer_mi_ksg_pca/Timer_MI_20260514_073311/ \
        --data_path ./datasets/ETTh1.csv \
        --seq_len 672 --pred_len 96 --prune_layer 6 \
        --output_dir ./results/timer_mi_hidden_pruning/
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_loader_benchmark import CIDatasetBenchmark


# ─────────────────────────────────────────────────────────────────────────────
# Config & Model
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer_model(ckpt_path: str, patch_len: int, stride: int,
                      d_model: int, d_ff: int, e_layers: int,
                      n_heads: int, dropout: float,
                      seq_len: int, pred_len: int):
    config = Config(
        task_name='forecast', ckpt_path=ckpt_path,
        patch_len=patch_len, stride=stride,
        d_model=d_model, d_ff=d_ff, e_layers=e_layers,
        n_heads=n_heads, dropout=dropout,
        output_attention=False, distil=True, use_revin=False,
        seq_len=seq_len, pred_len=pred_len,
        d_layers=1, factor=1, enc_in=1, dec_in=1, c_out=1,
        activation='gelu', use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(), devices='0',
        num_workers=4, freq='h', data='custom',
        embed='timeF', target='OT', features='M',
        des='Exp', lradj='type1', use_amp=False,
        is_finetuning=0, inverse=False,
        use_align_loss=False,
        align_loss_layers=list(range(e_layers)),
        label_len=pred_len, output_len=pred_len,
        batch_size=64, train_epochs=1, patience=3,
        learning_rate=3e-5, itr=1, use_ims=False,
    )
    from models.Timer import Model
    model = Model(config)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trues, preds):
    """Return dict of MSE, MAE, RMSE. NaN-safe."""
    mse = float(np.nanmean((trues - preds) ** 2))
    mae = float(np.nanmean(np.abs(trues - preds)))
    return {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse)}


def align_pred_shape(pred, true):
    """Ensure prediction shape matches ground truth."""
    if pred.shape != true.shape and pred.shape[-1] == true.shape[-1] and pred.shape[1] == true.shape[-1]:
        return np.transpose(pred, (0, 2, 1))
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Inference Functions
# ─────────────────────────────────────────────────────────────────────────────

def _run_hidden_pruning_layer(layer_module, dec_in, causal_mask,
                              mi_vec, keep_ratio, device, debug=False):
    """
    Compute a single decoder layer AND zero the hidden states of low-MI tokens.

    The key difference from attention masking:
      - Attention masking only removes cross-token communication (soft prune)
      - Hidden-state pruning sets entire token embeddings to zero, cutting both
        the residual shortcut AND the attention path

    Args:
        layer_module:  the decoder layer (EncoderLayer)
        dec_in:       [B*M, N, d_model] — input hidden states
        causal_mask:  TriangularCausalMask object
        mi_vec:       [N,] per-patch MI scores for this layer
        keep_ratio:   fraction of tokens to keep (0–1)
        device:       torch device

    Returns:
        dec_in after the layer and after zeroing low-MI token positions
    """
    B_l, N, D = dec_in.shape

    # ── Step 1: Zero low-MI token hidden states BEFORE entering the layer ──
    # This is the core of hidden-state pruning: the pruned tokens contribute
    # nothing — neither to attention (zero key/value) nor to the residual stream
    keep_k = max(1, int(N * keep_ratio))
    sorted_idx = np.argsort(mi_vec)[::-1][:keep_k]

    prune_mask_np = np.ones(N, dtype=np.bool_)   # True = prune (zero), False = keep
    prune_mask_np[sorted_idx] = False            # keep tokens → False
    prune_mask = torch.tensor(prune_mask_np, device=device, dtype=torch.bool)

    dec_in_pruned = dec_in.clone()
    dec_in_pruned[:, prune_mask, :] = 0.0        # zero the hidden states

    # ── Step 2: Run the decoder layer with zeroed tokens ────────────────────
    dec_out, attn, logits = layer_module(
        dec_in_pruned, attn_mask=causal_mask)

    return dec_out


def forecast_with_hidden_pruning(model, test_loader, device, mi_matrix,
                                  n_patches, keep_ratio=1.0,
                                  prune_layer=None, pred_len=96):
    """
    Inference with per-layer MI-guided hidden-state pruning.

    For each pruned layer:
      1. Clone the hidden states
      2. Zero the entries corresponding to low-MI tokens
      3. Run the decoder layer with the zeroed input
      4. Continue with normal forward for remaining layers

    mi_matrix:  [n_layers, n_patches]
    keep_ratio: fraction of highest-MI tokens to retain
    prune_layer: None = all layers, int = only that specific layer
    """
    core = _unwrap(model)

    preds_list = []
    inf_times = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
                test_loader, desc=f"Keep={keep_ratio*100:.0f}%", leave=False):

            B = batch_x.shape[0]

            seq_x = batch_x.float().to(device)
            seq_y = batch_y.float().to(device)
            bx_mark = (batch_x_mark.float().to(device)
                       if batch_x_mark is not None else None)
            by_mark = (batch_y_mark.float().to(device)
                       if batch_y_mark is not None else None)

            means = seq_x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_norm = (seq_x - means) / stdev

            x2 = x_norm.permute(0, 2, 1)
            dec_in, n_vars = core.enc_embedding(x2)

            BM, N, D = dec_in.shape
            B_times_vars = B * n_vars
            dec_in = dec_in.view(B_times_vars, N, D)

            from utils.masking import TriangularCausalMask
            causal_mask = TriangularCausalMask(B_times_vars, N, device=device)

            start_time = time.time()

            if keep_ratio < 1.0:
                # ── Pruned forward pass ──────────────────────────────────────
                if prune_layer is None:
                    # Mode 2: ALL layers receive hidden-state pruning
                    for li, layer_mod in enumerate(core.decoder.attn_layers):
                        dec_in = _run_hidden_pruning_layer(
                            layer_mod, dec_in, causal_mask,
                            mi_matrix[li], keep_ratio, device)

                    if core.decoder.norm is not None:
                        dec_in = core.decoder.norm(dec_in)
                else:
                    # Mode 1: only ONE layer receives hidden-state pruning
                    # Layers BEFORE the pruned layer: normal forward
                    for li, layer_mod in enumerate(core.decoder.attn_layers):
                        if li == prune_layer:
                            break
                        dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)

                    # The pruned layer: hidden-state zeroing
                    dec_in = _run_hidden_pruning_layer(
                        core.decoder.attn_layers[prune_layer],
                        dec_in, causal_mask,
                        mi_matrix[prune_layer], keep_ratio, device)

                    # Layers AFTER the pruned layer: normal forward
                    for li, layer_mod in enumerate(core.decoder.attn_layers):
                        if li <= prune_layer:
                            continue
                        dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)

                    if core.decoder.norm is not None:
                        dec_in = core.decoder.norm(dec_in)
            else:
                # ── Baseline (no pruning) ────────────────────────────────────
                for layer_mod in core.decoder.attn_layers:
                    dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)
                if core.decoder.norm is not None:
                    dec_in = core.decoder.norm(dec_in)

            if core.proj is not None:
                dec_in = core.proj(dec_in)

            dec_out = dec_in.view(B, n_vars, N, core.patch_len)
            dec_out = dec_out.mean(dim=1)
            dec_out = dec_out.reshape(B, n_vars, -1).transpose(1, 2)
            dec_out = dec_out[:, -pred_len:, :]
            dec_out = dec_out * stdev + means

            inf_times.append(time.time() - start_time)
            preds_list.append(dec_out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    return preds, np.mean(inf_times), np.sum(inf_times)


def forecast_baseline(model, test_loader, device, pred_len=96):
    """Standard forward pass — no pruning."""
    core = _unwrap(model)

    preds_list = []
    inf_times = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
                test_loader, desc="Baseline", leave=False):

            seq_x = batch_x.float().to(device)
            means = seq_x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_norm = (seq_x - means) / stdev

            x2 = x_norm.permute(0, 2, 1)
            dec_in, n_vars = core.enc_embedding(x2)

            BM, N, D = dec_in.shape

            from utils.masking import TriangularCausalMask
            causal_mask = TriangularCausalMask(BM, N, device=device)

            start_time = time.time()

            for layer_mod in core.decoder.attn_layers:
                dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)

            if core.decoder.norm is not None:
                dec_in = core.decoder.norm(dec_in)

            dec_out = core.proj(dec_in)

            dec_out = dec_out.view(BM // n_vars, n_vars, N, core.patch_len)
            dec_out = dec_out.mean(dim=1)
            dec_out = dec_out.reshape(BM // n_vars, n_vars, -1).transpose(1, 2)
            dec_out = dec_out[:, -pred_len:, :]
            dec_out = dec_out * stdev + means

            inf_times.append(time.time() - start_time)
            preds_list.append(dec_out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    return preds, np.mean(inf_times), np.sum(inf_times)


# ─────────────────────────────────────────────────────────────────────────────
# Nature-Style Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _nature_rc():
    """Nature journal figure defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _get_layer_colors(n_layers):
    """Colorblind-safe palette from Okabe-Ito."""
    palette = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999",
        "#F4A261", "#E76F51", "#264653", "#2A9D8F",
    ]
    return [palette[i % len(palette)] for i in range(n_layers)]


def _get_degradation_category(delta):
    """Classify MSE degradation severity."""
    if np.isnan(delta):
        return "nan"
    if delta <= 1:
        return "negligible"
    elif delta <= 5:
        return "moderate"
    elif delta <= 10:
        return "significant"
    else:
        return "severe"


def plot_pruning_results(results, output_dir, mask_ratios, layers_to_prune,
                          baseline_mse, baseline_time, dataset_name=""):
    """
    Generate Nature-style publication figures for hidden-state pruning.

    Generates 4 figures:
      fig1  — (a) MSE vs mask ratio  (b) MSE change % vs mask ratio
      fig2  — (a) Per-layer sensitivity bar chart  (b) all-layer MSE curve
      fig3  — Heatmap of MSE change per layer × mask ratio
      fig4  — "Pruning method comparison" summary (needs attention pruning data)
    """
    _nature_rc()
    mask_pct = [r * 100 for r in mask_ratios]
    n_layers = len(layers_to_prune)
    colors = _get_layer_colors(n_layers)

    # ── Figure 1: MSE vs Mask Ratio (2 panels) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        valid = np.isfinite(mse_vals)
        if np.any(valid):
            ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid[i]],
                    [mse_vals[i] for i in range(len(mse_vals)) if valid[i]],
                    "o-", color=colors[ci], label=f"L{li}", zorder=2)

    mse_all = [results['all_layers'].get(r, {}).get('metrics', {}).get('MSE', np.nan)
               for r in mask_ratios]
    valid = np.isfinite(mse_all)
    if np.any(valid):
        ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid[i]],
                [mse_all[i] for i in range(len(mse_all)) if valid[i]],
                "s--", color="black", linewidth=1.8, markersize=5,
                label="All-L", zorder=3)

    ax.axhline(baseline_mse, color="gray", linestyle="--", linewidth=1.2,
               label="Baseline", zorder=1)
    ax.set_xlabel("Mask Ratio (%)")
    ax.set_ylabel("MSE")
    ax.set_title("Forecast MSE vs Mask Ratio\n(Hidden-State Pruning)")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8)

    ax = axes[1]
    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        mse_vals_chg = [(m - baseline_mse) / baseline_mse * 100
                         if np.isfinite(m) else np.nan
                         for m in mse_vals]
        valid = np.isfinite(mse_vals_chg)
        if np.any(valid):
            ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid[i]],
                    [mse_vals_chg[i] for i in range(len(mse_vals_chg)) if valid[i]],
                    "o-", color=colors[ci], label=f"L{li}", zorder=2)

    mse_all_chg = [(m - baseline_mse) / baseline_mse * 100
                    if np.isfinite(m) else np.nan
                    for m in mse_all]
    valid = np.isfinite(mse_all_chg)
    if np.any(valid):
        ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid[i]],
                [mse_all_chg[i] for i in range(len(mse_all_chg)) if valid[i]],
                "s--", color="black", linewidth=1.8, markersize=5,
                label="All-L", zorder=3)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Mask Ratio (%)")
    ax.set_ylabel("MSE Change (%)")
    ax.set_title("MSE Degradation vs Baseline")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8)

    fig.savefig(os.path.join(output_dir, "fig1_mse.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig1_mse.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig1_mse.{{pdf,png}}")

    # ── Figure 2: Per-layer sensitivity and all-layer curve ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    mse_50_vals = []
    mse_50_valid = []
    for li in layers_to_prune:
        m = results['per_layer'][li].get(0.5, {}).get('metrics', {}).get('MSE', np.nan)
        mse_50_vals.append(m)
        mse_50_valid.append(li)
    valid_mask = np.isfinite(mse_50_vals)
    v_li = [l for l, v in zip(mse_50_valid, valid_mask) if v]
    v_ms = [m for m, v in zip(mse_50_vals, valid_mask) if v]
    v_col = [colors[layers_to_prune.index(l)] for l in v_li]

    bars = ax.bar(range(len(v_li)), v_ms, color=v_col,
                  edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2)

    all_50 = results['all_layers'].get(0.5, {}).get('metrics', {}).get('MSE', np.nan)
    if np.isfinite(all_50):
        ax.axhline(all_50, color="red", linestyle="--", linewidth=1.2,
                   label=f"All-L: {all_50:.4f}", zorder=1)
    if np.isfinite(baseline_mse):
        ax.axhline(baseline_mse, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Base: {baseline_mse:.4f}", zorder=1)

    ax.set_xticks(range(len(v_li)))
    ax.set_xticklabels([f"L{li}" for li in v_li])
    ax.set_xlabel("Layer")
    ax.set_ylabel("MSE (50% Mask)")
    ax.set_title("Per-Layer Sensitivity at 50% Mask\n(Hidden-State Pruning)")
    ax.legend(fontsize=7)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    ax = axes[1]
    valid_all = np.isfinite(mse_all)
    if np.any(valid_all):
        ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid_all[i]],
                [mse_all[i] for i in range(len(mse_all)) if valid_all[i]],
                "s-", color="black", linewidth=2.0, markersize=5,
                label="All-L", zorder=3)

    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        valid_pl = np.isfinite(mse_vals)
        if np.any(valid_pl):
            ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid_pl[i]],
                    [mse_vals[i] for i in range(len(mse_vals)) if valid_pl[i]],
                    "o-", color=colors[ci], label=f"L{li}", zorder=2,
                    alpha=0.7, linewidth=1.2)

    ax.axhline(baseline_mse, color="gray", linestyle="--", linewidth=1.2,
               label="Baseline", zorder=1)
    ax.set_xlabel("Mask Ratio (%)")
    ax.set_ylabel("MSE")
    ax.set_title("All-Layer vs Per-Layer MSE")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8)

    fig.savefig(os.path.join(output_dir, "fig2_sensitivity_alllayer.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig2_sensitivity_alllayer.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig2_sensitivity_alllayer.{{pdf,png}}")

    # ── Figure 3: Heatmap of MSE change per layer × mask ratio ───────────────
    fig, ax = plt.subplots(figsize=(5.5, 2.2))

    matrix = np.zeros((n_layers, len(mask_ratios)))
    for ci, li in enumerate(layers_to_prune):
        for ri, mr in enumerate(mask_ratios):
            m = results['per_layer'][li].get(mr, {}).get('metrics', {}).get('MSE', np.nan)
            matrix[ci, ri] = ((m - baseline_mse) / baseline_mse * 100
                               if np.isfinite(m) else np.nan)

    all_finite = np.all(np.isfinite(matrix))
    if all_finite:
        diverg = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 10
    else:
        diverg = 20
    diverg = max(diverg, 1.0)

    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-diverg, vmax=diverg)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("MSE Change (%)", fontsize=8)

    ax.set_xticks(range(len(mask_ratios)))
    ax.set_xticklabels([f"{int(r*100)}%" for r in mask_ratios])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{li}" for li in layers_to_prune])
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel("Layer")
    ax.set_title("MSE Change Heatmap\n(Hidden-State Pruning)")

    fig.savefig(os.path.join(output_dir, "fig3_heatmap.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig3_heatmap.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig3_heatmap.{{pdf,png}}")

    # ── Figure 4: Summary panel — degradation severity annotation ────────────
    fig, ax = plt.subplots(figsize=(7.0, 2.2))

    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        mse_chg = [(m - baseline_mse) / baseline_mse * 100
                    if np.isfinite(m) else np.nan
                    for m in mse_vals]
        valid = np.isfinite(mse_chg)
        if np.any(valid):
            ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid[i]],
                    [mse_chg[i] for i in range(len(mse_chg)) if valid[i]],
                    "o-", color=colors[ci], label=f"L{li}", zorder=2,
                    linewidth=1.5, markersize=4)

    valid_all = np.isfinite(mse_all_chg)
    if np.any(valid_all):
        ax.plot([mask_pct[i] for i in range(len(mask_pct)) if valid_all[i]],
                [mse_all_chg[i] for i in range(len(mse_all_chg)) if valid_all[i]],
                "s--", color="black", linewidth=2.0, markersize=5,
                label="All-L", zorder=3)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.axhline(5, color="#D55E00", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axhline(10, color="#CC79A7", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.fill_between([-5, 105], -1, 1, alpha=0.06, color="green",
                    label="Negligible (<1%)")
    ax.fill_between([-5, 105], 1, 5, alpha=0.06, color="#E69F00",
                    label="Moderate (1–5%)")
    ax.fill_between([-5, 105], 5, 10, alpha=0.06, color="#D55E00",
                    label="Significant (5–10%)")
    ax.fill_between([-5, 105], 10, 100, alpha=0.06, color="#CC79A7",
                    label="Severe (>10%)")

    ax.set_xlim(-2, 102)
    ax.set_ylim(
        min(-2, np.nanmin(mse_all_chg) - 2) if np.any(np.isfinite(mse_all_chg)) else -2,
        max(12, np.nanmax(mse_all_chg) + 2) if np.any(np.isfinite(mse_all_chg)) else 12
    )
    ax.set_xlabel("Mask Ratio (%)")
    ax.set_ylabel("MSE Change (%)")
    ax.set_title("Hidden-State Pruning: Degradation Severity Overview")
    ax.legend(loc="upper left", ncol=3, columnspacing=0.8,
              framealpha=0.9, edgecolor="lightgray")

    fig.savefig(os.path.join(output_dir, "fig4_severity_overview.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig4_severity_overview.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig4_severity_overview.{{pdf,png}}")

    # ── Figure 5: Top-k MI token retention analysis ───────────────────────────
    # Shows which keep_ratio thresholds are safe for each layer
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        keep_ratios = [(1.0 - r) * 100 for r in mask_ratios]
        mse_chg = [(m - baseline_mse) / baseline_mse * 100
                    if np.isfinite(m) else np.nan
                    for m in mse_vals]
        valid = np.isfinite(mse_chg)
        if np.any(valid):
            ax.plot(keep_ratios, mse_chg, "o-", color=colors[ci],
                    label=f"L{li}", zorder=2, linewidth=1.5, markersize=4)

    valid_all = np.isfinite(mse_all_chg)
    if np.any(valid_all):
        ax.plot(keep_ratios, mse_all_chg, "s--", color="black",
                linewidth=1.8, markersize=5, label="All-L", zorder=3)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Token Retention (%)")
    ax.set_ylabel("MSE Change (%)")
    ax.set_title("MSE vs Token Retention\n(Flipped x-axis)")
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8)
    ax.invert_xaxis()

    ax = axes[1]
    # Find the maximum safe mask ratio (where ΔMSE < 5%) for each layer
    safe_mask = {}
    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        mse_chg = [(m - baseline_mse) / baseline_mse * 100
                    if np.isfinite(m) else np.nan
                    for m in mse_vals]
        safe_mr = None
        for mr, delta in zip(mask_ratios, mse_chg):
            if np.isfinite(delta) and delta < 5.0:
                safe_mr = mr
        safe_mask[li] = safe_mr

    safe_vals = [safe_mask.get(li, np.nan) for li in layers_to_prune]
    valid_mask_s = np.isfinite(safe_vals)
    v_li_safe = [l for l, v in zip(layers_to_prune, valid_mask_s) if v]
    v_safe = [safe_mask[l] * 100 for l in v_li_safe]
    v_col_safe = [colors[layers_to_prune.index(l)] for l in v_li_safe]

    bars = ax.bar(range(len(v_li_safe)), v_safe, color=v_col_safe,
                  edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2)
    ax.set_xticks(range(len(v_li_safe)))
    ax.set_xticklabels([f"L{li}" for li in v_li_safe])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Safe Mask Ratio (%)")
    ax.set_title("Maximum Mask Ratio for <5% MSE Degradation")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    for bar, val in zip(bars, v_safe):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha='center', va='bottom', fontsize=7)

    fig.savefig(os.path.join(output_dir, "fig5_safe_mask_ratio.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig5_safe_mask_ratio.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig5_safe_mask_ratio.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Timer MI-Guided Hidden-State Pruning Experiment")
    # Paths
    parser.add_argument("--mi_dir", type=str,
                        default="./timer_mi_ksg_pca/Timer_MI_20260514_073311/",
                        help="Directory containing mi_matrix.npy or mi_hy_matrix.npy")
    parser.add_argument("--data_path", type=str, default="./datasets/ETTh1.csv")
    parser.add_argument("--data_type", type=str, default="ETTh1",
                        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])
    parser.add_argument("--output_dir", type=str,
                        default="./results/timer_mi_hidden_pruning/")
    # Model
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Data
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--freq", type=str, default="h")
    # Experiment
    parser.add_argument("--prune_layer", type=int, default=None,
                        help="Single layer to prune (0-indexed). None = all layers.")
    parser.add_argument("--all_layers", action="store_true", default=True,
                        help="Run per-layer pruning for all layers.")
    parser.add_argument("--mask_ratios", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        help="Mask ratios to evaluate")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Timer MI-Guided Hidden-State Token Pruning Experiment")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<22}: {v}")
    print("=" * 70)

    # ── Phase 1: Load and align MI matrix ──────────────────────────────────
    print("\n>>> [1/6] Loading MI matrix...")
    mi_path = os.path.join(args.mi_dir, "mi_hy_matrix.npy")
    if not os.path.exists(mi_path):
        mi_path = os.path.join(args.mi_dir, "mi_matrix.npy")
    if not os.path.exists(mi_path):
        raise FileNotFoundError(f"MI file not found in {args.mi_dir}")

    mi_matrix = np.load(mi_path)
    n_layers_raw, n_patches_raw = mi_matrix.shape
    print(f"  Raw MI matrix: {mi_matrix.shape}")

    # Load model to resolve shape ambiguities
    core_tmp = _unwrap(build_timer_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len, stride=args.patch_len,
        d_model=args.d_model, d_ff=args.d_ff,
        e_layers=args.e_layers, n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len, pred_len=args.pred_len,
    ))
    model_layers = core_tmp.layers
    model_heads = core_tmp.n_heads
    model_patches = args.seq_len // args.patch_len
    del core_tmp

    print(f"  Model: {model_layers} layers, {model_heads} heads, {model_patches} patches")

    # Auto-detect axis swap or per-head computation
    if n_layers_raw == model_heads and n_patches_raw == model_patches:
        print("  Detected: MI computed per-head. Averaging to per-layer.")
        mi_matrix = mi_matrix.mean(axis=0, keepdims=True)
        n_layers_raw = 1

    if n_layers_raw == model_patches and n_patches_raw == model_heads:
        print("  Detected: axes swapped. Transposing.")
        mi_matrix = mi_matrix.T
        n_layers_raw, n_patches_raw = n_patches_raw, n_layers_raw

    if n_patches_raw != model_patches and n_layers_raw == model_patches:
        print("  Auto-transposing MI matrix.")
        mi_matrix = mi_matrix.T
        n_layers_raw, n_patches_raw = n_patches_raw, n_layers_raw

    # Broadcast / truncate to match model layers
    if n_layers_raw != model_layers:
        print(f"  Adapting MI layers: {n_layers_raw} → {model_layers}")
        if n_layers_raw > model_layers:
            mi_matrix = mi_matrix[:model_layers]
        else:
            mi_matrix = np.broadcast_to(mi_matrix, (model_layers, mi_matrix.shape[1]))

    n_layers_final, n_patches_final = mi_matrix.shape
    print(f"  Final MI matrix: [{n_layers_final}, {n_patches_final}]")

    layers_to_prune = (
        list(range(n_layers_final)) if args.all_layers else [args.prune_layer]
    )
    if args.prune_layer is not None and args.prune_layer >= n_layers_final:
        print(f"  WARNING: prune_layer={args.prune_layer} >= n_layers={n_layers_final}")
        layers_to_prune = list(range(n_layers_final))

    # ── Phase 2: Load dataset ───────────────────────────────────────────────
    print("\n>>> [2/6] Loading dataset...")
    test_dataset = CIDatasetBenchmark(
        root_path=args.data_path,
        flag='test',
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = test_dataset.n_var
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"  Variables: {n_vars}, Test samples: {len(test_dataset)}")

    # ── Phase 3: Load model ─────────────────────────────────────────────────
    print("\n>>> [3/6] Loading Timer model...")
    model = build_timer_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len, stride=args.patch_len,
        d_model=args.d_model, d_ff=args.d_ff,
        e_layers=args.e_layers, n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len, pred_len=args.pred_len,
    ).to(device)
    model.eval()
    print(f"  Device: {device}")

    # ── Phase 4: Collect ground truth ───────────────────────────────────────
    print("\n>>> [4/6] Preparing ground truth...")
    trues_list = []
    with torch.no_grad():
        for batch_x, batch_y, *_ in tqdm(test_loader, desc="GT"):
            trues_list.append(batch_y.float().numpy())
    trues = np.concatenate(trues_list, axis=0)
    print(f"  Ground truth shape: {trues.shape}")

    # ── Phase 5: Run experiments ────────────────────────────────────────────
    print("\n>>> [5/6] Running hidden-state pruning experiments...")

    mask_ratios = [float(x) for x in args.mask_ratios]

    results = {
        'config': {
            'n_layers': n_layers_final,
            'n_patches': n_patches_final,
            'layers_to_prune': layers_to_prune,
            'mask_ratios': mask_ratios,
            'method': 'hidden_state_pruning',
            'description': (
                'Zero hidden states of low-MI tokens before entering each '
                'decoder layer. Cuts both attention contribution AND residual shortcut.'
            ),
        },
        'per_layer': {},
        'all_layers': {},
    }

    # ── Mode 1: Per-layer pruning ─────────────────────────────────────────
    print("\n  [Mode 1: Per-Layer Hidden-State Pruning]")
    for layer_idx in layers_to_prune:
        layer_mi = mi_matrix[layer_idx]
        results['per_layer'][layer_idx] = {}

        print(f"\n  Layer {layer_idx} | MI: [{layer_mi.min():.4f}, {layer_mi.max():.4f}]")
        print("  " + "-" * 60)

        for mask_ratio in mask_ratios:
            keep_ratio = 1.0 - mask_ratio
            n_keep = max(1, int(n_patches_final * keep_ratio))

            label = f"L{layer_idx} Mask={mask_ratio*100:.0f}%"
            if mask_ratio == 0.0:
                print(f"  [{label}] Baseline...")
                preds, avg_t, _ = forecast_baseline(
                    model, test_loader, device, pred_len=args.pred_len)
            else:
                print(f"  [{label}] Keep {n_keep}/{n_patches_final} tokens...")
                preds, avg_t, _ = forecast_with_hidden_pruning(
                    model, test_loader, device, mi_matrix,
                    n_patches_final, keep_ratio,
                    prune_layer=layer_idx, pred_len=args.pred_len)

            preds = align_pred_shape(preds, trues)
            metrics = compute_metrics(trues, preds)

            results['per_layer'][layer_idx][mask_ratio] = {
                'metrics': metrics,
                'avg_inference_time': avg_t,
                'n_keep': n_keep,
            }

            mse_ok = "" if np.isfinite(metrics['MSE']) else " ← NaN!"
            delta = ((metrics['MSE'] - results['per_layer'][layer_idx].get(0.0, {}).get('metrics', {}).get('MSE', np.nan))
                     / results['per_layer'][layer_idx].get(0.0, {}).get('metrics', {}).get('MSE', np.nan) * 100
                     if mask_ratio > 0.0 and np.isfinite(metrics['MSE']) else 0.0)
            print(f"    MSE: {metrics['MSE']:.6f}{mse_ok}  "
                  f"MAE: {metrics['MAE']:.6f}  "
                  f"ΔMSE%: {delta:+.2f}%  "
                  f"Time: {avg_t*1000:.2f}ms")

    # ── Mode 2: All-layer pruning ──────────────────────────────────────────
    print("\n  [Mode 2: All-Layer Hidden-State Pruning]")
    print("  " + "-" * 60)

    for mask_ratio in mask_ratios:
        keep_ratio = 1.0 - mask_ratio
        n_keep = max(1, int(n_patches_final * keep_ratio))

        label = f"All-L Mask={mask_ratio*100:.0f}%"
        if mask_ratio == 0.0:
            print(f"  [{label}] Baseline...")
            preds, avg_t, _ = forecast_baseline(
                model, test_loader, device, pred_len=args.pred_len)
        else:
            print(f"  [{label}] Keep {n_keep}/{n_patches_final} tokens per layer...")
            preds, avg_t, _ = forecast_with_hidden_pruning(
                model, test_loader, device, mi_matrix,
                n_patches_final, keep_ratio,
                prune_layer=None, pred_len=args.pred_len)

        preds = align_pred_shape(preds, trues)
        metrics = compute_metrics(trues, preds)

        results['all_layers'][mask_ratio] = {
            'metrics': metrics,
            'avg_inference_time': avg_t,
            'n_keep': n_keep,
        }

        mse_ok = "" if np.isfinite(metrics['MSE']) else " ← NaN!"
        delta = ((metrics['MSE'] - results['all_layers'].get(0.0, {}).get('metrics', {}).get('MSE', np.nan))
                 / results['all_layers'].get(0.0, {}).get('metrics', {}).get('MSE', np.nan) * 100
                 if mask_ratio > 0.0 and np.isfinite(metrics['MSE']) else 0.0)
        print(f"    MSE: {metrics['MSE']:.6f}{mse_ok}  "
              f"MAE: {metrics['MAE']:.6f}  "
              f"ΔMSE%: {delta:+.2f}%  "
              f"Time: {avg_t*1000:.2f}ms")

    # ── Extract baseline reference ────────────────────────────────────────
    baseline_mse = results['all_layers'].get(0.0, {}).get('metrics', {}).get('MSE', np.nan)
    baseline_time = results['all_layers'].get(0.0, {}).get('avg_inference_time', np.nan)
    if not np.isfinite(baseline_mse):
        baseline_mse = results['per_layer'][layers_to_prune[0]].get(0.0, {}).get('metrics', {}).get('MSE', np.nan)
        baseline_time = results['per_layer'][layers_to_prune[0]].get(0.0, {}).get('avg_inference_time', np.nan)

    # ── Phase 6: Save results ──────────────────────────────────────────────
    print("\n>>> [6/6] Saving results and plots...")
    results_path = os.path.join(args.output_dir, "hidden_pruning_results.pt")
    torch.save(results, results_path)
    print(f"  Saved: {results_path}")

    # ── Print summary tables ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Summary (Baseline MSE: {baseline_mse:.6f})")
    print("=" * 70)

    print(f"\n  Per-Layer @ 50% Mask:")
    print(f"  {'Layer':>7} | {'MSE':>12} | {'ΔMSE%':>10} | {'Time(ms)':>10}")
    print("  " + "-" * 46)
    for li in layers_to_prune:
        d = results['per_layer'][li].get(0.5, {})
        m = d.get('metrics', {})
        mse = m.get('MSE', np.nan)
        t = d.get('avg_inference_time', np.nan)
        delta = (mse - baseline_mse) / baseline_mse * 100 \
                if np.isfinite(mse) and np.isfinite(baseline_mse) else np.nan
        mse_s = f"{mse:.6f}" if np.isfinite(mse) else "NaN"
        delta_s = f"{delta:+.2f}%" if np.isfinite(delta) else "NaN"
        t_s = f"{t*1000:.2f}" if np.isfinite(t) else "NaN"
        print(f"  L{li:>6} | {mse_s:>12} | {delta_s:>10} | {t_s:>10}")

    print(f"\n  All-Layer Pruning:")
    print(f"  {'Mask%':>7} | {'MSE':>12} | {'ΔMSE%':>10} | {'Time(ms)':>10}")
    print("  " + "-" * 46)
    for mr in mask_ratios:
        d = results['all_layers'].get(mr, {})
        m = d.get('metrics', {})
        mse = m.get('MSE', np.nan)
        t = d.get('avg_inference_time', np.nan)
        delta = (mse - baseline_mse) / baseline_mse * 100 \
                if np.isfinite(mse) and np.isfinite(baseline_mse) else np.nan
        mse_s = f"{mse:.6f}" if np.isfinite(mse) else "NaN"
        delta_s = f"{delta:+.2f}%" if np.isfinite(delta) else "NaN"
        t_s = f"{t*1000:.2f}" if np.isfinite(t) else "NaN"
        print(f"  {mr*100:>6.0f}% | {mse_s:>12} | {delta_s:>10} | {t_s:>10}")

    # ── Generate Nature-style figures ──────────────────────────────────────
    try:
        plot_pruning_results(
            results=results,
            output_dir=args.output_dir,
            mask_ratios=mask_ratios,
            layers_to_prune=layers_to_prune,
            baseline_mse=baseline_mse,
            baseline_time=baseline_time,
        )
    except Exception as e:
        print(f"  WARNING: Plotting failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
