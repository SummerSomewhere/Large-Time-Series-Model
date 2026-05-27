#!/usr/bin/env python3
"""
Timer MI-Guided Attention Token Pruning Experiment

Pruning method: zero attention scores for low-MI tokens
- For the target layer(s), mask entire rows of the attention matrix for
  low-MI tokens so they contribute zero after softmax
- Sequence length is unchanged — this is a soft mask, not token deletion

Experiments:
  - Mode 1: per-layer pruning (one layer at a time)
  - Mode 2: all-layer pruning (all layers masked simultaneously)

Usage:
    python experiments/timer_mi_pruning_experiment.py \
        --mi_dir ./timer_mi_ksg_pca/Timer_MI_20260514_073311/ \
        --data_path ./datasets/ETTh1.csv \
        --seq_len 672 --pred_len 96 --prune_layer 6 \
        --output_dir ./results/timer_mi_pruning/
"""

import argparse
import json
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

def _run_manual_attention_layer(layer_module, dec_in, mask_obj, mi_vec,
                                keep_ratio, device, n_heads, debug=False):
    """
    Compute a single attention layer manually, masking low-MI token rows.

    dec_in:    [B_l, N, D]
    mask_obj:  TriangularCausalMask
    mi_vec:    [N,] per-patch MI for this layer
    keep_ratio: fraction of tokens to keep (0-1)
    """
    B_l, N, D = dec_in.shape

    q = layer_module.attention.query_projection(dec_in).view(B_l, N, n_heads, -1)
    k = layer_module.attention.key_projection(dec_in).view(B_l, N, n_heads, -1)
    v = layer_module.attention.value_projection(dec_in).view(B_l, N, n_heads, -1)

    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = torch.einsum("blhe,bshe->bhls", q, k)

    if mask_obj.mask is not None:
        scores = scores.masked_fill(mask_obj.mask, -float('inf'))

    scores = scale * scores

    # MI-guided token masking: shape [B_l, H, N, N]
    # token_mask[b,h,q,k] = True means token k cannot attend to query q
    # (i.e., scores[b,h,q,k] will be set to -inf)
    mask_np = np.zeros(N, dtype=np.bool_)   # False = attend, True = mask
    keep_k = max(1, int(N * keep_ratio))
    sorted_idx = np.argsort(mi_vec)[::-1][:keep_k]
    mask_np[sorted_idx] = True              # keep tokens False, drop tokens True
    token_mask = (torch.tensor(mask_np, device=device, dtype=torch.bool)
                  .unsqueeze(0).unsqueeze(1)      # [1, 1, N]
                  .expand(B_l, n_heads, N, N))   # [B_l, H, N, N]
    scores = scores.masked_fill(token_mask, -float('inf'))

    # Guard: if a query position has all keys masked (all -inf), softmax gives NaN.
    # Detect and replace with zeros (no attention) so the residual branch dominates.
    all_masked = token_mask.all(dim=-1)   # [B_l, H, N]
    if all_masked.any():
        n_bad = int(all_masked.sum().item())
        scores = scores.masked_fill(all_masked.unsqueeze(-1), 0.0)

    attn_weights = torch.softmax(scores, dim=-1)
    V = torch.einsum("bhls,bshd->blhd", attn_weights, v)
    attn_out = V.contiguous().view(B_l, N, -1)
    attn_out = layer_module.attention.out_projection(attn_out)

    dec_in = dec_in + layer_module.dropout(attn_out)
    dec_in = layer_module.norm1(dec_in)

    y = layer_module.dropout(
        layer_module.activation(
            layer_module.conv1(dec_in.transpose(-1, 1))))
    y = layer_module.dropout(
        layer_module.conv2(y).transpose(-1, 1))
    dec_in = layer_module.norm2(dec_in + y)

    return dec_in


def forecast_with_attention_pruning(model, test_loader, device, mi_matrix,
                                    n_patches, keep_ratio=1.0,
                                    prune_layer=None, pred_len=96):
    """
    Inference with per-head, per-token MI-guided attention masking.

    mi_matrix:  [n_layers, n_patches]
    keep_ratio: fraction of highest-MI tokens to retain
    prune_layer: None = all layers, int = only that layer
    """
    core = _unwrap(model)
    n_heads = core.n_heads

    preds_list = []
    inf_times = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
                test_loader, desc=f"Mask={keep_ratio*100:.0f}%", leave=False):

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
            dec_in, n_vars = core.enc_embedding(x2)   # [B, n_vars, N, D] → [B*n_vars, N, D]

            BM, N, D = dec_in.shape
            B_times_vars = B * n_vars
            dec_in = dec_in.view(B_times_vars, N, D)

            from utils.masking import TriangularCausalMask
            causal_mask = TriangularCausalMask(B_times_vars, N, device=device)

            start_time = time.time()

            if keep_ratio < 1.0:
                for li, layer_mod in enumerate(core.decoder.attn_layers):
                    if prune_layer is not None and li != prune_layer:
                        dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)
                        continue

                    dec_in = _run_manual_attention_layer(
                        layer_mod, dec_in, causal_mask,
                        mi_matrix[li], keep_ratio, device, n_heads,
                        debug=(li == 0 and B_times_vars == B_times_vars))

                # Apply remaining (non-pruned) layers normally
                for li, layer_mod in enumerate(core.decoder.attn_layers):
                    if prune_layer is not None and li == prune_layer:
                        break
                    dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)

                if core.decoder.norm is not None:
                    dec_in = core.decoder.norm(dec_in)
            else:
                for layer_mod in core.decoder.attn_layers:
                    dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)
                if core.decoder.norm is not None:
                    dec_in = core.decoder.norm(dec_in)

            if core.proj is not None:
                dec_in = core.proj(dec_in)

            # CRITICAL FIX: reshape back to [B, n_vars, N, patch_len] BEFORE projection
            # The manual loop left dec_in as [B*n_vars, N, patch_len] (already projected)
            # but the reshape expects [B*n_vars, N, patch_len] from the linear projection
            # Baseline proj gives [B*n_vars, N, patch_len]; manual loop also gives the same
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

            BM, N, D = dec_in.shape   # BM = B * n_vars

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


def plot_pruning_results(results, output_dir, mask_ratios, layers_to_prune,
                          baseline_mse, baseline_time, dataset_name=""):
    """
    Generate Nature-style publication figures.

    results must contain:
        per_layer:  {layer_idx: {mask_ratio: {'metrics': {...}, 'avg_inference_time': float}}}
        all_layers: {mask_ratio:   {'metrics': {...}, 'avg_inference_time': float}}
    """
    _nature_rc()
    mask_pct = [r * 100 for r in mask_ratios]
    n_layers = len(layers_to_prune)
    colors = _get_layer_colors(n_layers)

    # ── Figure 1: Main MSE and MSE change (two panels, full width) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        if np.any(np.isfinite(mse_vals)):
            ax.plot(mask_pct, mse_vals, "o-", color=colors[ci], label=f"L{li}", zorder=2)

    mse_all = [results['all_layers'].get(r, {}).get('metrics', {}).get('MSE', np.nan)
               for r in mask_ratios]
    valid = np.isfinite(mse_all)
    if np.any(valid):
        ax.plot([mask_pct[i] for i, v in enumerate(valid) if v],
                [mse_all[i] for i, v in enumerate(valid) if v],
                "s--", color="black", linewidth=1.8, markersize=5,
                label="All-L", zorder=3)

    ax.axhline(baseline_mse, color="gray", linestyle="--", linewidth=1.2,
               label="Baseline", zorder=1)
    ax.set_xlabel("Mask Ratio (%)")
    ax.set_ylabel("MSE")
    ax.set_title("Forecast MSE vs Mask Ratio")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8)

    ax = axes[1]
    for ci, li in enumerate(layers_to_prune):
        mse_vals = [
            results['per_layer'][li].get(r, {}).get('metrics', {}).get('MSE', np.nan)
            for r in mask_ratios
        ]
        mse_vals = [(m - baseline_mse) / baseline_mse * 100
                    if np.isfinite(m) else np.nan for m in mse_vals]
        if np.any(np.isfinite(mse_vals)):
            ax.plot(mask_pct, mse_vals, "o-", color=colors[ci], label=f"L{li}", zorder=2)

    mse_all_chg = [(m - baseline_mse) / baseline_mse * 100
                   if np.isfinite(results['all_layers'].get(r, {}).get('metrics', {}).get('MSE', np.nan))
                   else np.nan
                   for r, m in zip(mask_ratios, mse_all)]
    valid = np.isfinite(mse_all_chg)
    if np.any(valid):
        ax.plot([mask_pct[i] for i, v in enumerate(valid) if v],
                [mse_all_chg[i] for i, v in enumerate(valid) if v],
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

    # ── Figure 2: Speedup and per-layer sensitivity at 50% ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    if np.isfinite(baseline_time) and baseline_time > 0:
        for ci, li in enumerate(layers_to_prune):
            t_vals = [
                results['per_layer'][li].get(r, {}).get('avg_inference_time', np.nan)
                for r in mask_ratios
            ]
            speedups = [baseline_time / t if np.isfinite(t) and t > 0 else np.nan
                        for t in t_vals]
            if np.any(np.isfinite(speedups)):
                ax.plot(mask_pct, speedups, "o-", color=colors[ci],
                        label=f"L{li}", zorder=2)

        t_all = [results['all_layers'].get(r, {}).get('avg_inference_time', np.nan)
                 for r in mask_ratios]
        spd_all = [baseline_time / t if np.isfinite(t) and t > 0 else np.nan
                   for t in t_all]
        valid = np.isfinite(spd_all)
        if np.any(valid):
            ax.plot([mask_pct[i] for i, v in enumerate(valid) if v],
                    [spd_all[i] for i, v in enumerate(valid) if v],
                    "s--", color="black", linewidth=1.8, markersize=5,
                    label="All-L", zorder=3)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Mask Ratio (%)")
    ax.set_ylabel("Speedup (x)")
    ax.set_title("Inference Speedup")
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8)

    ax = axes[1]
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
    ax.set_title("Per-Layer Sensitivity at 50% Mask")
    ax.legend(fontsize=7)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    fig.savefig(os.path.join(output_dir, "fig2_speedup_sensitivity.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig2_speedup_sensitivity.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig2_speedup_sensitivity.{{pdf,png}}")

    # ── Figure 3: Summary heatmap of MSE change per layer × mask ratio ──────
    fig, ax = plt.subplots(figsize=(5.5, 2.2))

    matrix = np.zeros((n_layers, len(mask_ratios)))
    for ci, li in enumerate(layers_to_prune):
        for ri, mr in enumerate(mask_ratios):
            m = results['per_layer'][li].get(mr, {}).get('metrics', {}).get('MSE', np.nan)
            matrix[ci, ri] = (m - baseline_mse) / baseline_mse * 100 if np.isfinite(m) else np.nan

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    clim = max(abs(vmin), abs(vmax))
    diverg = np.nanmax(np.abs(matrix[~np.isnan(matrix)])) if np.any(np.isfinite(matrix)) else 10

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
    ax.set_title("MSE Change Heatmap")

    fig.savefig(os.path.join(output_dir, "fig3_heatmap.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig3_heatmap.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig3_heatmap.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Timer MI-Guided Attention Pruning Experiment")
    # Paths
    parser.add_argument("--mi_dir", type=str,
                        default="./timer_mi_ksg_pca/Timer_MI_20260514_073311/",
                        help="Directory containing mi_matrix.npy or mi_hy_matrix.npy")
    parser.add_argument("--data_path", type=str, default="./datasets/ETTh1.csv")
    parser.add_argument("--data_type", type=str, default="ETTh1",
                        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])
    parser.add_argument("--output_dir", type=str, default="./results/timer_mi_pruning/")
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
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
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
    print("Timer MI-Guided Attention Token Pruning Experiment")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
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
        mi_matrix = mi_matrix.mean(axis=0, keepdims=True)   # [1, n_patches]
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
    print("\n>>> [5/6] Running pruning experiments...")

    mask_ratios = [float(x) for x in args.mask_ratios]

    results = {
        'config': {
            'n_layers': n_layers_final,
            'n_patches': n_patches_final,
            'layers_to_prune': layers_to_prune,
            'mask_ratios': mask_ratios,
        },
        'per_layer': {},
        'all_layers': {},
    }

    # ── Mode 1: Per-layer pruning ─────────────────────────────────────────
    print("\n  [Mode 1: Per-Layer Pruning]")
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
                print(f"  [{label}] Keep {n_keep}/{n_patches_final}...")
                preds, avg_t, _ = forecast_with_attention_pruning(
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
            print(f"    MSE: {metrics['MSE']:.6f}{mse_ok}  "
                  f"MAE: {metrics['MAE']:.6f}  "
                  f"Time: {avg_t*1000:.2f}ms")

    # ── Mode 2: All-layer pruning ──────────────────────────────────────────
    print("\n  [Mode 2: All-Layer Pruning]")
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
            print(f"  [{label}] Keep {n_keep}/{n_patches_final} per layer...")
            preds, avg_t, _ = forecast_with_attention_pruning(
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
        print(f"    MSE: {metrics['MSE']:.6f}{mse_ok}  "
              f"MAE: {metrics['MAE']:.6f}  "
              f"Time: {avg_t*1000:.2f}ms")

    # ── Extract baseline reference ────────────────────────────────────────
    baseline_mse = results['all_layers'].get(0.0, {}).get('metrics', {}).get('MSE', np.nan)
    baseline_time = results['all_layers'].get(0.0, {}).get('avg_inference_time', np.nan)
    if not np.isfinite(baseline_mse):
        baseline_mse = results['per_layer'][layers_to_prune[0]].get(0.0, {}).get('metrics', {}).get('MSE', np.nan)
        baseline_time = results['per_layer'][layers_to_prune[0]].get(0.0, {}).get('avg_inference_time', np.nan)

    # ── Phase 6: Save results ──────────────────────────────────────────────
    print("\n>>> [6/6] Saving results and plots...")
    results_path = os.path.join(args.output_dir, "pruning_results.pt")
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
        delta = (mse - baseline_mse) / baseline_mse * 100 if np.isfinite(mse) and np.isfinite(baseline_mse) else np.nan
        mse_s = f"{mse:.6f}" if np.isfinite(mse) else "NaN"
        delta_s = f"{delta:+.2f}%" if np.isfinite(delta) else "NaN"
        t_s = f"{t*1000:.2f}" if np.isfinite(t) else "NaN"
        print(f"  L{li:>6} | {mse_s:>12} | {delta_s:>10} | {t_s:>10}")

    print(f"\n  All-Layer Pruning:")
    print(f"  {'Mask%':>7} | {'MSE':>12} | {'ΔMSE%':>10} | {'Speedup':>10}")
    print("  " + "-" * 46)
    for mr in mask_ratios:
        d = results['all_layers'].get(mr, {})
        m = d.get('metrics', {})
        mse = m.get('MSE', np.nan)
        t = d.get('avg_inference_time', np.nan)
        delta = (mse - baseline_mse) / baseline_mse * 100 if np.isfinite(mse) and np.isfinite(baseline_mse) else np.nan
        spd = baseline_time / t if np.isfinite(t) and t > 0 else np.nan
        mse_s = f"{mse:.6f}" if np.isfinite(mse) else "NaN"
        delta_s = f"{delta:+.2f}%" if np.isfinite(delta) else "NaN"
        spd_s = f"{spd:.2f}x" if np.isfinite(spd) else "NaN"
        print(f"  {mr*100:>6.0f}% | {mse_s:>12} | {delta_s:>10} | {spd_s:>10}")

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

    print("\nDone.")


if __name__ == "__main__":
    main()
