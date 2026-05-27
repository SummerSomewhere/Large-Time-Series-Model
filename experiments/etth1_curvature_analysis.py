#!/usr/bin/env python3
"""
Timer Decoder Curvature Analysis + Within/Between-Layer Metrics + Entropy.

Single-script generation of all analysis figures in one run:
  1. Curvature  vs decoder layer depth
  2. Curvature  per-patch distribution
  3. Within-layer trajectory: adjacent-patch cosine similarity
  4. Inter-layer CKA heatmap
  5. Hidden-state entropy per layer
"""

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


# ── Forward helpers ──────────────────────────────────────────────────────────

def forward_collect_layers(model, x_enc):
    """
    x_enc: [B, L, M] Timer convention.
    Returns: list of [B, N, D] tensors — pooled hidden states per decoder layer.
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
        h, _, _ = o3(h, attn_mask=mask)
        layers_out.append(pool(h.detach()))   # [B, N, D]
    return layers_out, int(n_vars), int(N)


# ── Curvature computation ────────────────────────────────────────────────────

def compute_sample_curvature(Z: torch.Tensor) -> torch.Tensor:
    """
    Compute curvature for a batch of samples.
    Z: [B, N, D] — hidden states after pooling over variates
    Returns: [B,] tensor of curvatures (one per sample)
    """
    B, N, D = Z.shape
    if N < 3:
        return torch.zeros(B, device=Z.device)

    Z_norm = Z / (torch.norm(Z, dim=-1, keepdim=True) + 1e-10)
    v = Z_norm[:, 1:, :] - Z_norm[:, :-1, :]
    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-10
    v = v / v_norm
    cos_angles = (v[:, 1:, :] * v[:, :-1, :]).sum(dim=-1)
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    angles = torch.acos(cos_angles)
    curvature = angles.mean(dim=1)
    return curvature


def compute_layer_curvature_all_samples(
    Z_layer: torch.Tensor,
    per_sample: bool = False,
) -> tuple[float, np.ndarray] | np.ndarray:
    """
    Compute curvature for one layer's hidden states.
    Z_layer: [B, N, D] Tensor on GPU
    Returns: (mean_curvature, per_sample_curvatures_array)
    """
    curvatures = compute_sample_curvature(Z_layer)
    curv_np = curvatures.detach().cpu().float().numpy()
    mean_c = float(curvatures.mean().item())
    if per_sample:
        return curv_np
    return mean_c, curv_np


def compute_curvature_per_position(Z: torch.Tensor) -> np.ndarray:
    """
    Compute the angle at each "elbow" between consecutive velocity vectors.
    Returns: [B, N-2] array of per-position angles.
    """
    B, N, D = Z.shape
    if N < 3:
        return np.zeros((B, 1))

    Z_norm = Z / (torch.norm(Z, dim=-1, keepdim=True) + 1e-10)
    v = Z_norm[:, 1:, :] - Z_norm[:, :-1, :]
    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-10
    v = v / v_norm
    cos_angles = (v[:, 1:, :] * v[:, :-1, :]).sum(dim=-1)
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    angles = torch.acos(cos_angles).detach().cpu().float().numpy()
    return angles


# ── Within-layer: adjacent-patch cosine similarity ───────────────────────────

def compute_adjacent_similarity(Z: torch.Tensor) -> np.ndarray:
    """
    Cosine similarity between consecutive patch embeddings (within a layer).
    Z: [B, N, D]
    Returns: [B,] per-sample mean similarity across patch positions
    """
    Z_norm = Z / (torch.norm(Z, dim=-1, keepdim=True) + 1e-10)
    sim = (Z_norm[:, :-1, :] * Z_norm[:, 1:, :]).sum(dim=-1)
    return sim.detach().cpu().float().numpy().mean(axis=1)


# ── Inter-layer: Linear CKA ─────────────────────────────────────────────────

def _median_sq_bandwidth(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    d2 = torch.cdist(X, X, p=2.0) ** 2
    triu_idx = torch.triu_indices(n, n, offset=1, device=X.device)
    return d2[triu_idx[0], triu_idx[1]].median().clamp(min=1e-12)


def rbf_kernel(X: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(X, X, p=2.0) ** 2
    return torch.exp(-d2 / (2.0 * sigma_sq))


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between X [n, D] and Y [n, D]. Returns scalar in [0, 1].
    """
    n = X.shape[0]
    if n < 4:
        return float("nan")
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    sx = _median_sq_bandwidth(K)
    sy = _median_sq_bandwidth(L)
    Krbf = rbf_kernel(K, sx)
    Lrbf = rbf_kernel(L, sy)
    H = torch.eye(n, device=K.device, dtype=K.dtype) - (1.0 / n)
    Kc = H @ Krbf @ H
    Lc = H @ Lrbf @ H
    hsic_xy = torch.trace(Kc @ Lc) / ((n - 1) ** 2)
    hsic_xx = torch.trace(Kc @ Kc) / ((n - 1) ** 2)
    hsic_yy = torch.trace(Lc @ Lc) / ((n - 1) ** 2)
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return float("nan")
    return (hsic_xy / denom).item()


# ── Entropy of hidden states ─────────────────────────────────────────────────

def compute_hidden_entropy(Z: torch.Tensor, n_bins: int = 20) -> np.ndarray:
    """
    Estimate marginal entropy of hidden dimensions per layer using histogram binning.
    Z: [B, N, D] — flatten all patch tokens per dimension
    Returns: scalar mean entropy across dimensions (in nats)
    """
    Z_flat = Z.reshape(-1, Z.shape[-1]).detach().cpu().float().numpy()  # [B*N, D]
    D = Z_flat.shape[1]
    entropies = np.zeros(D)
    for d in range(D):
        vals = Z_flat[:, d]
        hist, _ = np.histogram(vals, bins=n_bins, density=True)
        hist = hist[hist > 0]
        entropies[d] = -np.sum(hist * np.log(hist + 1e-12)) if len(hist) > 0 else 0.0
    return entropies


# ── Build namespace for model loading ───────────────────────────────────────

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
        "model_id": "curvature",
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


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_4panel_figure(
    layer_curvatures: list[float],
    layer_stds: list[float],
    layer_similarity: list[float],
    layer_similarity_stds: list[float],
    layer_entropy: list[float],
    layer_entropy_stds: list[float],
    per_sample_curvatures: list[np.ndarray],
    layer_indices: list[int],
    n_patches: int,
    n_layers: int,
    output_path: str,
):
    """
    Combined 4-panel figure in one run:
      (A) Curvature vs layer depth
      (B) Per-sample curvature violin per layer
      (C) Within-layer: adjacent-patch cosine similarity vs layer
      (D) Entropy vs layer depth
    """
    layers = np.arange(n_layers)
    curvatures = np.array(layer_curvatures)
    stds = np.array(layer_stds)
    similarity = np.array(layer_similarity)
    sim_stds = np.array(layer_similarity_stds)
    entropy = np.array(layer_entropy)
    ent_stds = np.array(layer_entropy_stds)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Timer Decoder: Layer-wise Geometric & Entropy Analysis (ETTh1)",
                 fontsize=15, fontweight="bold", y=0.99)

    # ── Panel A: Curvature by layer ──────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.errorbar(layers, curvatures, yerr=stds,
                 fmt="o-", color="#E63946", linewidth=2, markersize=7,
                 capsize=4, label="Mean ± Std", alpha=0.9)
    ax1.fill_between(layers, curvatures - stds, curvatures + stds,
                     alpha=0.15, color="#E63946")
    ax1.set_xlabel("Decoder Layer Index", fontsize=11)
    ax1.set_ylabel("Curvature (radians)", fontsize=11)
    ax1.set_title("(A) Curvature vs Layer Depth", fontsize=12, fontweight="bold")
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f"L{i}" for i in layers], fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    min_idx = int(np.argmin(curvatures))
    max_idx = int(np.argmax(curvatures))
    ax1.annotate(f"min={curvatures[min_idx]:.4f}",
                 xy=(min_idx, curvatures[min_idx]),
                 xytext=(min_idx + 0.5, curvatures[min_idx] + stds[min_idx]),
                 fontsize=8, color="#457B9D",
                 arrowprops=dict(arrowstyle="->", color="#457B9D", lw=0.8))
    ax1.annotate(f"max={curvatures[max_idx]:.4f}",
                 xy=(max_idx, curvatures[max_idx]),
                 xytext=(max_idx - 1.5, curvatures[max_idx] + stds[max_idx]),
                 fontsize=8, color="#1D3557",
                 arrowprops=dict(arrowstyle="->", color="#1D3557", lw=0.8))

    # ── Panel B: Curvature distribution ───────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    n_plyrs = len(per_sample_curvatures)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_plyrs))
    pos = np.arange(n_plyrs)
    vp = ax2.violinplot(per_sample_curvatures, positions=pos, showmeans=False, showmedians=False)
    for i, (pc, col) in enumerate(zip(vp["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for partname in ["cbars", "cmins", "cmaxes"]:
        if partname in vp:
            vp[partname].set_edgecolor("#888")
    for i, curv_np in enumerate(per_sample_curvatures):
        ax2.scatter(np.full_like(curv_np, i), curv_np, alpha=0.25, s=6, color=colors[i], zorder=3)
        ax2.scatter([i], [np.mean(curv_np)], color="white", s=30, zorder=4,
                    edgecolor="black", linewidth=1, marker="_")
    ax2.set_xlabel("Decoder Layer", fontsize=11)
    ax2.set_ylabel("Curvature (radians)", fontsize=11)
    ax2.set_title("(B) Per-Sample Curvature Distribution", fontsize=12, fontweight="bold")
    ax2.set_xticks(pos)
    ax2.set_xticklabels([f"L{li}" for li in layer_indices], fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel C: Within-layer adjacent similarity ─────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.errorbar(layers, similarity, yerr=sim_stds,
                 fmt="s-", color="#2A9D8F", linewidth=2, markersize=7,
                 capsize=4, label="Mean ± Std", alpha=0.9)
    ax3.fill_between(layers, similarity - sim_stds, similarity + sim_stds,
                     alpha=0.15, color="#2A9D8F")
    ax3.set_xlabel("Decoder Layer Index", fontsize=11)
    ax3.set_ylabel("Cosine Similarity", fontsize=11)
    ax3.set_title("(C) Within-Layer: Adjacent-Patch Cosine Similarity", fontsize=12, fontweight="bold")
    ax3.set_xticks(layers)
    ax3.set_xticklabels([f"L{i}" for i in layers], fontsize=9)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.axhline(0.0, color="gray", linestyle=":", lw=0.8, alpha=0.5)

    # ── Panel D: Entropy per layer ────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.errorbar(layers, entropy, yerr=ent_stds,
                 fmt="^-", color="#9B59B6", linewidth=2, markersize=7,
                 capsize=4, label="Mean ± Std (per dim)", alpha=0.9)
    ax4.fill_between(layers, entropy - ent_stds, entropy + ent_stds,
                     alpha=0.15, color="#9B59B6")
    ax4.set_xlabel("Decoder Layer Index", fontsize=11)
    ax4.set_ylabel("Entropy (nats, hist)", fontsize=11)
    ax4.set_title("(D) Hidden-State Marginal Entropy per Layer", fontsize=12, fontweight="bold")
    ax4.set_xticks(layers)
    ax4.set_xticklabels([f"L{i}" for i in layers], fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cka_heatmap(
    cka_mat: np.ndarray,
    layer_indices: list[int],
    output_path: str,
):
    """
    Inter-layer CKA heatmap: Linear CKA between layer representations.
    Each layer representation = mean-pool over patches per sample.
    """
    n = cka_mat.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cka_mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [f"L{li}" for li in layer_indices]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Decoder Layer", fontsize=11)
    ax.set_ylabel("Decoder Layer", fontsize=11)
    ax.set_title("Inter-Layer CKA: Linear Similarity Between Decoder Layers", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA", fontsize=10)

    for i in range(n):
        for j in range(n):
            val = cka_mat[i, j]
            text_color = "white" if cka_mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_normalized_comparison(
    layer_curvatures: list[float],
    n_layers: int,
    output_path: str,
):
    """
    Normalized curvature (0-1 per layer) vs layer depth.
    """
    layers = np.arange(n_layers)
    curvatures = np.array(layer_curvatures)

    curv_min, curv_max = curvatures.min(), curvatures.max()
    if curv_max > curv_min:
        curv_norm = (curvatures - curv_min) / (curv_max - curv_min)
    else:
        curv_norm = np.zeros_like(curvatures)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, curv_norm, "s-", color="#264653", linewidth=2.5, markersize=8,
            label="Normalized Curvature", zorder=3)
    ax.fill_between(layers, 0, curv_norm, alpha=0.15, color="#264653")

    ax.set_xlabel("Decoder Layer Index", fontsize=12)
    ax.set_ylabel("Normalized Curvature (0-1)", fontsize=12)
    ax.set_title("Normalized Curvature vs Decoder Layer Depth (Timer / ETTh1)", fontsize=13, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="best")

    min_idx = int(np.argmin(curv_norm))
    max_idx = int(np.argmax(curv_norm))
    ax.axhline(curv_norm[min_idx], color="#E76F51", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(curv_norm[max_idx], color="#2A9D8F", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(min_idx + 0.2, curv_norm[min_idx] + 0.03, f"smoothest (L{min_idx})", fontsize=8, color="#E76F51")
    ax.text(max_idx + 0.2, curv_norm[max_idx] + 0.03, f"sharpest (L{max_idx})", fontsize=8, color="#2A9D8F")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_within_layer_detail(
    layer_similarities: list[np.ndarray],
    layer_curvatures: list[np.ndarray],
    layer_indices: list[int],
    output_path: str,
):
    """
    Scatter + trend: within-layer adjacent similarity vs curvature per layer.
    One subplot per layer, showing per-sample (sim, curv) pairs.
    """
    n_layers = len(layer_similarities)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    colors_scatter = plt.cm.plasma(np.linspace(0.2, 0.9, n_layers))

    for col, (sim_np, curv_np) in enumerate(zip(layer_similarities, layer_curvatures)):
        ax = axes[col // n_cols][col % n_cols]
        li = layer_indices[col]
        ax.scatter(sim_np, curv_np, alpha=0.3, s=10, color=colors_scatter[col], zorder=3)
        ax.axhline(curv_np.mean(), color="red", linestyle="--", lw=1, alpha=0.7,
                   label=f"mean curv={curv_np.mean():.3f}")
        ax.axvline(sim_np.mean(), color="blue", linestyle="--", lw=1, alpha=0.7,
                   label=f"mean sim={sim_np.mean():.3f}")
        ax.set_xlabel("Adj-Cosine Sim", fontsize=8)
        ax.set_ylabel("Curvature (rad)", fontsize=8)
        ax.set_title(f"L{li}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.25)

    for col in range(n_layers, n_rows * n_cols):
        axes[col // n_cols][col % n_cols].axis("off")

    fig.suptitle("Within-Layer: Adjacent-Patch Similarity vs Curvature", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Timer Decoder Curvature + Layer Analysis")
    parser.add_argument("--ckpt_path", type=str, required=True)
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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max test samples to analyze")
    parser.add_argument("--out_dir", type=str, default="./results/curvature_etth1/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("Timer Decoder: Curvature + Within/Between-Layer + Entropy")
    print("=" * 60)
    print(f"  ckpt_path   : {args.ckpt_path}")
    print(f"  data        : {args.data}")
    print(f"  seq_len     : {args.seq_len}")
    print(f"  patch_len   : {args.patch_len}")
    print(f"  max_samples : {args.max_samples}")
    print(f"  device      : {device}")
    print(f"  out_dir     : {args.out_dir}")
    print("=" * 60)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("\n[1] Loading Timer model...")
    ns = build_namespace(args)
    model = TimerModel(ns)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    core = _unwrap_timer(model)
    print(f"  Loaded. Decoder layers: {len(core.decoder.attn_layers)}")

    # ── 2. Load test data ───────────────────────────────────────────────────
    print("\n[2] Building test data loader...")
    ns.batch_size = args.batch_size
    ns.num_workers = args.num_workers
    _, loader = data_provider(ns, "test")
    print(f"  Test loader: {len(loader)} batches")

    # ── 3. Collect hidden states per layer ───────────────────────────────────
    print("\n[3] Collecting layer representations...")
    t0 = time.time()
    per_layer_z: list[list[torch.Tensor]] = [[] for _ in range(MI_DECODER_LAYER_CAP)]
    n_extracted = 0

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        if n_extracted >= args.max_samples:
            break
        B = batch_x.shape[0]
        keep = min(B, args.max_samples - n_extracted)
        x_batch = batch_x[:keep].float().to(device)

        with torch.no_grad():
            layers_out, n_vars, N = forward_collect_layers(model, x_batch)

        for li in range(len(layers_out)):
            per_layer_z[li].append(layers_out[li][:keep].cpu())

        n_extracted += keep
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}: {n_extracted} samples collected")

        if n_extracted >= args.max_samples:
            break

    layer_z = {}
    for li in range(MI_DECODER_LAYER_CAP):
        if per_layer_z[li]:
            layer_z[li] = torch.cat(per_layer_z[li], dim=0)   # [S, N, D]

    n_layers = len(layer_z)
    n_patches = N
    n_samples = next(iter(layer_z.values())).shape[0]
    d_model = next(iter(layer_z.values())).shape[2]
    print(f"  Collected: {n_layers} layers, {n_samples} samples, {n_patches} patches, d_model={d_model}")
    print(f"  Collection time: {time.time()-t0:.1f}s")

    # ── 4. Compute all metrics per layer ──────────────────────────────────────
    print("\n[4] Computing metrics per layer...")
    t1 = time.time()

    layer_curvatures: list[float] = []
    layer_stds: list[float] = []
    per_sample_curvatures: list[np.ndarray] = []
    pos_curvatures_list: list[np.ndarray] = []

    layer_similarity: list[float] = []
    layer_similarity_stds: list[float] = []
    per_sample_similarity: list[np.ndarray] = []

    layer_entropy: list[float] = []
    layer_entropy_stds: list[float] = []

    for li in sorted(layer_z.keys()):
        Z = layer_z[li].to(device)   # [S, N, D]

        # ── Curvature ──
        mean_c, curv_np = compute_layer_curvature_all_samples(Z)
        std_c = float(curv_np.std())
        layer_curvatures.append(mean_c)
        layer_stds.append(std_c)
        per_sample_curvatures.append(curv_np)

        pos_angles = compute_curvature_per_position(Z)
        pos_mean = pos_angles.mean(axis=0)
        pos_curvatures_list.append(pos_mean)

        # ── Within-layer: adjacent-patch cosine similarity ──
        sim_np = compute_adjacent_similarity(Z)
        layer_similarity.append(float(sim_np.mean()))
        layer_similarity_stds.append(float(sim_np.std()))
        per_sample_similarity.append(sim_np)

        # ── Entropy ──
        ent_np = compute_hidden_entropy(Z, n_bins=20)
        layer_entropy.append(float(ent_np.mean()))
        layer_entropy_stds.append(float(ent_np.std()))

        print(f"  Layer {li}: curv={mean_c:.4f} (std={std_c:.4f}) | "
              f"adj_sim={sim_np.mean():.4f} | entropy={ent_np.mean():.4f}")

    print(f"  Metrics computation time: {time.time()-t1:.1f}s")

    pos_curvatures = np.stack(pos_curvatures_list, axis=0)
    layer_indices = sorted(layer_z.keys())

    # ── 5. Inter-layer CKA matrix ─────────────────────────────────────────────
    print("\n[5] Computing inter-layer CKA matrix...")
    t2 = time.time()
    n_lyrs = n_layers
    cka_mat = np.zeros((n_lyrs, n_lyrs), dtype=np.float64)

    Z_cpu = {}
    for li in layer_indices:
        h = layer_z[li]   # [S, N, D]
        rep = h.mean(dim=1).view(n_samples, -1)   # [S, D]
        Z_cpu[li] = rep.float()

    for i, li in enumerate(layer_indices):
        cka_mat[i, i] = 1.0
        for j, lj in enumerate(layer_indices):
            if j > i:
                val = linear_cka(Z_cpu[li], Z_cpu[lj])
                cka_mat[i, j] = val
                cka_mat[j, i] = val

    print(f"  CKA matrix computed in {time.time()-t2:.1f}s")
    print(f"  CKA diagonal: {np.diag(cka_mat)}")

    # ── 6. Save results ──────────────────────────────────────────────────────
    print("\n[6] Saving results...")
    results = {
        "config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                   for k, v in vars(args).items()},
        "n_layers": n_layers,
        "n_patches": n_patches,
        "n_samples": n_samples,
        "d_model": d_model,
        "layer_curvatures": layer_curvatures,
        "layer_stds": layer_stds,
        "layer_adj_similarity": layer_similarity,
        "layer_adj_similarity_stds": layer_similarity_stds,
        "layer_entropy": layer_entropy,
        "layer_entropy_stds": layer_entropy_stds,
        "per_position_curvature": pos_curvatures.tolist(),
        "cka_matrix": cka_mat.tolist(),
        "per_sample": {
            str(li): {"curvature": curv.tolist(), "similarity": sim.tolist()}
            for li, curv, sim in zip(layer_indices, per_sample_curvatures, per_sample_similarity)
        },
    }
    results_path = os.path.join(args.out_dir, "curvature_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\n[7] Generating plots...")

    # Combined 4-panel figure (curvature A+B + within-layer similarity C + entropy D)
    plot_4panel_figure(
        layer_curvatures, layer_stds,
        layer_similarity, layer_similarity_stds,
        layer_entropy, layer_entropy_stds,
        per_sample_curvatures,
        layer_indices, n_patches, n_layers,
        os.path.join(args.out_dir, "fig_combined_4panel.png"),
    )

    # Inter-layer CKA heatmap
    plot_cka_heatmap(
        cka_mat, layer_indices,
        os.path.join(args.out_dir, "fig_inter_layer_cka_heatmap.png"),
    )

    # Normalized curvature comparison
    plot_normalized_comparison(
        layer_curvatures, n_layers,
        os.path.join(args.out_dir, "figD_normalized_curvature.png"),
    )

    # Within-layer scatter: similarity vs curvature per layer
    plot_within_layer_detail(
        per_sample_similarity, per_sample_curvatures,
        layer_indices,
        os.path.join(args.out_dir, "fig_within_layer_sim_vs_curv.png"),
    )

    print(f"\n[Done] Total time: {time.time()-t0:.1f}s")
    print(f"  All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
