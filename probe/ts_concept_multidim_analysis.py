#!/usr/bin/env python3
"""
Step 4: Multi-Dimensional Analysis (Timer version)

Performs visualization and similarity analysis of Timer representations:

  1. CKA (Centered Kernel Alignment):
     - Compute CKA(H^(l1), H^(l2)) between all pairs of layers
     - Output similarity matrix heatmap

  2. UMAP / PCA Visualization:
     - Project specific layers to 2D
     - Color points by true generative parameter

  3. Disentanglement Analysis:
     - t-SNE / UMAP / PCA of pooled representations per layer
     - Color by concept type and by parameter values
     - Quantify separation using silhouette score

Usage:
    python probe/ts_concept_multidim_analysis.py \
        --rep_dir ./results/synthetic/representations/ \
        --output_dir ./results/synthetic/multidim_analysis/

    python probe/ts_concept_multidim_analysis.py \
        --rep_dir ./results/synthetic/representations/ \
        --output_dir ./results/synthetic/multidim_analysis/ \
        --cka_only
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ─────────────────────────────────────────────────────────────────────────────
# CKA
# ─────────────────────────────────────────────────────────────────────────────

def cka_gaussian(X: np.ndarray, Y: np.ndarray, sigma_X: float = None, sigma_Y: float = None) -> float:
    """
    Centered Kernel Alignment (CKA) with Gaussian (RBF) kernels.
    """
    n = X.shape[0]
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    if sigma_X is None:
        dist_X = torch.cdist(torch.from_numpy(X), torch.from_numpy(X))
        triu = torch.triu_indices(n, n, offset=1)
        sigma_X = max(torch.median(dist_X[triu[0], triu[1]]).item(), 1e-6)
    if sigma_Y is None:
        dist_Y = torch.cdist(torch.from_numpy(Y), torch.from_numpy(Y))
        triu = torch.triu_indices(n, n, offset=1)
        sigma_Y = max(torch.median(dist_Y[triu[0], triu[1]]).item(), 1e-6)

    K = np.exp(-np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2) / (2 * sigma_X ** 2))
    L = np.exp(-np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2) / (2 * sigma_Y ** 2))

    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic = np.trace(Kc @ Lc)
    norm = np.sqrt(np.trace(Kc @ Kc) * np.trace(Lc @ Lc))
    if norm < 1e-10:
        return 0.0
    return hsic / norm


def compute_cka_matrix(layer_reps: list[np.ndarray], max_samples: int = 2000) -> np.ndarray:
    """
    Compute CKA similarity matrix across all layers.
    """
    n_layers = len(layer_reps)
    cka = np.zeros((n_layers, n_layers))

    n_total = layer_reps[0].shape[0]
    n_use = min(max_samples, n_total)
    if n_use < n_total:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, n_use, replace=False)
        layer_reps = [rep[idx] for rep in layer_reps]

    for i in range(n_layers):
        for j in range(i, n_layers):
            score = cka_gaussian(layer_reps[i], layer_reps[j])
            cka[i, j] = score
            cka[j, i] = score
        if (i + 1) % 2 == 0 or i == n_layers - 1:
            print(f"    CKA: computed {i+1}/{n_layers} rows")

    return cka


def plot_cka_heatmap(cka_matrix: np.ndarray, output_path: str, n_layers: int):
    """Plot CKA similarity matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("CKA Similarity Matrix Across Timer Layers", fontsize=12, fontweight="bold")
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(n_layers)], fontsize=8)

    for i in range(n_layers):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.5))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved CKA heatmap: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality Reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_dimensions(
    reps: np.ndarray,
    method: str,
    n_components: int = 2,
    perplexity: float = 30.0,
) -> np.ndarray:
    """
    Reduce representations to n_components dimensions.
    """
    reps = StandardScaler().fit_transform(reps)

    if method == "pca":
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(reps)

    elif method == "tsne":
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
            learning_rate="auto",
            init="pca",
        )
        return tsne.fit_transform(reps)

    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
        )
        return reducer.fit_transform(reps)

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_layer_embeddings(
    layer_reps: list[np.ndarray],
    labels: list[dict],
    concept_idx: np.ndarray,
    concepts: list[str],
    output_path: str,
    max_samples: int = 500,
    method: str = "pca",
):
    """
    Plot dimensionality-reduced embeddings for each layer.
    Subplot per layer, colored by concept type.
    """
    n_layers = len(layer_reps)
    max_cols = 3
    n_rows = (n_layers + max_cols - 1) // max_cols

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(max_cols * 3.5, n_rows * 3.5))
    axes = axes.flatten() if n_layers > 1 else [axes]

    for l, rep in enumerate(layer_reps):
        ax = axes[l]

        n_total = rep.shape[0]
        n_use = min(max_samples, n_total)
        rng = np.random.default_rng(l * 42)
        idx = rng.choice(n_total, n_use, replace=False)
        rep_sub = rep[idx]
        ci_sub = concept_idx[idx]

        emb = reduce_dimensions(rep_sub, method=method)

        for c_i, concept in enumerate(concepts):
            mask = ci_sub == c_i
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                label=concept, alpha=0.5, s=8,
            )

        ax.set_title(f"Layer {l}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if l == 0:
            ax.legend(fontsize=5, loc="upper right", ncol=2)

    for j in range(l + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Timer Layer Embeddings ({method.upper()}, colored by concept)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved layer embeddings: {output_path}")


def plot_param_flow(
    layer_reps: list[np.ndarray],
    labels: list[dict],
    concepts: list[str],
    param_name: str,
    output_path: str,
    max_samples: int = 300,
    method: str = "pca",
    concept: str = None,
):
    """
    Plot parameter flow visualization for a specific concept.
    Color points by the parameter value to see if there's smooth gradient.
    """
    n_layers = len(layer_reps)

    if concept:
        target_concepts = [concept]
    else:
        target_concepts = [
            c for c in concepts
            if any(k == param_name for k in next(l for l in labels if l["concept"] == c).keys())
        ]

    fig, axes = plt.subplots(
        len(target_concepts), n_layers,
        figsize=(n_layers * 2.5, len(target_concepts) * 2.5),
        squeeze=False,
    )

    for row, tc in enumerate(target_concepts):
        tc_idx = [i for i, lbl in enumerate(labels) if lbl["concept"] == tc]
        tc_labels = [labels[i] for i in tc_idx]

        param_vals = []
        for lbl in tc_labels:
            if param_name in lbl:
                param_vals.append(lbl[param_name])
            else:
                param_vals.append(0.0)
        param_vals = np.array(param_vals)

        for l, rep in enumerate(layer_reps):
            ax = axes[row, l]
            rep_l = rep[tc_idx]

            n_use = min(max_samples, rep_l.shape[0])
            rng = np.random.default_rng(row * 42 + l)
            idx = rng.choice(rep_l.shape[0], n_use, replace=False)
            rep_sub = rep_l[idx]
            pv_sub = param_vals[idx]

            if rep_sub.shape[0] < 5:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"L{l}: {tc[:12]}", fontsize=7)
                ax.axis("off")
                continue

            try:
                emb = reduce_dimensions(rep_sub, method=method)
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            vmin, vmax = pv_sub.min(), pv_sub.max()
            if param_name in ("freq", "amp", "sigma_before", "sigma_after"):
                cmap = "viridis"
                vmin = max(0, vmin)
            elif param_name in ("phi", "mu", "beta", "warp"):
                cmap = "coolwarm"
                half = max(abs(vmin), abs(vmax))
                vmin, vmax = -half, half
            else:
                if vmin < 0 and vmax > 0:
                    cmap = "coolwarm"
                    half = max(abs(vmin), abs(vmax))
                    vmin, vmax = -half, half
                else:
                    cmap = "viridis"
                    vmin = max(0, vmin)

            scatter = ax.scatter(
                emb[:, 0], emb[:, 1],
                c=pv_sub, cmap=cmap, vmin=vmin, vmax=vmax, s=10, alpha=0.7,
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f"L{l}", fontsize=8, fontweight="bold")
            if l == 0:
                ax.set_ylabel(tc[:12], fontsize=7)

            if l == n_layers - 1:
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=5)
                cbar.set_label(param_name[:8], fontsize=5)

    plt.suptitle(
        f"Timer Parameter Flow: {param_name} ({method.upper()})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved param flow: {output_path}")


def compute_silhouette_scores(
    layer_reps: list[np.ndarray],
    concept_idx: np.ndarray,
    max_samples: int = 2000,
) -> dict[int, float]:
    """Compute silhouette score per layer (concept separation)."""
    scores = {}
    n_total = layer_reps[0].shape[0]
    n_use = min(max_samples, n_total)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_total, n_use, replace=False)
    ci = concept_idx[idx]

    for l, rep in enumerate(layer_reps):
        rep_sub = StandardScaler().fit_transform(rep[idx])
        try:
            score = silhouette_score(rep_sub, ci)
        except Exception:
            score = float("nan")
        scores[l] = score
        print(f"  Layer {l}: silhouette = {score:.4f}")

    return scores


def plot_silhouette_scores(
    scores: dict[int, float],
    output_path: str,
):
    layers = sorted(scores.keys())
    values = [scores[l] for l in layers]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, values, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("Layer Depth", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Concept Disentanglement: Silhouette Score per Timer Layer", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved silhouette: {output_path}")


def compute_procrustes_similarity(
    layer_reps: list[np.ndarray],
    max_samples: int = 2000,
) -> np.ndarray:
    """Compute Procrustes similarity between consecutive layers."""
    from scipy.linalg import orthogonal_procrustes
    n_layers = len(layer_reps)
    sims = np.zeros(n_layers - 1)

    n_total = layer_reps[0].shape[0]
    n_use = min(max_samples, n_total)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_total, n_use, replace=False)

    for i in range(n_layers - 1):
        X = StandardScaler().fit_transform(layer_reps[i][idx])
        Y = StandardScaler().fit_transform(layer_reps[i + 1][idx])
        try:
            R, _ = orthogonal_procrustes(X, Y)
            X_rot = X @ R
            mse = np.mean((X_rot - Y) ** 2)
            sim = 1.0 / (1.0 + mse)
        except Exception:
            sim = float("nan")
        sims[i] = sim
        print(f"  Layer {i} -> {i+1}: procrustes sim = {sim:.4f}")

    return sims


def plot_procrustes_similarity(
    sims: np.ndarray,
    output_path: str,
):
    layers = np.arange(len(sims))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(layers, sims, color="steelblue", alpha=0.7)
    ax.plot(layers, sims, "o-", color="crimson", linewidth=1.5, markersize=4)
    ax.set_xlabel("Layer Transition", fontsize=11)
    ax.set_ylabel("Procrustes Similarity", fontsize=11)
    ax.set_title("Timer Representation Continuity Between Layers", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}->L{i+1}" for i in layers], fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved procrustes: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Multi-Dimensional Analysis of Timer Representations"
    )
    parser.add_argument("--rep_dir", type=str, required=True,
                        help="Directory with layer_*_rep.pt or layer_representations.pt")
    parser.add_argument("--output_dir", type=str, default="./results/synthetic/multidim_analysis/",
                        help="Output directory")
    parser.add_argument("--viz_layers", type=int, nargs="*", default=None,
                        help="Specific layers to visualize (default: all)")
    parser.add_argument("--dim_method", type=str, default="pca",
                        choices=["pca", "tsne", "umap"],
                        help="Dimensionality reduction method")
    parser.add_argument("--max_samples_cka", type=int, default=1500,
                        help="Max samples for CKA computation")
    parser.add_argument("--max_samples_viz", type=int, default=500,
                        help="Max samples for visualization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cka_only", action="store_true",
                        help="Only compute CKA")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Step 4: Multi-Dimensional Analysis (Timer)")
    print("=" * 60)
    print(f"  rep_dir     : {args.rep_dir}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  dim_method  : {args.dim_method}")
    print(f"  viz_layers  : {args.viz_layers}")
    print(f"  cka_only    : {args.cka_only}")
    print("=" * 60)

    # ── 1. Load representations ─────────────────────────────────────────────────
    print("\n[1] Loading representations...")

    combined_path = os.path.join(args.rep_dir, "layer_representations.pt")
    if os.path.exists(combined_path):
        ckpt = torch.load(combined_path, map_location="cpu")
        layer_reps = ckpt["layer_reps"]
        params = ckpt["params"]
        concept_idx = ckpt["concept_idx"].numpy()
        concepts = ckpt["concepts"]
        labels = ckpt["labels"]
        d_model = ckpt["d_model"]
        n_layers_total = ckpt["n_layers"]
    else:
        layer_files = sorted(
            [f for f in os.listdir(args.rep_dir)
             if f.startswith("layer_") and f.endswith("_rep.pt")]
        )
        if not layer_files:
            raise FileNotFoundError(f"No representation files in {args.rep_dir}")
        layer_reps = []
        params = None
        concept_idx = None
        concepts = None
        labels = None
        d_model = None
        for f in layer_files:
            ckpt = torch.load(os.path.join(args.rep_dir, f), map_location="cpu", weights_only=False)
            layer_reps.append(ckpt["z"].numpy())
            if params is None:
                params = ckpt["params"]
                concept_idx = ckpt["concept_idx"].numpy()
                concepts = ckpt["concepts"]
                labels = ckpt["labels"]
                d_model = ckpt["d_model"]
        n_layers_total = len(layer_reps)

    # Convert to numpy
    layer_reps = [rep.numpy() if isinstance(rep, torch.Tensor) else rep for rep in layer_reps]

    n_samples = layer_reps[0].shape[0]
    print(f"  n_samples   : {n_samples}")
    print(f"  n_layers    : {n_layers_total}")
    print(f"  d_model     : {d_model}")
    print(f"  concepts    : {concepts}")

    # Filter layers
    if args.viz_layers is not None:
        layer_indices = [l for l in args.viz_layers if l < n_layers_total]
        layer_reps = [layer_reps[l] for l in layer_indices]
        n_layers = len(layer_indices)
    else:
        layer_indices = list(range(n_layers_total))
        n_layers = n_layers_total

    # ── 2. CKA Analysis ──────────────────────────────────────────────────────
    print("\n[2] Computing CKA similarity matrix...")
    cka_path = os.path.join(args.output_dir, "cka_matrix.npy")
    if os.path.exists(cka_path):
        print("  Loading cached CKA matrix...")
        cka_matrix = np.load(cka_path)
    else:
        cka_matrix = compute_cka_matrix(layer_reps, max_samples=args.max_samples_cka)
        np.save(cka_path, cka_matrix)
        print(f"  Saved CKA matrix: {cka_path}")

    if not args.no_plot:
        plot_cka_heatmap(cka_matrix, os.path.join(args.output_dir, "cka_heatmap.png"), n_layers)

    # ── 3. Layer Embeddings (colored by concept) ──────────────────────────────
    if not args.no_plot and not args.cka_only:
        print("\n[3] Plotting layer embeddings...")
        plot_layer_embeddings(
            layer_reps,
            labels,
            concept_idx,
            concepts,
            os.path.join(args.output_dir, f"layer_embeddings_{args.dim_method}.png"),
            max_samples=args.max_samples_viz,
            method=args.dim_method,
        )

    # ── 4. Parameter Flow ────────────────────────────────────────────────────
    if not args.no_plot and not args.cka_only:
        print("\n[4] Plotting parameter flow...")

        param_concept_pairs = [
            ("phi", "AR1"),
            ("mu", "RandomWalk"),
            ("beta", "DeterministicTrend"),
            ("freq", "Spectral"),
            ("warp", "TimeWarpedSinusoid"),
            ("delta", "LevelShift"),
            ("sigma_after", "VarianceShift"),
        ]
        for param_name, concept in param_concept_pairs:
            tc_indices = [i for i, lbl in enumerate(labels) if lbl["concept"] == concept]
            if len(tc_indices) < 20:
                continue
            try:
                plot_param_flow(
                    layer_reps,
                    labels,
                    concepts,
                    param_name,
                    os.path.join(args.output_dir, f"param_flow_{param_name}_{args.dim_method}.png"),
                    max_samples=args.max_samples_viz,
                    method=args.dim_method,
                    concept=concept,
                )
            except Exception as e:
                print(f"  Warning: could not plot {param_name}: {e}")

    # ── 5. Silhouette Scores ─────────────────────────────────────────────────
    silhouette_scores = None
    if not args.no_plot and not args.cka_only:
        print("\n[5] Computing silhouette scores...")
        silhouette_scores = compute_silhouette_scores(layer_reps, concept_idx)
        plot_silhouette_scores(
            silhouette_scores,
            os.path.join(args.output_dir, "silhouette_scores.png"),
        )

    # ── 6. Procrustes Analysis ────────────────────────────────────────────────
    sims = None
    if not args.no_plot and not args.cka_only:
        print("\n[6] Computing Procrustes similarity...")
        sims = compute_procrustes_similarity(layer_reps)
        plot_procrustes_similarity(
            sims,
            os.path.join(args.output_dir, "procrustes_similarity.png"),
        )

    # ── 7. Save summary ─────────────────────────────────────────────────────
    print("\n[7] Saving analysis summary...")
    summary_path = os.path.join(args.output_dir, "analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Multi-Dimensional Analysis Summary (Timer)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"n_samples      : {n_samples}\n")
        f.write(f"n_layers       : {n_layers_total}\n")
        f.write(f"d_model        : {d_model}\n")
        f.write(f"dim_method     : {args.dim_method}\n")
        f.write(f"concepts       : {concepts}\n\n")

        f.write("CKA Matrix (diagonal = 1.0):\n")
        f.write(f"  min off-diag: {np.min(np.where(np.eye(n_layers) == 0, cka_matrix, 1.0)):.4f}\n")
        f.write(f"  max off-diag: {np.max(cka_matrix):.4f}\n")
        f.write(f"  mean off-diag: {np.mean(cka_matrix):.4f}\n\n")

        if silhouette_scores is not None:
            f.write("Silhouette Scores (concept separation):\n")
            for l, s in silhouette_scores.items():
                f.write(f"  Layer {l}: {s:.4f}\n")

    print(f"  Saved: {summary_path}")
    print(f"\n[Done] Step 4 complete.")


if __name__ == "__main__":
    main()
