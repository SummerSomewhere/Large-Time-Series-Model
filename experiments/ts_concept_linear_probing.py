#!/usr/bin/env python3
"""
Step 3: Linear Probing Training

Trains a linear probe (one per model layer) to predict the generative
parameters of each synthetic concept from the pooled Timer representations.

Design:
  - For each layer l, define a linear probe W^(l) z^(l) + b^(l)
    that maps z^(l) in R^D to the parameter vector theta in R^P
  - 80/20 train/val split (stratified by concept)
  - MSE loss, only train linear probe (Timer is frozen)
  - 100 epochs
  - Output: MSE per layer per concept, "layer depth vs MSE" plot

Usage:
    python experiments/ts_concept_linear_probing.py \
        --rep_dir ./results/synthetic/representations/ \
        --output_dir ./results/synthetic/linear_probing/ \
        --epochs 100 --lr 1e-3 --batch_size 128

    # Run on specific layers
    python experiments/ts_concept_linear_probing.py \
        --rep_dir ./results/synthetic/representations/ \
        --layers 0 3 6 \
        --epochs 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class LinearProbe(nn.Module):
    """Single linear layer: R^D -> R^P"""

    def __init__(self, d_model: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_model, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def concept_collate_fn(
    z: torch.Tensor,
    params: torch.Tensor,
    concept_idx: torch.Tensor,
    train_indices: list,
    val_indices: list,
) -> tuple:
    """Split representations and params into train/val sets."""
    z_train = z[train_indices]
    z_val = z[val_indices]
    p_train = params[train_indices]
    p_val = params[val_indices]
    c_train = concept_idx[train_indices]
    c_val = concept_idx[val_indices]
    return z_train, z_val, p_train, p_val, c_train, c_val


def build_concept_masks(concept_idx: torch.Tensor, concepts: list[str]) -> dict:
    """Build per-concept boolean masks."""
    masks = {}
    for i, concept in enumerate(concepts):
        masks[concept] = concept_idx == i
    return masks


def train_linear_probe(
    z_train: torch.Tensor,
    p_train: torch.Tensor,
    z_val: torch.Tensor,
    p_val: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = False,
) -> tuple[float, list[float]]:
    """
    Train a linear probe on the given data.

    Returns:
        final_val_mse: final validation MSE
        train_mse_curve: list of train MSE per epoch
    """
    in_dim = z_train.shape[1]
    out_dim = p_train.shape[1]

    probe = LinearProbe(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_train = z_train.shape[0]
    batch_size = min(256, n_train)

    train_mse_curve = []
    for epoch in range(n_epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i : i + batch_size]
            xb = z_train[batch_idx].to(device)
            yb = p_train[batch_idx].to(device)

            optimizer.zero_grad()
            pred = probe(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_mse = epoch_loss / max(n_batches, 1)

        # Validation
        probe.eval()
        with torch.no_grad():
            val_pred = probe(z_val.to(device))
            val_mse = criterion(val_pred, p_val.to(device)).item()

        train_mse_curve.append(val_mse)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}")

    probe.eval()
    with torch.no_grad():
        final_pred = probe(z_val.to(device))
        final_val_mse = criterion(final_pred, p_val.to(device)).item()

    return final_val_mse, train_mse_curve


def train_per_concept(
    z_train: torch.Tensor,
    params_train: torch.Tensor,
    concept_idx_train: torch.Tensor,
    z_val: torch.Tensor,
    params_val: torch.Tensor,
    concept_idx_val: torch.Tensor,
    concepts: list[str],
    n_epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    """
    Train a separate linear probe for each concept.
    Only uses samples belonging to that concept.

    Returns:
        dict: concept -> final val MSE
    """
    results = {}
    for i, concept in enumerate(concepts):
        train_mask = concept_idx_train == i
        val_mask = concept_idx_val == i

        z_tr = z_train[train_mask]
        p_tr = params_train[train_mask]
        z_vl = z_val[val_mask]
        p_vl = params_val[val_mask]

        if z_tr.shape[0] < 10 or z_vl.shape[0] < 5:
            results[concept] = float("nan")
            continue

        mse, _ = train_linear_probe(
            z_tr, p_tr, z_vl, p_vl,
            n_epochs=n_epochs, lr=lr, device=device, verbose=False
        )
        results[concept] = mse
        print(f"  {concept:<25s}: val_mse={mse:.6f}  (n_train={z_tr.shape[0]}, n_val={z_vl.shape[0]})")

    return results


def plot_layer_mse_curve(
    layer_mses: list[float],
    layer_concept_mses: dict[str, list[float]],
    concepts: list[str],
    output_path: str,
):
    """Plot layer depth vs MSE curve."""
    n_layers = len(layer_mses)
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall mean MSE
    ax = axes[0]
    ax.plot(layers, layer_mses, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("Layer Depth", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("Layer Depth vs. Prediction MSE (Linear Probe)", fontsize=12)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Right: per-concept MSE curves
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))
    for c_idx, concept in enumerate(concepts):
        mses = layer_concept_mses.get(concept, [])
        if len(mses) == n_layers:
            ax.plot(layers, mses, "o-", label=concept,
                    color=colors[c_idx], linewidth=1.5, markersize=4)
    ax.set_xlabel("Layer Depth", fontsize=11)
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.set_title("Per-Concept MSE vs. Layer Depth", fontsize=12)
    ax.set_xticks(layers)
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved layer MSE curve: {output_path}")


def plot_epoch_curves(
    epoch_curves: dict[int, list[float]],
    output_path: str,
):
    """Plot training curves per layer."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for layer, curve in epoch_curves.items():
        ax.plot(curve, label=f"Layer {layer}", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation MSE", fontsize=11)
    ax.set_title("Linear Probe Training Curves (per Layer)", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved epoch curves: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Linear Probing on Timer Representations"
    )
    parser.add_argument("--rep_dir", type=str, required=True,
                        help="Directory containing layer_*_rep.pt files from Step 2")
    parser.add_argument("--output_dir", type=str, default="./results/synthetic/linear_probing/",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Specific layers to train probes on (default: all)")
    parser.add_argument("--per_concept", action="store_true",
                        help="Train separate probes per concept (default: global probe)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("Step 3: Linear Probing Training")
    print("=" * 60)
    print(f"  rep_dir     : {args.rep_dir}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  epochs      : {args.epochs}")
    print(f"  lr          : {args.lr}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  train_ratio : {args.train_ratio}")
    print(f"  device      : {device}")
    print(f"  per_concept : {args.per_concept}")
    print("=" * 60)

    # ── 1. Load representation data ──────────────────────────────────────────
    print("\n[1] Loading representation data...")

    # Try to load the combined file first
    combined_path = os.path.join(args.rep_dir, "layer_representations.pt")
    if os.path.exists(combined_path):
        ckpt = torch.load(combined_path, map_location="cpu")
        layer_reps = ckpt["layer_reps"]
        params = ckpt["params"]
        concept_idx = ckpt["concept_idx"]
        concepts = ckpt["concepts"]
        d_model = ckpt["d_model"]
        n_layers_total = ckpt["n_layers"]
        print(f"  Loaded combined file with {n_layers_total} layers")
    else:
        # Load individual layer files
        layer_files = sorted(
            [f for f in os.listdir(args.rep_dir) if f.startswith("layer_") and f.endswith("_rep.pt")]
        )
        if not layer_files:
            raise FileNotFoundError(f"No representation files found in {args.rep_dir}")
        layer_reps = []
        params = None
        concept_idx = None
        concepts = None
        for f in layer_files:
            ckpt = torch.load(os.path.join(args.rep_dir, f), map_location="cpu")
            layer_reps.append(ckpt["z"])
            if params is None:
                params = ckpt["params"]
                concept_idx = ckpt["concept_idx"]
                concepts = ckpt["concepts"]
                d_model = ckpt["d_model"]
        n_layers_total = len(layer_reps)
        print(f"  Loaded {n_layers_total} individual layer files")

    n_samples = layer_reps[0].shape[0]
    in_dim = layer_reps[0].shape[1]
    out_dim = params.shape[1]
    print(f"  n_samples   : {n_samples}")
    print(f"  in_dim (D)  : {in_dim}")
    print(f"  out_dim (P) : {out_dim}")
    print(f"  concepts    : {concepts}")

    # Determine which layers to train
    if args.layers is not None:
        layer_indices = [l for l in args.layers if l < n_layers_total]
        layer_reps = [layer_reps[l] for l in layer_indices]
        n_layers = len(layer_indices)
    else:
        layer_indices = list(range(n_layers_total))
        n_layers = n_layers_total

    print(f"  Training on layers: {layer_indices}")

    # ── 2. Train/val split (stratified by concept) ────────────────────────────
    print(f"\n[2] Creating {args.train_ratio:.0%}/{1-args.train_ratio:.0%} train/val split...")

    stratify_labels = concept_idx.numpy()
    train_idx, val_idx = train_test_split(
        np.arange(n_samples),
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=stratify_labels,
    )
    train_idx = torch.from_numpy(train_idx)
    val_idx = torch.from_numpy(val_idx)

    z_train = {l: rep[train_idx] for l, rep in zip(layer_indices, layer_reps)}
    z_val = {l: rep[val_idx] for l, rep in zip(layer_indices, layer_reps)}
    p_train = params[train_idx]
    p_val = params[val_idx]
    c_train = concept_idx[train_idx]
    c_val = concept_idx[val_idx]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # ── 3. Train probes per layer ────────────────────────────────────────────
    print(f"\n[3] Training linear probes ({args.epochs} epochs)...")
    layer_mse_results = {}
    layer_concept_mse_results = {c: [] for c in concepts}
    epoch_curves = {}

    for l_idx, layer in enumerate(layer_indices):
        print(f"\n  ── Layer {layer} ({l_idx+1}/{n_layers}) ──")

        z_tr = z_train[layer]
        z_vl = z_val[layer]

        if args.per_concept:
            # Train separate probe per concept
            concept_mses = train_per_concept(
                z_tr, p_train, c_train,
                z_vl, p_val, c_val,
                concepts, args.epochs, args.lr, device
            )
            # Compute weighted average MSE
            total_n = 0
            weighted_mse = 0.0
            for c, mse in concept_mses.items():
                if not np.isnan(mse):
                    n_c = (c_val == concepts.index(c)).sum().item()
                    weighted_mse += mse * n_c
                    total_n += n_c
            avg_mse = weighted_mse / total_n if total_n > 0 else float("nan")
            layer_mse_results[layer] = avg_mse

            for c in concepts:
                layer_concept_mse_results[c].append(concept_mses.get(c, float("nan")))
        else:
            # Train global probe
            mse, curve = train_linear_probe(
                z_tr, p_train, z_vl, p_val,
                n_epochs=args.epochs, lr=args.lr, device=device, verbose=True
            )
            layer_mse_results[layer] = mse
            epoch_curves[layer] = curve
            print(f"  Layer {layer}: overall val_mse = {mse:.6f}")

    # ── 4. Print results table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Layer':>6} | {'Val MSE':>12}")
    print("-" * 70)
    for layer in layer_indices:
        print(f"{layer:>6} | {layer_mse_results[layer]:>12.6f}")
    print("=" * 70)

    # ── 5. Save results ───────────────────────────────────────────────────────
    print("\n[5] Saving results...")
    results_path = os.path.join(args.output_dir, "linear_probing_results.pt")
    torch.save(
        {
            "layer_mse": layer_mse_results,
            "layer_concept_mse": layer_concept_mse_results if args.per_concept else None,
            "epoch_curves": epoch_curves if not args.per_concept else None,
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "n_layers": n_layers,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "per_concept": args.per_concept,
                "layer_indices": layer_indices,
            },
        },
        results_path,
    )
    print(f"  Saved: {results_path}")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        mse_curve_path = os.path.join(args.output_dir, "layer_mse_curve.png")
        plot_layer_mse_curve(
            [layer_mse_results[l] for l in layer_indices],
            layer_concept_mse_results,
            concepts,
            mse_curve_path,
        )

        if not args.per_concept:
            epoch_path = os.path.join(args.output_dir, "epoch_curves.png")
            plot_epoch_curves(epoch_curves, epoch_path)

    print(f"\n[Done] Step 3 complete.")
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
