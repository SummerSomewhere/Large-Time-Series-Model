#!/usr/bin/env python3
"""
Step 3: Linear Probing Training (Timer version)

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
    python probe/ts_concept_linear_probing.py \
        --rep_dir ./results/synthetic/representations/ \
        --output_dir ./results/synthetic/linear_probing/ \
        --epochs 100 --lr 1e-3 --batch_size 128

    # Run on specific layers
    python probe/ts_concept_linear_probing.py \
        --rep_dir ./results/synthetic/representations/ \
        --layers 0 3 6 \
        --epochs 100
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
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from probe.ts_concept_synthetic_dataset import CONCEPT_PARAM_SPEC


class LinearProbe(nn.Module):
    """Single linear layer: R^D -> R^P"""

    def __init__(self, d_model: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_model, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    z_train: torch.Tensor,
    p_train: torch.Tensor,
    z_val: torch.Tensor,
    p_val: torch.Tensor,
    n_epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = False,
    out_dim: int = None,
) -> tuple[float, list[float], torch.Tensor]:
    """
    Train a linear probe on the given data.

    Returns:
        final_val_mse: final validation MSE (averaged over predicted dims)
        train_mse_curve: list of validation MSE per epoch
        val_pred: final validation predictions [N_val, out_dim]
    """
    in_dim = z_train.shape[1]
    out_dim = out_dim or p_train.shape[1]

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

    return final_val_mse, train_mse_curve, final_pred.detach()


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
) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Train a separate linear probe for each concept.
    Only predicts the parameters relevant to that concept (no padding dims).
    """
    mse_results = {}
    val_preds = {}
    val_targets = {}
    for i, concept in enumerate(concepts):
        train_mask = concept_idx_train == i
        val_mask = concept_idx_val == i

        z_tr = z_train[train_mask]
        z_vl = z_val[val_mask]

        active_dims, _ = CONCEPT_PARAM_SPEC[concept]
        p_tr = params_train[train_mask][:, active_dims]
        p_vl = params_val[val_mask][:, active_dims]

        if z_tr.shape[0] < 10 or z_vl.shape[0] < 5:
            mse_results[concept] = float("nan")
            val_preds[concept] = torch.zeros(0, len(active_dims))
            val_targets[concept] = torch.zeros(0, len(active_dims))
            continue

        mse, _, pred = train_linear_probe(
            z_tr, p_tr, z_vl, p_vl,
            n_epochs=n_epochs, lr=lr, device=device, verbose=False
        )
        mse_results[concept] = mse
        val_preds[concept] = pred.detach().cpu()
        val_targets[concept] = p_vl.detach().cpu()
        print(f"  {concept:<25s}: val_mse={mse:.6f}  (dims={active_dims}, n_train={z_tr.shape[0]}, n_val={z_vl.shape[0]})")

    return mse_results, val_preds, val_targets


def compute_r2_from_predictions(
    layer_concept_preds: dict[int, dict[str, torch.Tensor]],
    layer_concept_targets: dict[int, dict[str, torch.Tensor]],
    concepts: list[str],
    layer_indices: list[int],
) -> tuple[list[float], dict[str, list[float]]]:
    """
    Compute R^2 scores from per-concept predictions.
    """
    overall_r2 = []
    concept_r2 = {c: [] for c in concepts}

    for layer in layer_indices:
        preds_per_c = layer_concept_preds[layer]
        targets_per_c = layer_concept_targets[layer]

        layer_ss_res_total = 0.0
        layer_ss_tot_total = 0.0

        for c in concepts:
            pred = preds_per_c.get(c, None)
            target = targets_per_c.get(c, None)
            if pred is None or target is None or pred.shape[0] < 5:
                concept_r2[c].append(float("nan"))
                continue

            ss_res = ((pred - target) ** 2).sum().item()
            ss_tot = ((target - target.mean(dim=0)) ** 2).sum().item()
            r2 = 1.0 - ss_res / (ss_tot + 1e-8)
            concept_r2[c].append(r2)

            layer_ss_res_total += ss_res
            layer_ss_tot_total += ss_tot

        overall_r2.append(1.0 - layer_ss_res_total / (layer_ss_tot_total + 1e-8))

    return overall_r2, concept_r2


def plot_layer_mse_curve(
    layer_mses: list[float],
    layer_concept_mses: dict[str, list[float]],
    concepts: list[str],
    output_path: str,
    layer_indices: list[int] = None,
    layer_r2: list[float] = None,
    layer_concept_r2: dict[str, list[float]] = None,
):
    """Plot layer depth vs MSE and R^2 curves."""
    n_layers = len(layer_mses)
    layers = layer_indices if layer_indices is not None else np.arange(n_layers)

    if layer_r2 is not None:
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
        ax_mse_all = fig.add_subplot(gs[0, 0])
        ax_mse_per = fig.add_subplot(gs[0, 1])
        ax_r2_all = fig.add_subplot(gs[0, 2])
        ax_r2_per = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_mse_all = axes[0]
        ax_mse_per = axes[1]
        ax_r2_all = ax_r2_per = None

    ax = ax_mse_all
    ax.plot(layers, layer_mses, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("Layer Depth", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("Layer Depth vs. Prediction MSE (Timer Linear Probe)", fontsize=12)
    ax.set_xticks(layers)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.grid(True, alpha=0.3)

    ax = ax_mse_per
    colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))
    for c_idx, concept in enumerate(concepts):
        mses = layer_concept_mses.get(concept, [])
        if len(mses) == n_layers:
            ax.plot(layers, mses, "o-", label=concept,
                    color=colors[c_idx], linewidth=1.5, markersize=4)
    ax.set_xlabel("Layer Depth", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("Per-Concept MSE vs. Layer Depth", fontsize=12)
    ax.set_xticks(layers)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    if layer_r2 is not None:
        ax = ax_r2_all
        ax.plot(layers, layer_r2, "o-", color="darkorange", linewidth=2, markersize=6)
        ax.set_xlabel("Layer Depth", fontsize=11)
        ax.set_ylabel(r"$R^2$", fontsize=11)
        ax.set_title(r"Layer Depth vs. $R^2$ (Timer Linear Probe)", fontsize=12)
        ax.set_xticks(layers)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.grid(True, alpha=0.3)

        ax = ax_r2_per
        for c_idx, concept in enumerate(concepts):
            r2s = layer_concept_r2.get(concept, [])
            if len(r2s) == n_layers:
                ax.plot(layers, r2s, "o-", label=concept,
                        color=colors[c_idx], linewidth=1.5, markersize=4)
        ax.set_xlabel("Layer Depth", fontsize=11)
        ax.set_ylabel(r"$R^2$", fontsize=11)
        ax.set_title(r"Per-Concept $R^2$ vs. Layer Depth", fontsize=12)
        ax.set_xticks(layers)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.legend(fontsize=7, loc="lower right")
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
    ax.set_title("Linear Probe Training Curves (Timer, per Layer)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved epoch curves: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# torchrun multi-GPU worker
# ─────────────────────────────────────────────────────────────────────────────

def _spawn_worker(
    rank,
    world_size,
    layer_reps,
    params,
    concept_idx,
    concepts,
    n_epochs,
    lr,
    per_concept,
    seed,
    output_dir,
):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    n_samples = params.shape[0]
    my_samples = torch.arange(n_samples)[rank::world_size]
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    print(f"[Rank {rank}] Training on {len(my_samples)} samples", flush=True)

    z_tr = {l: layer_reps[l][my_samples] for l in layer_reps}
    p_tr = params[my_samples]
    c_tr = concept_idx[my_samples]

    all_train_idx, all_val_idx = [], []
    for c_idx, concept in enumerate(concepts):
        mask = (c_tr == c_idx).numpy()
        n_c = mask.sum()
        split = np.random.RandomState(seed + rank).permutation(n_c)
        tr_n = int(n_c * 0.8)
        global_train = np.where(mask)[0][split[:tr_n]]
        global_val = np.where(mask)[0][split[tr_n:]]
        all_train_idx.append(global_train)
        all_val_idx.append(global_val)

    tr_idx = torch.from_numpy(np.concatenate(all_train_idx))
    vl_idx = torch.from_numpy(np.concatenate(all_val_idx))

    layer_mse_results = {}
    layer_concept_mse_results = {c: [] for c in concepts}
    layer_concept_val_preds = {}
    layer_concept_val_targets = {}

    for layer, z_layer in z_tr.items():
        z_tr_l = z_layer[tr_idx]
        z_vl_l = z_layer[vl_idx]
        p_tr_l = p_tr[tr_idx]
        p_vl_l = p_tr[vl_idx]
        c_tr_l = c_tr[tr_idx]
        c_vl_l = c_tr[vl_idx]

        if per_concept:
            concept_mses, val_preds_per_c, val_targets_per_c = train_per_concept(
                z_tr_l, p_tr_l, c_tr_l,
                z_vl_l, p_vl_l, c_vl_l,
                concepts, n_epochs, lr, device
            )
            total_n = 0
            weighted_mse = 0.0
            for c, mse in concept_mses.items():
                if not np.isnan(mse):
                    n_c = (c_vl_l == concepts.index(c)).sum().item()
                    weighted_mse += mse * n_c
                    total_n += n_c
            avg_mse = weighted_mse / total_n if total_n > 0 else float("nan")
            layer_mse_results[layer] = avg_mse
            for c in concepts:
                layer_concept_mse_results[c].append(concept_mses.get(c, float("nan")))
            layer_concept_val_preds[layer] = val_preds_per_c
            layer_concept_val_targets[layer] = val_targets_per_c
        else:
            mse, curve, val_pred = train_linear_probe(
                z_tr_l, p_tr_l, z_vl_l, p_vl_l,
                n_epochs=n_epochs, lr=lr, device=device, verbose=True
            )
            layer_mse_results[layer] = mse
            print(f"  Layer {layer}: overall mse = {mse:.6f}")

    rank_path = os.path.join(output_dir, f"lp_rank{rank}.pt")
    torch.save(
        {
            "rank": rank,
            "world_size": world_size,
            "layer_mse": layer_mse_results,
            "layer_concept_mse": layer_concept_mse_results if per_concept else None,
            "layer_concept_val_preds": layer_concept_val_preds if per_concept else None,
            "layer_concept_val_targets": layer_concept_val_targets if per_concept else None,
        },
        rank_path,
    )
    print(f"[Rank {rank}] Saved: {rank_path}", flush=True)


def _run_worker_single_process(args, output_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.torchrun_rank}")

    combined_path = os.path.join(args.rep_dir, "layer_representations.pt")
    if os.path.exists(combined_path):
        ckpt = torch.load(combined_path, map_location="cpu", weights_only=False)
        layer_reps = ckpt["layer_reps"]
        params = ckpt["params"]
        concept_idx = ckpt["concept_idx"]
        concepts = ckpt["concepts"]
        d_model = ckpt["d_model"]
        n_layers_total = ckpt["n_layers"]
        layer_indices = list(range(n_layers_total))
    else:
        layer_files = sorted(
            [f for f in os.listdir(args.rep_dir) if f.startswith("layer_") and f.endswith("_rep.pt")]
        )
        layer_data = {}
        params = None
        concept_idx = None
        concepts = None
        for f in layer_files:
            ckpt = torch.load(os.path.join(args.rep_dir, f), map_location="cpu", weights_only=False)
            real_layer = ckpt["layer"]
            layer_data[real_layer] = {"z": ckpt["z"]}
            if params is None:
                params = ckpt["params"]
                concept_idx = ckpt["concept_idx"]
                concepts = ckpt["concepts"]
        sorted_layers = sorted(layer_data.keys())
        layer_reps = {l: layer_data[l]["z"] for l in sorted_layers}
        layer_indices = sorted_layers
        n_layers_total = len(layer_reps)

    if args.layers is not None:
        available = set(layer_indices)
        layer_indices = [l for l in args.layers if l in available]
        orig_order = {l: i for i, l in enumerate(layer_indices)}
        layer_reps = {l: layer_reps[l] for l in layer_indices}

    n_samples = params.shape[0]

    my_samples = torch.arange(n_samples)[args.torchrun_rank::args.torchrun_world_size]
    print(f"[Rank {args.torchrun_rank}] Training on {len(my_samples)} samples", flush=True)

    z_tr = {l: layer_reps[l][my_samples] for l in layer_indices}
    p_tr = params[my_samples]
    c_tr = concept_idx[my_samples]

    all_train_idx, all_val_idx = [], []
    for c_idx, concept in enumerate(concepts):
        mask = (c_tr == c_idx).numpy()
        n_c = mask.sum()
        split = np.random.RandomState(args.seed + args.torchrun_rank).permutation(n_c)
        tr_n = int(n_c * args.train_ratio)
        global_train = np.where(mask)[0][split[:tr_n]]
        global_val = np.where(mask)[0][split[tr_n:]]
        all_train_idx.append(global_train)
        all_val_idx.append(global_val)

    tr_idx = torch.from_numpy(np.concatenate(all_train_idx))
    vl_idx = torch.from_numpy(np.concatenate(all_val_idx))
    print(f"  train={len(tr_idx)}, val={len(vl_idx)}", flush=True)

    layer_mse_results = {}
    layer_concept_mse_results = {c: [] for c in concepts}
    layer_concept_val_preds = {}
    layer_concept_val_targets = {}

    for layer in layer_indices:
        z_tr_l = z_tr[layer][tr_idx]
        z_vl_l = z_tr[layer][vl_idx]
        p_tr_l = p_tr[tr_idx]
        p_vl_l = p_tr[vl_idx]
        c_tr_l = c_tr[tr_idx]
        c_vl_l = c_tr[vl_idx]

        if args.per_concept:
            concept_mses, val_preds_per_c, val_targets_per_c = train_per_concept(
                z_tr_l, p_tr_l, c_tr_l,
                z_vl_l, p_vl_l, c_vl_l,
                concepts, args.epochs, args.lr, device
            )
            total_n = 0
            weighted_mse = 0.0
            for c, mse in concept_mses.items():
                if not np.isnan(mse):
                    n_c = (c_vl_l == concepts.index(c)).sum().item()
                    weighted_mse += mse * n_c
                    total_n += n_c
            avg_mse = weighted_mse / total_n if total_n > 0 else float("nan")
            layer_mse_results[layer] = avg_mse
            for c in concepts:
                layer_concept_mse_results[c].append(concept_mses.get(c, float("nan")))
            layer_concept_val_preds[layer] = val_preds_per_c
            layer_concept_val_targets[layer] = val_targets_per_c
        else:
            mse, curve, val_pred = train_linear_probe(
                z_tr_l, p_tr_l, z_vl_l, p_vl_l,
                n_epochs=args.epochs, lr=args.lr, device=device, verbose=True
            )
            layer_mse_results[layer] = mse
            print(f"  Layer {layer}: overall mse = {mse:.6f}", flush=True)

    rank_path = os.path.join(output_dir, f"lp_rank{args.torchrun_rank}.pt")
    torch.save(
        {
            "rank": args.torchrun_rank,
            "world_size": args.torchrun_world_size,
            "layer_mse": layer_mse_results,
            "layer_concept_mse": layer_concept_mse_results if args.per_concept else None,
            "layer_concept_val_preds": layer_concept_val_preds if args.per_concept else None,
            "layer_concept_val_targets": layer_concept_val_targets if args.per_concept else None,
        },
        rank_path,
    )
    print(f"[Rank {args.torchrun_rank}] Saved: {rank_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Linear Probing on Timer Representations"
    )
    parser.add_argument("--rep_dir", type=str, required=True,
                        help="Directory with layer_*_rep.pt or layer_representations.pt")
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
    parser.add_argument("--torchrun", action="store_true", default=False,
                        help="Enable torchrun multi-GPU mode: each rank trains a subset of layers")
    parser.add_argument("--torchrun_rank", type=int, default=None,
                        help="Internal: rank of this worker (set by spawn)")
    parser.add_argument("--torchrun_world_size", type=int, default=None,
                        help="Internal: world size (set by spawn)")
    parser.add_argument("--subproc_output_dir", type=str, default=None,
                        help="Internal: output_dir for subprocess workers (overrides run_ timestamp subdir)")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes (for torchrun, default: 1)")
    parser.add_argument("--nproc_per_node", type=int, default=None,
                        help="Processes per node (for torchrun, default: auto from CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subdir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_subdir, exist_ok=True)
    output_dir = run_subdir
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if args.subproc_output_dir is not None:
        output_dir = args.subproc_output_dir
        torch.cuda.set_device(args.torchrun_rank)
        device = torch.device(f"cuda:{args.torchrun_rank}")
        _run_worker_single_process(args, output_dir)
        return

    print("=" * 60)
    print("Step 3: Linear Probing Training (Timer)")
    print("=" * 60)
    print(f"  rep_dir     : {args.rep_dir}")
    print(f"  output_dir  : {output_dir}")
    print(f"  epochs      : {args.epochs}")
    print(f"  lr          : {args.lr}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  train_ratio : {args.train_ratio}")
    print(f"  device      : {device}")
    print(f"  per_concept : {args.per_concept}")
    print("=" * 60)

    # ── 1. Load representation data ──────────────────────────────────────────
    print("\n[1] Loading representation data...")

    combined_path = os.path.join(args.rep_dir, "layer_representations.pt")
    if os.path.exists(combined_path):
        ckpt = torch.load(combined_path, map_location="cpu", weights_only=False)
        layer_reps = ckpt["layer_reps"]
        params = ckpt["params"]
        concept_idx = ckpt["concept_idx"]
        concepts = ckpt["concepts"]
        d_model = ckpt["d_model"]
        n_layers_total = ckpt["n_layers"]
        layer_indices = list(range(n_layers_total))
        print(f"  Loaded combined file with {n_layers_total} layers, d_model={d_model}")
    else:
        layer_files = sorted(
            [f for f in os.listdir(args.rep_dir) if f.startswith("layer_") and f.endswith("_rep.pt")]
        )
        if not layer_files:
            raise FileNotFoundError(f"No representation files found in {args.rep_dir}")
        layer_data = {}
        params = None
        concept_idx = None
        concepts = None
        d_model = None
        for f in layer_files:
            ckpt = torch.load(os.path.join(args.rep_dir, f), map_location="cpu", weights_only=False)
            real_layer = ckpt["layer"]
            layer_data[real_layer] = {"z": ckpt["z"]}
            if params is None:
                params = ckpt["params"]
                concept_idx = ckpt["concept_idx"]
                concepts = ckpt["concepts"]
                d_model = ckpt["d_model"]

        sorted_layers = sorted(layer_data.keys())
        layer_reps = [layer_data[l]["z"] for l in sorted_layers]
        layer_indices = sorted_layers
        n_layers_total = len(layer_reps)
        print(f"  Loaded {n_layers_total} individual layer files: layers {sorted_layers}")

    n_samples = layer_reps[0].shape[0]
    in_dim = layer_reps[0].shape[1]
    out_dim = params.shape[1]
    print(f"  n_samples   : {n_samples}")
    print(f"  in_dim (D)  : {in_dim}")
    print(f"  out_dim (P) : {out_dim}")
    print(f"  concepts    : {concepts}")

    # Filter to requested layers
    if args.layers is not None:
        available = set(layer_indices)
        layer_indices = [l for l in args.layers if l in available]
        orig_order = {l: i for i, l in enumerate(layer_indices)}
        layer_reps = [layer_reps[orig_order[l]] for l in layer_indices]

    n_layers = len(layer_indices)
    print(f"  Training on layers: {layer_indices}")

    # ── 2. Train/val split (per concept) ─────────────────────────────────
    print(f"\n[2] Creating {args.train_ratio:.0%}/{1-args.train_ratio:.0%} train/val split (per concept)...")

    all_train_idx, all_val_idx = [], []
    for c_idx, concept in enumerate(concepts):
        mask = (concept_idx == c_idx).numpy()
        n_c = mask.sum()
        split = np.random.RandomState(args.seed).permutation(n_c)
        tr_n = int(n_c * args.train_ratio)
        global_train = np.where(mask)[0][split[:tr_n]]
        global_val = np.where(mask)[0][split[tr_n:]]
        all_train_idx.append(global_train)
        all_val_idx.append(global_val)
        print(f"  {concept:<25s}: n_train={tr_n}, n_val={n_c - tr_n}")

    train_idx = torch.from_numpy(np.concatenate(all_train_idx))
    val_idx = torch.from_numpy(np.concatenate(all_val_idx))

    z_train = {l: rep[train_idx] for l, rep in zip(layer_indices, layer_reps)}
    z_val = {l: rep[val_idx] for l, rep in zip(layer_indices, layer_reps)}
    p_train = params[train_idx]
    p_val = params[val_idx]
    c_train = concept_idx[train_idx]
    c_val = concept_idx[val_idx]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # ── 3a. Multi-GPU dispatch via subprocess ─────────────────────────────
    if args.torchrun_rank is not None and args.torchrun_world_size is not None:
        _run_worker_single_process(args, output_dir)
        return

    if args.torchrun:
        nproc = args.nproc_per_node or max(1, len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
        world_size = nproc

        print(f"\n[torchrun] Spawning {world_size} workers ...")
        import subprocess as _sp

        workers = []
        for rank in range(world_size):
            env = os.environ.copy()
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank)
            env["WORLD_SIZE"] = str(world_size)
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = "29500"
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--rep_dir", args.rep_dir, "--output_dir", output_dir,
                "--epochs", str(args.epochs), "--lr", str(args.lr),
                "--batch_size", str(args.batch_size),
                "--seed", str(args.seed), "--train_ratio", str(args.train_ratio),
                "--device", args.device,
                "--torchrun",
                "--subproc_output_dir", output_dir,
                "--torchrun_rank", str(rank),
                "--torchrun_world_size", str(world_size),
                "--layers",
            ] + [str(l) for l in layer_indices]
            if args.per_concept:
                cmd.append("--per_concept")
            if args.no_plot:
                cmd.append("--no_plot")
            p = _sp.Popen(cmd, env=env)
            workers.append(p)
        for p in workers:
            p.wait()

        print("\n[torchrun] All workers finished. Merging results ...")

        all_mse = {}
        all_concept_mse = {c: [] for c in concepts}
        all_concept_val_preds = {}
        all_concept_val_targets = {}

        for rank in range(world_size):
            rank_path = os.path.join(output_dir, f"lp_rank{rank}.pt")
            if os.path.exists(rank_path):
                ckpt = torch.load(rank_path, map_location="cpu")
                for layer, mse_val in ckpt["layer_mse"].items():
                    if layer not in all_mse:
                        all_mse[layer] = []
                    all_mse[layer].append(mse_val)
                if args.per_concept and ckpt.get("layer_concept_mse"):
                    for c in concepts:
                        all_concept_mse[c].extend(ckpt["layer_concept_mse"].get(c, []))
                if ckpt.get("layer_concept_val_preds"):
                    all_concept_val_preds.update(ckpt["layer_concept_val_preds"])
                if ckpt.get("layer_concept_val_targets"):
                    all_concept_val_targets.update(ckpt["layer_concept_val_targets"])

        layer_mse_results = {layer: np.mean(mses) for layer, mses in all_mse.items()}
        layer_concept_mse_results = all_concept_mse
        layer_concept_val_preds = all_concept_val_preds
        layer_concept_val_targets = all_concept_val_targets
        epoch_curves = {}

        layer_indices_sorted = sorted(layer_mse_results.keys())
        layer_r2, layer_concept_r2 = compute_r2_from_predictions(
            layer_concept_val_preds, layer_concept_val_targets, concepts, layer_indices_sorted
        )
        print("\n" + "=" * 85)
        print(f"{'Layer':>6} | {'Val MSE':>12} | {'Val R2':>10}")
        print("-" * 85)
        for l_idx, layer in enumerate(layer_indices_sorted):
            print(f"{layer:>6} | {layer_mse_results[layer]:>12.6f} | {layer_r2[l_idx]:>10.6f}")
        print("=" * 85)

        results_path = os.path.join(output_dir, "linear_probing_results.pt")
        torch.save(
            {
                "layer_mse": layer_mse_results,
                "layer_concept_mse": layer_concept_mse_results if args.per_concept else None,
                "layer_r2": dict(zip(layer_indices_sorted, layer_r2)),
                "layer_concept_r2": layer_concept_r2 if args.per_concept else None,
                "epoch_curves": None,
                "config": {
                    "epochs": args.epochs, "lr": args.lr, "train_ratio": args.train_ratio,
                    "seed": args.seed, "n_layers": n_layers, "in_dim": in_dim, "out_dim": out_dim,
                    "per_concept": args.per_concept, "layer_indices": layer_indices_sorted,
                },
            }, results_path,
        )
        if not args.no_plot:
            mse_curve_path = os.path.join(output_dir, "layer_mse_curve.png")
            plot_layer_mse_curve(
                [layer_mse_results[l] for l in layer_indices_sorted],
                layer_concept_mse_results, concepts, mse_curve_path,
                layer_indices=layer_indices_sorted,
                layer_r2=layer_r2, layer_concept_r2=layer_concept_r2,
            )
        print(f"\n[Done] Step 3 complete (torchrun mode).")
        print(f"  Results: {results_path}")
        return

    # ── 3b. Single-process training ───────────────────────────────────
    print(f"\n[3b] Training linear probes ({args.epochs} epochs, single-process) ...")
    layer_mse_results = {}
    layer_concept_mse_results = {c: [] for c in concepts}
    epoch_curves = {}
    layer_concept_val_preds = {}
    layer_concept_val_targets = {}

    for l_idx, layer in enumerate(layer_indices):
        print(f"\n  ── Layer {layer} ({l_idx+1}/{n_layers}) ──")

        z_tr = z_train[layer]
        z_vl = z_val[layer]

        if args.per_concept:
            concept_mses, val_preds_per_c, val_targets_per_c = train_per_concept(
                z_tr, p_train, c_train,
                z_vl, p_val, c_val,
                concepts, args.epochs, args.lr, device
            )
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

            layer_concept_val_preds[layer] = val_preds_per_c
            layer_concept_val_targets[layer] = val_targets_per_c
        else:
            mse, curve, val_pred = train_linear_probe(
                z_tr, p_train, z_vl, p_val,
                n_epochs=args.epochs, lr=args.lr, device=device, verbose=True
            )
            layer_mse_results[layer] = mse
            epoch_curves[layer] = curve
            print(f"  Layer {layer}: overall val_mse = {mse:.6f}")

    # ── 4. Compute R^2 scores ────────────────────────────────────────────
    print("\n[4] Computing R^2 scores...")

    if args.per_concept:
        layer_r2, layer_concept_r2 = compute_r2_from_predictions(
            layer_concept_val_preds,
            layer_concept_val_targets,
            concepts,
            layer_indices,
        )
    else:
        layer_r2 = [float("nan")] * n_layers
        layer_concept_r2 = {c: [float("nan")] * n_layers for c in concepts}

    # ── 5. Print results table ───────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"{'Layer':>6} | {'Val MSE':>12} | {'Val R2':>10}")
    print("-" * 85)
    for l_idx, layer in enumerate(layer_indices):
        print(f"{layer:>6} | {layer_mse_results[layer]:>12.6f} | {layer_r2[l_idx]:>10.6f}")
    print("=" * 85)

    # ── 6. Save results ───────────────────────────────────────────────
    print("\n[6] Saving results...")
    results_path = os.path.join(output_dir, "linear_probing_results.pt")
    torch.save(
        {
            "layer_mse": layer_mse_results,
            "layer_concept_mse": layer_concept_mse_results if args.per_concept else None,
            "layer_r2": dict(zip(layer_indices, layer_r2)),
            "layer_concept_r2": layer_concept_r2 if args.per_concept else None,
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

    # ── 7. Plot ───────────────────────────────────────────────────────
    if not args.no_plot:
        mse_curve_path = os.path.join(output_dir, "layer_mse_curve.png")
        plot_layer_mse_curve(
            [layer_mse_results[l] for l in layer_indices],
            layer_concept_mse_results,
            concepts,
            mse_curve_path,
            layer_indices=layer_indices,
            layer_r2=layer_r2,
            layer_concept_r2=layer_concept_r2,
        )

        if not args.per_concept:
            epoch_path = os.path.join(output_dir, "epoch_curves.png")
            plot_epoch_curves(epoch_curves, epoch_path)

    print(f"\n[Done] Step 3 complete.")
    print(f"  Results: {results_path}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
