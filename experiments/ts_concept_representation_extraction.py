#!/usr/bin/env python3
"""
Step 2: Representation Extraction & Pooling

Loads the trained Timer model and the synthetic dataset from Step 1,
extracts hidden states from each layer, and pools them into fixed-length
vectors z^(l) for use in linear probing.

Pipeline:
  1. Load synthetic dataset (X, labels, params) from Step 1
  2. Pre-process: z-score normalize all inputs (model-level normalization)
  3. Load Timer model in eval() mode, freeze all weights
  4. Forward pass with output_hidden_states=True
  5. Pool hidden states: Pool(H^(l)) = mean over sequence dimension
  6. Save per-layer pooled representations + labels

Usage:
    python experiments/ts_concept_representation_extraction.py \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --dataset_path ./results/synthetic/concepts_dataset.pt \
        --output_dir ./results/synthetic/representations/

    python experiments/ts_concept_representation_extraction.py \
        --ckpt_path random \
        --dataset_path ./results/synthetic/concepts_dataset.pt \
        --output_dir ./results/synthetic/representations/ \
        --d_model 1024 --e_layers 8
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

from models.Timer import Model
from experiments.ts_concept_synthetic_dataset import TSConceptGenerator, extract_param_vector


def build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    defaults = {
        "task_name": "forecast",
        "is_training": 0,
        "is_finetuning": 0,
        "train_test": 0,
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
        "des": "Exp",
        "num_workers": 4,
        "itr": 1,
        "dropout": 0.1,
        "activation": "gelu",
        "embed": "timeF",
        "freq": "h",
        "output_attention": False,
        "subset_rand_ratio": 1.0,
        "factor": 3,
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def pool_hidden_states(hidden_states: list[torch.Tensor], mode: str = "mean") -> list[torch.Tensor]:
    """
    Pool hidden states across the sequence/patch dimension.

    Args:
        hidden_states: list of [B, N, D] tensors, one per layer
        mode: 'mean', 'last', or 'cls'
            - 'mean': average over N
            - 'last': take the last token
            - 'cls': take the first token (if CLS was prepended)

    Returns:
        list of [B, D] pooled tensors, one per layer
    """
    pooled = []
    for h in hidden_states:
        if mode == "mean":
            p = h.mean(dim=1)          # [B, D]
        elif mode == "last":
            p = h[:, -1, :]            # [B, D]
        elif mode == "cls":
            p = h[:, 0, :]             # [B, D]
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        pooled.append(p)
    return pooled


def normalize_input(X: torch.Tensor) -> torch.Tensor:
    """Z-score normalize input over the time dimension (per sample)."""
    mean = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, keepdim=True) + 1e-8
    return (X - mean) / std


def extract_representations(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int = 128,
    device: torch.device = None,
) -> list[torch.Tensor]:
    """
    Extract per-layer pooled representations from the Timer model.

    Args:
        model: Timer model in eval mode
        X: [N, seq_len] time series tensor
        batch_size: batch size for processing
        device: target device

    Returns:
        list of [N, D] tensors, one per layer (pooled)
    """
    if device is None:
        device = next(model.parameters()).device

    N = X.shape[0]
    all_layer_pools = []

    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_x = X[i : i + batch_size]  # [B, seq_len]

            # Normalize input (model-level z-score)
            batch_x = normalize_input(batch_x)  # [B, seq_len]

            # Reshape to [B, T, M] with M=1 (single variate)
            batch_x = batch_x.unsqueeze(-1)  # [B, seq_len, 1]

            # Create dummy time marks (required by Timer interface)
            T = batch_x.shape[1]
            batch_x_mark = torch.zeros(B := batch_x.shape[0], T, 1, device=device)

            # Dummy decoder input (required by Timer forecast interface)
            label_len = 48
            dec_dummy = torch.zeros(B, label_len, 1, device=device)
            dec_mark_dummy = torch.zeros_like(dec_dummy)

            # Forward pass with hidden state extraction
            ret = model(
                batch_x.float().to(device),
                batch_x_mark.float().to(device),
                dec_dummy.float().to(device),
                dec_mark_dummy.float().to(device),
                output_hidden_states=True,
            )

            # Timer.forecast returns (dec_out, hidden_states) when output_hidden_states=True
            if isinstance(ret, tuple) and len(ret) >= 2:
                hidden_states = ret[1]
            else:
                hidden_states = None

            if hidden_states is None or len(hidden_states) == 0:
                raise RuntimeError("Failed to extract hidden states from model")

            # hidden_states 结构: [embedding, layer0, ..., layer7, LayerNorm]
            # 跳过 embedding 层，只取中间 8 层 decoder attention 输出
            decoder_states = hidden_states[1:9]

            # Pool hidden states per layer: mean over N patches
            # hidden_states[i] shape: [B*M, N, D] -> reshape to [B, M, N, D]
            # For M=1: [B, N, D]
            B_batch = batch_x.shape[0]
            M = 1  # single variate
            pooled_per_layer = []
            for h in decoder_states:
                # h: [B*M, N, D]
                h_reshaped = h.reshape(B_batch, M, -1, h.shape[-1])  # [B, M, N, D]
                h_reshaped = h_reshaped.squeeze(1)                    # [B, N, D]
                p = h_reshaped.mean(dim=1)                           # [B, D]
                pooled_per_layer.append(p)

            all_layer_pools.append(torch.stack(pooled_per_layer, dim=1))  # [B_batch, n_layers, D]

    # Concatenate all batches
    all_layer_pools = torch.cat(all_layer_pools, dim=0)  # [N, n_layers, D]
    # Transpose to list of [N, D] tensors
    return [all_layer_pools[:, l, :] for l in range(all_layer_pools.shape[1])]


def plot_representation_stats(
    layer_reps: list[torch.Tensor],
    labels: list[dict],
    concepts: list[str],
    output_path: str,
    max_samples: int = 200,
):
    """Plot per-layer representation statistics (norm, variance)."""
    n_layers = len(layer_reps)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    norms = []
    variances = []
    for rep in layer_reps:
        n = rep[:max_samples].cpu().norm(dim=1).numpy()
        v = rep[:max_samples].cpu().var(dim=1).numpy()
        norms.append(n)
        variances.append(v)

    layers = np.arange(n_layers)

    ax = axes[0]
    ax.boxplot(norms, labels=[str(i) for i in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Per-Layer Representation L2 Norm")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.boxplot(variances, labels=[str(i) for i in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance")
    ax.set_title("Per-Layer Representation Variance")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    mean_norms = [np.mean(n) for n in norms]
    ax.plot(layers, mean_norms, "o-", color="steelblue", linewidth=2)
    ax.fill_between(layers, [m - s for m, s in zip(mean_norms, [np.std(n) for n in norms])],
                    [m + s for m, s in zip(mean_norms, [np.std(n) for n in norms])],
                    alpha=0.2, color="steelblue")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Mean Norm per Layer (with std band)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved representation stats: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Extract Timer hidden states and pool them"
    )
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Timer model checkpoint path (or 'random' for uninitialized)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to concepts_dataset.pt from Step 1")
    parser.add_argument("--output_dir", type=str, default="./results/synthetic/representations/",
                        help="Output directory")
    parser.add_argument("--d_model", type=int, default=1024,
                        help="Model dimension (must match checkpoint)")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward dimension")
    parser.add_argument("--e_layers", type=int, default=8,
                        help="Number of encoder layers (must match checkpoint)")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--patch_len", type=int, default=96,
                        help="Patch length")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for representation extraction")
    parser.add_argument("--pooling_mode", type=str, default="mean",
                        choices=["mean", "last", "cls"],
                        help="Pooling strategy over sequence dimension")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per concept to process (None = all)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    print("=" * 60)
    print("Step 2: Representation Extraction & Pooling")
    print("=" * 60)
    print(f"  ckpt_path    : {args.ckpt_path}")
    print(f"  dataset_path : {args.dataset_path}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  d_model      : {args.d_model}")
    print(f"  e_layers     : {args.e_layers}")
    print(f"  patch_len    : {args.patch_len}")
    print(f"  pooling_mode : {args.pooling_mode}")
    print(f"  device       : {device}")
    print("=" * 60)

    # ── 1. Load synthetic dataset ────────────────────────────────────────────
    print("\n[1] Loading synthetic dataset...")
    ckpt = torch.load(args.dataset_path, map_location="cpu")
    X = ckpt["X"]           # [7*n_samples, seq_len]
    labels = ckpt["labels"]  # list of dicts
    concepts = ckpt["concepts"]
    n_samples_per_concept = ckpt["n_samples_per_concept"]
    seq_len = ckpt["seq_len"]
    print(f"  X shape      : {X.shape}")
    print(f"  concepts     : {concepts}")
    print(f"  samples/concept: {n_samples_per_concept}")

    # Limit samples if requested
    if args.max_samples is not None:
        selected = []
        for i, concept in enumerate(concepts):
            start = i * n_samples_per_concept
            end = start + n_samples_per_concept
            indices = np.random.choice(range(start, end), size=min(args.max_samples, n_samples_per_concept), replace=False)
            selected.extend(indices)
        selected = sorted(selected)
        X = X[selected]
        labels = [labels[i] for i in selected]
        print(f"  Limited to   : {len(X)} samples")

    # Build param vectors
    param_vectors = []
    for label in labels:
        pv = extract_param_vector(label)
        param_vectors.append(pv)
    params = torch.from_numpy(np.stack(param_vectors, axis=0)).float()

    concept_map = {c: i for i, c in enumerate(concepts)}
    concept_idx = torch.tensor(
        [concept_map[lbl["concept"]] for lbl in labels], dtype=torch.long
    )

    # ── 2. Load Timer model ──────────────────────────────────────────────────
    print("\n[2] Loading Timer model...")
    ns = build_namespace(argparse.Namespace(
        ckpt_path=args.ckpt_path,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        patch_len=args.patch_len,
        dropout=0.1,
        activation="gelu",
        factor=3,
    ))
    model = Model(ns).to(device)

    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded, total params: {n_params:,}")
    print(f"  d_model      : {model.d_model}")
    print(f"  e_layers     : {model.layers}")

    # ── 3. Extract hidden states ──────────────────────────────────────────────
    print(f"\n[3] Extracting hidden states (pooling={args.pooling_mode})...")
    print(f"  Processing {X.shape[0]} samples...")

    layer_reps = extract_representations(
        model=model,
        X=X,
        batch_size=args.batch_size,
        device=device,
    )

    n_layers = len(layer_reps)
    print(f"  Extracted {n_layers} layers, each with shape: {layer_reps[0].shape}")

    # Verify shapes
    for l, rep in enumerate(layer_reps):
        assert rep.shape[0] == X.shape[0], \
            f"Layer {l} sample count mismatch: {rep.shape[0]} vs {X.shape[0]}"
        assert rep.shape[1] == args.d_model, \
            f"Layer {l} dim mismatch: {rep.shape[1]} vs {args.d_model}"

    # ── 4. Save representations ──────────────────────────────────────────────
    print("\n[4] Saving representations...")

    # Save per-layer pooled representations
    rep_path = os.path.join(args.output_dir, "layer_representations.pt")
    torch.save(
        {
            "layer_reps": layer_reps,          # list of [N, D] tensors
            "params": params,                   # [N, n_params]
            "concept_idx": concept_idx,          # [N]
            "labels": labels,                   # list of dicts
            "concepts": concepts,
            "n_samples_per_concept": n_samples_per_concept,
            "seq_len": seq_len,
            "d_model": args.d_model,
            "n_layers": n_layers,
            "pooling_mode": args.pooling_mode,
            "ckpt_path": args.ckpt_path,
            "dataset_path": args.dataset_path,
        },
        rep_path,
    )
    print(f"  Saved: {rep_path}")

    # Also save as dict for linear probing (Step 3)
    for l in range(n_layers):
        torch.save(
            {
                "z": layer_reps[l],          # [N, D]
                "params": params,             # [N, n_params]
                "concept_idx": concept_idx,   # [N]
                "labels": labels,
                "concepts": concepts,
                "layer": l,
                "n_layers": n_layers,
                "d_model": args.d_model,
                "pooling_mode": args.pooling_mode,
            },
            os.path.join(args.output_dir, f"layer_{l}_rep.pt"),
        )

    # ── 5. Plot statistics ──────────────────────────────────────────────────
    if not args.no_plot:
        stats_path = os.path.join(args.output_dir, "repr_stats.png")
        plot_representation_stats(layer_reps, labels, concepts, stats_path)

    # Print per-layer norms
    print("\n[Summary] Per-layer L2 norm (mean ± std):")
    for l, rep in enumerate(layer_reps):
        norm_mean = rep.norm(dim=1).mean().item()
        norm_std = rep.norm(dim=1).std().item()
        print(f"  Layer {l:2d}: {norm_mean:.4f} ± {norm_std:.4f}")

    print(f"\n[Done] Step 2 complete.")
    print(f"  Representations: {rep_path}")
    print(f"  Layer count   : {n_layers}")
    print(f"  Samples       : {X.shape[0]}")


if __name__ == "__main__":
    main()
