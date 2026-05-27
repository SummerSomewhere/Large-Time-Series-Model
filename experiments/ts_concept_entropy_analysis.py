#!/usr/bin/env python3
"""
Step 5: Entropy Analysis for Timer

Computes and visualizes the matrix alpha entropy of hidden representations
after each decoder attention block. Uses repitl matrix_alpha_entropy if available,
otherwise falls back to binning-based entropy.

Reference: information_flow-main/experiments/utils/metrics/metric_functions.py

Usage:
    python experiments/ts_concept_entropy_analysis.py \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --root_path ./dataset/ \
        --data_path ETTh1.csv \
        --output_dir ./entropy_analysis \
        [--alpha 1] \
        [--normalization maxEntropy] \
        [--max_samples 2000]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_loader import Dataset_Custom
from experiments.etth1_mi_hsic_peaks import (
    MI_DECODER_LAYER_CAP,
    TriangularCausalMask,
    _unwrap_timer,
)

try:
    import repitl.matrix_itl as itl
    HAS_REPITL = True
except ImportError:
    HAS_REPITL = False
    print("Warning: repitl not installed. Using fallback binning entropy.")


def entropy_normalization(entropy: float, normalization: str, N: int, D: int) -> float:
    if normalization == 'maxEntropy':
        entropy /= min(math.log(N), math.log(D))
    elif normalization == 'logN':
        entropy /= math.log(N)
    elif normalization == 'logD':
        entropy /= math.log(D)
    elif normalization == 'logNlogD':
        entropy /= (math.log(N) * math.log(D))
    elif normalization == 'raw':
        pass
    elif normalization == 'length':
        entropy = N
    return entropy


def _matrix_alpha_entropy_eigen(
    cov: torch.Tensor, alpha: float
) -> float:
    """
    Compute matrix-based Rényi alpha entropy via eigenvalue decomposition.
    cov: [d, d] SPD matrix, already normalized (trace=1).
    Returns scalar entropy.
    """
    try:
        eigvals = torch.linalg.eigvalsh(cov.clamp(min=1e-10))
        eigvals = eigvals[eigvals > 1e-10]   # drop near-zero
        if eigvals.numel() == 0:
            return float('nan')
        if abs(alpha - 1.0) < 1e-6:
            # Shannon: -sum(p * log p)
            return float(-(eigvals * eigvals.log()).sum().item())
        else:
            # Rényi: 1/(1-alpha) * log(sum(p^alpha))
            return float((1.0 / (1.0 - alpha)) * (eigvals ** alpha).sum().log().item())
    except Exception:
        return float('nan')


def compute_matrix_entropy(
    hidden_states: torch.Tensor,
    alpha: float = 1,
    normalization: str = 'maxEntropy',
) -> float:
    """
    Compute matrix-based Rényi alpha entropy.
    If repitl is available, uses its optimized implementation; otherwise falls back
    to a pure-PyTorch eigenvalue-based method.
    hidden_states: [n_samples, n_patches, D]
    Returns scalar mean entropy across samples.
    """
    n, N, D = hidden_states.shape

    if N > D:
        cov = torch.matmul(hidden_states.transpose(1, 2), hidden_states)   # [n, D, D]
    else:
        cov = torch.matmul(hidden_states, hidden_states.transpose(1, 2))  # [n, N, N]

    cov = torch.clamp(cov, min=0)

    entropies = []
    for i in range(n):
        try:
            c = cov[i].double()
            trace = torch.trace(c)
            if trace <= 0:
                entropies.append(np.nan)
                continue
            c_norm = c / trace
            if HAS_REPITL:
                ent = itl.matrixAlphaEntropy(c_norm, alpha=alpha).item()
            else:
                ent = _matrix_alpha_entropy_eigen(c_norm, alpha=alpha)
            ent = entropy_normalization(ent, normalization, N, D)
            entropies.append(ent)
        except Exception:
            entropies.append(np.nan)

    valid = [e for e in entropies if not np.isnan(e)]
    return float(np.mean(valid)) if valid else float('nan')


def compute_fallback_entropy(
    hidden_states: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Simple binning-based entropy fallback when repitl is not available."""
    from scipy.stats import entropy as scipy_entropy

    n, N, D = hidden_states.shape
    flat = hidden_states.reshape(n, -1)   # [n, N*D]

    entropies = []
    for i in range(n):
        vals = flat[i]
        hist, _ = np.histogram(vals, bins=n_bins, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        entropies.append(scipy_entropy(hist))
    return float(np.mean(entropies))


def collect_timer_layer_representations(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> list[torch.Tensor]:
    """
    Forward the model through data_loader and collect representations
    after each of the first MI_DECODER_LAYER_CAP decoder attention blocks.

    Returns:
        layer_reps: list of K tensors, each [total_samples, N, D]
                     K = min(MI_DECODER_LAYER_CAP, num_decoder_layers)
    """
    model.eval()
    core = _unwrap_timer(model)

    layer_reps: list[list[torch.Tensor] | None] = [None] * MI_DECODER_LAYER_CAP
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x_enc = batch[0].to(device)   # [B, L, M]
            B = x_enc.shape[0]

            means = x_enc.mean(1, keepdim=True)
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

            x2 = x.permute(0, 2, 1).float()
            dec_in, n_vars = core.enc_embedding(x2)  # [BM, N, D]
            BM, N, D_dim = dec_in.shape

            def pool(z: torch.Tensor) -> torch.Tensor:
                return z.view(B, n_vars, -1, D_dim).mean(dim=1)   # [B, N, D]

            h = dec_in
            mask = TriangularCausalMask(BM, N, device=device)

            for li, o3 in enumerate(core.decoder.attn_layers):
                if li >= MI_DECODER_LAYER_CAP:
                    break
                h, _ = o3(h, attn_mask=mask)
                rep = pool(h.detach().cpu())   # [B, N, D]

                if layer_reps[li] is None:
                    layer_reps[li] = []
                layer_reps[li].append(rep)

            total_count += B

    # Concatenate batches
    result: list[torch.Tensor] = []
    for li in range(len(layer_reps)):
        if layer_reps[li] is not None:
            result.append(torch.cat(layer_reps[li], dim=0))
        else:
            result.append(torch.empty(0, N, D_dim))

    print(f"  Collected {total_count} samples across {len(result)} layers")
    print(f"  Layer shapes: {[r.shape for r in result]}")
    return result


def compute_layer_entropies(
    layer_reps: list[torch.Tensor],
    alpha: float = 1,
    normalization: str = 'maxEntropy',
    max_samples: int = 2000,
) -> tuple[list[int], list[float]]:
    """
    Compute entropy per layer across all collected samples.
    """
    entropies: list[float] = []
    layer_indices = list(range(len(layer_reps)))

    for layer_idx, rep in enumerate(layer_reps):
        rep_tensor = rep.cpu()

        if rep_tensor.shape[0] > max_samples:
            indices = np.random.choice(rep_tensor.shape[0], max_samples, replace=False)
            rep_tensor = rep_tensor[indices]

        ent = compute_matrix_entropy(rep_tensor, alpha=alpha, normalization=normalization)

        entropies.append(ent)
        print(f"  Layer {layer_idx}: entropy = {ent:.4f}  (n={rep_tensor.shape[0]})")

    return layer_indices, entropies


def plot_entropy_curve(
    layer_indices: list[int],
    entropies: list[float],
    output_path: str,
    title: str = "Decoder Layer Entropy vs. Layer Depth",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_idx = [i for i, e in zip(layer_indices, entropies) if not np.isnan(e)]
    valid_ent = [e for e in entropies if not np.isnan(e)]

    if valid_ent:
        ax.plot(valid_idx, valid_ent, "o-", color="steelblue", linewidth=2, markersize=8)
        ax.set_xticks(valid_idx)
        for i, ent in zip(valid_idx, valid_ent):
            ax.annotate(f"{ent:.3f}", (i, ent),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        ax.set_ylim(bottom=0)

    ax.set_xlabel("Decoder Layer Depth", fontsize=12)
    ax.set_ylabel("Entropy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_entropy_heatmap(
    layer_reps: list[torch.Tensor],
    entropies: list[float],
    output_path: str,
    max_samples_heatmap: int = 500,
):
    """
    Per-sample entropy heatmap: x=layer, y=sample, color=entropy.
    """
    layer_indices = list(range(len(layer_reps)))
    n_layers = len(layer_reps)

    sample_entropies: list[np.ndarray] = []
    for layer_idx, rep in enumerate(layer_reps):
        rep_tensor = rep.cpu()
        n = rep_tensor.shape[0]
        n_s = min(n, max_samples_heatmap)

        if n_s < n:
            indices = np.random.choice(n, n_s, replace=False)
            rep_tensor = rep_tensor[indices]

        ent = []
        for i in range(n_s):
            sample = rep_tensor[i:i+1]   # [1, N, D]
            e = compute_matrix_entropy(sample)
            ent.append(e)
        sample_entropies.append(np.array(ent))

    n_samples = max(s.shape[0] for s in sample_entropies)
    matrix = np.full((n_layers, n_samples), np.nan)
    for li, se in enumerate(sample_entropies):
        matrix[li, :len(se)] = se

    fig, ax = plt.subplots(figsize=(max(12, n_samples * 0.05), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Layer Depth", fontsize=12)
    ax.set_title("Per-Sample Entropy per Decoder Layer", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks(layer_indices)
    ax.set_yticklabels([f"L{i}" for i in layer_indices])
    plt.colorbar(im, ax=ax, label="Entropy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Timer Decoder Layer Entropy Analysis")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to Timer checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for entropy analysis")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to CSV data file (optional)")
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=["custom", "ettH1", "ettH2", "ettM1", "etth1", "etth2", "ettm1"],
                        help="Dataset type")
    parser.add_argument("--root_path", type=str, default="./dataset/",
                        help="Root path for data files")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Input sequence length")
    parser.add_argument("--pred_len", type=int, default=96,
                        help="Prediction length")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1,
                        help="Alpha parameter for matrix alpha entropy")
    parser.add_argument("--normalization", type=str, default="maxEntropy",
                        choices=['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length'])
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max samples for entropy computation")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Max batches to process (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch_len", type=int, default=96,
                        help="Patch length for Timer model")
    parser.add_argument("--d_model", type=int, default=1024,
                        help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--factor", type=int, default=1,
                        help="Factor for attention")
    parser.add_argument("--activation", type=str, default="gelu",
                        help="Activation function")
    parser.add_argument("--e_layers", type=int, default=8,
                        help="Number of encoder layers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Timer Decoder Layer Entropy Analysis")
    print("=" * 60)
    print(f"  ckpt_path     : {args.ckpt_path}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  dataset       : {args.dataset}")
    print(f"  seq_len       : {args.seq_len}")
    print(f"  pred_len      : {args.pred_len}")
    print(f"  alpha         : {args.alpha}")
    print(f"  normalization : {args.normalization}")
    print(f"  max_samples   : {args.max_samples}")
    print(f"  repitl        : {HAS_REPITL}")
    print(f"  device        : {args.device}")
    print("=" * 60)

    # Build minimal namespace for model + data (same pattern as etth1_mi_hsic_peaks.py)
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
        "model_id": "timer_entropy",
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

    # Load model
    print("\n[1] Loading Timer model...")
    device = torch.device(args.device)
    sys.path.insert(0, os.path.join(_ROOT, "models"))
    from Timer import Model as TimerModel
    model = TimerModel(ns)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    print("  Model loaded.")

    # Build dataset
    print("\n[2] Building dataset...")
    DataClass = Dataset_Custom
    data_set = DataClass(
        root_path=args.root_path,
        data_path=args.data_path,
        flag="test",
        size=[args.seq_len, 0, args.pred_len],
        features="M",
        target="OT",
    )
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(f"  Dataset: {len(data_set)} samples, {len(data_loader)} batches")

    # Collect representations
    print("\n[3] Collecting layer representations...")
    layer_reps = collect_timer_layer_representations(
        model, data_loader, device, max_batches=args.max_batches
    )

    # Compute entropy
    print("\n[4] Computing entropy per layer...")
    layer_indices, entropies = compute_layer_entropies(
        layer_reps,
        alpha=args.alpha,
        normalization=args.normalization,
        max_samples=args.max_samples,
    )

    # Save results
    print("\n[5] Saving results...")
    results = {
        "layer_indices": layer_indices,
        "entropies": entropies,
        "layer_shapes": [list(r.shape) for r in layer_reps],
        "config": {
            "alpha": args.alpha,
            "normalization": args.normalization,
            "max_samples": args.max_samples,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "use_matrix_entropy": HAS_REPITL,
        }
    }
    results_path = os.path.join(args.output_dir, "entropy_results.json")
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Plots
    print("\n[6] Creating visualizations...")
    plot_entropy_curve(
        layer_indices,
        entropies,
        os.path.join(args.output_dir, "entropy_curve.png"),
        title=f"Decoder Layer Entropy (alpha={args.alpha}, norm={args.normalization})",
    )

    if HAS_REPITL:
        plot_entropy_heatmap(
            layer_reps,
            entropies,
            os.path.join(args.output_dir, "entropy_heatmap.png"),
            max_samples_heatmap=500,
        )

    # Summary
    print("\n[Summary]")
    valid = [(i, e) for i, e in zip(layer_indices, entropies) if not np.isnan(e)]
    if valid:
        min_idx, min_ent = min(valid, key=lambda x: x[1])
        max_idx, max_ent = max(valid, key=lambda x: x[1])
        print(f"  Min entropy:  Layer {min_idx} = {min_ent:.4f}")
        print(f"  Max entropy:  Layer {max_idx} = {max_ent:.4f}")
        print(f"  Mean entropy: {np.mean([e for _, e in valid]):.4f}")

    print(f"\n[Done] Results: {results_path}")


if __name__ == "__main__":
    main()
