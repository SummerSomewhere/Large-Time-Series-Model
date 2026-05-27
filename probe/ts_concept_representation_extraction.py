#!/usr/bin/env python3
"""
Step 2: Representation Extraction & Pooling (Timer version)

Loads the Timer model and the synthetic dataset from Step 1, extracts
per-layer hidden states after each decoder attention block, and pools them
into fixed-length vectors z^(l) for linear probing.

Key design choices for Timer:
  - Uses Timer.Model from models/Timer.py
  - Applies z-score normalization on raw input (same as forward_collect_layers)
  - Feeds through enc_embedding then decoder.attn_layers
  - Extracts first MI_DECODER_LAYER_CAP=8 decoder layers
  - Pooling: mean over patch tokens
  - Supports pooling modes: mean, last, cls

Usage:
    python probe/ts_concept_representation_extraction.py \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --dataset_path ./results/synthetic/concepts_dataset.pt \
        --output_dir ./results/synthetic/representations/

    python probe/ts_concept_representation_extraction.py \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --dataset_path ./results/synthetic/concepts_dataset.pt \
        --output_dir ./results/synthetic/representations/ \
        --save_token_reps
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

from probe.ts_concept_synthetic_dataset import TSConceptGenerator, extract_param_vector
from utils.masking import TriangularCausalMask

# Only collect representations for the first N decoder attention blocks.
# Matches MI_DECODER_LAYER_CAP in etth1_mi_hsic_peaks.py
MI_DECODER_LAYER_CAP = 8


# ─────────────────────────────────────────────────────────────────────────────
# Timer model wrapper for representation extraction
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_timer(model: torch.nn.Module):
    """Unwrap DDP-wrapped model to get the bare Timer core."""
    return model.module if hasattr(model, "module") else model


class TimerExtractor(nn.Module):
    """
    Wrapper that exposes Timer's per-layer hidden states after each decoder
    attention block.

    Pipeline:
      1. Z-score normalize raw input x (per-sample, per-channel)
      2. Permute to [B, M, L], pass through enc_embedding → [BM, N, D]
      3. Reshape to [B, M, N, D], pool over M → [B, N, D]
      4. Run through first MI_DECODER_LAYER_CAP decoder attention blocks
      5. Extract hidden state after each block (before LayerNorm or FFN)

    For the synthetic dataset (M=1, univariate), step 3 pools over the single
    giving the same [B, N, D] shape as patch outputs.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        seq_len: int = 512,
        patch_len: int = 96,
        stride: int = 96,
        d_model: int = 1024,
        d_ff: int = 2048,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        e_layers: int = 8,
        factor: int = 3,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_layers = MI_DECODER_LAYER_CAP

        # Build minimal namespace for Timer.Model
        ns = argparse.Namespace(
            task_name="forecast",
            is_training=0,
            is_finetuning=0,
            train_test=0,
            use_multi_gpu=False,
            d_layers=1,
            target="OT",
            checkpoints="./checkpoints/",
            inverse=False,
            use_amp=False,
            use_weight_decay=0,
            weight_decay=0.01,
            loss="MSE",
            lradj="type1",
            train_epochs=0,
            patience=3,
            learning_rate=1e-4,
            itr=1,
            finetune_epochs=0,
            output_attention=False,
            distil=True,
            model_id="timer_probe",
            model="Timer",
            output_len_list=None,
            mask_rate=0.25,
            data_type="custom",
            decay_fac=0.75,
            cos_warm_up_steps=100,
            cos_max_decay_steps=60000,
            cos_max_decay_epoch=10,
            cos_max=1e-4,
            cos_min=2e-6,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            e_layers=e_layers,
            factor=factor,
            ckpt_path=ckpt_path,
        )
        for k, v in vars(args := argparse.Namespace()).items():
            if not hasattr(ns, k):
                setattr(ns, k, v)

        # Load model
        sys.path.insert(0, os.path.join(_ROOT, "models"))
        from Timer import Model as TimerModel
        self.model = TimerModel(ns).to(device)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"[TimerExtractor] Loaded checkpoint from {ckpt_path}")
        else:
            print(f"[TimerExtractor] Warning: checkpoint not found at {ckpt_path}, using random init")

        self.model.eval()
        self._freeze()

        # Access core
        core = _unwrap_timer(self.model)
        self.core = core
        self.d_model = core.d_model
        self.n_vars = 1  # synthetic dataset is univariate

        print(f"[TimerExtractor] d_model={self.d_model}, num_layers={self.num_layers}")

    def _freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def _manual_forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Forward x through Timer's encoder embedding + decoder layers.

        Args:
            x: [B, seq_len] raw time series (univariate)

        Returns:
            layer_hs: list of [B, N, D] per-layer hidden states (one per decoder block)
            patch_embeds: [B, N, D] encoder embedding output (Layer 0 input)
        """
        B = x.shape[0]
        device = x.device

        # Z-score normalization (same as Timer forward)
        means = x.mean(1, keepdim=True)
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        # Timer convention: permute to [B, M, L] then embed
        # univariate [B, L] -> unsqueeze to [B, 1, L]; multivariate [B, M, L] -> transpose to [B, M, L]
        if x_norm.dim() == 2:
            x2 = x_norm.unsqueeze(1).float()  # [B, 1, L]
        else:
            x2 = x_norm.permute(0, 2, 1).float()
        # For multivariate x [B, L, M], this stays [B, M, L]
        dec_in, n_vars = self.core.enc_embedding(x2)  # [BM, N, D]
        BM, N, D = dec_in.shape

        def pool(z: torch.Tensor) -> torch.Tensor:
            """Pool over variable dimension: [BM, N, D] → [B, N, D]"""
            return z.view(B, n_vars, N, D).mean(dim=1)

        # Patch embeddings (Layer 0 input)
        patch_embeds = pool(dec_in.detach().float().cpu())

        # Decoder forward through first MI_DECODER_LAYER_CAP blocks
        h = dec_in
        mask = TriangularCausalMask(BM, N, device=device)

        layer_hs: list[torch.Tensor] = []
        for i, o3 in enumerate(self.core.decoder.attn_layers):
            if i >= MI_DECODER_LAYER_CAP:
                break
            h, _, _ = o3(h, attn_mask=mask)
            rep = pool(h.detach().float().cpu())  # [B, N, D]
            layer_hs.append(rep)

        return layer_hs, patch_embeds


# ─────────────────────────────────────────────────────────────────────────────
# Representation extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_token_representations(
    extractor: TimerExtractor,
    X: torch.Tensor,
    batch_size: int = 128,
    device: torch.device = None,
):
    """
    Extract per-layer per-token (patch) representations and patch embeddings.

    Args:
        extractor: TimerExtractor model in eval mode
        X: [N, seq_len] time series tensor
        batch_size: batch size for processing
        device: target device

    Returns:
        layer_tokens: list of [N, N_patches, D] tensors, one per layer
        patch_tokens: [N, N_patches, D] patch embedding output (before Transformer layers)
    """
    if device is None:
        device = next(extractor.parameters()).device

    N = X.shape[0]
    all_layer_tokens = []

    _model = extractor.module if isinstance(extractor, torch.nn.DataParallel) else extractor
    extractor.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_x = X[i : i + batch_size].float().to(device)  # [B, seq_len]
            layer_hs, patch_embeds = _model._manual_forward(batch_x)

            if i == 0:
                all_layer_tokens = [h.float().cpu() for h in layer_hs]
                all_patch_tokens = patch_embeds.float().cpu()
            else:
                for li in range(len(layer_hs)):
                    all_layer_tokens[li] = torch.cat(
                        [all_layer_tokens[li], layer_hs[li].float().cpu()], dim=0
                    )
                all_patch_tokens = torch.cat([all_patch_tokens, patch_embeds.float().cpu()], dim=0)

    return all_layer_tokens, all_patch_tokens


def extract_representations(
    extractor: TimerExtractor,
    X: torch.Tensor,
    batch_size: int = 128,
    device: torch.device = None,
) -> list[torch.Tensor]:
    """
    Extract per-layer pooled representations from Timer.

    Args:
        extractor: TimerExtractor model in eval mode
        X: [N, seq_len] time series tensor
        batch_size: batch size for processing
        device: target device

    Returns:
        list of [N, D] pooled tensors, one per layer
    """
    if device is None:
        device = next(extractor.parameters()).device

    N = X.shape[0]
    all_layer_pools = []

    _model = extractor.module if isinstance(extractor, torch.nn.DataParallel) else extractor

    extractor.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_x = X[i : i + batch_size].float().to(device)  # [B, seq_len]
            layer_hs, _ = _model._manual_forward(batch_x)

            # layer_hs[i]: [B, N, D]  -> pooled: [B, D]
            pooled_per_layer = [h.mean(dim=1) for h in layer_hs]
            all_layer_pools.append(torch.stack(pooled_per_layer, dim=1))

    # Concatenate all batches: [N, n_layers, D]
    all_layer_pools = torch.cat(all_layer_pools, dim=0)

    # Transpose to list of [N, D] tensors, one per layer
    return [all_layer_pools[:, l, :] for l in range(all_layer_pools.shape[1])]


def plot_representation_stats(
    layer_reps: list[torch.Tensor],
    labels: list[dict],
    concepts: list[str],
    output_path: str,
    max_samples: int = 200,
):
    """Plot per-layer representation statistics."""
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
    ax.fill_between(layers,
                    [m - s for m, s in zip(mean_norms, [np.std(n) for n in norms])],
                    [m + s for m, s in zip(mean_norms, [np.std(n) for n in norms])],
                    alpha=0.2, color="steelblue")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Mean Norm per Layer (with std band)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Timer Layer Representations", fontsize=12, fontweight="bold")
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
                        help="Path to Timer checkpoint (.ckpt)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to concepts_dataset.pt from Step 1")
    parser.add_argument("--output_dir", type=str, default="./results/synthetic/representations/",
                        help="Output directory")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Input sequence length (Timer context)")
    parser.add_argument("--patch_len", type=int, default=96,
                        help="Patch length (Timer default: 96)")
    parser.add_argument("--stride", type=int, default=96,
                        help="Stride for patching")
    parser.add_argument("--d_model", type=int, default=1024,
                        help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--e_layers", type=int, default=8,
                        help="Number of encoder layers")
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for representation extraction")
    parser.add_argument("--pooling_mode", type=str, default="mean",
                        choices=["mean", "last", "cls"],
                        help="Pooling strategy over patch dimension")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per concept to process (None = all)")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--save_token_reps", action="store_true",
                        help="Also save per-token (unpooled) representations for token-level MI analysis")
    parser.add_argument("--use_dataparallel", action="store_true", default=False,
                        help="Wrap the model with torch.nn.DataParallel for multi-GPU inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    print("=" * 60)
    print("Step 2: Representation Extraction & Pooling (Timer)")
    print("=" * 60)
    print(f"  ckpt_path   : {args.ckpt_path}")
    print(f"  dataset_path : {args.dataset_path}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  seq_len     : {args.seq_len}")
    print(f"  patch_len   : {args.patch_len}")
    print(f"  pooling_mode: {args.pooling_mode}")
    print(f"  device      : {device}")
    print("=" * 60)

    # ── 1. Load synthetic dataset ────────────────────────────────────────────
    print("\n[1] Loading synthetic dataset...")
    ckpt_ds = torch.load(args.dataset_path, map_location="cpu")
    X = ckpt_ds["X"]           # [7*n_samples, seq_len]
    labels = ckpt_ds["labels"]
    concepts = ckpt_ds["concepts"]
    n_samples_per_concept = ckpt_ds["n_samples_per_concept"]
    seq_len_ds = ckpt_ds["seq_len"]
    print(f"  X shape     : {X.shape}")
    print(f"  seq_len     : {seq_len_ds}")
    print(f"  concepts    : {concepts}")
    print(f"  samples/concept: {n_samples_per_concept}")

    # Override seq_len to match dataset if needed
    if seq_len_ds != args.seq_len:
        print(f"  Warning: dataset seq_len={seq_len_ds}, using args seq_len={args.seq_len}")

    if args.max_samples is not None:
        selected = []
        for i, concept in enumerate(concepts):
            start = i * n_samples_per_concept
            end = start + n_samples_per_concept
            indices = np.random.choice(
                range(start, end), size=min(args.max_samples, n_samples_per_concept), replace=False
            )
            selected.extend(indices)
        selected = sorted(selected)
        X = X[selected]
        labels = [labels[i] for i in selected]
        print(f"  Limited to : {len(X)} samples")

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

    # ── 2. Load Timer model ───────────────────────────────────────────────
    print("\n[2] Loading Timer model...")

    extractor = TimerExtractor(
        ckpt_path=args.ckpt_path,
        device=device,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        dropout=args.dropout,
        activation=args.activation,
        e_layers=args.e_layers,
        factor=args.factor,
    ).to(device)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.use_dataparallel:
        print(f"  Wrapping with DataParallel across {n_gpus} GPUs ...")
        extractor = torch.nn.DataParallel(extractor)

    n_params = sum(p.numel() for p in extractor.parameters())
    print(f"  Model loaded, total params: {n_params:,}")
    d_model_val = extractor.module.d_model if isinstance(extractor, torch.nn.DataParallel) else extractor.d_model
    num_layers_val = extractor.module.num_layers if isinstance(extractor, torch.nn.DataParallel) else extractor.num_layers
    print(f"  d_model     : {d_model_val}")
    print(f"  num_layers  : {num_layers_val}")

    # ── 3. Extract hidden states ─────────────────────────────────────────────
    print(f"\n[3] Extracting hidden states (pooling={args.pooling_mode})...")
    print(f"  Processing {X.shape[0]} samples...")

    layer_reps = extract_representations(
        extractor=extractor,
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
        assert rep.shape[1] == d_model_val, \
            f"Layer {l} dim mismatch: {rep.shape[1]} vs {d_model_val}"

    # ── 4. Save representations ─────────────────────────────────────────────
    print("\n[4] Saving representations...")

    rep_path = os.path.join(args.output_dir, "layer_representations.pt")
    torch.save(
        {
            "layer_reps": layer_reps,
            "params": params,
            "concept_idx": concept_idx,
            "labels": labels,
            "concepts": concepts,
            "n_samples_per_concept": n_samples_per_concept,
            "seq_len": seq_len_ds,
            "d_model": d_model_val,
            "n_layers": n_layers,
            "pooling_mode": args.pooling_mode,
            "model_name": "Timer",
            "patch_len": args.patch_len,
            "dataset_path": args.dataset_path,
        },
        rep_path,
    )
    print(f"  Saved: {rep_path}")

    # Save per-token (unpooled) representations for token-level MI analysis
    if args.save_token_reps:
        print("\n[4b] Extracting per-token representations for MI analysis...")
        layer_tokens, patch_tokens = extract_token_representations(
            extractor=extractor,
            X=X,
            batch_size=args.batch_size,
            device=device,
        )
        n_patches = layer_tokens[0].shape[1]
        print(f"  Token reps shape: {n_layers} layers x [{X.shape[0]}, {n_patches}, {d_model_val}]")
        print(f"  Patch tokens shape: [{X.shape[0]}, {n_patches}, {d_model_val}]")

        token_rep_path = os.path.join(args.output_dir, "layer_token_representations.pt")
        torch.save(
            {
                "layer_tokens": layer_tokens,      # list of [N, N_patches, D]
                "patch_tokens": patch_tokens,       # [N, N_patches, D] — patch embedding output
                "params": params,
                "concept_idx": concept_idx,
                "labels": labels,
                "concepts": concepts,
                "n_samples_per_concept": n_samples_per_concept,
                "seq_len": seq_len_ds,
                "d_model": d_model_val,
                "n_layers": n_layers,
                "n_patches": n_patches,
                "model_name": "Timer",
                "patch_len": args.patch_len,
                "dataset_path": args.dataset_path,
            },
            token_rep_path,
        )
        print(f"  Saved: {token_rep_path}")

    # Also save as per-layer dict for linear probing
    for l in range(n_layers):
        torch.save(
            {
                "z": layer_reps[l],
                "params": params,
                "concept_idx": concept_idx,
                "labels": labels,
                "concepts": concepts,
                "layer": l,
                "n_layers": n_layers,
                "d_model": d_model_val,
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
