#!/usr/bin/env python3
"""Timer MI-guided *sample-level* sampling (Scheme B).

This experiment implements **sample-window sampling** driven by *high-MI patch difficulty*.

Core idea
---------
Given full-window training (sampling unit is a sample window), we compute a per-sample
score based on model error restricted to a subset of patches (e.g., high-MI patches):

    s_i = sum_{p in S} e_{i,p}
    P(i) ∝ (s_i + eps)^alpha

where S is one of:
  - high-MI patches
  - low-MI patches
  - random patches (same count as high) as a control

We then rebuild the DataLoader with WeightedRandomSampler.

Curriculum
----------
Alpha can be scheduled from high → 0 to gradually return to uniform sampling.

Outputs
-------
Writes per-run metrics JSON and optional cached sampling scores.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Make this script runnable via `python experiments/...py` or `bash experiments/...sh`
# by ensuring the project root is on sys.path.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


# --------------------------------------------------------------------------------------
# Data args safety net (fills what data_provider expects)
# --------------------------------------------------------------------------------------

def ensure_data_args(args):
    """Fill required args for data_provider with safe defaults.

    This prevents AttributeError when running this script standalone
    without all the flags that timer_mi_distillation_comparison.py defines.
    """
    defaults = {
        "embed": "timeF",
        "freq": "h",
        "features": "M",
        "target": "OT",
        "scale": True,
        "seasonal_patterns": "Monthly",
        "label_len": 0,
        "inverse": False,
        "cols": None,
        "task_name": "forecast",
        "stride": 1,
        "use_ims": False,
        "use_multi_gpu": False,
    }
    for k, v in defaults.items():
        if not hasattr(args, k) or getattr(args, k, None) is None:
            setattr(args, k, v)

    # timeenc is derived from embed in data_factory.py; mirror the logic here
    if getattr(args, "embed", "timeF") == "timeF":
        args.timeenc = 1
    else:
        args.timeenc = 0

    return args


# --------------------------------------------------------------------------------------
# Utilities (minimal, local)
# --------------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def safe_mkdir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


# --------------------------------------------------------------------------------------
# MI patch set helpers (re-use JSON structure from Scheme A)
# --------------------------------------------------------------------------------------

def load_global_mi_json(mi_dir: str | Path) -> Dict:
    mi_dir = Path(mi_dir)
    cands = sorted(mi_dir.glob("global_mi_peaks_*.json"))
    if not cands:
        raise FileNotFoundError(f"No global_mi_peaks_*.json found under: {mi_dir}")
    with open(cands[0], "r") as f:
        return json.load(f)


def build_patch_sets_from_mi_json(
    mi_data: Dict,
    n_patches: int,
    select_layer: str = "vote",
) -> Tuple[List[int], List[int]]:
    """Return (high_patches, low_patches) as global sets.

    - vote: union across layers' high_mi_patches / low_mi_patches.
    - first_layer: use layer '0' (or first existing) only.

    Notes:
      We intentionally keep this simple and robust; the goal here is to create a
      *consistent* patch subset for sampling.
    """

    layers = mi_data.get("layers", {})
    if not layers:
        # fallback: uniform partition
        half = max(1, n_patches // 2)
        return list(range(half)), list(range(half, n_patches))

    if select_layer == "first_layer":
        # pick first key deterministically
        k0 = sorted(layers.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))[0]
        info = layers[k0]
        high = [int(p) for p in info.get("high_mi_patches", [])]
        low = [int(p) for p in info.get("low_mi_patches", [])]
    else:
        high_set, low_set = set(), set()
        for _li, info in layers.items():
            for p in info.get("high_mi_patches", []):
                pi = int(p)
                if 0 <= pi < n_patches:
                    high_set.add(pi)
            for p in info.get("low_mi_patches", []):
                pi = int(p)
                if 0 <= pi < n_patches:
                    low_set.add(pi)
        high = sorted(high_set)
        low = sorted(low_set)

    # keep within range and unique
    high = sorted({p for p in high if 0 <= p < n_patches})
    low = sorted({p for p in low if 0 <= p < n_patches})

    # ensure non-empty (avoid sampler degeneracy)
    if len(high) == 0:
        high = list(range(max(1, n_patches // 4)))
    if len(low) == 0:
        low = list(range(n_patches - max(1, n_patches // 4), n_patches))

    return high, low


def sample_random_patches(n_patches: int, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    k = min(max(1, k), n_patches)
    return sorted(rng.sample(list(range(n_patches)), k=k))


# --------------------------------------------------------------------------------------
# Patch error -> sample score
# --------------------------------------------------------------------------------------


def compute_patch_mse_from_outputs(
    output: torch.Tensor,
    target: torch.Tensor,
    patch_len: int,
) -> torch.Tensor:
    """Compute per-patch MSE from prediction outputs.

    output/target: [B, pred_len, C]
    returns: [B, k] where k = pred_len/patch_len if divisible else 1.

    This intentionally mirrors the logic used in Scheme A's patch-weighted task loss.
    """

    B, pred_len, _C = output.shape
    mse_t = (output - target).pow(2).mean(dim=-1)  # [B, pred_len]

    if pred_len % patch_len != 0:
        return mse_t.mean(dim=1, keepdim=True)  # [B, 1]

    k = pred_len // patch_len
    mse_p = mse_t.reshape(B, k, patch_len).mean(dim=-1)  # [B, k]
    return mse_p


def score_samples_from_patch_mse(
    patch_mse: torch.Tensor,
    patch_indices: Sequence[int],
) -> torch.Tensor:
    """Reduce patch MSE into per-sample score.

    patch_mse: [B, k] where k is number of predicted patches.
    patch_indices: indices in [0, k) (within predicted horizon patch grid).

    returns: [B]
    """

    if patch_mse.ndim != 2:
        raise ValueError(f"patch_mse must be [B,k], got {tuple(patch_mse.shape)}")

    k = patch_mse.shape[1]
    use = [int(p) for p in patch_indices if 0 <= int(p) < k]
    if len(use) == 0:
        return patch_mse.mean(dim=1)
    return patch_mse[:, use].sum(dim=1)


# --------------------------------------------------------------------------------------
# Data / Model hooks: rely on existing codebase components via import
# --------------------------------------------------------------------------------------


def build_dataloaders_from_repo(
    args,
    sampling_weights: Optional[torch.Tensor] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create train/val/test loaders.

    If sampling_weights is provided, use WeightedRandomSampler for the training loader.

    Returns: (train_loader, val_loader, test_loader, train_size)
    """

    # Re-use the existing exp file's data provider to stay consistent.
    try:
        from experiments.timer_mi_distillation_comparison import data_provider  # type: ignore
    except ModuleNotFoundError:
        # Fallback when running without package context
        from timer_mi_distillation_comparison import data_provider  # type: ignore

    train_data, train_loader = data_provider(args, flag="train")
    val_data, val_loader = data_provider(args, flag="val")
    test_data, test_loader = data_provider(args, flag="test")

    train_size = len(train_data)

    if sampling_weights is None:
        return train_loader, val_loader, test_loader, train_size

    if sampling_weights.numel() != train_size:
        raise ValueError(
            f"sampling_weights length mismatch: got {sampling_weights.numel()}, expect {train_size}"
        )

    # Sampler expects CPU double/float weights
    weights_cpu = sampling_weights.detach().to("cpu").float()
    sampler = WeightedRandomSampler(
        weights=weights_cpu,
        num_samples=train_size,
        replacement=True,
    )

    train_loader_wrapped = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=getattr(args, "num_workers", 0),
        drop_last=True,
        pin_memory=True,
    )

    return train_loader_wrapped, val_loader, test_loader, train_size


def build_models_from_repo(args):
    """Build the Student model using the existing experiment helper.

    Note:
      This sampling experiment does not require the Timer teacher at all.
      Avoid importing teacher-building utilities to reduce dependency/version issues.
    """

    try:
        from experiments.timer_mi_distillation_comparison import StudentTransformer  # type: ignore
    except ModuleNotFoundError:
        from timer_mi_distillation_comparison import StudentTransformer  # type: ignore

    student = StudentTransformer(
        d_model=args.s_d_model,
        d_ff=args.s_d_ff,
        n_layers=args.s_n_layers,
        n_heads=args.s_n_heads,
        dropout=args.s_dropout,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    ).to(args.device)

    return student


def forward_batch_with_repo_model(student: nn.Module, batch, args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward a batch and return (pred, target, loss_task).

    data_provider returns (batch_x, batch_y, batch_x_mark, batch_y_mark).
    StudentTransformer.forward(x) expects [B, seq_len, 1].
    """

    # Unpack the batch tuple directly — no need for an import.
    if isinstance(batch, (tuple, list)):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    else:
        batch_x = batch

    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float().to(args.device)

    # StudentTransformer.forward: [B, seq_len, 1] -> [B, pred_len, 1]
    pred = student(batch_x)
    target = batch_y[:, :pred.shape[1], :]

    task_loss = F.mse_loss(pred, target)
    return pred, target, task_loss


# --------------------------------------------------------------------------------------
# Sampling weight computation loop
# --------------------------------------------------------------------------------------


@torch.no_grad()
def compute_sampling_weights(
    student: nn.Module,
    train_loader: DataLoader,
    args,
    patch_subset_in_pred_horizon: Sequence[int],
    eps: float,
    alpha: float,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """Compute P(i) weights for each training sample.

    Important:
      - We assume train_loader is in *sequential order* (i.e., no shuffle/sampler)
        so that the iteration order corresponds to dataset indices.
      - If the repo DataLoader is shuffled, pass an unshuffled loader.

    Returns:
      weights: [N_train] (unnormalized ok for WeightedRandomSampler)
    """

    student.eval()

    scores: List[torch.Tensor] = []
    seen = 0

    for bi, batch in enumerate(train_loader):
        if max_batches is not None and bi >= max_batches:
            break

        pred, target, _loss = forward_batch_with_repo_model(student, batch, args)
        patch_mse = compute_patch_mse_from_outputs(pred, target, patch_len=args.patch_len)
        s = score_samples_from_patch_mse(patch_mse, patch_subset_in_pred_horizon)  # [B]
        scores.append(s.detach().to("cpu"))
        seen += s.numel()

    if not scores:
        raise RuntimeError("No batches processed when computing sampling weights")

    s_all = torch.cat(scores, dim=0)

    # If we early-stopped, pad remaining with mean score (keeps length consistent only if needed).
    n_train = len(train_loader.dataset)
    if s_all.numel() != n_train:
        if s_all.numel() > n_train:
            s_all = s_all[:n_train]
        else:
            pad = torch.full((n_train - s_all.numel(),), float(s_all.mean().item()))
            s_all = torch.cat([s_all, pad], dim=0)

    w = (s_all + float(eps)).pow(float(alpha))

    # Avoid all-zeros/NaNs
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if float(w.sum().item()) <= 0:
        w = torch.ones_like(w)

    return w


def alpha_schedule(epoch: int, total_epochs: int, alpha_start: float, alpha_end: float) -> float:
    if total_epochs <= 1:
        return float(alpha_end)
    t = epoch / (total_epochs - 1)
    return float(alpha_start + (alpha_end - alpha_start) * t)


# --------------------------------------------------------------------------------------
# Training / Eval (thin wrapper around existing experiment logic)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate(student: nn.Module, loader: DataLoader, args) -> Dict[str, float]:
    student.eval()
    losses = []
    for batch in loader:
        _pred, _target, loss = forward_batch_with_repo_model(student, batch, args)
        losses.append(float(loss.item()))
    return {"mse": float(np.mean(losses))}


def train_one_epoch(student: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, args) -> Dict[str, float]:
    student.train()
    losses = []
    for batch in loader:
        pred, target, _ = forward_batch_with_repo_model(student, batch, args)
        loss = F.mse_loss(pred, target)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    return {"mse": float(np.mean(losses))}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Timer MI-guided sample-level sampling (Scheme B)")

    # Keep aligned with timer_mi_distillation_comparison.py (data/model args)
    p.add_argument("--mi_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="exp")

    p.add_argument("--ckpt_path", type=str, required=True)

    p.add_argument("--root_path", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)

    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=0)
    p.add_argument("--patch_len", type=int, default=96)

    # data_provider required args
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--target", type=str, default="OT")
    p.add_argument("--task_name", type=str, default="forecast")
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)

    # teacher
    p.add_argument("--d_model_t", type=int, default=1024)
    p.add_argument("--d_ff_t", type=int, default=2048)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    # student
    p.add_argument("--s_d_model", type=int, default=256)
    p.add_argument("--s_d_ff", type=int, default=512)
    p.add_argument("--s_n_layers", type=int, default=4)
    p.add_argument("--s_n_heads", type=int, default=4)
    p.add_argument("--s_dropout", type=float, default=0.1)

    # sampling
    p.add_argument("--sampler_mode", type=str, default="uniform",
                   choices=["uniform", "high", "low", "random"],
                   help="Which patch subset drives sampling score")
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--alpha_start", type=float, default=2.0)
    p.add_argument("--alpha_end", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--max_score_batches", type=int, default=0,
                   help="If >0, compute sampling weights using only first N batches (debug/fast)")

    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args = ensure_data_args(args)   # fill data_provider required fields

    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = Path(args.out_dir) / "timer_mi_sample_sampling" / args.model_id
    safe_mkdir(out_dir)

    # Patch subsets
    n_patches = args.seq_len // args.patch_len
    mi_data = load_global_mi_json(args.mi_dir)
    high_patches, low_patches = build_patch_sets_from_mi_json(mi_data, n_patches=n_patches)

    # Map global patch indices -> predicted horizon patch indices.
    # Most Timer settings predict only last patch (pred_len==patch_len), so k=1.
    k_pred = 1 if (args.pred_len % args.patch_len != 0) else (args.pred_len // args.patch_len)

    def to_pred_horizon_indices(global_patches: Sequence[int]) -> List[int]:
        if k_pred <= 1:
            return [0]
        # predicted horizon corresponds to last k_pred patches of the input window.
        start = max(n_patches - k_pred, 0)
        mapped = [p - start for p in global_patches if start <= p < start + k_pred]
        return sorted({int(x) for x in mapped if 0 <= int(x) < k_pred})

    high_in_pred = to_pred_horizon_indices(high_patches)
    low_in_pred = to_pred_horizon_indices(low_patches)
    rand_in_pred = sample_random_patches(k_pred, k=len(high_in_pred) if high_in_pred else 1, seed=args.seed)

    if args.sampler_mode == "high":
        subset = high_in_pred
    elif args.sampler_mode == "low":
        subset = low_in_pred
    elif args.sampler_mode == "random":
        subset = rand_in_pred
    else:
        subset = [0]  # unused for uniform

    # Build model
    student = build_models_from_repo(args)

    # Build base loaders (no sampler) for score computation + uniform training
    train_loader_base, val_loader, test_loader, train_size = build_dataloaders_from_repo(args, sampling_weights=None)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "args": vars(args),
        "mi": {
            "n_patches": n_patches,
            "k_pred": k_pred,
            "high_patches": high_patches,
            "low_patches": low_patches,
            "high_in_pred": high_in_pred,
            "low_in_pred": low_in_pred,
            "rand_in_pred": rand_in_pred,
        },
        "epochs": [],
    }

    best_val = float("inf")
    best_path = out_dir / "best_student.pt"

    for ep in range(args.epochs):
        if ep < args.warmup_epochs or args.sampler_mode == "uniform":
            train_loader = train_loader_base
            sampler_alpha = None
            weight_stats = None
        else:
            a = alpha_schedule(ep, args.epochs, args.alpha_start, args.alpha_end)
            sampler_alpha = a
            max_batches = args.max_score_batches if args.max_score_batches > 0 else None
            w = compute_sampling_weights(
                student,
                train_loader_base,
                args,
                patch_subset_in_pred_horizon=subset,
                eps=args.eps,
                alpha=a,
                max_batches=max_batches,
            )
            weight_stats = {
                "min": float(w.min().item()),
                "mean": float(w.mean().item()),
                "max": float(w.max().item()),
            }
            # rebuild loader with sampler
            train_loader, _, _, _ = build_dataloaders_from_repo(args, sampling_weights=w)

        tr = train_one_epoch(student, train_loader, optim, args)
        va = evaluate(student, val_loader, args)
        te = evaluate(student, test_loader, args)

        rec = {
            "epoch": ep,
            "train": tr,
            "val": va,
            "test": te,
            "sampler": {
                "mode": args.sampler_mode,
                "alpha": sampler_alpha,
                "weights": weight_stats,
                "subset_in_pred": list(subset),
            },
        }
        history["epochs"].append(rec)

        if va["mse"] < best_val:
            best_val = va["mse"]
            torch.save({"state_dict": student.state_dict(), "epoch": ep, "val": va}, best_path)

        # lightweight progress log
        print(
            f"[Ep {ep:03d}] train={tr['mse']:.6f} val={va['mse']:.6f} test={te['mse']:.6f} "
            f"sampler={args.sampler_mode} alpha={sampler_alpha if sampler_alpha is not None else 'NA'}"
        )

    with open(out_dir / f"metrics_{now_ts()}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Best val={best_val:.6f}. Saved: {best_path}")


if __name__ == "__main__":
    main()
