#!/usr/bin/env python3
"""
ETTh1 + Timer: ablation to test whether high-HSIC patches matter for forecast error.

Per test batch (needs B>=4 for HSIC):
  1) Run encoder stack up to the last attention block (pre final LayerNorm).
  2) Pool patch reps -> HSIC per patch vs golden future; take top-2 "peak" patch indices.
  3) Pick 2 random patch indices from the rest (same RNG seed for reproducibility).
  4) Baseline: norm + proj + denorm (matches Timer.forecast).
  5) Ablate: zero (or Gaussian noise) those patch rows in [B*M, N, D], then norm + proj + denorm.
  6) MSE on IMS-aligned pred_len window vs batch_y (same slicing as exp_forecast vali test).

If peak ablation hurts MSE more than random ablation, peaks behave like informative "thinking" patches.

Multi-GPU: pass --use_multi_gpu under torchrun; SSE/MAE sums and batch/skip counts are all-reduced (full test set).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse namespace + HSIC from MI script (no torchrun / matplotlib figures here).
_mi_path = os.path.join(_ROOT, "experiments", "etth1_mi_hsic_peaks.py")
_spec = importlib.util.spec_from_file_location("etth1_mi_hsic_peaks", _mi_path)
_mi = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mi)
build_namespace = _mi.build_namespace
mi_sequence_hsic = _mi.mi_sequence_hsic
forward_y_collect_layers = _mi.forward_y_collect_layers

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.masking import TriangularCausalMask


def _unwrap(m: nn.Module) -> Model:
    return m.module if hasattr(m, "module") else m


def timer_hidden_last_block_pre_norm(model: nn.Module, x_enc: torch.Tensor):
    """
    x_enc: [B, L, M] raw batch_x. Returns h [B*M, N, D] after all EncoderLayers, BEFORE Encoder.norm.
    """
    core = _unwrap(model)
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape
    mask = TriangularCausalMask(BM, N, device=dec_in.device)
    h = dec_in
    for lyr in core.decoder.attn_layers:
        h, _ = lyr(h, attn_mask=mask)
    return h, means, stdev, B, M, int(n_vars), int(N), core


def timer_decode_from_hidden(core: Model, h: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor, B: int, M: int):
    """Apply final LayerNorm, proj, reshape, denormalize (Timer.forecast tail)."""
    h = core.decoder.norm(h)
    dec_out = core.proj(h)
    dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)
    dec_out = dec_out * stdev + means
    return dec_out


def ablate_patches(h: torch.Tensor, patch_idxs: list[int] | np.ndarray, mode: str, noise_std: float) -> torch.Tensor:
    """h: [BM, N, D]. Zero or add noise on listed patch indices (all variates)."""
    out = h.clone()
    for p in patch_idxs:
        p = int(p)
        if 0 <= p < out.shape[1]:
            if mode == "zero":
                out[:, p, :] = 0
            elif mode == "noise":
                out[:, p, :] = out[:, p, :] + noise_std * torch.randn_like(out[:, p, :])
            else:
                raise ValueError(mode)
    return out


def ims_pred_true(outputs: torch.Tensor, batch_y: torch.Tensor, seq_len: int, pred_len: int):
    """Match exp_forecast vali test branch for use_ims."""
    pred = outputs[:, -seq_len:, :][:, -pred_len:, :]
    true = batch_y[:, -pred_len:, :]
    return pred, true


def sse_mae(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float, int]:
    """Sum of squared errors, sum abs error, element count."""
    d = pred - true
    n = d.numel()
    return (d.pow(2).sum().item(), d.abs().sum().item(), int(n))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--root_path", type=str, default="./datasets/")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--data", type=str, default="ETTh1")
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--label_len", type=int, default=576)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--output_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=96)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--subset_rand_ratio", type=float, default=1.0)
    p.add_argument("--use_ims", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test loader")
    p.add_argument("--ablate_k", type=int, default=2, help="how many peak patches and how many random patches")
    p.add_argument("--ablate_mode", type=str, default="zero", choices=("zero", "noise"))
    p.add_argument("--noise_std", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="DDP + DistributedSampler (launch with torchrun); metrics are all-reduced on all test shards.",
    )
    args = p.parse_args()

    rank = 0
    local_rank = 0
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "use_multi_gpu requires torchrun (WORLD_SIZE set). "
                "Example: torchrun --nproc_per_node=4 experiments/etth1_patch_ablation_mse.py ... --use_multi_gpu"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("use_multi_gpu requires CUDA.")
        n_visible = torch.cuda.device_count()
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > n_visible:
            # 最后一层删掉mi值最高的一个patch（暴涨）vs 删掉随机一个低的patch（略涨）
            raise RuntimeError(
                f"WORLD_SIZE={world_size} but only {n_visible} CUDA device(s) visible. "
                f"Use torchrun --nproc_per_node={n_visible}."
            )
        if local_rank >= n_visible:
            raise RuntimeError(f"LOCAL_RANK={local_rank} >= visible devices={n_visible}.")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

    if not args.use_ims:
        print("Warning: this script's MSE slice matches IMS; pass --use_ims for ETTh1 Timer setup.")

    ns = build_namespace(args)
    ns.use_multi_gpu = bool(args.use_multi_gpu)
    ns.use_gpu = bool(torch.cuda.is_available())

    device = torch.device(f"cuda:{local_rank}") if args.use_multi_gpu else torch.device(args.device)
    _, loader = data_provider(ns, flag="test")
    model = Model(ns).to(device)
    model.eval()
    if args.use_multi_gpu:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Different RNG stream per rank so random-patch control is independent across shards.
    rng = np.random.RandomState(args.seed + rank * 100003)

    sum_sse_b = sum_sse_p = sum_sse_r = 0.0
    sum_abs_b = sum_abs_p = sum_abs_r = 0.0
    n_el = 0
    n_batches_used = 0
    skipped = 0

    with torch.no_grad():
        for _, (batch_x, batch_y, _, _) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            B = batch_x.shape[0]
            if B < 4:
                skipped += 1
                continue

            if args.use_ims:
                y_future = batch_y[:, ns.label_len : ns.label_len + ns.pred_len, :]
            else:
                y_future = batch_y[:, -ns.pred_len :, :]
            Y = y_future.reshape(B, -1).contiguous()

            core = _unwrap(model)
            h, means, stdev, B, M, n_vars, N, _ = timer_hidden_last_block_pre_norm(model, batch_x)
            if N < args.ablate_k + 1:
                skipped += 1
                continue

            Hbm = h.view(B, n_vars, N, -1).mean(dim=1)

            h_y_layers = forward_y_collect_layers(model, y_future)
            h_y = h_y_layers[-1]

            mi = mi_sequence_hsic(Hbm.cpu(), h_y.cpu())
            mi_f = np.nan_to_num(mi, nan=-np.inf)
            k = min(args.ablate_k, N)
            peak_idx = list(np.argsort(mi_f)[-k:][::-1].astype(int))
            rest = [i for i in range(N) if i not in peak_idx]
            if len(rest) < k:
                skipped += 1
                continue
            rand_idx = list(rng.choice(rest, size=k, replace=False))

            out_b = timer_decode_from_hidden(core, h, means, stdev, B, M)
            out_p = timer_decode_from_hidden(
                core, ablate_patches(h, peak_idx, args.ablate_mode, args.noise_std), means, stdev, B, M
            )
            out_r = timer_decode_from_hidden(
                core, ablate_patches(h, rand_idx, args.ablate_mode, args.noise_std), means, stdev, B, M
            )

            pred_b, true = ims_pred_true(out_b, batch_y, ns.seq_len, ns.pred_len)
            pred_p, _ = ims_pred_true(out_p, batch_y, ns.seq_len, ns.pred_len)
            pred_r, _ = ims_pred_true(out_r, batch_y, ns.seq_len, ns.pred_len)

            sb, ab, nb = sse_mae(pred_b, true)
            sp, ap, _ = sse_mae(pred_p, true)
            sr, ar, _ = sse_mae(pred_r, true)
            sum_sse_b += sb
            sum_sse_p += sp
            sum_sse_r += sr
            sum_abs_b += ab
            sum_abs_p += ap
            sum_abs_r += ar
            n_el += nb
            n_batches_used += 1

            if args.max_batches and n_batches_used >= args.max_batches:
                break

    if args.use_multi_gpu:
        buf = torch.tensor(
            [
                sum_sse_b,
                sum_sse_p,
                sum_sse_r,
                sum_abs_b,
                sum_abs_p,
                sum_abs_r,
                float(n_el),
                float(n_batches_used),
                float(skipped),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        sum_sse_b, sum_sse_p, sum_sse_r = buf[0].item(), buf[1].item(), buf[2].item()
        sum_abs_b, sum_abs_p, sum_abs_r = buf[3].item(), buf[4].item(), buf[5].item()
        n_el = int(round(buf[6].item()))
        n_batches_used = int(round(buf[7].item()))
        skipped = int(round(buf[8].item()))

    mse_b = sum_sse_b / max(n_el, 1)
    mse_p = sum_sse_p / max(n_el, 1)
    mse_r = sum_sse_r / max(n_el, 1)
    mae_b = sum_abs_b / max(n_el, 1)
    mae_p = sum_abs_p / max(n_el, 1)
    mae_r = sum_abs_r / max(n_el, 1)

    if args.use_multi_gpu:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0 or not args.use_multi_gpu:
        print("=== ETTh1 patch ablation (last block pre-LN, IMS pred_len MSE/MAE) ===")
        if args.use_multi_gpu:
            print("(metrics all-reduced over ranks; test set sharded by DistributedSampler)")
        print(
            f"batches_used={n_batches_used} skipped={skipped} elements={n_el} "
            f"ablate_k={args.ablate_k} mode={args.ablate_mode}"
        )
        print(f"Baseline     MSE={mse_b:.6f}  MAE={mae_b:.6f}")
        print(f"Peak-{args.ablate_k}     MSE={mse_p:.6f}  MAE={mae_p:.6f}  (delta_MSE_vs_base={mse_p - mse_b:+.6f})")
        print(f"Random-{args.ablate_k}   MSE={mse_r:.6f}  MAE={mae_r:.6f}  (delta_MSE_vs_base={mse_r - mse_b:+.6f})")
        print(f"Peak_vs_Random delta_MSE={mse_p - mse_r:+.6f}  (positive => peak ablation hurts more)")


if __name__ == "__main__":
    main()
