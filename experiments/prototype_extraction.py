#!/usr/bin/env python3
"""
Prototype Extraction for Prototype Prompting (Inspired by 'Thinking Tokens are Information Peaks')

This script extracts prototype vectors from the LAST decoder layer using global HSIC:
1. Forward pass through the model to collect hidden representations at the last layer
2. Compute HSIC (MI surrogate) between patch representations and future values
3. Identify top-K patches with highest HSIC values
4. Average all selected patches across all samples to form prototype vector
5. Save as .pt file for later use in Prototype Prompting

Usage:
    python experiments/prototype_extraction.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1

Multi-GPU (torchrun):
    torchrun --nnodes=1 --nproc_per_node=4 experiments/prototype_extraction.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 --use_multi_gpu
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.masking import TriangularCausalMask


def _median_sq_bandwidth(X: torch.Tensor) -> torch.Tensor:
    """Median heuristic: median pairwise squared Euclidean distance."""
    n = X.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    d = torch.cdist(X, X, p=2.0)
    triu = torch.triu_indices(n, n, offset=1, device=X.device)
    sq = d[triu[0], triu[1]] ** 2
    med = torch.median(sq)
    return med.clamp(min=1e-12)


def rbf_kernel(X: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(X, X, p=2.0) ** 2
    return torch.exp(-d2 / (2.0 * sigma_sq))


def hsic_unbiased_gaussian(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Unbiased HSIC with Gaussian RBF kernels.
    Bandwidth sigma^2 = median pairwise squared Euclidean distance.
    """
    n = X.shape[0]
    if n < 4:
        return torch.tensor(float("nan"), device=X.device, dtype=X.dtype)
    sx = _median_sq_bandwidth(X)
    sy = _median_sq_bandwidth(Y)
    K = rbf_kernel(X, sx)
    Lm = rbf_kernel(Y, sy)
    H = torch.eye(n, device=X.device, dtype=X.dtype) - (1.0 / n)
    Kc = H @ K @ H
    Lc = H @ Lm @ H
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)


def zscore_cols(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = t.mean(dim=0, keepdim=True)
    s = t.std(dim=0, keepdim=True).clamp(min=eps)
    return (t - m) / s


def compute_hsic_per_patch(
    Hbm: torch.Tensor,
    Hy: torch.Tensor,
) -> np.ndarray:
    """
    Compute HSIC for each patch using ALL samples globally (no per-batch).
    Hbm: [total_samples, N, D] patch representations from input (all batches concatenated)
    Hy: [total_samples, N_y, D] representations from future (all batches concatenated)
    Returns: [N] array of HSIC values per patch (each computed across all samples)
    """
    total_samples, N_x, D = Hbm.shape
    _, N_y, _ = Hy.shape

    hy_mean = Hy.mean(dim=1, keepdim=True)
    Hy_expanded = hy_mean.expand(-1, N_x, -1)

    out = []
    for p in range(N_x):
        X = zscore_cols(Hbm[:, p, :].contiguous())
        Y = zscore_cols(Hy_expanded[:, p, :].contiguous())
        out.append(hsic_unbiased_gaussian(X, Y).detach().float().cpu().item())
    return np.asarray(out, dtype=np.float64)


def _unwrap_timer(model: torch.nn.Module) -> Model:
    return model.module if hasattr(model, "module") else model


def forward_collect_hiddens(
    model: torch.nn.Module,
    x_enc: torch.Tensor,
    target_layer: int,
) -> torch.Tensor:
    """
    Forward pass through encoder up to target_layer and return hidden states.
    x_enc: [B, L, M] in Timer convention
    Returns: [B, N, D] hidden states at target layer
    """
    core = _unwrap_timer(model)
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
    for i, attn_layer in enumerate(core.decoder.attn_layers):
        if i > target_layer:
            break
        h, _, _ = attn_layer(h, attn_mask=mask, tau=None, delta=None)

    h_pooled = h.view(B, n_vars, N, D).mean(dim=1)
    return h_pooled


def forward_y_collect(
    model: torch.nn.Module,
    y: torch.Tensor,
    target_layer: int,
) -> torch.Tensor:
    """
    Forward pass for future values y.
    y: [B, L, M] future ground truth
    Returns: [B, N, D] hidden state at target layer (averaged over variables)
    """
    core = _unwrap_timer(model)
    B, L, M = y.shape

    means = y.mean(1, keepdim=True).detach()
    x = y - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev

    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    mask = TriangularCausalMask(BM, N, device=dec_in.device)
    h = dec_in
    for i, attn_layer in enumerate(core.decoder.attn_layers):
        if i > target_layer:
            break
        h, _, _ = attn_layer(h, attn_mask=mask, tau=None, delta=None)

    h_pooled = h.view(B, n_vars, N, D).mean(dim=1)
    return h_pooled


def build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Build complete config namespace for Model and data_provider."""
    ns = argparse.Namespace(**vars(args))
    defaults = {
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
        "model_id": "prototype_extraction",
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
        "dropout": 0.1,
        "activation": "gelu",
        "embed": "timeF",
        "freq": "h",
        "features": "M",
        "stride": 24,
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def main() -> None:
    p = argparse.ArgumentParser(description="Prototype Extraction for Prototype Prompting")
    p.add_argument("--ckpt_path", type=str, required=True, help="Timer model checkpoint path")
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
    p.add_argument("--stride", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=64, help="HSIC needs B>=4")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="./results/prototype_extraction")
    p.add_argument("--top_k", type=int, default=1,
                   help="Number of top/bottom HSIC patches to average as prototype (default: 1)")
    p.add_argument("--target_layer", type=int, default=-1,
                   help="Decoder layer to extract (0-indexed). -1 = last layer (default)")
    p.add_argument("--mi_mode", type=str, default="high", choices=["high", "low"],
                   help='MI selection mode: "high" (top HSIC), "low" (bottom HSIC)')
    p.add_argument("--use_ims", action="store_true", help="Use informative missing value strategy")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_multi_gpu", action="store_true", help="DDP with torchrun")
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test set")
    args = p.parse_args()

    rank = 0
    local_rank = 0
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Multi-GPU requires torchrun")
        if not torch.cuda.is_available():
            raise RuntimeError("use_multi_gpu requires CUDA")
        n_visible = torch.cuda.device_count()
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > n_visible:
            raise RuntimeError(f"WORLD_SIZE={world_size} but only {n_visible} CUDA device(s)")
        if local_rank >= n_visible:
            raise RuntimeError(f"LOCAL_RANK={local_rank} >= visible devices={n_visible}")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        print(f"[Prototype Extraction] rank {rank}/{world_size}", flush=True)

    ns = build_namespace(args)
    ns.use_multi_gpu = bool(args.use_multi_gpu)
    ns.use_gpu = bool(torch.cuda.is_available())

    out_dir = args.out_dir
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if args.use_multi_gpu:
        dist.barrier()

    device = torch.device(f"cuda:{local_rank}" if args.use_multi_gpu else args.device)

    _, loader = data_provider(ns, flag="test")
    model = Model(ns).to(device)
    model.eval()
    if args.use_multi_gpu:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    core = _unwrap_timer(model)
    n_layers = len(core.decoder.attn_layers)

    if args.target_layer < 0:
        target_layer = n_layers - 1
    else:
        if args.target_layer >= n_layers:
            raise ValueError(f"target_layer {args.target_layer} out of range (0-{n_layers - 1})")
        target_layer = args.target_layer

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Prototype Extraction Configuration:")
        print(f"  Total decoder layers: {n_layers}")
        print(f"  Target layer: {target_layer} ({'last layer' if target_layer == n_layers - 1 else 'layer ' + str(target_layer) + ' (0-indexed)'})")
        print(f"  MI mode: {args.mi_mode} (high=highest HSIC, low=lowest HSIC)")
        print(f"  Top-K patches: {args.top_k}")
        print(f"  Out Dir: {out_dir}")
        print(f"{'='*60}\n")

    h_x_list: list[torch.Tensor] = []
    h_y_list: list[torch.Tensor] = []
    n_batches = 0

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        if args.use_ims:
            y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
        else:
            y_future = batch_y[:, -ns.pred_len:, :]

        with torch.no_grad():
            h_x = forward_collect_hiddens(model, batch_x, target_layer)
            h_y = forward_y_collect(model, y_future, target_layer)
            h_x_list.append(h_x.detach())
            h_y_list.append(h_y.detach())

        n_batches += 1
        if args.max_batches and n_batches >= args.max_batches:
            break

        if batch_idx % 20 == 0:
            print(f"[rank {rank}] Batch {batch_idx}", flush=True)

    print(f"[rank {rank}] Data collection done: {n_batches} batches")

    if args.use_multi_gpu:
        dist.barrier()
        world_size = int(os.environ["WORLD_SIZE"])

        local_h_x = torch.cat(h_x_list, dim=0)
        local_h_y = torch.cat(h_y_list, dim=0)

        local_num = torch.tensor([local_h_x.shape[0]], device=device, dtype=torch.long)
        num_list = [torch.zeros_like(local_num) for _ in range(world_size)]
        dist.all_gather(num_list, local_num)
        num_list = [n.item() for n in num_list]

        x_list = [torch.zeros(n, local_h_x.shape[1], local_h_x.shape[2], device=device, dtype=local_h_x.dtype) for n in num_list]
        y_list = [torch.zeros(n, local_h_y.shape[1], local_h_y.shape[2], device=device, dtype=local_h_y.dtype) for n in num_list]
        dist.all_gather(x_list, local_h_x)
        dist.all_gather(y_list, local_h_y)

        h_x_all = torch.cat(x_list, dim=0)
        h_y_all = torch.cat(y_list, dim=0)
        del x_list, y_list, local_h_x, local_h_y
        print(f"[rank {rank}] Gathered {h_x_all.shape[0]} samples across {world_size} GPUs")
    else:
        h_x_all = torch.cat(h_x_list, dim=0)
        h_y_all = torch.cat(h_y_list, dim=0)
        print(f"[rank {rank}] Total samples: {h_x_all.shape[0]}")

    hsic_per_patch = compute_hsic_per_patch(h_x_all, h_y_all)

    if args.use_multi_gpu:
        hsic_tensor = torch.from_numpy(hsic_per_patch).to(device)
        dist.broadcast(hsic_tensor, src=0)
        hsic_per_patch = hsic_tensor.cpu().numpy()

    top_k = min(args.top_k, h_x_all.shape[1])
    if args.mi_mode == "high":
        sel_indices = np.argsort(hsic_per_patch)[-top_k:][::-1]
    else:
        sel_indices = np.argsort(hsic_per_patch)[:top_k]
    sel_indices = np.sort(sel_indices)
    print(f"[rank {rank}] {args.mi_mode.capitalize()}-MI patch indices: {sel_indices.tolist()}")
    print(f"[rank {rank}] HSIC values: {hsic_per_patch[sel_indices].tolist()}")

    h_sel = h_x_all[:, sel_indices, :].reshape(-1, core.d_model)
    h_proto = h_sel.mean(dim=0, keepdim=True)

    del h_x_all, h_y_all

    if args.use_multi_gpu:
        dist.barrier()

    if rank == 0 or not args.use_multi_gpu:
        mode_prefix = "high" if args.mi_mode == "high" else "low"
        prototype_path = os.path.join(out_dir, f"h_proto_{mode_prefix}_top{top_k}_layer{target_layer}.pt")
        h_proto_save = h_proto.unsqueeze(0)
        torch.save(h_proto_save, prototype_path)

        metadata = {
            "h_proto": h_proto_save,
            "target_layer": target_layer,
            "mi_mode": args.mi_mode,
            "top_k": top_k,
            "sel_indices": sel_indices,
            "hsic_per_patch": hsic_per_patch,
            "d_model": core.d_model,
            "seq_len": ns.seq_len,
            "pred_len": ns.pred_len,
            "patch_len": ns.patch_len,
        }
        metadata_path = os.path.join(out_dir, f"prototype_metadata_{mode_prefix}_top{top_k}_layer{target_layer}.pt")
        torch.save(metadata, metadata_path)

        print(f"  Saved: {prototype_path}")
        print(f"  Prototype shape: {h_proto_save.shape}, selected patches: {sel_indices.tolist()}")

    if args.use_multi_gpu:
        dist.destroy_process_group()

    if rank == 0 or not args.use_multi_gpu:
        print(f"\n{'='*60}")
        print(f"Prototype extraction done! out_dir={out_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
