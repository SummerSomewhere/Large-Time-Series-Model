#!/usr/bin/env python3
"""
Native Timer: encoder self-attention heatmaps for every layer and head.
Uses default PatchEmbedding + sinusoidal PE (--geometric_hpe 0); no curvature scaling on patches.

Defaults align with scripts/forecast/ETTh1.sh (weather, seq_len=672, patch_len=96, ...).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from data_provider.data_factory import data_provider
from models.Timer import Model as TimerModel
from utils.metrics import metric as metric_np


def build_config_ns(args: argparse.Namespace) -> SimpleNamespace:
    """Standard Timer backbone (no Geometric-HPE)."""
    ns = SimpleNamespace(
        task_name=args.task_name,
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        output_len=args.output_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        factor=args.factor,
        activation=args.activation,
        output_attention=True,
        features=args.features,
        root_path=args.root_path,
        data_path=args.data_path,
        data=args.data,
        embed=args.embed,
        freq=args.freq,
        stride=args.stride,
        subset_rand_ratio=args.subset_rand_ratio,
        use_ims=args.use_ims,
        batch_size=1,
        num_workers=int(args.num_workers),
        use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(),
        gpu=args.gpu,
        inverse=bool(getattr(args, "inverse", False)),
        periodic_embedding_branch=getattr(args, "periodic_embedding_branch", 0),
        periodic_emb_bank_dim=getattr(args, "periodic_emb_bank_dim", 0),
        geometric_hpe=0,
        geometric_hpe_periods=getattr(args, "geometric_hpe_periods", "24,168"),
        geometric_hpe_curv_phase_scale=1.0,
        geometric_hpe_curv_residual=0,
        geometric_hpe_res_lambda=0.0,
        geometric_hpe_linear_k_patch_scale=0,
        geometric_hpe_linear_k_a_init=0.01,
        geometric_hpe_pe_curv_weighted=0,
        geometric_hpe_pe_curv_b_init=0.01,
        geometric_hpe_ab_fixed=0,
        geometric_hpe_ablate_k=0,
        geometric_hpe_ablate_omega=0,
        geometric_hpe_ablate_phi=0,
        geometric_hpe_pe_proj_ones_input=0,
        geometric_hpe_only_linear_trainable=0,
    )
    return ns


def parse_args():
    p = argparse.ArgumentParser(
        description="Native Timer: L×H self-attention maps (default embed + sin PE, no curvature scaling)"
    )
    p.set_defaults(use_ims=True)
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint (.ckpt or .pth) for standard Timer (geometric_hpe=0).",
    )
    p.add_argument("--root_path", type=str, default="./datasets")
    p.add_argument("--data_path", type=str, default="weather.csv")
    p.add_argument("--data", type=str, default="weather")
    p.add_argument("--inverse", action="store_true")
    p.add_argument("--task_name", type=str, default="forecast")
    p.add_argument("--features", type=str, default="M")
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
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--subset_rand_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_use_ims", action="store_false", dest="use_ims")
    p.add_argument("--batch_index", type=int, default=0)
    p.add_argument("--batch_eff_index", type=int, default=0)
    p.add_argument(
        "--metrics_max_batches",
        type=int,
        default=0,
        help="IMS test MSE/MAE: max batches (0 = full test).",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./figure_attention_timer_layers_heads")
    p.add_argument("--output_name", type=str, default="attention_layers_heads_native.png")
    p.add_argument("--periodic_embedding_branch", type=int, default=0)
    p.add_argument("--periodic_emb_bank_dim", type=int, default=0)
    return p.parse_args()


def attns_to_stack(attns: list, b_idx: int) -> np.ndarray:
    """[L, H, N, N] float."""
    t = torch.stack([layer[b_idx].detach().float().cpu() for layer in attns], dim=0)
    return np.nan_to_num(t.numpy(), nan=0.0, posinf=0.0, neginf=0.0)


def compute_ims_test_mse_mae(
    model: TimerModel,
    cfg: SimpleNamespace,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    cfg_m = copy.copy(cfg)
    if getattr(cfg_m, "output_len_list", None) is None:
        cfg_m.output_len_list = [cfg_m.output_len]

    test_data, test_loader = data_provider(cfg_m, flag="test")
    preds_list: list[torch.Tensor] = []
    trues_list: list[torch.Tensor] = []
    f_dim = -1 if cfg_m.features == "MS" else 0

    model.eval()
    with torch.no_grad():
        for bi, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -cfg_m.pred_len :, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, : cfg_m.label_len, :], dec_inp], dim=1
            ).float().to(device)

            inference_steps = cfg_m.output_len // cfg_m.pred_len
            dis = cfg_m.output_len - inference_steps * cfg_m.pred_len
            if dis != 0:
                inference_steps += 1
            pred_y: list[torch.Tensor] = []
            for j in range(inference_steps):
                if len(pred_y) != 0:
                    batch_x = torch.cat([batch_x[:, cfg_m.pred_len :, :], pred_y[-1]], dim=1)
                    tmp = batch_y_mark[:, j - 1 : j, :]
                    batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                if cfg_m.output_attention:
                    outputs, _att = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred_y.append(outputs[:, -cfg_m.pred_len :, :])
            pred_y = torch.cat(pred_y, dim=1)

            if dis != 0:
                pred_y = pred_y[:, : -cfg_m.pred_len + dis, :]

            if cfg_m.use_ims:
                batch_y_tgt = batch_y[
                    :, cfg_m.label_len : cfg_m.label_len + cfg_m.output_len, :
                ].to(device)
            else:
                batch_y_tgt = batch_y[:, : cfg_m.output_len, :].to(device)

            outputs = pred_y.detach().cpu()
            batch_y_tgt = batch_y_tgt.detach().cpu()

            if test_data.scale and bool(getattr(cfg_m, "inverse", False)):
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y_tgt = test_data.inverse_transform(
                    batch_y_tgt.squeeze(0)
                ).reshape(shape)

            outputs = outputs[:, :, f_dim:]
            batch_y_tgt = batch_y_tgt[:, :, f_dim:]

            preds_list.append(outputs)
            trues_list.append(batch_y_tgt)

            if max_batches > 0 and (bi + 1) >= max_batches:
                break

    if not preds_list:
        return float("nan"), float("nan")

    pred_all = torch.cat(preds_list, dim=0).numpy()
    true_all = torch.cat(trues_list, dim=0).numpy()
    mae, mse, _r, _mape, _mspe = metric_np(pred_all, true_all)
    return float(mse), float(mae)


def _integer_axis_ticks(ax, n_tokens: int, *, labelsize: float | None = None) -> None:
    ticks = np.arange(n_tokens)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=labelsize)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ckpt = os.path.abspath(os.path.expanduser(str(args.ckpt_path).strip()))
    if not ckpt or not os.path.isfile(ckpt):
        raise SystemExit(f"Missing checkpoint: {ckpt!r}")

    cfg = build_config_ns(args)
    cfg.ckpt_path = ckpt

    _, data_loader = data_provider(cfg, flag="test")
    it = iter(data_loader)
    for _ in range(args.batch_index + 1):
        batch = next(it)
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : cfg.label_len, :], dec_inp], dim=1).float().to(
        device
    )

    print("IMS test metrics (native Timer)...", flush=True)
    model = TimerModel(cfg).float().to(device)
    model.eval()
    mse_v, mae_v = compute_ims_test_mse_mae(
        model, cfg, device, int(args.metrics_max_batches)
    )
    print(f"  MSE={mse_v:.6f}  MAE={mae_v:.6f}", flush=True)

    print("Attention forward (one batch)...", flush=True)
    with torch.no_grad():
        _, attns = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    if not isinstance(attns, list) or len(attns) == 0:
        raise RuntimeError("No attention outputs; set output_attention and check model.")
    stack = attns_to_stack(attns, args.batch_eff_index)
    n_layers, n_heads, ny, nx = stack.shape
    assert ny == nx
    n_tok = int(ny)

    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    cmap = "coolwarm"

    fig_w = max(14.0, 1.1 * n_heads * 1.15)
    fig_h = max(10.0, 1.0 * n_layers * 1.05)
    fig = plt.figure(figsize=(fig_w, fig_h + 2.0), dpi=args.dpi)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.18],
        hspace=0.28,
        left=0.06,
        right=0.92,
        top=0.92,
        bottom=0.06,
    )
    inner = gs[0, 0].subgridspec(n_layers, n_heads, hspace=0.32, wspace=0.26)
    mappable = None
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = fig.add_subplot(inner[li, hi])
            mappable = ax.imshow(
                stack[li, hi],
                cmap=cmap,
                norm=norm,
                aspect="equal",
                origin="upper",
                interpolation="nearest",
            )
            if n_tok <= 32:
                _integer_axis_ticks(ax, n_tok, labelsize=4.5)
            if li == 0:
                ax.set_title(f"H{hi}", fontsize=8, fontweight="semibold")
            if hi == 0:
                ax.set_ylabel(f"L{li}", fontsize=8, fontweight="semibold")
            if li == n_layers - 1:
                ax.set_xlabel("tgt", fontsize=6)

    fig.suptitle(
        "Native Timer — PatchEmbedding + sinusoidal PE (no curvature scaling)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    cbar_ax = fig.add_axes([0.935, 0.18, 0.018, 0.62])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Attention score", rotation=270, labelpad=16, fontsize=10)

    foot = (
        f"IMS test: MSE={mse_v:.6f}, MAE={mae_v:.6f}  |  num_patches={n_tok}  |  ckpt: {ckpt}"
    )
    ax_note = fig.add_subplot(gs[1, 0])
    ax_note.axis("off")
    ax_note.text(0.5, 0.5, foot, ha="center", va="center", fontsize=8.0, wrap=True)

    out_path = os.path.join(args.output_dir, args.output_name)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "created": datetime.now().isoformat(),
        "ckpt_path": ckpt,
        "embedding": "native Timer (geometric_hpe=0)",
        "ims_test": {"mse": mse_v, "mae": mae_v},
        "metrics_max_batches": int(args.metrics_max_batches),
        "attn_shape": list(stack.shape),
        "seq_len": args.seq_len,
        "patch_len": args.patch_len,
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "subset_rand_ratio": float(args.subset_rand_ratio),
        "data": args.data,
        "data_path": args.data_path,
        "root_path": args.root_path,
        "num_patches": int(n_tok),
        "output_png": os.path.abspath(out_path),
    }
    meta_path = os.path.join(
        args.output_dir, args.output_name.replace(".png", "_meta.json")
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved: {os.path.abspath(out_path)}", flush=True)


if __name__ == "__main__":
    main()
