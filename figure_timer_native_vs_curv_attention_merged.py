#!/usr/bin/env python3
"""
Merged figure: left = native Timer (geometric_hpe=0), right = Geometric-HPE with
v' = v * (1 + a * k_norm) (same k definition as layers/Embed.py GeometricHPEPatchEmbedding).
Footer: per-patch mean |Δ²x| curvature k on normalized patches (shared input batch).

Single PNG: two L×H attention grids + curvature line + optional IMS metrics.
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
import torch.nn as nn

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


def curvature_per_patch_normalized(
    batch_x: torch.Tensor, patch_len: int, stride: int, padding: int = 0
) -> torch.Tensor:
    """Mean |Δ²x| per patch after instance normalization (same as GeometricHPEPatchEmbedding)."""
    b, l, m = batch_x.shape
    means = batch_x.mean(1, keepdim=True)
    x = batch_x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x = x / stdev
    x = x.permute(0, 2, 1)
    pad = nn.ReplicationPad1d((0, padding))
    x = pad(x)
    x = x.unfold(dimension=-1, size=patch_len, step=stride)
    x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
    bm, n, _pl = x.shape
    d2 = torch.diff(x, n=2, dim=-1)
    if d2.numel() == 0:
        return torch.zeros(bm, n, device=x.device, dtype=x.dtype)
    return torch.mean(torch.abs(d2), dim=-1)


def build_config_native(args: argparse.Namespace, ckpt: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_name=args.task_name,
        ckpt_path=ckpt,
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
        geometric_hpe_periods="24,168",
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


def build_config_geo(args: argparse.Namespace, ckpt: str) -> SimpleNamespace:
    pe_w = int(getattr(args, "geometric_hpe_pe_curv_weighted", 1))
    ab_fix = int(getattr(args, "geometric_hpe_ab_fixed", 0))
    return SimpleNamespace(
        task_name=args.task_name,
        ckpt_path=ckpt,
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
        geometric_hpe=1,
        geometric_hpe_periods=getattr(args, "geometric_hpe_periods", "24,168"),
        geometric_hpe_curv_phase_scale=float(
            getattr(args, "geometric_hpe_curv_phase_scale", 1.0)
        ),
        geometric_hpe_curv_residual=0,
        geometric_hpe_res_lambda=0.0,
        geometric_hpe_linear_k_patch_scale=1,
        geometric_hpe_linear_k_a_init=float(
            getattr(args, "geometric_hpe_linear_k_a_init", 0.0)
        ),
        geometric_hpe_pe_curv_weighted=pe_w,
        geometric_hpe_pe_curv_b_init=float(
            getattr(args, "geometric_hpe_pe_curv_b_init", 0.01)
        ),
        geometric_hpe_ab_fixed=ab_fix,
        geometric_hpe_ablate_k=0,
        geometric_hpe_ablate_omega=0,
        geometric_hpe_ablate_phi=0,
        geometric_hpe_pe_proj_ones_input=0,
        geometric_hpe_only_linear_trainable=0,
    )


def attns_to_stack(attns: list, b_idx: int) -> np.ndarray:
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


def parse_args():
    p = argparse.ArgumentParser(description="Merged native vs Geo-HPE L×H attention + patch k footer")
    p.set_defaults(use_ims=True)
    p.add_argument("--ckpt_native", type=str, required=True)
    p.add_argument("--ckpt_geo", type=str, required=True)
    p.add_argument("--root_path", type=str, default="./datasets")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--data", type=str, default="ETTh1")
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
    p.add_argument("--metrics_max_batches", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./figure_attention_native_vs_curv_merged")
    p.add_argument("--output_name", type=str, default="attention_native_vs_curv_merged.png")
    p.add_argument("--periodic_embedding_branch", type=int, default=0)
    p.add_argument("--periodic_emb_bank_dim", type=int, default=0)
    p.add_argument("--geometric_hpe_periods", type=str, default="24,168")
    p.add_argument("--geometric_hpe_curv_phase_scale", type=float, default=1.0)
    p.add_argument("--geometric_hpe_linear_k_a_init", type=float, default=0.0)
    p.add_argument("--geometric_hpe_pe_curv_weighted", type=int, default=1)
    p.add_argument("--geometric_hpe_pe_curv_b_init", type=float, default=0.01)
    p.add_argument("--geometric_hpe_ab_fixed", type=int, default=0)
    p.add_argument("--skip_ims_metrics", action="store_true", help="Skip IMS MSE/MAE (faster)")
    return p.parse_args()


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

    ck_n = os.path.abspath(os.path.expanduser(args.ckpt_native.strip()))
    ck_g = os.path.abspath(os.path.expanduser(args.ckpt_geo.strip()))
    for path, tag in ((ck_n, "--ckpt_native"), (ck_g, "--ckpt_geo")):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {tag}: {path!r}")

    cfg_n = build_config_native(args, ck_n)
    cfg_g = build_config_geo(args, ck_g)

    _, data_loader = data_provider(cfg_n, flag="test")
    it = iter(data_loader)
    for _ in range(args.batch_index + 1):
        batch = next(it)
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -cfg_n.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : cfg_n.label_len, :], dec_inp], dim=1).float().to(device)

    stride = int(args.patch_len)
    k = curvature_per_patch_normalized(batch_x, args.patch_len, stride, padding=0)
    k_np = k[args.batch_eff_index].detach().float().cpu().numpy()
    n_tok = int(k_np.shape[0])

    mse_n, mae_n = float("nan"), float("nan")
    mse_g, mae_g = float("nan"), float("nan")
    if not args.skip_ims_metrics:
        print("IMS test: native Timer...", flush=True)
        m_native = TimerModel(cfg_n).float().to(device)
        m_native.eval()
        mse_n, mae_n = compute_ims_test_mse_mae(
            m_native, cfg_n, device, int(args.metrics_max_batches)
        )
        print(f"  native MSE={mse_n:.6f} MAE={mae_n:.6f}", flush=True)
        del m_native
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("IMS test: Geo-HPE Timer...", flush=True)
        m_geo = TimerModel(cfg_g).float().to(device)
        m_geo.eval()
        mse_g, mae_g = compute_ims_test_mse_mae(
            m_geo, cfg_g, device, int(args.metrics_max_batches)
        )
        print(f"  geo MSE={mse_g:.6f} MAE={mae_g:.6f}", flush=True)
        del m_geo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Attention forward: native...", flush=True)
    m1 = TimerModel(cfg_n).float().to(device)
    m1.eval()
    with torch.no_grad():
        _, att_n = m1(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    stack_n = attns_to_stack(att_n, args.batch_eff_index)
    del m1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Attention forward: Geo-HPE...", flush=True)
    m2 = TimerModel(cfg_g).float().to(device)
    m2.eval()
    with torch.no_grad():
        _, att_g = m2(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    stack_g = attns_to_stack(att_g, args.batch_eff_index)
    del m2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert stack_n.shape == stack_g.shape
    n_layers, n_heads, ny, nx = stack_n.shape
    assert ny == nx == n_tok

    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    cmap = "coolwarm"
    fig_w = max(14.0, 1.05 * n_heads * 1.12) * 2
    fig_h = max(10.0, 1.0 * n_layers * 1.05)

    fig = plt.figure(figsize=(fig_w, fig_h + 3.0), dpi=args.dpi)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.32],
        hspace=0.35,
        left=0.04,
        right=0.91,
        top=0.93,
        bottom=0.05,
    )
    top_row = gs[0, 0].subgridspec(1, 2, wspace=0.28)
    inner_l = top_row[0, 0].subgridspec(n_layers, n_heads, hspace=0.35, wspace=0.28)
    inner_r = top_row[0, 1].subgridspec(n_layers, n_heads, hspace=0.35, wspace=0.28)

    mappable = None
    for li in range(n_layers):
        for hi in range(n_heads):
            axl = fig.add_subplot(inner_l[li, hi])
            mappable = axl.imshow(
                stack_n[li, hi],
                cmap=cmap,
                norm=norm,
                aspect="equal",
                origin="upper",
                interpolation="nearest",
            )
            axr = fig.add_subplot(inner_r[li, hi])
            axr.imshow(
                stack_g[li, hi],
                cmap=cmap,
                norm=norm,
                aspect="equal",
                origin="upper",
                interpolation="nearest",
            )
            if n_tok <= 32:
                _integer_axis_ticks(axl, n_tok, labelsize=4.0)
                _integer_axis_ticks(axr, n_tok, labelsize=4.0)
            if li == 0:
                axl.set_title(f"H{hi}", fontsize=7, fontweight="semibold")
                axr.set_title(f"H{hi}", fontsize=7, fontweight="semibold")
            if hi == 0:
                axl.set_ylabel(f"L{li}", fontsize=7, fontweight="semibold")
                axr.set_ylabel(f"L{li}", fontsize=7, fontweight="semibold")

    fig.text(
        0.27,
        0.97,
        "A. Native Timer — PatchEmbedding + sin PE",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.73,
        0.97,
        r"B. Geometric-HPE — $v'=v(1+a\cdot k_{\mathrm{norm}})$ + harmonic PE",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    cbar_ax = fig.add_axes([0.935, 0.15, 0.014, 0.65])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Attention score", rotation=270, labelpad=14, fontsize=10)

    k_parts = [f"p{i}: {float(k_np[i]):.6f}" for i in range(n_tok)]
    k_line = (
        "Per-patch curvature k (mean |Δ²x| on normalized patches, same batch for both models): "
        + "  ".join(k_parts)
    )
    ims_line = (
        f"IMS test — Native: MSE={mse_n:.6f}, MAE={mae_n:.6f}  |  "
        f"Geo-HPE: MSE={mse_g:.6f}, MAE={mae_g:.6f}"
        if not args.skip_ims_metrics
        else "IMS test metrics skipped (--skip_ims_metrics)."
    )
    foot = k_line + "\n\n" + ims_line
    ax_note = fig.add_subplot(gs[1, 0])
    ax_note.axis("off")
    ax_note.text(0.5, 0.5, foot, ha="center", va="center", fontsize=7.5, wrap=True)

    out_path = os.path.join(args.output_dir, args.output_name)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "created": datetime.now().isoformat(),
        "ckpt_native": ck_n,
        "ckpt_geo": ck_g,
        "k_per_patch": k_np.tolist(),
        "attn_shape": list(stack_n.shape),
        "ims_native": {"mse": mse_n, "mae": mae_n},
        "ims_geo": {"mse": mse_g, "mae": mae_g},
        "skip_ims_metrics": bool(args.skip_ims_metrics),
        "output_png": os.path.abspath(out_path),
    }
    with open(
        os.path.join(args.output_dir, args.output_name.replace(".png", "_meta.json")),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved: {os.path.abspath(out_path)}", flush=True)


if __name__ == "__main__":
    main()
