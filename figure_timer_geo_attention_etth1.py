#!/usr/bin/env python3
"""
Figure: Default Timer embed vs Geometric-HPE — full encoder self-attention
(7 patches with patch_len=96, seq_len=672 → (672−96)/96+1 = 7 when stride=patch_len).

Defaults align with scripts/forecast/ETTh1.sh: root_path ./datasets, data weather, weather.csv.

Left (--ckpt_timer): native Timer — default PatchEmbedding + sinusoidal PE (no v·(1+a·k)).
Right (--ckpt_geo): Geometric-HPE — harmonic PE + v·(1+a·k_norm) when trained so.

IMS test MSE/MAE use the same multi-step protocol as exp_forecast.test (use_ims=True).
Per-patch k is mean |Δ²x| on normalized patches (same as GeometricHPEPatchEmbedding).
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


def _integer_axis_ticks(ax, n_tokens: int, *, labelsize: float | None = None) -> None:
    ticks = np.arange(n_tokens)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=labelsize)


def curvature_per_patch_normalized(batch_x: torch.Tensor, patch_len: int, stride: int, padding: int = 0):
    """
    batch_x: [B, L, M] as in the dataloader. Same normalization + unfold as Timer.forecast + GeometricHPE.
    Returns k: [B*M, N] curvature per patch index.
    """
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
        k = torch.zeros(bm, n, device=x.device, dtype=x.dtype)
    else:
        k = torch.mean(torch.abs(d2), dim=-1)
    return k


def build_config_ns(args: argparse.Namespace, *, geo: bool) -> SimpleNamespace:
    gh = 1 if geo else 0
    lin_k = 1 if geo else 0
    # When geo: must match run.py finetune flags for CKPT_GEO (pe_curv_weighted=0 => feature-only v·(1+a·k_norm))
    pe_w = int(getattr(args, "geometric_hpe_pe_curv_weighted", 0)) if geo else 0
    ns = SimpleNamespace(
        task_name=args.task_name,
        ckpt_path=args.ckpt_geo if geo else args.ckpt_timer,
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
        geometric_hpe=gh,
        geometric_hpe_periods=getattr(args, "geometric_hpe_periods", "24,168"),
        geometric_hpe_curv_phase_scale=float(getattr(args, "geometric_hpe_curv_phase_scale", 1.0)),
        geometric_hpe_curv_residual=int(getattr(args, "geometric_hpe_curv_residual", 0)),
        geometric_hpe_res_lambda=float(getattr(args, "geometric_hpe_res_lambda", 0.0)),
        geometric_hpe_linear_k_patch_scale=lin_k,
        geometric_hpe_linear_k_a_init=float(getattr(args, "geometric_hpe_linear_k_a_init", 0.01)),
        geometric_hpe_pe_curv_weighted=pe_w,
        geometric_hpe_pe_curv_b_init=float(getattr(args, "geometric_hpe_pe_curv_b_init", 0.01)),
        geometric_hpe_ab_fixed=int(getattr(args, "geometric_hpe_ab_fixed", 0)),
        geometric_hpe_ablate_k=int(getattr(args, "geometric_hpe_ablate_k", 0)),
        geometric_hpe_ablate_omega=int(getattr(args, "geometric_hpe_ablate_omega", 0)),
        geometric_hpe_ablate_phi=int(getattr(args, "geometric_hpe_ablate_phi", 0)),
        geometric_hpe_pe_proj_ones_input=int(getattr(args, "geometric_hpe_pe_proj_ones_input", 0)),
        geometric_hpe_only_linear_trainable=int(getattr(args, "geometric_hpe_only_linear_trainable", 0)),
    )
    return ns


def parse_args():
    p = argparse.ArgumentParser(
        description="Native Timer (default embed) vs Geometric-HPE attention (defaults: ETTh1.sh weather data)"
    )
    p.set_defaults(use_ims=True)
    p.add_argument(
        "--ckpt_timer",
        type=str,
        required=True,
        help="Native Timer: default PatchEmbedding + sin PE; no v·(1+a·k); path to .ckpt or .pth",
    )
    p.add_argument(
        "--ckpt_geo",
        type=str,
        required=True,
        help="Geometric-HPE Timer: harmonic PE + v·(1+a·k_norm) when linear_k_patch_scale=1; align flags with training",
    )
    p.add_argument("--root_path", type=str, default="./datasets")
    p.add_argument("--data_path", type=str, default="weather.csv")
    p.add_argument("--data", type=str, default="weather")
    p.add_argument("--inverse", action="store_true", help="Inverse-transform preds/trues in IMS metrics (match run.py --inverse)")
    p.add_argument("--task_name", type=str, default="forecast")
    p.add_argument("--features", type=str, default="M")
    # Defaults: 7 patches with patch_len=96, stride=patch_len -> seq_len = 7*96 = 672 (same as ETTh1.sh)
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
    p.add_argument("--subset_rand_ratio", type=float, default=1.0, help="Match run.py / ETTh1.sh (subset_rand_ratio=1)")
    p.add_argument("--seed", type=int, default=1, help="Match ETTh1.sh --seed")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (ETTh1.sh --num_workers 4; test still uses batch_size=1)")
    p.add_argument("--no_use_ims", action="store_false", dest="use_ims")
    p.add_argument("--batch_index", type=int, default=0)
    p.add_argument("--batch_eff_index", type=int, default=0, help="Index along B*vars after patch embed")
    p.add_argument(
        "--metrics_max_batches",
        type=int,
        default=0,
        help="IMS test MSE/MAE: max test batches (0 = full test set; same as exp_forecast.test).",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./figure_attention_etth1_compare")
    p.add_argument("--output_name", type=str, default="figure_timer_vs_geo_tsfm_etth1.png")
    p.add_argument("--periodic_embedding_branch", type=int, default=0)
    p.add_argument("--periodic_emb_bank_dim", type=int, default=0)
    p.add_argument("--geometric_hpe_periods", type=str, default="24,168")
    p.add_argument("--geometric_hpe_curv_phase_scale", type=float, default=1.0)
    p.add_argument("--geometric_hpe_curv_residual", type=int, default=0)
    p.add_argument("--geometric_hpe_res_lambda", type=float, default=0.0)
    p.add_argument("--geometric_hpe_linear_k_a_init", type=float, default=0.01)
    p.add_argument(
        "--geometric_hpe_pe_curv_weighted",
        type=int,
        default=0,
        help="Geo model only: 1 = PE·(1+b·k_norm) like full GeoHPE; 0 = feature-only v·(1+a·k_norm)",
    )
    p.add_argument("--geometric_hpe_pe_curv_b_init", type=float, default=0.01)
    p.add_argument("--geometric_hpe_ab_fixed", type=int, default=0)
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
    """
    Same IMS protocol as exp_forecast.test when use_ims=True (multi-step rollout).
    Returns (mse, mae) in the same scale as exp (normalized unless inverse).
    """
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
            dec_inp = torch.cat([batch_y[:, : cfg_m.label_len, :], dec_inp], dim=1).float().to(device)

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
                batch_y_tgt = batch_y[:, cfg_m.label_len : cfg_m.label_len + cfg_m.output_len, :].to(device)
            else:
                batch_y_tgt = batch_y[:, : cfg_m.output_len, :].to(device)

            outputs = pred_y.detach().cpu()
            batch_y_tgt = batch_y_tgt.detach().cpu()

            if test_data.scale and bool(getattr(cfg_m, "inverse", False)):
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y_tgt = test_data.inverse_transform(batch_y_tgt.squeeze(0)).reshape(shape)

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

    ck_timer = os.path.abspath(os.path.expanduser(str(args.ckpt_timer).strip()))
    ck_geo = os.path.abspath(os.path.expanduser(str(args.ckpt_geo).strip()))
    for path, label in (
        (ck_timer, "--ckpt_timer (native Timer, default embed)"),
        (ck_geo, "--ckpt_geo (Geometric-HPE + v scaling)"),
    ):
        if not path or not os.path.isfile(path):
            raise SystemExit(
                f"Missing or invalid checkpoint for {label}: {path!r}. "
                "Set valid --ckpt_timer and --ckpt_geo paths."
            )

    cfg_timer = build_config_ns(args, geo=False)
    cfg_geo = build_config_ns(args, geo=True)
    cfg_timer.ckpt_path = ck_timer
    cfg_geo.ckpt_path = ck_geo

    _, data_loader = data_provider(cfg_timer, flag="test")
    it = iter(data_loader)
    for _ in range(args.batch_index + 1):
        batch = next(it)
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -cfg_timer.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : cfg_timer.label_len, :], dec_inp], dim=1).float().to(device)

    # Curvature on the same normalized patches as the model (Timer stride = patch_len in backbone)
    stride = cfg_timer.patch_len
    k = curvature_per_patch_normalized(batch_x, cfg_timer.patch_len, stride, padding=0)
    k_np = k[args.batch_eff_index].detach().float().cpu().numpy()
    n_tok = int(k_np.shape[0])
    # Timer backbone: stride = patch_len → num_patches = (seq_len - patch_len) // patch_len + 1 (default 7).
    n_from_cfg = (int(args.seq_len) - int(args.patch_len)) // int(args.patch_len) + 1
    if n_from_cfg != n_tok:
        raise SystemExit(f"Inconsistent patch count: curvature N={n_tok} vs formula ({args.seq_len},{args.patch_len})→{n_from_cfg}")
    print(f"Patches along sequence: {n_tok} (seq_len={args.seq_len}, patch_len={args.patch_len})", flush=True)

    print("IMS test metrics: native Timer...", flush=True)
    m_timer = TimerModel(cfg_timer).float().to(device)
    m_timer.eval()
    mse_t, mae_t = compute_ims_test_mse_mae(m_timer, cfg_timer, device, int(args.metrics_max_batches))
    print(f"  Native Timer  MSE={mse_t:.6f}  MAE={mae_t:.6f}", flush=True)

    print("Attention forward: native Timer (one batch)...", flush=True)
    with torch.no_grad():
        _out_t, att_timer = m_timer(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    if not isinstance(att_timer, list) or len(att_timer) == 0:
        raise RuntimeError("No attention from native Timer; check checkpoint.")
    stack_a = attns_to_stack(att_timer, args.batch_eff_index)
    del m_timer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("IMS test metrics: Geometric-HPE...", flush=True)
    m_geo = TimerModel(cfg_geo).float().to(device)
    m_geo.eval()
    mse_g, mae_g = compute_ims_test_mse_mae(m_geo, cfg_geo, device, int(args.metrics_max_batches))
    print(f"  Geometric-HPE  MSE={mse_g:.6f}  MAE={mae_g:.6f}", flush=True)

    print("Attention forward: Geometric-HPE (one batch)...", flush=True)
    with torch.no_grad():
        _out_g, att_geo = m_geo(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    if not isinstance(att_geo, list) or len(att_geo) == 0:
        raise RuntimeError("No attention from Geo model; check checkpoint.")
    stack_b = attns_to_stack(att_geo, args.batch_eff_index)
    del m_geo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert stack_a.shape == stack_b.shape
    n_layers, n_heads, ny, nx = stack_a.shape
    assert ny == nx == n_tok, f"Attention N={ny} vs curvature N={n_tok}; check seq_len/patch_len."
    idx_hi = max(0, n_tok - 1)
    axis_label_y = f"Patch Index 0 to {idx_hi} (Source)"
    axis_label_x = f"Patch Index 0 to {idx_hi} (Target)"

    # Shared Blue–White–Red scale on [0, 1] (softmax weights)
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    cmap = "coolwarm"

    fig = plt.figure(figsize=(20, 12), dpi=args.dpi)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.26],
        width_ratios=[1.0, 1.0],
        hspace=0.30,
        wspace=0.22,
        left=0.06,
        right=0.93,
        top=0.90,
        bottom=0.06,
    )

    mappable = None

    inner_l = gs[0, 0].subgridspec(n_layers, n_heads, hspace=0.35, wspace=0.28)
    inner_r = gs[0, 1].subgridspec(n_layers, n_heads, hspace=0.35, wspace=0.28)

    for li in range(n_layers):
        for hi in range(n_heads):
            axl = fig.add_subplot(inner_l[li, hi])
            axr = fig.add_subplot(inner_r[li, hi])
            mappable = axl.imshow(
                stack_a[li, hi],
                cmap=cmap,
                norm=norm,
                aspect="equal",
                origin="upper",
                interpolation="nearest",
            )
            axr.imshow(
                stack_b[li, hi],
                cmap=cmap,
                norm=norm,
                aspect="equal",
                origin="upper",
                interpolation="nearest",
            )
            if n_tok <= 32:
                _integer_axis_ticks(axl, n_tok, labelsize=5.0)
                _integer_axis_ticks(axr, n_tok, labelsize=5.0)
            if li == 0:
                axl.set_title(f"H{hi}", fontsize=8, fontweight="semibold")
                axr.set_title(f"H{hi}", fontsize=8, fontweight="semibold")
            if hi == 0:
                axl.set_ylabel(f"L{li}", fontsize=8, fontweight="semibold")
                axr.set_ylabel(f"L{li}", fontsize=8, fontweight="semibold")
            if li == n_layers - 1:
                axl.set_xlabel("Target", fontsize=6)
                axr.set_xlabel("Target", fontsize=6)

    # Panel titles: left = default PatchEmbedding + sin PE, no v boost; right = Geometric-HPE (not default PE)
    _pew = int(getattr(args, "geometric_hpe_pe_curv_weighted", 0))
    _b_pe_line = (
        "pe_curv_weighted=0: no PE·k scaling"
        if _pew == 0
        else "pe_curv_weighted=1: PE also scaled by k_norm"
    )
    fig.text(
        0.27,
        0.97,
        "A. Native Timer — default embed + sin PE\n(no v·(1+a·k), not Geometric-HPE)",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.73,
        0.97,
        r"B. Geometric-HPE — $v'=v(1+a\cdot k)$ + harmonic PE" + f"\n({_b_pe_line})",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
    )

    # Shared axis labels (outer)
    fig.text(
        0.02,
        0.52,
        axis_label_y,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(0.5, 0.02, axis_label_x, ha="center", va="bottom", fontsize=12)

    # Colorbar (attention score 0–1)
    cbar_ax = fig.add_axes([0.94, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Attention score", rotation=270, labelpad=18, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    k_parts = [f"p{i}: {float(k_np[i]):.6f}" for i in range(n_tok)]
    k_line = "Per-patch curvature k (mean |Δ²x| on normalized patches): " + "  ".join(k_parts)
    metrics_line = (
        f"IMS test — Native Timer: MSE={mse_t:.6f}, MAE={mae_t:.6f}    |    "
        f"Geometric-HPE: MSE={mse_g:.6f}, MAE={mae_g:.6f}"
    )
    foot_text = (
        k_line
        + "\n\n"
        + metrics_line
        + "\n\n"
        + "Implicit Spectral Equalization: Geometric Prior counteracts Spectral Bias, inducing stable, "
        "high-frequency focal points across foundation model depths."
    )

    ax_note = fig.add_subplot(gs[1, :])
    ax_note.axis("off")
    ax_note.text(
        0.5,
        0.5,
        foot_text,
        ha="center",
        va="center",
        fontsize=8.5,
        wrap=True,
    )

    out_path = os.path.join(args.output_dir, args.output_name)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "created": datetime.now().isoformat(),
        "ckpt_timer": ck_timer,
        "ckpt_geo": ck_geo,
        "note": "Left: native Timer (default embed). Right: Geometric-HPE + v·(1+a·k_norm).",
        "ims_test_native_timer": {"mse": mse_t, "mae": mae_t},
        "ims_test_geometric_hpe": {"mse": mse_g, "mae": mae_g},
        "metrics_max_batches": int(args.metrics_max_batches),
        "k_per_patch": k_np.tolist(),
        "attn_shape": list(stack_a.shape),
        "seq_len": args.seq_len,
        "patch_len": args.patch_len,
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "subset_rand_ratio": float(args.subset_rand_ratio),
        "data": args.data,
        "data_path": args.data_path,
        "root_path": args.root_path,
        "num_patches": int(stack_a.shape[2]),
        "output_png": os.path.abspath(out_path),
    }
    with open(os.path.join(args.output_dir, args.output_name.replace(".png", "_meta.json")), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved: {os.path.abspath(out_path)}", flush=True)


if __name__ == "__main__":
    main()
