#!/usr/bin/env python3
"""
Timer 极端对比掩码重建实验 (Extreme Contrast Reconstruction).

实验设计：
  三组相同的输入时间序列，对比模型在"只看骨架"vs"只看噪声"时的重建质量。

  组别 A（随机掩码 Random）：   随机遮蔽 50% 的 Patch
  组别 B（掩蔽低 MI）：         专门遮蔽掉 50% 的低 MI Patch，保留高 MI 骨架
  组别 C（掩蔽高 MI）：         专门遮蔽掉 50% 的高 MI Patch，保留平稳背景

  关键指标：
    - 组别 B 靠零星几个高 MI 骨架点 → 完美还原走势
    - 组别 C 保留了大部分数据 → 失去关键拐点，重建偏离相位
    - 组别 A 随机对比基准

核心机制：
  在 Patch Embedding 空间直接置零指定 Patch 的嵌入向量，模拟"缺失"。
  之后模型对剩余 Patch 进行推理，并在投影回原始空间时，
  被掩蔽位置自然地被周围 Patch 的注意力信息填充。

Usage:
    python experiments/timer_mi_extreme_reconstruction.py \
        --mi_result_dir ./results/timer_mi_ksg_pca/Timer_MI_20260516_133108 \
        --root_path ./datasets/ --data_path ETTh1.csv \
        --seq_len 672 --pred_len 96 --patch_len 96 \
        --batch_size 64 --max_samples 500 \
        --mask_ratio 0.5 \
        --ckpt_path ./checkpoints/Timer_forecast_1.0.ckpt \
        --e_layers 8 --d_model 1024 --d_ff 2048 --n_heads 8 \
        --gpu 0 --out_dir ./results/timer_mi_extreme_reconstruction/ \
        --model_id etth1 \
        --n_visual 3
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark


# ============================================================================
# Nature Figure Style
# ============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "orange":         "#E07B39",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}


def add_panel_label(ax, label, x=-0.08, y=1.06, fontsize=10,
                    fontweight="bold", color="black"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha="left", va="bottom")


def finalize_figure(fig, out_path, dpi=300, pad=1.2):
    from pathlib import Path
    fig.tight_layout(pad=pad)
    base = Path(out_path)
    os.makedirs(base.parent, exist_ok=True)
    base = base.with_suffix("")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {base}.{{svg,pdf,png}}")


# ============================================================================
# Config & Model
# ============================================================================

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(
    ckpt_path: str,
    patch_len: int,
    d_model: int,
    d_ff: int,
    e_layers: int,
    n_heads: int,
    dropout: float,
    seq_len: int,
    pred_len: int,
) -> Model:
    cfg = Config(
        task_name='forecast',
        ckpt_path=ckpt_path,
        patch_len=patch_len,
        stride=patch_len,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        n_heads=n_heads,
        dropout=dropout,
        output_attention=False,
        distil=True,
        use_revin=False,
        seq_len=seq_len,
        pred_len=pred_len,
        d_layers=1,
        factor=1,
        enc_in=1,
        dec_in=1,
        c_out=1,
        activation='gelu',
        use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(),
        devices='0',
        num_workers=4,
        freq='h',
        data='custom',
        embed='timeF',
        target='OT',
        features='M',
        des='Exp',
        lradj='type1',
        use_amp=False,
        is_finetuning=0,
        label_len=pred_len,
        output_len=pred_len,
        batch_size=64,
        train_epochs=1,
        patience=3,
        learning_rate=3e-5,
        itr=1,
        use_ims=False,
        inverse=False,
        use_align_loss=False,
        align_loss_layers=list(range(e_layers)),
    )
    model = Model(cfg)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ============================================================================
# MI Weight Loading
# ============================================================================

def load_mi_weights(mi_result_dir: str) -> tuple[np.ndarray, int]:
    """
    从 MI 结果目录（或直接文件路径）加载各 Patch 的平均 MI 权重。

    支持多种路径格式：
      1. 直接 JSON 文件：  ./results/global_mi_peaks_etth1.json
      2. 带时间戳的目录：  ./results/timer_mi_ksg_pca/Timer_MI_20260516_133108/
      3. 带模型 ID 的目录：./results/mi_hsic_etth1_decoder/

    支持多种数据格式（自动探测）：
      a. layers[l]["hsic_curve"]: 多层嵌套（timer_mi_ksg_pca 格式）
      b. top-level "hsic_curve": 单层列表（changepoint 格式）
      c. top-level "mi": 直接 MI 数组
      d. layers[l]["mi_curve"]:  变体字段名

    Returns:
        mi_per_patch: [n_patches] 平均 I(H,Y) 分数，越高说明该 Patch 对预测越重要
        n_patches: Patch 总数
    """
    import glob

    result_file = None

    # 格式 1：直接传入了 JSON 文件
    if mi_result_dir.endswith(".json") and os.path.isfile(mi_result_dir):
        result_file = mi_result_dir
        print(f"  MI 结果（直接文件）: {result_file}")

    # 格式 2 / 3：在目录中搜索
    if result_file is None:
        candidates = (
            sorted(glob.glob(os.path.join(mi_result_dir, "**", "global_mi_peaks_*.json"), recursive=True))
            + sorted(glob.glob(os.path.join(mi_result_dir, "**", "mi_result*.json"), recursive=True))
            + sorted(glob.glob(os.path.join(mi_result_dir, "**", "result*.json"), recursive=True))
            + sorted(glob.glob(os.path.join(mi_result_dir, "**", "*.json"), recursive=True))
        )
        candidates = [c for c in candidates if os.path.isfile(c)]
        if candidates:
            result_file = candidates[0]
            print(f"  MI 结果（目录扫描）: {result_file}")
        else:
            raise FileNotFoundError(
                f"在 {mi_result_dir} 下找不到 MI 结果 JSON 文件\n"
                f"请确保已运行 timer_mi_ksg_pca.py 并传入正确的 --mi_result_dir\n"
                f"或者直接传入 JSON 文件路径，如：./results/global_mi_peaks_etth1.json"
            )
    print(f"  加载 MI 结果: {result_file}")

    with open(result_file, "r") as f:
        data = json.load(f)

    # ── 自动探测 MI 数据格式 ─────────────────────────────────────────────────
    num_layers = data.get("num_layers", 8)
    n_patches = data.get("N", 0)
    layers = data.get("layers", {})
    mi_curves = []  # 必须提前声明，否则后续 mi_curves.append 会报 UnboundLocalError

    # 遍历 layers 中所有值，找 MI/HSIC/CKA 曲线字段
    layer_values = []
    layer_debug = []
    MI_FIELD_NAMES = ("hsic_curve", "cka_curve", "mi_curve", "mi", "mi_hy", "mi_xh", "cca_curve")
    for k, v in layers.items():
        if isinstance(v, dict):
            found_field = next((f for f in MI_FIELD_NAMES if f in v), None)
            if found_field:
                raw_curve = v[found_field]
                if isinstance(raw_curve, list) and len(raw_curve) > 0:
                    try:
                        curve = [float(x) for x in raw_curve]
                        layer_values.append((str(k), curve))
                        layer_debug.append(f"  layers['{k}']: OK ({found_field}), {len(curve)} values")
                    except (ValueError, TypeError) as e:
                        layer_debug.append(f"  layers['{k}']: {found_field} CONVERT FAILED ({e})")
                else:
                    layer_debug.append(f"  layers['{k}']: {found_field} is not a non-empty list")
            else:
                layer_debug.append(f"  layers['{k}']: no MI field, keys={list(v.keys())[:5]}")
        else:
            layer_debug.append(f"  layers['{k}']: not a dict, type={type(v).__name__}")

    if layer_debug:
        print("  [MI 加载调试]")
        for line in layer_debug[:15]:
            print(line)
        if len(layer_debug) > 15:
            print(f"  ... 还有 {len(layer_debug)-15} 层")

    if layer_values:
        # 按 key 排序（确保层顺序一致）
        layer_values.sort(key=lambda x: x[0])
        for k, curve in layer_values:
            mi_curves.append(curve)

    # 格式 b: top-level "hsic_curve"
    if not mi_curves:
        for field in ("hsic_curve", "mi_curve", "mi", "mi_hy", "mi_hsic", "curve"):
            curve = data.get(field, [])
            if isinstance(curve, list) and len(curve) > 0 and isinstance(curve[0], (int, float)):
                mi_curves = [curve]
                n_patches = len(curve)
                break

    # 格式 c: top-level "mi" 是嵌套 list of lists
    if not mi_curves:
        for field in ("mi", "mi_matrix", "mi_curves"):
            val = data.get(field, [])
            if isinstance(val, list) and val:
                if isinstance(val[0], list):
                    mi_curves = val
                    break
                elif isinstance(val[0], (int, float)):
                    mi_curves = [val]
                    n_patches = len(val)
                    break

    if not mi_curves:
        raise ValueError(
            f"MI 结果中没有有效的 hsic_curve 数据。\n"
            f"文件 keys: {list(data.keys())}\n"
            f"layers 类型: {type(layers).__name__}, layers 内容示例: "
            f"{dict(list(layers.items())[:3])}\n"
            f"请检查 MI 结果文件格式，或运行 timer_mi_ksg_pca.py 重新生成"
        )

    if n_patches == 0:
        n_patches = len(mi_curves[0]) if mi_curves else 0

    mi_matrix = np.array(mi_curves)
    mi_per_patch = mi_matrix.mean(axis=0)
    print(f"  MI 格式: {len(mi_curves)} layers, n_patches={n_patches}")
    print(f"  MI 权重范围: [{mi_per_patch.min():.6f}, {mi_per_patch.max():.6f}]")
    print(f"  MI 权重: {mi_per_patch.round(6)}")

    return mi_per_patch, n_patches


# ============================================================================
# Core Masked Reconstruction
# ============================================================================

def masked_forward(
    model: Model,
    x: torch.Tensor,
    mask_indices: np.ndarray,
    n_vars: int,
    n_patches: int,
    patch_len: int,
) -> torch.Tensor:
    """
    在 Patch Embedding 空间置零指定 Patch，然后前向推理重建。

    Timer 的 enc_embedding 将 [B, T, M] 映射为 [B*M, N, D]，
    其中 M = n_vars, N = n_patches。
    mask_indices 形状为 [n_patches,]，对每个变量应用相同的 patch 掩码。

    Args:
        model: Timer 模型
        x: 原始输入 [B, T, M]  (已归一化)，M = n_vars
        mask_indices: 要掩蔽的 Patch 索引 [n_patches,]  bool 或 int 数组
        n_vars: 变量数 M
        n_patches: 每个变量的 Patch 数 N
        patch_len: Patch 长度

    Returns:
        recon: 重建序列 [B, T, M] (反归一化)
    """
    core = _unwrap(model)
    B, T, M = x.shape

    # enc_embedding: [B, M, T] -> [B*M, N, D]
    dec_in, _ = core.enc_embedding(x.permute(0, 2, 1).float())
    dec_in = dec_in.float()
    BM, N, D = dec_in.shape

    # dec_in 形状: [B*M, N, D]，M=n_vars, N=n_patches
    # 重整为 [B, M, N, D] 以便按变量应用掩码
    dec_in = dec_in.reshape(B, M, N, D)

    # mask_indices: [n_patches,] -> [1, 1, n_patches, 1]
    # 先创建全 False 的完整掩码，再填入 True
    full_mask = np.zeros(n_patches, dtype=bool)
    full_mask[mask_indices] = True
    mask_patch = torch.from_numpy(full_mask).float().to(dec_in.device)
    mask_expanded = mask_patch.view(1, 1, N, 1)

    # 所有变量应用相同 patch 掩码
    dec_in_masked = dec_in * (1.0 - mask_expanded)
    dec_in_masked = dec_in_masked.reshape(BM, N, D)

    decoder_out = core.decoder(dec_in_masked, has_prototype=False, output_hidden_states=False)

    if isinstance(decoder_out, tuple):
        dec_out = decoder_out[0]
    else:
        dec_out = decoder_out

    recon = core.proj(dec_out)
    recon = recon.reshape(B, M, -1).transpose(1, 2)

    return recon


def run_group(
    model: torch.nn.Module,
    data_loader,
    mask_ratio: float,
    mi_per_patch: np.ndarray,
    n_vars: int,
    patch_len: int,
    device: torch.device,
    max_samples: int,
    group_name: str,
    mask_type: str,
    rng: np.random.Generator,
    show_progress: bool = True,
) -> dict:
    """
    对一组样本执行掩码重建。

    Timer 对多变量时序建模时，输入 [B, T, M] 经过 PatchEmbedding 后
    变成 [B*M, N, D]，其中 M=n_vars, N=T/patch_len。
    MI 权重 mi_per_patch 形状为 [N,]，是单变量的 patch 重要性分数，
    对所有变量应用相同的 patch 掩码索引。

    Args:
        mask_type: "random" | "low_mi" | "high_mi"
            - "random":    随机掩蔽 50% Patch（每样本独立随机）
            - "low_mi":    掩蔽最低 MI 的 50% Patch（保留骨架）
            - "high_mi":   掩蔽最高 MI 的 50% Patch（保留噪声）

    Returns:
        results dict with mse/mae metrics and recon/orig lists
    """
    n_patches = mi_per_patch.shape[0]
    n_mask = int(n_patches * mask_ratio)

    if mask_type == "low_mi":
        sorted_idx = np.argsort(mi_per_patch)
        masked_patches = np.sort(sorted_idx[:n_mask])
    elif mask_type == "high_mi":
        sorted_idx = np.argsort(mi_per_patch)[::-1]
        masked_patches = np.sort(sorted_idx[:n_mask])
    else:
        masked_patches = None

    kept_patches = np.array(
        [p for p in range(n_patches) if p not in masked_patches]
    ) if masked_patches is not None else None

    print(f"\n  [{group_name}] mask_type={mask_type}, "
          f"n_mask={n_mask}/{n_patches} per var, "
          f"masked_patches={list(masked_patches) if masked_patches is not None else 'per-sample random'}")

    recon_list, orig_list = [], []
    mse_masked_list, mse_kept_list = [], []
    mse_full_list, mae_masked_list, mae_kept_list = [], [], []

    desc = f"[{group_name}] Reconstruction"
    iterator = tqdm(data_loader, desc=desc, disable=not show_progress)

    total_samples = 0

    for seq_x, seq_y, seq_x_mark, seq_y_mark in iterator:
        B = seq_x.shape[0]
        total_samples += B
        if max_samples > 0 and total_samples > max_samples:
            break

        seq_x = seq_x.float().to(device)
        B_, T_full, M_ = seq_x.shape

        # 动态推断每个变量的 patch 数 N = T / patch_len
        # 截断到完整的 patch 倍数
        n_patches_actual = T_full // patch_len
        actual_T = n_patches_actual * patch_len
        seq_x_trim = seq_x[:, :actual_T, :]

        means = seq_x_trim.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(seq_x_trim, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        stdev = torch.clamp_min(stdev, 1e-5)
        x_norm = (seq_x_trim - means) / stdev

        # 对每个样本单独处理（随机掩码时每样本掩码不同）
        for b in range(B_):
            x_single = x_norm[b:b+1]

            if mask_type == "random":
                k = int(n_patches_actual * mask_ratio)
                mp = np.sort(rng.choice(n_patches_actual, size=k, replace=False))
            else:
                mp = masked_patches

            kp = np.array([p for p in range(n_patches_actual) if p not in mp])

            with torch.no_grad():
                recon_norm = masked_forward(
                    model, x_single,
                    mask_indices=mp,
                    n_vars=M_,
                    n_patches=n_patches_actual,
                    patch_len=patch_len,
                )

            recon_denorm = recon_norm * stdev[b:b+1] + means[b:b+1]

            orig_np = seq_x_trim[b, :, 0].cpu().numpy()
            recon_np = recon_denorm[0, :, 0].cpu().numpy()

            # 计算 patch 级别掩码时间索引
            masked_time = np.zeros(actual_T, dtype=bool)
            kept_time = np.zeros(actual_T, dtype=bool)
            for p in mp:
                s, e = p * patch_len, (p + 1) * patch_len
                masked_time[s:e] = True
            for p in kp:
                s, e = p * patch_len, (p + 1) * patch_len
                kept_time[s:e] = True

            err_full = orig_np - recon_np
            err_masked = orig_np[masked_time] - recon_np[masked_time]
            err_kept = orig_np[kept_time] - recon_np[kept_time]

            mse_full_list.append(float(np.mean(err_full ** 2)))
            if err_masked.size > 0:
                mse_masked_list.append(float(np.mean(err_masked ** 2)))
                mae_masked_list.append(float(np.mean(np.abs(err_masked))))
            else:
                mse_masked_list.append(np.nan)
                mae_masked_list.append(np.nan)
            if err_kept.size > 0:
                mse_kept_list.append(float(np.mean(err_kept ** 2)))
                mae_kept_list.append(float(np.mean(np.abs(err_kept))))
            else:
                mse_kept_list.append(np.nan)
                mae_kept_list.append(np.nan)

            recon_list.append(recon_np)
            orig_list.append(orig_np)

    def nanmean(arr):
        a = np.array(arr, dtype=float)
        return float(np.nanmean(a)) if np.any(~np.isnan(a)) else np.nan

    results = {
        "group": group_name,
        "mask_type": mask_type,
        "mask_ratio": mask_ratio,
        "n_mask": n_mask,
        "n_patches": n_patches_actual,
        "n_vars": M_,
        "n_samples": len(recon_list),
        "masked_patches": list(masked_patches) if masked_patches is not None else None,
        "kept_patches": list(kept_patches) if kept_patches is not None else None,
        "mse_masked": nanmean(mse_masked_list),
        "mse_kept": nanmean(mse_kept_list),
        "mse_total": nanmean(mse_full_list),
        "mae_masked": nanmean(mae_masked_list),
        "mae_kept": nanmean(mae_kept_list),
        "recon_list": recon_list,
        "orig_list": orig_list,
    }
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_extreme_reconstruction_comparison(
    results_by_group: dict,
    mi_per_patch: np.ndarray,
    n_patches: int,
    patch_len: int,
    output_dir: str,
    n_visual: int = 3,
    model_id: str = "",
):
    """
    绘制三组重建对比图 + 统计柱状图。

    Layout:
      Fig1 (重建曲线对比):
        左上  组别 A 随机掩码重建
        右上  组别 B 低 MI 掩码重建 (仅保留骨架)
        右下  组别 C 高 MI 掩码重建 (仅保留噪声)
        右下  原始序列

      Fig2 (统计对比):
        柱状图对比三组的 MSE 指标
    """
    os.makedirs(output_dir, exist_ok=True)

    group_order = ["Random", "Low-MI Masked (Keep Skeleton)", "High-MI Masked (Keep Noise)"]
    group_keys = ["random", "low_mi", "high_mi"]

    fig1 = plt.figure(figsize=(14, 4 * n_visual))
    gs = gridspec.GridSpec(n_visual, 4, figure=fig1, hspace=0.45, wspace=0.35)

    colors = {
        "orig": PALETTE["neutral_dark"],
        "random": PALETTE["blue_main"],
        "low_mi": PALETTE["teal"],
        "high_mi": PALETTE["red_strong"],
    }

    for row in range(n_visual):
        total_available = results_by_group["random"]["n_samples"]
        if total_available == 0:
            continue
        sample_idx = row % total_available

        for col_idx, (gkey, glabel) in enumerate(zip(group_keys, group_order)):
            ax = fig1.add_subplot(gs[row, col_idx])
            res = results_by_group[gkey]

            if sample_idx >= len(res["recon_list"]) or sample_idx >= len(res["orig_list"]):
                continue

            orig = res["orig_list"][sample_idx]
            recon = res["recon_list"][sample_idx]
            time_axis = np.arange(len(orig))

            masked_p = res.get("masked_patches")
            if masked_p is not None:
                masked_time = np.zeros(len(orig), dtype=bool)
                for p in masked_p:
                    s, e = p * patch_len, (p + 1) * patch_len
                    masked_time[s:e] = True
            else:
                masked_time = np.zeros(len(orig), dtype=bool)

            ax.fill_between(
                time_axis,
                orig * 0.9,
                orig * 1.1,
                alpha=0.08,
                color=PALETTE["neutral_light"],
                label="_nolegend_",
            )
            ax.plot(time_axis, orig, color=colors["orig"], lw=1.2, alpha=0.7, label="Ground Truth")
            ax.plot(time_axis, recon, color=colors[gkey], lw=1.2, alpha=0.9, label="Reconstruction")

            for p in masked_p or []:
                s, e = p * patch_len, (p + 1) * patch_len
                ax.axvspan(s, e, alpha=0.18, color=PALETTE["orange"], zorder=0)

            mse_t = float(np.mean((orig - recon) ** 2))
            ax.set_title(
                f"{glabel}\nMSE={mse_t:.4f}",
                fontsize=7.5,
                color=colors.get(gkey, "black"),
            )
            ax.set_xlabel("Time", fontsize=6)
            if col_idx == 0:
                ax.set_ylabel("Value", fontsize=6)
            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.2, lw=0.5)

            if col_idx == 0 and row == 0:
                add_panel_label(ax, "a", y=1.12)

    fig1.suptitle(
        f"Extreme Contrast Reconstruction ({model_id}) — "
        f"Random vs Low-MI vs High-MI Masking",
        fontsize=9,
        fontweight="bold",
    )
    finalize_figure(fig1, os.path.join(output_dir, "fig1_extreme_reconstruction_curves"), dpi=300, pad=0.8)

    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ("mse_total", "Total MSE", True),
        ("mse_masked", "Masked Region MSE", True),
        ("mse_kept", "Retained Region MSE", True),
    ]

    for ax_idx, (metric_key, metric_label, higher_worse) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = []
        labels_short = ["Random", "Low-MI\n(Keep Skeleton)", "High-MI\n(Keep Noise)"]
        bar_colors = [colors["random"], colors["low_mi"], colors["high_mi"]]

        for gkey in group_keys:
            v = results_by_group[gkey].get(metric_key, np.nan)
            vals.append(v if not np.isnan(v) else 0.0)

        bars = ax.bar(range(3), vals, color=bar_colors, edgecolor="white", linewidth=0.5)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=bar_colors[i],
            )

        ax.set_xticks(range(3))
        ax.set_xticklabels(labels_short, fontsize=7)
        ax.set_ylabel(metric_label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, axis="y", lw=0.5)
        ax.set_title(f"{chr(ord('a')+ax_idx)} {metric_label}", fontsize=7.5, pad=3)
        add_panel_label(ax, chr(ord('a') + ax_idx), y=1.08)

    fig2.suptitle(
        f"Reconstruction Quality Metrics ({model_id})",
        fontsize=9,
        fontweight="bold",
    )
    finalize_figure(fig2, os.path.join(output_dir, "fig2_metrics_comparison"), dpi=300, pad=0.8)

    fig3 = plt.figure(figsize=(10, 4))

    ax_mi = fig3.add_subplot(1, 2, 1)
    ax_hist = fig3.add_subplot(1, 2, 2)

    x_pos = np.arange(n_patches)
    ax_mi.bar(x_pos, mi_per_patch, color=PALETTE["blue_main"], alpha=0.7, edgecolor="white", lw=0.3)
    threshold_low = np.percentile(mi_per_patch, 50)
    ax_mi.axhline(threshold_low, color=PALETTE["red_strong"], ls="--", lw=1.2, label=f"P50={threshold_low:.3f}")
    ax_mi.set_xlabel("Patch Index", fontsize=7)
    ax_mi.set_ylabel("Mean I(H,Y) (bits)", fontsize=7)
    ax_mi.set_title("a  MI Weights per Patch", fontsize=7.5)
    ax_mi.tick_params(labelsize=6)
    ax_mi.legend(fontsize=6)
    add_panel_label(ax_mi, "a", y=1.08)

    for gkey, color, label in [
        ("random", colors["random"], "Random"),
        ("low_mi", colors["low_mi"], "Low-MI Mask"),
        ("high_mi", colors["high_mi"], "High-MI Mask"),
    ]:
        mse_vals = []
        for i in range(n_patches):
            sample = results_by_group[gkey]["orig_list"]
            recon = results_by_group[gkey]["recon_list"]
            if i < len(sample):
                mse_vals.append(float(np.mean((sample[i] - recon[i]) ** 2)))
        if mse_vals:
            ax_hist.plot(
                range(n_patches),
                np.convolve(mse_vals, np.ones(3) / 3, mode="same"),
                color=color,
                lw=1.5,
                alpha=0.8,
                label=label,
            )

    ax_hist.set_xlabel("Patch Index", fontsize=7)
    ax_hist.set_ylabel("MSE (rolling mean)", fontsize=7)
    ax_hist.set_title("b  Per-Patch Reconstruction Error", fontsize=7.5)
    ax_hist.tick_params(labelsize=6)
    ax_hist.legend(fontsize=6)
    ax_hist.grid(True, alpha=0.2, lw=0.5)
    add_panel_label(ax_hist, "b", y=1.08)

    fig3.suptitle(f"MI Distribution & Patch Error ({model_id})", fontsize=9, fontweight="bold")
    finalize_figure(fig3, os.path.join(output_dir, "fig3_mi_distribution"), dpi=300, pad=0.8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Timer 极端对比掩码重建实验"
    )
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="最大测试样本数 (0=全部)")
    parser.add_argument("--n_visual", type=int, default=3,
                        help="可视化展示的样本数")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="掩蔽 Patch 的比例")
    parser.add_argument("--mi_result_dir", type=str,
                        default="./results/timer_mi_ksg_pca/Timer_MI_20260516_133108")
    parser.add_argument("--ckpt_path", type=str,
                        default="./checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str,
                        default="./results/timer_mi_extreme_reconstruction/")
    parser.add_argument("--model_id", type=str, default="etth1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_type", type=str, default="ETTh1",
                        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Timer Extreme Contrast Reconstruction Experiment")
    print(f"{'='*60}")
    print(f"  Device:    {device}")
    print(f"  Dataset:   {args.data_path}")
    print(f"  seq_len:   {args.seq_len}, pred_len: {args.pred_len}")
    print(f"  patch_len: {args.patch_len}, mask_ratio: {args.mask_ratio}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  out_dir:   {args.out_dir}")

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = build_timer(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    model = model.to(device)
    model.eval()
    print("  模型加载完成")

    mi_per_patch, n_patches = load_mi_weights(args.mi_result_dir)

    data_csv = os.path.join(args.root_path, args.data_path)
    dataset = CIDatasetBenchmark(
        root_path=data_csv,
        flag="test",
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = dataset.n_var
    # MI 的 n_patches 对应 seq_len/patch_len（单变量的 patch 数）
    # 这里 n_patches 就是 mi_per_patch 的长度，不需要额外验证
    print(f"  n_vars={n_vars}, mi_n_patches={n_patches} (seq_len/patch_len={args.seq_len // args.patch_len})")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    print(f"  数据加载器就绪: {len(dataset)} 样本, n_vars={n_vars}")

    print("\n" + "=" * 60)
    print("  运行三组掩码重建实验")
    print("=" * 60)

    results_random = run_group(
        model, data_loader,
        mask_ratio=args.mask_ratio,
        mi_per_patch=mi_per_patch,
        n_vars=n_vars,
        patch_len=args.patch_len,
        device=device,
        max_samples=args.max_samples,
        group_name="Random",
        mask_type="random",
        rng=rng,
    )

    results_low_mi = run_group(
        model, data_loader,
        mask_ratio=args.mask_ratio,
        mi_per_patch=mi_per_patch,
        n_vars=n_vars,
        patch_len=args.patch_len,
        device=device,
        max_samples=args.max_samples,
        group_name="Low-MI Masked (Keep Skeleton)",
        mask_type="low_mi",
        rng=rng,
    )

    results_high_mi = run_group(
        model, data_loader,
        mask_ratio=args.mask_ratio,
        mi_per_patch=mi_per_patch,
        n_vars=n_vars,
        patch_len=args.patch_len,
        device=device,
        max_samples=args.max_samples,
        group_name="High-MI Masked (Keep Noise)",
        mask_type="high_mi",
        rng=rng,
    )

    results_by_group = {
        "random": results_random,
        "low_mi": results_low_mi,
        "high_mi": results_high_mi,
    }

    print("\n" + "=" * 60)
    print("  实验结果汇总")
    print("=" * 60)
    print(f"\n  {'Group':<40} {'MSE_Total':>10} {'MSE_Masked':>10} {'MSE_Kept':>10}")
    print(f"  {'-'*72}")
    for gkey, gres in results_by_group.items():
        print(
            f"  {gres['group']:<40} "
            f"{gres['mse_total']:>10.4f} "
            f"{gres['mse_masked']:>10.4f} "
            f"{gres['mse_kept']:>10.4f}"
        )

    plot_extreme_reconstruction_comparison(
        results_by_group,
        mi_per_patch,
        n_patches,
        args.patch_len,
        args.out_dir,
        n_visual=args.n_visual,
        model_id=args.model_id,
    )

    summary = {
        "model_id": args.model_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "patch_len": args.patch_len,
            "mask_ratio": args.mask_ratio,
            "n_patches": int(n_patches),
            "n_vars": int(n_vars),
            "max_samples": args.max_samples,
        },
        "mi_weights": {
            "range": [float(mi_per_patch.min()), float(mi_per_patch.max())],
            "mean": float(mi_per_patch.mean()),
            "std": float(mi_per_patch.std()),
        },
        "results": {
            gkey: {
                "group": gres["group"],
                "mask_type": gres["mask_type"],
                "n_samples": int(gres["n_samples"]),
                "mse_total": gres["mse_total"],
                "mse_masked": gres["mse_masked"],
                "mse_kept": gres["mse_kept"],
                "mae_masked": gres["mae_masked"],
                "mae_kept": gres["mae_kept"],
                "masked_patches": [int(p) for p in gres["masked_patches"]] if gres["masked_patches"] is not None else None,
                "kept_patches": [int(p) for p in gres["kept_patches"]] if gres["kept_patches"] is not None else None,
            }
            for gkey, gres in results_by_group.items()
        },
    }

    summary_path = os.path.join(args.out_dir, "extreme_reconstruction_summary.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  汇总结果已保存: {summary_path}")
    print(f"\n{'='*60}")
    print(f"  实验完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
