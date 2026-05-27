#!/usr/bin/env python3
"""
ETTh1 + Timer: 线性探针（Linear Probe）实验

五阶段实验流程：
  Stage 1: 语义标签提取（STL 分解 / 时间编码 / 波动率 / 未来真值）
  Stage 2: 特征提取（冻结 Timer 前向传播，提取每层隐藏状态 + MI 分数）
  Stage 3: 全量 Ridge Regression 基准扫描（R² per layer × semantic）
  Stage 4: MI 分组对比实验（High-MI vs Low-MI probe R²）
  Stage 5: 可视化（热力图 / 折线图 / Token 案例分析）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.masking import TriangularCausalMask

MI_DECODER_LAYER_CAP = 8


# ═══════════════════════════════════════════════════════════════════════
# 工具函数（从 etth1_mi_hsic_peaks.py 复制）
# ═══════════════════════════════════════════════════════════════════════

def _unwrap_timer(model):
    return model.module if hasattr(model, "module") else model


def load_hsic_mi_curve(hsic_file: str, layer_idx: int | str = -1) -> tuple[np.ndarray, float, float, int]:
    """
    加载 etth1_mi_hsic_peaks.py 输出的全局 MI JSON 文件。
    支持 layer_idx 为整数（指定层）或 "all"（所有层平均）。

    返回: (mi_curve, q3, q1, N)
    """
    with open(hsic_file, "r") as f:
        data = json.load(f)

    # JSON 结构: {"layers": {"0": {...}, "1": {...}, ...}, ...}
    layers_data = data.get("layers", data)
    # 过滤出实际的层数据（键为数字字符串）
    layer_keys = sorted([k for k in layers_data.keys() if k.isdigit()], key=int)
    n_layers = len(layer_keys)

    if layer_idx == "all":
        # 所有层的 hsic_curve 取平均
        all_curves = []
        for k in layer_keys:
            curve = layers_data[k]["hsic_curve"]
            all_curves.append(curve)
        # 按元素平均
        mi_curve = np.mean(all_curves, axis=0).astype(np.float64)
    else:
        if isinstance(layer_idx, int) and layer_idx < 0:
            layer_idx = n_layers - 1

        layer_key = str(layer_idx) if layer_idx < n_layers else str(n_layers - 1)
        layer_data = layers_data.get(layer_key, layers_data[layer_keys[-1]])

        mi_curve = np.array(layer_data["hsic_curve"], dtype=np.float64)

    sorted_mi = np.sort(mi_curve)
    n = len(sorted_mi)
    q3 = float(sorted_mi[int(np.ceil(0.75 * n)) - 1])
    q1 = float(sorted_mi[int(np.ceil(0.25 * n)) - 1])
    return mi_curve, q3, q1, n


def forward_collect_layers(model, x_enc):
    """
    Returns up to MI_DECODER_LAYER_CAP tensors [B, N, D] after each decoder block.
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
    assert BM == B * n_vars

    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers = []
    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _ = o3(h, attn_mask=mask)
        layers.append(pool(h.detach()))
    return layers, int(n_vars), int(N)


def hsic_score(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Unbiased HSIC (Gaussian kernels) as an MI-surrogate score."""
    n = X.shape[0]
    if n < 4:
        return float("nan")
    dX = torch.cdist(X, X, p=2.0).cpu()
    triu = torch.triu_indices(n, n, offset=1, device="cpu")
    medX = torch.median(dX[triu[0], triu[1]] ** 2).clamp(min=1e-12)

    dY = torch.cdist(Y, Y, p=2.0).cpu()
    medY = torch.median(dY[triu[0], triu[1]] ** 2).clamp(min=1e-12)

    K = torch.exp(-dX / (2 * medX))
    L = torch.exp(-dY / (2 * medY))
    H = torch.eye(n, device=X.device, dtype=X.dtype) - (1.0 / n)
    return float((H @ K @ H * (H @ L @ H)).trace().cpu().item())


def mi_sequence_hsic(h_x: torch.Tensor, h_y: torch.Tensor) -> np.ndarray:
    """Compute HSIC per patch position between h_x and h_y."""
    B, N, D = h_x.shape
    scores = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        if torch.isfinite(h_x[:, i, :]).all() and torch.isfinite(h_y[:, i, :]).all():
            scores[i] = hsic_score(h_x[:, i, :], h_y[:, i, :])
    return scores


# ═══════════════════════════════════════════════════════════════════════
# 阶段 1: 语义标签提取（从原始 CSV，未归一化）
# ═══════════════════════════════════════════════════════════════════════

def stl_decompose(series: np.ndarray, period: int = 24):
    """
    移动平均近似 STL，返回 (trend, seasonal, residual)。
    period: ETTh1 建议 24（小时粒度周期）。
    """
    n = len(series)
    trend = np.zeros(n)
    half = period // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        trend[i] = np.mean(series[lo:hi])

    seasonal = np.zeros(n)
    for k in range(period):
        season_vals = [series[j] - trend[j] for j in range(k, n, period)]
        if season_vals:
            seasonal[k::period] = np.mean(season_vals)

    residual = series - trend - seasonal
    return trend, seasonal, residual


def rolling_volatility(series: np.ndarray, window: int = 24):
    out = np.full(len(series), np.nan, dtype=np.float64)
    for i in range(window - 1, len(series)):
        out[i] = np.std(series[i - window + 1:i + 1])
    out[:window - 1] = out[window - 1]
    return out


def extract_raw_semantic_labels(
    raw_df: pd.DataFrame,
    seq_len: int,
    patch_len: int,
    stride: int,
    period: int = 24,
    pred_len: int = 96,
    data_type: str = "ETTh1",
):
    """
    从原始 DataFrame（未归一化）提取语义标签。
    raw_df: 原始 CSV DataFrame
    seq_len: 输入窗口长度（如 672）
    patch_len: patch 长度（如 96）
    stride: 滑动步长（用于 patch 划分，标签提取用密集窗口 stride=1）
    period: 周期（24 小时）
    pred_len: 预测步长

    返回:
      semantics: dict[str, np.ndarray]，每个 key 是 [n_vars * n_windows,] 扩展后的长向量
      N_patches: int（每个样本的 patch 数，基于 patch_len 的非重叠划分）
      border1_test: int（测试集起始索引）
      n_test_windows: int（测试集窗口总数，基于密集滑动）
    """
    n_vars = raw_df.shape[1]

    # N_patches: 每个窗口的 patch 数（模型内部用 patch_len 划分，非重叠）
    N_patches = (seq_len - patch_len) // patch_len + 1

    # 边界：与 CIDatasetBenchmark 完全一致
    total_len = len(raw_df)
    if data_type in ["ETTh1", "ETTh2", "ETTh"]:
        # ETTh 固定划分（12 月训练 + 4 月验证 + 4 月测试）
        num_train = 12 * 30 * 24
        num_val = 4 * 30 * 24
        num_test = 4 * 30 * 24
        border1_test = num_train + num_val - seq_len  # border1s[2]
        border2_test = num_train + num_val + num_test  # border2s[2]
    elif data_type in ["ETTm1", "ETTm2", "ETTm"]:
        num_train = 12 * 30 * 24 * 4
        num_val = 4 * 30 * 24 * 4
        num_test = 4 * 30 * 24 * 4
        border1_test = num_train + num_val - seq_len
        border2_test = num_train + num_val + num_test
    else:
        # 通用 7:3 划分（无验证集）
        num_train = int(total_len * 0.7)
        num_test = total_len - num_train
        border1_test = num_train - seq_len
        border2_test = total_len

    # 测试集窗口数（n_timepoint）与 CIDatasetBenchmark 完全一致
    n_test_windows = border2_test - border1_test

    # 截取测试集时间序列（长度 = border2_test - border1_test）
    numeric_df = raw_df.select_dtypes(include=[np.number])
    seq_test = numeric_df.mean(axis=1).values.astype(np.float64)[border1_test:border2_test]
    n_test_total = len(seq_test)

    semantics = {
        "trend": None, "seasonal": None, "residual": None,
        "volatility": None, "y_next1": None, "y_next24": None,
        "hour_sin": None, "hour_cos": None,
    }

    trend, seasonal, residual = stl_decompose(seq_test, period=period)
    vol = rolling_volatility(seq_test, window=period)

    t = np.arange(len(seq_test))
    hour_sin = np.sin(2 * np.pi * (t % period) / period)
    hour_cos = np.cos(2 * np.pi * (t % period) / period)

    y_next1 = np.roll(seq_test, -1)
    y_next24 = np.roll(seq_test, -24)
    y_next1[-24:] = seq_test[-24]
    y_next24[-24:] = seq_test[-24]

    full_curves = {
        "trend": trend, "seasonal": seasonal, "residual": residual,
        "volatility": vol, "y_next1": y_next1, "y_next24": y_next24,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
    }

    labels_per_patch: dict[str, list[np.ndarray]] = {k: [] for k in semantics}

    # 按滑动窗口切分（密集窗口，stride=1，匹配 CIDatasetBenchmark）
    # 注意：full_curves 基于 seq_test（border1_test 之后），索引从 0 开始
    for i in range(n_test_windows):
        center_local = i + seq_len // 2  # 在 seq_test 中的局部索引
        for key, curve in full_curves.items():
            if center_local < len(curve):
                labels_per_patch[key].append(curve[center_local])
            else:
                labels_per_patch[key].append(curve[-1])

    # 扩展标签：每个窗口有 n_vars 个变量，每个变量对应一个样本
    semantics_dict = {}
    for key in semantics:
        arr = np.array(labels_per_patch[key], dtype=np.float64)  # [n_windows]
        # 重复 n_vars 次：[n_windows] → [n_windows * n_vars]
        semantics_dict[key] = np.repeat(arr, n_vars)

    return semantics_dict, N_patches, border1_test, n_test_windows


# ═══════════════════════════════════════════════════════════════════════
# 阶段 3 & 4: Ridge Regression 探针
# ═══════════════════════════════════════════════════════════════════════

def ridge_probe(
    X_train, y_train, X_test, y_test, alpha=1.0
):
    """训练 Ridge 探针，返回 (r2_train, r2_test)。"""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_train)
    r2_tr = r2_score(y_train, model.predict(X_tr))
    r2_te = r2_score(y_test, model.predict(X_te))
    return r2_tr, r2_te


# ═══════════════════════════════════════════════════════════════════════
# 阶段 5: 可视化
# ═══════════════════════════════════════════════════════════════════════

def plot_heatmap_r2(results, out_dir, key="r2_full",
                    title="Layer × Semantic R²", cmap="RdYlGn"):
    semantics = results["semantics"]
    n_layers = results["n_layers"]
    mat = np.array([
        [results[key].get(str(li), {}).get(s, np.nan) for s in semantics]
        for li in range(n_layers)
    ], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(8, len(semantics) * 1.3), max(4, n_layers * 0.8)))
    vmax = 1.0
    vmin = max(-1.0, np.nanmin(mat))  # 允许负值，但不低于 -1
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(semantics)))
    ax.set_xticklabels(semantics, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)
    ax.set_title(title, fontsize=11)

    for li in range(n_layers):
        for si in range(len(semantics)):
            v = mat[li, si]
            if not np.isnan(v):
                color = "white" if v < 0.25 or v > 0.7 else "black"
                ax.text(si, li, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    path = os.path.join(out_dir, f"heatmap_{key}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] {path}")


def plot_high_low_line(results, out_dir, semantics_to_plot=None):
    if semantics_to_plot is None:
        semantics_to_plot = results["semantics"]
    n_layers = results["n_layers"]
    cols = min(2, len(semantics_to_plot))
    rows = (len(semantics_to_plot) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.2), squeeze=False)
    colors = plt.cm.tab10.colors

    for idx, sem in enumerate(semantics_to_plot):
        ax = axes[idx // cols][idx % cols]
        x = list(range(n_layers))
        r2_h = [results["r2_high"].get(str(i), {}).get(sem) for i in range(n_layers)]
        r2_l = [results["r2_low"].get(str(i), {}).get(sem) for i in range(n_layers)]
        r2_f = [results["r2_full"].get(str(i), {}).get(sem) for i in range(n_layers)]

        def to_arr(v):
            return np.array([np.nan if x is None else x for x in v])

        ax.plot(x, to_arr(r2_h), "o-", color=colors[idx % 10],
                label="High-MI", linewidth=2.2, markersize=6)
        ax.plot(x, to_arr(r2_l), "s--", color=colors[idx % 10],
                label="Low-MI", linewidth=2.2, markersize=6, alpha=0.75)
        ax.plot(x, to_arr(r2_f), "D-.", color="gray",
                label="Full", linewidth=1.5, markersize=5, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in x])
        ax.set_ylabel("R²")
        ax.set_ylim(-0.12, 1.05)
        ax.set_title(f"Semantic: {sem}", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for idx in range(len(semantics_to_plot), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.suptitle("High-MI vs Low-MI Ridge Probe R²", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "high_low_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {path}")


def plot_layer_bar(results, out_dir, semantic="trend"):
    n_layers = results["n_layers"]
    x = np.arange(n_layers)
    w = 0.22

    def v(semantic, key):
        arr = []
        for li in range(n_layers):
            arr.append(results[key].get(str(li), {}).get(semantic))
        return np.array([np.nan if a is None else a for a in arr])

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 1.3), 5))
    ax.bar(x - w,   v(semantic, "r2_high"), w, label="High-MI", color="#e74c3c", alpha=0.88)
    ax.bar(x,       v(semantic, "r2_full"), w, label="Full",    color="#7f8c8d", alpha=0.88)
    ax.bar(x + w,   v(semantic, "r2_low"),  w, label="Low-MI",  color="#3498db", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x])
    ax.set_ylabel("R²")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(f"Layer Semantic Scan — '{semantic}'", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"layer_bar_{semantic}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] {path}")


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def main():
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
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--use_ims", action="store_true")
    p.add_argument("--use_multi_gpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--out_dir", type=str, default="./results/linear_probe_etth1")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--val_ratio", type=float, default=0.2,
                   help="Validation set ratio (for 6:2:2 split)")
    p.add_argument("--max_batches", type=int, default=0, help="0=全部样本")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--period", type=int, default=24)
    p.add_argument("--semantics", type=str,
                   default="trend,seasonal,residual,volatility,y_next1,y_next24,hour_sin,hour_cos")
    p.add_argument("--stride", type=int, default=0, help="0=自动设为 patch_len")
    p.add_argument("--hsic_file", type=str, default=None,
                   help="etth1_mi_hsic_peaks 输出的 JSON 文件路径。若指定则直接加载全局 MI 曲线进行高/低分组。")
    p.add_argument("--hsic_layer", type=str, default="-1",
                   help="使用哪一层的 HSIC MI 曲线（-1=最后一层，'all'=所有层平均）")
    args = p.parse_args()

    if args.stride <= 0:
        args.stride = args.patch_len

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    semantics = [s.strip() for s in args.semantics.split(",") if s.strip()]

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Linear Probe — ETTh1")
        print(f"  Semantics: {semantics}")
        print(f"  Train/Val/Test ratio: {1-args.val_ratio-args.test_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%}")
        print(f"  Max batches: {args.max_batches} (0=全部)")
        print(f"  Ridge alpha: {args.alpha}")
        print(f"{'='*60}\n")

    # ── 阶段 1: 从原始 CSV 提取语义标签 ───────────────────────────────
    if rank == 0:
        print("[Stage 1] 从原始 CSV 提取语义标签...")
        raw_csv = os.path.join(args.root_path, args.data_path)
        raw_df = pd.read_csv(raw_csv)
        labels_dict, N_patches, border1_test, n_test_windows = extract_raw_semantic_labels(
            raw_df=raw_df,
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            stride=args.stride,
            period=args.period,
            pred_len=args.pred_len,
            data_type=args.data,
        )
        total_patch_labels = len(next(iter(labels_dict.values())))
        print(f"[Stage 1] N_patches={N_patches}, total_patch_labels={total_patch_labels}, "
              f"border1_test={border1_test}, n_test_windows={n_test_windows}")
    else:
        labels_dict = {}
        total_patch_labels = 0
        N_patches = (args.seq_len - args.patch_len) // args.stride + 1
        border1_test = 0
        n_test_windows = 0

    # ── 模型加载 ─────────────────────────────────────────────────────
    if rank == 0:
        print("[Init] 加载模型...")

    class C:
        pass
    ns = C()
    for k, v in vars(args).items():
        setattr(ns, k, v)
    # 单卡模式下强制禁用多卡
    ns.use_multi_gpu = False
    ns.task_name = "forecast"
    ns.freq = "h"
    ns.target = "OT"
    ns.embed = "timeF"
    ns.stride = args.stride if args.stride > 0 else args.patch_len
    ns.dropout = 0.1
    ns.output_attention = False
    ns.activation = "gelu"
    ns.factor = 3

    model = Model(ns).float()
    state = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)

    if args.use_multi_gpu and world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        model = DDP(model, device_ids=[local_rank])
        core = _unwrap_timer(model)
    else:
        core = _unwrap_timer(model)
    stride_model = int(core.backbone.patch_embedding.stride)

    # ── 数据加载 ─────────────────────────────────────────────────────
    if rank == 0:
        print("[Data] 加载测试集...")

    # 在 DDP 环境下减少 num_workers 避免进程竞争卡住
    effective_num_workers = 0 if args.use_multi_gpu and world_size > 1 else args.num_workers

    _, test_loader = data_provider(ns, "test", num_workers=effective_num_workers)

    n_test = len(test_loader.dataset)
    if rank == 0:
        print(f"[Data] Test set: {n_test} samples, N={N_patches}, num_workers={effective_num_workers}")

    # ── 阶段 2: 前向传播提取隐藏状态 ─────────────────────────────────
    if rank == 0:
        print("[Stage 2] 前向传播提取隐藏状态...")

    # 用 hook 捕获 decoder 各层输出
    hidden_states_per_layer: dict[int, list[torch.Tensor]] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output: tuple (hidden_state, attention) or just hidden_state
            if isinstance(output, tuple):
                hidden_states_per_layer[layer_idx] = output[0].detach()
            else:
                hidden_states_per_layer[layer_idx] = output.detach()
        return hook_fn

    # 注册 hook 到 decoder 的每一层
    decoder_layers = model.backbone.decoder.attn_layers
    handles = []
    for li, layer in enumerate(decoder_layers):
        h = layer.register_forward_hook(make_hook(li))
        handles.append(h)

    all_layer_states: list[list[np.ndarray]] = [[] for _ in range(len(decoder_layers))]  # [layer_idx][sample_idx] = [N, D]
    n_samples_total = 0

    # 遍历所有 batch 的所有样本
    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        B = batch_x.shape[0]

        # 处理 batch 内所有样本 - 确保数据类型为 float32
        for i in range(B):
            x_enc = batch_x[i].unsqueeze(0).to(device).float()       # [1, L, C]
            x_mark_enc = batch_x_mark[i].unsqueeze(0).to(device).float()
            x_dec = batch_y[i, :batch_y.shape[1], :].unsqueeze(0).to(device).float()
            x_mark_dec = batch_y_mark[i].unsqueeze(0).to(device).float()

            with torch.no_grad():
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # 从 hook 收集各层隐藏状态
            for li in range(len(decoder_layers)):
                h = hidden_states_per_layer[li]  # [B*M, N(+1), D]
                # 取该样本对应的 batch 维度（B*M=1 时）
                M = x_enc.shape[2]  # 变量数
                # h shape: [B*M, N(+1), D] → 取 [0:1, ...] 即第一个样本的所有变量
                # 然后 reshape 回 [M, N(+1), D] 再平均或取首个
                if h.shape[0] > M:
                    # 多个变量拼成 B*M > 1
                    h_sample = h[:M]  # [M, N(+1), D]
                else:
                    h_sample = h  # [1, N(+1), D]

                # 如果含 prototype，去掉第 0 个 token
                if h_sample.shape[1] == model.patch_len + 1:
                    h_sample = h_sample[:, 1:, :]  # [M, N, D]

                # 对变量维度取平均 → [N, D]
                h_avg = h_sample.float().mean(dim=0).cpu().numpy()
                all_layer_states[li].append(h_avg)

            n_samples_total += 1

        # 清理 hook 缓存
        hidden_states_per_layer.clear()

        if (batch_idx + 1) % 100 == 0 and rank == 0:
            print(f"  batch {batch_idx+1}/{len(test_loader)}, collected {n_samples_total} samples")

    # 移除 hook
    for handle in handles:
        handle.remove()

    if rank == 0:
        print(f"  total collected: {n_samples_total}")

    # ── 多卡汇总 ─────────────────────────────────────────────────────
    if args.use_multi_gpu and world_size > 1:
        dist.barrier()

        # 广播总数
        total_tensor = torch.tensor([n_samples_total], dtype=torch.long, device=device)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX)
        max_total = total_tensor.item()

        gathered: list[list[np.ndarray]] = [ [] for _ in range(len(all_layer_states)) ]
        gathered_n = 0

        if rank == 0:
            gathered = [[] for _ in range(len(all_layer_states))]

            local_flat = [
                np.concatenate(all_layer_states[li], axis=0)
                if all_layer_states[li] else np.zeros((0, N_patches, 1024), dtype=np.float64)
                for li in range(len(all_layer_states))
            ]
            for li in range(len(all_layer_states)):
                gathered[li] = [None] * world_size

            # all_gather 收集各 rank 数量
            gathered_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            local_n_tensor = torch.tensor([n_samples_total], dtype=torch.long, device=device)
            dist.all_gather(gathered_counts, local_n_tensor)
            counts = [t.item() for t in gathered_counts]

            # 收集 hidden states
            for li in range(len(all_layer_states)):
                gathered_li: list[torch.Tensor] = [torch.zeros((counts[r], N_patches, 1024),
                                                                 dtype=torch.float32, device=device)
                                                   for r in range(world_size)]
                h_tensor = torch.from_numpy(local_flat[li]).float().to(device)
                dist.all_gather(gathered_li, h_tensor)
                gathered[li] = [g.cpu().numpy() for g in gathered_li]

            # 拼接
            final_states: list[np.ndarray] = []
            for li in range(len(all_layer_states)):
                cat = np.concatenate(gathered[li], axis=0)  # [total_n, N, D]
                final_states.append(cat)
            gathered_n = sum(counts)

        else:
            # 其他 rank：发送本地数据
            local_flat = [
                np.concatenate(all_layer_states[li], axis=0)
                if all_layer_states[li] else np.zeros((0, N_patches, 1024), dtype=np.float64)
                for li in range(len(all_layer_states))
            ]
            for li in range(len(all_layer_states)):
                h_tensor = torch.from_numpy(local_flat[li]).float().to(device)
                zeros_list = [torch.zeros_like(h_tensor) for _ in range(world_size)]
                dist.all_gather(zeros_list, h_tensor)

            final_states = [np.zeros((0, N_patches, 1024), dtype=np.float64)
                           for _ in range(len(all_layer_states))]
            gathered_n = 0

        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return

        n_total = gathered_n
        all_layer_hidden = final_states
        all_layer_hidden_raw = final_states  # 保留未平均版本
    else:
        n_total = n_samples_total
        # all_layer_states 结构: [layer_idx][sample_idx] = [N, D]
        n_layers = len(all_layer_states)

        # 1) 构建未平均的 hidden states: [n_layers, n_total, N, D]
        all_layer_hidden_raw = []
        for li in range(n_layers):
            layer_states = all_layer_states[li]  # list of [N, D]
            all_layer_hidden_raw.append(np.stack(layer_states, axis=0))  # [n_total, N, D]

        # 2) 构建平均后的 hidden states: [n_layers, n_total, D]
        all_layer_hidden = []
        for li in range(n_layers):
            layer_states = all_layer_states[li]  # list of [N, D]
            layer_states_avg = [s.mean(axis=0) for s in layer_states]
            all_layer_hidden.append(np.stack(layer_states_avg, axis=0))  # [n_total, D]

    if rank == 0:
        print(f"[Stage 2] Hidden states: {[a.shape for a in all_layer_hidden]}")

    # ── 阶段 3 & 4: Ridge Regression 探针 ──────────────────────────────
    if rank == 0 and n_total > 0:
        print(f"[Stage 3+4] Ridge Regression 扫描（使用所有 {n_total} 个样本）...")

        n_layers = len(all_layer_hidden)
        D_dim = all_layer_hidden[0].shape[-1]

        # 对齐标签数量与 hidden states 数量
        n_labels = len(next(iter(labels_dict.values())))
        min_n = min(n_total, n_labels)

        # 应用 max_batches 限制（每 batch 的样本数）
        if args.max_batches > 0:
            max_samples = args.max_batches * args.batch_size
            min_n = min(min_n, max_samples)
            print(f"[Probe] max_batches={args.max_batches} → limiting to {max_samples} samples")

        probe_n = min_n
        probe_idx = np.arange(probe_n)

        for li in range(n_layers):
            all_layer_hidden[li] = all_layer_hidden[li][probe_idx]
            all_layer_hidden_raw[li] = all_layer_hidden_raw[li][probe_idx]
        for k in labels_dict:
            labels_dict[k] = labels_dict[k][probe_idx]

        print(f"[Probe] Using {probe_n} samples (n_total={n_total}, n_labels={n_labels})")

        # ── 加载全局 HSIC MI 曲线（从 JSON 文件）────────────────────────
        # 每层用自己的 HSIC 曲线
        hsic_layer_curves = [None] * n_layers

        if args.hsic_file and os.path.exists(args.hsic_file):
            if args.hsic_layer == "all":
                # 加载所有层的曲线
                for li in range(n_layers):
                    hsic_layer_curves[li], _, _, _ = load_hsic_mi_curve(args.hsic_file, li)
            else:
                li = int(args.hsic_layer)
                hsic_layer_curves[li], _, _, _ = load_hsic_mi_curve(args.hsic_file, li)
            print(f"[Probe] Loaded HSIC curves: layer-specific (all={args.hsic_layer})")
        else:
            if args.hsic_file:
                print(f"[Probe] Warning: HSIC file not found at {args.hsic_file}")
            print("[Probe] Falling back to runtime L2-norm MI proxy")
            for li in range(n_layers):
                # all_layer_hidden_raw[li]: [n_total, N, D]
                # L2 norm across D → [n_total, N]
                # 平均 across samples (axis=0) → [N]（每个 patch 位置）
                mi_li = np.linalg.norm(all_layer_hidden_raw[li], axis=-1).mean(axis=0)
                hsic_layer_curves[li] = mi_li
            print(f"[Probe] Using per-layer L2-norm MI proxy")

        # =========================================================================
        # 严格按时间顺序划分：训练集 / 验证集 / 测试集 = 6:2:2
        # =========================================================================
        idx = np.arange(probe_n)
        train_end = int(probe_n * (1 - args.val_ratio - args.test_ratio))
        val_end   = int(probe_n * (1 - args.test_ratio))
        train_idx = idx[:train_end]
        val_idx   = idx[train_end:val_end]   # 暂存，当前未使用
        test_idx  = idx[val_end:]

        print(f"[Probe] Time-order split: train={len(train_idx)} ({100*train_end/probe_n:.0f}%), "
              f"val={len(val_idx)} ({100*args.val_ratio:.0f}%), test={len(test_idx)} ({100*args.test_ratio:.0f}%)")

        results = {
            "n_layers": n_layers,
            "n_patches_per_sample": N_patches,
            "n_samples": probe_n,
            "probe_sample_indices": probe_idx.tolist(),
            "semantics": semantics,
            "r2_full": {str(i): {} for i in range(n_layers)},
            "r2_high": {str(i): {} for i in range(n_layers)},
            "r2_low":  {str(i): {} for i in range(n_layers)},
            "high_ratio": {str(i): None for i in range(n_layers)},
            "low_ratio":  {str(i): None for i in range(n_layers)},
        }

        for li in range(n_layers):
            lk = str(li)
            h = all_layer_hidden_raw[li]           # [n_total, N, D]（未平均）
            h_avg = all_layer_hidden[li]           # [n_total, D]（patch 平均）
            h_flat = h.reshape(h.shape[0], -1)     # [n_total, N*D]

            # 获取当前层的 MI 曲线（每 patch 的 MI 值）
            mi_curve_li = hsic_layer_curves[li]     # [N_patches]
            if mi_curve_li is not None:
                # 按 MI 值排序，取前 25% 作为 High-MI，后 25% 作为 Low-MI
                sorted_idx = np.argsort(mi_curve_li)
                n_patches = len(sorted_idx)
                n_quarter = max(1, n_patches // 4)

                high_patch_idx = sorted_idx[-n_quarter:]  # MI 最高的 25%
                low_patch_idx  = sorted_idx[:n_quarter]   # MI 最低的 25%
                print(f"[Probe] Layer {li}: High-MI patches: {len(high_patch_idx)}, Low-MI patches: {len(low_patch_idx)}")
            else:
                high_patch_idx = None
                low_patch_idx = None

            for sem in semantics:
                y_all = labels_dict.get(sem)
                if y_all is None or len(y_all) != min_n:
                    continue
                y_tr, y_te = y_all[train_idx], y_all[test_idx]

                # 全量
                r2_tr, r2_te = ridge_probe(h_flat[train_idx], y_tr,
                                            h_flat[test_idx], y_te, alpha=args.alpha)
                results["r2_full"][lk][sem] = round(r2_te, 4)

                # High MI 组（按 patch 位置筛选）
                if high_patch_idx is not None and len(high_patch_idx) > 0:
                    h_high = h[:, high_patch_idx, :].reshape(h.shape[0], -1)
                    r2_tr_h, r2_te_h = ridge_probe(
                        h_high[train_idx], y_tr,
                        h_high[test_idx], y_te, alpha=args.alpha)
                    results["r2_high"][lk][sem] = round(r2_te_h, 4)

                # Low MI 组（按 patch 位置筛选）
                if low_patch_idx is not None and len(low_patch_idx) > 0:
                    h_low = h[:, low_patch_idx, :].reshape(h.shape[0], -1)
                    r2_tr_l, r2_te_l = ridge_probe(
                        h_low[train_idx], y_tr,
                        h_low[test_idx], y_te, alpha=args.alpha)
                    results["r2_low"][lk][sem] = round(r2_te_l, 4)

        # ── 阶段 5: 可视化 ───────────────────────────────────────
        print("[Stage 5] 生成可视化...")

        plot_heatmap_r2(results, args.out_dir, "r2_full",
                        "Layer × Semantic R² (Full Set)")
        plot_heatmap_r2(results, args.out_dir, "r2_high",
                        "Layer × Semantic R² (High-MI Set)")
        plot_heatmap_r2(results, args.out_dir, "r2_low",
                        "Layer × Semantic R² (Low-MI Set)")

        plot_high_low_line(results, args.out_dir, semantics_to_plot=semantics)

        for sem_key in ["trend", "seasonal", "y_next1"]:
            if sem_key in semantics:
                plot_layer_bar(results, args.out_dir, semantic=sem_key)

        # 保存 JSON
        result_path = os.path.join(args.out_dir, "probe_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Result] {result_path}")

        # ── 打印摘要 ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  Ridge Probe Summary  (R² on test set)")
        print(f"{'='*60}")
        header = f"  {'Layer':<8}" + "".join(f"  {s:>12}" for s in semantics)
        print(header)
        for li in range(n_layers):
            lk = str(li)
            row = f"  L{li:<7}"
            for sem in semantics:
                rf = results["r2_full"].get(lk, {}).get(sem, "  -")
                rh = results["r2_high"].get(lk, {}).get(sem, "  -")
                rl = results["r2_low"].get(lk, {}).get(sem, "  -")
                row += f"  {str(rh):>4}h/{str(rf):>5}f/{str(rl):>5}l"
            print(row)
        print(f"{'='*60}\n")

    elif rank == 0:
        print("[Rank 0] 无数据，跳过计算。")


if __name__ == "__main__":
    main()