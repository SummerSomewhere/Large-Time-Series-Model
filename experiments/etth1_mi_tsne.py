#!/usr/bin/env python3
"""
ETTh1 + Timer: t-SNE 可视化 — 高/低 MI Patch 的隐藏状态分布分析

实验目标：
    使用 t-SNE 将高维隐藏状态向量降维到 2D 空间，
    分析 High-MI 和 Low-MI Patch 在特征空间中的分布差异。

操作步骤：
    1. 提取所有 patch 的最后一层隐藏状态向量
    2. 计算每个 patch 与未来真值的 HSIC MI 分数
    3. 将 patch 分为高 MI 组（>Q3）和低 MI 组（≤Q3）
    4. 使用 t-SNE 对所有隐藏状态进行降维
    5. 绘制 2D 散点图，按 MI 分组着色

用法（单卡）：
    python experiments/etth1_mi_tsne.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1

用法（多卡，必须用 torchrun）：
    torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_tsne.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 \
      --use_multi_gpu
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 尝试导入 MulticoreTSNE，如果不可用则使用标准 TSNE
try:
    from sklearn.manifold import TSNE
    USE_MULTICORE_TSNE = False
except ImportError:
    USE_MULTICORE_TSNE = False

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.masking import TriangularCausalMask

# 只使用最后一层（layer cap 之外的最后一层，即倒数第8层后的第一层）
MI_DECODER_LAYER_CAP = 8


# ─── HSIC / MI 相关函数 ────────────────────────────────────────────────────────

def _median_sq_bandwidth(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=X.device, dtype=torch.dtype)
    d = torch.cdist(X, X, p=2.0)
    triu = torch.triu_indices(n, n, offset=1, device=X.device)
    sq = d[triu[0], triu[1]] ** 2
    med = torch.median(sq)
    return med.clamp(min=1e-12)


def rbf_kernel(X: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(X, X, p=2.0) ** 2
    return torch.exp(-d2 / (2.0 * sigma_sq))


def hsic_unbiased_gaussian(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
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


def mi_sequence_hsic(Hbm: torch.Tensor, Hy: torch.Tensor) -> np.ndarray:
    B, N_x, D = Hbm.shape
    B2, _, _ = Hy.shape
    assert B == B2, f"Batch size mismatch: Hbm={Hbm.shape}, Hy={Hy.shape}"
    hy_single = Hy[:, 0, :].unsqueeze(1)
    Hy_expanded = hy_single.expand(-1, N_x, -1)
    out = []
    for p in range(N_x):
        X = zscore_cols(Hbm[:, p, :].contiguous())
        Y = zscore_cols(Hy_expanded[:, p, :].contiguous())
        out.append(hsic_unbiased_gaussian(X, Y).detach().float().cpu().item())
    return np.asarray(out, dtype=np.float64)


def forward_collect_layers(model, x_enc):
    """提取 decoder 各层的隐藏状态"""
    core = model.module if hasattr(model, "module") else model
    B, L, M = x_enc.shape
    means = x_enc.mean(1, keepdim=True).detach()
    x = x_enc - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers = []
    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _, _ = o3(h, attn_mask=mask)
        layers.append(pool(h.detach()))
    return layers, int(n_vars), int(N)


def forward_y_collect_layers(model, y):
    """提取真值 y 的 decoder 隐藏状态"""
    core = model.module if hasattr(model, "module") else model
    B, L, M = y.shape
    means = y.mean(1, keepdim=True).detach()
    x = y - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x = x / stdev
    x2 = x.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)
    BM, N, D = dec_in.shape

    def pool(z):
        return z.view(B, n_vars, N, D).mean(dim=1)

    h = dec_in
    layers = []
    mask = TriangularCausalMask(BM, N, device=h.device)
    for i, o3 in enumerate(core.decoder.attn_layers):
        if i >= MI_DECODER_LAYER_CAP:
            break
        h, _, _ = o3(h, attn_mask=mask)
        layers.append(pool(h.detach()))
    return layers


def build_namespace(args):
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
        "model_id": "tsne_analysis",
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
    }.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


# ─── t-SNE 可视化函数 ─────────────────────────────────────────────────────────

def run_tsne(
    hidden_states: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    对隐藏状态进行 t-SNE 降维。

    参数说明：
    - perplexity: 最近邻数量，控制 t-SNE 的局部/全局权衡。
      值越大，考虑更多全局结构；建议在 5-50 之间。
    - n_iter: 迭代次数，建议 >= 1000 以确保收敛。
    - random_state: 随机种子，保证结果可复现。

    返回: 降维后的 2D 坐标，shape [N_samples, 2]
    """
    print(f"[t-SNE] Input shape: {hidden_states.shape}")
    print(f"[t-SNE] perplexity={perplexity}, n_iter={n_iter}, random_state={random_state}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",       # 使用 PCA 初始化，比随机初始化更稳定
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(hidden_states)
    print(f"[t-SNE] Output shape: {embedding.shape}")
    return embedding


def plot_tsne_scatter(
    out_path: str,
    embedding: np.ndarray,
    mi_scores: np.ndarray,
    q3: float,
    label_high: str = "High-MI (Q3+)",
    label_low: str = "Low-MI (≤Q3)",
) -> None:
    """
    绘制 t-SNE 散点图，根据 MI 分组着色。

    embedding: 降维后的 2D 坐标，shape [N_samples, 2]
    mi_scores: MI 分数，shape [N_samples,]
    q3: Q3 分位阈值
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 根据 MI 阈值分组
    is_high_mi = mi_scores > q3

    # High-MI 组：鲜艳红色，较高不透明度
    high_coords = embedding[is_high_mi]
    ax.scatter(
        high_coords[:, 0],
        high_coords[:, 1],
        c="#e41a1c",           # 鲜红色
        alpha=0.8,
        s=60,
        label=f"{label_high} (n={len(high_coords)})",
        edgecolors="darkred",
        linewidths=0.5,
    )

    # Low-MI 组：淡蓝色，较低不透明度（防止遮挡 High-MI）
    low_coords = embedding[~is_high_mi]
    ax.scatter(
        low_coords[:, 0],
        low_coords[:, 1],
        c="#377eb8",           # 蓝色
        alpha=0.4,
        s=40,
        label=f"{label_low} (n={len(low_coords)})",
        edgecolors="steelblue",
        linewidths=0.3,
    )

    # 计算并标注质心
    centroid_high = high_coords.mean(axis=0)
    centroid_low = low_coords.mean(axis=0)

    ax.scatter(
        centroid_high[0], centroid_high[1],
        c="gold", marker="*", s=400, edgecolors="black", linewidths=2,
        zorder=10, label=f"High-MI Centroid"
    )
    ax.scatter(
        centroid_low[0], centroid_low[1],
        c="cyan", marker="*", s=400, edgecolors="black", linewidths=2,
        zorder=10, label=f"Low-MI Centroid"
    )

    # 绘制质心连线（显示偏移方向）
    ax.annotate(
        "",
        xy=centroid_high,
        xytext=centroid_low,
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

    # 计算质心距离
    centroid_distance = np.linalg.norm(centroid_high - centroid_low)
    print(f"[Stats] Centroid distance (High-MI vs Low-MI): {centroid_distance:.4f}")

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(
        "t-SNE Visualization of Hidden States:\n"
        "High vs. Low MI Patches",
        fontsize=14,
    )
    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = (
        f"Q3 Threshold: {q3:.4f}\n"
        f"High-MI Count: {len(high_coords)}\n"
        f"Low-MI Count: {len(low_coords)}\n"
        f"Centroid Distance: {centroid_distance:.4f}\n"
        f"Total Samples: {len(mi_scores)}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_tsne_density(
    out_path: str,
    embedding: np.ndarray,
    mi_scores: np.ndarray,
    q3: float,
) -> None:
    """
    绘制带密度云的 t-SNE 图，更直观展示分布。
    """
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    is_high_mi = mi_scores > q3

    for ax, is_high, title, color in [
        (axes[0], True, "High-MI Patches", "#e41a1c"),
        (axes[1], False, "Low-MI Patches", "#377eb8"),
    ]:
        coords = embedding[is_high]

        # 散点图
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=color, alpha=0.6, s=30,
        )

        # 如果样本足够多，添加密度等高线
        if len(coords) > 10:
            try:
                xy = np.vstack([coords[:, 0], coords[:, 1]])
                kde = gaussian_kde(xy)
                xmin, xmax = coords[:, 0].min() - 1, coords[:, 0].max() + 1
                ymin, ymax = coords[:, 1].min() - 1, coords[:, 1].max() + 1
                xx, yy = np.meshgrid(
                    np.linspace(xmin, xmax, 50),
                    np.linspace(ymin, ymax, 50)
                )
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors="black", alpha=0.3)
            except Exception as e:
                print(f"[Warning] Could not compute density: {e}")

        ax.set_title(f"{title} (n={len(coords)})", fontsize=12)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("t-SNE Density Analysis by MI Group", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_tsne_mi_gradient(
    out_path: str,
    embedding: np.ndarray,
    mi_scores: np.ndarray,
) -> None:
    """
    绘制按 MI 分数渐变着色的 t-SNE 图（不分 bin，用渐变色）。
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 按 MI 分数排序，取颜色映射
    mi_normalized = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-12)

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=mi_scores,
        cmap="RdYlBu_r",  # 红（高 MI）→ 蓝（低 MI）
        alpha=0.7,
        s=50,
        edgecolors="grey",
        linewidths=0.3,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("MI Score (HSIC)", fontsize=11)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(
        "t-SNE Visualization of Hidden States:\n"
        "MI Score Gradient",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─── 主函数 ────────────────────────────────────────────────────────────────────

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
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--subset_rand_ratio", type=float, default=1.0)
    p.add_argument("--use_ims", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", type=str, default="./results/tsne_etth1")
    p.add_argument("--max_batches", type=int, default=0, help="0 = full test loader")
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最大 t-SNE 采样数量，0=不限制（使用全部样本）",
    )
    # t-SNE 参数
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--tsne_random_state", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_multi_gpu", action="store_true")
    args = p.parse_args()

    rank = 0
    local_rank = 0
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "Multi-GPU requires torchrun. Example:\n"
                "  torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_mi_tsne.py \\\n"
                "    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt --use_multi_gpu ..."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("use_multi_gpu requires CUDA.")
        n_visible = torch.cuda.device_count()
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

    ns = build_namespace(args)
    ns.use_multi_gpu = bool(args.use_multi_gpu)
    ns.use_gpu = bool(torch.cuda.is_available())

    out_dir = args.out_dir
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if args.use_multi_gpu:
        dist.barrier()

    device = torch.device(f"cuda:{local_rank}") if args.use_multi_gpu else torch.device(args.device)

    _, loader = data_provider(ns, flag="test")
    model = Model(ns).to(device)
    model.eval()
    if args.use_multi_gpu:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── 多卡数据分割：每个 rank 通过 DistributedSampler 自动分配数据 ───────────
    if args.use_multi_gpu:
        total_samples = len(loader.dataset)
        samples_per_rank = total_samples // world_size
        remainder = total_samples % world_size

        # 计算当前 rank 的数据范围（用于日志）
        start_idx = rank * samples_per_rank + min(rank, remainder)
        end_idx = start_idx + samples_per_rank + (1 if rank < remainder else 0)
        num_local_samples = end_idx - start_idx

        print(f"[Rank {rank}] Total samples: {total_samples}, local range [{start_idx}, {end_idx}), count: {num_local_samples}")
    else:
        num_local_samples = len(loader.dataset)
        print(f"[Single GPU] Processing all {num_local_samples} samples")

    # ── 收集所有 patch 的隐藏状态和 MI 分数 ─────────────────────────────────
    all_hidden_states = []  # shape [total_patches, D_model]
    all_mi_scores = []      # shape [total_patches,]

    for batch_idx, (batch_x, batch_y, _, _) in enumerate(loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        B, Lx, C = batch_x.shape

        if args.use_ims:
            y_future = batch_y[:, ns.label_len: ns.label_len + ns.pred_len, :]
        else:
            y_future = batch_y[:, -ns.pred_len:, :]

        with torch.no_grad():
            layer_h_x, nvars, N = forward_collect_layers(model, batch_x)
            layer_h_y = forward_y_collect_layers(model, y_future)

        # 使用最后一层（layer -1）的隐藏状态
        h_last = layer_h_x[-1].cpu().numpy()  # shape: [B, N, D]
        B, N, D = h_last.shape

        # 重新组织：对每个 patch，取所有 batch 的平均值作为该 patch 的特征
        # shape: [B, N, D] -> [N, D]（取 batch 平均）
        h_patch = h_last.mean(axis=0)  # [N, D]

        # 计算该 batch 的 MI 分数
        mi_curve = mi_sequence_hsic(
            torch.from_numpy(h_last).float(),
            torch.from_numpy(layer_h_y[-1].cpu().numpy()).float()
        )

        for pi in range(N):
            all_hidden_states.append(h_patch[pi])
            all_mi_scores.append(mi_curve[pi])

        if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
            break

        # 进度提示
        if (batch_idx + 1) % 50 == 0:
            print(f"[Rank {rank}] Batch {batch_idx + 1}, collected {len(all_mi_scores)} patches")

    print(f"[Rank {rank}] Finished collection: {len(all_mi_scores)} patches")

    # ── 多卡汇总：所有 rank 的数据汇聚到 rank 0 ────────────────────────────────
    if args.use_multi_gpu:
        dist.barrier()  # 同步，确保所有 rank 都完成数据收集

        # 准备本地数据
        local_h = np.array(all_hidden_states, dtype=np.float64)
        local_m = np.array(all_mi_scores, dtype=np.float64)
        local_n = len(local_m)

        # 先同步每个 rank 的样本数
        local_n_tensor = torch.tensor([local_n], dtype=torch.long, device=device)
        count_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(count_list, local_n_tensor)
        counts = [c.item() for c in count_list]
        total_n = sum(counts)
        print(f"[Rank {rank}] Counts per rank: {counts}, total: {total_n}")

        # 准备接收 buffer（rank 0）
        if rank == 0:
            all_hidden_states_gathered = np.zeros((total_n, args.d_model), dtype=np.float64)
            all_mi_scores_gathered = np.zeros(total_n, dtype=np.float64)
        else:
            all_hidden_states_gathered = None
            all_mi_scores_gathered = None

        # 使用 all_gather 收集数据（统一使用 float32）
        gathered_h_list = [torch.zeros((counts[i], args.d_model), dtype=torch.float32, device=device) for i in range(world_size)]
        gathered_m_list = [torch.zeros(counts[i], dtype=torch.float32, device=device) for i in range(world_size)]

        local_h_tensor = torch.from_numpy(local_h).float().to(device)
        local_m_tensor = torch.from_numpy(local_m).float().to(device)

        dist.all_gather(gathered_h_list, local_h_tensor)
        dist.all_gather(gathered_m_list, local_m_tensor)

        # rank 0 拼接所有数据
        if rank == 0:
            for i, (h, m) in enumerate(zip(gathered_h_list, gathered_m_list)):
                start_idx = sum(counts[:i])
                end_idx = start_idx + counts[i]
                all_hidden_states_gathered[start_idx:end_idx] = h.cpu().numpy()
                all_mi_scores_gathered[start_idx:end_idx] = m.cpu().numpy()
            print(f"[Rank 0] Gathered {total_n} patches from {world_size} ranks")

        dist.barrier()  # 再次同步

        # rank 0 之外后续都不需要了，直接跳过
        if rank != 0:
            dist.destroy_process_group()
            return

        hidden_states_array = all_hidden_states_gathered
        mi_scores_array = all_mi_scores_gathered
    else:
        hidden_states_array = np.array(all_hidden_states, dtype=np.float64)
        mi_scores_array = np.array(all_mi_scores, dtype=np.float64)

    print(f"\n[Data] Total patches: {len(mi_scores_array)}")
    print(f"[Data] Hidden states shape: {hidden_states_array.shape}")
    print(f"[Data] MI scores shape: {mi_scores_array.shape}")

    # ── 数据准备：确保长度一致，处理异常值 ─────────────────────────────────
    assert len(hidden_states_array) == len(mi_scores_array), \
        "Hidden states and MI scores length mismatch!"

    # 过滤掉 NaN 的 MI 分数
    valid_mask = np.isfinite(mi_scores_array)
    hidden_states_array = hidden_states_array[valid_mask]
    mi_scores_array = mi_scores_array[valid_mask]
    print(f"[Data] After filtering NaN: {len(mi_scores_array)} samples")

    # ── 如果样本太多，进行随机采样 ───────────────────────────────────────────
    if args.max_samples > 0 and len(mi_scores_array) > args.max_samples:
        print(f"[Sampling] Too many samples ({len(mi_scores_array)}), sampling {args.max_samples}")
        indices = np.random.RandomState(args.tsne_random_state).choice(
            len(mi_scores_array), size=args.max_samples, replace=False
        )
        hidden_states_array = hidden_states_array[indices]
        mi_scores_array = mi_scores_array[indices]

    # 计算 Q3 阈值
    q3 = float(np.percentile(mi_scores_array, 75))
    print(f"[Threshold] Q3 = {q3:.6f}")

    # ── t-SNE 降维 ───────────────────────────────────────────────────────────
    if rank == 0:
        embedding_2d = run_tsne(
            hidden_states_array,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_n_iter,
            random_state=args.tsne_random_state,
        )

        # ── 可视化 ───────────────────────────────────────────────────────────
        # 图1：主散点图（按 MI 分组着色 + 质心标注）
        plot_tsne_scatter(
            out_path=os.path.join(out_dir, "tsne_mi_scatter.png"),
            embedding=embedding_2d,
            mi_scores=mi_scores_array,
            q3=q3,
        )

        # 图2：分组密度图
        plot_tsne_density(
            out_path=os.path.join(out_dir, "tsne_density.png"),
            embedding=embedding_2d,
            mi_scores=mi_scores_array,
            q3=q3,
        )

        # 图3：MI 渐变图
        plot_tsne_mi_gradient(
            out_path=os.path.join(out_dir, "tsne_mi_gradient.png"),
            embedding=embedding_2d,
            mi_scores=mi_scores_array,
        )

        # 打印统计摘要
        print("\n=== t-SNE Analysis Summary ===")
        print(f"Total samples: {len(mi_scores_array)}")
        print(f"High-MI patches (>Q3): {np.sum(mi_scores_array > q3)}")
        print(f"Low-MI patches (<=Q3): {np.sum(mi_scores_array <= q3)}")
        print(f"t-SNE perplexity: {args.tsne_perplexity}")
        print(f"t-SNE n_iter: {args.tsne_n_iter}")
        print(f"Output directory: {out_dir}/")

    if args.use_multi_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
