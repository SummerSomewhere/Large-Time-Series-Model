"""
基于信息瓶颈（IB）理论的 Top-K 物理 Token 剪枝实验
====================================================
验证假设：高互信息（High-MI）Token 是时间序列的预测锚点，
剪掉低互信息（Low-MI）Token 对端到端预测精度几乎无影响，
同时大幅提升 GPU 推理速度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 1. Token 剪枝模块
# ══════════════════════════════════════════════════════════════════════════════


class IBTokenPruningLayer(nn.Module):
    """
    基于互信息（IB）分数的 Token 剪枝层。

    Args:
        keep_ratio (float): 保留 Token 的比例，取值 (0, 1]。
                          例如 keep_ratio=0.8 表示保留前 80% 的高 MI Token。

    输入:
        x (Tensor): 隐藏表征，形状 (B, N, D)
        mi_scores (Tensor): 每个 Token 的互信息分数，形状 (B, N)

    返回:
        pruned_x (Tensor): 裁剪后的表征，形状 (B, K, D)
        topk_indices (LongTensor): 被保留 Token 的原始索引，形状 (B, K)，
                                   已按升序排列以保持时序顺序
    """

    def __init__(self, keep_ratio: float = 0.8):
        super().__init__()
        if not (0 < keep_ratio <= 1):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor, mi_scores: torch.Tensor):
        B, N, D = x.shape
        K = max(1, int(N * self.keep_ratio))

        # 在每个样本内独立取 top-K 高 MI Token
        # topk 返回 (values, indices)，indices 形状 (B, K)
        _, topk_indices = torch.topk(mi_scores, k=K, dim=1, largest=True)
        topk_indices, _ = torch.sort(topk_indices, dim=1)   # 升序排列，保持时序顺序

        # topk_indices 当前是 (B, K)，内部顺序已由 sorted=True 保证为升序，
        # 但不同样本间的索引无意义，需在序列维度 gather 时保持维度对齐。
        # 对每个样本独立 gather：
        # x[:, i, :] 中 i 取 topk_indices[b] 中的值
        pruned_x = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))

        return pruned_x, topk_indices


class RandomTokenPruningLayer(nn.Module):
    """
    随机 Token 剪枝层（Baseline），与 IB 剪枝形成对照。

    输入:
        x (Tensor): 形状 (B, N, D)
        mi_scores (Tensor): 此处不使用，仅保持接口一致性

    返回:
        pruned_x (Tensor): 裁剪后的表征，形状 (B, K, D)
        random_indices (LongTensor): 随机选取的索引，形状 (B, K)，已排序
    """

    def __init__(self, keep_ratio: float = 0.8):
        super().__init__()
        if not (0 < keep_ratio <= 1):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor, mi_scores: torch.Tensor = None):
        B, N, D = x.shape
        K = max(1, int(N * self.keep_ratio))

        # 为每个样本生成互不相同的随机索引
        # 先在 [0, N) 范围内采样 K 个不重复的索引，再排序以保持时序顺序
        indices = torch.arange(N, device=x.device, dtype=torch.long)
        random_indices_list = []
        for b in range(B):
            perm = torch.randperm(N, device=x.device)
            sampled = perm[:K]
            sorted_indices, _ = torch.sort(sampled)   # 升序排列，保持时序顺序
            random_indices_list.append(sorted_indices)

        random_indices = torch.stack(random_indices_list, dim=0)   # (B, K)

        pruned_x = torch.gather(x, 1, random_indices.unsqueeze(-1).expand(-1, -1, D))

        return pruned_x, random_indices


# ══════════════════════════════════════════════════════════════════════════════
# 2. 模拟时间序列预测模型
# ══════════════════════════════════════════════════════════════════════════════


class TransformerBlock(nn.Module):
    """简化的 Transformer Block：Multi-Head Attention + FFN"""

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm MHA
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class PrunableModelWrapper(nn.Module):
    """
    在指定层（第 prune_layer 层）插入 Token 剪枝模块的模拟时序预测模型。

    结构: Embedding → N 个 TransformerBlock → PruningLayer（第 prune_layer 层）
          → 剩余 Block → Output Head
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 6,
        prune_layer: int = 3,
        keep_ratio: float = 0.8,
        pruning_type: str = "ib",   # "ib" 或 "random"
    ):
        super().__init__()
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio
        self.pruning_type = pruning_type

        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # 剪枝层（第 prune_layer 个 block 之后）
        if pruning_type == "ib":
            self.pruner = IBTokenPruningLayer(keep_ratio)
        else:
            self.pruner = RandomTokenPruningLayer(keep_ratio)

        self.out_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mi_scores: torch.Tensor = None):
        """
        Args:
            x: (B, N, 1) 输入时间序列
            mi_scores: (B, N) 互信息分数，仅 pruning_type="ib" 时使用

        Returns:
            dict，包含预测结果、中间表征等
        """
        B, N, _ = x.shape
        x = self.input_proj(x)

        # 前半部分 block
        for i in range(self.prune_layer):
            x = self.blocks[i](x)

        # 剪枝
        if self.pruning_type == "ib":
            x, indices = self.pruner(x, mi_scores)
        else:
            x, indices = self.pruner(x)

        # 剩余 block（若 Token 数量减少，对序列长度做动态处理）
        for i in range(self.prune_layer, len(self.blocks)):
            x = self.blocks[i](x)

        # 全局平均池化 + 预测头
        pooled = x.mean(dim=1)          # (B, D)
        out = self.out_head(pooled)    # (B, 1)
        return {"pred": out, "pruned_indices": indices, "pruned_x": x}


# ══════════════════════════════════════════════════════════════════════════════
# 3. 模拟数据集与 MI 分数生成
# ══════════════════════════════════════════════════════════════════════════════


def generate_synthetic_data(
    B: int = 64,
    N: int = 512,
    seed: int = 42,
    mi_peak_ratio: float = 0.2,
    mi_peak_strength: float = 5.0,
):
    """
    生成模拟时间序列数据和 MI 分数。

    Args:
        B: 批大小
        N: 序列长度（Token 数量）
        seed: 随机种子
        mi_peak_ratio: 高 MI Token 占总 Token 的比例，默认 20%
        mi_peak_strength: 高 MI Token 分数相对背景的倍率

    Returns:
        x (B, N, 1): 输入序列（含周期信号 + 噪声）
        y (B, 1): 预测目标（取最后几个时间步的平均，这里简化为序列均值）
        mi_scores (B, N): 互信息分数，关键 Token 分数显著更高
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 输入序列 ──────────────────────────────────────────────────────────────
    # 构造一个带周期成分和局部突变的时间序列
    t = torch.linspace(0, 4 * np.pi, N).unsqueeze(0).expand(B, -1)  # (B, N)

    # 基础周期信号（模拟日/周周期）
    base = 0.5 * torch.sin(t) + 0.3 * torch.sin(2 * t + 0.5)
    # 添加少量噪声
    noise = torch.randn(B, N, 1) * 0.15
    # 随机局部突增（模拟异常事件，关键信息锚点）
    spike = torch.zeros(B, N, 1)
    spike_idx = torch.randint(N // 4, N, (B,))
    spike.scatter_(1, spike_idx.unsqueeze(-1), torch.randn(B, 1, 1) * 2.0)

    x = (base.unsqueeze(-1) + noise + spike)   # (B, N, 1)

    # ── 预测目标 ───────────────────────────────────────────────────────────────
    # 简化为序列整体均值的偏移
    y = x.mean(dim=1, keepdim=True) + torch.randn(B, 1) * 0.05   # (B, 1)

    # ── MI 分数 ───────────────────────────────────────────────────────────────
    # 背景分数（均匀随机噪声）
    bg = torch.rand(B, N)

    # 人工构造"高 MI 锚点"：在固定区间（如前 20%）内赋予高分
    n_peaks = max(1, int(N * mi_peak_ratio))
    peak_scores = torch.rand(B, N) * (mi_peak_strength - 1.0) + 1.0
    mask = torch.zeros(B, N, dtype=torch.bool)
    # 在序列前段和中段各放置若干高 MI 锚点
    peak_positions = torch.cat([
        torch.randint(0, N // 3, (B, n_peaks // 2)),
        torch.randint(N // 3, N, (B, n_peaks - n_peaks // 2)),
    ], dim=1) % N
    peak_positions = peak_positions.sort(dim=1).values
    mask.scatter_(1, peak_positions, True)

    mi_scores = torch.where(mask, peak_scores * bg, bg)

    # 每个样本内做 min-max 归一化到 [0, 1]，使 topk 在样本内公平比较
    mi_min = mi_scores.min(dim=1, keepdim=True).values
    mi_max = mi_scores.max(dim=1, keepdim=True).values
    mi_scores = (mi_scores - mi_min) / (mi_max - mi_min + 1e-8)

    return x, y, mi_scores


# ══════════════════════════════════════════════════════════════════════════════
# 4. 训练与评估
# ══════════════════════════════════════════════════════════════════════════════


def train_model(model, x, y, mi_scores, epochs=30, lr=1e-3, device="cuda"):
    """简单训练循环，返回训练好的模型。"""
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    mi_scores = mi_scores.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model(x, mi_scores)
        loss = criterion(out["pred"], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return model


@torch.no_grad()
def evaluate(model, x, y, mi_scores, device="cuda"):
    """评估模型，返回 MSE 和推理耗时。"""
    model.eval()
    model = model.to(device)

    x = x.to(device)
    y = y.to(device)
    mi_scores = mi_scores.to(device) if mi_scores is not None else None

    # 预热
    _ = model(x, mi_scores)

    # 计时
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.perf_counter()
    out = model(x, mi_scores)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t1 = time.perf_counter()

    mse = F.mse_loss(out["pred"], y).item()
    latency_ms = (t1 - t0) * 1000.0

    return mse, latency_ms


# ══════════════════════════════════════════════════════════════════════════════
# 5. 主实验
# ══════════════════════════════════════════════════════════════════════════════


def run_experiment(
    keep_ratios=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    n_trials=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
):
    """
    在不同 keep_ratio 下分别运行 IB 剪枝和随机剪枝实验，
    收集 MSE 和推理加速比，返回结果 DataFrame。
    """
    print(f"[设备] {device}")
    print(f"[Keep Ratios] {keep_ratios}")
    print(f"[每设置重复次数] {n_trials}\n")

    # 生成固定的测试集
    x_test, y_test, mi_test = generate_synthetic_data(B=128, N=512, seed=seed + 1)
    # 生成固定验证集
    x_val, y_val, mi_val = generate_synthetic_data(B=64, N=512, seed=seed + 2)

    results = []

    for kr in keep_ratios:
        drop_rate = 1.0 - kr
        print(f"\n{'='*60}")
        print(f"  Keep Ratio = {kr:.1f}  (Drop Rate = {drop_rate:.0%})")
        print(f"{'='*60}")

        ib_mses, ib_lats = [], []
        rand_mses, rand_lats = [], []

        for trial in range(n_trials):
            train_seed = seed + trial * 100

            # ── IB 剪枝 ────────────────────────────────────────────────────────
            x_train, y_train, mi_train = generate_synthetic_data(B=128, N=512, seed=train_seed)
            ib_model = PrunableModelWrapper(
                d_model=64, n_heads=4, n_layers=6,
                prune_layer=3, keep_ratio=kr, pruning_type="ib"
            )
            ib_model = train_model(ib_model, x_train, y_train, mi_train, epochs=30, device=device)
            ib_mse, ib_lat = evaluate(ib_model, x_val, y_val, mi_val, device=device)
            ib_mses.append(ib_mse)
            ib_lats.append(ib_lat)

            # ── Random 剪枝 ────────────────────────────────────────────────────
            x_train, y_train, _ = generate_synthetic_data(B=128, N=512, seed=train_seed)
            rand_model = PrunableModelWrapper(
                d_model=64, n_heads=4, n_layers=6,
                prune_layer=3, keep_ratio=kr, pruning_type="random"
            )
            rand_model = train_model(rand_model, x_train, y_train, None, epochs=30, device=device)
            rand_mse, rand_lat = evaluate(rand_model, x_val, y_val, None, device=device)
            rand_mses.append(rand_mse)
            rand_lats.append(rand_lat)

            print(f"  Trial {trial+1}: IB MSE={ib_mse:.6f} | Rand MSE={rand_mse:.6f}")

        ib_mse_mean = np.mean(ib_mses)
        ib_mse_std = np.std(ib_mses)
        rand_mse_mean = np.mean(rand_mses)
        rand_mse_std = np.std(rand_mses)
        ib_lat_mean = np.mean(ib_lats)
        rand_lat_mean = np.mean(rand_lats)

        # 以 keep_ratio=1.0 的推理延迟作为基线，计算加速比
        if not results:
            baseline_ib_lat = ib_lat_mean
            baseline_rand_lat = rand_lat_mean

        speedup_ib = baseline_ib_lat / ib_lat_mean if ib_lat_mean > 0 else 1.0
        speedup_rand = baseline_rand_lat / rand_lat_mean if rand_lat_mean > 0 else 1.0

        results.append({
            "keep_ratio": kr,
            "drop_rate": drop_rate,
            "ib_mse_mean": ib_mse_mean,
            "ib_mse_std": ib_mse_std,
            "rand_mse_mean": rand_mse_mean,
            "rand_mse_std": rand_mse_std,
            "speedup_ib": speedup_ib,
            "speedup_rand": speedup_rand,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. 可视化：双轴 Pareto 边界图
# ══════════════════════════════════════════════════════════════════════════════


def plot_pareto_frontier(results, save_path="mi_pruning_pareto.png", dpi=150):
    """
    绘制双轴折线图：
      - 主轴（左侧）：预测 MSE（IB Dropping vs Random Dropping）
      - 副轴（右侧）：推理加速比（以 Drop Rate=0 为基线）
    """
    drop_rates = [r["drop_rate"] * 100 for r in results]
    ib_mse = [r["ib_mse_mean"] for r in results]
    ib_std = [r["ib_mse_std"] for r in results]
    rand_mse = [r["rand_mse_mean"] for r in results]
    rand_std = [r["rand_mse_std"] for r in results]
    speedup_ib = [r["speedup_ib"] for r in results]
    speedup_rand = [r["speedup_rand"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ── 左轴：MSE ─────────────────────────────────────────────────────────────
    color_ib = "#1f77b4"   # 学术蓝
    color_rand = "#ff7f0e"  # 学术橙

    (line_mse_ib,) = ax1.plot(
        drop_rates, ib_mse,
        marker="o", markersize=8, linewidth=2.5,
        color=color_ib, label="IB-Guided Dropping"
    )
    ax1.fill_between(
        drop_rates,
        np.array(ib_mse) - np.array(ib_std),
        np.array(ib_mse) + np.array(ib_std),
        alpha=0.15, color=color_ib
    )

    (line_mse_rand,) = ax1.plot(
        drop_rates, rand_mse,
        marker="s", markersize=8, linewidth=2.5, linestyle="--",
        color=color_rand, label="Random Dropping"
    )
    ax1.fill_between(
        drop_rates,
        np.array(rand_mse) - np.array(rand_std),
        np.array(rand_mse) + np.array(rand_std),
        alpha=0.15, color=color_rand
    )

    ax1.set_xlabel("Token Drop Rate (%)", fontsize=13)
    ax1.set_ylabel("Prediction MSE", fontsize=13, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.25, axis="y")

    # ── 右轴：加速比 ───────────────────────────────────────────────────────────
    ax2 = ax1.twinx()
    (line_speedup_ib,) = ax2.plot(
        drop_rates, speedup_ib,
        marker="^", markersize=7, linewidth=2, linestyle=":",
        color="#2ca02c", label="IB Speedup"
    )
    (line_speedup_rand,) = ax2.plot(
        drop_rates, speedup_rand,
        marker="v", markersize=7, linewidth=2, linestyle=":",
        color="#d62728", label="Random Speedup"
    )
    ax2.set_ylabel("Throughput Speedup (×)", fontsize=13, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # 加速比基线（1.0）虚线
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    # ── 图例 ───────────────────────────────────────────────────────────────────
    lines = [line_mse_ib, line_mse_rand, line_speedup_ib, line_speedup_rand]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11, framealpha=0.9)

    # ── 标题与样式 ─────────────────────────────────────────────────────────────
    ax1.set_title(
        "IB Token Pruning: Accuracy vs. Speed Trade-off",
        fontsize=14, fontweight="bold", pad=12
    )
    plt.setp(ax1.get_xticklabels(), fontsize=11)
    plt.setp(ax1.get_yticklabels(), fontsize=11)
    plt.setp(ax2.get_yticklabels(), fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Pareto 图已保存] {save_path}")

    # ── 打印汇总表 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print(f"{'Drop%':>6} │ {'IB MSE':>10} {'±':>6} │ {'Rand MSE':>10} {'±':>6} │ {'IB Spd':>8} {'Rand Spd':>9}")
    print("─" * 75)
    for r in results:
        dr = f"{r['drop_rate']*100:>5.0f}%"
        ib = f"{r['ib_mse_mean']:>10.6f} {r['ib_mse_std']:>5.4f}"
        rm = f"{r['rand_mse_mean']:>10.6f} {r['rand_mse_std']:>5.4f}"
        si = f"{r['speedup_ib']:>8.2f}×"
        sr = f"{r['speedup_rand']:>9.2f}×"
        print(f"{dr} │ {ib} │ {rm} │ {si} {sr}")
    print("─" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# 7. 入口
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # ── 超参数 ────────────────────────────────────────────────────────────────
    KEEP_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    N_TRIALS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 2025
    OUTPUT_DIR = "./output_mi_pruning"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   IB Token Pruning Experiment — Time Series Foundation   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── 运行实验 ──────────────────────────────────────────────────────────────
    results = run_experiment(
        keep_ratios=KEEP_RATIOS,
        n_trials=N_TRIALS,
        device=DEVICE,
        seed=SEED,
    )

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(OUTPUT_DIR, "mi_pruning_pareto.png")
    plot_pareto_frontier(results, save_path=save_path, dpi=150)

    # 保存数值结果
    import json
    result_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(result_path, "w") as f:
        # 转换 numpy 类型为 Python 原生类型
        serializable = []
        for r in results:
            serializable.append({k: float(v) for k, v in r.items()})
        json.dump(serializable, f, indent=2)
    print(f"[数值结果已保存] {result_path}")
