#!/usr/bin/env python3
"""
表征垄断测试 (Information Monopoly Test) — Linear Probing
=========================================================

故事线：证明模型在深层并没有把信息均匀分布，而是把所有关于未来的
"天机"都压缩（垄断）在了少数高 MI 的 Patch 表征中。

实验设计：
  1. 冻结预训练好的 Timer 模型（不再更新 Transformer 权重）。
  2. 将序列输入模型，提取最后一层每个 Patch 对应的隐藏状态 H ∈ R^{d_model}。
  3. 从 timer_mi_ksg_pca.py 读取每 patch 的 I(H,Y) 曲线（MI 曲线）。
  4. 高 MI 探针：仅使用高 MI Patch 的 H，训练单层线性回归器，映射未来 96 步。
  5. 低 MI 探针：仅使用低 MI Patch 的 H，训练同样的线性回归器去预测未来。
  6. Baseline: 对所有 patch 做 mean pooling，训练全量 linear probe。

能突出什么贡献：
  Linear Probing 是深度学习评估表征质量的黄金标准。
  仅靠高 MI Patch 的一层线性映射，就能达到甚至超越全序列微调 80% 的性能；
  而低 MI Patch 的特征哪怕经过线性映射，预测结果也近乎瞎猜。
  这硬核地证明了：高 MI Token 就是任务相关信息的绝对载体。

展示方式：绘制三条重建曲线与真实曲线的叠加对比图。

依赖：
  - timer_mi_ksg_pca.py 输出的 global_mi_peaks_{model_id}.json
  - 预训练 Timer checkpoint
  - 测试集（与 MI 计算相同的测试集）

Usage:
  python experiments/timer_mi_monopoly_probe.py \
      --mi_result_dir ./results/timer_mi_ksg_pca/Timer_MI_20260516_114023/ \
      --root_path ./datasets/ --data_path ETTh1.csv \
      --seq_len 672 --pred_len 96 --patch_len 96 \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --out_dir ./outputs/timer_mi_monopoly_probe/
"""

import argparse
import gc
import json
import os
import sys
import datetime
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from utils.masking import TriangularCausalMask


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Config & Model Builder
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(ckpt_path: str, patch_len: int, stride: int,
                d_model: int, d_ff: int, e_layers: int,
                n_heads: int, dropout: float,
                seq_len: int, pred_len: int) -> Model:
    cfg = Config(
        task_name='forecast',
        ckpt_path=ckpt_path,
        patch_len=patch_len,
        stride=stride,
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Token Extraction (reused from timer_layer_inlayer_probe.py)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_layer_tokens(model, data_loader, device, n_layers: int):
    """
    提取每层每个 patch 的 hidden states。

    Returns:
        hist_tokens:  list of [N, n_patches, D]  per layer
        future_tokens: list of [N, n_future_patches, D] per layer
        x_patch_tokens: [N, n_patches, D] — patch embedding output
        n_patches: int
        n_future_patches: int
    """
    core = _unwrap(model)
    all_hist = [[] for _ in range(n_layers)]
    all_future = [[] for _ in range(n_layers)]
    all_x_patch = []
    n_patches = None
    n_future_patches = None

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(
                data_loader, desc="提取 token 表示"):
            B = seq_x.shape[0]
            sx = seq_x.float().to(device)
            sy = seq_y.float().to(device)

            means = sx.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            xn = (sx - means) / stdev
            yn = (sy - means) / stdev

            x2 = xn.permute(0, 2, 1)
            y2 = yn.permute(0, 2, 1)

            dec_in_x, _ = core.enc_embedding(x2)
            dec_in_y, _ = core.enc_embedding(y2)
            BM, N, D = dec_in_x.shape
            BM_y, N_y, _ = dec_in_y.shape

            if n_patches is None:
                n_patches = N
                n_future_patches = N_y
                print(f"  n_patches={n_patches}, n_future_patches={n_future_patches}, D={D}")

            derived_n_vars = BM // B
            x_patch_emb = dec_in_x.view(B, derived_n_vars, N, D).mean(dim=1).float().cpu()
            all_x_patch.append(x_patch_emb)

            mask_x = TriangularCausalMask(BM, N, device=device)
            mask_y = TriangularCausalMask(BM_y, N_y, device=device)

            hx = dec_in_x
            hy = dec_in_y
            for li, layer_module in enumerate(core.decoder.attn_layers):
                hx, _, _ = layer_module(hx, attn_mask=mask_x)
                hy, _, _ = layer_module(hy, attn_mask=mask_y)
                derived_n_vars_y = BM_y // B
                all_hist[li].append(
                    hx.view(B, derived_n_vars, N, D).mean(dim=1).float().cpu())
                all_future[li].append(
                    hy.view(B, derived_n_vars_y, N_y, D).mean(dim=1).float().cpu())

            del dec_in_x, dec_in_y, hx, hy
            gc.collect()
            torch.cuda.empty_cache()

    hist_tokens = [torch.cat(toks, dim=0) for toks in all_hist]
    future_tokens = [torch.cat(toks, dim=0) for toks in all_future]
    x_patch_tokens = torch.cat(all_x_patch, dim=0)
    return hist_tokens, future_tokens, x_patch_tokens, n_patches, n_future_patches


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MI Curve Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_mi_from_json(mi_result_dir: str, n_layers: int) -> dict:
    """从 timer_mi_ksg_pca.py 的输出 JSON 读取每层的 MI 曲线 (I(H,Y))。"""
    pattern = os.path.join(mi_result_dir, "global_mi_peaks_*.json")
    matched = glob.glob(pattern)
    if not matched:
        raise FileNotFoundError(
            f"未找到 MI 结果文件: {pattern}\n"
            f"请先运行: python experiments/timer_mi_ksg_pca.py --model_id <id> ..."
        )
    json_path = matched[0]
    print(f"  [MI Loader] 读取: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    layers_dict = data.get("layers", {})
    mi_curves = {}
    for li in range(n_layers):
        key = str(li)
        if key not in layers_dict:
            raise ValueError(f"JSON 中未找到 layer={li}，可用键: {list(layers_dict.keys())}")
        mi_curves[li] = np.array(layers_dict[key]["hsic_curve"], dtype=np.float64)
    return mi_curves, data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Linear Probe
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)

    def forward(self, x):
        return self.linear(x)


def train_probe(z_train, y_train, z_val, y_val, n_epochs: int, lr: float,
                device: torch.device, verbose: bool = False):
    """
    训练线性 probe，返回 (val_mse, val_r2, val_pred, train_curve)。
    """
    in_dim = z_train.shape[1]
    out_dim = y_train.shape[1]

    probe = LinearProbe(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    crit = nn.MSELoss()

    n = z_train.shape[0]
    bs = min(256, n)
    train_curve = []

    for ep in range(n_epochs):
        probe.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            bi = idx[i:i + bs]
            opt.zero_grad()
            loss = crit(probe(z_train[bi].to(device)), y_train[bi].to(device))
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            pred = probe(z_val.to(device))
            val_mse = crit(pred, y_val.to(device)).item()
        train_curve.append(val_mse)

        if verbose and (ep + 1) % 20 == 0:
            print(f"      Epoch {ep+1}/{n_epochs}, Val MSE: {val_mse:.6f}")

    probe.eval()
    with torch.no_grad():
        final_pred = probe(z_val.to(device))
        final_mse = crit(final_pred, y_val.to(device)).item()
        ss_res = ((final_pred.cpu() - y_val) ** 2).sum().item()
        ss_tot = ((y_val - y_val.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return final_mse, r2, final_pred.detach().cpu(), train_curve


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Monopoly Probe — Core Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_monopoly_probe(
    hist_tokens: list,
    future_tokens: list,
    y_flat: np.ndarray,
    mi_curves: dict,
    n_patches: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    output_dir: str,
):
    """
    表征垄断测试：对指定层，分别用高 MI patch 和低 MI patch 的 hidden state
    训练线性 probe，预测未来序列。
    """
    print(f"\n{'=' * 70}")
    print(f"  表征垄断测试 (Information Monopoly Probe)")
    print(f"  目标层: L{args.target_layer}, pred_len={args.pred_len}")
    print(f"{'=' * 70}")

    # ── Select target layer ───────────────────────────────────────────────
    li = args.target_layer
    if li >= len(hist_tokens):
        raise ValueError(f"目标层 {li} 超出范围 (max={len(hist_tokens)-1})")

    tok = hist_tokens[li]              # [N, n_patches, D]
    fut = future_tokens[li]           # [N, n_future_patches, D_future]
    mi_curve = mi_curves[li]           # [n_patches,]

    N, n_p, D = tok.shape
    print(f"  Hidden tokens: {tok.shape}, Future tokens: {fut.shape}")
    print(f"  MI curve range: [{mi_curve.min():.4f}, {mi_curve.max():.4f}] bits")

    # ── Build target: flatten future patches to [N, pred_len] ─────────────────
    # future_tokens: [N, n_future_patches, patch_len] (n_future_patches * patch_len = pred_len)
    y_all = fut.reshape(N, -1).float()  # [N, pred_len]
    y_t = torch.from_numpy(y_flat).float()  # [N, pred_len] from dataset

    # Use y_t (normalized future from data loader) as ground truth
    assert y_t.shape == y_all.shape, \
        f"Shape mismatch: y_t={y_t.shape}, y_all={y_all.shape}"

    y_train = y_t[train_idx]
    y_val = y_t[val_idx]

    # ── Split hidden states ───────────────────────────────────────────────
    tok_train = tok[train_idx]   # [N_train, n_patches, D]
    tok_val = tok[val_idx]       # [N_val, n_patches, D]

    # ── High / Low MI patch划分 ─────────────────────────────────────────────
    q_hi = max(1, int(n_patches * args.high_mi_ratio))
    q_lo = max(1, int(n_patches * args.low_mi_ratio))

    sorted_idx = np.argsort(mi_curve)
    high_mi_patch_idx = sorted_idx[-q_hi:].tolist()    # Top q_hi patches
    low_mi_patch_idx = sorted_idx[:q_lo].tolist()       # Bottom q_lo patches

    print(f"\n  Patch 划分:")
    print(f"    高 MI: top {q_hi}/{n_patches} patches → idx={high_mi_patch_idx}")
    print(f"    低 MI: bottom {q_lo}/{n_patches} patches → idx={low_mi_patch_idx}")
    print(f"    其余 : {n_patches - q_hi - q_lo} patches")

    # ── Representation: mean pooling over selected patches ────────────────────
    high_rep_train = tok_train[:, high_mi_patch_idx, :].mean(dim=1)  # [N_train, D]
    high_rep_val   = tok_val[:, high_mi_patch_idx, :].mean(dim=1)    # [N_val, D]
    low_rep_train  = tok_train[:, low_mi_patch_idx, :].mean(dim=1)    # [N_train, D]
    low_rep_val    = tok_val[:, low_mi_patch_idx, :].mean(dim=1)      # [N_val, D]
    all_rep_train  = tok_train.mean(dim=1)                             # [N_train, D]
    all_rep_val    = tok_val.mean(dim=1)                              # [N_val, D]

    # ── Train probes ─────────────────────────────────────────────────────────
    print(f"\n  训练 Linear Probe (epochs={args.probe_epochs}, lr={args.probe_lr})...")

    # High MI probe
    print(f"\n  [高 MI Patch Probe] (n={q_hi} patches)")
    mse_h, r2_h, pred_h, curve_h = train_probe(
        high_rep_train, y_train, high_rep_val, y_val,
        args.probe_epochs, args.probe_lr, args.device, verbose=True)

    # Low MI probe
    print(f"\n  [低 MI Patch Probe] (n={q_lo} patches)")
    mse_l, r2_l, pred_l, curve_l = train_probe(
        low_rep_train, y_train, low_rep_val, y_val,
        args.probe_epochs, args.probe_lr, args.device, verbose=True)

    # All-patch baseline probe
    print(f"\n  [All-Patch Baseline Probe] (n={n_patches} patches)")
    mse_a, r2_a, pred_a, curve_a = train_probe(
        all_rep_train, y_train, all_rep_val, y_val,
        args.probe_epochs, args.probe_lr, args.device, verbose=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'=' * 50}")
    print(f"  {'Probe Type':>20} | {'MSE':>12} | {'R²':>10} | {'n_patches':>10}")
    print(f"  {'-' * 50}")
    print(f"  {'High-MI Probe':>20} | {mse_h:>12.6f} | {r2_h:>10.4f} | {q_hi:>10}")
    print(f"  {'Low-MI Probe':>20} | {mse_l:>12.6f} | {r2_l:>10.4f} | {q_lo:>10}")
    print(f"  {'All-Patch Probe':>20} | {mse_a:>12.6f} | {r2_a:>10.4f} | {n_patches:>10}")
    print(f"  {'=' * 50}")

    # "80% performance" reference
    perf_80 = 0.80 * mse_a
    print(f"\n  80% 全量性能阈值: MSE = {perf_80:.6f}")
    print(f"  高 MI 是否达标: {'✓ YES' if mse_h <= perf_80 else '✗ NO'} "
          f"(高 MI MSE={mse_h:.6f}, ratio={mse_h/mse_a:.2f}x)")

    # Compute what % of all-patch performance high-MI achieves
    high_pct = (1 - mse_h / mse_a) * 100 if mse_a > 0 else 0
    print(f"  高 MI 相对全量 MSE 减少: {high_pct:.1f}%")
    print(f"  低 MI vs 全量 MSE 比值: {mse_l/mse_a:.2f}x")
    print(f"  低 MI R²: {r2_l:.4f} {'(近乎瞎猜)' if r2_l < 0.05 else ''}")

    results = {
        "high_mi": {
            "mse": mse_h, "r2": r2_h,
            "n_patches": q_hi, "patch_indices": high_mi_patch_idx,
            "mi_values": [float(mi_curve[i]) for i in high_mi_patch_idx],
        },
        "low_mi": {
            "mse": mse_l, "r2": r2_l,
            "n_patches": q_lo, "patch_indices": low_mi_patch_idx,
            "mi_values": [float(mi_curve[i]) for i in low_mi_patch_idx],
        },
        "all_patch": {
            "mse": mse_a, "r2": r2_a,
            "n_patches": n_patches,
        },
        "target_layer": li,
        "pred_len": args.pred_len,
        "high_mi_ratio": args.high_mi_ratio,
        "low_mi_ratio": args.low_mi_ratio,
    }

    return results, {
        "pred_h": pred_h, "pred_l": pred_l, "pred_a": pred_a,
        "y_val": y_val, "curve_h": curve_h, "curve_l": curve_l,
        "curve_a": curve_a, "mi_curve": mi_curve,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Plotting — The Key Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_monopoly_results(results: dict, curves: dict, output_dir: str,
                           n_patches: int, args, dataset_name: str = "ETTh1"):
    """
    核心可视化：三条重建曲线与真实曲线的叠加对比图。
    """
    pred_h = curves["pred_h"]    # [N_val, pred_len]
    pred_l = curves["pred_l"]
    pred_a = curves["pred_a"]
    y_val  = curves["y_val"]     # [N_val, pred_len]
    mi_curve = curves["mi_curve"]

    # ── Select representative samples ───────────────────────────────────────
    # Sample 1: best high-MI sample (lowest error), Sample 2: median, Sample 3: worst
    mse_per_sample_h = ((pred_h.numpy() - y_val.numpy()) ** 2).mean(axis=1)
    best_idx  = int(np.argmin(mse_per_sample_h))
    median_idx = int(np.median(np.argsort(mse_per_sample_h)))
    worst_idx = int(np.argmax(mse_per_sample_h))
    sample_indices = [best_idx, median_idx, worst_idx]
    sample_labels = ["Best (Lowest Error)", "Median", "Worst (Highest Error)"]

    # ── Figure 1: Three reconstruction curves overlay ────────────────────────
    t = np.arange(args.pred_len)

    for si, (idx, slabel) in enumerate(zip(sample_indices, sample_labels)):
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="white",
                                  gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(
            f"表征垄断测试 — {slabel}\n"
            f"Layer={args.target_layer} | High-MI: {results['high_mi']['n_patches']} patches | "
            f"Low-MI: {results['low_mi']['n_patches']} patches | pred_len={args.pred_len}",
            fontsize=12, fontweight="bold"
        )

        ax = axes[0]

        # Plot ground truth
        ax.plot(t, y_val[idx].numpy(), color="black", linewidth=2.5,
                 label="Ground Truth", zorder=5)

        # Plot high-MI prediction
        ax.plot(t, pred_h[idx].numpy(), color="steelblue", linewidth=1.8,
                 alpha=0.85, label=f"High-MI Probe (MSE={results['high_mi']['mse']:.4f}, R²={results['high_mi']['r2']:.3f})",
                 zorder=4)

        # Plot low-MI prediction
        ax.plot(t, pred_l[idx].numpy(), color="coral", linewidth=1.8,
                 alpha=0.85, label=f"Low-MI Probe  (MSE={results['low_mi']['mse']:.4f}, R²={results['low_mi']['r2']:.3f})",
                 zorder=3)

        # Plot all-patch baseline
        ax.plot(t, pred_a[idx].numpy(), color="gray", linewidth=1.5,
                 alpha=0.7, linestyle="--",
                 label=f"All-Patch Probe (MSE={results['all_patch']['mse']:.4f}, R²={results['all_patch']['r2']:.3f})",
                 zorder=2)

        ax.set_ylabel("Normalized Value", fontsize=11)
        ax.set_title("96-Step Future Forecast Comparison", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.pred_len - 1)

        # Per-step error (shaded)
        ax2 = axes[1]
        err_h = (pred_h[idx].numpy() - y_val[idx].numpy()) ** 2
        err_l = (pred_l[idx].numpy() - y_val[idx].numpy()) ** 2
        err_a = (pred_a[idx].numpy() - y_val[idx].numpy()) ** 2
        ax2.fill_between(t, 0, err_h, color="steelblue", alpha=0.3, label="High-MI Squared Error")
        ax2.fill_between(t, 0, err_l, color="coral", alpha=0.3, label="Low-MI Squared Error")
        ax2.plot(t, err_h, color="steelblue", linewidth=1, alpha=0.8)
        ax2.plot(t, err_l, color="coral", linewidth=1, alpha=0.8)
        ax2.plot(t, err_a, color="gray", linewidth=1, alpha=0.6, linestyle="--", label="All-Patch")
        ax2.set_ylabel("Squared Error", fontsize=10)
        ax2.set_xlabel("Forecast Step (t)", fontsize=11)
        ax2.set_xlim(0, args.pred_len - 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        suffix = ["best", "median", "worst"][si]
        path = os.path.join(output_dir, f"reconstruction_{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存: {path}")

    # ── Figure 2: Aggregate comparison bar chart ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    probe_types = ["High-MI\nProbe", "Low-MI\nProbe", "All-Patch\nBaseline"]
    mse_vals = [results["high_mi"]["mse"], results["low_mi"]["mse"], results["all_patch"]["mse"]]
    r2_vals  = [results["high_mi"]["r2"],  results["low_mi"]["r2"],  results["all_patch"]["r2"]]
    colors   = ["steelblue", "coral", "gray"]

    ax = axes[0]
    bars = ax.bar(probe_types, mse_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(y=0.8 * results["all_patch"]["mse"], color="green", linestyle="--",
               linewidth=1.5, label="80% Baseline")
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("MSE Comparison: Information Monopoly", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10)
    for bar, v in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mse_vals[2] * 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(probe_types, r2_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("R² Comparison: Prediction Quality", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, r2_vals):
        offset = 0.01 if v >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold")

    plt.suptitle(f"Layer {args.target_layer} — {dataset_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "monopoly_bar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

    # ── Figure 3: MI curve with high/low patch highlighting ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="white")

    ax = axes[0]
    patch_axis = np.arange(n_patches)
    ax.plot(patch_axis, mi_curve, color="black", linewidth=1.5,
            alpha=0.8, label="I(H, Y) per Patch")
    q_hi = results["high_mi"]["n_patches"]
    q_lo = results["low_mi"]["n_patches"]
    sorted_idx = np.argsort(mi_curve)
    high_idx = sorted_idx[-q_hi:]
    low_idx  = sorted_idx[:q_lo]

    ax.scatter(high_idx, mi_curve[high_idx], color="steelblue", s=60,
               zorder=5, label=f"High-MI Patches (n={q_hi})", marker="^", edgecolors="black")
    ax.scatter(low_idx, mi_curve[low_idx], color="coral", s=60,
               zorder=5, label=f"Low-MI Patches (n={q_lo})", marker="v", edgecolors="black")
    ax.axhline(mi_curve[high_idx].min(), color="steelblue", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axhline(mi_curve[low_idx].max(), color="coral", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_xlabel("Patch Index", fontsize=11)
    ax.set_ylabel("I(H, Y) — Mutual Information (bits)", fontsize=11)
    ax.set_title(f"Layer {args.target_layer}: MI Curve with High/Low Patch Selection", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: per-sample MSE distribution (histogram)
    ax = axes[1]
    mse_per_sample_h = ((pred_h.numpy() - y_val.numpy()) ** 2).mean(axis=1)
    mse_per_sample_l = ((pred_l.numpy() - y_val.numpy()) ** 2).mean(axis=1)
    mse_per_sample_a = ((pred_a.numpy() - y_val.numpy()) ** 2).mean(axis=1)

    bins = np.linspace(0, max(mse_per_sample_h.max(), mse_per_sample_l.max(),
                               mse_per_sample_a.max()) * 1.05, 40)
    ax.hist(mse_per_sample_h, bins=bins, color="steelblue", alpha=0.5,
            label=f"High-MI (mean={mse_per_sample_h.mean():.4f})")
    ax.hist(mse_per_sample_l, bins=bins, color="coral", alpha=0.5,
            label=f"Low-MI (mean={mse_per_sample_l.mean():.4f})")
    ax.hist(mse_per_sample_a, bins=bins, color="gray", alpha=0.4,
            label=f"All-Patch (mean={mse_per_sample_a.mean():.4f})")
    ax.set_xlabel("MSE per Sample", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Per-Sample MSE Distribution (Validation Set)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "mi_curve_and_mse_dist.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

    # ── Figure 4: Training convergence curves ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="white")
    curves_list = [curves["curve_h"], curves["curve_l"], curves["curve_a"]]
    labels = ["High-MI Probe", "Low-MI Probe", "All-Patch Probe"]
    colors_conv = ["steelblue", "coral", "gray"]

    for ax, c, label, col in zip(axes, curves_list, labels, colors_conv):
        ax.plot(c, color=col, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Val MSE", fontsize=10)
        ax.set_title(f"{label}\nFinal MSE: {c[-1]:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Probe Training Convergence — Layer {args.target_layer}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "probe_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")

    # ── Figure 5: Multi-panel summary figure ────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: three reconstruction curves (best, median, worst)
    for si, (idx, slabel) in enumerate(zip(sample_indices, sample_labels)):
        ax = fig.add_subplot(gs[0, si])
        ax.plot(t, y_val[idx].numpy(), color="black", linewidth=2, label="GT")
        ax.plot(t, pred_h[idx].numpy(), color="steelblue", linewidth=1.5,
                alpha=0.85, label="High-MI")
        ax.plot(t, pred_l[idx].numpy(), color="coral", linewidth=1.5,
                alpha=0.85, label="Low-MI")
        ax.plot(t, pred_a[idx].numpy(), color="gray", linewidth=1.2,
                alpha=0.7, linestyle="--", label="All")
        ax.set_title(f"{slabel}", fontsize=10, fontweight="bold")
        ax.set_xlabel("t", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Row 2: MSE bar chart + R² bar chart + MI curve
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(probe_types, mse_vals, color=colors, alpha=0.85)
    ax.axhline(y=0.8 * results["all_patch"]["mse"], color="green", linestyle="--",
               linewidth=1.5, label="80% Baseline")
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_title("MSE Comparison", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mse_vals[2] * 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    bars = ax.bar(probe_types, r2_vals, color=colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("R² Comparison", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, r2_vals):
        offset = 0.01 if v >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(patch_axis, mi_curve, color="black", linewidth=1.5, alpha=0.8)
    ax.scatter(high_idx, mi_curve[high_idx], color="steelblue", s=40,
               zorder=5, marker="^", edgecolors="black", label=f"High (n={q_hi})")
    ax.scatter(low_idx, mi_curve[low_idx], color="coral", s=40,
               zorder=5, marker="v", edgecolors="black", label=f"Low (n={q_lo})")
    ax.set_xlabel("Patch Index", fontsize=9)
    ax.set_ylabel("I(H,Y) (bits)", fontsize=9)
    ax.set_title("MI Curve — Layer Selection", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3: convergence curves + MSE distribution
    for ci, (c, label, col) in enumerate(zip(curves_list, labels, colors_conv)):
        ax = fig.add_subplot(gs[2, ci])
        ax.plot(c, color=col, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Val MSE", fontsize=9)
        ax.set_title(f"{label} Convergence", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"表征垄断测试 — Information Monopoly Probe\n"
        f"Layer={args.target_layer} | {dataset_name} | pred_len={args.pred_len} | "
        f"High: {results['high_mi']['n_patches']} patches → MSE={results['high_mi']['mse']:.4f} | "
        f"Low: {results['low_mi']['n_patches']} patches → MSE={results['low_mi']['mse']:.4f}",
        fontsize=12, fontweight="bold", y=1.01
    )

    path = os.path.join(output_dir, "monopoly_summary.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Multi-Layer Comparison Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multilayer_comparison(all_layer_results: list, output_dir: str,
                                 dataset_name: str = "ETTh1"):
    """
    绘制多层对比：如果对多个层都跑了垄断测试，画出每层的高 MI vs 低 MI MSE。
    """
    if len(all_layer_results) <= 1:
        return

    layers = [r["target_layer"] for r in all_layer_results]
    mse_h = [r["high_mi"]["mse"] for r in all_layer_results]
    mse_l = [r["low_mi"]["mse"] for r in all_layer_results]
    mse_a = [r["all_patch"]["mse"] for r in all_layer_results]
    r2_h  = [r["high_mi"]["r2"] for r in all_layer_results]
    r2_l  = [r["low_mi"]["r2"] for r in all_layer_results]
    r2_a  = [r["all_patch"]["r2"] for r in all_layer_results]

    x = np.arange(len(layers))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    ax = axes[0]
    ax.bar(x - w, mse_h, w, label="High-MI Probe", color="steelblue", alpha=0.85)
    ax.bar(x,     mse_l, w, label="Low-MI Probe",  color="coral",     alpha=0.85)
    ax.bar(x + w, mse_a, w, label="All-Patch",    color="gray",      alpha=0.85)
    ax.axhline(y=0.8 * np.mean(mse_a), color="green", linestyle="--",
               linewidth=1.5, label="80% Baseline")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Cross-Layer MSE: Information Monopoly", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - w, r2_h, w, label="High-MI Probe", color="steelblue", alpha=0.85)
    ax.bar(x,     r2_l, w, label="Low-MI Probe",  color="coral",     alpha=0.85)
    ax.bar(x + w, r2_a, w, label="All-Patch",    color="gray",      alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Cross-Layer R²: Information Monopoly", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Multi-Layer Information Monopoly — {dataset_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "multilayer_monopoly_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="表征垄断测试 — Linear Probing for Information Monopoly")
    parser.add_argument("--mi_result_dir", type=str,
                        default="./results/timer_mi_ksg_pca",
                        help="timer_mi_ksg_pca.py 输出目录（含 global_mi_peaks_*.json）")
    parser.add_argument("--root_path",   type=str, default="./datasets/")
    parser.add_argument("--data_path",   type=str, default="ETTh1.csv")
    parser.add_argument("--data_type",   type=str, default="ETTh1")
    parser.add_argument("--seq_len",     type=int, default=672)
    parser.add_argument("--pred_len",    type=int, default=96)
    parser.add_argument("--patch_len",   type=int, default=96)
    parser.add_argument("--stride",      type=int, default=96)
    parser.add_argument("--d_model",     type=int, default=1024)
    parser.add_argument("--d_ff",        type=int, default=2048)
    parser.add_argument("--e_layers",    type=int, default=8)
    parser.add_argument("--n_heads",     type=int, default=8)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--ckpt_path",  type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_lr",   type=float, default=1e-3)
    parser.add_argument("--high_mi_ratio", type=float, default=0.15,
                        help="高 MI patch 比例（默认 15%）")
    parser.add_argument("--low_mi_ratio", type=float, default=0.15,
                        help="低 MI patch 比例（默认 15%）")
    parser.add_argument("--target_layer", type=int, default=-1,
                        help="目标层（-1=最后一层，0=第一层，依次类推）")
    parser.add_argument("--layers", type=str, default=None,
                        help="多层模式，如 '0,1,2,3,7'；若设置则覆盖 --target_layer")
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--freq",       type=str, default="h")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out_dir",    type=str,
                        default="./outputs/timer_mi_monopoly_probe/")
    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine target layers
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(",")]
        layer_desc = f"multi-layer [{args.layers}]"
    elif args.target_layer == -1:
        target_layers = [args.e_layers - 1]
        layer_desc = f"last_layer={target_layers[0]}"
    else:
        target_layers = [args.target_layer]
        layer_desc = f"layer={target_layers[0]}"

    print("=" * 70)
    print("表征垄断测试 — Linear Probing for Information Monopoly")
    print("=" * 70)
    print(f"  mi_result_dir  : {args.mi_result_dir}")
    print(f"  ckpt_path      : {args.ckpt_path}")
    print(f"  data_path      : {args.data_path}")
    print(f"  data_type      : {args.data_type}")
    print(f"  seq_len        : {args.seq_len}")
    print(f"  pred_len       : {args.pred_len}")
    print(f"  patch_len      : {args.patch_len}")
    print(f"  e_layers       : {args.e_layers}")
    print(f"  target_layer   : {layer_desc}")
    print(f"  high_mi_ratio  : {args.high_mi_ratio}")
    print(f"  low_mi_ratio   : {args.low_mi_ratio}")
    print(f"  probe_epochs   : {args.probe_epochs}")
    print(f"  probe_lr       : {args.probe_lr}")
    print(f"  device         : {args.device}")
    print(f"  out_dir        : {output_dir}")
    print("=" * 70)

    # ── Phase 1: Load MI curves ──────────────────────────────────────────────
    print("\n>>> Phase 1: 从 JSON 加载 MI 曲线...")
    mi_curves, mi_json_data = load_mi_from_json(args.mi_result_dir, args.e_layers)
    n_patches_mi = len(mi_curves[0])
    print(f"  MI 曲线: {args.e_layers} 层, 每层 {n_patches_mi} patches")

    # ── Phase 2: Load dataset ─────────────────────────────────────────────────
    print("\n>>> Phase 2: 加载数据集...")
    test_dataset = CIDatasetBenchmark(
        root_path=os.path.join(args.root_path, args.data_path),
        flag="test",
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = test_dataset.n_var
    total_samples = len(test_dataset)
    print(f"  变量数: {n_vars}, 测试集总样本数: {total_samples}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # ── Phase 3: Load Timer model ─────────────────────────────────────────────
    print("\n>>> Phase 3: 加载 Timer 模型...")
    model = build_timer(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        stride=args.patch_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    model = model.to(args.device)
    model.eval()
    print(f"  模型加载完成，设备: {args.device}")

    # ── Phase 4: Extract layer tokens ─────────────────────────────────────────
    print("\n>>> Phase 4: 提取测试集每层 token 表示...")
    hist_tokens, future_tokens, x_patch_tokens, n_patches, n_future_patches = \
        extract_layer_tokens(model, test_loader, args.device, args.e_layers)

    N = hist_tokens[0].shape[0]
    D = hist_tokens[0].shape[2]
    print(f"  提取完成: N={N}, n_patches={n_patches}, D={D}, n_layers={len(hist_tokens)}")

    # ── Phase 5: Prepare ground truth ─────────────────────────────────────────
    print("\n>>> Phase 5: 构建 Ground Truth 目标...")
    y_flat_all = []
    for i in range(N):
        _, seq_y, _, _ = test_dataset[i]
        y_flat_all.append(np.array(seq_y, dtype=np.float32)[:, 0])
    y_flat = np.stack(y_flat_all, axis=0)
    print(f"  Ground Truth shape: {y_flat.shape}")

    # train/val split
    all_idx = np.arange(N)
    train_idx, val_idx = train_test_split(
        all_idx, train_size=0.8, random_state=args.seed)
    print(f"  训练集: {len(train_idx)}, 验证集: {len(val_idx)}")

    # ── Phase 6: Run monopoly probe per layer ──────────────────────────────────
    all_layer_results = []

    for li in target_layers:
        args.target_layer = li
        result, curves = run_monopoly_probe(
            hist_tokens=hist_tokens,
            future_tokens=future_tokens,
            y_flat=y_flat,
            mi_curves=mi_curves,
            n_patches=n_patches,
            train_idx=train_idx,
            val_idx=val_idx,
            args=args,
            output_dir=output_dir,
        )

        # ── Plotting for this layer ─────────────────────────────────────────
        plot_monopoly_results(
            results=result,
            curves=curves,
            output_dir=output_dir,
            n_patches=n_patches,
            args=args,
            dataset_name=args.data_type,
        )

        all_layer_results.append(result)

    # ── Phase 7: Multi-layer comparison ──────────────────────────────────────
    if len(all_layer_results) > 1:
        print("\n>>> Phase 7: 多层对比绘图...")
        plot_multilayer_comparison(all_layer_results, output_dir, args.data_type)

    # ── Phase 8: Save all results ─────────────────────────────────────────────
    print(f"\n>>> Phase 8: 保存结果...")
    final_results = {
        "layers": all_layer_results,
        "config": {
            "n_layers": args.e_layers,
            "n_patches": n_patches,
            "d_model": int(D),
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "high_mi_ratio": args.high_mi_ratio,
            "low_mi_ratio": args.low_mi_ratio,
            "target_layers": target_layers,
            "data_path": args.data_path,
            "data_type": args.data_type,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "mi_result_dir": args.mi_result_dir,
        }
    }

    results_path = os.path.join(output_dir, "monopoly_results.pt")
    torch.save(final_results, results_path)
    print(f"  保存: {results_path}")

    # JSON summary
    summary = {
        "dataset": args.data_type,
        "pred_len": args.pred_len,
        "layers": [
            {
                "layer": r["target_layer"],
                "high_mi": {"mse": r["high_mi"]["mse"], "r2": r["high_mi"]["r2"],
                            "n_patches": r["high_mi"]["n_patches"]},
                "low_mi":  {"mse": r["low_mi"]["mse"],  "r2": r["low_mi"]["r2"],
                            "n_patches": r["low_mi"]["n_patches"]},
                "all_patch": {"mse": r["all_patch"]["mse"], "r2": r["all_patch"]["r2"]},
            }
            for r in all_layer_results
        ]
    }
    summary_path = os.path.join(output_dir, "monopoly_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  保存: {summary_path}")

    # Config snapshot
    cfg_path = os.path.join(output_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  保存: {cfg_path}")

    print("\n" + "=" * 70)
    print("  表征垄断测试完成！")
    print(f"  结果目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
