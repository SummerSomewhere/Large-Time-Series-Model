#!/usr/bin/env python3
"""
合成数据 Layer-wise Linear Probe。

对合成数据的 Timer hidden states 做 Ridge/MLP 探测：
  - 输入：每层的 Mean-Pooled 特征 Z [n, D]
  - 目标：三个 ground truth 标签
      trend     = 线性趋势斜率 a
      periodic  = 周期成分振幅 A
      noise_std = 噪声水平 σ

时序切分：前 60% train，后 40% test。
切勿使用 shuffle 切分（合成数据的标签有明确的物理含义）。
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import warnings

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_features_and_labels(pt_path: str, n_samples: int = 0):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    layer_hidden = {}
    for k, v in data["layer_hidden"].items():
        if n_samples > 0 and v.shape[0] > n_samples:
            v = v[:n_samples]
        layer_hidden[k] = v.numpy()
    labels = {}
    for k, v in data["labels"].items():
        if n_samples > 0 and v.shape[0] > n_samples:
            v = v[:n_samples]
        labels[k] = v.numpy()
    config = data.get("config", {})
    return layer_hidden, labels, config


def prepare_features(layer_hidden: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """按 layer index 排序特征，返回 {layer_X: [n, D]}"""
    features = {}
    for k, v in layer_hidden.items():
        idx = int(k.split("_")[1])
        features[idx] = np.array(v)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Ridge Probe
# ─────────────────────────────────────────────────────────────────────────────

def run_ridge_probing(
    features: Dict[int, np.ndarray],
    labels: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float = 1.0,
) -> Dict:
    layer_keys = sorted(features.keys())
    label_names = list(labels.keys())

    results = {
        "r2": {name: [] for name in label_names},
        "mse": {name: [] for name in label_names},
        "layer_keys": [f"layer_{k}" for k in layer_keys],
        "label_names": label_names,
    }

    for li in layer_keys:
        h = features[li]
        for name in label_names:
            y = labels[name]
            scaler_X = StandardScaler()
            h_tr = scaler_X.fit_transform(h[train_idx])
            h_te = scaler_X.transform(h[test_idx])

            scaler_y = StandardScaler()
            y_tr = scaler_y.fit_transform(y[train_idx].reshape(-1, 1)).ravel()

            ridge = Ridge(alpha=alpha)
            ridge.fit(h_tr, y_tr)
            y_pred = scaler_y.inverse_transform(ridge.predict(h_te).reshape(-1, 1)).ravel()

            r2 = r2_score(y[test_idx], y_pred)
            mse = mean_squared_error(y[test_idx], y_pred)
            results["r2"][name].append(r2)
            results["mse"][name].append(mse)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MLP Probe
# ─────────────────────────────────────────────────────────────────────────────

class MLPProbe(torch.nn.Module):
    def __init__(self, D: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = D
        for h in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev, h),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            ])
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_mlp_probing(
    features: Dict[int, np.ndarray],
    labels: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden_dims: List[int] = [256, 128],
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 256,
    dropout: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    layer_keys = sorted(features.keys())
    label_names = list(labels.keys())

    results = {
        "r2": {name: [] for name in label_names},
        "mse": {name: [] for name in label_names},
        "layer_keys": [f"layer_{k}" for k in layer_keys],
        "label_names": label_names,
    }

    dev = torch.device(device)
    scaler_X = StandardScaler()

    for li in layer_keys:
        h = features[li]
        h_all = scaler_X.fit_transform(h)
        X_tr = torch.from_numpy(h_all[train_idx]).float().to(dev)
        X_te = torch.from_numpy(h_all[test_idx]).float().to(dev)

        for name in label_names:
            y_all = labels[name]
            Y_tr = torch.from_numpy(y_all[train_idx]).float().to(dev)
            Y_te = torch.from_numpy(y_all[test_idx]).float().to(dev)

            model = MLPProbe(h.shape[1], hidden_dims, dropout).to(dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(X_tr, Y_tr)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            best_r2 = -1e9
            best_state = None
            patience = 15
            no_improve = 0

            for ep in range(epochs):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss_fn(model(xb), yb).backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred_te = model(X_te).cpu().numpy()
                    r2 = r2_score(Y_te.cpu().numpy(), pred_te)

                if r2 > best_r2:
                    best_r2 = r2
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred_te = model(X_te).cpu().numpy()
                y_te = Y_te.cpu().numpy()
                r2 = r2_score(y_te, pred_te)
                mse = mean_squared_error(y_te, pred_te)
            results["r2"][name].append(r2)
            results["mse"][name].append(mse)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_layerwise_r2(results: Dict, out_dir: str, prefix: str):
    layer_names = results["layer_keys"]
    label_names = results["label_names"]
    n_layers = len(layer_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for ax, metric_name, metric_key in zip(
        axes, ["R² Score", "MSE", "R² Heatmap"],
        ["r2", "mse", "r2"]
    ):
        if metric_key == "r2":
            for i, name in enumerate(label_names):
                ax.plot(range(n_layers), results["r2"][name], "o-",
                        color=colors[i % len(colors)], label=name, linewidth=2, markersize=5)
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel("R² Score")
            ax.set_title("Layer-wise Ridge Probe R²")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
        elif metric_key == "mse":
            for i, name in enumerate(label_names):
                mse_vals = results["mse"][name]
                ax.plot(range(n_layers), mse_vals, "o-",
                        color=colors[i % len(colors)], label=name, linewidth=2, markersize=5)
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel("MSE")
            ax.set_title("Layer-wise Ridge Probe MSE")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
        else:
            data = np.array([results["r2"][name] for name in label_names])
            im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.0)
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
            ax.set_yticks(range(len(label_names)))
            ax.set_yticklabels(label_names)
            ax.set_xlabel("Decoder Layer")
            ax.set_title("R² Heatmap")
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_layerwise_r2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")


def plot_label_distributions(labels: Dict[str, np.ndarray], out_dir: str):
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, labels.items()):
        ax.hist(arr, bins=40, alpha=0.7, edgecolor="black")
        ax.set_title(f"{name}: mean={arr.mean():.3f}, std={arr.std():.3f}")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "label_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 跨 Mode 对比可视化
# ─────────────────────────────────────────────────────────────────────────────

MODE_COLORS = {
    "trend":          "#e74c3c",  # 红
    "periodic":        "#3498db",  # 蓝
    "noise":           "#2ecc71",  # 绿
    "ar1":            "#f39c12",  # 橙
    "level_shift":     "#9b59b6",  # 紫
    "random_walk":     "#1abc9c",  # 青
    "spectral":        "#e67e22",  # 深橙
    "time_warp":       "#34495e",  # 深灰蓝
    "variance":        "#d35400",  # 橙红
    "trend_periodic":  "#27ae60",  # 深绿
    "all":             "#2c3e50",  # 深灰
}
MODE_LINESTYLE = {
    "trend":          "-",
    "periodic":        "--",
    "noise":           ":",
    "ar1":            "-.",
    "level_shift":     (0, (5, 2)),
    "random_walk":     (0, (3, 1, 1, 1)),
    "spectral":        (0, (1, 1)),
    "time_warp":       (0, (5, 1, 2, 1)),
    "variance":        (0, (3, 3)),
    "trend_periodic":  (0, (4, 1)),
    "all":             "-",
}
MODE_MARKER = {
    "trend":          "o",
    "periodic":        "s",
    "noise":           "^",
    "ar1":            "D",
    "level_shift":     "v",
    "random_walk":     "p",
    "spectral":        "h",
    "time_warp":       "*",
    "variance":        "X",
    "trend_periodic":  "P",
    "all":             "o",
}


def load_results_json(json_path: str) -> Dict:
    with open(json_path) as f:
        return json.load(f)


def plot_cross_mode_comparison(
    probe_base_dir: str,
    modes: List[str],
    probe_type: str,
    out_dir: str,
):
    """
    从各 mode 目录读取结果，画 R² 和 MSE 对比图。
    目录结构：probe_base_dir/mode_{m}/{probe_type}_results.json
    每个 mode 只有一个 label。
    """
    mode_data = {}
    for m in modes:
        json_path = os.path.join(probe_base_dir, f"mode_{m}", f"{probe_type}_results.json")
        if not os.path.exists(json_path):
            print(f"[WARN] 跳过 {m}：文件不存在 {json_path}")
            continue
        data = load_results_json(json_path)
        label_names = [ln for ln in data["label_names"] if ln != "ground_truth"]
        if not label_names:
            print(f"[WARN] 跳过 {m}：没有有效 label")
            continue
        label = label_names[0]
        mode_data[m] = {
            "r2": data["r2"].get(label, []),
            "mse": data["mse"].get(label, []),
            "label": label,
        }

    if not mode_data:
        print("[WARN] 没有找到任何 mode 结果，跳过跨 mode 对比图")
        return

    # 从第一个有效 mode 推断层数
    n_layers = None
    for m in modes:
        if m in mode_data and mode_data[m]["r2"]:
            n_layers = len(mode_data[m]["r2"])
            break
    if n_layers is None:
        return

    layers = list(range(n_layers))
    layer_labels = [f"L{i}" for i in layers]

    # ── 图 1: R² vs Layer ────────────────────────────────────────────────
    fig_r2, ax_r2 = plt.subplots(figsize=(11, 6))
    for m in modes:
        if m not in mode_data or not mode_data[m]["r2"]:
            continue
        ax_r2.plot(
            layers, mode_data[m]["r2"],
            linestyle=MODE_LINESTYLE.get(m, "-"),
            color=MODE_COLORS.get(m, "gray"),
            marker=MODE_MARKER.get(m, "o"),
            markersize=6, linewidth=2,
            label=f"{m} ({mode_data[m]['label']})",
        )
    ax_r2.set_xlabel("Decoder Layer", fontsize=13)
    ax_r2.set_ylabel("R² Score", fontsize=13)
    ax_r2.set_title(f"{probe_type.upper()} Probe — R² vs Layer (11 Data Modes)", fontsize=14)
    ax_r2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="Data Mode")
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_xticks(layers)
    ax_r2.set_xticklabels(layer_labels)
    ax_r2.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    r2_path = os.path.join(out_dir, f"{probe_type}_cross_mode_r2.png")
    fig_r2.savefig(r2_path, dpi=150, bbox_inches="tight")
    plt.close(fig_r2)
    print(f"[Plot] R² 对比图: {r2_path}")

    # ── 图 2: MSE vs Layer ──────────────────────────────────────────────
    fig_mse, ax_mse = plt.subplots(figsize=(11, 6))
    for m in modes:
        if m not in mode_data or not mode_data[m]["mse"]:
            continue
        ax_mse.plot(
            layers, mode_data[m]["mse"],
            linestyle=MODE_LINESTYLE.get(m, "-"),
            color=MODE_COLORS.get(m, "gray"),
            marker=MODE_MARKER.get(m, "o"),
            markersize=6, linewidth=2,
            label=f"{m} ({mode_data[m]['label']})",
        )
    ax_mse.set_xlabel("Decoder Layer", fontsize=13)
    ax_mse.set_ylabel("MSE", fontsize=13)
    ax_mse.set_title(f"{probe_type.upper()} Probe — MSE vs Layer (11 Data Modes)", fontsize=14)
    ax_mse.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="Data Mode")
    ax_mse.grid(True, alpha=0.3)
    ax_mse.set_xticks(layers)
    ax_mse.set_xticklabels(layer_labels)
    plt.tight_layout()
    mse_path = os.path.join(out_dir, f"{probe_type}_cross_mode_mse.png")
    fig_mse.savefig(mse_path, dpi=150, bbox_inches="tight")
    plt.close(fig_mse)
    print(f"[Plot] MSE 对比图: {mse_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="合成数据 Layer-wise Probe")
    parser.add_argument("--features_pt_path", type=str, default=None,
                        help="由 synthetic_layerwise_probe_features.py 生成的 .pt 文件（compare 模式下不需要）")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="使用的样本数（0=全部）")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--probe_type", type=str, default="both",
                        choices=["ridge", "mlp", "both"])
    parser.add_argument("--mlp_hidden_dims", type=str, default="256,128")
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="训练集比例（默认 0.6，后 0.4 为测试集）")
    parser.add_argument("--out_dir", type=str,
                        default="./results/synthetic_layerwise_probe")
    parser.add_argument("--compare_modes", action="store_true",
                        help="跨 mode 对比模式：从各 mode 子目录读取结果并画对比图")
    parser.add_argument("--modes", type=str,
                        default="trend,periodic,noise,ar1,level_shift,random_walk,spectral,time_warp,variance,trend_periodic,all",
                        help="--compare_modes 时指定 mode 列表，逗号分隔")
    parser.add_argument("--compare_probe_type", type=str, default="both",
                        choices=["ridge", "mlp", "both"],
                        help="--compare_modes 时指定生成哪种 probe 的对比图")
    args = parser.parse_args()

    # ── 跨 Mode 对比模式 ─────────────────────────────────────────────
    if args.compare_modes:
        modes = [m.strip() for m in args.modes.split(",")]
        probe_types = []
        if args.compare_probe_type in ("ridge", "both"):
            probe_types.append("ridge")
        if args.compare_probe_type in ("mlp", "both"):
            probe_types.append("mlp")
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("  跨 Mode 对比可视化")
        print(f"{'='*60}")
        print(f"  Probe Base: {args.out_dir}")
        print(f"  Modes:     {modes}")
        print(f"  Probe Types:{probe_types}")
        for pt in probe_types:
            plot_cross_mode_comparison(
                probe_base_dir=args.out_dir,
                modes=modes,
                probe_type=pt,
                out_dir=args.out_dir,
            )
        print(f"\n✅ 对比图完成: {args.out_dir}")
        return

    if args.features_pt_path is None:
        raise ValueError("--features_pt_path 是必填参数（使用 --compare_sources 除外）")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. 加载 ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  合成数据 Layer-wise Probe")
    print(f"{'='*60}")
    print(f"\n[1] 加载特征: {args.features_pt_path}")
    layer_hidden, labels, config = load_features_and_labels(
        args.features_pt_path, n_samples=args.n_samples
    )
    n_total = next(iter(layer_hidden.values())).shape[0]
    D = next(iter(layer_hidden.values())).shape[-1]
    print(f"  n_samples={n_total}, feat_dim={D}, layers={len(layer_hidden)}")
    print(f"  标签: {list(labels.keys())}")

    # ── 2. 整理特征 ──────────────────────────────────────────────────────
    features = prepare_features(layer_hidden)

    # ── 3. 时序切分（60/40，不 shuffle）──────────────────────────────────
    train_end = int(n_total * args.train_ratio)
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(train_end, n_total)
    print(f"\n[2] 时序切分: train={len(train_idx)}, test={len(test_idx)} "
          f"(ratio={args.train_ratio})")

    # ── 4. 标签分布可视化 ────────────────────────────────────────────────
    print(f"\n[3] 标签分布:")
    for name, arr in labels.items():
        print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")
    plot_label_distributions(labels, args.out_dir)

    # ── 5. Ridge Probe ──────────────────────────────────────────────────
    ridge_results = None
    if args.probe_type in ("ridge", "both"):
        print(f"\n[4] Ridge Probe (alpha={args.alpha})...")
        ridge_results = run_ridge_probing(
            features, labels, train_idx, test_idx, alpha=args.alpha
        )
        print(f"  Ridge 结果:")
        for li, lname in enumerate(ridge_results["layer_keys"]):
            vals = {name: ridge_results["r2"][name][li] for name in ridge_results["label_names"]}
            best = max(vals, key=vals.get)
            print(f"  {lname}: R² = " + ", ".join(f"{name}={v:.4f}" for name, v in vals.items())
                  + f"  ← best: {best}")

        plot_layerwise_r2(ridge_results, args.out_dir, "ridge")
        _save_results(ridge_results, os.path.join(args.out_dir, "ridge_results.json"))

    # ── 6. MLP Probe ────────────────────────────────────────────────────
    mlp_results = None
    if args.probe_type in ("mlp", "both"):
        hidden_dims = [int(x) for x in args.mlp_hidden_dims.split(",") if x.strip()]
        print(f"\n[5] MLP Probe (hidden={hidden_dims}, epochs={args.mlp_epochs})...")
        mlp_results = run_mlp_probing(
            features, labels, train_idx, test_idx,
            hidden_dims=hidden_dims,
            lr=args.mlp_lr,
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            dropout=args.mlp_dropout,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"  MLP 结果:")
        for li, lname in enumerate(mlp_results["layer_keys"]):
            vals = {name: mlp_results["r2"][name][li] for name in mlp_results["label_names"]}
            best = max(vals, key=vals.get)
            print(f"  {lname}: R² = " + ", ".join(f"{name}={v:.4f}" for name, v in vals.items())
                  + f"  ← best: {best}")

        plot_layerwise_r2(mlp_results, args.out_dir, "mlp")
        _save_results(mlp_results, os.path.join(args.out_dir, "mlp_results.json"))

    # ── 7. Ridge vs MLP 对比 ────────────────────────────────────────────
    if ridge_results and mlp_results:
        print(f"\n[6] Ridge vs MLP 对比 (R²):")
        for name in ridge_results["label_names"]:
            r2_r = ridge_results["r2"][name]
            r2_m = mlp_results["r2"][name]
            print(f"  {name}:")
            for li, lname in enumerate(ridge_results["layer_keys"]):
                diff = r2_m[li] - r2_r[li]
                winner = "MLP" if diff > 0 else "Ridge"
                print(f"    {lname}: Ridge={r2_r[li]:.4f}, MLP={r2_m[li]:.4f}  "
                      f"({winner} +{abs(diff):.4f})")

    # ── 8. 解读 ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  解读")
    print(f"{'='*60}")
    if ridge_results:
        best_layer_per_label = {}
        for name in ridge_results["label_names"]:
            vals = ridge_results["r2"][name]
            best_idx = int(np.argmax(vals))
            best_layer_per_label[name] = (best_idx, vals[best_idx])
        print(f"\n  各标签最佳层（Ridge R²）:")
        for name, (li, r2) in best_layer_per_label.items():
            trend = "上升" if all(ridge_results["r2"][name][i] < ridge_results["r2"][name][i+1]
                                   for i in range(len(ridge_results["r2"][name])-1)) else "非单调"
            print(f"  {name:12s}: layer_{li} (R²={r2:.4f}) — {trend}")

    print(f"\n✅ 完成！结果保存在: {args.out_dir}")


def _save_results(results: Dict, path: str):
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: _to_list(vv) for kk, vv in v.items()}
        else:
            serializable[k] = _to_list(v)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Save] {path}")


def _to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, list):
        return [_to_list(x) for x in obj]
    return obj


if __name__ == "__main__":
    main()
