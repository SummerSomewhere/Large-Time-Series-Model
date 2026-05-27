#!/usr/bin/env python3
"""
合成数据 Layer-wise Probe 多 source 对比可视化。

读取 4 种 source (all/trend/periodic/noise) 的 probe 结果，
在两张子图上分别绘制 R² 和 MSE 随层数变化的曲线。

用法:
  python experiments/synthetic_probe_compare_sources.py \
      --probe_dir results/synthetic_layerwise_probe \
      --n_samples 10000 \
      --freq_hz 0.042 \
      --probe_type ridge \
      --out_dir results/synthetic_layerwise_probe/comparison
"""

import argparse
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCES = ["all", "trend", "periodic", "noise"]

SOURCE_COLORS = {
    "all":      "#2c3e50",
    "trend":    "#e74c3c",
    "periodic": "#27ae60",
    "noise":    "#8e44ad",
}

SOURCE_LINESTYLE = {
    "all":      "-",
    "trend":    "--",
    "periodic": "-.",
    "noise":    ":",
}

SOURCE_LABELS = {
    "all":      "all (all components vary)",
    "trend":    "trend varies (A=1.0, σ=0.175)",
    "periodic": "periodic varies (a=0, σ=0.175)",
    "noise":    "noise varies (a=0, A=1.0)",
}

LABEL_TEX = {
    "trend":     r"$\mathrm{trend}$",
    "periodic":  r"$\mathrm{periodic}$",
    "noise_std": r"$\mathrm{noise\_std}$",
}


def load_results(probe_dir: str, source: str, probe_type: str) -> dict:
    """加载单个 source 的 ridge/mlp results JSON。"""
    path = os.path.join(probe_dir, f"source_{source}", f"{probe_type}_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"结果文件不存在: {path}")
    with open(path) as f:
        return json.load(f)


def load_all_sources(probe_dir: str, probe_type: str):
    """返回 {source: (results_dict, label_names)}。"""
    all_results = {}
    for src in SOURCES:
        path = os.path.join(probe_dir, f"source_{src}", f"{probe_type}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            all_results[src] = results
        else:
            print(f"[Warning] 跳过 {src}，文件不存在: {path}")
    return all_results


def make_comparison_plot(
    all_results: dict,
    probe_type: str,
    out_dir: str,
    n_samples: int,
    freq_hz: float,
    rows: int = 1,
    cols: int = 3,
):
    """
    画 2 张大图，每张按 rows×cols 排列子图（每个子图对应一个 label）。
      - 图1: R² vs layer (越高越好)
      - 图2: MSE vs layer (越低越好)
    """
    os.makedirs(out_dir, exist_ok=True)

    label_names = ["trend", "periodic", "noise_std"]
    n_layers = len(next(iter(all_results.values()))["layer_keys"])

    # ── 图1: R² ────────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if cols == 1 and rows == 1:
        axes1 = np.array([axes1])
    axes1 = axes1.flatten()

    for col, label in enumerate(label_names):
        ax = axes1[col]
        for src in SOURCES:
            if src not in all_results:
                continue
            r2_vals = all_results[src]["r2"].get(label, [])
            if not r2_vals:
                continue
            is_active = (src == "all" or
                         (src == "trend" and label == "trend") or
                         (src == "periodic" and label == "periodic") or
                         (src == "noise" and label == "noise_std"))

            ax.plot(
                range(n_layers), r2_vals,
                color=SOURCE_COLORS[src],
                linestyle=SOURCE_LINESTYLE[src],
                linewidth=2 if is_active else 1.2,
                alpha=1.0 if is_active else 0.35,
                marker="o",
                markersize=4,
                label=SOURCE_LABELS[src] if is_active else None,
            )

        ax.set_xlabel("Decoder Layer", fontsize=10)
        ax.set_ylabel(r"$R^2$ Score", fontsize=10)
        ax.set_title(f"{LABEL_TEX.get(label, label)}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    fig1.suptitle(
        f"Layer-wise {probe_type.capitalize()} Probe — $R^2$ Score\n"
        f"(n={n_samples}, freq_hz={freq_hz})",
        fontsize=13, fontweight="bold",
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    path1 = os.path.join(out_dir, f"{probe_type}_r2_comparison.png")
    fig1.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Saved] {path1}")

    # ── 图2: MSE ────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if cols == 1 and rows == 1:
        axes2 = np.array([axes2])
    axes2 = axes2.flatten()

    for col, label in enumerate(label_names):
        ax = axes2[col]
        for src in SOURCES:
            if src not in all_results:
                continue
            mse_vals = all_results[src]["mse"].get(label, [])
            if not mse_vals:
                continue
            is_active = (src == "all" or
                         (src == "trend" and label == "trend") or
                         (src == "periodic" and label == "periodic") or
                         (src == "noise" and label == "noise_std"))

            ax.plot(
                range(n_layers), mse_vals,
                color=SOURCE_COLORS[src],
                linestyle=SOURCE_LINESTYLE[src],
                linewidth=2 if is_active else 1.2,
                alpha=1.0 if is_active else 0.35,
                marker="o",
                markersize=4,
                label=SOURCE_LABELS[src] if is_active else None,
            )

        ax.set_xlabel("Decoder Layer", fontsize=10)
        ax.set_ylabel("MSE", fontsize=10)
        ax.set_title(f"{LABEL_TEX.get(label, label)}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    fig2.suptitle(
        f"Layer-wise {probe_type.capitalize()} Probe — MSE\n"
        f"(n={n_samples}, freq_hz={freq_hz})",
        fontsize=13, fontweight="bold",
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = os.path.join(out_dir, f"{probe_type}_mse_comparison.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Saved] {path2}")


def make_summary_table(all_results: dict, probe_type: str, out_dir: str):
    """打印并保存每层各 source 的 R² 摘要表。"""
    label_names = ["trend", "periodic", "noise_std"]
    layer_keys = next(iter(all_results.values()))["layer_keys"]

    lines = []
    lines.append(f"# {probe_type.upper()} Probe — R² Summary by Layer")
    lines.append(f"# {'Layer':<10} " + "  ".join(f"{src:>12}" for src in SOURCES))
    lines.append("-" * (10 + 14 * len(SOURCES)))

    for li, lname in enumerate(layer_keys):
        row = f"{lname:<10}"
        for src in SOURCES:
            if src not in all_results:
                row += f"{'N/A':>12}"
                continue
            # 四个 source 中该层的 R² 均值
            vals = [all_results[src]["r2"].get(lbl, [float("nan")])[li]
                    for lbl in label_names]
            mean_val = np.nanmean(vals)
            row += f"{mean_val:>12.4f}"
        lines.append(row)

    text = "\n".join(lines) + "\n"
    path = os.path.join(out_dir, f"{probe_type}_r2_summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"[Saved] {path}")
    print(text)


def main():
    parser = argparse.ArgumentParser(
        description="合成数据 Layer-wise Probe 多 source 对比图"
    )
    parser.add_argument(
        "--probe_dir", type=str,
        default="results/synthetic_layerwise_probe",
        help="包含 source_*/ 子目录的 probe 结果根目录",
    )
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--freq_hz", type=float, default=0.042)
    parser.add_argument(
        "--probe_type", type=str, default="ridge",
        choices=["ridge", "mlp", "both"],
        help="绘制哪种 probe 结果（ridge / mlp / both）",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="输出目录（默认 probe_dir/comparison）",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.probe_dir, "comparison")

    print(f"\n{'='*60}")
    print("  Layer-wise Probe 多 Source 对比")
    print(f"{'='*60}")
    print(f"  probe_dir:   {args.probe_dir}")
    print(f"  n_samples:  {args.n_samples}")
    print(f"  freq_hz:     {args.freq_hz}")
    print(f"  probe_type:  {args.probe_type}")
    print(f"  out_dir:     {out_dir}")
    print(f"  sources:     {SOURCES}")
    print()

    for ptype in (["ridge", "mlp"] if args.probe_type == "both" else [args.probe_type]):
        print(f"\n── {ptype.upper()} ──")
        all_results = load_all_sources(args.probe_dir, ptype)
        if not all_results:
            print(f"[Skip] {ptype}: 无可用结果")
            continue

        make_comparison_plot(
            all_results, ptype, out_dir,
            n_samples=args.n_samples, freq_hz=args.freq_hz,
        )
        make_summary_table(all_results, ptype, out_dir)

    print(f"\n✅ 完成！图片保存在: {out_dir}")


if __name__ == "__main__":
    main()
