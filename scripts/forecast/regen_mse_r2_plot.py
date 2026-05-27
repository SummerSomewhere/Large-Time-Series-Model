#!/usr/bin/env python3
"""
Regenerate the layer MSE and R^2 plot from existing linear probing results.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RESULTS_PATH = "./synthetic/linear_probing/run_20260423_235428/linear_probing_results.pt"
OUTPUT_PATH = "./synthetic/linear_probing/layer_mse_curve.png"


def main():
    d = torch.load(RESULTS_PATH, map_location="cpu", weights_only=False)

    layer_indices = sorted(d["layer_mse"].keys())
    n_layers = len(layer_indices)

    layer_mse = [d["layer_mse"][l] for l in layer_indices]
    layer_r2 = [d["layer_r2"][l] for l in layer_indices]
    layer_concept_mse = d["layer_concept_mse"]
    layer_concept_r2 = d["layer_concept_r2"]

    concepts = list(layer_concept_mse.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))

    # ── Left: MSE curves ──────────────────────────────────────────
    ax = axes[0]
    for c_idx, concept in enumerate(concepts):
        mses = layer_concept_mse.get(concept, [])
        if len(mses) == n_layers:
            ax.plot(layer_indices, mses, "o-", label=concept,
                    color=colors[c_idx], linewidth=1.5, markersize=5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Per-Concept MSE vs. Layer", fontsize=13)
    ax.set_xticks(layer_indices)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    # Y-axis tick step = 0.01
    ymax = float(ax.get_ylim()[1])
    n_ticks = int(round(ymax / 0.01)) + 1
    ax.set_yticks(np.round(np.linspace(0, n_ticks * 0.01, n_ticks + 1), 3))

    # ── Right: R^2 curves ───────────────────────────────────────
    ax = axes[1]
    for c_idx, concept in enumerate(concepts):
        r2s = layer_concept_r2.get(concept, [])
        if len(r2s) == n_layers:
            ax.plot(layer_indices, r2s, "o-", label=concept,
                    color=colors[c_idx], linewidth=1.5, markersize=5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(r"$R^2$", fontsize=12)
    ax.set_title(r"Per-Concept $R^2$ vs. Layer", fontsize=13)
    ax.set_xticks(layer_indices)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    # Y-axis tick step = 0.1, show negative values
    ymin_r2 = float(ax.get_ylim()[0])
    ymax_r2 = float(ax.get_ylim()[1])
    ymin_r2 = np.floor(ymin_r2 / 0.1) * 0.1
    ymax_r2 = np.ceil(ymax_r2 / 0.1) * 0.1
    n_ticks_r2 = int(round((ymax_r2 - ymin_r2) / 0.1)) + 1
    ax.set_ylim(ymin_r2, ymax_r2)
    ax.set_yticks(np.round(np.linspace(ymin_r2, ymax_r2, n_ticks_r2), 2))

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
