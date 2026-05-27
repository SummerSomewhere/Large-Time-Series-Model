#!/usr/bin/env python3
"""
Timer MI Validation via ROCKET Features & Change-Point Alignment

Validates that high-MI tokens identified by the KSG-PCA MI estimator
have genuinely richer structural information, using two aeon methods:

  1. ROCKET Feature Discriminability
     - Extract per-patch hidden states from Timer's decoder layers
     - Partition patches into high-MI and low-MI groups
     - Apply aeon/MiniRocket features to the raw patch subsequences
     - Measure group separability: Fisher's discriminant ratio on features
       and k-NN classification accuracy

  2. Change-Point Alignment
     - Run ClaSPSegmenter on each full test sequence
     - Align detected change points with MI score profiles
     - Quantify whether MI spikes co-locate with regime boundaries

Output: Nature-style multi-panel figures + statistical summary tables

Usage:
    python experiments/timer_mi_rocket_validation.py \
        --mi_dir ./outputs/timer_mi_ksg_pca/Timer_MI_20260516_133108/ \
        --data_path ./datasets/ETTh1.csv \
        --seq_len 672 --pred_len 96 \
        --output_dir ./results/timer_mi_rocket_validation/

Dependencies:
    pip install aeon scikit-learn umap-learn
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_loader_benchmark import CIDatasetBenchmark

# ─────────────────────────────────────────────────────────────────────────────
# Config & Model
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer_model(ckpt_path: str, patch_len: int, stride: int,
                      d_model: int, d_ff: int, e_layers: int,
                      n_heads: int, dropout: float,
                      seq_len: int, pred_len: int):
    config = Config(
        task_name='forecast', ckpt_path=ckpt_path,
        patch_len=patch_len, stride=stride,
        d_model=d_model, d_ff=d_ff, e_layers=e_layers,
        n_heads=n_heads, dropout=dropout,
        output_attention=False, distil=True, use_revin=False,
        seq_len=seq_len, pred_len=pred_len,
        d_layers=1, factor=1, enc_in=1, dec_in=1, c_out=1,
        activation='gelu', use_multi_gpu=False,
        use_gpu=torch.cuda.is_available(), devices='0',
        num_workers=4, freq='h', data='custom',
        embed='timeF', target='OT', features='M',
        des='Exp', lradj='type1', use_amp=False,
        is_finetuning=0, inverse=False,
        use_align_loss=False,
        align_loss_layers=list(range(e_layers)),
        label_len=pred_len, output_len=pred_len,
        batch_size=64, train_epochs=1, patience=3,
        learning_rate=3e-5, itr=1, use_ims=False,
    )
    from models.Timer import Model
    model = Model(config)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ─────────────────────────────────────────────────────────────────────────────
# Hidden State Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_patch_hidden_states(model, test_loader, device, n_layers):
    """
    Extract per-layer, per-patch hidden states from the Timer decoder.

    Returns:
        all_hidden: list of n_layers arrays, each [n_samples, n_patches, d_model]
        all_raw_patches: [n_samples, n_patches, patch_len] raw subsequences
    """
    core = _unwrap(model)
    patch_len = core.patch_len

    hidden_states_per_layer = [[] for _ in range(n_layers)]
    raw_patches = []
    all_raw_ts = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
                test_loader, desc="Extract hidden states", leave=False):

            B = batch_x.shape[0]

            seq_x = batch_x.float().to(device)
            bx_mark = (batch_x_mark.float().to(device)
                       if batch_x_mark is not None else None)

            means = seq_x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_norm = (seq_x - means) / stdev

            x2 = x_norm.permute(0, 2, 1)
            dec_in, n_vars = core.enc_embedding(x2)

            BM, N, D = dec_in.shape
            B_times_vars = B * n_vars
            dec_in = dec_in.view(B_times_vars, N, D)

            from utils.masking import TriangularCausalMask
            causal_mask = TriangularCausalMask(B_times_vars, N, device=device)

            _, _, logits_list, layer_hidden = core.decoder(
                dec_in, attn_mask=causal_mask, output_hidden_states=True)

            # layer_hidden: list of (e_layers + 1) tensors, each [BM, N, D]
            # We only want the first e_layers outputs (exclude final norm)
            for li in range(n_layers):
                hs = layer_hidden[li].cpu().numpy()
                hidden_states_per_layer[li].append(hs)

            # Raw patches: dec_in is already patched [BM, N, patch_len]
            dec_in_np = dec_in.cpu().numpy()
            for i in range(B):
                for v in range(n_vars):
                    idx = i * n_vars + v
                    raw_patches.append(dec_in_np[idx])  # [N, patch_len]
                    all_raw_ts.append(seq_x[i, 0].cpu().numpy())  # [seq_len]

    # Concatenate: each layer list -> [n_samples * n_vars * n_patches, N, D]
    all_hidden = []
    for li in range(n_layers):
        arr = np.concatenate(hidden_states_per_layer[li], axis=0)
        n_samples_total = arr.shape[0]
        n_patches = arr.shape[1]
        all_hidden.append(arr)  # [N_samples, N_patches, D]

    raw_patches = np.array(raw_patches)  # [N_samples, N_patches, patch_len]
    all_raw_ts = np.array(all_raw_ts)   # [N_samples, seq_len]

    return all_hidden, raw_patches, all_raw_ts


# ─────────────────────────────────────────────────────────────────────────────
# ROCKET Feature Validation
# ─────────────────────────────────────────────────────────────────────────────

def _nature_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _import_rocket_transformer():
    """
    Import the best available ROCKET transformer with version-robust fallback.
    Tries multiple naming conventions across aeon versions.
    Returns the transformer class and a boolean is_fitted_api (whether fit_transform
    returns a 3D array that needs reshaping after).
    """
    candidates = [
        # aeon >= 1.4.0: class is named 'MiniRocket'
        ("aeon.transformations.collection.convolution_based", "MiniRocket"),
        ("aeon.transformations.collection.convolution_based", "Rocket"),
        # aeon < 1.0 / early naming
        ("aeon.transformations.collection.convolution_based", "MiniRocketTransformer"),
        ("aeon.transformations.collection.convolution_based", "RocketTransformer"),
        # Classifier wrappers as last resort (they have .transform too)
        ("aeon.classification.convolution_based", "MiniRocketClassifier"),
        ("aeon.classification.convolution_based", "RocketClassifier"),
    ]
    for module_name, cls_name in candidates:
        try:
            module = __import__(module_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            # Check it has fit_transform
            if hasattr(cls, 'fit_transform'):
                is_classifier = 'Classifier' in cls_name
                return cls, not is_classifier  # classifier returns 2D, transformer returns 3D
        except (ImportError, AttributeError):
            continue
    return None, None


def compute_rocket_features(patch_sequences, n_kernels=10000, random_state=42):
    """
    Compute MiniRocket features for a collection of patch subsequences.

    patch_sequences: [n_samples, n_patches, patch_len]
    Returns: [n_samples * n_patches, n_features]

    Version-robust: tries aeon MiniRocket/Rocket (various naming conventions),
    then falls back to a lightweight statistical fingerprint if aeon is unavailable.
    """
    RocketCls, needs_3d_input = _import_rocket_transformer()

    if RocketCls is None:
        warnings.warn(
            "aeon ROCKET not available. Using statistical fingerprint fallback."
        )
        return _statistical_fingerprint(patch_sequences)

    N, Np, L = patch_sequences.shape
    X_2d = patch_sequences.reshape(-1, L)

    if needs_3d_input:
        X_input = X_2d[:, np.newaxis, :]
    else:
        X_input = X_2d

    rocket = RocketCls(n_jobs=-1, random_state=random_state)
    # Set kernel count (varies by version)
    for param in ['n_kernels', 'num_kernels', 'n_estimators']:
        if hasattr(rocket, param):
            setattr(rocket, param, n_kernels)
            break

    return rocket.fit_transform(X_input)


def _statistical_fingerprint(patch_sequences):
    """
    Lightweight fallback when aeon is not available.
    Extracts 16 statistical features per patch subsequence.
    """
    N, Np, L = patch_sequences.shape
    X = patch_sequences.reshape(-1, L)  # [N_total, L]
    n_total = X.shape[0]
    feats = []
    for row in X:
        r = np.asarray(row, dtype=np.float64)
        f = [
            np.mean(r),
            np.std(r),
            np.min(r),
            np.max(r),
            np.median(r),
            np.percentile(r, 25),
            np.percentile(r, 75),
            np.percentile(r, 90),
            np.percentile(r, 10),
            np.mean(np.abs(r - np.mean(r))),       # MAD
            np.sum(np.diff(r) ** 2) / (len(r) - 1), # second-diff variance
            r[-1] - r[0],                          # range trend
            np.mean(np.sign(np.diff(r)) != 0),       # zero-crossing rate of diff
            np.argmax(r) / len(r),                  # argmax position
            np.argmax(np.abs(np.fft.rfft(r)[1:])) / (len(r) // 2),  # dominant freq pos
            np.abs(np.fft.rfft(r)[1:]).mean(),       # spectral energy
        ]
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def fisher_discriminant_ratio(X_high, X_low):
    """
    Compute Fisher's linear discriminant ratio for two groups.
    Higher = better separation.
    """
    X_h = np.asarray(X_high)
    X_l = np.asarray(X_low)

    if X_h.shape[1] == 0 or X_l.shape[1] == 0:
        return np.nan

    mean_h = X_h.mean(axis=0)
    mean_l = X_l.mean(axis=0)
    var_h = X_h.var(axis=0) + 1e-8
    var_l = X_l.var(axis=0) + 1e-8

    fdr_per_feat = ((mean_h - mean_l) ** 2) / (var_h + var_l)
    return float(np.mean(fdr_per_feat))


def knn_classification_accuracy(X_high, X_low, n_neighbors=5):
    """
    k-NN classification between high-MI and low-MI groups.
    Returns accuracy (0.5 = random).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X_h = np.asarray(X_high)
    X_l = np.asarray(X_low)

    if X_h.shape[0] < n_neighbors or X_l.shape[0] < n_neighbors:
        return np.nan

    n_h, n_f = X_h.shape
    n_l = X_l.shape[0]

    X = np.vstack([X_h, X_l])
    y = np.array([1] * n_h + [0] * n_l)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = KNeighborsClassifier(n_neighbors=min(n_neighbors, min(n_h, n_l)))
    scores = cross_val_score(clf, X_scaled, y, cv=min(5, min(n_h, n_l)), scoring='accuracy')
    return float(scores.mean())


def run_rocket_validation(hidden_states, raw_patches, mi_matrix,
                          n_layers, mi_threshold_percentile=50,
                          n_kernels=10000, random_state=42):
    """
    Full ROCKET validation pipeline.

    Returns a dict of per-layer statistics:
        {layer_idx: {'fdr': float, 'knn_acc': float, 'n_high': int, 'n_low': int}}
    """
    from sklearn.decomposition import PCA

    results = {}
    # Use the last decoder layer hidden states (semantic level)
    hs_last = hidden_states[-1]  # [n_samples, n_patches, D]
    mi_last = mi_matrix[-1]       # [n_patches,]

    n_samples, n_patches, D = hs_last.shape
    N_total = n_samples * n_patches

    # Flatten for grouping
    flat_hs = hs_last.reshape(N_total, D)
    flat_patches = raw_patches.reshape(N_total, raw_patches.shape[-1])
    sample_idx = np.arange(N_total) // n_patches

    # Per-patch MI: broadcast across samples
    flat_mi = np.tile(mi_last, n_samples)  # [N_total,]

    threshold = np.percentile(flat_mi, 100 - mi_threshold_percentile)
    high_mask = flat_mi >= threshold
    low_mask = flat_mi < np.percentile(flat_mi, mi_threshold_percentile)

    print(f"\n  ROCKET Validation (top {mi_threshold_percentile}% MI)")
    print(f"  High-MI patches: {high_mask.sum()}, Low-MI patches: {low_mask.sum()}")

    # Compute MiniRocket features for raw patch subsequences
    print("  Computing MiniRocket features...")
    t0 = time.time()
    rocket_feats = compute_rocket_features(
        raw_patches, n_kernels=n_kernels, random_state=random_state)
    print(f"  ROCKET features: {rocket_feats.shape} in {time.time()-t0:.1f}s")

    # FDR on ROCKET features
    fdr = fisher_discriminant_ratio(rocket_feats[high_mask], rocket_feats[low_mask])
    print(f"  Fisher FDR (ROCKET feats): {fdr:.4f}")

    # k-NN accuracy on ROCKET features
    knn_acc = knn_classification_accuracy(
        rocket_feats[high_mask], rocket_feats[low_mask])
    print(f"  k-NN accuracy (ROCKET feats): {knn_acc:.4f}")

    # Also compute on hidden states (sanity: are hidden states separable?)
    hs_scaler = StandardScaler()
    hs_scaled = hs_scaler.fit_transform(flat_hs)
    hs_knn = KNeighborsClassifier(n_neighbors=5)
    hs_scores = cross_val_score(hs_knn, hs_scaled, flat_mi > threshold, cv=5, scoring='accuracy')
    hs_knn_acc = float(hs_scores.mean())
    print(f"  k-NN accuracy (hidden states): {hs_knn_acc:.4f}")

    # FDR per layer (hidden states)
    print(f"\n  Per-layer FDR (hidden states):")
    layer_fdr = {}
    for li in range(n_layers):
        hs_li = hidden_states[li].reshape(N_total, -1)
        fdr_li = fisher_discriminant_ratio(hs_li[high_mask], hs_li[low_mask])
        layer_fdr[li] = fdr_li
        print(f"    L{li}: FDR={fdr_li:.4f}")

    results['global'] = {
        'fdr_rocket': fdr,
        'knn_acc_rocket': knn_acc,
        'knn_acc_hidden': hs_knn_acc,
        'n_high': int(high_mask.sum()),
        'n_low': int(low_mask.sum()),
        'threshold_pct': mi_threshold_percentile,
    }
    results['per_layer_fdr'] = layer_fdr
    results['rocket_features'] = rocket_feats
    results['high_mask'] = high_mask
    results['low_mask'] = low_mask

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Change-Point Alignment
# ─────────────────────────────────────────────────────────────────────────────

def run_change_point_analysis(all_raw_ts, mi_matrix, seq_len, patch_len,
                               stride=None, n_segments=5):
    """
    Run ClaSPSegmenter on each test sequence and align change points
    with MI score profiles.

    Returns:
        results: dict with alignment statistics
    """
    if stride is None:
        stride = patch_len

    try:
        from aeon.segmentation import ClaSPSegmenter, BinSegmenter
    except ImportError:
        warnings.warn("aeon not installed. Skipping change-point analysis.")
        return None

    n_patches = seq_len // patch_len
    patch_centers = np.array([
        i * patch_len + patch_len // 2 for i in range(n_patches)
    ])

    # Aggregate change points across all test samples
    all_cps = []
    cp_per_mi_high = []   # MI value at nearest patch to each CP
    cp_per_mi_layer = []   # per-layer MI

    print(f"\n  Change-Point Analysis (n={len(all_raw_ts)} sequences)")
    print("  Running ClaSPSegmenter on each sequence...")

    for ts in tqdm(all_raw_ts[:500], desc="ClaSP", leave=False):  # cap for speed
        try:
            seg = ClaSPSegmenter()
            cps = seg.fit_predict(ts)
            all_cps.extend(cps)
            for cp in cps:
                # Find nearest patch center
                nearest_patch = np.argmin(np.abs(patch_centers - cp))
                for li in range(mi_matrix.shape[0]):
                    cp_per_mi_layer.append(mi_matrix[li, nearest_patch])
                    cp_per_mi_high.append(mi_matrix[li, nearest_patch])
        except Exception:
            continue

    if not all_cps:
        print("  No change points detected.")
        return None

    all_cps = np.array(all_cps)
    cp_per_mi_layer = np.array(cp_per_mi_layer)

    # Baseline: random MI values (sample from MI matrix uniformly)
    np.random.seed(42)
    n_random = len(cp_per_mi_layer)
    random_mi_flat = mi_matrix.flatten()
    baseline_samples = np.random.choice(random_mi_flat, size=n_random, replace=True)

    # Compare: are CPs at higher-MI locations than random?
    cp_mean = cp_per_mi_layer.mean()
    random_mean = baseline_samples.mean()
    cp_median = np.median(cp_per_mi_layer)
    random_median = np.median(baseline_samples)

    # Wilcoxon rank-sum test
    from scipy.stats import mannwhitneyu
    stat, pvalue = mannwhitneyu(cp_per_mi_layer, baseline_samples, alternative='greater')

    print(f"\n  Change-Point Alignment Results:")
    print(f"    N change points detected: {len(all_cps)}")
    print(f"    CP MI mean: {cp_mean:.4f}  |  Random baseline: {random_mean:.4f}")
    print(f"    CP MI median: {cp_median:.4f} |  Random median: {random_median:.4f}")
    print(f"    Mann-Whitney U p-value: {pvalue:.4e}")

    # Fraction of CPs within 1 patch of high-MI boundary
    n_patches = seq_len // patch_len
    patch_boundaries = set(range(0, seq_len, patch_len))
    cp_near_boundary = sum(
        any(abs(cp - b) <= patch_len // 2 for b in patch_boundaries)
        for cp in all_cps
    )
    frac_boundary = cp_near_boundary / len(all_cps) if all_cps.size > 0 else 0.0
    print(f"    CPs near patch boundary: {frac_boundary:.1%}")

    return {
        'all_cps': all_cps,
        'cp_mi_mean': float(cp_mean),
        'cp_mi_median': float(cp_median),
        'random_mi_mean': float(random_mean),
        'random_mi_median': float(random_median),
        'pvalue': float(pvalue),
        'frac_boundary': float(frac_boundary),
        'n_cps_total': len(all_cps),
        'patch_centers': patch_centers,
        'cp_per_mi_layer': cp_per_mi_layer,
        'baseline_samples': baseline_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _get_layer_colors(n_layers):
    palette = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999",
    ]
    return [palette[i % len(palette)] for i in range(n_layers)]


def plot_rocket_validation(rocket_results, mi_matrix, output_dir, dataset_name=""):
    """Generate Nature-style figures for ROCKET validation."""
    _nature_rc()
    n_layers = mi_matrix.shape[0]
    colors = _get_layer_colors(n_layers)

    # ── Figure 1: Panel A — MI score profile per layer ──────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.5))

    ax = axes[0]
    n_patches = mi_matrix.shape[1]
    x = np.arange(n_patches)
    for li in range(n_layers):
        ax.plot(x, mi_matrix[li], "o-", color=colors[li],
                label=f"L{li}", linewidth=1.4, markersize=4, zorder=2)
    ax.set_xlabel("Patch Index")
    ax.set_ylabel("MI (nats)")
    ax.set_title(f"{dataset_name}: Per-Layer MI Score Profile")
    ax.legend(loc="upper right", ncol=min(n_layers, 4), fontsize=7)
    ax.set_xticks(x)

    # ── Panel B — FDR per layer ────────────────────────────────────────────
    ax = axes[1]
    fdr_vals = [rocket_results['per_layer_fdr'].get(li, np.nan) for li in range(n_layers)]
    valid = np.isfinite(fdr_vals)
    ax.bar(np.arange(n_layers)[valid], np.array(fdr_vals)[valid],
           color=[colors[li] for li in range(n_layers) if valid[li]],
           edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Fisher FDR")
    ax.set_title("Fisher Discriminant Ratio: High-MI vs Low-MI Patches (Hidden States)")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])

    fig.savefig(os.path.join(output_dir, "fig1_mi_profile_fdr.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig1_mi_profile_fdr.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig1_mi_profile_fdr.{{pdf,png}}")

    # ── Figure 2: Panel A — k-NN accuracy comparison ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    gl = rocket_results['global']
    acc_vals = [gl['knn_acc_hidden'], gl['knn_acc_rocket']]
    lbl_vals = ["Hidden States\n(k-NN)", "ROCKET Features\n(k-NN)"]
    bar_colors = ["#0072B2", "#009E73"]
    bars = ax.bar(range(len(acc_vals)), acc_vals, color=bar_colors,
                  edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Random (50%)")
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(lbl_vals)))
    ax.set_xticklabels(lbl_vals)
    ax.set_ylabel("k-NN Accuracy")
    ax.set_title("Discriminability: High-MI vs Low-MI Patches")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)

    # ── Panel B — Per-layer FDR bar chart ──────────────────────────────────
    ax = axes[1]
    fdr_vals = [rocket_results['per_layer_fdr'].get(li, 0) for li in range(n_layers)]
    bars = ax.bar(range(n_layers), fdr_vals, color=colors,
                  edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2)
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Fisher FDR")
    ax.set_title("Per-Layer FDR (Hidden States)")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])

    fig.savefig(os.path.join(output_dir, "fig2_discriminability.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig2_discriminability.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig2_discriminability.{{pdf,png}}")


def plot_change_point_analysis(cp_results, mi_matrix, output_dir, dataset_name=""):
    """Generate Nature-style figures for change-point alignment."""
    _nature_rc()
    n_layers = mi_matrix.shape[0]
    colors = _get_layer_colors(n_layers)

    if cp_results is None:
        return

    # ── Figure 3: Panel A — MI distribution at CPs vs random ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    cp_mi = cp_results['cp_per_mi_layer']
    baseline = cp_results['baseline_samples']

    bins = np.linspace(min(cp_mi.min(), baseline.min()),
                       max(cp_mi.max(), baseline.max()), 30)
    ax.hist(cp_mi, bins=bins, alpha=0.7, label=f"CPs (n={len(cp_mi)})",
            color="#D55E00", density=True, zorder=2)
    ax.hist(baseline, bins=bins, alpha=0.5, label="Random baseline",
            color="#999999", density=True, zorder=1)
    ax.set_xlabel("MI Score")
    ax.set_ylabel("Density")
    ax.set_title("MI Distribution: Change Points vs Random Baseline")
    ax.legend(fontsize=7)

    ax = axes[1]
    vals = [cp_results['cp_mi_mean'], cp_results['random_mi_mean']]
    lbls = ["Change Points", "Random Baseline"]
    bar_colors = ["#D55E00", "#999999"]
    bars = ax.bar(lbls, vals, color=bar_colors, edgecolor="white", linewidth=0.5,
                  alpha=0.85, zorder=2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Mean MI Score")
    ax.set_title(f"Mean MI at CPs (p={cp_results['pvalue']:.2e})")
    if cp_results['cp_mi_mean'] > cp_results['random_mi_mean']:
        ax.annotate("CPs enriched\nfor high MI", xy=(0.5, 0.95),
                    xycoords='axes fraction', ha='center', va='top',
                    fontsize=7, color="green")

    fig.savefig(os.path.join(output_dir, "fig3_change_point_alignment.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig3_change_point_alignment.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig3_change_point_alignment.{{pdf,png}}")

    # ── Figure 4: MI profile with CP overlay on first test sample ───────────
    if 'all_cps' in cp_results and len(cp_results['all_cps']) > 0:
        fig, ax = plt.subplots(figsize=(7.0, 2.4))
        n_patches = mi_matrix.shape[1]
        patch_len_approx = 96
        seq_len_approx = n_patches * patch_len_approx
        patch_centers = np.array([
            i * patch_len_approx + patch_len_approx // 2 for i in range(n_patches)
        ])

        for li in range(n_layers):
            ax.plot(patch_centers, mi_matrix[li], "o-", color=colors[li],
                    label=f"L{li}", linewidth=1.4, markersize=4, zorder=2)

        # Overlay CPs as vertical lines
        cp_x = cp_results['all_cps']
        if len(cp_x) > 0:
            for cp in cp_x[:50]:  # show first 50
                ax.axvline(cp, color="red", alpha=0.3, linewidth=0.8, zorder=1)

        red_patch = mpatches.Patch(color="red", alpha=0.3, label=f"Detected CPs (n={len(cp_x)})")
        ax.legend(loc="upper right", fontsize=7, ncol=min(n_layers + 1, 5))
        ax.set_xlabel("Time Index")
        ax.set_ylabel("MI (nats)")
        ax.set_title(f"{dataset_name}: MI Profile with Change Points")

        fig.savefig(os.path.join(output_dir, "fig4_mi_with_change_points.pdf"), dpi=300)
        fig.savefig(os.path.join(output_dir, "fig4_mi_with_change_points.png"), dpi=300)
        plt.close()
        print(f"  [Saved] fig4_mi_with_change_points.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Timer MI Validation: ROCKET Features & Change-Point Alignment")
    # Paths
    parser.add_argument("--mi_dir", type=str,
                        default="./timer_mi_ksg_pca/Timer_MI_20260514_073311/")
    parser.add_argument("--data_path", type=str, default="./datasets/ETTh1.csv")
    parser.add_argument("--data_type", type=str, default="ETTh1",
                        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])
    parser.add_argument("--output_dir", type=str,
                        default="./results/timer_mi_rocket_validation/")
    # Model
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Data
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--freq", type=str, default="h")
    # Validation
    parser.add_argument("--mi_percentile", type=float, default=50,
                        help="Percentile threshold for high/low MI split")
    parser.add_argument("--n_kernels", type=int, default=10000,
                        help="Number of ROCKET kernels")
    parser.add_argument("--n_cp_samples", type=int, default=500,
                        help="Max test sequences for change-point detection")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Timer MI Validation: ROCKET Features & Change-Point Alignment")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<25}: {v}")
    print("=" * 70)

    # ── Phase 1: Load MI matrix ────────────────────────────────────────────
    print("\n>>> [1/5] Loading MI matrix...")
    mi_path = os.path.join(args.mi_dir, "mi_hy_matrix.npy")
    if not os.path.exists(mi_path):
        mi_path = os.path.join(args.mi_dir, "mi_matrix.npy")
    if not os.path.exists(mi_path):
        raise FileNotFoundError(f"MI file not found in {args.mi_dir}")

    mi_matrix = np.load(mi_path)
    n_layers_raw, n_patches_raw = mi_matrix.shape
    print(f"  Raw MI matrix: {mi_matrix.shape}")

    # Load model to resolve shape
    core_tmp = _unwrap(build_timer_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len, stride=args.patch_len,
        d_model=args.d_model, d_ff=args.d_ff,
        e_layers=args.e_layers, n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len, pred_len=args.pred_len,
    ))
    model_layers = core_tmp.layers
    model_heads = core_tmp.n_heads
    model_patches = args.seq_len // args.patch_len
    del core_tmp

    print(f"  Model: {model_layers} layers, {model_heads} heads, {model_patches} patches")

    # Shape alignment
    if n_layers_raw == model_heads and n_patches_raw == model_patches:
        print("  Detected: MI computed per-head. Averaging to per-layer.")
        mi_matrix = mi_matrix.mean(axis=0, keepdims=True)
        n_layers_raw = 1

    if n_layers_raw == model_patches and n_patches_raw == model_heads:
        print("  Auto-transposing MI matrix.")
        mi_matrix = mi_matrix.T
        n_layers_raw, n_patches_raw = n_patches_raw, n_layers_raw

    if n_patches_raw != model_patches and n_layers_raw == model_patches:
        print("  Auto-transposing MI matrix.")
        mi_matrix = mi_matrix.T
        n_layers_raw, n_patches_raw = n_patches_raw, n_layers_raw

    if n_layers_raw != model_layers:
        print(f"  Adapting MI layers: {n_layers_raw} -> {model_layers}")
        if n_layers_raw > model_layers:
            mi_matrix = mi_matrix[:model_layers]
        else:
            mi_matrix = np.broadcast_to(mi_matrix, (model_layers, mi_matrix.shape[1]))

    n_layers_final, n_patches_final = mi_matrix.shape
    print(f"  Final MI matrix: [{n_layers_final}, {n_patches_final}]")

    # ── Phase 2: Load dataset ──────────────────────────────────────────────
    print("\n>>> [2/5] Loading dataset...")
    test_dataset = CIDatasetBenchmark(
        root_path=args.data_path,
        flag='test',
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = test_dataset.n_var
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"  Variables: {n_vars}, Test samples: {len(test_dataset)}")

    # ── Phase 3: Load Timer model ──────────────────────────────────────────
    print("\n>>> [3/5] Loading Timer model...")
    model = build_timer_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len, stride=args.patch_len,
        d_model=args.d_model, d_ff=args.d_ff,
        e_layers=args.e_layers, n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len, pred_len=args.pred_len,
    ).to(device)
    model.eval()
    print(f"  Device: {device}")

    # ── Phase 4: Extract hidden states ─────────────────────────────────────
    print("\n>>> [4/5] Extracting hidden states...")
    all_hidden, raw_patches, all_raw_ts = extract_patch_hidden_states(
        model, test_loader, device, n_layers_final)

    n_samples = len(test_dataset)
    print(f"  Hidden states: {[h.shape for h in all_hidden]}")
    print(f"  Raw patches: {raw_patches.shape}")
    print(f"  Raw TS: {all_raw_ts.shape}")

    # ── Phase 5: ROCKET Validation ─────────────────────────────────────────
    print("\n>>> [5/5] Running ROCKET validation...")
    rocket_results = run_rocket_validation(
        hidden_states=all_hidden,
        raw_patches=raw_patches,
        mi_matrix=mi_matrix,
        n_layers=n_layers_final,
        mi_threshold_percentile=args.mi_percentile,
        n_kernels=args.n_kernels,
        random_state=args.seed,
    )

    # ── Phase 6: Change-Point Analysis ────────────────────────────────────
    print("\n>>> [6/5] Running change-point alignment...")
    # Cap test samples for speed
    all_raw_ts_capped = all_raw_ts[:args.n_cp_samples]
    cp_results = run_change_point_analysis(
        all_raw_ts=all_raw_ts_capped,
        mi_matrix=mi_matrix,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        n_segments=5,
    )

    # ── Save results ───────────────────────────────────────────────────────
    print("\n>>> [Saving] Writing results...")
    results = {
        'config': vars(args),
        'rocket': {k: v for k, v in rocket_results.items()
                   if k not in ['rocket_features', 'high_mask', 'low_mask']},
        'change_point': {k: v for k, v in (cp_results or {}).items()
                         if k not in ['all_cps', 'cp_per_mi_layer',
                                      'baseline_samples', 'patch_centers']},
    }

    results_path = os.path.join(args.output_dir, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else list(x) if isinstance(x, np.ndarray) else x)
    print(f"  Saved: {results_path}")

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\n>>> [Plotting] Generating figures...")
    dataset_name = args.data_type
    try:
        plot_rocket_validation(rocket_results, mi_matrix, args.output_dir, dataset_name)
    except Exception as e:
        print(f"  WARNING: ROCKET plot failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_change_point_analysis(cp_results, mi_matrix, args.output_dir, dataset_name)
    except Exception as e:
        print(f"  WARNING: CP plot failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    print(f"\n  Dataset: {dataset_name}")
    print(f"  MI percentile threshold: {args.mi_percentile}%")
    print(f"  High-MI patches: {rocket_results['global']['n_high']}")
    print(f"  Low-MI patches:  {rocket_results['global']['n_low']}")

    print(f"\n  ROCKET Feature Discriminability:")
    print(f"    Fisher FDR (ROCKET):    {rocket_results['global']['fdr_rocket']:.4f}")
    print(f"    k-NN acc (ROCKET):      {rocket_results['global']['knn_acc_rocket']:.4f}")
    print(f"    k-NN acc (hidden):      {rocket_results['global']['knn_acc_hidden']:.4f}")

    print(f"\n  Per-Layer Fisher FDR (Hidden States):")
    for li in range(n_layers_final):
        fdr_li = rocket_results['per_layer_fdr'].get(li, np.nan)
        print(f"    L{li:2d}: {fdr_li:.4f}")

    if cp_results:
        print(f"\n  Change-Point Alignment:")
        print(f"    N CPs detected:         {cp_results['n_cps_total']}")
        print(f"    Mean MI at CPs:         {cp_results['cp_mi_mean']:.4f}")
        print(f"    Mean MI (random):       {cp_results['random_mi_mean']:.4f}")
        print(f"    Mann-Whitney p-value:    {cp_results['pvalue']:.4e}")
        print(f"    CPs near patch boundary: {cp_results['frac_boundary']:.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
