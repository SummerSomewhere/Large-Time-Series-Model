#!/usr/bin/env python3
"""
Timer MI-变点对齐 & STL语义成分分析实验。

模仿 sundial_mi_changepoint_alignment.py，为 Timer 模型定制。

研究问题：
  1. 高 MI patch 是否对应于数据中的结构性变点（趋势转折点、异常值等）？
  2. MI 分数能否解释 Trend/Seasonal/Residual 成分在原始时序中的分布？
  3. 将 MI 峰值位置与 ruptures 变点检测结果对齐，量化重叠率。

实验设计：
  [模块A] 变点检测与 MI 峰值对齐
    - 使用 ruptures 变点检测（或梯度备选法）
    - 将变点位置映射到 patch 索引
    - 计算 MI 分数在变点附近的统计特性（均值、方差、峰值密度）
    - 蒙特卡洛置换检验：随机基线 vs 变点位置

  [模块B] STL 语义成分与 MI 的对应关系
    - 对每个 patch 对应的原始子序列做 STL 分解
    - 检验：高 MI patch 的子序列是否 trend 成分更显著？seasonal 更强？
    - 对比层间 MI 与各成分强度的相关性

  [模块C] MI 峰值的时序结构解读
    - 识别 MI 曲线中的尖峰位置
    - 分析尖峰位置对应的原始子序列特征（方差、趋势斜率、周期强度）
    - 构建 "MI 峰值 ←→ 时序结构" 的映射

Usage:
    python experiments/timer_mi_changepoint_alignment.py \
        --mi_result_dir ./results/timer_mi_ksg_pca/Timer_MI_20260517_112544 \
        --root_path ./datasets/ --data_path ETTh1.csv \
        --seq_len 672 --pred_len 96 --patch_len 96 \
        --batch_size 64 --max_samples 500 \
        --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
        --e_layers 8 --d_model 1024 --d_ff 2048 --n_heads 8 \
        --stl_period 24 \
        --gpu 0 --out_dir ./results/timer_mi_changepoint/ --model_id etth1

Dependencies:
    pip install aeon ruptures matplotlib numpy pandas scipy torch tqdm scikit-learn statsmodels
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import gc
import datetime
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from data_provider.data_loader_benchmark import CIDatasetBenchmark
from utils.masking import TriangularCausalMask


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
    "blue_secondary":  "#3775BA",
    "green_3":       "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "orange":          "#E07B39",
    "neutral_light":   "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}


def add_panel_label(ax, label, x=-0.08, y=1.06, fontsize=10,
                    fontweight="bold", color="black"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha="left", va="bottom")


def finalize_figure(fig, out_path, dpi=300, pad=1.2):
    import os as _os
    from pathlib import Path
    fig.tight_layout(pad=pad)
    base = Path(out_path)
    _os.makedirs(base.parent, exist_ok=True)
    base = base.with_suffix("")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {base}.{{svg,pdf,png}}")


# ============================================================================
# Config & Model Builder
# ============================================================================

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(ckpt_path: str, patch_len: int, stride: int,
                d_model: int, d_ff: int, e_layers: int,
                n_heads: int, dropout: float,
                seq_len: int, pred_len: int,
                enc_in: int = 1) -> Model:
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
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
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
# TIMER Token Extraction
# ============================================================================

def extract_raw_patches_and_mi(
    model, data_loader, num_layers, patch_len,
    n_vars, device, max_samples=0,
):
    """
    提取原始 patches + 每层 hidden states（与 timer_mi_ksg_pca.py 逻辑一致）。

    CIDatasetBenchmark 的 seq_x 形状为 [B, seq_len, 1]（即 [B, T, M]，M=1）。
    我们直接从 seq_x 切 raw patch（未标准化）用于变点/STL 分析；
    对标准化后的序列提取 hidden states 用于 MI 分析。

    Returns:
        hist_tokens:  list of [N_total, n_patches, D] per layer
        raw_patches: [N_total, n_patches, patch_len] — 原始时序 patch（未标准化）
        n_patches:  int
    """
    core = _unwrap(model)
    all_hist_tokens = [[] for _ in range(num_layers)]
    all_raw_patches = []

    n_patches = None
    total_samples = 0

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(
            data_loader, desc="提取 patch 表示"):
            B = seq_x.shape[0]
            total_samples += B
            if max_samples > 0 and total_samples > max_samples:
                break

            # seq_x: [B, seq_len, n_vars] = [B, T, M]
            # For raw patches: mean over variables (dim M), then patch the time series.
            # seq_x [B, T, M] -> mean over M -> [B, T] -> reshape to patches [B, N, patch_len]
            B_, T_full, M_ = seq_x.shape
            n_p = T_full // patch_len
            actual_T = n_p * patch_len
            raw_patch = seq_x[:, :actual_T, :].float().mean(dim=2)  # [B, T] = mean over M
            raw_patch = raw_patch.view(B_, n_p, patch_len)       # [B, N, patch_len]
            all_raw_patches.append(raw_patch)

            if n_patches is None:
                n_patches = n_p
                print(f"  检测到 n_patches={n_patches}, B={B_}, T={T_full}, M={M_}")

            # Normalization（与 Timer.forecast 一致）：[B, T, M] -> mean over T
            sx = seq_x.float().to(device)  # [B, T, M]
            means = sx.mean(dim=1, keepdim=True).detach()  # [B, 1, M]
            stdev = torch.sqrt(
                torch.var(sx, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            stdev = torch.clamp_min(stdev, 1e-5)
            x_norm = (sx - means) / stdev  # [B, T, M]

            # enc_embedding expects [B, M, T]
            x2 = x_norm.permute(0, 2, 1)  # [B, M, T]
            dec_in, n_vars_det = core.enc_embedding(x2)  # [B*M, N, D]
            BM, N, D = dec_in.shape
            derived_n_vars = BM // B

            mask = TriangularCausalMask(BM, N, device=device)

            # 逐层提取 hidden states（TIMER 是 encoder，每层顺序处理）
            h = dec_in
            for li, layer_module in enumerate(core.decoder.attn_layers):
                if li >= num_layers:
                    break
                h, _, _ = layer_module(h, attn_mask=mask)
                # Reshape: [B*M, N, D] -> [B, M, N, D] -> [B, N, D]（mean over M）
                h_avg = h.view(B, derived_n_vars, N, D).mean(dim=1).float().cpu()
                all_hist_tokens[li].append(h_avg)

            del dec_in, h, x_norm
            gc.collect()
            torch.cuda.empty_cache()

    hist_tokens = [torch.cat(toks, dim=0) for toks in all_hist_tokens]
    raw_patches = torch.cat(all_raw_patches, dim=0)

    return hist_tokens, raw_patches, n_patches


# ============================================================================
# Module A: Change Point Detection & MI Peak Alignment
# ============================================================================

def detect_change_points_ruptures(raw_seq, method="Pelt", min_size=5):
    """
    使用 ruptures 检测变点。
    Available methods: Pelt, Binseg, BottomUp, Window

    策略：
    - 先尝试较低 penalty 的 BIC (2*log(n)) 检测变点
    - 若找到过多 (>n_patches*2)，改用更严格 penalty
    - 若仍无检测结果，回退到梯度法
    """
    try:
        import ruptures as rpt
    except ImportError:
        print("  WARNING: ruptures 未安装. 请运行: pip install ruptures")
        print("  使用梯度法作为备选...")
        return _detect_change_points_gradient(raw_seq)

    seq = np.asarray(raw_seq).flatten()
    if len(seq) < min_size * 3:
        return _detect_change_points_gradient(raw_seq)

    seq_norm = (seq - seq.mean()) / (seq.std() + 1e-8)
    n = len(seq_norm)

    # BIC-like penalty: 2*log(n) — detects more change points than 3*log(n)
    pen_bic = 2.0 * np.log(n)

    for pen_val in [pen_bic, pen_bic * 1.5, pen_bic * 2.0]:
        try:
            algo = rpt.Pelt(model="l2", min_size=min_size).fit(seq_norm.reshape(-1, 1))
            cps = algo.predict(pen=pen_val)
            # Exclude last (always the end), require at least 1 valid CP
            cps = np.array(cps[:-1])
            if len(cps) > 0:
                return cps
        except Exception:
            pass

    # Fallback to gradient method
    return _detect_change_points_gradient(raw_seq)


def _detect_change_points_gradient(raw_seq, window=5, threshold_ratio=0.9):
    """
    备选：基于梯度的变点检测。
    """
    seq = np.asarray(raw_seq).flatten()
    grad = np.abs(np.diff(seq))
    from scipy.ndimage import uniform_filter1d
    grad_smooth = uniform_filter1d(grad, size=window, mode="reflect")
    threshold = np.percentile(grad_smooth, threshold_ratio * 100)
    cps = np.where(grad_smooth > threshold)[0]
    if len(cps) == 0:
        return np.array([])
    clustered = [cps[0]]
    for c in cps[1:]:
        if c - clustered[-1] > window:
            clustered.append(c)
    return np.array(clustered)


def align_mi_peaks_with_change_points(
    mi_matrix: np.ndarray,
    raw_patches: torch.Tensor,
    n_patches: int,
    patch_len: int,
    seq_len: int,
    max_samples: int = 2000,
    mi_n_patches: int = None,
) -> dict:
    """
    核心对齐函数：验证 MI 峰值位置与数据变点的重叠关系。
    """
    from scipy.stats import mannwhitneyu, ttest_ind

    N_total, raw_n_patches, p_len = raw_patches.shape

    # n_patches 必须是 MI summary 中的 N，不能用 raw_patches.shape 的值覆盖
    # raw_patches.shape[1] 可能在不同 run 之间不一致
    actual_n_patches = mi_n_patches if mi_n_patches is not None else n_patches

    if mi_matrix.ndim == 2:
        mi_mean_per_patch = mi_matrix.mean(axis=0)
    else:
        mi_mean_per_patch = mi_matrix

    # 长度对齐检查
    if len(mi_mean_per_patch) != actual_n_patches:
        print(f"  [ModuleA Debug] MI per-patch length ({len(mi_mean_per_patch)}) != n_patches ({actual_n_patches}), "
              f"raw_patches N={raw_n_patches}). Skipping change-point alignment.")
        print(f"  [ModuleA Debug] MI first 3: {mi_mean_per_patch[:3]}")
        return {}

    all_near = []
    all_far = []
    samples_processed = 0
    samples_skipped_len = 0
    samples_skipped_cp = 0
    samples_skipped_filter = 0
    cp_counts = []

    for sample_idx in range(min(N_total, max_samples)):
        raw_seq = raw_patches[sample_idx]
        raw_seq_concat = raw_seq.view(-1).numpy()

        if len(raw_seq_concat) < patch_len * 2:
            samples_skipped_len += 1
            continue

        cps = detect_change_points_ruptures(raw_seq_concat)
        cp_counts.append(len(cps))
        if len(cps) == 0:
            samples_skipped_cp += 1
            continue

        cp_patch_indices = set()
        for cp_pos in cps:
            pi = min(cp_pos // patch_len, actual_n_patches - 1)
            cp_patch_indices.add(pi)
            for delta in [-1, 1]:
                nb = pi + delta
                if 0 <= nb < actual_n_patches:
                    cp_patch_indices.add(nb)

        if len(cp_patch_indices) == 0:
            samples_skipped_cp += 1
            continue

        sample_mi = mi_mean_per_patch

        near_mask = np.array([i in cp_patch_indices for i in range(actual_n_patches)])
        mi_near = sample_mi[near_mask]
        mi_far = sample_mi[~near_mask]

        # min 1 patch each side is sufficient for statistical testing with n_permutations
        if len(mi_near) >= 1 and len(mi_far) >= 1:
            all_near.append(mi_near)
            all_far.append(mi_far)
        else:
            samples_skipped_filter += 1

        samples_processed += 1

    # Debug summary
    print(f"  [ModuleA Debug] samples_processed={samples_processed}, "
          f"skipped(len)={samples_skipped_len}, skipped(cp=0)={samples_skipped_cp}, "
          f"skipped(filter)={samples_skipped_filter}")
    print(f"  [ModuleA Debug] MI n_patches={actual_n_patches}, raw_patches N={raw_n_patches}")
    if cp_counts:
        print(f"  [ModuleA Debug] CP counts: min={min(cp_counts)}, max={max(cp_counts)}, "
              f"mean={np.mean(cp_counts):.1f}")
    if all_near:
        n_near_total = sum(len(x) for x in all_near)
        n_far_total = sum(len(x) for x in all_far)
        print(f"  [ModuleA Debug] near/far patches: near_total={n_near_total}, far_total={n_far_total}, "
              f"near_per_sample={n_near_total/max(samples_processed,1):.1f}")

    if not all_near:
        print("  WARNING: 变点对齐无有效样本")
        return {}

    near_flat = np.concatenate(all_near)
    far_flat = np.concatenate(all_far)

    stat_mw, p_mw = mannwhitneyu(near_flat, far_flat, alternative="greater")
    stat_t, p_t = ttest_ind(near_flat, far_flat, alternative="greater")

    mean_near = float(np.mean(near_flat))
    mean_far = float(np.mean(far_flat))
    std_pooled = float(np.std(np.concatenate([near_flat, far_flat])))
    cohen_d = (mean_near - mean_far) / (std_pooled + 1e-8)

    n_permutations = 2000
    observed_diff = mean_near - mean_far
    all_vals = np.concatenate([near_flat, far_flat])
    n_near = len(near_flat)

    perm_diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        perm = rng.permutation(len(all_vals))
        perm_near = all_vals[perm[:n_near]]
        perm_far = all_vals[perm[n_near:]]
        perm_diffs.append(np.mean(perm_near) - np.mean(perm_far))

    perm_diffs = np.array(perm_diffs)
    p_perm = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

    results = {
        "mean_mi_near_cp": mean_near,
        "mean_mi_far_cp": mean_far,
        "std_mi_near_cp": float(np.std(near_flat)),
        "std_mi_far_cp": float(np.std(far_flat)),
        "n_near_cp": len(near_flat),
        "n_far_cp": len(far_flat),
        "cohen_d": cohen_d,
        "mann_whitney_u": float(stat_mw),
        "mann_whitney_p": float(p_mw),
        "ttest_t": float(stat_t),
        "ttest_p": float(p_t),
        "permutation_test_p": p_perm,
        "n_permutations": n_permutations,
        "n_samples": samples_processed,
        "n_cp_per_sample": float(np.mean(cp_counts)) if cp_counts else 0.0,
    }

    return results


# ============================================================================
# Module B: STL Semantic Components & MI
# ============================================================================

def compute_stl_components_per_patch(
    raw_patches: torch.Tensor,
    mi_scores: np.ndarray,
    seq_len: int,
    stl_period: int = 24,
    stl_robust: bool = True,
    max_samples_for_stl: int = 500,
) -> dict:
    """
    对每个样本独立做 STL 分解，再将 trend/seasonal/residual 成分映射到每个 patch，
    计算 per-patch 的 trend strength 和 seasonal strength。
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        print("  WARNING: statsmodels 未安装，请运行: pip install statsmodels")
        return {}

    N_total, n_patches, patch_len = raw_patches.shape

    from scipy.stats import skew as _skew, kurtosis as _kurtosis

    if mi_scores.ndim == 2:
        mi_per_patch = mi_scores.mean(axis=0)
    else:
        mi_per_patch = mi_scores

    n_samples_stl = min(N_total, max_samples_for_stl)
    print(f"  [STL] 参与 STL 分解的样本数: {n_samples_stl}/{N_total}, period={stl_period}")

    accum_ts    = np.zeros(n_patches)
    accum_ss    = np.zeros(n_patches)
    accum_ts_sh = np.zeros(n_patches)
    accum_ss_sh = np.zeros(n_patches)
    accum_rs_sh = np.zeros(n_patches)
    accum_rs_sd = np.zeros(n_patches)
    accum_var   = np.zeros(n_patches)
    accum_skew  = np.zeros(n_patches)
    accum_kurt  = np.zeros(n_patches)

    sample_ts_list    = []
    sample_ss_list    = []
    sample_ts_sh_list = []
    sample_ss_sh_list = []
    sample_rs_sh_list = []
    sample_rs_sd_list = []
    sample_var_list   = []
    sample_skew_list  = []
    sample_kurt_list  = []

    global_var_trend     = []
    global_var_seasonal = []
    global_var_resid    = []
    global_var_total    = []

    samples_valid = 0

    for si in range(n_samples_stl):
        sample_patches = raw_patches[si]
        full_seq = sample_patches.view(-1).numpy()

        if len(full_seq) != seq_len:
            continue

        try:
            if stl_robust:
                stl = STL(full_seq, period=stl_period, robust=True).fit()
            else:
                stl = STL(full_seq, period=stl_period, robust=False).fit()
        except Exception as e:
            print(f"  WARNING: 样本 {si} STL 分解失败: {e}")
            continue

        trend_full    = stl.trend
        seasonal_full = stl.seasonal
        resid_full    = stl.resid

        var_trend    = float(np.var(trend_full))
        var_seasonal = float(np.var(seasonal_full))
        var_resid    = float(np.var(resid_full))
        var_total    = float(np.var(full_seq))

        global_var_trend.append(var_trend)
        global_var_seasonal.append(var_seasonal)
        global_var_resid.append(var_resid)
        global_var_total.append(var_total)

        samples_valid += 1

        s_ts    = np.empty(n_patches)
        s_ss    = np.empty(n_patches)
        s_ts_sh = np.empty(n_patches)
        s_ss_sh = np.empty(n_patches)
        s_rs_sh = np.empty(n_patches)
        s_rs_sd = np.empty(n_patches)
        s_var   = np.empty(n_patches)
        s_skew  = np.empty(n_patches)
        s_kurt  = np.empty(n_patches)

        for pi in range(n_patches):
            start = pi * patch_len
            end   = (pi + 1) * patch_len

            trend_pi    = trend_full[start:end]
            seasonal_pi = seasonal_full[start:end]
            resid_pi    = resid_full[start:end]
            raw_pi      = full_seq[start:end]

            var_trend_pi    = float(np.var(trend_pi))
            var_seasonal_pi = float(np.var(seasonal_pi))
            var_resid_pi    = float(np.var(resid_pi))

            var_deseasonal = var_trend_pi + var_resid_pi
            var_detrend    = var_seasonal_pi + var_resid_pi

            ts_local = max(0.0, 1.0 - var_resid_pi / var_deseasonal) if var_deseasonal > 1e-10 else 0.0
            ss_local = max(0.0, 1.0 - var_resid_pi / var_detrend)    if var_detrend    > 1e-10 else 0.0

            ts_share = var_trend_pi    / (var_trend    + 1e-10)
            ss_share = var_seasonal_pi / (var_seasonal + 1e-10)
            rs_share = var_resid_pi    / (var_resid    + 1e-10)

            raw_var   = float(np.var(raw_pi))
            raw_skew  = float(_skew(raw_pi))
            raw_kurt  = float(_kurtosis(raw_pi))

            accum_ts[pi]    += ts_local
            accum_ss[pi]    += ss_local
            accum_ts_sh[pi] += ts_share
            accum_ss_sh[pi] += ss_share
            accum_rs_sh[pi] += rs_share
            accum_rs_sd[pi] += float(np.std(resid_pi))
            accum_var[pi]   += raw_var
            accum_skew[pi]  += raw_skew
            accum_kurt[pi]  += raw_kurt

            s_ts[pi]    = ts_local
            s_ss[pi]    = ss_local
            s_ts_sh[pi] = ts_share
            s_ss_sh[pi] = ss_share
            s_rs_sh[pi] = rs_share
            s_rs_sd[pi] = float(np.std(resid_pi))
            s_var[pi]   = raw_var
            s_skew[pi]  = raw_skew
            s_kurt[pi]  = raw_kurt

        sample_ts_list.append(s_ts)
        sample_ss_list.append(s_ss)
        sample_ts_sh_list.append(s_ts_sh)
        sample_ss_sh_list.append(s_ss_sh)
        sample_rs_sh_list.append(s_rs_sh)
        sample_rs_sd_list.append(s_rs_sd)
        sample_var_list.append(s_var)
        sample_skew_list.append(s_skew)
        sample_kurt_list.append(s_kurt)

    if samples_valid == 0:
        print("  WARNING: 所有样本 STL 分解均失败")
        return {}

    accum_ts    /= samples_valid
    accum_ss    /= samples_valid
    accum_ts_sh /= samples_valid
    accum_ss_sh /= samples_valid
    accum_rs_sh /= samples_valid
    accum_rs_sd /= samples_valid
    accum_var   /= samples_valid
    accum_skew  /= samples_valid
    accum_kurt  /= samples_valid

    var_trend_full_mean    = float(np.mean(global_var_trend))
    var_seasonal_full_mean = float(np.mean(global_var_seasonal))
    var_resid_full_mean    = float(np.mean(global_var_resid))
    var_total_full_mean    = float(np.mean(global_var_total))

    print(f"  [STL] 全局分解（跨{samples_valid}样本均值）: "
          f"var_total={var_total_full_mean:.4f}, "
          f"var_trend={var_trend_full_mean:.4f}, "
          f"var_seasonal={var_seasonal_full_mean:.4f}, "
          f"var_resid={var_resid_full_mean:.4f}")

    from scipy.stats import spearmanr, pearsonr

    results = {"per_patch": {}, "global": {}}

    metrics = {
        "trend_strength":    accum_ts,
        "seasonal_strength": accum_ss,
        "trend_share":       accum_ts_sh,
        "seasonal_share":    accum_ss_sh,
        "residual_share":    accum_rs_sh,
        "residual_std":      accum_rs_sd,
        "raw_variance":      accum_var,
        "raw_skewness":      accum_skew,
        "raw_kurtosis":      accum_kurt,
    }

    for metric_name, metric_arr in metrics.items():
        valid = ~(np.isnan(metric_arr) | np.isnan(mi_per_patch))
        n_valid = valid.sum()
        if n_valid < 3:
            results["per_patch"][metric_name] = {
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
            }
            if n_valid == 0:
                print(f"  WARNING: {metric_name}: 全部 NaN，跳过 (n_valid=0)")
            else:
                print(f"  WARNING: {metric_name}: 有效点数不足 (n_valid={n_valid}<3)，跳过")
            continue
        try:
            rho_sp, p_sp = spearmanr(metric_arr[valid], mi_per_patch[valid])
            r_pr, p_pr   = pearsonr(metric_arr[valid],   mi_per_patch[valid])
        except Exception as e:
            print(f"  WARNING: {metric_name}: 相关性计算失败 ({e})，跳过")
            results["per_patch"][metric_name] = {
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "pearson_r": np.nan, "pearson_p": np.nan,
            }
            continue

        results["per_patch"][metric_name] = {
                "spearman_rho": float(rho_sp),
                "spearman_p":   float(p_sp),
                "pearson_r":    float(r_pr),
                "pearson_p":    float(p_pr),
                "_raw_array":    metric_arr.copy(),
            }

    results["global"] = {
        "var_total":    var_total_full_mean,
        "var_trend":    var_trend_full_mean,
        "var_seasonal": var_seasonal_full_mean,
        "var_resid":    var_resid_full_mean,
        "stl_period":   stl_period,
        "patch_len":    patch_len,
        "seq_len":      seq_len,
        "n_samples_stl": samples_valid,
    }

    n_samples_stored = len(sample_ts_list)
    results["_sample_level"] = {
        "trend_strength":    np.array(sample_ts_list),
        "seasonal_strength": np.array(sample_ss_list),
        "trend_share":      np.array(sample_ts_sh_list),
        "seasonal_share":    np.array(sample_ss_sh_list),
        "residual_share":   np.array(sample_rs_sh_list),
        "residual_std":     np.array(sample_rs_sd_list),
        "raw_variance":     np.array(sample_var_list),
        "raw_skewness":     np.array(sample_skew_list),
        "raw_kurtosis":     np.array(sample_kurt_list),
        "mi":               np.broadcast_to(mi_per_patch, (n_samples_stored, len(mi_per_patch))),
    }

    results["_mi_per_patch"] = mi_per_patch
    return results


# ============================================================================
# Module C: MI Peak Characterization
# ============================================================================

def characterize_mi_peaks(
    mi_matrix: np.ndarray,
    raw_patches: torch.Tensor,
    n_patches: int,
    patch_len: int,
    top_k_ratio: float = 0.15,
    mi_n_patches: int = None,
) -> dict:
    """
    识别 MI 曲线中的峰值位置，分析其对应的时序结构特征。

    方法：
    1. 找到每层 MI 曲线中 top-K% 的峰值位置
    2. 对每个峰值位置，分析其对应原始子序列的特征：
       - 方差（volatility）
       - 趋势斜率（trend slope）
       - 周期峰值数（seasonal peaks）
       - 与邻居的跳变（mean_jump, max_jump）
    3. 对比峰值位置 vs 非峰值位置的统计差异（Mann-Whitney + Cohen's d）
    """
    from scipy.stats import mannwhitneyu
    from scipy.signal import find_peaks

    N_total, raw_n_patches, _ = raw_patches.shape

    # Use MI summary N, not raw_patches.shape, for consistency with Module A
    actual_n_patches = mi_n_patches if mi_n_patches is not None else n_patches

    if mi_matrix.ndim == 2:
        mi_mean = mi_matrix.mean(axis=0)
    else:
        mi_mean = mi_matrix

    if len(mi_mean) != actual_n_patches:
        print(f"  [ModuleC Debug] MI length ({len(mi_mean)}) != n_patches ({actual_n_patches}), "
              f"raw_patches N={raw_n_patches}. Skipping peak analysis.")
        return {}

    k = max(1, int(actual_n_patches * top_k_ratio))
    peak_threshold = np.percentile(mi_mean, 100 * (1 - top_k_ratio))
    peak_indices = np.where(mi_mean >= peak_threshold)[0]
    non_peak_indices = np.setdiff1d(np.arange(actual_n_patches), peak_indices)

    print(f"  [ModuleC Debug] MI n_patches={actual_n_patches}, raw_patches N={raw_n_patches}, "
          f"top_k_ratio={top_k_ratio}, k={k}, threshold={peak_threshold:.4f}")
    print(f"  [ModuleC Debug] Peak patches: {list(peak_indices)}, Non-peak: {len(non_peak_indices)} patches")
    print(f"  [ModuleC Debug] MI mean values: {mi_mean.round(4)}")

    def patch_features(patch_seq):
        x = np.asarray(patch_seq)
        features = {}
        features["variance"] = float(np.var(x))
        features["mean"] = float(np.mean(x))
        features["std"] = float(np.std(x))

        t = np.arange(len(x))
        x_centered = x - x.mean()
        t_centered = t - t.mean()
        slope = float(np.sum(x_centered * t_centered) / (np.sum(t_centered ** 2) + 1e-10))
        features["trend_slope"] = slope
        features["trend_abs_slope"] = abs(slope)

        diffs = np.abs(np.diff(x))
        features["mean_jump"] = float(np.mean(diffs))
        features["max_jump"] = float(np.max(diffs))

        peaks, _ = find_peaks(x)
        features["n_peaks"] = len(peaks)

        features["cross_sample_std"] = float(np.std(x))

        return features

    peak_features = {"variance": [], "trend_abs_slope": [], "mean_jump": [],
                     "max_jump": [], "n_peaks": [], "cross_sample_std": []}
    non_peak_features = {"variance": [], "trend_abs_slope": [], "mean_jump": [],
                          "max_jump": [], "n_peaks": [], "cross_sample_std": []}

    for pi in peak_indices:
        patch_seq = raw_patches[:, pi, :].mean(dim=0).numpy()
        f = patch_features(patch_seq)
        for k_feat in peak_features:
            peak_features[k_feat].append(f[k_feat])

    for pi in non_peak_indices:
        patch_seq = raw_patches[:, pi, :].mean(dim=0).numpy()
        f = patch_features(patch_seq)
        for k_feat in non_peak_features:
            non_peak_features[k_feat].append(f[k_feat])

    results = {"comparison": {}, "peak_n": len(peak_indices), "non_peak_n": len(non_peak_indices)}

    for feat_name in peak_features:
        peak_vals = np.array(peak_features[feat_name])
        non_peak_vals = np.array(non_peak_features[feat_name])

        if len(peak_vals) < 2 or len(non_peak_vals) < 2:
            print(f"  [ModuleC Debug] {feat_name}: peak_n={len(peak_vals)}, non_peak_n={len(non_peak_vals)} — 跳过统计")
            results["comparison"][feat_name] = {
                "peak_mean": float(np.mean(peak_vals)) if len(peak_vals) > 0 else np.nan,
                "non_peak_mean": float(np.mean(non_peak_vals)) if len(non_peak_vals) > 0 else np.nan,
                "peak_std": float(np.std(peak_vals)) if len(peak_vals) > 0 else np.nan,
                "non_peak_std": float(np.std(non_peak_vals)) if len(non_peak_vals) > 0 else np.nan,
                "mann_whitney_p": np.nan,
                "cohen_d": np.nan,
                "diff_pct": np.nan,
                "_skipped": True,
            }
            continue

        mean_peak = float(np.mean(peak_vals))
        mean_non = float(np.mean(non_peak_vals))
        std_peak = float(np.std(peak_vals))
        std_non = float(np.std(non_peak_vals))

        try:
            stat, p = mannwhitneyu(peak_vals, non_peak_vals, alternative="two-sided")
        except Exception:
            stat, p = 0.0, 1.0

        cohen = (mean_peak - mean_non) / (np.std(np.concatenate([peak_vals, non_peak_vals])) + 1e-8)

        results["comparison"][feat_name] = {
            "peak_mean": mean_peak,
            "non_peak_mean": mean_non,
            "peak_std": std_peak,
            "non_peak_std": std_non,
            "mann_whitney_p": float(p),
            "cohen_d": float(cohen),
            "diff_pct": float((mean_peak - mean_non) / (abs(mean_non) + 1e-8) * 100),
        }

    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_change_point_results(cp_results: dict, output_dir: str):
    """变点对齐结果可视化。"""
    if not cp_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))

    ax = axes[0]
    labels = ["Near CP", "Far from CP"]
    means = [cp_results["mean_mi_near_cp"], cp_results["mean_mi_far_cp"]]
    stds = [cp_results["std_mi_near_cp"], cp_results["std_mi_far_cp"]]
    colors = [PALETTE["red_strong"], PALETTE["neutral_light"]]
    bars = ax.bar(range(2), means, yerr=stds, color=colors,
                  edgecolor="white", lw=0.5, capsize=3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean MI Score", fontsize=7)
    ax.set_title("MI Near vs Far from Change Points", fontsize=7.5, pad=3)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    add_panel_label(ax, "a")

    ax = axes[1]
    d = cp_results["cohen_d"]
    color = PALETTE["teal"] if d > 0 else PALETTE["neutral_mid"]
    ax.barh(0, d, color=color, edgecolor="white", lw=0.5, height=0.4)
    ax.axvline(0, color=PALETTE["neutral_dark"], ls="--", lw=0.8)
    ax.axvline(0.2, color=PALETTE["neutral_mid"], ls=":", lw=0.8, label="small (0.2)")
    ax.axvline(0.5, color=PALETTE["neutral_mid"], ls="--", lw=0.8, label="medium (0.5)")
    ax.set_xlabel("Cohen's d", fontsize=7)
    ax.set_yticks([0])
    ax.set_yticklabels(["Near - Far"], fontsize=7)
    ax.set_title(f"Effect Size (d={d:.3f})", fontsize=7.5, pad=3)
    ax.grid(True, alpha=0.25, axis="x", lw=0.5)
    ax.legend(fontsize=5, handlelength=1.5)
    add_panel_label(ax, "b")

    ax = axes[2]
    p_vals = {
        "Mann-Whitney": cp_results["mann_whitney_p"],
        "t-test": cp_results["ttest_p"],
        "Permutation": cp_results["permutation_test_p"],
    }
    names = list(p_vals.keys())
    p_log = [-np.log10(max(p, 1e-10)) for p in p_vals.values()]
    bar_colors = [PALETTE["blue_main"] if p < 0.05 else PALETTE["neutral_light"] for p in p_vals.values()]
    bars = ax.bar(names, p_log, color=bar_colors, edgecolor="white", lw=0.5)
    ax.axhline(-np.log10(0.05), color=PALETTE["red_strong"], ls="--", lw=1.0, label="p=0.05")
    ax.axhline(-np.log10(0.01), color=PALETTE["red_strong"], ls=":", lw=1.0, label="p=0.01")
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=7)
    ax.set_title("Statistical Significance", fontsize=7.5, pad=3)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, handlelength=1.5)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    add_panel_label(ax, "c")

    fig.tight_layout(pad=1.0)
    finalize_figure(fig, os.path.join(output_dir, "fig_changepoint_alignment"), dpi=300)


def plot_stl_mi_correlations(stl_results: dict, output_dir: str):
    """STL成分与MI相关性可视化。"""
    if not stl_results or not stl_results.get("per_patch"):
        return

    per_patch = stl_results["per_patch"]
    metrics = list(per_patch.keys())

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))

    ax = axes[0]
    rhos = [per_patch[m]["spearman_rho"] for m in metrics]
    ps = [per_patch[m]["spearman_p"] for m in metrics]
    bar_colors = [
        PALETTE["green_3"] if p < 0.05 else PALETTE["neutral_light"]
        for p in ps
    ]
    x = range(len(metrics))
    ax.bar(x, rhos, color=bar_colors, edgecolor="white", lw=0.5)
    ax.axhline(0, color=PALETTE["neutral_dark"], ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=5, rotation=30, ha="right")
    ax.set_ylabel(r"Spearman $\rho$", fontsize=7)
    ax.set_title("STL Component vs MI (Spearman)", fontsize=7.5, pad=3)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    add_panel_label(ax, "a")

    ax = axes[1]
    rs = [per_patch[m]["pearson_r"] for m in metrics]
    bar_colors = [
        PALETTE["teal"] if p < 0.05 else PALETTE["neutral_light"]
        for p in ps
    ]
    ax.bar(x, rs, color=bar_colors, edgecolor="white", lw=0.5)
    ax.axhline(0, color=PALETTE["neutral_dark"], ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=5, rotation=30, ha="right")
    ax.set_ylabel("Pearson r", fontsize=7)
    ax.set_title("STL Component vs MI (Pearson)", fontsize=7.5, pad=3)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    add_panel_label(ax, "b")

    fig.tight_layout(pad=1.0)
    finalize_figure(fig, os.path.join(output_dir, "fig_stl_mi_correlation"), dpi=300)


def plot_peak_characterization(peak_results: dict, output_dir: str):
    """MI峰值特征分析可视化。"""
    if not peak_results or not peak_results.get("comparison"):
        return

    comp = peak_results["comparison"]
    features = list(comp.keys())
    n_feats = len(features)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))

    ax = axes[0]
    peak_means = [comp[f]["peak_mean"] for f in features]
    non_peak_means = [comp[f]["non_peak_mean"] for f in features]
    x = np.arange(len(features))
    width = 0.35
    ax.bar(x - width / 2, peak_means, width, label="MI Peak",
           color=PALETTE["blue_main"], edgecolor="white", lw=0.5)
    ax.bar(x + width / 2, non_peak_means, width, label="Non-peak",
           color=PALETTE["neutral_light"], edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=5, rotation=30, ha="right")
    ax.set_ylabel("Feature Value", fontsize=7)
    ax.set_title("Peak vs Non-peak Patch Features", fontsize=7.5, pad=3)
    ax.legend(fontsize=6, handlelength=1.5)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    add_panel_label(ax, "a")

    ax = axes[1]
    cohen_ds = [comp[f]["cohen_d"] for f in features]
    bar_colors = [PALETTE["teal"] if d > 0 else PALETTE["neutral_mid"] for d in cohen_ds]
    ax.bar(range(len(features)), cohen_ds, color=bar_colors, edgecolor="white", lw=0.5)
    ax.axhline(0, color=PALETTE["neutral_dark"], ls="--", lw=0.8)
    ax.axhline(0.2, color=PALETTE["neutral_mid"], ls=":", lw=0.8, label="small=0.2")
    ax.axhline(0.5, color=PALETTE["neutral_mid"], ls="--", lw=0.8, label="medium=0.5")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, fontsize=5, rotation=30, ha="right")
    ax.set_ylabel("Cohen's d", fontsize=7)
    ax.set_title("Peak Characterization Effect Sizes", fontsize=7.5, pad=3)
    ax.legend(fontsize=5, handlelength=1.5)
    ax.grid(True, alpha=0.25, axis="y", lw=0.5)
    add_panel_label(ax, "b")

    fig.tight_layout(pad=1.0)
    finalize_figure(fig, os.path.join(output_dir, "fig_peak_characterization"), dpi=300)


def plot_combined_dashboard(
    cp_results, stl_results, peak_results, mi_matrix, output_dir
):
    """汇总仪表盘。"""
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.4)

    ax_hm = fig.add_subplot(gs[0, :2])
    im = ax_hm.imshow(mi_matrix, aspect="auto", cmap="YlOrRd")
    ax_hm.set_xlabel("Patch index", fontsize=6)
    ax_hm.set_ylabel("Layer", fontsize=6)
    ax_hm.set_title("I(H,Y) Heatmap (Timer)", fontsize=7.5, pad=3)
    plt.colorbar(im, ax=ax_hm, shrink=0.8, label="MI (bits)")
    add_panel_label(ax_hm, "a", y=1.15)

    ax_cp = fig.add_subplot(gs[0, 2])
    if cp_results:
        means = [cp_results["mean_mi_near_cp"], cp_results["mean_mi_far_cp"]]
        ax_cp.bar(range(2), means, color=[PALETTE["red_strong"], PALETTE["neutral_light"]],
                  edgecolor="white", lw=0.5)
        ax_cp.set_xticks(range(2))
        ax_cp.set_xticklabels(["Near\nCP", "Far\nCP"], fontsize=6)
        ax_cp.set_ylabel("Mean MI", fontsize=6)
        ax_cp.set_title(f"CP Alignment\nd={cp_results.get('cohen_d', 0):.2f}", fontsize=7, pad=3)
        ax_cp.tick_params(labelsize=5)
        ax_cp.grid(True, alpha=0.25, axis="y", lw=0.5)
    else:
        ax_cp.text(0.5, 0.5, "No CP data", ha="center", va="center",
                   transform=ax_cp.transAxes, fontsize=7)
        ax_cp.set_title("Change Point", fontsize=7, pad=3)
    add_panel_label(ax_cp, "b", y=1.15)

    ax_stl1 = fig.add_subplot(gs[1, 0])
    ax_stl2 = fig.add_subplot(gs[1, 1])
    if stl_results and stl_results.get("per_patch"):
        pp = stl_results["per_patch"]
        metrics = list(pp.keys())
        rhos = [pp[m]["spearman_rho"] for m in metrics]
        rs = [pp[m]["pearson_r"] for m in metrics]
        x = range(len(metrics))
        bar_colors = [PALETTE["green_3"] if pp[m]["spearman_p"] < 0.05 else PALETTE["neutral_light"]
                      for m in metrics]
        ax_stl1.bar(x, rhos, color=bar_colors, edgecolor="white", lw=0.3)
        ax_stl1.axhline(0, ls="--", lw=0.6, color=PALETTE["neutral_dark"])
        ax_stl1.set_xticks(x)
        ax_stl1.set_xticklabels(metrics, fontsize=4, rotation=45, ha="right")
        ax_stl1.set_ylabel(r"Spearman $\rho$", fontsize=6)
        ax_stl1.set_title("STL vs MI", fontsize=7, pad=3)
        ax_stl1.tick_params(labelsize=4)
        ax_stl1.grid(True, alpha=0.25, axis="y", lw=0.5)
        bar_colors2 = [PALETTE["teal"] if pp[m]["pearson_p"] < 0.05 else PALETTE["neutral_light"]
                      for m in metrics]
        ax_stl2.bar(x, rs, color=bar_colors2, edgecolor="white", lw=0.3)
        ax_stl2.axhline(0, ls="--", lw=0.6, color=PALETTE["neutral_dark"])
        ax_stl2.set_xticks(x)
        ax_stl2.set_xticklabels(metrics, fontsize=4, rotation=45, ha="right")
        ax_stl2.set_ylabel("Pearson r", fontsize=6)
        ax_stl2.set_title("STL vs MI (Pearson)", fontsize=7, pad=3)
        ax_stl2.tick_params(labelsize=4)
        ax_stl2.grid(True, alpha=0.25, axis="y", lw=0.5)
    else:
        ax_stl1.text(0.5, 0.5, "No STL data", ha="center", va="center",
                       transform=ax_stl1.transAxes, fontsize=7)
        ax_stl1.set_title("STL vs MI", fontsize=7, pad=3)
        ax_stl2.text(0.5, 0.5, "No STL data", ha="center", va="center",
                      transform=ax_stl2.transAxes, fontsize=7)
        ax_stl2.set_title("STL vs MI (Pearson)", fontsize=7, pad=3)
    add_panel_label(ax_stl1, "c")
    add_panel_label(ax_stl2, "d")

    ax_pk = fig.add_subplot(gs[1, 2])
    if peak_results and peak_results.get("comparison"):
        comp = peak_results["comparison"]
        feats = list(comp.keys())[:5]
        ds = [comp[f]["cohen_d"] for f in feats]
        bar_colors = [PALETTE["teal"] if d > 0.2 else PALETTE["neutral_light"] for d in ds]
        ax_pk.barh(range(len(feats)), ds, color=bar_colors, edgecolor="white", lw=0.3)
        ax_pk.axvline(0, ls="--", lw=0.6, color=PALETTE["neutral_dark"])
        ax_pk.axvline(0.2, ls=":", lw=0.6, color=PALETTE["neutral_mid"])
        ax_pk.set_yticks(range(len(feats)))
        ax_pk.set_yticklabels(feats, fontsize=5)
        ax_pk.set_xlabel("Cohen's d", fontsize=6)
        ax_pk.set_title("Peak Features", fontsize=7, pad=3)
        ax_pk.tick_params(labelsize=5)
        ax_pk.grid(True, alpha=0.25, axis="x", lw=0.5)
    else:
        ax_pk.text(0.5, 0.5, "No peak data", ha="center", va="center",
                    transform=ax_pk.transAxes, fontsize=7)
        ax_pk.set_title("Peak Characterization", fontsize=7, pad=3)
    add_panel_label(ax_pk, "e")

    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    table_data = []
    headers = ["Analysis", "Metric", "Value", "p-value", "Interpretation"]

    if cp_results:
        table_data.append([
            "Change Point",
            "Cohen's d",
            f"{cp_results['cohen_d']:.3f}",
            f"{cp_results['permutation_test_p']:.4f}",
            "Significant" if cp_results['permutation_test_p'] < 0.05 else "N.S.",
        ])

    if stl_results and stl_results.get("per_patch"):
        best_metric = max(stl_results["per_patch"].items(),
                           key=lambda x: abs(x[1]["spearman_rho"]))
        table_data.append([
            "STL",
            f"{best_metric[0]} (Spearman)",
            f"{best_metric[1]['spearman_rho']:.3f}",
            f"{best_metric[1]['spearman_p']:.4f}",
            "Significant" if best_metric[1]["spearman_p"] < 0.05 else "N.S.",
        ])

    if peak_results and peak_results.get("comparison"):
        best_peak = max(peak_results["comparison"].items(),
                         key=lambda x: abs(x[1]["cohen_d"]))
        table_data.append([
            "Peak",
            f"{best_peak[0]} (Cohen d)",
            f"{best_peak[1]['cohen_d']:.3f}",
            f"{best_peak[1]['mann_whitney_p']:.4f}",
            "Significant" if best_peak[1]["mann_whitney_p"] < 0.05 else "N.S.",
        ])

    if table_data:
        tbl = ax_tbl.table(
            cellText=table_data, colLabels=headers,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.5)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor(PALETTE["blue_main"])
                cell.set_text_props(color="white", fontweight="bold")
            elif col == 4:
                val = table_data[row - 1][4]
                cell.set_facecolor(PALETTE["green_3"] if val == "Significant" else PALETTE["neutral_light"])
                cell.set_text_props(color="white" if val == "Significant" else "black")
            else:
                cell.set_facecolor(PALETTE["neutral_light"] if row % 2 == 0 else "white")

    add_panel_label(ax_tbl, "f")

    finalize_figure(fig, os.path.join(output_dir, "fig_changepoint_stl_dashboard"), dpi=300, pad=0.8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Timer MI-变点对齐 & STL分析")
    # Data
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=500,
                        help="最大测试样本数")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--data_type", type=str, default="ETTh1")
    parser.add_argument("--enc_in", type=int, default=None,
                        help="输入变量数（默认从数据集自动推断）")
    # Model
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # MI result
    parser.add_argument("--mi_result_dir", type=str,
                        default="./results",
                        help="包含 global_mi_peaks_*.json 的目录")
    parser.add_argument("--model_id", type=str, default="etth1")
    # Analysis params
    parser.add_argument("--stl_period", type=int, default=24,
                        help="STL 分解周期")
    parser.add_argument("--top_k_ratio", type=float, default=0.15,
                        help="MI峰值比例")
    parser.add_argument("--max_samples_stl", type=int, default=500,
                        help="参与 STL 分解的最大样本数")
    # Runtime
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./results/timer_mi_changepoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"Timer_ChangePoint_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  Timer MI-变点对齐 & STL分析实验")
    print("=" * 70)

    # ── Load MI results from timer_mi_ksg_pca.py ─────────────────────────────
    print("\n>>> Phase 0: 加载 MI 结果...")

    # Auto-discovery: scan multiple likely locations for the MI JSON
    candidates = []
    if os.path.isdir(args.mi_result_dir):
        candidates.append(args.mi_result_dir)

    # Scan root-level Timer_MI_* folders (no prefix)
    for sub in sorted(glob.glob("Timer_MI_*")):
        if os.path.isdir(sub):
            candidates.append(sub)
    # Scan timer_mi_ksg_pca/ sub-folders
    for sub in sorted(glob.glob("timer_mi_ksg_pca/Timer_MI_*")):
        if os.path.isdir(sub):
            candidates.append(sub)
    # Scan results/
    for sub in sorted(glob.glob("results/timer_mi_ksg_pca/Timer_MI_*")):
        if os.path.isdir(sub):
            candidates.append(sub)

    # De-duplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    mi_json_path = None
    mi_json_dir = None
    for cand_dir in unique_candidates:
        pattern = os.path.join(cand_dir, "global_mi_peaks_*.json")
        matched = sorted(glob.glob(pattern))
        if matched:
            # Prefer exact model_id match
            exact = [m for m in matched if args.model_id in os.path.basename(m)]
            if exact:
                mi_json_path = exact[-1]
                mi_json_dir = cand_dir
                break
            # Otherwise take newest
            mi_json_path = matched[-1]
            mi_json_dir = cand_dir
            break

    if mi_json_path is None:
        print(f"  ERROR: 未找到 MI 结果文件 (tried dirs: {unique_candidates})")
        print(f"  请确保先运行 timer_mi_ksg_pca.py 生成 MI 结果。")
        print(f"  或手动指定 --mi_result_dir 指向包含 global_mi_peaks_*.json 的目录。")
        sys.exit(1)

    print(f"  [MI Loader] 使用: {mi_json_path}")

    with open(mi_json_path) as f:
        mi_summary = json.load(f)

    mi_n_layers = mi_summary['num_layers']
    mi_n_patches = mi_summary['N']
    mi_patch_len = mi_summary.get('patch_len', None)
    mi_stride = mi_summary.get('stride', None)
    mi_n_vars = mi_summary.get('n_vars', None)
    layers_dict = mi_summary['layers']

    mi_matrix = np.array([
        layers_dict[str(li)]['hsic_curve'] for li in range(mi_n_layers)
    ])

    print(f"  MI 矩阵: {mi_matrix.shape}, 补丁数={mi_n_patches}, "
          f"patch_len={mi_patch_len}, stride={mi_stride}, n_vars={mi_n_vars}")

    # Compute n_patches from seq_len for validation
    expected_n_patches = args.seq_len // args.patch_len
    if mi_n_patches != expected_n_patches:
        print(f"  [WARN] MI JSON patch 数 (N={mi_n_patches}) 与数据集不匹配 "
              f"(seq_len={args.seq_len} // patch_len={args.patch_len} = {expected_n_patches})！")
        print(f"  [WARN] 这会导致 STL 相关性分析结果为 NaN。")
        print(f"  [WARN] 请使用与当前 seq_len/patch_len 配置一致的 MI 结果。")
        print(f"  [WARN] 当前 MI 结果: patch_len={mi_patch_len}, stride={mi_stride}")
        # Continue anyway so partial results (partial correlation, stratified) still run
    else:
        print(f"  [OK] MI patch 数 ({mi_n_patches}) 与数据集匹配。")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n>>> Phase 1: 加载数据集...")

    test_dataset = CIDatasetBenchmark(
        root_path=os.path.join(args.root_path, args.data_path),
        flag='test',
        input_len=args.seq_len,
        pred_len=args.pred_len,
        data_type=args.data_type,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    n_vars = test_dataset.n_var
    if args.enc_in is not None:
        n_vars_actual = args.enc_in
        print(f"  [WARN] 用户指定 enc_in={n_vars_actual}，数据集报告 n_var={n_vars}")
    else:
        n_vars_actual = n_vars
    print(f"  变量数: {n_vars_actual}, 测试集: {len(test_dataset)}")

    # seq_len divisibility check
    if args.seq_len % args.patch_len != 0:
        print(f"  [WARN] seq_len={args.seq_len} 不是 patch_len={args.patch_len} 的整数倍。")
        n_p = args.seq_len // args.patch_len
        actual_seq_len = n_p * args.patch_len
        print(f"  [INFO] 自动截断为 {actual_seq_len} ({n_p} patches)。请确认数据长度足够。")
    else:
        n_p = args.seq_len // args.patch_len
        actual_seq_len = args.seq_len
        print(f"  seq_len={args.seq_len}, patch_len={args.patch_len} -> n_patches={n_p}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    # ── Load Timer model ─────────────────────────────────────────────────────
    print("\n>>> Phase 2: 加载 Timer 模型...")

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
        enc_in=n_vars_actual,
    )
    model = model.to(device)
    model.eval()

    print(f"  Timer 模型加载完成，设备: {device}")
    print(f"  patch_len={args.patch_len}, d_model={args.d_model}, e_layers={args.e_layers}")

    # ── Extract representations ───────────────────────────────────────────────
    print("\n>>> Phase 3: 提取 patch 表示...")

    hist_tokens, raw_patches, n_patches = extract_raw_patches_and_mi(
        model=model,
        data_loader=test_loader,
        num_layers=args.e_layers,
        patch_len=args.patch_len,
        n_vars=n_vars_actual,
        device=device,
        max_samples=args.max_samples,
    )

    print(f"  历史 tokens: {[t.shape for t in hist_tokens]}")
    print(f"  原始 patches: {raw_patches.shape}")

    all_results = {}

    # ============================================================
    # Module A: Change Point Alignment
    # ============================================================
    print("\n" + "=" * 70)
    print("  模块A: 变点检测与 MI 峰值对齐")
    print("=" * 70)

    cp_results = align_mi_peaks_with_change_points(
        mi_matrix=mi_matrix,
        raw_patches=raw_patches,
        n_patches=n_patches,
        patch_len=args.patch_len,
        seq_len=actual_seq_len,
        max_samples=args.max_samples,
        mi_n_patches=mi_n_patches,
    )

    if cp_results:
        all_results["changepoint"] = cp_results
        print(f"\n  Near CP: {cp_results['mean_mi_near_cp']:.4f} ± {cp_results['std_mi_near_cp']:.4f}")
        print(f"  Far CP:  {cp_results['mean_mi_far_cp']:.4f} ± {cp_results['std_mi_far_cp']:.4f}")
        print(f"  Cohen's d: {cp_results['cohen_d']:.4f}")
        print(f"  Permutation p: {cp_results['permutation_test_p']:.4f}")

    # ============================================================
    # Module B: STL Semantic Components
    # ============================================================
    print("\n" + "=" * 70)
    print("  模块B: STL 语义成分与 MI 对应关系")
    print("=" * 70)

    stl_results = compute_stl_components_per_patch(
        raw_patches=raw_patches,
        mi_scores=mi_matrix,
        seq_len=actual_seq_len,
        stl_period=args.stl_period,
        max_samples_for_stl=args.max_samples_stl,
    )

    if stl_results:
        all_results["stl"] = stl_results
        print("\n  STL vs MI 相关性:")
        for metric, vals in stl_results.get("per_patch", {}).items():
            sig = "***" if vals["spearman_p"] < 0.001 else "**" if vals["spearman_p"] < 0.01 else "*" if vals["spearman_p"] < 0.05 else ""
            print(f"    {metric}: Spearman rho={vals['spearman_rho']:.4f}{sig}, Pearson r={vals['pearson_r']:.4f}")

    # Partial Correlation + Stratified Analysis
    if stl_results and "_sample_level" in stl_results:
        sl = stl_results["_sample_level"]

        print("\n  偏相关分析: raw_kurtosis vs MI (控制 trend_share)...")
        kurt_flat = sl["raw_kurtosis"].flatten()
        ts_flat   = sl["trend_share"].flatten()
        mi_flat   = sl["mi"].flatten()
        valid = ~(np.isnan(kurt_flat) | np.isnan(ts_flat) | np.isnan(mi_flat))
        kurt_v, ts_v, mi_v = kurt_flat[valid], ts_flat[valid], mi_flat[valid]
        n_flat = len(kurt_v)

        if n_flat >= 50:
            from scipy.stats import pearsonr
            r_kurt_mi, _ = pearsonr(kurt_v, mi_v)
            r_kurt_ts, _ = pearsonr(kurt_v, ts_v)
            r_mi_ts, _   = pearsonr(mi_v, ts_v)
            denom = np.sqrt(max(1e-10, (1 - r_kurt_ts**2) * (1 - r_mi_ts**2)))
            r_partial = (r_kurt_mi - r_kurt_ts * r_mi_ts) / denom

            rng_boot = np.random.default_rng(42)
            boot_partials = []
            for _ in range(1000):
                idx = rng_boot.integers(0, n_flat, n_flat)
                r_km, _ = pearsonr(kurt_v[idx], mi_v[idx])
                r_kt, _ = pearsonr(kurt_v[idx], ts_v[idx])
                r_mt, _ = pearsonr(mi_v[idx], ts_v[idx])
                d = np.sqrt(max(1e-10, (1 - r_kt**2) * (1 - r_mt**2)))
                boot_partials.append((r_km - r_kt * r_mt) / d)
            boot_partials = np.array(boot_partials)
            ci_low, ci_high = float(np.percentile(boot_partials, 2.5)), float(np.percentile(boot_partials, 97.5))

            all_results["partial_correlation"] = {
                "raw_kurtosis__vs__mi": {
                    "controlling_for": "trend_share",
                    "partial_r": float(r_partial),
                    "simple_r": float(r_kurt_mi),
                    "ci_95": [ci_low, ci_high],
                    "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
                    "n_flat": int(n_flat),
                }
            }
            sig_str = "不显著" if ci_low <= 0 <= ci_high else "显著"
            print(f"    样本×patch总点数: {n_flat}")
            print(f"    简单相关: r(kurt,MI)={r_kurt_mi:.4f}")
            print(f"    偏相关 r(kurt,MI | trend_share)={r_partial:.4f} "
                  f"[95% CI: {ci_low:.4f}, {ci_high:.4f}] ({sig_str})")

        # ── 分层相关性（高 MI patch vs 低 MI patch 特征对比）─────────────────
        print("\n  分层分析: 高 MI patch vs 低 MI patch 特征对比 (per-sample paired)...")
        feature_names = [
            "trend_strength", "seasonal_strength", "trend_share",
            "seasonal_share", "residual_share", "residual_std",
            "raw_variance", "raw_skewness", "raw_kurtosis",
        ]

        mi_patch = sl["mi"][0]
        n_p = len(mi_patch)

        # 方法1：Q1 vs Q4 四分位（极值对比，效应量最大）
        q25 = float(np.nanpercentile(mi_patch, 25))
        q75 = float(np.nanpercentile(mi_patch, 75))
        q1_patch = mi_patch <= q25
        q4_patch = mi_patch >= q75
        n_q1 = int(q1_patch.sum())
        n_q4 = int(q4_patch.sum())

        # 方法2：中位数（对半切）
        mi_med_patch = float(np.nanmedian(mi_patch))
        med_high_patch = mi_patch >= mi_med_patch
        med_low_patch  = mi_patch <  mi_med_patch
        n_high_patch = int(med_high_patch.sum())
        n_low_patch  = int(med_low_patch.sum())

        print(f"    [Q1/Q4 极值分组] Q1(n={n_q1}, MI≤{q25:.4f}) vs Q4(n={n_q4}, MI≥{q75:.4f})")
        print(f"    [中位数分组] high(n={n_high_patch}, MI≥{mi_med_patch:.4f}) vs low(n={n_low_patch})")

        strat_results = {
            "q1_q4": {"n_q1": n_q1, "n_q4": n_q4, "q25": q25, "q75": q75, "groups": {}},
            "median": {"n_high": n_high_patch, "n_low": n_low_patch, "median": mi_med_patch, "groups": {}},
        }

        def _run_paired_analysis(feat, idx_a, idx_b, label_a, label_b):
            """对一对 patch 索引做配对分析。返回均值差、效应量、p值。"""
            from scipy.stats import ttest_rel, wilcoxon

            a_mean = np.nanmean(feat[:, idx_a], axis=1)
            b_mean = np.nanmean(feat[:, idx_b], axis=1)
            diff = a_mean - b_mean
            valid = ~np.isnan(diff)
            if valid.sum() < 10:
                return None

            mean_diff = float(np.mean(diff[valid]))
            std_diff  = float(np.std(diff[valid], ddof=1))
            sem_diff  = std_diff / np.sqrt(valid.sum())

            coh_d = mean_diff / (std_diff + 1e-10)

            # Cliff's delta (非参效应量)
            a_v, b_v = a_mean[valid], b_mean[valid]
            n1, n2 = len(a_v), len(b_v)
            dom = 0.0
            for x in a_v:
                for y in b_v:
                    if x > y: dom += 1
                    elif x < y: dom -= 1
            cliffs_delta = dom / (n1 * n2)

            t_stat, t_p = ttest_rel(a_mean[valid], b_mean[valid])
            try:
                w_stat, w_p = wilcoxon(a_mean[valid], b_mean[valid])
                w_sig = bool(w_p < 0.05)
            except Exception:
                w_p = np.nan; w_sig = False
            t_sig = bool(t_p < 0.05)

            return {
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "sem": float(sem_diff),
                "cohen_d": coh_d,
                "cliffs_delta": cliffs_delta,
                "t_stat": float(t_stat),
                "ttest_p": float(t_p),
                "wilcoxon_p": float(w_p) if not np.isnan(w_p) else np.nan,
                "ttest_sig": t_sig,
                "wilcoxon_sig": w_sig,
                "n_valid": int(valid.sum()),
                "mean_a": float(np.mean(a_mean[valid])),
                "mean_b": float(np.mean(b_mean[valid])),
            }

        print(f"\n  {'特征':20s}  {'Δmean±SEM':>14s}  {'Cohen d':>9s}  {'Cliff δ':>7s}  {'t_p':>6s}  {'W_p':>6s}  {'显著?'}")
        print(f"  {'-'*80}")

        for feat_name in feature_names:
            feat = sl[feat_name]

            r_q  = _run_paired_analysis(feat, q4_patch, q1_patch, "Q4", "Q1")
            r_md = _run_paired_analysis(feat, med_high_patch, med_low_patch, "High", "Low")

            strat_results["q1_q4"]["groups"][feat_name]  = r_q
            strat_results["median"]["groups"][feat_name] = r_md

            def _fmt(r, sig_only=False):
                if r is None: return "  N/A  "
                sig = "*" if r["ttest_sig"] else " "
                return f"{r['mean_diff']:+.4f}±{r['sem']:.4f}  {r['cohen_d']:>8.3f}  {r['cliffs_delta']:>7.3f}  {r['ttest_p']:.4f}  {r['wilcoxon_p']:.4f}  {sig}"

            q_line  = f"  [Q1/Q4] {feat_name:20s}  {_fmt(r_q)}"
            md_line = f"  [中位数] {feat_name:20s}  {_fmt(r_md)}"
            print(q_line)
            print(md_line)

        all_results["stratified_correlation"] = strat_results

    # ============================================================
    # Module C: MI Peak Characterization
    # ============================================================
    print("\n" + "=" * 70)
    print("  模块C: MI 峰值特征分析")
    print("=" * 70)

    peak_results = characterize_mi_peaks(
        mi_matrix=mi_matrix,
        raw_patches=raw_patches,
        n_patches=n_patches,
        patch_len=args.patch_len,
        top_k_ratio=args.top_k_ratio,
        mi_n_patches=mi_n_patches,
    )

    if peak_results:
        all_results["peak"] = peak_results
        print(f"\n  Peak patches: {peak_results['peak_n']}, Non-peak: {peak_results['non_peak_n']}")
        for feat, vals in peak_results.get("comparison", {}).items():
            skipped = vals.get("_skipped", False)
            if skipped:
                print(f"    {feat}: SKIPPED (peak_n={int(peak_results['peak_n'])}, "
                      f"peak_mean={vals['peak_mean']:.4f}, non_peak_mean={vals['non_peak_mean']:.4f})")
            else:
                sig = "***" if vals["mann_whitney_p"] < 0.001 else "**" if vals["mann_whitney_p"] < 0.01 else "*" if vals["mann_whitney_p"] < 0.05 else ""
                print(f"    {feat}: d={vals['cohen_d']:.3f}{sig}")

    # ── Plotting ────────────────────────────────────────────────────────────
    print("\n>>> Phase 4: 绘图...")

    if all_results.get("changepoint"):
        plot_change_point_results(all_results["changepoint"], output_dir)

    if all_results.get("stl"):
        plot_stl_mi_correlations(all_results["stl"], output_dir)

    if all_results.get("peak"):
        plot_peak_characterization(all_results["peak"], output_dir)

    plot_combined_dashboard(
        all_results.get("changepoint", {}),
        all_results.get("stl", {}),
        all_results.get("peak", {}),
        mi_matrix, output_dir,
    )

    # ── Save results ──────────────────────────────────────────────────────
    print("\n>>> Phase 5: 保存结果...")

    def make_serializable(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    results_clean = json.loads(json.dumps(all_results, default=make_serializable))
    results_path = os.path.join(output_dir, "changepoint_stl_results.json")
    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"  结果已保存: {results_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)

    if cp_results:
        print(f"\n  [变点对齐] Cohen's d={cp_results['cohen_d']:.4f}, perm-p={cp_results['permutation_test_p']:.4f}")
    if stl_results and stl_results.get("per_patch"):
        best = max(stl_results["per_patch"].items(), key=lambda x: abs(x[1]["spearman_rho"]))
        print(f"\n  [STL] Best correlation: {best[0]} (rho={best[1]['spearman_rho']:.4f}, p={best[1]['spearman_p']:.4f})")
    if peak_results and peak_results.get("comparison"):
        best_peak = max(peak_results["comparison"].items(), key=lambda x: abs(x[1]["cohen_d"]))
        print(f"\n  [Peak] Best effect: {best_peak[0]} (d={best_peak[1]['cohen_d']:.4f})")

    print(f"\n  输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
