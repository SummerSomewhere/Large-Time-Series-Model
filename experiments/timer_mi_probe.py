#!/usr/bin/env python3
"""
Timer 层间 MI + Linear Probing 实验

第一组（基础架构与 Baseline）:
  - 生成 7 种合成时间序列概念数据集 (AR1, Level Shift, Random Walk, Spectral,
    Time Warped, Trend, Variance Shift)
  - 每层提取所有 Patch 的隐藏状态 H^{(l)} ∈ [B × S × D]
  - 层内均值池化得到 z^{(l)} ∈ [B × D]
  - 为每个概念独立训练一个线性探针（Ridge Regression），输入 z^{(l)}，目标为 θ
  - 计算每层 MSE 和 R²，绘制类似论文 Figure 2 的双纵轴图

第二组（锚点 vs 噪声对比实验）:
  - 基于 SPI 或 I(Hx, hY) 指标筛选 Anchor 组（Top 10%）和 Noise 组（Bottom 10%）
  - 对每层分别做 token 筛选，不均化，展平为独立样本 (B*S, D)
  - 对应的 θ 复制 S 倍
  - 分别为 Anchor 组和 Noise 组训练独立 Ridge 探针
  - 将两组 MSE/R² 曲线也绘制在第一组图表中进行对比

Usage:
  python experiments/timer_mi_probe.py \
    --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
    --out_dir ./outputs/timer_mi_probe/ \
    --e_layers 8 --seq_len 672 --pred_len 96 --patch_len 96 \
    --n_samples 2048 --select_metric spi --run_group2

  # 快速调试（随机模型，较小规模）:
  python experiments/timer_mi_probe.py \
    --ckpt_path random --out_dir ./outputs/timer_mi_probe_debug/ \
    --e_layers 4 --seq_len 192 --pred_len 48 --patch_len 48 \
    --n_samples 512 --select_metric ihy --n_concepts 3
"""

import argparse
import json
import math
import os
import gc
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as ss
import scipy.special as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model


# ─── Concept names ────────────────────────────────────────────────────────────

CONCEPT_NAMES = [
    "AR1", "LevelShift", "RandomWalk", "Spectral",
    "TimeWarp", "Trend", "VarianceShift",
]
CONCEPT_DISPLAY = {
    "AR1":           "AR(1)",
    "LevelShift":    "Level Shift",
    "RandomWalk":    "Random Walk",
    "Spectral":      "Spectral",
    "TimeWarp":      "Time Warped",
    "Trend":         "Trend",
    "VarianceShift": "Variance Shift",
}

METRIC_DISPLAY = {
    "spi": "SPI",
    "ihy": "I(h_x, h_y)",
    "ixh": "I(e_x, h_x)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Synthetic Dataset Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class SyntheticSingleConceptDataset(Dataset):
    """
    每个样本是一个单独的概念实例。
    返回 (seq_x, seq_y, concept_name, param_value)
      - seq_x: [1, seq_len]
      - seq_y: [1, pred_len]
      - concept_name: str
      - param_value: float
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_samples_per_concept: int,
        concepts: List[str],
        split: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.concepts = concepts
        self.n_c = len(concepts)
        self.n_per = n_samples_per_concept
        self.n_total = self.n_c * self.n_per
        self.rng = _rng(seed + hash(split) % 1000)

        # Generate per-sample params so each sample has a unique theta
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Tuple[np.ndarray, np.ndarray, str, float]]:
        samples = []
        for ci, cname in enumerate(self.concepts):
            seed_i = 42 + ci * 100 + hash(cname) % 1000
            rng_i = _rng(seed_i)
            for _ in range(self.n_per):
                seq, param = self._gen_one(cname, rng_i)
                samples.append((seq[:self.seq_len], seq[self.seq_len:], cname, param))
        return samples

    def _gen_one(self, cname: str, rng
                 ) -> Tuple[np.ndarray, float]:
        seg_len = max(self.total_len * 3, 8192)
        if cname == "AR1":
            phi = rng.uniform(0.3, 0.9)
            x = np.zeros(seg_len)
            x[0] = rng.normal(0, 1.0)
            for t in range(1, seg_len):
                x[t] = phi * x[t - 1] + rng.normal(0, 0.5)
            x = (x - x.mean()) / (x.std() + 1e-8)
            return x, float(phi)
        elif cname == "LevelShift":
            tau = rng.integers(int(seg_len * 0.3), int(seg_len * 0.7))
            delta = rng.uniform(-3.0, 3.0)
            x = rng.normal(0, 0.2, seg_len)
            x[tau:] += delta
            return x, float(delta)
        elif cname == "RandomWalk":
            mu = rng.uniform(-0.05, 0.05)
            sigma = rng.uniform(0.3, 1.0)
            x = np.zeros(seg_len)
            for t in range(1, seg_len):
                x[t] = x[t - 1] + mu + rng.normal(0, sigma)
            return x, float(mu)
        elif cname == "Spectral":
            freq = rng.uniform(0.05, 0.3)
            amp = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * math.pi)
            t = np.arange(seg_len)
            x = amp * np.sin(2 * math.pi * freq * t + phase) + rng.normal(0, 0.05, seg_len)
            x = (x - x.mean()) / (x.std() + 1e-8)
            return x, float(freq)
        elif cname == "TimeWarp":
            freq = rng.uniform(0.05, 0.2)
            phase = rng.uniform(0, 2 * math.pi)
            warp_s = rng.uniform(0.5, 2.0)
            steps = rng.gamma(warp_s, scale=1.0, size=seg_len)
            u = np.cumsum(steps)
            u = (u - u.min()) / (u.max() - u.min() + 1e-8) * (seg_len - 1)
            base = np.sin(2 * math.pi * freq * np.arange(seg_len) + phase)
            x = np.interp(np.arange(seg_len), u, base) + rng.normal(0, 0.05, seg_len)
            x = (x - x.mean()) / (x.std() + 1e-8)
            return x, float(warp_s)
        elif cname == "Trend":
            beta = rng.uniform(-0.05, 0.05)
            x = beta * np.arange(seg_len) + rng.normal(0, 0.2, seg_len)
            x = (x - x.mean()) / (x.std() + 1e-8)
            return x, float(beta)
        elif cname == "VarianceShift":
            tau = rng.integers(int(seg_len * 0.3), int(seg_len * 0.7))
            s1 = rng.uniform(0.3, 1.0)
            s2 = rng.uniform(0.3, 1.0)
            x = np.zeros(seg_len)
            x[:tau] = rng.normal(0, s1, tau)
            x[tau:] = rng.normal(0, s2, seg_len - tau)
            return x, float(s2 / (s1 + 1e-8))
        raise ValueError(f"Unknown concept: {cname}")

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx: int):
        sx, sy, cname, param = self.samples[idx]
        return (
            torch.tensor(sx, dtype=torch.float32).unsqueeze(0),
            torch.tensor(sy, dtype=torch.float32).unsqueeze(0),
            cname,
            param,
        )


def collate_single(sequences):
    """Stack samples as-is for Timer ([B, 1, T])."""
    xs, ys, cnames, params = zip(*sequences)
    return (
        torch.stack(xs, dim=0),
        torch.stack(ys, dim=0),
        list(cnames),
        np.array(params, dtype=np.float32),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: KSG MI Estimator
# ═══════════════════════════════════════════════════════════════════════════════

def _mi_worker_single(task):
    """
    KSG worker for a single (layer, patch, concept) slice.
    Computes I(e_x, h_x) and I(h_x, h_y) for one patch position.
    Must be top-level for multiprocessing pickling.
    """
    import numpy as np
    import scipy.spatial as ss
    import scipy.special as sp

    li, pi, ex_patch, hx_patch, hy_patch, k = task
    N = hx_patch.shape[0]

    # I(e_x; h_x) at this patch
    xy = np.concatenate((ex_patch, hx_patch), axis=1)
    tree_xy = ss.cKDTree(xy)
    tree_x = ss.cKDTree(ex_patch)
    tree_y = ss.cKDTree(hx_patch)

    dist_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.maximum(dist_xy[:, k] - 1e-10, 0)
    dist_x, _ = tree_x.query(ex_patch, k=k + 1, p=np.inf)
    dist_y, _ = tree_y.query(hx_patch, k=k + 1, p=np.inf)

    nx = np.maximum(np.sum(dist_x < eps[:, None], axis=1) - 1, 0)
    ny = np.maximum(np.sum(dist_y < eps[:, None], axis=1) - 1, 0)
    m_exh = max(0.0, (sp.digamma(k) - np.mean(sp.digamma(nx + 1) + sp.digamma(ny + 1)) + sp.digamma(N)) / math.log(2))

    # I(h_x; h_y) at this patch
    xy2 = np.concatenate((hx_patch, hy_patch), axis=1)
    tree_xy2 = ss.cKDTree(xy2)
    tree_h = ss.cKDTree(hx_patch)
    tree_p = ss.cKDTree(hy_patch)

    dist_xy2, _ = tree_xy2.query(xy2, k=k + 1, p=np.inf)
    eps2 = np.maximum(dist_xy2[:, k] - 1e-10, 0)
    dist_h, _ = tree_h.query(hx_patch, k=k + 1, p=np.inf)
    dist_p, _ = tree_p.query(hy_patch, k=k + 1, p=np.inf)

    nh = np.maximum(np.sum(dist_h < eps2[:, None], axis=1) - 1, 0)
    np_ = np.maximum(np.sum(dist_p < eps2[:, None], axis=1) - 1, 0)
    m_hxh = max(0.0, (sp.digamma(k) - np.mean(sp.digamma(nh + 1) + sp.digamma(np_ + 1)) + sp.digamma(N)) / math.log(2))

    return li, pi, m_exh, m_hxh


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Model builder
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_timer(ckpt_path, patch_len, stride, d_model, d_ff, e_layers,
                 n_heads, dropout, seq_len, pred_len):
    cfg = Config(
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
        is_finetuning=0, label_len=pred_len,
        output_len=pred_len, batch_size=64,
        train_epochs=1, patience=3, learning_rate=3e-5,
        itr=1, use_ims=False, inverse=False,
        use_align_loss=False, align_loss_layers=list(range(e_layers)),
    )
    model = Model(cfg)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Token Extraction + per-token MI
# ═══════════════════════════════════════════════════════════════════════════════

def extract_layer_tokens_and_mi(model, data_loader, device, n_layers,
                                 patch_len, k_neighbors=5, pca_dim=32,
                                 sample_ratio=1.0):
    """
    Returns a list of dicts (one per layer):
      layer_data[l] = {
        "hist_tokens":  Tensor [N, n_patches, D],   per-layer hidden state from history
        "x_emb_tokens": Tensor [N, n_patches, D],   patch embedding (Encoder input)
        "param_tokens":  Tensor [N, D],              first-token param rep per layer
        "ixh":          np [n_patches],   I(e_x, h_x) per patch (shared across concepts)
        "ihxh":         np [n_patches],   I(h_x, h_y) per patch (shared across concepts)
        "spi":          np [n_patches],   I(h_x, h_y) / (I(e_x, h_x) + bias)
        "n_patches":    int,
        "n":            int,
        "D":            int,
        "cname":        list[str],
        "param":        np [N],
      }
    """
    core = _unwrap(model)
    rng = np.random.default_rng(42)

    all_hist = [[] for _ in range(n_layers)]
    all_param_rep = [[] for _ in range(n_layers)]
    all_emb = []
    all_cnames, all_params = [], []

    n_patches = None

    with torch.no_grad():
        for seq_x, seq_y, cnames, params in tqdm(data_loader, desc="提取 token"):
            B = seq_x.shape[0]

            sx = seq_x.float().to(device)
            sy = seq_y.float().to(device)

            xm = sx.mean(-1, keepdim=True).detach()
            xs = torch.sqrt(torch.var(sx, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
            xn = (sx - xm) / xs

            # History branch: patch embed -> decoder layers
            dx, _ = core.enc_embedding(xn)
            BM, N, D = dx.shape

            if n_patches is None:
                n_patches = N
                print(f"  B={B}, N={N}, D={D}")

            all_emb.append(dx.float().cpu())

            hx = dx
            for li, layer in enumerate(core.decoder.attn_layers):
                hx, _, _ = layer(hx, attn_mask=None)
                all_hist[li].append(hx.float().cpu())

            # Param branch: param scalar -> expand to [B, 1, seq_len] -> enc_embedding -> decoder layers
            # h_y^{(l)} = mean pool over all patches after decoder layer l (逐层累积)
            # Skip normalization: a constant signal has zero variance, normalization yields NaN
            param_seq = torch.tensor(params, dtype=torch.float32, device=device)  # [B]
            param_expanded = param_seq.unsqueeze(1).unsqueeze(2)                  # [B, 1, 1]
            param_expanded = param_expanded.expand(-1, 1, dx.shape[2])            # [B, 1, seq_len]
            # Scale so the constant has non-trivial magnitude for patch embedding
            dy, _ = core.enc_embedding(param_expanded * 1e2)
            hy = dy
            for li, layer in enumerate(core.decoder.attn_layers):
                hy, _, _ = layer(hy, attn_mask=None)
                # Mean pool over all patches for this layer's param representation
                all_param_rep[li].append(hy.mean(dim=1).float().cpu())

            all_cnames.extend(cnames)
            all_params.extend(params.tolist() if isinstance(params, np.ndarray) else params)

            del dx, hx, dy, hy, param_expanded
            gc.collect()
            torch.cuda.empty_cache()

    hist_list = [torch.cat(toks, dim=0) for toks in all_hist]
    param_rep_list = [torch.cat(toks, dim=0) for toks in all_param_rep]
    x_emb = torch.cat(all_emb, dim=0)
    cnames_arr = np.array(all_cnames)
    params_arr = np.array(all_params, dtype=np.float32)

    N_total = x_emb.shape[0]
    print(f"  Total N={N_total}, n_patches={n_patches}, D={x_emb.shape[2]}")

    n_s = N_total
    if sample_ratio < 1.0:
        n_s = max(int(N_total * sample_ratio), 100)
        idx = np.sort(rng.choice(N_total, size=n_s, replace=False))
        hist_list = [h[idx] for h in hist_list]
        param_rep_list = [p[idx] for p in param_rep_list]
        x_emb = x_emb[idx]
        cnames_arr = cnames_arr[idx]
        params_arr = params_arr[idx]

    # ── Per-concept MI computation ─────────────────────────────────────────────
    # I(e_x; h_x) and I(h_x; h_y) for each (layer, patch, concept) slice.
    # PCA is fit per slice; bias is per-concept mean of I(e_x, h_x).
    print(f"  Computing per-concept I(e_x,h_x)+I(h_x,h_y)+SPI for {n_s} samples...")

    unique_concepts = list(set(cnames_arr.tolist()))
    ixh_concept = {}   # {cname: [n_layers, n_patches]}
    ihxh_concept = {}
    spi_concept = {}

    n_workers = min(mp.cpu_count(), n_layers * n_patches)
    print(f"  Distributing {n_layers * n_patches} tasks across {n_workers} CPU workers...")

    for cname in unique_concepts:
        c_mask_global = cnames_arr == cname
        N_c = int(c_mask_global.sum())
        print(f"\n  Concept [{cname}]: N={N_c}")

        x_emb_c = x_emb.numpy()[c_mask_global]
        hist_c = [h.numpy()[c_mask_global] for h in hist_list]
        # h_y: first token from each layer's param representation
        param_rep_c = [p.numpy()[c_mask_global] for p in param_rep_list]

        ixh_c = np.zeros((n_layers, n_patches))
        ihxh_c = np.zeros((n_layers, n_patches))

        all_tasks = []
        for li in range(n_layers):
            hl = hist_c[li]
            pl = param_rep_c[li]

            for pi in range(n_patches):
                ex_patch = x_emb_c[:, pi, :]
                hx_patch = hl[:, pi, :]
                hy_patch = pl
                all_tasks.append((li, pi, ex_patch, hx_patch, hy_patch, k_neighbors))

        task_results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_mi_worker_single, t): (t[0], t[1]) for t in all_tasks}
            for res in tqdm(as_completed(futures), total=len(futures), desc=f"  {cname} MI"):
                li_r, pi_r, m_exh, m_hxh = res.result()
                task_results[(li_r, pi_r)] = (m_exh, m_hxh)

        for li in range(n_layers):
            for pi in range(n_patches):
                m_exh, m_hxh = task_results[(li, pi)]
                ixh_c[li, pi] = m_exh
                ihxh_c[li, pi] = m_hxh

        bias_c = 0.15 * float(ixh_c.mean())
        spi_c = ihxh_c / (ixh_c + bias_c + 1e-12)
        print(f"    I(e_x,h_x) mean={ixh_c.mean():.4f}, bias={bias_c:.4f}, "
              f"I(h_x,h_y) mean={ihxh_c.mean():.4f}, SPI mean={spi_c.mean():.4f}")

        ixh_concept[cname] = ixh_c
        ihxh_concept[cname] = ihxh_c
        spi_concept[cname] = spi_c

    # Build layer_data: ixh/ihxh/spi are per-concept dicts now
    layer_data = []
    for li in range(n_layers):
        layer_data.append({
            "hist_tokens":  hist_list[li],
            "x_emb_tokens": x_emb,
            "param_tokens": param_rep_list[li],
            "ixh":          ixh_concept,
            "ihxh":         ihxh_concept,
            "spi":          spi_concept,
            "n_patches":    n_patches,
            "n":            n_s,
            "D":            x_emb.shape[2],
            "cname":        cnames_arr.tolist(),
            "param":        params_arr,
        })

    return layer_data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Probing — Group 1 Baseline (Mean Pooling)
# ═══════════════════════════════════════════════════════════════════════════════

def probe_baseline(layer_data, concepts, alpha=1.0, train_frac=0.8, seed=42):
    """
    Group 1: layer-wise mean pool within each sample + Ridge probe.

    For each layer and each concept:
      - Filter samples belonging to that concept
      - Mean pool all patches per sample -> (n_c, D)
      - Split train/test, train Ridge, compute MSE + R²
    """
    rng = np.random.default_rng(seed)
    results = {}

    for cname in concepts:
        mse_per_layer, r2_per_layer = [], []

        for li, ld in enumerate(layer_data):
            c_mask = np.array(ld["cname"]) == cname
            if not c_mask.any():
                mse_per_layer.append(float("nan"))
                r2_per_layer.append(float("nan"))
                continue

            # Mean pool all patches within each sample
            H = ld["hist_tokens"].numpy()[c_mask]   # [n_c, n_patches, D]
            z = H.mean(axis=1)                        # [n_c, D]
            theta = ld["param"][c_mask]               # [n_c]

            # Z-score normalize theta for cross-concept fairness
            th_mean = theta.mean()
            th_std = theta.std(ddof=1)
            theta_norm = (theta - th_mean) / (th_std + 1e-12)

            n = len(z)
            perm = rng.permutation(n)
            n_train = max(1, int(n * train_frac))
            z_tr = z[perm[:n_train]]
            z_te = z[perm[n_train:]]
            th_tr = theta_norm[perm[:n_train]]
            th_te = theta_norm[perm[n_train:]]

            if len(z_te) == 0 or z_tr.shape[0] < 3:
                mse_per_layer.append(float("nan"))
                r2_per_layer.append(float("nan"))
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(z_tr, th_tr)
            pred = ridge.predict(z_te)

            mse = float(np.mean((pred - th_te) ** 2))
            ss_r = np.sum((th_te - pred) ** 2)
            ss_t = np.sum((th_te - th_te.mean()) ** 2)
            r2 = float(1 - ss_r / (ss_t + 1e-12))

            mse_per_layer.append(mse)
            r2_per_layer.append(r2)

        results[cname] = {
            "layer_mse": mse_per_layer,
            "layer_r2":  r2_per_layer,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Probing — Group 2 Anchor vs Noise
# ═══════════════════════════════════════════════════════════════════════════════

def probe_anchor_vs_noise(layer_data, concepts, metric="spi",
                            anchor_ratio=0.10, alpha=1.0,
                            train_frac=0.8, seed=42):
    """
    Group 2: per-sample mean of high/low SPI/I(h_x,h_y)/I(e_x,h_x) tokens + Ridge probe.

    For each layer and each concept:
      1. Filter samples belonging to that concept
      2. Look up per-concept scores dict (spi / ihxh / ixh) for this layer
      3. Pick top/bottom ratio patch positions as anchor/noise
      4. Per sample: mean-pool selected tokens -> z [n_c, D]
         (if no tokens selected in a sample, skip it)
      5. Train Ridge per group

    Returns (anchor_results, noise_results)
    """
    rng = np.random.default_rng(seed)
    anchor_res, noise_res = {}, {}

    for cname in concepts:
        anc_mse, anc_r2 = [], []
        noi_mse, noi_r2 = [], []

        for li, ld in enumerate(layer_data):
            c_mask = np.array(ld["cname"]) == cname
            if not c_mask.any():
                anc_mse.append(float("nan")); anc_r2.append(float("nan"))
                noi_mse.append(float("nan")); noi_r2.append(float("nan"))
                continue

            if metric == "spi":
                scores_all = ld["spi"][cname]
            elif metric == "ihy":
                scores_all = ld["ihxh"][cname]
            else:
                scores_all = ld["ixh"][cname]

            H = ld["hist_tokens"].numpy()[c_mask]   # [n_c, n_patches, D]
            theta = ld["param"][c_mask]              # [n_c]
            S = ld["n_patches"]
            n_c = H.shape[0]

            th_mean = theta.mean()
            th_std = theta.std(ddof=1)
            theta_norm = (theta - th_mean) / (th_std + 1e-12)

            n_select = max(1, int(S * anchor_ratio))

            if S <= n_select:
                mask_a = np.ones(S, dtype=bool)
                mask_n = np.ones(S, dtype=bool)
            else:
                scores = scores_all[li]   # [n_patches], per-layer scores
                hi_thr = np.partition(scores, -n_select)[-n_select]
                lo_thr = np.partition(scores, n_select - 1)[n_select - 1]
                mask_a = scores >= hi_thr
                mask_n = scores <= lo_thr

            # Per-sample mean of selected tokens
            z_anc = H[:, mask_a].mean(axis=1)   # [n_c, D]
            z_noi = H[:, mask_n].mean(axis=1)   # [n_c, D]

            def train_and_eval(z, y):
                n = len(z)
                if n < 3:
                    return float("nan"), float("nan")
                perm = rng.permutation(n)
                n_tr = max(1, int(n * train_frac))
                z_tr, z_te = z[perm[:n_tr]], z[perm[n_tr:]]
                y_tr, y_te = y[perm[:n_tr]], y[perm[n_tr:]]
                if len(z_te) == 0 or z_tr.shape[0] < 3:
                    return float("nan"), float("nan")
                ridge = Ridge(alpha=alpha)
                ridge.fit(z_tr, y_tr)
                pred = ridge.predict(z_te)
                mse = float(np.mean((pred - y_te) ** 2))
                ss_r = np.sum((y_te - pred) ** 2)
                ss_t = np.sum((y_te - y_te.mean()) ** 2)
                r2 = float(1 - ss_r / (ss_t + 1e-12))
                return mse, r2

            mse_a, r2_a = train_and_eval(z_anc, theta_norm)
            mse_n, r2_n = train_and_eval(z_noi, theta_norm)

            anc_mse.append(mse_a); anc_r2.append(r2_a)
            noi_mse.append(mse_n); noi_r2.append(r2_n)

        anchor_res[cname] = {"layer_mse": anc_mse, "layer_r2": anc_r2}
        noise_res[cname]  = {"layer_mse": noi_mse, "layer_r2": noi_r2}

    return anchor_res, noise_res


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _color(idx, total):
    cmap = plt.get_cmap("tab10")
    return cmap(idx % 10)


def _make_layer_x(n_layers):
    return np.arange(n_layers)


def _plot_metric_dual_axis(out_dir, fname, results, n_layers, concepts, metric="mse"):
    """Single panel: MSE or R² per layer, all concepts, dual y-axes."""
    layers = _make_layer_x(n_layers)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for ci, cn in enumerate(concepts):
        col = _color(ci, len(concepts))
        vals = results[cn][f"layer_{metric}"]
        if metric == "mse":
            ax1.plot(layers, vals, color=col, linewidth=2.0, marker='o',
                    markersize=6, label=CONCEPT_DISPLAY[cn])
            ax1.set_ylabel("MSE", fontsize=12)
        else:
            ax2.plot(layers, vals, color=col, linewidth=2.0, marker='s',
                    markersize=6, linestyle='--', label=CONCEPT_DISPLAY[cn])
            ax2.set_ylabel("R²", fontsize=12)

    ax1.set_xlabel("Layer Depth", fontsize=12)
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f"L{i}" for i in layers])
    title = "Group 1 — Baseline Linear Probe" if metric == "mse" else ""
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    if metric == "r2":
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles1 + handles2, labels1 + labels2,
                   loc="upper right", fontsize=9, ncol=2)
    else:
        ax1.legend(loc="upper right", fontsize=9, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def _plot_combined(out_dir, baseline, anchor, noise, n_layers, concepts,
                    metric="mse", select_metric="spi", anchor_ratio=0.20):
    """Per-concept subplot: Baseline vs Anchor vs Noise, with per-layer win-rate table."""
    layers = _make_layer_x(n_layers)
    n_c = len(concepts)
    n_rows = math.ceil(n_c / 3)
    table_row_h = 0.22  # fraction of figure height for the table row

    fig_h_per_row = 4.5
    fig_h = (fig_h_per_row * n_rows + 0.8) / (1 - table_row_h)
    fig, plot_axes = plt.subplots(n_rows, 3, figsize=(15, fig_h), sharex=True)
    if n_rows == 1:
        plot_axes = plot_axes[np.newaxis, :]
    plot_axes = plot_axes.flatten()

    styles = {
        "Baseline": {"marker": "o", "ls": "-",  "lw": 2.0, "alpha": 1.0},
        "Anchor":   {"marker": "^", "ls": "-.", "lw": 1.8, "alpha": 0.85},
        "Noise":    {"marker": "v", "ls": ":",  "lw": 1.5, "alpha": 0.7},
    }

    for ci, cn in enumerate(concepts):
        ax = plot_axes[ci]
        col = _color(ci, n_c)

        for label, res in [("Baseline", baseline), ("Anchor", anchor), ("Noise", noise)]:
            st = styles[label]
            vals = res[cn][f"layer_{metric}"]
            ax.plot(layers, vals, color=col, marker=st["marker"],
                    linestyle=st["ls"], linewidth=st["lw"], alpha=st["alpha"],
                    label=label)

        ax.set_title(CONCEPT_DISPLAY[cn], fontsize=10, fontweight='bold')
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{i}" for i in layers], fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ai in range(n_c, len(plot_axes)):
        plot_axes[ai].axis("off")

    ylabel = "MSE" if metric == "mse" else "R²"
    _metric_label = METRIC_DISPLAY.get(select_metric, select_metric.upper())
    _ratio_pct = int(anchor_ratio * 100)

    # Table axis at the bottom — use figure fractions so it spans the full width
    fig.subplots_adjust(hspace=0.55)
    box = plot_axes[0].get_position()
    tbl_bottom = 0.01
    tbl_height = 0.12
    tbl_ax = fig.add_axes([0.01, tbl_bottom, 0.98, tbl_height])
    tbl_ax.axis("off")

    row_labels = ["Best @ L", "Avg", "Win vs Baseline"]
    if metric == "mse":
        row_vals = ["Min MSE", "Mean MSE", "Win Rate"]
    else:
        row_vals = ["Max R\u00b2", "Mean R\u00b2", "Win Rate"]

    col_labels = [CONCEPT_DISPLAY[cn] for cn in concepts]
    cell_data = []

    for ri, rv in enumerate(row_labels):
        row = [rv]
        for cn in concepts:
            vals = {
                "Baseline": np.array(baseline[cn][f"layer_{metric}"], dtype=float),
                "Anchor":   np.array(anchor[cn][f"layer_{metric}"], dtype=float),
                "Noise":    np.array(noise[cn][f"layer_{metric}"], dtype=float),
            }
            if ri == 0:
                bl_best = np.nanmin(vals["Baseline"])
                an_best = np.nanmin(vals["Anchor"])
                no_best = np.nanmin(vals["Noise"])
                best = min(bl_best, an_best, no_best)
                if best == bl_best:
                    cell = f"Base: {bl_best:.4f}"
                elif best == an_best:
                    cell = f"Anchor: {an_best:.4f}"
                else:
                    cell = f"Noise: {no_best:.4f}"
            elif ri == 1:
                cell = (f"B:{np.nanmean(vals['Baseline']):.4f}  "
                       f"A:{np.nanmean(vals['Anchor']):.4f}  "
                       f"N:{np.nanmean(vals['Noise']):.4f}")
            else:
                if metric == "mse":
                    n_win_anc = sum(
                        vals["Anchor"][li] < vals["Baseline"][li]
                        for li in range(n_layers)
                        if not (np.isnan(vals["Anchor"][li]) or np.isnan(vals["Baseline"][li]))
                    )
                else:
                    n_win_anc = sum(
                        vals["Anchor"][li] > vals["Baseline"][li]
                        for li in range(n_layers)
                        if not (np.isnan(vals["Anchor"][li]) or np.isnan(vals["Baseline"][li]))
                    )
                n_total = sum(
                    not (np.isnan(vals["Anchor"][li]) or np.isnan(vals["Baseline"][li]))
                    for li in range(n_layers)
                )
                pct = n_win_anc / n_total * 100 if n_total > 0 else 0
                cell = f"{pct:.0f}%"
            row.append(cell)
        cell_data.append(row)

    tbl_ax.set_xlim(0, 1)
    tbl_ax.set_ylim(0, 1)

    n_cols = n_c + 1
    col_w = 1.0 / n_cols
    row_h_frac = 0.22
    tbl_h_frac = 4 * row_h_frac + 0.03

    for r in range(len(cell_data) + 1):
        for c in range(n_cols):
            is_header = r == 0
            is_rowlabel = c == 0

            if is_header:
                text = col_labels[c - 1] if not is_rowlabel else ""
                bg = "#404040"
                color = "white"
                weight = "bold"
            elif is_rowlabel:
                text = row_labels[r - 1]
                bg = "#d0d0d0"
                color = "black"
                weight = "bold"
            else:
                text = cell_data[r - 1][c - 1]
                bg = "#f5f5f5"
                color = "black"
                weight = "normal"

                if r == 3:  # Win Rate row
                    if text.endswith("%"):
                        try:
                            pct = float(text.rstrip("%"))
                            if pct >= 60:
                                bg = "#c8e6c9"
                            elif pct <= 40:
                                bg = "#ffcdd2"
                            else:
                                bg = "#fff9c4"
                        except ValueError:
                            pass

            x = c * col_w
            y = 1.0 - (r + 1) * row_h_frac / tbl_h_frac

            rect = plt.Rectangle(
                (x + 0.001, y + 0.005),
                col_w - 0.003, row_h_frac / tbl_h_frac - 0.01,
                transform=tbl_ax.transAxes,
                facecolor=bg, edgecolor="white", linewidth=0.5,
                zorder=1,
            )
            tbl_ax.add_patch(rect)

            fontsize = 7.5 if (is_header or is_rowlabel) else 7
            tbl_ax.text(
                x + col_w / 2, y + row_h_frac / tbl_h_frac / 2 + 0.005,
                text,
                transform=tbl_ax.transAxes,
                ha="center", va="center",
                fontsize=fontsize,
                color=color,
                fontweight=weight,
                clip_on=True,
                zorder=2,
            )

    fig.suptitle(
        f"Group 1 vs Group 2 — Linear Probe — {ylabel} per Layer\n"
        f"Anchor/Noise selected by {_metric_label} (Top/Bottom {_ratio_pct}%)",
        fontsize=13, fontweight='bold', y=0.99
    )
    fig.savefig(os.path.join(out_dir, f"probe_combined_{metric}.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: probe_combined_{metric}.png")


def _plot_delta_mse(out_dir, baseline, anchor, noise, n_layers, concepts, select_metric, anchor_ratio=0.20):
    """Delta MSE (Anchor - Baseline, Noise - Baseline) per layer per concept."""
    layers = _make_layer_x(n_layers)
    n_c = len(concepts)
    n_rows = math.ceil(n_c / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows), sharex=True)
    axes = axes.flatten()
    width = 0.35

    for ci, cn in enumerate(concepts):
        ax = axes[ci]
        col = _color(ci, n_c)
        bl = np.array(baseline[cn]["layer_mse"], dtype=float)
        anc = np.array(anchor[cn]["layer_mse"], dtype=float)
        noi = np.array(noise[cn]["layer_mse"], dtype=float)

        delta_anc = anc - bl
        delta_noi = noi - bl

        ax.bar(layers - width/2, delta_anc, width, label="Anchor - Baseline",
               color=col, alpha=0.8)
        ax.bar(layers + width/2, delta_noi, width, label="Noise - Baseline",
               color=col, alpha=0.4, hatch="//")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        ax.set_title(CONCEPT_DISPLAY[cn], fontsize=10, fontweight='bold')
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{i}" for i in layers], fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=7)

    for ai in range(n_c, len(axes)):
        axes[ai].axis("off")

    _metric_label = METRIC_DISPLAY.get(select_metric, select_metric.upper())
    _ratio_pct = int(anchor_ratio * 100)
    fig.suptitle(
        f"Group 2 \u0394MSE vs Baseline\n(Selected by {_metric_label}, Top/Bottom {_ratio_pct}%)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_deltamse.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: probe_deltamse.png")


def _plot_mse_heatmap(out_dir, baseline, n_layers, concepts):
    """Heatmap of baseline MSE: concepts × layers."""
    n_c = len(concepts)
    matrix = np.zeros((n_c, n_layers))
    for ci, cn in enumerate(concepts):
        matrix[ci] = baseline[cn]["layer_mse"]

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 1.5), n_c * 0.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_yticks(range(n_c))
    ax.set_yticklabels([CONCEPT_DISPLAY[cn] for cn in concepts])
    ax.set_xlabel("Layer Depth", fontsize=11)
    ax.set_ylabel("Concept", fontsize=11)
    ax.set_title("Group 1 — Baseline MSE Heatmap (concepts × layers)",
                  fontsize=12, fontweight='bold')

    vmax = matrix.max()
    for ci in range(n_c):
        for li in range(n_layers):
            val = matrix[ci, li]
            if not np.isnan(val):
                fc = "white" if val > vmax * 0.65 else "black"
                ax.text(li, ci, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, color=fc)

    plt.colorbar(im, ax=ax, label="MSE")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_baseline_mse_heatmap.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: probe_baseline_mse_heatmap.png")


def _plot_cross_concept_comparison(out_dir, baseline, anchor, noise,
                                   n_layers, concepts, select_metric):
    """
    Two-panel comparison: all concepts aggregated, Anchor (high MI) vs Noise (low MI).
    Panel 1: mean R² per layer across all concepts (higher is better).
    Panel 2: mean MSE delta (Anchor MSE - Noise MSE) per layer (more negative = Anchor better).

    Both panels on a single figure.
    """
    layers = _make_layer_x(n_layers)
    n_c = len(concepts)

    mean_anc_r2 = np.zeros(n_layers)
    mean_noi_r2 = np.zeros(n_layers)
    mean_anc_mse = np.zeros(n_layers)
    mean_noi_mse = np.zeros(n_layers)

    for cn in concepts:
        mean_anc_r2 += np.array(anchor[cn]["layer_r2"], dtype=float)
        mean_noi_r2 += np.array(noise[cn]["layer_r2"], dtype=float)
        mean_anc_mse += np.array(anchor[cn]["layer_mse"], dtype=float)
        mean_noi_mse += np.array(noise[cn]["layer_mse"], dtype=float)

    mean_anc_r2 /= n_c
    mean_noi_r2 /= n_c
    mean_anc_mse /= n_c
    mean_noi_mse /= n_c

    delta_mse = mean_anc_mse - mean_noi_mse

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_r2, ax_delta = axes

    ax_r2.plot(layers, mean_anc_r2, marker='^', color='#e74c3c',
               linewidth=2.0, label='High-MI (Anchor)')
    ax_r2.plot(layers, mean_noi_r2, marker='v', color='#3498db',
               linewidth=2.0, linestyle='--', label='Low-MI (Noise)')
    ax_r2.set_xlabel("Layer Depth", fontsize=11)
    ax_r2.set_ylabel("Mean R² (all concepts)", fontsize=11)
    ax_r2.set_title("High-MI vs Low-MI: Mean R² per Layer", fontsize=12, fontweight='bold')
    ax_r2.set_xticks(layers)
    ax_r2.set_xticklabels([f"L{i}" for i in layers])
    ax_r2.legend(fontsize=9)
    ax_r2.grid(True, alpha=0.3)

    bars = ax_delta.bar(layers, delta_mse, width=0.5,
                        color=['#27ae60' if v <= 0 else '#e74c3c' for v in delta_mse],
                        alpha=0.8)
    ax_delta.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_delta.set_xlabel("Layer Depth", fontsize=11)
    ax_delta.set_ylabel("\u0394MSE = High-MI \u2212 Low-MI", fontsize=11)
    ax_delta.set_title("High-MI vs Low-MI: MSE Difference per Layer", fontsize=12, fontweight='bold')
    ax_delta.set_xticks(layers)
    ax_delta.set_xticklabels([f"L{i}" for i in layers])
    ax_delta.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, delta_mse):
        ypos = bar.get_height()
        ax_delta.text(bar.get_x() + bar.get_width() / 2,
                      ypos + 0.002 * np.sign(ypos) if ypos != 0 else 0.001,
                      f"{val:.4f}",
                      ha='center', va='bottom' if ypos >= 0 else 'top',
                      fontsize=7, color='black')

    _metric_label = METRIC_DISPLAY.get(select_metric, select_metric.upper())
    fig.suptitle(
        f"High-MI vs Low-MI Group Comparison (all {n_c} concepts)\n"
        f"Top/Bottom 20% selected by {_metric_label}",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_cross_concept_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: probe_cross_concept_comparison.png")


def plot_all_results(out_dir, baseline, anchor, noise, n_layers, concepts, select_metric, anchor_ratio=0.20):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Baseline MSE
    _plot_metric_dual_axis(out_dir, "probe_baseline_mse.png",
                             baseline, n_layers, concepts, "mse")
    # 2. Baseline R²
    _plot_metric_dual_axis(out_dir, "probe_baseline_r2.png",
                             baseline, n_layers, concepts, "r2")
    # 3. Combined MSE
    if anchor and noise:
        _plot_combined(out_dir, baseline, anchor, noise, n_layers, concepts,
                        "mse", select_metric, anchor_ratio)
        _plot_combined(out_dir, baseline, anchor, noise, n_layers, concepts,
                        "r2", select_metric, anchor_ratio)
        _plot_delta_mse(out_dir, baseline, anchor, noise, n_layers,
                          concepts, select_metric, anchor_ratio)
        _plot_cross_concept_comparison(out_dir, baseline, anchor, noise,
                                        n_layers, concepts, select_metric)
    # 4. Heatmap
    _plot_mse_heatmap(out_dir, baseline, n_layers, concepts)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    concepts = CONCEPT_NAMES[:args.n_concepts]

    print("=" * 70)
    print("  Timer MI + Linear Probing (Synthetic Concepts)")
    print("=" * 70)
    print(f"  Concepts : {concepts}")
    print(f"  seq_len  : {args.seq_len}")
    print(f"  pred_len : {args.pred_len}")
    print(f"  patch_len: {args.patch_len}")
    print(f"  e_layers : {args.e_layers}")
    print(f"  n_samples: {args.n_samples} per concept")
    print(f"  metric   : {args.select_metric}")
    print(f"  anchor   : {args.anchor_ratio}")
    print(f"  device   : cuda:{args.gpu}" if torch.cuda.is_available() else "  device   : cpu")
    print("=" * 70)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n>>> [1/4] 构建合成数据集...")
    ds = SyntheticSingleConceptDataset(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_samples_per_concept=args.n_samples,
        concepts=concepts,
        split="all",
        seed=42,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, collate_fn=collate_single)
    print(f"  总样本数: {len(ds)} ({args.n_samples} × {len(concepts)} concepts)")

    # ── 2. Model ───────────────────────────────────────────────────────────────
    print("\n>>> [2/4] 加载 Timer 模型...")
    model = build_timer(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len,
        stride=args.patch_len,
        d_model=args.d_model, d_ff=args.d_ff,
        e_layers=args.e_layers, n_heads=args.n_heads,
        dropout=args.dropout,
        seq_len=args.seq_len, pred_len=args.pred_len,
    )
    model = model.to(device)
    model.eval()
    print(f"  模型就绪，设备: {device}")

    # ── 3. Extract tokens + MI ─────────────────────────────────────────────────
    print("\n>>> [3/4] 提取 Token 表示并计算 per-token MI...")
    layer_data = extract_layer_tokens_and_mi(
        model=model,
        data_loader=loader,
        device=device,
        n_layers=args.e_layers,
        patch_len=args.patch_len,
        k_neighbors=args.k_neighbors,
        pca_dim=args.pca_dim,
        sample_ratio=args.sample_ratio,
    )
    n_layers_actual = len(layer_data)
    print(f"  提取完成，共 {n_layers_actual} 层")

    # ── 4. Group 1: Baseline ───────────────────────────────────────────────────
    print("\n>>> [4/4a] Group 1: Baseline Linear Probe (Mean Pooling)...")
    baseline = probe_baseline(
        layer_data=layer_data,
        concepts=concepts,
        alpha=args.probe_alpha,
        train_frac=0.8,
        seed=42,
    )
    for cn in concepts:
        arr = np.array(baseline[cn]["layer_mse"], dtype=float)
        arr_r = np.array(baseline[cn]["layer_r2"], dtype=float)
        best = int(np.nanargmin(arr))
        print(f"  {CONCEPT_DISPLAY[cn]:16s} | Best L{best} MSE={arr[best]:.4f} R²={arr_r[best]:.4f}")

    # ── 5. Group 2: Anchor vs Noise ─────────────────────────────────────────────
    anchor, noise = None, None
    if args.run_group2:
        print(f"\n>>> [4/4b] Group 2: Anchor vs Noise (metric={args.select_metric})...")
        anchor, noise = probe_anchor_vs_noise(
            layer_data=layer_data,
            concepts=concepts,
            metric=args.select_metric,
            anchor_ratio=args.anchor_ratio,
            alpha=args.probe_alpha,
            train_frac=0.8,
            seed=42,
        )
        for cn in concepts:
            am = np.array(anchor[cn]["layer_mse"], dtype=float)
            nm = np.array(noise[cn]["layer_mse"], dtype=float)
            bm = np.array(baseline[cn]["layer_mse"], dtype=float)
            ba = int(np.nanargmin(am))
            bn = int(np.nanargmin(nm))
            print(f"  {CONCEPT_DISPLAY[cn]:16s} | Anchor L{ba} MSE={am[ba]:.4f} | "
                  f"Noise L{bn} MSE={nm[bn]:.4f} | Baseline L{int(np.nanargmin(bm))} MSE={bm.min():.4f}")

    # ── 6. Plot ────────────────────────────────────────────────────────────────
    print("\n>>> [5/5] 绘图...")
    plot_all_results(
        out_dir=args.out_dir,
        baseline=baseline,
        anchor=anchor,
        noise=noise,
        n_layers=n_layers_actual,
        concepts=concepts,
        select_metric=args.select_metric,
        anchor_ratio=args.anchor_ratio,
    )

    # ── 7. Save JSON ───────────────────────────────────────────────────────────
    print("\n>>> [6/6] 保存结果...")
    results = {
        "args": {k: v for k, v in vars(args).items()
                 if not callable(v) and not k.startswith("_")},
        "timestamp": datetime.now().isoformat(),
        "concepts": concepts,
        "n_layers": n_layers_actual,
        "baseline": baseline,
    }
    if anchor:
        results["anchor"] = anchor
        results["noise"] = noise

    # Convert numpy types for JSON
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = _clean(results)
    json_path = os.path.join(args.out_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_clean)
    print(f"  JSON: {json_path}")

    print("\n" + "=" * 70)
    print(f"  完成！结果: {args.out_dir}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Timer MI + Linear Probing")
    p.add_argument("--ckpt_path",  type=str, default="checkpoints/Timer_forecast_1.0.ckpt")
    p.add_argument("--d_model",    type=int, default=1024)
    p.add_argument("--d_ff",       type=int, default=2048)
    p.add_argument("--e_layers",   type=int, default=8)
    p.add_argument("--n_heads",    type=int, default=8)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--seq_len",    type=int, default=672)
    p.add_argument("--pred_len",   type=int, default=96)
    p.add_argument("--patch_len",  type=int, default=96)
    p.add_argument("--n_samples",  type=int, default=2048,
                   help="每概念样本数")
    p.add_argument("--n_concepts", type=int, default=7,
                   help="概念数量 1-7")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--k_neighbors", type=int, default=5)
    p.add_argument("--pca_dim",    type=int, default=32)
    p.add_argument("--sample_ratio", type=float, default=1.0)
    p.add_argument("--probe_alpha", type=float, default=1.0)
    p.add_argument("--select_metric", type=str, default="spi",
                   choices=["spi", "ihy", "ixh"],
                   help="'spi' = SPI (I(h_x,h_y)/I(e_x,h_x)); 'ihy' = I(h_x,h_y); 'ixh' = I(e_x,h_x) for token selection")
    p.add_argument("--anchor_ratio", type=float, default=0.20)
    p.add_argument("--run_group2",  action="store_true", default=True)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--out_dir",     type=str, default="./outputs/timer_mi_probe")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    run_pipeline(args)
