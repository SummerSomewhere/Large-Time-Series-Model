#!/usr/bin/env python3
"""
合成数据 Layer-wise Probe 特征提取。

11 种数据模式（--data_mode）：
  1. trend          : X = β*t + ε,  label=slope
  2. periodic       : X = A*sin(2πf*t) + ε,  label=amplitude
  3. noise          : X = ε,  label=noise_std
  4. ar1            : X_t = φ*X_{t-1} + ε_t,  label=phi
  5. level_shift    : X = η + Δ*1{t≥τ} + ε,  label=delta
  6. random_walk    : X_t = X_{t-1} + µ + ε_t,  label=drift
  7. spectral       : X = Σ a_j*sin(2πf_j*t) + ε,  label=amplitude
  8. time_warp      : warped sinusoid,  label=warp_factor
  9. variance       : X ~ N(0, σ₁²) (t<τ) / N(0, σ₂²) (t≥τ),  label=sigma2_ratio
 10. trend_periodic : X = β*t + A*sin(2πf*t) + ε,  labels=[slope, amplitude]
 11. all            : X = β*t + A*sin + ε,  labels=[slope, amplitude, noise_std]

得到 [n, D] 特征向量，配合 ground truth 标签一起保存为 .pt。

运行时：
  python experiments/synthetic_layerwise_probe_features.py --ckpt_path checkpoints/...
"""

from __future__ import annotations

import argparse
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Timer import Model
from utils.masking import TriangularCausalMask


# ─────────────────────────────────────────────────────────────────────────────
# 合成数据生成器
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticTimeSeriesDataset(torch.utils.data.Dataset):
    """
    11 种数据模式，每种模式生成独立信号并输出对应的 ground truth 标签。
    """

    # mode -> list of label names
    MODE_LABELS = {
        "trend":          ["slope"],
        "periodic":        ["amplitude"],
        "noise":           ["noise_std"],
        "ar1":            ["phi"],
        "level_shift":     ["delta"],
        "random_walk":     ["drift"],
        "spectral":        ["amplitude"],
        "time_warp":       ["warp_factor"],
        "variance":        ["sigma2_ratio"],
        "trend_periodic":  ["slope", "amplitude"],
        "all":             ["slope", "amplitude", "noise_std"],
    }

    # 参数采样范围
    MODE_RANGES = {
        "slope":          (-0.01, 0.01),
        "amplitude":      (0.5, 1.5),
        "noise_std":      (0.05, 0.3),
        "phi":            (0.3, 0.9),
        "delta":          (-1.5, 1.5),
        "drift":          (-0.005, 0.005),
        "freq_low":       (0.01, 0.05),
        "freq_high":      (0.1, 0.3),
        "n_components":   (1, 3),
        "warp_shape":     (0.5, 2.0),
        "sigma2_before":  (0.5, 1.0),
        "sigma2_after":   (0.5, 2.0),
    }

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        pred_len: int,
        data_mode: str = "all",
        freq_hz: float = 1/24,
        seed: int = 42,
        train: bool = True,
        train_ratio: float = 0.7,
    ):
        assert data_mode in self.MODE_LABELS, f"Unknown mode: {data_mode}"
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_mode = data_mode
        self.freq_hz = freq_hz
        self.rng = np.random.default_rng(seed)
        self.label_names = self.MODE_LABELS[data_mode]

        # ── 生成标签和信号 ────────────────────────────────────────────────
        labels_dict, X_full = self._generate(n_samples, seq_len, pred_len)

        # split
        split = int(n_samples * train_ratio)
        if train:
            self.X_full = X_full[:split]
            self.labels = {k: v[:split] for k, v in labels_dict.items()}
        else:
            self.X_full = X_full[split:]
            self.labels = {k: v[split:] for k, v in labels_dict.items()}

        self.n_samples = len(self.X_full)

    # ── 统一的采样接口 ────────────────────────────────────────────────────
    def _sample(self, name: str, n: int) -> np.ndarray:
        lo, hi = self.MODE_RANGES[name]
        return self.rng.uniform(lo, hi, n).astype(np.float32)

    # ── 各模式生成器 ──────────────────────────────────────────────────────
    def _gen_trend(self, n, seq_len, pred_len):
        slope = self._sample("slope", n)
        noise_std = self._sample("noise_std", n)
        t = np.linspace(0, 1, seq_len + pred_len)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            signal = slope[i] * t
            X[i] = signal + self.rng.normal(0, noise_std[i], seq_len + pred_len)
        return {"slope": slope, "noise_std": noise_std}, X

    def _gen_periodic(self, n, seq_len, pred_len):
        amplitude = self._sample("amplitude", n)
        noise_std = self._sample("noise_std", n)
        t = np.linspace(0, 1, seq_len + pred_len)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            signal = amplitude[i] * np.sin(2 * np.pi * self.freq_hz * t)
            X[i] = signal + self.rng.normal(0, noise_std[i], seq_len + pred_len)
        return {"amplitude": amplitude, "noise_std": noise_std}, X

    def _gen_noise(self, n, seq_len, pred_len):
        noise_std = self._sample("noise_std", n)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            X[i] = self.rng.normal(0, noise_std[i], seq_len + pred_len)
        return {"noise_std": noise_std}, X

    def _gen_ar1(self, n, seq_len, pred_len):
        phi = self._sample("phi", n)
        noise_std = self._sample("noise_std", n)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            x0 = self.rng.normal(0, 1.0 / np.sqrt(1 - phi[i]**2 + 1e-12))
            x = np.empty(seq_len + pred_len)
            x[0] = x0
            for t in range(1, seq_len + pred_len):
                x[t] = phi[i] * x[t-1] + self.rng.normal(0, noise_std[i])
            X[i] = x
        return {"phi": phi, "noise_std": noise_std}, X

    def _gen_level_shift(self, n, seq_len, pred_len):
        delta = self._sample("delta", n)
        noise_std = self._sample("noise_std", n)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            tau = int(self.rng.uniform(0.2, 0.8) * (seq_len + pred_len))
            base = self.rng.normal(0, 0.5)
            x = np.full(seq_len + pred_len, base)
            x[tau:] += delta[i]
            x += self.rng.normal(0, noise_std[i], seq_len + pred_len)
            X[i] = x
        return {"delta": delta, "noise_std": noise_std}, X

    def _gen_random_walk(self, n, seq_len, pred_len):
        drift = self._sample("drift", n)
        noise_std = self._sample("noise_std", n)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            x = np.empty(seq_len + pred_len)
            x[0] = 0.0
            for t in range(1, seq_len + pred_len):
                x[t] = x[t-1] + drift[i] + self.rng.normal(0, noise_std[i])
            X[i] = x
        return {"drift": drift, "noise_std": noise_std}, X

    def _gen_spectral(self, n, seq_len, pred_len):
        amplitude = self._sample("amplitude", n)
        noise_std = self._sample("noise_std", n)
        n_comp = self.rng.integers(1, 4, n)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            t = np.linspace(0, 1, seq_len + pred_len)
            k = n_comp[i]
            freqs = self.rng.uniform(self.MODE_RANGES["freq_low"][0],
                                     self.MODE_RANGES["freq_high"][0], k)
            phases = self.rng.uniform(0, 2*np.pi, k)
            signal = np.zeros(seq_len + pred_len)
            for j in range(k):
                signal += amplitude[i] * np.sin(2*np.pi * freqs[j] * t + phases[j])
            X[i] = signal + self.rng.normal(0, noise_std[i], seq_len + pred_len)
        return {"amplitude": amplitude, "noise_std": noise_std}, X

    def _gen_time_warp(self, n, seq_len, pred_len):
        warp_factor = self._sample("warp_shape", n)
        noise_std = self._sample("noise_std", n)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            t_orig = np.linspace(0, 1, seq_len + pred_len)
            # monotone warp: cumsum of positive gamma steps
            k = seq_len + pred_len
            steps = self.rng.gamma(warp_factor[i] * 2, 1.0, k)
            steps = steps / (steps.max() + 1e-12)
            warp = np.cumsum(steps)
            warp = warp / warp.max() * (seq_len + pred_len - 1)
            base = np.sin(2 * np.pi * self.freq_hz * t_orig)
            x_interp = np.interp(np.linspace(0, warp[-1], seq_len + pred_len),
                                   warp, base)
            X[i] = x_interp + self.rng.normal(0, noise_std[i], seq_len + pred_len)
        return {"warp_factor": warp_factor, "noise_std": noise_std}, X

    def _gen_variance(self, n, seq_len, pred_len):
        noise_std_before = self._sample("sigma2_before", n)
        noise_std_after = self._sample("sigma2_after", n)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            tau = int(self.rng.uniform(0.3, 0.7) * (seq_len + pred_len))
            x = np.empty(seq_len + pred_len)
            x[:tau] = self.rng.normal(0, noise_std_before[i], tau)
            x[tau:] = self.rng.normal(0, noise_std_after[i], seq_len + pred_len - tau)
            X[i] = x
        sigma2_ratio = (noise_std_after / (noise_std_before + 1e-12)).astype(np.float32)
        return {"sigma2_ratio": sigma2_ratio, "noise_std_before": noise_std_before,
                "noise_std_after": noise_std_after}, X

    def _gen_trend_periodic(self, n, seq_len, pred_len):
        slope = self._sample("slope", n)
        amplitude = self._sample("amplitude", n)
        noise_std = self._sample("noise_std", n)
        t = np.linspace(0, 1, seq_len + pred_len)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            signal = slope[i] * t + amplitude[i] * np.sin(2 * np.pi * self.freq_hz * t)
            X[i] = signal + self.rng.normal(0, noise_std[i], seq_len + pred_len)
        return {"slope": slope, "amplitude": amplitude, "noise_std": noise_std}, X

    def _gen_all(self, n, seq_len, pred_len):
        slope = self._sample("slope", n)
        amplitude = self._sample("amplitude", n)
        noise_std = self._sample("noise_std", n)
        t = np.linspace(0, 1, seq_len + pred_len)
        X = np.empty((n, seq_len + pred_len), dtype=np.float32)
        for i in range(n):
            signal = slope[i] * t + amplitude[i] * np.sin(2 * np.pi * self.freq_hz * t)
            X[i] = signal + self.rng.normal(0, noise_std[i], seq_len + pred_len)
        return {"slope": slope, "amplitude": amplitude, "noise_std": noise_std}, X

    def _generate(self, n, seq_len, pred_len):
        gen_map = {
            "trend":          self._gen_trend,
            "periodic":       self._gen_periodic,
            "noise":          self._gen_noise,
            "ar1":            self._gen_ar1,
            "level_shift":    self._gen_level_shift,
            "random_walk":    self._gen_random_walk,
            "spectral":       self._gen_spectral,
            "time_warp":      self._gen_time_warp,
            "variance":       self._gen_variance,
            "trend_periodic": self._gen_trend_periodic,
            "all":            self._gen_all,
        }
        labels, X_2d = gen_map[self.data_mode](n, seq_len, pred_len)
        return labels, X_2d.reshape(n, -1, 1)   # -> (n, T, 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = self.X_full[idx, : self.seq_len, :]    # [seq_len, 1]
        y = self.X_full[idx, self.seq_len :, :]   # [pred_len, 1]
        x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        y_mark = np.zeros((self.pred_len, 4), dtype=np.float32)
        return x, y, x_mark, y_mark


# ─────────────────────────────────────────────────────────────────────────────
# Timer 每层特征提取
# ─────────────────────────────────────────────────────────────────────────────

def collect_layerwise_features(
    model: nn.Module,
    dataset: SyntheticTimeSeriesDataset,
    batch_size: int,
    device: torch.device,
    e_layers: int,
    seq_len: int,
    pred_len: int,
) -> dict:
    """
    与 etth1_mi_hsic_peaks.py 合成数据路径完全对齐：
      - 同一份 dataset（train=False, train_ratio=0.7）
      - 同一份 Normalization 逻辑
      - 同一份标签提取逻辑（batch_y[:, -pred_len:, 0]）
      - 同一份 decoder forward 逻辑（enc_embedding -> decoder 各层）
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    backbone = model.module.backbone if hasattr(model, "module") else model.backbone

    # layer_idx -> list[Tensor[B, D]]
    hidden_lists: dict = {li: [] for li in range(e_layers)}
    label_names = dataset.label_names
    y_labels: dict = {name: [] for name in label_names}
    y_labels["ground_truth"] = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            B = batch_x.shape[0]

            # Normalization — 与 HSIC 脚本 forward_collect_layers 完全一致
            x = batch_x.squeeze(-1).float().to(device)
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdev
            x = x.unsqueeze(-1)

            # Patch embedding：与 HSIC 脚本完全一致
            x2d = x.permute(0, 2, 1)
            dec_in, n_vars = backbone.patch_embedding(x2d)
            BM = dec_in.shape[0]

            # Decoder forward with hidden states（与 HSIC 脚本完全一致）
            dec_out, _, hidden_states = backbone.decoder(
                dec_in, has_prototype=False, output_hidden_states=True
            )

            for li in range(min(len(hidden_states), e_layers)):
                h = hidden_states[li]
                h_view = h.view(B, n_vars, -1, h.shape[-1])
                h_pooled = h_view.mean(dim=[1, 2])
                hidden_lists[li].append(h_pooled.cpu())

            # 标签收集：取 batch_y 未来窗口均值作为 ground_truth
            base = len(y_labels[label_names[0]])
            y_future = batch_y[:, -pred_len:, 0]
            for i in range(B):
                idx = base + i
                for name in label_names:
                    y_labels[name].append(dataset.labels[name][idx])
                y_labels["ground_truth"].append(float(y_future[i].mean()))

    # concat
    layer_hidden: dict = {}
    for li in range(e_layers):
        if not hidden_lists[li]:
            continue
        stacked = torch.cat(hidden_lists[li], dim=0)
        layer_hidden[f"layer_{li}"] = stacked
        print(f"  [Layer {li}] shape={stacked.shape}, D={stacked.shape[-1]}")

    labels: dict = {k: np.array(v) for k, v in y_labels.items()}
    return layer_hidden, labels, label_names


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def build_model_config(args: argparse.Namespace) -> argparse.Namespace:
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
        "model_id": "synthetic_probe",
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
    }.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def main() -> None:
    parser = argparse.ArgumentParser(description="合成数据 Layer-wise Probe 特征提取")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--freq_hz", type=float, default=0.042,
                        help="周期成分频率 (Hz)，默认 0.5")
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="train/test 切分比例，默认 0.7（与 HSIC 脚本一致）")
    parser.add_argument("--data_mode", type=str, default="all",
                        choices=["trend", "periodic", "noise",
                                 "ar1", "level_shift", "random_walk",
                                 "spectral", "time_warp", "variance",
                                 "trend_periodic", "all"],
                        help="数据模式：trend/periodic/noise/ar1/level_shift/random_walk/spectral/time_warp/variance/trend_periodic/all")
    parser.add_argument("--out_dir", type=str,
                        default="./results/synthetic_layerwise_probe_features")
    parser.add_argument("--out_name", type=str, default=None,
                        help="输出文件名（不含扩展名），默认 synth_{n_samples}")
    args = parser.parse_args()

    # Device
    gpu_ids_raw = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if gpu_ids_raw:
        gpu_ids = [int(x) for x in gpu_ids_raw]
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    fname = args.out_name or f"synth_{args.n_samples}_f{args.freq_hz}_{args.data_mode}"
    out_path = os.path.join(out_dir, f"{fname}.pt")
    json_path = os.path.join(out_dir, f"{fname}_labels.json")

    print(f"{'='*60}")
    print(f"  合成数据 Layer-wise Feature Extraction")
    print(f"  n_samples={args.n_samples}, seq_len={args.seq_len}, pred_len={args.pred_len}")
    print(f"  data_mode={args.data_mode}, freq_hz={args.freq_hz}")
    print(f"{'='*60}\n")

    # ── 1. 生成合成数据 ────────────────────────────────────────────────
    print("[Step 1] 生成合成时序数据...")
    dataset = SyntheticTimeSeriesDataset(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        data_mode=args.data_mode,
        freq_hz=args.freq_hz,
        seed=args.seed,
        train=False,
        train_ratio=args.train_ratio,
    )
    print(f"  n_samples (after split): {len(dataset)}")
    print(f"  X shape per sample: {dataset[0][0].shape}")
    print(f"  data_mode: {dataset.data_mode}")
    print(f"  label_names: {dataset.label_names}")
    for name in dataset.label_names:
        vals = dataset.labels[name]
        print(f"  {name}: [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}")

    # ── 2. 加载 Timer ───────────────────────────────────────────────────────
    print("\n[Step 2] 加载 Timer 模型...")
    config = build_model_config(args)
    model = Model(config).to(device)
    model.eval()
    backbone = model.module.backbone if hasattr(model, "module") else model.backbone
    stride = int(backbone.patch_embedding.stride)
    print(f"  Timer loaded (stride={stride})")

    # ── 3. 提取各层特征 ────────────────────────────────────────────────────
    print("\n[Step 3] 提取 layer-wise hidden states...")
    layer_hidden, labels, label_names = collect_layerwise_features(
        model, dataset,
        batch_size=args.batch_size,
        device=device,
        e_layers=args.e_layers,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )

    if not layer_hidden:
        raise RuntimeError("No layer features collected!")

    n_collected = labels[label_names[0]].shape[0]
    n_layers_out = len(layer_hidden)
    D = next(iter(layer_hidden.values())).shape[-1]

    # ── 4. 保存 .pt ───────────────────────────────────────────────────────
    print(f"\n[Step 4] 保存特征到 {out_path} ...")
    save_dict = {
        "layer_hidden": {k: v.clone() for k, v in layer_hidden.items()},
        "labels": {k: torch.from_numpy(v).clone() for k, v in labels.items()},
        "config": {
            "n_samples": n_collected,
            "n_layers": n_layers_out,
            "feat_dim": D,
            "pred_len": args.pred_len,
            "seq_len": args.seq_len,
            "patch_len": args.patch_len,
            "stride": stride,
            "freq_hz": args.freq_hz,
            "data_mode": args.data_mode,
            "label_names": label_names,
            "data": "synthetic",
            "ckpt_path": args.ckpt_path,
        },
    }
    torch.save(save_dict, out_path)

    json_labels = {k: v.tolist() for k, v in labels.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_samples": n_collected, "labels": json_labels, "config": save_dict["config"]}, f, indent=2)

    # ── 5. 摘要 ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  摘要")
    print(f"{'='*60}")
    print(f"  样本数:    {n_collected}")
    print(f"  Decoder层: {n_layers_out}")
    print(f"  特征维度:  {D} (Mean Pool over N patches)")
    print(f"  标签:")
    for name, arr in labels.items():
        print(f"    {name}: mean={arr.mean():.4f}, std={arr.std():.4f}")
    print(f"  输出文件:  {out_path}")
    print(f"\n✅ 完成！")


if __name__ == "__main__":
    main()
