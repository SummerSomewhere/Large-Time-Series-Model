#!/usr/bin/env python3
"""
Layer-wise Probing 特征提取脚本。

通过 forward 的 output_hidden_states=True 参数获取 Timer 所有 decoder 层的隐藏状态，
对每层做 Mean Pooling over patches，得到 [n_samples, D] 的特征向量，
配合 STL 分解标签一起 torch.save 到本地，供后续 Probe 训练使用。

输出文件 (.pt):
    {
        "layer_hidden": {
            "layer_0": torch.Tensor [n, D],
            "layer_1": torch.Tensor [n, D],
            ...
        },
        "y_true": torch.Tensor [n, pred_len],
        "labels": {
            "trend":     torch.Tensor [n],
            "seasonal":  torch.Tensor [n],
            "noise":     torch.Tensor [n],
            "volatility":torch.Tensor [n],
            "ground_truth": torch.Tensor [n],
        },
        "config": { n_samples, n_layers, pred_len, patch_len, stride, seq_len, ... }
    }
"""


from __future__ import annotations

import argparse
import os
import sys
import json
import warnings
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.masking import TriangularCausalMask


# ─────────────────────────────────────────────────────────────────────────────
# STL 标签
# ─────────────────────────────────────────────────────────────────────────────

def get_stl_labels(
    y_seq: np.ndarray,
    period: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    对预测窗口 y_seq 进行 STL 分解，生成语义标签。

    y_seq: [n_samples, pred_len]
    返回: {label_name: [n_samples]}
    """
    from statsmodels.tsa.seasonal import STL

    n_samples, pred_len = y_seq.shape

    def _infer_period(seq: np.ndarray, max_period: int = 48) -> int:
        fft_vals = np.abs(np.fft.rfft(seq - seq.mean()))
        freqs = np.fft.rfftfreq(len(seq))
        if len(fft_vals) < 2:
            return max(2, min(pred_len // 2, 24))
        fft_vals[0] = 0
        peak_idx = int(np.argmax(fft_vals))
        period_est = int(round(1.0 / freqs[peak_idx])) if freqs[peak_idx] > 0 else 24
        return max(2, min(period_est, max_period))

    period_used = period if period is not None else _infer_period(y_seq[0]) if n_samples > 0 else 24
    print(f"  [STL] period={period_used}, pred_len={pred_len}")

    trend_arr     = np.zeros(n_samples)
    seasonal_arr  = np.zeros(n_samples)
    noise_arr      = np.zeros((n_samples, pred_len))
    volatility_arr = np.zeros(n_samples)
    gt_arr         = np.zeros(n_samples)

    for i in range(n_samples):
        window = y_seq[i]
        try:
            stl = STL(window, period=period_used, robust=True)
            res = stl.fit()
            trend_arr[i]     = np.mean(res.trend)
            seasonal_arr[i]  = np.std(res.seasonal)
            noise_arr[i]     = res.resid
        except Exception:
            trend_arr[i]     = np.mean(window)
            seasonal_arr[i]  = 0.0
            noise_arr[i]     = window - np.mean(window)
        volatility_arr[i] = np.std(window)
        gt_arr[i]          = np.mean(window)

    noise_std = np.std(noise_arr, axis=1)

    return {
        "trend":        trend_arr,
        "seasonal":     seasonal_arr,
        "noise":        noise_std,
        "volatility":   volatility_arr,
        "ground_truth": gt_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 构建 Model 配置
# ─────────────────────────────────────────────────────────────────────────────

def build_model_namespace(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    for k, v in {
        "task_name":         "forecast",
        "is_training":       0,
        "is_finetuning":     0,
        "train_test":        0,
        "use_multi_gpu":     False,
        "d_layers":          1,
        "target":            "OT",
        "checkpoints":       "./checkpoints/",
        "inverse":           False,
        "use_amp":           False,
        "use_weight_decay":  0,
        "weight_decay":      0.01,
        "loss":              "MSE",
        "lradj":             "type1",
        "train_epochs":      0,
        "patience":          3,
        "learning_rate":     1e-4,
        "itr":               1,
        "finetune_epochs":   0,
        "output_attention":  False,
        "distil":            True,
        "model_id":          "layerwise_probe",
        "model":             "Timer",
        "output_len_list":   None,
        "mask_rate":         0.25,
        "data_type":         "custom",
        "decay_fac":         0.75,
        "cos_warm_up_steps": 100,
        "cos_max_decay_steps": 60000,
        "cos_max_decay_epoch": 10,
        "cos_max":           1e-4,
        "cos_min":           2e-6,
        "stride":            getattr(args, "stride", args.patch_len if hasattr(args, "patch_len") else 96),
    }.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    ns.use_gpu = bool(torch.cuda.is_available())
    return ns


# ─────────────────────────────────────────────────────────────────────────────
# 核心提取函数
# ─────────────────────────────────────────────────────────────────────────────

def collect_layerwise_features(
    args: argparse.Namespace,
    n_samples: int,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    通过 forward(output_hidden_states=True) 收集所有 decoder 层的 hidden_states，
    每层做 Mean Pooling over patches，得到 {layer_0: [n, D], ...}。

    hidden_states 列表结构:
      每层输出 [B*M, N, D]（prototype 注入后可能为 N+1）
      pool 后 -> [B, N, D] -> mean over N -> [B, D]

    返回: {"layer_0": Tensor[n,D], "layer_1": ..., "y_true": Tensor[n,pred_len]}
    """
    ns = build_model_namespace(args)
    _, test_loader = data_provider(ns, "test", num_workers=0)

    # 初始化模型
    model = Model(ns).to(device)
    model.eval()
    backbone = model.module.backbone if hasattr(model, "module") else model.backbone
    stride   = int(backbone.patch_embedding.stride)

    # 收集容器: layer_idx -> list[Tensor[B, D]]
    hidden_lists: Dict[int, list[torch.Tensor]] = {
        li: [] for li in range(args.e_layers)
    }
    y_true_list: List[np.ndarray] = []
    n_collected  = 0
    pred_len     = args.pred_len

    def _pool(z: torch.Tensor, B: int, n_vars: int) -> torch.Tensor:
        """[BM, N, D] -> [B, D]  mean over N and n_vars"""
        return z.view(B, n_vars, -1, z.shape[-1]).mean(dim=[1, 2])

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if n_collected >= n_samples:
                break

            B_full    = batch_x.shape[0]
            n_vars_cur = batch_x.shape[-1]

            # 截取剩余所需量
            remain = n_samples - n_collected
            B      = B_full if remain >= B_full else remain
            if B < B_full:
                batch_x      = batch_x[:B]
                batch_y      = batch_y[:B]
                batch_x_mark = batch_x_mark[:B]
                batch_y_mark = batch_y_mark[:B]

            x_enc      = batch_x.float().to(device)
            x_mark_enc = batch_x_mark.float().to(device)
            x_dec      = batch_y[:, :batch_y.shape[1], :].float().to(device)
            x_mark_dec = batch_y_mark.float().to(device)

            # forward with hidden_states
            _, hidden_states = model(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                output_hidden_states=True
            )
            # hidden_states: list[Tensor[BM, N_or_N+1, D]]

            # n_vars 可能每次 batch 不同，动态推断
            BM0  = hidden_states[0].shape[0]
            n_vars_actual = n_vars_cur
            B_actual      = BM0 // n_vars_actual if BM0 % n_vars_actual == 0 else B

            for li, h in enumerate(hidden_states):
                if li >= args.e_layers:
                    break
                h_pooled = _pool(h, B_actual, n_vars_actual)
                hidden_lists[li].append(h_pooled.cpu())

            # 收集 y_true（最后一列通道的预测窗口均值作为默认标签）
            for i in range(B):
                y_window = batch_y[i, -pred_len:, 0].cpu().numpy()
                y_true_list.append(y_window)

            n_collected += B

            if verbose and (batch_idx + 1) % 20 == 0:
                print(f"  [Feature] {n_collected}/{n_samples} (batch {batch_idx+1})")

    # concat
    layer_hidden: Dict[str, torch.Tensor] = {}
    for li in range(args.e_layers):
        if not hidden_lists[li]:
            continue
        stacked = torch.cat(hidden_lists[li], dim=0)       # [n_collected, D]
        if stacked.shape[0] > n_samples:
            stacked = stacked[:n_samples]
        layer_hidden[f"layer_{li}"] = stacked
        if verbose:
            print(f"  [Layer {li}] shape={stacked.shape}, D={stacked.shape[-1]}")

    y_true = np.stack(y_true_list, axis=0)[:n_collected]

    if verbose:
        print(f"[Feature] Done. n_samples={n_collected}, y_true shape={y_true.shape}")

    return layer_hidden, y_true, stride


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Timer Layer-wise Probing — 特征提取")
    # 模型 / 数据路径
    parser.add_argument("--ckpt_path",     type=str,  required=True)
    parser.add_argument("--root_path",    type=str,  default="./datasets/")
    parser.add_argument("--data_path",    type=str,  default="ETTh1.csv")
    parser.add_argument("--data",         type=str,  default="ETTh1")
    parser.add_argument("--features",     type=str,  default="M")
    parser.add_argument("--embed",        type=str,  default="timeF")
    parser.add_argument("--freq",         type=str,  default="h")

    # 模型结构
    parser.add_argument("--seq_len",      type=int,  default=672)
    parser.add_argument("--label_len",     type=int,  default=576)
    parser.add_argument("--pred_len",      type=int,  default=96)
    parser.add_argument("--output_len",    type=int,  default=96)
    parser.add_argument("--patch_len",     type=int,  default=96)
    parser.add_argument("--d_model",      type=int,  default=1024)
    parser.add_argument("--d_ff",         type=int,  default=2048)
    parser.add_argument("--e_layers",     type=int,  default=8)
    parser.add_argument("--n_heads",      type=int,  default=8)
    parser.add_argument("--factor",       type=int,  default=3)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--activation",   type=str,  default="gelu")

    # 提取参数
    parser.add_argument("--n_samples",    type=int,  default=20000,
                        help="最大采样数量（0 = 全部）")
    parser.add_argument("--batch_size",    type=int,  default=64)
    parser.add_argument("--num_workers",   type=int,  default=4)
    parser.add_argument("--gpu_ids",       type=str,  default="0",
                        help="逗号分隔的 GPU ID，如 '0,1,2,3'")
    parser.add_argument("--device",        type=str,  default="cuda",
                        help="fallback 设备（gpu_ids 为空时使用）")

    # 输出路径
    parser.add_argument("--out_dir",       type=str,
                        default="./results/layerwise_probe_features",
                        help="特征输出目录")
    parser.add_argument("--features_name", type=str,
                        default=None,
                        help="输出 .pt 文件名（不含扩展名）；默认 {data}_{n_samples}.pt")

    args = parser.parse_args()

    # GPU 设备
    gpu_ids_raw = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if gpu_ids_raw:
        gpu_ids = [int(x) for x in gpu_ids_raw]
        device  = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        gpu_ids = []
        device  = torch.device(args.device if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    print(f"[Device] Primary={device}, visible={n_gpus} GPU(s), using={gpu_ids or [device]}")

    # 输出文件名
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    fname = args.features_name or f"{args.data}_{args.n_samples}"
    out_path = os.path.join(out_dir, f"{fname}.pt")
    json_path = os.path.join(out_dir, f"{fname}_labels.json")

    # ── 1. 收集特征 ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Layer-wise Feature Extraction")
    print(f"  n_samples={args.n_samples}, e_layers={args.e_layers}, pred_len={args.pred_len}")
    print(f"{'='*60}\n")

    layer_hidden, y_true, stride = collect_layerwise_features(
        args, n_samples=args.n_samples, device=device
    )

    if not layer_hidden:
        raise RuntimeError("No layer features collected! Check e_layers and model checkpoint.")

    n_collected  = y_true.shape[0]
    n_layers_out = len(layer_hidden)
    D            = next(iter(layer_hidden.values())).shape[-1]

    # ── 2. STL 标签 ──────────────────────────────────────────────────────────
    print(f"\n[Step 2] STL 分解生成语义标签...")
    labels = get_stl_labels(y_true)
    for name, arr in labels.items():
        print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}, shape={arr.shape}")

    # ── 3. torch.save ───────────────────────────────────────────────────────
    print(f"\n[Step 3] 保存特征到 {out_path} ...")

    save_dict = {
        "layer_hidden": {
            k: v.clone() for k, v in layer_hidden.items()
        },
        "y_true":       torch.from_numpy(y_true).clone(),
        "labels": {
            k: torch.from_numpy(v).clone() for k, v in labels.items()
        },
        "config": {
            "n_samples":  n_collected,
            "n_layers":   n_layers_out,
            "feat_dim":   D,
            "pred_len":   args.pred_len,
            "seq_len":    args.seq_len,
            "patch_len":  args.patch_len,
            "stride":     stride,
            "data":       args.data,
            "ckpt_path":  args.ckpt_path,
        },
    }
    torch.save(save_dict, out_path)
    print(f"[Save] Features saved: {out_path}")

    # 同时保存一份 JSON（不含 Tensor，仅供快速预览标签）
    json_labels = {
        k: v.tolist() for k, v in labels.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_samples": n_collected, "labels": json_labels, "config": save_dict["config"]}, f, indent=2)
    print(f"[Save] Labels JSON saved: {json_path}")

    # ── 4. 摘要 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  摘要")
    print(f"{'='*60}")
    print(f"  样本数:    {n_collected}")
    print(f"  Decoder层: {n_layers_out}")
    print(f"  特征维度:  {D} (Mean Pool over N patches)")
    print(f"  Patch len: {args.patch_len}, stride={stride}")
    print(f"  输出文件:  {out_path}")
    print(f"  标签文件:  {json_path}")
    print(f"\n✅ 完成！")


if __name__ == "__main__":
    main()
