#!/usr/bin/env python3
"""
ETTh1 + Timer: 特征注入（Feature Injection）对比实验

实验设计：
- 对照组：标准 Timer 推理（is_injection_test=False）
- 实验组：启用特征注入（is_injection_test=True）

特征注入机制（仅第 1 层执行一次）：
- 加载 etth1_mi_hsic_peaks 输出的全局 HSIC MI 曲线
- 高 MI patch（>Q3）均值作为 h_high 代表向量
- 低 MI patch（<Q1）更新为 h_low = h_low + w * h_high
  其中 w = alpha * |MI - Q1| / (Q3 - Q1)，按距离阈值加权
- 后续层保持注入结果继续传递

用法（单卡）：
    python experiments/etth1_feature_injection.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1

用法（多卡，必须用 torchrun）：
    torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_feature_injection.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 \
      --use_multi_gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model


# ─── 加载全局 HSIC JSON ──────────────────────────────────────────────────────

def load_hsic_mi_curve(hsic_file: str, layer_idx: int = -1) -> tuple[np.ndarray, float, float, int]:
    """
    加载 etth1_mi_hsic_peaks.py 输出的全局 MI JSON 文件。

    返回: (mi_curve, q3_threshold, q1_threshold, N) — MI 曲线、Q3 阈值、Q1 阈值、patch 数
    """
    with open(hsic_file, "r") as f:
        data = json.load(f)

    num_layers = data.get("num_layers", 0)
    layers = data.get("layers", {})

    if layer_idx < 0:
        actual_layer = num_layers + layer_idx
    else:
        actual_layer = layer_idx

    layer_key = str(actual_layer)
    if layer_key not in layers:
        available = list(layers.keys())
        raise ValueError(f"Layer {layer_key} not found. Available layers: {available}")

    layer_data = layers[layer_key]
    mi_curve = np.array(layer_data["hsic_curve"], dtype=np.float64)
    q3_threshold = layer_data["q3_threshold"]
    valid_mi = mi_curve[np.isfinite(mi_curve)]
    q1_threshold = float(np.percentile(valid_mi, 25))

    return mi_curve, q3_threshold, q1_threshold, len(mi_curve)


def build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """构建完整的配置 namespace"""
    ns = argparse.Namespace(**vars(args))
    defaults = {
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
        "model_id": "feature_injection_exp",
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
        "dropout": 0.1,
        "activation": "gelu",
        "embed": "timeF",
        "freq": "h",
        "features": "M",
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def metric(y_true, y_pred):
    """计算预测评估指标"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": float(rmse)}


def run_experiment(ns: argparse.Namespace, device: torch.device,
                   is_injection_test: bool, injection_alpha: float,
                   hsic_mi_curve, hsic_q3, hsic_q1,
                   rank: int = 0) -> dict:
    """运行单次实验"""
    ns = build_namespace(ns)
    ns.is_injection_test = is_injection_test
    ns.injection_alpha = injection_alpha
    ns.hsic_mi_curve = hsic_mi_curve
    ns.hsic_q3 = hsic_q3
    ns.hsic_q1 = hsic_q1

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"实验配置:")
        print(f"  is_injection_test = {is_injection_test}")
        print(f"  injection_alpha   = {injection_alpha}")
        if is_injection_test and hsic_q3 is not None:
            print(f"  hsic_q3           = {hsic_q3:.6f}")
            print(f"  hsic_q1           = {hsic_q1:.6f}")
        print(f"{'='*60}\n")

    model = Model(ns).to(device)
    model.eval()

    if hasattr(ns, 'use_multi_gpu') and ns.use_multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    _, loader = data_provider(ns, flag="test")

    preds, trues = [], []
    inference_time = 0.0

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            t_start = time.time()
            dec_inp = batch_y[:, :ns.label_len, :]

            if getattr(ns, 'use_ims', False):
                y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
            else:
                y_future = batch_y[:, -ns.pred_len:, :]

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            inference_time += time.time() - t_start

            if getattr(ns, 'use_ims', False):
                pred = outputs[:, -ns.pred_len:, :]
            else:
                pred = outputs[:, -ns.pred_len:, :]

            preds.append(pred.cpu().numpy())
            trues.append(y_future.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    metrics = metric(trues, preds)

    if rank == 0:
        metrics["inference_time"] = inference_time
        metrics["avg_time_per_sample"] = inference_time / len(preds)
        print(f"\n结果:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


def main():
    p = argparse.ArgumentParser(description="ETTh1 特征注入对比实验")
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Timer 模型检查点路径")
    p.add_argument("--root_path", type=str, default="./datasets/")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--data", type=str, default="ETTh1")
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--label_len", type=int, default=576)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--output_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=24)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--e_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./results/feature_injection_exp/")
    p.add_argument("--use_ims", action="store_true",
                   help="启用 IMS (Iterated Multi-Step) 模式，与原型提取保持一致")
    p.add_argument("--use_multi_gpu", action="store_true",
                   help="使用多 GPU (torchrun)")
    p.add_argument("--skip_baseline", action="store_true",
                   help="跳过基线实验（仅运行特征注入）")
    p.add_argument("--skip_injection", action="store_true",
                   help="跳过特征注入实验（仅运行基线）")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="特征注入系数 alpha")
    p.add_argument("--hsic_file", type=str, default=None,
                   help="etth1_mi_hsic_peaks 输出的 JSON 文件路径")
    p.add_argument("--hsic_layer", type=int, default=-1,
                   help="使用哪一层的 HSIC MI 曲线（-1=最后一层）")
    args = p.parse_args()

    rank = 0
    local_rank = 0
    if args.use_multi_gpu:
        if "WORLD_SIZE" not in os.environ:
            raise RuntimeError("多 GPU 需要 torchrun 设置 WORLD_SIZE/RANK/LOCAL_RANK")
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if args.use_multi_gpu else args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 加载全局 HSIC MI 曲线 ────────────────────────────────────────────────
    hsic_mi_curve = None
    hsic_q3 = None
    hsic_q1 = None
    if args.hsic_file and os.path.exists(args.hsic_file):
        hsic_mi_curve, hsic_q3, hsic_q1, N_hsic = load_hsic_mi_curve(args.hsic_file, args.hsic_layer)
        if rank == 0:
            print(f"[Feature Injection] Loaded HSIC: layer={args.hsic_layer}, N={N_hsic}, Q3={hsic_q3:.6f}, Q1={hsic_q1:.6f}")
    else:
        if rank == 0:
            print("[Feature Injection] WARNING: --hsic_file not provided; falling back to positional injection")
        if args.skip_injection:
            pass
        else:
            print("[Feature Injection] WARNING: --hsic_file required for injection; skipping injection group")
            args.skip_injection = True

    results = {}

    if not args.skip_baseline:
        baseline_metrics = run_experiment(
            args, device,
            is_injection_test=False,
            injection_alpha=0.0,
            hsic_mi_curve=None, hsic_q3=None, hsic_q1=None,
            rank=rank
        )
        results["baseline"] = baseline_metrics

    if not args.skip_injection:
        injection_metrics = run_experiment(
            args, device,
            is_injection_test=True,
            injection_alpha=args.alpha,
            hsic_mi_curve=hsic_mi_curve, hsic_q3=hsic_q3, hsic_q1=hsic_q1,
            rank=rank
        )
        results["injection"] = injection_metrics

    if rank == 0 and len(results) == 2:
        print("\n" + "="*80)
        print("│                     Baseline vs Feature Injection 对比分析                    │")
        print("="*80)
        print(f"{'指标':<10} {'Baseline':>15} {'Feature Injection':>18} {'差值 (Δ)':>15} {'改进率':>12}")
        print("-"*80)
        
        improvement_dict = {}
        for key in ["MSE", "MAE", "RMSE"]:
            baseline_val = results["baseline"].get(key, float('inf'))
            inject_val = results["injection"].get(key, float('inf'))
            if baseline_val != float('inf') and inject_val != float('inf'):
                delta = inject_val - baseline_val
                pct = delta / baseline_val * 100
                improvement_dict[key] = {
                    "baseline": baseline_val,
                    "injection": inject_val,
                    "delta": delta,
                    "pct": pct
                }
                # 改进率：MSE/MAE/RMSE 越小越好，所以改进为负值表示变好
                arrow = "↓" if delta < 0 else "↑"
                print(f"{key:<10} {baseline_val:>15.6f} {inject_val:>18.6f} {delta:>+15.6f} {pct:>+11.2f}% {arrow}")

        print("="*80)
        
        # 打印总结
        print("\n📊 实验总结:")
        mse_improve = improvement_dict.get("MSE", {}).get("pct", 0)
        mae_improve = improvement_dict.get("MAE", {}).get("pct", 0)
        rmse_improve = improvement_dict.get("RMSE", {}).get("pct", 0)
        
        if mse_improve < 0:
            print(f"  ✅ MSE 降低 {-mse_improve:.2f}%（预测精度提升）")
        else:
            print(f"  ⚠️  MSE 上升 {mse_improve:.2f}%（预测精度下降）")
            
        if mae_improve < 0:
            print(f"  ✅ MAE 降低 {-mae_improve:.2f}%（预测误差减少）")
        else:
            print(f"  ⚠️  MAE 上升 {mae_improve:.2f}%（预测误差增加）")
        
        # 保存结果到文件
        output_file = os.path.join(args.output_dir, "comparison_results.txt")
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("        Feature Injection 对比实验结果 (Baseline vs Feature Injection)\n")
            f.write("="*80 + "\n\n")
            
            f.write("【各实验详细指标】\n")
            for exp_name, metrics in results.items():
                f.write(f"\n实验: {exp_name}\n")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {v:.6f}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("【差值分析 (Delta = Injection - Baseline)】\n")
            f.write("="*80 + "\n")
            f.write(f"{'指标':<10} {'Baseline':>15} {'Injection':>15} {'差值 (Δ)':>15} {'改进率':>12}\n")
            f.write("-"*80 + "\n")
            
            for key in ["MSE", "MAE", "RMSE"]:
                if key in improvement_dict:
                    d = improvement_dict[key]
                    arrow = "↓" if d["delta"] < 0 else "↑"
                    f.write(f"{key:<10} {d['baseline']:>15.6f} {d['injection']:>15.6f} "
                           f"{d['delta']:>+15.6f} {d['pct']:>+11.2f}% {arrow}\n")
            
            f.write("="*80 + "\n")
            f.write(f"\n总结: MSE 改进率 = {mse_improve:+.2f}%, MAE 改进率 = {mae_improve:+.2f}%\n")
            if mse_improve < 0 and mae_improve < 0:
                f.write("结论: Feature Injection 有效提升了预测精度\n")
            elif mse_improve > 0 or mae_improve > 0:
                f.write("结论: Feature Injection 导致预测精度下降\n")
            else:
                f.write("结论: Feature Injection 对预测精度无明显影响\n")

        print(f"\n📁 详细结果已保存到: {output_file}")

    if args.use_multi_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
