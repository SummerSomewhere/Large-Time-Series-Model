#!/usr/bin/env python3
"""
ETTh1 + Timer: Prototype Prompting 对比实验

基于论文《Thinking Tokens are Information Peaks》思想实现的原型提示机制：

第一阶段：原型提取（需先运行 prototype_extraction.py）
  - 提取高 MI (HSIC) patch 的隐藏表示
  - 识别 MI > Q3 的峰值 patch
  - 计算峰值 patch 的全局平均值，生成原型向量 h_proto

第二阶段：原型注入评估
  - 加载预计算的 h_proto
  - 将原型向量作为 Prefix 拼接到输入序列最前面（位置 0）
  - 原型作为全局引导，所有 patch 都能看到它
  - 预测时截断原型，只保留原始 T 个时间步的预测

实验设计：
- 对照组：标准 Timer 推理（use_prototype=False）
- 实验组：启用原型注入（use_prototype=True）

用法（单卡）：
    python experiments/etth1_prototype_injection.py --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --prototype_path ./results/prototype_extraction/h_proto_layer7.pt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1

用法（多卡，必须用 torchrun）：
    torchrun --nnodes=1 --nproc_per_node=4 experiments/etth1_prototype_injection.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --prototype_path ./results/prototype_extraction/h_proto_layer7.pt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 \
      --use_multi_gpu

绘制注意力热力图（可选，会运行两次推理）：
    python experiments/etth1_prototype_injection.py \
      --ckpt_path checkpoints/Timer_forecast_1.0.ckpt \
      --prototype_path ./results/prototype_extraction/h_proto_layer7.pt \
      --root_path ./datasets/ --data_path ETTh1.csv --data ETTh1 \
      --plot_attention \
      --output_dir ./results/prototype_injection_exp/

热力图输出位置：./results/prototype_injection_exp/attention_heatmaps/
  - baseline_attention.png   （7×7 patch 注意力矩阵，8层）
  - prototype_attention.png  （8×8 patch（含原型），8层）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider
from models.Timer import Model
from utils.tools import EarlyStopping, LargeScheduler


def plot_combined_attention_heatmaps(
    baseline_attns: list,
    prototype_attns: list,
    low_prototype_attns: list | None,
    n_heads: int,
    n_patches: int,
    output_path: str,
    baseline_title: str = "Baseline",
    prototype_title: str = "High-MI Proto",
    low_prototype_title: str = "Low-MI Proto"
):
    """
    将 Baseline / High-MI Prototype / Low-MI Prototype 的注意力热力图（每层每头独立）
    合并绘制在一张大图中，便于三组横向对比。

    布局: n_groups 行 × (n_layers × n_heads) 列
    - 第1行: Baseline（L = n_patches）
    - 第2行: High-MI Prototype（L = n_patches + 1）
    - 第3行: Low-MI Prototype（L = n_patches + 1，可选）
    - 每行每列对应一个 [L, L] 或 [L+1, L+1] 的单头注意力热力图

    参数:
        baseline_attns:     list of [H, L, L] 张量，len = n_layers
        prototype_attns:   list of [H, L+1, L+1] 张量，len = n_layers
        low_prototype_attns: list of [H, L+1, L+1] 张量，len = n_layers，或 None
        n_heads:           注意力头数量
        n_patches:         原始 patch 数量（不含 prototype）
        output_path:       保存路径
        baseline_title:    图总标题（Baseline 行）
        prototype_title:   图总标题（High-MI 行）
        low_prototype_title: 图总标题（Low-MI 行）
    """
    n_layers = len(baseline_attns)
    seq_len_base = n_patches
    seq_len_proto = n_patches + 1
    n_groups = 3 if low_prototype_attns is not None else 2

    cell_w = 2.5
    cell_h = 2.2

    fig, axes = plt.subplots(
        n_groups, n_layers * n_heads,
        figsize=(n_layers * n_heads * cell_w + 1.5, n_groups * cell_h + 1.5),
        squeeze=False
    )
    title_parts = ["Baseline", "High-MI Proto", "Low-MI Proto"]
    fig.suptitle(
        f"Decoder Attention Heatmaps: {title_parts[0]} (top) → {title_parts[n_groups - 1]} (bottom)",
        fontsize=14, fontweight="bold", y=0.99
    )

    # 行配置: (attns, seq_len, tick_labels, row_title)
    row_configs = [
        (baseline_attns, seq_len_base, [f"P{i}" for i in range(n_patches)], baseline_title),
        (prototype_attns, seq_len_proto, ["Proto"] + [f"P{i}" for i in range(n_patches)], prototype_title),
    ]
    if low_prototype_attns is not None:
        row_configs.append(
            (low_prototype_attns, seq_len_proto, ["Proto"] + [f"P{i}" for i in range(n_patches)], low_prototype_title)
        )

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            col = layer_idx * n_heads + head_idx

            for row_idx, (attns, seq_len, tick_labels, row_title) in enumerate(row_configs):
                ax = axes[row_idx][col]
                mat: np.ndarray = attns[layer_idx].cpu().numpy()[head_idx]
                vmax = mat.max()

                ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
                ax.set_xticks(range(seq_len))
                ax.set_yticks(range(seq_len))
                ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=45, ha="right")
                ax.set_yticklabels(tick_labels, fontsize=5.5)

                if seq_len <= 10:
                    for i in range(seq_len):
                        for j in range(seq_len):
                            val = mat[i, j]
                            c = "white" if val > vmax * 0.65 else "black"
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                   fontsize=4, color=c)

                # 行标题（列标题只在第一列标注层号，行标题在行首标注组名）
                if layer_idx == 0 and head_idx == 0:
                    ax.set_title(f"{row_title}", fontsize=9, pad=4)
                elif head_idx == 0:
                    ax.set_title(f"L{layer_idx + 1}", fontsize=8, pad=3)
                elif layer_idx == 0:
                    ax.set_title(f"H{head_idx}", fontsize=7, pad=2)

                if col == 0:
                    ax.set_ylabel(row_title, fontsize=8, labelpad=3)

    plt.subplots_adjust(hspace=0.45, wspace=0.25, left=0.04, right=0.98, top=0.92, bottom=0.04)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Attention] Saved combined heatmap: {output_path}")


def plot_attention_heatmaps(all_attns: list, n_heads: int, has_prototype: bool,
                              n_patches: int, output_path: str, title: str):
    """
    为 Decoder 的每一层、每一个 head 绘制独立的热力图。

    布局: n_layers 行 × n_heads 列
    每个格子为 [L, L]（Baseline）或 [L+1, L+1]（Prototype）的单头注意力矩阵。

    参数:
        all_attns: list of [H, L, L] 张量，len = n_layers
        n_heads:   注意力头数量
        has_prototype: 是否注入了原型 token（L = n_patches+1 时为 True）
        n_patches: 原始 patch 数量（不含 prototype）
        output_path: 保存路径
        title: 图标题
    """
    n_layers = len(all_attns)
    seq_len = n_patches + (1 if has_prototype else 0)

    cell_w = 2.5
    cell_h = 2.2
    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(n_heads * cell_w + 1.0, n_layers * cell_h + 1.5),
        squeeze=False
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    for layer_idx in range(n_layers):
        attn_layer: np.ndarray = all_attns[layer_idx].cpu().numpy()  # [H, L, L] 或 [H, L+1, L+1]

        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            mat = attn_layer[head_idx]  # [L, L]
            vmax = mat.max()

            ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))

            if has_prototype:
                tick_labels = ["Proto"] + [f"P{i}" for i in range(n_patches)]
            else:
                tick_labels = [f"P{i}" for i in range(n_patches)]

            ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")
            ax.set_yticklabels(tick_labels, fontsize=6)

            if seq_len <= 10:
                for i in range(seq_len):
                    for j in range(seq_len):
                        val = mat[i, j]
                        c = "white" if val > vmax * 0.65 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=4.5, color=c)

            # 标题和标签
            if layer_idx == 0:
                ax.set_title(f"H{head_idx}", fontsize=8, pad=3)
            if head_idx == 0:
                ax.set_ylabel(f"L{layer_idx + 1}", fontsize=8)

    plt.subplots_adjust(hspace=0.4, wspace=0.25, left=0.05, right=0.98, top=0.93, bottom=0.05)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Attention] Saved: {output_path}")


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
        "model_id": "prototype_injection_exp",
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
        "stride": 96,
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


def plot_sample_prediction_comparison(
    baseline_preds: np.ndarray,
    baseline_trues: np.ndarray,
    proto_preds: np.ndarray | None,
    low_proto_preds: np.ndarray | None,
    proto_trues: np.ndarray | None,
    n_show: int = 10,
    save_path: str = "./results/prototype_injection_exp/sample_comparison.png"
):
    """
    随机抽取 n_show 个样本，绘制 Baseline vs Prototype 的预测曲线对比图。

    参数:
        baseline_preds:   [N, pred_len, 1] Baseline 预测
        baseline_trues:  [N, pred_len, 1] 真值
        proto_preds:     [N, pred_len, 1] High-MI 原型引导预测，可为 None
        low_proto_preds: [N, pred_len, 1] Low-MI 原型引导预测，可为 None
        proto_trues:     [N, pred_len, 1] 原型组真值（与 baseline_trues 相同）
        n_show:          展示的样本数
        save_path:       保存路径
    """
    N = baseline_preds.shape[0]
    n_show = min(n_show, N)
    indices = np.random.choice(N, n_show, replace=False)

    pred_len = baseline_preds.shape[1]
    t_pred = np.linspace(0, 1, pred_len)

    # 确定列数：Baseline | High-MI | Low-MI | All（叠在一起）
    has_high = proto_preds is not None
    has_low = low_proto_preds is not None
    n_cols = 1 + int(has_high) + int(has_low) + 1  # baseline + high + low + all
    n_rows = n_show

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), squeeze=False)
    fig.suptitle("Prototype Injection: Baseline vs High-MI / Low-MI Proto (Random 10 Samples)", fontsize=13, fontweight="bold")

    for row_i, idx in enumerate(indices):
        bl_pred = baseline_preds[idx, :, 0] if baseline_preds.ndim == 3 else baseline_preds[idx]
        bl_true = baseline_trues[idx, :, 0] if baseline_trues.ndim == 3 else baseline_trues[idx]

        col = 0
        ax = axes[row_i, col]
        ax.plot(t_pred, bl_pred, 'b-', alpha=0.85, lw=1.4, label='Baseline')
        ax.plot(t_pred, bl_true, 'k--', alpha=0.35, lw=1, label='True')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        if row_i == 0:
            ax.set_title('Baseline', fontsize=9, fontweight='bold')
        ax.set_ylabel(f"#{idx}", fontsize=7)

        if has_high:
            col += 1
            ax = axes[row_i, col]
            hi_pred = proto_preds[idx, :, 0] if proto_preds.ndim == 3 else proto_preds[idx]
            ax.plot(t_pred, hi_pred, 'r-', alpha=0.85, lw=1.4, label='High-MI')
            ax.plot(t_pred, bl_true, 'k--', alpha=0.35, lw=1, label='True')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3)
            if row_i == 0:
                ax.set_title('High-MI Proto', fontsize=9, fontweight='bold')

        if has_low:
            col += 1
            ax = axes[row_i, col]
            lo_pred = low_proto_preds[idx, :, 0] if low_proto_preds.ndim == 3 else low_proto_preds[idx]
            ax.plot(t_pred, lo_pred, 'g-', alpha=0.85, lw=1.4, label='Low-MI')
            ax.plot(t_pred, bl_true, 'k--', alpha=0.35, lw=1, label='True')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3)
            if row_i == 0:
                ax.set_title('Low-MI Proto', fontsize=9, fontweight='bold')

        # All 列：叠在一起
        col += 1
        ax = axes[row_i, col]
        ax.plot(t_pred, bl_pred, 'b-', alpha=0.7, lw=1, label='Baseline')
        if has_high:
            ax.plot(t_pred, hi_pred, 'r-', alpha=0.7, lw=1, label='High-MI')
        if has_low:
            ax.plot(t_pred, lo_pred, 'g-', alpha=0.7, lw=1, label='Low-MI')
        ax.plot(t_pred, bl_true, 'k--', alpha=0.5, lw=1.1, label='True')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        if row_i == 0:
            ax.set_title('All', fontsize=9, fontweight='bold')

    # 图例
    handles, labels = axes[0, n_cols - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.01), fontsize=8, ncol=4)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Sample comparison saved: {save_path}")


def finetune_model(
    args: argparse.Namespace,
    device: torch.device,
    use_prototype: bool,
    prototype_path: str | None,
    rank: int = 0,
    world_size: int = 1,
) -> Model:
    """
    微调模型，支持原型注入。

    参数:
        args: 实验参数
        device: 设备
        use_prototype: 是否使用原型注入
        prototype_path: 原型向量路径
        rank: 当前进程 rank
        world_size: 总进程数

    返回:
        微调后的模型
    """
    ns = build_namespace(args)
    ns.use_prototype = use_prototype
    ns.prototype_path = prototype_path
    ns.finetune_epochs = getattr(args, 'finetune_epochs', 5)
    ns.learning_rate = getattr(args, 'finetune_lr', 3e-5)
    ns.patience = getattr(args, 'finetune_patience', 3)
    ns.checkpoints = getattr(args, 'checkpoints', './checkpoints/')

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"微调配置:")
        print(f"  use_prototype   = {use_prototype}")
        print(f"  prototype_path  = {prototype_path}")
        print(f"  finetune_epochs = {ns.finetune_epochs}")
        print(f"  learning_rate   = {ns.learning_rate}")
        print(f"  patience        = {ns.patience}")
        print(f"{'='*60}\n")

    # 构建模型
    model = Model(ns).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 获取数据加载器
    train_data, train_loader = data_provider(ns, flag='train')
    val_data, val_loader = data_provider(ns, flag='val')

    # 设置优化器
    model_optim = optim.Adam(model.parameters(), lr=ns.learning_rate)
    criterion = nn.MSELoss()
    scheduler = LargeScheduler(ns, model_optim)

    # 创建保存路径
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    proto_suffix = "high_mi" if use_prototype else "baseline"
    if prototype_path and "low" in prototype_path.lower():
        proto_suffix = "low_mi"
    setting_name = f"proto_{proto_suffix}_ft"
    ckpt_path = os.path.join(ns.checkpoints, f"{setting_name}_{timestamp}")
    if rank == 0:
        os.makedirs(ckpt_path, exist_ok=True)

    # 早停
    early_stopping = EarlyStopping(patience=ns.patience, verbose=True)

    for epoch in range(ns.finetune_epochs):
        iter_count = 0
        loss_val = torch.tensor(0., device=device)
        count = torch.tensor(0., device=device)

        model.train()
        epoch_time = time.time()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{ns.finetune_epochs} 开始训练...")

        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -ns.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :ns.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if getattr(ns, 'use_ims', False):
                loss = criterion(outputs[:, -ns.seq_len:, :], batch_y)
            else:
                loss = criterion(outputs[:, -ns.pred_len:, :], batch_y[:, -ns.pred_len:, :])

            loss_val += loss.item()
            count += 1

            if batch_idx % 100 == 0 and rank == 0:
                print(f"  iter {batch_idx}, loss = {loss.item():.6f}")

            loss.backward()
            model_optim.step()

        # 同步多卡损失
        if world_size > 1:
            dist.barrier()
            dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        train_loss = loss_val.item() / max(count.item(), 1)

        # 验证
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -ns.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :ns.label_len, :], dec_inp], dim=1).float().to(device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if getattr(ns, 'use_ims', False):
                    loss = criterion(outputs[:, -ns.seq_len:, :], batch_y)
                else:
                    loss = criterion(outputs[:, -ns.pred_len:, :], batch_y[:, -ns.pred_len:, :])

                val_loss += loss.item()
                val_count += 1

        if world_size > 1:
            dist.all_reduce(torch.tensor([val_loss, val_count], device=device), op=dist.ReduceOp.SUM)
        val_loss = val_loss / max(val_count, 1)

        if rank == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        early_stopping(val_loss, model, ckpt_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.schedule_epoch(epoch)

    # 加载最佳模型
    best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        if rank == 0:
            print(f"加载最佳模型: {best_model_path}")

    if world_size > 1:
        dist.barrier()

    return model


def run_experiment(ns: argparse.Namespace, device: torch.device,
                   use_prototype: bool, prototype_path: str | None,
                   rank: int = 0,
                   collect_attention: bool = False,
                   collect_preds: bool = False) -> dict | tuple[dict, list] | tuple[dict, np.ndarray, np.ndarray]:
    """运行单次实验，可选收集注意力权重和预测结果用于可视化。"""
    ns = build_namespace(ns)
    ns.use_prototype = use_prototype
    ns.prototype_path = prototype_path
    ns.output_attention = collect_attention  # 启用注意力收集

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"实验配置:")
        print(f"  use_prototype   = {use_prototype}")
        print(f"  prototype_path  = {prototype_path}")
        print(f"  output_attention= {collect_attention}")
        print(f"{'='*60}\n")

    model = Model(ns).to(device)
    model.eval()

    if hasattr(ns, 'use_multi_gpu') and ns.use_multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    _, loader = data_provider(ns, flag="test")

    preds, trues = [], []
    inference_time = 0.0
    all_attns = []  # 存储每层的注意力权重

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

            ret = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            inference_time += time.time() - t_start

            if collect_attention:
                outputs, attns = ret  # ret 是 (dec_out, attns) 元组
                # attns: list of [B*M, H, L, L]，len = n_layers
                # 只对 batch*M 求平均，保留 H 维度 -> [H, L, L]
                cap = 8
                for layer_idx, a in enumerate(attns[:cap]):
                    avg_a = a.mean(dim=0)  # [H, L, L]
                    if batch_idx == 0:
                        all_attns.append(avg_a)
                    else:
                        all_attns[layer_idx] = all_attns[layer_idx] + avg_a
            else:
                outputs = ret

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

    # 返回前归一化：所有 batch 的注意力权重相加后除以 batch 数
    all_attns = [a / (batch_idx + 1) for a in all_attns]
    if collect_attention and collect_preds:
        return metrics, all_attns, preds, trues
    if collect_attention:
        return metrics, all_attns
    if collect_preds:
        return metrics, preds, trues
    return metrics


def main():
    p = argparse.ArgumentParser(description="ETTh1 原型注入对比实验")
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Timer 模型检查点路径")
    p.add_argument("--prototype_path", type=str, required=True,
                   help="原型向量 .pt 文件路径")
    p.add_argument("--root_path", type=str, default="./datasets/")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--data", type=str, default="ETTh1")
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--label_len", type=int, default=576)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--output_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=96)
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
    p.add_argument("--stride", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./results/prototype_injection_exp/")
    p.add_argument("--use_ims", action="store_true",
                   help="启用 IMS (Iterated Multi-Step) 模式，与原型提取保持一致")
    p.add_argument("--use_multi_gpu", action="store_true",
                   help="使用多 GPU (torchrun)")
    p.add_argument("--skip_baseline", action="store_true",
                   help="跳过基线实验（仅运行原型注入）")
    p.add_argument("--skip_prototype", action="store_true",
                   help="跳过原型注入实验（仅运行基线）")
    p.add_argument("--low_proto_path", type=str, default=None,
                   help="Low-MI 原型向量 .pt 文件路径（可选，MI <= Q1 的 patch 融合）")
    p.add_argument("--skip_low_prototype", action="store_true",
                   help="跳过 Low-MI 原型注入实验")
    p.add_argument("--plot_attention", action="store_true",
                   help="绘制 Baseline 和 Prototype 的 Decoder 注意力热力图（每层每头独立）")
    p.add_argument("--plot_samples", action="store_true",
                   help="随机抽取 10 个样本绘制 Baseline vs Prototype 预测曲线对比图")
    p.add_argument("--n_show_samples", type=int, default=10,
                   help="随机抽取的样本数（默认 10）")
    # ── 微调参数 ────────────────────────────────────────────────────────────────
    p.add_argument("--finetune", action="store_true",
                   help="启用微调模式（先微调再测试）")
    p.add_argument("--finetune_epochs", type=int, default=5,
                   help="微调轮数（默认 5）")
    p.add_argument("--finetune_lr", type=float, default=3e-5,
                   help="微调学习率（默认 3e-5）")
    p.add_argument("--finetune_patience", type=int, default=3,
                   help="微调早停耐心值（默认 3）")
    p.add_argument("--checkpoints", type=str, default="./checkpoints/",
                   help="模型检查点保存路径")
    p.add_argument("--skip_finetune_baseline", action="store_true",
                   help="跳过基线微调（仅在 --finetune 时有效）")
    p.add_argument("--skip_finetune_high", action="store_true",
                   help="跳过 High-MI 原型微调（仅在 --finetune 时有效）")
    p.add_argument("--skip_finetune_low", action="store_true",
                   help="跳过 Low-MI 原型微调（仅在 --finetune 时有效）")
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

    results = {}

    # n_patches = seq_len / patch_len = 672 / 96 = 7
    n_patches = args.seq_len // args.patch_len
    n_heads = args.n_heads

    world_size = int(os.environ.get("WORLD_SIZE", "1")) if args.use_multi_gpu else 1

    # ═══════════════════════════════════════════════════════════════════════════
    # 微调模式：先微调三个模型（Baseline / High-MI / Low-MI），再测试对比
    # ═══════════════════════════════════════════════════════════════════════════
    finetuned_models = {}

    if args.finetune:
        # ── 1. 基线微调 ───────────────────────────────────────────────────────
        if not args.skip_finetune_baseline:
            print("\n" + "="*60)
            print("  [微调] Baseline 模型")
            print("="*60)
            baseline_model = finetune_model(
                args=args,
                device=device,
                use_prototype=False,
                prototype_path=None,
                rank=rank,
                world_size=world_size,
            )
            finetuned_models["baseline"] = baseline_model

        # ── 2. High-MI 原型微调 ──────────────────────────────────────────────
        if not args.skip_finetune_high:
            print("\n" + "="*60)
            print("  [微调] High-MI Prototype 模型")
            print("="*60)
            high_model = finetune_model(
                args=args,
                device=device,
                use_prototype=True,
                prototype_path=args.prototype_path,
                rank=rank,
                world_size=world_size,
            )
            finetuned_models["high_mi"] = high_model

        # ── 3. Low-MI 原型微调 ────────────────────────────────────────────────
        if args.low_proto_path and not args.skip_finetune_low:
            print("\n" + "="*60)
            print("  [微调] Low-MI Prototype 模型")
            print("="*60)
            low_model = finetune_model(
                args=args,
                device=device,
                use_prototype=True,
                prototype_path=args.low_proto_path,
                rank=rank,
                world_size=world_size,
            )
            finetuned_models["low_mi"] = low_model

        # ── 使用微调后的模型进行测试 ─────────────────────────────────────────
        print("\n" + "="*60)
        print("  [测试] 使用微调后的模型进行测试对比")
        print("="*60)

        # 重新构建 namespace（确保参数正确）
        from models.Timer import Model as TimerModel
        import copy

        # Baseline 测试
        if "baseline" in finetuned_models and not args.skip_baseline:
            print("\n  [测试] Baseline 模型...")
            baseline_model = finetuned_models["baseline"]
            ns = build_namespace(args)
            ns.use_prototype = False
            ns.prototype_path = None
            ns.output_attention = False

            _, loader = data_provider(ns, flag="test")
            baseline_model.eval()

            preds, trues = [], []
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)

                    dec_inp = batch_y[:, :ns.label_len, :]
                    if getattr(ns, 'use_ims', False):
                        y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
                    else:
                        y_future = batch_y[:, -ns.pred_len:, :]

                    outputs = baseline_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if getattr(ns, 'use_ims', False):
                        pred = outputs[:, -ns.pred_len:, :]
                    else:
                        pred = outputs[:, -ns.pred_len:, :]

                    preds.append(pred.cpu().numpy())
                    trues.append(y_future.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            results["baseline"] = metric(trues, preds)

        # High-MI Prototype 测试
        if "high_mi" in finetuned_models and not args.skip_prototype:
            print("\n  [测试] High-MI Prototype 模型...")
            high_model = finetuned_models["high_mi"]
            ns = build_namespace(args)
            ns.use_prototype = True
            ns.prototype_path = args.prototype_path
            ns.output_attention = False

            _, loader = data_provider(ns, flag="test")
            high_model.eval()

            preds, trues = [], []
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)

                    dec_inp = batch_y[:, :ns.label_len, :]
                    if getattr(ns, 'use_ims', False):
                        y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
                    else:
                        y_future = batch_y[:, -ns.pred_len:, :]

                    outputs = high_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if getattr(ns, 'use_ims', False):
                        pred = outputs[:, -ns.pred_len:, :]
                    else:
                        pred = outputs[:, -ns.pred_len:, :]

                    preds.append(pred.cpu().numpy())
                    trues.append(y_future.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            results["prototype"] = metric(trues, preds)

        # Low-MI Prototype 测试
        if "low_mi" in finetuned_models and args.low_proto_path and not args.skip_low_prototype:
            print("\n  [测试] Low-MI Prototype 模型...")
            low_model = finetuned_models["low_mi"]
            ns = build_namespace(args)
            ns.use_prototype = True
            ns.prototype_path = args.low_proto_path
            ns.output_attention = False

            _, loader = data_provider(ns, flag="test")
            low_model.eval()

            preds, trues = [], []
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)

                    dec_inp = batch_y[:, :ns.label_len, :]
                    if getattr(ns, 'use_ims', False):
                        y_future = batch_y[:, ns.label_len:ns.label_len + ns.pred_len, :]
                    else:
                        y_future = batch_y[:, -ns.pred_len:, :]

                    outputs = low_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if getattr(ns, 'use_ims', False):
                        pred = outputs[:, -ns.pred_len:, :]
                    else:
                        pred = outputs[:, -ns.pred_len:, :]

                    preds.append(pred.cpu().numpy())
                    trues.append(y_future.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            results["low_prototype"] = metric(trues, preds)

    # ── 注意力可视化：Baseline × High-MI Prototype × Low-MI Prototype ──────────
    elif args.plot_attention:
        print("\n" + "="*60)
        print("  [Attention] 收集 Baseline 注意力权重 ...")
        print("="*60)
        baseline_metrics, baseline_attns = run_experiment(
            args, device,
            use_prototype=False, prototype_path=None,
            rank=rank, collect_attention=True
        )
        results["baseline"] = baseline_metrics

        print("\n" + "="*60)
        print("  [Attention] 收集 High-MI Prototype 注意力权重 ...")
        print("="*60)
        prototype_metrics, prototype_attns = run_experiment(
            args, device,
            use_prototype=True, prototype_path=args.prototype_path,
            rank=rank, collect_attention=True
        )
        results["prototype"] = prototype_metrics

        # Low-MI Prototype 实验（可选）
        low_prototype_attns = None
        if args.low_proto_path and not args.skip_low_prototype:
            print("\n" + "="*60)
            print("  [Attention] 收集 Low-MI Prototype 注意力权重 ...")
            print("="*60)
            low_proto_metrics, low_prototype_attns = run_experiment(
                args, device,
                use_prototype=True, prototype_path=args.low_proto_path,
                rank=rank, collect_attention=True
            )
            results["low_prototype"] = low_proto_metrics

        # ── 绘制热力图 ──────────────────────────────────────────────────────────
        if rank == 0:
            attn_dir = os.path.join(args.output_dir, "attention_heatmaps")
            os.makedirs(attn_dir, exist_ok=True)

            # 合并大图：三组对比（Baseline / High-MI / Low-MI）
            plot_combined_attention_heatmaps(
                baseline_attns=baseline_attns,
                prototype_attns=prototype_attns,
                low_prototype_attns=low_prototype_attns,
                n_heads=n_heads,
                n_patches=n_patches,
                output_path=os.path.join(attn_dir, "attention_comparison.png"),
                baseline_title="Baseline",
                prototype_title="High-MI Proto",
                low_prototype_title="Low-MI Proto"
            )

            # 保留单独的图（可选）
            plot_attention_heatmaps(
                all_attns=baseline_attns,
                n_heads=n_heads,
                has_prototype=False,
                n_patches=n_patches,
                output_path=os.path.join(attn_dir, "baseline_attention.png"),
                title="Baseline Decoder Attention Heatmaps"
            )
            plot_attention_heatmaps(
                all_attns=prototype_attns,
                n_heads=n_heads,
                has_prototype=True,
                n_patches=n_patches,
                output_path=os.path.join(attn_dir, "high_mi_prototype_attention.png"),
                title="High-MI Prototype Decoder Attention Heatmaps"
            )
            if low_prototype_attns is not None:
                plot_attention_heatmaps(
                    all_attns=low_prototype_attns,
                    n_heads=n_heads,
                    has_prototype=True,
                    n_patches=n_patches,
                    output_path=os.path.join(attn_dir, "low_mi_prototype_attention.png"),
                    title="Low-MI Prototype Decoder Attention Heatmaps"
                )

    else:
        # ── 常规实验（不收集注意力） ───────────────────────────────────────────
        baseline_preds_np = None
        baseline_trues_np = None
        proto_preds_np = None
        low_proto_preds_np = None

        if not args.skip_baseline:
            ret = run_experiment(
                args, device,
                use_prototype=False, prototype_path=None,
                rank=rank, collect_attention=False,
                collect_preds=args.plot_samples
            )
            if args.plot_samples:
                baseline_metrics, baseline_preds_np, baseline_trues_np = ret
            else:
                baseline_metrics = ret
            results["baseline"] = baseline_metrics

        if not args.skip_prototype:
            ret = run_experiment(
                args, device,
                use_prototype=True, prototype_path=args.prototype_path,
                rank=rank, collect_attention=False,
                collect_preds=args.plot_samples
            )
            if args.plot_samples:
                prototype_metrics, proto_preds_np, _ = ret
            else:
                prototype_metrics = ret
            results["prototype"] = prototype_metrics

        if args.low_proto_path and not args.skip_low_prototype:
            ret = run_experiment(
                args, device,
                use_prototype=True, prototype_path=args.low_proto_path,
                rank=rank, collect_attention=False,
                collect_preds=args.plot_samples
            )
            if args.plot_samples:
                low_proto_metrics, low_proto_preds_np, _ = ret
            else:
                low_proto_metrics = ret
            results["low_prototype"] = low_proto_metrics

        # ── 绘制随机样本对比图 ───────────────────────────────────────────────
        if args.plot_samples and rank == 0 and baseline_preds_np is not None:
            plot_sample_prediction_comparison(
                baseline_preds=baseline_preds_np,
                baseline_trues=baseline_trues_np,
                proto_preds=proto_preds_np,
                low_proto_preds=low_proto_preds_np,
                proto_trues=None,
                n_show=args.n_show_samples,
                save_path=os.path.join(args.output_dir, "sample_prediction_comparison.png")
            )

    # ── 打印对比分析 ───────────────────────────────────────────────────────────
    if rank == 0 and len(results) >= 2:
        n_groups = len(results)
        col_names = ["Baseline", "High-MI", "Low-MI"]
        col_keys = ["baseline", "prototype", "low_prototype"]

        # 根据是否是微调模式，设置标题
        exp_type = "微调后" if args.finetune else "推理"
        print("\n" + "="*90)
        print(f"│    {exp_type}对比分析: Baseline vs High-MI Prototype vs Low-MI Prototype      │")
        print("="*90)

        # 动态构造表头
        header = f"{'指标':<10}"
        for i in range(n_groups):
            header += f" {col_names[i]:>15}"
        if n_groups >= 2:
            header += f" {'Δ High-MI':>13} {'Δ Low-MI':>13}"
        print(header)
        print("-"*90)

        improvement_dict = {}
        for key in ["MSE", "MAE", "RMSE"]:
            row = f"{key:<10}"
            vals = {}
            for i, k in enumerate(col_keys):
                v = results.get(k, {}).get(key, float('inf'))
                vals[k] = v
                row += f" {v:>15.6f}" if v != float('inf') else f" {'N/A':>15}"

            base_val = vals.get("baseline", float('inf'))
            if base_val != float('inf') and n_groups >= 2:
                hi_val = vals.get("prototype", float('inf'))
                lo_val = vals.get("low_prototype", float('inf'))
                if hi_val != float('inf'):
                    row += f" {hi_val - base_val:>+13.6f}"
                    improvement_dict.setdefault(key, {})["prototype"] = {
                        "delta": hi_val - base_val,
                        "pct": (hi_val - base_val) / base_val * 100
                    }
                if lo_val != float('inf'):
                    row += f" {lo_val - base_val:>+13.6f}"
                    improvement_dict.setdefault(key, {})["low_prototype"] = {
                        "delta": lo_val - base_val,
                        "pct": (lo_val - base_val) / base_val * 100
                    }
            print(row)

        print("="*90)

        # 根据是否是微调模式，设置标题
        exp_type = "微调后" if args.finetune else "推理"
        output_file = os.path.join(args.output_dir, f"comparison_results_{exp_type}.txt")

        # 保存结果
        with open(output_file, 'w') as f:
            f.write("="*90 + "\n")
            f.write(f"    {exp_type}对比实验: Baseline vs High-MI Proto vs Low-MI Proto\n")
            f.write("="*90 + "\n\n")
            f.write(header + "\n")

            for key in ["MSE", "MAE", "RMSE"]:
                for grp, grp_name in [("prototype", "High-MI"), ("low_prototype", "Low-MI")]:
                    d = improvement_dict.get(key, {}).get(grp, {})
                    delta = d.get("delta", 0)
                    pct = d.get("pct", 0)
                    arrow = "↓" if delta < 0 else "↑"
                    f.write(f"  {key} {grp_name}: Δ={delta:+.6f} ({pct:+.2f}%) {arrow}\n")

            f.write("="*90 + "\n")
            mse_hi = improvement_dict.get("MSE", {}).get("prototype", {}).get("pct", 0)
            mse_lo = improvement_dict.get("MSE", {}).get("low_prototype", {}).get("pct", 0)
            f.write(f"\n结论:\n")
            if mse_hi < mse_lo:
                f.write(f"  High-MI Prototype 效果更好（MSE Δ={mse_hi:+.2f}%），Low-MI 次之（Δ={mse_lo:+.2f}%）\n")
            else:
                f.write(f"  Low-MI Prototype 效果更好（MSE Δ={mse_lo:+.2f}%），High-MI 次之（Δ={mse_hi:+.2f}%）\n")

        print(f"\n详细结果已保存到: {output_file}")

    if args.use_multi_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
