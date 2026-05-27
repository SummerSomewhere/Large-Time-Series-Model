#!/usr/bin/env python3
"""
Sundial MI-Guided Knowledge Distillation Comparison Experiment

对齐 timer_mi_distillation_comparison.py 的结构和风格。

对比三种训练方式：
  1. 从零训练学生模型 (StudentTransformer)
  2. 普通知识蒸馏 (均匀权重)
  3. MI 引导知识蒸馏 (按层 MI 加权)

教师: Sundial (thuml/sundial-base-128m, 冻结 backbone)
学生: 轻量 StudentTransformer (从头训练或蒸馏自教师)

Usage:
    python experiments/sundial_mi_distillation_comparison.py \
        --mi_dir ./results/sundial_mi_ksg_pca/Sundial_MI_xxx \
        --data_path ./datasets/ETTh1.csv \
        --seq_len 512 --pred_len 96 --label_len 48 --patch_len 16 \
        --ckpt_path checkpoints/sundial-base-128m.pt \
        --output_dir ./results/sundial_mi_distillation/ \
        --model_id etth1 --gpu 0

Dependencies:
    pip install numpy torch matplotlib tqdm scikit-learn
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_loader_benchmark import CIDatasetBenchmark


# =============================================================================
# 1. Config & Sundial Model Utilities
# =============================================================================

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_sundial_model(ckpt_path: str, seq_len: int, pred_len: int, label_len: int,
                       enc_in: int, d_model: int, n_layers: int, n_heads: int,
                       d_ff: int, dropout: float):
    """构建 Sundial 模型，对齐 Timer 的 build_timer_model 风格."""
    config = Config(
        task_name='zero_shot_finetune',
        pred_len=pred_len,
        num_samples=20,
        enc_in=enc_in,
        # model hyperparameters (passed to base_model)
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        seq_len=seq_len,
        label_len=label_len,
    )
    from models.Sundial import Model as SundialModel
    model = SundialModel(config)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  Loading Sundial from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# =============================================================================
# 2. Student Transformer (与 timer_mi_distillation_comparison.py 完全一致)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Projector(nn.Module):
    """Maps teacher hidden dimension to student hidden dimension per layer."""
    def __init__(self, layer_mapping, D_t, D_s):
        super().__init__()
        self.projs = nn.ModuleDict()
        for s_idx in layer_mapping:
            if D_t != D_s:
                self.projs[str(s_idx)] = nn.Linear(D_t, D_s)

    def forward(self, t_h, s_idx):
        if str(s_idx) in self.projs:
            return self.projs[str(s_idx)](t_h)
        return t_h


class StudentTransformer(nn.Module):
    """
    Lightweight Transformer student model.

    Processes time series with patch embedding + positional encoding + stacked
    TransformerEncoder layers, then projects to prediction horizon.
    与 timer_mi_distillation_comparison.py 中的 StudentTransformer 完全一致。
    """
    def __init__(self, seq_len: int = 512, patch_len: int = 16, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, d_ff: int = 512,
                 dropout: float = 0.1, pred_len: int = 96, n_channels: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_patches = seq_len // patch_len
        self.pred_len = pred_len
        self.n_channels = n_channels

        self.patch_embedding = nn.Linear(patch_len * n_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.n_patches, pred_len * n_channels)

    def forward(self, x, return_hidden: bool = False):
        B, L, C = x.shape
        n_patches = L // self.patch_len

        x_patched = x.reshape(B, n_patches, self.patch_len * C)
        x_flat = x_patched.reshape(B * n_patches, self.patch_len * C)
        h = self.patch_embedding(x_flat)
        h = h.reshape(B, n_patches, self.d_model)
        h = self.pos_encoding(h)

        hidden_states = [h]
        for layer in self.encoder.layers:
            h = layer(h)
            hidden_states.append(h)

        output = h.reshape(B, -1)
        forecast = self.head(output)
        forecast = forecast.reshape(B, self.pred_len, C)

        if return_hidden:
            return forecast, hidden_states
        return forecast


# =============================================================================
# 3. Teacher Hidden-State Extraction (Sundial)
# =============================================================================

def extract_sundial_hidden_states(model, batch_x, device):
    """
    Extract per-layer hidden states from Sundial's base_model.

    Returns a list of [B, N, D] tensors, one per encoder layer.
    Sundial uses bidirectional attention (no causal mask needed).
    Each channel is processed independently, then averaged across channels.

    Args:
        model: Sundial Model instance
        batch_x: [B, seq_len, n_channels] input time series

    Returns:
        list of [B, n_patches, D] tensors, one per layer
    """
    core = _unwrap(model)
    base = core.base_model

    seq_x = batch_x.float().to(device)
    B, seq_len, n_channels = seq_x.shape

    enc_means = seq_x.mean(dim=1, keepdim=True).detach()
    enc_stds = seq_x.std(dim=1, keepdim=False, unbiased=False) + 1e-5
    enc_stds = torch.clamp_min(enc_stds, 1e-2)
    x_norm = (seq_x - enc_means) / enc_stds.unsqueeze(1)

    all_layer_hiddens = []

    for ch in range(n_channels):
        ch_input = x_norm[..., ch:ch+1]
        with torch.no_grad():
            out = base(ch_input, output_hidden_states=True)

        if hasattr(out, 'hidden_states'):
            hs_list = out.hidden_states
        elif isinstance(out, tuple) and len(out) >= 2:
            hs_list = out[1]
        else:
            hs_list = list(out)

        if ch == 0:
            num_layers = len(hs_list)
            n_patches = hs_list[0].shape[1]
            D = hs_list[0].shape[2]
            hidden_per_channel = [
                torch.zeros(B, num_layers, n_patches, D, device=device)
                for _ in range(n_channels)
            ]

        for layer_idx, layer_hs in enumerate(hs_list):
            hidden_per_channel[ch][:, layer_idx] = layer_hs

    for layer_idx in range(num_layers):
        stacked = torch.stack([hidden_per_channel[ch][:, layer_idx] for ch in range(n_channels)], dim=0)
        all_layer_hiddens.append(stacked.mean(dim=0))

    return all_layer_hiddens


# =============================================================================
# 4. Distillation Loss (对齐 timer_mi_distillation_comparison.py)
# =============================================================================

def compute_distillation_loss(student_output, target, teacher_hiddens, student_hiddens,
                               mi_weights, layer_mapping, projector, alpha=1.0,
                               layer_weights=None):
    """
    Compute combined task loss + feature distillation loss.
    Layer internal patches contribute equally (simple mean).
    Layer-level contributions are weighted by layer_weights.

    与 timer_mi_distillation_comparison.py 中的 compute_distillation_loss 完全对齐。

    Args:
        student_output: [B, pred_len, C] forecast from student
        target:         [B, pred_len, C] ground truth
        teacher_hiddens: list of [B, N, D_t] tensors from teacher
        student_hiddens: list of [B, N, D_s] tensors from student
        mi_weights:      [n_teacher_layers, n_patches] per-patch MI scores.
                         Used only to build layer_weights; layer internal is uniform average.
        layer_mapping:   dict {student_idx: teacher_idx}
        projector:       Projector module
        alpha:          weight of feature distillation loss
        layer_weights:  dict {teacher_idx: weight}. If None, uniform (1 / n_layers).

    Returns:
        total_loss, loss_dict
    """
    task_loss = F.mse_loss(student_output, target)

    if teacher_hiddens is None or student_hiddens is None:
        return task_loss, {'task': task_loss.item(), 'feature': 0.0, 'total': task_loss.item()}

    unique_t = sorted({t for t in layer_mapping.values()})
    n_layers = len(unique_t)

    feature_loss = 0.0

    for s_idx, t_idx in layer_mapping.items():
        if t_idx >= len(teacher_hiddens) or s_idx >= len(student_hiddens):
            continue

        t_h = teacher_hiddens[t_idx]
        s_h = student_hiddens[s_idx]

        B_s, N_s, D_s = s_h.shape
        B_t, N_t, D_t = t_h.shape

        if N_t != N_s:
            continue

        t_h = projector(t_h, s_idx)

        per_patch_mse = F.mse_loss(t_h, s_h, reduction='none').mean(dim=-1)
        layer_loss = per_patch_mse.mean()

        if layer_weights is not None:
            lw = layer_weights.get(t_idx, 1.0 / n_layers)
        else:
            lw = 1.0 / n_layers
        feature_loss += layer_loss * lw

    total_loss = task_loss + alpha * feature_loss
    return total_loss, {
        'task': task_loss.item(),
        'feature': feature_loss.item(),
        'total': total_loss.item()
    }


# =============================================================================
# 5. Evaluation Utilities (与 Timer 版本完全一致)
# =============================================================================

def compute_metrics(trues, preds):
    mse = float(np.nanmean((trues - preds) ** 2))
    mae = float(np.nanmean(np.abs(trues - preds)))
    return {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse)}


def align_pred_shape(pred, true):
    if pred.shape != true.shape and pred.shape[-1] == true.shape[-1] and pred.shape[1] == true.shape[-1]:
        return np.transpose(pred, (0, 2, 1))
    return pred


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'  Val loss {self.val_loss_min:.6f} -> {val_loss:.6f}, saving model')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# =============================================================================
# 6. Training Loop (对齐 timer_mi_distillation_comparison.py)
# =============================================================================

def train_student(student_model, teacher_model, train_loader, val_loader, test_loader,
                  device, args, mi_weights, layer_mapping, projector,
                  mode='scratch', model_save_path="best_model.pth",
                  layer_weights=None):
    """
    Train student model in one of three modes:
      - 'scratch':       MSE task loss only
      - 'uniform':       MSE + uniform feature distillation (all patches)
      - 'mi_weighted':   MSE + MI-guided feature distillation

    与 timer_mi_distillation_comparison.py 中的 train_student 完全对齐。
    """
    params = list(student_model.parameters())
    if projector is not None:
        params += list(projector.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    history = {'train_loss': [], 'train_task': [], 'train_feat': [],
               'val_mse': [], 'val_mae': []}

    for epoch in range(args.epochs):
        student_model.train()
        if projector is not None:
            projector.train()

        train_loss = 0.0
        train_task = 0.0
        train_feat = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [{mode}]")

        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            else:
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else batch

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if mode == 'scratch':
                output = student_model(batch_x)
                target_y = batch_y[:, :output.shape[1], :]
                loss, loss_dict = F.mse_loss(output, target_y), \
                                  {'task': F.mse_loss(output, target_y).item(),
                                   'feature': 0.0, 'total': F.mse_loss(output, target_y).item()}
            else:
                with torch.no_grad():
                    teacher_hiddens = extract_sundial_hidden_states(
                        teacher_model, batch_x, device)

                output, student_hiddens = student_model(batch_x, return_hidden=True)
                target_y = batch_y[:, :output.shape[1], :]

                loss, loss_dict = compute_distillation_loss(
                    output, target_y,
                    teacher_hiddens=teacher_hiddens,
                    student_hiddens=student_hiddens,
                    mi_weights=mi_weights if mode == 'mi_weighted' else None,
                    layer_mapping=layer_mapping,
                    projector=projector,
                    alpha=args.alpha,
                    layer_weights=layer_weights
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            train_loss += loss_dict['total']
            train_task += loss_dict['task']
            train_feat += loss_dict['feature']
            n_batches += 1
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'task': f"{loss_dict['task']:.4f}",
                'feat': f"{loss_dict['feature']:.4f}"
            })

        train_loss /= n_batches
        train_task /= n_batches
        train_feat /= n_batches
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_task'].append(train_task)
        history['train_feat'].append(train_feat)

        student_model.eval()
        if projector is not None:
            projector.eval()

        val_mse, val_mae = 0.0, 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                else:
                    batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    batch_y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else batch

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = student_model(batch_x)
                target_y = batch_y[:, :output.shape[1], :]
                val_mse += F.mse_loss(output, target_y).item()
                val_mae += F.l1_loss(output, target_y).item()
                n_val += 1

        val_mse /= max(n_val, 1)
        val_mae /= max(n_val, 1)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f} "
              f"(task={train_task:.4f}, feat={train_feat:.4f}) | "
              f"val_mse={val_mse:.4f}, val_mae={val_mae:.4f}")

        early_stopping(val_mse, student_model, model_save_path)
        if early_stopping.early_stop:
            print("  Early stopping triggered.")
            break

    student_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    student_model.eval()

    trues_list, preds_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            else:
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else batch

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = student_model(batch_x)
            target_y = batch_y[:, :output.shape[1], :]
            preds_list.append(output.cpu().numpy())
            trues_list.append(target_y.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    preds = align_pred_shape(preds, trues)
    metrics = compute_metrics(trues, preds)

    print(f"  Test: MSE={metrics['MSE']:.6f}, MAE={metrics['MAE']:.6f}")
    return metrics, history


# =============================================================================
# 7. Plotting (与 timer_mi_distillation_comparison.py 完全一致)
# =============================================================================

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
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def plot_comparison(results, output_dir, dataset_name=""):
    """Generate comparison figures for the three training strategies."""
    _nature_rc()

    methods = ['scratch', 'uniform', 'mi_weighted']
    method_labels = {
        'scratch': 'Train from Scratch',
        'uniform': 'Uniform KD',
        'mi_weighted': 'MI-Guided KD'
    }
    colors = {
        'scratch': '#999999',
        'uniform': '#56B4E9',
        'mi_weighted': '#E69F00',
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))

    for ax, metric, ylabel in zip(
            axes,
            ['MSE', 'MAE', 'train_loss'],
            ['Test MSE', 'Test MAE', 'Train Loss']
    ):
        for method in methods:
            hist = results[method].get('history', {})
            vals = hist.get(metric, [])
            if vals:
                ax.plot(vals, 'o-', color=colors[method],
                        label=method_labels[method], linewidth=1.5, markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7)

    fig.savefig(os.path.join(output_dir, "fig_training_curves.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig_training_curves.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig_training_curves.{{pdf,png}}")

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    methods_run = [m for m in methods if m in results]
    mse_vals = [results[m]['metrics']['MSE'] for m in methods_run]
    bar_width = 0.35
    x = np.arange(len(methods_run))

    ax.bar(x - bar_width/2, mse_vals, bar_width,
           label='MSE', color=[colors[m] for m in methods_run], alpha=0.85,
           edgecolor='white', linewidth=0.5)
    ax.set_ylabel('MSE')
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels[m] for m in methods_run], fontsize=8)
    ax.set_title('Test MSE Comparison')

    for i, val in enumerate(mse_vals):
        ax.text(i - bar_width/2, val + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    fig.savefig(os.path.join(output_dir, "fig_mse_comparison.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig_mse_comparison.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig_mse_comparison.{{pdf,png}}")

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    mse_baseline = results.get('scratch', {}).get('metrics', {}).get('MSE', None)
    if mse_baseline is not None:
        improvements = []
        labels = []
        bar_colors = []
        for m, label, color in [('uniform', 'Uniform KD', colors['uniform']),
                                 ('mi_weighted', 'MI-Guided KD', colors['mi_weighted'])]:
            if m in results:
                improvements.append((mse_baseline - results[m]['metrics']['MSE']) / mse_baseline * 100)
                labels.append(label)
                bar_colors.append(color)
        if improvements:
            bars = ax.bar(labels, improvements, color=bar_colors, alpha=0.85,
                          edgecolor='white', linewidth=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)
            ax.set_ylabel('MSE Improvement vs Scratch (%)')
            ax.set_title('Relative Improvement')
            for bar, val in zip(bars, improvements):
                va = 'bottom' if val >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.2 if val >= 0 else -0.2),
                        f'{val:+.2f}%', ha='center', va=va, fontsize=8)

    fig.savefig(os.path.join(output_dir, "fig_improvement.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig_improvement.png"), dpi=300)
    plt.close()
    print(f"  [Saved] fig_improvement.{{pdf,png}}")


# =============================================================================
# 8. Main (对齐 timer_mi_distillation_comparison.py 的 main 函数结构)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sundial MI-Guided Knowledge Distillation Comparison")
    # Paths
    parser.add_argument("--mi_dir", type=str,
                       default="./results/sundial_mi_ksg_pca/Sundial_MI_20260520_120000",
                       help="Directory containing MI matrix")
    parser.add_argument("--data_path", type=str, default="./datasets/ETTh1.csv")
    parser.add_argument("--data_type", type=str, default="ETTh1",
                       choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])
    parser.add_argument("--output_dir", type=str,
                       default="./results/sundial_mi_distillation/")
    # Model (teacher)
    parser.add_argument("--ckpt_path", type=str,
                       default="checkpoints/sundial-base-128m.pt",
                       help="Sundial checkpoint path")
    parser.add_argument("--d_model_t", type=int, default=768,
                       help="Teacher d_model (Sundial)")
    parser.add_argument("--d_ff_t", type=int, default=3072)
    parser.add_argument("--e_layers", type=int, default=12,
                       help="Teacher number of layers (Sundial base: 12)")
    parser.add_argument("--n_heads", type=int, default=12)
    # Model (student)
    parser.add_argument("--s_d_model", type=int, default=256,
                       help="Student d_model")
    parser.add_argument("--s_d_ff", type=int, default=512,
                       help="Student feed-forward dim")
    parser.add_argument("--s_n_layers", type=int, default=4,
                       help="Student number of layers")
    parser.add_argument("--s_n_heads", type=int, default=4,
                       help="Student number of heads")
    parser.add_argument("--s_dropout", type=float, default=0.1)
    # Data
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--patch_len", type=int, default=16,
                       help="Student patch length")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--freq", type=str, default="h")
    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Weight of feature distillation loss")
    parser.add_argument("--experiments", type=str, nargs='+',
                       default=['scratch', 'uniform', 'mi_weighted'],
                       choices=['scratch', 'uniform', 'mi_weighted'],
                       help="Which experiments to run (default: all three)")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="etth1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.output_dir, f"{args.model_id}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    print("=" * 70)
    print("Sundial MI-Guided Knowledge Distillation Comparison")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<24}: {v}")
    print(f"  exp_dir                : {exp_dir}")
    print(f"  device                 : {device}")
    print("=" * 70)

    # ── Phase 1: Load MI weights ─────────────────────────────────────────────
    print("\n>>> [1/6] Loading MI weights...")
    mi_weights = None

    for fname in ["mi_hy_matrix.npy", "mi_matrix.npy", "mi_matrix_future.npy"]:
        p = os.path.join(args.mi_dir, fname)
        if os.path.exists(p):
            mi_matrix = np.load(p)
            print(f"  Loaded MI matrix from {p}: shape={mi_matrix.shape}")
            n_t_layers, n_patches = mi_matrix.shape
            mi_weights = torch.from_numpy(mi_matrix).float()
            print(f"  In-layer patch weights: shape={mi_weights.shape}")
            break

    if mi_weights is None:
        json_path = os.path.join(args.mi_dir, f"global_mi_peaks_{args.model_id}.json")
        if not os.path.exists(json_path):
            candidates = [f for f in os.listdir(args.mi_dir)
                          if f.startswith("global_mi_peaks_") and f.endswith(".json")]
            if candidates:
                json_path = os.path.join(args.mi_dir, candidates[0])
        if os.path.exists(json_path):
            with open(json_path) as f:
                mi_data = json.load(f)
            n_t_layers = mi_data["num_layers"]
            n_patches = mi_data["N"]
            mi_vals = np.zeros((n_t_layers, n_patches))
            for li in range(n_t_layers):
                mi_vals[li] = mi_data["layers"][str(li)]["hsic_curve"]
            mi_weights = torch.from_numpy(mi_vals).float()
            print(f"  Loaded MI from JSON: {n_t_layers} layers x {n_patches} patches")
            print(f"  In-layer patch weights: shape={mi_weights.shape}")
        else:
            raise FileNotFoundError(
                f"MI file not found in {args.mi_dir}\n"
                f"Expected: mi_hy_matrix.npy / mi_matrix.npy / global_mi_peaks_*.json"
            )

    n_t_layers = mi_weights.shape[0]
    n_s_patches = args.seq_len // args.patch_len
    print(f"  Teacher layers: {n_t_layers}, patches: {n_s_patches}")

    # ── Phase 2: Load dataset ─────────────────────────────────────────────────
    print("\n>>> [2/6] Loading dataset...")
    train_dataset = CIDatasetBenchmark(
        root_path=args.data_path, flag='train',
        input_len=args.seq_len, pred_len=args.pred_len,
        data_type=args.data_type, scale=True, timeenc=1, freq=args.freq,
    )
    val_dataset = CIDatasetBenchmark(
        root_path=args.data_path, flag='val',
        input_len=args.seq_len, pred_len=args.pred_len,
        data_type=args.data_type, scale=True, timeenc=1, freq=args.freq,
    )
    test_dataset = CIDatasetBenchmark(
        root_path=args.data_path, flag='test',
        input_len=args.seq_len, pred_len=args.pred_len,
        data_type=args.data_type, scale=True, timeenc=1, freq=args.freq,
    )

    n_vars = train_dataset.n_var
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Variables: {n_vars}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Phase 3: Load teacher model (Sundial, frozen) ─────────────────────────
    print("\n>>> [3/6] Loading Sundial teacher (frozen)...")
    teacher_model = build_sundial_model(
        ckpt_path=args.ckpt_path,
        seq_len=args.seq_len, pred_len=args.pred_len, label_len=args.label_len,
        enc_in=n_vars, d_model=args.d_model_t,
        n_layers=args.e_layers, n_heads=args.n_heads,
        d_ff=args.d_ff_t, dropout=0.1,
    ).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    core = _unwrap(teacher_model)
    teacher_d_model = core.d_model
    teacher_n_layers = core.n_layers
    print(f"  Teacher: d_model={teacher_d_model}, layers={teacher_n_layers}")

    # Build layer mapping: student layer i -> teacher layer (evenly distributed)
    layer_mapping = {}
    for s_idx in range(args.s_n_layers):
        t_idx = min((s_idx + 1) * (teacher_n_layers // args.s_n_layers),
                    teacher_n_layers - 1)
        layer_mapping[s_idx] = t_idx
    print(f"  Layer mapping: {layer_mapping}")

    # Build projector
    projector = Projector(layer_mapping, teacher_d_model, args.s_d_model).to(device)

    results = {}

    # ── Phase 4: Experiment 1 — Scratch ──────────────────────────────────────
    if 'scratch' in args.experiments:
        print("\n" + "=" * 70)
        print("Experiment 1/3: Train Student from Scratch")
        print("=" * 70)

        student_scratch = StudentTransformer(
            seq_len=args.seq_len, patch_len=args.patch_len,
            d_model=args.s_d_model, n_layers=args.s_n_layers,
            n_heads=args.s_n_heads, d_ff=args.s_d_ff,
            dropout=args.s_dropout, pred_len=args.pred_len,
            n_channels=1
        ).to(device)
        print(f"  Student: d_model={args.s_d_model}, n_layers={args.s_n_layers}, "
              f"n_heads={args.s_n_heads}, d_ff={args.s_d_ff}")

        metrics_s, hist_s = train_student(
            student_scratch, teacher_model, train_loader, val_loader, test_loader,
            device, args, mi_weights=None, layer_mapping=None, projector=None,
            mode='scratch',
            model_save_path=os.path.join(exp_dir, "best_scratch.pth")
        )
        results['scratch'] = {'metrics': metrics_s, 'history': hist_s}
    else:
        print("\n[skip] Train from Scratch (not in --experiments)")
        metrics_s = None

    # ── Phase 5: Experiment 2 — Uniform KD ───────────────────────────────────
    if 'uniform' in args.experiments:
        print("\n" + "=" * 70)
        print("Experiment 2/3: Uniform Feature Distillation")
        print("=" * 70)

        student_uniform = StudentTransformer(
            seq_len=args.seq_len, patch_len=args.patch_len,
            d_model=args.s_d_model, n_layers=args.s_n_layers,
            n_heads=args.s_n_heads, d_ff=args.s_d_ff,
            dropout=args.s_dropout, pred_len=args.pred_len,
            n_channels=1
        ).to(device)

        projector_uniform = Projector(layer_mapping, teacher_d_model, args.s_d_model).to(device)

        metrics_u, hist_u = train_student(
            student_uniform, teacher_model, train_loader, val_loader, test_loader,
            device, args, mi_weights=None, layer_mapping=layer_mapping,
            projector=projector_uniform,
            mode='uniform',
            model_save_path=os.path.join(exp_dir, "best_uniform.pth"),
            layer_weights=None
        )
        results['uniform'] = {'metrics': metrics_u, 'history': hist_u}
    else:
        print("\n[skip] Uniform Feature Distillation (not in --experiments)")
        metrics_u = None

    # ── Phase 6: Experiment 3 — MI-Weighted KD ───────────────────────────────
    if 'mi_weighted' in args.experiments:
        print("\n" + "=" * 70)
        print("Experiment 3/3: MI-Guided Feature Distillation")
        print("=" * 70)

        student_mi = StudentTransformer(
            seq_len=args.seq_len, patch_len=args.patch_len,
            d_model=args.s_d_model, n_layers=args.s_n_layers,
            n_heads=args.s_n_heads, d_ff=args.s_d_ff,
            dropout=args.s_dropout, pred_len=args.pred_len,
            n_channels=1
        ).to(device)

        projector_mi = Projector(layer_mapping, teacher_d_model, args.s_d_model).to(device)

        unique_t = sorted({t for t in layer_mapping.values()})
        if mi_weights is not None:
            layer_weights_mi = {}
            for t_idx in unique_t:
                if t_idx < mi_weights.shape[0]:
                    layer_weights_mi[t_idx] = float(mi_weights[t_idx].sum().item())
            print(f"  MI layer weights (sum of patch MI): {layer_weights_mi}")
        else:
            layer_weights_mi = None

        metrics_m, hist_m = train_student(
            student_mi, teacher_model, train_loader, val_loader, test_loader,
            device, args, mi_weights=mi_weights, layer_mapping=layer_mapping,
            projector=projector_mi,
            mode='mi_weighted',
            model_save_path=os.path.join(exp_dir, "best_mi.pth"),
            layer_weights=layer_weights_mi
        )
        results['mi_weighted'] = {'metrics': metrics_m, 'history': hist_m}
    else:
        print("\n[skip] MI-Guided Feature Distillation (not in --experiments)")
        metrics_m = None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Comparison Summary (Test Set)")
    print("=" * 70)
    print(f"{'Method':<28} | {'MSE':<12} | {'MAE':<12} | {'Improvement vs Scratch'}")
    print("-" * 72)

    baseline_mse = metrics_s['MSE'] if metrics_s is not None else None
    for method, label in [
        ('scratch', 'Train from Scratch'),
        ('uniform', 'Uniform KD'),
        ('mi_weighted', 'MI-Guided KD'),
    ]:
        if method not in results:
            print(f"  {label:<26} | {'(skipped)':<12}")
            continue
        m = results[method]['metrics']
        if baseline_mse is not None and method != 'scratch':
            imp = (baseline_mse - m['MSE']) / baseline_mse * 100
            imp_str = f"{imp:+.2f}%"
        else:
            imp_str = "baseline" if method == 'scratch' else "N/A"
        print(f"  {label:<26} | {m['MSE']:<12.6f} | {m['MAE']:<12.6f} | {imp_str}")

    print("=" * 70)

    results['comparison'] = {'experiments_run': list(results.keys())}
    if baseline_mse is not None:
        results['comparison']['baseline_mse'] = float(baseline_mse)
        if 'uniform' in results and metrics_u is not None:
            results['comparison']['improvement_uniform'] = float(
                (baseline_mse - metrics_u['MSE']) / baseline_mse * 100)
        if 'mi_weighted' in results and metrics_m is not None:
            results['comparison']['improvement_mi'] = float(
                (baseline_mse - metrics_m['MSE']) / baseline_mse * 100)
            if 'uniform' in results and metrics_u is not None:
                results['comparison']['improvement_mi_vs_uniform'] = float(
                    (metrics_u['MSE'] - metrics_m['MSE']) / metrics_u['MSE'] * 100)
        results['comparison']['layer_mapping'] = {str(k): v for k, v in layer_mapping.items()}
        if mi_weights is not None:
            results['comparison']['mi_weights'] = mi_weights.tolist()

    results_path = os.path.join(exp_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    try:
        plot_comparison(results, exp_dir, args.model_id)
    except Exception as e:
        print(f"WARNING: Plotting failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nExperiment complete. Output: {exp_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
