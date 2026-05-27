#!/usr/bin/env python3
"""
Timer MI Token-Level Masked 知识蒸馏对比实验

对比五种训练方式：
  1. 从零训练学生模型（StudentTransformer）
  2. 普通知识蒸馏（均匀权重，per-token）
  3. 层级别 MI 加权蒸馏（按层归一化 MI 加权，per-token）
  4. Token 级别高 MI Mask 蒸馏（只蒸馏 high-MI tokens）
  5. Token 级别高 MI Mask + 层级别 MI 加权蒸馏

关键改进（相对于 timer_mi_distillation_comparison.py）：
  - 蒸馏损失在 token/patch 粒度上计算，而非整个 hidden state 均一化
  - uniform 模式：所有 token 的蒸馏损失均匀平均
  - layer_mi 模式：层间 MI 加权 + 所有 token 均匀平均
  - token_mask 模式：层间均匀加权 + 只蒸馏 high-MI tokens（masked）
  - token_mask_layer_mi 模式：层间 MI 加权 + 只蒸馏 high-MI tokens（双重加权）

教师：Pretrained Timer（thuml/Timer，冻结backbone）
学生：轻量 StudentTransformer（从头训练，或蒸馏自教师）

Usage:
    python experiments/timer_token_mi_distillation_comparison.py \
        --mi_dir ./timer_mi_ksg_pca/Timer_MI_20260514_103927 \
        --root_path ./datasets/ --data ETTh1 --data_path ETTh1.csv \
        --seq_len 672 --pred_len 96 --label_len 48 \
        --batch_size 32 --epochs 15 --gpu 0 \
        --out_dir ./results/timer_token_mi_distillation/ --model_id etth1
"""

import argparse
import json
import os
import sys
from datetime import datetime

os.environ["TORCHDYNAMO_AUTOJECT"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


if not hasattr(torch.utils._pytree, "register_pytree_node"):
    def _noop_register_pytree_node(*args, **kwargs):
        pass
    torch.utils._pytree.register_pytree_node = _noop_register_pytree_node

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_provider.data_factory import data_provider


# ==========================================
# 1. 辅助工具与网络模块
# ==========================================

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


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
    """将教师隐层维度映射到学生维度"""
    def __init__(self, layer_mapping, D_t, D_s):
        super().__init__()
        self.projs = nn.ModuleDict({
            str(s_idx): nn.Linear(D_t, D_s) for s_idx in layer_mapping.keys() if D_t != D_s
        })

    def forward(self, t_h, s_idx):
        if str(s_idx) in self.projs:
            return self.projs[str(s_idx)](t_h)
        return t_h


class StudentTransformer(nn.Module):
    """
    轻量 Transformer 学生模型。
    处理时序序列：patch embedding + 位置编码 + TransformerEncoder 层堆叠 + 预测头。
    """
    def __init__(self, seq_len: int = 96, patch_len: int = 8, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, d_ff: int = 512,
                 dropout: float = 0.1, pred_len: int = 96, n_channels: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_patches = seq_len // patch_len
        self.pred_len = pred_len
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.n_patches, pred_len)

    def forward(self, x, return_hidden: bool = False):
        B, L, C = x.shape
        n_patches = L // self.patch_len

        x_patched = x.reshape(B, n_patches, self.patch_len)
        x_flat = x_patched.reshape(B * n_patches, self.patch_len)

        h = self.patch_embedding(x_flat)
        h = h.reshape(B, n_patches, self.d_model)
        h = self.pos_encoding(h)

        hidden_states = [h]
        for layer in self.encoder.layers:
            h = layer(h)
            hidden_states.append(h)

        output = h.reshape(B, -1)
        forecast = self.head(output)
        forecast = forecast.reshape(B, self.pred_len, 1)

        if return_hidden:
            return forecast, hidden_states
        return forecast


# ==========================================
# 2. 教师模型逻辑（Timer）
# ==========================================

def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def load_teacher_model(ckpt_path: str, patch_len: int, stride: int,
                       d_model: int, d_ff: int, e_layers: int,
                       n_heads: int, seq_len: int, pred_len: int, device):
    """加载 Timer 教师模型（冻结，全部参数不可学习）"""
    class Config:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    config = Config(
        task_name='forecast', ckpt_path=ckpt_path,
        patch_len=patch_len, stride=stride,
        d_model=d_model, d_ff=d_ff, e_layers=e_layers,
        n_heads=n_heads, dropout=0.1,
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
    from models.Timer import Model as TimerModel
    model = TimerModel(config).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_timer_hidden_states(model, batch_x, device):
    """
    提取 Timer 教师模型的隐层状态（per-channel forward，与 Timer 原生行为对齐）。
    返回 list of [B, N, D] tensors（每个 decoder 层一个）。
    """
    core = _unwrap(model)

    seq_x = batch_x.float().to(device)
    B, seq_len, num_channels = seq_x.shape

    means = seq_x.mean(dim=1, keepdim=True).detach()
    stdev = torch.sqrt(
        torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    stdev = torch.clamp_min(stdev, 1e-5)
    x_norm = (seq_x - means) / stdev

    x2 = x_norm.permute(0, 2, 1)
    dec_in, n_vars = core.enc_embedding(x2)

    BM, N, D = dec_in.shape
    B, M = B * n_vars, BM // B

    dec_in = dec_in.reshape(B, M, N, D)
    dec_in = dec_in.mean(dim=1)

    from utils.masking import TriangularCausalMask
    causal_mask = TriangularCausalMask(B, N, device=device)

    hidden_states = [dec_in]
    for layer_mod in core.decoder.attn_layers:
        dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)
        dec_in = dec_in.reshape(B, M, N, D).mean(dim=1)
        hidden_states.append(dec_in)

    if core.decoder.norm is not None:
        dec_in = core.decoder.norm(dec_in)
        hidden_states.append(dec_in)

    return hidden_states


# ==========================================
# 3. Token 级别蒸馏损失（核心差异）
# ==========================================

def build_mi_token_mask(mi_json_path: str, model_id: str) -> dict:
    """
    从 MI JSON 构建每层的高 MI token 掩码。

    Returns:
        {
            'n_layers': 8,
            'n_patches': 7,
            'layer_token_mask': dict of {t_idx: [bool, ...]}  # True = high-MI token
            'high_token_ratio': float   # 平均每层高 MI token 比例
            'all_high_patches': [int, ...]  # 所有层一致的高 MI patches
        }
    """
    candidates = [f for f in os.listdir(mi_json_path)
                  if f.startswith("global_mi_peaks_") and f.endswith(".json")]
    if model_id:
        json_path = os.path.join(mi_json_path, f"global_mi_peaks_{model_id}.json")
        if not os.path.exists(json_path):
            if candidates:
                json_path = os.path.join(mi_json_path, candidates[0])
    else:
        json_path = os.path.join(mi_json_path, candidates[0]) if candidates else None

    if not json_path or not os.path.exists(json_path):
        raise FileNotFoundError(f"MI JSON not found in {mi_json_path}")

    with open(json_path) as f:
        mi_data = json.load(f)

    n_layers = mi_data["num_layers"]
    n_patches = mi_data["N"]

    layer_token_mask = {}
    total_high = 0
    all_consistent_high = None

    for li in range(n_layers):
        layer_info = mi_data["layers"][str(li)]
        high_patches = layer_info["high_mi_patches"]
        curve = layer_info["hsic_curve"]
        q3 = layer_info["q3_threshold"]

        mask = [curve[p] >= q3 for p in range(n_patches)]
        layer_token_mask[li] = mask
        total_high += sum(mask)

        if all_consistent_high is None:
            all_consistent_high = set(high_patches)
        else:
            all_consistent_high &= set(high_patches)

    high_ratio = total_high / (n_layers * n_patches)
    all_consistent_high = sorted(all_consistent_high)

    return {
        'n_layers': n_layers,
        'n_patches': n_patches,
        'layer_token_mask': layer_token_mask,
        'high_token_ratio': high_ratio,
        'all_high_patches': all_consistent_high,
        'json_path': json_path,
    }


def compute_distillation_loss_token_level(
    student_output, target,
    teacher_hiddens=None, student_hiddens=None,
    mi_layer_weights=None,
    layer_mapping=None, projector=None,
    token_mask_info=None,
    mode='uniform',
    alpha=1.0,
):
    """
    Token 级别的蒸馏损失计算。

    支持四种模式（对应实验 2-5）：
      - 'uniform':         层间均匀加权 + 所有 token 均匀平均（baseline）
      - 'layer_mi':        层间 MI 加权（每层一个标量）+ 所有 token 均匀平均
      - 'token_mask':      层间均匀加权 + 只蒸馏 high-MI tokens
      - 'token_mask_layer_mi': 层间 MI 加权 + 只蒸馏 high-MI tokens

    核心变化：per-token MSE（而非 per-layer 均一化）
      - teacher_hiddens[t_idx]: shape [B, N, D_t]  N = n_patches
      - student_hiddens[s_idx]: shape [B, N, D_s]
      - 对每个 token position 计算 MSE，然后按 mask 加权
    """
    task_loss = F.mse_loss(student_output, target)

    if teacher_hiddens is None or student_hiddens is None or projector is None:
        return task_loss, {'task': task_loss.item(), 'feature': 0.0}

    feature_loss = 0.0
    n_active = 0

    for s_idx, t_idx in layer_mapping.items():
        if t_idx >= len(teacher_hiddens) or s_idx >= len(student_hiddens):
            continue

        t_h = teacher_hiddens[t_idx]
        s_h = student_hiddens[s_idx]

        B_t, N_t, D_t = t_h.shape
        B_s, N_s, D_s = s_h.shape

        if N_t != N_s:
            continue

        t_h = projector(t_h, s_idx)

        per_token_mse = F.mse_loss(t_h, s_h, reduction='none').mean(dim=-1)

        if mode in ('token_mask', 'token_mask_layer_mi') and token_mask_info is not None:
            mask = token_mask_info['layer_token_mask'].get(t_idx, [True] * N_t)
            mask_t = torch.tensor(mask, dtype=torch.bool,
                                  device=s_h.device).unsqueeze(0)
            mask_t = mask_t.expand(B_s, -1)

            if mask_t.sum() < 1:
                continue

            layer_loss = per_token_mse[mask_t].mean()
        else:
            layer_loss = per_token_mse.mean()

        if mi_layer_weights is not None and mode in ('layer_mi', 'token_mask_layer_mi'):
            w = float(mi_layer_weights[t_idx]) if t_idx < len(mi_layer_weights) else 1.0
        else:
            w = 1.0 / len(layer_mapping)

        feature_loss += layer_loss * w
        n_active += 1

    if n_active > 0:
        feature_loss = feature_loss / n_active

    total_loss = task_loss + alpha * feature_loss
    return total_loss, {
        'task': task_loss.item(),
        'feature': feature_loss.item(),
        'total': total_loss.item()
    }


# ==========================================
# 4. 训练循环
# ==========================================

def train_model(student_model, teacher_model, train_loader, val_loader, test_loader, device,
                args, mode='scratch', mi_layer_weights=None, layer_mapping=None,
                token_mask_info=None, model_save_path="best_model.pth"):
    """
    训练学生模型。

    五种模式：
      - 'scratch':              仅 MSE 任务损失
      - 'uniform':              MSE + 均匀特征蒸馏（token-level）
      - 'layer_mi':             MSE + 层级别 MI 加权蒸馏（token-level）
      - 'token_mask':           MSE + token 高 MI mask 蒸馏（token-level）
      - 'token_mask_layer_mi':  MSE + token mask + 层 MI 加权蒸馏
    """
    projector = None
    params = list(student_model.parameters())

    if mode != 'scratch' and layer_mapping is not None:
        core = _unwrap(teacher_model)
        teacher_d_model = core.d_model
        projector = Projector(layer_mapping, teacher_d_model, args.s_d_model).to(device)
        params += list(projector.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, verbose=False)

    history = {
        'step': [], 'epoch': [],
        'train_loss': [], 'train_task': [], 'train_feat': [],
        'val_mse': [], 'val_mae': []
    }

    global_step = 0
    batches_per_epoch = len(train_loader)
    log_interval = max(1, batches_per_epoch // 5)

    for epoch in range(args.epochs):
        student_model.train()
        if projector is not None:
            projector.train()

        train_loss = 0.0
        train_task = 0.0
        train_feat = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs} [{mode}]", leave=False)

        for batch_x, batch_y, batch_x_mark, batch_y_mark in pbar:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if mode == 'scratch':
                output = student_model(batch_x)
                target_y = batch_y[:, :output.shape[1], :]
                loss = F.mse_loss(output, target_y)
                loss_dict = {'task': loss.item(), 'feature': 0.0, 'total': loss.item()}
            else:
                with torch.no_grad():
                    teacher_hiddens = get_timer_hidden_states(teacher_model, batch_x, device)
                output, student_hiddens = student_model(batch_x, return_hidden=True)
                target_y = batch_y[:, :output.shape[1], :]

                loss, loss_dict = compute_distillation_loss_token_level(
                    student_output=output, target=target_y,
                    teacher_hiddens=teacher_hiddens, student_hiddens=student_hiddens,
                    mi_layer_weights=mi_layer_weights,
                    layer_mapping=layer_mapping, projector=projector,
                    token_mask_info=token_mask_info,
                    mode=mode, alpha=args.alpha
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            global_step += 1
            train_loss += loss_dict['total']
            train_task += loss_dict['task']
            train_feat += loss_dict['feature']
            n_batches += 1

            if global_step % log_interval == 0:
                history['step'].append(global_step)
                history['epoch'].append(epoch)
                history['train_loss'].append(train_loss / max(n_batches, 1))
                history['train_task'].append(train_task / max(n_batches, 1))
                history['train_feat'].append(train_feat / max(n_batches, 1))

            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'step': global_step
            })

        train_loss /= max(n_batches, 1)
        train_task /= max(n_batches, 1)
        train_feat /= max(n_batches, 1)
        scheduler.step()

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_task'].append(train_task)
        history['train_feat'].append(train_feat)

        student_model.eval()
        if projector is not None:
            projector.eval()

        val_mse, val_mae = 0.0, 0.0
        n_val = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
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

        print(f"  Ep {epoch+1}: train={train_loss:.4f} "
              f"(task={train_task:.4f}, feat={train_feat:.4f}) | "
              f"Val MSE={val_mse:.4f}, MAE={val_mae:.4f} | step={global_step}")

        early_stopping(val_mse, student_model, model_save_path)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    student_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    student_model.eval()

    test_mse, test_mae = 0.0, 0.0
    n_test = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = student_model(batch_x)
            target_y = batch_y[:, :output.shape[1], :]
            test_mse += F.mse_loss(output, target_y).item()
            test_mae += F.l1_loss(output, target_y).item()
            n_test += 1

    test_mse /= max(n_test, 1)
    test_mae /= max(n_test, 1)
    print(f"  >>> Test: MSE={test_mse:.6f}, MAE={test_mae:.6f}")

    return test_mse, test_mae, history, global_step


# ==========================================
# 5. 主流程
# ==========================================

MODE_META = {
    'scratch':             {'label': 'From Scratch',     'distill': False,  'layer_mi': False, 'token_mask': False},
    'uniform':              {'label': 'Uniform Distill',  'distill': True,   'layer_mi': False, 'token_mask': False},
    'layer_mi':            {'label': 'Layer-MI Distill', 'distill': True,   'layer_mi': True,  'token_mask': False},
    'token_mask':          {'label': 'Token-Mask Dist',   'distill': True,   'layer_mi': False, 'token_mask': True},
    'token_mask_layer_mi': {'label': 'Token+Layer Dist',  'distill': True,   'layer_mi': True,  'token_mask': True},
}


def run_comparison_experiment(args):
    print("=" * 70)
    print("Timer Token-Level MI Masked Knowledge Distillation Comparison")
    print("=" * 70)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n>>> [1] Loading MI token masks from: {args.mi_dir}")
    token_mask_info = build_mi_token_mask(args.mi_dir, args.model_id)
    print(f"  Layers={token_mask_info['n_layers']}, Patches={token_mask_info['n_patches']}")
    print(f"  High-MI token ratio: {token_mask_info['high_token_ratio']:.1%}")
    print(f"  Consistent high-MI patches across layers: {token_mask_info['all_high_patches']}")
    print(f"  Per-layer high patches:")
    for li in sorted(token_mask_info['layer_token_mask'].keys(), key=int):
        mask = token_mask_info['layer_token_mask'][li]
        high_indices = [i for i, v in enumerate(mask) if v]
        print(f"    Layer {li}: high={high_indices}")

    n_t_layers = token_mask_info['n_layers']
    n_patches = token_mask_info['n_patches']

    mi_matrix = None
    for fname in ["mi_matrix_future.npy", "mi_hy_matrix.npy", "mi_matrix.npy"]:
        p = os.path.join(args.mi_dir, fname)
        if os.path.exists(p):
            mi_matrix = np.load(p)
            print(f"  Loaded MI matrix from {p}: shape={mi_matrix.shape}")
            break

    if mi_matrix is not None:
        mi_mean_per_layer = mi_matrix.mean(axis=1)
        mi_layer_weights = mi_mean_per_layer / mi_mean_per_layer.sum()
        mi_layer_weights_tensor = torch.from_numpy(mi_layer_weights).float()
    else:
        mi_layer_weights = None
        mi_layer_weights_tensor = None

    print(f"\n>>> [2] Loading dataset: {args.data}")
    class _Args:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    ds_args = _Args(
        data=args.data, root_path=args.root_path, data_path=args.data_path,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        stride=args.patch_len, enc_in=args.enc_in, dec_in=args.enc_in,
        c_out=args.enc_in, features=args.features, target=args.target,
        embed=args.embed, freq=args.freq, batch_size=args.batch_size,
        num_workers=args.num_workers, augmentation_ratio=0,
        model_id=args.model_id, use_ddp=False,
        task_name="forecast", seasonal_patterns="Monthly",
    )

    train_data, train_loader = data_provider(ds_args, flag="train")
    _, val_loader = data_provider(ds_args, flag="val")
    _, test_loader = data_provider(ds_args, flag="test")

    n_vars = train_data.n_var
    print(f"  Variables: {n_vars}")
    print(f"  Train: {len(train_data)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    print(f"\n>>> [3] Loading Timer teacher (frozen)...")
    teacher_model = load_teacher_model(
        ckpt_path=args.ckpt_path,
        patch_len=args.patch_len, stride=args.patch_len,
        d_model=args.d_model_t, d_ff=args.d_ff_t,
        e_layers=args.e_layers, n_heads=args.n_heads,
        seq_len=args.seq_len, pred_len=args.pred_len,
        device=device
    )
    core = _unwrap(teacher_model)
    teacher_d_model = core.d_model
    teacher_n_layers = core.layers
    print(f"  Teacher: d_model={teacher_d_model}, layers={teacher_n_layers}")

    layer_mapping = {}
    for s_idx in range(args.s_n_layers):
        t_idx = min((s_idx + 1) * (teacher_n_layers // args.s_n_layers),
                    teacher_n_layers - 1)
        layer_mapping[s_idx] = t_idx
    print(f"  Layer mapping: {layer_mapping}")

    output_dir = os.path.join(args.out_dir, args.model_id,
                              f"token_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}")

    results = {}
    max_steps = {}

    run_modes = ['scratch', 'uniform', 'layer_mi', 'token_mask', 'token_mask_layer_mi']

    for mode in run_modes:
        meta = MODE_META[mode]
        print(f"\n{'='*70}")
        print(f"  [{mode.upper()}] {meta['label']}")
        print(f"{'='*70}")

        student = StudentTransformer(
            seq_len=args.seq_len, patch_len=args.patch_len,
            d_model=args.s_d_model, n_layers=args.s_n_layers,
            n_heads=args.s_n_heads, d_ff=args.s_d_ff,
            dropout=args.s_dropout, pred_len=args.pred_len,
            n_channels=n_vars
        ).to(device)
        print(f"  Student: d_model={args.s_d_model}, layers={args.s_n_layers}, "
              f"heads={args.s_n_heads}, d_ff={args.s_d_ff}")

        save_path = os.path.join(output_dir, f"best_{mode}.pth")

        if not meta['distill']:
            mi_w = None
            tok_mask = None
        else:
            mi_w = mi_layer_weights_tensor if meta['layer_mi'] else None
            tok_mask = token_mask_info if meta['token_mask'] else None

        mse, mae, history, total_steps = train_model(
            student, teacher_model, train_loader, val_loader, test_loader,
            device, args,
            mode=mode,
            mi_layer_weights=mi_w,
            layer_mapping=layer_mapping,
            token_mask_info=tok_mask,
            model_save_path=save_path
        )

        results[mode] = {
            'mse': float(mse), 'mae': float(mae),
            'total_steps': total_steps,
            'history': {k: [float(x) for x in v] for k, v in history.items()}
        }
        max_steps[mode] = total_steps

        with open(os.path.join(output_dir, f"curve_{mode}.json"), 'w') as f:
            json.dump({
                'mode': mode,
                'total_steps': total_steps,
                'curve': [
                    dict(zip(history.keys(), t))
                    for t in zip(*[history[k] for k in history])
                ]
            }, f, indent=2)

    print(f"\n{'='*70}")
    print("  FINAL COMPARISON (Test Set)")
    print(f"{'='*70}")
    print(f"{'Method':<28} | {'Test MSE':<12} | {'Test MAE':<12} | {'vs Scratch':<12} | {'Steps':<8}")
    print("-" * 80)

    scratch_mse = results['scratch']['mse']
    for mode in run_modes:
        mse = results[mode]['mse']
        mae = results[mode]['mae']
        rel = (scratch_mse - mse) / scratch_mse * 100
        steps = results[mode]['total_steps']
        label = MODE_META[mode]['label']
        print(f"{label:<28} | {mse:<12.6f} | {mae:<12.6f} | {rel:>+.2f}%     | {steps:<8}")

    print(f"\n{'='*70}")
    print("  CONVERGENCE ANALYSIS (Time-to-Threshold)")
    print(f"{'='*70}")

    thresholds = [0.80, 0.70, 0.60, 0.55, 0.50, 0.45]
    print(f"{'Threshold':<10}", end="")
    for mode in run_modes:
        print(f" | {MODE_META[mode]['label'][:12]:<12}", end="")
    print()
    print("-" * 100)

    for thresh in thresholds:
        print(f"{thresh:<10}", end="")
        for mode in run_modes:
            hist = results[mode]['history']
            val_mses = hist.get('val_mse', [])
            found_step = None
            for i, v in enumerate(val_mses):
                if v <= thresh:
                    steps_list = hist.get('step', [])
                    if i < len(steps_list):
                        found_step = steps_list[i]
                    else:
                        found_step = (i + 1) * (max_steps[mode] // max(len(val_mses), 1))
                    break
            if found_step is not None:
                print(f" | {found_step:<12}", end="")
            else:
                print(f" | {'N/A':<12}", end="")
        print()

    summary = {
        'model_id': args.model_id,
        'timestamp': datetime.now().isoformat(),
        'modes': run_modes,
        'mode_meta': MODE_META,
        'test_results': {m: {'mse': results[m]['mse'], 'mae': results[m]['mae']} for m in run_modes},
        'total_steps': max_steps,
        'token_mask_info': {
            'high_token_ratio': token_mask_info['high_token_ratio'],
            'all_high_patches': token_mask_info['all_high_patches'],
            'n_layers': token_mask_info['n_layers'],
            'n_patches': token_mask_info['n_patches'],
        },
        'layer_mapping': {str(k): v for k, v in layer_mapping.items()},
        'mi_layer_weights': mi_layer_weights.tolist() if mi_layer_weights is not None else None,
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - curve_*.json          (per-mode learning curves)")
    print(f"  - best_*.pth            (best checkpoints)")
    print(f"  - summary.json          (final comparison)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timer Token-Level MI Masked KD Comparison")
    parser.add_argument("--mi_dir", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./datasets/")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--out_dir", type=str, default="./results/timer_token_mi_distillation/")
    parser.add_argument("--model_id", type=str, default="etth1")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/Timer_forecast_1.0.ckpt")
    parser.add_argument("--d_model_t", type=int, default=1024)
    parser.add_argument("--d_ff_t", type=int, default=2048)
    parser.add_argument("--e_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--s_d_model", type=int, default=256)
    parser.add_argument("--s_d_ff", type=int, default=512)
    parser.add_argument("--s_n_layers", type=int, default=4)
    parser.add_argument("--s_n_heads", type=int, default=4)
    parser.add_argument("--s_dropout", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--patch_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--features", type=str, default="M", choices=["S", "M"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    run_comparison_experiment(args)
