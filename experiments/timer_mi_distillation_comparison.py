#!/usr/bin/env python3
"""
Timer MI引导知识蒸馏对比实验

对比三种训练方式：
  1. 从零训练学生模型（StudentTransformer）
  2. 普通知识蒸馏（均匀权重）
  3. MI引导知识蒸馏（按层归一化 MI 加权）

教师：Pretrained Timer（thuml/Timer，冻结backbone）
学生：轻量 StudentTransformer（从头训练，或蒸馏自教师）

与 sundial_mi_distillation_comparison.py 对齐：
  - 单入口 run_comparison_experiment()
  - MI权重：按层归一化的标量（而非 [n_layers, n_patches] 矩阵）
  - Timer per-channel forward 提取教师隐状态
  - 格式化结果汇总 + JSON 保存

Usage:
    python experiments/timer_mi_distillation_comparison.py \
        --mi_dir ./results/timer_mi_ksg_pca/run_xxx \
        --root_path ./datasets/ --data ETTh1 --data_path ETTh1.csv \
        --seq_len 672 --pred_len 96 --label_len 48 \
        --batch_size 32 --epochs 10 --gpu 0 \
        --out_dir ./results/timer_mi_distillation/ --model_id etth1
"""

import argparse
import json
import os
import sys
from datetime import datetime

# 禁用 torch._dynamo，避免 PyTorch/transformers 版本不兼容导致的崩溃
# 必须在 import torch 之前设置，因为访问 torch._dynamo 会触发级联导入
os.environ["TORCHDYNAMO_AUTOJECT"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 修复: torch.utils._pytree 缺少 register_pytree_node
# 导致 transformers 导入时报 AttributeError
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
                print(f'  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'  Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
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
        # data_provider 返回 [B, seq_len, 1]，patch embedding 永远只看 1 通道
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

        # CIDatasetBenchmark: x shape = [B, seq_len, 1]
        # Reshape to [B, n_patches, patch_len], then flatten to [B*n_patches, patch_len]
        x_patched = x.reshape(B, n_patches, self.patch_len)
        x_flat = x_patched.reshape(B * n_patches, self.patch_len)

        h = self.patch_embedding(x_flat)                       # [B*n_patches, d_model]
        h = h.reshape(B, n_patches, self.d_model)             # [B, n_patches, d_model]
        h = self.pos_encoding(h)

        hidden_states = [h]
        for layer in self.encoder.layers:
            h = layer(h)
            hidden_states.append(h)

        output = h.reshape(B, -1)
        forecast = self.head(output)  # [B, pred_len]
        forecast = forecast.reshape(B, self.pred_len, 1)  # [B, pred_len, 1]

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

    Timer 对每个变量单独处理（univariate per pass），最后沿变量维度取平均。
    返回 list of [B, N, D] tensors（每个 decoder 层一个）。

    与 sundial_mi_distillation_comparison.py 中 get_teacher_hidden_states 的
    逻辑完全对齐：per-channel forward + avg across channels。
    """
    core = _unwrap(model)

    seq_x = batch_x.float().to(device)
    B, seq_len, num_channels = seq_x.shape

    # Non-stationary Normalization（与 Timer.forecast 对齐）
    means = seq_x.mean(dim=1, keepdim=True).detach()
    stdev = torch.sqrt(
        torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    stdev = torch.clamp_min(stdev, 1e-5)
    x_norm = (seq_x - means) / stdev

    # Patch embedding: [B, T, M] -> [B*M, N, D]
    x2 = x_norm.permute(0, 2, 1)  # [B, M, T]
    dec_in, n_vars = core.enc_embedding(x2)  # [B*M, N, D]

    BM, N, D = dec_in.shape
    B, M = B * n_vars, BM // B  # recover B and M from B*M

    # 将 [B*M, N, D] 重排为 [B, M, N, D]，沿 M 取平均得到 [B, N, D]
    # 这与 Student hidden states 格式一致
    dec_in = dec_in.reshape(B, M, N, D)  # [B, M, N, D]
    dec_in = dec_in.mean(dim=1)            # [B, N, D] — 跨变量平均

    from utils.masking import TriangularCausalMask
    causal_mask = TriangularCausalMask(B, N, device=device)

    # 逐层提取隐状态（与 TimerDecoder.forward 对齐）
    hidden_states = [dec_in]
    for layer_mod in core.decoder.attn_layers:
        dec_in, _, _ = layer_mod(dec_in, attn_mask=causal_mask)
        dec_in = dec_in.reshape(B, M, N, D).mean(dim=1)  # 始终保持 [B, N, D]
        hidden_states.append(dec_in)

    if core.decoder.norm is not None:
        dec_in = core.decoder.norm(dec_in)
        hidden_states.append(dec_in)

    return hidden_states


# ==========================================
# 3. Patch-importance 加权任务损失（方案 A）
# ==========================================


def compute_task_loss_patch_weighted(student_output, target, seq_len, patch_len,
                                     mode="uniform", patch_weights=None,
                                     high_patches=None, low_patches=None):
    """Patch-importance 加权的预测损失（不改 DataLoader）。

    约定：
      - student_output/target: [B, pred_len, C]
      - 当 pred_len==patch_len 时，只覆盖最后一个 patch（index = seq_len/patch_len - 1）。
      - 当 pred_len>patch_len 且能整除 patch_len 时，按 patch 切块加权。

    mode:
      - uniform: 全 1
      - high_only: high_patches=1, others=0
      - low_only: low_patches=1, others=0
      - mi_weighted: 使用 patch_weights（长度 N=seq_len/patch_len，或仅覆盖预测 patch 的子集）
    """
    B, pred_len, C = student_output.shape
    n_patches = seq_len // patch_len
    assert n_patches > 0

    mse_per_t = (student_output - target).pow(2).mean(dim=-1)  # [B, pred_len]

    # 预测覆盖的 patch 范围（按 seq_len 的 patch index 计）
    if pred_len == patch_len:
        pred_patch_indices = [n_patches - 1]
        mse_per_patch = mse_per_t.mean(dim=1, keepdim=True)  # [B, 1]
    else:
        if pred_len % patch_len != 0:
            # 最稳妥：退化为逐时间点 uniform（避免隐式错切）
            return mse_per_t.mean()
        k = pred_len // patch_len
        start = max(n_patches - k, 0)
        pred_patch_indices = list(range(start, start + k))
        mse_per_patch = mse_per_t.reshape(B, k, patch_len).mean(dim=-1)  # [B, k]

    # 构造 patch 权重向量（对应 pred_patch_indices）
    if mode == "uniform":
        w = torch.ones(len(pred_patch_indices), device=student_output.device, dtype=student_output.dtype)
    elif mode == "high_only":
        hs = set(high_patches or [])
        w = torch.tensor([1.0 if p in hs else 0.0 for p in pred_patch_indices],
                         device=student_output.device, dtype=student_output.dtype)
    elif mode == "low_only":
        ls = set(low_patches or [])
        w = torch.tensor([1.0 if p in ls else 0.0 for p in pred_patch_indices],
                         device=student_output.device, dtype=student_output.dtype)
    elif mode == "mi_weighted":
        if patch_weights is None:
            raise ValueError("mode=mi_weighted 需要 patch_weights")
        # patch_weights 可以是全长 N，也可以只给预测覆盖的子集
        if len(patch_weights) == n_patches:
            w = torch.tensor([float(patch_weights[p]) for p in pred_patch_indices],
                             device=student_output.device, dtype=student_output.dtype)
        elif len(patch_weights) == len(pred_patch_indices):
            w = torch.tensor([float(x) for x in patch_weights],
                             device=student_output.device, dtype=student_output.dtype)
        else:
            raise ValueError(f"patch_weights 长度不匹配: got {len(patch_weights)}, expect {n_patches} or {len(pred_patch_indices)}")
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 归一化（避免不同模式下 loss 尺度差异过大；同时保留 0 权重情形）
    w_sum = w.sum().clamp_min(1e-12)
    w = w / w_sum

    return (mse_per_patch * w.unsqueeze(0)).sum(dim=1).mean()


def build_patch_weights_from_mi_json(mi_data, strategy="diff_mean", eps=1e-6):
    """从 global_mi_peaks_*.json 构造全局 patch 权重 w_p（长度 N）。

    strategy:
      - diff_mean: 对每层 curve 做 (curve-mean) 的正部，然后跨层平均
      - high_vote: 统计每层 high_mi_patches 出现次数作为权重
    """
    n_patches = int(mi_data.get("N"))
    layers = mi_data.get("layers", {})

    if strategy == "high_vote":
        counts = np.zeros(n_patches, dtype=np.float32)
        for li, info in layers.items():
            for p in info.get("high_mi_patches", []):
                if 0 <= int(p) < n_patches:
                    counts[int(p)] += 1.0
        w = counts + eps
        return (w / w.sum()).tolist()

    if strategy == "diff_mean":
        diffs = []
        for li, info in layers.items():
            curve = info.get("hsic_curve", None)
            if curve is None:
                curve = info.get("cka_curve", None)
            if curve is None:
                continue
            curve = np.asarray(curve, dtype=np.float32)
            if curve.shape[0] != n_patches:
                continue
            d = curve - curve.mean()
            d = np.clip(d, a_min=0.0, a_max=None)
            diffs.append(d)
        if not diffs:
            # fallback: uniform
            return (np.ones(n_patches, dtype=np.float32) / n_patches).tolist()
        w = np.mean(np.stack(diffs, axis=0), axis=0) + eps
        return (w / w.sum()).tolist()

    raise ValueError(f"未知 strategy: {strategy}")

def compute_distillation_loss(student_output, target, teacher_hiddens=None, student_hiddens=None,
                              mi_weights=None, layer_mapping=None, projector=None, alpha=1.0):
    """
    计算蒸馏损失（与 sundial 版本完全对齐）。

    MI权重策略：
      - mi_weights: [n_layers] tensor，每层归一化后的 MI 权重（标量）
      - 每个教师层的 patch 内部使用均匀权重（不区分 patch 粒度）
      - 层间按 mi_weights 加权

    Args:
        student_output: [B, pred_len, C] 学生预测
        target:         [B, pred_len, C] 真实值
        teacher_hiddens: list of [B*M, N, D_t] 教师隐状态
        student_hiddens: list of [B, N, D_s] 学生隐状态
        mi_weights:      [n_layers] per-layer MI 权重（归一化标量）
        layer_mapping:   dict {student_idx: teacher_idx}
        projector:       Projector 模块
        alpha:          特征蒸馏权重

    Returns:
        total_loss, loss_dict
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

        # Patch 数不匹配则跳过
        if N_t != N_s:
            continue

        # 维度映射
        t_h = projector(t_h, s_idx)

        # Per-patch MSE：[B, N, D] -> [B, N]
        per_patch_mse = F.mse_loss(t_h, s_h, reduction='none').mean(dim=-1)

        # 层内均匀加权
        layer_loss = per_patch_mse.mean()

        # 层间按 MI 权重加权
        if mi_weights is not None:
            w = float(mi_weights[t_idx]) if t_idx < len(mi_weights) else 1.0
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
                args, mode='scratch', mi_weights=None, layer_mapping=None,
                model_save_path="best_model.pth"):
    """
    训练学生模型（与 sundial 版本完全对齐）。

    三种模式：
      - 'scratch':       仅 MSE 任务损失（从头训练）
      - 'uniform':       MSE + 均匀特征蒸馏
      - 'mi_weighted':   MSE + MI 加权特征蒸馏
    """
    projector = None
    params = list(student_model.parameters())
    teacher_d_model = None

    if mode != 'scratch' and layer_mapping is not None:
        # 从教师模型获取维度信息
        core = _unwrap(teacher_model)
        teacher_d_model = core.d_model
        projector = Projector(layer_mapping, teacher_d_model, args.s_d_model).to(device)
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [{mode}]")

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

                loss, loss_dict = compute_distillation_loss(
                    student_output=output, target=target_y,
                    teacher_hiddens=teacher_hiddens, student_hiddens=student_hiddens,
                    mi_weights=mi_weights if mode == 'mi_weighted' else None,
                    layer_mapping=layer_mapping, projector=projector,
                    alpha=args.alpha
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

        train_loss /= max(n_batches, 1)
        train_task /= max(n_batches, 1)
        train_feat /= max(n_batches, 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_task'].append(train_task)
        history['train_feat'].append(train_feat)

        # ── 验证集评估 ──────────────────────────────────────────
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

        print(f"  Epoch {epoch + 1}: train={train_loss:.4f} "
              f"(task={train_task:.4f}, feat={train_feat:.4f}) | "
              f"Val MSE={val_mse:.4f}, Val MAE={val_mae:.4f}")

        early_stopping(val_mse, student_model, model_save_path)
        if early_stopping.early_stop:
            print("  触发早停 (Early stopping)!")
            break

    # ── 测试集评估 ──────────────────────────────────────────────
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
    print(f"  >>> 测试集结果: MSE={test_mse:.6f}, MAE={test_mae:.6f}")

    return test_mse, test_mae, history


# ==========================================
# 5. 主流程（与 sundial 版本完全对齐）
# ==========================================

def run_comparison_experiment(args):
    print("=" * 70)
    print("Timer MI引导知识蒸馏对比实验")
    print("=" * 70)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── Phase 1: 加载 MI 权重 ───────────────────────────────────
    print("\n>>> Phase 1: 加载MI权重...")

    mi_matrix = None
    for fname in ["mi_matrix_future.npy", "mi_hy_matrix.npy", "mi_matrix.npy"]:
        p = os.path.join(args.mi_dir, fname)
        if os.path.exists(p):
            mi_matrix = np.load(p)
            print(f"  从 {p} 加载 MI 矩阵: shape={mi_matrix.shape}")
            break

    if mi_matrix is None:
        json_path = os.path.join(args.mi_dir, f"global_mi_peaks_{args.model_id}.json")
        if not os.path.exists(json_path):
            candidates = [f for f in os.listdir(args.mi_dir)
                          if f.startswith("global_mi_peaks_") and f.endswith(".json")]
            if candidates:
                json_path = os.path.join(args.mi_dir, candidates[0])
        if os.path.exists(json_path):
            with open(json_path) as f:
                mi_data = json.load(f)
            n_layers = mi_data["num_layers"]
            n_patches = mi_data["N"]
            mi_matrix = np.zeros((n_layers, n_patches))
            for li in range(n_layers):
                mi_matrix[li] = mi_data["layers"][str(li)]["hsic_curve"]
            print(f"  从 {json_path} 重建 MI 矩阵: {n_layers} layers x {n_patches} patches")
        else:
            raise FileNotFoundError(
                f"未找到 MI 矩阵文件。mi_dir={args.mi_dir}\n"
                f"期望: mi_matrix_future.npy / mi_hy_matrix.npy / mi_matrix.npy / global_mi_peaks_*.json"
            )

    n_t_layers, n_patches = mi_matrix.shape
    print(f"  MI矩阵形状: {n_t_layers} layers x {n_patches} patches")

    # 与 sundial 版本对齐：按层归一化为标量权重
    mi_mean_per_layer = mi_matrix.mean(axis=1)  # [n_layers,]
    mi_weights = mi_mean_per_layer / mi_mean_per_layer.sum()  # 归一化到和为1
    mi_weights_tensor = torch.from_numpy(mi_weights).float()
    print(f"  MI层权重（归一化）: {mi_weights.round(4)}")

    # ── Phase 2: 加载数据集 ─────────────────────────────────────
    print("\n>>> Phase 2: 加载数据集...")

    class _Args:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    ds_args = _Args(
        data=args.data,
        root_path=args.root_path,
        data_path=args.data_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        stride=args.patch_len,
        enc_in=args.enc_in,
        dec_in=args.enc_in,
        c_out=args.enc_in,
        features=args.features,
        target=args.target,
        embed=args.embed,
        freq=args.freq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_ratio=0,
        model_id=args.model_id,
        use_ddp=False,
        task_name="forecast",
        seasonal_patterns="Monthly",
    )

    train_data, train_loader = data_provider(ds_args, flag="train")
    _, val_loader = data_provider(ds_args, flag="val")
    _, test_loader = data_provider(ds_args, flag="test")

    n_vars = train_data.n_var
    print(f"  变量数: {n_vars}")
    print(f"  训练集: {len(train_data)} | 验证集: {len(val_loader.dataset)} | 测试集: {len(test_loader.dataset)}")

    # ── Phase 3: 加载教师模型（Timer 冻结）───────────────────────
    print("\n>>> Phase 3: 加载教师模型 (Timer 冻结)...")
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
    print(f"  教师隐层维度: {teacher_d_model}, 层数: {teacher_n_layers}")

    # 层映射：教师层数 -> 学生层数（均匀映射）
    layer_mapping = {}
    for s_idx in range(args.s_n_layers):
        t_idx = min((s_idx + 1) * (teacher_n_layers // args.s_n_layers),
                    teacher_n_layers - 1)
        layer_mapping[s_idx] = t_idx
    print(f"  层映射: {layer_mapping}")

    # ── 创建输出目录 ────────────────────────────────────────────
    output_dir = os.path.join(args.out_dir, args.model_id,
                              f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    results = {}

    # ── 实验 1：从零训练 ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("实验 1: 从零训练学生模型 (StudentTransformer)")
    print("=" * 70)
    student_scratch = StudentTransformer(
        seq_len=args.seq_len, patch_len=args.patch_len,
        d_model=args.s_d_model, n_layers=args.s_n_layers,
        n_heads=args.s_n_heads, d_ff=args.s_d_ff,
        dropout=args.s_dropout, pred_len=args.pred_len,
        n_channels=n_vars
    ).to(device)
    print(f"  学生模型: d_model={args.s_d_model}, n_layers={args.s_n_layers}, "
          f"n_heads={args.s_n_heads}, d_ff={args.s_d_ff}, n_channels={n_vars}")

    mse_s, mae_s, hist_s = train_model(
        student_scratch, teacher_model, train_loader, val_loader, test_loader,
        device, args, mode='scratch',
        model_save_path=os.path.join(output_dir, "best_scratch.pth")
    )
    results['scratch'] = {'mse': float(mse_s), 'mae': float(mae_s), 'history': hist_s}

    # ── 实验 2：普通蒸馏（均匀权重）─────────────────────────────
    print("\n" + "=" * 70)
    print("实验 2: 普通知识蒸馏（均匀权重）")
    print("=" * 70)
    student_u = StudentTransformer(
        seq_len=args.seq_len, patch_len=args.patch_len,
        d_model=args.s_d_model, n_layers=args.s_n_layers,
        n_heads=args.s_n_heads, d_ff=args.s_d_ff,
        dropout=args.s_dropout, pred_len=args.pred_len,
        n_channels=n_vars
    ).to(device)

    mse_u, mae_u, hist_u = train_model(
        student_u, teacher_model, train_loader, val_loader, test_loader,
        device, args, mode='uniform',
        layer_mapping=layer_mapping,
        model_save_path=os.path.join(output_dir, "best_uniform.pth")
    )
    results['uniform'] = {'mse': float(mse_u), 'mae': float(mae_u), 'history': hist_u}

    # ── 实验 3：MI 引导蒸馏 ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("实验 3: MI引导知识蒸馏（按层归一化 MI 加权）")
    print("=" * 70)
    student_mi = StudentTransformer(
        seq_len=args.seq_len, patch_len=args.patch_len,
        d_model=args.s_d_model, n_layers=args.s_n_layers,
        n_heads=args.s_n_heads, d_ff=args.s_d_ff,
        dropout=args.s_dropout, pred_len=args.pred_len,
        n_channels=n_vars
    ).to(device)

    mse_m, mae_m, hist_m = train_model(
        student_mi, teacher_model, train_loader, val_loader, test_loader,
        device, args, mode='mi_weighted',
        mi_weights=mi_weights_tensor, layer_mapping=layer_mapping,
        model_save_path=os.path.join(output_dir, "best_mi.pth")
    )
    results['mi_weighted'] = {'mse': float(mse_m), 'mae': float(mae_m), 'history': hist_m}

    # ── 结果汇总 ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("对比结果汇总 (测试集)")
    print("=" * 70)
    print(f"{'方法':<30} | {'MSE':<12} | {'MAE':<12} | {'MSE相对提升':<15}")
    print("-" * 75)
    print(f"{'从零训练 (Scratch)':<30} | {mse_s:<12.4f} | {mae_s:<12.4f} | {'-':<15}")

    imp_u = (mse_s - mse_u) / mse_s * 100
    print(f"{'普通蒸馏 (Uniform)':<30} | {mse_u:<12.4f} | {mae_u:<12.4f} | {imp_u:>+.2f}%")

    imp_m = (mse_s - mse_m) / mse_s * 100
    print(f"{'MI引导蒸馏 (MI)':<30} | {mse_m:<12.4f} | {mae_m:<12.4f} | {imp_m:>+.2f}%")

    imp_m_vs_u = (mse_u - mse_m) / mse_u * 100
    print(f"\nMI蒸馏 vs 普通蒸馏: {imp_m_vs_u:>+.2f}%")

    results['comparison'] = {
        'scratch': {'mse': float(mse_s), 'mae': float(mae_s)},
        'uniform': {'mse': float(mse_u), 'mae': float(mae_u)},
        'mi_weighted': {'mse': float(mse_m), 'mae': float(mae_m)},
        'improve_uniform_vs_scratch': float(imp_u),
        'improve_mi_vs_scratch': float(imp_m),
        'improve_mi_vs_uniform': float(imp_m_vs_u),
        'mi_layer_weights': mi_weights.tolist(),
        'layer_mapping': {str(k): v for k, v in layer_mapping.items()},
    }

    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果与模型检查点已保存到: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timer MI引导知识蒸馏对比实验")
    # 路径
    parser.add_argument("--mi_dir", type=str, required=True,
                        help="MI矩阵目录（包含 mi_matrix.npy 或 global_mi_peaks_*.json）")
    parser.add_argument("--root_path", type=str, default="./datasets/", help="数据根目录")
    parser.add_argument("--data", type=str, default="ETTh1", help="数据集名称")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="数据文件")
    parser.add_argument("--out_dir", type=str, default="./results/timer_mi_distillation/",
                        help="输出目录")
    parser.add_argument("--model_id", type=str, default="etth1", help="模型ID")
    # 教师模型（Timer）
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/Timer_forecast_1.0.ckpt",
                        help="Timer checkpoint 路径")
    parser.add_argument("--d_model_t", type=int, default=1024, help="教师 d_model (Timer)")
    parser.add_argument("--d_ff_t", type=int, default=2048, help="教师前馈维度")
    parser.add_argument("--e_layers", type=int, default=8, help="教师层数")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    # 学生模型
    parser.add_argument("--s_d_model", type=int, default=256, help="学生 d_model")
    parser.add_argument("--s_d_ff", type=int, default=512, help="学生前馈维度")
    parser.add_argument("--s_n_layers", type=int, default=4, help="学生层数")
    parser.add_argument("--s_n_heads", type=int, default=4, help="学生注意力头数")
    parser.add_argument("--s_dropout", type=float, default=0.1, help="Dropout率")
    # 数据
    parser.add_argument("--seq_len", type=int, default=672, help="输入序列长度")
    parser.add_argument("--pred_len", type=int, default=96, help="预测长度")
    parser.add_argument("--label_len", type=int, default=48, help="标签长度")
    parser.add_argument("--patch_len", type=int, default=96, help="Patch 长度")
    parser.add_argument("--enc_in", type=int, default=7, help="输入变量数")
    parser.add_argument("--features", type=str, default="M", choices=["S", "M"],
                        help="S=单变量, M=多变量")
    parser.add_argument("--target", type=str, default="OT", help="目标变量")
    parser.add_argument("--embed", type=str, default="timeF", help="时间特征嵌入方式")
    parser.add_argument("--freq", type=str, default="h", help="数据频率")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    # 训练
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--alpha", type=float, default=1.0, help="特征蒸馏权重")
    # 其他
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")

    args = parser.parse_args()
    run_comparison_experiment(args)
