import os
import sys
import time
import warnings
import json
import pathlib
from typing import Optional

# Clear stale cached bytecode to prevent import errors from outdated .pyc files
for cache_dir in pathlib.Path("./exp").rglob("__pycache__"):
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map

warnings.filterwarnings('ignore')


def _apply_region_pooling(
    attn_or_logits: torch.Tensor,
    p_mi: torch.Tensor,
    high_mi_mask: torch.Tensor,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply 1D adaptive average pooling to all alignment inputs.

    This implements region-wise alignment: instead of matching token-by-token,
    both distributions are coarsened so that each pool of W consecutive tokens
    maps to a single region, allowing the model to freely shift attention
    within a region without penalty.

    Math:
        L_orig  = Loss(A, MI)                          (token-level)
        L_pooled = Loss(AvgPool(A, W), AvgPool(MI, W)) (region-level)
                   where W = pool_size, S' = floor(S / W)

    Args:
        attn_or_logits: [B, H, L, S] attention or logits tensor
        p_mi:          [S] target MI distribution
        high_mi_mask:  [L] binary mask for active query tokens (unchanged)
        pool_size:     window size; 1 = no pooling (default)

    Returns:
        (pooled_attn_logits, pooled_p_mi, pooled_high_mi_mask)
        - pooled_attn_logits: [B, H, L, S']  where S' = floor(S / W)
        - pooled_p_mi:  [S']
        - pooled_high_mi_mask: [L]  (unchanged — pooling is over the sequence dim only)
    """
    if pool_size <= 1:
        return attn_or_logits, p_mi, high_mi_mask

    S = attn_or_logits.shape[-1]
    S_pooled = S // pool_size
    if S_pooled == 0:
        return attn_or_logits, p_mi, high_mi_mask

    B, H, L, _ = attn_or_logits.shape

    # Pool the sequence dimension (last axis) of attention / logits
    # [B, H, L, S] -> flatten to [B*H*L, 1, S] for adaptive_avg_pool1d
    flat = attn_or_logits.permute(0, 1, 2, 3).reshape(B * H * L, 1, S)   # [BHL, 1, S]
    pooled_flat = F.adaptive_avg_pool1d(flat, output_size=S_pooled)         # [BHL, 1, S']
    pooled_tensor = pooled_flat.reshape(B, H, L, S_pooled)                  # [B, H, L, S']

    # Pool MI distribution: [S] -> [S']
    pooled_p_mi = F.adaptive_avg_pool1d(
        p_mi.unsqueeze(0).unsqueeze(0),   # [1, 1, S]
        output_size=S_pooled,
    ).squeeze(0).squeeze(0)                # [S']

    return pooled_tensor, pooled_p_mi, high_mi_mask


def _layer_align_loss(p_mi: torch.Tensor, attn: torch.Tensor,
                       high_mi_mask: torch.Tensor,
                       pool_size: int = 1,
                       eps: float = 1e-8) -> torch.Tensor:
    """
    Row-level KL alignment with per-layer high-MI mask and optional region pooling:

        L_align = KL( AvgPool(P_MI, W) || AvgPool(A, W)_{i,:} )
        W = pool_size

    - p_mi: [S] — target distribution from CKA/MI curve of THIS layer
    - high_mi_mask: [L] — 1 for high-MI rows (contribute to loss), 0 for low-MI rows (free)
    - attn: [B, H, L, S]
    - pool_size: window for region-wise alignment; 1 = token-to-token (no pooling)
    """
    attn, p_mi, high_mi_mask = _apply_region_pooling(attn, p_mi, high_mi_mask, pool_size)
    B, H, L, S = attn.shape
    p_mi = p_mi.to(attn.device)
    high_mi_mask = high_mi_mask.to(attn.device)

    p_mi_clamped = p_mi.clamp(min=eps)
    p_mi_norm = p_mi_clamped / p_mi_clamped.sum(dim=-1, keepdim=True)

    log_attn = torch.log_softmax(attn, dim=-1)
    kl_full = F.kl_div(log_attn, p_mi_norm.unsqueeze(0).unsqueeze(0),
                       reduction='none').sum(dim=-1)

    kl_row = kl_full.mean(dim=(0, 1))

    masked_kl = kl_row * high_mi_mask
    denom = high_mi_mask.sum().clamp(min=eps)
    return masked_kl.sum() / denom


def _layer_logit_align_loss(
        p_mi: torch.Tensor,
        logits: torch.Tensor,
        high_mi_mask: torch.Tensor,
        gamma: torch.Tensor,
        sharp_k: int = 0,
        pool_size: int = 1,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Probability-space MSE alignment with optional region pooling.

    Formula:
        L_align(l) = (1 / sum(M)) * sum_i M_i * sum_h
                     MSE( softmax(logits_i^{(h)} / temp), softmax(p_mi / tau) )

    When pool_size > 1, both sequences are pooled before comparison:
        L_align(l) = MSE( AvgPool(softmax(...)), AvgPool(softmax(p_mi)) )

    - p_mi:      [S] raw MI energy scores
    - logits:    [B, H, L, S] raw (pre-softmax) attention scores
    - high_mi_mask: [L] binary mask; M_i = 1 for high-MI tokens
    - gamma:     temperature for softmax normalisation
    - sharp_k:   if > 0, average only the top-K highest-variance heads; 0 = all heads
    - pool_size: window for region-wise alignment; 1 = token-to-token (no pooling)
    """
    logits, p_mi, high_mi_mask = _apply_region_pooling(logits, p_mi, high_mi_mask, pool_size)
    B, H, L, S = logits.shape
    p_mi = p_mi.to(logits.device)
    high_mi_mask = high_mi_mask.to(logits.device)
    gamma = gamma.to(logits.device)

    # Convert target MI scores to probability distribution via softmax
    p_mi_shifted = p_mi - p_mi.max()
    p_mi_prob = torch.softmax(p_mi_shifted / gamma.clamp(min=eps), dim=-1)   # [S], sum=1

    if sharp_k > 0 and sharp_k < H:
        head_var = logits.var(dim=(0, 2, 3))
        _, topk_idx = head_var.topk(sharp_k, largest=True)
        logits_subset = logits[:, topk_idx, :, :]
        logits_head_avg = logits_subset.mean(dim=1)   # [B, sharp_k, L, S]
    else:
        logits_head_avg = logits.mean(dim=1)           # [B, H, L, S]

    # Convert attention logits to probability distributions via softmax
    attn_prob = torch.softmax(logits_head_avg / gamma.clamp(min=eps), dim=-1)  # [B, H, L, S], sum over S = 1

    # MSE per token position per head: mean over S (probability dimension)
    mse_per_head_row = ((attn_prob - p_mi_prob.unsqueeze(0).unsqueeze(0)) ** 2).mean(dim=-1)   # [B, H, L]

    # Mask and average
    denom = high_mi_mask.sum().clamp(min=eps)
    masked_mse = mse_per_head_row * high_mi_mask.unsqueeze(0).unsqueeze(0)  # [B, H, L]
    layer_loss = masked_mse.sum(dim=(0, 1, 2)) / (denom * H)

    # NaN guard
    if torch.isnan(layer_loss) or torch.isinf(layer_loss):
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return layer_loss


def _soft_dtw_batch(
    p_mi: torch.Tensor,
    attn_rows: torch.Tensor,
    gamma: float = 1.0,
    bandwidth: int = -1,
) -> torch.Tensor:
    """
    Fully-vectorized Soft-DTW over a batch of (reference, query) pairs.

    Args:
        p_mi:      [B, S] — reference distribution (target MI, one per batch element)
        attn_rows: [B, S] — query distribution (attention row, one per batch element)
        gamma:     Soft-DTW temperature.
        bandwidth: Sakoe-Chiba band. -1 = full matrix.

    Returns:
        [B] tensor of Soft-DTW distances.
    """
    if p_mi.dim() == 1:
        p_mi = p_mi.unsqueeze(0)
    if attn_rows.dim() == 1:
        attn_rows = attn_rows.unsqueeze(0)

    B, S = p_mi.shape  # S = sequence length

    # Squared cost matrix: [B, S, S]
    cost = (p_mi[:, :, None] - attn_rows[:, None, :]) ** 2

    # Sakoe-Chiba band mask: [B, S, S]
    if bandwidth >= 0:
        mask = torch.ones(B, S, S, device=cost.device, dtype=torch.bool)
        for i in range(S):
            j_min = max(0, i - bandwidth)
            j_max = min(S, i + bandwidth + 1)
            mask[:, i, :j_min] = False
            mask[:, i, j_max:] = False
        cost = cost.masked_fill(~mask, float('inf'))

    # ── Log-space DP with one loop over the matrix dimension ──────────────────
    g = max(gamma, 1e-8)
    D = torch.full((B, S + 1, S + 1), float('inf'), device=cost.device)
    D[:, 0, 0] = 0.0
    # First row and first column are 0 (start state: can step right or down from origin)
    D[:, 0, 1:] = 0.0
    D[:, 1:, 0] = 0.0

    for i in range(1, S + 1):
        for j in range(1, S + 1):
            c = cost[:, i - 1, j - 1]
            # Stack up to 3 neighbours and compute softmin in one op
            if i == 1 and j == 1:
                # D[1,1] = c + D[0,0] = c (origin state)
                D[:, 1, 1] = c
                continue
            elif i == 1:
                # D[1,j] = c + D[0,j] = c + 0 = c  (first row: only from left)
                D[:, 1, j] = c
                continue
            elif j == 1:
                # D[i,1] = c + D[i,0] = c + 0 = c  (first col: only from above)
                D[:, i, 1] = c
                continue
            # Interior: soft-min over three neighbours
            neighbors = torch.stack(
                [D[:, i - 1, j], D[:, i, j - 1], D[:, i - 1, j - 1]], dim=0
            )  # [3, B]
            neg = -neighbors / g
            log_sum = torch.logcumsumexp(neg, dim=0)[-1]
            D[:, i, j] = c - g * log_sum

    return D[:, -1, -1]  # [B]


def _layer_softdtw_align_loss(
        p_mi: torch.Tensor,
        attn: torch.Tensor,
        high_mi_mask: torch.Tensor,
        gamma: float = 1.0,
        bandwidth: int = -1,
        pool_size: int = 1,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Fully-vectorized Soft-DTW alignment loss.
    All (batch, query_token) pairs are processed in a single GPU call.

    Args:
        p_mi:        [S]  — target MI distribution
        attn:        [B, H, L, S] — post-softmax attention per head and query token
        high_mi_mask:[L]  — binary mask; M_i=1 for active query tokens
        gamma:       Soft-DTW temperature
        bandwidth:   Sakoe-Chiba band
        pool_size:   window for region-wise pooling; 1 = token-to-token
        eps:         Numerical floor

    Returns:
        scalar alignment loss
    """
    attn, p_mi, high_mi_mask = _apply_region_pooling(attn, p_mi, high_mi_mask, pool_size)
    B, H, L, S = attn.shape
    p_mi = p_mi.to(attn.device)
    high_mi_mask = high_mi_mask.to(attn.device)

    # Normalize MI distribution and ensure 1D: [S]
    p_mi = p_mi.squeeze().clamp(min=eps)
    p_mi_norm = p_mi / p_mi.sum(dim=-1, keepdim=True)   # [S]

    # Average over heads: [B, L, S]
    attn_head_avg = attn.mean(dim=1)

    # Select only high-MI token positions
    active_idx = high_mi_mask.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        return torch.tensor(0.0, device=attn.device)

    L_active = active_idx.shape[0]
    attn_active = attn_head_avg[:, active_idx, :]       # [B, L_active, S]

    # Expand p_mi_norm: each (batch, token) pair gets its own copy of the same reference
    # p_mi_norm [S] → [1, 1, S] → [B, L_active, S] → [B*L, S]
    p_mi_expanded = p_mi_norm.unsqueeze(0).unsqueeze(0).expand(B, L_active, -1)
    p_mi_flat = p_mi_expanded.reshape(B * L_active, S)
    attn_flat = attn_active.reshape(B * L_active, S)    # [B*L, S]

    # Soft-DTW for all pairs at once → [B * L_active]
    losses = _soft_dtw_batch(p_mi_flat, attn_flat, gamma=gamma, bandwidth=bandwidth)

    return losses.mean()


class Exp_Forecast(Exp_Basic):

    def __init__(self, args):
        super().__init__(args)
        self._load_align_mi_scores()

    def _load_align_mi_scores(self):
        """Load per-layer CKA/MI curves and high-MI row masks from JSON file."""
        self.align_mi_raw: dict[int, list[float]] = {}
        self.align_high_mi_mask: dict[int, torch.Tensor] = {}
        self._align_mi_tensors: dict[int, torch.Tensor] = {}
        if not getattr(self.args, 'use_align_loss', False):
            return
        align_file = getattr(self.args, 'align_loss_file', '')
        if not align_file:
            print("[AlignLoss] WARNING: use_align_loss=True but align_loss_file is empty. "
                  "Skipping alignment loss.")
            return
        if not os.path.exists(align_file):
            print(f"[AlignLoss] WARNING: align_loss_file not found: {align_file}. "
                  "Skipping alignment loss.")
            return

        align_layers = getattr(self.args, 'align_loss_layers', [0, 1, 2, 3, 4, 5, 6, 7])

        with open(align_file, 'r') as f:
            data = json.load(f)

        for li in align_layers:
            key = str(li)
            if key not in data.get('layers', {}):
                print(f"[AlignLoss] WARNING: layer {li} not found in {align_file}. Skipping.")
                continue
            layer_entry = data['layers'][key]
            hsic_curve = layer_entry.get('hsic_curve', []) or layer_entry.get('hsic_curve1', [])
            if not hsic_curve:
                print(f"[AlignLoss] WARNING: hsic_curve empty for layer {li}. Skipping.")
                continue
            self.align_mi_raw[li] = hsic_curve

            high_patches = layer_entry.get('high_mi_patches', [])
            T = len(hsic_curve)
            mask = torch.zeros(T, dtype=torch.float32)
            for idx in high_patches:
                if 0 <= idx < T:
                    mask[idx] = 1.0
            self.align_high_mi_mask[li] = mask

        if self.align_mi_raw:
            print(f"[AlignLoss] Loaded per-layer HSIC curves for {len(self.align_mi_raw)} layers: "
                  f"layers={sorted(self.align_mi_raw.keys())}")

    def _ensure_mi_tensor(self, li: int, device: torch.device) -> torch.Tensor:
        """
        Lazily convert raw MI scores to a tensor on the target device.

        - KL mode: returns softmax(MI / tau) — a probability distribution.
        - Logit-MSE mode: returns raw shifted MI scores — an energy template
          (no softmax, so the full dynamic range is preserved).
        """
        if li not in self._align_mi_tensors:
            mi_t = torch.tensor(self.align_mi_raw[li], dtype=torch.float32, device=device)
            mode = getattr(self.args, 'align_loss_mode', 'kl')
            if mode == 'logit_mse':
                # Logit-MSE: use raw shifted MI as energy template (no softmax)
                mi_t = mi_t - mi_t.max()
                self._align_mi_tensors[li] = mi_t
            elif mode == 'softdtw':
                # Soft-DTW: softmax-normalised probability distribution
                # (required for squared-error cost matrix to be well-defined)
                mi_t = mi_t - mi_t.max()
                self._align_mi_tensors[li] = torch.softmax(mi_t / 1.0, dim=-1)
            else:
                # KL mode: softmax over sequence dimension as probability distribution
                tau = self.model.align_tau.item() if hasattr(self.model, 'align_tau') else getattr(self.args, 'align_loss_tau', 0.001)
                mi_t = mi_t - mi_t.max()
                self._align_mi_tensors[li] = torch.softmax(mi_t / tau, dim=-1)
        return self._align_mi_tensors[li]
    def _compute_align_loss(self, attns: list, logits_list: Optional[list] = None) -> torch.Tensor:
        """
        Compute hierarchical alignment loss.

        KL mode (default):
            L_align = sum_{l in align_layers} KL(P_MI^{(l)} || P_attn^{(l)})

        Logit-MSE mode:
            L_align = sum_{l in align_layers} (1/sum(M)) * sum_i M_i *
                                       sum_h MSE(Logits_i^{(h,l)}, gamma * P_MI^{(l)})

        Soft-DTW mode:
            L_align = sum_{l in align_layers} (1/sum(M)) * sum_i M_i *
                                       SoftDTW( P_MI^{(l)}, mean_h(attn_i^{(h,l)}) )
            where SoftDTW uses logcumsumexp for fully differentiable dynamic alignment.

        Region-wise pooling (pool_size > 1):
            All modes above first apply AvgPool(W) to both distributions before comparison,
            coarsening from token-level to region-level alignment (W = align_pool_size).

        attns:        list of [B, H, L, S] softmax attention tensors, one per decoder layer
        logits_list:  list of [B, H, L, S] raw (pre-softmax) logits, one per decoder layer
                      Required when align_loss_mode == 'logit_mse'.
        """
        if not self.align_mi_raw:
            return torch.tensor(0.0, device=self.device)

        mode = getattr(self.args, 'align_loss_mode', 'kl')
        total_align = torch.tensor(0.0, device=self.device)
        align_layers = getattr(self.args, 'align_loss_layers', [])
        valid_count = 0
        _debug_skips = {}   # track why layers are skipped

        for li in align_layers:
            if li >= len(attns):
                _debug_skips[li] = 'no_attn'
                continue
            if li not in self.align_mi_raw:
                _debug_skips[li] = 'no_mi'
                continue
            if li not in self.align_high_mi_mask:
                _debug_skips[li] = 'no_mask'
                continue
            attn = attns[li]
            if attn is None:
                _debug_skips[li] = 'attn_none'
                continue
            T = attn.shape[-1]
            p_mi = self._ensure_mi_tensor(li, attn.device)
            if p_mi.shape[-1] != T:
                _debug_skips[li] = 'shape_mismatch_pmi_T{}_attn_T{}'.format(p_mi.shape[-1], T)
                continue
            high_mi_mask = self.align_high_mi_mask[li]
            if high_mi_mask.shape[-1] != T:
                _debug_skips[li] = 'shape_mismatch_mask_T{}_attn_T{}'.format(high_mi_mask.shape[-1], T)
                continue

            if mode == 'logit_mse':
                if logits_list is None:
                    _debug_skips[li] = 'logits_none'
                    continue
                if li >= len(logits_list):
                    _debug_skips[li] = 'logits_missing_idx{}_len{}'.format(li, len(logits_list))
                    continue
                logits = logits_list[li]
                if logits is None:
                    _debug_skips[li] = 'logits_is_none'
                    continue
                gamma = self.model.align_gamma if hasattr(self.model, 'align_gamma') else torch.tensor(
                    getattr(self.args, 'align_gamma', 0.0), device=attn.device)
                sharp_k = getattr(self.args, 'align_sharp_heads', 0)
                pool_size = getattr(self.args, 'align_pool_size', 1)
                layer_loss = _layer_logit_align_loss(p_mi, logits, high_mi_mask, gamma,
                                                     sharp_k=sharp_k, pool_size=pool_size)
            elif mode == 'softdtw':
                dtw_gamma = getattr(self.args, 'align_dtw_gamma', 1.0)
                dtw_bw = getattr(self.args, 'align_dtw_bw', -1)
                pool_size = getattr(self.args, 'align_pool_size', 1)
                layer_loss = _layer_softdtw_align_loss(
                    p_mi, attn, high_mi_mask,
                    gamma=dtw_gamma, bandwidth=dtw_bw, pool_size=pool_size
                )
            else:
                pool_size = getattr(self.args, 'align_pool_size', 1)
                layer_loss = _layer_align_loss(p_mi, attn, high_mi_mask, pool_size=pool_size)

            total_align = total_align + layer_loss
            valid_count += 1

        if valid_count > 0:
            total_align = total_align / valid_count
        else:
            # Print debug info so we know exactly which guard triggered
            print('[DEBUG _compute_align_loss] mode={} valid=0 | attns_len={} | '
                  'logits_list_len={} | skips={} | align_layers={}'.format(
                      mode, len(attns),
                      len(logits_list) if logits_list else None,
                      list(_debug_skips.items()), align_layers))

        return total_align

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        use_align = getattr(self.args, 'use_align_loss', False) and self.align_mi_raw
        mode = getattr(self.args, 'align_loss_mode', 'kl')
        rank_sq_error = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        rank_abs_error = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        rank_count = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                if use_align:
                    if mode == 'logit_mse':
                        model_result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  output_hidden_states=False,
                                                  output_attention_override=True)
                        if len(model_result) == 3:
                            outputs, attns, logits_list = model_result
                        else:
                            outputs, attns = model_result
                            logits_list = None
                    else:
                        outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                    output_hidden_states=False)
                        logits_list = None
                    align_loss = self._compute_align_loss(attns, logits_list)
                    beta = self.model.align_beta
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         output_attention_override=False)
                    logits_list = None

                if self.args.use_ims:
                    pred = outputs[:, -self.args.seq_len:, :]
                    true = batch_y
                    if flag == 'vali':
                        pred = pred[:, -self.args.pred_len:, :]
                        true = batch_y[:, -self.args.pred_len:, :]
                    elif flag == 'test':
                        pred = pred[:, -self.args.pred_len:, :]
                        true = batch_y[:, -self.args.pred_len:, :]
                else:
                    pred = outputs[:, -self.args.pred_len:, :]
                    true = batch_y[:, -self.args.pred_len:, :]

                se = (pred - true) ** 2
                ae = torch.abs(pred - true)
                rank_sq_error += se.sum().double()
                rank_abs_error += ae.sum().double()
                rank_count += pred.numel()

                if use_align:
                    rank_sq_error += (beta * align_loss).detach().double() * pred.numel()

        if self.args.use_multi_gpu:
            dist.barrier()
            sq_list = [torch.zeros_like(rank_sq_error) for _ in range(dist.get_world_size())]
            ae_list = [torch.zeros_like(rank_abs_error) for _ in range(dist.get_world_size())]
            cnt_list = [torch.zeros_like(rank_count) for _ in range(dist.get_world_size())]
            dist.all_gather(sq_list, rank_sq_error)
            dist.all_gather(ae_list, rank_abs_error)
            dist.all_gather(cnt_list, rank_count)
            total_sq = sum(t.item() for t in sq_list)
            total_ae = sum(t.item() for t in ae_list)
            total_cnt = sum(t.item() for t in cnt_list)
        else:
            total_sq = rank_sq_error.item()
            total_ae = rank_abs_error.item()
            total_cnt = rank_count.item()

        total_loss = total_sq / total_cnt if total_cnt > 0 else 0.0
        self.model.train()
        return total_loss

    def finetune(self, setting):
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if getattr(self.args, 'use_align_loss', False):
            init_beta = getattr(self.args, 'align_loss_weight', 1.0)
            init_tau = getattr(self.args, 'align_loss_tau', 0.001)
            mode = getattr(self.args, 'align_loss_mode', 'kl')

            if mode == 'logit_mse':
                # logit_mse uses z-score normalization — no softmax, tau is irrelevant.
                # Store beta and tau as plain tensors (non-learnable) to avoid NaN
                # from gradient corruption in early steps.
                self.model.align_beta = torch.tensor(init_beta, device=self.device)
                self.model.align_tau = torch.tensor(init_tau, device=self.device)

                init_gamma = getattr(self.args, 'align_gamma', 0.0)
                if init_gamma <= 0.0:
                    d_model = self.args.d_model
                    n_heads = getattr(self.args, 'n_heads', 8)
                    init_gamma = (d_model / n_heads) ** 0.5
                if getattr(self.args, 'align_gamma_learnable', False):
                    self.model.align_gamma = nn.Parameter(torch.tensor(init_gamma, device=self.device))
                    model_optim.param_groups[0]['params'].append(self.model.align_gamma)
                else:
                    self.model.align_gamma = torch.tensor(init_gamma, device=self.device)
                print(f"[AlignLoss] Mode: {mode} | beta={init_beta:.4f} | gamma={init_gamma:.4f} "
                      f"(gamma_learnable={getattr(self.args, 'align_gamma_learnable', False)})")
            else:
                # KL mode: beta and tau are learnable (original behavior)
                self.model.align_beta = nn.Parameter(torch.tensor(init_beta, device=self.device))
                self.model.align_tau = nn.Parameter(torch.tensor(init_tau, device=self.device))
                model_optim.param_groups[0]['params'].extend([self.model.align_beta, self.model.align_tau])

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)


        for epoch in range(self.args.finetune_epochs):
            # tau is learnable — must invalidate cached softmax tensors each epoch
            self._align_mi_tensors.clear()

            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            align_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            align_freq = getattr(self.args, 'align_loss_freq', 1)
            mode = getattr(self.args, 'align_loss_mode', 'kl')

            print(f"Epoch {epoch+1} | mode={mode} | align_layers={getattr(self.args, 'align_loss_layers', [])} | "
                  f"beta={self.model.align_beta if hasattr(self.model, 'align_beta') else 'N/A'}")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                use_align = getattr(self.args, 'use_align_loss', False) and self.align_mi_raw
                mode = getattr(self.args, 'align_loss_mode', 'kl')

                if use_align:
                    if mode == 'logit_mse':
                        model_result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  output_hidden_states=False,
                                                  output_attention_override=True)
                        if len(model_result) == 3:
                            outputs, attns, logits_list = model_result
                        else:
                            outputs, attns = model_result
                            logits_list = None
                    else:
                        outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                    output_hidden_states=False,
                                                    output_attention_override=True)
                        logits_list = None
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         output_attention_override=False)
                    logits_list = None

                if self.args.use_ims:
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                # NaN/Inf guard: skip this step if task loss is corrupted.
                # This prevents the optimizer from receiving bad gradients that destroy the model.
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[WARN] NaN/Inf in task loss at iter {iter_count}, epoch {epoch+1}. "
                          f"Skipping update. (loss={loss.item():.4f})")
                    model_optim.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                if use_align and (iter_count % align_freq == 0):
                    align_loss = self._compute_align_loss(attns, logits_list)
                    # Guard: if alignment loss itself is NaN/Inf, fall back to 0 so the
                    # task loss gradient is still propagated (prevents model corruption).
                    if torch.isnan(align_loss) or torch.isinf(align_loss):
                        align_loss = torch.tensor(0.0, device=loss.device)
                        print(f"[WARN] NaN in align_loss at iter {iter_count}, epoch {epoch+1}. Using 0.")
                    beta = self.model.align_beta
                    total_loss = loss + beta * align_loss
                    align_loss_val = align_loss.detach()
                else:
                    total_loss = loss
                    align_loss_val = torch.tensor(0.0, device=loss.device)

                loss_val += loss.item()
                align_val += align_loss_val.item()
                count += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    align_str = f"| align_loss: {align_loss_val.item():.7f}" if use_align else ""
                    print(
                        f"\titers: {i}, epoch: {epoch + 1} | loss: {loss.item():.7f}{align_str} | "
                        f"cost_time: {cost_time:.0f} | memory: allocated {torch.cuda.memory_allocated() / 1024 / 1024:.0f}MB, "
                        f"reserved {torch.cuda.memory_reserved() / 1024 / 1024:.0f}MB, "
                        f"cached {torch.cuda.memory_cached() / 1024 / 1024:.0f}MB")
                    time_now = time.time()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(align_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()
            train_align = align_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if getattr(self.args, 'use_align_loss', False):
                align_beta_val = self.model.align_beta.item()
                align_tau_val = self.model.align_tau.item()
                mode = getattr(self.args, 'align_loss_mode', 'kl')
                if mode == 'logit_mse':
                    gamma_val = self.model.align_gamma.item() if hasattr(self.model.align_gamma, 'item') else float(self.model.align_gamma)
                    beta_str = f" | align_beta: {align_beta_val:.6f}, align_tau: {align_tau_val:.6f}, gamma: {gamma_val:.4f}"
                else:
                    beta_str = f" | align_beta: {align_beta_val:.6f}, align_tau: {align_tau_val:.6f}"
            else:
                beta_str = ""
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                align_str = f" | Align: {train_align:.7f}" if getattr(self.args, 'use_align_loss', False) else ""
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}{4} | Vali Loss: {3:.7f} | Test Loss: {5:.7f}{6}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, align_str, test_loss, beta_str))
            else:
                align_str = f" | Align: {train_align:.7f}" if getattr(self.args, 'use_align_loss', False) else ""
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}{4} | Vali Loss: {3:.7f}{5}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, align_str, beta_str))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        self.args.output_len_list.sort()

        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')

                rank_sq_error = torch.tensor(0.0, dtype=torch.float64, device=self.device)
                rank_abs_error = torch.tensor(0.0, dtype=torch.float64, device=self.device)
                rank_count = torch.tensor(0, dtype=torch.int64, device=self.device)

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    inference_steps = self.args.output_len // self.args.pred_len
                    dis = self.args.output_len - inference_steps * self.args.pred_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            tmp = batch_y_mark[:, j - 1:j, :]
                            batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                        mode_test = getattr(self.args, 'align_loss_mode', 'kl')
                        use_output_attn = self.args.output_attention or getattr(self.args, 'use_align_loss', False)
                        if use_output_attn:
                            model_result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                     output_hidden_states=False)
                            if mode_test == 'logit_mse' and len(model_result) == 3:
                                outputs, attns, logits_list = model_result
                            elif len(model_result) >= 2:
                                outputs, attns = model_result[:2]
                            else:
                                outputs = model_result
                                attns = None
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 output_attention_override=False)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred_y.append(outputs[:, -self.args.pred_len:, :])
                    pred_y = torch.cat(pred_y, dim=1)

                    if dis != 0:
                        pred_y = pred_y[:, :-self.args.pred_len+dis, :]

                    if self.args.use_ims:
                        batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                            self.device)
                    else:
                        batch_y = batch_y[:, :self.args.output_len, :].to(self.device)
                    outputs = pred_y.detach().cpu()
                    batch_y = batch_y.detach().cpu()

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    se = (pred - true) ** 2
                    ae = torch.abs(pred - true)
                    rank_sq_error += se.sum().double()
                    rank_abs_error += ae.sum().double()
                    rank_count += pred.numel()

                    if i % 10 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                        rank0 = int(os.environ.get("LOCAL_RANK", "0"))
                        if rank0 == 0:
                            if self.args.output_attention:
                                attn = attns[0].cpu().numpy()[0, 0, :, :]
                                attn_map(attn, os.path.join(folder_path, f'attn_{i}_{rank0}.pdf'))
                            visual(gt, pd, os.path.join(folder_path, f'{i}_{rank0}.pdf'))

                # ── all-reduce 聚合 ───────────────────────────────────────────────
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                if world_size > 1:
                    sq_list = [torch.zeros_like(rank_sq_error) for _ in range(world_size)]
                    ae_list = [torch.zeros_like(rank_abs_error) for _ in range(world_size)]
                    cnt_list = [torch.zeros_like(rank_count) for _ in range(world_size)]
                    dist.all_gather(sq_list, rank_sq_error)
                    dist.all_gather(ae_list, rank_abs_error)
                    dist.all_gather(cnt_list, rank_count)
                    total_sq = sum(t.item() for t in sq_list)
                    total_ae = sum(t.item() for t in ae_list)
                    total_cnt = sum(t.item() for t in cnt_list)
                else:
                    total_sq = rank_sq_error.item()
                    total_ae = rank_abs_error.item()
                    total_cnt = rank_count.item()

                mse = total_sq / total_cnt if total_cnt > 0 else 0.0
                mae = total_ae / total_cnt if total_cnt > 0 else 0.0

                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(f"output_len: {self.args.output_len_list[output_ptr]}")
                    print('mse:{}, mae:{}'.format(mse, mae))
                    f = open("result_long_term_forecast.txt", 'a')
                    f.write(setting + "  \n")
                    f.write('mse:{}, mae:{}'.format(mse, mae))
                    f.write('\n')
                    f.write('\n')
                    f.close()

        return
