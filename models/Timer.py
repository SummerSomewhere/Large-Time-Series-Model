from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.spectral_residual import SpectralResidualBranch
from models import TimerBackbone
from models.checkpoint_utils import load_backbone_state_dict
from models.periodic_time_utils import (
    marks_to_hour_dow_indices,
    patch_center_timestep_indices,
)


class HMISRRefiner(nn.Module):
    """
    Hard-masked in-situ refiner: x += lambda * ((alpha * hard_mask) * MLP(x)).
    Only selected patch indices receive learnable gates; others are forced to zero (spatial isolation).
    MLP: Linear(d, d//2) -> GELU -> Linear(d//2, d) -> LayerNorm.
    Alpha: normal_(std=1e-5) kickstart so the gate has a non-zero gradient path from step 0.
    """

    def __init__(
        self,
        d_model: int,
        num_patches: int,
        lambda_scale: float = 10.0,
        patch_indices: tuple[int, ...] = (3, 6),
    ):
        super().__init__()
        h = max(1, d_model // 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, d_model),
            nn.LayerNorm(d_model),
        )
        self.alpha = nn.Parameter(torch.empty(num_patches, 1))
        nn.init.normal_(self.alpha, mean=0.0, std=1e-5)
        self.register_buffer("lambda_scale", torch.tensor(float(lambda_scale), dtype=torch.float32))
        mask = torch.zeros(num_patches, 1, dtype=torch.float32)
        for idx in patch_indices:
            ii = int(idx)
            if 0 <= ii < num_patches:
                mask[ii, 0] = 1.0
        self.register_buffer("patch_hard_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, D]; gate [P,1] * mask -> broadcast
        refined_x = self.mlp(x)
        g = (self.alpha * self.patch_hard_mask).unsqueeze(0)
        ls = self.lambda_scale.to(dtype=x.dtype, device=x.device)
        return x + ls * (g * refined_x)


class SIGRefiner(nn.Module):
    """
    Legacy post-stack gate: injects proj(first_layer) into final encoder output (full-patch alpha).
    x += lambda_scale * (alpha * refined(recycle_feat)). Alpha: normal_(std=1e-5) kickstart.
    """

    def __init__(self, d_model: int, num_patches: int, lambda_scale: float = 10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(num_patches, 1))
        nn.init.normal_(self.alpha, mean=0.0, std=1e-5)
        self.register_buffer("lambda_scale", torch.tensor(float(lambda_scale), dtype=torch.float32))
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, current_feat: torch.Tensor, recycle_feat: torch.Tensor) -> torch.Tensor:
        refined_x = self.proj(recycle_feat)
        g = self.alpha.unsqueeze(0)
        ls = self.lambda_scale.to(dtype=current_feat.dtype, device=current_feat.device)
        return current_feat + ls * (g * refined_x)


# Backward-compatible name (legacy refiner class).
SIGGate = SIGRefiner


def _cross_attn_bridge_num_heads(d_model: int, requested: int) -> int:
    """Pick head count in [1, 4] with d_model % n_heads == 0 (fallback downward)."""
    r = int(max(1, min(4, requested)))
    for nh in range(r, 0, -1):
        if d_model % nh == 0:
            return nh
    return 1


class CrossAttentionBridge(nn.Module):
    """
    MI-inspired cross-layer retrieval: Query = final encoder output, Key/Value = first-layer memory.
    Scalar alpha (normal std=1e-5) scales the cross-attn residual.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        nh = _cross_attn_bridge_num_heads(d_model, n_heads)
        self.n_heads = nh
        self.mha = nn.MultiheadAttention(d_model, nh, dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.alpha, mean=0.0, std=1e-5)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # query, memory: [B, N, D]
        attn_out, _ = self.mha(query, memory, memory, need_weights=False)
        return query + self.alpha * attn_out


class Model(nn.Module):
    """
    Timer: Generative Pre-trained Transformers Are Large Time Series Models (ICML 2024)

    Optional periodic embedding residual (PEFT): hour/day nn.Embedding + gamma * sum,
    applied after the Transformer stack and before proj (see periodic_embedding_branch).
    """

    # First N training forwards with SIG_REFINER_DEBUG=1 (avoid log flood).
    _sig_gate_debug_fwd_prints_left: int | None = None

    def _debug_print_sig_gate_if_enabled(self) -> None:
        """SIG_REFINER_DEBUG=1, LOCAL_RANK=0: print alpha mean; grad is usually None until backward."""
        if int(os.environ.get("SIG_REFINER_DEBUG", "0")) == 0:
            return
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        if self.sig_gate is None or not self.training:
            return
        rem = Model._sig_gate_debug_fwd_prints_left
        if rem is None:
            Model._sig_gate_debug_fwd_prints_left = 10
            rem = 10
        if rem <= 0:
            return
        Model._sig_gate_debug_fwd_prints_left = rem - 1
        a = self.sig_gate.alpha
        grad_str = "None" if a.grad is None else f"{a.grad.mean().item():.8e}"
        print(
            f">>> DEBUG: alpha mean: {a.data.mean().item():.8f}, grad: {grad_str}",
            flush=True,
        )

    @staticmethod
    def _parse_hm_isr_patch_indices(configs) -> tuple[int, ...]:
        raw = str(getattr(configs, "hm_isr_patch_indices", "") or "").strip()
        if not raw:
            return (3, 6)
        out: list[int] = []
        for part in raw.replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                continue
        return tuple(out) if out else (3, 6)

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.ckpt_path = configs.ckpt_path
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        self.output_attention = configs.output_attention
        self.periodic_embedding_branch = bool(int(getattr(configs, "periodic_embedding_branch", 0)))
        self.geometric_hpe = bool(int(getattr(configs, "geometric_hpe", 0)))
        self.data_freq = getattr(configs, "freq", "h")

        self.backbone = TimerBackbone.Model(configs)
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj
        self.enc_embedding = self.backbone.patch_embedding

        if self.periodic_embedding_branch:
            d_model = configs.d_model
            bank = int(getattr(configs, "periodic_emb_bank_dim", 0) or 0)
            if bank > 0:
                self.hour_embed = nn.Embedding(24, bank)
                self.day_embed = nn.Embedding(7, bank)
                self.periodic_to_hidden = nn.Linear(bank, d_model)
            else:
                self.hour_embed = nn.Embedding(24, d_model)
                self.day_embed = nn.Embedding(7, d_model)
                self.periodic_to_hidden = None
            self.periodic_gamma = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        if self.ckpt_path != '':
            if self.ckpt_path == 'random':
                print('loading model randomly')
            else:
                print('loading model: ', self.ckpt_path)
                strict_load = not (self.periodic_embedding_branch or self.geometric_hpe)
                if not strict_load:
                    print(
                        'Note: strict=False (periodic_embedding_branch or geometric_hpe): load compatible '
                        'backbone keys from ckpt; new/reshaped modules keep init (e.g. hour/day/gamma or HPE).'
                    )
                if self.ckpt_path.endswith('.pth'):
                    sd = load_backbone_state_dict(self.ckpt_path, from_lightning_ckpt=False)
                    self.backbone.load_state_dict(sd, strict=strict_load)
                elif self.ckpt_path.endswith('.ckpt'):
                    sd = load_backbone_state_dict(self.ckpt_path, from_lightning_ckpt=True)
                    self.backbone.load_state_dict(sd, strict=strict_load)
                else:
                    raise NotImplementedError

        # HM-ISR: hard-masked in-situ refiners after layer 0 and before last encoder layer (forecast).
        self.use_hm_isr = bool(int(getattr(configs, "hm_isr", 0)))
        self.hm_isr_entry: HMISRRefiner | None = None
        self.hm_isr_exit: HMISRRefiner | None = None
        # Legacy post-stack SIG-Refiner (disabled when HM-ISR is on).
        self.use_sig_gate = bool(int(getattr(configs, "sig_gate", 0)))
        if self.use_hm_isr and self.use_sig_gate:
            print(
                "NOTE: hm_isr is on; legacy post-stack sig_gate is not instantiated (use hm_isr OR sig_gate).",
                flush=True,
            )
        self.sig_gate: SIGRefiner | None = None
        if self.task_name == "forecast":
            pe = self.enc_embedding
            pad = pe.padding_patch_layer.padding
            pad_r = pad[-1] if isinstance(pad, tuple) and len(pad) >= 2 else 0
            L = int(getattr(configs, "seq_len", 0))
            Lp = L + pad_r
            if Lp >= pe.patch_len:
                n_sig = (Lp - pe.patch_len) // pe.stride + 1
                if self.use_hm_isr:
                    _hl = float(getattr(configs, "hm_isr_lambda_scale", 10.0))
                    _hpi = self._parse_hm_isr_patch_indices(configs)
                    self.hm_isr_entry = HMISRRefiner(
                        self.d_model, n_sig, lambda_scale=_hl, patch_indices=_hpi
                    )
                    self.hm_isr_exit = HMISRRefiner(
                        self.d_model, n_sig, lambda_scale=_hl, patch_indices=_hpi
                    )
                    print(
                        f"HM-ISR: num_patches={n_sig}, lambda_scale={_hl}, "
                        f"hard_mask_patch_indices={_hpi}, refiner_entry+exit (in-stack)",
                        flush=True,
                    )
                elif self.use_sig_gate:
                    _lg = float(getattr(configs, "sig_gate_lambda_scale", 10.0))
                    self.sig_gate = SIGRefiner(self.d_model, n_sig, lambda_scale=_lg)
                    print(
                        f"SIGRefiner (legacy post-stack): num_patches={n_sig}, lambda_scale={_lg}",
                        flush=True,
                    )
            else:
                if self.use_hm_isr or self.use_sig_gate:
                    print("WARNING: hm_isr / sig_gate disabled (seq_len too short for patch grid).")

        # Lightweight representation recycle (forecast only): at listed encoder layers, for each
        # selected patch run K rounds: h <- h + alpha_k * (TF_l(h) - h) on length-1 seq, then
        # F.layer_norm on that row (before stack final norm).
        self.recycle_encoder_layer = int(getattr(configs, "recycle_encoder_layer", -1))
        _rls = str(getattr(configs, "recycle_encoder_layers", "") or "").strip()
        self.recycle_encoder_layers: list[int] = []
        if _rls:
            for s in _rls.replace(",", " ").split():
                s = s.strip()
                if not s:
                    continue
                try:
                    self.recycle_encoder_layers.append(int(s))
                except ValueError:
                    pass
            self.recycle_encoder_layers = sorted(set(self.recycle_encoder_layers))
        elif self.recycle_encoder_layer >= 0:
            self.recycle_encoder_layers = [self.recycle_encoder_layer]

        _rp = getattr(configs, "recycle_patch_indices", "") or ""
        if isinstance(_rp, str) and _rp.strip():
            self._recycle_patch_indices_cfg = []
            for s in _rp.replace(",", " ").split():
                s = s.strip()
                if not s:
                    continue
                try:
                    self._recycle_patch_indices_cfg.append(int(s))
                except ValueError:
                    pass
        elif isinstance(_rp, (list, tuple)):
            self._recycle_patch_indices_cfg = [int(x) for x in _rp]
        else:
            self._recycle_patch_indices_cfg = []
        self.recycle_hsic_mean_npy = str(getattr(configs, "recycle_hsic_mean_npy", "") or "").strip()
        # tukey: Q3 + 1.5*IQR (same as MI experiment iqr_peak_mask); mean_iqr: mean + 1.5*IQR along patches
        self.recycle_peak_mode = str(getattr(configs, "recycle_peak_mode", "tukey")).lower()
        self._recycle_hsic_arr: np.ndarray | None = None
        if self.recycle_hsic_mean_npy:
            self._recycle_hsic_arr = np.load(self.recycle_hsic_mean_npy)

        self.recycle_alpha = float(getattr(configs, "recycle_alpha", 0.05))
        _ras = str(getattr(configs, "recycle_round_alphas", "") or "").strip()
        self.recycle_round_alphas: list[float] = []
        if _ras:
            for s in _ras.replace(",", " ").split():
                s = s.strip()
                if not s:
                    continue
                try:
                    self.recycle_round_alphas.append(float(s))
                except ValueError:
                    pass
        if not self.recycle_round_alphas:
            self.recycle_round_alphas = [self.recycle_alpha]

        self._recycle_wants_active = bool(
            self.recycle_encoder_layer >= 0
            or self.recycle_encoder_layers
            or self._recycle_patch_indices_cfg
            or self._recycle_hsic_arr is not None
        )

        # Cross-attention bridge: replaces linear recycle when enabled (first-layer memory -> final query).
        self.use_cross_attn_bridge = bool(int(getattr(configs, "cross_attn_bridge", 0)))
        self.cross_attn_bridge: CrossAttentionBridge | None = None
        if self.use_cross_attn_bridge and self.task_name == "forecast":
            _bh = int(getattr(configs, "cross_attn_bridge_heads", 2))
            _cab_drop = float(getattr(configs, "cross_attn_bridge_dropout", -1.0))
            if _cab_drop < 0.0:
                _cab_drop = float(self.dropout)
            self.cross_attn_bridge = CrossAttentionBridge(self.d_model, _bh, dropout=_cab_drop)
            print(
                f"CrossAttentionBridge: d_model={self.d_model}, n_heads={self.cross_attn_bridge.n_heads}, "
                f"dropout={_cab_drop} (linear recycle disabled when bridge is on)",
                flush=True,
            )

        # Variational information bottleneck before patch Linear head (forecast only).
        self.use_timer_vib = bool(int(getattr(configs, "timer_vib", 0)))
        if self.use_timer_vib:
            self.mu_layer = nn.Linear(self.d_model, self.d_model)
            self.logvar_layer = nn.Linear(self.d_model, self.d_model)

        # MI preservation (forecast training only): align early vs late encoder hidden states at selected patches.
        self.use_mi_preservation = (
            self.task_name == "forecast" and bool(int(getattr(configs, "mi_preservation", 0)))
        )
        self.mi_proj: nn.Sequential | None = None
        if self.use_mi_preservation:
            self.mi_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            self._mi_preserve_patch_idx: list[int] = list(
                Model._parse_mi_patch_indices_static(configs)
            )
            print("MI preservation: mi_proj enabled (train with return_mi_feats=True only)", flush=True)
        else:
            self._mi_preserve_patch_idx = []

        # Spectral Residual saliency prompt after patch embedding [B,P,D], before encoder stack.
        self.use_spectral_residual = bool(int(getattr(configs, "spectral_residual_branch", 0)))
        self.spectral_residual_branch: SpectralResidualBranch | None = None
        if self.use_spectral_residual:
            _srk = int(getattr(configs, "sr_smooth_kernel", 3))
            self.spectral_residual_branch = SpectralResidualBranch(self.d_model, smooth_kernel=_srk)
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    f"SpectralResidualBranch: d_model={self.d_model}, sr_smooth_kernel={_srk}",
                    flush=True,
                )

    @staticmethod
    def _parse_mi_patch_indices_static(configs) -> tuple[int, ...]:
        raw = str(getattr(configs, "mi_patch_indices", "") or "").strip()
        if not raw:
            return (3, 6)
        out: list[int] = []
        for part in raw.replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                continue
        return tuple(out) if out else (3, 6)

    def _mi_preservation_aux_loss(
        self, feat_early: torch.Tensor, feat_late: torch.Tensor
    ) -> torch.Tensor:
        """Cosine MI loss; early stop-grad; single mi_proj forward (DDP-safe). Must run inside model.forward."""
        assert self.mi_proj is not None
        p = int(feat_early.shape[1])
        idx = [i for i in self._mi_preserve_patch_idx if 0 <= i < p]
        if not idx:
            return feat_early.new_zeros(())
        early = feat_early[:, idx, :].detach()
        late = feat_late[:, idx, :]
        d = int(early.shape[-1])
        n_early = early.numel() // d
        both = torch.cat([early.reshape(-1, d), late.reshape(-1, d)], dim=0)
        proj_both = self.mi_proj(both)
        early_p = proj_both[:n_early].view_as(early)
        late_p = proj_both[n_early:].view_as(late)
        cos = F.cosine_similarity(early_p, late_p, dim=-1, eps=1e-8)
        return (1.0 - cos).mean()

    def _apply_spectral_residual(self, dec_in: torch.Tensor) -> torch.Tensor:
        if self.spectral_residual_branch is None:
            return dec_in
        return self.spectral_residual_branch(dec_in)

    def _vib_transform(
        self, dec_out: torch.Tensor, return_vib_kl: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reparameterize z ~ q(z|x); optional analytic KL to N(0,I) per latent dim (summed, then mean over tokens)."""
        mu = self.mu_layer(dec_out)
        logvar = self.logvar_layer(dec_out)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        need_kl = bool(return_vib_kl or self.training)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu
        vib_kl = None
        if need_kl:
            vib_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return z, vib_kl

    def _recycle_patch_indices_for_layer(self, n_patches: int, layer_idx: int) -> list[int]:
        """
        Patch indices to recycle at encoder layer layer_idx.
        If hsic_mean.npy is set, use row layer_idx for peak detection; else use configured list.
        """
        indices: list[int] = []
        if self._recycle_hsic_arr is not None:
            arr = self._recycle_hsic_arr
            if arr.ndim == 2 and 0 <= layer_idx < arr.shape[0]:
                row = np.asarray(arr[layer_idx], dtype=np.float64).ravel()
                finite = row[np.isfinite(row)]
                if finite.size >= 2:
                    q1, q3 = np.percentile(finite, [25, 75])
                    iqr = q3 - q1
                    if self.recycle_peak_mode == "mean_iqr":
                        mu = float(np.mean(finite))
                        thresh = mu + 1.5 * iqr
                    else:
                        thresh = q3 + 1.5 * iqr
                    mask = np.logical_and(np.isfinite(row), row > thresh)
                    indices = np.where(mask)[0].tolist()
        if not indices and self._recycle_patch_indices_cfg:
            indices = list(self._recycle_patch_indices_cfg)
        return [i for i in indices if 0 <= i < n_patches]

    def _decoder_forward_with_optional_recycle(
        self, x: torch.Tensor, capture_mi_feats: bool = False
    ) -> tuple[torch.Tensor, list, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Encoder forward; optional linear recycle (off if cross_attn_bridge); optional layer-0
        capture for SIG-Gate / CrossAttentionBridge.
        Returns (x, attns, first_layer_feat, feat_early, feat_late). feat_* are None unless
        capture_mi_feats: pure outputs of first / last EncoderLayer (before HM-ISR / recycle on those steps).
        """
        recycle_active = self._recycle_wants_active and not self.use_cross_attn_bridge
        need_first_layer = (self.sig_gate is not None) or self.use_cross_attn_bridge
        use_hm_isr_stack = self.hm_isr_entry is not None

        if not recycle_active and not need_first_layer and not use_hm_isr_stack and not capture_mi_feats:
            x, attns = self.decoder(x)
            return x, attns, None, None, None

        first_layer_feat: torch.Tensor | None = None
        feat_early: torch.Tensor | None = None
        feat_late: torch.Tensor | None = None
        n_enc = len(self.decoder.attn_layers)
        last_i = n_enc - 1
        if recycle_active:
            if self.recycle_encoder_layers:
                layer_set = frozenset(i for i in self.recycle_encoder_layers if 0 <= i < n_enc)
            elif 0 <= self.recycle_encoder_layer < n_enc:
                layer_set = frozenset([self.recycle_encoder_layer])
            else:
                layer_set = frozenset()
            if not layer_set:
                layer_set = frozenset([last_i])
        else:
            layer_set = frozenset()

        d_model = x.shape[-1]
        norm_eps = self.decoder.norm.eps if self.decoder.norm is not None else 1e-5
        alphas = self.recycle_round_alphas

        attns = []
        for i, layer in enumerate(self.decoder.attn_layers):
            x, attn = layer(x)
            attns.append(attn)
            if capture_mi_feats:
                if i == 0:
                    feat_early = x.clone()
                if i == last_i:
                    feat_late = x.clone()
            if need_first_layer and i == 0:
                first_layer_feat = x.clone()
            if use_hm_isr_stack and self.hm_isr_entry is not None and i == 0:
                x = self.hm_isr_entry(x)
            if recycle_active and i in layer_set:
                patch_idx = self._recycle_patch_indices_for_layer(x.shape[1], i)
                for pi in patch_idx:
                    for alpha in alphas:
                        seg = x[:, pi : pi + 1, :].contiguous()
                        h = seg[:, 0, :]
                        h_tf, _ = layer(seg)
                        h_final = h + alpha * (h_tf[:, 0, :] - h)
                        x[:, pi, :] = F.layer_norm(h_final, (d_model,), eps=norm_eps)
            if (
                use_hm_isr_stack
                and self.hm_isr_exit is not None
                and n_enc >= 2
                and i == n_enc - 2
            ):
                x = self.hm_isr_exit(x)
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x, attns, first_layer_feat, feat_early, feat_late

    def _apply_periodic_residual(
        self,
        dec_out: torch.Tensor,
        x_mark_enc: torch.Tensor,
        batch_size: int,
        seq_len: int,
        n_vars: int,
    ) -> torch.Tensor:
        """dec_out [B*M, N, D]; x_mark_enc [B, L, F]."""
        if not self.periodic_embedding_branch or x_mark_enc is None:
            return dec_out
        B, N, D = dec_out.shape
        pe = self.enc_embedding
        pad = pe.padding_patch_layer.padding
        pad_r = pad[-1] if isinstance(pad, tuple) and len(pad) >= 2 else 0
        Lp = seq_len + pad_r
        if Lp < pe.patch_len:
            return dec_out
        n_patches = (Lp - pe.patch_len) // pe.stride + 1
        if n_patches != N:
            return dec_out

        device = dec_out.device
        centers = patch_center_timestep_indices(
            seq_len, n_patches, pe.patch_len, pe.stride, pad_r, device
        )
        hour_l, dow_l = marks_to_hour_dow_indices(x_mark_enc, self.data_freq)
        centers_e = centers.unsqueeze(0).expand(batch_size, -1)
        hour_bn = hour_l.gather(1, centers_e)
        dow_bn = dow_l.gather(1, centers_e)
        hour_flat = hour_bn.unsqueeze(1).expand(batch_size, n_vars, n_patches).reshape(-1, n_patches)
        dow_flat = dow_bn.unsqueeze(1).expand(batch_size, n_vars, n_patches).reshape(-1, n_patches)

        hvec = self.hour_embed(hour_flat)
        dvec = self.day_embed(dow_flat)
        periodic = hvec + dvec
        if self.periodic_to_hidden is not None:
            periodic = self.periodic_to_hidden(periodic)
        gamma = self.periodic_gamma.to(dtype=dec_out.dtype)
        return dec_out + gamma * periodic

    def forecast(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        return_vib_kl: bool = False,
        return_mi_feats: bool = False,
    ):
        B, L, M = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        dec_in, n_vars = self.enc_embedding(x_enc)
        dec_in = self._apply_spectral_residual(dec_in)

        cap_mi = bool(return_mi_feats) and self.use_mi_preservation
        dec_out, attns, first_layer_feat, feat_early, feat_late = (
            self._decoder_forward_with_optional_recycle(dec_in, capture_mi_feats=cap_mi)
        )
        if self.cross_attn_bridge is not None and first_layer_feat is not None:
            dec_out = self.cross_attn_bridge(dec_out, first_layer_feat)
        if self.sig_gate is not None and first_layer_feat is not None:
            dec_out = self.sig_gate(dec_out, first_layer_feat)
            self._debug_print_sig_gate_if_enabled()
        dec_out = self._apply_periodic_residual(dec_out, x_mark_enc, B, L, n_vars)

        vib_kl = None
        if self.use_timer_vib:
            dec_out, vib_kl = self._vib_transform(dec_out, return_vib_kl)

        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)

        dec_out = dec_out * stdev + means
        # MI aux loss must be computed inside this forward so DDP sees all mi_proj use in one wrapped call.
        mi_loss_out = dec_out.new_zeros(())
        if cap_mi and feat_early is not None and feat_late is not None and self.mi_proj is not None:
            if self.training:
                mi_loss_out = self._mi_preservation_aux_loss(feat_early, feat_late)
            else:
                mi_loss_out = feat_early.new_zeros(())

        if return_mi_feats and self.use_mi_preservation:
            if self.output_attention:
                if return_vib_kl and self.use_timer_vib and vib_kl is not None:
                    return dec_out, attns, vib_kl, feat_early, feat_late, mi_loss_out
                return dec_out, attns, feat_early, feat_late, mi_loss_out
            if return_vib_kl and self.use_timer_vib and vib_kl is not None:
                return dec_out, vib_kl, feat_early, feat_late, mi_loss_out
            return dec_out, feat_early, feat_late, mi_loss_out
        if self.output_attention:
            if return_vib_kl and self.use_timer_vib and vib_kl is not None:
                return dec_out, attns, vib_kl
            return dec_out, attns
        if return_vib_kl and self.use_timer_vib and vib_kl is not None:
            return dec_out, vib_kl
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        dec_in, n_vars = self.enc_embedding(x_enc)
        dec_in = self._apply_spectral_residual(dec_in)

        dec_out, attns = self.decoder(dec_in)
        dec_out = self._apply_periodic_residual(dec_out, x_mark_enc, B, L, n_vars)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)

        dec_out = dec_out * stdev + means
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        dec_in, n_vars = self.enc_embedding(x_enc)
        dec_in = self._apply_spectral_residual(dec_in)
        x_mark_enc = None

        dec_out, attns = self.decoder(dec_in)
        dec_out = self._apply_periodic_residual(dec_out, x_mark_enc, B, L, n_vars)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)

        dec_out = dec_out * stdev + means
        return dec_out

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        return_vib_kl: bool = False,
        return_mi_feats: bool = False,
    ):
        if self.task_name == 'forecast':
            return self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                return_vib_kl=return_vib_kl,
                return_mi_feats=return_mi_feats,
            )
        if self.task_name == 'imputation':
            return self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)

        raise NotImplementedError
