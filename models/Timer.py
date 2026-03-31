from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TimerBackbone
from models.checkpoint_utils import load_backbone_state_dict
from models.periodic_time_utils import (
    marks_to_hour_dow_indices,
    patch_center_timestep_indices,
)


class Model(nn.Module):
    """
    Timer: Generative Pre-trained Transformers Are Large Time Series Models (ICML 2024)

    Optional periodic embedding residual (PEFT): hour/day nn.Embedding + gamma * sum,
    applied after the Transformer stack and before proj (see periodic_embedding_branch).
    """
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
                strict_load = not self.periodic_embedding_branch
                if not strict_load:
                    print(
                        'Note: strict=False (periodic_embedding_branch): backbone keys from ckpt; '
                        'hour_embed/day_embed/gamma init in Timer.'
                    )
                if self.ckpt_path.endswith('.pth'):
                    sd = load_backbone_state_dict(self.ckpt_path, from_lightning_ckpt=False)
                    self.backbone.load_state_dict(sd, strict=strict_load)
                elif self.ckpt_path.endswith('.ckpt'):
                    sd = load_backbone_state_dict(self.ckpt_path, from_lightning_ckpt=True)
                    self.backbone.load_state_dict(sd, strict=strict_load)
                else:
                    raise NotImplementedError

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

        # Variational information bottleneck before patch Linear head (forecast only).
        self.use_timer_vib = bool(int(getattr(configs, "timer_vib", 0)))
        if self.use_timer_vib:
            self.mu_layer = nn.Linear(self.d_model, self.d_model)
            self.logvar_layer = nn.Linear(self.d_model, self.d_model)

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

    def _decoder_forward_with_optional_recycle(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Encoder forward; lightweight recycle on configured layers (e.g. first + last):
        for each round alpha_k: h <- h + alpha_k * (TF_l(h) - h), then F.layer_norm(h, (D,)).
        """
        if not self._recycle_wants_active:
            return self.decoder(x)

        n_enc = len(self.decoder.attn_layers)
        last_i = n_enc - 1
        if self.recycle_encoder_layers:
            layer_set = frozenset(i for i in self.recycle_encoder_layers if 0 <= i < n_enc)
        elif 0 <= self.recycle_encoder_layer < n_enc:
            layer_set = frozenset([self.recycle_encoder_layer])
        else:
            layer_set = frozenset()
        if not layer_set:
            layer_set = frozenset([last_i])

        d_model = x.shape[-1]
        norm_eps = self.decoder.norm.eps if self.decoder.norm is not None else 1e-5
        alphas = self.recycle_round_alphas

        attns = []
        for i, layer in enumerate(self.decoder.attn_layers):
            x, attn = layer(x)
            attns.append(attn)
            if i in layer_set:
                patch_idx = self._recycle_patch_indices_for_layer(x.shape[1], i)
                for pi in patch_idx:
                    for alpha in alphas:
                        seg = x[:, pi : pi + 1, :].contiguous()
                        h = seg[:, 0, :]
                        h_tf, _ = layer(seg)
                        h_final = h + alpha * (h_tf[:, 0, :] - h)
                        x[:, pi, :] = F.layer_norm(h_final, (d_model,), eps=norm_eps)
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x, attns

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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_vib_kl: bool = False):
        B, L, M = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self._decoder_forward_with_optional_recycle(dec_in)
        dec_out = self._apply_periodic_residual(dec_out, x_mark_enc, B, L, n_vars)

        vib_kl = None
        if self.use_timer_vib:
            dec_out, vib_kl = self._vib_transform(dec_out, return_vib_kl)

        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)

        dec_out = dec_out * stdev + means
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
        x_mark_enc = None

        dec_out, attns = self.decoder(dec_in)
        dec_out = self._apply_periodic_residual(dec_out, x_mark_enc, B, L, n_vars)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)

        dec_out = dec_out * stdev + means
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_vib_kl: bool = False):
        if self.task_name == 'forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_vib_kl=return_vib_kl)
        if self.task_name == 'imputation':
            return self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)

        raise NotImplementedError
