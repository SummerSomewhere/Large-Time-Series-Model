import torch
import torch.nn as nn

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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self.decoder(dec_in)
        dec_out = self._apply_periodic_residual(dec_out, x_mark_enc, B, L, n_vars)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)

        dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'imputation':
            return self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)

        raise NotImplementedError
