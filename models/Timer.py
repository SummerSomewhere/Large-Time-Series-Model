import torch
from torch import nn

from models import TimerBackbone
from models.checkpoint_utils import load_backbone_state_dict


class Model(nn.Module):
    """
    Timer: Generative Pre-trained Transformers Are Large Time Series Models (ICML 2024)

    Paper: https://arxiv.org/abs/2402.02368
    
    GitHub: https://github.com/thuml/Large-Time-Series-Model
    
    Citation: @inproceedings{liutimer,
        title={Timer: Generative Pre-trained Transformers Are Large Time Series Models},
        author={Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        booktitle={Forty-first International Conference on Machine Learning}
    }
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
        self.resonance_last_layer = bool(getattr(configs, "resonance_last_layer", 0))
        self.resonance_dt_hours = float(getattr(configs, "resonance_dt_hours", 1.0))

        self.backbone = TimerBackbone.Model(configs)
        # Decoder
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj
        self.enc_embedding = self.backbone.patch_embedding


        if self.ckpt_path != '':
            if self.ckpt_path == 'random':
                print('loading model randomly')
            else:
                print('loading model: ', self.ckpt_path)
                # Pretrained Timer ckpts have no last-layer resonance params; strict=False keeps finetune working.
                strict_load = not self.resonance_last_layer
                if not strict_load:
                    print(
                        'Note: strict=False (resonance_last_layer=1): ω/λ/φ and mask init from module; '
                        'other weights loaded from checkpoint.'
                    )
                if self.ckpt_path.endswith('.pth'):
                    sd = load_backbone_state_dict(self.ckpt_path, from_lightning_ckpt=False)
                    self.backbone.load_state_dict(sd, strict=strict_load)
                elif self.ckpt_path.endswith('.ckpt'):
                    sd = load_backbone_state_dict(self.ckpt_path, from_lightning_ckpt=True)
                    self.backbone.load_state_dict(sd, strict=strict_load)

                else:
                    raise NotImplementedError

    def _patch_center_physical_timestamps(self, batch_size, seq_len, n_vars, dtype, device):
        """
        Physical time at each patch center: T_p = (p*stride + (patch_len-1)/2) * dt_hours.
        Monotonic in p; matches cos(2πω|T_i-T_j|+φ) on the causal lower triangle when time increases with index.
        Returns [B * n_vars, N] or None if resonance is off.
        """
        if not self.resonance_last_layer:
            return None
        pe = self.enc_embedding
        patch_len = float(pe.patch_len)
        stride = float(pe.stride)
        pad = pe.padding_patch_layer.padding
        pad_r = pad[-1] if isinstance(pad, tuple) and len(pad) >= 2 else 0
        L = seq_len
        L_pad = L + pad_r
        if L_pad < pe.patch_len:
            return None
        N = (L_pad - pe.patch_len) // pe.stride + 1
        dt = self.resonance_dt_hours
        p_idx = torch.arange(N, device=device, dtype=torch.float64)
        centers = (p_idx * stride + 0.5 * (patch_len - 1.0)) * dt
        centers = centers.to(dtype=dtype).unsqueeze(0).expand(batch_size, n_vars, N)
        centers = centers.reshape(batch_size * n_vars, N)
        return centers

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        phys_t = self._patch_center_physical_timestamps(
            B, L, n_vars, dec_in.dtype, dec_in.device
        )

        # Transformer Blocks (diurnal on inner layers; last layer optional resonance bias)
        dec_out, attns = self.decoder(dec_in, physical_timestamps=phys_t) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        phys_t = self._patch_center_physical_timestamps(
            B, L, n_vars, dec_in.dtype, dec_in.device
        )

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in, physical_timestamps=phys_t) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        phys_t = self._patch_center_physical_timestamps(
            B, L, n_vars, dec_in.dtype, dec_in.device
        )

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in, physical_timestamps=phys_t) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, T, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, T, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, T, D]

        raise NotImplementedError

