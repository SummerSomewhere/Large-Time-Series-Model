import os
import torch
from torch import nn

from models import TimerBackbone


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

    Prototype Prompting Extension:
    - use_prototype: Enable prototype injection mechanism
    - prototype_path: Path to the pre-computed prototype .pt file
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

        self.output_attention = configs.output_attention or getattr(configs, 'use_align_loss', False)
        self.align_loss_mode = getattr(configs, 'align_loss_mode', 'kl')

        # Prototype prompting parameters
        self.use_prototype = getattr(configs, 'use_prototype', False)
        self.prototype_path = getattr(configs, 'prototype_path', None)
        self.prototype_scale = getattr(configs, 'prototype_scale', 1.0)

        # Load prototype vector if enabled
        self.prototype: torch.Tensor | None = None
        if self.use_prototype and self.prototype_path:
            self._load_prototype()

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
                if self.ckpt_path.endswith('.pth'):
                    sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
                    sd = {k.replace("model.", ""): v for k, v in sd.items()}
                elif self.ckpt_path.endswith('.ckpt'):
                    sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
                    for prefix in ["module.", "model.", "backbone."]:
                        sd = {k.replace(prefix, ""): v for k, v in sd.items()}
                    sd = {k.replace("enc_embedding.", "patch_embedding."): v for k, v in sd.items()}
                else:
                    raise NotImplementedError
                self.backbone.load_state_dict(sd, strict=True)

    def _load_prototype(self) -> None:
        """加载预计算的原型向量 h_proto [1, 1, D]"""
        if self.prototype_path and os.path.exists(self.prototype_path):
            checkpoint = torch.load(self.prototype_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and 'h_proto' in checkpoint:
                h_proto = checkpoint['h_proto']
            else:
                h_proto = checkpoint

            # 确保原型维度为 [1, 1, D]
            # 处理各种可能的错误维度格式：[*, *, D], [*, D], [D], [1, 1, D], [1, 1, 2, D] 等
            if h_proto.dim() > 3:
                # 展平多余维度
                while h_proto.dim() > 3:
                    last_dim = h_proto.shape[-1]
                    second_last = h_proto.shape[-2]
                    h_proto = h_proto.reshape(-1, second_last, last_dim)
            elif h_proto.dim() == 2:
                # [1, D] -> [1, 1, D]
                h_proto = h_proto.unsqueeze(1)
            elif h_proto.dim() == 1:
                # [D] -> [1, 1, D]
                h_proto = h_proto.unsqueeze(0).unsqueeze(0)

            # 确保前两个维度是 [1, 1]
            if h_proto.shape[0] != 1:
                h_proto = h_proto[:1, :, :]
            if h_proto.shape[1] != 1:
                h_proto = h_proto[:, :1, :]

            self.prototype = h_proto
            print(f"[Prototype] Loaded prototype from: {self.prototype_path}, shape: {self.prototype.shape}")
        else:
            print(f"[Prototype] WARNING: Prototype path not found: {self.prototype_path}")
            self.prototype = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                output_hidden_states: bool = False, layer_guide: torch.Tensor = None,
                output_attention_override=None):
        """
        layer_guide: [B*M, num_layers, D] — 第 0 层不需要，引导从第 1 层开始注入。
        即第 i 层（i>=1）注入 layer_guide[:, i-1, :]。
        """
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)  # [B * M, N, D]

        # ── Prototype Injection (Prefix Prompting) ───────────────────────────────
        n_guide_tokens = 0
        if self.use_prototype and self.prototype is not None:
            prototype_expanded = self.prototype.expand(B * n_vars, -1, -1).to(dec_in.device)
            dec_in = torch.cat([prototype_expanded, dec_in], dim=1)  # [B*M, N+1, D]
            n_guide_tokens += 1

        # Transformer Blocks — only pass layer_guide when it is not None
        decoder_kwargs = dict(
            has_prototype=(n_guide_tokens > 0) or (layer_guide is not None),
            output_hidden_states=output_hidden_states,
        )
        if layer_guide is not None:
            decoder_kwargs["layer_guide"] = layer_guide
        decoder_out = self.decoder(dec_in, **decoder_kwargs)

        use_logit_mode = getattr(self, 'align_loss_mode', 'kl') == 'logit_mse'

        if output_hidden_states and len(decoder_out) == 4:
            dec_out, attns, logits_list, hidden_states = decoder_out
        elif len(decoder_out) == 3:
            dec_out, attns, logits_list = decoder_out
            if not use_logit_mode:
                logits_list = None
            hidden_states = None
        else:
            dec_out, attns = decoder_out
            logits_list = None
            hidden_states = None

        # Remove injected guide token(s) from decoder output only
        if n_guide_tokens > 0:
            dec_out = dec_out[:, n_guide_tokens:, :]

        dec_out = self.proj(dec_out)  # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

        # De-Normalization
        dec_out = dec_out * stdev + means
        want_attn = output_attention_override if output_attention_override is not None else self.output_attention
        if want_attn:
            if use_logit_mode:
                if output_hidden_states:
                    return dec_out, attns, logits_list, hidden_states
                return dec_out, attns, logits_list
            if output_hidden_states:
                return dec_out, attns, hidden_states
            return dec_out, attns
        if output_hidden_states:
            return dec_out, hidden_states
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

        # Transformer Blocks
        dec_out, attns, _ = self.decoder(dec_in) # [B * M, N, D]
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

        # Transformer Blocks
        dec_out, attns, _ = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                output_hidden_states: bool = False, layer_guide: torch.Tensor = None,
                output_attention_override=None):
        print(f"[DEBUG Timer.forward] task_name={self.task_name!r}")
        if self.task_name == 'forecast':
            result = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                output_hidden_states=output_hidden_states,
                layer_guide=layer_guide,
                output_attention_override=output_attention_override
            )
            return result
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, T, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, T, D]

        raise NotImplementedError

