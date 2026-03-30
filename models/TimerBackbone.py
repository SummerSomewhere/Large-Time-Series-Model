import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    FullAttentionLastLayerResonance,
)
from layers.Transformer_EncDec import Encoder, EncoderLayer

# Per-layer head indices for cos(2*pi*|i-j|/period) attention bias (period in patch tokens); layers not listed get no bias.
_TIMER_DIURNAL_HEADS_BY_LAYER = {
    0: frozenset({2, 4, 5}),
    1: frozenset({0, 2, 7}),
    2: frozenset({2, 5}),
    3: frozenset({4}),
}


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, self.dropout)

        diurnal_on = bool(getattr(configs, "diurnal_attn_bias", 0))
        diurnal_lambda = float(getattr(configs, "diurnal_lambda", 1.0))
        diurnal_period = float(getattr(configs, "diurnal_period", 24.0))

        def _diurnal_heads(layer_idx):
            if not diurnal_on:
                return frozenset()
            return _TIMER_DIURNAL_HEADS_BY_LAYER.get(layer_idx, frozenset())

        diurnal_heads_per_layer = [_diurnal_heads(l) for l in range(configs.e_layers)]

        resonance_on = bool(getattr(configs, "resonance_last_layer", 0))
        n_heads = configs.n_heads
        last_idx = configs.e_layers - 1

        def _parse_resonance_head_mask():
            raw = getattr(configs, "resonance_head_mask", None)
            if raw is None or raw == "":
                return None
            if isinstance(raw, (list, tuple)):
                return [bool(int(x)) for x in raw]
            s = str(raw).replace(",", " ").split()
            return [bool(int(x)) for x in s]

        rmask = _parse_resonance_head_mask()
        if rmask is not None and len(rmask) != n_heads:
            raise ValueError(
                f"resonance_head_mask length {len(rmask)} must equal n_heads {n_heads}"
            )

        omega_cfg = getattr(configs, "resonance_omega_init", None)
        if omega_cfg is not None:
            omega_init = float(omega_cfg)
        else:
            period_h = getattr(configs, "resonance_period_hours", None)
            if period_h is not None and float(period_h) > 0:
                omega_init = 1.0 / float(period_h)
            else:
                omega_init = None
        lambda_init = float(getattr(configs, "resonance_lambda_init", 0.1))
        phi_init = float(getattr(configs, "resonance_phi_init", 0.0))

        def _make_attention(layer_idx: int):
            if resonance_on and layer_idx == last_idx:
                return FullAttentionLastLayerResonance(
                    True,
                    configs.factor,
                    attention_dropout=configs.dropout,
                    output_attention=True,
                    n_heads=n_heads,
                    resonance_head_mask=rmask,
                    omega_init=omega_init,
                    lambda_init=lambda_init,
                    phi_init=phi_init,
                )
            diurnal_set = diurnal_heads_per_layer[layer_idx]
            return FullAttention(
                True,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=True,
                diurnal_heads=diurnal_set,
                diurnal_lambda=diurnal_lambda if diurnal_set else 0.0,
                diurnal_period=diurnal_period,
            )

        # Decoder-only Transformer: Refer to issue: https://github.com/thuml/Large-Time-Series-Model/issues/23
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        _make_attention(l),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Prediction Head
        self.proj = nn.Linear(self.d_model, configs.patch_len, bias=True)