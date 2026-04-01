import os

import torch
import torch.nn as nn

from layers.Embed import GeometricHPEPatchEmbedding, PatchEmbedding, parse_geometric_hpe_periods_hours
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


def _parse_mi_att_bias_patches(configs) -> tuple[int, ...]:
    raw = str(getattr(configs, "mi_att_bias_patches", "") or "").strip()
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

        use_geometric_hpe = bool(int(getattr(configs, "geometric_hpe", 0)))
        L_seq = int(getattr(configs, "seq_len", 0))
        Lp = L_seq + int(padding)
        pl, st = self.patch_len, self.stride
        n_patches_emb = (Lp - pl) // st + 1 if Lp >= pl else 0

        if use_geometric_hpe and n_patches_emb > 0:
            periods = parse_geometric_hpe_periods_hours(
                str(getattr(configs, "geometric_hpe_periods", "") or "")
            )
            curv_ps = float(getattr(configs, "geometric_hpe_curv_phase_scale", 1.0))
            self.patch_embedding = GeometricHPEPatchEmbedding(
                self.d_model,
                self.patch_len,
                self.stride,
                padding,
                self.dropout,
                num_patches=n_patches_emb,
                periods_hours=periods,
                curv_phase_scale_init=curv_ps,
            )
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    f"Geometric-HPE patch embedding: num_patches={n_patches_emb}, "
                    f"periods_hours={periods}, curv_phase_scale_init={curv_ps}",
                    flush=True,
                )
        else:
            if use_geometric_hpe and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    "geometric_hpe=1 but num_patches<=0; falling back to standard PatchEmbedding",
                    flush=True,
                )
            self.patch_embedding = PatchEmbedding(
                self.d_model, self.patch_len, self.stride, padding, self.dropout
            )

        use_mi_bias = bool(int(getattr(configs, "mi_att_bias", 0)))
        n_mi_patches = 0
        if use_mi_bias:
            L = int(getattr(configs, "seq_len", 0))
            Lp = L + int(padding)
            pl, st = self.patch_embedding.patch_len, self.patch_embedding.stride
            if Lp >= pl:
                n_mi_patches = (Lp - pl) // st + 1
        mi_idx = _parse_mi_att_bias_patches(configs)
        mi_init = float(getattr(configs, "mi_att_bias_init", 0.5))
        _npc = n_mi_patches if (use_mi_bias and n_mi_patches > 0) else None
        _nh = configs.n_heads if _npc else None

        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=True,
                            num_patches=_npc,
                            n_heads=_nh,
                            mi_bias_patch_indices=mi_idx if _npc else None,
                            mi_bias_init_val=mi_init,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        if _npc and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                f"MI-guided attention bias: num_patches={_npc}, n_heads={configs.n_heads}, "
                f"key_boost_indices={mi_idx}, init={mi_init} (per encoder layer)",
                flush=True,
            )

        self.proj = nn.Linear(self.d_model, configs.patch_len, bias=True)
