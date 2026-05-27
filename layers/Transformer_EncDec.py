import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn, logits = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, logits


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, has_prototype: bool = False,
                output_hidden_states: bool = False, layer_guide: torch.Tensor = None,
                output_attention_override: bool = False):
        # layer_guide is accepted (but ignored) here; only InjectionEncoder uses it.
        # output_attention_override: if True, force attention return even without FullAttention.output_attention flag.
        # x [B, L, D] or [B, L+1, D] if has_prototype=True
        attns = []
        logits_list = []
        hidden_states = [] if output_hidden_states else None

        # Adjust mask size if prototype was prepended
        if has_prototype and attn_mask is not None:
            # attn_mask needs to be adjusted for the longer sequence
            from utils.masking import TriangularCausalMask
            B = x.shape[0]
            L = x.shape[1]
            attn_mask = TriangularCausalMask(B, L, device=x.device)

        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn, logits = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
                if logits is not None:
                    logits_list.append(logits)
                if output_hidden_states:
                    hidden_states.append(x)
            x, attn, logits = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
            if logits is not None:
                logits_list.append(logits)
            if output_hidden_states:
                hidden_states.append(x)
        else:
            for attn_layer in self.attn_layers:
                x, attn, logits = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
                if logits is not None:
                    logits_list.append(logits)
                if output_hidden_states:
                    hidden_states.append(x)

        if self.norm is not None:
            x = self.norm(x)
            if output_hidden_states:
                hidden_states.append(x)

        if output_hidden_states:
            return x, attns, logits_list, hidden_states
        return x, attns, logits_list


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        sa_out, _, _ = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )
        x = x + self.dropout(sa_out)
        x = self.norm1(x)

        ca_out, ca_attn, ca_logits = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(ca_out)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), ca_attn, ca_logits


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None,
                output_hidden_states: bool = False, has_prototype: bool = False,
                layer_guide: torch.Tensor = None):
        if cross is None:
            cross = x
        attns = []
        logits_list = []
        hidden_states = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            layer_out = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
            if isinstance(layer_out, tuple) and len(layer_out) >= 3:
                x = layer_out[0]
                attns.append(layer_out[1])
                logits_list.append(layer_out[2])
            else:
                x = layer_out
            if output_hidden_states:
                hidden_states.append(x)

        if self.norm is not None:
            x = self.norm(x)
            if output_hidden_states:
                hidden_states.append(x)

        if self.projection is not None:
            x = self.projection(x)

        if output_hidden_states:
            if logits_list and any(l is not None for l in logits_list):
                return x, attns, logits_list, hidden_states
            return x, attns, None, hidden_states
        if logits_list and any(l is not None for l in logits_list):
            return x, attns, logits_list
        return x, attns
