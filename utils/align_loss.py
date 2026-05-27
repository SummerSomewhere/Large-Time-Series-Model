"""
Hierarchical Alignment Loss: Aligns per-layer attention distributions with
pre-computed HSIC/MI prior distributions.

Mathematical Framework
---------------------
For model layer l:

  1. Construct layer-wise MI target distribution:
       P_MI^{(l)}(j) = softmax(MI_j^{(l)} / tau)     [T patches, tau temperature]

  2. Extract layer-wise attention distribution:
       P_attn^{(l)}(j) = (1/(H*T)) * sum_{h,i} A_{h,i,j}^{(l)}   [marginal over heads & query positions]

  3. KL divergence alignment loss:
       L_align^{(l)} = KL(P_MI^{(l)} || P_attn^{(l)})
                     = sum_j P_MI^{(l)}(j) * log[P_MI^{(l)}(j) / P_attn^{(l)}(j)]

  4. Total loss:
       L_total = L_task + beta * sum_{l in L} L_align^{(l)}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_align_loss_per_layer(
    attn_matrix: torch.Tensor,
    mi_scores: torch.Tensor,
    tau: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute KL(P_MI || P_attn) for a single layer.

    Args:
        attn_matrix: Attention matrix of shape [B, H, T, T].
                     Usually the self-attention of a decoder/encoder layer.
        mi_scores:   Pre-computed MI scores for T patches, shape [T] or [B, T].
                     If [T], broadcast across batch dimension.
        tau:         Temperature for softmax on MI scores.
        eps:         Small constant for numerical stability.

    Returns:
        KL divergence (scalar tensor), mean over batch dimension.
    """
    B, H, T, _ = attn_matrix.shape

    # P_attn^{(l)}(j) = marginal over heads & query positions
    # mean(dim=2) averages over query dimension T (i=1..T)
    # mean(dim=1) averages over heads H
    # Result: [B, T]
    P_attn = attn_matrix.mean(dim=2).mean(dim=1)  # [B, T]
    P_attn = P_attn + eps
    P_attn = P_attn / P_attn.sum(dim=-1, keepdim=True)

    # P_MI^{(l)}(j) = softmax(MI_j / tau)
    if mi_scores.dim() == 1:
        mi_scores = mi_scores.unsqueeze(0).expand(B, -1)  # [B, T]
    P_mi = F.softmax(mi_scores / tau, dim=-1)  # [B, T]
    P_mi = P_mi + eps
    P_mi = P_mi / P_mi.sum(dim=-1, keepdim=True)

    # KL(P_MI || P_attn) = sum_j P_MI * log(P_MI / P_attn)
    kl = P_mi * (torch.log(P_mi) - torch.log(P_attn))
    kl_loss = kl.sum(dim=-1)  # [B]
    return kl_loss.mean()


def compute_align_loss_multi_layer(
    attn_list: list[torch.Tensor],
    mi_curves: dict[int, list[float]],
    layers: list[int],
    tau: float = 0.1,
    device: torch.device = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute total alignment loss across multiple layers.

    Args:
        attn_list:   List of attention matrices, one per layer.
                     Each element: [B, H, T, T].
        mi_curves:   Dict mapping layer_index -> list of T MI scores.
                     layer_index must be in [0, len(attn_list)-1].
        layers:      List of layer indices to compute alignment for.
                     E.g., [0, 1, 2, 3, 4, 5, 6, 7].
        tau:         Temperature for softmax on MI scores.
        device:      Target device for MI score tensors.
        eps:         Numerical stability constant.

    Returns:
        Total alignment loss (scalar tensor), mean over layers.
    """
    total_loss = torch.tensor(0.0, device=device or torch.device("cpu"))
    n_layers = 0

    for l in layers:
        if l >= len(attn_list):
            continue
        attn = attn_list[l]  # [B, H, T, T]
        if attn is None:
            continue

        mi_scores = torch.tensor(
            mi_curves[l], dtype=torch.float32, device=device or attn.device
        )  # [T]

        layer_loss = compute_align_loss_per_layer(attn, mi_scores, tau=tau, eps=eps)
        total_loss = total_loss + layer_loss
        n_layers += 1

    if n_layers == 0:
        return total_loss

    return total_loss / n_layers


def load_mi_curves_from_json(json_path: str) -> dict[int, list[float]]:
    """
    Load MI curves from a JSON file produced by the HSIC analysis pipeline.

    Expected JSON structure:
    {
        "layers": {
            "0": {"hsic_curve": [score_0, score_1, ..., score_T-1]},
            "1": {"hsic_curve": [...]},
            ...
        }
    }

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dict mapping layer_index (int) -> list of MI scores per patch.
    """
    import json

    with open(json_path, "r") as f:
        data = json.load(f)

    mi_curves = {}
    layers_data = data.get("layers", data)
    for layer_str, layer_info in layers_data.items():
        layer_idx = int(layer_str)
        mi_curves[layer_idx] = layer_info["hsic_curve"]

    return mi_curves
