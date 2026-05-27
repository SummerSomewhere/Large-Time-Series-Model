"""
HSIC / MI Estimation Utilities

Implements HSIC (Hilbert-Schmidt Independence Criterion) as a non-parametric
proxy for mutual information, using Gaussian RBF kernels with median bandwidth.
"""

from __future__ import annotations

import torch
import numpy as np


def median_sq_bandwidth(X: torch.Tensor) -> float:
    """
    Median heuristic: median of upper-triangle pairwise squared Euclidean distances.

    X: [n, D]
    Returns: scalar sigma_sq (float)
    """
    n = X.shape[0]
    if n < 2:
        return 1.0
    d = torch.cdist(X, X, p=2.0)
    triu = torch.triu_indices(n, n, offset=1, device=X.device)
    sq = d[triu[0], triu[1]] ** 2
    med = torch.median(sq)
    return float(med.clamp(min=1e-12).item())


def hsic_with_separate_sigmas(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x_sq: float,
    sigma_y_sq: float,
) -> float:
    """
    Biased HSIC with separate Gaussian RBF bandwidths for X and Y.

    HSIC(X, Y) = (1/(n-1)^2) * tr(H K H @ H L H)
    where K_ij = exp(-||X[i]-X[j]||^2 / (2*sigma_x_sq))
          L_ij = exp(-||Y[i]-Y[j]||^2 / (2*sigma_y_sq))
          H    = I - 1/n * 11^T  (centering matrix)

    Args:
        X:         [n, d_x] tensor
        Y:         [n, d_y] tensor
        sigma_x_sq: squared bandwidth for X kernel (float)
        sigma_y_sq: squared bandwidth for Y kernel (float)

    Returns:
        HSIC score (float)
    """
    n = X.shape[0]
    if n < 2:
        return 0.0

    d2_x = torch.cdist(X, X, p=2.0) ** 2
    d2_y = torch.cdist(Y, Y, p=2.0) ** 2

    K = torch.exp(-d2_x / (2.0 * sigma_x_sq))
    L = torch.exp(-d2_y / (2.0 * sigma_y_sq))

    H = torch.eye(n, device=X.device, dtype=X.dtype) - (1.0 / n)
    Kc = H @ K @ H
    Lc = H @ L @ H

    return float((torch.trace(Kc @ Lc) / ((n - 1) ** 2)).item())


def rbf_kernel(X: torch.Tensor, sigma_sq: float) -> torch.Tensor:
    """RBF kernel matrix."""
    d2 = torch.cdist(X, X, p=2.0) ** 2
    return torch.exp(-d2 / (2.0 * sigma_sq))


def zscore_cols(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Column-wise z-score normalization."""
    m = t.mean(dim=0, keepdim=True)
    s = t.std(dim=0, keepdim=True).clamp(min=eps)
    return (t - m) / s


def compute_stable_rank(w_q: torch.Tensor, w_k: torch.Tensor, d_head: int) -> float:
    """
    Compute the stable rank of the projected representation M = W_Q @ W_K^T / sqrt(d_head).

    Stable rank is defined as:
        sr(M) = ||M||_F^2 / ||M||_2^2 = sum(s_i^2) / max(s_i)^2

    The division by sqrt(d_head) normalises the Q-K product so that the stable
    rank is comparable across heads with different dimensions and across layers
    with different d_model / n_heads configurations.

    This is a data-independent proxy for the "effective dimensionality" of the
    query-key interaction space. It measures how spread-out the rank is relative
    to the dominant singular value, independent of the ambient dimension.

    In d-dimensional space, average pairwise distances scale as ~sqrt(d).
    By normalising bandwidth via sqrt(sr_l / sr_ref), we give each layer a
    sigma proportional to the layer's own effective dimensionality, correcting
    for the "space inflation / deflation" bias across transformer layers.

    Args:
        w_q:    [D, D] Query projection weight matrix
        w_k:    [D, D] Key projection weight matrix
        d_head: Head dimension d_model // n_heads

    Returns:
        Stable rank (float), clipped to [1.0, d_head] for numerical safety.
    """
    # M = W_Q @ W_K^T / sqrt(d_head)  →  [D, D]
    m = w_q @ w_k.T / np.sqrt(d_head)
    # torch.svd is available in all PyTorch versions
    _, s, _ = torch.svd(m)
    s_sq = s**2
    fro_sq = s_sq.sum()
    spec_norm_sq = s[0] ** 2
    if spec_norm_sq < 1e-12:
        return 1.0
    sr = fro_sq / spec_norm_sq
    return float(sr.clamp(min=1.0).item())
