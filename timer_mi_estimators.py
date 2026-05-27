#!/usr/bin/env python3
"""
Timer 版本的 MI (HSIC) 估计器

模仿 MI-Peaks/src/mi_estimators.py，为时间序列模型 Timer 定制。

核心改进：
1. 使用 batch 维度计算 HSIC（需要 B >= 4）
2. 支持 Gaussian RBF 核
3. 使用中位数带宽估计
"""

import torch
import numpy as np


def distmat(X: torch.Tensor) -> torch.Tensor:
    """
    计算距离矩阵 D_ij = ||x_i - x_j||^2
    
    Args:
        X: [n, d] 张量，n 个样本，d 维特征
    
    Returns:
        D: [n, n] 距离矩阵
    """
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j
    r = torch.sum(X * X, 1)  # [n]
    r = r.view([-1, 1])       # [n, 1]
    a = torch.mm(X, torch.transpose(X, 0, 1))  # [n, n] 内积矩阵
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    return D


def median_bandwidth(X: torch.Tensor) -> torch.Tensor:
    """
    中位数带宽估计：median(||x_i - x_j||^2) for i < j
    
    Args:
        X: [n, d] 张量
    
    Returns:
        sigma_sq: 带宽的平方（用于 RBF 核）
    """
    D = distmat(torch.cat([X, X]))  # [2n, 2n]
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return torch.tensor(med, dtype=X.dtype, device=X.device)


def rbf_kernel(X: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
    """
    RBF (Gaussian) 核矩阵：K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    
    Args:
        X: [n, d] 张量
        sigma_sq: 带宽的平方
    
    Returns:
        K: [n, n] 核矩阵
    """
    d2 = torch.cdist(X, X, p=2.0) ** 2  # [n, n] 欧氏距离平方
    return torch.exp(-d2 / (2.0 * sigma_sq))


def centering_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    中心化矩阵：H = I - 1/n * 1 * 1^T
    
    使得 Kc = H @ K @ H，去均值化
    """
    H = torch.eye(n, device=device, dtype=dtype) - (1.0 / n)
    return H


def hsic_unbiased(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    无偏 HSIC 估计量（作为依赖性/MI 的代理指标）
    
    使用 Gaussian RBF 核，中位数带宽。
    公式：HSIC = tr(H @ K @ H @ H @ L @ H) / (n-1)^2
    
    Args:
        X: [n, d_x] 第一个变量，n 个样本
        Y: [n, d_y] 第二个变量，n 个样本
    
    Returns:
        HSIC 值（标量）
    """
    n = X.shape[0]
    if n < 4:
        return torch.tensor(float("nan"), device=X.device, dtype=X.dtype)
    
    # 计算两个变量的带宽
    sigma_x_sq = median_bandwidth(X)
    sigma_y_sq = median_bandwidth(Y)
    
    # RBF 核矩阵
    K = rbf_kernel(X, sigma_x_sq)  # [n, n]
    L = rbf_kernel(Y, sigma_y_sq)  # [n, n]
    
    # 中心化
    H = centering_matrix(n, X.device, X.dtype)
    Kc = H @ K @ H  # [n, n]
    Lc = H @ L @ H  # [n, n]
    
    # HSIC = trace(Kc @ Lc) / (n-1)^2
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)


def zscore(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    对每个特征维度进行 Z-score 标准化
    
    Args:
        t: [n, d] 张量
        eps: 防止除零的小常数
    
    Returns:
        标准化后的张量
    """
    m = t.mean(dim=0, keepdim=True)
    s = t.std(dim=0, keepdim=True).clamp(min=eps)
    return (t - m) / s


def hsic_normalized_cca(x: torch.Tensor, y: torch.Tensor, sigma: float = 50.0) -> torch.Tensor:
    """
    MI-Peaks 原始版本的 HSIC（CCA 归一化版本）
    
    注意：此版本当 m=1 时会失效，只用于对比参考
    
    Args:
        x: [m, d_x] 输入
        y: [m, d_y] 目标
        sigma: 固定带宽参数
    
    Returns:
        归一化 HSIC 值
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    m = int(x.size()[0])
    Kx = rbf_kernel(x, torch.tensor(2. * sigma * sigma, device=x.device, dtype=x.dtype))
    Ky = rbf_kernel(y, torch.tensor(2. * sigma * sigma, device=y.device, dtype=y.dtype))
    
    H = centering_matrix(m, x.device, x.dtype)
    Kxc = H @ Kx @ H
    Kyc = H @ Ky @ H
    
    epsilon = 1E-5
    K_I = torch.eye(m, device=x.device, dtype=x.dtype)
    
    try:
        Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
        Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
        Rx = Kxc @ Kxc_i
        Ry = Kyc @ Kyc_i
        Pxy = torch.sum(torch.mul(Rx, Ry.t()))
        return Pxy
    except RuntimeError:
        return torch.tensor(0.0)


def estimate_mi_hsic(x: torch.Tensor, y: torch.Tensor, 
                     ktype: str = 'gaussian', sigma: float = 50.) -> torch.Tensor:
    """
    MI 估计的入口函数（兼容 MI-Peaks 接口）
    
    当 batch size >= 4 时使用无偏 HSIC，否则返回 NaN
    """
    if ktype == 'gaussian':
        if x.shape[0] >= 4:
            return hsic_unbiased(x, y)
        else:
            return hsic_normalized_cca(x, y, sigma)
    else:
        return hsic_normalized_cca(x, y, sigma)
