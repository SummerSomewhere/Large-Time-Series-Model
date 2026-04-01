from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralResidualBranch(nn.Module):
    """
    Spectral Residual saliency along the patch axis: log-amplitude FFT, smooth, subtract,
    reconstruct complex spectrum with original phase and residual-shaped magnitude, IFFT to time
    (patch) domain. Injected as x + sigmoid(Linear(S)) * x (Saliency Prompt).
    """

    def __init__(self, d_model: int, smooth_kernel: int = 3, log_amp_eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.smooth_kernel = int(smooth_kernel)
        self.log_amp_eps = float(log_amp_eps)
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, D]
        B, P, D = x.shape
        if P < 2:
            s = self.proj(x)
            return x + torch.sigmoid(s) * x

        X = torch.fft.rfft(x, dim=1, norm="ortho")
        amp = X.abs().clamp_min(self.log_amp_eps)
        L = torch.log(amp)
        Freq = L.shape[1]
        k = self.smooth_kernel
        pad = k // 2
        L_perm = L.permute(0, 2, 1)
        L_in = L_perm.reshape(B * D, 1, Freq)
        L_s = F.avg_pool1d(L_in, kernel_size=k, stride=1, padding=pad)
        if L_s.shape[-1] != Freq:
            L_s = F.interpolate(L_s, size=Freq, mode="linear", align_corners=False)
        L_s = L_s.view(B, D, Freq).permute(0, 2, 1)
        R = L - L_s
        R = R.clamp(-8.0, 8.0)
        phase = torch.angle(X)
        X_new = torch.exp(R) * torch.exp(1j * phase)
        S = torch.fft.irfft(X_new, n=P, dim=1, norm="ortho")
        S = self.proj(S)
        return x + torch.sigmoid(S) * x
