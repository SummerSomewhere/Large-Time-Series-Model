"""
Input-Level Dual-Path (Trend + Seasonal) Decomposition for Time Series Forecasting.

Architecture:
    Input X [B*M, N, D]  (after patch embedding)
         │
         ├──► SeriesDecomp (Moving Average, kernel=K)
         │         │
         │    ┌────┴────┐
         │  Seasonal  Trend
         │    │         │
         │  ┌─▼──────┐ ┌▼───────┐
         │  │ CA      │ │Linear/ │
         │  │ Encoder │ │MLP     │
         │  └─┬──────┘ └───┬───┘
         │    │            │
         │    └──────┬─────┘
         │           ▼
         │    Recomposition: α·T_trend + (1-α)·T_seasonal
         │
         ▼
Output fused [B*M, N, D]

Decomposition formula:
    X = Trend + Seasonal
    Trend = MovingAvg(X)
    Seasonal = X - Trend

The decomposition is applied to the raw embedded patches (after PatchEmbedding),
which ensures the attention mechanism sees cleaner signals:
    - Seasonal path: focuses on periodic/oscillatory patterns
    - Trend path: focuses on smooth/long-range trends
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomp(nn.Module):
    """
    FFT-based STL-like seasonal decomposition (fully differentiable).

    Extracts periodic (seasonal) component via FFT band-pass and smoothes
    the remaining low-frequency component as the trend.

    Args:
        period:  Period of the seasonal component. If <= 0, uses adaptive
                 period = N // 4 (empirically good for most forecasting tasks).
        decomp_mode: 'stl' (periodic seasonal + low-freq trend) or
                     'ma' (simple moving average, DLinear style).
    """

    def __init__(self, period: int = 0, decomp_mode: str = 'stl'):
        super().__init__()
        self.period = period      # 0 = adaptive (N // 4)
        self.decomp_mode = decomp_mode

        if self.decomp_mode == 'ma':
            self._ma_ks = 25

    def _ema_trend(self, x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """
        Exponential moving average as a differentiable trend extractor.

        E_t = α·x_t + (1-α)·E_{t-1}, initialized with E_0 = x_0
        """
        alpha_v = x.new_tensor(alpha)
        o = x.new_zeros(x.shape)
        o[..., 0] = x[..., 0]
        for t in range(1, x.shape[-1]):
            o[..., t] = alpha_v * x[..., t] + (1 - alpha_v) * o[..., t-1]
        return o

    def _auto_period(self, residual: torch.Tensor, N: int) -> int:
        """
        Hybrid FFT + ACF period detection.

        Strategy:
            Stage 1 — FFT Periodogram:
                Zero-pad the signal to improve frequency resolution.
                Find the top-K energy peaks in the frequency domain.
                Convert each peak's frequency bin k to a period candidate: P = N / k.
                Filter candidates to only those with P >= min_period.

            Stage 2 — ACF Validation:
                Compute ACF only at the FFT-derived candidate periods (±1 neighbors).
                Pick the candidate with the highest ACF score.

        This avoids the classic ACF failure modes on short/noisy sequences:
            - ACF alone picks lag=2,3 due to neighboring-sample correlation.
            - FFT alone picks harmonics or DC.
            - Hybrid uses FFT to narrow the search, ACF to validate periodicity.

        Args:
            residual: [B, D, N] — deseasonalized signal in patch space
            N:       number of patches

        Returns:
            period:  int (period in patch space)
        """
        min_period = 10          # periods shorter than this are ignored
        top_k = 5                # how many FFT peaks to consider

        if N < 4:
            return max(min_period, N // 2)

        # ── Stage 1: FFT Periodogram — find candidate periods ───────────────────
        # Zero-pad to 4x for better frequency resolution (still fully differentiable)
        pad_len = max(N * 4, 32)
        x_pad = F.pad(residual, (0, pad_len - N))           # [B, D, pad_len]
        X = torch.fft.rfft(x_pad, dim=-1)                    # [B, D, pad_len//2+1]

        # Power spectral density: mean over B and D, ignore DC (bin 0)
        psd = (X.abs() ** 2).mean(dim=1).mean(dim=0)         # [pad_len//2+1]
        psd_nodc = psd[1:]                                    # drop DC bin

        # Guard: need at least 2 non-DC bins
        if psd_nodc.numel() < 2:
            return max(min_period, N // 2)

        # Find top-K peaks in the periodogram (local maxima)
        # Extend with repeated endpoints for safe diff
        padded_psd = torch.cat([psd_nodc[1:2].expand(1), psd_nodc, psd_nodc[-2:-1].expand(1)], dim=0)
        diff_l = padded_psd[2:] - padded_psd[1:-1]
        diff_r = padded_psd[1:-1] - padded_psd[:-2]
        is_peak = (diff_l > 0) & (diff_r > 0)                # [num_bins]

        if is_peak.any():
            peak_vals    = psd_nodc[is_peak]
            peak_indices = torch.where(is_peak)[0] + 1        # +1 because psd_nodc starts at bin 1
        else:
            peak_vals    = psd_nodc
            peak_indices = torch.arange(psd_nodc.size(0), device=psd_nodc.device) + 1

        # Sort by energy descending, take top K
        top_k_cand = min(top_k, peak_vals.numel())
        order = peak_vals.sort(descending=True).indices[:top_k_cand]
        cand_freq_bins = peak_indices[order]                   # frequency bin indices

        # Convert frequency bin → period: P = pad_len / k
        # (k=1 → P=pad_len, k=pad_len/2 → P=2)
        cand_periods_float = pad_len / cand_freq_bins.float()   # [top_k]
        cand_periods_int   = cand_periods_float.round().long()  # nearest-int period

        # Filter: only keep periods in [min_period, N//2] and > 1
        mask = (cand_periods_int >= min_period) & (cand_periods_int <= N // 2) & (cand_periods_int > 1)
        if not mask.any().item():
            # All candidates too short → fall back to N//2 clamped by min_period
            return max(min_period, N // 2)

        cand_periods = cand_periods_int[mask].unique()          # deduplicate
        device = residual.device

        # ── Stage 2: ACF validation at candidate periods ──────────────────────
        def _ema(x, alpha=0.3):
            o = x.new_zeros(x.shape)
            o[..., 0] = x[..., 0]
            a = x.new_tensor(alpha)
            for t in range(1, x.shape[-1]):
                o[..., t] = a * x[..., t] + (1 - a) * o[..., t-1]
            return o

        resid = _ema(residual, alpha=0.3)
        resid_mean = resid.mean(dim=-1, keepdim=True)
        resid_c = resid - resid_mean
        var = (resid_c ** 2).sum(dim=-1).clamp(min=1e-8)

        acf_scores = []
        for P in cand_periods:
            for delta in (-1, 0, 1):          # ±1 neighborhood for robustness
                lag = P + delta
                if lag < 2 or lag >= N:
                    continue
                shifted = resid_c.roll(shifts=-lag, dims=-1)
                shifted[..., -lag:] = 0.0
                ac = (resid_c * shifted).sum(dim=-1).mean() / var.mean()
                acf_scores.append((P.item(), ac.item(), abs(delta)))

        if not acf_scores:
            return max(min_period, N // 2)

        # Sort: primary key = -delta (prefer delta=0), secondary key = -ACF score
        acf_scores.sort(key=lambda x: (x[2], -x[1]))
        best_period, best_acf, _ = acf_scores[0]

        # Weighted refinement: average with neighbors weighted by ACF
        neighbors = [(p, s) for p, s, d in acf_scores if abs(p - best_period) <= 1 and d <= 1]
        if len(neighbors) > 1:
            total_w = sum(s for _, s in neighbors)
            best_period = int(round(sum(p * s for p, s in neighbors) / total_w))

        return max(min_period, best_period)

    def _detect_period_from_raw(self, raw_x: torch.Tensor) -> int:
        """
        Detect period on the raw (pre-patch-embedding) time series.

        This is the recommended approach because:
            - Period is a physical property of the original signal, not the patch space.
            - With 7 patches of length 96, patch-level detection is too coarse.
            - Raw T=672 gives much better frequency resolution.

        Algorithm (Hybrid FFT + ACF):
            Stage 1 — FFT Periodogram: zero-pad to improve freq resolution,
                      find top-K energy peaks, convert to period candidates.
            Stage 2 — ACF Validation: score each candidate's autocorrelation,
                      pick the best with ±1 tolerance.

        Args:
            raw_x: [B, T, M] or [B*T*M] — z-normalized raw time series
        Returns:
            period: int in original time steps
        """
        # Reshape to [B, T, M] — handle both flat B*T*M and full B,T,M
        if raw_x.dim() == 1:
            # single flat vector [T] → treat as B=1, M=1
            x_seq = raw_x.unsqueeze(0).unsqueeze(-1)        # [1, T, 1]
        elif raw_x.dim() == 2:
            x_seq = raw_x.unsqueeze(-1)                     # [B, T, 1] or [T, M]
        else:
            x_seq = raw_x                                   # [B, T, M]

        B_raw, T, M = x_seq.shape
        if T < 4:
            return 0

        # Aggregate across variables (mean over M) for a single ACF/PSD
        x_agg = x_seq.mean(dim=-1)   # [B, T]
        x_mean = x_agg.mean(dim=-1, keepdim=True)
        x_c = x_agg - x_mean
        x_var = (x_c ** 2).sum(dim=-1).clamp(min=1e-8)   # [B]

        min_period = 10
        top_k = 5

        # ── Stage 1: FFT Periodogram ───────────────────────────────────────────
        pad_len = max(T * 4, 64)
        x_pad = F.pad(x_c, (0, pad_len - T))               # [B, pad_len]
        X = torch.fft.rfft(x_pad, dim=-1)                   # [B, pad_len//2+1]
        psd = (X.abs() ** 2).mean(dim=0)                   # [pad_len//2+1]
        psd_nodc = psd[1:]                                  # drop DC

        if psd_nodc.numel() < 2:
            return 0

        # Find local maxima in periodogram
        padded_psd = torch.cat([
            psd_nodc[1:2].expand(1), psd_nodc, psd_nodc[-2:-1].expand(1)
        ], dim=0)
        diff_l = padded_psd[2:] - padded_psd[1:-1]
        diff_r = padded_psd[1:-1] - padded_psd[:-2]
        is_peak = (diff_l > 0) & (diff_r > 0)              # [num_bins]

        if is_peak.any():
            peak_vals    = psd_nodc[is_peak]
            peak_indices = torch.where(is_peak)[0] + 1     # +1: psd_nodc offset
        else:
            peak_vals    = psd_nodc
            peak_indices = torch.arange(psd_nodc.size(0), device=psd.device) + 1

        order = peak_vals.sort(descending=True).indices[:top_k]
        cand_freq_bins = peak_indices[order]                  # [top_k]

        # Convert frequency bin → period: P = pad_len / k
        cand_periods_float = pad_len / cand_freq_bins.float()
        cand_periods_int   = cand_periods_float.round().long()

        mask = (cand_periods_int >= min_period) & (cand_periods_int <= T // 2) & (cand_periods_int > 1)
        if not mask.any().item():
            return 0

        cand_periods = cand_periods_int[mask].unique()

        # ── Stage 2: ACF validation at candidate periods ──────────────────────
        acf_scores = []
        for P in cand_periods:
            for delta in (-1, 0, 1):
                lag = P + delta
                if lag < 2 or lag >= T:
                    continue
                shifted = x_c.roll(shifts=-lag.item(), dims=-1)
                shifted[:, -lag.item():] = 0.0
                ac = (x_c * shifted).sum(dim=-1).mean() / x_var.mean()
                acf_scores.append((P.item(), ac.item(), abs(delta)))

        if not acf_scores:
            return 0

        # Prefer delta=0, then highest ACF
        acf_scores.sort(key=lambda x: (x[2], -x[1]))
        best_period, best_acf, _ = acf_scores[0]

        # Weighted refinement with neighbors
        neighbors = [(p, s) for p, s, d in acf_scores
                     if abs(p - best_period) <= 1 and d <= 1]
        if len(neighbors) > 1:
            total_w = sum(s for _, s in neighbors)
            best_period = int(round(sum(p * s for p, s in neighbors) / total_w))

        return max(min_period, best_period)

    def _fft_stl(self, x: torch.Tensor,
                  raw_period: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Two-stage STL decomposition via FFT (fully differentiable).

        Algorithm:
            1. EMA trend extraction → remove trend from raw signal
            2. FFT on residual → PSD → detect dominant period (or use pre-detected raw_period)
            3. Use period to build Gaussian band-pass mask → seasonal
            4. Remaining low-freq bins → refined trend

        Args:
            x: [B, D, N]
            raw_period: pre-detected period from raw sequence (>0 means use it)
        Returns:
            trend:    [B, D, N]
            seasonal: [B, D, N]
        """
        B, D, N = x.shape

        # ── Stage 1: Detrend via EMA ─────────────────────────────────────────
        trend = self._ema_trend(x, alpha=0.1)              # [B, D, N]
        residual = x - trend                               # [B, D, N] — dirty seasonal

        # ── Stage 2: Period detection ─────────────────────────────────────────
        if self.period > 0:
            period = self.period
        elif raw_period > 0:
            period = raw_period
        else:
            period = self._auto_period(residual, N)
        print(f"[SeriesDecomp] period = {period}  (raw_period={raw_period}, self.period={self.period}, N={N})")

        # ── Stage 3: Full FFT decomposition with detected period ─────────────
        pad_len = 2 * N
        x_pad = F.pad(x, (0, pad_len - N))
        X = torch.fft.rfft(x_pad, dim=-1)                  # [B, D, pad_len//2+1]

        freq_res = 1.0 / pad_len
        seasonal_center_bin = max(1, int(round(1.0 / period / freq_res)))
        seasonal_half_bins = max(1, seasonal_center_bin // 2)

        freq_bins = torch.arange(X.size(-1), device=x.device)
        dist = torch.abs(freq_bins - seasonal_center_bin).float()
        sigma = max(1.0, seasonal_half_bins * 0.5)
        seasonal_mask = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        seasonal_mask[0] = 0.0  # zero out DC

        trend_mask = (freq_bins < (seasonal_center_bin - seasonal_half_bins)).float()

        X_seasonal = X * seasonal_mask.unsqueeze(0).unsqueeze(0)
        X_trend    = X * trend_mask.unsqueeze(0).unsqueeze(0)

        seasonal_pad = torch.fft.irfft(X_seasonal, n=pad_len, dim=-1)
        trend_pad    = torch.fft.irfft(X_trend,    n=pad_len, dim=-1)

        seasonal = seasonal_pad[..., :N]
        trend    = trend_pad[...,    :N]

        # Circular-shift blending to reduce boundary artifacts
        seasonal_shifted = torch.roll(seasonal, shifts=1, dims=-1)
        seasonal = 0.8 * seasonal + 0.2 * seasonal_shifted

        return trend, seasonal

    def _ma_trend(self, x: torch.Tensor) -> torch.Tensor:
        """Simple moving average (DLinear style) as fallback."""
        k = self._ma_ks
        pad = k // 2
        x_t = x.transpose(1, 2)                        # [B, D, N]
        x_pad = F.pad(x_t, (pad, pad), mode='replicate')
        trend = F.avg_pool1d(x_pad, kernel_size=k)     # [B, D, N]
        return trend.transpose(1, 2)                   # [B, N, D]

    def forward(self, x: torch.Tensor,
                raw_x: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] — patch-embedded sequence (note: this is [B, N, D]
                          coming from DualPathSeriesDecomposer where B already = B*M)
            raw_x: [B*T*M or None] — raw time series before patch embedding.
                   If provided, period is detected on this instead of the patch space.
                   Expected shape: [B, T, M] (z-normalized but otherwise raw).
                   When B*M batches are flattened, raw_x is duplicated accordingly.

        Returns:
            trend:    [B, N, D]
            seasonal: [B, N, D]
        """
        # Ensure 3D [B, D, N]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 4:
            x = x.squeeze(1) if x.size(1) == 1 else x.permute(0, 2, 3, 1).reshape(-1, x.size(1), x.size(-1))

        x_3d = x                 # [B, N, D]
        x_t  = x_3d.transpose(1, 2)  # [B, D, N]

        raw_period = 0
        if raw_x is not None and self.decomp_mode != 'ma':
            # raw_x should be [B, T, M]. If it's [B, M, T] (common mistake with patching
            # inputs), detection silently returns 0 — warn here to catch shape bugs.
            if raw_x.dim() == 3 and raw_x.shape[1] < raw_x.shape[2]:
                import warnings
                warnings.warn(
                    f"SeriesDecomp: raw_x may have wrong shape [B,M,T]={tuple(raw_x.shape)} "
                    f"— expected [B,T,M]. Period detection will likely return 0. "
                    f"Fix: transpose(1,2) the input before passing here."
                )
            raw_period = self._detect_period_from_raw(raw_x)

        if self.decomp_mode == 'ma':
            trend = self._ma_trend(x_t).transpose(1, 2)  # [B, N, D]
        else:
            trend, _ = self._fft_stl(x_t, raw_period=raw_period)     # [B, D, N] each
            trend = trend.transpose(1, 2)      # [B, N, D]

        seasonal = x_3d - trend
        return trend, seasonal



class TrendProjector(nn.Module):
    """
    Lightweight projector for the trend component.

    Args:
        d_model: Embedding dimension.
        trend_mode: 'linear' (Linear projection) or 'mlp' (2-layer MLP with SiLU).
                     'mlp' gives more capacity to learn non-trivial trend transformations.
    """

    def __init__(self, d_model: int, trend_mode: str = 'mlp'):
        super().__init__()
        self.trend_mode = trend_mode

        if trend_mode == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.net = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*M, N, D] — trend component from SeriesDecomp

        Returns:
            projected: [B*M, N, D] — transformed trend representation
        """
        return self.net(x)


class Recomposer(nn.Module):
    """
    Learnable recombination of trend and seasonal components.

    Output = α · T_trend + (1 - α) · T_seasonal

    Where α is a learnable scalar constrained to [0, 1] via sigmoid.

    Args:
        d_model: Embedding dimension (for tensor device/dtype).
        alpha_init: Initial α value (default 0.5, equal weight).
        learnable: If True, α is a learnable parameter; else fixed.
    """

    def __init__(self, d_model: int, alpha_init: float = 0.5,
                 learnable: bool = True):
        super().__init__()
        self.d_model = d_model
        if learnable:
            # sigmoid(0) = 0.5; store as unconstrained logit
            self._logit_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self._logit_alpha = None
            self._fixed_alpha = alpha_init

    @property
    def alpha(self) -> float:
        if self._logit_alpha is not None:
            return torch.sigmoid(self._logit_alpha).item()
        return self._fixed_alpha

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trend:    [B*M, N, D] — projected trend
            seasonal: [B*M, N, D] — projected seasonal

        Returns:
            fused: [B*M, N, D] — α·trend + (1-α)·seasonal
        """
        alpha = (torch.sigmoid(self._logit_alpha)
                 if self._logit_alpha is not None
                 else trend.new_tensor(self._fixed_alpha))
        return alpha * trend + (1 - alpha) * seasonal


class DualPathSeriesDecomposer(nn.Module):
    """
    Full input-level dual-path decomposition pipeline.

    Flow:
        X ──► SeriesDecomp ──► [Trend, Seasonal]
                                       │
                                       ├──► TrendProjector ──► T_trend
                                       │
                                       ├──► CA Encoder ──► T_seasonal
                                       │
                                       └─► Recomposer ──► α·T_trend + (1-α)·T_seasonal

    Args:
        decomp: SeriesDecomp instance.
        trend_proj: TrendProjector instance.
        recomposer: Recomposer instance.
        seasonal_encoder_layers: List of EncoderLayer for seasonal path.
        norm_layer: LayerNorm to apply after seasonal encoder.
        kernel_size: Passed to SeriesDecomp if not provided.
    """

    def __init__(self, d_model: int,
                 seasonal_encoder_layers,
                 trend_mode: str = 'mlp',
                 alpha_init: float = 0.5,
                 norm_layer=None,
                 decomp_mode: str = 'stl',
                 period: int = 0):
        super().__init__()
        self.d_model = d_model

        # ── Decomposition (STL or Moving Average) ──────────────────────────────
        self.decomp = SeriesDecomp(
            period=period,
            decomp_mode=decomp_mode,
        )

        # ── Trend Projector ───────────────────────────────────────────────
        self.trend_proj = TrendProjector(d_model=d_model, trend_mode=trend_mode)

        # ── Seasonal Encoder ─────────────────────────────────────────────
        self.seasonal_encoder = nn.ModuleList(seasonal_encoder_layers)

        # ── Norm ──────────────────────────────────────────────────────────
        self.norm = norm_layer

        # ── Recomposer ───────────────────────────────────────────────────
        self.recomposer = Recomposer(d_model=d_model, alpha_init=alpha_init)

    def forward(self, x, attn_mask=None, tau=None, delta=None,
                has_prototype: bool = False, output_hidden_states: bool = False,
                raw_x: torch.Tensor = None):
        """
        Args:
            x: [B, N, D] or [B, N+1, D] if has_prototype=True
            attn_mask: optional attention mask
            has_prototype: whether prototype was prepended to x
            output_hidden_states: if True, return all layer outputs
            raw_x: [B, T, M] — raw time series before patch embedding.
                   If provided, period is detected on this instead of the patch space.

        Returns:
            fused_output, attns, hidden_states
        """
        attns = []
        hidden_states = [] if output_hidden_states else None

        # ── Step 1: Series Decomposition ────────────────────────────────────
        if has_prototype:
            prototype = x[:, :1, :]          # [B, 1, D] — global summary token, DO NOT decompose
            patches   = x[:, 1:, :]         # [B, N, D]
            trend_raw, seasonal_raw = self.decomp(patches, raw_x=raw_x)  # each [B, N, D]
            # Trend: project trend through MLP
            trend_proj_out = self.trend_proj(trend_raw)    # [B, N, D]
            # Seasonal: seasonal component goes through encoder, with prototype prepended
            seasonal_out = torch.cat([prototype, seasonal_raw], dim=1)  # [B, N+1, D]
        else:
            trend_raw, seasonal_raw = self.decomp(x, raw_x=raw_x)       # each [B, N, D]
            trend_proj_out = self.trend_proj(trend_raw)    # [B, N, D]
            seasonal_out  = seasonal_raw                    # [B, N, D]

        # ── Step 2: Seasonal Encoder ────────────────────────────────────────
        for attn_layer in self.seasonal_encoder:
            seasonal_out, attn = attn_layer(
                seasonal_out, attn_mask=attn_mask, tau=tau, delta=delta
            )
            attns.append(attn)
            if output_hidden_states:
                hidden_states.append(seasonal_out)

        if self.norm is not None:
            seasonal_out = self.norm(seasonal_out)

        # ── Step 3: Prepare for fusion ─────────────────────────────────────
        # After the seasonal encoder, the prototype at position 0 carries enriched
        # information (attended to patches). We preserve it for the final output.
        # For the seasonal patch tokens (positions 1+), we use the encoder output.
        # For the trend patch tokens, trend_proj_out has shape [B, N, D].
        #
        # Final fused sequence shape: [B, N+1, D] (prototype at pos 0, N fused patches)
        if has_prototype:
            # seasonal_out[:, 0:1, :] — encoder-updated prototype (keep as-is)
            # seasonal_out[:, 1:, :]   — encoded seasonal patches
            # trend_proj_out           — projected trend patches [B, N, D]
            prototype_enc = seasonal_out[:, :1, :]          # [B, 1, D] — enriched prototype
            seasonal_patches = seasonal_out[:, 1:, :]        # [B, N, D] — encoded seasonal
            fused_patches = self.recomposer(trend_proj_out, seasonal_patches)  # [B, N, D]
            fused = torch.cat([prototype_enc, fused_patches], dim=1)          # [B, N+1, D]
        else:
            fused = self.recomposer(trend_proj_out, seasonal_out)  # [B, N, D]

        if output_hidden_states:
            hidden_states.append(fused)

        return fused, attns, hidden_states
