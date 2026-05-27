#!/usr/bin/env python3
"""
Step 1: Synthetic Time Series Concept Dataset Generation (Timer version)

Generates synthetic time series data for 7 concept types for linear probing
experiments on Timer. Designed for Timer (seq_len=512 by default).

  1. AR(1)                - Autoregressive model with coefficient phi
  2. Level Shift          - Abrupt level change at random position tau with magnitude Delta
  3. Random Walk          - Random walk with drift mu
  4. Spectral             - Superposition of sine waves with frequency f and amplitude
  5. Time-Warped Sinusoid - Sinusoid with nonlinear time warping
  6. Deterministic Trend  - Linear trend with slope beta
  7. Variance Shift       - Variance change at position tau

Normalization rules (per paper):
  - AR(1), Spectral, Trend, Time-Warped: z-score normalization
  - Level Shift, Random Walk, Variance Shift: NO normalization (preserve scale)

Usage:
    python probe/ts_concept_synthetic_dataset.py \
        --n_samples 1000 --seq_len 512 --seed 42 --output_dir ./results/synthetic/

Output:
    results/synthetic/concepts_dataset.pt
    results/synthetic/concepts_visualization.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalization: (x - mean) / std"""
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


class TSConceptGenerator:
    """Generate synthetic time series with 7 concept types.

    Design principles:
      - High SNR: each parameter drives a clearly visible, large-scale feature.
      - Z-score only for concepts where the feature is in the SHAPE (AR, Spectral, Trend).
        For concepts where the feature is in the AMPLITUDE/SCALE (LevelShift, RandomWalk,
        VarianceShift), NO normalization preserves the parameter signal.
      - All parameters are scaled to [0, 1] for linear probing compatibility.
      - Noise is tuned per concept to keep SNR >> 1 across the full parameter range.
    """

    CONCEPTS = [
        "AR1", "LevelShift", "RandomWalk", "Spectral",
        "TimeWarpedSinusoid", "DeterministicTrend", "VarianceShift"
    ]

    # Which concepts get z-score normalization (shape-based, not amplitude-based)
    # - Shape-based concepts (True): AR1, Spectral, TimeWarpedSinusoid, DeterministicTrend
    #   -> z-score removes amplitude so only shape/pattern matters
    # - Amplitude-based concepts (False): LevelShift, RandomWalk, VarianceShift
    #   -> keep raw amplitude; the parameter itself encodes physical scale
    # All parameters are separately normalized to [0, 1] (or [-1, 1] for signed)
    # before saving, so the linear probe treats every dimension uniformly.
    NORM_FLAGS = {
        "AR1": True,                   # z-score: AR(1) structure, not amplitude
        "LevelShift": False,           # NO norm: amplitude jump is the signal
        "RandomWalk": False,           # NO norm: accumulated drift is the signal
        "Spectral": True,              # z-score: shape by frequency content
        "TimeWarpedSinusoid": True,    # z-score: shape deformation via phase/warp
        "DeterministicTrend": True,    # z-score: slope shape, not absolute amplitude
        "VarianceShift": False,       # NO norm: variance ratio is the signal
    }

    # Per-concept noise defaults (scaled for high SNR across parameter range)
    NOISE_DEFAULTS = {
        "AR1": 0.05,
        "LevelShift": 0.05,
        "RandomWalk": 0.01,
        "Spectral": 0.02,
        "TimeWarpedSinusoid": 0.02,
        "DeterministicTrend": 0.02,
        "VarianceShift": 0.005,
    }

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 512,
        seed: int = 42,
        noise_std: float = None,
        noise_per_concept: dict = None,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.seed = seed
        self.noise_per_concept = noise_per_concept or {}
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _noise(self, concept: str) -> float:
        return self.noise_per_concept.get(concept, self.NOISE_DEFAULTS.get(concept, 0.1))

    # ─────────────────────────────────────────────────────────────────────────
    # Individual concept generators
    # ─────────────────────────────────────────────────────────────────────────

    def _ar1(self, phi: float, noise_std: float) -> np.ndarray:
        """AR(1): x_t = phi * x_{t-1} + epsilon_t. phi in [0.5, 0.99] for high SNR."""
        x = np.zeros(self.seq_len)
        # Steady-state variance = sigma^2 / (1 - phi^2); initialize from that
        x[0] = np.random.randn() * noise_std / np.sqrt(max(1 - phi ** 2, 0.01))
        for t in range(1, self.seq_len):
            x[t] = phi * x[t - 1] + np.random.randn() * noise_std
        return x

    def _level_shift(self, tau: int, delta: float, noise_std: float) -> np.ndarray:
        """Level Shift: x_t = noise_t + delta * 1_{t >= tau}. delta in [-5, -1] ∪ [1, 5]."""
        x = np.random.randn(self.seq_len) * noise_std
        x[tau:] += delta
        return x

    def _random_walk(self, mu: float, sigma: float, noise_std: float) -> np.ndarray:
        """Random Walk with drift: x_t = x_{t-1} + mu + epsilon_t.
        mu in [-1.0, -0.2] or [0.2, 1.0] — strong directional drift over 512 steps."""
        x = np.zeros(self.seq_len)
        for t in range(1, self.seq_len):
            x[t] = x[t - 1] + mu + np.random.randn() * noise_std
        return x

    def _spectral(self, freqs: list, amps: list, phases: list, noise_std: float) -> np.ndarray:
        """Spectral: superposition of sine waves. Frequencies well-separated for interpretability."""
        t = np.arange(self.seq_len)
        x = np.zeros(self.seq_len)
        for f, a, p in zip(freqs, amps, phases):
            x += a * np.sin(2 * np.pi * f * t + p)
        x += np.random.randn(self.seq_len) * noise_std
        return x

    def _time_warped_sinusoid(
        self, freq: float, amp: float, phase: float, warp_strength: float, noise_std: float
    ) -> np.ndarray:
        """Time-Warped Sinusoid with stronger warp for clear signal."""
        n_warp = max(8, self.seq_len // 32)
        steps = np.abs(np.random.gamma(shape=2.0, scale=1.0, size=n_warp))
        cumsum = np.cumsum(steps)
        u_warp = cumsum / cumsum[-1] * (self.seq_len - 1)

        b = amp * np.sin(2 * np.pi * freq * u_warp + phase)
        t_reg = np.arange(self.seq_len)
        x = np.interp(t_reg, u_warp, b)

        u_uni = t_reg.astype(float)
        alpha = min(abs(warp_strength) / 1.0, 1.0)   # stronger warp range
        x = (1 - alpha) * amp * np.sin(2 * np.pi * freq * u_uni + phase) + alpha * x
        x += np.random.randn(self.seq_len) * noise_std
        return x

    def _deterministic_trend(self, beta: float, noise_std: float) -> np.ndarray:
        """Deterministic Trend: x_t = beta * t + epsilon_t.
        beta in [-1, -0.05] ∪ [0.05, 1] for clear linear trend over 512 steps."""
        t = np.arange(self.seq_len).astype(float)
        x = beta * t + np.random.randn(self.seq_len) * noise_std
        return x

    def _variance_shift(self, tau: int, sigma_before: float, sigma_after: float, noise_std: float) -> np.ndarray:
        """Variance Shift: variance changes at tau. Sigmas in [0.2, 2.5] with clear contrast."""
        x = np.zeros(self.seq_len)
        x[:tau] = np.random.randn(tau) * sigma_before
        x[tau:] = np.random.randn(self.seq_len - tau) * sigma_after
        return x

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self) -> tuple[torch.Tensor, list[dict]]:
        """
        Generate dataset for all 7 concepts.

        Returns:
            X: [n_samples * 7, seq_len] tensor
            labels: list of dicts
        """
        all_X = []
        all_labels = []

        X, labels = self._generate_ar1()
        all_X.append(X)
        all_labels.extend(labels)

        X, labels = self._generate_level_shift()
        all_X.append(X)
        all_labels.extend(labels)

        X, labels = self._generate_random_walk()
        all_X.append(X)
        all_labels.extend(labels)

        X, labels = self._generate_spectral()
        all_X.append(X)
        all_labels.extend(labels)

        X, labels = self._generate_time_warped_sinusoid()
        all_X.append(X)
        all_labels.extend(labels)

        X, labels = self._generate_deterministic_trend()
        all_X.append(X)
        all_labels.extend(labels)

        X, labels = self._generate_variance_shift()
        all_X.append(X)
        all_labels.extend(labels)

        X_all = np.concatenate(all_X, axis=0)
        return torch.from_numpy(X_all).float(), all_labels

    def _generate_ar1(self) -> tuple[np.ndarray, list[dict]]:
        # phi in [0.7, 0.99] — high phi means strong autocorrelation; signal is unmistakable
        phis = np.random.uniform(0.7, 0.99, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        noise = self._noise("AR1")
        for i, phi in enumerate(phis):
            x = self._ar1(phi, noise)
            if self.NORM_FLAGS["AR1"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({"concept": "AR1", "phi": float(phi)})
        return X, labels

    def _generate_level_shift(self) -> tuple[np.ndarray, list[dict]]:
        # tau in [64, 448] — change point well inside the sequence
        # delta in [-8, -2] or [2, 8] — large, unambiguous jumps
        taus = np.random.randint(64, self.seq_len - 64, self.n_samples)
        signs = np.random.choice([-1, 1], self.n_samples)
        mag = np.random.uniform(2.0, 8.0, self.n_samples)
        deltas = signs * mag
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        noise = self._noise("LevelShift")
        for i in range(self.n_samples):
            x = self._level_shift(int(taus[i]), float(deltas[i]), noise)
            X[i] = x
            labels.append({"concept": "LevelShift", "tau": int(taus[i]), "delta": float(deltas[i])})
        return X, labels

    def _generate_random_walk(self) -> tuple[np.ndarray, list[dict]]:
        # mu in [-2.0, -0.5] or [0.5, 2.0] — very strong drift
        # sigma in [0.05, 0.3] — step noise reduced further
        mus = np.random.uniform(-2.0, 2.0, self.n_samples)
        mus = np.where(np.abs(mus) < 0.5, np.where(mus >= 0, 0.5, -0.5), mus)
        sigmas = np.random.uniform(0.05, 0.3, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        noise = self._noise("RandomWalk")
        for i in range(self.n_samples):
            x = self._random_walk(float(mus[i]), float(sigmas[i]), noise)
            X[i] = x
            labels.append({"concept": "RandomWalk", "mu": float(mus[i]), "sigma": float(sigmas[i])})
        return X, labels

    def _generate_spectral(self) -> tuple[np.ndarray, list[dict]]:
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        noise = self._noise("Spectral")
        for i in range(self.n_samples):
            # 2 components with well-separated frequencies for clarity
            n_components = 2
            freqs = sorted(np.random.uniform(0.05, 0.45, n_components).tolist())
            amps = np.random.uniform(2.0, 5.0, n_components).tolist()
            phases = np.random.uniform(0, 2 * np.pi, n_components).tolist()
            x = self._spectral(freqs, amps, phases, noise)
            if self.NORM_FLAGS["Spectral"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({
                "concept": "Spectral",
                "n_components": n_components,
                "freqs": freqs,
                "amps": amps,
                "phases": phases,
            })
        return X, labels

    def _generate_time_warped_sinusoid(self) -> tuple[np.ndarray, list[dict]]:
        # freq in [0.05, 0.4] — multiple full cycles in 512 steps
        # warp in [-1.0, -0.4] or [0.4, 1.0] — strong, unambiguous warping
        freqs = np.random.uniform(0.05, 0.4, self.n_samples)
        amps = np.random.uniform(2.0, 5.0, self.n_samples)
        phases = np.random.uniform(0, 2 * np.pi, self.n_samples)
        warps = np.random.uniform(-1.0, 1.0, self.n_samples)
        warps = np.where(np.abs(warps) < 0.4,
                         np.sign(warps + 1e-8) * 0.4 + np.random.uniform(-0.1, 0.1, self.n_samples),
                         warps)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        noise = self._noise("TimeWarpedSinusoid")
        for i in range(self.n_samples):
            x = self._time_warped_sinusoid(
                float(freqs[i]), float(amps[i]), float(phases[i]), float(warps[i]), noise
            )
            if self.NORM_FLAGS["TimeWarpedSinusoid"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({
                "concept": "TimeWarpedSinusoid",
                "freq": float(freqs[i]),
                "amp": float(amps[i]),
                "phase": float(phases[i]),
                "warp": float(warps[i]),
            })
        return X, labels

    def _generate_deterministic_trend(self) -> tuple[np.ndarray, list[dict]]:
        # beta in [-2.0, -0.2] or [0.2, 2.0] — strong slope dominates noise over 512 steps
        betas = np.random.uniform(-2.0, 2.0, self.n_samples)
        betas = np.where(np.abs(betas) < 0.2,
                         np.sign(betas + 1e-8) * 0.2,
                         betas)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        noise = self._noise("DeterministicTrend")
        for i, beta in enumerate(betas):
            x = self._deterministic_trend(beta, noise)
            if self.NORM_FLAGS["DeterministicTrend"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({"concept": "DeterministicTrend", "beta": float(beta)})
        return X, labels

    def _generate_variance_shift(self) -> tuple[np.ndarray, list[dict]]:
        # tau in [64, 448]
        # sigma_before and sigma_after in [0.2, 2.5] with STRICT 3x contrast for ALL samples
        taus = np.random.randint(64, self.seq_len - 64, self.n_samples)
        sigma_before = np.random.uniform(0.2, 2.5, self.n_samples)
        sigma_after = np.random.uniform(0.2, 2.5, self.n_samples)
        # Force at least 3x contrast ratio for every sample
        for i in range(self.n_samples):
            ratio = sigma_before[i] / (sigma_after[i] + 1e-8)
            if ratio >= 1.0:
                # sigma_before >= sigma_after: push sigma_after down to ensure 3x
                sigma_after[i] = sigma_before[i] / 3.0
            else:
                # sigma_after > sigma_before: push sigma_before down to ensure 3x
                sigma_before[i] = sigma_after[i] / 3.0
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        noise = self._noise("VarianceShift")
        for i in range(self.n_samples):
            x = self._variance_shift(
                int(taus[i]), float(sigma_before[i]), float(sigma_after[i]), noise
            )
            X[i] = x
            labels.append({
                "concept": "VarianceShift",
                "tau": int(taus[i]),
                "sigma_before": float(sigma_before[i]),
                "sigma_after": float(sigma_after[i]),
            })
        return X, labels

    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────

    def plot_samples(
        self,
        X: torch.Tensor,
        labels: list[dict],
        output_path: str,
        n_per_concept: int = 4,
    ):
        """Plot sample time series from each concept."""
        fig, axes = plt.subplots(7, n_per_concept, figsize=(n_per_concept * 3, 7 * 2))
        concept_indices = {}
        current_idx = 0
        for concept in self.CONCEPTS:
            concept_indices[concept] = (current_idx, current_idx + self.n_samples)
            current_idx += self.n_samples

        for row, concept in enumerate(self.CONCEPTS):
            start, end = concept_indices[concept]
            indices = np.random.choice(
                range(start, end), size=n_per_concept, replace=False
            )
            for col, idx in enumerate(indices):
                ax = axes[row, col]
                ax.plot(X[idx].numpy(), linewidth=0.8)
                ax.set_title(
                    f"{concept}\n{self._label_summary(labels[idx])}",
                    fontsize=7,
                )
                ax.tick_params(labelsize=5)
                ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Synthetic Time Series Concepts (Timer probe dataset)",
            fontsize=10,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved visualization: {output_path}")

    def _label_summary(self, label: dict) -> str:
        parts = []
        for k, v in label.items():
            if k == "concept":
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            elif isinstance(v, list):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts) if parts else ""


def extract_param_vector(label: dict, max_len: int = 4) -> np.ndarray:
    """
    Convert a label dict to a fixed-length numeric parameter vector for linear probing.

    All parameters are normalized to [0, 1] (or [-1, 1] for signed parameters) so that
    the linear probe treats all dimensions equally, regardless of their physical scale.

    Conventions:
      - Parameters with symmetric ranges (e.g. [-1, 1]) → mapped to [-1, 1]
      - Parameters with positive ranges (e.g. [0, 1]) → mapped to [0, 1]
      - Phase (periodic) → mapped to [0, 1]
    """
    concept = label["concept"]
    vec = np.zeros(max_len, dtype=np.float32)

    if concept == "AR1":
        # phi in [0.7, 0.99] → [0, 1]
        vec[0] = (label["phi"] - 0.7) / (0.99 - 0.7)

    elif concept == "LevelShift":
        # tau in [64, 448] (relative to seq_len=512) → [0, 1]
        vec[0] = (label["tau"] - 64) / (448 - 64)
        # delta in [-8, -2] or [2, 8] → sign-preserving magnitude to [0, 1]
        delta = label["delta"]
        sign = 1 if delta > 0 else -1
        mag = abs(delta)
        vec[1] = sign * (mag - 2.0) / (8.0 - 2.0)

    elif concept == "RandomWalk":
        # mu in [-2.0, 2.0] (with zero band removed) → [-1, 1]
        vec[0] = label["mu"] / 2.0
        # sigma in [0.05, 0.3] → [0, 1]
        vec[1] = (label["sigma"] - 0.05) / (0.3 - 0.05)

    elif concept == "Spectral":
        # freq_0, freq_1 in [0.05, 0.45] → [0, 1]
        freqs = label["freqs"]
        vec[0] = (freqs[0] - 0.05) / (0.45 - 0.05)
        vec[1] = (freqs[1] - 0.05) / (0.45 - 0.05)
        # amp_0 in [2.0, 5.0] → [0, 1]
        amps = label["amps"]
        vec[2] = (amps[0] - 2.0) / (5.0 - 2.0)

    elif concept == "TimeWarpedSinusoid":
        # freq in [0.05, 0.4] → [0, 1]
        vec[0] = (label["freq"] - 0.05) / (0.4 - 0.05)
        # amp in [2.0, 5.0] → [0, 1]
        vec[1] = (label["amp"] - 2.0) / (5.0 - 2.0)
        # phase already [0, 1]
        vec[2] = label["phase"] / (2 * np.pi)
        # warp in [-1, 1] → [-1, 1]
        vec[3] = label["warp"] / 1.0

    elif concept == "DeterministicTrend":
        # beta in [-2.0, 2.0] → [-1, 1]
        vec[0] = label["beta"] / 2.0

    elif concept == "VarianceShift":
        # tau in [64, 448] → [0, 1]
        vec[0] = (label["tau"] - 64) / (448 - 64)
        # sigma_before in [0.2, 2.5] → [0, 1]
        vec[1] = (label["sigma_before"] - 0.2) / (2.5 - 0.2)
        # sigma_after in [0.2, 2.5] → [0, 1]
        vec[2] = (label["sigma_after"] - 0.2) / (2.5 - 0.2)

    return vec


def main():
    parser = argparse.ArgumentParser(description="Synthetic Time Series Concept Generation (Timer)")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples per concept")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Length of each time series (Timer default: 512)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--noise_std", type=float, default=None,
                        help="Global noise std (overridden by --noise_per_concept)")
    parser.add_argument("--output_dir", type=str, default="./results/synthetic/",
                        help="Output directory")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Step 1: Synthetic Time Series Concept Dataset Generation (Timer)")
    print("=" * 60)
    print(f"  n_samples   : {args.n_samples}")
    print(f"  seq_len     : {args.seq_len}  (Timer context length)")
    print(f"  seed        : {args.seed}")
    print(f"  noise_std   : {args.noise_std}  (per-concept overrides below)")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  concepts    : {TSConceptGenerator.CONCEPTS}")
    print(f"  noise/concept: {TSConceptGenerator.NOISE_DEFAULTS}")
    print("=" * 60)

    generator = TSConceptGenerator(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        noise_std=args.noise_std,
        noise_per_concept=TSConceptGenerator.NOISE_DEFAULTS,
    )
    X, labels = generator.generate()
    print(f"\nGenerated dataset shape: {X.shape}")

    param_vectors = []
    for label in labels:
        param_vectors.append(extract_param_vector(label))
    param_vectors = np.stack(param_vectors, axis=0)
    param_tensors = torch.from_numpy(param_vectors).float()

    concept_map = {c: i for i, c in enumerate(TSConceptGenerator.CONCEPTS)}
    concept_indices = torch.tensor(
        [concept_map[lbl["concept"]] for lbl in labels], dtype=torch.long
    )

    dataset_path = os.path.join(args.output_dir, "concepts_dataset.pt")
    torch.save(
        {
            "X": X,
            "labels": labels,
            "params": param_tensors,
            "concept_idx": concept_indices,
            "concepts": TSConceptGenerator.CONCEPTS,
            "norm_flags": TSConceptGenerator.NORM_FLAGS,
            "n_samples_per_concept": args.n_samples,
            "seq_len": args.seq_len,
            "seed": args.seed,
        },
        dataset_path,
    )
    print(f"[Save] Dataset saved: {dataset_path}")

    for concept in TSConceptGenerator.CONCEPTS:
        count = sum(1 for lbl in labels if lbl["concept"] == concept)
        norm_flag = TSConceptGenerator.NORM_FLAGS[concept]
        print(f"  {concept:<25s}: {count} samples  (normalized={norm_flag})")

    if not args.no_plot:
        plot_path = os.path.join(args.output_dir, "concepts_visualization.png")
        generator.plot_samples(X, labels, plot_path, n_per_concept=4)
        print(f"\nDataset ready: {dataset_path}")
        print(f"Labels shape: {param_tensors.shape}")

    print("\n[Done] Step 1 complete.")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Shared concept specs (used by all downstream probe scripts)
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_PARAM_SPEC: dict[str, tuple[list[int], str]] = {
    "AR1":                   ([0],                               "phi"),
    "LevelShift":           ([0, 1],                            "tau, delta"),
    "RandomWalk":           ([0, 1],                            "mu, sigma"),
    "Spectral":             ([0, 1, 2],                         "freq_0, freq_1, amp_0"),
    "TimeWarpedSinusoid":   ([0, 1, 2, 3],                     "freq, amp, phase, warp"),
    "DeterministicTrend":    ([0],                               "beta"),
    "VarianceShift":        ([0, 1, 2],                         "tau, sigma_before, sigma_after"),
}
