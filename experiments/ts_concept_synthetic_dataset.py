#!/usr/bin/env python3
"""
Step 1: Synthetic Time Series Concept Dataset Generation

Generates synthetic time series data for 7 concept types following the paper's framework:
  1. AR(1)       - Autoregressive model with coefficient phi
  2. Level Shift - Abrupt level change at random position tau with magnitude Delta
  3. Random Walk - Random walk with drift mu
  4. Spectral    - Superposition of sine waves with frequency f and amplitude
  5. Time-Warped Sinusoid - Sinusoid with nonlinear time warping
  6. Deterministic Trend - Linear trend with slope beta
  7. Variance Shift - Variance change at position tau

Normalization rules (per paper):
  - AR(1), Spectral, Trend, Time-Warped: z-score normalization
  - Level Shift, Random Walk, Variance Shift: NO normalization (preserve scale)

Returns: (X, labels) where X is [n_samples, seq_len] and labels is [n_samples, n_params]

Usage:
    python experiments/ts_concept_synthetic_dataset.py \
        --n_samples 1000 --seq_len 256 --seed 42 --output_dir ./results/synthetic/

Output:
    results/synthetic/concepts_dataset.pt  - torch.save({'X': tensor, 'labels': tensor, ...})
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
    """Generate synthetic time series with 7 concept types."""

    CONCEPTS = [
        "AR1", "LevelShift", "RandomWalk", "Spectral",
        "TimeWarpedSinusoid", "DeterministicTrend", "VarianceShift"
    ]

    # Normalization flag per concept: True = z-score, False = raw
    NORM_FLAGS = {
        "AR1": True,
        "LevelShift": False,
        "RandomWalk": False,
        "Spectral": True,
        "TimeWarpedSinusoid": True,
        "DeterministicTrend": True,
        "VarianceShift": False,
    }

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 256,
        seed: int = 42,
        noise_std: float = 0.1,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.seed = seed
        self.noise_std = noise_std
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Individual concept generators
    # ─────────────────────────────────────────────────────────────────────────

    def _ar1(self, phi: float) -> np.ndarray:
        """AR(1): x_t = phi * x_{t-1} + epsilon_t"""
        x = np.zeros(self.seq_len)
        x[0] = np.random.randn() * self.noise_std
        for t in range(1, self.seq_len):
            x[t] = phi * x[t - 1] + np.random.randn() * self.noise_std
        return x

    def _level_shift(self, tau: int, delta: float) -> np.ndarray:
        """Level Shift: add Delta to x[t] for t >= tau"""
        x = np.random.randn(self.seq_len) * self.noise_std
        x[tau:] += delta
        return x

    def _random_walk(self, mu: float) -> np.ndarray:
        """Random Walk with drift: x_t = x_{t-1} + mu + epsilon_t"""
        x = np.zeros(self.seq_len)
        for t in range(1, self.seq_len):
            x[t] = x[t - 1] + mu + np.random.randn() * self.noise_std
        return x

    def _spectral(self, freqs: list, amps: list, phases: list) -> np.ndarray:
        """Spectral: superposition of sine waves"""
        t = np.arange(self.seq_len)
        x = np.zeros(self.seq_len)
        for f, a, p in zip(freqs, amps, phases):
            x += a * np.sin(2 * np.pi * f * t + p)
        x += np.random.randn(self.seq_len) * self.noise_std
        return x

    def _time_warped_sinusoid(
        self, freq: float, amp: float, phase: float, warp_strength: float
    ) -> np.ndarray:
        """Time-Warped Sinusoid: nonlinear time warping of sine wave"""
        t = np.arange(self.seq_len).astype(float)
        warped_t = t + warp_strength * np.sin(2 * np.pi * 0.02 * t)
        warped_t = warped_t / (self.seq_len - 1)  # Normalize to [0,1]
        x = amp * np.sin(2 * np.pi * freq * warped_t + phase)
        x += np.random.randn(self.seq_len) * self.noise_std
        return x

    def _deterministic_trend(self, beta: float) -> np.ndarray:
        """Deterministic Trend: x_t = beta * t + epsilon_t"""
        t = np.arange(self.seq_len).astype(float)
        x = beta * t + np.random.randn(self.seq_len) * self.noise_std
        return x

    def _variance_shift(self, tau: int, sigma_before: float, sigma_after: float) -> np.ndarray:
        """Variance Shift: change noise std at position tau"""
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
            X: [n_samples * 7, seq_len] tensor of time series
            labels: list of dicts with concept name and parameters
        """
        all_X = []
        all_labels = []

        # AR(1)
        X, labels = self._generate_ar1()
        all_X.append(X)
        all_labels.extend(labels)

        # Level Shift
        X, labels = self._generate_level_shift()
        all_X.append(X)
        all_labels.extend(labels)

        # Random Walk
        X, labels = self._generate_random_walk()
        all_X.append(X)
        all_labels.extend(labels)

        # Spectral
        X, labels = self._generate_spectral()
        all_X.append(X)
        all_labels.extend(labels)

        # Time-Warped Sinusoid
        X, labels = self._generate_time_warped_sinusoid()
        all_X.append(X)
        all_labels.extend(labels)

        # Deterministic Trend
        X, labels = self._generate_deterministic_trend()
        all_X.append(X)
        all_labels.extend(labels)

        # Variance Shift
        X, labels = self._generate_variance_shift()
        all_X.append(X)
        all_labels.extend(labels)

        X_all = np.concatenate(all_X, axis=0)  # [7*n_samples, seq_len]
        return torch.from_numpy(X_all).float(), all_labels

    def _generate_ar1(self) -> tuple[np.ndarray, list[dict]]:
        """Generate AR(1) samples with varying phi."""
        phis = np.random.uniform(-0.9, 0.9, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        for i, phi in enumerate(phis):
            x = self._ar1(phi)
            if self.NORM_FLAGS["AR1"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({"concept": "AR1", "phi": float(phi)})
        return X, labels

    def _generate_level_shift(self) -> tuple[np.ndarray, list[dict]]:
        """Generate Level Shift samples with varying tau and Delta."""
        taus = np.random.randint(20, self.seq_len - 20, self.n_samples)
        deltas = np.random.uniform(-5, 5, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        for i in range(self.n_samples):
            x = self._level_shift(int(taus[i]), float(deltas[i]))
            if self.NORM_FLAGS["LevelShift"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({"concept": "LevelShift", "tau": int(taus[i]), "delta": float(deltas[i])})
        return X, labels

    def _generate_random_walk(self) -> tuple[np.ndarray, list[dict]]:
        """Generate Random Walk samples with varying drift mu."""
        mus = np.random.uniform(-0.1, 0.1, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        for i, mu in enumerate(mus):
            x = self._random_walk(mu)
            if self.NORM_FLAGS["RandomWalk"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({"concept": "RandomWalk", "mu": float(mu)})
        return X, labels

    def _generate_spectral(self) -> tuple[np.ndarray, list[dict]]:
        """Generate Spectral samples with varying frequencies, amplitudes, phases."""
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        for i in range(self.n_samples):
            n_components = np.random.choice([2, 3])
            freqs = np.random.uniform(0.01, 0.3, n_components).tolist()
            amps = np.random.uniform(0.5, 2.0, n_components).tolist()
            phases = np.random.uniform(0, 2 * np.pi, n_components).tolist()
            x = self._spectral(freqs, amps, phases)
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
        """Generate Time-Warped Sinusoid samples."""
        freqs = np.random.uniform(0.05, 0.2, self.n_samples)
        amps = np.random.uniform(1.0, 3.0, self.n_samples)
        phases = np.random.uniform(0, 2 * np.pi, self.n_samples)
        warps = np.random.uniform(-0.3, 0.3, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        for i in range(self.n_samples):
            x = self._time_warped_sinusoid(
                float(freqs[i]), float(amps[i]), float(phases[i]), float(warps[i])
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
        """Generate Deterministic Trend samples with varying slope beta."""
        betas = np.random.uniform(-0.1, 0.1, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        for i, beta in enumerate(betas):
            x = self._deterministic_trend(beta)
            if self.NORM_FLAGS["DeterministicTrend"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
            X[i] = x
            labels.append({"concept": "DeterministicTrend", "beta": float(beta)})
        return X, labels

    def _generate_variance_shift(self) -> tuple[np.ndarray, list[dict]]:
        """Generate Variance Shift samples with varying tau and sigma."""
        taus = np.random.randint(20, self.seq_len - 20, self.n_samples)
        sigma_before = np.random.uniform(0.1, 1.0, self.n_samples)
        sigma_after = np.random.uniform(0.1, 1.0, self.n_samples)
        X = np.zeros((self.n_samples, self.seq_len))
        labels = []
        for i in range(self.n_samples):
            x = self._variance_shift(
                int(taus[i]), float(sigma_before[i]), float(sigma_after[i])
            )
            if self.NORM_FLAGS["VarianceShift"]:
                x = zscore_normalize(x.reshape(1, -1)).flatten()
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
            # Randomly pick n_per_concept samples
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
            "Synthetic Time Series Concepts (normalized flag per concept)",
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
    All concepts are padded to max_len dimensions.
    """
    concept = label["concept"]
    vec = np.zeros(max_len, dtype=np.float32)

    if concept == "AR1":
        vec[0] = label["phi"]
    elif concept == "LevelShift":
        vec[0] = label["tau"] / 256.0
        vec[1] = label["delta"]
    elif concept == "RandomWalk":
        vec[0] = label["mu"]
    elif concept == "Spectral":
        freqs = label["freqs"]
        for i, f in enumerate(freqs[:max_len]):
            vec[i] = f
    elif concept == "TimeWarpedSinusoid":
        vec[0] = label["freq"]
        vec[1] = label["amp"]
        vec[2] = label["phase"] / (2 * np.pi)
        vec[3] = label["warp"]
    elif concept == "DeterministicTrend":
        vec[0] = label["beta"]
    elif concept == "VarianceShift":
        vec[0] = label["tau"] / 256.0
        vec[1] = label["sigma_before"]
        vec[2] = label["sigma_after"]

    return vec


def main():
    parser = argparse.ArgumentParser(description="Synthetic Time Series Concept Generation")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples per concept")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Length of each time series")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Base noise standard deviation")
    parser.add_argument("--output_dir", type=str, default="./results/synthetic/",
                        help="Output directory")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Step 1: Synthetic Time Series Concept Dataset Generation")
    print("=" * 60)
    print(f"  n_samples   : {args.n_samples}")
    print(f"  seq_len     : {args.seq_len}")
    print(f"  seed        : {args.seed}")
    print(f"  noise_std   : {args.noise_std}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  concepts    : {TSConceptGenerator.CONCEPTS}")
    print("=" * 60)

    # Generate dataset
    generator = TSConceptGenerator(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        noise_std=args.noise_std,
    )
    X, labels = generator.generate()
    print(f"\nGenerated dataset shape: {X.shape}")  # [7*n_samples, seq_len]

    # Build param vectors for each sample
    param_vectors = []
    for label in labels:
        param_vectors.append(extract_param_vector(label))
    param_vectors = np.stack(param_vectors, axis=0)  # [7*n_samples, n_params]
    param_tensors = torch.from_numpy(param_vectors).float()

    # Build concept index tensor
    concept_map = {c: i for i, c in enumerate(TSConceptGenerator.CONCEPTS)}
    concept_indices = torch.tensor(
        [concept_map[lbl["concept"]] for lbl in labels], dtype=torch.long
    )

    # Save dataset
    dataset_path = os.path.join(args.output_dir, "concepts_dataset.pt")
    torch.save(
        {
            "X": X,                     # [7*n_samples, seq_len]
            "labels": labels,           # list of dicts
            "params": param_tensors,    # [7*n_samples, max_n_params]
            "concept_idx": concept_indices,  # [7*n_samples]
            "concepts": TSConceptGenerator.CONCEPTS,
            "norm_flags": TSConceptGenerator.NORM_FLAGS,
            "n_samples_per_concept": args.n_samples,
            "seq_len": args.seq_len,
            "seed": args.seed,
        },
        dataset_path,
    )
    print(f"[Save] Dataset saved: {dataset_path}")

    # Print summary
    for concept in TSConceptGenerator.CONCEPTS:
        count = sum(1 for lbl in labels if lbl["concept"] == concept)
        norm_flag = TSConceptGenerator.NORM_FLAGS[concept]
        print(f"  {concept:<25s}: {count} samples  (normalized={norm_flag})")

    # Plot
    if not args.no_plot:
        plot_path = os.path.join(args.output_dir, "concepts_visualization.png")
        generator.plot_samples(X, labels, plot_path, n_per_concept=4)
        print(f"\nDataset ready: {dataset_path}")
        print(f"Labels shape: {param_tensors.shape}")

    print("\n[Done] Step 1 complete.")


if __name__ == "__main__":
    main()
