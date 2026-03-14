"""Experiment: Real data vs synthetic data consciousness response.

Compares how ConsciousKernel responds to structured external data
(colored noise with 1/f spectral properties) versus random synthetic stimuli.

Measures: free_energy, prediction_error, dream_count, consciousness_index,
and detects phase transitions in consciousness metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from zeta_life.datasets import ColoredNoiseSource, DatasetAdapter
from zeta_life.kernel.conscious_kernel import ConsciousKernel


def run_kernel_session(
    kernel: ConsciousKernel,
    stimulus_fn,
    n_steps: int,
    label: str,
) -> dict:
    """Run kernel for n_steps, collecting metrics."""
    free_energies = []
    pred_errors = []
    consciousness = []
    dream_count = 0

    for t in range(n_steps):
        stim = stimulus_fn()
        result = kernel.step(stim)
        free_energies.append(result.free_energy)
        pred_errors.append(sum(result.errors.values()))
        consciousness.append(result.consciousness)
        if result.dreamed:
            dream_count += 1

    fe = np.array(free_energies)
    pe = np.array(pred_errors)
    ci = np.array(consciousness)

    return {
        "label": label,
        "n_steps": n_steps,
        "free_energy_mean": float(fe.mean()),
        "free_energy_std": float(fe.std()),
        "free_energy_final": float(fe[-100:].mean()),
        "pred_error_mean": float(pe.mean()),
        "pred_error_final": float(pe[-100:].mean()),
        "consciousness_mean": float(ci.mean()),
        "consciousness_max": float(ci.max()),
        "dream_count": dream_count,
        "fe_series": fe,
        "ci_series": ci,
    }


def detect_phase_transitions(series: np.ndarray, window: int = 100) -> list[int]:
    """Detect abrupt changes in a time series via rolling std deviation."""
    if len(series) < 2 * window:
        return []
    transitions = []
    rolling_mean = np.convolve(series, np.ones(window) / window, mode="valid")
    diff = np.abs(np.diff(rolling_mean))
    threshold = diff.mean() + 2 * diff.std()
    for i, d in enumerate(diff):
        if d > threshold:
            transitions.append(i + window)
    return transitions


def print_comparison(real: dict, synthetic: dict) -> None:
    """Print side-by-side comparison table."""
    metrics = [
        ("Free Energy (mean)", "free_energy_mean", ".4f"),
        ("Free Energy (std)", "free_energy_std", ".4f"),
        ("Free Energy (final 100)", "free_energy_final", ".4f"),
        ("Prediction Error (mean)", "pred_error_mean", ".4f"),
        ("Prediction Error (final)", "pred_error_final", ".4f"),
        ("Consciousness (mean)", "consciousness_mean", ".4f"),
        ("Consciousness (max)", "consciousness_max", ".4f"),
        ("Dream Count", "dream_count", "d"),
    ]

    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Real Data':>15} {'Synthetic':>15}")
    print("-" * 70)
    for name, key, fmt in metrics:
        rv = real[key]
        sv = synthetic[key]
        print(f"{name:<30} {rv:>{15}{fmt}} {sv:>{15}{fmt}}")
    print("=" * 70)


def main():
    N_STEPS = 2000
    OBS_DIM = 4
    SEED = 42

    print("Experiment: Real Data vs Synthetic Consciousness Response")
    print(f"Steps per condition: {N_STEPS}")
    print()

    # --- Condition 1: Real data (pink noise) ---
    print("[1/2] Running with colored noise (pink, 1/f)...")
    source = ColoredNoiseSource(
        n_samples=N_STEPS + 500,
        n_channels=OBS_DIM,
        noise_type="pink",
        seed=SEED,
    )
    adapter = DatasetAdapter(source, obs_dim=OBS_DIM, projection="identity")
    kernel_real = ConsciousKernel(obs_dim=OBS_DIM)

    real_results = run_kernel_session(
        kernel_real,
        stimulus_fn=lambda: adapter.get_stimulus()[0],
        n_steps=N_STEPS,
        label="pink_noise",
    )
    print(f"  Done. Final free energy: {real_results['free_energy_final']:.4f}")

    # --- Condition 2: Synthetic baseline (random uniform) ---
    print("[2/2] Running with random synthetic stimuli (baseline)...")
    rng = np.random.default_rng(SEED)
    kernel_synth = ConsciousKernel(obs_dim=OBS_DIM)

    def synthetic_stimulus():
        return torch.tensor(
            np.abs(rng.standard_normal(OBS_DIM)) + 0.01, dtype=torch.float32
        )

    synth_results = run_kernel_session(
        kernel_synth,
        stimulus_fn=synthetic_stimulus,
        n_steps=N_STEPS,
        label="random_synthetic",
    )
    print(f"  Done. Final free energy: {synth_results['free_energy_final']:.4f}")

    # --- Comparison ---
    print_comparison(real_results, synth_results)

    # --- Phase transitions ---
    print("\nPhase Transition Detection:")
    for label, results in [("Real", real_results), ("Synthetic", synth_results)]:
        fe_transitions = detect_phase_transitions(results["fe_series"])
        ci_transitions = detect_phase_transitions(results["ci_series"])
        print(f"  {label}:")
        print(f"    Free energy transitions: {len(fe_transitions)} at steps {fe_transitions[:5]}")
        print(f"    Consciousness transitions: {len(ci_transitions)} at steps {ci_transitions[:5]}")

    # --- Convergence check ---
    fe_ratio = real_results["free_energy_final"] / max(synth_results["free_energy_final"], 1e-10)
    print(f"\nFree energy ratio (real/synthetic): {fe_ratio:.3f}")
    if fe_ratio < 1.0:
        print("-> WorldModel learned real data patterns better (lower free energy)")
    else:
        print("-> Random data produced lower free energy (unexpected)")

    print("\nExperiment complete.")
    return {"real": real_results, "synthetic": synth_results}


if __name__ == "__main__":
    main()
