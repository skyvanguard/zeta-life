"""Experiment: Real data vs synthetic data consciousness response.

Compares how ConsciousOrganism (multi-kernel Darwinian Brain) responds to
structured external data (colored noise with 1/f spectral properties) versus
random synthetic stimuli.

Uses ConsciousOrganism instead of bare ConsciousKernel because consciousness
emerges from multi-agent dynamics (diversity, coherence, workspace competition),
not from individual kernel processing.

Measures: free_energy, prediction_error, dream_count, consciousness_index,
diversity, coherence, population dynamics, and detects phase transitions.
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
from zeta_life.kernel.conscious_organism import ConsciousOrganism


def run_organism_session(
    organism: ConsciousOrganism,
    stimulus_fn,
    n_steps: int,
    label: str,
) -> dict:
    """Run organism for n_steps, collecting metrics."""
    free_energies = []
    consciousness = []
    diversity = []
    coherence = []
    population = []

    for t in range(n_steps):
        stim = stimulus_fn()
        result = organism.step(stim)

        # Average free energy across all kernels
        avg_fe = np.mean(list(result.free_energies.values())) if result.free_energies else 0.0
        free_energies.append(avg_fe)
        consciousness.append(result.psi)
        diversity.append(result.diversity)
        coherence.append(result.coherence)
        population.append(result.population)

    fe = np.array(free_energies)
    ci = np.array(consciousness)
    div = np.array(diversity)
    coh = np.array(coherence)
    pop = np.array(population)

    return {
        "label": label,
        "n_steps": n_steps,
        "free_energy_mean": float(fe.mean()),
        "free_energy_std": float(fe.std()),
        "free_energy_final": float(fe[-100:].mean()),
        "consciousness_mean": float(ci.mean()),
        "consciousness_max": float(ci.max()),
        "consciousness_final": float(ci[-100:].mean()),
        "diversity_mean": float(div.mean()),
        "coherence_mean": float(coh.mean()),
        "population_final": int(pop[-1]),
        "population_max": int(pop.max()),
        "fe_series": fe,
        "ci_series": ci,
        "div_series": div,
        "coh_series": coh,
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
        ("Consciousness (mean)", "consciousness_mean", ".4f"),
        ("Consciousness (max)", "consciousness_max", ".4f"),
        ("Consciousness (final 100)", "consciousness_final", ".4f"),
        ("Diversity (mean)", "diversity_mean", ".4f"),
        ("Coherence (mean)", "coherence_mean", ".4f"),
        ("Population (final)", "population_final", "d"),
        ("Population (max)", "population_max", "d"),
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

    print("Experiment: Real Data vs Synthetic — Consciousness Response")
    print(f"Steps per condition: {N_STEPS}")
    print(f"Using ConsciousOrganism (multi-kernel Darwinian Brain)")
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
    org_real = ConsciousOrganism(obs_dim=OBS_DIM, initial_kernels=3, total_energy=15.0)

    real_results = run_organism_session(
        org_real,
        stimulus_fn=lambda: adapter.get_stimulus()[0],
        n_steps=N_STEPS,
        label="pink_noise",
    )
    print(f"  Done. Consciousness final: {real_results['consciousness_final']:.4f}")

    # --- Condition 2: Synthetic baseline (random) ---
    print("[2/2] Running with random synthetic stimuli (baseline)...")
    rng = np.random.default_rng(SEED)
    org_synth = ConsciousOrganism(obs_dim=OBS_DIM, initial_kernels=3, total_energy=15.0)

    def synthetic_stimulus():
        return torch.tensor(
            np.abs(rng.standard_normal(OBS_DIM)) + 0.01, dtype=torch.float32
        )

    synth_results = run_organism_session(
        org_synth,
        stimulus_fn=synthetic_stimulus,
        n_steps=N_STEPS,
        label="random_synthetic",
    )
    print(f"  Done. Consciousness final: {synth_results['consciousness_final']:.4f}")

    # --- Comparison ---
    print_comparison(real_results, synth_results)

    # --- Phase transitions ---
    print("\nPhase Transition Detection:")
    for label, results in [("Real", real_results), ("Synthetic", synth_results)]:
        fe_trans = detect_phase_transitions(results["fe_series"])
        ci_trans = detect_phase_transitions(results["ci_series"])
        div_trans = detect_phase_transitions(results["div_series"])
        print(f"  {label}:")
        print(f"    Free energy transitions:  {len(fe_trans)} at steps {fe_trans[:5]}")
        print(f"    Consciousness transitions: {len(ci_trans)} at steps {ci_trans[:5]}")
        print(f"    Diversity transitions:     {len(div_trans)} at steps {div_trans[:5]}")

    # --- Convergence analysis ---
    fe_ratio = real_results["free_energy_final"] / max(synth_results["free_energy_final"], 1e-10)
    ci_ratio = real_results["consciousness_final"] / max(synth_results["consciousness_final"], 1e-10)
    print(f"\nFree energy ratio (real/synthetic): {fe_ratio:.3f}")
    print(f"Consciousness ratio (real/synthetic): {ci_ratio:.3f}")

    if real_results["consciousness_final"] > synth_results["consciousness_final"]:
        print("-> Structured data produced HIGHER consciousness (more integration)")
    elif real_results["consciousness_final"] < synth_results["consciousness_final"]:
        print("-> Random data produced higher consciousness (more diversity)")
    else:
        print("-> Similar consciousness levels between conditions")

    print("\nExperiment complete.")
    return {"real": real_results, "synthetic": synth_results}


if __name__ == "__main__":
    main()
