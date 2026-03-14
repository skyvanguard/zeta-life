"""Experiment: Phase transitions in hierarchical consciousness with real data.

Tests the formal equation Psi = B^3 + Phi by feeding structured data
(colored noise) into HierarchicalSimulation with use_formal_psi=True.

Sweeps coupling parameter alpha and noise type to map the phase boundary
between subcritical (Psi=0) and supercritical (Psi>0) regimes.

Key predictions:
- Pink noise (1/f correlations) should lower coherence cost C, enabling
  supercritical transitions at lower alpha values.
- White noise (uncorrelated) should require higher alpha for emergence.
- Brown noise (over-correlated) may suppress diversity, limiting Psi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from zeta_life.consciousness.hierarchical_simulation import (
    HierarchicalSimulation,
    SimulationConfig,
)
from zeta_life.datasets import ColoredNoiseSource, DatasetAdapter


def run_sweep(
    noise_type: str,
    alpha: float,
    n_steps: int = 500,
    n_cells: int = 60,
) -> dict:
    """Run one condition: noise_type x alpha."""
    config = SimulationConfig(
        n_cells=n_cells,
        n_clusters=4,
        n_steps=n_steps,
        use_formal_psi=True,
        consciousness_alpha=alpha,
        enable_resilience=True,
        resilience_preset="optimal",
        enable_perturbations=False,
    )

    sim = HierarchicalSimulation(config)
    sim.initialize()

    source = ColoredNoiseSource(
        n_samples=n_steps + 100,
        n_channels=4,
        noise_type=noise_type,
        seed=42,
    )
    adapter = DatasetAdapter(source, obs_dim=4, projection="identity")

    for _ in range(n_steps):
        stimulus = adapter.get_hierarchical_stimulus()
        sim.step(external_stimulus=stimulus)

    history = sim.metrics_history
    if not history:
        return _empty_result(noise_type, alpha, n_steps)

    # Collect final-quarter metrics
    quarter = max(1, len(history) // 4)
    final = history[-quarter:]

    psi_values = [m.psi_raw for m in final]
    phi_values = [m.phi_global for m in final]
    ci_values = [m.consciousness_index for m in final]
    b_values = [m.B_value for m in final]
    supercritical = [m.is_supercritical for m in final]
    corruption = [m.corruption_warning for m in final]

    return {
        "noise_type": noise_type,
        "alpha": alpha,
        "n_steps": n_steps,
        "psi_mean": float(np.mean(psi_values)),
        "psi_max": float(np.max(psi_values)),
        "phi_mean": float(np.mean(phi_values)),
        "ci_mean": float(np.mean(ci_values)),
        "B_mean": float(np.mean(b_values)),
        "supercritical_ratio": float(np.mean(supercritical)),
        "final_corruption": corruption[-1],
        "psi_series": [m.psi_raw for m in history],
        "phi_series": [m.phi_global for m in history],
    }


def _empty_result(noise_type: str, alpha: float, n_steps: int) -> dict:
    return {
        "noise_type": noise_type, "alpha": alpha, "n_steps": n_steps,
        "psi_mean": 0.0, "psi_max": 0.0, "phi_mean": 0.0, "ci_mean": 0.0,
        "B_mean": 0.0, "supercritical_ratio": 0.0, "final_corruption": "N/A",
        "psi_series": [], "phi_series": [],
    }


def print_heatmap(results: list[dict]) -> None:
    """Print ASCII heatmap of (noise_type x alpha) -> Psi."""
    noise_types = sorted(set(r["noise_type"] for r in results))
    alphas = sorted(set(r["alpha"] for r in results))

    lookup = {(r["noise_type"], r["alpha"]): r for r in results}

    # Header
    print(f"\n{'':>12}", end="")
    for a in alphas:
        print(f"{'a=' + str(a):>10}", end="")
    print()
    print("-" * (12 + 10 * len(alphas)))

    # Psi values
    print("\nPsi (mean):")
    for nt in noise_types:
        print(f"{nt:>12}", end="")
        for a in alphas:
            r = lookup.get((nt, a))
            val = r["psi_mean"] if r else 0.0
            print(f"{val:>10.4f}", end="")
        print()

    # Supercritical ratio
    print("\nSupercritical %:")
    for nt in noise_types:
        print(f"{nt:>12}", end="")
        for a in alphas:
            r = lookup.get((nt, a))
            val = r["supercritical_ratio"] * 100 if r else 0.0
            print(f"{val:>9.1f}%", end="")
        print()

    # Phi values
    print("\nPhi (mean):")
    for nt in noise_types:
        print(f"{nt:>12}", end="")
        for a in alphas:
            r = lookup.get((nt, a))
            val = r["phi_mean"] if r else 0.0
            print(f"{val:>10.4f}", end="")
        print()

    # B factor
    print("\nB factor (mean):")
    for nt in noise_types:
        print(f"{nt:>12}", end="")
        for a in alphas:
            r = lookup.get((nt, a))
            val = r["B_mean"] if r else 0.0
            print(f"{val:>10.4f}", end="")
        print()


def detect_critical_alpha(results: list[dict]) -> None:
    """Find the alpha at which each noise type becomes supercritical."""
    noise_types = sorted(set(r["noise_type"] for r in results))

    print("\nCritical Alpha (first supercritical > 50%):")
    print("-" * 50)
    for nt in noise_types:
        nt_results = sorted(
            [r for r in results if r["noise_type"] == nt],
            key=lambda r: r["alpha"],
        )
        critical = None
        for r in nt_results:
            if r["supercritical_ratio"] > 0.5:
                critical = r["alpha"]
                break
        if critical:
            print(f"  {nt:>8}: alpha_c = {critical:.1f}")
        else:
            print(f"  {nt:>8}: never supercritical (needs higher alpha)")


def analyze_time_series(results: list[dict]) -> None:
    """Analyze Psi time series for phase transition signatures."""
    print("\nPhase Transition Signatures:")
    print("-" * 50)
    for r in results:
        psi = np.array(r["psi_series"])
        if len(psi) < 100:
            continue
        # Check for onset: first sustained Psi > 0
        nonzero = np.where(psi > 0)[0]
        onset = int(nonzero[0]) if len(nonzero) > 0 else -1
        # Check for fluctuations in final quarter
        final = psi[-len(psi) // 4:]
        std = float(final.std()) if len(final) > 0 else 0.0
        if onset >= 0 or r["psi_mean"] > 0:
            print(
                f"  {r['noise_type']:>8} a={r['alpha']:.1f}: "
                f"onset=step {onset}, final_std={std:.4f}, "
                f"psi_final={float(final.mean()) if len(final) else 0:.4f}"
            )


def main():
    NOISE_TYPES = ["white", "pink", "brown"]
    ALPHAS = [0.5, 1.0, 1.5, 2.0, 2.5]
    N_STEPS = 500
    N_CELLS = 60

    total = len(NOISE_TYPES) * len(ALPHAS)
    print("Experiment: Phase Transitions in Hierarchical Consciousness")
    print(f"Formal equation: Psi = B^3 + Phi (phase transition at Phi = Phi_c)")
    print(f"Grid: {len(NOISE_TYPES)} noise types x {len(ALPHAS)} alpha values = {total} conditions")
    print(f"Steps per condition: {N_STEPS}, Cells: {N_CELLS}")
    print()

    results = []
    idx = 0
    for noise_type in NOISE_TYPES:
        for alpha in ALPHAS:
            idx += 1
            print(f"[{idx}/{total}] noise={noise_type:>5}, alpha={alpha:.1f}...", end=" ", flush=True)
            r = run_sweep(noise_type, alpha, n_steps=N_STEPS, n_cells=N_CELLS)
            results.append(r)
            supercrit = "SUPERCRITICAL" if r["supercritical_ratio"] > 0.5 else "subcritical"
            print(f"Psi={r['psi_mean']:.4f} B={r['B_mean']:.4f} [{supercrit}]")

    print_heatmap(results)
    detect_critical_alpha(results)
    analyze_time_series(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for nt in NOISE_TYPES:
        nt_results = [r for r in results if r["noise_type"] == nt]
        max_psi = max(r["psi_mean"] for r in nt_results)
        best_alpha = max(nt_results, key=lambda r: r["psi_mean"])["alpha"]
        print(f"  {nt:>8}: peak Psi={max_psi:.4f} at alpha={best_alpha:.1f}")

    print("\nExperiment complete.")
    return results


if __name__ == "__main__":
    main()
