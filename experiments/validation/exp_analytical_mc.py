"""
Experiment: Analytical M_c vs Empirical Goldilocks Zone

Validates the formal equation M_c = F_i / (alpha - C) by comparing
its predictions with empirical Goldilocks zone observations from
evolved IPUESA configurations.

The empirical Goldilocks zone was found at damage_multiplier ~ 3.9x.
This experiment tests whether M_c correctly predicts this sweet spot.

Usage:
    python -m experiments.validation.exp_analytical_mc
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from zeta_life.consciousness.formal_equations import (
    compute_M_c,
    compute_phi_c,
    compute_psi,
    predict_system_stability,
)
from zeta_life.consciousness.hierarchical_simulation import (
    HierarchicalSimulation,
    SimulationConfig,
)


def run_simulation_sweep(
    damage_multipliers: list[float],
    n_steps: int = 100,
    n_cells: int = 80,
) -> list[dict]:
    """Run simulations across damage multiplier range."""
    results = []

    for dm in damage_multipliers:
        config = SimulationConfig(
            n_cells=n_cells,
            n_steps=n_steps,
            enable_resilience=True,
            resilience_preset='optimal',
            use_formal_psi=True,
            consciousness_alpha=1.0,
        )

        # Override damage multiplier in resilience config
        sim = HierarchicalSimulation(config)
        if sim.resilience_config:
            sim.resilience_config['damage']['multiplier'] = dm

        sim.initialize()
        metrics_list = sim.run(verbose=False)

        if metrics_list:
            final = metrics_list[-1]
            peak_consciousness = max(m.consciousness_index for m in metrics_list)
            peak_phi = max(m.phi_global for m in metrics_list)
            final_hs = final.holographic_survival

            # Analytical prediction
            phi_c = compute_phi_c(F_i=peak_phi * 0.5, alpha=1.0, C=0.3)
            M_c = phi_c

            results.append({
                'damage_multiplier': dm,
                'peak_consciousness': peak_consciousness,
                'peak_phi': peak_phi,
                'final_hs': final_hs,
                'phi_c_analytical': phi_c,
                'M_c_analytical': M_c,
                'final_psi_raw': final.psi_raw,
                'corruption_warning': final.corruption_warning,
            })

    return results


def main():
    print("=" * 65)
    print("  EXPERIMENT: Analytical M_c vs Empirical Goldilocks Zone")
    print("  Validating M_c = F_i / (alpha - C)")
    print("=" * 65)

    # Sweep damage multiplier from 1.5 to 5.0
    multipliers = np.arange(1.5, 5.5, 0.5).tolist()

    print(f"\nRunning {len(multipliers)} simulations...")
    print(f"  damage_multiplier range: {multipliers[0]:.1f} - {multipliers[-1]:.1f}")
    print()

    results = run_simulation_sweep(multipliers, n_steps=100)

    # Display results
    print(f"{'DM':>6} {'PeakCI':>8} {'PeakPhi':>8} {'FinalHS':>8} "
          f"{'PsiRaw':>8} {'M_c':>8} {'Status':>12}")
    print("-" * 65)

    best_dm = 0.0
    best_ci = 0.0

    for r in results:
        print(f"{r['damage_multiplier']:>6.1f} "
              f"{r['peak_consciousness']:>8.3f} "
              f"{r['peak_phi']:>8.3f} "
              f"{r['final_hs']:>8.3f} "
              f"{r['final_psi_raw']:>8.3f} "
              f"{r['M_c_analytical']:>8.3f} "
              f"{r['corruption_warning']:>12}")

        if r['peak_consciousness'] > best_ci:
            best_ci = r['peak_consciousness']
            best_dm = r['damage_multiplier']

    print(f"\nEmpirical Goldilocks (known): damage_multiplier ~ 3.9x")
    print(f"Best consciousness found at:  damage_multiplier = {best_dm:.1f}x")
    print(f"  Peak consciousness index: {best_ci:.4f}")

    # Correlation between M_c and peak consciousness
    if len(results) >= 3:
        mc_vals = [r['M_c_analytical'] for r in results]
        ci_vals = [r['peak_consciousness'] for r in results]
        corr = np.corrcoef(mc_vals, ci_vals)[0, 1]
        print(f"\nCorrelation M_c vs peak consciousness: {corr:.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
