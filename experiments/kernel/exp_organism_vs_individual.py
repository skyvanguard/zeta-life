"""
Organism vs Individual — Comparative experiment
=================================================

Compares a ConsciousOrganism (multi-agent) against a single
ConsciousKernel receiving the same stimuli sequence.

Measures:
1. Free energy trajectory (learning quality)
2. Action diversity (behavioral richness)
3. Adaptation speed to phase changes
4. Final consciousness metrics

Success criterion from design doc:
    "organism free energy < individual kernel FE"
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zeta_life.kernel import ConsciousKernel, ConsciousOrganism
from zeta_life.kernel.spawn_controller import SpawnEvent, MergeEvent, DeathEvent


# ---------------------------------------------------------------------------
# Shared stimulus sequence
# ---------------------------------------------------------------------------

class StimulusEnvironment:
    """Deterministic stimulus sequence for fair comparison."""

    def __init__(self, obs_dim: int = 4, seed: int = 42):
        self.obs_dim = obs_dim
        self.rng = torch.Generator().manual_seed(seed)
        self.t = 0

    def get_stimulus(self) -> tuple[torch.Tensor, str]:
        self.t += 1

        if self.t <= 2000:
            pattern = torch.tensor([0.7, 0.1, 0.1, 0.1])
            noise = torch.randn(self.obs_dim, generator=self.rng) * 0.02
            return (pattern + noise).abs(), "stable"

        if self.t <= 5000:
            if self.t % 100 < 50:
                pattern = torch.tensor([0.6, 0.2, 0.1, 0.1])
            else:
                pattern = torch.tensor([0.1, 0.1, 0.6, 0.2])
            noise = torch.randn(self.obs_dim, generator=self.rng) * 0.05
            return (pattern + noise).abs(), "alternating"

        if self.t <= 8000:
            phase = (self.t % 200) / 200.0
            pattern = torch.tensor([
                0.3 + 0.3 * torch.sin(torch.tensor(phase * 6.28)).item(),
                0.3 + 0.3 * torch.cos(torch.tensor(phase * 6.28)).item(),
                0.2, 0.2,
            ])
            noise = torch.randn(self.obs_dim, generator=self.rng) * 0.03
            return (pattern + noise).abs(), "oscillating"

        noise = torch.randn(self.obs_dim, generator=self.rng) * 0.15
        pattern = torch.tensor([0.25, 0.25, 0.25, 0.25])
        return (pattern + noise).abs(), "chaotic"


def run_individual(stimuli: list[torch.Tensor]) -> dict:
    """Run a single ConsciousKernel on the stimulus sequence."""
    ck = ConsciousKernel(obs_dim=4, latent_dim=32, embed_dim=16,
                         reflect_interval=5, dream_interval=50)
    fes = []
    actions = []
    for s in stimuli:
        r = ck.step(s)
        fes.append(r.free_energy)
        actions.append(r.action.detach())
    return {'free_energies': fes, 'actions': actions}


def run_organism(stimuli: list[torch.Tensor]) -> dict:
    """Run a ConsciousOrganism on the stimulus sequence."""
    org = ConsciousOrganism(obs_dim=4, initial_kernels=2, total_energy=10.0)
    fes = []
    actions = []
    consciousnesses = []
    diversities = []
    populations = []
    events_total = {'spawn': 0, 'merge': 0, 'death': 0}

    for s in stimuli:
        r = org.step(s)
        # Use avg free energy across all kernels (collective performance)
        avg_fe = sum(r.free_energies.values()) / max(len(r.free_energies), 1)
        fes.append(avg_fe)
        actions.append(org.gw.broadcast_signal.clone().detach())
        consciousnesses.append(r.psi)
        diversities.append(r.diversity)
        populations.append(r.population)
        for e in r.events:
            if isinstance(e, SpawnEvent):
                events_total['spawn'] += 1
            elif isinstance(e, MergeEvent):
                events_total['merge'] += 1
            elif isinstance(e, DeathEvent):
                events_total['death'] += 1

    return {
        'free_energies': fes,
        'actions': actions,
        'consciousnesses': consciousnesses,
        'diversities': diversities,
        'populations': populations,
        'events': events_total,
    }


def window_avg(data: list[float], start: int, end: int) -> float:
    segment = data[start:end]
    return sum(segment) / max(len(segment), 1)


def action_diversity(actions: list[torch.Tensor], start: int, end: int) -> float:
    """Measure how diverse actions are in a window (std of action vectors)."""
    segment = actions[start:end]
    if len(segment) < 2:
        return 0.0
    stacked = torch.stack(segment)
    return stacked.std(dim=0).mean().item()


def main(n_steps: int = 10000):
    print("=" * 70)
    print("  Organism vs Individual — Comparative Experiment")
    print("=" * 70)

    # Generate stimuli sequence
    print(f"\n  Generating {n_steps} stimuli...")
    env = StimulusEnvironment(seed=42)
    stimuli = []
    phases = []
    for _ in range(n_steps):
        s, phase = env.get_stimulus()
        stimuli.append(s)
        phases.append(phase)

    # Run individual
    print("  Running Individual ConsciousKernel...")
    t0 = time.time()
    ind = run_individual(stimuli)
    t_ind = time.time() - t0
    print(f"    Done in {t_ind:.1f}s ({n_steps/t_ind:.0f} steps/s)")

    # Run organism
    print("  Running ConsciousOrganism...")
    t0 = time.time()
    org = run_organism(stimuli)
    t_org = time.time() - t0
    print(f"    Done in {t_org:.1f}s ({n_steps/t_org:.0f} steps/s)")

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Phase boundaries
    phase_windows = [
        ("stable", 0, 2000),
        ("alternating", 2000, 5000),
        ("oscillating", 5000, 8000),
        ("chaotic", 8000, n_steps),
    ]

    print(f"\n  {'Phase':<15} {'Individual FE':>15} {'Organism FE':>15} {'Winner':>10}")
    print("  " + "-" * 57)

    for phase_name, start, end in phase_windows:
        ind_fe = window_avg(ind['free_energies'], start, end)
        org_fe = window_avg(org['free_energies'], start, end)
        winner = "ORGANISM" if org_fe < ind_fe else "INDIVIDUAL"
        print(f"  {phase_name:<15} {ind_fe:>15.4f} {org_fe:>15.4f} {winner:>10}")

    # Overall
    overall_ind = window_avg(ind['free_energies'], 0, n_steps)
    overall_org = window_avg(org['free_energies'], 0, n_steps)
    winner = "ORGANISM" if overall_org < overall_ind else "INDIVIDUAL"
    print("  " + "-" * 57)
    print(f"  {'OVERALL':<15} {overall_ind:>15.4f} {overall_org:>15.4f} {winner:>10}")

    # Action diversity
    print(f"\n  {'Phase':<15} {'Ind ActionDiv':>15} {'Org ActionDiv':>15}")
    print("  " + "-" * 47)
    for phase_name, start, end in phase_windows:
        ind_ad = action_diversity(ind['actions'], start, end)
        org_ad = action_diversity(org['actions'], start, end)
        print(f"  {phase_name:<15} {ind_ad:>15.4f} {org_ad:>15.4f}")

    # Organism-specific metrics
    print(f"\n  Organism Metrics:")
    print(f"    Avg consciousness: {sum(org['consciousnesses'])/n_steps:.3f}")
    print(f"    Avg diversity:     {sum(org['diversities'])/n_steps:.3f}")
    print(f"    Population range:  {min(org['populations'])}-{max(org['populations'])}")
    print(f"    Events: spawn={org['events']['spawn']} merge={org['events']['merge']} death={org['events']['death']}")

    # Adaptation speed: how quickly FE drops after phase change
    print(f"\n  Adaptation Speed (steps to reach 90% of phase minimum FE):")
    for phase_name, start, end in phase_windows:
        for label, fes in [("individual", ind['free_energies']), ("organism", org['free_energies'])]:
            segment = fes[start:end]
            if not segment:
                continue
            min_fe = min(segment)
            threshold = segment[0] - 0.9 * (segment[0] - min_fe)
            steps_to_adapt = len(segment)
            for i, fe in enumerate(segment):
                if fe <= threshold:
                    steps_to_adapt = i
                    break
            print(f"    {phase_name:<15} {label:<12} {steps_to_adapt:>5} steps")

    # --- Success criteria ---
    print("\n" + "-" * 70)
    print("  SUCCESS CRITERIA")
    print("-" * 70)

    org_wins_fe = overall_org < overall_ind
    print(f"  [{'PASS' if org_wins_fe else 'FAIL'}] Organism FE < Individual FE "
          f"({overall_org:.4f} vs {overall_ind:.4f})")

    # Try to save plot
    try:
        _save_plot(ind, org, n_steps, phase_windows)
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")


def _save_plot(ind, org, n_steps, phase_windows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ConsciousOrganism vs Individual Kernel", fontsize=14)

    x = range(n_steps)

    # Smoothing helper
    def smooth(data, window=200):
        out = []
        for i in range(len(data)):
            start = max(0, i - window)
            out.append(sum(data[start:i+1]) / (i - start + 1))
        return out

    # FE comparison
    axes[0, 0].plot(x, smooth(ind['free_energies']), label='Individual', alpha=0.8)
    axes[0, 0].plot(x, smooth(org['free_energies']), label='Organism', alpha=0.8)
    axes[0, 0].set_title("Free Energy (smoothed)")
    axes[0, 0].legend()
    for _, start, _ in phase_windows[1:]:
        axes[0, 0].axvline(start, color='gray', linestyle='--', alpha=0.3)

    # Consciousness
    axes[0, 1].plot(x, smooth(org['consciousnesses']), color='purple')
    axes[0, 1].set_title("Organism Consciousness Index")

    # Population
    axes[1, 0].plot(x, org['populations'], alpha=0.5, linewidth=0.5)
    axes[1, 0].set_title("Organism Population")

    # Diversity
    axes[1, 1].plot(x, smooth(org['diversities']), color='green')
    axes[1, 1].set_title("Organism Diversity")

    for ax in axes.flat:
        ax.set_xlabel("Step")

    plt.tight_layout()
    out = Path("results") / "organism_vs_individual.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    main(n_steps=args.steps)
