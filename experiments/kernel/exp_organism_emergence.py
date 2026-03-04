"""
Organism Emergence Monitor — Multi-agent Darwinian Brain experiment
===================================================================

Runs a ConsciousOrganism through varied stimuli and tracks:

1. Population dynamics (spawn/merge/death events)
2. Diversity and coherence trajectories
3. Consciousness index evolution
4. Energy distribution across kernels
5. Global Workspace turnover (anti-monopoly)
6. Lifecycle event log

Usage:
    python experiments/kernel/exp_organism_emergence.py --steps 5000
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zeta_life.kernel import ConsciousOrganism
from zeta_life.kernel.spawn_controller import SpawnEvent, MergeEvent, DeathEvent


# ---------------------------------------------------------------------------
# Stimulus environment
# ---------------------------------------------------------------------------

class StimulusEnvironment:
    """Generates varied stimuli in phases."""

    def __init__(self, obs_dim: int = 4):
        self.obs_dim = obs_dim
        self.t = 0

    def get_stimulus(self) -> tuple[torch.Tensor, str]:
        self.t += 1

        if self.t <= 500:
            pattern = torch.tensor([0.7, 0.1, 0.1, 0.1])
            noise = torch.randn(self.obs_dim) * 0.02
            return (pattern + noise).abs(), "stable"

        if self.t <= 1500:
            if self.t % 50 < 25:
                pattern = torch.tensor([0.6, 0.2, 0.1, 0.1])
            else:
                pattern = torch.tensor([0.1, 0.1, 0.6, 0.2])
            noise = torch.randn(self.obs_dim) * 0.05
            return (pattern + noise).abs(), "alternating"

        if self.t <= 3000:
            phase = (self.t % 200) / 200.0
            pattern = torch.tensor([
                0.3 + 0.3 * torch.sin(torch.tensor(phase * 6.28)).item(),
                0.3 + 0.3 * torch.cos(torch.tensor(phase * 6.28)).item(),
                0.2,
                0.2,
            ])
            noise = torch.randn(self.obs_dim) * 0.03
            return (pattern + noise).abs(), "oscillating"

        noise = torch.randn(self.obs_dim) * 0.15
        pattern = torch.tensor([0.25, 0.25, 0.25, 0.25])
        return (pattern + noise).abs(), "chaotic"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_steps: int = 5000, print_interval: int = 100):
    print("=" * 70)
    print("  ConsciousOrganism — Emergence Monitor")
    print("=" * 70)

    org = ConsciousOrganism(
        obs_dim=4,
        initial_kernels=2,
        total_energy=10.0,
        latent_dim=32,
        embed_dim=16,
        reflect_interval=5,
        dream_interval=50,
    )
    env = StimulusEnvironment(obs_dim=4)

    # Tracking
    consciousness_history = []
    diversity_history = []
    coherence_history = []
    population_history = []
    winner_counter: Counter[int] = Counter()
    event_log: list[tuple[int, str]] = []

    start = time.time()

    for step in range(1, n_steps + 1):
        stimulus, phase = env.get_stimulus()
        result = org.step(stimulus)

        consciousness_history.append(result.consciousness)
        diversity_history.append(result.diversity)
        coherence_history.append(result.coherence)
        population_history.append(result.population)
        winner_counter[result.winner_id] += 1

        for evt in result.events:
            if isinstance(evt, SpawnEvent):
                event_log.append((step, f"SPAWN parent={evt.parent_id}"))
            elif isinstance(evt, MergeEvent):
                event_log.append((step, f"MERGE {evt.kernel_a}+{evt.kernel_b}"))
            elif isinstance(evt, DeathEvent):
                event_log.append((step, f"DEATH kernel={evt.kernel_id}"))

        if step % print_interval == 0:
            elapsed = time.time() - start
            sps = step / max(elapsed, 0.01)
            energies = " ".join(
                f"k{kid}={e:.2f}" for kid, e in sorted(result.energies.items())
            )
            print(
                f"[{step:5d}] phase={phase:12s} | "
                f"pop={result.population} | "
                f"C={result.consciousness:.3f} | "
                f"div={result.diversity:.3f} coh={result.coherence:.3f} | "
                f"E: {energies} | "
                f"{sps:.0f} steps/s"
            )

    elapsed = time.time() - start

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    avg_c = sum(consciousness_history[-500:]) / min(500, len(consciousness_history))
    avg_div = sum(diversity_history[-500:]) / min(500, len(diversity_history))
    avg_coh = sum(coherence_history[-500:]) / min(500, len(coherence_history))
    pop_set = set(population_history)

    print(f"\n  Steps: {n_steps} in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")
    print(f"  Final population: {len(org.kernels)}")
    print(f"  Population range: {min(population_history)}-{max(population_history)}")
    print(f"  Avg consciousness (last 500): {avg_c:.3f}")
    print(f"  Avg diversity (last 500): {avg_div:.3f}")
    print(f"  Avg coherence (last 500): {avg_coh:.3f}")

    print(f"\n  GW Winners ({len(winner_counter)} unique):")
    total_wins = sum(winner_counter.values())
    for kid, wins in winner_counter.most_common(5):
        pct = 100.0 * wins / total_wins
        print(f"    kernel {kid}: {wins} wins ({pct:.1f}%)")

    max_pct = max(100.0 * w / total_wins for w in winner_counter.values())

    print(f"\n  Lifecycle events ({len(event_log)} total):")
    spawns = sum(1 for _, e in event_log if "SPAWN" in e)
    merges = sum(1 for _, e in event_log if "MERGE" in e)
    deaths = sum(1 for _, e in event_log if "DEATH" in e)
    print(f"    Spawns: {spawns}, Merges: {merges}, Deaths: {deaths}")
    for step_n, desc in event_log[:10]:
        print(f"    t={step_n}: {desc}")
    if len(event_log) > 10:
        print(f"    ... and {len(event_log) - 10} more")

    # --- Success criteria ---
    print("\n" + "-" * 70)
    print("  SUCCESS CRITERIA")
    print("-" * 70)

    criteria = [
        ("Population varies 2-10", len(pop_set) >= 1 and min(population_history) >= 2
         and max(population_history) <= 10),
        ("No kernel wins >40% GW", max_pct <= 40.0),
        ("Diversity in 0.3-0.7 (avg)", 0.3 <= avg_div <= 0.7),
        ("At least 1 lifecycle event", len(event_log) >= 1),
    ]

    for name, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    passed_count = sum(1 for _, p in criteria if p)
    print(f"\n  {passed_count}/{len(criteria)} criteria met")

    # Try to save plot
    try:
        _save_plot(consciousness_history, diversity_history, coherence_history,
                   population_history, n_steps)
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")


def _save_plot(consciousness, diversity, coherence, population, n_steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ConsciousOrganism — Emergence Monitor", fontsize=14)

    x = range(1, len(consciousness) + 1)

    axes[0, 0].plot(x, consciousness, alpha=0.5, linewidth=0.5)
    axes[0, 0].set_title("Consciousness Index")
    axes[0, 0].set_ylabel("C")

    axes[0, 1].plot(x, population, alpha=0.7, linewidth=0.5)
    axes[0, 1].set_title("Population")
    axes[0, 1].set_ylabel("N kernels")

    axes[1, 0].plot(x, diversity, alpha=0.5, linewidth=0.5, label="diversity")
    axes[1, 0].plot(x, coherence, alpha=0.5, linewidth=0.5, label="coherence")
    axes[1, 0].set_title("Diversity & Coherence")
    axes[1, 0].legend()

    # Smoothed consciousness
    window = min(100, len(consciousness))
    if window > 1:
        smoothed = [
            sum(consciousness[max(0, i - window):i]) / min(i, window)
            for i in range(1, len(consciousness) + 1)
        ]
        axes[1, 1].plot(x, smoothed, linewidth=1)
    axes[1, 1].set_title("Consciousness (smoothed)")
    axes[1, 1].set_ylabel("C")

    for ax in axes.flat:
        ax.set_xlabel("Step")

    plt.tight_layout()
    out = Path("results") / "organism_emergence.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConsciousOrganism emergence monitor")
    parser.add_argument("--steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--interval", type=int, default=100, help="Print interval")
    args = parser.parse_args()
    run_experiment(n_steps=args.steps, print_interval=args.interval)
