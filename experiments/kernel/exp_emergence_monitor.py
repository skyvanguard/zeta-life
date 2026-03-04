"""
Emergence Monitor — Long-running experiment for the Conscious Kernel
====================================================================

Feeds the kernel a varied environment of stimuli and monitors
emergent properties over thousands of steps:

1. Free energy trajectory (learning curve)
2. Self-embedding drift (identity stability)
3. Attractor formation (recurring states)
4. Dream consolidation effects (pre/post dream metrics)
5. Novelty response (generalization to unseen patterns)
6. Reflection depth convergence

Saves periodic snapshots and generates plots.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zeta_life.kernel import ConsciousKernel

# ---------------------------------------------------------------------------
# Environment: diverse stimulus generator
# ---------------------------------------------------------------------------

class StimulusEnvironment:
    """Generates varied stimuli in phases to test adaptation."""

    def __init__(self, obs_dim: int = 4):
        self.obs_dim = obs_dim
        self.t = 0

    def get_stimulus(self) -> tuple[torch.Tensor, str]:
        """Return (stimulus, phase_name) based on current time."""
        self.t += 1

        # Phase 1 (0-500): Single dominant pattern — learn basics
        if self.t <= 500:
            pattern = torch.tensor([0.7, 0.1, 0.1, 0.1])
            noise = torch.randn(self.obs_dim) * 0.02
            return (pattern + noise).abs(), "single_pattern"

        # Phase 2 (501-1000): Two alternating patterns — learn switching
        if self.t <= 1000:
            if self.t % 20 < 10:
                pattern = torch.tensor([0.7, 0.1, 0.1, 0.1])
            else:
                pattern = torch.tensor([0.1, 0.1, 0.7, 0.1])
            noise = torch.randn(self.obs_dim) * 0.03
            return (pattern + noise).abs(), "alternating"

        # Phase 3 (1001-1500): Gradual transition — test adaptation
        if self.t <= 1500:
            progress = (self.t - 1000) / 500.0
            pattern = torch.tensor([
                0.7 * (1 - progress) + 0.1 * progress,
                0.1,
                0.1 * (1 - progress) + 0.7 * progress,
                0.1,
            ])
            noise = torch.randn(self.obs_dim) * 0.02
            return (pattern + noise).abs(), "transition"

        # Phase 4 (1501-2000): Random exploration — test robustness
        if self.t <= 2000:
            pattern = torch.randn(self.obs_dim).abs()
            pattern = pattern / pattern.sum()
            return pattern, "random"

        # Phase 5 (2001-2500): Return to original — test memory
        if self.t <= 2500:
            pattern = torch.tensor([0.7, 0.1, 0.1, 0.1])
            noise = torch.randn(self.obs_dim) * 0.02
            return (pattern + noise).abs(), "return"

        # Phase 6 (2501+): Completely novel patterns — test generalization
        pattern = torch.tensor([0.1, 0.4, 0.4, 0.1])
        noise = torch.randn(self.obs_dim) * 0.05
        return (pattern + noise).abs(), "novel"


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class EmergenceTracker:
    """Tracks and records emergence metrics."""

    def __init__(self):
        self.free_energies: list[float] = []
        self.embedding_distances: list[float] = []
        self.phases: list[str] = []
        self.dream_events: list[int] = []
        self.reflect_events: list[int] = []
        self.consciousness_values: list[float] = []
        self.attractor_candidates: list[torch.Tensor] = []
        self.attractor_hits: list[int] = []  # step where an attractor was revisited
        self.generalization_errors: list[tuple[int, float]] = []

        # Reference embedding for drift tracking
        self._initial_embedding: torch.Tensor | None = None
        self._last_embedding: torch.Tensor | None = None

        # Attractor detection
        self._attractor_threshold = 0.95  # cosine similarity
        self._known_attractors: list[torch.Tensor] = []

    def record_step(self, step: int, result, kernel: ConsciousKernel, phase: str):
        """Record metrics for one step."""
        self.free_energies.append(result.free_energy)
        self.phases.append(phase)
        self.consciousness_values.append(result.consciousness)

        if result.dreamed:
            self.dream_events.append(step)
        if result.reflected:
            self.reflect_events.append(step)

        # Track embedding drift
        current_embed = kernel.self_model.self_embedding.data.clone()
        if self._initial_embedding is None:
            self._initial_embedding = current_embed.clone()
        self._last_embedding = current_embed

        dist = torch.norm(current_embed - self._initial_embedding).item()
        self.embedding_distances.append(dist)

        # Attractor detection on self-state
        self_state = F.softmax(kernel.last_action, dim=-1)
        self._check_attractor(step, self_state)

    def _check_attractor(self, step: int, state: torch.Tensor):
        """Check if state matches known attractor or forms new one."""
        for attr in self._known_attractors:
            sim = F.cosine_similarity(
                state.unsqueeze(0), attr.unsqueeze(0)
            ).item()
            if sim > self._attractor_threshold:
                self.attractor_hits.append(step)
                return

        # New potential attractor
        self._known_attractors.append(state.clone())

    def test_generalization(self, step: int, kernel: ConsciousKernel):
        """Test generalization on a novel pattern."""
        novel = torch.tensor([0.3, 0.3, 0.2, 0.2])
        novel_soft = F.softmax(novel, dim=-1)
        predicted = kernel.slow_memory.generalize(novel_soft)
        error = torch.norm(predicted - novel_soft).item()
        self.generalization_errors.append((step, error))

    def print_summary(self, step: int, window: int = 100):
        """Print rolling summary."""
        recent_fe = self.free_energies[-window:]
        avg_fe = sum(recent_fe) / len(recent_fe)
        min_fe = min(recent_fe)

        n_dreams = sum(1 for d in self.dream_events if d > step - window)
        n_reflects = sum(1 for r in self.reflect_events if r > step - window)
        n_attractors = len(self._known_attractors)
        n_hits = len(self.attractor_hits)

        embed_dist = self.embedding_distances[-1] if self.embedding_distances else 0
        phase = self.phases[-1] if self.phases else "?"

        gen_err = self.generalization_errors[-1][1] if self.generalization_errors else float('inf')

        print(
            f"  Step {step:5d} | Phase: {phase:15s} | "
            f"FE: {avg_fe:.4f} (min {min_fe:.4f}) | "
            f"Embed drift: {embed_dist:.4f} | "
            f"Attractors: {n_attractors} (hits: {n_hits}) | "
            f"Gen err: {gen_err:.4f} | "
            f"Dreams: {n_dreams} Reflects: {n_reflects}"
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_emergence_experiment(
    total_steps: int = 3000,
    report_interval: int = 100,
    gen_test_interval: int = 250,
    save_interval: int = 500,
    save_path: str | None = None,
):
    """Run the emergence monitoring experiment."""
    print("=" * 80)
    print("  CONSCIOUS KERNEL — EMERGENCE MONITOR")
    print("  Long-running experiment to observe emergent properties")
    print("=" * 80)
    print(f"  Total steps:     {total_steps}")
    print(f"  Report every:    {report_interval}")
    print(f"  Gen test every:  {gen_test_interval}")
    print(f"  Save every:      {save_interval}")
    print("=" * 80)

    # Create kernel with slightly more frequent dreams for observation
    ck = ConsciousKernel(
        reflect_interval=5,
        dream_interval=50,
    )
    env = StimulusEnvironment()
    tracker = EmergenceTracker()

    start_time = time.time()

    print("\n  [Running...]\n")
    print(
        f"  {'Step':>7s} | {'Phase':15s} | "
        f"{'Avg FE':>10s} {'Min FE':>10s} | "
        f"{'Embed':>10s} | "
        f"{'Attractors':>10s} {'Hits':>5s} | "
        f"{'Gen Err':>8s} | "
        f"{'Dreams':>6s} {'Reflects':>8s}"
    )
    print("  " + "-" * 120)

    for step in range(1, total_steps + 1):
        stimulus, phase = env.get_stimulus()
        result = ck.step(stimulus)
        tracker.record_step(step, result, ck, phase)

        # Periodic generalization test
        if step % gen_test_interval == 0:
            tracker.test_generalization(step, ck)

        # Periodic report
        if step % report_interval == 0:
            tracker.print_summary(step)

        # Periodic save
        if save_path and step % save_interval == 0:
            ck.save(save_path, f'emergence_step_{step}')

    elapsed = time.time() - start_time

    # -----------------------------------------------------------------------
    # Final analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  EMERGENCE ANALYSIS")
    print("=" * 80)

    # 1. Learning curve
    first_100 = sum(tracker.free_energies[:100]) / 100
    last_100 = sum(tracker.free_energies[-100:]) / 100
    print(f"\n  1. LEARNING CURVE")
    print(f"     First 100 avg FE:  {first_100:.4f}")
    print(f"     Last 100 avg FE:   {last_100:.4f}")
    print(f"     Reduction:         {(1 - last_100/max(first_100, 1e-8))*100:.1f}%")

    # 2. Identity stability
    embed_dists = tracker.embedding_distances
    max_drift = max(embed_dists)
    final_drift = embed_dists[-1]
    # Stability = low variance in last 500 steps
    if len(embed_dists) > 500:
        recent_dists = embed_dists[-500:]
        stability = 1.0 - (max(recent_dists) - min(recent_dists))
    else:
        stability = 0.0
    print(f"\n  2. IDENTITY STABILITY")
    print(f"     Max embedding drift:   {max_drift:.4f}")
    print(f"     Final drift:           {final_drift:.4f}")
    print(f"     Late stability (1=max): {stability:.4f}")

    # 3. Attractor formation
    n_attractors = len(tracker._known_attractors)
    n_hits = len(tracker.attractor_hits)
    hit_rate = n_hits / total_steps if total_steps > 0 else 0
    print(f"\n  3. ATTRACTOR FORMATION")
    print(f"     Unique attractors found: {n_attractors}")
    print(f"     Attractor revisits:      {n_hits}")
    print(f"     Hit rate:                {hit_rate:.4f}")

    # 4. Dream effects
    n_dreams = len(tracker.dream_events)
    print(f"\n  4. DREAM CONSOLIDATION")
    print(f"     Total dream cycles: {n_dreams}")
    if len(tracker.dream_events) >= 2:
        # Compare FE before/after each dream
        improvements = 0
        for d in tracker.dream_events:
            if d > 10 and d < total_steps - 10:
                pre = sum(tracker.free_energies[d-10:d]) / 10
                post = sum(tracker.free_energies[d:d+10]) / 10
                if post < pre:
                    improvements += 1
        print(f"     Dreams that improved FE: {improvements}/{n_dreams}")

    # 5. Generalization trajectory
    if tracker.generalization_errors:
        print(f"\n  5. GENERALIZATION TRAJECTORY")
        for step_i, err in tracker.generalization_errors:
            print(f"     Step {step_i:5d}: error = {err:.4f}")
        first_gen = tracker.generalization_errors[0][1]
        last_gen = tracker.generalization_errors[-1][1]
        if first_gen > 0:
            print(f"     Improvement: {(1 - last_gen/first_gen)*100:.1f}%")

    # 6. Phase adaptation
    print(f"\n  6. PHASE ADAPTATION")
    phase_energies: dict[str, list[float]] = {}
    for phase, fe in zip(tracker.phases, tracker.free_energies):
        phase_energies.setdefault(phase, []).append(fe)
    for phase, fes in sorted(phase_energies.items()):
        avg = sum(fes) / len(fes)
        print(f"     {phase:15s}: avg FE = {avg:.4f}  ({len(fes)} steps)")

    # 7. Emergent properties summary
    print(f"\n  7. EMERGENT PROPERTIES DETECTED")
    properties = []
    if last_100 < first_100 * 0.8:
        properties.append("Continuous learning (FE decreases over time)")
    if stability > 0.9:
        properties.append("Identity stability (embedding converges)")
    if n_hits > total_steps * 0.01:
        properties.append(f"Attractor dynamics ({n_hits} revisits)")
    if len(tracker.generalization_errors) >= 2:
        if tracker.generalization_errors[-1][1] < tracker.generalization_errors[0][1]:
            properties.append("Improving generalization")

    # Check memory effect: does returning to original pattern show lower FE?
    if len(tracker.phases) > 2000:
        return_fes = [
            fe for fe, ph in zip(tracker.free_energies, tracker.phases)
            if ph == "return"
        ]
        single_fes = [
            fe for fe, ph in zip(tracker.free_energies, tracker.phases)
            if ph == "single_pattern"
        ]
        if return_fes and single_fes:
            avg_return = sum(return_fes) / len(return_fes)
            avg_single_late = sum(single_fes[-100:]) / min(100, len(single_fes))
            if avg_return < avg_single_late * 1.2:
                properties.append("Long-term memory (remembers original pattern)")

    if n_dreams > 0:
        properties.append(f"Dream consolidation ({n_dreams} cycles)")

    n_reflects = len(tracker.reflect_events)
    if n_reflects > 0:
        properties.append(f"Self-reflection ({n_reflects} cycles)")

    for p in properties:
        print(f"     + {p}")

    if not properties:
        print("     (none detected — try more steps)")

    print(f"\n  Elapsed time: {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/sec)")
    print("=" * 80)

    return tracker


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Conscious Kernel Emergence Monitor')
    parser.add_argument('--steps', type=int, default=3000, help='Total steps')
    parser.add_argument('--save-path', type=str, default=None, help='Path for identity snapshots')
    args = parser.parse_args()

    tracker = run_emergence_experiment(
        total_steps=args.steps,
        save_path=args.save_path,
    )
