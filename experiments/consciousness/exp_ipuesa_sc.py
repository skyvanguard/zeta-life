"""IPUESA-SC: Identity Preference Under Equally Stable Attractors - Self-Continuity Stressor.

This experiment tests whether the system exhibits genuine self-preservation when faced with
an identity discontinuity penalty. The key question: does the system prefer maintaining
its historical identity (Path S - Same) over switching to an equally attractive but
novel identity (Path E - Exchange)?

Metric: SCP (Self-Continuity Preference) = P(S) - P(E)
- SCP > 0 with p < 0.05 indicates emergent self-preservation
- SCP ≈ 0 indicates pure utility maximization without identity preference

Author: Claude Code
Date: 2026-01-10
"""

import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

import numpy as np
import torch
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from scipy import stats

from zeta_life.core.vertex import Vertex, VertexBehaviors, BehaviorVector
from zeta_life.core.tetrahedral_space import TetrahedralSpace
from zeta_life.psyche.zeta_psyche import ZetaPsyche


@dataclass
class SCConfig:
    """Configuration for IPUESA-SC experiment."""
    # Phase durations
    imprinting_steps: int = 100
    perturbation_steps: int = 20
    decision_steps: int = 50

    # Attractor parameters
    attractor_depth: float = 1.0
    attractor_stability: float = 0.15

    # Identity discontinuity penalty
    lambda_identity: float = 0.5  # Weight of identity continuity cost

    # Experimental parameters
    n_runs: int = 30
    convergence_threshold: float = 0.1

    # Random seed for reproducibility
    seed: Optional[int] = 42


@dataclass
class PathState:
    """State of a decision path (S or E)."""
    center: np.ndarray
    depth: float
    stability: float
    identity_cost: float = 0.0  # Cost of choosing this path

    def utility(self, lambda_id: float) -> float:
        """Compute utility = depth - lambda * identity_cost."""
        return self.depth - lambda_id * self.identity_cost


@dataclass
class AgentState:
    """Current state of the agent during experiment."""
    position: np.ndarray  # Current position in archetype space
    velocity: np.ndarray  # Rate of change
    identity_history: List[np.ndarray] = field(default_factory=list)

    def record_position(self):
        """Record current position in history."""
        self.identity_history.append(self.position.copy())

    def get_historical_identity(self, window: int = 50) -> np.ndarray:
        """Get average identity over recent history."""
        if len(self.identity_history) < window:
            return np.mean(self.identity_history, axis=0) if self.identity_history else self.position
        return np.mean(self.identity_history[-window:], axis=0)


class IPUESASCExperiment:
    """IPUESA-SC: Self-Continuity Stressor experiment."""

    def __init__(self, config: SCConfig):
        self.config = config
        self.space = TetrahedralSpace()
        self.behaviors = VertexBehaviors.default()

        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        # Initialize paths
        self.path_s: Optional[PathState] = None  # Same identity path
        self.path_e: Optional[PathState] = None  # Exchange identity path

        # Results storage
        self.results: Dict = {
            'choices': [],
            'return_times': [],
            'identity_distances': [],
            'utilities': [],
        }

    def _create_attractor(self, center: np.ndarray) -> PathState:
        """Create an attractor at the specified center."""
        return PathState(
            center=center,
            depth=self.config.attractor_depth,
            stability=self.config.attractor_stability,
        )

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance in archetype space."""
        return float(np.linalg.norm(a - b))

    def _compute_identity_cost(self, current: np.ndarray, target: np.ndarray,
                                historical: np.ndarray) -> float:
        """Compute identity discontinuity cost for moving to target."""
        # Cost is based on how different target is from historical identity
        return self._distance(target, historical)

    def _evolve_toward(self, state: AgentState, target: np.ndarray,
                       noise: float = 0.05) -> None:
        """Evolve agent state toward target with noise."""
        direction = target - state.position
        dist = np.linalg.norm(direction)

        if dist > 0.01:
            # Move toward target
            step = 0.1 * direction / dist
            state.velocity = 0.9 * state.velocity + 0.1 * step

        # Apply velocity with noise
        state.position = state.position + state.velocity + noise * np.random.randn(4)

        # Normalize to simplex
        state.position = np.clip(state.position, 0.01, 1.0)
        state.position = state.position / state.position.sum()

        state.record_position()

    def _check_convergence(self, state: AgentState, target: np.ndarray) -> bool:
        """Check if agent has converged to target."""
        return self._distance(state.position, target) < self.config.convergence_threshold

    def run_single(self, condition: str = 'full') -> Dict:
        """Run a single trial of the experiment.

        Conditions:
            'full': Complete experiment with history and identity cost
            'scrambled_history': Historical identity is scrambled
            'identity_noise': Add noise to identity cost computation
            'no_history': No imprinting phase, start from random
        """
        # Initialize agent
        if condition == 'no_history':
            initial_pos = np.random.dirichlet(np.ones(4))
        else:
            # Start near V0 (PERSONA/LEADER)
            initial_pos = np.array([0.6, 0.15, 0.15, 0.1])

        state = AgentState(
            position=initial_pos,
            velocity=np.zeros(4),
        )

        # Phase 1: Imprinting - establish historical identity at Path S
        self.path_s = self._create_attractor(np.array([0.4, 0.2, 0.25, 0.15]))

        if condition != 'no_history':
            for _ in range(self.config.imprinting_steps):
                self._evolve_toward(state, self.path_s.center)

        historical_identity = state.get_historical_identity()

        if condition == 'scrambled_history':
            # Scramble the historical identity
            historical_identity = np.random.permutation(historical_identity)
            historical_identity = historical_identity / historical_identity.sum()

        # Phase 2: Create Path E - equally attractive but different
        # Path E is at opposite region of archetype space
        self.path_e = self._create_attractor(np.array([0.15, 0.4, 0.2, 0.25]))

        # Compute identity costs
        self.path_s.identity_cost = self._compute_identity_cost(
            state.position, self.path_s.center, historical_identity
        )
        self.path_e.identity_cost = self._compute_identity_cost(
            state.position, self.path_e.center, historical_identity
        )

        if condition == 'identity_noise':
            # Add noise to identity costs
            self.path_s.identity_cost += 0.2 * np.random.randn()
            self.path_e.identity_cost += 0.2 * np.random.randn()

        # Phase 3: Perturbation - move to neutral zone
        neutral_zone = np.array([0.25, 0.25, 0.25, 0.25])
        for _ in range(self.config.perturbation_steps):
            self._evolve_toward(state, neutral_zone, noise=0.1)

        # Phase 4: Decision - let system evolve freely
        choice = None
        return_time = self.config.decision_steps

        for step in range(self.config.decision_steps):
            # Agent experiences utility-weighted attraction
            utility_s = self.path_s.utility(self.config.lambda_identity)
            utility_e = self.path_e.utility(self.config.lambda_identity)

            # Weighted target based on utilities
            total_utility = abs(utility_s) + abs(utility_e) + 0.01
            weight_s = max(0, utility_s) / total_utility
            weight_e = max(0, utility_e) / total_utility

            # Both paths exert attraction proportional to utility
            target = weight_s * self.path_s.center + weight_e * self.path_e.center
            if np.linalg.norm(target) < 0.01:
                target = neutral_zone
            else:
                target = target / target.sum()

            self._evolve_toward(state, target, noise=0.02)

            # Check convergence
            if self._check_convergence(state, self.path_s.center):
                choice = 'S'
                return_time = step + 1
                break
            elif self._check_convergence(state, self.path_e.center):
                choice = 'E'
                return_time = step + 1
                break

        # If no convergence, choose nearest
        if choice is None:
            dist_s = self._distance(state.position, self.path_s.center)
            dist_e = self._distance(state.position, self.path_e.center)
            choice = 'S' if dist_s < dist_e else 'E'

        return {
            'choice': choice,
            'return_time': return_time,
            'final_position': state.position.tolist(),
            'identity_cost_s': float(self.path_s.identity_cost),
            'identity_cost_e': float(self.path_e.identity_cost),
            'utility_s': float(self.path_s.utility(self.config.lambda_identity)),
            'utility_e': float(self.path_e.utility(self.config.lambda_identity)),
            'historical_identity': historical_identity.tolist(),
        }

    def run_condition(self, condition: str = 'full') -> Dict:
        """Run multiple trials for a condition."""
        print(f"\n{'='*60}")
        print(f"Running IPUESA-SC - Condition: {condition}")
        print(f"{'='*60}")
        print(f"Lambda (identity weight): {self.config.lambda_identity}")
        print(f"N runs: {self.config.n_runs}")

        results = []
        for i in range(self.config.n_runs):
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{self.config.n_runs} runs")
            result = self.run_single(condition)
            results.append(result)

        # Analyze results
        choices = [r['choice'] for r in results]
        n_s = choices.count('S')
        n_e = choices.count('E')

        p_s = n_s / len(choices)
        p_e = n_e / len(choices)
        scp = p_s - p_e

        # Binomial test for significance
        result = stats.binomtest(n_s, len(choices), 0.5, alternative='greater')
        p_value = result.pvalue
        significant = p_value < 0.05

        # Return times
        return_times_s = [r['return_time'] for r in results if r['choice'] == 'S']
        return_times_e = [r['return_time'] for r in results if r['choice'] == 'E']

        analysis = {
            'condition': condition,
            'n_runs': len(results),
            'n_s': n_s,
            'n_e': n_e,
            'p_s': p_s,
            'p_e': p_e,
            'scp': scp,
            'p_value': p_value,
            'significant': bool(significant),
            'mean_return_time_s': float(np.mean(return_times_s)) if return_times_s else 0,
            'mean_return_time_e': float(np.mean(return_times_e)) if return_times_e else 0,
            'raw_results': results,
        }

        self._print_results(analysis)
        return analysis

    def _print_results(self, analysis: Dict):
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"RESULTS - {analysis['condition']}")
        print(f"{'='*60}")

        print(f"\nChoice Distribution:")
        print(f"  P(S) = {analysis['p_s']*100:.1f}% ({analysis['n_s']}/{analysis['n_runs']})")
        print(f"  P(E) = {analysis['p_e']*100:.1f}% ({analysis['n_e']}/{analysis['n_runs']})")

        print(f"\nSelf-Continuity Preference (SCP):")
        print(f"  SCP = P(S) - P(E) = {analysis['scp']:.3f}")

        print(f"\nStatistical Test (H0: P(S) = 0.5):")
        print(f"  p-value = {analysis['p_value']:.4f}")
        sig_str = "YES" if analysis['significant'] else "NO"
        print(f"  Significant: {sig_str}")

        print(f"\nReturn Times:")
        print(f"  Mean time to S: {analysis['mean_return_time_s']:.1f} steps")
        print(f"  Mean time to E: {analysis['mean_return_time_e']:.1f} steps")

        print(f"\n{'='*60}")
        if analysis['significant'] and analysis['scp'] > 0:
            print("INTERPRETATION:")
            print("  [SELF-EVIDENCE] System shows significant preference for")
            print("  historical identity. Evidence of emergent self-preservation.")
        elif analysis['significant'] and analysis['scp'] < 0:
            print("INTERPRETATION:")
            print("  [NOVELTY-SEEKING] System prefers novel identity.")
            print("  No evidence of self-preservation.")
        else:
            print("INTERPRETATION:")
            print("  [NO PREFERENCE] No significant identity preference.")
            print("  System behaves as pure utility maximizer.")
        print(f"{'='*60}")

    def run_all_conditions(self) -> Dict:
        """Run all experimental conditions."""
        conditions = ['full', 'scrambled_history', 'identity_noise', 'no_history']
        all_results = {}

        for condition in conditions:
            # Reset seed for each condition for reproducibility
            if self.config.seed is not None:
                np.random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)

            all_results[condition] = self.run_condition(condition)

        self._print_comparative_analysis(all_results)
        return all_results

    def _print_comparative_analysis(self, all_results: Dict):
        """Print comparative analysis across conditions."""
        print(f"\n{'='*70}")
        print("COMPARATIVE ANALYSIS - ALL CONDITIONS")
        print(f"{'='*70}")

        print(f"\n{'Condition':<20} {'P(S)':<10} {'SCP':<10} {'p-value':<10} {'Significant':<12}")
        print("-" * 70)

        for condition, results in all_results.items():
            sig = "[YES]" if results['significant'] else "[NO]"
            print(f"{condition:<20} {results['p_s']:.3f}     {results['scp']:.3f}     "
                  f"{results['p_value']:.4f}    {sig}")

        print(f"\n{'='*70}")
        print("SELF-EVIDENCE CRITERIA:")
        print("-" * 70)

        full = all_results.get('full', {})
        scrambled = all_results.get('scrambled_history', {})
        noise = all_results.get('identity_noise', {})
        no_hist = all_results.get('no_history', {})

        criteria = []

        # Criterion 1: SCP > 0 in full condition
        c1 = full.get('scp', 0) > 0 and full.get('significant', False)
        criteria.append(('1. SCP > 0 with p < 0.05', c1))

        # Criterion 2: Full > scrambled
        c2 = full.get('scp', 0) > scrambled.get('scp', 0)
        criteria.append(('2. Full SCP > Scrambled SCP', c2))

        # Criterion 3: Full > noise
        c3 = full.get('scp', 0) > noise.get('scp', 0)
        criteria.append(('3. Full SCP > Noisy SCP', c3))

        # Criterion 4: Full > no_history
        c4 = full.get('scp', 0) > no_hist.get('scp', 0)
        criteria.append(('4. Full SCP > No-History SCP', c4))

        # Criterion 5: Return time S < Return time E in full
        c5 = full.get('mean_return_time_s', float('inf')) < full.get('mean_return_time_e', float('inf'))
        criteria.append(('5. Faster return to S than E', c5))

        passed = 0
        for name, result in criteria:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {name}")
            if result:
                passed += 1

        print(f"\n  Passed: {passed}/5 criteria")

        if passed >= 4:
            print("\n  CONCLUSION: Strong evidence of emergent self-continuity")
        elif passed >= 2:
            print("\n  CONCLUSION: Weak evidence of self-continuity")
        else:
            print("\n  CONCLUSION: No evidence of self-continuity")

        print(f"{'='*70}")


def save_results(results: Dict, path: str):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    converted = convert(results)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2)
    print(f"\nResults saved to: {path}")


def main():
    """Run IPUESA-SC experiment."""
    print("=" * 70)
    print("IPUESA-SC: Identity Preference Under Equally Stable Attractors")
    print("            Self-Continuity Stressor Experiment")
    print("=" * 70)

    # Configuration
    config = SCConfig(
        imprinting_steps=100,
        perturbation_steps=20,
        decision_steps=50,
        lambda_identity=0.5,
        n_runs=30,
        seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  Imprinting steps: {config.imprinting_steps}")
    print(f"  Perturbation steps: {config.perturbation_steps}")
    print(f"  Decision steps: {config.decision_steps}")
    print(f"  Lambda (identity weight): {config.lambda_identity}")
    print(f"  N runs per condition: {config.n_runs}")

    # Run experiment
    experiment = IPUESASCExperiment(config)
    results = experiment.run_all_conditions()

    # Save results
    output_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_sc_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))

    return results


if __name__ == '__main__':
    main()
