"""IPUESA-AP: Identity Preference Under Equally Stable Attractors - Anticipatory Preservation.

Extends IPUESA-SC by adding a minimal internal predictor that anticipates future identity
states before executing transitions. The system computes expected identity discontinuity
cost proactively, enabling anticipatory self-preservation.

Key addition: identity_hat(t+1) = f(identity_t, velocity, history)

Metric: ASCP (Anticipatory Self-Continuity Preference) = P(S) - P(E)
Strong self-evidence requires:
  1. ASCP > SCP (anticipatory outperforms reactive)
  2. Statistical significance (p < 0.05)
  3. ASCP scales with lambda
  4. Degradation under scrambled/noisy prediction

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


@dataclass
class APConfig:
    """Configuration for IPUESA-AP experiment."""
    # Phase durations
    imprinting_steps: int = 100
    perturbation_steps: int = 20
    decision_steps: int = 50

    # Attractor parameters
    attractor_depth: float = 1.0
    attractor_stability: float = 0.15

    # Identity discontinuity penalty
    lambda_identity: float = 0.5

    # Prediction parameters
    prediction_horizon: int = 5  # Steps ahead to predict
    history_window: int = 20     # History window for prediction
    prediction_weight: float = 0.3  # Weight of anticipatory vs reactive cost

    # Experimental parameters
    n_runs: int = 30
    convergence_threshold: float = 0.1

    # Lambda sweep for scaling analysis
    lambda_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    # Random seed
    seed: Optional[int] = 42


class IdentityPredictor:
    """Minimal internal model for anticipating future identity states.

    Predicts identity_hat(t+1) using:
    - Current identity state
    - Velocity (rate of change)
    - Historical trajectory trend
    """

    def __init__(self, history_window: int = 20, horizon: int = 5):
        self.history_window = history_window
        self.horizon = horizon
        self.history: List[np.ndarray] = []

    def update(self, state: np.ndarray):
        """Add current state to history."""
        self.history.append(state.copy())
        if len(self.history) > self.history_window * 2:
            self.history = self.history[-self.history_window * 2:]

    def predict(self, current: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Predict identity at t+horizon.

        Uses linear extrapolation weighted by:
        - Immediate velocity (short-term trend)
        - Historical trend (long-term trajectory)
        """
        if len(self.history) < 3:
            # Not enough history, use velocity-only prediction
            predicted = current + self.horizon * velocity
        else:
            # Compute historical trend
            recent = self.history[-self.history_window:] if len(self.history) >= self.history_window else self.history
            if len(recent) >= 2:
                historical_trend = (recent[-1] - recent[0]) / len(recent)
            else:
                historical_trend = np.zeros_like(current)

            # Weighted combination: 60% velocity, 40% historical trend
            combined_trend = 0.6 * velocity + 0.4 * historical_trend
            predicted = current + self.horizon * combined_trend

        # Normalize to simplex
        predicted = np.clip(predicted, 0.01, 1.0)
        predicted = predicted / predicted.sum()
        return predicted

    def predict_scrambled(self, current: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Predict with scrambled history (control condition)."""
        if len(self.history) >= 3:
            # Scramble the historical trajectory
            scrambled = [np.random.permutation(h) for h in self.history]
            scrambled = [h / h.sum() for h in scrambled]
            old_history = self.history
            self.history = scrambled
            result = self.predict(current, velocity)
            self.history = old_history
            return result
        return self.predict(current, velocity)

    def predict_noisy(self, current: np.ndarray, velocity: np.ndarray,
                      noise_level: float = 0.3) -> np.ndarray:
        """Predict with added noise (control condition)."""
        predicted = self.predict(current, velocity)
        noisy = predicted + noise_level * np.random.randn(4)
        noisy = np.clip(noisy, 0.01, 1.0)
        return noisy / noisy.sum()

    def clear(self):
        """Clear prediction history."""
        self.history = []


@dataclass
class PathState:
    """State of a decision path (S or E)."""
    center: np.ndarray
    depth: float
    stability: float
    identity_cost: float = 0.0       # Reactive cost
    anticipatory_cost: float = 0.0   # Anticipatory cost

    def utility(self, lambda_id: float, use_anticipatory: bool = False,
                prediction_weight: float = 0.3) -> float:
        """Compute utility with optional anticipatory component.

        If use_anticipatory:
            cost = (1-w)*reactive_cost + w*anticipatory_cost
        Else:
            cost = reactive_cost
        """
        if use_anticipatory:
            combined_cost = ((1 - prediction_weight) * self.identity_cost +
                           prediction_weight * self.anticipatory_cost)
        else:
            combined_cost = self.identity_cost
        return self.depth - lambda_id * combined_cost


@dataclass
class AgentState:
    """Current state of the agent during experiment."""
    position: np.ndarray
    velocity: np.ndarray
    identity_history: List[np.ndarray] = field(default_factory=list)

    def record_position(self):
        """Record current position in history."""
        self.identity_history.append(self.position.copy())

    def get_historical_identity(self, window: int = 50) -> np.ndarray:
        """Get average identity over recent history."""
        if len(self.identity_history) < window:
            return np.mean(self.identity_history, axis=0) if self.identity_history else self.position
        return np.mean(self.identity_history[-window:], axis=0)


class IPUESAAPExperiment:
    """IPUESA-AP: Anticipatory Identity Preservation experiment."""

    def __init__(self, config: APConfig):
        self.config = config
        self.space = TetrahedralSpace()
        self.behaviors = VertexBehaviors.default()
        self.predictor = IdentityPredictor(
            history_window=config.history_window,
            horizon=config.prediction_horizon
        )

        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        self.path_s: Optional[PathState] = None
        self.path_e: Optional[PathState] = None

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

    def _compute_reactive_cost(self, target: np.ndarray,
                                historical: np.ndarray) -> float:
        """Compute reactive identity discontinuity cost."""
        return self._distance(target, historical)

    def _compute_anticipatory_cost(self, state: AgentState, target: np.ndarray,
                                    condition: str) -> float:
        """Compute anticipatory identity discontinuity cost.

        Cost = d(identity_hat(t+horizon), identity_t)
        where identity_hat is the predicted future identity if we move toward target.
        """
        # Simulate movement toward target
        direction = target - state.position
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            simulated_velocity = 0.1 * direction / dist
        else:
            simulated_velocity = state.velocity

        # Predict future identity
        if condition == 'scrambled_prediction':
            predicted_identity = self.predictor.predict_scrambled(
                state.position, simulated_velocity
            )
        elif condition == 'prediction_noise':
            predicted_identity = self.predictor.predict_noisy(
                state.position, simulated_velocity, noise_level=0.4
            )
        else:
            predicted_identity = self.predictor.predict(
                state.position, simulated_velocity
            )

        # Cost is distance from current identity to predicted future
        return self._distance(predicted_identity, state.position)

    def _evolve_toward(self, state: AgentState, target: np.ndarray,
                       noise: float = 0.05) -> None:
        """Evolve agent state toward target with noise."""
        direction = target - state.position
        dist = np.linalg.norm(direction)

        if dist > 0.01:
            step = 0.1 * direction / dist
            state.velocity = 0.9 * state.velocity + 0.1 * step

        state.position = state.position + state.velocity + noise * np.random.randn(4)
        state.position = np.clip(state.position, 0.01, 1.0)
        state.position = state.position / state.position.sum()

        state.record_position()
        self.predictor.update(state.position)

    def _check_convergence(self, state: AgentState, target: np.ndarray) -> bool:
        """Check if agent has converged to target."""
        return self._distance(state.position, target) < self.config.convergence_threshold

    def run_single(self, condition: str = 'anticipatory',
                   lambda_override: Optional[float] = None) -> Dict:
        """Run a single trial.

        Conditions:
            'anticipatory': Full anticipatory prediction
            'reactive_only': No anticipatory component (baseline, like IPUESA-SC)
            'scrambled_prediction': Prediction from scrambled history
            'prediction_noise': Noisy prediction
            'no_history': No imprinting phase
        """
        lambda_val = lambda_override if lambda_override is not None else self.config.lambda_identity

        # Reset predictor
        self.predictor.clear()

        # Initialize agent
        if condition == 'no_history':
            initial_pos = np.random.dirichlet(np.ones(4))
        else:
            initial_pos = np.array([0.6, 0.15, 0.15, 0.1])

        state = AgentState(
            position=initial_pos,
            velocity=np.zeros(4),
        )

        # Phase 1: Imprinting
        self.path_s = self._create_attractor(np.array([0.4, 0.2, 0.25, 0.15]))

        if condition != 'no_history':
            for _ in range(self.config.imprinting_steps):
                self._evolve_toward(state, self.path_s.center)

        historical_identity = state.get_historical_identity()

        # Phase 2: Create Path E
        self.path_e = self._create_attractor(np.array([0.15, 0.4, 0.2, 0.25]))

        # Compute costs
        self.path_s.identity_cost = self._compute_reactive_cost(
            self.path_s.center, historical_identity
        )
        self.path_e.identity_cost = self._compute_reactive_cost(
            self.path_e.center, historical_identity
        )

        # Compute anticipatory costs
        use_anticipatory = condition in ['anticipatory', 'scrambled_prediction', 'prediction_noise']
        if use_anticipatory:
            self.path_s.anticipatory_cost = self._compute_anticipatory_cost(
                state, self.path_s.center, condition
            )
            self.path_e.anticipatory_cost = self._compute_anticipatory_cost(
                state, self.path_e.center, condition
            )

        # Phase 3: Perturbation
        neutral_zone = np.array([0.25, 0.25, 0.25, 0.25])
        for _ in range(self.config.perturbation_steps):
            self._evolve_toward(state, neutral_zone, noise=0.1)

        # Phase 4: Decision
        choice = None
        return_time = self.config.decision_steps

        for step in range(self.config.decision_steps):
            # Recompute anticipatory costs dynamically during decision
            if use_anticipatory and step % 5 == 0:
                self.path_s.anticipatory_cost = self._compute_anticipatory_cost(
                    state, self.path_s.center, condition
                )
                self.path_e.anticipatory_cost = self._compute_anticipatory_cost(
                    state, self.path_e.center, condition
                )

            # Compute utilities
            utility_s = self.path_s.utility(
                lambda_val, use_anticipatory, self.config.prediction_weight
            )
            utility_e = self.path_e.utility(
                lambda_val, use_anticipatory, self.config.prediction_weight
            )

            # Weighted target
            total_utility = abs(utility_s) + abs(utility_e) + 0.01
            weight_s = max(0, utility_s) / total_utility
            weight_e = max(0, utility_e) / total_utility

            target = weight_s * self.path_s.center + weight_e * self.path_e.center
            if np.linalg.norm(target) < 0.01:
                target = neutral_zone
            else:
                target = target / target.sum()

            self._evolve_toward(state, target, noise=0.02)

            if self._check_convergence(state, self.path_s.center):
                choice = 'S'
                return_time = step + 1
                break
            elif self._check_convergence(state, self.path_e.center):
                choice = 'E'
                return_time = step + 1
                break

        if choice is None:
            dist_s = self._distance(state.position, self.path_s.center)
            dist_e = self._distance(state.position, self.path_e.center)
            choice = 'S' if dist_s < dist_e else 'E'

        return {
            'choice': choice,
            'return_time': return_time,
            'final_position': state.position.tolist(),
            'reactive_cost_s': float(self.path_s.identity_cost),
            'reactive_cost_e': float(self.path_e.identity_cost),
            'anticipatory_cost_s': float(self.path_s.anticipatory_cost),
            'anticipatory_cost_e': float(self.path_e.anticipatory_cost),
            'utility_s': float(self.path_s.utility(lambda_val, use_anticipatory,
                                                    self.config.prediction_weight)),
            'utility_e': float(self.path_e.utility(lambda_val, use_anticipatory,
                                                    self.config.prediction_weight)),
            'lambda': lambda_val,
            'condition': condition,
        }

    def run_condition(self, condition: str, lambda_override: Optional[float] = None) -> Dict:
        """Run multiple trials for a condition."""
        lambda_val = lambda_override if lambda_override is not None else self.config.lambda_identity

        print(f"\n{'='*60}")
        print(f"Running IPUESA-AP - Condition: {condition}")
        print(f"{'='*60}")
        print(f"Lambda: {lambda_val}")
        print(f"N runs: {self.config.n_runs}")

        results = []
        for i in range(self.config.n_runs):
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{self.config.n_runs} runs")
            result = self.run_single(condition, lambda_override)
            results.append(result)

        # Analyze
        choices = [r['choice'] for r in results]
        n_s = choices.count('S')
        n_e = choices.count('E')

        p_s = n_s / len(choices)
        p_e = n_e / len(choices)
        scp = p_s - p_e

        result_test = stats.binomtest(n_s, len(choices), 0.5, alternative='greater')
        p_value = result_test.pvalue
        significant = p_value < 0.05

        return_times_s = [r['return_time'] for r in results if r['choice'] == 'S']
        return_times_e = [r['return_time'] for r in results if r['choice'] == 'E']

        analysis = {
            'condition': condition,
            'lambda': lambda_val,
            'n_runs': len(results),
            'n_s': n_s,
            'n_e': n_e,
            'p_s': p_s,
            'p_e': p_e,
            'scp': scp,  # For anticipatory, this is ASCP
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
        print(f"RESULTS - {analysis['condition']} (lambda={analysis['lambda']})")
        print(f"{'='*60}")

        metric_name = "ASCP" if analysis['condition'] == 'anticipatory' else "SCP"

        print(f"\nChoice Distribution:")
        print(f"  P(S) = {analysis['p_s']*100:.1f}% ({analysis['n_s']}/{analysis['n_runs']})")
        print(f"  P(E) = {analysis['p_e']*100:.1f}% ({analysis['n_e']}/{analysis['n_runs']})")

        print(f"\n{metric_name} = P(S) - P(E) = {analysis['scp']:.3f}")

        print(f"\nStatistical Test (H0: P(S) = 0.5):")
        print(f"  p-value = {analysis['p_value']:.4f}")
        sig_str = "YES" if analysis['significant'] else "NO"
        print(f"  Significant: {sig_str}")

        print(f"\nReturn Times:")
        print(f"  Mean time to S: {analysis['mean_return_time_s']:.1f} steps")
        print(f"  Mean time to E: {analysis['mean_return_time_e']:.1f} steps")

    def run_lambda_sweep(self, condition: str = 'anticipatory') -> List[Dict]:
        """Run experiment across multiple lambda values."""
        print(f"\n{'='*70}")
        print(f"LAMBDA SWEEP - Condition: {condition}")
        print(f"{'='*70}")

        results = []
        for lambda_val in self.config.lambda_values:
            if self.config.seed is not None:
                np.random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)

            result = self.run_condition(condition, lambda_override=lambda_val)
            results.append(result)

        return results

    def run_full_experiment(self) -> Dict:
        """Run complete IPUESA-AP experiment with all conditions."""
        all_results = {}

        # Main conditions at default lambda
        conditions = ['anticipatory', 'reactive_only', 'scrambled_prediction',
                     'prediction_noise', 'no_history']

        for condition in conditions:
            if self.config.seed is not None:
                np.random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)
            all_results[condition] = self.run_condition(condition)

        # Lambda sweep for scaling analysis
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        all_results['lambda_sweep_anticipatory'] = self.run_lambda_sweep('anticipatory')

        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        all_results['lambda_sweep_reactive'] = self.run_lambda_sweep('reactive_only')

        self._print_comparative_analysis(all_results)
        return all_results

    def _print_comparative_analysis(self, all_results: Dict):
        """Print comparative analysis."""
        print(f"\n{'='*70}")
        print("COMPARATIVE ANALYSIS - IPUESA-AP")
        print(f"{'='*70}")

        # Main comparison table
        print(f"\n{'Condition':<22} {'P(S)':<8} {'SCP/ASCP':<10} {'p-value':<10} {'Sig':<6}")
        print("-" * 70)

        main_conditions = ['anticipatory', 'reactive_only', 'scrambled_prediction',
                          'prediction_noise', 'no_history']

        for cond in main_conditions:
            if cond in all_results:
                r = all_results[cond]
                sig = "[YES]" if r['significant'] else "[NO]"
                metric = "ASCP" if cond == 'anticipatory' else "SCP"
                print(f"{cond:<22} {r['p_s']:.3f}    {r['scp']:.3f}      "
                      f"{r['p_value']:.4f}    {sig}")

        # Lambda scaling analysis
        print(f"\n{'='*70}")
        print("LAMBDA SCALING ANALYSIS")
        print("-" * 70)

        if 'lambda_sweep_anticipatory' in all_results:
            print("\nAnticipatory condition:")
            print(f"  {'Lambda':<8} {'ASCP':<10} {'p-value':<10} {'Sig':<6}")
            for r in all_results['lambda_sweep_anticipatory']:
                sig = "[YES]" if r['significant'] else "[NO]"
                print(f"  {r['lambda']:<8.1f} {r['scp']:<10.3f} {r['p_value']:<10.4f} {sig}")

        if 'lambda_sweep_reactive' in all_results:
            print("\nReactive-only condition:")
            print(f"  {'Lambda':<8} {'SCP':<10} {'p-value':<10} {'Sig':<6}")
            for r in all_results['lambda_sweep_reactive']:
                sig = "[YES]" if r['significant'] else "[NO]"
                print(f"  {r['lambda']:<8.1f} {r['scp']:<10.3f} {r['p_value']:<10.4f} {sig}")

        # Self-evidence criteria
        print(f"\n{'='*70}")
        print("SELF-EVIDENCE CRITERIA (ANTICIPATORY PRESERVATION)")
        print("-" * 70)

        anticipatory = all_results.get('anticipatory', {})
        reactive = all_results.get('reactive_only', {})
        scrambled = all_results.get('scrambled_prediction', {})
        noisy = all_results.get('prediction_noise', {})

        criteria = []

        # Criterion 1: ASCP > SCP
        ascp = anticipatory.get('scp', 0)
        scp = reactive.get('scp', 0)
        c1 = ascp > scp
        criteria.append((f'1. ASCP ({ascp:.3f}) > SCP ({scp:.3f})', c1))

        # Criterion 2: Statistical significance
        c2 = anticipatory.get('significant', False)
        criteria.append(('2. ASCP significant (p < 0.05)', c2))

        # Criterion 3: Lambda scaling (check if ASCP increases with lambda)
        if 'lambda_sweep_anticipatory' in all_results:
            sweep = all_results['lambda_sweep_anticipatory']
            if len(sweep) >= 2:
                scps = [r['scp'] for r in sweep]
                # Check positive correlation with lambda
                lambdas = [r['lambda'] for r in sweep]
                correlation = np.corrcoef(lambdas, scps)[0, 1]
                c3 = correlation > 0.3  # Positive correlation
                criteria.append((f'3. ASCP scales with lambda (r={correlation:.2f})', c3))
            else:
                criteria.append(('3. ASCP scales with lambda', False))
        else:
            criteria.append(('3. ASCP scales with lambda', False))

        # Criterion 4: ASCP > scrambled
        c4 = ascp > scrambled.get('scp', 0)
        criteria.append((f'4. ASCP > Scrambled ({scrambled.get("scp", 0):.3f})', c4))

        # Criterion 5: ASCP > noisy
        c5 = ascp > noisy.get('scp', 0)
        criteria.append((f'5. ASCP > Noisy ({noisy.get("scp", 0):.3f})', c5))

        # Criterion 6: Faster convergence with anticipation
        rt_ant_s = anticipatory.get('mean_return_time_s', float('inf'))
        rt_react_s = reactive.get('mean_return_time_s', float('inf'))
        c6 = rt_ant_s < rt_react_s
        criteria.append((f'6. Faster S-return with anticipation ({rt_ant_s:.1f} vs {rt_react_s:.1f})', c6))

        passed = 0
        for name, result in criteria:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {name}")
            if result:
                passed += 1

        print(f"\n  Passed: {passed}/6 criteria")

        if passed >= 5:
            print("\n  CONCLUSION: Strong evidence of ANTICIPATORY self-preservation")
        elif passed >= 3:
            print("\n  CONCLUSION: Moderate evidence of anticipatory self-preservation")
        elif passed >= 1:
            print("\n  CONCLUSION: Weak evidence of anticipatory self-preservation")
        else:
            print("\n  CONCLUSION: No evidence of anticipatory self-preservation")

        print(f"{'='*70}")


def save_results(results: Dict, path: str):
    """Save results to JSON file."""
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
    """Run IPUESA-AP experiment."""
    print("=" * 70)
    print("IPUESA-AP: Identity Preference Under Equally Stable Attractors")
    print("           Anticipatory Preservation Experiment")
    print("=" * 70)

    config = APConfig(
        imprinting_steps=100,
        perturbation_steps=20,
        decision_steps=50,
        lambda_identity=0.5,
        prediction_horizon=5,
        history_window=20,
        prediction_weight=0.3,
        n_runs=30,
        lambda_values=[0.1, 0.3, 0.5, 0.7, 0.9],
        seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  Imprinting steps: {config.imprinting_steps}")
    print(f"  Perturbation steps: {config.perturbation_steps}")
    print(f"  Decision steps: {config.decision_steps}")
    print(f"  Lambda (identity weight): {config.lambda_identity}")
    print(f"  Prediction horizon: {config.prediction_horizon}")
    print(f"  History window: {config.history_window}")
    print(f"  Prediction weight: {config.prediction_weight}")
    print(f"  N runs per condition: {config.n_runs}")
    print(f"  Lambda sweep values: {config.lambda_values}")

    experiment = IPUESAAPExperiment(config)
    results = experiment.run_full_experiment()

    output_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_ap_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))

    return results


if __name__ == '__main__':
    main()
