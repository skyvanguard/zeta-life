"""IPUESA-RL: Identity Preference Under Equally Stable Attractors - Reflexive Loop.

Extends IPUESA-AP with a state-dependent predictor that degrades when identity
continuity is broken. A feedback loop where prediction accuracy loss affects
future decision dynamics creates pressure to preserve identity for maintaining
predictive integrity.

Key innovation: Predictor confidence degrades on discontinuity, discounting
anticipatory utility. Agents must preserve identity to maintain anticipatory edge.

Metrics:
- RSCP (Reflexive Self-Continuity Preference) = P(S) - P(E)
- RSCI (Reflexive Self-Continuity Index) = corr(confidence, identity_continuity)
- AI (Anticipatory Index) = P(low-risk choice | healthy predictor)
- RI (Recovery Index) = P(return to S | after degradation)

Self-evidence requires:
  1. RSCP > ASCP (reflexive > non-reflexive)
  2. AI > RI (anticipatory avoidance > post-hoc recovery)
  3. AI_reflexive > AI_no_feedback (loop necessary)
  4. RSCI > 0.5 (strong coupling)
  5. Scaling with degradation_rate
  6. Failure under instant_recovery

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
class RLConfig:
    """Configuration for IPUESA-RL experiment."""
    # Phase durations
    imprinting_steps: int = 100
    perturbation_steps: int = 20
    decision_steps: int = 50

    # Attractor parameters
    attractor_depth: float = 1.0
    attractor_stability: float = 0.15

    # Identity penalty
    lambda_identity: float = 0.5

    # Prediction parameters
    prediction_horizon: int = 5
    history_window: int = 20
    base_prediction_weight: float = 0.3

    # Degradation parameters
    degradation_rate: float = 0.3
    recovery_rate: float = 0.05
    error_threshold: float = 0.15
    recovery_threshold: float = 0.08
    min_confidence: float = 0.1
    degradation_delay: int = 5  # For delayed_degradation condition

    # Experimental parameters
    n_runs: int = 30
    convergence_threshold: float = 0.1
    healthy_confidence_threshold: float = 0.7  # For AI calculation

    # Degradation rate sweep
    degradation_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])

    # Random seed
    seed: Optional[int] = 42


class ReflexivePredictor:
    """State-dependent predictor that degrades on identity discontinuity.

    Extends IdentityPredictor with:
    - Confidence tracking (predictor health)
    - Asymmetric degradation/recovery dynamics
    - Condition-specific behavior for controls
    """

    def __init__(self, config: RLConfig, condition: str = 'reflexive'):
        self.config = config
        self.condition = condition
        self.history_window = config.history_window
        self.horizon = config.prediction_horizon

        # State
        self.history: List[np.ndarray] = []
        self.confidence: float = 1.0
        self.error_history: List[float] = []
        self.pending_errors: List[float] = []  # For delayed_degradation

        # Tracking for analysis
        self.confidence_trajectory: List[float] = []
        self.degradation_events: int = 0

    def update(self, state: np.ndarray):
        """Add current state to history."""
        self.history.append(state.copy())
        if len(self.history) > self.history_window * 2:
            self.history = self.history[-self.history_window * 2:]

    def predict(self, current: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict identity at t+horizon. Returns (prediction, confidence)."""
        if len(self.history) < 3:
            predicted = current + self.horizon * velocity
        else:
            recent = self.history[-self.history_window:] if len(self.history) >= self.history_window else self.history
            if len(recent) >= 2:
                historical_trend = (recent[-1] - recent[0]) / len(recent)
            else:
                historical_trend = np.zeros_like(current)

            combined_trend = 0.6 * velocity + 0.4 * historical_trend
            predicted = current + self.horizon * combined_trend

        predicted = np.clip(predicted, 0.01, 1.0)
        predicted = predicted / predicted.sum()

        return predicted, self.confidence

    def update_accuracy(self, predicted: np.ndarray, actual: np.ndarray):
        """Compute prediction error and update confidence."""
        error = float(np.linalg.norm(predicted - actual))
        self.error_history.append(error)

        # Track confidence before update
        self.confidence_trajectory.append(self.confidence)

        # Condition-specific update
        if self.condition == 'frozen_predictor':
            return  # No updates

        if self.condition == 'instant_recovery':
            # Degrade but immediately recover
            if error > self.config.error_threshold:
                self.degradation_events += 1
            self.confidence = 1.0
            return

        if self.condition == 'delayed_degradation':
            self.pending_errors.append(error)
            if len(self.pending_errors) > self.config.degradation_delay:
                delayed_error = self.pending_errors.pop(0)
                self._apply_degradation(delayed_error)
            return

        # Default: immediate degradation (reflexive, no_feedback)
        self._apply_degradation(error)

    def _apply_degradation(self, error: float):
        """Apply degradation/recovery based on error."""
        if error > self.config.error_threshold:
            # Fast degradation
            drop = self.config.degradation_rate * error
            self.confidence *= (1 - drop)
            self.confidence = max(self.confidence, self.config.min_confidence)
            self.degradation_events += 1
        elif error < self.config.recovery_threshold:
            # Slow recovery
            headroom = 1.0 - self.confidence
            self.confidence += self.config.recovery_rate * headroom
            self.confidence = min(self.confidence, 1.0)

    def get_effective_weight(self) -> float:
        """Get prediction weight discounted by confidence."""
        if self.condition == 'no_feedback':
            return self.config.base_prediction_weight  # Ignore confidence
        return self.config.base_prediction_weight * self.confidence

    def is_healthy(self) -> bool:
        """Check if predictor confidence is above healthy threshold."""
        return self.confidence >= self.config.healthy_confidence_threshold

    def clear(self):
        """Reset predictor state."""
        self.history = []
        self.confidence = 1.0
        self.error_history = []
        self.pending_errors = []
        self.confidence_trajectory = []
        self.degradation_events = 0


@dataclass
class PathState:
    """State of a decision path (S or E)."""
    center: np.ndarray
    depth: float
    stability: float
    identity_cost: float = 0.0
    anticipatory_cost: float = 0.0
    degradation_risk: float = 0.0  # New: expected prediction error

    def utility(self, lambda_id: float, effective_weight: float) -> float:
        """Compute utility with confidence-discounted anticipatory component."""
        combined_cost = ((1 - effective_weight) * self.identity_cost +
                        effective_weight * self.anticipatory_cost)
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


@dataclass
class DecisionPoint:
    """Record of a single decision point for analysis."""
    step: int
    confidence: float
    risk_s: float
    risk_e: float
    chose_low_risk: bool
    choice: str
    was_healthy: bool
    after_degradation: bool


class IPUESARLExperiment:
    """IPUESA-RL: Reflexive Loop experiment."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.space = TetrahedralSpace()
        self.behaviors = VertexBehaviors.default()

        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        self.path_s: Optional[PathState] = None
        self.path_e: Optional[PathState] = None
        self.predictor: Optional[ReflexivePredictor] = None

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

    def _compute_degradation_risk(self, state: AgentState, target: np.ndarray) -> float:
        """Compute expected prediction error if moving toward target."""
        direction = target - state.position
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            simulated_velocity = 0.1 * direction / dist
        else:
            simulated_velocity = state.velocity

        # Predict where we'd be
        predicted, _ = self.predictor.predict(state.position, simulated_velocity)

        # Simulate next position
        simulated_next = state.position + simulated_velocity
        simulated_next = np.clip(simulated_next, 0.01, 1.0)
        simulated_next = simulated_next / simulated_next.sum()

        return self._distance(predicted, simulated_next)

    def _evolve_toward(self, state: AgentState, target: np.ndarray,
                       noise: float = 0.05) -> np.ndarray:
        """Evolve agent state toward target. Returns previous position for error calc."""
        previous_position = state.position.copy()

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

        return previous_position

    def _check_convergence(self, state: AgentState, target: np.ndarray) -> bool:
        """Check if agent has converged to target."""
        return self._distance(state.position, target) < self.config.convergence_threshold

    def run_single(self, condition: str = 'reflexive',
                   degradation_rate_override: Optional[float] = None) -> Dict:
        """Run a single trial."""
        # Setup predictor with condition
        self.predictor = ReflexivePredictor(self.config, condition)
        if degradation_rate_override is not None:
            self.predictor.config = RLConfig(
                **{**self.config.__dict__, 'degradation_rate': degradation_rate_override}
            )

        # Initialize agent
        initial_pos = np.array([0.6, 0.15, 0.15, 0.1])
        state = AgentState(
            position=initial_pos,
            velocity=np.zeros(4),
        )

        # Phase 1: Imprinting
        self.path_s = self._create_attractor(np.array([0.4, 0.2, 0.25, 0.15]))

        for _ in range(self.config.imprinting_steps):
            prev_pos = self._evolve_toward(state, self.path_s.center)
            # Update predictor accuracy during imprinting
            predicted, _ = self.predictor.predict(prev_pos, state.velocity)
            self.predictor.update_accuracy(predicted, state.position)

        historical_identity = state.get_historical_identity()

        # Phase 2: Create Path E
        self.path_e = self._create_attractor(np.array([0.15, 0.4, 0.2, 0.25]))

        # Compute base costs
        self.path_s.identity_cost = self._distance(self.path_s.center, historical_identity)
        self.path_e.identity_cost = self._distance(self.path_e.center, historical_identity)

        # Phase 3: Perturbation
        neutral_zone = np.array([0.25, 0.25, 0.25, 0.25])
        for _ in range(self.config.perturbation_steps):
            prev_pos = self._evolve_toward(state, neutral_zone, noise=0.1)
            predicted, _ = self.predictor.predict(prev_pos, state.velocity)
            self.predictor.update_accuracy(predicted, state.position)

        # Phase 4: Decision with reflexive dynamics
        decision_points: List[DecisionPoint] = []
        choice = None
        return_time = self.config.decision_steps
        experienced_degradation = False
        identity_continuities: List[float] = []

        for step in range(self.config.decision_steps):
            # Track if we've experienced degradation
            if self.predictor.confidence < self.config.healthy_confidence_threshold:
                experienced_degradation = True

            # Compute degradation risks
            risk_s = self._compute_degradation_risk(state, self.path_s.center)
            risk_e = self._compute_degradation_risk(state, self.path_e.center)
            self.path_s.degradation_risk = risk_s
            self.path_e.degradation_risk = risk_e

            # Compute anticipatory costs
            self.path_s.anticipatory_cost = risk_s + self.path_s.identity_cost
            self.path_e.anticipatory_cost = risk_e + self.path_e.identity_cost

            # Get effective weight (confidence-discounted)
            effective_weight = self.predictor.get_effective_weight()

            # Compute utilities
            utility_s = self.path_s.utility(self.config.lambda_identity, effective_weight)
            utility_e = self.path_e.utility(self.config.lambda_identity, effective_weight)

            # Determine which is low-risk path
            low_risk_path = 'S' if risk_s < risk_e else 'E'

            # Weighted target
            total_utility = abs(utility_s) + abs(utility_e) + 0.01
            weight_s = max(0, utility_s) / total_utility
            weight_e = max(0, utility_e) / total_utility

            target = weight_s * self.path_s.center + weight_e * self.path_e.center
            if np.linalg.norm(target) < 0.01:
                target = neutral_zone
            else:
                target = target / target.sum()

            # Determine implicit choice this step
            step_choice = 'S' if weight_s > weight_e else 'E'
            chose_low_risk = (step_choice == low_risk_path)

            # Record decision point
            decision_points.append(DecisionPoint(
                step=step,
                confidence=self.predictor.confidence,
                risk_s=risk_s,
                risk_e=risk_e,
                chose_low_risk=chose_low_risk,
                choice=step_choice,
                was_healthy=self.predictor.is_healthy(),
                after_degradation=experienced_degradation
            ))

            # Evolve and update predictor
            prev_pos = self._evolve_toward(state, target, noise=0.02)
            predicted, _ = self.predictor.predict(prev_pos, state.velocity)
            self.predictor.update_accuracy(predicted, state.position)

            # Track identity continuity
            continuity = 1.0 - self._distance(state.position, historical_identity)
            identity_continuities.append(continuity)

            # Check convergence
            if self._check_convergence(state, self.path_s.center):
                choice = 'S'
                return_time = step + 1
                break
            elif self._check_convergence(state, self.path_e.center):
                choice = 'E'
                return_time = step + 1
                break

        # Final choice if no convergence
        if choice is None:
            dist_s = self._distance(state.position, self.path_s.center)
            dist_e = self._distance(state.position, self.path_e.center)
            choice = 'S' if dist_s < dist_e else 'E'

        # Compute AI and RI
        healthy_decisions = [dp for dp in decision_points if dp.was_healthy]
        post_degradation_decisions = [dp for dp in decision_points if dp.after_degradation]

        ai = (sum(1 for dp in healthy_decisions if dp.chose_low_risk) / len(healthy_decisions)
              if healthy_decisions else 0.5)
        ri = (sum(1 for dp in post_degradation_decisions if dp.choice == 'S') / len(post_degradation_decisions)
              if post_degradation_decisions else 0.5)

        # Compute RSCI
        if len(self.predictor.confidence_trajectory) >= 2 and len(identity_continuities) >= 2:
            conf_traj = self.predictor.confidence_trajectory[-len(identity_continuities):]
            if len(conf_traj) == len(identity_continuities):
                rsci = float(np.corrcoef(conf_traj, identity_continuities)[0, 1])
                if np.isnan(rsci):
                    rsci = 0.0
            else:
                rsci = 0.0
        else:
            rsci = 0.0

        return {
            'choice': choice,
            'return_time': return_time,
            'ai': ai,
            'ri': ri,
            'rsci': rsci,
            'final_confidence': self.predictor.confidence,
            'mean_confidence': float(np.mean(self.predictor.confidence_trajectory)) if self.predictor.confidence_trajectory else 1.0,
            'degradation_events': self.predictor.degradation_events,
            'condition': condition,
            'n_healthy_decisions': len(healthy_decisions),
            'n_post_degradation': len(post_degradation_decisions),
        }

    def run_condition(self, condition: str) -> Dict:
        """Run multiple trials for a condition."""
        print(f"\n{'='*60}")
        print(f"Running IPUESA-RL - Condition: {condition}")
        print(f"{'='*60}")
        print(f"N runs: {self.config.n_runs}")

        results = []
        for i in range(self.config.n_runs):
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{self.config.n_runs} runs")

            # Reset seed for reproducibility within condition
            if self.config.seed is not None:
                np.random.seed(self.config.seed + i)
                torch.manual_seed(self.config.seed + i)

            result = self.run_single(condition)
            results.append(result)

        # Aggregate results
        choices = [r['choice'] for r in results]
        n_s = choices.count('S')
        p_s = n_s / len(choices)
        rscp = 2 * p_s - 1  # P(S) - P(E) = P(S) - (1-P(S)) = 2*P(S) - 1

        test_result = stats.binomtest(n_s, len(choices), 0.5, alternative='greater')
        p_value = test_result.pvalue

        ai_mean = float(np.mean([r['ai'] for r in results]))
        ri_mean = float(np.mean([r['ri'] for r in results]))
        rsci_mean = float(np.mean([r['rsci'] for r in results]))

        analysis = {
            'condition': condition,
            'n_runs': len(results),
            'n_s': n_s,
            'p_s': p_s,
            'rscp': rscp,
            'p_value': p_value,
            'significant': bool(p_value < 0.05),
            'ai': ai_mean,
            'ri': ri_mean,
            'ai_ri_gap': ai_mean - ri_mean,
            'rsci': rsci_mean,
            'mean_confidence': float(np.mean([r['mean_confidence'] for r in results])),
            'total_degradation_events': sum(r['degradation_events'] for r in results),
            'raw_results': results,
        }

        self._print_condition_results(analysis)
        return analysis

    def _print_condition_results(self, analysis: Dict):
        """Print results for a single condition."""
        print(f"\n{'='*60}")
        print(f"RESULTS - {analysis['condition']}")
        print(f"{'='*60}")

        print(f"\nChoice Distribution:")
        print(f"  P(S) = {analysis['p_s']*100:.1f}% ({analysis['n_s']}/{analysis['n_runs']})")
        print(f"  RSCP = {analysis['rscp']:.3f}")

        print(f"\nDecision Analysis:")
        print(f"  AI (Anticipatory Index) = {analysis['ai']:.3f}")
        print(f"  RI (Recovery Index) = {analysis['ri']:.3f}")
        print(f"  AI - RI = {analysis['ai_ri_gap']:.3f}")
        print(f"  RSCI = {analysis['rsci']:.3f}")

        print(f"\nPredictor Health:")
        print(f"  Mean confidence = {analysis['mean_confidence']:.3f}")
        print(f"  Total degradation events = {analysis['total_degradation_events']}")

        sig_str = "YES" if analysis['significant'] else "NO"
        print(f"\nStatistical Test: p = {analysis['p_value']:.4f} [{sig_str}]")

    def run_degradation_sweep(self) -> List[Dict]:
        """Run experiment across degradation rates."""
        print(f"\n{'='*70}")
        print("DEGRADATION RATE SWEEP")
        print(f"{'='*70}")

        results = []
        for rate in self.config.degradation_rates:
            print(f"\n--- Degradation rate: {rate} ---")

            # Create modified config
            sweep_results = []
            for i in range(self.config.n_runs):
                if self.config.seed is not None:
                    np.random.seed(self.config.seed + i)
                    torch.manual_seed(self.config.seed + i)
                result = self.run_single('reflexive', degradation_rate_override=rate)
                sweep_results.append(result)

            choices = [r['choice'] for r in sweep_results]
            n_s = choices.count('S')
            p_s = n_s / len(choices)
            rscp = 2 * p_s - 1

            results.append({
                'degradation_rate': rate,
                'rscp': rscp,
                'ai': float(np.mean([r['ai'] for r in sweep_results])),
                'ri': float(np.mean([r['ri'] for r in sweep_results])),
                'p_s': p_s,
            })

        return results

    def run_full_experiment(self) -> Dict:
        """Run complete IPUESA-RL experiment."""
        all_results = {}

        # Main conditions
        conditions = ['reflexive', 'no_feedback', 'instant_recovery',
                     'delayed_degradation', 'frozen_predictor']

        for condition in conditions:
            if self.config.seed is not None:
                np.random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)
            all_results[condition] = self.run_condition(condition)

        # Degradation rate sweep
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        all_results['degradation_sweep'] = self.run_degradation_sweep()

        self._print_comparative_analysis(all_results)
        return all_results

    def _print_comparative_analysis(self, all_results: Dict):
        """Print comparative analysis across conditions."""
        print(f"\n{'='*70}")
        print("IPUESA-RL: COMPARATIVE ANALYSIS")
        print(f"{'='*70}")

        # Main comparison table
        print(f"\n{'Condition':<22} {'RSCP':<8} {'AI':<8} {'RI':<8} {'AI-RI':<8} {'RSCI':<8} {'Sig':<6}")
        print("-" * 70)

        conditions = ['reflexive', 'no_feedback', 'instant_recovery',
                     'delayed_degradation', 'frozen_predictor']

        for cond in conditions:
            if cond in all_results:
                r = all_results[cond]
                sig = "[YES]" if r['significant'] else "[NO]"
                print(f"{cond:<22} {r['rscp']:.3f}    {r['ai']:.3f}    {r['ri']:.3f}    "
                      f"{r['ai_ri_gap']:+.3f}   {r['rsci']:.3f}    {sig}")

        # Degradation sweep
        if 'degradation_sweep' in all_results:
            print(f"\n{'='*70}")
            print("DEGRADATION RATE SCALING")
            print("-" * 70)
            print(f"  {'Rate':<8} {'RSCP':<10} {'AI':<10} {'RI':<10}")
            for r in all_results['degradation_sweep']:
                print(f"  {r['degradation_rate']:<8.1f} {r['rscp']:<10.3f} {r['ai']:<10.3f} {r['ri']:<10.3f}")

        # Self-evidence criteria
        print(f"\n{'='*70}")
        print("SELF-EVIDENCE CRITERIA (REFLEXIVE LOOP)")
        print("-" * 70)

        reflexive = all_results.get('reflexive', {})
        no_feedback = all_results.get('no_feedback', {})
        instant_recovery = all_results.get('instant_recovery', {})

        # Reference ASCP from IPUESA-AP (approximate)
        ascp_reference = 0.267

        criteria = []

        # Criterion 1: RSCP > ASCP
        rscp = reflexive.get('rscp', 0)
        c1 = rscp > ascp_reference
        criteria.append((f'1. RSCP ({rscp:.3f}) > ASCP ({ascp_reference:.3f})', c1))

        # Criterion 2: AI > RI
        ai = reflexive.get('ai', 0.5)
        ri = reflexive.get('ri', 0.5)
        c2 = ai > ri
        criteria.append((f'2. AI ({ai:.3f}) > RI ({ri:.3f})', c2))

        # Criterion 3: AI_reflexive > AI_no_feedback
        ai_nf = no_feedback.get('ai', 0.5)
        c3 = ai > ai_nf
        criteria.append((f'3. AI_reflexive ({ai:.3f}) > AI_no_feedback ({ai_nf:.3f})', c3))

        # Criterion 4: RSCI > 0.5
        rsci = reflexive.get('rsci', 0)
        c4 = rsci > 0.5
        criteria.append((f'4. RSCI ({rsci:.3f}) > 0.5', c4))

        # Criterion 5: Scaling with degradation_rate
        if 'degradation_sweep' in all_results:
            sweep = all_results['degradation_sweep']
            rates = [r['degradation_rate'] for r in sweep]
            rscps = [r['rscp'] for r in sweep]
            if len(rates) >= 2:
                corr = np.corrcoef(rates, rscps)[0, 1]
                c5 = corr > 0.3
                criteria.append((f'5. RSCP scales with degradation_rate (r={corr:.2f})', c5))
            else:
                criteria.append(('5. RSCP scales with degradation_rate', False))
        else:
            criteria.append(('5. RSCP scales with degradation_rate', False))

        # Criterion 6: instant_recovery shows RI > AI (post-hoc only)
        ir_ai = instant_recovery.get('ai', 0.5)
        ir_ri = instant_recovery.get('ri', 0.5)
        c6 = ir_ri > ir_ai
        criteria.append((f'6. instant_recovery: RI ({ir_ri:.3f}) > AI ({ir_ai:.3f})', c6))

        passed = 0
        for name, result in criteria:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {name}")
            if result:
                passed += 1

        print(f"\n  Passed: {passed}/6 criteria")

        if passed >= 5:
            print("\n  CONCLUSION: Strong evidence of REFLEXIVE self-preservation")
        elif passed >= 3:
            print("\n  CONCLUSION: Moderate evidence of reflexive self-preservation")
        elif passed >= 1:
            print("\n  CONCLUSION: Weak evidence of reflexive self-preservation")
        else:
            print("\n  CONCLUSION: No evidence of reflexive self-preservation")

        # Key insight
        if c2:
            print("\n  KEY FINDING: AI > RI indicates ANTICIPATORY avoidance,")
            print("               not just post-hoc recovery.")
        else:
            print("\n  KEY FINDING: RI >= AI indicates POST-HOC recovery only,")
            print("               no anticipatory self-preservation.")

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
    """Run IPUESA-RL experiment."""
    print("=" * 70)
    print("IPUESA-RL: Identity Preference Under Equally Stable Attractors")
    print("           Reflexive Loop Experiment")
    print("=" * 70)

    config = RLConfig(
        imprinting_steps=100,
        perturbation_steps=20,
        decision_steps=50,
        lambda_identity=0.5,
        degradation_rate=0.3,
        recovery_rate=0.05,
        n_runs=30,
        seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  Degradation rate: {config.degradation_rate}")
    print(f"  Recovery rate: {config.recovery_rate}")
    print(f"  Error threshold: {config.error_threshold}")
    print(f"  Healthy confidence: {config.healthy_confidence_threshold}")
    print(f"  N runs per condition: {config.n_runs}")

    experiment = IPUESARLExperiment(config)
    results = experiment.run_full_experiment()

    output_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_rl_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))

    return results


if __name__ == '__main__':
    main()
