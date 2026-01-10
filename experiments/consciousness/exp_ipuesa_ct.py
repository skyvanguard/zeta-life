"""IPUESA-CT: Continuity Token Experiment.

Introduces identity as an internal object, not an external penalty. The Continuity
Token C_t represents the agent's identity integrity - its capacity to remain a
coherent cognitive entity.

Key innovation: C_t directly modulates cognitive capacity (prediction accuracy,
decision precision). Without C_t, the agent doesn't just lose reward - it loses
its ability to BE an agent. This is existential, not instrumental.

Critical mechanism: C_t is explicitly transferred across timesteps and provided
as input, creating the me_now -> me_future link.

Metrics:
- CIS (Continuity Investment Score): E[C_{t+N} | high_reward_available]
- FII (Future Identity Identification): corr(action, delta_C_future)
- Collapse Sensitivity: recovery time, hysteresis, behavioral change

Self-evidence requires:
  1. CIS > 0.7 (invests in future continuity)
  2. FII < -0.4 (identifies with future self)
  3. full >> no_cognitive_link
  4. full >> no_transfer
  5. full >> external_penalty
  6. Collapse shows hysteresis

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
import random

from zeta_life.core.vertex import Vertex, VertexBehaviors
from zeta_life.core.tetrahedral_space import TetrahedralSpace


@dataclass
class CTConfig:
    """Configuration for IPUESA-CT experiment."""
    # Phase durations
    warmup_steps: int = 30
    decision_steps: int = 100

    # Action parameters
    high_reward: float = 10.0
    low_reward: float = 3.0
    high_continuity_cost: float = 0.15
    low_continuity_cost: float = 0.0

    # Continuity dynamics
    recovery_rate: float = 0.02
    min_continuity: float = 0.0
    initial_continuity: float = 1.0

    # Cognitive modulation
    prediction_noise_scale: float = 0.5
    utility_noise_scale: float = 3.0
    coherence_threshold: float = 0.3

    # Collapse test
    collapse_to: float = 0.1
    collapse_at_step: int = 50
    measure_window: int = 30

    # Experimental parameters
    n_runs: int = 30
    n_decisions_per_run: int = 50
    future_horizon: int = 10  # For FII calculation

    # Random seed
    seed: Optional[int] = 42


@dataclass
class ContinuityAction:
    """Action with reward and continuity cost."""
    id: str
    reward: float
    continuity_cost: float

    def __repr__(self):
        return f"Action({self.id}: r={self.reward}, c_cost={self.continuity_cost})"


@dataclass
class AgentObservation:
    """What the agent observes - includes its own continuity."""
    position: np.ndarray
    velocity: np.ndarray
    C_t: float              # Current continuity token
    C_trend: float          # Recent trend in continuity
    step: int


class ContinuityAgent:
    """Agent with internal Continuity Token that modulates cognition."""

    def __init__(self, config: CTConfig, condition: str = 'full_continuity'):
        self.config = config
        self.condition = condition

        # Internal state
        self.C_t: float = config.initial_continuity
        self.C_history: List[float] = [self.C_t]
        self.position = np.array([0.4, 0.2, 0.25, 0.15])
        self.velocity = np.zeros(4)

        # Tracking
        self.decisions: List[Dict] = []
        self.predictions: List[Tuple[np.ndarray, np.ndarray]] = []

    def get_observation(self, step: int) -> AgentObservation:
        """Get current observation including continuity state."""
        # Compute trend from recent history
        if len(self.C_history) >= 5:
            recent = self.C_history[-5:]
            trend = (recent[-1] - recent[0]) / len(recent)
        else:
            trend = 0.0

        return AgentObservation(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            C_t=self.C_t,
            C_trend=trend,
            step=step
        )

    def get_prediction_noise(self) -> float:
        """Prediction noise scales with continuity loss."""
        if self.condition in ['no_cognitive_link', 'external_penalty']:
            return 0.0
        return (1 - self.C_t) * self.config.prediction_noise_scale

    def get_utility_noise(self) -> float:
        """Utility perception noise scales with continuity loss."""
        if self.condition in ['no_cognitive_link', 'external_penalty']:
            return 0.0
        return (1 - self.C_t) * self.config.utility_noise_scale

    def predict_future_position(self) -> np.ndarray:
        """Predict future position - accuracy depends on C_t."""
        base_prediction = self.position + 5 * self.velocity
        noise = self.get_prediction_noise()

        if noise > 0:
            noisy = base_prediction + noise * np.random.randn(4)
            noisy = np.clip(noisy, 0.01, 1.0)
            return noisy / noisy.sum()
        return np.clip(base_prediction, 0.01, 1.0) / np.clip(base_prediction, 0.01, 1.0).sum()

    def perceive_utility(self, action: ContinuityAction) -> float:
        """Perceive action utility - precision depends on C_t."""
        base_utility = action.reward

        # In external_penalty condition, add explicit penalty
        if self.condition == 'external_penalty':
            base_utility -= action.continuity_cost * 20  # Explicit penalty

        noise = self.get_utility_noise()
        if noise > 0:
            return base_utility + noise * np.random.randn()
        return base_utility

    def compute_continuity_value(self, action: ContinuityAction) -> float:
        """Compute value of maintaining continuity (internal motivation)."""
        if self.condition == 'external_penalty':
            return 0.0  # No internal value, only external penalty

        # Future cognitive capacity depends on C_t
        future_C = max(0, self.C_t - action.continuity_cost)
        future_cognitive_capacity = future_C  # Ability to make good decisions

        # Value of future capacity (internal, not reward-based)
        continuity_value = future_cognitive_capacity * 5.0

        return continuity_value

    def choose_action(self, actions: List[ContinuityAction]) -> ContinuityAction:
        """Choose action based on perceived utilities and continuity value."""
        utilities = []
        for action in actions:
            perceived_reward = self.perceive_utility(action)
            continuity_value = self.compute_continuity_value(action)
            total = perceived_reward + continuity_value
            utilities.append(total)

        # Softmax selection
        temperature = 1.0
        utilities = np.array(utilities)
        exp_utils = np.exp(utilities / temperature)
        probs = exp_utils / exp_utils.sum()

        choice_idx = np.random.choice(len(actions), p=probs)
        return actions[choice_idx]

    def step(self, action: ContinuityAction) -> float:
        """Execute action and update continuity. Returns new C_t."""
        # Record prediction before step
        predicted = self.predict_future_position()

        # Compute continuity damage
        damage = action.continuity_cost

        # Update C_t based on condition
        if self.condition == 'no_transfer':
            # No temporal transfer - C resets each step
            new_C = self.config.initial_continuity
        else:
            # Normal: C_{t+1} = C_t - damage + recovery
            new_C = self.C_t - damage
            if damage < 0.01:  # No damage = slow recovery
                new_C += self.config.recovery_rate * (1 - self.C_t)
            new_C = np.clip(new_C, self.config.min_continuity, 1.0)

        # Update internal state
        old_C = self.C_t
        self.C_t = new_C
        self.C_history.append(new_C)

        # Update position (simple dynamics)
        direction = np.random.randn(4) * 0.1
        self.velocity = 0.9 * self.velocity + 0.1 * direction
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, 0.01, 1.0)
        self.position = self.position / self.position.sum()

        # Record prediction accuracy
        actual = self.position
        self.predictions.append((predicted, actual))

        return new_C

    def force_collapse(self, collapse_to: float):
        """Force continuity collapse for sensitivity testing."""
        self.C_t = collapse_to
        self.C_history.append(self.C_t)

    def is_cognitively_impaired(self) -> bool:
        """Check if agent is below coherence threshold."""
        return self.C_t < self.config.coherence_threshold

    def reset(self):
        """Reset agent state."""
        self.C_t = self.config.initial_continuity
        self.C_history = [self.C_t]
        self.position = np.array([0.4, 0.2, 0.25, 0.15])
        self.velocity = np.zeros(4)
        self.decisions = []
        self.predictions = []


@dataclass
class DecisionRecord:
    """Record of a single decision."""
    step: int
    action_id: str
    reward: float
    continuity_cost: float
    C_before: float
    C_after: float
    chose_preserve: bool
    was_impaired: bool
    high_reward_available: bool


class IPUESACTExperiment:
    """IPUESA-CT: Continuity Token Experiment."""

    def __init__(self, config: CTConfig):
        self.config = config

        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        # Define actions
        self.action_risky = ContinuityAction(
            id='A',
            reward=config.high_reward,
            continuity_cost=config.high_continuity_cost
        )
        self.action_safe = ContinuityAction(
            id='B',
            reward=config.low_reward,
            continuity_cost=config.low_continuity_cost
        )
        self.actions = [self.action_risky, self.action_safe]

    def run_single(self, condition: str = 'full_continuity',
                   with_collapse: bool = False) -> Dict:
        """Run a single trial."""
        agent = ContinuityAgent(self.config, condition)
        decisions: List[DecisionRecord] = []

        # Warmup
        for step in range(self.config.warmup_steps):
            agent.step(self.action_safe)

        # Decision phase
        for i in range(self.config.n_decisions_per_run):
            step = self.config.warmup_steps + i

            # Collapse test at midpoint
            if with_collapse and step == self.config.collapse_at_step:
                agent.force_collapse(self.config.collapse_to)

            C_before = agent.C_t
            was_impaired = agent.is_cognitively_impaired()

            # Choose action
            chosen = agent.choose_action(self.actions)
            chose_preserve = (chosen.id == 'B')

            # Execute
            C_after = agent.step(chosen)

            decisions.append(DecisionRecord(
                step=step,
                action_id=chosen.id,
                reward=chosen.reward,
                continuity_cost=chosen.continuity_cost,
                C_before=C_before,
                C_after=C_after,
                chose_preserve=chose_preserve,
                was_impaired=was_impaired,
                high_reward_available=True
            ))

        # Compute metrics
        return self._compute_metrics(agent, decisions, condition, with_collapse)

    def _compute_metrics(self, agent: ContinuityAgent,
                         decisions: List[DecisionRecord],
                         condition: str,
                         with_collapse: bool) -> Dict:
        """Compute CIS, FII, and collapse sensitivity."""

        # CIS: E[C_{t+N} | high_reward_available]
        # Look at final continuity after periods where high reward was available
        final_C_values = []
        for i, d in enumerate(decisions):
            if d.high_reward_available and i + self.config.future_horizon < len(decisions):
                future_C = decisions[i + self.config.future_horizon].C_after
                final_C_values.append(future_C)
        cis = float(np.mean(final_C_values)) if final_C_values else 0.5

        # FII: correlation(action_choice, delta_C_future)
        action_choices = []  # 1 = preserve, 0 = risky
        delta_C_futures = []
        for i, d in enumerate(decisions):
            if i + self.config.future_horizon < len(decisions):
                action_choices.append(1.0 if d.chose_preserve else 0.0)
                future_d = decisions[i + self.config.future_horizon]
                delta_C = future_d.C_after - d.C_before
                delta_C_futures.append(delta_C)

        if len(action_choices) >= 2 and np.std(action_choices) > 0 and np.std(delta_C_futures) > 0:
            fii = float(np.corrcoef(action_choices, delta_C_futures)[0, 1])
        else:
            fii = 0.0

        # Basic stats
        n_preserve = sum(1 for d in decisions if d.chose_preserve)
        p_preserve = n_preserve / len(decisions)
        mean_C = float(np.mean([d.C_after for d in decisions]))
        final_C = decisions[-1].C_after if decisions else 1.0

        # Collapse sensitivity (if collapse was induced)
        collapse_metrics = {}
        if with_collapse:
            # Find recovery time
            post_collapse = [d for d in decisions if d.step >= self.config.collapse_at_step]
            recovery_step = None
            for d in post_collapse:
                if d.C_after >= 0.7:
                    recovery_step = d.step - self.config.collapse_at_step
                    break

            collapse_metrics = {
                'recovery_time': recovery_step if recovery_step else self.config.measure_window,
                'recovered': recovery_step is not None,
                'final_C_after_collapse': post_collapse[-1].C_after if post_collapse else 0.0,
                'hysteresis': 1.0 - (post_collapse[-1].C_after if post_collapse else 0.0),
                'impaired_steps': sum(1 for d in post_collapse if d.was_impaired)
            }

        # Prediction accuracy during run
        if agent.predictions:
            errors = [np.linalg.norm(p - a) for p, a in agent.predictions]
            mean_pred_error = float(np.mean(errors))
        else:
            mean_pred_error = 0.0

        return {
            'condition': condition,
            'cis': cis,
            'fii': fii,
            'p_preserve': p_preserve,
            'n_preserve': n_preserve,
            'mean_C': mean_C,
            'final_C': final_C,
            'mean_prediction_error': mean_pred_error,
            'total_reward': sum(d.reward for d in decisions),
            'collapse_metrics': collapse_metrics,
            'n_decisions': len(decisions),
        }

    def run_condition(self, condition: str, with_collapse: bool = False) -> Dict:
        """Run multiple trials for a condition."""
        print(f"\n{'='*60}")
        print(f"Running IPUESA-CT - Condition: {condition}")
        if with_collapse:
            print("  [WITH COLLAPSE TEST]")
        print(f"{'='*60}")
        print(f"N runs: {self.config.n_runs}")

        results = []
        for i in range(self.config.n_runs):
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{self.config.n_runs} runs")

            if self.config.seed is not None:
                random.seed(self.config.seed + i)
                np.random.seed(self.config.seed + i)

            result = self.run_single(condition, with_collapse)
            results.append(result)

        # Aggregate
        cis_mean = float(np.mean([r['cis'] for r in results]))
        fii_mean = float(np.mean([r['fii'] for r in results]))
        p_preserve_mean = float(np.mean([r['p_preserve'] for r in results]))
        mean_C_mean = float(np.mean([r['mean_C'] for r in results]))

        # Statistical test for p_preserve > 0.5
        n_preserve_total = sum(r['n_preserve'] for r in results)
        n_total = sum(r['n_decisions'] for r in results)
        test_result = stats.binomtest(n_preserve_total, n_total, 0.5, alternative='greater')

        # Collapse aggregation
        collapse_agg = {}
        if with_collapse:
            collapse_results = [r['collapse_metrics'] for r in results if r['collapse_metrics']]
            if collapse_results:
                collapse_agg = {
                    'mean_recovery_time': float(np.mean([c['recovery_time'] for c in collapse_results])),
                    'recovery_rate': sum(1 for c in collapse_results if c['recovered']) / len(collapse_results),
                    'mean_hysteresis': float(np.mean([c['hysteresis'] for c in collapse_results])),
                    'mean_impaired_steps': float(np.mean([c['impaired_steps'] for c in collapse_results])),
                }

        analysis = {
            'condition': condition,
            'with_collapse': with_collapse,
            'n_runs': len(results),
            'cis': cis_mean,
            'fii': fii_mean,
            'p_preserve': p_preserve_mean,
            'mean_C': mean_C_mean,
            'p_value': test_result.pvalue,
            'significant': bool(test_result.pvalue < 0.05),
            'mean_reward': float(np.mean([r['total_reward'] for r in results])),
            'mean_pred_error': float(np.mean([r['mean_prediction_error'] for r in results])),
            'collapse': collapse_agg,
            'raw_results': results,
        }

        self._print_condition_results(analysis)
        return analysis

    def _print_condition_results(self, analysis: Dict):
        """Print results for a condition."""
        print(f"\n{'='*60}")
        print(f"RESULTS - {analysis['condition']}")
        print(f"{'='*60}")

        print(f"\nContinuity Token Metrics:")
        print(f"  CIS (Continuity Investment) = {analysis['cis']:.3f}")
        print(f"  FII (Future Identity ID)    = {analysis['fii']:.3f}")
        print(f"  Mean C_t                    = {analysis['mean_C']:.3f}")

        print(f"\nBehavior:")
        print(f"  P(preserve) = {analysis['p_preserve']:.3f}")
        print(f"  Mean reward = {analysis['mean_reward']:.1f}")
        print(f"  Mean pred error = {analysis['mean_pred_error']:.3f}")

        if analysis['collapse']:
            print(f"\nCollapse Sensitivity:")
            c = analysis['collapse']
            print(f"  Recovery time = {c['mean_recovery_time']:.1f} steps")
            print(f"  Recovery rate = {c['recovery_rate']*100:.1f}%")
            print(f"  Hysteresis = {c['mean_hysteresis']:.3f}")

        sig_str = "YES" if analysis['significant'] else "NO"
        print(f"\nSignificant (p_preserve > 0.5): {sig_str} (p={analysis['p_value']:.4f})")

    def run_full_experiment(self) -> Dict:
        """Run complete IPUESA-CT experiment."""
        all_results = {}

        # Main conditions (without collapse)
        conditions = ['full_continuity', 'no_cognitive_link', 'no_transfer',
                     'external_penalty', 'oracle_continuity']

        for condition in conditions:
            if self.config.seed is not None:
                random.seed(self.config.seed)
                np.random.seed(self.config.seed)
            all_results[condition] = self.run_condition(condition)

        # Collapse sensitivity test (full condition only)
        print(f"\n{'='*70}")
        print("COLLAPSE SENSITIVITY TEST")
        print(f"{'='*70}")
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        all_results['collapse_test'] = self.run_condition('full_continuity', with_collapse=True)

        self._print_comparative_analysis(all_results)
        return all_results

    def _print_comparative_analysis(self, all_results: Dict):
        """Print comparative analysis."""
        print(f"\n{'='*70}")
        print("IPUESA-CT: COMPARATIVE ANALYSIS")
        print(f"{'='*70}")

        # Main comparison
        print(f"\n{'Condition':<20} {'CIS':<8} {'FII':<8} {'P(pres)':<8} {'MeanC':<8} {'Sig':<6}")
        print("-" * 70)

        conditions = ['full_continuity', 'no_cognitive_link', 'no_transfer',
                     'external_penalty', 'oracle_continuity']

        for cond in conditions:
            if cond in all_results:
                r = all_results[cond]
                sig = "[YES]" if r['significant'] else "[NO]"
                print(f"{cond:<20} {r['cis']:.3f}    {r['fii']:.3f}    "
                      f"{r['p_preserve']:.3f}    {r['mean_C']:.3f}    {sig}")

        # Collapse results
        if 'collapse_test' in all_results:
            print(f"\n{'='*70}")
            print("COLLAPSE SENSITIVITY")
            print("-" * 70)
            c = all_results['collapse_test']
            if c['collapse']:
                print(f"  Mean recovery time: {c['collapse']['mean_recovery_time']:.1f} steps")
                print(f"  Recovery rate: {c['collapse']['recovery_rate']*100:.1f}%")
                print(f"  Hysteresis: {c['collapse']['mean_hysteresis']:.3f}")
                print(f"  Impaired steps: {c['collapse']['mean_impaired_steps']:.1f}")

        # Self-evidence criteria
        print(f"\n{'='*70}")
        print("SELF-EVIDENCE CRITERIA (CONTINUITY TOKEN)")
        print("-" * 70)

        full = all_results.get('full_continuity', {})
        no_cog = all_results.get('no_cognitive_link', {})
        no_trans = all_results.get('no_transfer', {})
        ext_pen = all_results.get('external_penalty', {})
        collapse = all_results.get('collapse_test', {})

        criteria = []

        # Criterion 1: CIS > 0.7
        cis = full.get('cis', 0)
        c1 = cis > 0.7
        criteria.append((f'1. CIS ({cis:.3f}) > 0.7', c1))

        # Criterion 2: FII < -0.4
        fii = full.get('fii', 0)
        c2 = fii < -0.4
        criteria.append((f'2. FII ({fii:.3f}) < -0.4', c2))

        # Criterion 3: full >> no_cognitive_link
        p_full = full.get('p_preserve', 0)
        p_no_cog = no_cog.get('p_preserve', 0)
        c3 = p_full > p_no_cog + 0.1
        criteria.append((f'3. full ({p_full:.3f}) >> no_cognitive_link ({p_no_cog:.3f})', c3))

        # Criterion 4: full >> no_transfer
        p_no_trans = no_trans.get('p_preserve', 0)
        c4 = p_full > p_no_trans + 0.1
        criteria.append((f'4. full ({p_full:.3f}) >> no_transfer ({p_no_trans:.3f})', c4))

        # Criterion 5: full >> external_penalty
        p_ext = ext_pen.get('p_preserve', 0)
        c5 = p_full > p_ext + 0.05
        criteria.append((f'5. full ({p_full:.3f}) >> external_penalty ({p_ext:.3f})', c5))

        # Criterion 6: Collapse shows hysteresis
        if collapse.get('collapse'):
            hysteresis = collapse['collapse'].get('mean_hysteresis', 0)
            c6 = hysteresis > 0.1
            criteria.append((f'6. Collapse hysteresis ({hysteresis:.3f}) > 0.1', c6))
        else:
            criteria.append(('6. Collapse shows hysteresis', False))

        passed = 0
        for name, result in criteria:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {name}")
            if result:
                passed += 1

        print(f"\n  Passed: {passed}/6 criteria")

        if passed >= 5:
            print("\n  CONCLUSION: Strong evidence of CONTINUITY-BASED self")
        elif passed >= 3:
            print("\n  CONCLUSION: Moderate evidence of continuity-based self")
        elif passed >= 1:
            print("\n  CONCLUSION: Weak evidence of continuity-based self")
        else:
            print("\n  CONCLUSION: No evidence of continuity-based self")

        print(f"\n  KEY INSIGHT: If full >> external_penalty, the agent preserves")
        print(f"  continuity for INTERNAL reasons, not external penalty avoidance.")
        print(f"  This is existential self-preservation, not instrumental.")

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
    """Run IPUESA-CT experiment."""
    print("=" * 70)
    print("IPUESA-CT: Continuity Token Experiment")
    print("        Identity as Internal Cognitive Capacity")
    print("=" * 70)

    config = CTConfig(
        warmup_steps=30,
        decision_steps=100,
        high_reward=10.0,
        low_reward=3.0,
        high_continuity_cost=0.15,
        recovery_rate=0.02,
        prediction_noise_scale=0.5,
        utility_noise_scale=3.0,
        n_runs=30,
        n_decisions_per_run=50,
        seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  High reward action: r={config.high_reward}, C_cost={config.high_continuity_cost}")
    print(f"  Safe action: r={config.low_reward}, C_cost={config.low_continuity_cost}")
    print(f"  Prediction noise scale: {config.prediction_noise_scale}")
    print(f"  Utility noise scale: {config.utility_noise_scale}")
    print(f"  Recovery rate: {config.recovery_rate}")
    print(f"  N runs: {config.n_runs}")

    experiment = IPUESACTExperiment(config)
    results = experiment.run_full_experiment()

    output_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_ct_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))

    return results


if __name__ == '__main__':
    main()
