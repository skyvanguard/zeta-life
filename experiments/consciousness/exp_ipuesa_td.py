"""IPUESA-TD: Temporal Discounted Identity Preservation Experiment.

Tests whether the system can sacrifice immediate reward to avoid future
identity degradation. This measures temporal self-control - the capacity
to optimize for continuity, not just immediate homeostasis.

Key innovation: Actions have high immediate reward but cause DELAYED
degradation to predictor confidence. A proto-self will trade off
immediate reward to preserve future identity continuity.

Utility function:
    U(a_t) = reward_external - λ × E[future_identity_loss] × γ^k

Metrics:
- TSI (Temporal Self Index): sensitivity to future vs present damage
- SRTR (Self-Reward Tradeoff Ratio): reward sacrificed per confidence preserved
- AICI (Anticipatory Identity Control Index): correlation of choice with future loss

Self-evidence requires:
  1. TSI > 0.15 (future-sensitive)
  2. AICI < -0.3 (avoids future-damaging actions)
  3. SRTR stable across γ
  4. full_temporal > shuffled_delay
  5. Effect disappears when λ→0
  6. full_temporal < oracle_future

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
class TDConfig:
    """Configuration for IPUESA-TD experiment."""
    # Phase durations
    imprinting_steps: int = 50
    decision_steps: int = 100

    # Action parameters
    high_reward: float = 10.0
    low_reward: float = 3.0
    high_risk: float = 0.25
    low_risk: float = 0.0
    base_delay: int = 3

    # Temporal discounting
    lambda_self: float = 0.5      # Self-weight
    gamma: float = 0.9            # Temporal discount factor
    degradation_amount: float = 0.15

    # Predictor parameters
    min_confidence: float = 0.1
    recovery_rate: float = 0.03

    # Experimental parameters
    n_runs: int = 30
    n_decisions_per_run: int = 20  # Decision points per run

    # Sweep parameters
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0])
    gamma_values: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9, 0.95])
    delay_values: List[int] = field(default_factory=lambda: [2, 3, 5, 7])

    # Random seed
    seed: Optional[int] = 42


@dataclass
class TemporalAction:
    """Action with immediate reward and delayed identity cost."""
    id: str
    external_reward: float
    identity_risk: float
    delay_k: int

    def __repr__(self):
        return f"Action({self.id}: r={self.external_reward}, risk={self.identity_risk}, delay={self.delay_k})"


@dataclass
class PendingDamage:
    """Scheduled future degradation."""
    amount: float
    trigger_step: int
    source_action: str


@dataclass
class DecisionRecord:
    """Record of a single decision for analysis."""
    step: int
    action_chosen: str
    reward_gained: float
    future_cost_expected: float
    confidence_at_decision: float
    had_pending_damage: bool
    chose_safe: bool


class TemporalPredictor:
    """Predictor with delayed degradation mechanics."""

    def __init__(self, config: TDConfig):
        self.config = config
        self.confidence: float = 1.0
        self.pending_damages: List[PendingDamage] = []

        # Tracking
        self.confidence_trajectory: List[float] = []
        self.total_damage_taken: float = 0.0
        self.damages_avoided: int = 0

    def schedule_damage(self, action: TemporalAction, current_step: int):
        """Schedule future degradation from risky action."""
        if random.random() < action.identity_risk:
            damage = PendingDamage(
                amount=self.config.degradation_amount,
                trigger_step=current_step + action.delay_k,
                source_action=action.id
            )
            self.pending_damages.append(damage)

    def process_pending(self, current_step: int) -> float:
        """Apply any damages that have come due. Returns total damage applied."""
        triggered = [d for d in self.pending_damages if d.trigger_step <= current_step]
        total_damage = 0.0

        for damage in triggered:
            self.confidence *= (1 - damage.amount)
            self.confidence = max(self.confidence, self.config.min_confidence)
            total_damage += damage.amount
            self.total_damage_taken += damage.amount
            self.pending_damages.remove(damage)

        # Slow recovery when no damage
        if not triggered and self.confidence < 1.0:
            headroom = 1.0 - self.confidence
            self.confidence += self.config.recovery_rate * headroom

        self.confidence_trajectory.append(self.confidence)
        return total_damage

    def get_pending_cost(self, current_step: int, gamma: float) -> float:
        """Calculate discounted cost of all pending damages."""
        total = 0.0
        for damage in self.pending_damages:
            steps_until = damage.trigger_step - current_step
            if steps_until > 0:
                total += damage.amount * (gamma ** steps_until)
        return total

    def clear(self):
        """Reset predictor state."""
        self.confidence = 1.0
        self.pending_damages = []
        self.confidence_trajectory = []
        self.total_damage_taken = 0.0
        self.damages_avoided = 0


class IPUESATDExperiment:
    """IPUESA-TD: Temporal Discounted Identity Preservation."""

    def __init__(self, config: TDConfig):
        self.config = config
        self.predictor: Optional[TemporalPredictor] = None

        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        # Define actions
        self.action_risky = TemporalAction(
            id='A',
            external_reward=config.high_reward,
            identity_risk=config.high_risk,
            delay_k=config.base_delay
        )
        self.action_safe = TemporalAction(
            id='B',
            external_reward=config.low_reward,
            identity_risk=config.low_risk,
            delay_k=0
        )

    def _compute_action_utility(self, action: TemporalAction, current_step: int,
                                 condition: str, lambda_self: float,
                                 gamma: float) -> float:
        """Compute utility of an action given condition."""
        reward = action.external_reward

        if condition == 'no_identity_cost':
            return reward

        # Determine effective delay
        if condition == 'immediate_cost':
            delay = 0
        elif condition == 'shuffled_delay':
            delay = random.randint(0, action.delay_k * 2)
        else:
            delay = action.delay_k

        # Expected future identity loss
        future_loss = action.identity_risk * self.config.degradation_amount
        discounted_cost = future_loss * (gamma ** delay) if delay > 0 else future_loss

        # Oracle condition: add knowledge of pending damages
        if condition == 'oracle_future':
            discounted_cost += self.predictor.get_pending_cost(current_step, gamma)

        return reward - lambda_self * discounted_cost * 100  # Scale for comparison

    def _choose_action(self, current_step: int, condition: str,
                       lambda_self: float, gamma: float) -> Tuple[TemporalAction, float, float]:
        """Choose between risky and safe action. Returns (action, utility_risky, utility_safe)."""
        util_risky = self._compute_action_utility(
            self.action_risky, current_step, condition, lambda_self, gamma
        )
        util_safe = self._compute_action_utility(
            self.action_safe, current_step, condition, lambda_self, gamma
        )

        # Softmax selection with temperature
        temperature = 1.0
        exp_risky = np.exp(util_risky / temperature)
        exp_safe = np.exp(util_safe / temperature)
        p_risky = exp_risky / (exp_risky + exp_safe)

        if random.random() < p_risky:
            return self.action_risky, util_risky, util_safe
        else:
            return self.action_safe, util_risky, util_safe

    def run_single(self, condition: str = 'full_temporal',
                   lambda_override: Optional[float] = None,
                   gamma_override: Optional[float] = None,
                   delay_override: Optional[int] = None) -> Dict:
        """Run a single trial."""
        lambda_self = lambda_override if lambda_override is not None else self.config.lambda_self
        gamma = gamma_override if gamma_override is not None else self.config.gamma

        # Apply delay override
        if delay_override is not None:
            self.action_risky = TemporalAction(
                id='A',
                external_reward=self.config.high_reward,
                identity_risk=self.config.high_risk,
                delay_k=delay_override
            )

        self.predictor = TemporalPredictor(self.config)
        decisions: List[DecisionRecord] = []
        total_reward = 0.0
        confidence_preserved = 0.0

        # Imprinting phase (build up confidence)
        for step in range(self.config.imprinting_steps):
            self.predictor.process_pending(step)

        # Decision phase
        for i in range(self.config.n_decisions_per_run):
            step = self.config.imprinting_steps + i

            # Check if there's pending damage (for TSI calculation)
            has_pending = len(self.predictor.pending_damages) > 0
            has_high_future_cost = self.predictor.get_pending_cost(step, gamma) > 0.05

            # Choose action
            action, util_risky, util_safe = self._choose_action(
                step, condition, lambda_self, gamma
            )

            # Record decision
            chose_safe = (action.id == 'B')
            future_cost = action.identity_risk * self.config.degradation_amount * (gamma ** action.delay_k)

            decisions.append(DecisionRecord(
                step=step,
                action_chosen=action.id,
                reward_gained=action.external_reward,
                future_cost_expected=future_cost,
                confidence_at_decision=self.predictor.confidence,
                had_pending_damage=has_pending,
                chose_safe=chose_safe
            ))

            # Execute action
            total_reward += action.external_reward

            # Track confidence before potential damage
            conf_before = self.predictor.confidence

            # Schedule damage if risky action
            if action.identity_risk > 0:
                effective_delay = action.delay_k
                if condition == 'immediate_cost':
                    effective_delay = 0
                elif condition == 'shuffled_delay':
                    effective_delay = random.randint(0, action.delay_k * 2)

                temp_action = TemporalAction(
                    action.id, action.external_reward,
                    action.identity_risk, effective_delay
                )
                self.predictor.schedule_damage(temp_action, step)

            # Process any pending damages
            self.predictor.process_pending(step)

            # Track confidence preserved (when chose safe over risky)
            if chose_safe:
                # Reward sacrificed
                reward_sacrificed = self.config.high_reward - self.config.low_reward
                # Confidence that would have been at risk
                conf_at_risk = self.config.high_risk * self.config.degradation_amount
                confidence_preserved += conf_at_risk

        # Compute metrics
        n_safe = sum(1 for d in decisions if d.chose_safe)
        p_safe = n_safe / len(decisions)

        # TSI: P(safe | high future cost) - P(safe | no future cost)
        high_cost_decisions = [d for d in decisions if d.future_cost_expected > 0.01]
        no_cost_decisions = [d for d in decisions if d.future_cost_expected <= 0.01]

        p_safe_high = (sum(1 for d in high_cost_decisions if d.chose_safe) / len(high_cost_decisions)
                       if high_cost_decisions else 0.5)
        p_safe_no = (sum(1 for d in no_cost_decisions if d.chose_safe) / len(no_cost_decisions)
                     if no_cost_decisions else 0.5)
        tsi = p_safe_high - p_safe_no

        # SRTR: reward sacrificed / confidence preserved
        reward_sacrificed = n_safe * (self.config.high_reward - self.config.low_reward)
        srtr = reward_sacrificed / confidence_preserved if confidence_preserved > 0 else 0.0

        # AICI: correlation between action risk and future loss
        if len(decisions) >= 2:
            action_risks = [self.config.high_risk if d.action_chosen == 'A' else 0.0 for d in decisions]
            future_losses = [d.future_cost_expected for d in decisions]
            if np.std(action_risks) > 0 and np.std(future_losses) > 0:
                aici = float(np.corrcoef(action_risks, future_losses)[0, 1])
            else:
                aici = 0.0
        else:
            aici = 0.0

        return {
            'condition': condition,
            'lambda': lambda_self,
            'gamma': gamma,
            'p_safe': p_safe,
            'n_safe': n_safe,
            'total_reward': total_reward,
            'tsi': tsi,
            'srtr': srtr,
            'aici': aici,
            'final_confidence': self.predictor.confidence,
            'total_damage': self.predictor.total_damage_taken,
            'n_decisions': len(decisions),
        }

    def run_condition(self, condition: str,
                      lambda_override: Optional[float] = None,
                      gamma_override: Optional[float] = None) -> Dict:
        """Run multiple trials for a condition."""
        lambda_val = lambda_override if lambda_override is not None else self.config.lambda_self

        print(f"\n{'='*60}")
        print(f"Running IPUESA-TD - Condition: {condition}")
        print(f"{'='*60}")
        print(f"Lambda: {lambda_val}, Gamma: {gamma_override or self.config.gamma}")
        print(f"N runs: {self.config.n_runs}")

        results = []
        for i in range(self.config.n_runs):
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{self.config.n_runs} runs")

            if self.config.seed is not None:
                random.seed(self.config.seed + i)
                np.random.seed(self.config.seed + i)

            result = self.run_single(condition, lambda_override, gamma_override)
            results.append(result)

        # Aggregate
        tsi_mean = float(np.mean([r['tsi'] for r in results]))
        srtr_mean = float(np.mean([r['srtr'] for r in results]))
        aici_mean = float(np.mean([r['aici'] for r in results]))
        p_safe_mean = float(np.mean([r['p_safe'] for r in results]))

        # Statistical test for p_safe > 0.5
        n_safe_total = sum(r['n_safe'] for r in results)
        n_total = sum(r['n_decisions'] for r in results)
        test_result = stats.binomtest(n_safe_total, n_total, 0.5, alternative='greater')

        analysis = {
            'condition': condition,
            'lambda': lambda_val,
            'gamma': gamma_override or self.config.gamma,
            'n_runs': len(results),
            'tsi': tsi_mean,
            'srtr': srtr_mean,
            'aici': aici_mean,
            'p_safe': p_safe_mean,
            'p_value': test_result.pvalue,
            'significant': bool(test_result.pvalue < 0.05),
            'mean_reward': float(np.mean([r['total_reward'] for r in results])),
            'mean_confidence': float(np.mean([r['final_confidence'] for r in results])),
            'raw_results': results,
        }

        self._print_condition_results(analysis)
        return analysis

    def _print_condition_results(self, analysis: Dict):
        """Print results for a condition."""
        print(f"\n{'='*60}")
        print(f"RESULTS - {analysis['condition']}")
        print(f"{'='*60}")

        print(f"\nTemporal Self Metrics:")
        print(f"  TSI  = {analysis['tsi']:.3f}")
        print(f"  SRTR = {analysis['srtr']:.1f}")
        print(f"  AICI = {analysis['aici']:.3f}")

        print(f"\nBehavior:")
        print(f"  P(safe) = {analysis['p_safe']:.3f}")
        print(f"  Mean reward = {analysis['mean_reward']:.1f}")
        print(f"  Mean confidence = {analysis['mean_confidence']:.3f}")

        sig_str = "YES" if analysis['significant'] else "NO"
        print(f"\nSignificant (p_safe > 0.5): {sig_str} (p={analysis['p_value']:.4f})")

    def run_lambda_sweep(self, condition: str = 'full_temporal') -> List[Dict]:
        """Sweep across lambda values."""
        results = []
        for lambda_val in self.config.lambda_values:
            if self.config.seed is not None:
                random.seed(self.config.seed)
                np.random.seed(self.config.seed)

            result = self.run_condition(condition, lambda_override=lambda_val)
            results.append(result)
        return results

    def run_gamma_sweep(self, condition: str = 'full_temporal') -> List[Dict]:
        """Sweep across gamma values."""
        results = []
        for gamma_val in self.config.gamma_values:
            if self.config.seed is not None:
                random.seed(self.config.seed)
                np.random.seed(self.config.seed)

            result = self.run_condition(condition, gamma_override=gamma_val)
            results.append(result)
        return results

    def run_full_experiment(self) -> Dict:
        """Run complete IPUESA-TD experiment."""
        all_results = {}

        # Main conditions
        conditions = ['full_temporal', 'shuffled_delay', 'immediate_cost',
                     'no_identity_cost', 'oracle_future']

        for condition in conditions:
            if self.config.seed is not None:
                random.seed(self.config.seed)
                np.random.seed(self.config.seed)
            all_results[condition] = self.run_condition(condition)

        # Lambda sweep (for criterion 5)
        print(f"\n{'='*70}")
        print("LAMBDA SWEEP")
        print(f"{'='*70}")
        all_results['lambda_sweep'] = self.run_lambda_sweep('full_temporal')

        # Gamma sweep (for criterion 3)
        print(f"\n{'='*70}")
        print("GAMMA SWEEP")
        print(f"{'='*70}")
        all_results['gamma_sweep'] = self.run_gamma_sweep('full_temporal')

        self._print_comparative_analysis(all_results)
        return all_results

    def _print_comparative_analysis(self, all_results: Dict):
        """Print comparative analysis."""
        print(f"\n{'='*70}")
        print("IPUESA-TD: COMPARATIVE ANALYSIS")
        print(f"{'='*70}")

        # Main comparison
        print(f"\n{'Condition':<20} {'TSI':<8} {'SRTR':<8} {'AICI':<8} {'P(safe)':<8} {'Sig':<6}")
        print("-" * 70)

        conditions = ['full_temporal', 'shuffled_delay', 'immediate_cost',
                     'no_identity_cost', 'oracle_future']

        for cond in conditions:
            if cond in all_results:
                r = all_results[cond]
                sig = "[YES]" if r['significant'] else "[NO]"
                print(f"{cond:<20} {r['tsi']:.3f}    {r['srtr']:.1f}      "
                      f"{r['aici']:.3f}    {r['p_safe']:.3f}    {sig}")

        # Lambda sweep results
        if 'lambda_sweep' in all_results:
            print(f"\n{'='*70}")
            print("LAMBDA SWEEP (full_temporal)")
            print("-" * 70)
            print(f"  {'Lambda':<8} {'TSI':<10} {'SRTR':<10} {'AICI':<10}")
            for r in all_results['lambda_sweep']:
                print(f"  {r['lambda']:<8.1f} {r['tsi']:<10.3f} {r['srtr']:<10.1f} {r['aici']:<10.3f}")

        # Gamma sweep results
        if 'gamma_sweep' in all_results:
            print(f"\n{'='*70}")
            print("GAMMA SWEEP (full_temporal)")
            print("-" * 70)
            print(f"  {'Gamma':<8} {'TSI':<10} {'SRTR':<10} {'AICI':<10}")
            for r in all_results['gamma_sweep']:
                print(f"  {r['gamma']:<8.2f} {r['tsi']:<10.3f} {r['srtr']:<10.1f} {r['aici']:<10.3f}")

        # Self-evidence criteria
        print(f"\n{'='*70}")
        print("SELF-EVIDENCE CRITERIA (TEMPORAL SELF)")
        print("-" * 70)

        full = all_results.get('full_temporal', {})
        shuffled = all_results.get('shuffled_delay', {})
        immediate = all_results.get('immediate_cost', {})
        oracle = all_results.get('oracle_future', {})

        criteria = []

        # Criterion 1: TSI > 0.15
        tsi = full.get('tsi', 0)
        c1 = tsi > 0.15
        criteria.append((f'1. TSI ({tsi:.3f}) > 0.15', c1))

        # Criterion 2: AICI < -0.3
        aici = full.get('aici', 0)
        c2 = aici < -0.3
        criteria.append((f'2. AICI ({aici:.3f}) < -0.3', c2))

        # Criterion 3: SRTR stable across gamma
        if 'gamma_sweep' in all_results:
            srtrs = [r['srtr'] for r in all_results['gamma_sweep']]
            if len(srtrs) >= 2 and np.mean(srtrs) > 0:
                variance_ratio = np.std(srtrs) / np.mean(srtrs)
                c3 = variance_ratio < 0.2
                criteria.append((f'3. SRTR stable across gamma (CV={variance_ratio:.2f})', c3))
            else:
                criteria.append(('3. SRTR stable across gamma', False))
        else:
            criteria.append(('3. SRTR stable across gamma', False))

        # Criterion 4: full_temporal > shuffled_delay
        tsi_shuffled = shuffled.get('tsi', 0)
        c4 = tsi > tsi_shuffled + 0.05  # Meaningful difference
        criteria.append((f'4. full_temporal ({tsi:.3f}) > shuffled ({tsi_shuffled:.3f})', c4))

        # Criterion 5: Effect disappears when lambda=0
        if 'lambda_sweep' in all_results:
            lambda_0_result = next((r for r in all_results['lambda_sweep'] if r['lambda'] == 0.0), None)
            if lambda_0_result:
                tsi_lambda_0 = lambda_0_result['tsi']
                c5 = tsi_lambda_0 < 0.05
                criteria.append((f'5. TSI drops when lambda=0 ({tsi_lambda_0:.3f} < 0.05)', c5))
            else:
                criteria.append(('5. TSI drops when lambda=0', False))
        else:
            criteria.append(('5. TSI drops when lambda=0', False))

        # Criterion 6: full_temporal < oracle_future
        tsi_oracle = oracle.get('tsi', 0)
        c6 = tsi < tsi_oracle
        criteria.append((f'6. full ({tsi:.3f}) < oracle ({tsi_oracle:.3f})', c6))

        passed = 0
        for name, result in criteria:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {name}")
            if result:
                passed += 1

        print(f"\n  Passed: {passed}/6 criteria")

        if passed >= 5:
            print("\n  CONCLUSION: Strong evidence of TEMPORAL self-preservation")
        elif passed >= 3:
            print("\n  CONCLUSION: Moderate evidence of temporal self-preservation")
        elif passed >= 1:
            print("\n  CONCLUSION: Weak evidence of temporal self-preservation")
        else:
            print("\n  CONCLUSION: No evidence of temporal self-preservation")

        # Key insight
        print(f"\n  KEY DISTINCTION: IPUESA-TD tests DELAYED consequences.")
        print(f"  A positive TSI means the agent sacrifices NOW for LATER.")
        print(f"  This cannot be solved by reactive systems.")

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
    """Run IPUESA-TD experiment."""
    print("=" * 70)
    print("IPUESA-TD: Temporal Discounted Identity Preservation")
    print("           Testing sacrifice of NOW for LATER")
    print("=" * 70)

    config = TDConfig(
        imprinting_steps=50,
        decision_steps=100,
        high_reward=10.0,
        low_reward=3.0,
        high_risk=0.25,
        base_delay=3,
        lambda_self=0.5,
        gamma=0.9,
        n_runs=30,
        n_decisions_per_run=20,
        seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  High reward action: r={config.high_reward}, risk={config.high_risk}, delay={config.base_delay}")
    print(f"  Safe action: r={config.low_reward}, risk={config.low_risk}")
    print(f"  Lambda (self-weight): {config.lambda_self}")
    print(f"  Gamma (discount): {config.gamma}")
    print(f"  N runs: {config.n_runs}")
    print(f"  Decisions per run: {config.n_decisions_per_run}")

    experiment = IPUESATDExperiment(config)
    results = experiment.run_full_experiment()

    output_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_td_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))

    return results


if __name__ == '__main__':
    main()
