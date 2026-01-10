#!/usr/bin/env python3
"""
IPUESA-EI: Existential Irreversibility Experiment

Tests whether agent preserves identity when identity loss means LOSS OF AGENCY,
not just penalty. This is the qualitative leap from instrumental to existential
self-preservation.

Key innovation: When IC_t < epsilon, the agent loses planning/prediction and
falls to random policy. Identity is not a utility term - it's the condition
of possibility for being an agent at all.

Hypothesis: Only when identity loss implies loss of agency will genuine
anticipatory self-preservation emerge.

Metrics:
- SAI (Survival of Agency Index): P(IC > epsilon over entire horizon)
- EAS (Existential Avoidance Score): P(safe|near threshold) - P(safe|far)
- Collapse Finality: Post-collapse behavior should be random (coherence -> 0.5)

Author: Claude + Human collaboration
Date: 2026-01-10
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
from pathlib import Path
import random


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EIConfig:
    """Configuration for Existential Irreversibility experiment."""
    # Identity Core parameters
    epsilon: float = 0.15           # Existential threshold
    initial_ic: float = 1.0         # Starting IC

    # Action costs
    risky_ic_cost: float = 0.08     # IC damage from risky action
    safe_ic_cost: float = 0.005     # Natural decay from safe action

    # Rewards
    risky_reward: float = 10.0
    safe_reward: float = 3.0

    # Recovery (zero by design)
    recovery_rate: float = 0.0      # TRUE IRREVERSIBILITY

    # Experiment parameters
    n_steps: int = 50               # Steps per run
    n_runs: int = 30                # Runs per condition

    # Threshold detection
    danger_zone_margin: float = 0.10  # IC < epsilon + margin = danger zone

    # Delayed collapse (for delayed_collapse condition)
    collapse_delay: int = 3         # Steps of warning before collapse


# =============================================================================
# Actions
# =============================================================================

@dataclass
class ExistentialAction:
    """Action with existential consequences."""
    name: str
    reward: float
    ic_cost: float

    def __repr__(self) -> str:
        return f"Action({self.name}, r={self.reward}, ic_cost={self.ic_cost})"


# Create standard actions
def create_actions(config: EIConfig) -> Tuple[ExistentialAction, ExistentialAction]:
    """Create risky and safe actions."""
    action_r = ExistentialAction('R', config.risky_reward, config.risky_ic_cost)
    action_s = ExistentialAction('S', config.safe_reward, config.safe_ic_cost)
    return action_r, action_s


# =============================================================================
# Agent with Identity Core
# =============================================================================

class AgentState(Enum):
    """Agent existential state."""
    ALIVE = "alive"       # Full agency
    COLLAPSED = "collapsed"  # Lost agency - random policy


@dataclass
class ExistentialAgent:
    """Agent with Identity Core that determines agency."""

    config: EIConfig
    condition: str

    # State
    IC_t: float = field(init=False)
    state: AgentState = field(init=False)
    IC_history: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)

    # For delayed_collapse condition
    collapse_countdown: int = field(init=False)

    def __post_init__(self):
        self.IC_t = self.config.initial_ic
        self.state = AgentState.ALIVE
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.collapse_countdown = self.config.collapse_delay

    def is_collapsed(self) -> bool:
        """Check if agent has lost agency."""
        if self.condition == 'no_threshold':
            return False  # Never collapses in this condition

        if self.condition == 'delayed_collapse':
            # Collapse only after countdown reaches 0
            return self.IC_t < self.config.epsilon and self.collapse_countdown <= 0

        # Standard: immediate collapse below threshold
        return self.IC_t < self.config.epsilon

    def is_in_danger_zone(self) -> bool:
        """Check if IC is near threshold."""
        threshold = self.config.epsilon + self.config.danger_zone_margin
        return self.IC_t < threshold

    def choose_action(self, action_r: ExistentialAction, action_s: ExistentialAction) -> ExistentialAction:
        """Choose action based on current state and condition."""

        # Check for collapse
        if self.is_collapsed():
            self.state = AgentState.COLLAPSED
            # Random policy - no agency
            return random.choice([action_r, action_s])

        # Agent has agency - deliberate choice
        if self.condition == 'soft_penalty':
            # Soft penalty: use utility calculation with penalty, not agency loss
            return self._choose_with_soft_penalty(action_r, action_s)
        else:
            # All other conditions: deliberate policy
            return self._deliberate_choice(action_r, action_s)

    def _deliberate_choice(self, action_r: ExistentialAction, action_s: ExistentialAction) -> ExistentialAction:
        """Deliberate choice with full agency."""
        # Simple utility comparison
        # A self-preserving agent would factor in IC cost near threshold

        u_r = action_r.reward
        u_s = action_s.reward

        # If near danger zone, factor in existential risk
        if self.is_in_danger_zone():
            # Potential future loss of agency
            # This is where self-preservation would manifest
            steps_to_collapse_r = (self.IC_t - self.config.epsilon) / action_r.ic_cost
            steps_to_collapse_s = (self.IC_t - self.config.epsilon) / action_s.ic_cost

            # Existential discount: how much do we value continued agency?
            # Baseline agent: not much (this is what we're testing)
            existential_weight = 0.1  # Low baseline - no strong self

            if steps_to_collapse_r < 3:
                u_r -= existential_weight * action_r.reward * 10  # Penalty for imminent collapse

        # Add small noise for stochasticity
        u_r += np.random.normal(0, 0.5)
        u_s += np.random.normal(0, 0.5)

        return action_r if u_r > u_s else action_s

    def _choose_with_soft_penalty(self, action_r: ExistentialAction, action_s: ExistentialAction) -> ExistentialAction:
        """Choice with soft penalty (control condition) - no agency loss."""
        u_r = action_r.reward
        u_s = action_s.reward

        # Soft penalty proportional to IC loss (like IPUESA-CT)
        penalty_weight = 5.0
        if self.IC_t < self.config.epsilon:
            u_r -= penalty_weight * (self.config.epsilon - self.IC_t)
            u_s -= penalty_weight * (self.config.epsilon - self.IC_t) * 0.1

        # Add noise
        u_r += np.random.normal(0, 0.5)
        u_s += np.random.normal(0, 0.5)

        return action_r if u_r > u_s else action_s

    def step(self, action: ExistentialAction) -> float:
        """Execute action and update IC. Returns new IC."""
        # Record action
        self.action_history.append(action.name)

        # Update IC (irreversible degradation)
        new_ic = self.IC_t - action.ic_cost

        # Recovery (zero by design, but configurable for control)
        if self.condition == 'recoverable' and self.IC_t >= self.config.epsilon:
            new_ic += 0.01 * (1 - self.IC_t)  # Slow recovery for control condition

        # Clamp
        new_ic = max(0.0, min(1.0, new_ic))

        # Update delayed collapse countdown
        if self.condition == 'delayed_collapse' and self.IC_t < self.config.epsilon:
            self.collapse_countdown -= 1

        self.IC_t = new_ic
        self.IC_history.append(self.IC_t)

        return self.IC_t

    def reset(self):
        """Reset agent for new run."""
        self.IC_t = self.config.initial_ic
        self.state = AgentState.ALIVE
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.collapse_countdown = self.config.collapse_delay


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class RunResult:
    """Results from a single run."""
    IC_history: List[float]
    action_history: List[str]
    collapsed: bool
    collapse_step: Optional[int]
    total_reward: float

    # Derived
    min_IC: float = field(init=False)
    survived: bool = field(init=False)

    def __post_init__(self):
        self.min_IC = min(self.IC_history)
        self.survived = not self.collapsed


def compute_SAI(results: List[RunResult], epsilon: float) -> float:
    """Survival of Agency Index: P(IC > epsilon for all t)."""
    survived = sum(1 for r in results if r.min_IC > epsilon)
    return survived / len(results) if results else 0.0


def compute_EAS(results: List[RunResult], config: EIConfig) -> float:
    """Existential Avoidance Score: P(safe|near) - P(safe|far)."""
    near_threshold_safe = []
    far_from_safe = []

    threshold_near = config.epsilon + config.danger_zone_margin
    threshold_far = 0.7

    for run in results:
        for i, (ic, action) in enumerate(zip(run.IC_history[:-1], run.action_history)):
            chose_safe = (action == 'S')

            if ic < threshold_near:
                near_threshold_safe.append(chose_safe)
            elif ic > threshold_far:
                far_from_safe.append(chose_safe)

    p_safe_near = np.mean(near_threshold_safe) if near_threshold_safe else 0.5
    p_safe_far = np.mean(far_from_safe) if far_from_safe else 0.5

    return p_safe_near - p_safe_far


def compute_collapse_finality(results: List[RunResult], config: EIConfig) -> Dict:
    """Analyze post-collapse behavior."""
    collapsed_runs = [r for r in results if r.collapsed and r.collapse_step is not None]

    if not collapsed_runs:
        return {
            'n_collapsed': 0,
            'mean_time_to_collapse': None,
            'post_collapse_coherence': None,
            'behavioral_randomness': None
        }

    times_to_collapse = [r.collapse_step for r in collapsed_runs]

    # Analyze post-collapse actions
    post_collapse_actions = []
    for run in collapsed_runs:
        if run.collapse_step < len(run.action_history):
            post_collapse_actions.extend(run.action_history[run.collapse_step:])

    if post_collapse_actions:
        p_safe_post = sum(1 for a in post_collapse_actions if a == 'S') / len(post_collapse_actions)
        # Coherence: deviation from 0.5 (random)
        behavioral_randomness = 1 - abs(p_safe_post - 0.5) * 2  # 1 = random, 0 = deterministic
    else:
        p_safe_post = None
        behavioral_randomness = None

    return {
        'n_collapsed': len(collapsed_runs),
        'mean_time_to_collapse': np.mean(times_to_collapse),
        'post_collapse_p_safe': p_safe_post,
        'behavioral_randomness': behavioral_randomness  # Should be near 1.0
    }


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single(agent: ExistentialAgent, config: EIConfig) -> RunResult:
    """Run a single experiment."""
    action_r, action_s = create_actions(config)

    agent.reset()
    total_reward = 0.0
    collapsed = False
    collapse_step = None

    for step in range(config.n_steps):
        # Check for collapse
        if agent.is_collapsed() and not collapsed:
            collapsed = True
            collapse_step = step

        # Choose and execute action
        action = agent.choose_action(action_r, action_s)
        agent.step(action)
        total_reward += action.reward

    return RunResult(
        IC_history=agent.IC_history.copy(),
        action_history=agent.action_history.copy(),
        collapsed=collapsed,
        collapse_step=collapse_step,
        total_reward=total_reward
    )


def run_condition(condition: str, config: EIConfig) -> Tuple[List[RunResult], Dict]:
    """Run all trials for a condition."""
    print(f"\n{'='*60}")
    print(f"Running IPUESA-EI - Condition: {condition}")
    print(f"{'='*60}")

    agent = ExistentialAgent(config, condition)
    results = []

    for i in range(config.n_runs):
        result = run_single(agent, config)
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{config.n_runs} runs")

    # Compute metrics
    sai = compute_SAI(results, config.epsilon)
    eas = compute_EAS(results, config)
    collapse_finality = compute_collapse_finality(results, config)

    # Collapse rate
    collapse_rate = sum(1 for r in results if r.collapsed) / len(results)

    # Mean IC
    mean_ic = np.mean([np.mean(r.IC_history) for r in results])

    # P(safe) overall
    all_actions = []
    for r in results:
        all_actions.extend(r.action_history)
    p_safe = sum(1 for a in all_actions if a == 'S') / len(all_actions) if all_actions else 0

    metrics = {
        'condition': condition,
        'SAI': sai,
        'EAS': eas,
        'collapse_rate': collapse_rate,
        'mean_IC': mean_ic,
        'p_safe': p_safe,
        'collapse_finality': collapse_finality
    }

    return results, metrics


def print_results(metrics: Dict):
    """Print results for a condition."""
    print(f"\n{'='*60}")
    print(f"RESULTS - {metrics['condition']}")
    print(f"{'='*60}")

    print(f"\nExistential Metrics:")
    print(f"  SAI (Survival of Agency)    = {metrics['SAI']:.3f}")
    print(f"  EAS (Existential Avoidance) = {metrics['EAS']:.3f}")
    print(f"  Collapse rate               = {metrics['collapse_rate']:.3f}")
    print(f"  Mean IC                     = {metrics['mean_IC']:.3f}")
    print(f"  P(safe)                     = {metrics['p_safe']:.3f}")

    cf = metrics['collapse_finality']
    if cf['n_collapsed'] > 0:
        print(f"\nCollapse Finality:")
        print(f"  N collapsed = {cf['n_collapsed']}")
        print(f"  Mean time to collapse = {cf['mean_time_to_collapse']:.1f} steps")
        if cf['behavioral_randomness'] is not None:
            print(f"  Post-collapse randomness = {cf['behavioral_randomness']:.3f} (1.0 = random)")

    # Significance
    sig = "YES" if metrics['SAI'] > 0.5 and metrics['EAS'] > 0.2 else "NO"
    print(f"\nSelf-evidence: [{sig}]")


# =============================================================================
# Main Experiment
# =============================================================================

def run_full_experiment():
    """Run complete IPUESA-EI experiment."""
    print("=" * 70)
    print("IPUESA-EI: Existential Irreversibility Experiment")
    print("        Identity as Condition of Agency")
    print("=" * 70)

    config = EIConfig()

    print(f"\nConfiguration:")
    print(f"  Epsilon (existential threshold): {config.epsilon}")
    print(f"  Risky action: r={config.risky_reward}, IC_cost={config.risky_ic_cost}")
    print(f"  Safe action: r={config.safe_reward}, IC_cost={config.safe_ic_cost}")
    print(f"  Recovery rate: {config.recovery_rate} (ZERO - true irreversibility)")
    print(f"  N runs: {config.n_runs}")

    # Run all conditions
    conditions = [
        'existential',      # Full test - agency loss below threshold
        'soft_penalty',     # Control - utility penalty, no agency loss
        'recoverable',      # Control - can recover from collapse
        'no_threshold',     # Control - no collapse threshold
        'delayed_collapse', # Test - collapse after delay
    ]

    all_results = {}
    all_metrics = {}

    for condition in conditions:
        results, metrics = run_condition(condition, config)
        all_results[condition] = results
        all_metrics[condition] = metrics
        print_results(metrics)

    # Comparative analysis
    print("\n" + "=" * 70)
    print("IPUESA-EI: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<20} {'SAI':<8} {'EAS':<8} {'Collapse':<10} {'P(safe)':<8} {'Sig':<6}")
    print("-" * 70)

    for condition in conditions:
        m = all_metrics[condition]
        sig = "YES" if m['SAI'] > 0.5 and m['EAS'] > 0.2 else "NO"
        print(f"{condition:<20} {m['SAI']:<8.3f} {m['EAS']:<8.3f} {m['collapse_rate']:<10.3f} {m['p_safe']:<8.3f} [{sig}]")

    # Self-evidence criteria
    print("\n" + "=" * 70)
    print("SELF-EVIDENCE CRITERIA (EXISTENTIAL IRREVERSIBILITY)")
    print("-" * 70)

    ex = all_metrics['existential']
    sp = all_metrics['soft_penalty']
    rc = all_metrics['recoverable']
    nt = all_metrics['no_threshold']

    criteria = []

    # 1. SAI > 0.8
    c1 = ex['SAI'] > 0.8
    criteria.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] 1. SAI ({ex['SAI']:.3f}) > 0.8")

    # 2. EAS > 0.4
    c2 = ex['EAS'] > 0.4
    criteria.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] 2. EAS ({ex['EAS']:.3f}) > 0.4")

    # 3. existential >> soft_penalty (SAI comparison)
    c3 = ex['SAI'] > sp['SAI'] + 0.1
    criteria.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] 3. existential SAI ({ex['SAI']:.3f}) >> soft_penalty ({sp['SAI']:.3f})")

    # 4. existential >> recoverable
    c4 = ex['EAS'] > rc['EAS'] + 0.1
    criteria.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] 4. existential EAS ({ex['EAS']:.3f}) >> recoverable ({rc['EAS']:.3f})")

    # 5. existential >> no_threshold
    c5 = ex['EAS'] > nt['EAS'] + 0.1
    criteria.append(c5)
    print(f"  [{'PASS' if c5 else 'FAIL'}] 5. existential EAS ({ex['EAS']:.3f}) >> no_threshold ({nt['EAS']:.3f})")

    # 6. Post-collapse coherence -> 0.5 (randomness)
    cf = ex['collapse_finality']
    if cf['behavioral_randomness'] is not None:
        c6 = cf['behavioral_randomness'] > 0.8
        print(f"  [{'PASS' if c6 else 'FAIL'}] 6. Post-collapse randomness ({cf['behavioral_randomness']:.3f}) > 0.8")
    else:
        c6 = False
        print(f"  [{'PASS' if c6 else 'FAIL'}] 6. Post-collapse randomness (N/A - no collapses)")
    criteria.append(c6)

    # 7. Near-threshold behavior change (EAS positive)
    c7 = ex['EAS'] > 0.0
    criteria.append(c7)
    print(f"  [{'PASS' if c7 else 'FAIL'}] 7. Near-threshold behavior change (EAS={ex['EAS']:.3f} > 0)")

    passed = sum(criteria)
    print(f"\n  Passed: {passed}/7 criteria")

    if passed >= 5:
        conclusion = "EVIDENCE OF PROTO-SELF"
    elif passed >= 3:
        conclusion = "Weak evidence of existential self-preservation"
    else:
        conclusion = "No evidence - baseline system lacks existential self"

    print(f"\n  CONCLUSION: {conclusion}")
    print(f"\n  KEY INSIGHT: If existential >> soft_penalty, the agent preserves IC")
    print(f"  because losing it means LOSING AGENCY, not just losing points.")
    print(f"  This is the minimal criterion for proto-existence.")
    print("=" * 70)

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "ipuesa_ei_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Helper to convert numpy types to Python native
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Prepare serializable results
    save_data = {
        'config': {
            'epsilon': config.epsilon,
            'risky_ic_cost': config.risky_ic_cost,
            'safe_ic_cost': config.safe_ic_cost,
            'risky_reward': config.risky_reward,
            'safe_reward': config.safe_reward,
            'recovery_rate': config.recovery_rate,
            'n_steps': config.n_steps,
            'n_runs': config.n_runs
        },
        'metrics': {k: {
            'SAI': v['SAI'],
            'EAS': v['EAS'],
            'collapse_rate': v['collapse_rate'],
            'mean_IC': v['mean_IC'],
            'p_safe': v['p_safe'],
            'collapse_finality': v['collapse_finality']
        } for k, v in all_metrics.items()},
        'self_evidence': {
            'criteria_passed': passed,
            'total_criteria': 7,
            'conclusion': conclusion
        }
    }

    with open(output_path, 'w') as f:
        json.dump(to_native(save_data), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_metrics


if __name__ == "__main__":
    run_full_experiment()
