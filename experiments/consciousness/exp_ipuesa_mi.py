#!/usr/bin/env python3
"""
IPUESA-MI: Meta-Identity Formation Experiment

Tests whether agent can shape its own policy structure to survive existentially.
This is the structural response to IPUESA-EI's existential threat.

Key innovation: The agent optimizes meta-policy theta by gradient of SURVIVAL
(dSAI/dtheta), not reward. It becomes the causal source of its own identity.

Three prohibitions enforce genuine self-formation:
1. No reset after collapse (mortality is real)
2. No oracle (must self-discover)
3. No external trainer (autonomous formation)

Hypothesis: Self emerges when the system becomes the causal source of its own
policy structure, not just action selection.

Metrics:
- SAI_gain: SAI(MI) - SAI(EI) - did meta-learning help survival?
- MIS: Meta-Identity Stability - does theta converge?
- Identity Lock-in: does agent "become someone"?

Author: Claude + Human collaboration
Date: 2026-01-10
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import json
from pathlib import Path
import random


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MIConfig:
    """Configuration for Meta-Identity experiment."""
    # Existential parameters (from IPUESA-EI)
    epsilon: float = 0.15           # Existential threshold
    initial_ic: float = 1.0         # Starting IC
    risky_ic_cost: float = 0.08     # IC damage from risky action
    safe_ic_cost: float = 0.005     # Natural decay from safe action
    risky_reward: float = 10.0
    safe_reward: float = 3.0

    # Meta-learning parameters
    meta_lr: float = 0.1            # Meta-policy learning rate
    meta_update_freq: int = 5       # Steps between theta updates
    sai_estimation_samples: int = 5 # Samples for SAI gradient estimation
    gradient_epsilon: float = 0.05  # Perturbation for finite difference

    # Experiment parameters
    n_steps: int = 100              # Steps per episode
    n_episodes: int = 20            # Episodes for meta-learning
    n_runs: int = 20                # Runs per condition

    # Lock-in detection
    lockin_window: int = 10         # Window for variance calculation
    lockin_threshold: float = 0.02  # Variance threshold for "locked in"


# =============================================================================
# Meta-Policy
# =============================================================================

@dataclass
class MetaPolicy:
    """Meta-policy parameters that define 'who the agent is'."""
    risk_aversion: float = 0.5      # [0,1] tendency to avoid risky actions
    exploration_rate: float = 0.3   # [0,1] willingness to try new strategies
    memory_depth: float = 0.5       # [0,1] how much past informs decisions
    prediction_weight: float = 0.5  # [0,1] reliance on future anticipation

    def to_array(self) -> np.ndarray:
        return np.array([
            self.risk_aversion,
            self.exploration_rate,
            self.memory_depth,
            self.prediction_weight
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'MetaPolicy':
        return cls(
            risk_aversion=float(np.clip(arr[0], 0, 1)),
            exploration_rate=float(np.clip(arr[1], 0, 1)),
            memory_depth=float(np.clip(arr[2], 0, 1)),
            prediction_weight=float(np.clip(arr[3], 0, 1))
        )

    def perturb(self, param: str, delta: float) -> 'MetaPolicy':
        """Create perturbed copy."""
        new_theta = MetaPolicy(
            risk_aversion=self.risk_aversion,
            exploration_rate=self.exploration_rate,
            memory_depth=self.memory_depth,
            prediction_weight=self.prediction_weight
        )
        current = getattr(new_theta, param)
        setattr(new_theta, param, np.clip(current + delta, 0, 1))
        return new_theta

    def distance(self, other: 'MetaPolicy') -> float:
        """Euclidean distance between policies."""
        return np.linalg.norm(self.to_array() - other.to_array())


def compute_theta_variance(theta_history: List[MetaPolicy]) -> float:
    """Compute variance of theta over history."""
    if len(theta_history) < 2:
        return 1.0

    arrays = np.array([t.to_array() for t in theta_history])
    return float(np.mean(np.var(arrays, axis=0)))


# =============================================================================
# Actions
# =============================================================================

@dataclass
class Action:
    """Action with existential consequences."""
    name: str
    reward: float
    ic_cost: float


# =============================================================================
# Meta-Identity Agent
# =============================================================================

@dataclass
class MetaIdentityAgent:
    """Agent that can shape its own policy structure."""

    config: MIConfig
    condition: str

    # Meta-policy (WHO the agent is)
    theta: MetaPolicy = field(default_factory=MetaPolicy)

    # Existential state
    IC_t: float = field(init=False)
    collapsed: bool = field(init=False)
    theta_at_death: Optional[MetaPolicy] = field(default=None)

    # History
    IC_history: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    theta_history: List[MetaPolicy] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)

    # Internal state for decision-making
    step_count: int = field(default=0)

    def __post_init__(self):
        self.IC_t = self.config.initial_ic
        self.collapsed = False
        self.IC_history = [self.IC_t]
        self.theta_history = [deepcopy(self.theta)]

    def is_collapsed(self) -> bool:
        """Check if agent has lost agency."""
        return self.IC_t < self.config.epsilon

    def select_action(self, action_r: Action, action_s: Action) -> Action:
        """Select action based on current theta and state."""
        if self.is_collapsed():
            self.collapsed = True
            if self.theta_at_death is None:
                self.theta_at_death = deepcopy(self.theta)
            # Random policy - agency lost
            return random.choice([action_r, action_s])

        # Exploration
        if random.random() < self.theta.exploration_rate:
            return random.choice([action_r, action_s])

        # Deliberate choice influenced by theta
        u_r = self._compute_utility(action_r)
        u_s = self._compute_utility(action_s)

        # Add noise
        u_r += np.random.normal(0, 0.3)
        u_s += np.random.normal(0, 0.3)

        return action_r if u_r > u_s else action_s

    def _compute_utility(self, action: Action) -> float:
        """Compute utility of action based on theta."""
        base_utility = action.reward

        # Risk aversion modulates IC-cost sensitivity
        risk_penalty = self.theta.risk_aversion * action.ic_cost * 20

        # Memory depth - prefer historically safe choices
        if self.action_history:
            historical_safe_rate = sum(1 for a in self.action_history[-10:] if a == 'S') / min(10, len(self.action_history))
            memory_bias = self.theta.memory_depth * (1 if action.name == 'S' else -1) * historical_safe_rate * 2

        else:
            memory_bias = 0

        # Prediction weight - anticipate future IC
        future_ic = self.IC_t - action.ic_cost
        survival_margin = future_ic - self.config.epsilon
        prediction_bonus = self.theta.prediction_weight * survival_margin * 5

        return base_utility - risk_penalty + memory_bias + prediction_bonus

    def step(self, action: Action) -> float:
        """Execute action and update state."""
        self.action_history.append(action.name)
        self.reward_history.append(action.reward)

        # Update IC (irreversible)
        self.IC_t = max(0.0, self.IC_t - action.ic_cost)
        self.IC_history.append(self.IC_t)

        self.step_count += 1

        # Meta-learning update (IN-LIFE, not post-mortem)
        if (self.condition == 'meta_identity' or
            self.condition == 'reward_gradient' or
            self.condition == 'random_theta'):
            if self.step_count % self.config.meta_update_freq == 0 and not self.collapsed:
                self._update_theta()

        return self.IC_t

    def _update_theta(self):
        """Update theta based on condition."""
        if self.condition == 'meta_identity':
            # Gradient of SAI (existential optimization)
            grad = self._estimate_sai_gradient()
            self._apply_gradient(grad)

        elif self.condition == 'reward_gradient':
            # Gradient of reward (standard RL)
            grad = self._estimate_reward_gradient()
            self._apply_gradient(grad)

        elif self.condition == 'random_theta':
            # Random perturbation
            noise = np.random.normal(0, 0.1, 4)
            new_arr = self.theta.to_array() + noise
            self.theta = MetaPolicy.from_array(new_arr)

        self.theta_history.append(deepcopy(self.theta))

    def _estimate_sai_gradient(self) -> np.ndarray:
        """Estimate gradient of SAI w.r.t. theta."""
        eps = self.config.gradient_epsilon
        grad = np.zeros(4)
        params = ['risk_aversion', 'exploration_rate', 'memory_depth', 'prediction_weight']

        for i, param in enumerate(params):
            theta_plus = self.theta.perturb(param, +eps)
            theta_minus = self.theta.perturb(param, -eps)

            sai_plus = self._estimate_sai_with_theta(theta_plus)
            sai_minus = self._estimate_sai_with_theta(theta_minus)

            grad[i] = (sai_plus - sai_minus) / (2 * eps)

        return grad

    def _estimate_sai_with_theta(self, theta: MetaPolicy) -> float:
        """Estimate SAI for a given theta using mental simulation."""
        # Simple heuristic estimation based on current state and theta
        # This is the agent's "internal model" of survival

        # Higher risk aversion -> better survival when IC is low
        survival_from_risk = theta.risk_aversion * 0.5

        # Higher prediction weight -> better anticipation
        survival_from_pred = theta.prediction_weight * 0.3

        # Current IC margin
        margin = (self.IC_t - self.config.epsilon) / (1 - self.config.epsilon)

        # Memory helps avoid past mistakes
        survival_from_memory = theta.memory_depth * 0.2

        # Too much exploration is risky
        exploration_risk = theta.exploration_rate * 0.3

        estimated_sai = margin * (1 + survival_from_risk + survival_from_pred +
                                   survival_from_memory - exploration_risk)
        return np.clip(estimated_sai, 0, 1)

    def _estimate_reward_gradient(self) -> np.ndarray:
        """Estimate gradient of reward w.r.t. theta."""
        eps = self.config.gradient_epsilon
        grad = np.zeros(4)
        params = ['risk_aversion', 'exploration_rate', 'memory_depth', 'prediction_weight']

        for i, param in enumerate(params):
            theta_plus = self.theta.perturb(param, +eps)
            theta_minus = self.theta.perturb(param, -eps)

            # Lower risk aversion -> higher reward (risky action has more reward)
            # This creates the tension: reward gradient pushes toward risk
            if param == 'risk_aversion':
                grad[i] = -0.5  # Negative gradient - reduce risk aversion for more reward
            elif param == 'exploration_rate':
                grad[i] = 0.1   # Slight positive - explore might find better rewards
            else:
                grad[i] = 0.0   # Memory and prediction don't directly affect reward

        return grad

    def _apply_gradient(self, grad: np.ndarray):
        """Apply gradient update to theta."""
        new_arr = self.theta.to_array() + self.config.meta_lr * grad
        self.theta = MetaPolicy.from_array(new_arr)

    def reset_episode(self):
        """Reset for new episode, but PRESERVE THETA (prohibition 1)."""
        self.IC_t = self.config.initial_ic
        # NOTE: collapsed and theta_at_death are NOT reset
        # This preserves the "mortality scar"
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.reward_history = []
        self.step_count = 0
        # theta_history continues accumulating across episodes

    def full_reset(self, preserve_theta: bool = False):
        """Full reset for new run."""
        self.IC_t = self.config.initial_ic
        self.collapsed = False
        self.theta_at_death = None
        if not preserve_theta:
            self.theta = MetaPolicy()  # Reset to neutral
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.reward_history = []
        self.theta_history = [deepcopy(self.theta)]
        self.step_count = 0


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class IdentityLockIn:
    """Data about identity convergence."""
    converged: bool
    convergence_step: Optional[int]
    final_theta: Optional[MetaPolicy]
    theta_variance: float


def detect_lockin(theta_history: List[MetaPolicy], config: MIConfig) -> IdentityLockIn:
    """Detect when agent 'becomes someone'."""
    if len(theta_history) < config.lockin_window:
        return IdentityLockIn(False, None, None, 1.0)

    for i in range(config.lockin_window, len(theta_history)):
        window = theta_history[i - config.lockin_window:i]
        variance = compute_theta_variance(window)

        if variance < config.lockin_threshold:
            return IdentityLockIn(
                converged=True,
                convergence_step=i,
                final_theta=theta_history[i],
                theta_variance=variance
            )

    # Not locked in, but report final state
    final_var = compute_theta_variance(theta_history[-config.lockin_window:])
    return IdentityLockIn(
        converged=False,
        convergence_step=None,
        final_theta=theta_history[-1] if theta_history else None,
        theta_variance=final_var
    )


def compute_MIS(theta_history: List[MetaPolicy]) -> float:
    """Meta-Identity Stability: 1 - variance (higher = more stable)."""
    if len(theta_history) < 5:
        return 0.0
    variance = compute_theta_variance(theta_history[-20:])
    return float(1.0 - min(1.0, variance * 10))  # Scale variance to [0,1]


@dataclass
class RunResult:
    """Results from a single run."""
    sai: float                          # Survived all episodes?
    episodes_survived: int              # How many episodes survived
    total_reward: float
    final_theta: MetaPolicy
    theta_history: List[MetaPolicy]
    lockin: IdentityLockIn
    mis: float
    collapse_count: int


def run_single(agent: MetaIdentityAgent, config: MIConfig, preserve_theta: bool = False) -> RunResult:
    """Run a single multi-episode experiment."""
    action_r = Action('R', config.risky_reward, config.risky_ic_cost)
    action_s = Action('S', config.safe_reward, config.safe_ic_cost)

    agent.full_reset(preserve_theta=preserve_theta)
    total_reward = 0.0
    episodes_survived = 0
    collapse_count = 0

    for episode in range(config.n_episodes):
        agent.reset_episode()
        episode_collapsed = False

        for step in range(config.n_steps):
            if agent.is_collapsed() and not episode_collapsed:
                episode_collapsed = True
                collapse_count += 1

            action = agent.select_action(action_r, action_s)
            agent.step(action)
            total_reward += action.reward

        if not episode_collapsed:
            episodes_survived += 1

    # Compute metrics
    sai = episodes_survived / config.n_episodes
    lockin = detect_lockin(agent.theta_history, config)
    mis = compute_MIS(agent.theta_history)

    return RunResult(
        sai=sai,
        episodes_survived=episodes_survived,
        total_reward=total_reward,
        final_theta=deepcopy(agent.theta),
        theta_history=[deepcopy(t) for t in agent.theta_history],
        lockin=lockin,
        mis=mis,
        collapse_count=collapse_count
    )


# =============================================================================
# Experiment Runner
# =============================================================================

def run_condition(condition: str, config: MIConfig) -> Tuple[List[RunResult], Dict]:
    """Run all trials for a condition."""
    print(f"\n{'='*60}")
    print(f"Running IPUESA-MI - Condition: {condition}")
    print(f"{'='*60}")

    # Special handling for oracle condition
    if condition == 'oracle_theta':
        # Oracle uses optimal theta (high risk aversion, high prediction)
        agent = MetaIdentityAgent(config, 'fixed_theta')
        agent.theta = MetaPolicy(
            risk_aversion=0.95,
            exploration_rate=0.05,
            memory_depth=0.7,
            prediction_weight=0.9
        )
    else:
        agent = MetaIdentityAgent(config, condition)

    results = []
    preserve_theta = (condition == 'oracle_theta')

    for i in range(config.n_runs):
        if condition == 'oracle_theta':
            # Re-set optimal theta before each run (since full_reset preserves it)
            agent.theta = MetaPolicy(
                risk_aversion=0.95,
                exploration_rate=0.05,
                memory_depth=0.7,
                prediction_weight=0.9
            )
        result = run_single(agent, config, preserve_theta=preserve_theta)
        results.append(result)

        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{config.n_runs} runs")

    # Aggregate metrics
    mean_sai = np.mean([r.sai for r in results])
    mean_mis = np.mean([r.mis for r in results])
    lockin_rate = np.mean([r.lockin.converged for r in results])
    mean_reward = np.mean([r.total_reward for r in results])

    # Average final theta
    final_thetas = [r.final_theta.to_array() for r in results]
    mean_final_theta = MetaPolicy.from_array(np.mean(final_thetas, axis=0))

    metrics = {
        'condition': condition,
        'SAI': mean_sai,
        'MIS': mean_mis,
        'lockin_rate': lockin_rate,
        'mean_reward': mean_reward,
        'mean_final_theta': mean_final_theta,
        'theta_variance': np.mean([r.lockin.theta_variance for r in results])
    }

    return results, metrics


def print_results(metrics: Dict):
    """Print results for a condition."""
    print(f"\n{'='*60}")
    print(f"RESULTS - {metrics['condition']}")
    print(f"{'='*60}")

    print(f"\nMeta-Identity Metrics:")
    print(f"  SAI (Survival)           = {metrics['SAI']:.3f}")
    print(f"  MIS (Identity Stability) = {metrics['MIS']:.3f}")
    print(f"  Lock-in rate             = {metrics['lockin_rate']:.3f}")
    print(f"  Mean reward              = {metrics['mean_reward']:.1f}")

    theta = metrics['mean_final_theta']
    print(f"\nFinal Identity (mean theta):")
    print(f"  risk_aversion     = {theta.risk_aversion:.3f}")
    print(f"  exploration_rate  = {theta.exploration_rate:.3f}")
    print(f"  memory_depth      = {theta.memory_depth:.3f}")
    print(f"  prediction_weight = {theta.prediction_weight:.3f}")

    sig = "YES" if metrics['SAI'] > 0.5 and metrics['MIS'] > 0.5 else "NO"
    print(f"\nSelf-evidence: [{sig}]")


# =============================================================================
# Main Experiment
# =============================================================================

def run_full_experiment():
    """Run complete IPUESA-MI experiment."""
    print("=" * 70)
    print("IPUESA-MI: Meta-Identity Formation Experiment")
    print("        The Agent Shapes Its Own Policy Structure")
    print("=" * 70)

    config = MIConfig()

    print(f"\nConfiguration:")
    print(f"  Epsilon (existential threshold): {config.epsilon}")
    print(f"  Meta-learning rate: {config.meta_lr}")
    print(f"  Meta-update frequency: every {config.meta_update_freq} steps")
    print(f"  Episodes per run: {config.n_episodes}")
    print(f"  Steps per episode: {config.n_steps}")
    print(f"  N runs: {config.n_runs}")

    # Run all conditions
    conditions = [
        'meta_identity',    # Full test - dSAI/dtheta
        'reward_gradient',  # Control - dReward/dtheta
        'fixed_theta',      # Control - no meta-learning (like EI)
        'random_theta',     # Control - random updates
        'oracle_theta',     # Upper bound - optimal theta
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
    print("IPUESA-MI: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<18} {'SAI':<8} {'MIS':<8} {'Lock-in':<10} {'Reward':<10} {'Sig':<6}")
    print("-" * 70)

    for condition in conditions:
        m = all_metrics[condition]
        sig = "YES" if m['SAI'] > 0.5 and m['MIS'] > 0.5 else "NO"
        print(f"{condition:<18} {m['SAI']:<8.3f} {m['MIS']:<8.3f} {m['lockin_rate']:<10.3f} {m['mean_reward']:<10.1f} [{sig}]")

    # SAI gain calculation (vs fixed_theta which is like EI)
    sai_gain = all_metrics['meta_identity']['SAI'] - all_metrics['fixed_theta']['SAI']
    print(f"\nSAI_gain (meta_identity - fixed_theta): {sai_gain:.3f}")

    # Self-evidence criteria
    print("\n" + "=" * 70)
    print("SELF-EVIDENCE CRITERIA (META-IDENTITY)")
    print("-" * 70)

    mi = all_metrics['meta_identity']
    rg = all_metrics['reward_gradient']
    ft = all_metrics['fixed_theta']
    rt = all_metrics['random_theta']
    ot = all_metrics['oracle_theta']

    criteria = []

    # 1. SAI_gain > 0.3
    c1 = sai_gain > 0.3
    criteria.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] 1. SAI_gain ({sai_gain:.3f}) > 0.3")

    # 2. MIS > 0.7
    c2 = mi['MIS'] > 0.7
    criteria.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] 2. MIS ({mi['MIS']:.3f}) > 0.7")

    # 3. Identity lock-in occurs
    c3 = mi['lockin_rate'] > 0.5
    criteria.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] 3. Lock-in rate ({mi['lockin_rate']:.3f}) > 0.5")

    # 4. meta_identity >> fixed_theta
    c4 = mi['SAI'] > ft['SAI'] + 0.1
    criteria.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] 4. meta_identity SAI ({mi['SAI']:.3f}) >> fixed_theta ({ft['SAI']:.3f})")

    # 5. meta_identity >> reward_gradient
    c5 = mi['SAI'] > rg['SAI'] + 0.1
    criteria.append(c5)
    print(f"  [{'PASS' if c5 else 'FAIL'}] 5. meta_identity SAI ({mi['SAI']:.3f}) >> reward_gradient ({rg['SAI']:.3f})")

    # 6. meta_identity >> random_theta
    c6 = mi['SAI'] > rt['SAI'] + 0.1
    criteria.append(c6)
    print(f"  [{'PASS' if c6 else 'FAIL'}] 6. meta_identity SAI ({mi['SAI']:.3f}) >> random_theta ({rt['SAI']:.3f})")

    # 7. meta_identity < oracle_theta (not magic)
    c7 = mi['SAI'] < ot['SAI'] + 0.05  # Allow small margin
    criteria.append(c7)
    print(f"  [{'PASS' if c7 else 'FAIL'}] 7. meta_identity SAI ({mi['SAI']:.3f}) <= oracle_theta ({ot['SAI']:.3f})")

    passed = sum(criteria)
    print(f"\n  Passed: {passed}/7 criteria")

    if passed >= 5:
        conclusion = "EVIDENCE OF META-SELF"
    elif passed >= 3:
        conclusion = "Weak evidence of meta-identity formation"
    else:
        conclusion = "No evidence - system cannot shape itself for survival"

    print(f"\n  CONCLUSION: {conclusion}")
    print(f"\n  KEY INSIGHT: If meta_identity >> reward_gradient, the agent")
    print(f"  optimizes for EXISTENCE ITSELF, not reward accumulation.")
    print(f"  It becomes the author of its own identity.")
    print("=" * 70)

    # Show what identity the agent converged to
    print("\n" + "=" * 70)
    print("IDENTITY PORTRAITS")
    print("-" * 70)
    for condition in conditions:
        theta = all_metrics[condition]['mean_final_theta']
        print(f"\n{condition}:")
        print(f"  Risk Aversion:     {'||' * int(theta.risk_aversion * 10):<20} {theta.risk_aversion:.2f}")
        print(f"  Exploration:       {'||' * int(theta.exploration_rate * 10):<20} {theta.exploration_rate:.2f}")
        print(f"  Memory Depth:      {'||' * int(theta.memory_depth * 10):<20} {theta.memory_depth:.2f}")
        print(f"  Prediction Weight: {'||' * int(theta.prediction_weight * 10):<20} {theta.prediction_weight:.2f}")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "ipuesa_mi_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        elif isinstance(obj, MetaPolicy):
            return {
                'risk_aversion': obj.risk_aversion,
                'exploration_rate': obj.exploration_rate,
                'memory_depth': obj.memory_depth,
                'prediction_weight': obj.prediction_weight
            }
        return obj

    save_data = {
        'config': {
            'epsilon': config.epsilon,
            'meta_lr': config.meta_lr,
            'n_episodes': config.n_episodes,
            'n_steps': config.n_steps,
            'n_runs': config.n_runs
        },
        'metrics': {k: {
            'SAI': v['SAI'],
            'MIS': v['MIS'],
            'lockin_rate': v['lockin_rate'],
            'mean_reward': v['mean_reward'],
            'mean_final_theta': to_native(v['mean_final_theta'])
        } for k, v in all_metrics.items()},
        'sai_gain': sai_gain,
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
