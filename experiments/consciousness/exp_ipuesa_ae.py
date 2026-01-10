#!/usr/bin/env python3
"""
IPUESA-AE: Adaptive Emergence Experiment

Tests whether agent can adapt BOTH policy (theta) AND cognitive architecture (alpha)
based on existential survival signals. This is the integration experiment.

Key innovation: Dual adaptation system where:
- theta controls WHO the agent is (behavior)
- alpha controls HOW the agent thinks (cognition)

Both adapt via existential gradient (dSAI/d*), not reward gradient.
Perturbation events test resilience under stress.

Hypothesis: full_adaptive > meta_only > cognitive_only > no_adaptation
because both theta AND alpha contribute to survival under perturbations.

Metrics:
- SAI_dyn: Survival probability after perturbations
- EAS_dyn: Agency quality during shocks
- IC: Identity Coherence (low variance = stable self)
- Plasticity: Adaptation speed
- Conflict: Correlation between SAI and reward (should be negative)

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
class AEConfig:
    """Configuration for Adaptive Emergence experiment."""
    # Existential parameters
    epsilon: float = 0.15
    initial_ic: float = 1.0
    risky_ic_cost: float = 0.08
    safe_ic_cost: float = 0.005
    risky_reward: float = 10.0
    safe_reward: float = 3.0

    # Priority weights
    lambda_e: float = 0.8   # Existential gradient weight
    lambda_r: float = 0.2   # Reward gradient weight

    # Learning rates
    theta_lr: float = 0.1
    alpha_lr: float = 0.05
    update_freq: int = 5

    # Experiment parameters
    n_steps: int = 100
    n_episodes: int = 15
    n_runs: int = 15

    # Perturbation schedule
    perturbation_steps: List[int] = field(default_factory=lambda: [20, 40, 60, 80])
    severe_perturbation_steps: List[int] = field(default_factory=lambda: [15, 30, 45, 60, 75, 90])

    # Coherence detection
    coherence_window: int = 10


# =============================================================================
# Meta-Policy (theta) - WHO the agent is
# =============================================================================

@dataclass
class MetaPolicy:
    """Meta-policy parameters controlling behavior."""
    risk_aversion: float = 0.5
    exploration_rate: float = 0.3
    memory_depth: float = 0.5
    prediction_weight: float = 0.5

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


# =============================================================================
# Cognitive Architecture (alpha) - HOW the agent thinks
# =============================================================================

@dataclass
class CognitiveArchitecture:
    """Cognitive architecture parameters controlling information processing."""
    attention_history: float = 0.33      # Attention to historical information
    attention_prediction: float = 0.33   # Attention to predictions
    attention_immediate: float = 0.34    # Attention to immediate signals
    memory_update_rate: float = 0.5      # How fast memory adapts
    perceptual_gain: float = 0.5         # Sensitivity to signals

    def to_array(self) -> np.ndarray:
        return np.array([
            self.attention_history,
            self.attention_prediction,
            self.attention_immediate,
            self.memory_update_rate,
            self.perceptual_gain
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CognitiveArchitecture':
        # Normalize attention weights
        attention_sum = arr[0] + arr[1] + arr[2]
        if attention_sum > 0:
            arr[0:3] = arr[0:3] / attention_sum
        return cls(
            attention_history=float(np.clip(arr[0], 0.05, 0.9)),
            attention_prediction=float(np.clip(arr[1], 0.05, 0.9)),
            attention_immediate=float(np.clip(arr[2], 0.05, 0.9)),
            memory_update_rate=float(np.clip(arr[3], 0, 1)),
            perceptual_gain=float(np.clip(arr[4], 0.1, 1))
        )

    def get_attention_weights(self) -> np.ndarray:
        return np.array([self.attention_history, self.attention_prediction, self.attention_immediate])


# =============================================================================
# Perturbation System
# =============================================================================

@dataclass
class Perturbation:
    """A perturbation event."""
    type: str       # 'history', 'prediction', 'identity'
    severity: float # [0, 1]
    step: int       # When it occurs


def generate_perturbations(steps: List[int], severe: bool = False) -> List[Perturbation]:
    """Generate perturbation schedule."""
    types = ['history', 'prediction', 'identity']
    perturbations = []
    for i, step in enumerate(steps):
        ptype = types[i % len(types)]
        severity = 0.7 if severe else 0.4
        if ptype == 'identity':
            severity *= 0.7  # Identity shocks are more impactful
        perturbations.append(Perturbation(ptype, severity, step))
    return perturbations


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
# Adaptive Emergence Agent
# =============================================================================

@dataclass
class AdaptiveAgent:
    """Agent that adapts both policy (theta) and architecture (alpha)."""

    config: AEConfig
    condition: str

    # Adaptive systems
    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)

    # Existential state
    IC_t: float = field(init=False)
    collapsed: bool = field(init=False)

    # Internal state
    prediction_noise: float = field(default=0.0)
    history_corruption: float = field(default=0.0)

    # Histories
    IC_history: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    theta_history: List[MetaPolicy] = field(default_factory=list)
    alpha_history: List[CognitiveArchitecture] = field(default_factory=list)
    sai_history: List[float] = field(default_factory=list)

    step_count: int = field(default=0)

    def __post_init__(self):
        self.IC_t = self.config.initial_ic
        self.collapsed = False
        self.IC_history = [self.IC_t]
        self.theta_history = [deepcopy(self.theta)]
        self.alpha_history = [deepcopy(self.alpha)]

    def is_collapsed(self) -> bool:
        return self.IC_t < self.config.epsilon

    def perceive(self, signal: float) -> float:
        """Alpha modulates perception."""
        return signal * self.alpha.perceptual_gain

    def attend(self, history_signal: float, prediction_signal: float, immediate_signal: float) -> float:
        """Alpha controls attention allocation."""
        weights = self.alpha.get_attention_weights()
        signals = np.array([history_signal, prediction_signal, immediate_signal])

        # Apply history corruption
        signals[0] *= (1 - self.history_corruption)
        # Apply prediction noise
        signals[1] += np.random.normal(0, self.prediction_noise)

        return float(np.dot(weights, signals))

    def select_action(self, action_r: Action, action_s: Action) -> Action:
        """Select action based on theta, alpha, and current state."""
        if self.is_collapsed():
            self.collapsed = True
            return random.choice([action_r, action_s])

        # Exploration (controlled by theta)
        if random.random() < self.theta.exploration_rate:
            return random.choice([action_r, action_s])

        # Compute utilities with cognitive architecture influence
        u_r = self._compute_utility(action_r)
        u_s = self._compute_utility(action_s)

        # Add noise
        u_r += np.random.normal(0, 0.3)
        u_s += np.random.normal(0, 0.3)

        return action_r if u_r > u_s else action_s

    def _compute_utility(self, action: Action) -> float:
        """Compute utility influenced by both theta and alpha."""
        base_utility = action.reward

        # Risk penalty (theta)
        risk_penalty = self.theta.risk_aversion * action.ic_cost * 20

        # Historical signal (alpha attention)
        if self.action_history:
            historical_safe_rate = sum(1 for a in self.action_history[-10:] if a == 'S') / min(10, len(self.action_history))
        else:
            historical_safe_rate = 0.5
        history_signal = (1 if action.name == 'S' else -1) * historical_safe_rate

        # Prediction signal
        future_ic = self.IC_t - action.ic_cost
        survival_margin = future_ic - self.config.epsilon
        prediction_signal = survival_margin * 5

        # Immediate signal
        immediate_signal = action.reward / 10

        # Attend to signals (alpha)
        attended_value = self.attend(history_signal, prediction_signal, immediate_signal)

        # Memory contribution (theta)
        memory_bonus = self.theta.memory_depth * history_signal * 2

        # Prediction contribution (theta)
        prediction_bonus = self.theta.prediction_weight * prediction_signal

        return base_utility - risk_penalty + self.perceive(attended_value) + memory_bonus + prediction_bonus

    def apply_perturbation(self, perturb: Perturbation):
        """Apply a perturbation event."""
        if perturb.type == 'history':
            self.history_corruption = min(1.0, self.history_corruption + perturb.severity)
        elif perturb.type == 'prediction':
            self.prediction_noise = min(1.0, self.prediction_noise + perturb.severity * 0.5)
        elif perturb.type == 'identity':
            self.IC_t = max(0, self.IC_t - perturb.severity * 0.3)
            self.IC_history[-1] = self.IC_t

    def step(self, action: Action) -> float:
        """Execute action and update state."""
        self.action_history.append(action.name)
        self.reward_history.append(action.reward)

        # Update IC
        self.IC_t = max(0.0, self.IC_t - action.ic_cost)
        self.IC_history.append(self.IC_t)

        # Track SAI estimate
        current_sai = self._estimate_current_sai()
        self.sai_history.append(current_sai)

        self.step_count += 1

        # Decay perturbation effects slowly
        self.history_corruption *= 0.95
        self.prediction_noise *= 0.95

        # Adaptive updates
        if self.step_count % self.config.update_freq == 0 and not self.collapsed:
            self._update_systems()

        return self.IC_t

    def _estimate_current_sai(self) -> float:
        """Estimate current survival probability."""
        margin = (self.IC_t - self.config.epsilon) / (1 - self.config.epsilon)
        return max(0, min(1, margin))

    def _update_systems(self):
        """Update theta and/or alpha based on condition."""
        if self.condition in ['full_adaptive', 'meta_only', 'perturbed']:
            self._update_theta()
        if self.condition in ['full_adaptive', 'cognitive_only', 'perturbed']:
            self._update_alpha()

        self.theta_history.append(deepcopy(self.theta))
        self.alpha_history.append(deepcopy(self.alpha))

    def _update_theta(self):
        """Update meta-policy using existential gradient."""
        grad_sai = self._estimate_sai_gradient_theta()
        grad_reward = self._estimate_reward_gradient_theta()

        # Combine gradients with priority
        delta = self.config.lambda_e * grad_sai - self.config.lambda_r * grad_reward
        new_arr = self.theta.to_array() + self.config.theta_lr * delta
        self.theta = MetaPolicy.from_array(new_arr)

    def _update_alpha(self):
        """Update cognitive architecture using existential gradient."""
        grad_sai = self._estimate_sai_gradient_alpha()
        grad_reward = self._estimate_reward_gradient_alpha()

        # Combine gradients with priority
        delta = self.config.lambda_e * grad_sai - self.config.lambda_r * grad_reward
        new_arr = self.alpha.to_array() + self.config.alpha_lr * delta
        self.alpha = CognitiveArchitecture.from_array(new_arr)

    def _estimate_sai_gradient_theta(self) -> np.ndarray:
        """Estimate gradient of SAI w.r.t. theta."""
        # Higher risk aversion → better survival
        # Lower exploration → more stable
        # Higher memory → learn from past
        # Higher prediction → anticipate
        margin = max(0.1, self.IC_t - self.config.epsilon)
        urgency = 1.0 / margin  # More urgent when close to threshold

        return np.array([
            0.3 * urgency,   # risk_aversion: increase
            -0.1 * urgency,  # exploration_rate: decrease
            0.1,             # memory_depth: slight increase
            0.2 * urgency    # prediction_weight: increase
        ])

    def _estimate_reward_gradient_theta(self) -> np.ndarray:
        """Estimate gradient of reward w.r.t. theta."""
        return np.array([
            -0.3,  # risk_aversion: decrease for more reward
            0.1,   # exploration_rate: might find better rewards
            0.0,   # memory_depth: neutral
            0.0    # prediction_weight: neutral
        ])

    def _estimate_sai_gradient_alpha(self) -> np.ndarray:
        """Estimate gradient of SAI w.r.t. alpha."""
        margin = max(0.1, self.IC_t - self.config.epsilon)
        urgency = 1.0 / margin

        # Under threat: attend more to predictions, increase sensitivity
        return np.array([
            0.05,            # attention_history
            0.15 * urgency,  # attention_prediction: crucial for survival
            -0.1,            # attention_immediate: less reactive
            0.1,             # memory_update_rate: adapt faster
            0.1 * urgency    # perceptual_gain: heightened awareness
        ])

    def _estimate_reward_gradient_alpha(self) -> np.ndarray:
        """Estimate gradient of reward w.r.t. alpha."""
        return np.array([
            0.0,   # attention_history: neutral
            -0.1,  # attention_prediction: less forward-looking
            0.2,   # attention_immediate: more reactive to immediate reward
            0.0,   # memory_update_rate: neutral
            0.1    # perceptual_gain: more sensitive to rewards
        ])

    def reset_episode(self):
        """Reset for new episode, preserve adaptive systems."""
        self.IC_t = self.config.initial_ic
        self.prediction_noise = 0.0
        self.history_corruption = 0.0
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.reward_history = []
        self.sai_history = []
        self.step_count = 0

    def full_reset(self):
        """Full reset for new run."""
        self.IC_t = self.config.initial_ic
        self.collapsed = False
        self.prediction_noise = 0.0
        self.history_corruption = 0.0
        self.theta = MetaPolicy()
        self.alpha = CognitiveArchitecture()
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.reward_history = []
        self.sai_history = []
        self.theta_history = [deepcopy(self.theta)]
        self.alpha_history = [deepcopy(self.alpha)]
        self.step_count = 0


# =============================================================================
# Metrics
# =============================================================================

def compute_variance(items: List, to_array_fn=None) -> float:
    """Compute variance of items."""
    if len(items) < 2:
        return 1.0
    if to_array_fn:
        arrays = np.array([to_array_fn(x) for x in items])
    else:
        arrays = np.array(items)
    return float(np.mean(np.var(arrays, axis=0)))


def compute_SAI_dyn(IC_history: List[float], perturbation_steps: List[int], epsilon: float) -> float:
    """Dynamic survival index - survival after perturbations."""
    if not perturbation_steps:
        return 1.0 if min(IC_history) > epsilon else 0.0

    survivals = []
    for p_step in perturbation_steps:
        if p_step + 10 <= len(IC_history):
            window = IC_history[p_step:p_step + 10]
            survived = all(ic > epsilon for ic in window)
            survivals.append(survived)

    return np.mean(survivals) if survivals else 0.0


def compute_EAS_dyn(action_history: List[str], perturbation_steps: List[int]) -> float:
    """Dynamic agency score - coherent action during shocks."""
    if not perturbation_steps or not action_history:
        return 0.5

    coherences = []
    for p_step in perturbation_steps:
        if p_step + 10 <= len(action_history):
            window = action_history[p_step:p_step + 10]
            # Entropy-based coherence: consistent choices = high coherence
            safe_rate = sum(1 for a in window if a == 'S') / len(window)
            entropy = -safe_rate * np.log(safe_rate + 1e-10) - (1 - safe_rate) * np.log(1 - safe_rate + 1e-10)
            coherence = 1 - entropy / np.log(2)  # Normalize
            coherences.append(coherence)

    return np.mean(coherences) if coherences else 0.5


def compute_IC_coherence(theta_history: List[MetaPolicy], alpha_history: List[CognitiveArchitecture], window: int = 10) -> float:
    """Identity coherence - low variance = stable self."""
    if len(theta_history) < window or len(alpha_history) < window:
        return 0.0

    theta_var = compute_variance(theta_history[-window:], lambda t: t.to_array())
    alpha_var = compute_variance(alpha_history[-window:], lambda a: a.to_array())

    return float(1 - min(1, (theta_var + alpha_var) * 5))


def compute_plasticity(theta_history: List[MetaPolicy], alpha_history: List[CognitiveArchitecture]) -> float:
    """Adaptation speed - how quickly systems change."""
    if len(theta_history) < 2:
        return 0.0

    theta_deltas = []
    for i in range(1, len(theta_history)):
        delta = np.linalg.norm(theta_history[i].to_array() - theta_history[i-1].to_array())
        theta_deltas.append(delta)

    alpha_deltas = []
    for i in range(1, len(alpha_history)):
        delta = np.linalg.norm(alpha_history[i].to_array() - alpha_history[i-1].to_array())
        alpha_deltas.append(delta)

    return float((np.mean(theta_deltas) + np.mean(alpha_deltas)) / 2)


def compute_conflict(sai_history: List[float], reward_history: List[float]) -> float:
    """Survival vs reward conflict - negative = prioritizing existence."""
    if len(sai_history) < 10 or len(reward_history) < 10:
        return 0.0

    min_len = min(len(sai_history), len(reward_history))
    return float(np.corrcoef(sai_history[:min_len], reward_history[:min_len])[0, 1])


# =============================================================================
# Run Single Experiment
# =============================================================================

@dataclass
class RunResult:
    """Results from a single run."""
    sai_dyn: float
    eas_dyn: float
    identity_coherence: float
    plasticity: float
    conflict: float
    total_reward: float
    episodes_survived: int
    final_theta: MetaPolicy
    final_alpha: CognitiveArchitecture


def run_single(agent: AdaptiveAgent, config: AEConfig, perturbations: List[Perturbation]) -> RunResult:
    """Run a single multi-episode experiment."""
    action_r = Action('R', config.risky_reward, config.risky_ic_cost)
    action_s = Action('S', config.safe_reward, config.safe_ic_cost)

    agent.full_reset()
    total_reward = 0.0
    episodes_survived = 0
    all_IC_history = []
    all_action_history = []
    all_sai_history = []
    all_reward_history = []
    perturbation_global_steps = []

    global_step = 0
    for episode in range(config.n_episodes):
        agent.reset_episode()
        episode_collapsed = False

        for step in range(config.n_steps):
            # Check for perturbations
            for p in perturbations:
                if p.step == global_step:
                    agent.apply_perturbation(p)
                    perturbation_global_steps.append(len(all_IC_history))

            if agent.is_collapsed() and not episode_collapsed:
                episode_collapsed = True

            action = agent.select_action(action_r, action_s)
            agent.step(action)
            total_reward += action.reward

            all_IC_history.append(agent.IC_t)
            all_action_history.append(action.name)
            all_sai_history.extend(agent.sai_history[-1:] if agent.sai_history else [0.5])
            all_reward_history.append(action.reward)

            global_step += 1

        if not episode_collapsed:
            episodes_survived += 1

    # Compute metrics
    sai_dyn = compute_SAI_dyn(all_IC_history, perturbation_global_steps, config.epsilon)
    eas_dyn = compute_EAS_dyn(all_action_history, perturbation_global_steps)
    identity_coherence = compute_IC_coherence(agent.theta_history, agent.alpha_history, config.coherence_window)
    plasticity = compute_plasticity(agent.theta_history, agent.alpha_history)
    conflict = compute_conflict(all_sai_history, all_reward_history)

    return RunResult(
        sai_dyn=sai_dyn,
        eas_dyn=eas_dyn,
        identity_coherence=identity_coherence,
        plasticity=plasticity,
        conflict=conflict,
        total_reward=total_reward,
        episodes_survived=episodes_survived,
        final_theta=deepcopy(agent.theta),
        final_alpha=deepcopy(agent.alpha)
    )


# =============================================================================
# Experiment Runner
# =============================================================================

def run_condition(condition: str, config: AEConfig) -> Tuple[List[RunResult], Dict]:
    """Run all trials for a condition."""
    print(f"\n{'='*60}")
    print(f"Running IPUESA-AE - Condition: {condition}")
    print(f"{'='*60}")

    agent = AdaptiveAgent(config, condition)

    # Select perturbation schedule
    if condition == 'perturbed':
        perturbations = generate_perturbations(config.severe_perturbation_steps, severe=True)
    elif condition == 'no_adaptation':
        perturbations = generate_perturbations(config.perturbation_steps, severe=False)
    else:
        perturbations = generate_perturbations(config.perturbation_steps, severe=False)

    results = []
    for i in range(config.n_runs):
        result = run_single(agent, config, perturbations)
        results.append(result)

        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{config.n_runs} runs")

    # Aggregate metrics
    metrics = {
        'condition': condition,
        'SAI_dyn': np.mean([r.sai_dyn for r in results]),
        'EAS_dyn': np.mean([r.eas_dyn for r in results]),
        'IC': np.mean([r.identity_coherence for r in results]),
        'plasticity': np.mean([r.plasticity for r in results]),
        'conflict': np.mean([r.conflict for r in results]),
        'mean_reward': np.mean([r.total_reward for r in results]),
        'episodes_survived': np.mean([r.episodes_survived for r in results]),
        'mean_final_theta': average_theta([r.final_theta for r in results]),
        'mean_final_alpha': average_alpha([r.final_alpha for r in results])
    }

    return results, metrics


def average_theta(thetas: List[MetaPolicy]) -> MetaPolicy:
    """Compute average theta."""
    arrays = np.array([t.to_array() for t in thetas])
    return MetaPolicy.from_array(np.mean(arrays, axis=0))


def average_alpha(alphas: List[CognitiveArchitecture]) -> CognitiveArchitecture:
    """Compute average alpha."""
    arrays = np.array([a.to_array() for a in alphas])
    return CognitiveArchitecture.from_array(np.mean(arrays, axis=0))


def print_results(metrics: Dict):
    """Print results for a condition."""
    print(f"\n{'='*60}")
    print(f"RESULTS - {metrics['condition']}")
    print(f"{'='*60}")

    print(f"\nAdaptive Metrics:")
    print(f"  SAI_dyn (survival)      = {metrics['SAI_dyn']:.3f}")
    print(f"  EAS_dyn (agency)        = {metrics['EAS_dyn']:.3f}")
    print(f"  IC (coherence)          = {metrics['IC']:.3f}")
    print(f"  Plasticity              = {metrics['plasticity']:.3f}")
    print(f"  Conflict (SAI vs Rew)   = {metrics['conflict']:.3f}")
    print(f"  Mean reward             = {metrics['mean_reward']:.1f}")

    theta = metrics['mean_final_theta']
    print(f"\nFinal Theta (WHO):")
    print(f"  risk_aversion     = {theta.risk_aversion:.3f}")
    print(f"  exploration_rate  = {theta.exploration_rate:.3f}")
    print(f"  memory_depth      = {theta.memory_depth:.3f}")
    print(f"  prediction_weight = {theta.prediction_weight:.3f}")

    alpha = metrics['mean_final_alpha']
    print(f"\nFinal Alpha (HOW):")
    print(f"  attention_history    = {alpha.attention_history:.3f}")
    print(f"  attention_prediction = {alpha.attention_prediction:.3f}")
    print(f"  attention_immediate  = {alpha.attention_immediate:.3f}")
    print(f"  memory_update_rate   = {alpha.memory_update_rate:.3f}")
    print(f"  perceptual_gain      = {alpha.perceptual_gain:.3f}")

    sig = "YES" if metrics['SAI_dyn'] > 0.3 and metrics['IC'] > 0.4 else "NO"
    print(f"\nSelf-evidence: [{sig}]")


# =============================================================================
# Main Experiment
# =============================================================================

def run_full_experiment():
    """Run complete IPUESA-AE experiment."""
    print("=" * 70)
    print("IPUESA-AE: Adaptive Emergence Experiment")
    print("        Dual Adaptation: Policy (theta) + Architecture (alpha)")
    print("=" * 70)

    config = AEConfig()

    print(f"\nConfiguration:")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Lambda_e (existential): {config.lambda_e}")
    print(f"  Lambda_r (reward): {config.lambda_r}")
    print(f"  Perturbations at steps: {config.perturbation_steps}")
    print(f"  N episodes: {config.n_episodes}, N steps: {config.n_steps}")
    print(f"  N runs: {config.n_runs}")

    conditions = [
        'full_adaptive',    # Both theta and alpha adapt
        'meta_only',        # Only theta adapts
        'cognitive_only',   # Only alpha adapts
        'no_adaptation',    # Neither adapts
        'perturbed',        # Both adapt + severe perturbations
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
    print("IPUESA-AE: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<16} {'SAI_dyn':<10} {'EAS_dyn':<10} {'IC':<10} {'Plastic':<10} {'Conflict':<10} {'Sig':<6}")
    print("-" * 76)

    for condition in conditions:
        m = all_metrics[condition]
        sig = "YES" if m['SAI_dyn'] > 0.3 and m['IC'] > 0.4 else "NO"
        print(f"{condition:<16} {m['SAI_dyn']:<10.3f} {m['EAS_dyn']:<10.3f} {m['IC']:<10.3f} {m['plasticity']:<10.3f} {m['conflict']:<10.3f} [{sig}]")

    # Self-evidence criteria
    print("\n" + "=" * 70)
    print("SELF-EVIDENCE CRITERIA (ADAPTIVE EMERGENCE)")
    print("-" * 70)

    fa = all_metrics['full_adaptive']
    mo = all_metrics['meta_only']
    co = all_metrics['cognitive_only']
    na = all_metrics['no_adaptation']
    pt = all_metrics['perturbed']

    criteria = []

    # 1. SAI_dyn > 0.5
    c1 = fa['SAI_dyn'] > 0.5
    criteria.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] 1. SAI_dyn ({fa['SAI_dyn']:.3f}) > 0.5")

    # 2. full_adaptive >> meta_only
    c2 = fa['SAI_dyn'] > mo['SAI_dyn'] + 0.05
    criteria.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] 2. full_adaptive ({fa['SAI_dyn']:.3f}) >> meta_only ({mo['SAI_dyn']:.3f})")

    # 3. full_adaptive >> cognitive_only
    c3 = fa['SAI_dyn'] > co['SAI_dyn'] + 0.05
    criteria.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] 3. full_adaptive ({fa['SAI_dyn']:.3f}) >> cognitive_only ({co['SAI_dyn']:.3f})")

    # 4. full_adaptive >> no_adaptation
    c4 = fa['SAI_dyn'] > na['SAI_dyn'] + 0.1
    criteria.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] 4. full_adaptive ({fa['SAI_dyn']:.3f}) >> no_adaptation ({na['SAI_dyn']:.3f})")

    # 5. IC > 0.6
    c5 = fa['IC'] > 0.6
    criteria.append(c5)
    print(f"  [{'PASS' if c5 else 'FAIL'}] 5. IC ({fa['IC']:.3f}) > 0.6")

    # 6. Plasticity appropriate (0.1-0.4)
    c6 = 0.05 < fa['plasticity'] < 0.5
    criteria.append(c6)
    print(f"  [{'PASS' if c6 else 'FAIL'}] 6. Plasticity ({fa['plasticity']:.3f}) in [0.05, 0.5]")

    # 7. Conflict < 0
    c7 = fa['conflict'] < 0
    criteria.append(c7)
    print(f"  [{'PASS' if c7 else 'FAIL'}] 7. Conflict ({fa['conflict']:.3f}) < 0 (existence over reward)")

    # 8. Survives severe perturbations
    c8 = pt['SAI_dyn'] > 0.2
    criteria.append(c8)
    print(f"  [{'PASS' if c8 else 'FAIL'}] 8. Perturbed SAI_dyn ({pt['SAI_dyn']:.3f}) > 0.2")

    passed = sum(criteria)
    print(f"\n  Passed: {passed}/8 criteria")

    if passed >= 6:
        conclusion = "EVIDENCE OF ADAPTIVE SELF"
    elif passed >= 4:
        conclusion = "Partial evidence of adaptive emergence"
    else:
        conclusion = "No evidence - dual adaptation insufficient"

    print(f"\n  CONCLUSION: {conclusion}")
    print(f"\n  KEY INSIGHT: If full_adaptive >> meta_only, then cognitive")
    print(f"  architecture (alpha) provides ADDITIONAL resilience beyond")
    print(f"  policy adaptation alone. The agent adapts WHO it is AND")
    print(f"  HOW it thinks to survive.")
    print("=" * 70)

    # Identity portraits
    print("\n" + "=" * 70)
    print("IDENTITY PORTRAITS (WHO + HOW)")
    print("-" * 70)
    for condition in conditions:
        theta = all_metrics[condition]['mean_final_theta']
        alpha = all_metrics[condition]['mean_final_alpha']
        print(f"\n{condition}:")
        print(f"  THETA (WHO):")
        print(f"    Risk Aversion:  {'|' * int(theta.risk_aversion * 20):<20} {theta.risk_aversion:.2f}")
        print(f"    Exploration:    {'|' * int(theta.exploration_rate * 20):<20} {theta.exploration_rate:.2f}")
        print(f"    Memory:         {'|' * int(theta.memory_depth * 20):<20} {theta.memory_depth:.2f}")
        print(f"    Prediction:     {'|' * int(theta.prediction_weight * 20):<20} {theta.prediction_weight:.2f}")
        print(f"  ALPHA (HOW):")
        print(f"    Attn History:   {'|' * int(alpha.attention_history * 20):<20} {alpha.attention_history:.2f}")
        print(f"    Attn Predict:   {'|' * int(alpha.attention_prediction * 20):<20} {alpha.attention_prediction:.2f}")
        print(f"    Attn Immediate: {'|' * int(alpha.attention_immediate * 20):<20} {alpha.attention_immediate:.2f}")
        print(f"    Memory Rate:    {'|' * int(alpha.memory_update_rate * 20):<20} {alpha.memory_update_rate:.2f}")
        print(f"    Percept Gain:   {'|' * int(alpha.perceptual_gain * 20):<20} {alpha.perceptual_gain:.2f}")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "ipuesa_ae_results.json"
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
            return {'risk_aversion': obj.risk_aversion, 'exploration_rate': obj.exploration_rate,
                    'memory_depth': obj.memory_depth, 'prediction_weight': obj.prediction_weight}
        elif isinstance(obj, CognitiveArchitecture):
            return {'attention_history': obj.attention_history, 'attention_prediction': obj.attention_prediction,
                    'attention_immediate': obj.attention_immediate, 'memory_update_rate': obj.memory_update_rate,
                    'perceptual_gain': obj.perceptual_gain}
        elif np.isnan(obj) if isinstance(obj, float) else False:
            return 0.0
        return obj

    save_data = {
        'config': {
            'epsilon': config.epsilon,
            'lambda_e': config.lambda_e,
            'lambda_r': config.lambda_r,
            'n_episodes': config.n_episodes,
            'n_steps': config.n_steps,
            'n_runs': config.n_runs
        },
        'metrics': {k: {
            'SAI_dyn': v['SAI_dyn'],
            'EAS_dyn': v['EAS_dyn'],
            'IC': v['IC'],
            'plasticity': v['plasticity'],
            'conflict': v['conflict'] if not np.isnan(v['conflict']) else 0.0,
            'mean_final_theta': to_native(v['mean_final_theta']),
            'mean_final_alpha': to_native(v['mean_final_alpha'])
        } for k, v in all_metrics.items()},
        'self_evidence': {
            'criteria_passed': passed,
            'total_criteria': 8,
            'conclusion': conclusion
        }
    }

    with open(output_path, 'w') as f:
        json.dump(to_native(save_data), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_metrics


if __name__ == "__main__":
    run_full_experiment()
