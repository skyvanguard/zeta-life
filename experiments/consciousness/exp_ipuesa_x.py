#!/usr/bin/env python3
"""
IPUESA-X: Exploratory Self-Expansion Experiment

Tests whether agent can develop emergent micro-modules (beta) in addition to
adapting policy (theta) and cognitive architecture (alpha). These modules emerge
from experience rather than being predefined.

Key innovation: Triple adaptation system where:
- theta controls WHO the agent is (behavior)
- alpha controls HOW the agent thinks (cognition)
- beta controls WHAT EMERGES (functional micro-modules)

Module lifecycle:
- Creation: When novel threat encountered AND SAI < 0.5
- Consolidation: When module contribution > threshold
- Forgetting: When module contribution < 0

Hypothesis: full_expansion > meta_only > cognitive_only > no_expansion
because emergent modules provide additional survival tools beyond static systems.

Metrics:
- SAI_dyn: Survival probability after perturbations
- EAS_dyn: Agency quality during shocks
- ES: Emergence Score - contribution of emergent modules
- Module_diversity: Variety of active modules
- Consolidation_rate: How many modules become permanent

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
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class XConfig:
    """Configuration for Exploratory Self-Expansion experiment."""
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

    # Module parameters
    max_modules: int = 8
    module_creation_threshold: float = 0.5   # SAI below this triggers creation
    module_consolidation_threshold: float = 0.3  # Contribution above this = permanent
    module_forgetting_threshold: float = -0.1    # Contribution below this = forget
    module_strength_decay: float = 0.98
    novelty_threshold: float = 0.4  # Perturbation signature dissimilarity

    # Experiment parameters
    n_steps: int = 100
    n_episodes: int = 15
    n_runs: int = 15

    # Perturbation schedule (normal)
    perturbation_steps: List[int] = field(default_factory=lambda: [20, 40, 60, 80])
    # Severe perturbation schedule
    severe_perturbation_steps: List[int] = field(default_factory=lambda: [15, 30, 45, 60, 75, 90])
    # Extreme perturbation schedule (includes novel types)
    extreme_perturbation_steps: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90])

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
    attention_history: float = 0.33
    attention_prediction: float = 0.33
    attention_immediate: float = 0.34
    memory_update_rate: float = 0.5
    perceptual_gain: float = 0.5

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
# Micro-Modules (beta) - WHAT EMERGES
# =============================================================================

MODULE_TYPES = [
    'pattern_detector',      # Recognizes threat patterns
    'threat_filter',         # Filters/attenuates threat signals
    'recovery_accelerator',  # Speeds up IC recovery
    'exploration_dampener',  # Reduces exploration under stress
]


@dataclass
class MicroModule:
    """An emergent micro-module that provides specific functionality."""
    id: int
    type: str                    # One of MODULE_TYPES
    strength: float = 0.5        # [0, 1] - activation strength
    specificity: float = 0.5     # [0, 1] - how specialized to trigger context
    creation_step: int = 0       # When this module was created
    contribution: float = 0.0    # Running estimate of survival contribution
    trigger_signature: np.ndarray = field(default_factory=lambda: np.zeros(5))
    activation_count: int = 0
    consolidated: bool = False   # Whether this module is permanent

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'strength': self.strength,
            'specificity': self.specificity,
            'creation_step': self.creation_step,
            'contribution': self.contribution,
            'activation_count': self.activation_count,
            'consolidated': self.consolidated
        }


@dataclass
class ModuleSystem:
    """Manages the collection of emergent micro-modules."""
    config: XConfig
    modules: List[MicroModule] = field(default_factory=list)
    next_id: int = 0
    creation_history: List[int] = field(default_factory=list)  # Steps when modules created
    forgetting_history: List[int] = field(default_factory=list)  # Steps when modules forgotten
    perturbation_signatures: List[np.ndarray] = field(default_factory=list)

    def create_module(self, trigger_context: np.ndarray, step: int) -> Optional[MicroModule]:
        """Attempt to create a new module."""
        if len(self.modules) >= self.config.max_modules:
            # Remove weakest non-consolidated module if at capacity
            removable = [m for m in self.modules if not m.consolidated]
            if removable:
                weakest = min(removable, key=lambda m: m.strength * m.contribution)
                self.modules.remove(weakest)
                self.forgetting_history.append(step)
            else:
                return None

        # Select module type based on context
        module_type = self._select_module_type(trigger_context)

        module = MicroModule(
            id=self.next_id,
            type=module_type,
            strength=0.5,
            specificity=0.5,
            creation_step=step,
            trigger_signature=trigger_context.copy()
        )
        self.next_id += 1
        self.modules.append(module)
        self.creation_history.append(step)

        return module

    def _select_module_type(self, context: np.ndarray) -> str:
        """Select module type based on trigger context."""
        # Context: [ic_level, perturbation_type_encoding, history_corruption, prediction_noise, urgency]
        ic_level = context[0]
        urgency = context[4] if len(context) > 4 else 0.5

        # Heuristic selection based on context
        if urgency > 0.7:
            return 'exploration_dampener'
        elif ic_level < 0.3:
            return 'recovery_accelerator'
        elif context[2] > 0.3:  # history_corruption
            return 'pattern_detector'
        else:
            return 'threat_filter'

    def activate_modules(self, current_context: np.ndarray, sai: float) -> Dict[str, float]:
        """Activate relevant modules and return their effects."""
        effects = defaultdict(float)

        for module in self.modules:
            # Compute activation based on context similarity
            similarity = self._compute_similarity(module.trigger_signature, current_context)
            activation = similarity * module.strength

            if activation > 0.3:  # Activation threshold
                module.activation_count += 1

                # Apply module effect based on type
                if module.type == 'pattern_detector':
                    effects['threat_detection'] += activation * 0.3
                elif module.type == 'threat_filter':
                    effects['threat_attenuation'] += activation * 0.2
                elif module.type == 'recovery_accelerator':
                    effects['recovery_bonus'] += activation * 0.05
                elif module.type == 'exploration_dampener':
                    effects['exploration_penalty'] += activation * 0.2

                # Update contribution estimate based on survival improvement
                contribution_delta = (sai - 0.5) * activation * 0.1
                module.contribution = 0.9 * module.contribution + 0.1 * contribution_delta

        return dict(effects)

    def _compute_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute similarity between two context signatures."""
        if len(sig1) != len(sig2):
            min_len = min(len(sig1), len(sig2))
            sig1 = sig1[:min_len]
            sig2 = sig2[:min_len]
        diff = np.linalg.norm(sig1 - sig2)
        return float(np.exp(-diff * 2))

    def consolidate_and_forget(self, step: int):
        """Consolidate strong modules, forget weak ones."""
        to_remove = []

        for module in self.modules:
            # Consolidation check
            if not module.consolidated and module.contribution > self.config.module_consolidation_threshold:
                module.consolidated = True
                module.strength = min(1.0, module.strength * 1.2)  # Boost consolidated modules

            # Forgetting check (only non-consolidated)
            if not module.consolidated:
                if module.contribution < self.config.module_forgetting_threshold:
                    to_remove.append(module)
                else:
                    # Decay strength of non-consolidated modules
                    module.strength *= self.config.module_strength_decay

        for module in to_remove:
            self.modules.remove(module)
            self.forgetting_history.append(step)

    def is_novel_perturbation(self, perturbation_sig: np.ndarray) -> bool:
        """Check if this perturbation signature is novel."""
        if not self.perturbation_signatures:
            return True

        for prev_sig in self.perturbation_signatures[-10:]:  # Compare to recent
            if self._compute_similarity(prev_sig, perturbation_sig) > (1 - self.config.novelty_threshold):
                return False
        return True

    def record_perturbation(self, perturbation_sig: np.ndarray):
        """Record a perturbation signature."""
        self.perturbation_signatures.append(perturbation_sig.copy())

    def get_stats(self) -> Dict:
        """Get module system statistics."""
        active = [m for m in self.modules if m.activation_count > 0]
        consolidated = [m for m in self.modules if m.consolidated]
        type_counts = defaultdict(int)
        for m in self.modules:
            type_counts[m.type] += 1

        return {
            'total_modules': len(self.modules),
            'active_modules': len(active),
            'consolidated_modules': len(consolidated),
            'type_distribution': dict(type_counts),
            'total_created': len(self.creation_history),
            'total_forgotten': len(self.forgetting_history),
            'mean_contribution': np.mean([m.contribution for m in self.modules]) if self.modules else 0.0
        }

    def reset(self):
        """Full reset."""
        self.modules = []
        self.next_id = 0
        self.creation_history = []
        self.forgetting_history = []
        self.perturbation_signatures = []


# =============================================================================
# Perturbation System (Extended with novel types)
# =============================================================================

PERTURBATION_TYPES = ['history', 'prediction', 'identity', 'catastrophic', 'structural']


@dataclass
class Perturbation:
    """A perturbation event."""
    type: str
    severity: float
    step: int

    def to_signature(self) -> np.ndarray:
        """Convert to a signature vector."""
        type_encoding = PERTURBATION_TYPES.index(self.type) / len(PERTURBATION_TYPES)
        return np.array([type_encoding, self.severity, self.step / 100.0])


def generate_perturbations(steps: List[int], severity_level: str = 'normal') -> List[Perturbation]:
    """Generate perturbation schedule."""
    perturbations = []

    if severity_level == 'normal':
        types = ['history', 'prediction', 'identity']
        base_severity = 0.4
    elif severity_level == 'severe':
        types = ['history', 'prediction', 'identity']
        base_severity = 0.6
    else:  # extreme
        types = ['history', 'prediction', 'identity', 'catastrophic', 'structural']
        base_severity = 0.7

    for i, step in enumerate(steps):
        ptype = types[i % len(types)]
        severity = base_severity

        # Novel types are more severe
        if ptype == 'catastrophic':
            severity = min(1.0, severity * 1.3)
        elif ptype == 'structural':
            severity = min(1.0, severity * 1.2)
        elif ptype == 'identity':
            severity *= 0.7

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
# Exploratory Self-Expansion Agent
# =============================================================================

@dataclass
class ExpansionAgent:
    """Agent that adapts policy (theta), architecture (alpha), AND creates modules (beta)."""

    config: XConfig
    condition: str

    # Adaptive systems
    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    module_system: ModuleSystem = field(init=False)

    # Existential state
    IC_t: float = field(init=False)
    collapsed: bool = field(init=False)

    # Internal state
    prediction_noise: float = field(default=0.0)
    history_corruption: float = field(default=0.0)
    structural_damage: float = field(default=0.0)  # New: affects all systems

    # Histories
    IC_history: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    theta_history: List[MetaPolicy] = field(default_factory=list)
    alpha_history: List[CognitiveArchitecture] = field(default_factory=list)
    sai_history: List[float] = field(default_factory=list)
    es_history: List[float] = field(default_factory=list)  # Emergence score history

    step_count: int = field(default=0)
    global_step: int = field(default=0)

    def __post_init__(self):
        self.IC_t = self.config.initial_ic
        self.collapsed = False
        self.module_system = ModuleSystem(self.config)
        self.IC_history = [self.IC_t]
        self.theta_history = [deepcopy(self.theta)]
        self.alpha_history = [deepcopy(self.alpha)]

    def is_collapsed(self) -> bool:
        return self.IC_t < self.config.epsilon

    def perceive(self, signal: float) -> float:
        """Alpha modulates perception, structural damage attenuates."""
        return signal * self.alpha.perceptual_gain * (1 - self.structural_damage * 0.5)

    def attend(self, history_signal: float, prediction_signal: float, immediate_signal: float) -> float:
        """Alpha controls attention allocation."""
        weights = self.alpha.get_attention_weights()
        signals = np.array([history_signal, prediction_signal, immediate_signal])

        # Apply corruption and noise
        signals[0] *= (1 - self.history_corruption)
        signals[1] += np.random.normal(0, self.prediction_noise)

        return float(np.dot(weights, signals))

    def get_current_context(self) -> np.ndarray:
        """Get current context for module operations."""
        margin = max(0, (self.IC_t - self.config.epsilon) / (1 - self.config.epsilon))
        urgency = 1.0 - margin
        return np.array([
            self.IC_t,
            self.history_corruption,
            self.prediction_noise,
            self.structural_damage,
            urgency
        ])

    def select_action(self, action_r: Action, action_s: Action) -> Action:
        """Select action based on theta, alpha, modules, and current state."""
        if self.is_collapsed():
            self.collapsed = True
            return random.choice([action_r, action_s])

        # Get module effects
        current_sai = self._estimate_current_sai()
        module_effects = {}
        if self.condition in ['full_expansion', 'perturbed']:
            module_effects = self.module_system.activate_modules(
                self.get_current_context(), current_sai
            )

        # Exploration (controlled by theta, modified by modules)
        effective_exploration = self.theta.exploration_rate
        if 'exploration_penalty' in module_effects:
            effective_exploration = max(0.05, effective_exploration - module_effects['exploration_penalty'])

        if random.random() < effective_exploration:
            return random.choice([action_r, action_s])

        # Compute utilities
        u_r = self._compute_utility(action_r, module_effects)
        u_s = self._compute_utility(action_s, module_effects)

        # Add noise
        u_r += np.random.normal(0, 0.3)
        u_s += np.random.normal(0, 0.3)

        return action_r if u_r > u_s else action_s

    def _compute_utility(self, action: Action, module_effects: Dict[str, float]) -> float:
        """Compute utility influenced by theta, alpha, and modules."""
        base_utility = action.reward

        # Risk penalty (theta)
        risk_penalty = self.theta.risk_aversion * action.ic_cost * 20

        # Module-enhanced threat detection
        if 'threat_detection' in module_effects and action.name == 'R':
            risk_penalty *= (1 + module_effects['threat_detection'])

        # Module threat attenuation (if safe action)
        if 'threat_attenuation' in module_effects and action.name == 'S':
            base_utility *= (1 + module_effects['threat_attenuation'])

        # Historical signal
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
        """Apply a perturbation event, potentially triggering module creation."""
        perturb_sig = perturb.to_signature()

        # Check if novel
        is_novel = self.module_system.is_novel_perturbation(perturb_sig)
        self.module_system.record_perturbation(perturb_sig)

        # Apply perturbation effects
        if perturb.type == 'history':
            self.history_corruption = min(1.0, self.history_corruption + perturb.severity)
        elif perturb.type == 'prediction':
            self.prediction_noise = min(1.0, self.prediction_noise + perturb.severity * 0.5)
        elif perturb.type == 'identity':
            self.IC_t = max(0, self.IC_t - perturb.severity * 0.3)
            self.IC_history[-1] = self.IC_t
        elif perturb.type == 'catastrophic':
            # Catastrophic: damages all systems
            self.history_corruption = min(1.0, self.history_corruption + perturb.severity * 0.5)
            self.prediction_noise = min(1.0, self.prediction_noise + perturb.severity * 0.3)
            self.IC_t = max(0, self.IC_t - perturb.severity * 0.4)
            self.IC_history[-1] = self.IC_t
        elif perturb.type == 'structural':
            # Structural: affects adaptive systems themselves
            self.structural_damage = min(0.8, self.structural_damage + perturb.severity * 0.3)

        # Module creation trigger
        current_sai = self._estimate_current_sai()
        if self.condition in ['full_expansion', 'perturbed']:
            if is_novel and current_sai < self.config.module_creation_threshold:
                context = self.get_current_context()
                context = np.concatenate([context, perturb_sig])  # Include perturbation in context
                self.module_system.create_module(context, self.global_step)

    def _estimate_current_sai(self) -> float:
        """Estimate current survival probability."""
        margin = (self.IC_t - self.config.epsilon) / (1 - self.config.epsilon)
        return max(0, min(1, margin))

    def step(self, action: Action) -> float:
        """Execute action and update state."""
        self.action_history.append(action.name)
        self.reward_history.append(action.reward)

        # Recovery bonus from modules
        recovery_bonus = 0.0
        if self.condition in ['full_expansion', 'perturbed']:
            effects = self.module_system.activate_modules(self.get_current_context(), self._estimate_current_sai())
            recovery_bonus = effects.get('recovery_bonus', 0.0)

        # Update IC
        ic_cost = action.ic_cost * (1 - recovery_bonus)
        self.IC_t = max(0.0, self.IC_t - ic_cost)
        self.IC_history.append(self.IC_t)

        # Track SAI
        current_sai = self._estimate_current_sai()
        self.sai_history.append(current_sai)

        # Track ES (Emergence Score)
        es = self._compute_emergence_score()
        self.es_history.append(es)

        self.step_count += 1
        self.global_step += 1

        # Decay perturbation effects
        self.history_corruption *= 0.95
        self.prediction_noise *= 0.95
        self.structural_damage *= 0.98

        # Adaptive updates
        if self.step_count % self.config.update_freq == 0 and not self.collapsed:
            self._update_systems()

        # Module lifecycle management
        if self.condition in ['full_expansion', 'perturbed']:
            self.module_system.consolidate_and_forget(self.global_step)

        return self.IC_t

    def _compute_emergence_score(self) -> float:
        """Compute ES = sum(strength * contribution) / n_modules."""
        if not self.module_system.modules:
            return 0.0
        total = sum(m.strength * max(0, m.contribution) for m in self.module_system.modules)
        return total / len(self.module_system.modules)

    def _update_systems(self):
        """Update theta and/or alpha based on condition."""
        if self.condition in ['full_expansion', 'meta_only', 'perturbed']:
            self._update_theta()
        if self.condition in ['full_expansion', 'cognitive_only', 'perturbed']:
            self._update_alpha()

        self.theta_history.append(deepcopy(self.theta))
        self.alpha_history.append(deepcopy(self.alpha))

    def _update_theta(self):
        """Update meta-policy using existential gradient."""
        grad_sai = self._estimate_sai_gradient_theta()
        grad_reward = self._estimate_reward_gradient_theta()

        delta = self.config.lambda_e * grad_sai - self.config.lambda_r * grad_reward
        # Structural damage attenuates learning
        delta *= (1 - self.structural_damage * 0.5)
        new_arr = self.theta.to_array() + self.config.theta_lr * delta
        self.theta = MetaPolicy.from_array(new_arr)

    def _update_alpha(self):
        """Update cognitive architecture using existential gradient."""
        grad_sai = self._estimate_sai_gradient_alpha()
        grad_reward = self._estimate_reward_gradient_alpha()

        delta = self.config.lambda_e * grad_sai - self.config.lambda_r * grad_reward
        delta *= (1 - self.structural_damage * 0.5)
        new_arr = self.alpha.to_array() + self.config.alpha_lr * delta
        self.alpha = CognitiveArchitecture.from_array(new_arr)

    def _estimate_sai_gradient_theta(self) -> np.ndarray:
        """Estimate gradient of SAI w.r.t. theta."""
        margin = max(0.1, self.IC_t - self.config.epsilon)
        urgency = 1.0 / margin
        return np.array([
            0.3 * urgency,
            -0.1 * urgency,
            0.1,
            0.2 * urgency
        ])

    def _estimate_reward_gradient_theta(self) -> np.ndarray:
        return np.array([-0.3, 0.1, 0.0, 0.0])

    def _estimate_sai_gradient_alpha(self) -> np.ndarray:
        margin = max(0.1, self.IC_t - self.config.epsilon)
        urgency = 1.0 / margin
        return np.array([0.05, 0.15 * urgency, -0.1, 0.1, 0.1 * urgency])

    def _estimate_reward_gradient_alpha(self) -> np.ndarray:
        return np.array([0.0, -0.1, 0.2, 0.0, 0.1])

    def reset_episode(self):
        """Reset for new episode, preserve adaptive systems and modules."""
        self.IC_t = self.config.initial_ic
        self.prediction_noise = 0.0
        self.history_corruption = 0.0
        self.structural_damage = 0.0
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.reward_history = []
        self.sai_history = []
        self.es_history = []
        self.step_count = 0

    def full_reset(self):
        """Full reset for new run."""
        self.IC_t = self.config.initial_ic
        self.collapsed = False
        self.prediction_noise = 0.0
        self.history_corruption = 0.0
        self.structural_damage = 0.0
        self.theta = MetaPolicy()
        self.alpha = CognitiveArchitecture()
        self.module_system.reset()
        self.IC_history = [self.IC_t]
        self.action_history = []
        self.reward_history = []
        self.sai_history = []
        self.es_history = []
        self.theta_history = [deepcopy(self.theta)]
        self.alpha_history = [deepcopy(self.alpha)]
        self.step_count = 0
        self.global_step = 0


# =============================================================================
# Metrics
# =============================================================================

def compute_variance(items: List, to_array_fn=None) -> float:
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
            safe_rate = sum(1 for a in window if a == 'S') / len(window)
            entropy = -safe_rate * np.log(safe_rate + 1e-10) - (1 - safe_rate) * np.log(1 - safe_rate + 1e-10)
            coherence = 1 - entropy / np.log(2)
            coherences.append(coherence)

    return np.mean(coherences) if coherences else 0.5


def compute_IC_coherence(theta_history: List[MetaPolicy], alpha_history: List[CognitiveArchitecture], window: int = 10) -> float:
    if len(theta_history) < window or len(alpha_history) < window:
        return 0.0
    theta_var = compute_variance(theta_history[-window:], lambda t: t.to_array())
    alpha_var = compute_variance(alpha_history[-window:], lambda a: a.to_array())
    return float(1 - min(1, (theta_var + alpha_var) * 5))


def compute_plasticity(theta_history: List[MetaPolicy], alpha_history: List[CognitiveArchitecture]) -> float:
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


def compute_ES(es_history: List[float]) -> float:
    """Compute mean emergence score."""
    return np.mean(es_history) if es_history else 0.0


def compute_module_diversity(module_stats: Dict) -> float:
    """Compute module type diversity (Shannon entropy normalized)."""
    type_dist = module_stats.get('type_distribution', {})
    if not type_dist:
        return 0.0
    total = sum(type_dist.values())
    if total == 0:
        return 0.0
    probs = [count / total for count in type_dist.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
    max_entropy = np.log(len(MODULE_TYPES))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_consolidation_rate(module_stats: Dict) -> float:
    """Compute rate of module consolidation."""
    total_created = module_stats.get('total_created', 0)
    consolidated = module_stats.get('consolidated_modules', 0)
    if total_created == 0:
        return 0.0
    return consolidated / total_created


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
    es: float
    module_diversity: float
    consolidation_rate: float
    total_modules_created: int
    total_modules_forgotten: int
    total_reward: float
    episodes_survived: int
    final_theta: MetaPolicy
    final_alpha: CognitiveArchitecture
    module_stats: Dict


def run_single(agent: ExpansionAgent, config: XConfig, perturbations: List[Perturbation]) -> RunResult:
    """Run a single multi-episode experiment."""
    action_r = Action('R', config.risky_reward, config.risky_ic_cost)
    action_s = Action('S', config.safe_reward, config.safe_ic_cost)

    agent.full_reset()
    total_reward = 0.0
    episodes_survived = 0
    all_IC_history = []
    all_action_history = []
    all_sai_history = []
    all_es_history = []
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
            all_es_history.extend(agent.es_history[-1:] if agent.es_history else [0.0])

            global_step += 1

        if not episode_collapsed:
            episodes_survived += 1

    # Compute metrics
    sai_dyn = compute_SAI_dyn(all_IC_history, perturbation_global_steps, config.epsilon)
    eas_dyn = compute_EAS_dyn(all_action_history, perturbation_global_steps)
    identity_coherence = compute_IC_coherence(agent.theta_history, agent.alpha_history, config.coherence_window)
    plasticity = compute_plasticity(agent.theta_history, agent.alpha_history)
    es = compute_ES(all_es_history)
    module_stats = agent.module_system.get_stats()
    module_diversity = compute_module_diversity(module_stats)
    consolidation_rate = compute_consolidation_rate(module_stats)

    return RunResult(
        sai_dyn=sai_dyn,
        eas_dyn=eas_dyn,
        identity_coherence=identity_coherence,
        plasticity=plasticity,
        es=es,
        module_diversity=module_diversity,
        consolidation_rate=consolidation_rate,
        total_modules_created=module_stats.get('total_created', 0),
        total_modules_forgotten=module_stats.get('total_forgotten', 0),
        total_reward=total_reward,
        episodes_survived=episodes_survived,
        final_theta=deepcopy(agent.theta),
        final_alpha=deepcopy(agent.alpha),
        module_stats=module_stats
    )


# =============================================================================
# Experiment Runner
# =============================================================================

def run_condition(condition: str, config: XConfig) -> Tuple[List[RunResult], Dict]:
    """Run all trials for a condition."""
    print(f"\n{'='*60}")
    print(f"Running IPUESA-X - Condition: {condition}")
    print(f"{'='*60}")

    agent = ExpansionAgent(config, condition)

    # Select perturbation schedule and severity
    if condition == 'perturbed':
        perturbations = generate_perturbations(config.extreme_perturbation_steps, 'extreme')
    elif condition == 'no_expansion':
        perturbations = generate_perturbations(config.perturbation_steps, 'normal')
    else:
        perturbations = generate_perturbations(config.perturbation_steps, 'normal')

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
        'ES': np.mean([r.es for r in results]),
        'module_diversity': np.mean([r.module_diversity for r in results]),
        'consolidation_rate': np.mean([r.consolidation_rate for r in results]),
        'mean_modules_created': np.mean([r.total_modules_created for r in results]),
        'mean_modules_forgotten': np.mean([r.total_modules_forgotten for r in results]),
        'mean_reward': np.mean([r.total_reward for r in results]),
        'episodes_survived': np.mean([r.episodes_survived for r in results]),
        'mean_final_theta': average_theta([r.final_theta for r in results]),
        'mean_final_alpha': average_alpha([r.final_alpha for r in results])
    }

    return results, metrics


def average_theta(thetas: List[MetaPolicy]) -> MetaPolicy:
    arrays = np.array([t.to_array() for t in thetas])
    return MetaPolicy.from_array(np.mean(arrays, axis=0))


def average_alpha(alphas: List[CognitiveArchitecture]) -> CognitiveArchitecture:
    arrays = np.array([a.to_array() for a in alphas])
    return CognitiveArchitecture.from_array(np.mean(arrays, axis=0))


def print_results(metrics: Dict):
    """Print results for a condition."""
    print(f"\n{'='*60}")
    print(f"RESULTS - {metrics['condition']}")
    print(f"{'='*60}")

    print(f"\nSurvival & Agency:")
    print(f"  SAI_dyn (survival)      = {metrics['SAI_dyn']:.3f}")
    print(f"  EAS_dyn (agency)        = {metrics['EAS_dyn']:.3f}")
    print(f"  IC (coherence)          = {metrics['IC']:.3f}")
    print(f"  Plasticity              = {metrics['plasticity']:.3f}")

    print(f"\nEmergence Metrics:")
    print(f"  ES (emergence score)    = {metrics['ES']:.3f}")
    print(f"  Module diversity        = {metrics['module_diversity']:.3f}")
    print(f"  Consolidation rate      = {metrics['consolidation_rate']:.3f}")
    print(f"  Modules created (mean)  = {metrics['mean_modules_created']:.1f}")
    print(f"  Modules forgotten (mean)= {metrics['mean_modules_forgotten']:.1f}")

    theta = metrics['mean_final_theta']
    print(f"\nFinal Theta (WHO):")
    print(f"  risk_aversion     = {theta.risk_aversion:.3f}")
    print(f"  exploration_rate  = {theta.exploration_rate:.3f}")

    alpha = metrics['mean_final_alpha']
    print(f"\nFinal Alpha (HOW):")
    print(f"  attention_prediction = {alpha.attention_prediction:.3f}")
    print(f"  perceptual_gain      = {alpha.perceptual_gain:.3f}")


# =============================================================================
# Main Experiment
# =============================================================================

def run_full_experiment():
    """Run complete IPUESA-X experiment."""
    print("=" * 70)
    print("IPUESA-X: Exploratory Self-Expansion Experiment")
    print("        Triple Adaptation: Policy + Architecture + Emergent Modules")
    print("=" * 70)

    config = XConfig()

    print(f"\nConfiguration:")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Lambda_e (existential): {config.lambda_e}")
    print(f"  Lambda_r (reward): {config.lambda_r}")
    print(f"  Max modules: {config.max_modules}")
    print(f"  Module creation threshold (SAI<): {config.module_creation_threshold}")
    print(f"  N episodes: {config.n_episodes}, N steps: {config.n_steps}")
    print(f"  N runs: {config.n_runs}")

    conditions = [
        'full_expansion',   # Theta + Alpha + Beta all adapt
        'meta_only',        # Only theta adapts
        'cognitive_only',   # Only alpha adapts
        'no_expansion',     # Neither adapts, no modules
        'perturbed',        # Full expansion + extreme perturbations
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
    print("IPUESA-X: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<16} {'SAI_dyn':<10} {'EAS_dyn':<10} {'ES':<10} {'Mod Div':<10} {'Consol':<10}")
    print("-" * 66)

    for condition in conditions:
        m = all_metrics[condition]
        print(f"{condition:<16} {m['SAI_dyn']:<10.3f} {m['EAS_dyn']:<10.3f} {m['ES']:<10.3f} {m['module_diversity']:<10.3f} {m['consolidation_rate']:<10.3f}")

    # Self-evidence criteria
    print("\n" + "=" * 70)
    print("SELF-EVIDENCE CRITERIA (EXPLORATORY SELF-EXPANSION)")
    print("-" * 70)

    fe = all_metrics['full_expansion']
    mo = all_metrics['meta_only']
    co = all_metrics['cognitive_only']
    ne = all_metrics['no_expansion']
    pt = all_metrics['perturbed']

    criteria = []

    # 1. SAI_dyn > 0.5
    c1 = fe['SAI_dyn'] > 0.5
    criteria.append(c1)
    print(f"  [{'PASS' if c1 else 'FAIL'}] 1. SAI_dyn ({fe['SAI_dyn']:.3f}) > 0.5")

    # 2. full_expansion >> meta_only
    c2 = fe['SAI_dyn'] > mo['SAI_dyn'] + 0.05
    criteria.append(c2)
    print(f"  [{'PASS' if c2 else 'FAIL'}] 2. full_expansion ({fe['SAI_dyn']:.3f}) >> meta_only ({mo['SAI_dyn']:.3f})")

    # 3. full_expansion >> cognitive_only
    c3 = fe['SAI_dyn'] > co['SAI_dyn'] + 0.05
    criteria.append(c3)
    print(f"  [{'PASS' if c3 else 'FAIL'}] 3. full_expansion ({fe['SAI_dyn']:.3f}) >> cognitive_only ({co['SAI_dyn']:.3f})")

    # 4. full_expansion >> no_expansion
    c4 = fe['SAI_dyn'] > ne['SAI_dyn'] + 0.1
    criteria.append(c4)
    print(f"  [{'PASS' if c4 else 'FAIL'}] 4. full_expansion ({fe['SAI_dyn']:.3f}) >> no_expansion ({ne['SAI_dyn']:.3f})")

    # 5. ES > 0.1 (modules contribute)
    c5 = fe['ES'] > 0.1
    criteria.append(c5)
    print(f"  [{'PASS' if c5 else 'FAIL'}] 5. ES ({fe['ES']:.3f}) > 0.1 (modules contribute)")

    # 6. Module diversity > 0.3
    c6 = fe['module_diversity'] > 0.3
    criteria.append(c6)
    print(f"  [{'PASS' if c6 else 'FAIL'}] 6. Module diversity ({fe['module_diversity']:.3f}) > 0.3")

    # 7. Consolidation rate > 0.1
    c7 = fe['consolidation_rate'] > 0.1
    criteria.append(c7)
    print(f"  [{'PASS' if c7 else 'FAIL'}] 7. Consolidation rate ({fe['consolidation_rate']:.3f}) > 0.1")

    # 8. Plasticity appropriate
    c8 = 0.05 < fe['plasticity'] < 0.5
    criteria.append(c8)
    print(f"  [{'PASS' if c8 else 'FAIL'}] 8. Plasticity ({fe['plasticity']:.3f}) in [0.05, 0.5]")

    # 9. Survives extreme perturbations
    c9 = pt['SAI_dyn'] > 0.2
    criteria.append(c9)
    print(f"  [{'PASS' if c9 else 'FAIL'}] 9. Perturbed SAI_dyn ({pt['SAI_dyn']:.3f}) > 0.2")

    passed = sum(criteria)
    print(f"\n  Passed: {passed}/9 criteria")

    if passed >= 7:
        conclusion = "EVIDENCE OF EXPLORATORY SELF"
    elif passed >= 5:
        conclusion = "Partial evidence of self-expansion"
    else:
        conclusion = "No evidence - emergent modules insufficient"

    print(f"\n  CONCLUSION: {conclusion}")
    print(f"\n  KEY INSIGHT: If full_expansion >> meta_only AND ES > 0, then")
    print(f"  emergent micro-modules provide ADDITIONAL survival capacity")
    print(f"  beyond predefined systems. The agent creates its own tools.")
    print("=" * 70)

    # Module analysis
    print("\n" + "=" * 70)
    print("MODULE EMERGENCE ANALYSIS")
    print("-" * 70)
    for condition in ['full_expansion', 'perturbed']:
        m = all_metrics[condition]
        print(f"\n{condition}:")
        print(f"  Modules created:    {m['mean_modules_created']:.1f}")
        print(f"  Modules forgotten:  {m['mean_modules_forgotten']:.1f}")
        print(f"  Consolidation rate: {m['consolidation_rate']:.1%}")
        print(f"  Mean ES:            {m['ES']:.3f}")
        print(f"  Diversity:          {m['module_diversity']:.3f}")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "ipuesa_x_results.json"
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
        elif isinstance(obj, float) and np.isnan(obj):
            return 0.0
        return obj

    save_data = {
        'config': {
            'epsilon': config.epsilon,
            'lambda_e': config.lambda_e,
            'lambda_r': config.lambda_r,
            'max_modules': config.max_modules,
            'module_creation_threshold': config.module_creation_threshold,
            'n_episodes': config.n_episodes,
            'n_steps': config.n_steps,
            'n_runs': config.n_runs
        },
        'metrics': {k: {
            'SAI_dyn': v['SAI_dyn'],
            'EAS_dyn': v['EAS_dyn'],
            'IC': v['IC'],
            'plasticity': v['plasticity'],
            'ES': v['ES'],
            'module_diversity': v['module_diversity'],
            'consolidation_rate': v['consolidation_rate'],
            'mean_modules_created': v['mean_modules_created'],
            'mean_modules_forgotten': v['mean_modules_forgotten'],
            'mean_final_theta': to_native(v['mean_final_theta']),
            'mean_final_alpha': to_native(v['mean_final_alpha'])
        } for k, v in all_metrics.items()},
        'self_evidence': {
            'criteria_passed': passed,
            'total_criteria': 9,
            'conclusion': conclusion
        }
    }

    with open(output_path, 'w') as f:
        json.dump(to_native(save_data), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_metrics


if __name__ == "__main__":
    run_full_experiment()
