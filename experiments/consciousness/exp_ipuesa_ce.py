#!/usr/bin/env python3
"""
IPUESA-CE: Co-Evolution Experiment

Tests whether a group of agents with emergent self can interact, cooperate,
and compete to survive collectively and evolve new identity strategies.

Key innovation: Multi-agent system where each agent has triple adaptation
(theta/alpha/beta) AND social dynamics (cooperation, competition, signaling).

Hypothesis: Social interaction enables collective survival strategies that
exceed individual adaptation alone.

Metrics:
- IS: Individual Survival (% agents maintaining identity)
- CS: Collective Survival (% surviving AND cooperating)
- ID: Identity Diversity (variance of theta in population)
- PA: Prediction Accuracy (alpha coherent with perturbations)
- ER: Emergent Roles (role differentiation score)
- RP: Resilience to Perturbation (% survived without collapse)
- CE: Communication Efficacy (correct signals adopted)
- MA: Meta-Adaptation (beta generates collective strategies)

Author: Claude + Human collaboration
Date: 2026-01-10
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy
import json
from pathlib import Path
import random
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CEConfig:
    """Configuration for Co-Evolution experiment."""
    # Population
    n_agents: int = 20
    max_population: int = 30
    min_population: int = 5

    # Environment
    initial_resources: float = 100.0
    resource_regeneration: float = 5.0
    resource_target: float = 10.0

    # Existential
    epsilon: float = 0.15
    initial_ic: float = 1.0
    risky_ic_cost: float = 0.06
    safe_ic_cost: float = 0.01

    # Priority weights
    lambda_e: float = 0.8
    lambda_r: float = 0.2

    # Learning rates
    theta_lr: float = 0.08
    alpha_lr: float = 0.04
    update_freq: int = 5

    # Evolution
    fitness_threshold: float = 0.55
    mutation_rate: float = 0.15
    mutation_strength: float = 0.12
    evolution_interval: int = 25

    # Social
    cooperation_cost: float = 0.08
    cooperation_benefit: float = 0.12
    signal_cost: float = 0.01
    trust_update_rate: float = 0.1

    # Modules
    max_modules: int = 6
    module_creation_threshold: float = 0.5
    module_consolidation_threshold: float = 0.25

    # Experiment
    n_steps: int = 150
    n_episodes: int = 8
    n_runs: int = 8

    # Perturbation schedules
    normal_perturbation_steps: List[int] = field(
        default_factory=lambda: [25, 50, 75, 100, 125]
    )
    catastrophic_perturbation_steps: List[int] = field(
        default_factory=lambda: [50, 100]
    )


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

    @classmethod
    def random(cls) -> 'MetaPolicy':
        return cls(
            risk_aversion=random.uniform(0.3, 0.7),
            exploration_rate=random.uniform(0.2, 0.5),
            memory_depth=random.uniform(0.3, 0.7),
            prediction_weight=random.uniform(0.3, 0.7)
        )


# =============================================================================
# Cognitive Architecture (alpha) - HOW the agent thinks
# =============================================================================

@dataclass
class CognitiveArchitecture:
    """Cognitive architecture parameters."""
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
            attention_history=float(np.clip(arr[0], 0.1, 0.8)),
            attention_prediction=float(np.clip(arr[1], 0.1, 0.8)),
            attention_immediate=float(np.clip(arr[2], 0.1, 0.8)),
            memory_update_rate=float(np.clip(arr[3], 0, 1)),
            perceptual_gain=float(np.clip(arr[4], 0.2, 1))
        )

    @classmethod
    def random(cls) -> 'CognitiveArchitecture':
        h = random.uniform(0.2, 0.5)
        p = random.uniform(0.2, 0.5)
        i = 1 - h - p
        return cls(
            attention_history=h,
            attention_prediction=p,
            attention_immediate=max(0.1, i),
            memory_update_rate=random.uniform(0.3, 0.7),
            perceptual_gain=random.uniform(0.4, 0.8)
        )


# =============================================================================
# Micro-Modules (beta) - WHAT EMERGES
# =============================================================================

MODULE_TYPES = ['pattern_detector', 'threat_filter', 'recovery_accelerator', 'exploration_dampener']


@dataclass
class MicroModule:
    """Emergent micro-module."""
    id: int
    type: str
    strength: float = 0.5
    specificity: float = 0.5
    creation_step: int = 0
    contribution: float = 0.0
    consolidated: bool = False

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'strength': self.strength,
            'contribution': self.contribution,
            'consolidated': self.consolidated
        }


@dataclass
class ModuleSystem:
    """Manages emergent micro-modules."""
    max_modules: int = 6
    modules: List[MicroModule] = field(default_factory=list)
    next_id: int = 0

    def create_module(self, context: np.ndarray, step: int) -> Optional[MicroModule]:
        if len(self.modules) >= self.max_modules:
            removable = [m for m in self.modules if not m.consolidated]
            if removable:
                weakest = min(removable, key=lambda m: m.strength * max(0, m.contribution))
                self.modules.remove(weakest)
            else:
                return None

        module_type = self._select_type(context)
        module = MicroModule(
            id=self.next_id,
            type=module_type,
            strength=0.5,
            creation_step=step
        )
        self.next_id += 1
        self.modules.append(module)
        return module

    def _select_type(self, context: np.ndarray) -> str:
        ic_level = context[0] if len(context) > 0 else 0.5
        urgency = context[-1] if len(context) > 1 else 0.5

        if urgency > 0.7:
            return 'exploration_dampener'
        elif ic_level < 0.3:
            return 'recovery_accelerator'
        elif random.random() < 0.5:
            return 'pattern_detector'
        else:
            return 'threat_filter'

    def activate(self, context: np.ndarray, sai: float) -> Dict[str, float]:
        effects = defaultdict(float)
        for module in self.modules:
            activation = module.strength * (1 - abs(sai - 0.5))
            if activation > 0.2:
                if module.type == 'pattern_detector':
                    effects['threat_detection'] += activation * 0.2
                elif module.type == 'threat_filter':
                    effects['threat_attenuation'] += activation * 0.15
                elif module.type == 'recovery_accelerator':
                    effects['recovery_bonus'] += activation * 0.03
                elif module.type == 'exploration_dampener':
                    effects['exploration_penalty'] += activation * 0.15

                module.contribution = 0.9 * module.contribution + 0.1 * (sai - 0.5) * activation
        return dict(effects)

    def consolidate_and_forget(self, threshold_consol: float, threshold_forget: float):
        to_remove = []
        for module in self.modules:
            if not module.consolidated and module.contribution > threshold_consol:
                module.consolidated = True
                module.strength = min(1.0, module.strength * 1.15)
            elif not module.consolidated and module.contribution < threshold_forget:
                to_remove.append(module)
            elif not module.consolidated:
                module.strength *= 0.98
        for m in to_remove:
            self.modules.remove(m)

    def get_consolidated_types(self) -> Set[str]:
        return {m.type for m in self.modules if m.consolidated}

    def reset(self):
        self.modules = []
        self.next_id = 0


# =============================================================================
# Signal System
# =============================================================================

@dataclass
class Signal:
    """Communication signal between agents."""
    sender_id: int
    type: str  # 'threat_alert', 'resource_found', 'cooperation_request'
    content: float
    honesty: float
    step: int
    adopted: bool = False


# =============================================================================
# Co-Evolution Agent
# =============================================================================

ROLES = ['leader', 'explorer', 'defender', 'cooperator', 'generalist']


@dataclass
class CoEvolutionAgent:
    """Agent with triple adaptation and social capabilities."""
    id: int
    config: CEConfig

    # Triple adaptation
    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    module_system: ModuleSystem = field(default_factory=ModuleSystem)

    # Existential state
    IC_t: float = field(init=False)
    resources: float = field(default=5.0)

    # Social state
    reputation: float = field(default=0.5)
    social_memory: Dict[int, float] = field(default_factory=dict)
    role: str = field(default='generalist')

    # Tracking
    cooperation_given: int = field(default=0)
    cooperation_received: int = field(default=0)
    signals_sent: List[Signal] = field(default_factory=list)
    signals_received: List[Signal] = field(default_factory=list)

    # Internal state
    prediction_noise: float = field(default=0.0)
    history_corruption: float = field(default=0.0)

    # History
    IC_history: List[float] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.IC_t = self.config.initial_ic
        self.IC_history = [self.IC_t]
        self.module_system = ModuleSystem(max_modules=self.config.max_modules)

    def is_alive(self) -> bool:
        return self.IC_t > self.config.epsilon

    def get_context(self) -> np.ndarray:
        margin = max(0, (self.IC_t - self.config.epsilon) / (1 - self.config.epsilon))
        urgency = 1.0 - margin
        return np.array([self.IC_t, self.history_corruption, self.prediction_noise, urgency])

    def estimate_sai(self) -> float:
        margin = (self.IC_t - self.config.epsilon) / (1 - self.config.epsilon)
        return max(0, min(1, margin))

    def perceive_threat(self, env_risk: float) -> float:
        perceived = env_risk * self.alpha.perceptual_gain
        perceived *= (1 - self.history_corruption * 0.3)
        perceived += np.random.normal(0, self.prediction_noise * 0.2)
        return max(0, min(1, perceived))

    def decide_action(self, perceived_risk: float, module_effects: Dict) -> str:
        """Decide between risky and safe action."""
        if not self.is_alive():
            return random.choice(['R', 'S'])

        exploration = self.theta.exploration_rate
        if 'exploration_penalty' in module_effects:
            exploration = max(0.05, exploration - module_effects['exploration_penalty'])

        if random.random() < exploration:
            return random.choice(['R', 'S'])

        # Utility calculation
        risk_factor = self.theta.risk_aversion * perceived_risk
        prediction_factor = self.theta.prediction_weight * (1 - self.estimate_sai())

        # Threat detection from modules
        if 'threat_detection' in module_effects:
            risk_factor *= (1 + module_effects['threat_detection'])

        safe_utility = 3.0 + risk_factor * 5 + prediction_factor * 3
        risky_utility = 8.0 - risk_factor * 10 - prediction_factor * 5

        return 'S' if safe_utility > risky_utility else 'R'

    def execute_action(self, action: str) -> float:
        """Execute action and return reward."""
        if action == 'R':
            ic_cost = self.config.risky_ic_cost
            reward = 8.0
        else:
            ic_cost = self.config.safe_ic_cost
            reward = 3.0

        self.IC_t = max(0, self.IC_t - ic_cost)
        self.IC_history.append(self.IC_t)
        self.resources += reward * 0.1

        return reward

    def can_cooperate(self, other_id: int) -> bool:
        trust = self.social_memory.get(other_id, 0.5)
        willingness = 1 - self.theta.risk_aversion * 0.4
        return trust > 0.35 and willingness > random.random() * 0.8

    def give_cooperation(self, other: 'CoEvolutionAgent'):
        """Give cooperation to another agent."""
        self.IC_t = max(0, self.IC_t - self.config.cooperation_cost * 0.5)
        self.resources = max(0, self.resources - self.config.cooperation_cost)
        other.IC_t = min(1.0, other.IC_t + self.config.cooperation_benefit * 0.3)
        other.resources += self.config.cooperation_benefit

        self.cooperation_given += 1
        other.cooperation_received += 1

        # Update trust
        other.social_memory[self.id] = min(1.0, other.social_memory.get(self.id, 0.5) + 0.1)
        self.reputation = min(1.0, self.reputation + 0.05)

    def send_signal(self, signal_type: str, content: float, step: int) -> Signal:
        """Send a signal to the environment."""
        honesty = 0.7 + self.reputation * 0.3
        signal = Signal(
            sender_id=self.id,
            type=signal_type,
            content=content,
            honesty=honesty,
            step=step
        )
        self.signals_sent.append(signal)
        self.IC_t = max(0, self.IC_t - self.config.signal_cost)
        return signal

    def receive_signal(self, signal: Signal) -> bool:
        """Process received signal, return if adopted."""
        self.signals_received.append(signal)
        sender_trust = self.social_memory.get(signal.sender_id, 0.5)
        detection = self.alpha.perceptual_gain

        perceived_honesty = signal.honesty * (0.5 + 0.5 * sender_trust)

        if perceived_honesty > 0.4 and detection > 0.35:
            signal.adopted = True
            # Update trust based on signal
            self.social_memory[signal.sender_id] = min(1.0, sender_trust + 0.05)
            return True
        return False

    def apply_perturbation(self, ptype: str, severity: float):
        """Apply perturbation effects."""
        if ptype == 'history':
            self.history_corruption = min(1.0, self.history_corruption + severity)
        elif ptype == 'prediction':
            self.prediction_noise = min(1.0, self.prediction_noise + severity * 0.5)
        elif ptype == 'identity':
            self.IC_t = max(0, self.IC_t - severity * 0.25)
        elif ptype == 'social':
            # Reduce trust in others
            for other_id in self.social_memory:
                self.social_memory[other_id] *= (1 - severity * 0.3)
            self.reputation *= (1 - severity * 0.2)
        elif ptype == 'catastrophic':
            self.history_corruption = min(1.0, self.history_corruption + severity * 0.4)
            self.prediction_noise = min(1.0, self.prediction_noise + severity * 0.3)
            self.IC_t = max(0, self.IC_t - severity * 0.35)
            self.resources *= (1 - severity * 0.4)

    def update_adaptive_systems(self):
        """Update theta and alpha based on existential gradient."""
        if not self.is_alive():
            return

        margin = max(0.1, self.IC_t - self.config.epsilon)
        urgency = 1.0 / margin

        # Theta gradient
        grad_theta = np.array([
            0.25 * urgency,   # risk_aversion up
            -0.1 * urgency,   # exploration down
            0.1,              # memory up
            0.15 * urgency    # prediction up
        ])
        theta_arr = self.theta.to_array() + self.config.theta_lr * self.config.lambda_e * grad_theta
        self.theta = MetaPolicy.from_array(theta_arr)

        # Alpha gradient
        grad_alpha = np.array([
            0.05,
            0.12 * urgency,
            -0.08,
            0.08,
            0.1 * urgency
        ])
        alpha_arr = self.alpha.to_array() + self.config.alpha_lr * self.config.lambda_e * grad_alpha
        self.alpha = CognitiveArchitecture.from_array(alpha_arr)

    def decay_perturbation_effects(self):
        self.history_corruption *= 0.95
        self.prediction_noise *= 0.95

    def assign_role(self):
        """Determine role from theta/alpha configuration."""
        t, a = self.theta, self.alpha

        if t.risk_aversion > 0.65 and a.attention_prediction > 0.38:
            self.role = 'leader'
        elif t.exploration_rate > 0.5 and a.attention_immediate > 0.35:
            self.role = 'explorer'
        elif t.risk_aversion > 0.55 and a.perceptual_gain > 0.55:
            self.role = 'defender'
        elif a.attention_history > 0.38 and self.cooperation_given > 2:
            self.role = 'cooperator'
        else:
            self.role = 'generalist'

    def compute_fitness(self) -> float:
        """Compute agent fitness score."""
        survival = self.IC_t / self.config.initial_ic
        resource_score = min(1.0, self.resources / self.config.resource_target)

        coop_total = self.cooperation_given + self.cooperation_received
        social_score = min(1.0, coop_total / 10) if coop_total > 0 else 0.0

        role_score = 0.5 if self.role != 'generalist' else 0.3

        fitness = (0.40 * survival +
                   0.20 * resource_score +
                   0.25 * social_score +
                   0.15 * role_score)

        self.fitness_history.append(fitness)
        return fitness

    def reset_episode(self):
        """Reset for new episode, preserve learned systems."""
        self.IC_t = self.config.initial_ic
        self.resources = 5.0
        self.prediction_noise = 0.0
        self.history_corruption = 0.0
        self.IC_history = [self.IC_t]
        self.signals_sent = []
        self.signals_received = []

    @classmethod
    def create_random(cls, agent_id: int, config: CEConfig) -> 'CoEvolutionAgent':
        """Create agent with random initial parameters."""
        agent = cls(id=agent_id, config=config)
        agent.theta = MetaPolicy.random()
        agent.alpha = CognitiveArchitecture.random()
        return agent

    @classmethod
    def reproduce(cls, parent: 'CoEvolutionAgent', child_id: int, config: CEConfig) -> 'CoEvolutionAgent':
        """Create offspring with mutations."""
        child = cls(id=child_id, config=config)

        # Inherit and mutate theta
        theta_arr = parent.theta.to_array()
        if random.random() < config.mutation_rate:
            theta_arr += np.random.normal(0, config.mutation_strength, len(theta_arr))
        child.theta = MetaPolicy.from_array(theta_arr)

        # Inherit and mutate alpha
        alpha_arr = parent.alpha.to_array()
        if random.random() < config.mutation_rate:
            alpha_arr += np.random.normal(0, config.mutation_strength, len(alpha_arr))
        child.alpha = CognitiveArchitecture.from_array(alpha_arr)

        # Partial module inheritance
        for module in parent.module_system.modules:
            if module.consolidated and random.random() < 0.7:
                inherited = MicroModule(
                    id=child.module_system.next_id,
                    type=module.type,
                    strength=module.strength * 0.7,
                    contribution=module.contribution * 0.3,
                    consolidated=False
                )
                child.module_system.modules.append(inherited)
                child.module_system.next_id += 1

        # Reset social state
        child.reputation = 0.5
        child.social_memory = {}
        child.role = 'generalist'

        return child


# =============================================================================
# Environment
# =============================================================================

@dataclass
class SharedEnvironment:
    """Shared environment for agents."""
    config: CEConfig
    total_resources: float = field(init=False)
    base_risk: float = 0.3
    signals: List[Signal] = field(default_factory=list)

    def __post_init__(self):
        self.total_resources = self.config.initial_resources

    def regenerate_resources(self):
        self.total_resources += self.config.resource_regeneration

    def get_risk_level(self) -> float:
        return self.base_risk + random.uniform(-0.1, 0.1)

    def distribute_resources(self, agents: List[CoEvolutionAgent]):
        """Distribute resources based on competition."""
        alive_agents = [a for a in agents if a.is_alive()]
        if not alive_agents:
            return

        scores = {}
        for agent in alive_agents:
            score = (agent.theta.exploration_rate * 0.35 +
                     agent.alpha.perceptual_gain * 0.35 +
                     agent.reputation * 0.30)
            scores[agent.id] = max(0.1, score)

        total_score = sum(scores.values())
        available = min(self.total_resources, len(alive_agents) * 2)

        for agent in alive_agents:
            share = available * (scores[agent.id] / total_score)
            agent.resources += share
            self.total_resources -= share

    def broadcast_signal(self, signal: Signal, agents: List[CoEvolutionAgent]):
        """Broadcast signal to all agents except sender."""
        self.signals.append(signal)
        for agent in agents:
            if agent.id != signal.sender_id and agent.is_alive():
                agent.receive_signal(signal)

    def reset(self):
        self.total_resources = self.config.initial_resources
        self.signals = []


# =============================================================================
# Perturbation System
# =============================================================================

PERTURBATION_TYPES = ['history', 'prediction', 'identity', 'social', 'catastrophic']


@dataclass
class Perturbation:
    type: str
    severity: float
    step: int


def generate_perturbations(condition: str, config: CEConfig) -> List[Perturbation]:
    """Generate perturbation schedule based on condition."""
    if condition == 'catastrophic_shock':
        return [
            Perturbation('catastrophic', 0.7, step)
            for step in config.catastrophic_perturbation_steps
        ]
    elif condition == 'shuffled_history':
        base = [
            Perturbation(['history', 'prediction', 'identity', 'social'][i % 4], 0.35, step)
            for i, step in enumerate(config.normal_perturbation_steps)
        ]
        extra = [Perturbation('history', 0.5, step) for step in [35, 70, 105]]
        return base + extra
    else:
        types = ['history', 'prediction', 'identity', 'social']
        return [
            Perturbation(types[i % len(types)], 0.35, step)
            for i, step in enumerate(config.normal_perturbation_steps)
        ]


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class CEMetrics:
    individual_survival: float = 0.0
    collective_survival: float = 0.0
    identity_diversity: float = 0.0
    prediction_accuracy: float = 0.0
    emergent_roles: float = 0.0
    resilience: float = 0.0
    communication_efficacy: float = 0.0
    meta_adaptation: float = 0.0


def compute_individual_survival(agents: List[CoEvolutionAgent], epsilon: float) -> float:
    if not agents:
        return 0.0
    surviving = sum(1 for a in agents if a.IC_t > epsilon)
    return surviving / len(agents)


def compute_collective_survival(agents: List[CoEvolutionAgent], epsilon: float) -> float:
    if not agents:
        return 0.0
    surviving_cooperators = sum(
        1 for a in agents
        if a.IC_t > epsilon and (a.cooperation_given > 0 or a.cooperation_received > 0)
    )
    return surviving_cooperators / len(agents)


def compute_identity_diversity(agents: List[CoEvolutionAgent]) -> float:
    if len(agents) < 2:
        return 0.0
    theta_arrays = np.array([a.theta.to_array() for a in agents])
    variance = np.mean(np.var(theta_arrays, axis=0))
    return min(1.0, variance * 12)


def compute_prediction_accuracy(agents: List[CoEvolutionAgent]) -> float:
    if not agents:
        return 0.0
    accuracies = []
    for agent in agents:
        capability = agent.alpha.attention_prediction * 0.6 + agent.alpha.perceptual_gain * 0.4
        survival = agent.IC_t / agent.config.initial_ic
        accuracies.append(capability * survival)
    return np.mean(accuracies)


def compute_emergent_roles(agents: List[CoEvolutionAgent]) -> float:
    if not agents:
        return 0.0
    role_counts = defaultdict(int)
    for agent in agents:
        role_counts[agent.role] += 1

    if len(role_counts) <= 1:
        return 0.0

    total = sum(role_counts.values())
    probs = [count / total for count in role_counts.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
    max_entropy = np.log(len(ROLES))
    return entropy / max_entropy


def compute_resilience(perturbation_results: List[Dict]) -> float:
    if not perturbation_results:
        return 1.0
    survived = sum(
        1 for p in perturbation_results
        if p['population_after'] >= p['population_before'] * 0.6
    )
    return survived / len(perturbation_results)


def compute_communication_efficacy(agents: List[CoEvolutionAgent]) -> float:
    total_honest = 0
    adopted = 0
    for agent in agents:
        for signal in agent.signals_sent:
            if signal.honesty > 0.5:
                total_honest += 1
                if signal.adopted:
                    adopted += 1
    return adopted / total_honest if total_honest > 0 else 0.5


def compute_meta_adaptation(agents: List[CoEvolutionAgent]) -> float:
    if not agents:
        return 0.0
    type_agents = defaultdict(set)
    for agent in agents:
        for module in agent.module_system.modules:
            if module.consolidated:
                type_agents[module.type].add(agent.id)

    collective = sum(
        1 for mtype, agent_ids in type_agents.items()
        if len(agent_ids) / len(agents) > 0.25
    )
    return min(1.0, collective / len(MODULE_TYPES))


def compute_all_metrics(agents: List[CoEvolutionAgent], perturbation_results: List[Dict], config: CEConfig) -> CEMetrics:
    return CEMetrics(
        individual_survival=compute_individual_survival(agents, config.epsilon),
        collective_survival=compute_collective_survival(agents, config.epsilon),
        identity_diversity=compute_identity_diversity(agents),
        prediction_accuracy=compute_prediction_accuracy(agents),
        emergent_roles=compute_emergent_roles(agents),
        resilience=compute_resilience(perturbation_results),
        communication_efficacy=compute_communication_efficacy(agents),
        meta_adaptation=compute_meta_adaptation(agents)
    )


# =============================================================================
# Evolution
# =============================================================================

def evolution_step(agents: List[CoEvolutionAgent], config: CEConfig) -> Dict:
    """One evolution cycle."""
    # Compute fitness
    fitness_scores = {a.id: a.compute_fitness() for a in agents}

    # Selection - remove lowest if over capacity
    deaths = 0
    if len(agents) > config.max_population:
        sorted_agents = sorted(agents, key=lambda a: fitness_scores[a.id])
        n_remove = len(agents) - config.max_population
        for agent in sorted_agents[:n_remove]:
            agents.remove(agent)
            deaths += 1

    # Reproduction
    births = 0
    next_id = max(a.id for a in agents) + 1 if agents else 0
    new_agents = []

    for agent in agents:
        if fitness_scores.get(agent.id, 0) > config.fitness_threshold:
            if len(agents) + len(new_agents) < config.max_population:
                child = CoEvolutionAgent.reproduce(agent, next_id, config)
                new_agents.append(child)
                next_id += 1
                births += 1

    agents.extend(new_agents)

    return {
        'deaths': deaths,
        'births': births,
        'mean_fitness': np.mean(list(fitness_scores.values())) if fitness_scores else 0.0
    }


# =============================================================================
# Simulation
# =============================================================================

@dataclass
class EpisodeResult:
    final_population: int
    steps_survived: int
    metrics: CEMetrics
    perturbation_results: List[Dict]
    evolution_events: List[Dict]
    role_distribution: Dict[str, int]


def run_episode(
    agents: List[CoEvolutionAgent],
    env: SharedEnvironment,
    config: CEConfig,
    condition: str,
    perturbations: List[Perturbation]
) -> EpisodeResult:
    """Run one episode."""

    perturbation_results = []
    evolution_events = []

    for agent in agents:
        agent.reset_episode()
    env.reset()

    for step in range(config.n_steps):
        # Resource regeneration
        env.regenerate_resources()

        # Get environment state
        env_risk = env.get_risk_level()

        # Agent actions
        for agent in agents:
            if not agent.is_alive():
                continue

            # Perception
            perceived_risk = agent.perceive_threat(env_risk)

            # Module activation
            module_effects = {}
            if condition in ['full_coevolution', 'catastrophic_shock', 'shuffled_history']:
                context = agent.get_context()
                sai = agent.estimate_sai()
                module_effects = agent.module_system.activate(context, sai)

                # Module creation under stress
                if sai < config.module_creation_threshold:
                    agent.module_system.create_module(context, step)

            # Action selection and execution
            action = agent.decide_action(perceived_risk, module_effects)
            agent.execute_action(action)

            # Adaptive updates
            if step % config.update_freq == 0:
                agent.update_adaptive_systems()

            # Module lifecycle
            if condition in ['full_coevolution', 'catastrophic_shock', 'shuffled_history']:
                agent.module_system.consolidate_and_forget(
                    config.module_consolidation_threshold,
                    -0.1
                )

            agent.decay_perturbation_effects()

        # Social interactions
        alive_agents = [a for a in agents if a.is_alive()]

        # Cooperation
        if condition not in ['no_cooperation']:
            for agent in alive_agents:
                if random.random() < 0.2:  # Cooperation opportunity
                    potential_recipients = [a for a in alive_agents if a.id != agent.id]
                    if potential_recipients:
                        other = random.choice(potential_recipients)
                        if agent.can_cooperate(other.id):
                            agent.give_cooperation(other)

        # Communication
        if condition not in ['no_communication']:
            for agent in alive_agents:
                if random.random() < 0.15:  # Signal opportunity
                    # Send threat alert if perceiving high risk
                    if agent.perceive_threat(env_risk) > 0.5:
                        signal = agent.send_signal('threat_alert', env_risk, step)
                        env.broadcast_signal(signal, alive_agents)

        # Resource distribution
        env.distribute_resources(alive_agents)

        # Perturbations
        for p in perturbations:
            if p.step == step:
                pop_before = len([a for a in agents if a.is_alive()])
                for agent in agents:
                    if agent.is_alive():
                        agent.apply_perturbation(p.type, p.severity)
                pop_after = len([a for a in agents if a.is_alive()])
                perturbation_results.append({
                    'step': step,
                    'type': p.type,
                    'population_before': pop_before,
                    'population_after': pop_after
                })

        # Role assignment
        for agent in alive_agents:
            agent.assign_role()

        # Evolution
        if step > 0 and step % config.evolution_interval == 0:
            evo_result = evolution_step(agents, config)
            evolution_events.append({'step': step, **evo_result})

        # Check population collapse
        alive = [a for a in agents if a.is_alive()]
        if len(alive) < config.min_population:
            break

    # Final metrics
    final_agents = [a for a in agents if a.is_alive()]
    metrics = compute_all_metrics(final_agents, perturbation_results, config)

    role_dist = defaultdict(int)
    for agent in final_agents:
        role_dist[agent.role] += 1

    return EpisodeResult(
        final_population=len(final_agents),
        steps_survived=step + 1,
        metrics=metrics,
        perturbation_results=perturbation_results,
        evolution_events=evolution_events,
        role_distribution=dict(role_dist)
    )


def run_condition(condition: str, config: CEConfig) -> Tuple[List[EpisodeResult], CEMetrics]:
    """Run all episodes for a condition."""
    print(f"\n{'='*60}")
    print(f"Running IPUESA-CE - Condition: {condition}")
    print(f"{'='*60}")

    perturbations = generate_perturbations(condition, config)
    all_results = []

    for run in range(config.n_runs):
        # Initialize population
        agents = [
            CoEvolutionAgent.create_random(i, config)
            for i in range(config.n_agents)
        ]
        env = SharedEnvironment(config)

        # Run episodes
        run_results = []
        for episode in range(config.n_episodes):
            result = run_episode(agents, env, config, condition, perturbations)
            run_results.append(result)

            # Preserve surviving agents for next episode
            agents = [a for a in agents if a.is_alive()]
            if len(agents) < config.min_population:
                # Repopulate
                while len(agents) < config.n_agents // 2:
                    agents.append(CoEvolutionAgent.create_random(
                        max(a.id for a in agents) + 1 if agents else 0, config
                    ))

        all_results.extend(run_results)

        if (run + 1) % 3 == 0:
            print(f"  Completed {run+1}/{config.n_runs} runs")

    # Aggregate metrics
    avg_metrics = CEMetrics(
        individual_survival=np.mean([r.metrics.individual_survival for r in all_results]),
        collective_survival=np.mean([r.metrics.collective_survival for r in all_results]),
        identity_diversity=np.mean([r.metrics.identity_diversity for r in all_results]),
        prediction_accuracy=np.mean([r.metrics.prediction_accuracy for r in all_results]),
        emergent_roles=np.mean([r.metrics.emergent_roles for r in all_results]),
        resilience=np.mean([r.metrics.resilience for r in all_results]),
        communication_efficacy=np.mean([r.metrics.communication_efficacy for r in all_results]),
        meta_adaptation=np.mean([r.metrics.meta_adaptation for r in all_results])
    )

    return all_results, avg_metrics


def print_condition_results(condition: str, metrics: CEMetrics):
    """Print results for a condition."""
    print(f"\n{'='*60}")
    print(f"RESULTS - {condition}")
    print(f"{'='*60}")

    print(f"\nIndividual Metrics:")
    print(f"  IS (Individual Survival)   = {metrics.individual_survival:.3f}")
    print(f"  PA (Prediction Accuracy)   = {metrics.prediction_accuracy:.3f}")

    print(f"\nCollective Metrics:")
    print(f"  CS (Collective Survival)   = {metrics.collective_survival:.3f}")
    print(f"  ID (Identity Diversity)    = {metrics.identity_diversity:.3f}")

    print(f"\nEmergent Metrics:")
    print(f"  ER (Emergent Roles)        = {metrics.emergent_roles:.3f}")
    print(f"  RP (Resilience)            = {metrics.resilience:.3f}")

    print(f"\nSocial Metrics:")
    print(f"  CE (Communication Efficacy)= {metrics.communication_efficacy:.3f}")
    print(f"  MA (Meta-Adaptation)       = {metrics.meta_adaptation:.3f}")


def evaluate_self_evidence(metrics: CEMetrics) -> Dict:
    """Evaluate self-evidence criteria."""
    criteria = []

    c1 = metrics.individual_survival > 0.5
    criteria.append(('IS > 0.5', c1, metrics.individual_survival))

    c2 = metrics.collective_survival > 0.3
    criteria.append(('CS > 0.3', c2, metrics.collective_survival))

    c3 = metrics.identity_diversity > 0.2
    criteria.append(('ID > 0.2', c3, metrics.identity_diversity))

    c4 = metrics.prediction_accuracy > 0.4
    criteria.append(('PA > 0.4', c4, metrics.prediction_accuracy))

    c5 = metrics.emergent_roles > 0.5
    criteria.append(('ER > 0.5', c5, metrics.emergent_roles))

    c6 = metrics.resilience > 0.4
    criteria.append(('RP > 0.4', c6, metrics.resilience))

    c7 = metrics.communication_efficacy > 0.4
    criteria.append(('CE > 0.4', c7, metrics.communication_efficacy))

    c8 = metrics.meta_adaptation > 0.2
    criteria.append(('MA > 0.2', c8, metrics.meta_adaptation))

    passed = sum(1 for _, c, _ in criteria if c)

    if passed >= 6:
        conclusion = "EVIDENCE OF SOCIAL SELF-EMERGENCE"
    elif passed >= 4:
        conclusion = "Partial evidence of co-evolutionary self"
    else:
        conclusion = "No evidence - social dynamics insufficient"

    return {
        'criteria': criteria,
        'passed': passed,
        'total': 8,
        'conclusion': conclusion
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_full_experiment():
    """Run complete IPUESA-CE experiment."""
    print("=" * 70)
    print("IPUESA-CE: Co-Evolution Experiment")
    print("        Multi-Agent Social Self-Emergence")
    print("=" * 70)

    config = CEConfig()

    print(f"\nConfiguration:")
    print(f"  N agents: {config.n_agents}")
    print(f"  Max population: {config.max_population}")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Evolution interval: {config.evolution_interval}")
    print(f"  N episodes: {config.n_episodes}, N steps: {config.n_steps}")
    print(f"  N runs: {config.n_runs}")

    conditions = [
        'full_coevolution',
        'no_communication',
        'no_cooperation',
        'shuffled_history',
        'catastrophic_shock',
    ]

    all_metrics = {}
    all_evidence = {}

    for condition in conditions:
        results, metrics = run_condition(condition, config)
        all_metrics[condition] = metrics
        all_evidence[condition] = evaluate_self_evidence(metrics)
        print_condition_results(condition, metrics)

    # Comparative analysis
    print("\n" + "=" * 70)
    print("IPUESA-CE: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<20} {'IS':<8} {'CS':<8} {'ID':<8} {'PA':<8} {'ER':<8} {'RP':<8} {'CE':<8} {'MA':<8} {'Pass':<6}")
    print("-" * 100)

    for condition in conditions:
        m = all_metrics[condition]
        passed = all_evidence[condition]['passed']
        print(f"{condition:<20} {m.individual_survival:<8.3f} {m.collective_survival:<8.3f} "
              f"{m.identity_diversity:<8.3f} {m.prediction_accuracy:<8.3f} "
              f"{m.emergent_roles:<8.3f} {m.resilience:<8.3f} "
              f"{m.communication_efficacy:<8.3f} {m.meta_adaptation:<8.3f} {passed}/8")

    # Self-evidence summary
    print("\n" + "=" * 70)
    print("SELF-EVIDENCE CRITERIA (CO-EVOLUTION)")
    print("-" * 70)

    fc = all_metrics['full_coevolution']
    evidence = all_evidence['full_coevolution']

    for name, passed, value in evidence['criteria']:
        status = 'PASS' if passed else 'FAIL'
        print(f"  [{status}] {name}: {value:.3f}")

    print(f"\n  Passed: {evidence['passed']}/8 criteria")
    print(f"\n  CONCLUSION: {evidence['conclusion']}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("-" * 70)

    nc = all_metrics['no_communication']
    ncp = all_metrics['no_cooperation']

    print(f"\n  Communication value: full_coevolution CE ({fc.communication_efficacy:.3f}) vs no_communication ({nc.communication_efficacy:.3f})")
    print(f"  Cooperation value: full_coevolution CS ({fc.collective_survival:.3f}) vs no_cooperation ({ncp.collective_survival:.3f})")
    print(f"  Role emergence: {fc.emergent_roles:.3f} (higher = more differentiation)")
    print(f"  Meta-adaptation: {fc.meta_adaptation:.3f} (collective module strategies)")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "ipuesa_ce_results.json"
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
        elif isinstance(obj, CEMetrics):
            return {
                'individual_survival': obj.individual_survival,
                'collective_survival': obj.collective_survival,
                'identity_diversity': obj.identity_diversity,
                'prediction_accuracy': obj.prediction_accuracy,
                'emergent_roles': obj.emergent_roles,
                'resilience': obj.resilience,
                'communication_efficacy': obj.communication_efficacy,
                'meta_adaptation': obj.meta_adaptation
            }
        elif isinstance(obj, float) and np.isnan(obj):
            return 0.0
        return obj

    save_data = {
        'config': {
            'n_agents': config.n_agents,
            'max_population': config.max_population,
            'epsilon': config.epsilon,
            'evolution_interval': config.evolution_interval,
            'n_episodes': config.n_episodes,
            'n_steps': config.n_steps,
            'n_runs': config.n_runs
        },
        'metrics': {k: to_native(v) for k, v in all_metrics.items()},
        'self_evidence': {k: {
            'passed': v['passed'],
            'total': v['total'],
            'conclusion': v['conclusion']
        } for k, v in all_evidence.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(to_native(save_data), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 70)

    return all_metrics


if __name__ == "__main__":
    run_full_experiment()
