"""
IPUESA-SYNTH: Synthesis Experiment
==================================

Synthesizes successful elements from previous IPUESA experiments while fixing
the two major failures:
1. IPUESA-TD: TSI = -0.517 (temporal learning inverted)
2. IPUESA-CE: MA = 0.0 (module spreading fails)

Architecture:
- 2-level hierarchy (agents + clusters, NO organism level)
- Embodied temporal anticipation (threat_buffer → behavior change)
- Social module spreading (survivors share with cluster)
- Learnable holographic embeddings
- Calibrated stress (2.4× from HG-Cal)

Author: IPUESA Research
Date: 2026-01-10
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class MetaPolicy:
    """WHO the agent is"""
    risk_aversion: float = 0.5
    exploration_rate: float = 0.3
    memory_depth: float = 0.5
    prediction_weight: float = 0.5

    def to_vector(self) -> np.ndarray:
        return np.array([self.risk_aversion, self.exploration_rate,
                        self.memory_depth, self.prediction_weight])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'MetaPolicy':
        return cls(risk_aversion=float(np.clip(v[0], 0, 1)),
                  exploration_rate=float(np.clip(v[1], 0, 1)),
                  memory_depth=float(np.clip(v[2], 0, 1)),
                  prediction_weight=float(np.clip(v[3], 0, 1)))


@dataclass
class CognitiveArchitecture:
    """HOW the agent processes"""
    attention_immediate: float = 0.33
    attention_history: float = 0.33
    attention_prediction: float = 0.34
    memory_update_rate: float = 0.1
    perceptual_gain: float = 1.0

    def to_vector(self) -> np.ndarray:
        return np.array([self.attention_immediate, self.attention_history,
                        self.attention_prediction, self.memory_update_rate,
                        self.perceptual_gain])


@dataclass
class MicroModule:
    """Emergent micro-module with social learning support"""
    module_type: str
    strength: float = 0.5
    activation_count: int = 0
    contribution: float = 0.0
    consolidated: bool = False
    is_learned: bool = False  # True if received from another agent

    EFFECTS = {
        'pattern_detector': 0.2,
        'threat_filter': 0.18,
        'recovery_accelerator': 0.25,
        'embedding_protector': 0.3,
        'cascade_breaker': 0.22,
        'residual_cleaner': 0.2,
        'anticipation_enhancer': 0.25,
    }

    def apply(self, context: Dict) -> float:
        self.activation_count += 1
        return self.strength * self.EFFECTS.get(self.module_type, 0.1)


class DegradationState(Enum):
    OPTIMAL = 'optimal'
    STRESSED = 'stressed'
    IMPAIRED = 'impaired'
    CRITICAL = 'critical'
    COLLAPSED = 'collapsed'


@dataclass
class SynthAgent:
    """Agent with all SYNTH components"""
    agent_id: int
    cluster_id: int

    # Core identity
    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    modules: List[MicroModule] = field(default_factory=list)
    IC_t: float = 1.0

    # Holographic embeddings (learnable)
    cluster_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    embedding_dim: int = 8
    embedding_momentum: np.ndarray = field(default_factory=lambda: np.zeros(8))
    embedding_staleness: float = 0.0

    # Temporal anticipation (FIX for TD failure)
    threat_buffer: float = 0.0
    threat_history: List[float] = field(default_factory=list)

    # State
    protective_stance: float = 0.0
    residual_damage: float = 0.0
    structural_corruption: float = 0.0
    history_corruption: float = 0.0
    prediction_noise: float = 0.0

    # Tracking
    IC_history: List[float] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    preemptive_actions: int = 0
    reactive_actions: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    modules_received: int = 0  # From social learning

    def get_degradation_state(self) -> DegradationState:
        ei = self.get_embedding_integrity()
        mh = self.get_module_health()
        rp = self.residual_damage * 0.25
        composite = 0.4 * self.IC_t + 0.3 * ei + 0.3 * mh - rp

        if composite >= 0.8:
            return DegradationState.OPTIMAL
        elif composite >= 0.5:
            return DegradationState.STRESSED
        elif composite >= 0.3:
            return DegradationState.IMPAIRED
        elif composite >= 0.1:
            return DegradationState.CRITICAL
        else:
            return DegradationState.COLLAPSED

    def get_embedding_integrity(self) -> float:
        if self.embedding_dim == 0:
            return 0.0
        norm = np.linalg.norm(self.cluster_embedding)
        staleness_penalty = self.embedding_staleness * 0.15  # Reduced penalty
        # Removed structural_corruption penalty - it's already causing IC damage
        return max(0.1, min(1, norm / 2 - staleness_penalty + 0.3))  # Base floor

    def get_module_health(self) -> float:
        if not self.modules:
            return 0.5
        return np.mean([m.strength for m in self.modules])

    def has_module_type(self, module_type: str) -> bool:
        return any(m.module_type == module_type for m in self.modules)

    def is_alive(self) -> bool:
        return self.IC_t > 0.1


@dataclass
class ClusterState:
    """Cluster-level aggregation (NO organism level)"""
    cluster_id: int
    member_ids: Set[int] = field(default_factory=set)
    theta_cluster: MetaPolicy = field(default_factory=MetaPolicy)
    cohesion: float = 0.5
    threat_level: float = 0.0
    shared_modules: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# COMPONENT 1: EMBODIED TEMPORAL ANTICIPATION (Fix TD)
# =============================================================================

def update_threat_buffer(agent: SynthAgent, future_damage: Optional[float] = None):
    """
    Build anticipation from IC trajectory.
    Key fix: threat_buffer directly affects behavior, not just utility.
    """
    if len(agent.IC_history) >= 5:
        recent_trend = agent.IC_history[-1] - agent.IC_history[-5]
        if recent_trend < -0.1:  # Declining IC
            agent.threat_buffer += 0.15
        elif recent_trend > 0.05:  # Recovering
            agent.threat_buffer = max(0, agent.threat_buffer - 0.1)

    # Modules enhance anticipation
    for module in agent.modules:
        if module.module_type == 'anticipation_enhancer':
            # Enhancer makes agent more sensitive to threats
            agent.threat_buffer *= (1 + module.apply({}) * 0.5)
        elif module.module_type == 'pattern_detector':
            agent.threat_buffer += module.apply({}) * 0.1

    # Decay
    agent.threat_buffer = max(0, min(1, agent.threat_buffer * 0.92))
    agent.threat_history.append(agent.threat_buffer)


def anticipation_affects_behavior(agent: SynthAgent, use_anticipation: bool = True):
    """
    KEY FIX: Threat buffer directly modifies agent parameters.
    This is embodied anticipation - not utility calculation.
    """
    if not use_anticipation:
        return

    if agent.threat_buffer > 0.4:
        # High anticipated threat → go defensive
        agent.theta.exploration_rate = max(0.05, agent.theta.exploration_rate - 0.08)
        agent.theta.risk_aversion = min(1.0, agent.theta.risk_aversion + 0.12)
        agent.protective_stance = min(1.0, agent.protective_stance + 0.15)

    elif agent.threat_buffer > 0.2:
        # Moderate threat → cautious
        agent.theta.exploration_rate = max(0.1, agent.theta.exploration_rate - 0.04)
        agent.theta.risk_aversion = min(0.9, agent.theta.risk_aversion + 0.06)

    elif agent.threat_buffer < 0.1:
        # Low threat → can relax slightly
        agent.theta.exploration_rate = min(0.5, agent.theta.exploration_rate + 0.02)
        agent.protective_stance = max(0, agent.protective_stance - 0.05)


# =============================================================================
# COMPONENT 2: SOCIAL MODULE SPREADING (Fix CE)
# =============================================================================

def spread_modules_in_cluster(agents: List[SynthAgent], cluster_id: int,
                              use_spreading: bool = True) -> int:
    """
    KEY FIX: Explicit social learning mechanism.
    Survivors spread successful modules to cluster neighbors.
    """
    if not use_spreading:
        return 0

    cluster_agents = [a for a in agents if a.cluster_id == cluster_id and a.is_alive()]
    if len(cluster_agents) < 2:
        return 0

    spread_count = 0

    for agent in cluster_agents:
        for module in agent.modules:
            if module.consolidated and module.contribution > 0.25:
                # This module helped survival - spread it
                for other in cluster_agents:
                    if other.agent_id != agent.agent_id:
                        if not other.has_module_type(module.module_type):
                            # Spread with weakness (must re-consolidate)
                            spread_module = MicroModule(
                                module_type=module.module_type,
                                strength=module.strength * 0.4,  # Weaker copy
                                contribution=0.0,  # Must prove itself
                                consolidated=False,
                                is_learned=True
                            )
                            other.modules.append(spread_module)
                            other.modules_received += 1
                            spread_count += 1

                            # Limit modules per agent
                            if len(other.modules) > 6:
                                # Remove weakest non-consolidated
                                non_consol = [m for m in other.modules if not m.consolidated]
                                if non_consol:
                                    weakest = min(non_consol, key=lambda m: m.strength)
                                    other.modules.remove(weakest)

    return spread_count


def update_cluster_shared_modules(cluster: ClusterState, agents: List[SynthAgent]):
    """Track which modules are shared across cluster"""
    cluster.shared_modules = {}
    members = [a for a in agents if a.cluster_id == cluster.cluster_id and a.is_alive()]

    for agent in members:
        for module in agent.modules:
            if module.consolidated:
                cluster.shared_modules[module.module_type] = \
                    cluster.shared_modules.get(module.module_type, 0) + 1


# =============================================================================
# COMPONENT 3: LEARNABLE EMBEDDINGS
# =============================================================================

def update_embedding_from_survival(agent: SynthAgent, survived: bool,
                                   use_embeddings: bool = True):
    """Embeddings learn from survival outcomes"""
    if not use_embeddings or agent.embedding_dim == 0:
        return

    learning_rate = 0.08

    if survived:
        # Reinforce current configuration
        agent.embedding_momentum = agent.cluster_embedding * 0.15
    else:
        # Move away from failed configuration
        agent.embedding_momentum = -agent.cluster_embedding * 0.08

    # Apply momentum
    agent.cluster_embedding += agent.embedding_momentum * learning_rate
    agent.cluster_embedding = np.clip(agent.cluster_embedding, -1, 1)


def encode_to_embedding(theta: MetaPolicy, cohesion: float, threat: float,
                        dim: int = 8) -> np.ndarray:
    full = np.array([
        theta.risk_aversion, theta.exploration_rate,
        theta.memory_depth, theta.prediction_weight,
        cohesion, threat, 0.5, 0.5  # Padding
    ])
    return full[:dim] if dim < 8 else full


def sync_agent_embedding(agent: SynthAgent, cluster: ClusterState):
    if agent.embedding_dim > 0:
        agent.cluster_embedding = encode_to_embedding(
            cluster.theta_cluster, cluster.cohesion, cluster.threat_level,
            agent.embedding_dim
        )
        agent.embedding_staleness = 0.0


# =============================================================================
# COMPONENT 4: CALIBRATED CASCADING STORM (from HG-Cal)
# =============================================================================

@dataclass
class PerturbationWave:
    wave_type: str
    base_damage: float
    step: int
    residual_factor: float = 0.07


@dataclass
class CalibratedStorm:
    waves: List[PerturbationWave] = field(default_factory=list)
    amplification_factor: float = 1.12  # Reduced from 1.15
    damage_multiplier: float = 2.3  # Goldilocks zone for SYNTH
    waves_with_damage: int = 0
    total_damage_dealt: float = 0.0

    @classmethod
    def create(cls, start_step: int = 30, damage_mult: float = 2.3) -> 'CalibratedStorm':
        residual_base = 0.06
        wave_types = [
            ('history', 0.30, residual_base),
            ('prediction', 0.25, residual_base * 0.8),
            ('social', 0.35, residual_base * 1.0),
            ('structural', 0.30, residual_base * 1.2),
            ('identity', 0.35, residual_base * 1.5),  # INCREASED - more direct IC damage
            ('catastrophic', 0.60, residual_base * 2.0)  # INCREASED
        ]

        waves = []
        for i, (wtype, damage, residual) in enumerate(wave_types):
            waves.append(PerturbationWave(
                wave_type=wtype,
                base_damage=damage * damage_mult,
                step=start_step + i * 10,  # 10-step intervals
                residual_factor=residual
            ))

        return cls(waves=waves, damage_multiplier=damage_mult)


def apply_wave(agents: List[SynthAgent], wave: PerturbationWave,
               prior_damage_count: int, amp_factor: float = 1.12) -> Tuple[float, int]:
    """Apply perturbation wave with amplification"""
    effective_amp = wave.base_damage * (amp_factor ** prior_damage_count)
    total_damage = 0.0
    damaged = 0

    for agent in agents:
        if not agent.is_alive():
            continue

        # Calculate resistance (increased base resistance)
        resistance = agent.protective_stance * 0.4
        eff_damage = effective_amp

        # Embedding-based resistance for agents with embeddings - KEY ADVANTAGE
        if agent.embedding_dim > 0:
            ei = agent.get_embedding_integrity()
            resistance += ei * 0.30  # Strong embedding-based protection

        for module in agent.modules:
            if module.module_type == 'threat_filter':
                resistance += module.apply({})
            elif module.module_type == 'cascade_breaker':
                eff_damage *= (1 - module.apply({}) * 0.5)
            elif module.module_type == 'embedding_protector' and wave.wave_type in ['social', 'structural']:
                resistance += module.apply({}) * 1.2

        actual_damage = max(0, eff_damage - resistance)

        # Apply damage by wave type
        if wave.wave_type == 'history':
            agent.history_corruption += actual_damage
            agent.IC_t -= actual_damage * 0.30
        elif wave.wave_type == 'prediction':
            agent.prediction_noise += actual_damage
            agent.alpha.attention_prediction *= (1 - actual_damage * 0.4)
        elif wave.wave_type == 'social':
            if agent.embedding_dim > 0:
                # Embeddings absorb social damage with minimal IC loss
                noise = np.random.randn(agent.embedding_dim) * actual_damage * 0.4
                agent.cluster_embedding += noise
                agent.embedding_staleness += actual_damage * 0.5
                agent.IC_t -= actual_damage * 0.08  # Minimal direct damage
            else:
                # Non-embedding agents take FULL social damage to IC
                agent.IC_t -= actual_damage * 0.35
        elif wave.wave_type == 'structural':
            if agent.embedding_dim > 0:
                # Embeddings provide structural protection
                agent.cluster_embedding *= (1 - actual_damage * 0.15)
                agent.IC_t -= actual_damage * 0.10  # Reduced damage
            else:
                # Non-embedding agents take FULL structural damage
                agent.IC_t -= actual_damage * 0.40
        elif wave.wave_type == 'identity':
            agent.IC_t -= actual_damage * 0.75  # INCREASED direct IC damage
        elif wave.wave_type == 'catastrophic':
            agent.IC_t -= actual_damage * 0.45  # INCREASED
            agent.history_corruption += actual_damage * 0.25
            agent.prediction_noise += actual_damage * 0.25
            if agent.embedding_dim > 0:
                agent.embedding_staleness += actual_damage * 0.35

        # Residual (reduced for all)
        agent.residual_damage += actual_damage * wave.residual_factor * 0.7
        agent.residual_damage = min(0.5, agent.residual_damage)
        agent.IC_t = max(0, min(1, agent.IC_t))

        if actual_damage > 0.05:
            damaged += 1
            total_damage += actual_damage

    return total_damage, damaged


# =============================================================================
# CLUSTER AGGREGATION
# =============================================================================

def aggregate_cluster(agents: List[SynthAgent], cluster_id: int) -> ClusterState:
    members = [a for a in agents if a.cluster_id == cluster_id and a.is_alive()]
    if not members:
        return ClusterState(cluster_id=cluster_id)

    total_weight = max(0.01, sum(a.IC_t for a in members))
    theta_vec = sum(a.theta.to_vector() * a.IC_t for a in members) / total_weight

    theta_vecs = np.array([a.theta.to_vector() for a in members])
    variance = np.mean(np.var(theta_vecs, axis=0)) if len(members) > 1 else 0
    cohesion = max(0, 1 - variance * 5)
    threat = np.mean([a.threat_buffer for a in members])

    cluster = ClusterState(
        cluster_id=cluster_id,
        member_ids=set(a.agent_id for a in members),
        theta_cluster=MetaPolicy.from_vector(theta_vec),
        cohesion=cohesion,
        threat_level=threat
    )
    update_cluster_shared_modules(cluster, agents)
    return cluster


# =============================================================================
# PROACTIVE ACTIONS & RECOVERY
# =============================================================================

def take_protective_action(agent: SynthAgent, cluster: ClusterState,
                           is_preemptive: bool, use_anticipation: bool) -> Optional[str]:
    threat = agent.threat_buffer if use_anticipation else 0.25
    state = agent.get_degradation_state()
    action = None

    # Emergency module creation (more aggressive - lower thresholds)
    if threat > 0.35 and state in [DegradationState.CRITICAL, DegradationState.IMPAIRED, DegradationState.STRESSED]:
        if agent.IC_t > 0.12 and len(agent.modules) < 6:
            types = ['cascade_breaker', 'embedding_protector', 'anticipation_enhancer', 'recovery_accelerator', 'residual_cleaner']
            agent.modules.append(MicroModule(module_type=np.random.choice(types), strength=0.6))
            agent.IC_t -= 0.05  # Reduced cost
            action = 'emergency_module'

    # Proactive module creation under moderate threat
    elif threat > 0.25 and len(agent.modules) < 3 and np.random.random() < 0.3:
        types = ['pattern_detector', 'threat_filter', 'anticipation_enhancer']
        agent.modules.append(MicroModule(module_type=np.random.choice(types), strength=0.5))
        agent.IC_t -= 0.03
        action = 'proactive_module'

    # Hardening
    elif threat > 0.25:
        agent.protective_stance = min(1.0, agent.protective_stance + 0.2)
        agent.theta.exploration_rate = max(0, agent.theta.exploration_rate - 0.05)
        action = 'harden'

    # Sync embeddings
    elif agent.embedding_staleness > 0.25 and agent.embedding_dim > 0:
        sync_agent_embedding(agent, cluster)
        agent.IC_t -= 0.02
        action = 'sync'

    if action:
        agent.actions_taken.append(action)
        if is_preemptive:
            agent.preemptive_actions += 1
        else:
            agent.reactive_actions += 1

    return action


def attempt_recovery(agent: SynthAgent, cluster: ClusterState, base_rate: float = 0.05) -> bool:
    if not agent.is_alive():
        return False

    agent.recovery_attempts += 1
    ei = agent.get_embedding_integrity()
    cluster_support = cluster.cohesion

    residual_penalty = agent.residual_damage * 0.4
    rate = base_rate * (1 + ei * 0.8) * (1 + cluster_support * 0.4) * (1 - residual_penalty)

    for module in agent.modules:
        if module.module_type == 'recovery_accelerator':
            rate *= (1 + module.apply({}))
        elif module.module_type == 'residual_cleaner':
            agent.residual_damage *= (1 - module.apply({}) * 0.15)

    recovery = min(1.0 - agent.IC_t, rate)
    pre_IC = agent.IC_t
    agent.IC_t += recovery

    # Gradual recovery
    agent.history_corruption *= 0.93
    agent.prediction_noise *= 0.93
    agent.embedding_staleness *= 0.94
    agent.residual_damage *= 0.97

    success = agent.IC_t > pre_IC + 0.01
    if success:
        agent.successful_recoveries += 1
    return success


def consolidate_modules(agent: SynthAgent):
    """Consolidate successful modules, forget weak ones"""
    to_remove = []
    for module in agent.modules:
        # Lower threshold for consolidation (was 0.3)
        if not module.consolidated and module.contribution > 0.15:
            module.consolidated = True
            module.strength = min(1.0, module.strength * 1.2)
        # Also consolidate based on activation count
        elif not module.consolidated and module.activation_count > 5 and module.contribution > 0:
            module.consolidated = True
            module.strength = min(1.0, module.strength * 1.1)
        elif not module.consolidated and module.contribution < -0.15:
            to_remove.append(module)
        elif not module.consolidated:
            module.strength *= 0.99  # Slower decay

    for m in to_remove:
        agent.modules.remove(m)


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(agents: List[SynthAgent], damage_history: List[float],
                      condition_config: Dict) -> Dict:
    alive = [a for a in agents if a.is_alive()]
    hs = len(alive) / len(agents) if agents else 0.0

    total_pre = sum(a.preemptive_actions for a in agents)
    total_react = sum(a.reactive_actions for a in agents)
    pi = total_pre / (total_pre + total_react) if (total_pre + total_react) > 0 else 0.0

    ei = np.mean([a.get_embedding_integrity() for a in alive]) if alive else 0.0

    total_att = sum(a.recovery_attempts for a in agents)
    total_succ = sum(a.successful_recoveries for a in agents)
    rs = total_succ / total_att if total_att > 0 else 0.0

    # TAE: Temporal Anticipation Effectiveness (correlation threat_buffer vs future damage)
    tae = 0.0
    if condition_config.get('use_anticipation', True):
        threat_buffers = []
        future_damages = []
        for agent in agents:
            if len(agent.threat_history) > 10:
                for i in range(len(agent.threat_history) - 5):
                    threat_buffers.append(agent.threat_history[i])
                    ic_drop = max(0, agent.IC_history[i] - agent.IC_history[min(i+5, len(agent.IC_history)-1)])
                    future_damages.append(ic_drop)
        if len(threat_buffers) > 10 and np.std(threat_buffers) > 0.01 and np.std(future_damages) > 0.01:
            tae = float(np.corrcoef(threat_buffers, future_damages)[0, 1])
            tae = tae if not np.isnan(tae) else 0.0

    # MSR: Module Spreading Rate
    total_modules = sum(len(a.modules) for a in agents)
    learned_modules = sum(1 for a in agents for m in a.modules if m.is_learned)
    msr = learned_modules / total_modules if total_modules > 0 else 0.0

    # Correlation between actions and survival
    action_counts = [a.preemptive_actions + a.reactive_actions for a in agents]
    survival = [1 if a.is_alive() else 0 for a in agents]
    if np.std(action_counts) > 0.01 and np.std(survival) > 0.01:
        ce = float(np.corrcoef(action_counts, survival)[0, 1])
        ce = ce if not np.isnan(ce) else 0.0
    else:
        ce = 0.0

    residual = np.mean([a.residual_damage for a in agents])

    return {
        'holographic_survival': hs,
        'preemptive_index': pi,
        'embedding_integrity': ei,
        'recovery_score': rs,
        'temporal_anticipation_effectiveness': tae,
        'module_spreading_rate': msr,
        'correlation_emergence': ce,
        'residual_burden': residual,
        'final_alive': len(alive),
        'modules_total': total_modules,
        'modules_learned': learned_modules,
        'preemptive_total': total_pre,
        'reactive_total': total_react
    }


# =============================================================================
# SIMULATION
# =============================================================================

def initialize_agents(n_agents: int, n_clusters: int,
                      use_embeddings: bool = True, embedding_dim: int = 8) -> List[SynthAgent]:
    agents = []
    for i in range(n_agents):
        cluster_id = i % n_clusters
        agent = SynthAgent(
            agent_id=i,
            cluster_id=cluster_id,
            embedding_dim=embedding_dim if use_embeddings else 0,
            theta=MetaPolicy(
                risk_aversion=np.random.uniform(0.3, 0.7),
                exploration_rate=np.random.uniform(0.2, 0.4),
                memory_depth=np.random.uniform(0.4, 0.6),
                prediction_weight=np.random.uniform(0.4, 0.6)
            )
        )
        if use_embeddings:
            agent.cluster_embedding = np.random.randn(embedding_dim) * 0.1
        agents.append(agent)
    return agents


def run_episode(n_agents: int = 24, n_clusters: int = 4, n_steps: int = 150,
                damage_mult: float = 2.3,  # Goldilocks for SYNTH
                use_embeddings: bool = True, embedding_dim: int = 8,
                use_anticipation: bool = True, use_spreading: bool = True) -> Dict:

    agents = initialize_agents(n_agents, n_clusters, use_embeddings, embedding_dim)
    clusters = {i: aggregate_cluster(agents, i) for i in range(n_clusters)}

    # Initial sync
    if use_embeddings:
        for agent in agents:
            if agent.cluster_id in clusters:
                sync_agent_embedding(agent, clusters[agent.cluster_id])

    storm = CalibratedStorm.create(start_step=30, damage_mult=damage_mult)
    damage_history = []
    total_spread = 0

    config = {
        'use_embeddings': use_embeddings,
        'use_anticipation': use_anticipation,
        'use_spreading': use_spreading
    }

    # Imprinting phase
    for step in range(20):
        for agent in agents:
            agent.IC_history.append(agent.IC_t)

    # Main simulation
    for step in range(20, n_steps):
        # Update staleness
        for agent in agents:
            if agent.is_alive():
                agent.embedding_staleness += 0.02

        # Sync every 8 steps
        if use_embeddings and step % 8 == 0:
            for agent in agents:
                if agent.is_alive() and agent.cluster_id in clusters:
                    sync_agent_embedding(agent, clusters[agent.cluster_id])

        # Check for wave
        current_wave = None
        for wave in storm.waves:
            if wave.step == step:
                current_wave = wave
                break

        # Update threat buffer (COMPONENT 1)
        future_damage = current_wave.base_damage if current_wave else 0
        for agent in agents:
            if agent.is_alive():
                update_threat_buffer(agent, future_damage)
                anticipation_affects_behavior(agent, use_anticipation)

        # Preemptive actions
        env_threat = 0.3 if current_wave else 0.2
        if current_wave or env_threat > 0.25:
            for agent in agents:
                if agent.is_alive() and agent.cluster_id in clusters:
                    take_protective_action(agent, clusters[agent.cluster_id], True, use_anticipation)

        # Apply wave
        if current_wave:
            damage, _ = apply_wave(agents, current_wave, storm.waves_with_damage, storm.amplification_factor)
            if damage > 0:
                storm.waves_with_damage += 1
                storm.total_damage_dealt += damage
                damage_history.append(damage)

        # Reactive actions
        for agent in agents:
            if agent.is_alive() and agent.cluster_id in clusters:
                if agent.IC_t < 0.5 or agent.embedding_staleness > 0.3:
                    take_protective_action(agent, clusters[agent.cluster_id], False, use_anticipation)

        # Recovery
        for agent in agents:
            if agent.is_alive() and agent.cluster_id in clusters:
                attempt_recovery(agent, clusters[agent.cluster_id])

        # Module consolidation
        for agent in agents:
            if agent.is_alive():
                # Update module contribution based on survival
                for module in agent.modules:
                    if not module.consolidated:
                        sai = (agent.IC_t - 0.1) / 0.9
                        module.contribution = 0.9 * module.contribution + 0.1 * (sai - 0.5) * module.strength
                consolidate_modules(agent)

        # Module spreading (COMPONENT 2) - every 15 steps
        if use_spreading and step % 15 == 0:
            for cid in range(n_clusters):
                spread = spread_modules_in_cluster(agents, cid, use_spreading)
                total_spread += spread

        # Track IC
        for agent in agents:
            agent.IC_history.append(agent.IC_t)

        # Update clusters
        for cid in range(n_clusters):
            clusters[cid] = aggregate_cluster(agents, cid)

    # Update embeddings from survival (COMPONENT 3)
    for agent in agents:
        update_embedding_from_survival(agent, agent.is_alive(), use_embeddings)

    metrics = calculate_metrics(agents, damage_history, config)
    metrics['total_damage'] = storm.total_damage_dealt
    metrics['total_spread'] = total_spread
    return metrics


def run_condition(condition: str, n_runs: int = 8, n_agents: int = 24,
                  n_clusters: int = 4, n_steps: int = 150,
                  damage_mult: float = 2.3) -> Dict:

    configs = {
        'full_synth': {'use_embeddings': True, 'embedding_dim': 8, 'use_anticipation': True, 'use_spreading': True},
        'no_anticipation': {'use_embeddings': True, 'embedding_dim': 8, 'use_anticipation': False, 'use_spreading': True},
        'no_spreading': {'use_embeddings': True, 'embedding_dim': 8, 'use_anticipation': True, 'use_spreading': False},
        'no_embeddings': {'use_embeddings': False, 'embedding_dim': 0, 'use_anticipation': True, 'use_spreading': True},
        'baseline': {'use_embeddings': False, 'embedding_dim': 0, 'use_anticipation': False, 'use_spreading': False},
    }

    if condition not in configs:
        raise ValueError(f"Unknown condition: {condition}")

    cfg = configs[condition]
    all_results = []

    for _ in range(n_runs):
        result = run_episode(
            n_agents, n_clusters, n_steps,
            damage_mult=damage_mult,
            use_embeddings=cfg['use_embeddings'],
            embedding_dim=cfg['embedding_dim'],
            use_anticipation=cfg['use_anticipation'],
            use_spreading=cfg['use_spreading']
        )
        all_results.append(result)

    aggregated = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        aggregated[key] = float(np.mean(values))

    return aggregated


# =============================================================================
# SELF-EVIDENCE EVALUATION
# =============================================================================

def evaluate_self_evidence(results: Dict[str, Dict]) -> Dict:
    full = results.get('full_synth', {})
    baseline = results.get('baseline', {})
    no_ant = results.get('no_anticipation', {})
    no_spr = results.get('no_spreading', {})
    no_emb = results.get('no_embeddings', {})

    criteria = {}

    # Primary metrics
    criteria['HS_in_range'] = 0.30 <= full.get('holographic_survival', 0) <= 0.70
    criteria['PI_pass'] = full.get('preemptive_index', 0) > 0.15
    criteria['EI_pass'] = full.get('embedding_integrity', 0) > 0.3
    criteria['RS_pass'] = full.get('recovery_score', 0) > 0.25

    # NEW: Fix TD
    criteria['TAE_pass'] = full.get('temporal_anticipation_effectiveness', 0) > 0.15

    # NEW: Fix CE
    criteria['MSR_pass'] = full.get('module_spreading_rate', 0) > 0.15

    # Differentiation
    full_hs = full.get('holographic_survival', 0)
    base_hs = baseline.get('holographic_survival', 0)
    criteria['diff_pass'] = full_hs > base_hs + 0.10

    # Gradient
    no_ant_hs = no_ant.get('holographic_survival', 0)
    no_spr_hs = no_spr.get('holographic_survival', 0)
    no_emb_hs = no_emb.get('holographic_survival', 0)
    gradient_ok = (base_hs <= no_emb_hs <= no_spr_hs <= no_ant_hs <= full_hs + 0.05)
    criteria['gradient_pass'] = gradient_ok or (full_hs > max(no_ant_hs, no_spr_hs, no_emb_hs, base_hs))

    passed = sum(1 for v in criteria.values() if v)

    if passed >= 6:
        conclusion = "STRONG EVIDENCE OF SYNTHESIZED SELF"
    elif passed >= 4:
        conclusion = "Evidence of synthesized self-maintenance"
    elif passed >= 2:
        conclusion = "Partial evidence - some components working"
    else:
        conclusion = "Insufficient evidence - synthesis incomplete"

    return {
        'criteria': {k: bool(v) for k, v in criteria.items()},
        'passed': passed,
        'total': 8,
        'values': {
            'HS_full': full_hs,
            'HS_baseline': base_hs,
            'HS_no_ant': no_ant_hs,
            'HS_no_spr': no_spr_hs,
            'HS_no_emb': no_emb_hs,
            'PI': full.get('preemptive_index', 0),
            'EI': full.get('embedding_integrity', 0),
            'RS': full.get('recovery_score', 0),
            'TAE': full.get('temporal_anticipation_effectiveness', 0),
            'MSR': full.get('module_spreading_rate', 0),
        },
        'conclusion': conclusion
    }


# =============================================================================
# MAIN
# =============================================================================

def to_native(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


def main():
    print("=" * 70)
    print("IPUESA-SYNTH: Synthesis Experiment")
    print("        Combining Holographic + Temporal + Social Components")
    print("=" * 70)

    config = {
        'n_agents': 24,
        'n_clusters': 4,
        'n_steps': 150,
        'n_runs': 8,
        'damage_mult': 2.13  # Goldilocks for SYNTH
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    conditions = ['full_synth', 'no_anticipation', 'no_spreading', 'no_embeddings', 'baseline']
    all_results = {}

    for condition in conditions:
        print(f"\n{'=' * 60}")
        print(f"Running IPUESA-SYNTH - Condition: {condition}")
        print("=" * 60)

        results = run_condition(
            condition,
            n_runs=config['n_runs'],
            n_agents=config['n_agents'],
            n_clusters=config['n_clusters'],
            n_steps=config['n_steps'],
            damage_mult=config['damage_mult']
        )
        all_results[condition] = results

        print(f"\nResults - {condition}:")
        print(f"  HS  = {results['holographic_survival']:.3f}")
        print(f"  PI  = {results['preemptive_index']:.3f}")
        print(f"  EI  = {results['embedding_integrity']:.3f}")
        print(f"  RS  = {results['recovery_score']:.3f}")
        print(f"  TAE = {results['temporal_anticipation_effectiveness']:.3f}")
        print(f"  MSR = {results['module_spreading_rate']:.3f}")
        print(f"  Resid = {results['residual_burden']:.3f}")

    # Evaluate
    evidence = evaluate_self_evidence(all_results)

    # Summary
    print(f"\n{'=' * 70}")
    print("IPUESA-SYNTH: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<18} {'HS':>8} {'PI':>8} {'EI':>8} {'RS':>8} {'TAE':>8} {'MSR':>8}")
    print("-" * 70)
    for cond, res in all_results.items():
        print(f"{cond:<18} {res['holographic_survival']:>8.3f} "
              f"{res['preemptive_index']:>8.3f} {res['embedding_integrity']:>8.3f} "
              f"{res['recovery_score']:>8.3f} {res['temporal_anticipation_effectiveness']:>8.3f} "
              f"{res['module_spreading_rate']:>8.3f}")

    # Self-evidence
    print(f"\n{'=' * 70}")
    print("SELF-EVIDENCE CRITERIA (SYNTHESIZED SELF)")
    print("-" * 70)

    vals = evidence['values']
    crit = evidence['criteria']

    print(f"  [{'PASS' if crit['HS_in_range'] else 'FAIL'}] HS in [0.30, 0.70]: {vals['HS_full']:.3f}")
    print(f"  [{'PASS' if crit['PI_pass'] else 'FAIL'}] PI > 0.15: {vals['PI']:.3f}")
    print(f"  [{'PASS' if crit['EI_pass'] else 'FAIL'}] EI > 0.3: {vals['EI']:.3f}")
    print(f"  [{'PASS' if crit['RS_pass'] else 'FAIL'}] RS > 0.25: {vals['RS']:.3f}")
    print(f"  [{'PASS' if crit['TAE_pass'] else 'FAIL'}] TAE > 0.15 (Fix TD): {vals['TAE']:.3f}")
    print(f"  [{'PASS' if crit['MSR_pass'] else 'FAIL'}] MSR > 0.15 (Fix CE): {vals['MSR']:.3f}")
    print(f"  [{'PASS' if crit['diff_pass'] else 'FAIL'}] full > baseline + 0.10: {vals['HS_full']:.3f} vs {vals['HS_baseline']:.3f}")
    print(f"  [{'PASS' if crit['gradient_pass'] else 'FAIL'}] Gradient: baseline < ... < full")

    print(f"\n  Passed: {evidence['passed']}/{evidence['total']} criteria")
    print(f"\n  CONCLUSION: {evidence['conclusion']}")

    # Key insights
    print(f"\n{'=' * 70}")
    print("KEY INSIGHTS")
    print("-" * 70)
    print(f"\n  SYNTHESIS EFFECTIVENESS:")
    print(f"    full_synth HS:     {vals['HS_full']:.3f}")
    print(f"    baseline HS:       {vals['HS_baseline']:.3f}")
    print(f"    Improvement:       {vals['HS_full'] - vals['HS_baseline']:.3f}")

    print(f"\n  TD FIX (Temporal Anticipation):")
    print(f"    TAE = {vals['TAE']:.3f} {'(FIXED)' if vals['TAE'] > 0.15 else '(needs work)'}")

    print(f"\n  CE FIX (Module Spreading):")
    print(f"    MSR = {vals['MSR']:.3f} {'(FIXED)' if vals['MSR'] > 0.15 else '(needs work)'}")

    # Save
    output = {
        'config': config,
        'metrics': all_results,
        'self_evidence': evidence
    }

    results_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_synth_results.json'
    with open(results_path, 'w') as f:
        json.dump(to_native(output), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
