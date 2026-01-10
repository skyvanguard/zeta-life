"""
IPUESA-HG+: Holographic Self Stress Test
=========================================

Enhanced stress test of holographic embeddings under extreme conditions.
Designed to create clear differentiation between embedding and no-embedding conditions.

Key modifications from IPUESA-HG:
- 2× base damage for all perturbations
- Reduced wave interval: 15→10 steps
- Cumulative damage (residual that doesn't fully recover)
- New perturbation: structural (embedding alterations)
- New conditions: partial_hg (4-dim), high_stress (3×, 8-step)
- New metrics: RS (Recovery Score), CE (Correlation Emergence vs Damage)

Author: IPUESA Research
Date: 2026-01-10
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class MetaPolicy:
    """WHO the agent is - policy parameters"""
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

    def distance(self, other: 'MetaPolicy') -> float:
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))


@dataclass
class CognitiveArchitecture:
    """HOW the agent processes - cognitive parameters"""
    attention_immediate: float = 0.33
    attention_history: float = 0.33
    attention_prediction: float = 0.34
    memory_update_rate: float = 0.1
    perceptual_gain: float = 1.0

    def to_vector(self) -> np.ndarray:
        return np.array([self.attention_immediate, self.attention_history,
                        self.attention_prediction, self.memory_update_rate,
                        self.perceptual_gain])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'CognitiveArchitecture':
        return cls(attention_immediate=float(np.clip(v[0], 0, 1)),
                  attention_history=float(np.clip(v[1], 0, 1)),
                  attention_prediction=float(np.clip(v[2], 0, 1)),
                  memory_update_rate=float(np.clip(v[3], 0, 1)),
                  perceptual_gain=float(np.clip(v[4], 0.1, 2)))


@dataclass
class MicroModule:
    """Emergent micro-module (beta system)"""
    module_type: str
    strength: float = 0.5
    activation_count: int = 0
    creation_step: int = 0

    def apply(self, context: Dict) -> float:
        """Apply module effect based on type"""
        self.activation_count += 1
        effects = {
            'pattern_detector': 0.2,
            'threat_filter': 0.15,
            'recovery_accelerator': 0.25,
            'embedding_protector': 0.3,
            'cascade_breaker': 0.2,
            'residual_cleaner': 0.2,  # NEW: helps clear cumulative damage
        }
        return self.strength * effects.get(self.module_type, 0.1)


class DegradationState(Enum):
    """Agent degradation levels"""
    OPTIMAL = 'optimal'
    STRESSED = 'stressed'
    IMPAIRED = 'impaired'
    CRITICAL = 'critical'
    COLLAPSED = 'collapsed'


@dataclass
class HolographicAgent:
    """Agent with holographic embeddings for self-maintenance"""
    agent_id: int
    cluster_id: int

    # Triple adaptation
    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    modules: List[MicroModule] = field(default_factory=list)

    # Identity core
    IC_t: float = 1.0

    # Holographic embeddings (configurable dimension)
    cluster_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    collective_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    embedding_dim: int = 8
    embedding_staleness: float = 0.0

    # Self-maintenance state
    threat_anticipation: float = 0.0
    protective_stance: float = 0.0

    # CUMULATIVE DAMAGE (NEW in HG+)
    residual_damage: float = 0.0  # Doesn't fully recover
    structural_corruption: float = 0.0  # Embedding structural damage

    # History
    history: List[float] = field(default_factory=list)
    prediction_noise: float = 0.0
    history_corruption: float = 0.0

    # Tracking
    IC_history: List[float] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    preemptive_actions: int = 0
    reactive_actions: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0

    def get_degradation_state(self) -> DegradationState:
        """Get current degradation state based on composite health"""
        embedding_integrity = self.get_embedding_integrity()
        module_health = self.get_module_health()
        # Include residual damage in composite
        residual_penalty = self.residual_damage * 0.3
        composite = 0.4 * self.IC_t + 0.3 * embedding_integrity + 0.3 * module_health - residual_penalty

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
        """How intact are the holographic embeddings"""
        cluster_norm = np.linalg.norm(self.cluster_embedding)
        collective_norm = np.linalg.norm(self.collective_embedding)
        staleness_penalty = self.embedding_staleness * 0.3
        structural_penalty = self.structural_corruption * 0.5
        base = (cluster_norm + collective_norm) / 4
        return max(0, min(1, base - staleness_penalty - structural_penalty))

    def get_module_health(self) -> float:
        """Average strength of active modules"""
        if not self.modules:
            return 0.5
        return np.mean([m.strength for m in self.modules])

    def is_alive(self) -> bool:
        return self.IC_t > 0.1


@dataclass
class ClusterState:
    """Cluster-level state for aggregation"""
    cluster_id: int
    member_ids: Set[int] = field(default_factory=set)
    theta_cluster: MetaPolicy = field(default_factory=MetaPolicy)
    alpha_cluster: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    cohesion: float = 0.5
    threat_level: float = 0.0


@dataclass
class CollectiveState:
    """Collective-level state"""
    theta_collective: MetaPolicy = field(default_factory=MetaPolicy)
    alpha_collective: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    global_coherence: float = 0.5
    threat_level: float = 0.0


# =============================================================================
# HOLOGRAPHIC EMBEDDING SYSTEM
# =============================================================================

def encode_to_embedding(theta: MetaPolicy, alpha: CognitiveArchitecture,
                        threat: float, cohesion: float, dim: int = 8) -> np.ndarray:
    """Encode theta/alpha + context into embedding"""
    full = np.array([
        theta.risk_aversion,
        theta.exploration_rate,
        theta.memory_depth,
        theta.prediction_weight,
        alpha.attention_prediction,
        alpha.perceptual_gain,
        threat,
        cohesion
    ])
    if dim < 8:
        # Partial embedding: compress by averaging pairs
        return full[:dim]
    return full


def decode_threat_from_embedding(embedding: np.ndarray) -> float:
    """Extract threat signal from embedding"""
    if len(embedding) >= 7:
        return float(embedding[6])
    elif len(embedding) >= 4:
        # Partial embedding: infer from available data
        return float(np.mean(embedding) * 0.5)
    return 0.0


def sync_agent_embeddings(agent: HolographicAgent,
                          cluster: ClusterState,
                          collective: CollectiveState):
    """Refresh agent's holographic embeddings"""
    dim = agent.embedding_dim
    agent.cluster_embedding = encode_to_embedding(
        cluster.theta_cluster, cluster.alpha_cluster,
        cluster.threat_level, cluster.cohesion, dim
    )
    agent.collective_embedding = encode_to_embedding(
        collective.theta_collective, collective.alpha_collective,
        collective.threat_level, collective.global_coherence, dim
    )
    agent.embedding_staleness = 0.0


def aggregate_cluster_state(agents: List[HolographicAgent],
                            cluster_id: int) -> ClusterState:
    """Aggregate agent states to cluster level"""
    members = [a for a in agents if a.cluster_id == cluster_id and a.is_alive()]
    if not members:
        return ClusterState(cluster_id=cluster_id)

    total_weight = sum(a.IC_t for a in members)
    if total_weight < 0.01:
        total_weight = 1.0

    theta_vec = sum(a.theta.to_vector() * a.IC_t for a in members) / total_weight
    alpha_vec = sum(a.alpha.to_vector() * a.IC_t for a in members) / total_weight

    theta_vecs = np.array([a.theta.to_vector() for a in members])
    variance = np.mean(np.var(theta_vecs, axis=0)) if len(members) > 1 else 0
    cohesion = max(0, 1 - variance * 5)

    threat = np.mean([a.threat_anticipation for a in members])

    return ClusterState(
        cluster_id=cluster_id,
        member_ids=set(a.agent_id for a in members),
        theta_cluster=MetaPolicy.from_vector(theta_vec),
        alpha_cluster=CognitiveArchitecture.from_vector(alpha_vec),
        cohesion=cohesion,
        threat_level=threat
    )


def aggregate_collective_state(clusters: List[ClusterState]) -> CollectiveState:
    """Aggregate cluster states to collective level"""
    active_clusters = [c for c in clusters if len(c.member_ids) > 0]
    if not active_clusters:
        return CollectiveState()

    weights = [c.cohesion * len(c.member_ids) for c in active_clusters]
    total_weight = sum(weights)
    if total_weight < 0.01:
        total_weight = 1.0

    theta_vec = sum(c.theta_cluster.to_vector() * w
                    for c, w in zip(active_clusters, weights)) / total_weight
    alpha_vec = sum(c.alpha_cluster.to_vector() * w
                    for c, w in zip(active_clusters, weights)) / total_weight

    coherence = np.mean([c.cohesion for c in active_clusters])
    threat = np.mean([c.threat_level for c in active_clusters])

    return CollectiveState(
        theta_collective=MetaPolicy.from_vector(theta_vec),
        alpha_collective=CognitiveArchitecture.from_vector(alpha_vec),
        global_coherence=coherence,
        threat_level=threat
    )


# =============================================================================
# CASCADING STORM PERTURBATION SYSTEM (ENHANCED)
# =============================================================================

@dataclass
class PerturbationWave:
    """Single wave in cascading storm"""
    wave_type: str
    base_damage: float
    step: int
    amplification: float = 1.0
    residual_factor: float = 0.0  # How much becomes permanent


@dataclass
class CascadingStorm:
    """Enhanced multi-wave perturbation sequence"""
    waves: List[PerturbationWave] = field(default_factory=list)
    amplification_factor: float = 1.2
    wave_interval: int = 10  # Reduced from 15
    current_wave: int = 0
    total_damage_dealt: float = 0.0
    waves_with_damage: int = 0
    damage_multiplier: float = 2.0  # NEW: base damage multiplier

    @classmethod
    def create_stress_test(cls, start_step: int = 30,
                           damage_mult: float = 2.0,
                           interval: int = 10) -> 'CascadingStorm':
        """Create enhanced stress test storm"""
        # 6 wave types including structural
        wave_types = [
            ('history', 0.3, 0.15),       # (type, base_damage, residual_factor)
            ('prediction', 0.25, 0.1),
            ('social', 0.35, 0.2),
            ('structural', 0.3, 0.25),    # NEW: embedding structural damage
            ('identity', 0.2, 0.15),
            ('catastrophic', 0.5, 0.3)
        ]
        waves = []
        for i, (wtype, damage, residual) in enumerate(wave_types):
            waves.append(PerturbationWave(
                wave_type=wtype,
                base_damage=damage * damage_mult,
                step=start_step + i * interval,
                residual_factor=residual
            ))
        return cls(waves=waves, wave_interval=interval, damage_multiplier=damage_mult)

    @classmethod
    def create_high_stress(cls, start_step: int = 30) -> 'CascadingStorm':
        """3× damage, 8-step intervals"""
        return cls.create_stress_test(start_step, damage_mult=3.0, interval=8)

    @classmethod
    def create_normal(cls, start_step: int = 30) -> 'CascadingStorm':
        """Standard stress test (2×, 10-step)"""
        return cls.create_stress_test(start_step, damage_mult=2.0, interval=10)


def apply_perturbation_wave(agents: List[HolographicAgent],
                            wave: PerturbationWave,
                            prior_damage_count: int,
                            amp_factor: float = 1.2) -> Tuple[float, int]:
    """Apply single wave with cumulative damage"""
    effective_amp = wave.amplification * (amp_factor ** prior_damage_count)
    effective_damage = wave.base_damage * effective_amp

    total_damage = 0.0
    damaged_count = 0

    for agent in agents:
        if not agent.is_alive():
            continue

        # Calculate resistance
        resistance = agent.protective_stance * 0.3
        for module in agent.modules:
            if module.module_type == 'threat_filter':
                resistance += module.apply({})
            elif module.module_type == 'cascade_breaker':
                effective_damage *= (1 - module.apply({}))
            elif module.module_type == 'embedding_protector' and wave.wave_type in ['social', 'structural']:
                resistance += module.apply({})

        actual_damage = max(0, effective_damage - resistance)

        # Apply damage based on type
        if wave.wave_type == 'history':
            agent.history_corruption += actual_damage
            agent.IC_t -= actual_damage * 0.4

        elif wave.wave_type == 'prediction':
            agent.prediction_noise += actual_damage
            agent.alpha.attention_prediction *= (1 - actual_damage * 0.6)

        elif wave.wave_type == 'social':
            noise = np.random.randn(len(agent.cluster_embedding)) * actual_damage
            agent.cluster_embedding += noise
            agent.collective_embedding += noise
            agent.embedding_staleness += actual_damage

        elif wave.wave_type == 'structural':  # NEW
            # Directly corrupt embedding structure
            corruption = np.random.randn(len(agent.cluster_embedding)) * actual_damage * 0.5
            agent.cluster_embedding *= (1 - actual_damage * 0.3)
            agent.collective_embedding *= (1 - actual_damage * 0.3)
            agent.structural_corruption += actual_damage * 0.5
            agent.IC_t -= actual_damage * 0.2

        elif wave.wave_type == 'identity':
            agent.IC_t -= actual_damage

        elif wave.wave_type == 'catastrophic':
            agent.IC_t -= actual_damage * 0.5
            agent.history_corruption += actual_damage * 0.4
            agent.prediction_noise += actual_damage * 0.4
            agent.embedding_staleness += actual_damage * 0.6
            agent.structural_corruption += actual_damage * 0.3

        # Add residual damage (cumulative, doesn't fully recover)
        residual = actual_damage * wave.residual_factor
        agent.residual_damage += residual

        # Clamp values
        agent.IC_t = max(0, min(1, agent.IC_t))
        agent.residual_damage = min(0.8, agent.residual_damage)  # Cap at 0.8

        if actual_damage > 0.05:
            damaged_count += 1
            total_damage += actual_damage

    return total_damage, damaged_count


# =============================================================================
# PROACTIVE SELF-MAINTENANCE
# =============================================================================

def anticipate_threats(agent: HolographicAgent,
                       environment_threat: float) -> float:
    """Use holographic embedding to predict incoming damage"""
    local_threat = environment_threat * agent.alpha.attention_prediction

    cluster_threat = decode_threat_from_embedding(agent.cluster_embedding)
    collective_threat = decode_threat_from_embedding(agent.collective_embedding)

    # Account for embedding dimension (partial embeddings less accurate)
    embedding_factor = agent.embedding_dim / 8.0

    agent.threat_anticipation = (
        0.4 * local_threat +
        0.35 * cluster_threat * embedding_factor +
        0.25 * collective_threat * embedding_factor
    )

    for module in agent.modules:
        if module.module_type == 'pattern_detector':
            agent.threat_anticipation += module.apply({})

    return agent.threat_anticipation


def take_protective_action(agent: HolographicAgent,
                           clusters: Dict[int, ClusterState],
                           collective: CollectiveState,
                           is_preemptive: bool) -> Optional[str]:
    """Agent takes protective action based on threat level"""
    threat = agent.threat_anticipation
    state = agent.get_degradation_state()

    action_taken = None

    # Emergency module creation
    if threat > 0.7 and state in [DegradationState.CRITICAL, DegradationState.IMPAIRED]:
        if agent.IC_t > 0.15 and len(agent.modules) < 5:
            module_types = ['cascade_breaker', 'embedding_protector', 'residual_cleaner']
            module_type = np.random.choice(module_types)
            agent.modules.append(MicroModule(
                module_type=module_type,
                strength=0.6,
                creation_step=0
            ))
            agent.IC_t -= 0.1
            action_taken = 'emergency_module'

    # Isolation
    elif threat > 0.6 and state == DegradationState.CRITICAL:
        agent.cluster_id = -1
        action_taken = 'isolate'

    # Sync embeddings
    elif threat > 0.4 and agent.embedding_staleness > 0.2:
        if agent.cluster_id >= 0 and agent.cluster_id in clusters:
            sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)
            agent.IC_t -= 0.03
            action_taken = 'sync_embeddings'

    # Harden
    elif threat > 0.25:
        agent.protective_stance = min(1.0, agent.protective_stance + 0.4)
        agent.theta.exploration_rate = max(0, agent.theta.exploration_rate - 0.1)
        action_taken = 'harden'

    if action_taken:
        agent.actions_taken.append(action_taken)
        if is_preemptive:
            agent.preemptive_actions += 1
        else:
            agent.reactive_actions += 1

    return action_taken


# =============================================================================
# RECOVERY SYSTEM (ENHANCED)
# =============================================================================

def attempt_recovery(agent: HolographicAgent,
                     cluster: Optional[ClusterState],
                     base_rate: float = 0.04) -> Tuple[float, bool]:
    """Attempt recovery with cumulative damage consideration"""
    if not agent.is_alive():
        return 0.0, False

    agent.recovery_attempts += 1
    embedding_integrity = agent.get_embedding_integrity()
    cluster_support = cluster.cohesion if cluster else 0.2

    # Recovery rate reduced by residual damage
    residual_penalty = agent.residual_damage * 0.5
    recovery_rate = base_rate * (1 + embedding_integrity) * (1 + cluster_support * 0.5) * (1 - residual_penalty)

    # Modules can help
    for module in agent.modules:
        if module.module_type == 'recovery_accelerator':
            recovery_rate *= (1 + module.apply({}))
        elif module.module_type == 'residual_cleaner':
            agent.residual_damage *= (1 - module.apply({}) * 0.1)

    # Apply recovery
    recovery = min(1.0 - agent.IC_t, recovery_rate)
    pre_IC = agent.IC_t
    agent.IC_t += recovery

    # Partial recovery of other damages
    agent.history_corruption *= 0.92
    agent.prediction_noise *= 0.92
    agent.embedding_staleness *= 0.95
    agent.structural_corruption *= 0.97  # Slow recovery
    agent.residual_damage *= 0.98  # Very slow recovery

    success = agent.IC_t > pre_IC + 0.01
    if success:
        agent.successful_recoveries += 1

    return recovery, success


def apply_degradation_effects(agent: HolographicAgent):
    """Apply state-dependent effects"""
    state = agent.get_degradation_state()

    if state == DegradationState.STRESSED:
        agent.theta.risk_aversion = min(1.0, agent.theta.risk_aversion + 0.15)
        agent.theta.exploration_rate = max(0, agent.theta.exploration_rate - 0.08)

    elif state == DegradationState.IMPAIRED:
        agent.alpha.attention_prediction *= 0.85
        agent.modules = [m for m in agent.modules if m.strength > 0.25]

    elif state == DegradationState.CRITICAL:
        agent.theta.risk_aversion = 1.0
        agent.theta.exploration_rate = 0.0
        agent.embedding_staleness = max(0, agent.embedding_staleness - 0.05)

    elif state == DegradationState.COLLAPSED:
        # Agency lost
        agent.theta = MetaPolicy(
            risk_aversion=np.random.random(),
            exploration_rate=np.random.random() * 0.2,
            memory_depth=np.random.random() * 0.3,
            prediction_weight=np.random.random() * 0.3
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_holographic_survival(agents: List[HolographicAgent]) -> float:
    """HS: Fraction of agents surviving"""
    alive = sum(1 for a in agents if a.is_alive())
    return alive / len(agents) if agents else 0.0


def calculate_preemptive_index(agents: List[HolographicAgent]) -> float:
    """PI: Ratio of preemptive to total protective actions"""
    total_preemptive = sum(a.preemptive_actions for a in agents)
    total_reactive = sum(a.reactive_actions for a in agents)
    total = total_preemptive + total_reactive
    return total_preemptive / total if total > 0 else 0.0


def calculate_degradation_smoothness(agents: List[HolographicAgent]) -> float:
    """DS: Smoothness of IC decline"""
    max_derivatives = []
    for agent in agents:
        if len(agent.IC_history) < 2:
            continue
        derivatives = np.abs(np.diff(agent.IC_history))
        if len(derivatives) > 0:
            max_derivatives.append(np.max(derivatives))

    if not max_derivatives:
        return 1.0

    return max(0, 1 - np.mean(max_derivatives) * 2)


def calculate_embedding_integrity(agents: List[HolographicAgent]) -> float:
    """EI: Average embedding integrity"""
    alive = [a for a in agents if a.is_alive()]
    if not alive:
        return 0.0
    return np.mean([a.get_embedding_integrity() for a in alive])


def calculate_recovery_score(agents: List[HolographicAgent]) -> float:
    """RS: Recovery success rate"""
    total_attempts = sum(a.recovery_attempts for a in agents)
    total_successes = sum(a.successful_recoveries for a in agents)
    return total_successes / total_attempts if total_attempts > 0 else 0.0


def calculate_correlation_emergence(agents: List[HolographicAgent],
                                    damage_history: List[float]) -> float:
    """CE: Correlation between protective actions and damage mitigation"""
    if not damage_history or len(damage_history) < 2:
        return 0.0

    # Count modules created per agent
    module_counts = [len(a.modules) for a in agents]
    action_counts = [a.preemptive_actions + a.reactive_actions for a in agents]

    # Survival correlation
    survival = [1 if a.is_alive() else 0 for a in agents]

    if np.std(action_counts) < 0.01 or np.std(survival) < 0.01:
        return 0.0

    correlation = np.corrcoef(action_counts, survival)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def calculate_residual_burden(agents: List[HolographicAgent]) -> float:
    """Average residual damage across agents"""
    if not agents:
        return 0.0
    return np.mean([a.residual_damage for a in agents])


# =============================================================================
# SIMULATION
# =============================================================================

def initialize_agents(n_agents: int, n_clusters: int,
                      use_embeddings: bool = True,
                      embedding_dim: int = 8) -> List[HolographicAgent]:
    """Initialize agents"""
    agents = []
    for i in range(n_agents):
        cluster_id = i % n_clusters
        agent = HolographicAgent(
            agent_id=i,
            cluster_id=cluster_id,
            embedding_dim=embedding_dim if use_embeddings else 0,
            theta=MetaPolicy(
                risk_aversion=np.random.uniform(0.3, 0.7),
                exploration_rate=np.random.uniform(0.2, 0.4),
                memory_depth=np.random.uniform(0.4, 0.6),
                prediction_weight=np.random.uniform(0.4, 0.6)
            ),
            alpha=CognitiveArchitecture(
                attention_immediate=np.random.uniform(0.25, 0.4),
                attention_history=np.random.uniform(0.25, 0.4),
                attention_prediction=np.random.uniform(0.25, 0.4),
                memory_update_rate=np.random.uniform(0.08, 0.12),
                perceptual_gain=np.random.uniform(0.9, 1.1)
            )
        )

        if use_embeddings:
            agent.cluster_embedding = np.random.randn(embedding_dim) * 0.1
            agent.collective_embedding = np.random.randn(embedding_dim) * 0.1

        agents.append(agent)

    return agents


def run_simulation_step(agents: List[HolographicAgent],
                        clusters: Dict[int, ClusterState],
                        collective: CollectiveState,
                        storm: CascadingStorm,
                        step: int,
                        sync_interval: int = 8,
                        use_embeddings: bool = True) -> Dict:
    """Run single simulation step"""
    metrics = {
        'alive': sum(1 for a in agents if a.is_alive()),
        'wave_damage': 0.0,
        'wave_type': None,
        'preemptive_actions': 0,
        'reactive_actions': 0
    }

    # Update embedding staleness
    for agent in agents:
        if agent.is_alive():
            agent.embedding_staleness += 0.03  # Faster staleness

    # Periodic embedding sync
    if use_embeddings and step % sync_interval == 0:
        for agent in agents:
            if agent.is_alive() and agent.cluster_id >= 0 and agent.cluster_id in clusters:
                sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)

    # Check for incoming wave
    current_wave = None
    for wave in storm.waves:
        if wave.step == step:
            current_wave = wave
            break

    # Anticipate threats
    environment_threat = 0.2
    if current_wave:
        environment_threat = 0.4 + current_wave.base_damage

    for agent in agents:
        if agent.is_alive():
            anticipate_threats(agent, environment_threat)

    # Preemptive actions
    if current_wave or environment_threat > 0.3:
        for agent in agents:
            if agent.is_alive():
                action = take_protective_action(agent, clusters, collective, is_preemptive=True)
                if action:
                    metrics['preemptive_actions'] += 1

    # Apply wave
    if current_wave:
        damage, damaged = apply_perturbation_wave(
            agents, current_wave, storm.waves_with_damage, storm.amplification_factor
        )
        metrics['wave_damage'] = damage
        metrics['wave_type'] = current_wave.wave_type
        if damaged > 0:
            storm.waves_with_damage += 1
        storm.total_damage_dealt += damage

    # Reactive actions
    for agent in agents:
        if agent.is_alive():
            if agent.IC_t < 0.6 or agent.embedding_staleness > 0.4 or agent.residual_damage > 0.2:
                action = take_protective_action(agent, clusters, collective, is_preemptive=False)
                if action:
                    metrics['reactive_actions'] += 1

    # Apply degradation effects
    for agent in agents:
        if agent.is_alive():
            apply_degradation_effects(agent)

    # Recovery attempts
    for agent in agents:
        if agent.is_alive() and agent.cluster_id >= 0:
            cluster = clusters.get(agent.cluster_id)
            attempt_recovery(agent, cluster)

    # Track IC history
    for agent in agents:
        agent.IC_history.append(agent.IC_t)

    # Update cluster and collective states
    cluster_ids = set(a.cluster_id for a in agents if a.cluster_id >= 0)
    for cid in cluster_ids:
        clusters[cid] = aggregate_cluster_state(agents, cid)
    collective_new = aggregate_collective_state(list(clusters.values()))
    collective.theta_collective = collective_new.theta_collective
    collective.alpha_collective = collective_new.alpha_collective
    collective.global_coherence = collective_new.global_coherence
    collective.threat_level = collective_new.threat_level

    metrics['alive'] = sum(1 for a in agents if a.is_alive())
    return metrics


def run_episode(n_agents: int = 24,
                n_clusters: int = 4,
                n_steps: int = 150,
                storm_type: str = 'normal',
                use_embeddings: bool = True,
                embedding_dim: int = 8,
                sync_interval: int = 8) -> Dict:
    """Run single episode"""

    agents = initialize_agents(n_agents, n_clusters, use_embeddings, embedding_dim)
    clusters = {i: aggregate_cluster_state(agents, i) for i in range(n_clusters)}
    collective = aggregate_collective_state(list(clusters.values()))

    if use_embeddings:
        for agent in agents:
            if agent.cluster_id >= 0:
                sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)

    # Create storm
    if storm_type == 'high_stress':
        storm = CascadingStorm.create_high_stress(start_step=25)
    else:
        storm = CascadingStorm.create_normal(start_step=30)

    damage_history = []

    # Imprinting phase
    for step in range(20):
        run_simulation_step(agents, clusters, collective, storm, step,
                           sync_interval, use_embeddings)

    # Main simulation
    for step in range(20, n_steps):
        metrics = run_simulation_step(agents, clusters, collective, storm, step,
                                     sync_interval, use_embeddings)
        if metrics['wave_damage'] > 0:
            damage_history.append(metrics['wave_damage'])

    # Calculate final metrics
    results = {
        'holographic_survival': calculate_holographic_survival(agents),
        'preemptive_index': calculate_preemptive_index(agents),
        'degradation_smoothness': calculate_degradation_smoothness(agents),
        'embedding_integrity': calculate_embedding_integrity(agents),
        'recovery_score': calculate_recovery_score(agents),
        'correlation_emergence': calculate_correlation_emergence(agents, damage_history),
        'residual_burden': calculate_residual_burden(agents),
        'total_damage': storm.total_damage_dealt,
        'final_alive': sum(1 for a in agents if a.is_alive()),
        'modules_created': sum(len(a.modules) for a in agents),
        'preemptive_total': sum(a.preemptive_actions for a in agents),
        'reactive_total': sum(a.reactive_actions for a in agents)
    }

    return results


def run_condition(condition: str, n_runs: int = 8, n_agents: int = 24,
                  n_clusters: int = 4, n_steps: int = 150) -> Dict:
    """Run multiple episodes for a condition"""

    # Configure based on condition
    if condition == 'full_hg':
        use_embeddings = True
        embedding_dim = 8
        storm_type = 'normal'
        sync_interval = 8
    elif condition == 'no_emb':
        use_embeddings = False
        embedding_dim = 0
        storm_type = 'normal'
        sync_interval = 8
    elif condition == 'partial_hg':
        use_embeddings = True
        embedding_dim = 4  # Reduced dimension
        storm_type = 'normal'
        sync_interval = 8
    elif condition == 'high_stress':
        use_embeddings = True
        embedding_dim = 8
        storm_type = 'high_stress'
        sync_interval = 6
    else:
        raise ValueError(f"Unknown condition: {condition}")

    all_results = []
    for run in range(n_runs):
        results = run_episode(n_agents, n_clusters, n_steps, storm_type,
                             use_embeddings, embedding_dim, sync_interval)
        all_results.append(results)

    # Aggregate results
    aggregated = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        aggregated[key] = float(np.mean(values))

    return aggregated


# =============================================================================
# SELF-EVIDENCE EVALUATION
# =============================================================================

def evaluate_self_evidence(results: Dict[str, Dict]) -> Dict:
    """Evaluate 8 self-evidence criteria for HG+"""
    criteria = {}

    full = results.get('full_hg', {})
    no_emb = results.get('no_emb', {})
    partial = results.get('partial_hg', {})
    high = results.get('high_stress', {})

    # Primary metrics (adjusted thresholds for harder test)
    criteria['HS_pass'] = full.get('holographic_survival', 0) > 0.3
    criteria['PI_pass'] = full.get('preemptive_index', 0) > 0.2
    criteria['DS_pass'] = full.get('degradation_smoothness', 0) > 0.5
    criteria['EI_pass'] = full.get('embedding_integrity', 0) > 0.4
    criteria['RS_pass'] = full.get('recovery_score', 0) > 0.3
    criteria['CE_pass'] = full.get('correlation_emergence', 0) > 0.2

    # Comparative metrics
    hs_diff = full.get('holographic_survival', 0) - no_emb.get('holographic_survival', 0)
    criteria['HS_diff_pass'] = hs_diff > 0.1

    # Partial should be between full and no_emb
    partial_hs = partial.get('holographic_survival', 0)
    full_hs = full.get('holographic_survival', 0)
    no_emb_hs = no_emb.get('holographic_survival', 0)
    criteria['partial_gradient_pass'] = (partial_hs > no_emb_hs) and (partial_hs <= full_hs + 0.05)

    passed = sum(1 for v in criteria.values() if v)

    return {
        'criteria': criteria,
        'passed': passed,
        'total': 8,
        'values': {
            'HS_full': full.get('holographic_survival', 0),
            'HS_no_emb': no_emb.get('holographic_survival', 0),
            'HS_partial': partial.get('holographic_survival', 0),
            'HS_diff': hs_diff,
            'PI': full.get('preemptive_index', 0),
            'DS': full.get('degradation_smoothness', 0),
            'EI_full': full.get('embedding_integrity', 0),
            'EI_no_emb': no_emb.get('embedding_integrity', 0),
            'RS': full.get('recovery_score', 0),
            'CE': full.get('correlation_emergence', 0),
            'residual_full': full.get('residual_burden', 0),
            'residual_no_emb': no_emb.get('residual_burden', 0)
        }
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def to_native(obj):
    """Convert numpy types to native Python for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    return obj


def main():
    print("=" * 70)
    print("IPUESA-HG+: Holographic Self Stress Test")
    print("        Enhanced Cascading Storm Resilience")
    print("=" * 70)

    config = {
        'n_agents': 24,
        'n_clusters': 4,
        'n_steps': 150,
        'n_runs': 8,
        'damage_multiplier': '2x (normal), 3x (high_stress)',
        'wave_interval': '10 (normal), 8 (high_stress)',
        'perturbation_types': 6
    }

    print(f"\nConfiguration:")
    print(f"  N agents: {config['n_agents']}")
    print(f"  N clusters: {config['n_clusters']}")
    print(f"  N steps: {config['n_steps']}")
    print(f"  N runs: {config['n_runs']}")
    print(f"  Damage multiplier: {config['damage_multiplier']}")
    print(f"  Wave interval: {config['wave_interval']}")
    print(f"  Perturbation types: {config['perturbation_types']} (incl. structural)")

    conditions = ['full_hg', 'no_emb', 'partial_hg', 'high_stress']

    all_results = {}

    for condition in conditions:
        print(f"\n{'=' * 60}")
        print(f"Running IPUESA-HG+ - Condition: {condition}")
        print("=" * 60)

        results = run_condition(
            condition,
            n_runs=config['n_runs'],
            n_agents=config['n_agents'],
            n_clusters=config['n_clusters'],
            n_steps=config['n_steps']
        )

        all_results[condition] = results

        print(f"\n{'=' * 60}")
        print(f"RESULTS - {condition}")
        print("=" * 60)
        print(f"\nPrimary Metrics:")
        print(f"  HS (Holographic Survival)   = {results['holographic_survival']:.3f}")
        print(f"  PI (Preemptive Index)       = {results['preemptive_index']:.3f}")
        print(f"  DS (Degradation Smoothness) = {results['degradation_smoothness']:.3f}")
        print(f"  EI (Embedding Integrity)    = {results['embedding_integrity']:.3f}")
        print(f"  RS (Recovery Score)         = {results['recovery_score']:.3f}")
        print(f"  CE (Correlation Emergence)  = {results['correlation_emergence']:.3f}")
        print(f"\nCumulative Damage:")
        print(f"  Residual Burden             = {results['residual_burden']:.3f}")
        print(f"  Total Damage                = {results['total_damage']:.2f}")
        print(f"\nActions:")
        print(f"  Preemptive Actions          = {results['preemptive_total']:.0f}")
        print(f"  Reactive Actions            = {results['reactive_total']:.0f}")
        print(f"  Modules Created             = {results['modules_created']:.0f}")

    # Evaluate self-evidence
    evidence = evaluate_self_evidence(all_results)

    # Comparative analysis
    print(f"\n{'=' * 70}")
    print("IPUESA-HG+: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<15} {'HS':>8} {'PI':>8} {'DS':>8} {'EI':>8} {'RS':>8} {'Resid':>8}")
    print("-" * 70)
    for cond, res in all_results.items():
        print(f"{cond:<15} {res['holographic_survival']:>8.3f} "
              f"{res['preemptive_index']:>8.3f} {res['degradation_smoothness']:>8.3f} "
              f"{res['embedding_integrity']:>8.3f} {res['recovery_score']:>8.3f} "
              f"{res['residual_burden']:>8.3f}")

    # Self-evidence criteria
    print(f"\n{'=' * 70}")
    print("SELF-EVIDENCE CRITERIA (HOLOGRAPHIC SELF STRESS TEST)")
    print("-" * 70)

    vals = evidence['values']
    crit = evidence['criteria']

    print(f"  [{'PASS' if crit['HS_pass'] else 'FAIL'}] HS > 0.3: {vals['HS_full']:.3f}")
    print(f"  [{'PASS' if crit['PI_pass'] else 'FAIL'}] PI > 0.2: {vals['PI']:.3f}")
    print(f"  [{'PASS' if crit['DS_pass'] else 'FAIL'}] DS > 0.5: {vals['DS']:.3f}")
    print(f"  [{'PASS' if crit['EI_pass'] else 'FAIL'}] EI > 0.4: {vals['EI_full']:.3f}")
    print(f"  [{'PASS' if crit['RS_pass'] else 'FAIL'}] RS > 0.3: {vals['RS']:.3f}")
    print(f"  [{'PASS' if crit['CE_pass'] else 'FAIL'}] CE > 0.2: {vals['CE']:.3f}")
    print(f"  [{'PASS' if crit['HS_diff_pass'] else 'FAIL'}] HS(full) - HS(no_emb) > 0.1: {vals['HS_diff']:.3f}")
    print(f"  [{'PASS' if crit['partial_gradient_pass'] else 'FAIL'}] Partial gradient: no_emb < partial <= full")

    print(f"\n  Passed: {evidence['passed']}/{evidence['total']} criteria")

    if evidence['passed'] >= 6:
        conclusion = "STRONG EVIDENCE OF HOLOGRAPHIC SELF"
    elif evidence['passed'] >= 4:
        conclusion = "Evidence of holographic self-maintenance"
    elif evidence['passed'] >= 2:
        conclusion = "Partial evidence - embedding provides advantage"
    else:
        conclusion = "No evidence - mechanism insufficient under stress"

    print(f"\n  CONCLUSION: {conclusion}")

    # Key insights
    print(f"\n{'=' * 70}")
    print("KEY INSIGHTS")
    print("-" * 70)

    full = all_results['full_hg']
    no_emb = all_results['no_emb']
    partial = all_results['partial_hg']

    print(f"\n  Survival differentiation:")
    print(f"    full_hg:    {full['holographic_survival']:.3f}")
    print(f"    partial_hg: {partial['holographic_survival']:.3f}")
    print(f"    no_emb:     {no_emb['holographic_survival']:.3f}")

    print(f"\n  Embedding integrity under stress:")
    print(f"    full_hg:    {full['embedding_integrity']:.3f}")
    print(f"    no_emb:     {no_emb['embedding_integrity']:.3f}")

    print(f"\n  Recovery effectiveness:")
    print(f"    full_hg RS: {full['recovery_score']:.3f}")
    print(f"    no_emb RS:  {no_emb['recovery_score']:.3f}")

    print(f"\n  Cumulative damage burden:")
    print(f"    full_hg:    {full['residual_burden']:.3f}")
    print(f"    no_emb:     {no_emb['residual_burden']:.3f}")

    # Save results
    output = {
        'config': config,
        'metrics': all_results,
        'self_evidence': {
            'passed': evidence['passed'],
            'total': evidence['total'],
            'criteria': {k: bool(v) for k, v in evidence['criteria'].items()},
            'values': evidence['values'],
            'conclusion': conclusion
        }
    }

    results_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_hg_plus_results.json'
    with open(results_path, 'w') as f:
        json.dump(to_native(output), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
