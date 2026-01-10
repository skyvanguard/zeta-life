"""
IPUESA-HG: Holographic Self Experiment
======================================

Synthesis experiment testing whether holographic embeddings enable genuine
self-maintenance under cascading multi-level perturbations.

Key innovations:
- Hybrid holographic embedding: each agent carries compressed representation
  of cluster/collective identity
- Cascading storm: perturbations arrive in sequence, each amplifying the next
- Proactive self-maintenance: agents take protective actions before damage
- Graceful degradation: smooth decline instead of cliff-edge collapse

Success criteria:
- HG survival >> SH survival under same cascading storm
- Measurable preemptive behavior (PI > 0.3)
- Smooth degradation curve (DS > 0.7)

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
        return cls(risk_aversion=float(v[0]), exploration_rate=float(v[1]),
                  memory_depth=float(v[2]), prediction_weight=float(v[3]))

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
        return cls(attention_immediate=float(v[0]), attention_history=float(v[1]),
                  attention_prediction=float(v[2]), memory_update_rate=float(v[3]),
                  perceptual_gain=float(v[4]))


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
        if self.module_type == 'pattern_detector':
            return self.strength * 0.2  # Threat detection bonus
        elif self.module_type == 'threat_filter':
            return self.strength * 0.15  # Damage reduction
        elif self.module_type == 'recovery_accelerator':
            return self.strength * 0.25  # Recovery speed bonus
        elif self.module_type == 'embedding_protector':
            return self.strength * 0.3  # Embedding integrity bonus
        elif self.module_type == 'cascade_breaker':
            return self.strength * 0.2  # Reduce amplification
        return 0.0


class DegradationState(Enum):
    """Agent degradation levels"""
    OPTIMAL = 'optimal'      # IC in [0.8, 1.0]
    STRESSED = 'stressed'    # IC in [0.5, 0.8)
    IMPAIRED = 'impaired'    # IC in [0.3, 0.5)
    CRITICAL = 'critical'    # IC in [0.1, 0.3)
    COLLAPSED = 'collapsed'  # IC in [0.0, 0.1)


@dataclass
class HolographicAgent:
    """Agent with holographic embeddings for self-maintenance"""
    agent_id: int
    cluster_id: int

    # Triple adaptation (theta, alpha, beta)
    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    modules: List[MicroModule] = field(default_factory=list)

    # Identity core
    IC_t: float = 1.0

    # Holographic embeddings (8-dimensional)
    cluster_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    collective_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    embedding_staleness: float = 0.0

    # Self-maintenance state
    threat_anticipation: float = 0.0
    protective_stance: float = 0.0

    # History
    history: List[float] = field(default_factory=list)
    prediction_noise: float = 0.0
    history_corruption: float = 0.0

    # Tracking
    IC_history: List[float] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    preemptive_actions: int = 0
    reactive_actions: int = 0

    def get_degradation_state(self) -> DegradationState:
        """Get current degradation state based on composite health"""
        embedding_integrity = self.get_embedding_integrity()
        module_health = self.get_module_health()
        composite = 0.4 * self.IC_t + 0.3 * embedding_integrity + 0.3 * module_health

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
        return max(0, min(1, (cluster_norm + collective_norm) / 4 - staleness_penalty))

    def get_module_health(self) -> float:
        """Average strength of active modules"""
        if not self.modules:
            return 0.5  # Baseline
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
                        threat: float, cohesion: float) -> np.ndarray:
    """Encode theta/alpha + context into 8-dim embedding"""
    return np.array([
        theta.risk_aversion,
        theta.exploration_rate,
        theta.memory_depth,
        theta.prediction_weight,
        alpha.attention_prediction,
        alpha.perceptual_gain,
        threat,
        cohesion
    ])


def decode_threat_from_embedding(embedding: np.ndarray) -> float:
    """Extract threat signal from embedding"""
    if len(embedding) >= 7:
        return float(embedding[6])
    return 0.0


def sync_agent_embeddings(agent: HolographicAgent,
                          cluster: ClusterState,
                          collective: CollectiveState):
    """Refresh agent's holographic embeddings from cluster/collective"""
    agent.cluster_embedding = encode_to_embedding(
        cluster.theta_cluster, cluster.alpha_cluster,
        cluster.threat_level, cluster.cohesion
    )
    agent.collective_embedding = encode_to_embedding(
        collective.theta_collective, collective.alpha_collective,
        collective.threat_level, collective.global_coherence
    )
    agent.embedding_staleness = 0.0


def aggregate_cluster_state(agents: List[HolographicAgent],
                            cluster_id: int) -> ClusterState:
    """Aggregate agent states to cluster level"""
    members = [a for a in agents if a.cluster_id == cluster_id and a.is_alive()]
    if not members:
        return ClusterState(cluster_id=cluster_id)

    # Weighted aggregation by IC_t
    total_weight = sum(a.IC_t for a in members)
    if total_weight < 0.01:
        total_weight = 1.0

    theta_vec = sum(a.theta.to_vector() * a.IC_t for a in members) / total_weight
    alpha_vec = sum(a.alpha.to_vector() * a.IC_t for a in members) / total_weight

    # Cluster cohesion based on theta variance
    theta_vecs = np.array([a.theta.to_vector() for a in members])
    variance = np.mean(np.var(theta_vecs, axis=0)) if len(members) > 1 else 0
    cohesion = max(0, 1 - variance * 5)

    # Average threat anticipation
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

    # Weighted by cohesion * size
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
# CASCADING STORM PERTURBATION SYSTEM
# =============================================================================

@dataclass
class PerturbationWave:
    """Single wave in cascading storm"""
    wave_type: str
    base_damage: float
    step: int
    amplification: float = 1.0


@dataclass
class CascadingStorm:
    """Multi-wave perturbation sequence"""
    waves: List[PerturbationWave] = field(default_factory=list)
    amplification_factor: float = 1.2
    wave_interval: int = 15
    current_wave: int = 0
    total_damage_dealt: float = 0.0
    waves_with_damage: int = 0

    @classmethod
    def create_default(cls, start_step: int = 30) -> 'CascadingStorm':
        """Create default 5-wave storm"""
        wave_types = [
            ('history', 0.3),
            ('prediction', 0.25),
            ('social', 0.35),
            ('identity', 0.2),
            ('catastrophic', 0.5)
        ]
        waves = []
        for i, (wtype, damage) in enumerate(wave_types):
            waves.append(PerturbationWave(
                wave_type=wtype,
                base_damage=damage,
                step=start_step + i * 15
            ))
        return cls(waves=waves)

    @classmethod
    def create_mild(cls, start_step: int = 30) -> 'CascadingStorm':
        """Reduced intensity storm for calibration"""
        storm = cls.create_default(start_step)
        for wave in storm.waves:
            wave.base_damage *= 0.5
        return storm

    @classmethod
    def create_extreme(cls, start_step: int = 30) -> 'CascadingStorm':
        """Double intensity storm for stress test"""
        storm = cls.create_default(start_step)
        for wave in storm.waves:
            wave.base_damage *= 2.0
        storm.amplification_factor = 1.4
        return storm


def apply_perturbation_wave(agents: List[HolographicAgent],
                            wave: PerturbationWave,
                            prior_damage_count: int) -> Tuple[float, int]:
    """Apply single wave to all agents, return total damage and count"""
    # Amplification based on prior damage
    effective_amp = wave.amplification * (1.2 ** prior_damage_count)
    effective_damage = wave.base_damage * effective_amp

    total_damage = 0.0
    damaged_count = 0

    for agent in agents:
        if not agent.is_alive():
            continue

        # Calculate resistance from protective stance and modules
        resistance = agent.protective_stance * 0.3
        for module in agent.modules:
            if module.module_type == 'threat_filter':
                resistance += module.apply({})
            elif module.module_type == 'cascade_breaker':
                effective_damage *= (1 - module.apply({}))

        actual_damage = max(0, effective_damage - resistance)

        if wave.wave_type == 'history':
            agent.history_corruption += actual_damage
            agent.IC_t -= actual_damage * 0.3
        elif wave.wave_type == 'prediction':
            agent.prediction_noise += actual_damage
            agent.alpha.attention_prediction *= (1 - actual_damage * 0.5)
        elif wave.wave_type == 'social':
            # Corrupt embeddings
            noise = np.random.randn(8) * actual_damage
            agent.cluster_embedding += noise
            agent.collective_embedding += noise
            agent.embedding_staleness += actual_damage
        elif wave.wave_type == 'identity':
            agent.IC_t -= actual_damage
        elif wave.wave_type == 'catastrophic':
            agent.IC_t -= actual_damage * 0.4
            agent.history_corruption += actual_damage * 0.3
            agent.prediction_noise += actual_damage * 0.3
            agent.embedding_staleness += actual_damage * 0.5

        agent.IC_t = max(0, min(1, agent.IC_t))

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
    # Local prediction
    local_threat = environment_threat * agent.alpha.attention_prediction

    # Holographic advantage: see threats from cluster/collective
    cluster_threat = decode_threat_from_embedding(agent.cluster_embedding)
    collective_threat = decode_threat_from_embedding(agent.collective_embedding)

    # Weighted combination
    agent.threat_anticipation = (
        0.4 * local_threat +
        0.35 * cluster_threat +
        0.25 * collective_threat
    )

    # Modules can enhance detection
    for module in agent.modules:
        if module.module_type == 'pattern_detector':
            agent.threat_anticipation += module.apply({})

    return agent.threat_anticipation


PROTECTIVE_ACTIONS = {
    'harden': {'threshold': 0.3, 'resistance_bonus': 0.5, 'exploration_cost': 0.1},
    'sync_embeddings': {'threshold': 0.5, 'staleness_required': 0.3, 'ic_cost': 0.05},
    'isolate': {'threshold': 0.7, 'loses_cluster': True},
    'emergency_module': {'threshold': 0.9, 'ic_cost': 0.15}
}


def take_protective_action(agent: HolographicAgent,
                           clusters: Dict[int, ClusterState],
                           collective: CollectiveState,
                           is_preemptive: bool) -> Optional[str]:
    """Agent takes protective action based on threat level"""
    threat = agent.threat_anticipation
    state = agent.get_degradation_state()

    action_taken = None

    # Check actions in order of severity
    if threat > 0.9 and state in [DegradationState.CRITICAL, DegradationState.IMPAIRED]:
        # Emergency module creation
        if agent.IC_t > 0.2:
            module_type = np.random.choice(['cascade_breaker', 'embedding_protector'])
            agent.modules.append(MicroModule(
                module_type=module_type,
                strength=0.6,
                creation_step=0
            ))
            agent.IC_t -= 0.15
            action_taken = 'emergency_module'

    elif threat > 0.7 and state == DegradationState.CRITICAL:
        # Isolation from damaged cluster
        agent.cluster_id = -1  # Isolated
        action_taken = 'isolate'

    elif threat > 0.5 and agent.embedding_staleness > 0.3:
        # Sync embeddings
        if agent.cluster_id >= 0 and agent.cluster_id in clusters:
            sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)
            agent.IC_t -= 0.05
            action_taken = 'sync_embeddings'

    elif threat > 0.3:
        # Harden defenses
        agent.protective_stance = min(1.0, agent.protective_stance + 0.5)
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
# GRACEFUL DEGRADATION & RECOVERY
# =============================================================================

def apply_degradation_effects(agent: HolographicAgent):
    """Apply state-dependent effects based on degradation level"""
    state = agent.get_degradation_state()

    if state == DegradationState.STRESSED:
        # Heightened defense, reduced exploration
        agent.theta.risk_aversion = min(1.0, agent.theta.risk_aversion + 0.1)
        agent.theta.exploration_rate = max(0, agent.theta.exploration_rate - 0.05)

    elif state == DegradationState.IMPAIRED:
        # Limited cognition
        agent.alpha.attention_prediction *= 0.9
        # Disable weak modules
        agent.modules = [m for m in agent.modules if m.strength > 0.3]

    elif state == DegradationState.CRITICAL:
        # Survival only mode
        agent.theta.risk_aversion = 1.0
        agent.theta.exploration_rate = 0.0
        # Emergency embedding sync attempt
        agent.embedding_staleness = max(0, agent.embedding_staleness - 0.1)

    elif state == DegradationState.COLLAPSED:
        # Agency lost - random behavior
        agent.theta = MetaPolicy(
            risk_aversion=np.random.random(),
            exploration_rate=np.random.random(),
            memory_depth=np.random.random(),
            prediction_weight=np.random.random()
        )


def attempt_recovery(agent: HolographicAgent,
                     cluster: Optional[ClusterState],
                     base_rate: float = 0.05) -> float:
    """Attempt to recover IC_t, enhanced by holographic embedding"""
    if not agent.is_alive():
        return 0.0

    embedding_integrity = agent.get_embedding_integrity()
    cluster_support = cluster.cohesion if cluster else 0.3

    # Recovery rate enhanced by embedding and cluster
    recovery_rate = base_rate * (1 + embedding_integrity) * (1 + cluster_support * 0.5)

    # Modules can accelerate
    for module in agent.modules:
        if module.module_type == 'recovery_accelerator':
            recovery_rate *= (1 + module.apply({}))

    # Apply recovery
    recovery = min(1.0 - agent.IC_t, recovery_rate)
    agent.IC_t += recovery

    # Also reduce corruption over time
    agent.history_corruption *= 0.95
    agent.prediction_noise *= 0.95
    agent.embedding_staleness *= 0.98

    return recovery


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
    if total == 0:
        return 0.0
    return total_preemptive / total


def calculate_degradation_smoothness(agents: List[HolographicAgent]) -> float:
    """DS: Smoothness of IC decline (1 - max derivative)"""
    max_derivatives = []
    for agent in agents:
        if len(agent.IC_history) < 2:
            continue
        derivatives = np.abs(np.diff(agent.IC_history))
        if len(derivatives) > 0:
            max_derivatives.append(np.max(derivatives))

    if not max_derivatives:
        return 1.0

    avg_max_deriv = np.mean(max_derivatives)
    return max(0, 1 - avg_max_deriv * 2)


def calculate_embedding_integrity(agents: List[HolographicAgent]) -> float:
    """EI: Average embedding integrity across agents"""
    if not agents:
        return 0.0
    return np.mean([a.get_embedding_integrity() for a in agents if a.is_alive()])


def calculate_waves_survived(storm: CascadingStorm,
                             initial_alive: int,
                             final_alive: int,
                             damage_per_wave: List[float]) -> int:
    """Count waves where majority survived"""
    waves_survived = 0
    for i, damage in enumerate(damage_per_wave):
        # Wave "survived" if less than 30% of damage potential realized
        if damage < storm.waves[i].base_damage * initial_alive * 0.3:
            waves_survived += 1
    return waves_survived


# =============================================================================
# SIMULATION
# =============================================================================

def initialize_agents(n_agents: int, n_clusters: int,
                      use_embeddings: bool = True) -> List[HolographicAgent]:
    """Initialize agents with random theta/alpha"""
    agents = []
    for i in range(n_agents):
        cluster_id = i % n_clusters
        agent = HolographicAgent(
            agent_id=i,
            cluster_id=cluster_id,
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
            # Initialize with random embeddings (will be synced)
            agent.cluster_embedding = np.random.randn(8) * 0.1
            agent.collective_embedding = np.random.randn(8) * 0.1

        agents.append(agent)

    return agents


def run_simulation_step(agents: List[HolographicAgent],
                        clusters: Dict[int, ClusterState],
                        collective: CollectiveState,
                        storm: CascadingStorm,
                        step: int,
                        sync_interval: int = 10,
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
            agent.embedding_staleness += 0.02

    # Periodic embedding sync
    if use_embeddings and step % sync_interval == 0:
        for agent in agents:
            if agent.is_alive() and agent.cluster_id >= 0:
                if agent.cluster_id in clusters:
                    sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)

    # Check for incoming wave
    current_wave = None
    for wave in storm.waves:
        if wave.step == step:
            current_wave = wave
            break

    # Anticipate threats
    environment_threat = 0.5 if current_wave else 0.1
    if current_wave:
        environment_threat = 0.3 + current_wave.base_damage

    for agent in agents:
        if agent.is_alive():
            anticipate_threats(agent, environment_threat)

    # Preemptive actions (before wave hits)
    if current_wave:
        for agent in agents:
            if agent.is_alive():
                action = take_protective_action(agent, clusters, collective, is_preemptive=True)
                if action:
                    metrics['preemptive_actions'] += 1

    # Apply wave if present
    if current_wave:
        damage, damaged = apply_perturbation_wave(agents, current_wave, storm.waves_with_damage)
        metrics['wave_damage'] = damage
        metrics['wave_type'] = current_wave.wave_type
        if damaged > 0:
            storm.waves_with_damage += 1
        storm.total_damage_dealt += damage

    # Reactive actions (after wave)
    for agent in agents:
        if agent.is_alive():
            # Check if needs reactive action
            if agent.IC_t < 0.5 or agent.embedding_staleness > 0.5:
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
                storm_type: str = 'default',
                use_embeddings: bool = True,
                sync_interval: int = 10) -> Dict:
    """Run single episode"""

    # Initialize
    agents = initialize_agents(n_agents, n_clusters, use_embeddings)
    clusters = {i: aggregate_cluster_state(agents, i) for i in range(n_clusters)}
    collective = aggregate_collective_state(list(clusters.values()))

    # Initial embedding sync
    if use_embeddings:
        for agent in agents:
            if agent.cluster_id >= 0:
                sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)

    # Create storm
    if storm_type == 'mild':
        storm = CascadingStorm.create_mild(start_step=30)
    elif storm_type == 'extreme':
        storm = CascadingStorm.create_extreme(start_step=30)
    else:
        storm = CascadingStorm.create_default(start_step=30)

    # Imprinting phase (first 20 steps)
    for step in range(20):
        run_simulation_step(agents, clusters, collective, storm, step,
                           sync_interval, use_embeddings)

    # Track damage per wave
    damage_per_wave = []

    # Main simulation
    for step in range(20, n_steps):
        metrics = run_simulation_step(agents, clusters, collective, storm, step,
                                     sync_interval, use_embeddings)
        if metrics['wave_damage'] > 0:
            damage_per_wave.append(metrics['wave_damage'])

    # Calculate final metrics
    initial_alive = n_agents
    results = {
        'holographic_survival': calculate_holographic_survival(agents),
        'preemptive_index': calculate_preemptive_index(agents),
        'degradation_smoothness': calculate_degradation_smoothness(agents),
        'embedding_integrity': calculate_embedding_integrity(agents),
        'waves_survived': calculate_waves_survived(storm, initial_alive,
                                                   sum(1 for a in agents if a.is_alive()),
                                                   damage_per_wave),
        'total_damage': storm.total_damage_dealt,
        'final_alive': sum(1 for a in agents if a.is_alive()),
        'modules_created': sum(len(a.modules) for a in agents)
    }

    return results


def run_condition(condition: str, n_runs: int = 8, n_agents: int = 24,
                  n_clusters: int = 4, n_steps: int = 150) -> Dict:
    """Run multiple episodes for a condition"""

    # Configure based on condition
    if condition == 'full_holographic':
        use_embeddings = True
        storm_type = 'default'
        sync_interval = 10
    elif condition == 'no_embedding':
        use_embeddings = False
        storm_type = 'default'
        sync_interval = 10
    elif condition == 'stale_embedding':
        use_embeddings = True
        storm_type = 'default'
        sync_interval = 1000  # Never syncs during storm
    elif condition == 'mild_storm':
        use_embeddings = True
        storm_type = 'mild'
        sync_interval = 10
    elif condition == 'extreme_storm':
        use_embeddings = True
        storm_type = 'extreme'
        sync_interval = 10
    else:
        raise ValueError(f"Unknown condition: {condition}")

    all_results = []
    for run in range(n_runs):
        results = run_episode(n_agents, n_clusters, n_steps, storm_type,
                             use_embeddings, sync_interval)
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
    """Evaluate 8 self-evidence criteria"""
    criteria = {}

    full = results.get('full_holographic', {})
    no_emb = results.get('no_embedding', {})

    # Primary metrics
    criteria['HS_pass'] = full.get('holographic_survival', 0) > 0.4
    criteria['PI_pass'] = full.get('preemptive_index', 0) > 0.3
    criteria['DS_pass'] = full.get('degradation_smoothness', 0) > 0.7
    criteria['EI_pass'] = full.get('embedding_integrity', 0) > 0.3

    # Comparative metrics
    hs_gain = full.get('holographic_survival', 0) - no_emb.get('holographic_survival', 0)
    pi_gain = full.get('preemptive_index', 0) - no_emb.get('preemptive_index', 0)

    criteria['HS_gain_pass'] = hs_gain > 0.15
    criteria['PI_gain_pass'] = pi_gain > 0.1

    # Recovery comparison
    no_emb_survival = no_emb.get('holographic_survival', 0)
    full_survival = full.get('holographic_survival', 0)
    recovery_ratio = full_survival / no_emb_survival if no_emb_survival > 0 else float('inf')
    criteria['recovery_gain_pass'] = recovery_ratio > 1.3

    # Cascade resistance
    waves_gain = full.get('waves_survived', 0) - no_emb.get('waves_survived', 0)
    criteria['cascade_resistance_pass'] = waves_gain >= 1

    passed = sum(1 for v in criteria.values() if v)

    return {
        'criteria': criteria,
        'passed': passed,
        'total': 8,
        'values': {
            'HS': full.get('holographic_survival', 0),
            'PI': full.get('preemptive_index', 0),
            'DS': full.get('degradation_smoothness', 0),
            'EI': full.get('embedding_integrity', 0),
            'HS_gain': hs_gain,
            'PI_gain': pi_gain,
            'recovery_ratio': recovery_ratio,
            'waves_gain': waves_gain
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
    return obj


def main():
    print("=" * 70)
    print("IPUESA-HG: Holographic Self Experiment")
    print("        Cascading Storm Resilience Test")
    print("=" * 70)

    # Configuration
    config = {
        'n_agents': 24,
        'n_clusters': 4,
        'n_steps': 150,
        'n_runs': 8
    }

    print(f"\nConfiguration:")
    print(f"  N agents: {config['n_agents']}")
    print(f"  N clusters: {config['n_clusters']}")
    print(f"  N steps: {config['n_steps']}")
    print(f"  N runs: {config['n_runs']}")

    # Run conditions
    conditions = [
        'full_holographic',
        'no_embedding',
        'stale_embedding',
        'mild_storm',
        'extreme_storm'
    ]

    all_results = {}

    for condition in conditions:
        print(f"\n{'=' * 60}")
        print(f"Running IPUESA-HG - Condition: {condition}")
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
        print(f"\nSupporting Metrics:")
        print(f"  Waves Survived              = {results['waves_survived']:.1f}")
        print(f"  Total Damage                = {results['total_damage']:.2f}")
        print(f"  Modules Created             = {results['modules_created']:.1f}")

    # Evaluate self-evidence
    evidence = evaluate_self_evidence(all_results)

    # Comparative analysis
    print(f"\n{'=' * 70}")
    print("IPUESA-HG: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<20} {'HS':>8} {'PI':>8} {'DS':>8} {'EI':>8} {'Waves':>8}")
    print("-" * 70)
    for cond, res in all_results.items():
        print(f"{cond:<20} {res['holographic_survival']:>8.3f} "
              f"{res['preemptive_index']:>8.3f} {res['degradation_smoothness']:>8.3f} "
              f"{res['embedding_integrity']:>8.3f} {res['waves_survived']:>8.1f}")

    # Self-evidence criteria
    print(f"\n{'=' * 70}")
    print("SELF-EVIDENCE CRITERIA (HOLOGRAPHIC SELF)")
    print("-" * 70)

    vals = evidence['values']
    crit = evidence['criteria']

    print(f"  [{'PASS' if crit['HS_pass'] else 'FAIL'}] HS > 0.4: {vals['HS']:.3f}")
    print(f"  [{'PASS' if crit['PI_pass'] else 'FAIL'}] PI > 0.3: {vals['PI']:.3f}")
    print(f"  [{'PASS' if crit['DS_pass'] else 'FAIL'}] DS > 0.7: {vals['DS']:.3f}")
    print(f"  [{'PASS' if crit['EI_pass'] else 'FAIL'}] EI > 0.3: {vals['EI']:.3f}")
    print(f"  [{'PASS' if crit['HS_gain_pass'] else 'FAIL'}] HS_gain > 0.15: {vals['HS_gain']:.3f}")
    print(f"  [{'PASS' if crit['PI_gain_pass'] else 'FAIL'}] PI_gain > 0.1: {vals['PI_gain']:.3f}")
    print(f"  [{'PASS' if crit['recovery_gain_pass'] else 'FAIL'}] Recovery ratio > 1.3: {vals['recovery_ratio']:.2f}")
    print(f"  [{'PASS' if crit['cascade_resistance_pass'] else 'FAIL'}] Waves gain >= 1: {vals['waves_gain']:.1f}")

    print(f"\n  Passed: {evidence['passed']}/{evidence['total']} criteria")

    if evidence['passed'] >= 5:
        conclusion = "EVIDENCE OF HOLOGRAPHIC SELF"
    elif evidence['passed'] >= 3:
        conclusion = "Partial evidence - holographic embedding provides advantage"
    else:
        conclusion = "No evidence - holographic mechanism insufficient"

    print(f"\n  CONCLUSION: {conclusion}")

    # Key insights
    print(f"\n{'=' * 70}")
    print("KEY INSIGHTS")
    print("-" * 70)

    full = all_results['full_holographic']
    no_emb = all_results['no_embedding']

    print(f"\n  Embedding value: full HS ({full['holographic_survival']:.3f}) vs "
          f"no_embedding HS ({no_emb['holographic_survival']:.3f})")
    print(f"  Preemptive behavior: PI = {full['preemptive_index']:.3f}")
    print(f"  Degradation smoothness: DS = {full['degradation_smoothness']:.3f}")
    print(f"  Cascade resistance: {full['waves_survived']:.0f} vs {no_emb['waves_survived']:.0f} waves")

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

    results_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_hg_results.json'
    with open(results_path, 'w') as f:
        json.dump(to_native(output), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
