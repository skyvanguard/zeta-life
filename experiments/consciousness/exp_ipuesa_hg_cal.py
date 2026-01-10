"""
IPUESA-HG-Calibrated: Holographic Self Goldilocks Zone
=======================================================

Calibration experiment to find optimal stress parameters where holographic
embeddings show differentiated resilience.

Target:
- full_hg survival: 30-70%
- no_emb survival: 10-40%
- Clear differentiation (DE > 0.15)

Key changes from HG+:
- Damage multiplier: 1.4× (binary searchable)
- Residual factor: 0.05-0.10 (reduced from 0.15-0.30)
- Wave interval: 10 steps
- New metric: DE (Differentiation Effectiveness)

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

    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'CognitiveArchitecture':
        return cls(attention_immediate=float(np.clip(v[0], 0, 1)),
                  attention_history=float(np.clip(v[1], 0, 1)),
                  attention_prediction=float(np.clip(v[2], 0, 1)),
                  memory_update_rate=float(np.clip(v[3], 0, 1)),
                  perceptual_gain=float(np.clip(v[4], 0.1, 2)))


@dataclass
class MicroModule:
    """Emergent micro-module"""
    module_type: str
    strength: float = 0.5
    activation_count: int = 0

    def apply(self, context: Dict) -> float:
        self.activation_count += 1
        effects = {
            'pattern_detector': 0.2,
            'threat_filter': 0.18,
            'recovery_accelerator': 0.25,
            'embedding_protector': 0.3,
            'cascade_breaker': 0.22,
            'residual_cleaner': 0.2,
        }
        return self.strength * effects.get(self.module_type, 0.1)


class DegradationState(Enum):
    OPTIMAL = 'optimal'
    STRESSED = 'stressed'
    IMPAIRED = 'impaired'
    CRITICAL = 'critical'
    COLLAPSED = 'collapsed'


@dataclass
class HolographicAgent:
    """Agent with holographic embeddings"""
    agent_id: int
    cluster_id: int

    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    modules: List[MicroModule] = field(default_factory=list)

    IC_t: float = 1.0

    cluster_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    collective_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    embedding_dim: int = 8
    embedding_staleness: float = 0.0

    threat_anticipation: float = 0.0
    protective_stance: float = 0.0

    residual_damage: float = 0.0
    structural_corruption: float = 0.0

    history_corruption: float = 0.0
    prediction_noise: float = 0.0

    IC_history: List[float] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    preemptive_actions: int = 0
    reactive_actions: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0

    def get_degradation_state(self) -> DegradationState:
        embedding_integrity = self.get_embedding_integrity()
        module_health = self.get_module_health()
        residual_penalty = self.residual_damage * 0.25
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
        if self.embedding_dim == 0:
            return 0.0
        cluster_norm = np.linalg.norm(self.cluster_embedding)
        collective_norm = np.linalg.norm(self.collective_embedding)
        staleness_penalty = self.embedding_staleness * 0.25
        structural_penalty = self.structural_corruption * 0.4
        base = (cluster_norm + collective_norm) / 4
        return max(0, min(1, base - staleness_penalty - structural_penalty))

    def get_module_health(self) -> float:
        if not self.modules:
            return 0.5
        return np.mean([m.strength for m in self.modules])

    def is_alive(self) -> bool:
        return self.IC_t > 0.1


@dataclass
class ClusterState:
    cluster_id: int
    member_ids: Set[int] = field(default_factory=set)
    theta_cluster: MetaPolicy = field(default_factory=MetaPolicy)
    alpha_cluster: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    cohesion: float = 0.5
    threat_level: float = 0.0


@dataclass
class CollectiveState:
    theta_collective: MetaPolicy = field(default_factory=MetaPolicy)
    alpha_collective: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    global_coherence: float = 0.5
    threat_level: float = 0.0


# =============================================================================
# HOLOGRAPHIC EMBEDDING SYSTEM
# =============================================================================

def encode_to_embedding(theta: MetaPolicy, alpha: CognitiveArchitecture,
                        threat: float, cohesion: float, dim: int = 8) -> np.ndarray:
    full = np.array([
        theta.risk_aversion, theta.exploration_rate,
        theta.memory_depth, theta.prediction_weight,
        alpha.attention_prediction, alpha.perceptual_gain,
        threat, cohesion
    ])
    return full[:dim] if dim < 8 else full


def decode_threat_from_embedding(embedding: np.ndarray) -> float:
    if len(embedding) >= 7:
        return float(embedding[6])
    elif len(embedding) >= 4:
        return float(np.mean(embedding) * 0.5)
    return 0.0


def sync_agent_embeddings(agent: HolographicAgent,
                          cluster: ClusterState,
                          collective: CollectiveState):
    dim = agent.embedding_dim
    if dim > 0:
        agent.cluster_embedding = encode_to_embedding(
            cluster.theta_cluster, cluster.alpha_cluster,
            cluster.threat_level, cluster.cohesion, dim
        )
        agent.collective_embedding = encode_to_embedding(
            collective.theta_collective, collective.alpha_collective,
            collective.threat_level, collective.global_coherence, dim
        )
        agent.embedding_staleness = 0.0


def aggregate_cluster_state(agents: List[HolographicAgent], cluster_id: int) -> ClusterState:
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
    active = [c for c in clusters if len(c.member_ids) > 0]
    if not active:
        return CollectiveState()

    weights = [c.cohesion * len(c.member_ids) for c in active]
    total = sum(weights)
    if total < 0.01:
        total = 1.0

    theta_vec = sum(c.theta_cluster.to_vector() * w for c, w in zip(active, weights)) / total
    alpha_vec = sum(c.alpha_cluster.to_vector() * w for c, w in zip(active, weights)) / total

    return CollectiveState(
        theta_collective=MetaPolicy.from_vector(theta_vec),
        alpha_collective=CognitiveArchitecture.from_vector(alpha_vec),
        global_coherence=np.mean([c.cohesion for c in active]),
        threat_level=np.mean([c.threat_level for c in active])
    )


# =============================================================================
# CALIBRATED CASCADING STORM
# =============================================================================

@dataclass
class PerturbationWave:
    wave_type: str
    base_damage: float
    step: int
    residual_factor: float = 0.0


@dataclass
class CalibratedStorm:
    """Calibrated storm with tunable parameters"""
    waves: List[PerturbationWave] = field(default_factory=list)
    amplification_factor: float = 1.15
    damage_multiplier: float = 1.4
    residual_base: float = 0.07
    wave_interval: int = 10
    waves_with_damage: int = 0
    total_damage_dealt: float = 0.0

    @classmethod
    def create(cls, start_step: int = 30,
               damage_mult: float = 1.4,
               residual_base: float = 0.07,
               interval: int = 10) -> 'CalibratedStorm':
        """Create calibrated storm with specified parameters"""
        # 6 wave types with REDUCED residual factors
        wave_types = [
            ('history', 0.3, residual_base),
            ('prediction', 0.25, residual_base * 0.8),
            ('social', 0.35, residual_base * 1.2),
            ('structural', 0.3, residual_base * 1.5),
            ('identity', 0.2, residual_base),
            ('catastrophic', 0.5, residual_base * 2.0)
        ]

        waves = []
        for i, (wtype, damage, residual) in enumerate(wave_types):
            waves.append(PerturbationWave(
                wave_type=wtype,
                base_damage=damage * damage_mult,
                step=start_step + i * interval,
                residual_factor=residual
            ))

        return cls(
            waves=waves,
            damage_multiplier=damage_mult,
            residual_base=residual_base,
            wave_interval=interval
        )


def apply_perturbation_wave(agents: List[HolographicAgent],
                            wave: PerturbationWave,
                            prior_damage_count: int,
                            amp_factor: float = 1.15) -> Tuple[float, int]:
    """Apply wave with calibrated damage"""
    effective_amp = wave.base_damage * (amp_factor ** prior_damage_count)

    total_damage = 0.0
    damaged_count = 0

    for agent in agents:
        if not agent.is_alive():
            continue

        # Calculate resistance
        resistance = agent.protective_stance * 0.35
        effective_damage = effective_amp

        for module in agent.modules:
            if module.module_type == 'threat_filter':
                resistance += module.apply({})
            elif module.module_type == 'cascade_breaker':
                effective_damage *= (1 - module.apply({}) * 0.5)
            elif module.module_type == 'embedding_protector' and wave.wave_type in ['social', 'structural']:
                resistance += module.apply({}) * 1.2

        actual_damage = max(0, effective_damage - resistance)

        # Apply damage by type
        if wave.wave_type == 'history':
            agent.history_corruption += actual_damage
            agent.IC_t -= actual_damage * 0.35

        elif wave.wave_type == 'prediction':
            agent.prediction_noise += actual_damage
            agent.alpha.attention_prediction *= (1 - actual_damage * 0.5)

        elif wave.wave_type == 'social':
            if agent.embedding_dim > 0:
                noise = np.random.randn(agent.embedding_dim) * actual_damage * 0.8
                agent.cluster_embedding += noise
                agent.collective_embedding += noise
            agent.embedding_staleness += actual_damage

        elif wave.wave_type == 'structural':
            if agent.embedding_dim > 0:
                agent.cluster_embedding *= (1 - actual_damage * 0.25)
                agent.collective_embedding *= (1 - actual_damage * 0.25)
            agent.structural_corruption += actual_damage * 0.4
            agent.IC_t -= actual_damage * 0.15

        elif wave.wave_type == 'identity':
            agent.IC_t -= actual_damage * 0.8

        elif wave.wave_type == 'catastrophic':
            agent.IC_t -= actual_damage * 0.4
            agent.history_corruption += actual_damage * 0.3
            agent.prediction_noise += actual_damage * 0.3
            agent.embedding_staleness += actual_damage * 0.4
            agent.structural_corruption += actual_damage * 0.2

        # Calibrated residual (much lower than HG+)
        residual = actual_damage * wave.residual_factor
        agent.residual_damage += residual

        # Clamp values
        agent.IC_t = max(0, min(1, agent.IC_t))
        agent.residual_damage = min(0.6, agent.residual_damage)  # Lower cap

        if actual_damage > 0.05:
            damaged_count += 1
            total_damage += actual_damage

    return total_damage, damaged_count


# =============================================================================
# PROACTIVE SELF-MAINTENANCE
# =============================================================================

def anticipate_threats(agent: HolographicAgent, environment_threat: float) -> float:
    local_threat = environment_threat * agent.alpha.attention_prediction

    if agent.embedding_dim > 0:
        cluster_threat = decode_threat_from_embedding(agent.cluster_embedding)
        collective_threat = decode_threat_from_embedding(agent.collective_embedding)
        embedding_factor = agent.embedding_dim / 8.0
    else:
        cluster_threat = 0.0
        collective_threat = 0.0
        embedding_factor = 0.0

    agent.threat_anticipation = (
        0.5 * local_threat +
        0.3 * cluster_threat * embedding_factor +
        0.2 * collective_threat * embedding_factor
    )

    for module in agent.modules:
        if module.module_type == 'pattern_detector':
            agent.threat_anticipation += module.apply({})

    return agent.threat_anticipation


def take_protective_action(agent: HolographicAgent,
                           clusters: Dict[int, ClusterState],
                           collective: CollectiveState,
                           is_preemptive: bool) -> Optional[str]:
    threat = agent.threat_anticipation
    state = agent.get_degradation_state()

    action_taken = None

    # Emergency module
    if threat > 0.6 and state in [DegradationState.CRITICAL, DegradationState.IMPAIRED]:
        if agent.IC_t > 0.15 and len(agent.modules) < 5:
            module_types = ['cascade_breaker', 'embedding_protector', 'residual_cleaner', 'recovery_accelerator']
            module_type = np.random.choice(module_types)
            agent.modules.append(MicroModule(module_type=module_type, strength=0.55))
            agent.IC_t -= 0.08
            action_taken = 'emergency_module'

    # Isolation
    elif threat > 0.55 and state == DegradationState.CRITICAL:
        agent.cluster_id = -1
        action_taken = 'isolate'

    # Sync embeddings
    elif threat > 0.35 and agent.embedding_staleness > 0.2 and agent.embedding_dim > 0:
        if agent.cluster_id >= 0 and agent.cluster_id in clusters:
            sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)
            agent.IC_t -= 0.02
            action_taken = 'sync_embeddings'

    # Harden
    elif threat > 0.2:
        agent.protective_stance = min(1.0, agent.protective_stance + 0.35)
        agent.theta.exploration_rate = max(0, agent.theta.exploration_rate - 0.08)
        action_taken = 'harden'

    if action_taken:
        agent.actions_taken.append(action_taken)
        if is_preemptive:
            agent.preemptive_actions += 1
        else:
            agent.reactive_actions += 1

    return action_taken


# =============================================================================
# RECOVERY SYSTEM
# =============================================================================

def attempt_recovery(agent: HolographicAgent,
                     cluster: Optional[ClusterState],
                     base_rate: float = 0.05) -> Tuple[float, bool]:
    if not agent.is_alive():
        return 0.0, False

    agent.recovery_attempts += 1

    embedding_integrity = agent.get_embedding_integrity()
    cluster_support = cluster.cohesion if cluster else 0.2

    # Recovery rate with embedding bonus
    residual_penalty = agent.residual_damage * 0.4
    recovery_rate = base_rate * (1 + embedding_integrity * 0.8) * (1 + cluster_support * 0.4) * (1 - residual_penalty)

    for module in agent.modules:
        if module.module_type == 'recovery_accelerator':
            recovery_rate *= (1 + module.apply({}))
        elif module.module_type == 'residual_cleaner':
            agent.residual_damage *= (1 - module.apply({}) * 0.15)

    recovery = min(1.0 - agent.IC_t, recovery_rate)
    pre_IC = agent.IC_t
    agent.IC_t += recovery

    # Gradual recovery of damages
    agent.history_corruption *= 0.93
    agent.prediction_noise *= 0.93
    agent.embedding_staleness *= 0.94
    agent.structural_corruption *= 0.96
    agent.residual_damage *= 0.97  # Slow but possible recovery

    success = agent.IC_t > pre_IC + 0.01
    if success:
        agent.successful_recoveries += 1

    return recovery, success


def apply_degradation_effects(agent: HolographicAgent):
    state = agent.get_degradation_state()

    if state == DegradationState.STRESSED:
        agent.theta.risk_aversion = min(1.0, agent.theta.risk_aversion + 0.12)
        agent.theta.exploration_rate = max(0, agent.theta.exploration_rate - 0.06)

    elif state == DegradationState.IMPAIRED:
        agent.alpha.attention_prediction *= 0.88
        agent.modules = [m for m in agent.modules if m.strength > 0.25]

    elif state == DegradationState.CRITICAL:
        agent.theta.risk_aversion = 1.0
        agent.theta.exploration_rate = 0.0

    elif state == DegradationState.COLLAPSED:
        agent.theta = MetaPolicy(
            risk_aversion=np.random.random(),
            exploration_rate=np.random.random() * 0.2,
            memory_depth=np.random.random() * 0.3,
            prediction_weight=np.random.random() * 0.3
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(agents: List[HolographicAgent], damage_history: List[float]) -> Dict:
    alive = [a for a in agents if a.is_alive()]

    hs = len(alive) / len(agents) if agents else 0.0

    total_preemptive = sum(a.preemptive_actions for a in agents)
    total_reactive = sum(a.reactive_actions for a in agents)
    pi = total_preemptive / (total_preemptive + total_reactive) if (total_preemptive + total_reactive) > 0 else 0.0

    max_derivatives = []
    for agent in agents:
        if len(agent.IC_history) >= 2:
            derivatives = np.abs(np.diff(agent.IC_history))
            if len(derivatives) > 0:
                max_derivatives.append(np.max(derivatives))
    ds = max(0, 1 - np.mean(max_derivatives) * 2) if max_derivatives else 1.0

    ei = np.mean([a.get_embedding_integrity() for a in alive]) if alive else 0.0

    total_attempts = sum(a.recovery_attempts for a in agents)
    total_successes = sum(a.successful_recoveries for a in agents)
    rs = total_successes / total_attempts if total_attempts > 0 else 0.0

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
        'degradation_smoothness': ds,
        'embedding_integrity': ei,
        'recovery_score': rs,
        'correlation_emergence': ce,
        'residual_burden': residual,
        'final_alive': len(alive),
        'modules_created': sum(len(a.modules) for a in agents),
        'preemptive_total': total_preemptive,
        'reactive_total': total_reactive
    }


# =============================================================================
# SIMULATION
# =============================================================================

def initialize_agents(n_agents: int, n_clusters: int,
                      use_embeddings: bool = True,
                      embedding_dim: int = 8) -> List[HolographicAgent]:
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


def run_episode(n_agents: int = 24,
                n_clusters: int = 4,
                n_steps: int = 150,
                damage_mult: float = 1.4,
                residual_base: float = 0.07,
                use_embeddings: bool = True,
                embedding_dim: int = 8,
                sync_interval: int = 8) -> Dict:

    agents = initialize_agents(n_agents, n_clusters, use_embeddings, embedding_dim)
    clusters = {i: aggregate_cluster_state(agents, i) for i in range(n_clusters)}
    collective = aggregate_collective_state(list(clusters.values()))

    if use_embeddings:
        for agent in agents:
            if agent.cluster_id >= 0:
                sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)

    storm = CalibratedStorm.create(
        start_step=30,
        damage_mult=damage_mult,
        residual_base=residual_base,
        interval=10
    )

    damage_history = []

    # Imprinting phase
    for step in range(20):
        for agent in agents:
            agent.IC_history.append(agent.IC_t)

    # Main simulation
    for step in range(20, n_steps):
        # Update staleness
        for agent in agents:
            if agent.is_alive():
                agent.embedding_staleness += 0.025

        # Sync embeddings periodically
        if use_embeddings and step % sync_interval == 0:
            for agent in agents:
                if agent.is_alive() and agent.cluster_id >= 0 and agent.cluster_id in clusters:
                    sync_agent_embeddings(agent, clusters[agent.cluster_id], collective)

        # Check for wave
        current_wave = None
        for wave in storm.waves:
            if wave.step == step:
                current_wave = wave
                break

        # Anticipate threats
        env_threat = 0.2
        if current_wave:
            env_threat = 0.35 + current_wave.base_damage * 0.5

        for agent in agents:
            if agent.is_alive():
                anticipate_threats(agent, env_threat)

        # Preemptive actions
        if current_wave or env_threat > 0.25:
            for agent in agents:
                if agent.is_alive():
                    take_protective_action(agent, clusters, collective, is_preemptive=True)

        # Apply wave
        if current_wave:
            damage, _ = apply_perturbation_wave(
                agents, current_wave, storm.waves_with_damage, storm.amplification_factor
            )
            if damage > 0:
                storm.waves_with_damage += 1
                storm.total_damage_dealt += damage
                damage_history.append(damage)

        # Reactive actions
        for agent in agents:
            if agent.is_alive():
                if agent.IC_t < 0.55 or agent.embedding_staleness > 0.35 or agent.residual_damage > 0.15:
                    take_protective_action(agent, clusters, collective, is_preemptive=False)

        # Degradation effects
        for agent in agents:
            if agent.is_alive():
                apply_degradation_effects(agent)

        # Recovery
        for agent in agents:
            if agent.is_alive() and agent.cluster_id >= 0:
                cluster = clusters.get(agent.cluster_id)
                attempt_recovery(agent, cluster)

        # Track IC
        for agent in agents:
            agent.IC_history.append(agent.IC_t)

        # Update hierarchy
        cluster_ids = set(a.cluster_id for a in agents if a.cluster_id >= 0)
        for cid in cluster_ids:
            clusters[cid] = aggregate_cluster_state(agents, cid)
        collective_new = aggregate_collective_state(list(clusters.values()))
        collective.theta_collective = collective_new.theta_collective
        collective.alpha_collective = collective_new.alpha_collective
        collective.global_coherence = collective_new.global_coherence
        collective.threat_level = collective_new.threat_level

    metrics = calculate_metrics(agents, damage_history)
    metrics['total_damage'] = storm.total_damage_dealt
    return metrics


def run_condition(condition: str, n_runs: int = 8, n_agents: int = 24,
                  n_clusters: int = 4, n_steps: int = 150,
                  damage_mult: float = 1.4) -> Dict:

    if condition == 'full_hg':
        use_embeddings = True
        embedding_dim = 8
        residual_base = 0.07
    elif condition == 'no_emb':
        use_embeddings = False
        embedding_dim = 0
        residual_base = 0.07
    elif condition == 'partial_hg':
        use_embeddings = True
        embedding_dim = 4
        residual_base = 0.07
    elif condition == 'high_stress':
        use_embeddings = True
        embedding_dim = 8
        residual_base = 0.10
        damage_mult = 1.5
    else:
        raise ValueError(f"Unknown condition: {condition}")

    all_results = []
    for _ in range(n_runs):
        results = run_episode(n_agents, n_clusters, n_steps,
                             damage_mult, residual_base,
                             use_embeddings, embedding_dim)
        all_results.append(results)

    aggregated = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        aggregated[key] = float(np.mean(values))

    return aggregated


def calibration_search(n_runs: int = 6) -> Dict:
    """Binary search to find optimal damage multiplier"""
    print("\n" + "=" * 60)
    print("CALIBRATION SEARCH: Finding Goldilocks Zone")
    print("=" * 60)

    target_full_min, target_full_max = 0.30, 0.70
    target_no_min, target_no_max = 0.10, 0.40

    results_history = []

    # Test range of damage multipliers (fine-tuned around Goldilocks zone)
    test_mults = [2.1, 2.2, 2.3, 2.35, 2.4]

    for mult in test_mults:
        print(f"\n  Testing damage_mult = {mult:.1f}...")

        full_hs = []
        no_hs = []

        for _ in range(n_runs):
            full_res = run_episode(damage_mult=mult, use_embeddings=True, embedding_dim=8)
            no_res = run_episode(damage_mult=mult, use_embeddings=False, embedding_dim=0)
            full_hs.append(full_res['holographic_survival'])
            no_hs.append(no_res['holographic_survival'])

        full_mean = np.mean(full_hs)
        no_mean = np.mean(no_hs)
        diff = full_mean - no_mean

        results_history.append({
            'damage_mult': mult,
            'full_hs': full_mean,
            'no_hs': no_mean,
            'diff': diff,
            'in_target': (target_full_min <= full_mean <= target_full_max and
                         target_no_min <= no_mean <= target_no_max)
        })

        print(f"    full_hg: {full_mean:.3f}, no_emb: {no_mean:.3f}, diff: {diff:.3f}")

    # Find best
    best = None
    best_score = -1

    for r in results_history:
        if r['in_target']:
            score = r['diff']
            if score > best_score:
                best_score = score
                best = r

    if best is None:
        # Find closest to target
        for r in results_history:
            dist = abs(r['full_hs'] - 0.5) + abs(r['no_hs'] - 0.25)
            score = 1 / (dist + 0.01)
            if score > best_score:
                best_score = score
                best = r

    return {
        'history': results_history,
        'best': best,
        'optimal_damage_mult': best['damage_mult'] if best else 1.4
    }


# =============================================================================
# SELF-EVIDENCE EVALUATION
# =============================================================================

def evaluate_self_evidence(results: Dict[str, Dict]) -> Dict:
    criteria = {}

    full = results.get('full_hg', {})
    no_emb = results.get('no_emb', {})
    partial = results.get('partial_hg', {})

    # Primary metrics
    criteria['HS_in_range'] = 0.30 <= full.get('holographic_survival', 0) <= 0.70
    criteria['PI_pass'] = full.get('preemptive_index', 0) > 0.15
    criteria['DS_pass'] = full.get('degradation_smoothness', 0) > 0.4
    criteria['EI_pass'] = full.get('embedding_integrity', 0) > 0.3
    criteria['RS_pass'] = full.get('recovery_score', 0) > 0.25
    criteria['CE_pass'] = full.get('correlation_emergence', 0) > 0.15

    # Differentiation
    full_hs = full.get('holographic_survival', 0)
    no_hs = no_emb.get('holographic_survival', 0)
    de = full_hs - no_hs
    criteria['DE_pass'] = de > 0.15

    # Gradient
    partial_hs = partial.get('holographic_survival', 0)
    criteria['gradient_pass'] = (no_hs < partial_hs) and (partial_hs <= full_hs + 0.05)

    passed = sum(1 for v in criteria.values() if v)

    return {
        'criteria': criteria,
        'passed': passed,
        'total': 8,
        'values': {
            'HS_full': full_hs,
            'HS_no_emb': no_hs,
            'HS_partial': partial_hs,
            'DE': de,
            'PI': full.get('preemptive_index', 0),
            'DS': full.get('degradation_smoothness', 0),
            'EI_full': full.get('embedding_integrity', 0),
            'RS': full.get('recovery_score', 0),
            'CE': full.get('correlation_emergence', 0),
            'residual_full': full.get('residual_burden', 0),
            'residual_no_emb': no_emb.get('residual_burden', 0)
        }
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
    print("IPUESA-HG-Calibrated: Holographic Self Goldilocks Zone")
    print("        Finding Optimal Stress Parameters")
    print("=" * 70)

    config = {
        'n_agents': 24,
        'n_clusters': 4,
        'n_steps': 150,
        'n_runs': 8,
        'target_full_hs': '30-70%',
        'target_no_emb_hs': '10-40%',
        'residual_base': 0.07
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # First, calibration search
    calibration = calibration_search(n_runs=4)
    optimal_mult = calibration['optimal_damage_mult']

    print(f"\n{'=' * 60}")
    print(f"Using optimal damage_mult: {optimal_mult}")
    print("=" * 60)

    # Run full experiment with optimal parameters
    conditions = ['full_hg', 'no_emb', 'partial_hg', 'high_stress']
    all_results = {}

    for condition in conditions:
        print(f"\n{'=' * 60}")
        print(f"Running IPUESA-HG-Cal - Condition: {condition}")
        print("=" * 60)

        results = run_condition(
            condition,
            n_runs=config['n_runs'],
            n_agents=config['n_agents'],
            n_clusters=config['n_clusters'],
            n_steps=config['n_steps'],
            damage_mult=optimal_mult
        )

        all_results[condition] = results

        print(f"\nResults - {condition}:")
        print(f"  HS = {results['holographic_survival']:.3f}")
        print(f"  PI = {results['preemptive_index']:.3f}")
        print(f"  DS = {results['degradation_smoothness']:.3f}")
        print(f"  EI = {results['embedding_integrity']:.3f}")
        print(f"  RS = {results['recovery_score']:.3f}")
        print(f"  Residual = {results['residual_burden']:.3f}")

    # Evaluate
    evidence = evaluate_self_evidence(all_results)

    # Summary
    print(f"\n{'=' * 70}")
    print("IPUESA-HG-Calibrated: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<15} {'HS':>8} {'PI':>8} {'DS':>8} {'EI':>8} {'RS':>8} {'Resid':>8}")
    print("-" * 70)
    for cond, res in all_results.items():
        print(f"{cond:<15} {res['holographic_survival']:>8.3f} "
              f"{res['preemptive_index']:>8.3f} {res['degradation_smoothness']:>8.3f} "
              f"{res['embedding_integrity']:>8.3f} {res['recovery_score']:>8.3f} "
              f"{res['residual_burden']:>8.3f}")

    # Self-evidence
    print(f"\n{'=' * 70}")
    print("SELF-EVIDENCE CRITERIA (CALIBRATED HOLOGRAPHIC SELF)")
    print("-" * 70)

    vals = evidence['values']
    crit = evidence['criteria']

    print(f"  [{'PASS' if crit['HS_in_range'] else 'FAIL'}] HS in [0.30, 0.70]: {vals['HS_full']:.3f}")
    print(f"  [{'PASS' if crit['PI_pass'] else 'FAIL'}] PI > 0.15: {vals['PI']:.3f}")
    print(f"  [{'PASS' if crit['DS_pass'] else 'FAIL'}] DS > 0.4: {vals['DS']:.3f}")
    print(f"  [{'PASS' if crit['EI_pass'] else 'FAIL'}] EI > 0.3: {vals['EI_full']:.3f}")
    print(f"  [{'PASS' if crit['RS_pass'] else 'FAIL'}] RS > 0.25: {vals['RS']:.3f}")
    print(f"  [{'PASS' if crit['CE_pass'] else 'FAIL'}] CE > 0.15: {vals['CE']:.3f}")
    print(f"  [{'PASS' if crit['DE_pass'] else 'FAIL'}] DE (full - no_emb) > 0.15: {vals['DE']:.3f}")
    print(f"  [{'PASS' if crit['gradient_pass'] else 'FAIL'}] Gradient: no_emb < partial <= full")

    print(f"\n  Passed: {evidence['passed']}/{evidence['total']} criteria")

    if evidence['passed'] >= 6:
        conclusion = "STRONG EVIDENCE OF HOLOGRAPHIC SELF"
    elif evidence['passed'] >= 4:
        conclusion = "Evidence of holographic self-maintenance"
    elif evidence['passed'] >= 2:
        conclusion = "Partial evidence - embedding provides advantage"
    else:
        conclusion = "Calibration needed - parameters not yet optimal"

    print(f"\n  CONCLUSION: {conclusion}")

    # Key insights
    print(f"\n{'=' * 70}")
    print("KEY INSIGHTS")
    print("-" * 70)
    print(f"\n  Optimal damage multiplier: {optimal_mult:.1f}x")
    print(f"\n  Survival differentiation (DE):")
    print(f"    full_hg:    {vals['HS_full']:.3f}")
    print(f"    partial_hg: {vals['HS_partial']:.3f}")
    print(f"    no_emb:     {vals['HS_no_emb']:.3f}")
    print(f"    DE (diff):  {vals['DE']:.3f}")
    print(f"\n  Embedding preserves integrity under calibrated stress:")
    print(f"    full_hg EI: {vals['EI_full']:.3f}")

    # Save
    output = {
        'config': config,
        'calibration': to_native(calibration),
        'optimal_damage_mult': optimal_mult,
        'metrics': all_results,
        'self_evidence': {
            'passed': evidence['passed'],
            'total': evidence['total'],
            'criteria': {k: bool(v) for k, v in evidence['criteria'].items()},
            'values': evidence['values'],
            'conclusion': conclusion
        }
    }

    results_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_hg_cal_results.json'
    with open(results_path, 'w') as f:
        json.dump(to_native(output), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
