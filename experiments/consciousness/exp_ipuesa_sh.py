#!/usr/bin/env python3
"""
IPUESA-SH: Self-Hierarchy Experiment

Tests whether hierarchical identity can emerge from local interactions without
explicit programming. Each agent maintains individual identity (theta/alpha/beta)
and participates in cluster and collective identities.

Key innovation: Three-level identity hierarchy with bi-directional influence:
- Individual (theta_i, alpha_i, beta_i)
- Cluster (aggregated from members)
- Collective (aggregated from clusters)

Hypothesis: Vertical coherence emerges when levels are aligned, and hierarchical
resilience exceeds flat structures under perturbation.

Metrics:
- VC: Vertical Coherence (alignment across levels)
- HR: Hierarchical Resilience (identity survival at each level)
- ED: Emergent Diversity (functional differentiation)
- AD: Alignment/Dissonance (level agreement on decisions)

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
class SHConfig:
    """Configuration for Self-Hierarchy experiment."""
    # Population
    n_agents: int = 24
    n_initial_clusters: int = 4
    min_cluster_size: int = 3
    max_cluster_size: int = 8

    # Existential thresholds
    epsilon: float = 0.15
    initial_ic: float = 1.0
    cluster_ic_threshold: float = 0.2
    collective_ic_threshold: float = 0.25

    # Hierarchy parameters
    aggregation_frequency: int = 5
    top_down_strength: float = 0.25
    bottom_up_weight: float = 0.7
    update_freq: int = 5

    # Dynamic clustering
    cohesion_threshold_split: float = 0.3
    cohesion_threshold_merge: float = 0.8
    migration_threshold: float = 0.4
    similarity_threshold: float = 0.6

    # Autonomy-Conformity
    initial_autonomy: float = 0.5
    initial_conformity: float = 0.5

    # Learning
    theta_lr: float = 0.06
    alpha_lr: float = 0.03

    # Experiment parameters
    n_steps: int = 150
    n_episodes: int = 6
    n_runs: int = 6

    # Decision tracking
    decision_interval: int = 15


# =============================================================================
# Meta-Policy (theta) - WHO
# =============================================================================

@dataclass
class MetaPolicy:
    """Meta-policy parameters."""
    risk_aversion: float = 0.5
    exploration_rate: float = 0.3
    memory_depth: float = 0.5
    prediction_weight: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([self.risk_aversion, self.exploration_rate,
                        self.memory_depth, self.prediction_weight])

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
# Cognitive Architecture (alpha) - HOW
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
        return np.array([self.attention_history, self.attention_prediction,
                        self.attention_immediate, self.memory_update_rate,
                        self.perceptual_gain])

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
        i = max(0.1, 1 - h - p)
        return cls(
            attention_history=h, attention_prediction=p, attention_immediate=i,
            memory_update_rate=random.uniform(0.3, 0.7),
            perceptual_gain=random.uniform(0.4, 0.8)
        )


# =============================================================================
# Individual Identity
# =============================================================================

@dataclass
class IndividualIdentity:
    """Level 0: Single agent identity."""
    agent_id: int
    theta: MetaPolicy
    alpha: CognitiveArchitecture
    IC_t: float = 1.0

    # Perturbation effects
    history_corruption: float = 0.0
    prediction_noise: float = 0.0


# =============================================================================
# Cluster Identity
# =============================================================================

CLUSTER_SPECIALIZATIONS = ['explorers', 'defenders', 'cooperators', 'balanced', 'adaptive']


@dataclass
class ClusterIdentity:
    """Level 1: Emergent subgroup identity."""
    cluster_id: int
    member_ids: Set[int] = field(default_factory=set)

    theta_cluster: MetaPolicy = field(default_factory=MetaPolicy)
    alpha_cluster: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)

    cohesion: float = 0.5
    specialization: str = 'balanced'
    IC_cluster: float = 1.0


def determine_cluster_specialization(theta: MetaPolicy) -> str:
    """Determine cluster role from aggregated theta."""
    if theta.exploration_rate > 0.55:
        return 'explorers'
    elif theta.risk_aversion > 0.6:
        return 'defenders'
    elif theta.memory_depth > 0.55:
        return 'cooperators'
    elif theta.prediction_weight > 0.55:
        return 'adaptive'
    else:
        return 'balanced'


# =============================================================================
# Collective Identity
# =============================================================================

@dataclass
class CollectiveIdentity:
    """Level 2: Global population identity."""
    theta_collective: MetaPolicy = field(default_factory=MetaPolicy)
    alpha_collective: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)

    global_coherence: float = 0.5
    collective_purpose: str = 'survival'
    IC_collective: float = 1.0


# =============================================================================
# Hierarchical Agent
# =============================================================================

@dataclass
class HierarchicalAgent:
    """Agent aware of its position in identity hierarchy."""
    individual: IndividualIdentity
    config: SHConfig

    cluster_id: Optional[int] = None
    alignment_to_cluster: float = 0.5
    alignment_to_collective: float = 0.5

    autonomy: float = 0.5
    conformity: float = 0.5

    seeking_new_cluster: bool = False
    resources: float = 5.0

    def is_alive(self) -> bool:
        return self.individual.IC_t > self.config.epsilon

    def execute_action(self, action: str) -> float:
        """Execute action and return reward."""
        if action == 'R':
            ic_cost = 0.05
            reward = 7.0
        else:
            ic_cost = 0.008
            reward = 3.0

        self.individual.IC_t = max(0, self.individual.IC_t - ic_cost)
        self.resources += reward * 0.1
        return reward

    def select_hierarchical_action(self, context: Dict, config: SHConfig) -> str:
        """Select action based on individual + hierarchy influence."""
        if not self.is_alive():
            return random.choice(['R', 'S'])

        # Base decision from individual theta
        ind_preference = self.individual.theta.risk_aversion

        # Cluster influence
        cluster_preference = context.get('cluster_risk_aversion', 0.5)

        # Collective influence
        collective_preference = context.get('collective_risk_aversion', 0.5)

        # Weighted combination based on autonomy/conformity
        total_weight = self.autonomy + self.conformity
        if total_weight > 0:
            ind_weight = self.autonomy / total_weight
            hier_weight = self.conformity / total_weight
        else:
            ind_weight, hier_weight = 0.5, 0.5

        hier_preference = 0.6 * cluster_preference + 0.4 * collective_preference
        combined = ind_weight * ind_preference + hier_weight * hier_preference

        # Exploration
        if random.random() < self.individual.theta.exploration_rate * 0.5:
            return random.choice(['R', 'S'])

        return 'S' if combined > 0.5 else 'R'

    def update_adaptive_systems(self, context: Dict):
        """Update theta/alpha based on existential gradient."""
        if not self.is_alive():
            return

        margin = max(0.1, self.individual.IC_t - self.config.epsilon)
        urgency = 1.0 / margin

        # Theta gradient
        grad_theta = np.array([
            0.2 * urgency, -0.08 * urgency, 0.08, 0.12 * urgency
        ])
        theta_arr = self.individual.theta.to_array() + self.config.theta_lr * grad_theta
        self.individual.theta = MetaPolicy.from_array(theta_arr)

        # Alpha gradient
        grad_alpha = np.array([0.04, 0.1 * urgency, -0.06, 0.06, 0.08 * urgency])
        alpha_arr = self.individual.alpha.to_array() + self.config.alpha_lr * grad_alpha
        self.individual.alpha = CognitiveArchitecture.from_array(alpha_arr)

    def decay_effects(self):
        """Decay perturbation effects."""
        self.individual.history_corruption *= 0.95
        self.individual.prediction_noise *= 0.95

    def get_hierarchical_context(
        self,
        clusters: Dict[int, ClusterIdentity],
        collective: CollectiveIdentity,
        condition: str
    ) -> Dict:
        """Get context from hierarchy position."""
        context = {
            'individual_ic': self.individual.IC_t,
            'cluster_risk_aversion': 0.5,
            'collective_risk_aversion': 0.5
        }

        if condition != 'no_cluster' and self.cluster_id in clusters:
            cluster = clusters[self.cluster_id]
            context['cluster_risk_aversion'] = cluster.theta_cluster.risk_aversion
            context['cluster_cohesion'] = cluster.cohesion

        if condition != 'no_collective':
            context['collective_risk_aversion'] = collective.theta_collective.risk_aversion
            context['collective_coherence'] = collective.global_coherence

        return context

    @classmethod
    def create_random(cls, agent_id: int, config: SHConfig) -> 'HierarchicalAgent':
        """Create agent with random initial parameters."""
        individual = IndividualIdentity(
            agent_id=agent_id,
            theta=MetaPolicy.random(),
            alpha=CognitiveArchitecture.random(),
            IC_t=config.initial_ic
        )
        return cls(
            individual=individual,
            config=config,
            autonomy=config.initial_autonomy,
            conformity=config.initial_conformity
        )


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_to_cluster(members: List[HierarchicalAgent]) -> ClusterIdentity:
    """Aggregate individual identities into cluster identity."""
    if not members:
        return ClusterIdentity(cluster_id=-1)

    cluster_id = members[0].cluster_id

    # Weight by IC
    weights = np.array([max(0.1, m.individual.IC_t) for m in members])
    weights = weights / (weights.sum() + 1e-10)

    # Aggregate theta
    theta_arrays = np.array([m.individual.theta.to_array() for m in members])
    theta_cluster = MetaPolicy.from_array(np.average(theta_arrays, weights=weights, axis=0))

    # Aggregate alpha
    alpha_arrays = np.array([m.individual.alpha.to_array() for m in members])
    alpha_cluster = CognitiveArchitecture.from_array(np.average(alpha_arrays, weights=weights, axis=0))

    # Compute cohesion
    theta_variance = np.mean(np.var(theta_arrays, axis=0))
    cohesion = 1.0 / (1.0 + theta_variance * 10)

    # Specialization
    specialization = determine_cluster_specialization(theta_cluster)

    # Cluster IC
    IC_cluster = np.average([m.individual.IC_t for m in members], weights=weights) * cohesion

    return ClusterIdentity(
        cluster_id=cluster_id,
        member_ids={m.individual.agent_id for m in members},
        theta_cluster=theta_cluster,
        alpha_cluster=alpha_cluster,
        cohesion=cohesion,
        specialization=specialization,
        IC_cluster=IC_cluster
    )


def aggregate_to_collective(clusters: List[ClusterIdentity]) -> CollectiveIdentity:
    """Aggregate cluster identities into collective identity."""
    if not clusters:
        return CollectiveIdentity()

    # Weight by cluster IC and size
    weights = np.array([c.IC_cluster * len(c.member_ids) for c in clusters])
    weights = weights / (weights.sum() + 1e-10)

    # Aggregate theta
    theta_arrays = np.array([c.theta_cluster.to_array() for c in clusters])
    theta_collective = MetaPolicy.from_array(np.average(theta_arrays, weights=weights, axis=0))

    # Aggregate alpha
    alpha_arrays = np.array([c.alpha_cluster.to_array() for c in clusters])
    alpha_collective = CognitiveArchitecture.from_array(np.average(alpha_arrays, weights=weights, axis=0))

    # Global coherence
    mean_cohesion = np.mean([c.cohesion for c in clusters])
    inter_cluster_var = np.mean(np.var(theta_arrays, axis=0))
    inter_alignment = 1.0 / (1.0 + inter_cluster_var * 5)
    global_coherence = mean_cohesion * inter_alignment

    # Collective IC
    IC_collective = np.average([c.IC_cluster for c in clusters], weights=weights) * global_coherence

    return CollectiveIdentity(
        theta_collective=theta_collective,
        alpha_collective=alpha_collective,
        global_coherence=global_coherence,
        collective_purpose='survival' if IC_collective > 0.3 else 'fragmented',
        IC_collective=IC_collective
    )


def aggregate_individuals_to_collective(agents: List[HierarchicalAgent]) -> CollectiveIdentity:
    """Direct aggregation from individuals (no cluster level)."""
    if not agents:
        return CollectiveIdentity()

    weights = np.array([max(0.1, a.individual.IC_t) for a in agents])
    weights = weights / (weights.sum() + 1e-10)

    theta_arrays = np.array([a.individual.theta.to_array() for a in agents])
    theta_collective = MetaPolicy.from_array(np.average(theta_arrays, weights=weights, axis=0))

    alpha_arrays = np.array([a.individual.alpha.to_array() for a in agents])
    alpha_collective = CognitiveArchitecture.from_array(np.average(alpha_arrays, weights=weights, axis=0))

    variance = np.mean(np.var(theta_arrays, axis=0))
    global_coherence = 1.0 / (1.0 + variance * 8)

    IC_collective = np.average([a.individual.IC_t for a in agents], weights=weights) * global_coherence

    return CollectiveIdentity(
        theta_collective=theta_collective,
        alpha_collective=alpha_collective,
        global_coherence=global_coherence,
        IC_collective=IC_collective
    )


# =============================================================================
# Top-Down Influence
# =============================================================================

def apply_top_down_to_cluster(
    cluster: ClusterIdentity,
    collective: CollectiveIdentity,
    config: SHConfig
):
    """Apply collective influence to cluster."""
    theta_diff = collective.theta_collective.to_array() - cluster.theta_cluster.to_array()
    strength = config.top_down_strength * collective.global_coherence

    theta_arr = cluster.theta_cluster.to_array() + theta_diff * strength * 0.2
    cluster.theta_cluster = MetaPolicy.from_array(theta_arr)


def apply_top_down_to_agent(
    agent: HierarchicalAgent,
    cluster: ClusterIdentity,
    config: SHConfig
):
    """Apply cluster influence to agent."""
    theta_diff = cluster.theta_cluster.to_array() - agent.individual.theta.to_array()

    acceptance = agent.conformity * cluster.cohesion
    resistance = agent.autonomy

    effective = acceptance / (acceptance + resistance + 1e-10) * config.top_down_strength

    theta_arr = agent.individual.theta.to_array() + theta_diff * effective * 0.15
    agent.individual.theta = MetaPolicy.from_array(theta_arr)

    # Update alignment
    agent.alignment_to_cluster = 1.0 - min(1.0, np.linalg.norm(theta_diff))


# =============================================================================
# Dissonance
# =============================================================================

@dataclass
class DissonanceMetrics:
    """Measures misalignment across hierarchy levels."""
    ind_cluster_total: float = 0.0
    cluster_collective_total: float = 0.0
    vertical_coherence: float = 0.5
    dissonance_type: str = 'none'


def compute_dissonance(
    agent: HierarchicalAgent,
    cluster: ClusterIdentity,
    collective: CollectiveIdentity
) -> DissonanceMetrics:
    """Compute dissonance at all levels."""
    theta_i = agent.individual.theta.to_array()
    theta_c = cluster.theta_cluster.to_array()
    theta_col = collective.theta_collective.to_array()

    ind_cluster = np.linalg.norm(theta_i - theta_c)
    cluster_collective = np.linalg.norm(theta_c - theta_col)

    total = ind_cluster + cluster_collective
    vc = 1.0 / (1.0 + total * 3)

    if total < 0.3:
        dtype = 'none'
    elif ind_cluster > cluster_collective * 2:
        dtype = 'local'
    elif cluster_collective > ind_cluster * 2:
        dtype = 'systemic'
    else:
        dtype = 'crisis'

    return DissonanceMetrics(
        ind_cluster_total=ind_cluster,
        cluster_collective_total=cluster_collective,
        vertical_coherence=vc,
        dissonance_type=dtype
    )


def resolve_dissonance(
    agent: HierarchicalAgent,
    dissonance: DissonanceMetrics,
    config: SHConfig
):
    """Resolve dissonance through adaptation."""
    if dissonance.dissonance_type == 'none':
        return

    if dissonance.dissonance_type == 'local':
        if agent.autonomy > 0.65:
            agent.individual.IC_t = min(1.0, agent.individual.IC_t * 1.05)
        elif dissonance.ind_cluster_total > config.migration_threshold:
            agent.seeking_new_cluster = True
        else:
            agent.conformity = min(1.0, agent.conformity + 0.08)

    elif dissonance.dissonance_type == 'crisis':
        agent.conformity = min(0.85, agent.conformity + 0.15)
        agent.autonomy = max(0.15, agent.autonomy - 0.1)


# =============================================================================
# Dynamic Clustering
# =============================================================================

class ClusterManager:
    """Manages dynamic cluster operations."""

    def __init__(self, config: SHConfig):
        self.config = config
        self.next_cluster_id = config.n_initial_clusters

    def update_clusters(
        self,
        agents: List[HierarchicalAgent],
        clusters: Dict[int, ClusterIdentity]
    ) -> Dict:
        """Perform clustering operations."""
        ops = {'migrations': 0, 'splits': 0, 'merges': 0}

        # Handle migrations
        for agent in agents:
            if agent.seeking_new_cluster:
                best = self._find_best_cluster(agent, clusters)
                if best is not None and best != agent.cluster_id:
                    self._migrate_agent(agent, best, clusters)
                    ops['migrations'] += 1
                agent.seeking_new_cluster = False

        # Split low-cohesion clusters
        for cluster_id in list(clusters.keys()):
            cluster = clusters[cluster_id]
            if cluster.cohesion < self.config.cohesion_threshold_split:
                if len(cluster.member_ids) >= self.config.min_cluster_size * 2:
                    self._split_cluster(cluster_id, agents, clusters)
                    ops['splits'] += 1

        # Merge similar clusters
        cluster_list = list(clusters.values())
        merged = set()
        for i, c1 in enumerate(cluster_list):
            if c1.cluster_id in merged:
                continue
            for c2 in cluster_list[i+1:]:
                if c2.cluster_id in merged:
                    continue
                if self._should_merge(c1, c2):
                    self._merge_clusters(c1, c2, agents, clusters)
                    merged.add(c2.cluster_id)
                    ops['merges'] += 1
                    break

        return ops

    def _find_best_cluster(
        self,
        agent: HierarchicalAgent,
        clusters: Dict[int, ClusterIdentity]
    ) -> Optional[int]:
        """Find most compatible cluster."""
        best_fit = None
        best_sim = 0

        agent_theta = agent.individual.theta.to_array()

        for cluster_id, cluster in clusters.items():
            if len(cluster.member_ids) >= self.config.max_cluster_size:
                continue

            cluster_theta = cluster.theta_cluster.to_array()
            sim = 1.0 / (1.0 + np.linalg.norm(agent_theta - cluster_theta))

            if sim > best_sim and sim > self.config.similarity_threshold:
                best_sim = sim
                best_fit = cluster_id

        return best_fit

    def _migrate_agent(
        self,
        agent: HierarchicalAgent,
        new_cluster_id: int,
        clusters: Dict[int, ClusterIdentity]
    ):
        """Move agent to new cluster."""
        if agent.cluster_id in clusters:
            clusters[agent.cluster_id].member_ids.discard(agent.individual.agent_id)

        agent.cluster_id = new_cluster_id
        if new_cluster_id in clusters:
            clusters[new_cluster_id].member_ids.add(agent.individual.agent_id)

    def _split_cluster(
        self,
        cluster_id: int,
        agents: List[HierarchicalAgent],
        clusters: Dict[int, ClusterIdentity]
    ):
        """Split cluster into two."""
        members = [a for a in agents if a.cluster_id == cluster_id]
        if len(members) < self.config.min_cluster_size * 2:
            return

        # Split by risk_aversion median
        members_sorted = sorted(members, key=lambda a: a.individual.theta.risk_aversion)
        mid = len(members_sorted) // 2

        group_a = members_sorted[:mid]
        group_b = members_sorted[mid:]

        # Create new cluster for group_b
        new_id = self.next_cluster_id
        self.next_cluster_id += 1

        for agent in group_b:
            agent.cluster_id = new_id

        clusters[cluster_id] = aggregate_to_cluster(group_a)
        clusters[new_id] = aggregate_to_cluster(group_b)

    def _should_merge(self, c1: ClusterIdentity, c2: ClusterIdentity) -> bool:
        """Check if clusters should merge."""
        if c1.cohesion < self.config.cohesion_threshold_merge * 0.9:
            return False
        if c2.cohesion < self.config.cohesion_threshold_merge * 0.9:
            return False
        if len(c1.member_ids) + len(c2.member_ids) > self.config.max_cluster_size:
            return False

        theta_sim = 1.0 / (1.0 + np.linalg.norm(
            c1.theta_cluster.to_array() - c2.theta_cluster.to_array()))

        return theta_sim > self.config.similarity_threshold

    def _merge_clusters(
        self,
        c1: ClusterIdentity,
        c2: ClusterIdentity,
        agents: List[HierarchicalAgent],
        clusters: Dict[int, ClusterIdentity]
    ):
        """Merge c2 into c1."""
        for agent in agents:
            if agent.cluster_id == c2.cluster_id:
                agent.cluster_id = c1.cluster_id

        if c2.cluster_id in clusters:
            del clusters[c2.cluster_id]

        members = [a for a in agents if a.cluster_id == c1.cluster_id]
        clusters[c1.cluster_id] = aggregate_to_cluster(members)


# =============================================================================
# Perturbations
# =============================================================================

@dataclass
class HierarchicalPerturbation:
    """Multi-level perturbation."""
    type: str
    level: str
    severity: float
    step: int


def generate_perturbations(condition: str, config: SHConfig) -> List[HierarchicalPerturbation]:
    """Generate perturbation schedule."""
    if condition == 'full_hierarchy':
        return [
            HierarchicalPerturbation('history_ind', 'individual', 0.3, 25),
            HierarchicalPerturbation('cohesion_attack', 'cluster', 0.35, 50),
            HierarchicalPerturbation('purpose_loss', 'collective', 0.3, 75),
            HierarchicalPerturbation('identity_ind', 'individual', 0.35, 100),
            HierarchicalPerturbation('fragmentation', 'collective', 0.3, 125),
        ]
    elif condition == 'no_cluster':
        return [
            HierarchicalPerturbation('history_ind', 'individual', 0.4, 30),
            HierarchicalPerturbation('identity_ind', 'individual', 0.4, 60),
            HierarchicalPerturbation('purpose_loss', 'collective', 0.4, 90),
            HierarchicalPerturbation('history_ind', 'individual', 0.35, 120),
        ]
    elif condition == 'no_collective':
        return [
            HierarchicalPerturbation('history_ind', 'individual', 0.35, 30),
            HierarchicalPerturbation('cohesion_attack', 'cluster', 0.4, 60),
            HierarchicalPerturbation('cohesion_attack', 'cluster', 0.4, 90),
            HierarchicalPerturbation('history_ind', 'individual', 0.35, 120),
        ]
    elif condition == 'shuffled_links':
        return [
            HierarchicalPerturbation('isolation', 'individual', 0.5, 25),
            HierarchicalPerturbation('vertical_tear', 'cross_level', 0.4, 50),
            HierarchicalPerturbation('fragmentation', 'collective', 0.45, 75),
            HierarchicalPerturbation('cohesion_attack', 'cluster', 0.4, 100),
            HierarchicalPerturbation('vertical_tear', 'cross_level', 0.35, 125),
        ]
    elif condition == 'catastrophic_multi':
        return [
            HierarchicalPerturbation('catastrophic_cascade', 'cross_level', 0.6, 40),
            HierarchicalPerturbation('catastrophic_cascade', 'cross_level', 0.65, 80),
            HierarchicalPerturbation('vertical_tear', 'cross_level', 0.5, 120),
        ]
    return []


def apply_perturbation(
    p: HierarchicalPerturbation,
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity],
    collective: CollectiveIdentity
) -> Dict:
    """Apply perturbation and return effects."""
    effects = {'agents_affected': 0, 'clusters_affected': 0, 'damage': 0.0}

    if p.level == 'individual':
        n_targets = max(1, int(len(agents) * p.severity * 0.4))
        targets = random.sample(agents, min(n_targets, len(agents)))

        for agent in targets:
            if p.type == 'history_ind':
                agent.individual.history_corruption = min(1.0,
                    agent.individual.history_corruption + p.severity)
            elif p.type == 'identity_ind':
                agent.individual.IC_t = max(0, agent.individual.IC_t - p.severity * 0.2)
            elif p.type == 'isolation':
                agent.alignment_to_cluster *= (1 - p.severity)
                agent.conformity *= (1 - p.severity * 0.4)

            effects['agents_affected'] += 1
            effects['damage'] += p.severity * 0.2

    elif p.level == 'cluster':
        n_targets = max(1, int(len(clusters) * p.severity * 0.5))
        target_ids = random.sample(list(clusters.keys()), min(n_targets, len(clusters)))

        for cid in target_ids:
            cluster = clusters[cid]
            if p.type == 'cohesion_attack':
                cluster.cohesion = max(0.1, cluster.cohesion - p.severity * 0.35)
                cluster.IC_cluster *= (1 - p.severity * 0.25)

            effects['clusters_affected'] += 1
            effects['damage'] += p.severity * 0.3

            # Propagate to members
            for agent in agents:
                if agent.cluster_id == cid:
                    agent.alignment_to_cluster *= (1 - p.severity * 0.15)

    elif p.level == 'collective':
        if p.type == 'purpose_loss':
            collective.collective_purpose = 'fragmented'
            collective.global_coherence *= (1 - p.severity * 0.4)
            collective.IC_collective *= (1 - p.severity * 0.35)

        elif p.type == 'fragmentation':
            collective.global_coherence *= (1 - p.severity * 0.5)
            for cluster in clusters.values():
                noise = np.random.normal(0, p.severity * 0.15, 4)
                theta_arr = cluster.theta_cluster.to_array() + noise
                cluster.theta_cluster = MetaPolicy.from_array(theta_arr)

        effects['damage'] += p.severity * 0.4

        # Propagate down
        for agent in agents:
            agent.alignment_to_collective *= (1 - p.severity * 0.2)

    elif p.level == 'cross_level':
        if p.type == 'catastrophic_cascade':
            for agent in agents:
                agent.individual.IC_t *= (1 - p.severity * 0.25)
                agent.individual.history_corruption += p.severity * 0.3
            for cluster in clusters.values():
                cluster.cohesion *= (1 - p.severity * 0.35)
                cluster.IC_cluster *= (1 - p.severity * 0.3)
            collective.global_coherence *= (1 - p.severity * 0.4)
            collective.IC_collective *= (1 - p.severity * 0.35)

            effects['agents_affected'] = len(agents)
            effects['clusters_affected'] = len(clusters)
            effects['damage'] = p.severity * 0.6

        elif p.type == 'vertical_tear':
            for agent in agents:
                agent.alignment_to_cluster *= (1 - p.severity * 0.5)
                agent.alignment_to_collective *= (1 - p.severity * 0.6)
                agent.autonomy = min(1.0, agent.autonomy + p.severity * 0.25)
                agent.conformity = max(0.1, agent.conformity - p.severity * 0.3)

            collective.global_coherence *= (1 - p.severity * 0.4)
            effects['damage'] = p.severity * 0.5

    return effects


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class SHMetrics:
    """Metrics for Self-Hierarchy experiment."""
    vertical_coherence: float = 0.0
    hierarchical_resilience: float = 0.0
    emergent_diversity: float = 0.0
    alignment_dissonance: float = 0.0

    individual_survival: float = 0.0
    cluster_stability: float = 0.0
    collective_integrity: float = 0.0

    migration_rate: float = 0.0
    restructuring_events: int = 0


def compute_vertical_coherence(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity],
    collective: CollectiveIdentity
) -> float:
    """VC: Alignment from individual → cluster → collective."""
    if not agents or not clusters:
        return 0.0

    # Individual-to-cluster
    ind_cluster_aligns = []
    for agent in agents:
        if agent.cluster_id and agent.cluster_id in clusters:
            cluster = clusters[agent.cluster_id]
            sim = 1.0 - np.linalg.norm(
                agent.individual.theta.to_array() - cluster.theta_cluster.to_array()
            ) / 2.0
            ind_cluster_aligns.append(max(0, sim))

    ic_vc = np.mean(ind_cluster_aligns) if ind_cluster_aligns else 0.0

    # Cluster-to-collective
    cc_aligns = []
    for cluster in clusters.values():
        sim = 1.0 - np.linalg.norm(
            cluster.theta_cluster.to_array() - collective.theta_collective.to_array()
        ) / 2.0
        cc_aligns.append(max(0, sim))

    cc_vc = np.mean(cc_aligns) if cc_aligns else 0.0

    return float(np.sqrt(ic_vc * cc_vc))


def compute_hierarchical_resilience(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity],
    collective: CollectiveIdentity,
    config: SHConfig
) -> float:
    """HR: Identity survival at each level."""
    # Individual
    ind_surviving = sum(1 for a in agents if a.individual.IC_t > config.epsilon)
    ind_res = ind_surviving / len(agents) if agents else 0.0

    # Cluster
    cluster_surviving = sum(1 for c in clusters.values()
                           if c.IC_cluster > config.cluster_ic_threshold and c.cohesion > 0.3)
    cluster_res = cluster_surviving / len(clusters) if clusters else 0.0

    # Collective
    col_res = 1.0 if collective.IC_collective > config.collective_ic_threshold else 0.0
    col_res *= collective.global_coherence

    return float(0.25 * ind_res + 0.35 * cluster_res + 0.40 * col_res)


def compute_emergent_diversity(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity]
) -> float:
    """ED: Functional differentiation."""
    if not clusters:
        return 0.0

    # Specialization diversity
    specs = [c.specialization for c in clusters.values()]
    spec_counts = defaultdict(int)
    for s in specs:
        spec_counts[s] += 1

    total = len(specs)
    entropy = 0.0
    for count in spec_counts.values():
        p = count / total
        entropy -= p * np.log(p + 1e-10)

    max_entropy = np.log(len(CLUSTER_SPECIALIZATIONS))
    cluster_div = entropy / max_entropy if max_entropy > 0 else 0.0

    # Individual diversity within clusters
    role_divs = []
    for cluster in clusters.values():
        members = [a for a in agents if a.cluster_id == cluster.cluster_id]
        if len(members) < 2:
            continue
        theta_arrays = np.array([m.individual.theta.to_array() for m in members])
        variance = np.mean(np.var(theta_arrays, axis=0))
        div_score = 1.0 - abs(variance - 0.08) * 6
        role_divs.append(max(0, min(1, div_score)))

    ind_div = np.mean(role_divs) if role_divs else 0.0

    return float(0.5 * cluster_div + 0.5 * ind_div)


def compute_alignment_metric(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity],
    collective: CollectiveIdentity,
    decision_history: List[Dict]
) -> float:
    """AD: Level agreement on decisions."""
    if not decision_history:
        return 0.5

    alignments = []
    for decision in decision_history[-15:]:
        ind_decisions = decision.get('individual_decisions', {})
        cluster_decisions = decision.get('cluster_decisions', {})
        col_decision = decision.get('collective_decision')

        # Individual-cluster alignment
        ic_aligns = []
        for cid, c_dec in cluster_decisions.items():
            members = [aid for aid, d in ind_decisions.items()
                      if any(a.cluster_id == cid and a.individual.agent_id == aid for a in agents)]
            if members:
                agreement = sum(1 for mid in members if ind_decisions.get(mid) == c_dec) / len(members)
                ic_aligns.append(agreement)

        ic_align = np.mean(ic_aligns) if ic_aligns else 0.5

        # Cluster-collective alignment
        if col_decision and cluster_decisions:
            cc_align = sum(1 for cd in cluster_decisions.values() if cd == col_decision) / len(cluster_decisions)
        else:
            cc_align = 0.5

        alignments.append(0.5 * ic_align + 0.5 * cc_align)

    return float(np.mean(alignments)) if alignments else 0.5


def compute_all_metrics(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity],
    collective: CollectiveIdentity,
    config: SHConfig,
    decision_history: List[Dict],
    restructuring_events: int,
    migration_count: int,
    condition: str
) -> SHMetrics:
    """Compute all metrics."""
    if condition == 'no_cluster':
        vc = compute_vertical_coherence_no_cluster(agents, collective)
        hr = compute_hr_no_cluster(agents, collective, config)
        ed = 0.0
    elif condition == 'no_collective':
        vc = compute_vc_no_collective(agents, clusters)
        hr = compute_hr_no_collective(agents, clusters, config)
        ed = compute_emergent_diversity(agents, clusters)
    else:
        vc = compute_vertical_coherence(agents, clusters, collective)
        hr = compute_hierarchical_resilience(agents, clusters, collective, config)
        ed = compute_emergent_diversity(agents, clusters)

    return SHMetrics(
        vertical_coherence=vc,
        hierarchical_resilience=hr,
        emergent_diversity=ed,
        alignment_dissonance=compute_alignment_metric(agents, clusters, collective, decision_history),
        individual_survival=sum(1 for a in agents if a.individual.IC_t > config.epsilon) / len(agents) if agents else 0.0,
        cluster_stability=sum(1 for c in clusters.values() if c.cohesion > 0.4) / len(clusters) if clusters else 0.0,
        collective_integrity=collective.IC_collective * collective.global_coherence,
        migration_rate=migration_count / max(1, config.n_steps),
        restructuring_events=restructuring_events
    )


def compute_vertical_coherence_no_cluster(
    agents: List[HierarchicalAgent],
    collective: CollectiveIdentity
) -> float:
    """VC without cluster level."""
    if not agents:
        return 0.0
    aligns = []
    for agent in agents:
        sim = 1.0 - np.linalg.norm(
            agent.individual.theta.to_array() - collective.theta_collective.to_array()
        ) / 2.0
        aligns.append(max(0, sim))
    return float(np.mean(aligns))


def compute_hr_no_cluster(
    agents: List[HierarchicalAgent],
    collective: CollectiveIdentity,
    config: SHConfig
) -> float:
    """HR without cluster level."""
    ind_res = sum(1 for a in agents if a.individual.IC_t > config.epsilon) / len(agents) if agents else 0.0
    col_res = collective.IC_collective * collective.global_coherence
    return float(0.5 * ind_res + 0.5 * col_res)


def compute_vc_no_collective(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity]
) -> float:
    """VC without collective level."""
    if not agents or not clusters:
        return 0.0
    aligns = []
    for agent in agents:
        if agent.cluster_id and agent.cluster_id in clusters:
            cluster = clusters[agent.cluster_id]
            sim = 1.0 - np.linalg.norm(
                agent.individual.theta.to_array() - cluster.theta_cluster.to_array()
            ) / 2.0
            aligns.append(max(0, sim))
    return float(np.mean(aligns)) if aligns else 0.0


def compute_hr_no_collective(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity],
    config: SHConfig
) -> float:
    """HR without collective level."""
    ind_res = sum(1 for a in agents if a.individual.IC_t > config.epsilon) / len(agents) if agents else 0.0
    cluster_res = sum(1 for c in clusters.values() if c.cohesion > 0.3) / len(clusters) if clusters else 0.0
    return float(0.5 * ind_res + 0.5 * cluster_res)


# =============================================================================
# Decision Simulation
# =============================================================================

def simulate_collective_decision(
    agents: List[HierarchicalAgent],
    clusters: Dict[int, ClusterIdentity],
    collective: CollectiveIdentity,
    condition: str
) -> Dict:
    """Simulate collective decision."""
    individual_decisions = {}
    for agent in agents:
        if agent.individual.IC_t > 0.15:
            decision = 'safe' if agent.individual.theta.risk_aversion > 0.5 else 'risk'
            individual_decisions[agent.individual.agent_id] = decision

    cluster_decisions = {}
    if condition != 'no_cluster':
        for cid, cluster in clusters.items():
            decision = 'safe' if cluster.theta_cluster.risk_aversion > 0.5 else 'risk'
            cluster_decisions[cid] = decision

    collective_decision = None
    if condition != 'no_collective':
        collective_decision = 'safe' if collective.theta_collective.risk_aversion > 0.5 else 'risk'

    return {
        'individual_decisions': individual_decisions,
        'cluster_decisions': cluster_decisions,
        'collective_decision': collective_decision
    }


# =============================================================================
# Episode Runner
# =============================================================================

@dataclass
class EpisodeResult:
    """Result of one episode."""
    final_agents: int
    final_clusters: int
    collective_intact: bool
    metrics: SHMetrics
    restructuring_events: int
    migration_count: int


def run_episode(
    config: SHConfig,
    condition: str,
    perturbations: List[HierarchicalPerturbation]
) -> EpisodeResult:
    """Run one episode."""

    # Initialize agents
    agents = [HierarchicalAgent.create_random(i, config) for i in range(config.n_agents)]

    # Initialize clusters
    clusters: Dict[int, ClusterIdentity] = {}
    if condition != 'no_cluster':
        agents_per_cluster = config.n_agents // config.n_initial_clusters
        for i in range(config.n_initial_clusters):
            start = i * agents_per_cluster
            end = start + agents_per_cluster if i < config.n_initial_clusters - 1 else config.n_agents
            for j in range(start, end):
                agents[j].cluster_id = i
            members = agents[start:end]
            clusters[i] = aggregate_to_cluster(members)

    # Initialize collective
    if condition != 'no_collective':
        if clusters:
            collective = aggregate_to_collective(list(clusters.values()))
        else:
            collective = aggregate_individuals_to_collective(agents)
    else:
        collective = CollectiveIdentity()

    cluster_manager = ClusterManager(config)
    decision_history = []
    restructuring_events = 0
    migration_count = 0

    for step in range(config.n_steps):
        # Phase 1: Bottom-up aggregation
        if step % config.aggregation_frequency == 0:
            if condition != 'no_cluster':
                for cid in list(clusters.keys()):
                    members = [a for a in agents if a.cluster_id == cid]
                    if members:
                        clusters[cid] = aggregate_to_cluster(members)
                    else:
                        del clusters[cid]

            if condition != 'no_collective':
                if clusters:
                    collective = aggregate_to_collective(list(clusters.values()))
                else:
                    collective = aggregate_individuals_to_collective(agents)

        # Phase 2: Top-down influence
        if condition not in ['no_collective'] and clusters:
            for cluster in clusters.values():
                apply_top_down_to_cluster(cluster, collective, config)

        if condition not in ['no_cluster']:
            for agent in agents:
                if agent.cluster_id and agent.cluster_id in clusters:
                    apply_top_down_to_agent(agent, clusters[agent.cluster_id], config)

        # Phase 3: Individual actions
        for agent in agents:
            if not agent.is_alive():
                continue

            context = agent.get_hierarchical_context(clusters, collective, condition)
            action = agent.select_hierarchical_action(context, config)
            agent.execute_action(action)

            if step % config.update_freq == 0:
                agent.update_adaptive_systems(context)

            agent.decay_effects()

        # Phase 4: Dissonance resolution
        if condition not in ['no_cluster', 'no_collective']:
            for agent in agents:
                if agent.cluster_id and agent.cluster_id in clusters:
                    dissonance = compute_dissonance(agent, clusters[agent.cluster_id], collective)
                    if dissonance.dissonance_type != 'none':
                        resolve_dissonance(agent, dissonance, config)
                        if agent.seeking_new_cluster:
                            migration_count += 1

        # Phase 5: Dynamic clustering
        if condition != 'no_cluster' and step % 12 == 0:
            ops = cluster_manager.update_clusters(agents, clusters)
            restructuring_events += ops['splits'] + ops['merges']
            migration_count += ops['migrations']

        # Phase 6: Perturbations
        for p in perturbations:
            if p.step == step:
                apply_perturbation(p, agents, clusters, collective)

        # Phase 7: Decisions
        if step % config.decision_interval == 0:
            decision = simulate_collective_decision(agents, clusters, collective, condition)
            decision_history.append(decision)

        # Check collapse
        alive = [a for a in agents if a.is_alive()]
        if len(alive) < config.min_cluster_size:
            break

    # Compute metrics
    metrics = compute_all_metrics(
        agents, clusters, collective, config,
        decision_history, restructuring_events, migration_count, condition
    )

    return EpisodeResult(
        final_agents=len([a for a in agents if a.is_alive()]),
        final_clusters=len([c for c in clusters.values() if c.IC_cluster > config.cluster_ic_threshold]),
        collective_intact=collective.IC_collective > config.collective_ic_threshold,
        metrics=metrics,
        restructuring_events=restructuring_events,
        migration_count=migration_count
    )


# =============================================================================
# Condition Runner
# =============================================================================

def run_condition(condition: str, config: SHConfig) -> Tuple[List[EpisodeResult], SHMetrics]:
    """Run all episodes for a condition."""
    print(f"\n{'='*60}")
    print(f"Running IPUESA-SH - Condition: {condition}")
    print(f"{'='*60}")

    perturbations = generate_perturbations(condition, config)
    all_results = []

    for run in range(config.n_runs):
        for episode in range(config.n_episodes):
            result = run_episode(config, condition, perturbations)
            all_results.append(result)

        if (run + 1) % 2 == 0:
            print(f"  Completed {run+1}/{config.n_runs} runs")

    # Aggregate
    avg_metrics = SHMetrics(
        vertical_coherence=np.mean([r.metrics.vertical_coherence for r in all_results]),
        hierarchical_resilience=np.mean([r.metrics.hierarchical_resilience for r in all_results]),
        emergent_diversity=np.mean([r.metrics.emergent_diversity for r in all_results]),
        alignment_dissonance=np.mean([r.metrics.alignment_dissonance for r in all_results]),
        individual_survival=np.mean([r.metrics.individual_survival for r in all_results]),
        cluster_stability=np.mean([r.metrics.cluster_stability for r in all_results]),
        collective_integrity=np.mean([r.metrics.collective_integrity for r in all_results]),
        migration_rate=np.mean([r.metrics.migration_rate for r in all_results]),
        restructuring_events=int(np.mean([r.restructuring_events for r in all_results]))
    )

    return all_results, avg_metrics


def print_condition_results(condition: str, metrics: SHMetrics):
    """Print results."""
    print(f"\n{'='*60}")
    print(f"RESULTS - {condition}")
    print(f"{'='*60}")

    print(f"\nPrimary Metrics:")
    print(f"  VC (Vertical Coherence)     = {metrics.vertical_coherence:.3f}")
    print(f"  HR (Hierarchical Resilience)= {metrics.hierarchical_resilience:.3f}")
    print(f"  ED (Emergent Diversity)     = {metrics.emergent_diversity:.3f}")
    print(f"  AD (Alignment)              = {metrics.alignment_dissonance:.3f}")

    print(f"\nSupporting Metrics:")
    print(f"  Individual Survival         = {metrics.individual_survival:.3f}")
    print(f"  Cluster Stability           = {metrics.cluster_stability:.3f}")
    print(f"  Collective Integrity        = {metrics.collective_integrity:.3f}")

    print(f"\nDynamic Metrics:")
    print(f"  Migration Rate              = {metrics.migration_rate:.3f}")
    print(f"  Restructuring Events        = {metrics.restructuring_events}")


def evaluate_self_evidence(metrics: SHMetrics, all_conditions: Dict[str, SHMetrics]) -> Dict:
    """Evaluate self-evidence criteria."""
    criteria = []
    fc = metrics

    c1 = fc.vertical_coherence > 0.5
    criteria.append(('VC > 0.5', c1, fc.vertical_coherence))

    c2 = fc.hierarchical_resilience > 0.4
    criteria.append(('HR > 0.4', c2, fc.hierarchical_resilience))

    c3 = fc.emergent_diversity > 0.3
    criteria.append(('ED > 0.3', c3, fc.emergent_diversity))

    c4 = fc.alignment_dissonance > 0.5
    criteria.append(('AD > 0.5', c4, fc.alignment_dissonance))

    nc = all_conditions.get('no_cluster', SHMetrics())
    c5 = fc.hierarchical_resilience > nc.hierarchical_resilience + 0.08
    criteria.append(('full >> no_cluster', c5,
                    f"{fc.hierarchical_resilience:.3f} vs {nc.hierarchical_resilience:.3f}"))

    ncol = all_conditions.get('no_collective', SHMetrics())
    c6 = fc.vertical_coherence > ncol.vertical_coherence + 0.08
    criteria.append(('full >> no_collective', c6,
                    f"{fc.vertical_coherence:.3f} vs {ncol.vertical_coherence:.3f}"))

    cat = all_conditions.get('catastrophic_multi', SHMetrics())
    c7 = cat.hierarchical_resilience > 0.2
    criteria.append(('catastrophic HR > 0.2', c7, cat.hierarchical_resilience))

    c8 = fc.cluster_stability > 0.5
    criteria.append(('Cluster stability > 0.5', c8, fc.cluster_stability))

    passed = sum(1 for _, c, _ in criteria if c)

    if passed >= 6:
        conclusion = "EVIDENCE OF HIERARCHICAL SELF-EMERGENCE"
    elif passed >= 4:
        conclusion = "Partial evidence of hierarchical self"
    else:
        conclusion = "No evidence - hierarchy insufficient"

    return {
        'criteria': criteria,
        'passed': passed,
        'total': 8,
        'conclusion': conclusion
    }


# =============================================================================
# Main
# =============================================================================

def run_full_experiment():
    """Run complete IPUESA-SH experiment."""
    print("=" * 70)
    print("IPUESA-SH: Self-Hierarchy Experiment")
    print("        Three-Level Identity Emergence")
    print("=" * 70)

    config = SHConfig()

    print(f"\nConfiguration:")
    print(f"  N agents: {config.n_agents}")
    print(f"  N initial clusters: {config.n_initial_clusters}")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  N episodes: {config.n_episodes}, N steps: {config.n_steps}")
    print(f"  N runs: {config.n_runs}")

    conditions = [
        'full_hierarchy',
        'no_cluster',
        'no_collective',
        'shuffled_links',
        'catastrophic_multi',
    ]

    all_metrics = {}
    all_evidence = {}

    for condition in conditions:
        results, metrics = run_condition(condition, config)
        all_metrics[condition] = metrics
        print_condition_results(condition, metrics)

    # Evaluate
    for condition in conditions:
        all_evidence[condition] = evaluate_self_evidence(all_metrics[condition], all_metrics)

    # Comparative
    print("\n" + "=" * 70)
    print("IPUESA-SH: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Condition':<20} {'VC':<8} {'HR':<8} {'ED':<8} {'AD':<8} {'ClustStab':<10} {'Pass':<6}")
    print("-" * 80)

    for condition in conditions:
        m = all_metrics[condition]
        passed = all_evidence[condition]['passed']
        print(f"{condition:<20} {m.vertical_coherence:<8.3f} {m.hierarchical_resilience:<8.3f} "
              f"{m.emergent_diversity:<8.3f} {m.alignment_dissonance:<8.3f} "
              f"{m.cluster_stability:<10.3f} {passed}/8")

    # Self-evidence
    print("\n" + "=" * 70)
    print("SELF-EVIDENCE CRITERIA (HIERARCHICAL SELF)")
    print("-" * 70)

    evidence = all_evidence['full_hierarchy']
    for name, passed, value in evidence['criteria']:
        status = 'PASS' if passed else 'FAIL'
        print(f"  [{status}] {name}: {value}")

    print(f"\n  Passed: {evidence['passed']}/8 criteria")
    print(f"\n  CONCLUSION: {evidence['conclusion']}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("-" * 70)

    fc = all_metrics['full_hierarchy']
    nc = all_metrics['no_cluster']
    ncol = all_metrics['no_collective']

    print(f"\n  Cluster value: full HR ({fc.hierarchical_resilience:.3f}) vs no_cluster HR ({nc.hierarchical_resilience:.3f})")
    print(f"  Collective value: full VC ({fc.vertical_coherence:.3f}) vs no_collective VC ({ncol.vertical_coherence:.3f})")
    print(f"  Vertical coherence: {fc.vertical_coherence:.3f}")
    print(f"  Emergent diversity: {fc.emergent_diversity:.3f}")

    # Save
    output_path = Path(__file__).parent.parent.parent / "results" / "ipuesa_sh_results.json"
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
        elif isinstance(obj, SHMetrics):
            return {
                'vertical_coherence': obj.vertical_coherence,
                'hierarchical_resilience': obj.hierarchical_resilience,
                'emergent_diversity': obj.emergent_diversity,
                'alignment_dissonance': obj.alignment_dissonance,
                'individual_survival': obj.individual_survival,
                'cluster_stability': obj.cluster_stability,
                'collective_integrity': obj.collective_integrity,
                'migration_rate': obj.migration_rate,
                'restructuring_events': obj.restructuring_events
            }
        elif isinstance(obj, float) and np.isnan(obj):
            return 0.0
        return obj

    save_data = {
        'config': {
            'n_agents': config.n_agents,
            'n_initial_clusters': config.n_initial_clusters,
            'epsilon': config.epsilon,
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
