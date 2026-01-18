"""
IPUESA Evolvable - Parameterized IPUESA Simulation

Adapts IPUESA-SYNTH-v2 to accept external configuration,
allowing evolutionary optimization of hyperparameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from .config_space import EvolvableConfig

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DegradationState(Enum):
    OPTIMAL = 'optimal'
    STRESSED = 'stressed'
    IMPAIRED = 'impaired'
    CRITICAL = 'critical'
    COLLAPSED = 'collapsed'

@dataclass
class EvolvableMicroModule:
    """MicroModule with configurable effects."""
    module_type: str
    strength: float = 0.5
    activation_count: int = 0
    contribution: float = 0.0
    consolidated: bool = False
    is_learned: bool = False
    age: int = 0

    def apply(self, config: EvolvableConfig) -> float:
        """Apply module effect using config-defined strengths."""
        self.activation_count += 1
        effects = config.get_module_effects()
        base_effect = effects.get(self.module_type, 0.15)
        return self.strength * base_effect

    def age_tick(self):
        """Module ages and weakens over time."""
        self.age += 1
        if self.age > 50 and not self.consolidated:
            self.strength *= 0.98

@dataclass
class MetaPolicy:
    """WHO the agent is."""
    risk_aversion: float = 0.5
    exploration_rate: float = 0.3
    memory_depth: float = 0.5
    prediction_weight: float = 0.5

    def to_vector(self) -> np.ndarray:
        return np.array([self.risk_aversion, self.exploration_rate,
                        self.memory_depth, self.prediction_weight])

@dataclass
class CognitiveArchitecture:
    """HOW the agent processes."""
    attention_immediate: float = 0.33
    attention_history: float = 0.33
    attention_prediction: float = 0.34
    memory_update_rate: float = 0.1
    perceptual_gain: float = 1.0

@dataclass
class EvolvableAgent:
    """Agent with evolvable configuration."""
    agent_id: int
    cluster_id: int

    # Core identity
    theta: MetaPolicy = field(default_factory=MetaPolicy)
    alpha: CognitiveArchitecture = field(default_factory=CognitiveArchitecture)
    modules: list[EvolvableMicroModule] = field(default_factory=list)
    IC_t: float = 1.0

    # Holographic embeddings
    cluster_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    embedding_dim: int = 8
    embedding_staleness: float = 0.0

    # Temporal anticipation
    threat_buffer: float = 0.0
    threat_history: list[float] = field(default_factory=list)
    anticipated_damage: float = 0.0

    # Gradual degradation
    degradation_level: float = 0.0
    residual_damage: float = 0.0

    # State
    protective_stance: float = 0.0
    structural_corruption: float = 0.0
    history_corruption: float = 0.0
    prediction_noise: float = 0.0

    # Tracking
    IC_history: list[float] = field(default_factory=list)
    preemptive_actions: int = 0
    reactive_actions: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    modules_created: int = 0
    modules_received: int = 0
    proactive_modules_created: int = 0

    def is_alive(self) -> bool:
        return self.IC_t > 0.05

    def get_embedding_integrity(self) -> float:
        if self.embedding_dim == 0:
            return 0.0
        norm = np.linalg.norm(self.cluster_embedding)
        return min(1.0, norm / np.sqrt(self.embedding_dim)) * (1 - self.embedding_staleness)

    def has_module_type(self, module_type: str) -> bool:
        return any(m.module_type == module_type for m in self.modules)

    def get_degradation_state(self) -> DegradationState:
        if self.degradation_level < 0.15:
            return DegradationState.OPTIMAL
        elif self.degradation_level < 0.35:
            return DegradationState.STRESSED
        elif self.degradation_level < 0.55:
            return DegradationState.IMPAIRED
        elif self.degradation_level < 0.80:
            return DegradationState.CRITICAL
        else:
            return DegradationState.COLLAPSED

@dataclass
class ClusterState:
    """Cluster state for agent grouping."""
    cluster_id: int
    cohesion: float = 0.5
    specialization: float = 0.0
    size: int = 0

@dataclass
class PerturbationWave:
    """A wave of perturbation in the storm."""
    wave_type: str
    base_damage: float
    step: int
    residual_factor: float = 0.05

# =============================================================================
# CORE FUNCTIONS (Parameterized)
# =============================================================================

def gradual_damage(agent: EvolvableAgent, damage: float,
                   config: EvolvableConfig) -> float:
    """
    Apply gradual damage using config parameters.
    """
    base_degrad_rate = config.base_degrad_rate
    individual_factor = 1.0

    # Embedding protection
    if agent.embedding_dim > 0:
        ei = agent.get_embedding_integrity()
        individual_factor *= (1.0 - ei * config.embedding_protection)

    # Protective stance
    individual_factor *= (1.0 - agent.protective_stance * config.stance_protection)

    # Compound effect (degraded agents degrade faster)
    individual_factor *= (1.0 + agent.degradation_level * config.compound_factor)

    # Module protection - activate modules when used
    for module in agent.modules:
        if module.module_type in ['threat_filter', 'cascade_breaker']:
            effect = module.apply(config)  # Increment activation_count
            individual_factor *= (1.0 - effect)
            # Track contribution (positive = helped reduce damage)
            module.contribution += 0.1 * module.strength

    # Random resilience
    np.random.seed(agent.agent_id + int(damage * 1000))
    resilience = config.resilience_min + np.random.random() * config.resilience_range
    individual_factor *= resilience

    # Noise
    np.random.seed(agent.agent_id * 7 + int(agent.IC_t * 100))
    noise = (np.random.random() - 0.5) * damage * config.noise_scale

    # Cluster modifier for variance
    cluster_modifier = 0.8 + (agent.cluster_id % 4) * 0.15
    degradation_increment = damage * base_degrad_rate * individual_factor * cluster_modifier + noise

    agent.degradation_level += max(0, degradation_increment)
    agent.degradation_level = min(1.0, agent.degradation_level)

    # IC damage scaled by degradation
    effective_damage = damage * (1 + agent.degradation_level * 0.3)
    agent.IC_t -= effective_damage
    agent.IC_t = max(0, min(1, agent.IC_t))

    # Residual damage
    agent.residual_damage += damage * 0.04
    agent.residual_damage = min(config.residual_cap, agent.residual_damage)

    return effective_damage

def gradual_recovery(agent: EvolvableAgent, cluster: ClusterState,
                     config: EvolvableConfig) -> tuple[float, bool]:
    """
    Recovery using config parameters.
    """
    if not agent.is_alive():
        return 0.0, False

    agent.recovery_attempts += 1
    base_rate = config.base_recovery_rate

    ei = agent.get_embedding_integrity()
    cluster_support = cluster.cohesion

    # Bonuses from config
    rate = base_rate * (1 + ei * config.embedding_bonus)
    rate *= (1 + cluster_support * config.cluster_bonus)

    # Module bonuses - activate modules when used
    for module in agent.modules:
        if module.module_type == 'recovery_accelerator':
            effect = module.apply(config)
            rate *= (1 + effect)
            module.contribution += 0.1 * module.strength
        elif module.module_type == 'residual_cleaner':
            effect = module.apply(config)
            agent.residual_damage *= (1 - effect * 0.5)
            module.contribution += 0.1 * module.strength
        elif module.module_type == 'embedding_protector':
            effect = module.apply(config)
            agent.embedding_staleness *= (1 - effect * 0.3)
            module.contribution += 0.05 * module.strength
        elif module.module_type == 'pattern_detector':
            # Pattern detectors help predict and prepare
            effect = module.apply(config)
            agent.protective_stance = min(1.0, agent.protective_stance + effect * 0.1)
            module.contribution += 0.05 * module.strength
        elif module.module_type == 'anticipation_enhancer':
            effect = module.apply(config)
            # Improves threat prediction accuracy
            module.contribution += 0.05 * module.strength

    # Degradation penalty
    rate *= (1 - agent.degradation_level * config.degradation_penalty)

    recovery = min(1.0 - agent.IC_t, rate)
    pre_IC = agent.IC_t
    agent.IC_t += recovery

    # Degradation recovery (very slow)
    if agent.degradation_level > 0:
        agent.degradation_level *= config.degrad_recovery_factor

    # Corruption decay
    agent.history_corruption *= config.corruption_decay
    agent.prediction_noise *= config.corruption_decay
    agent.embedding_staleness *= 0.95
    agent.residual_damage *= 0.97

    success = agent.IC_t > pre_IC + 0.01
    if success:
        agent.successful_recoveries += 1

    return recovery, success

def spread_modules(agents: list[EvolvableAgent], cluster_id: int,
                   config: EvolvableConfig) -> int:
    """
    Module spreading with configurable thresholds.
    """
    cluster_agents = [a for a in agents if a.cluster_id == cluster_id and a.is_alive()]
    if len(cluster_agents) < 2:
        return 0

    spread_count = 0

    for agent in cluster_agents:
        for module in agent.modules:
            should_spread = (
                module.consolidated or
                (module.contribution > config.spread_threshold and
                 module.activation_count > config.min_activations)
            )

            if should_spread:
                for other in cluster_agents:
                    if other.agent_id != agent.agent_id:
                        if not other.has_module_type(module.module_type):
                            if np.random.random() < config.spread_probability:
                                spread_module = EvolvableMicroModule(
                                    module_type=module.module_type,
                                    strength=module.strength * config.spread_strength_factor,
                                    is_learned=True
                                )
                                other.modules.append(spread_module)
                                other.modules_received += 1
                                spread_count += 1

                                # Cap modules
                                if len(other.modules) > config.module_cap:
                                    weakest = min(other.modules, key=lambda m: m.strength)
                                    other.modules.remove(weakest)

    return spread_count

def consolidate_modules(agent: EvolvableAgent, config: EvolvableConfig):
    """Consolidate successful modules."""
    to_remove = []
    for module in agent.modules:
        module.age_tick()

        if not module.consolidated and module.contribution > config.consolidation_threshold:
            module.consolidated = True
            module.strength = min(1.0, module.strength * 1.15)
        elif not module.consolidated and module.activation_count > 8 and module.contribution > 0:
            module.consolidated = True
            module.strength = min(1.0, module.strength * 1.10)
        elif not module.consolidated and module.contribution < -0.2:
            to_remove.append(module)
        elif module.strength < 0.15:
            to_remove.append(module)

    for m in to_remove:
        if m in agent.modules:
            agent.modules.remove(m)

def create_proactive_module(agent: EvolvableAgent, config: EvolvableConfig) -> bool:
    """Create proactive module based on threat anticipation."""
    if len(agent.modules) >= config.module_cap:
        return False

    # Choose module type based on vulnerability
    if agent.get_embedding_integrity() < 0.5:
        module_type = 'embedding_protector'
    elif agent.degradation_level > 0.3:
        module_type = 'cascade_breaker'
    elif agent.anticipated_damage > 0.2:
        module_type = 'threat_filter'
    else:
        module_type = np.random.choice([
            'pattern_detector', 'recovery_accelerator',
            'exploration_dampener', 'anticipation_enhancer'
        ])

    if not agent.has_module_type(module_type):
        new_module = EvolvableMicroModule(
            module_type=module_type,
            strength=0.4 + np.random.random() * 0.2
        )
        agent.modules.append(new_module)
        agent.modules_created += 1
        agent.proactive_modules_created += 1
        return True

    return False

def update_temporal_anticipation(agent: EvolvableAgent, damage: float):
    """Update threat anticipation based on recent damage."""
    agent.threat_history.append(damage)
    if len(agent.threat_history) > 10:
        agent.threat_history.pop(0)

    if len(agent.threat_history) >= 3:
        recent = agent.threat_history[-3:]
        trend = (recent[-1] - recent[0]) / max(0.01, recent[0] + 0.01)
        agent.anticipated_damage = max(0, recent[-1] * (1 + trend * 0.5))
    else:
        agent.anticipated_damage = damage

    # Update threat buffer
    agent.threat_buffer = 0.7 * agent.threat_buffer + 0.3 * damage

def update_cluster(cluster: ClusterState, agents: list[EvolvableAgent]):
    """Update cluster state based on member agents."""
    members = [a for a in agents if a.cluster_id == cluster.cluster_id and a.is_alive()]
    cluster.size = len(members)

    if cluster.size == 0:
        cluster.cohesion = 0.0
        return

    # Cohesion based on IC similarity
    ics = [a.IC_t for a in members]
    if len(ics) > 1:
        cluster.cohesion = 1.0 - np.std(ics)
    else:
        cluster.cohesion = 0.5

# =============================================================================
# STORM GENERATION
# =============================================================================

def create_storm(damage_multiplier: float, n_steps: int = 150) -> list[PerturbationWave]:
    """Create cascading storm with 5 wave types."""
    waves = []
    # Base damages calibrated for survival with damage_multiplier ~2-4
    # These are much lower than original to allow evolution to explore
    base_damages = {
        'history': 0.015,
        'prediction': 0.020,
        'social': 0.012,
        'identity': 0.025,
        'catastrophic': 0.030
    }

    wave_types = list(base_damages.keys())
    interval = n_steps // (len(wave_types) + 1)

    for i, wave_type in enumerate(wave_types):
        step = (i + 1) * interval
        waves.append(PerturbationWave(
            wave_type=wave_type,
            base_damage=base_damages[wave_type] * damage_multiplier,
            step=step
        ))

    return waves

def apply_wave_damage(agent: EvolvableAgent, wave: PerturbationWave,
                      config: EvolvableConfig, step: int) -> float:
    """Apply wave-specific damage to agent."""
    if step < wave.step or step > wave.step + 15:
        return 0.0

    # Damage decays over wave duration
    progress = (step - wave.step) / 15
    damage = wave.base_damage * (1 - progress * 0.5)

    # Wave-specific effects
    if wave.wave_type == 'history':
        agent.history_corruption += damage * 0.3
    elif wave.wave_type == 'prediction':
        agent.prediction_noise += damage * 0.4
    elif wave.wave_type == 'social':
        agent.embedding_staleness += damage * 0.2
    elif wave.wave_type == 'identity':
        damage *= 1.2  # Identity waves hit harder

    return gradual_damage(agent, damage, config)

# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def create_agents(n_agents: int, n_clusters: int,
                  config: EvolvableConfig) -> list[EvolvableAgent]:
    """Create agents distributed across clusters."""
    agents = []
    for i in range(n_agents):
        agent = EvolvableAgent(
            agent_id=i,
            cluster_id=i % n_clusters,
            cluster_embedding=np.random.randn(8) * 0.5
        )
        agents.append(agent)
    return agents

def create_clusters(n_clusters: int) -> list[ClusterState]:
    """Create cluster states."""
    return [ClusterState(cluster_id=i) for i in range(n_clusters)]

def run_single_simulation(agents: list[EvolvableAgent],
                          clusters: list[ClusterState],
                          storm: list[PerturbationWave],
                          config: EvolvableConfig,
                          n_steps: int,
                          seed: int) -> dict[str, float]:
    """Run a single simulation and return metrics."""
    np.random.seed(seed)

    total_damage = 0.0
    total_spread = 0
    total_modules_created = 0

    for step in range(n_steps):
        # Apply storm waves
        for wave in storm:
            for agent in agents:
                if agent.is_alive():
                    damage = apply_wave_damage(agent, wave, config, step)
                    total_damage += damage
                    update_temporal_anticipation(agent, damage)

        # Proactive module creation
        for agent in agents:
            if agent.is_alive():
                if agent.threat_buffer > 0.1 or agent.anticipated_damage > 0.15:
                    if create_proactive_module(agent, config):
                        total_modules_created += 1

        # Recovery
        for agent in agents:
            if agent.is_alive():
                cluster = clusters[agent.cluster_id]
                gradual_recovery(agent, cluster, config)

        # Module spreading
        for cluster in clusters:
            spread = spread_modules(agents, cluster.cluster_id, config)
            total_spread += spread

        # Consolidate modules
        for agent in agents:
            consolidate_modules(agent, config)

        # Update clusters
        for cluster in clusters:
            update_cluster(cluster, agents)

        # Track IC history
        for agent in agents:
            agent.IC_history.append(agent.IC_t)

    # Calculate final metrics
    alive_agents = [a for a in agents if a.is_alive()]
    n_alive = len(alive_agents)

    # Holographic survival
    hs = n_alive / len(agents)

    # Embedding integrity (average of alive)
    if alive_agents:
        ei = np.mean([a.get_embedding_integrity() for a in alive_agents])
    else:
        ei = 0.0

    # Module spreading rate
    total_modules = sum(len(a.modules) for a in agents)
    modules_learned = sum(sum(1 for m in a.modules if m.is_learned) for a in agents)
    msr = modules_learned / max(1, total_modules) if total_modules > 0 else 0.0

    # Temporal anticipation effectiveness
    proactive_total = sum(a.proactive_modules_created for a in agents)
    tae = proactive_total / max(1, total_modules_created) if total_modules_created > 0 else 0.0

    # Emergent differentiation (variance in degradation)
    if alive_agents:
        degradations = [a.degradation_level for a in alive_agents]
        ed = np.std(degradations) if len(degradations) > 1 else 0.0
    else:
        ed = 0.0

    # Degradation variance
    all_degradations = [a.degradation_level for a in agents]
    deg_var = np.var(all_degradations) if len(all_degradations) > 1 else 0.0

    return {
        'holographic_survival': hs,
        'embedding_integrity': ei,
        'module_spreading_rate': msr,
        'temporal_anticipation_effectiveness': tae,
        'emergent_differentiation': ed,
        'degradation_variance': deg_var,
        'final_alive': n_alive,
        'total_damage': total_damage,
        'total_spread': total_spread,
        'modules_total': total_modules,
        'modules_learned': modules_learned,
        'proactive_created': proactive_total,
    }

def run_baseline_simulation(n_agents: int, n_clusters: int,
                            n_steps: int, seed: int,
                            damage_multiplier: float) -> dict[str, float]:
    """Run baseline simulation without advanced features."""
    np.random.seed(seed)

    # Simple agents without embeddings or modules
    agents = []
    for i in range(n_agents):
        agent = EvolvableAgent(
            agent_id=i,
            cluster_id=i % n_clusters,
            embedding_dim=0  # No embeddings
        )
        agents.append(agent)

    clusters = create_clusters(n_clusters)
    storm = create_storm(damage_multiplier, n_steps)

    # Simple config with no protection
    simple_config = EvolvableConfig(
        embedding_protection=0.0,
        stance_protection=0.0,
        module_protection=0.0
    )

    for step in range(n_steps):
        for wave in storm:
            for agent in agents:
                if agent.is_alive():
                    apply_wave_damage(agent, wave, simple_config, step)

        # Minimal recovery
        for agent in agents:
            if agent.is_alive():
                agent.IC_t = min(1.0, agent.IC_t + 0.02)

    alive = len([a for a in agents if a.is_alive()])
    return {'baseline_survival': alive / n_agents}

def run_ipuesa_with_config(config: dict[str, Any],
                           n_agents: int = 24,
                           n_clusters: int = 4,
                           n_steps: int = 150,
                           n_runs: int = 8) -> dict[str, float]:
    """
    Main entry point for OpenAlpha evaluation.

    Runs IPUESA simulation with injected config and returns
    aggregated metrics.
    """
    cfg = EvolvableConfig.from_dict(config)

    all_results = []

    for run in range(n_runs):
        agents = create_agents(n_agents, n_clusters, cfg)
        clusters = create_clusters(n_clusters)
        storm = create_storm(cfg.damage_multiplier, n_steps)

        result = run_single_simulation(
            agents=agents,
            clusters=clusters,
            storm=storm,
            config=cfg,
            n_steps=n_steps,
            seed=run * 42
        )
        all_results.append(result)

    # Also run baseline for comparison
    baseline = run_baseline_simulation(
        n_agents, n_clusters, n_steps,
        seed=999, damage_multiplier=cfg.damage_multiplier
    )

    # Run no-embedding variant for gradient check
    no_emb_results = []
    for run in range(min(3, n_runs)):
        agents = create_agents(n_agents, n_clusters, cfg)
        for a in agents:
            a.embedding_dim = 0
            a.cluster_embedding = np.zeros(8)
        clusters = create_clusters(n_clusters)
        storm = create_storm(cfg.damage_multiplier, n_steps)

        result = run_single_simulation(
            agents=agents,
            clusters=clusters,
            storm=storm,
            config=cfg,
            n_steps=n_steps,
            seed=run * 42 + 1000
        )
        no_emb_results.append(result)

    # Aggregate results
    aggregated = {}
    keys = all_results[0].keys()
    for key in keys:
        values = [r[key] for r in all_results]
        aggregated[key] = float(np.mean(values))

    # Add baseline and no-embedding
    aggregated['baseline_survival'] = baseline['baseline_survival']
    if no_emb_results:
        aggregated['no_embedding_survival'] = float(np.mean(
            [r['holographic_survival'] for r in no_emb_results]
        ))
    else:
        aggregated['no_embedding_survival'] = 0.0

    return aggregated

if __name__ == '__main__':
    # Quick test
    print("Testing IPUESA Evolvable...")

    config = EvolvableConfig()
    results = run_ipuesa_with_config(
        config.to_dict(),
        n_agents=12,
        n_clusters=3,
        n_steps=50,
        n_runs=2
    )

    print("\nResults:")
    for key, value in sorted(results.items()):
        print(f"  {key}: {value:.4f}")
