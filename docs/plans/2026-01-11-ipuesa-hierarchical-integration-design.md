# IPUESA → HierarchicalSimulation Integration Design

**Date**: 2026-01-11
**Status**: Approved
**Author**: Claude + Francisco Ruiz

## Overview

This document describes the integration of IPUESA (Identity-Preserving Unified Emergent Self-Architecture) mechanisms into the existing HierarchicalSimulation consciousness system. The goal is to port the successful resilience mechanisms (gradual damage, micro-modules, temporal anticipation) that achieved 8/8 self-evidence criteria in IPUESA-SYNTH-v2.

### Integration Approach

**Selected**: Port mechanisms (not merge systems)
- Add CellResilience to existing ConsciousCell
- Integrate DamageSystem into HierarchicalSimulation.step()
- Keep 3-level hierarchy (Cells → Clusters → Organism)
- Use evolved parameters from 50-generation optimization

### Key Benefits

1. Proven resilience mechanics (fitness=0.9993)
2. Module spreading within cluster structure
3. Temporal anticipation for proactive protection
4. Gradual degradation states (not binary alive/dead)

---

## Section 1: Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL SIMULATION                       │
│                    (3-level consciousness)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   ORGANISM    │     │   CLUSTERS    │     │    CELLS      │
│               │     │               │     │               │
│ global_state  │◄────│ archetype_    │◄────│ archetype_    │
│ coherence     │     │ profile       │     │ weights       │
│               │     │ cohesion      │     │               │
│ +-----------+ │     │ +-----------+ │     │ +-----------+ │
│ |Organism   | │     │ |Cluster    | │     │ |Cell       | │
│ |Resilience | │     │ |Resilience | │     │ |Resilience | │
│ +-----------+ │     │ +-----------+ │     │ +-----------+ │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │   DAMAGE SYSTEM   │
                    │                   │
                    │ - apply_damage()  │
                    │ - apply_recovery()│
                    │ - spread_modules()│
                    │                   │
                    │ Config: Evolved   │
                    │ (50 generations)  │
                    └───────────────────┘
```

### Resilience at Each Level

| Level | Resilience Component | Key Metrics |
|-------|---------------------|-------------|
| Cell | CellResilience | degradation, modules, threat_buffer |
| Cluster | cluster_resilience (property) | mean degradation, functional_ratio |
| Organism | organism_resilience (property) | global coherence, total survival |

---

## Section 2: New Components

### 2.1 CellResilience

```python
@dataclass
class CellResilience:
    """Resilience state for a single cell."""

    # Core state
    degradation_level: float = 0.0  # 0=optimal, 1=collapsed
    residual_damage: float = 0.0    # Accumulated damage that persists

    # Micro-modules
    modules: List[MicroModule] = field(default_factory=list)

    # Temporal anticipation
    threat_buffer: float = 0.0      # EMA of recent damage
    anticipated_damage: float = 0.0  # Predicted future damage
    protective_stance: float = 0.0   # Proactive protection level

    # Optional embedding (for holographic protection)
    embedding: np.ndarray = None
    embedding_strength: float = 1.0

    @property
    def state(self) -> str:
        """Degradation state category."""
        if self.degradation_level < 0.2:
            return 'OPTIMAL'
        elif self.degradation_level < 0.4:
            return 'STRESSED'
        elif self.degradation_level < 0.6:
            return 'IMPAIRED'
        elif self.degradation_level < 0.8:
            return 'CRITICAL'
        else:
            return 'COLLAPSED'

    @property
    def is_functional(self) -> bool:
        """Cell is functional if not collapsed."""
        return self.degradation_level < 0.8
```

### 2.2 MicroModule

```python
@dataclass
class MicroModule:
    """Emergent protective module."""

    module_type: str  # One of 8 types
    strength: float = 0.5
    activations: int = 0
    contribution: float = 0.0  # Net contribution to survival

    # Module types and their effects:
    # - pattern_detector: Recognize threat patterns
    # - threat_filter: Reduce incoming damage
    # - recovery_accelerator: Speed up recovery
    # - exploration_dampener: Reduce exploration under stress
    # - embedding_protector: Preserve embedding integrity
    # - cascade_breaker: Prevent damage cascades
    # - residual_cleaner: Clear accumulated residual
    # - anticipation_enhancer: Improve threat prediction

    def apply(self, config: dict) -> float:
        """Apply module effect, return effect magnitude."""
        effect_key = f'effect_{self.module_type}'
        base_effect = config.get(effect_key, 0.1)

        self.activations += 1
        return base_effect * self.strength

    def decay(self, rate: float = 0.95):
        """Decay strength from disuse."""
        self.strength *= rate

    def is_consolidated(self, min_activations: int = 3) -> bool:
        """Check if module is consolidated (eligible for spreading)."""
        return self.activations >= min_activations and self.contribution > 0
```

### 2.3 DamageSystem

```python
class DamageSystem:
    """Manages damage and recovery for cells."""

    def __init__(self, config: dict):
        self.config = config

    def apply_damage(self, cell, resilience: CellResilience,
                     base_damage: float) -> float:
        """Apply damage to cell, return actual damage dealt."""
        cfg = self.config

        # 1. Base calculation with multiplier
        damage = base_damage * cfg['damage']['multiplier']

        # 2. Module mitigation
        for module in resilience.modules:
            if module.module_type in ['threat_filter', 'cascade_breaker']:
                effect = module.apply(cfg['modules']['effects'])
                damage *= (1.0 - effect)
                module.contribution += 0.1 * module.strength

        # 3. Protection factors
        damage *= (1.0 - cfg['protection']['cell']['embedding'] * resilience.embedding_strength)
        damage *= (1.0 - cfg['protection']['cell']['stance'] * resilience.protective_stance)

        # 4. Noise for anti-fragility
        noise = np.random.normal(0, cfg['damage']['noise_scale'])
        damage *= (1.0 + noise)
        damage = max(0, damage)

        # 5. Apply to degradation
        resilience.degradation_level += damage * cfg['damage']['base_degrad_rate']
        resilience.degradation_level = min(1.0, resilience.degradation_level)

        # 6. Accumulate residual
        resilience.residual_damage += damage * cfg['damage']['compound_factor']
        resilience.residual_damage = min(cfg['damage']['residual_cap'],
                                         resilience.residual_damage)

        # 7. Update threat buffer for TAE
        alpha = cfg['anticipation']['buffer_alpha']
        resilience.threat_buffer = alpha * damage + (1 - alpha) * resilience.threat_buffer

        return damage

    def apply_recovery(self, cell, resilience: CellResilience,
                       cluster_cohesion: float = 0.5) -> float:
        """Apply recovery to cell, return recovery amount."""
        cfg = self.config

        # 1. Base recovery rate
        rate = cfg['recovery']['base_rate']

        # 2. Module bonuses
        for module in resilience.modules:
            if module.module_type == 'recovery_accelerator':
                effect = module.apply(cfg['modules']['effects'])
                rate *= (1 + effect)
                module.contribution += 0.1 * module.strength
            elif module.module_type == 'residual_cleaner':
                effect = module.apply(cfg['modules']['effects'])
                resilience.residual_damage *= (1 - effect)
                module.contribution += 0.05 * module.strength

        # 3. Cluster cohesion bonus
        rate += cfg['recovery']['cluster_bonus'] * cluster_cohesion

        # 4. Embedding bonus
        rate += cfg['recovery']['embedding_bonus'] * resilience.embedding_strength * 0.1

        # 5. Degradation penalty (harder to recover when damaged)
        penalty = resilience.degradation_level * cfg['recovery']['degradation_penalty']
        rate *= (1 - penalty)

        # 6. Apply recovery
        recovery = resilience.degradation_level * rate
        resilience.degradation_level -= recovery
        resilience.degradation_level = max(0, resilience.degradation_level)

        # 7. Residual decay
        resilience.residual_damage *= cfg['recovery']['corruption_decay']

        return recovery
```

---

## Section 3: Modifications to Existing Classes

### 3.1 ConsciousCell (micro_psyche.py)

```python
@dataclass
class ConsciousCell:
    # Existing fields
    position: np.ndarray
    archetype_weights: np.ndarray
    plasticity: float = 1.0
    accumulated_surprise: float = 0.0

    # NEW: Resilience state
    resilience: CellResilience = field(default_factory=CellResilience)

    @property
    def effective_plasticity(self) -> float:
        """Plasticity modulated by degradation."""
        degradation_penalty = self.resilience.degradation_level * 0.5
        return self.plasticity * (1.0 - degradation_penalty)

    @property
    def is_functional(self) -> bool:
        """Cell is functional if degradation < critical threshold."""
        return self.resilience.degradation_level < 0.8

    def update_with_resilience(self, stimulus: np.ndarray,
                                damage_system: DamageSystem,
                                cluster_cohesion: float = 0.5):
        """Update that includes damage/recovery cycle."""
        # 1. Apply damage if stressful stimulus
        stress_level = np.linalg.norm(stimulus)
        if stress_level > 0.3:
            damage_system.apply_damage(self, self.resilience, stress_level)

        # 2. Natural recovery
        damage_system.apply_recovery(self, self.resilience, cluster_cohesion)

        # 3. Normal archetype update (only if functional)
        if self.is_functional:
            self._update_archetypes(stimulus)
```

### 3.2 Cluster (cluster.py)

```python
@dataclass
class Cluster:
    cells: List[ConsciousCell]
    cluster_id: int

    @property
    def cluster_resilience(self) -> float:
        """Average resilience of functional cells."""
        functional = [c for c in self.cells if c.is_functional]
        if not functional:
            return 0.0
        return 1.0 - np.mean([c.resilience.degradation_level
                              for c in functional])

    @property
    def functional_ratio(self) -> float:
        """Proportion of functional cells."""
        return sum(1 for c in self.cells if c.is_functional) / len(self.cells)

    def spread_modules(self, config: dict):
        """Spread consolidated modules to neighbors."""
        # Find consolidated modules
        consolidated = []
        for cell in self.cells:
            for module in cell.resilience.modules:
                if module.activations >= config['modules']['spreading']['min_activations']:
                    if module.contribution > 0:
                        consolidated.append((cell, module))

        # Spread to neighbors
        for source_cell, module in consolidated:
            if np.random.random() < config['modules']['spreading']['probability']:
                for target_cell in self.cells:
                    if target_cell is source_cell:
                        continue
                    has_type = any(m.module_type == module.module_type
                                   for m in target_cell.resilience.modules)
                    if not has_type:
                        # Create weakened copy
                        new_module = MicroModule(
                            module_type=module.module_type,
                            strength=module.strength * config['modules']['spreading']['strength_factor']
                        )
                        target_cell.resilience.modules.append(new_module)
                        break
```

### 3.3 HierarchicalSimulation (hierarchical_simulation.py)

```python
class HierarchicalSimulation:
    def __init__(self, n_cells: int, config: dict = None, preset: str = 'optimal'):
        # Existing initialization
        self.cells = [ConsciousCell(...) for _ in range(n_cells)]
        self.clusters = []
        self.organism = OrganismConsciousness()

        # NEW: Damage system with evolved config
        from zeta_life.consciousness.resilience_config import get_preset_config
        self.resilience_config = config or get_preset_config(preset)
        self.damage_system = DamageSystem(self.resilience_config)

        # NEW: Resilience metrics history
        self.resilience_history = []

    def step(self, external_stimulus: np.ndarray = None):
        """Step with resilience integration."""
        # 1. Bottom-up: cells → clusters → organism
        self._bottom_up_pass()

        # 2. NEW: Damage and recovery per cell
        for cluster in self.clusters:
            cohesion = cluster.cohesion
            for cell in cluster.cells:
                if external_stimulus is not None:
                    self.damage_system.apply_damage(
                        cell, cell.resilience,
                        np.linalg.norm(external_stimulus) * 0.5
                    )
                self.damage_system.apply_recovery(
                    cell, cell.resilience, cohesion
                )

        # 3. NEW: Module spreading intra-cluster
        for cluster in self.clusters:
            cluster.spread_modules(self.resilience_config)

        # 4. Top-down: organism → clusters → cells
        self._top_down_pass()

        # 5. NEW: Record resilience metrics
        self._record_resilience_metrics()

    def _record_resilience_metrics(self):
        """Record IPUESA-compatible metrics."""
        functional = [c for c in self.cells if c.is_functional]

        metrics = {
            'HS': len(functional) / len(self.cells),
            'mean_degradation': np.mean([c.resilience.degradation_level
                                          for c in self.cells]),
            'total_modules': sum(len(c.resilience.modules) for c in self.cells),
            'functional_clusters': sum(1 for cl in self.clusters
                                        if cl.functional_ratio > 0.5)
        }
        self.resilience_history.append(metrics)

    @classmethod
    def from_evolved_config(cls, n_cells: int = 100, **kwargs):
        """Factory using evolved configuration."""
        from zeta_life.evolution import get_optimized_dict
        from zeta_life.consciousness.resilience_config import create_hierarchical_config
        config = create_hierarchical_config(get_optimized_dict())
        return cls(n_cells=n_cells, config=config, **kwargs)
```

---

## Section 4: Data Flow

### 4.1 Main Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL SIMULATION STEP                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. BOTTOM-UP PASS (existing)                                         │
│    Cell.archetype_weights ──aggregate──► Cluster.archetype_profile   │
│    Cluster.archetype_profile ──aggregate──► Organism.global_state    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. DAMAGE PHASE (NEW)                                                │
│                                                                      │
│    external_stimulus ───┐                                            │
│                         ▼                                            │
│    For each cell:                                                    │
│      base_damage = stimulus * multiplier                             │
│      → Module mitigation (threat_filter, cascade_breaker)            │
│      → Protection factors (embedding, stance)                        │
│      → Apply noise for anti-fragility                                │
│      → Update degradation_level and residual_damage                  │
│      → Update threat_buffer for TAE                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. RECOVERY PHASE (NEW)                                              │
│                                                                      │
│    For each cell:                                                    │
│      base_rate = config.recovery_rate                                │
│      → Module bonuses (recovery_accelerator, residual_cleaner)       │
│      → Cluster cohesion bonus                                        │
│      → Embedding bonus                                               │
│      → Degradation penalty                                           │
│      → Apply recovery to degradation_level                           │
│      → Decay residual_damage                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. MODULE SPREADING (NEW)                                            │
│                                                                      │
│    For each cluster:                                                 │
│      Find consolidated modules (activations >= threshold, contrib>0) │
│      For each consolidated module:                                   │
│        if random() < spread_probability:                             │
│          Copy to neighbor cell (strength * 0.5)                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. TOP-DOWN PASS (existing, modified)                                │
│                                                                      │
│    Organism.global_state ──modulate──► Cluster.target_profile        │
│    Cluster.target_profile ──modulate──► Cell.archetype_weights       │
│                                                                      │
│    NEW: Modulation scaled by functional_ratio                        │
│         top_down_strength *= cluster.functional_ratio                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. METRICS RECORDING (NEW)                                           │
│                                                                      │
│    HS  = functional_cells / total_cells                              │
│    MSR = modules_spread / eligible_modules                           │
│    EI  = mean(embedding_strength)                                    │
│    ED  = std(degradation_level)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module Lifecycle

```
    CREATION                CONSOLIDATION              DECAY
        │                        │                        │
        ▼                        ▼                        ▼
  ┌───────────┐           ┌───────────┐           ┌───────────┐
  │ strength  │           │ strength  │           │ strength  │
  │ = 0.5     │──use───►  │ += 0.1    │──disuse─► │ *= 0.95   │
  │ activ = 0 │           │ activ++   │           │           │
  └───────────┘           └───────────┘           └───────────┘
        │                        │                        │
        │                        ▼                        ▼
        │               ┌─────────────────┐      ┌─────────────────┐
        │               │ If activ >= 3   │      │ If strength     │
        │               │ && contrib > 0  │      │ < 0.1           │
        │               │ → CONSOLIDATED  │      │ → REMOVE        │
        │               └─────────────────┘      └─────────────────┘
        │                        │
        │                        ▼
        │               ┌─────────────────┐
        └──────────────►│ ELIGIBLE FOR    │
                        │ SPREADING       │
                        └─────────────────┘
```

---

## Section 5: Configuration

### 5.1 Parameter Mapping

```python
def create_hierarchical_config(
    base_config: dict = None,
    scale_factor: float = 1.0
) -> dict:
    """
    Create configuration for HierarchicalSimulation from evolved params.

    scale_factor allows intensity adjustment:
    - scale_factor < 1.0: Softer (for demos/visualization)
    - scale_factor = 1.0: Optimal IPUESA calibration
    - scale_factor > 1.0: More intense (for stress tests)
    """
    from zeta_life.evolution import get_optimized_dict
    evolved = base_config or get_optimized_dict()

    return {
        'damage': {
            'multiplier': evolved['damage_multiplier'] * 0.88 * scale_factor,
            'base_degrad_rate': evolved['base_degrad_rate'],
            'compound_factor': evolved['compound_factor'],
            'noise_scale': evolved['noise_scale'],
            'residual_cap': evolved['residual_cap'],
        },
        'protection': {
            'cell': {
                'embedding': evolved['embedding_protection'],
                'stance': evolved['stance_protection'],
                'module': evolved['module_protection'],
                'resilience_min': evolved['resilience_min'],
                'resilience_range': evolved['resilience_range'],
            },
            'cluster': {
                'cohesion_bonus': evolved['cluster_bonus'],
                'min_functional_ratio': 0.3,
            },
            'organism': {
                'coherence_bonus': 0.1,
                'top_down_protection': 0.05,
            },
        },
        'recovery': {
            'base_rate': evolved['base_recovery_rate'],
            'embedding_bonus': evolved['embedding_bonus'],
            'cluster_bonus': evolved['cluster_bonus'],
            'degradation_penalty': evolved['degradation_penalty'],
            'degrad_recovery_factor': evolved['degrad_recovery_factor'],
            'corruption_decay': evolved['corruption_decay'],
        },
        'modules': {
            'effects': {
                'pattern_detector': evolved['effect_pattern_detector'],
                'threat_filter': evolved['effect_threat_filter'],
                'recovery_accelerator': evolved['effect_recovery_accelerator'],
                'exploration_dampener': evolved['effect_exploration_dampener'],
                'embedding_protector': evolved['effect_embedding_protector'],
                'cascade_breaker': evolved['effect_cascade_breaker'],
                'residual_cleaner': evolved['effect_residual_cleaner'],
                'anticipation_enhancer': evolved['effect_anticipation_enhancer'],
            },
            'spreading': {
                'threshold': evolved['spread_threshold'],
                'probability': evolved['spread_probability'],
                'strength_factor': evolved['spread_strength_factor'],
                'min_activations': int(evolved['min_activations']),
                'max_per_cell': int(evolved['module_cap']),
            },
            'consolidation_threshold': evolved['consolidation_threshold'],
        },
        'anticipation': {
            'buffer_alpha': 0.3,
            'vulnerability_threshold': 0.5,
            'creation_probability': 0.4,
            'momentum_factor': 1.2,
        },
    }
```

### 5.2 Presets

| Preset | scale_factor | Use Case |
|--------|--------------|----------|
| `demo` | 0.6 | Visualization, soft dynamics |
| `optimal` | 1.0 | Goldilocks zone, 8/8 criteria |
| `stress` | 1.5 | Stress testing, find limits |
| `validation` | 1.0 + override | Exact reproduction of experiments |

### 5.3 Parameter Reference

| Evolved Parameter | Optimal Value | Hierarchical Use | Effect |
|-------------------|---------------|------------------|--------|
| `damage_multiplier` | 4.425 | `damage.multiplier` × 0.88 | Base damage intensity |
| `base_recovery_rate` | 0.098 | `recovery.base_rate` | Recovery speed |
| `embedding_protection` | 0.05 | `protection.cell.embedding` | Damage reduction from embedding |
| `spread_probability` | 0.48 | `modules.spreading.probability` | Chance to copy module |
| `min_activations` | 4.57→5 | `modules.spreading.min_activations` | Uses before consolidation |
| `compound_factor` | 0.25 | `damage.compound_factor` | Damage cascade factor |
| `cluster_bonus` | 0.20 | `recovery.cluster_bonus` | Cohesion bonus |
| `noise_scale` | 0.37 | `damage.noise_scale` | Variability (anti-fragility) |

---

## Section 6: Testing Strategy

### 6.1 Unit Tests

- `TestCellResilience`: Initial state, degradation states, is_functional
- `TestMicroModule`: apply(), decay(), is_consolidated()
- `TestDamageSystem`: apply_damage(), apply_recovery(), module mitigation

### 6.2 Integration Tests

- `TestHierarchicalIntegration`: cells_have_resilience, damage_system_initialized, step_applies_damage_and_recovery, module_spreading_occurs, metrics_recorded
- `TestPresets`: All presets load, stress more damaging than demo

### 6.3 Self-Evidence Validation (8 Criteria)

| # | Criterion | Test |
|---|-----------|------|
| 1 | HS in [0.30, 0.70] | Survival in Goldilocks zone |
| 2 | MSR > 0.15 | Modules spread |
| 3 | EI > 0.3 | Embeddings preserved |
| 4 | ED > 0.10 | Cell differentiation |
| 5 | full > baseline | Full config beats baseline |
| 6 | Gradient valid | demo > optimal > stress |
| 7 | deg_var > 0.02 | Smooth (not bistable) |
| 8 | Modules emerge | > 10 modules created |

### 6.4 Validation Script

```bash
python experiments/consciousness/exp_validate_integration.py
```

Expected output:
```
VALIDACIÓN DE INTEGRACIÓN IPUESA → HIERARCHICAL
============================================================

Resultados con preset 'optimal':
  HS  = 0.396
  MSR = 0.501
  EI  = 1.000
  ED  = 0.360

Criterios de Self-Evidence:
  ✓ PASS: HS en [0.30, 0.70]
  ✓ PASS: MSR > 0.15
  ✓ PASS: EI > 0.3
  ✓ PASS: ED > 0.10
  ✓ PASS: full > baseline
  ✓ PASS: Gradiente válido
  ✓ PASS: deg_var > 0.02
  ✓ PASS: Módulos emergen

Total: 8/8 criterios
```

---

## Implementation Plan

### Phase 1: Core Components
1. Create `src/zeta_life/consciousness/resilience.py` with CellResilience, MicroModule
2. Create `src/zeta_life/consciousness/damage_system.py` with DamageSystem
3. Create `src/zeta_life/consciousness/resilience_config.py` with config mapping

### Phase 2: Integration
4. Modify `ConsciousCell` to include resilience field
5. Modify `Cluster` to add spread_modules() and cluster_resilience
6. Modify `HierarchicalSimulation` to integrate DamageSystem in step()

### Phase 3: Testing
7. Write unit tests for new components
8. Write integration tests
9. Write self-evidence validation tests

### Phase 4: Validation
10. Run full validation with cascading storm
11. Verify 8/8 criteria pass
12. Document results

---

## References

- `experiments/consciousness/exp_ipuesa_synth_v2.py` - Source of mechanics
- `src/zeta_life/evolution/optimized_config.py` - Evolved parameters
- `docs/plans/2026-01-10-ipuesa-synth-v2-design.md` - SYNTH-v2 design
- `CLAUDE.md` Section 6.17 - IPUESA-SYNTH-v2 results
