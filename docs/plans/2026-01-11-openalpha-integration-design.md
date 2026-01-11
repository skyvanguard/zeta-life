# OpenAlpha_Evolve Integration Design - Phase 1

**Date:** 2026-01-11
**Author:** Claude + Francisco Ruiz
**Status:** Approved

## Overview

This document describes the integration of [OpenAlpha_Evolve](https://github.com/shyamsaktawat/OpenAlpha_Evolve) with Zeta Life to automatically optimize IPUESA hyperparameters using evolutionary algorithms.

### Objective

Use evolutionary optimization to discover the optimal configuration of ~30 IPUESA-SYNTH hyperparameters that maximize self-evidence criteria, replacing manual trial-and-error tuning.

### Why OpenAlpha_Evolve?

| Aspect | OpenAlpha_Evolve | Zeta Life |
|--------|------------------|-----------|
| **Paradigm** | Code evolution with LLMs | Artificial life + emergence |
| **Architecture** | Specialized multi-agent | Emergent multi-agent |
| **Self-improvement** | Code evolves iteratively | Consciousness/identity emerges |
| **Stack** | Python, LiteLLM, Docker | Python, PyTorch, NumPy |

The concepts are deeply compatible - both use evolutionary principles (selection, mutation, fitness) and seek emergence from iterative processes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Evolution Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Population  │  │ Selection   │  │ Mutation +          │ │
│  │ Manager     │→ │ (Tournament)│→ │ Crossover           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         ↑                                    │              │
│         └────────── Fitness ←────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Zeta Life                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  EvolvableConfig (30 parameters)                    │   │
│  │  - damage_multiplier, base_degrad_rate              │   │
│  │  - recovery_factors, thresholds                     │   │
│  │  - MicroModule.EFFECTS                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  IPUESA Evolvable Simulation                        │   │
│  │  → Runs with injected config                        │   │
│  │  → Returns: passed_criteria, metrics                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
zeta-life/
├── src/zeta_life/evolution/           # NEW: Evolution module
│   ├── __init__.py
│   ├── config_space.py               # PARAM_RANGES, EvolvableConfig
│   ├── fitness_evaluator.py          # evaluate_config(), FitnessResult
│   └── ipuesa_evolvable.py           # Parameterized IPUESA functions
├── experiments/evolution/             # NEW: Evolution experiments
│   └── exp_evolve_ipuesa.py          # Main orchestrator
├── tasks/                             # NEW: OpenAlpha task definitions
│   └── ipuesa_optimization.yaml      # Task definition (optional)
├── evolved_configs/                   # NEW: Discovered configs
│   ├── checkpoint_gen*.json
│   ├── best_config_*.json
│   └── evolved_config_*.json
└── docs/plans/
    └── 2026-01-11-openalpha-integration-design.md
```

## Parameter Space

### Group A: Damage and Degradation (10 params)

| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `damage_multiplier` | 3.9 | [1.5, 5.0] | Storm damage scaling |
| `base_degrad_rate` | 0.18 | [0.05, 0.30] | Base degradation rate |
| `embedding_protection` | 0.15 | [0.05, 0.40] | Embedding damage reduction |
| `stance_protection` | 0.12 | [0.05, 0.25] | Protective stance effect |
| `compound_factor` | 0.5 | [0.2, 0.8] | Degradation compounding |
| `module_protection` | 0.08 | [0.03, 0.20] | Module damage reduction |
| `resilience_min` | 0.3 | [0.1, 0.5] | Minimum resilience |
| `resilience_range` | 1.4 | [0.8, 2.0] | Resilience variation |
| `noise_scale` | 0.25 | [0.10, 0.40] | Damage noise |
| `residual_cap` | 0.35 | [0.20, 0.50] | Max residual damage |

### Group B: Recovery (6 params)

| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `base_recovery_rate` | 0.06 | [0.03, 0.12] | Base recovery rate |
| `embedding_bonus` | 0.6 | [0.3, 0.9] | Embedding recovery bonus |
| `cluster_bonus` | 0.3 | [0.1, 0.5] | Cluster support bonus |
| `degradation_penalty` | 0.4 | [0.2, 0.6] | Degradation recovery penalty |
| `degrad_recovery_factor` | 0.998 | [0.990, 0.999] | Degradation decay |
| `corruption_decay` | 0.94 | [0.90, 0.98] | Corruption recovery |

### Group C: Module Effects (8 params)

| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `effect_pattern_detector` | 0.20 | [0.10, 0.35] | Pattern detection strength |
| `effect_threat_filter` | 0.18 | [0.10, 0.30] | Threat filtering strength |
| `effect_recovery_accelerator` | 0.25 | [0.15, 0.40] | Recovery acceleration |
| `effect_exploration_dampener` | 0.15 | [0.08, 0.25] | Exploration dampening |
| `effect_embedding_protector` | 0.30 | [0.15, 0.45] | Embedding protection |
| `effect_cascade_breaker` | 0.22 | [0.12, 0.35] | Cascade breaking |
| `effect_residual_cleaner` | 0.20 | [0.10, 0.35] | Residual cleaning |
| `effect_anticipation_enhancer` | 0.25 | [0.15, 0.40] | Anticipation enhancement |

### Group D: Thresholds and Spreading (6 params)

| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `consolidation_threshold` | 0.10 | [0.05, 0.25] | Module consolidation threshold |
| `spread_threshold` | 0.15 | [0.08, 0.30] | Module spreading threshold |
| `spread_probability` | 0.30 | [0.15, 0.50] | Spread probability |
| `spread_strength_factor` | 0.45 | [0.30, 0.70] | Spread strength retention |
| `module_cap` | 6 | [4, 10] | Max modules per agent |
| `min_activations` | 3 | [2, 6] | Min activations for spread |

**Total: 30 evolvable parameters**

## Fitness Function

### Self-Evidence Criteria (8 total)

```python
criteria = {
    'HS_in_range': 0.30 <= HS <= 0.70,      # Goldilocks zone
    'MSR_pass': MSR > 0.15,                  # Module spreading
    'TAE_pass': TAE > 0.15,                  # Temporal anticipation
    'EI_pass': EI > 0.30,                    # Embedding integrity
    'ED_pass': ED > 0.10,                    # Emergent differentiation
    'diff_pass': HS > baseline_HS,           # Better than baseline
    'gradient_pass': gradient_valid,         # Valid gradient
    'smooth_transition': deg_var > 0.02,     # Not bistable
}
```

### Composite Fitness Score

```python
# 70% binary (criteria passed)
binary_score = sum(criteria.values()) / 8

# 30% continuous (for gradient)
hs_score = 1.0 - abs(HS - 0.5) * 2  # Peak at 0.5
continuous_score = (
    hs_score * 0.25 +
    min(MSR / 0.50, 1.0) * 0.25 +
    min(TAE / 0.30, 1.0) * 0.25 +
    min(ED / 0.30, 1.0) * 0.25
)

fitness = 0.70 * binary_score + 0.30 * continuous_score
```

## Evolution Algorithm

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `population_size` | 20 | Individuals per generation |
| `elite_size` | 2 | Preserved best individuals |
| `mutation_rate` | 0.3 | Probability of mutating each param |
| `mutation_strength` | 0.15 | Gaussian mutation σ (as % of range) |
| `crossover_rate` | 0.7 | Probability of crossover per param |
| `tournament_size` | 3 | Tournament selection size |

### Operators

1. **Initialization**: Variations around baseline (±25%)
2. **Selection**: Tournament selection (k=3)
3. **Crossover**: Arithmetic crossover (α random per individual)
4. **Mutation**: Gaussian mutation with range clamping
5. **Elitism**: Top 2 individuals preserved

### Termination

- Max generations reached (default: 50)
- Target criteria achieved (8/8)
- No improvement for N generations (optional)

## Implementation Components

### 1. EvolvableConfig (config_space.py)

```python
@dataclass
class EvolvableConfig:
    # Group A: Damage
    damage_multiplier: float = 3.9
    base_degrad_rate: float = 0.18
    # ... (30 parameters)

    @classmethod
    def from_dict(cls, d: Dict) -> 'EvolvableConfig':
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def get_module_effects(self) -> Dict[str, float]:
        return {
            'pattern_detector': self.effect_pattern_detector,
            # ...
        }
```

### 2. FitnessEvaluator (fitness_evaluator.py)

```python
def evaluate_config(config: Dict, n_runs: int = 5) -> FitnessResult:
    # 1. Validate config ranges
    # 2. Run IPUESA simulation
    # 3. Calculate self-evidence criteria
    # 4. Compute composite fitness
    return FitnessResult(
        fitness_score=fitness,
        criteria_passed=passed,
        metrics=metrics
    )
```

### 3. IPUESA Evolvable (ipuesa_evolvable.py)

Parameterized versions of SYNTH-v2 functions:
- `gradual_damage_evolvable(agent, damage, config)`
- `gradual_recovery_evolvable(agent, cluster, config)`
- `spread_modules_evolvable(agents, cluster_id, config)`
- `run_ipuesa_with_config(config, n_agents, n_steps, n_runs)`

### 4. Orchestrator (exp_evolve_ipuesa.py)

```python
class IPUESAEvolver:
    def initialize_population(self) -> List[Individual]
    def evaluate_population(self, population, gen) -> List[Individual]
    def evolve_generation(self, population, gen) -> List[Individual]
    def run(self, generations, target_criteria) -> EvolutionState
```

Features:
- Checkpoint/resume support
- CLI with all hyperparameters
- Progress logging
- Early stopping

## Usage

### Quick Validation (5 generations)

```bash
python experiments/evolution/exp_evolve_ipuesa.py \
    --generations 5 \
    --population 10 \
    --eval-runs 3 \
    --eval-steps 50
```

### Full Evolution Run

```bash
python experiments/evolution/exp_evolve_ipuesa.py \
    --generations 50 \
    --population 20 \
    --eval-runs 5 \
    --eval-steps 100 \
    --checkpoint-interval 5
```

### Resume from Checkpoint

```bash
python experiments/evolution/exp_evolve_ipuesa.py \
    --resume evolved_configs/checkpoint_gen25.json \
    --generations 50
```

### Use Evolved Config

```python
from zeta_life.evolution.ipuesa_evolvable import run_ipuesa_with_config
import json

config = json.load(open('evolved_configs/evolved_config_20260111.json'))
results = run_ipuesa_with_config(config, n_runs=10)
print(f'Results: {results}')
```

## Success Metrics

| Metric | Baseline (SYNTH-v2) | Target |
|--------|---------------------|--------|
| Criteria passed | 8/8 | 8/8 (maintain) |
| HS | 0.396 | 0.45-0.55 (more centered) |
| MSR | 0.501 | > 0.50 |
| TAE | 0.215 | > 0.25 |
| ED | 0.360 | > 0.35 |
| Fitness score | ~0.85 | > 0.90 |

## Completion Checklist

- [ ] Create `evolution/` module
- [ ] Implement `EvolvableConfig`
- [ ] Implement `fitness_evaluator.py`
- [ ] Adapt IPUESA to `ipuesa_evolvable.py`
- [ ] Create orchestrator `exp_evolve_ipuesa.py`
- [ ] Unit tests pass
- [ ] Validation run (5 gen) executes without errors
- [ ] Full run finds config with 8/8 criteria
- [ ] Document evolved config in CLAUDE.md
- [ ] Compare baseline vs evolved results

## Future Phases

| Phase | What Evolves | Fitness | Complexity |
|-------|--------------|---------|------------|
| **1** (this) | 30 IPUESA hyperparameters | criteria/8 | Low |
| **2** | + MicroModule.EFFECTS (8 values) | + emergent_differentiation | Medium |
| **3** | + Zeta kernel (σ, M, R) | + cellular_survival | Medium |
| **4** | Multi-objective co-evolution | Pareto front | High |

## References

- [OpenAlpha_Evolve](https://github.com/shyamsaktawat/OpenAlpha_Evolve)
- [IPUESA-SYNTH-v2 Results](../results/ipuesa_synth_v2_results.json)
- [Zeta Life CLAUDE.md](../../CLAUDE.md)
