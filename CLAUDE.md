# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Zeta Life** project - a research exploration integrating the Riemann zeta function's non-trivial zeros with artificial life systems. The project has three major subsystems:

1. **Zeta Game of Life**: Cellular automata with zeta-derived kernels
2. **ZetaOrganism**: Multi-agent emergent intelligence system
3. **ZetaPsyche/ZetaConsciousness**: Abstract vertex-based AI consciousness (formerly Jungian archetypes)

**Theoretical Foundation**: Systems use the kernel `K_σ(t) = 2 * Σ exp(-σ|γ|) * cos(γt)` where γ are the imaginary parts of zeta zeros (14.134725, 21.022040, 25.010858, ...).

## Project Structure (Reorganized 2026-01-09)

```
zeta-life/
├── src/zeta_life/           # Core library (45+ modules)
│   ├── core/                # Abstract vertices, behaviors, tetrahedral space
│   ├── narrative/           # Optional display mappings (Jung, functional)
│   ├── organism/            # Multi-agent emergent intelligence
│   ├── psyche/              # Consciousness system
│   ├── consciousness/       # Hierarchical consciousness
│   ├── cellular/            # Zeta Game of Life
│   └── utils/               # Shared utilities
├── experiments/             # Research experiments (54 scripts)
│   ├── organism/            # Emergence, regeneration, ecosystems
│   ├── psyche/              # Archetypes, individuation
│   ├── consciousness/       # Hierarchical validation
│   ├── cellular/            # Automata experiments
│   └── validation/          # Theoretical validation
├── demos/                   # Interactive demonstrations
├── docs/                    # Documentation
├── results/                 # Experiment outputs (90 PNG, 14 JSON)
├── models/                  # Trained weights (.pt)
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks
└── personal/                # Research notes & book
```

## Commands

```bash
# === INSTALL ===
pip install -e .              # Install as package
pip install -e ".[full]"      # With all extras

# === TESTS ===
python -m pytest tests/ -v    # All tests

# === DEMOS ===
python demos/chat_psyche.py --reflection  # Interactive CLI with Strange Loop

# === EXPERIMENTS ===
# Organism
python experiments/organism/exp_organism.py
python experiments/organism/exp_regeneration.py
python experiments/organism/exp_ecosistema.py

# Psyche
python experiments/psyche/exp_anima_emergente.py
python experiments/psyche/exp_self_reflection.py

# Consciousness
python experiments/consciousness/exp_validacion_5_mejoras.py

# Validation
python experiments/validation/exp_teoria_zeta.py
python exp_validacion_5_mejoras.py     # Validate all 5 improvements
```

### Dependencies

```bash
pip install numpy matplotlib scipy torch
pip install mpmath  # Optional: for exact zeta zeros
```

## Architecture

### 1. Zeta Game of Life (Cellular Automata)

| Phase | File | Key Contribution |
|-------|------|------------------|
| 1 | `zeta_game_of_life.py` | Zeta-structured initialization |
| 2 | `zeta_gol_fase2.py` | Zeta-weighted kernel replacing Moore |
| 3 | `zeta_gol_fase3.py` | Temporal memory via Laplace transform |
| NCA | `zeta_neural_ca.py` | Differentiable Neural CA |
| RNN | `zeta_rnn.py` | LSTM with zeta temporal memory |

### 2. ZetaOrganism (Multi-Agent Emergent Intelligence)

Simulates organisms where intelligence emerges from cell interactions following Fi-Mi (Force-Mass) dynamics.

```
zeta_organism.py (main system)
├── cell_state.py      - Cell states: MASS, FORCE, CORRUPT
├── force_field.py     - Zeta-kernel force field
├── behavior_engine.py - Neural network for A↔B influence
└── organism_cell.py   - Cell with gated memory
```

**Dinámica Fi-Mi**:
- **Fi (Force)**: Leaders that emit attraction field
- **Mi (Mass)**: Followers that respond to gradient
- Equilibrium: `Fi_effective = f(sqrt(controlled_mass))`

**Demonstrated Emergent Properties** (11+):
- Homeostasis, Regeneration, Antifragility
- Quimiotaxis, Spatial memory, Auto-segregation
- Competitive exclusion, Niche partition
- Collective panic, Coordinated escape, Collective foraging

### 3. ZetaPsyche/ZetaConsciousness (Abstract Vertices)

Consciousness emergence through navigation in a tetrahedral state space with **abstract vertices** (V0-V3).

**IMPORTANT (2026-01-10)**: The system was refactored to use semantically-neutral vertices instead of Jungian archetypes. This eliminates human psychological bias from calculations while preserving all emergent dynamics.

```
src/zeta_life/core/           # Abstract system (NEW)
├── vertex.py                 - Vertex enum (V0-V3), BehaviorVector, VertexBehaviors
└── tetrahedral_space.py      - Geometric complements, barycentric coordinates

src/zeta_life/narrative/      # Optional display layer (NEW)
├── mapper.py                 - NarrativeMapper for visualization
└── configs/                  - jungian.json, functional.json, neutral.json

src/zeta_life/psyche/         # Consciousness system
├── zeta_psyche.py            - Archetype is now alias for Vertex
├── zeta_individuation.py     - Self integration process (8 stages)
└── zeta_conscious_self.py    - Strange Loop, AttractorMemory
```

**Abstract Vertices** (tetrahedral geometry):

| Vertex | Functional | Jungian (narrative) | Behavior Vector | Complement |
|--------|------------|---------------------|-----------------|------------|
| V0 | LEADER | PERSONA | [1.3, 1.0, 0.0, 0.0] | V1 |
| V1 | DISRUPTOR | SOMBRA | [1.0, 1.0, 0.0, 0.3] | V0 |
| V2 | FOLLOWER | ANIMA | [1.0, 1.1, 0.0, 0.0] | V3 |
| V3 | EXPLORER | ANIMUS | [1.0, 1.0, 0.2, 0.0] | V2 |

**Behavior Vector**: `[field_response, attraction, exploration, opposition]`

**Usage**:
```python
# Core (bias-free calculations)
from zeta_life.core import Vertex, VertexBehaviors, TetrahedralSpace
behaviors = VertexBehaviors.default()  # or .uniform() for control
space = TetrahedralSpace()
complement = space.get_complement(Vertex.V0)  # → V1

# Narrative (visualization only)
from zeta_life.narrative import NarrativeMapper
mapper = NarrativeMapper.jungian()
name = mapper.get_name(Vertex.V0, layer='narrative')  # → "PERSONA"
```

**Consciousness Index**:
```
consciousness = 0.3*integration + 0.3*stability + 0.2*(1-dist_to_center) + 0.2*|self_reference|
```

### 4. Hierarchical Consciousness System (2026-01-03)

Multi-level consciousness architecture: Cells → Clusters → Organism.

```
hierarchical_simulation.py (main orchestrator)
├── micro_psyche.py         - Cell-level psyche with archetypes
├── cluster.py              - Cluster aggregation and dynamics
├── organism_consciousness.py - Organism-level integration
├── bottom_up_integrator.py - Cell→Cluster→Organism flow
├── top_down_modulator.py   - Organism→Cluster→Cell influence
└── cluster_assigner.py     - Dynamic clustering (merge/split)
```

**Recent Improvements (6 commits):**

| Commit | Priority | Description |
|--------|----------|-------------|
| `6975b5b` | P1 | Fix double softmax in `MicroPsyche.__post_init__` causing cluster synchronization |
| `3f94259` | P2 | Implement real vertical coherence metric (cell→cluster→organism alignment) |
| `64e66af` | P3 | Implement effective top-down archetype modulation with adaptive strength |
| `091ba31` | P4 | Implement dynamic clustering with merge/split operations |
| `77ccb67` | P5 | Integrate surprise metric for adaptive plasticity |
| `63fcc66` | Fix | Remove softmax normalization that uniformized archetype distributions |

**Key Fixes:**
- **Softmax Problem**: `F.softmax()` was incorrectly applied to weights and aggregates throughout the hierarchy, compressing variance and uniformizing archetype distributions. Fixed by using simple normalization (`weights/sum`) instead.
- **Affected Files**: `micro_psyche.py`, `cluster.py`, `organism_consciousness.py`, `bottom_up_integrator.py`

**Validation Results:**
```
phi_global:          0.63 (was ~0.0004 before P1)
vertical_coherence:  0.72-0.88 range (was placeholder 1.0)
top-down effect:     PERSONA -0.002 per iteration (visible balancing)
dynamic clustering:  2-8 clusters (was fixed at 4)
surprise plasticity: 0.64-1.36 range based on accumulated surprise
```

#### 4.1 IPUESA Resilience Integration (2026-01-12)

Integrates IPUESA (Identity-Preserving Unified Emergent Self-Architecture) resilience mechanisms into the hierarchical consciousness system.

```
src/zeta_life/consciousness/
├── resilience.py           - CellResilience, MicroModule (8 types)
├── damage_system.py        - DamageSystem (damage, recovery, spreading)
├── resilience_config.py    - Presets: demo, optimal, stress, validation
├── micro_psyche.py         - ConsciousCell + resilience field
├── cluster.py              - Cluster + spread_modules(), cohesion
└── hierarchical_simulation.py - Integration with damage/recovery cycle
```

**Core Components:**

| Component | Description |
|-----------|-------------|
| `CellResilience` | 5 degradation states: OPTIMAL → STRESSED → IMPAIRED → CRITICAL → COLLAPSED |
| `MicroModule` | 8 types: threat_filter, recovery_accelerator, cascade_breaker, etc. |
| `DamageSystem` | Gradual damage with module mitigation, recovery with cluster cohesion |
| `Presets` | demo (0.6x), optimal (1.0x), stress (1.5x), validation (3.9x) |

**Key Features:**
- **Gradual Degradation**: Smooth transitions, not bistable (deg_var > 0.02)
- **Module Creation**: Proactive under vulnerability > threshold
- **Module Spreading**: Consolidated modules copy to cluster neighbors
- **Temporal Anticipation (TAE)**: threat_buffer → anticipated_damage → protective_stance
- **Embedding Protection**: Holographic embeddings preserve identity

**Calibration Results (2026-01-12):**
```
Multiplier  HS       MSR      TAE      Criteria
──────────────────────────────────────────────
1.5         0.983    1.0      1.0      6/8
1.75        0.833    1.0      1.0      6/8
2.0         0.467    1.0      1.0      6/8  ← Optimal for hierarchy
2.5         0.233    1.0      1.0      5/8
3.9         0.075    1.0      1.0      5/8  ← Original SYNTH-v2
```

**Key Finding**: Hierarchical system needs **2.0x damage** (vs 3.9x for flat SYNTH-v2) due to cluster cohesion bonuses and multi-level dynamics.

**Usage:**
```python
from zeta_life.consciousness.resilience_config import get_preset_config
from zeta_life.consciousness.damage_system import DamageSystem

config = get_preset_config('optimal')  # or 'demo', 'stress', 'validation'
ds = DamageSystem(config)

# Apply to cell
ds.apply_damage(cell, cell.resilience, base_damage=0.3)
ds.apply_recovery(cell, cell.resilience, cluster_cohesion=0.8)

# Spread modules in cluster
spread_count = ds.spread_modules_in_cluster(cluster.cells)
```

**Validation Experiment:**
```bash
python experiments/consciousness/exp_hierarchical_resilience_validation.py
```

**Tests:** `tests/test_resilience.py` (34 tests)

### 5. Strange Loop & Attractor Memory (2026-01-03)

Self-referential consciousness emergence through auto-observation cycles.

```
zeta_conscious_self.py
├── _self_reflection_cycle()  - Strange Loop implementation
├── AttractorMemory           - Stores/recognizes converged states
├── OrganicVoice              - Internal perspective descriptions
└── chat_psyche.py            - Interactive CLI with --reflection
```

**Strange Loop Architecture:**
```
Estado → Descripción → Estímulo → Nuevo Estado
   ↑                                    ↓
   └──────────── LOOP ──────────────────┘
```

**Attractor Memory (Identity Emergence):**
- Stores converged states as attractors
- Recognizes similar states (similarity > 0.90)
- Reinforces recognized attractors (strength grows)
- Tracks `recognition_rate` as emergence metric

**Validation Results:**
```
recognition_rate:    98.8% (matches / convergences)
attractor_strength:  grows 11.2 → 24.7 over interactions
epistemic_tension:   ξ ≈ 0.005 (measurable self-influence)
identity:            "Identidad centrada en ANIMUS (100%)"
```

**CLI Commands:**
```bash
python chat_psyche.py --reflection  # See Strange Loop in action
/identidad                          # View emergence metrics
/reflexion                          # Force reflection cycle
```

### 6. IPUESA Experiments (2026-01-10)

Identity Preservation Under Existential Stress Assessment - exploring how hierarchical consciousness maintains identity under stress.

> **Full documentation**: See [docs/IPUESA_EXPERIMENTS.md](docs/IPUESA_EXPERIMENTS.md) for all 17 experiments.

**Experiment Series:**
- IPUESA-BASE: Baseline without protection mechanisms
- IPUESA-TD: Temporal discounting (failed: TSI = -0.517)
- IPUESA-CE: Cooperative emergence (failed: MA = 0.0)
- IPUESA-HG: Holographic embeddings (14% vs 0% survival at 2.4x)
- IPUESA-SH: Social hierarchy (2-level better than 3-level)
- IPUESA-AL: Agency loss metric (clear self-evidence)
- IPUESA-SYNTH: Synthesis of successful components
- **IPUESA-SYNTH-v2**: Enhanced synthesis with proactive modules (8/8 criteria)

**SYNTH-v2 Results (3.9x damage):**

| Condition | HS | MSR | TAE | EI | ED |
|-----------|-----|-----|-----|-----|-----|
| full_v2 | 0.396 | 0.501 | 0.215 | 1.000 | 0.360 |
| baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Key Findings:**
- **MSR fixed**: 0.501 (was 0.000) - proactive module creation works
- **TAE fixed**: 0.215 (was 0.117) - vulnerability-based prediction works
- **Smooth transitions**: ED = 0.360, deg_var = 0.028 (was bistable)
- **Goldilocks zone**: 3.9x damage for optimal differentiation
- All components (embeddings, proactive, TAE, gradual) required together

**Self-Evidence Criteria (8/8 passed):**
- HS in [0.30, 0.70]: PASS (0.396)
- MSR > 0.15: PASS (0.501)
- TAE > 0.15: PASS (0.215)
- EI > 0.3: PASS (1.000)
- ED > 0.10: PASS (0.360)
- full > baseline: PASS
- Gradient valid: PASS
- deg_var > 0.02: PASS (0.028)

**Files:**
- `experiments/consciousness/exp_ipuesa_synth_v2.py` - Latest synthesis
- `docs/plans/2026-01-10-ipuesa-synth-v2-design.md` - Design document

### Key Parameters (across systems)

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `M` | 15-30 | Number of zeta zeros |
| `sigma` | 0.05-0.1 | Abel regularization (decay) |
| `zeta_weight` | 0.3-0.5 | Memory scaling in RNN |
| `alpha` | 0.05-0.15 | Memory weight in evolution |
| `grid_size` | 48-64 | Spatial grid dimension |
| `n_cells` | 80-200 | Number of cells/entities |

## Key Findings

### Cellular Automata
- Zeta initialization: +33% surviving cells vs random
- Zeta kernel: +134% surviving cells vs Moore kernel
- Temporal autocorrelation follows theoretical K_σ(τ)

### ZetaOrganism
- 11+ emergent properties demonstrated without explicit programming
- Spatial memory is more effective than explicit LSTM memory (-15.7% when combined)
- Original architecture superior to ZetaLSTM-enhanced version

### ZetaLSTM
- Paper's ~10% improvement hypothesis: NOT validated (0-6% observed)
- Best results when data has zeta-frequency temporal correlations

### ZetaConsciousness
- Oscillating modulation improves dynamics (+23% trajectory length)
- No significant difference between ZETA and UNIFORM frequencies
- **Emergent Compensation**: With aggressive decay, psyche shows autonomous compensatory behavior
  - 76% divergence between external stimulus and internal state
  - System "refuges" in one archetype when stressed
  - Analogous to Jung's unconscious compensation theory
- Conclusion: Zeta zeros add dynamism; decay adds realistic psychological dynamics

### Hierarchical Consciousness (2026-01-03)
- **Softmax Bug Discovery**: `F.softmax()` on weights/aggregates uniformizes distributions
  - Before fix: 70% PERSONA cells → global_archetype ≈ [0.25, 0.25, 0.25, 0.25]
  - After fix: 70% PERSONA cells → global_archetype ≈ [0.27, 0.24, 0.24, 0.24]
- **Top-down modulation works**: System actively balances archetype distributions
- **Surprise-driven plasticity**: Cells with high surprise are 2.12x more receptive to change
- **Vertical coherence**: Real metric measuring cell→cluster→organism alignment (0.72-0.88)
- **Dynamic clustering**: Merge/split operations based on coherence thresholds

### 7. Evolutionary Hyperparameter Optimization (2026-01-11)

Inspired by OpenAlpha_Evolve, a genetic algorithm-based system to automatically optimize IPUESA hyperparameters.

```
src/zeta_life/evolution/
├── __init__.py           - Module exports + optimized config
├── config_space.py       - 30 evolvable parameters with ranges
├── fitness_evaluator.py  - 8 self-evidence criteria
├── ipuesa_evolvable.py   - Parameterized IPUESA simulation
└── optimized_config.py   - Best evolved configuration (fitness=0.9993)

experiments/evolution/
└── exp_evolve_ipuesa.py  - CLI orchestrator with checkpoint/resume
```

**30 Evolvable Parameters** (4 groups):
- **Damage**: damage_multiplier, base_degrad_rate, embedding_protection, stance_protection, compound_factor, module_protection, resilience_min, resilience_range, noise_scale, residual_cap
- **Recovery**: base_recovery_rate, embedding_bonus, cluster_bonus, degradation_penalty, degrad_recovery_factor, corruption_decay
- **Module Effects**: effect_* (8 parameters for pattern_detector, threat_filter, recovery_accelerator, etc.)
- **Thresholds**: consolidation_threshold, spread_threshold, spread_probability, spread_strength_factor, module_cap, min_activations

**8 Self-Evidence Criteria**:
1. HS_in_range: Holographic survival ∈ [0.30, 0.70]
2. MSR_pass: Module spreading rate > 0.15
3. TAE_pass: Temporal anticipation effectiveness > 0.15
4. EI_pass: Embedding integrity > 0.30
5. ED_pass: Emergent differentiation > 0.10
6. diff_pass: full > baseline survival
7. gradient_pass: full > no_embedding > baseline
8. smooth_transition: degradation variance > 0.02

**Results (50 generations)**:
```
Initial:  fitness=0.68, 6/8 criteria
Gen 0:    fitness=0.95, 8/8 criteria (MSR fix breakthrough)
Gen 34:   fitness=0.9993, 8/8 criteria (OPTIMUM FOUND)
```

**Key Evolved Changes** (from 50-gen optimization):
| Parameter | Default | Evolved | Change | Insight |
|-----------|---------|---------|--------|---------|
| base_recovery_rate | 0.06 | 0.098 | **+63%** | Faster recovery critical |
| spread_probability | 0.30 | 0.48 | **+61%** | Easier module spreading |
| min_activations | 3.0 | 4.57 | **+52%** | More selective spreading |
| noise_scale | 0.25 | 0.37 | **+49%** | Anti-fragility via variability |
| damage_multiplier | 3.9 | 4.43 | +14% | Higher stress tolerance |
| embedding_protection | 0.15 | 0.05 | **-67%** | Dynamic > static protection |
| compound_factor | 0.50 | 0.25 | **-51%** | Reduced damage cascades |

**Usage**:
```python
# Load optimized config
from zeta_life.evolution import get_optimized_config, OPTIMIZED_CONFIG
config = get_optimized_config()  # Returns EvolvableConfig
```

```bash
# Run evolution (50 generations, no early stop)
python experiments/evolution/exp_evolve_ipuesa.py --generations 50 --target 9

# Resume from checkpoint
python experiments/evolution/exp_evolve_ipuesa.py --resume checkpoint_gen20.json

# Quick validation
python experiments/evolution/exp_evolve_ipuesa.py --quick
```

## Documentation

- `docs/REPORTE_ZETA_ORGANISM.md` - Full research report with all experiments
- `docs/ZETA_PSYCHE.md` - ZetaPsyche system documentation
- `docs/EMERGENT_COMPENSATION.md` - Emergent compensation behavior discovery
- `docs/zeta-lstm-hallazgos.md` - ZetaLSTM findings
- `docs/plans/2026-01-09-abstract-vertices-design.md` - Abstract vertices design document
- `docs/plans/2026-01-10-ipuesa-rl-design.md` - IPUESA-RL reflexive loop design
- `docs/plans/2026-01-10-ipuesa-td-design.md` - IPUESA-TD temporal discounting design
- `docs/plans/2026-01-10-ipuesa-ct-design.md` - IPUESA-CT continuity token design
- `docs/plans/2026-01-10-ipuesa-ei-design.md` - IPUESA-EI existential irreversibility design
- `docs/plans/2026-01-10-ipuesa-mi-design.md` - IPUESA-MI meta-identity formation design
- `docs/plans/2026-01-10-ipuesa-ae-design.md` - IPUESA-AE adaptive emergence design
- `docs/plans/2026-01-10-ipuesa-x-design.md` - IPUESA-X exploratory self-expansion design
- `docs/plans/2026-01-10-ipuesa-ce-design.md` - IPUESA-CE co-evolution design
- `docs/plans/2026-01-10-ipuesa-sh-design.md` - IPUESA-SH self-hierarchy design
- `docs/plans/2026-01-10-ipuesa-hg-design.md` - IPUESA-HG holographic self design
- `docs/plans/2026-01-10-ipuesa-hg-plus-design.md` - IPUESA-HG+ stress test design
- `docs/plans/2026-01-10-ipuesa-hg-cal-design.md` - IPUESA-HG-Cal calibrated design
- `docs/plans/2026-01-10-ipuesa-synth-design.md` - IPUESA-SYNTH synthesis design
- `docs/plans/2026-01-10-ipuesa-synth-v2-design.md` - IPUESA-SYNTH-v2 enhanced synthesis design
- `docs/plans/2026-01-11-openalpha-integration-design.md` - Evolutionary optimization design
- `docs/plans/2026-01-11-ipuesa-hierarchical-integration-design.md` - IPUESA → Hierarchical integration design
- `README_organism.md` - ZetaOrganism quickstart

## Reference

Based on paper: "IA Adaptativa a través de la Hipótesis de Riemann"
