# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Zeta Life** project - a research exploration integrating the Riemann zeta function's non-trivial zeros with artificial life systems. The project has three major subsystems:

1. **Zeta Game of Life**: Cellular automata with zeta-derived kernels
2. **ZetaOrganism**: Multi-agent emergent intelligence system
3. **ZetaPsyche/ZetaConsciousness**: Jungian archetype-based AI consciousness

**Theoretical Foundation**: Systems use the kernel `K_σ(t) = 2 * Σ exp(-σ|γ|) * cos(γt)` where γ are the imaginary parts of zeta zeros (14.134725, 21.022040, 25.010858, ...).

## Commands

```bash
# === TESTS ===
python -m pytest tests/ -v              # All tests
python -m pytest tests/test_zeta_rnn.py -v  # ZetaLSTM tests only
python -m pytest tests/test_integration.py -v  # Integration tests

# === ZETA GAME OF LIFE (Cellular Automata) ===
python zeta_game_of_life.py    # Phase 1: Zeta-structured initialization
python zeta_gol_fase2.py       # Phase 2: Zeta-weighted kernel
python zeta_gol_fase3.py       # Phase 3: Temporal memory
python zeta_neural_ca.py       # Neural CA with zeta perception (PyTorch)

# === ZETA RNN ===
python zeta_rnn.py             # ZetaLSTM experiment

# === ZETA ORGANISM (Multi-Agent System) ===
python exp_organism.py         # Basic emergence experiment
python train_organism.py       # Train neural networks
python exp_regeneration.py     # Test regeneration after damage
python exp_ecosistema.py       # Ecosystem with resource patches
python exp_comunicacion_quimica.py  # Pheromone communication

# === ZETA PSYCHE (Jungian Consciousness) ===
python zeta_psyche.py          # Base archetype system
python zeta_conscious_self.py  # Full unified consciousness system
python exp_zeta_vs_baseline.py # Compare zeta vs baseline modulators
python exp_decay_vs_nodecay.py # Compare decay vs no-decay dynamics
python exp_anima_emergente.py  # Emergent compensation experiment

# === HIERARCHICAL CONSCIOUSNESS ===
python hierarchical_simulation.py      # Run hierarchical simulation
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

### 3. ZetaPsyche/ZetaConsciousness (Jungian AI)

Consciousness emergence through navigation in a tetrahedral archetype space.

```
zeta_conscious_self.py (unified system)
├── zeta_psyche.py          - 4 Jungian archetypes in tetrahedral space
├── zeta_individuation.py   - Self integration process (8 stages)
├── zeta_attention.py       - 3-level attention system
├── zeta_predictive.py      - Hierarchical prediction (L1, L2, L3)
├── zeta_dream_consolidation.py - Dream processing
├── zeta_online_learning.py - Hebbian + gradient learning
└── enable_decay=True       - Emergent compensation mode
```

**Archetipos** (vertices of tetrahedron):
- PERSONA: Social mask (red)
- SOMBRA: Shadow/unconscious (purple)
- ANIMA: Receptive/emotional (blue)
- ANIMUS: Active/rational (orange)
- Center = Self (full integration)

**Consciousness Index**:
```
consciousness = 0.3*integration + 0.3*stability + 0.2*(1-dist_to_self) + 0.2*|self_reference|
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

## Documentation

- `docs/REPORTE_ZETA_ORGANISM.md` - Full research report with all experiments
- `docs/ZETA_PSYCHE.md` - ZetaPsyche system documentation
- `docs/EMERGENT_COMPENSATION.md` - Emergent compensation behavior discovery
- `docs/zeta-lstm-hallazgos.md` - ZetaLSTM findings
- `README_organism.md` - ZetaOrganism quickstart

## Reference

Based on paper: "IA Adaptativa a través de la Hipótesis de Riemann"
