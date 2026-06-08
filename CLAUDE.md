# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Status note (2026-06-08):** This file was rewritten to match the repository's
> actual current state. The project's center of gravity has moved to the
> **active-inference Conscious Kernel** (`src/zeta_life/kernel/`). Several earlier
> subsystems (psyche, hierarchical/IPUESA integration, evolution, organism) are
> **legacy** — frozen, with no current experiments — and are slated for archival.
> See "Core vs Legacy" below.

## What this project actually is

An **active-inference "Conscious Kernel"** for AI: a single adaptive unit that
runs a perception→prediction→action→reflection→dream loop with a learned world
model, a recursive self-model, precision-weighted prediction errors,
complementary (fast/slow) memory, expected-free-energy action selection, and
persistent identity — plus a Darwinian multi-kernel organism built on top.

The project grew out of earlier work integrating the **Riemann zeta zeros** with
artificial-life systems. That heritage survives in the name and in the
`K_σ(t)` kernel, but **zeta is no longer the thesis**: see "Where zeta actually
matters" below. The real, current research program is emergent coherent
integration via active inference.

**Theoretical foundation (zeta kernel):**
`K_σ(t) = 2 · Σ exp(-σ|γ|) · cos(γt)`, where γ are the imaginary parts of the
non-trivial zeta zeros (14.134725, 21.022040, 25.010858, …). Used in the dream
consolidation rhythm and in the cellular automata; optional in the kernel.

## Repository structure (verified)

```
zeta-life/
├── src/zeta_life/
│   ├── kernel/          # CORE — active-inference Conscious Kernel (14 files)
│   ├── integration/     # formal_equations.py (CORE) + hierarchical/IPUESA stack (LEGACY)
│   ├── datasets/        # SUPPORTING — real/synthetic signal loaders for Psi validation
│   ├── core/            # zeta_constants/vertex/tetrahedral (SUPPORTING) + zeta_memory/rnn/resonance (LEGACY)
│   ├── utils/           # SUPPORTING — statistics helpers
│   ├── organism/        # LEGACY — Fi-Mi swarm artificial life
│   ├── psyche/          # LEGACY — Jungian/archetype consciousness (superseded by kernel)
│   └── evolution/       # LEGACY — GA hyperparameter optimizer for IPUESA
├── experiments/
│   ├── kernel/          # 11 experiments (the live research)
│   └── datasets/        # 2 experiments (Psi on real data, phase transitions)
├── tests/               # 40 test files (~800 tests)
├── results/             # experiment outputs (PNG + run .txt)
├── docs/                # reports, papers, plans, theory
├── models/              # trained weights (.pt)
└── demos/               # interactive demonstrations
```

## Commands

```bash
# === INSTALL ===
pip install -e .                 # makes `zeta_life` importable
pip install -e ".[full]"         # with extras
# (mpmath optional, for exact zeta zeros; otherwise hardcoded values are used)

# === TESTS ===
# Tests import `zeta_life...`; either install (above) or set PYTHONPATH:
PYTHONPATH=src python -m pytest tests/ -q          # full suite
PYTHONPATH=src python -m pytest tests/test_conscious_kernel.py -q   # single file

# === EXPERIMENTS (self-pathing; run directly) ===
# Kernel (the live research):
PYTHONPATH=src python experiments/kernel/exp_conscious_kernel_validation.py
PYTHONPATH=src python experiments/kernel/exp_agency.py
PYTHONPATH=src python experiments/kernel/exp_zeta_vs_baselines.py    # zeta vs fourier/random/learned/rnn
PYTHONPATH=src python experiments/kernel/exp_spacing_statistics.py --kernel  # GUE vs Poisson vs lattice
PYTHONPATH=src python experiments/kernel/exp_grounding.py
PYTHONPATH=src python experiments/kernel/exp_organism_vs_individual.py
# Datasets:
PYTHONPATH=src python experiments/datasets/exp_real_data_psi.py
```

### Dependencies
`numpy`, `torch`, `matplotlib`, `scipy` (required). `mpmath` optional.

## Architecture — the Conscious Kernel (`src/zeta_life/kernel/`)

The active-inference cycle, one `ConsciousKernel.step(stimulus)` per tick:

```
PERCEIVE → PREDICT → COMPARE → UPDATE → MEMORIZE → ACT → REFLECT → DREAM
```

| File | Role |
|------|------|
| `conscious_kernel.py` | Orchestrator; the step loop, Psi computation, EFE agency |
| `world_model.py` | Learned latent dynamics (encoder + GRUCell transition + predictor); `imagine()` for counterfactual rollouts |
| `self_model.py` | Recursive self-model / identity embedding (Strange-Loop-flavored EMA + trained self-prediction) |
| `prediction_error.py` | Multi-channel precision-weighted errors; **precision learning** toward inverse error variance |
| `complementary_memory.py` | `FastMemory` (episodic deque, surprise-gated) + `SlowMemory` (slow-lr semantic net) — CLS |
| `dream_engine.py` | Zeta-rhythm sleep consolidation (fast→slow transfer, identity replay) |
| `temporal_features.py` | `OscillatorBank` — optional time code fed to the world model (see below) |
| `global_workspace.py`, `energy_pool.py`, `spawn_controller.py`, `organism_state.py`, `conscious_organism.py` | Darwinian multi-kernel organism (winner-take-all GW, energy, spawn/merge/death) |
| `persistence.py` | Save/load identity across sessions |

**Consciousness index Ψ** (in `integration/formal_equations.py`, imported by the
kernel): `Ψ = B³ + Φ` (cubic) or a bounded **Hill** variant (default), with a
critical threshold `Φ_c = F_i/(α − C)`. Note: Ψ is a bespoke, hand-tuned
heuristic (not IIT/FEP); treat it as a monotone integration signal, not a proven
consciousness measure.

**`OscillatorBank` temporal bases** (`temporal_features.py`):
```python
OscillatorBank.fourier(M)      # equispaced lattice — RECOMMENDED fixed basis
OscillatorBank.log_spaced(M)   # multi-scale (Transformer-style)
OscillatorBank.learned(M)      # trainable frequencies (adaptive)
OscillatorBank.zeta(M)         # the zeta zeros (kept for the comparison studies)
# spacing-statistic banks for the decisive test:
OscillatorBank.by_spacing("gue"|"poisson"|"uniform"|"zeta", M)
```
Default for `ConsciousKernel` is `temporal_features=None` (byte-identical to the
pre-temporal kernel).

## Where zeta actually matters (tested, honest)

This session's experiments (and the project's own prior results) settled it:

- **Cellular automata (spatial):** zeta genuinely wins (+134% survival vs Moore,
  beats UNIFORM, p<0.001). This is the one place the specific spectrum earns its
  keep.
- **Kernel / temporal prediction:** zeta's apparent edge is **basis-matching**,
  not specialness. A `fourier` (equispaced) lattice matches or beats zeta even on
  a zeta-structured signal (`results/zeta_vs_baselines_run.txt`).
- **Spacing statistics:** zeta's GUE level repulsion is real but **≤ a rigid
  lattice** on covering radius and conditioning, and **functionally flat** inside
  the kernel (`results/spacing_statistics_run.txt`).
- **Consciousness/psyche (legacy):** ZETA == UNIFORM (p=1.0). Structure matters,
  the specific zeta frequencies do not.

**Recommendation in code:** fixed basis → `fourier`/`log_spaced`; adaptive →
`learned`. Zeta is an optional, documented-and-falsified design choice.

## Core vs Legacy

**Core / keep:** `kernel/`, `integration/formal_equations.py`, `datasets/`,
`core/{zeta_constants,vertex,tetrahedral_space}.py`, `utils/`.

**Legacy (frozen, no current experiments — slated for archival to a `legacy/` branch):**
- `psyche/` — Jungian/archetype consciousness (the original formalism; superseded by the kernel)
- `integration/` hierarchical + IPUESA resilience stack (a parallel consciousness formalism the live kernel does not use)
- `evolution/` — GA optimizer that only tuned IPUESA hyperparameters
- `organism/` — Fi-Mi swarm artificial life (tangent to consciousness)
- `core/{zeta_memory,zeta_rnn,zeta_resonance}.py` — effectively unused (~0 imports)

There are **four competing "consciousness" formalisms** in the tree; the
canonical one is **Ψ** (`formal_equations.py`, used by the live kernel). The
psyche `ConsciousnessIndex`, the hierarchical `phi_global`, and
`OrganismState.integration_index` are redundant/superseded.

## Key parameters

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `M` | 15–40 | Number of oscillators / zeta zeros |
| `sigma` | 0.0–0.1 | Abel decay (0 = flat, all M live; 0.1 = ~4 effective) |
| `latent_dim` | 32 | World model latent dim |
| `obs_dim` | 4 | Observation/action dim |
| `reflect_interval` | 5 | Self-reflection cadence |
| `dream_interval` | 50 | Dream consolidation cadence |
| `action_mode` | reactive / efe | Reactive softmax vs expected-free-energy planning |

## Documentation

- `docs/AUDIT_FIXES_2026.md` — 11 audited kernel implementation fixes (with before/after metrics)
- `docs/AGENCY_2026.md` — active-inference agency investigation (honest negative results)
- `docs/YVYRA_BRIDGE.md` — contract for feeding a live LLM agent's experience into the kernel (the intended application)
- `docs/theory/EXPERIMENTO_ZETA_VS_BASELINE.md` — zeta vs uniform/none/random (zeta == uniform)
- `docs/papers/` — the original "zeta unification" paper (predates the kernel; its zeta thesis is now partly falsified by the project's own evidence)
- `docs/REPORTE_ZETA_ORGANISM.md`, `docs/ZETA_PSYCHE.md` — legacy subsystem reports
