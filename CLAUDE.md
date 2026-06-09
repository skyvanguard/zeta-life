# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Status note (2026-06-08):** This file was rewritten to match the repository's
> actual current state. The project's center of gravity has moved to the
> **active-inference Conscious Kernel** (`src/zeta_life/kernel/`). Several earlier
> subsystems (psyche, hierarchical/IPUESA integration, evolution, organism) were
> **archived on 2026-06-08** to the `legacy/pre-refocus-snapshot` branch and
> removed from the working tree. See "Core vs Archived" below.

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
│   ├── kernel/          # CORE — active-inference Conscious Kernel (19 files)
│   ├── bridge/          # Yvyra coupling — feed a live agent's experience to the kernel
│   ├── integration/     # formal_equations.py — the integration index Psi
│   ├── datasets/        # real/synthetic signal loaders for Psi validation
│   ├── core/            # zeta_constants, vertex, tetrahedral geometry
│   └── utils/           # statistics helpers
├── experiments/
│   ├── kernel/          # 22 experiments (the live research)
│   └── datasets/        # 1 experiment (Psi on real data)
├── tests/               # 28 test files (515 tests)
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
| `policy.py`, `replay.py`, `dynamics_ensemble.py` | Dreamer amortized actor/critic, transition replay, independent dynamics ensemble (curiosity) used by `action_mode="dreamer"` |
| `rssm.py`, `dreamerv3_agent.py` | **Reference** DreamerV2/V3-style RSSM agent (NOT the kernel) — bounds the kernel's CartPole limit: a recurrent state-space model trained on sequences + learned reward **solves CartPole** where the kernel's 1-step model plateaus (`exp_dreamerv3.py`, §3.10) |
| `rssm_kernel.py` | **Integration** — `RSSMConsciousKernel`: the kernel's faculties (identity, CLS memory, dream, **Ψ**) layered on the RSSM world model + controller; reaches CartPole's ceiling with Ψ live over the recurrent state (`exp_rssm_kernel.py`, §3.11) |

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

## Core vs Archived

**Core (this is the whole project now):** `kernel/`,
`integration/formal_equations.py`, `datasets/`,
`core/{zeta_constants,vertex,tetrahedral_space}.py`, `utils/`.

**Archived 2026-06-08** — removed from the working tree, preserved on the
`legacy/pre-refocus-snapshot` branch (retrieve with
`git show legacy/pre-refocus-snapshot:<path>`):
- `psyche/` — Jungian/archetype consciousness (original formalism; superseded by the kernel)
- `integration/` hierarchical + IPUESA resilience stack (a parallel consciousness formalism the kernel never used)
- `evolution/` — GA optimizer that only tuned IPUESA hyperparameters
- `organism/` — Fi-Mi swarm artificial life (tangent to consciousness)
- `core/{zeta_memory,zeta_rnn,zeta_resonance}.py` — effectively unused

The competing consciousness formalisms (psyche `ConsciousnessIndex`, hierarchical
`phi_global`) were archived with their packages. The canonical index is **Ψ**
(`formal_equations.py`, used by the live kernel); `OrganismState.integration_index`
(`kernel/organism_state.py`) remains as the organism-level aggregate.

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
| `efe_n_samples` | 0 / 48 | EFE: add N sampled CONTINUOUS candidate actions (0 = one-hots only). Continuous + `efe_obs_norm="l1"` lets the planner reach non-vertex targets (see `exp_control.py`) |
| `efe_horizon` | 1 | EFE planning horizon (sustained-action rollout). >1 found YAGNI in the 4-D env |
| `efe_cem_iters` | 0 | EFE: Cross-Entropy Method refinement (0 = random shooting). Capability for hard action landscapes; no reliable gain in the unimodal control task (`exp_cem.py`) |
| `wm_disagreement_heads` | 0 | World-model ensemble heads for an epistemic (disagreement) signal (0 = off). Under a *controlled* comparison it gives **no reliable** exploration gain in the 4-D regime (`exp_curiosity.py`; an earlier apparent ~2x was an RNG confound, now fixed). Head masking uses a dedicated RNG; kept as a capability |
| `efe_epistemic_mode` | entropy / disagreement | EFE epistemic term: coarse outcome-entropy proxy vs real world-model disagreement |
| `dynamics_ensemble` / `wm_disagreement_heads` | 0 / 0 | epistemic source: **independent** one-step dynamics models (Plan2Explore-faithful) vs shared-latent readout heads. With a commensurate `efe_epistemic_weight`, disagreement-curiosity reliably drives exploration (`exp_curiosity.py`) |
| `action_mode` | reactive / efe / **dreamer** | `dreamer` = amortized actor+critic trained in imagination (value gradients); O(1) action cost, matches/beats search on control (`exp_dreamer.py`) |
| `imag_horizon` / `imag_rollouts` | 5 / 8 | Dreamer imagination horizon and rollout batch (per-step behaviour learning) |
| `critic_tau` / `return_norm` / `actor_grad_clip` | 0.98 / True / 100 | Dreamer stabilizers: EMA target critic, return-scale normalization, gradient clipping |
| `replay_capacity` / `replay_wm` | 10000 / True | DreamerV3 transition replay: imagine behaviour from re-encoded replayed states + ground the world model on diverse transitions (improves CartPole curve/peak; doesn't reach the ceiling — late collapse persists) |
| `action_dim` / `dreamer_reward` | None / kl | decouple action from obs space; `neg_distance` reward for regulation to a raw goal state (e.g. CartPole) |

## Documentation

- `docs/AUDIT_FIXES_2026.md` — 11 audited kernel implementation fixes (with before/after metrics)
- `docs/AGENCY_2026.md` — active-inference agency investigation (honest negative results)
- `docs/YVYRA_BRIDGE.md` — contract for feeding a live agent's experience into the kernel; the zeta-life side is implemented in `src/zeta_life/bridge/` (demo: `experiments/kernel/exp_yvyra_bridge.py`)
- `docs/theory/EXPERIMENTO_ZETA_VS_BASELINE.md` — zeta vs uniform/none/random (zeta == uniform)
- `docs/papers/conscious-kernel-paper.md` — **current thesis**: the active-inference Conscious Kernel (architecture, honest results, the "what helped / what didn't" ledger)
- `docs/RELATED_WORK.md` — curated literature scan mapped to each kernel component (Dreamer, Plan2Explore, CLS, Butlin indicator properties, LLM+active-inference) with validate/inspire/SOTA-gap takeaways and a ranked "what to adopt" list
- `docs/INDICATOR_PROPERTIES.md` — honest, conservative audit of the kernel against Butlin et al. (2023) consciousness *indicator properties* (strong on PP/agency/embodiment, partial on GWT/recurrence/HOT, absent AST/HOT-4/GWT-4); the rigorous framework replacing Ψ-as-consciousness. Explicitly: indicators ≠ consciousness
- `docs/papers/zeta-life-framework-paper.md` — the original "zeta unification" paper (predates the kernel; its zeta thesis is partly falsified by the project's own evidence; superseded by the kernel paper)
- `docs/REPORTE_ZETA_ORGANISM.md`, `docs/ZETA_PSYCHE.md` — legacy subsystem reports
