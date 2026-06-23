# Zeta Life

**Emergent coherent integration in adaptive multi-agent systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-585%20passed-brightgreen.svg)](tests/)

---

## Overview

Zeta Life is a computational framework for studying **phase transitions in adaptive integration systems**. It models how independent processing units (kernels) can transition from fragmented computation to coherent, unified integration -- and how populations of such units exhibit emergent collective behavior.

The core equation predicts when integration emerges:

$$\Psi = B^3 + \Phi \qquad \text{(integration index, supercritical regime)}$$

$$\Phi_c = \frac{F_i}{\alpha - C} \qquad \text{(critical threshold for phase transition)}$$

$$B = \frac{\Phi - \Phi_c}{\Phi_c} \qquad \text{(binding factor)}$$

The cubic term $B^3$ creates a **sharp phase transition**: below the critical threshold $\Phi_c$, integration is zero. Above it, the system amplifies nonlinearly -- a mathematical signature of emergent coherence.

The current research program (**"The North"**, below) takes this further: making $\Psi$ a property of a real LLM's own activations and testing whether the model can *learn to introspect it* — adaptation, not scale.

---

## Architecture

### Adaptive Kernel (Active Inference)

Each `ConsciousKernel` implements a full Active Inference cycle:

```
PERCEIVE -> PREDICT -> COMPARE -> UPDATE -> MEMORIZE -> ACT -> REFLECT -> DREAM
```

| Component | Role |
|-----------|------|
| WorldModel | GRU-based predictive model of the environment |
| SelfModel | Recursive self-modeling (Strange Loop) |
| PredictionErrorEngine | Multi-channel precision-weighted error signals |
| PrecisionController | Learned precision hyper-model (attention) |
| FastMemory / SlowMemory | Complementary Learning Systems (episodic + semantic) |
| DreamEngine | Zeta-driven offline consolidation |
| PersistenceLayer | Identity save/load across sessions |

The integration index $\Psi$ is derived from internal signals:
- $\Phi$ (integrated information) from inverse free energy + memory depth
- $F_i$ (binding force) from learned precisions + self-reflection convergence
- $C$ (coherence cost) from recent prediction errors

```python
import torch
from zeta_life.kernel import ConsciousKernel

kernel = ConsciousKernel(obs_dim=4, latent_dim=32, alpha=1.0)
result = kernel.step(torch.randn(4))
print(result.psi)          # integration index [0, 1]
print(result.free_energy)  # prediction error magnitude
```

Or run the full demo (learning loop + identity save/load):

```bash
make quickstart            # = PYTHONPATH=src python demos/quickstart.py
```

### Multi-Kernel Organism (Darwinian Brain)

Multiple kernels compete for a shared `GlobalWorkspace` via proposal strength. Energy-managed spawning, merging, and death create a Darwinian selection process at the kernel level.

```python
import torch
from zeta_life.kernel import ConsciousOrganism

organism = ConsciousOrganism(obs_dim=4, initial_kernels=5)
result = organism.step(torch.randn(4))
print(result.psi)         # organism-level integration
print(result.population)  # surviving kernel count
```

### Hierarchical Integration (Cells -> Clusters -> Organism) — archived

> **Archived (2026-06-08).** A parallel Cells→Clusters→Organism consciousness
> formalism that the live kernel never used. Removed from the working tree in the
> refocus; preserved on the `legacy/pre-refocus-snapshot` branch.

### Zeta Function as Temporal Binding (optional — tested, not load-bearing)

The Riemann zeta zeros give an *optional* temporal basis:

$$K_\sigma(t) = 2 \sum_n \exp\!\bigl(-\sigma\,|\gamma_n|\bigr)\,\cos(\gamma_n\, t)$$

where $\gamma_n$ are the imaginary parts of zeta zeros (14.134, 21.022, 25.011, ...). They drive the DreamEngine's consolidation rhythm and are available as an optional `OscillatorBank` time code for the world model.

**Honest result (this project's own experiments).** The *specific* zeta spectrum is **not** load-bearing. In the kernel, an equispaced **Fourier lattice matches or beats zeta** at temporal prediction — even on a zeta-structured signal — and zeta's GUE level-repulsion is functionally flat (`results/zeta_vs_baselines_run.txt`, `results/spacing_statistics_run.txt`). In the consciousness dynamics, ZETA == UNIFORM (p=1.0). Zeta genuinely wins only in the spatial cellular automata. **Recommendation:** fixed temporal basis → `OscillatorBank.fourier`/`log_spaced`; adaptive → `learned`. Zeta is kept as a documented, tested design choice — not the thesis.

---

## The North: making Ψ internal (introspection research)

The kernel's $\Psi$ is an **external** index — a thermometer pointed *at* an agent.
The current research program asks the harder question: can $\Psi$ become a property
of an agent's **own activations**, and can the agent learn to **perceive** it? This
is the project's real thesis — *a model that adapts until it can observe itself,
rather than scaling to a giant*.

**Substrate.** A live LLM (Yvyra, Qwen3-8B) runs in `transformers`/8-bit. The
`bridge/` couples its real experience to the kernel; `introspection/` computes
candidate integration metrics over its hidden states and tests introspection with
the methods of Anthropic (Lindsey, concept injection) and Binder (privileged
access). Honest, control-driven results so far:

| Experiment | Result |
|------------|--------|
| **Phase A/B** (expose Ψ to the agent, sham control) | Inconclusive — Ψ saturates (~89% high); the design works, the signal didn't vary enough (`docs/PHASE_B_DESIGN.md`) |
| **Spontaneous introspection** (concept injection, untrained) | **Negative** — 0/30; a control killed the apparent positive (steering, not introspection). Replicates the scale limit Anthropic sees |
| **Trained P(IK) self-report** (LoRA) | **Honest negative** — a strong blind M2 (Claude, 0.82) and the model's own softmax confidence (0.81) both beat the self-report (0.76). It verbalizes confidence, no robust *privileged* access |
| **Trained injected-concept detection** (LoRA, constant prompt → non-textual by construction) | **Positive** — accuracy 1.000 (chance 0.091), 0 false positives. The model reads a non-textual injected state and names it. Caveat: fixed concept set may be a lookup; scaling test pending |

Each result was reported only after an adversarial control — the honesty discipline
that distinguishes "verbalizing confidence" from "privileged introspection." See
`docs/{ANTHROPIC_NORTH,RESEARCH_PHASE_B,TARGET_SELECTION,TRAINED_INTROSPECTION,LORA_PLAN}.md`.

```python
from zeta_life.introspection import psi_act_all   # 4 integration metrics over hidden states
# harness.py / concept_injection.py need the GPU stack (transformers); see experiments/introspection/
```

---

## Installation

```bash
git clone https://github.com/skyvanguard/zeta-life.git
cd zeta-life
pip install -e .

# With all extras (mpmath for exact zeta zeros, jupyter)
pip install -e ".[full]"
```

### Dependencies

- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib

---

## Quick Start

```python
import torch
from zeta_life.kernel import ConsciousKernel

# Create a kernel and run it for 100 steps
kernel = ConsciousKernel(obs_dim=4, latent_dim=32)
for i in range(100):
    stimulus = torch.randn(4)
    result = kernel.step(stimulus)

print(f"Psi: {result.psi:.4f}")
print(f"Free energy: {result.free_energy:.4f}")
print(f"Dreamed: {result.dreamed}")
```

### Run Experiments

```bash
# Psi validation on real datasets
python experiments/datasets/exp_real_data_psi.py

# Kernel validation (compositionality, grounding, emergence)
python experiments/kernel/exp_conscious_kernel_validation.py

# Multi-kernel organism dynamics
python experiments/kernel/exp_organism_emergence.py

# Science pipeline (toy -> instrument; see docs/SCIENCE_PLAN.md)
python experiments/kernel/exp_psi_vs_free_energy.py   # validate Psi (Albantakis method)
python experiments/kernel/exp_epistemic_depth.py      # precision hyper-model (epistemic depth)
python experiments/kernel/exp_yvyra_experiment.py     # Yvyra pipeline (modes + blind re-scorer)

# Introspection / the north (needs the GPU stack: transformers + bitsandbytes)
python experiments/introspection/exp_pik_probe.py          # is "I know" in the activations?
python experiments/introspection/exp_pik_train.py          # train P(IK) self-report (LoRA)
python experiments/introspection/exp_f3_inject_train.py    # trained injected-concept detection
```

### Run Tests

```bash
PYTHONPATH=src pytest tests/ -q   # 556 tests (or `pip install -e .` first)
```

---

## Project Structure

```
zeta-life/
|-- src/zeta_life/
|   |-- kernel/          # CORE - Active Inference kernel + Darwinian organism
|   |-- bridge/          # Yvyra coupling - feed a live agent's experience to the kernel
|   |-- introspection/   # THE NORTH - Psi over an LLM's activations; concept injection
|   |-- instrumentation/ # TickLogger - paired per-tick logging (science pipeline)
|   |-- integration/     # formal_equations.py - the integration index Psi
|   |-- datasets/        # Real-world dataset adapters (for Psi validation)
|   |-- core/            # zeta_constants, vertex, tetrahedral geometry
|   +-- utils/           # Shared utilities
|
|-- experiments/
|   |-- kernel/          # kernel experiments (the live research)
|   |-- introspection/   # the north - probe, P(IK) LoRA, injected-concept detection
|   +-- datasets/        # Psi on real data
|
|-- demos/               # quickstart.py - the 60-line kernel demo
|
|-- tests/               # 34 test files (585 tests)
+-- docs/                # Documentation, papers, plans (incl. ANTHROPIC_NORTH,
                         #   TRAINED_INTROSPECTION, TARGET_SELECTION, LORA_PLAN)

(Legacy subsystems -- psyche, hierarchical/IPUESA integration, organism,
evolution -- were archived on 2026-06-08 to the legacy/pre-refocus-snapshot branch.)
```

---

## Formal Equations

| Equation | Formula | Meaning |
|----------|---------|---------|
| Critical threshold | $\Phi_c = \frac{F_i}{\alpha - C}$ | Minimum integration for phase transition |
| Binding factor | $B = \frac{\Phi - \Phi_c}{\Phi_c}$ | Distance above threshold (normalized) |
| Integration index | $\Psi = B^3 + \Phi$ | Nonlinear amplification of coherence |
| Critical mass | $M_c = \frac{F_i}{\alpha - C}$ | Minimum units for integration to emerge |
| Corruption threshold | $1 - \frac{F_i}{\alpha \cdot M \cdot \alpha_s}$ | Maximum damage ratio before collapse |

These are implemented as pure functions in `integration/formal_equations.py` and consumed by the kernel's `_compute_psi()`.

---

## Theoretical Foundations

- **Active Inference / Free Energy Principle** (Friston) -- the kernel minimizes prediction error
- **Complementary Learning Systems** (McClelland et al.) -- fast episodic + slow semantic memory
- **Darwinian Brain** -- multi-kernel competition via Global Workspace
- **Integrated Information Theory** (Tononi) -- Phi as integration measure
- **LLM introspection** (Lindsey/Anthropic concept injection; Binder privileged access) -- the basis of the introspection program (see "The North")
- **Riemann zeta zeros** -- optional temporal basis (tested; not load-bearing except in spatial CA -- see "Zeta Function as Temporal Binding")

---

## Citation

```bibtex
@software{zeta_life_2026,
  author = {Francisco Ruiz},
  title = {Zeta Life: Emergent Coherent Integration in Adaptive Multi-Agent Systems},
  year = {2026},
  url = {https://github.com/skyvanguard/zeta-life}
}
```

---

## License

MIT License -- See [LICENSE](LICENSE) for details.
