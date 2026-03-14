# Zeta Life

**Emergent coherent integration in adaptive multi-agent systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-766%20passed-brightgreen.svg)](tests/)

---

## Overview

Zeta Life is a computational framework for studying **phase transitions in adaptive integration systems**. It models how independent processing units (kernels) can transition from fragmented computation to coherent, unified integration -- and how populations of such units exhibit emergent collective behavior.

The core equation predicts when integration emerges:

$$\Psi = B^3 + \Phi \qquad \text{(integration index, supercritical regime)}$$

$$\Phi_c = \frac{F_i}{\alpha - C} \qquad \text{(critical threshold for phase transition)}$$

$$B = \frac{\Phi - \Phi_c}{\Phi_c} \qquad \text{(binding factor)}$$

The cubic term $B^3$ creates a **sharp phase transition**: below the critical threshold $\Phi_c$, integration is zero. Above it, the system amplifies nonlinearly -- a mathematical signature of emergent coherence.

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
from zeta_life.kernel import ConsciousKernel

kernel = ConsciousKernel(state_dim=32, alpha=1.0)
result = kernel.step(stimulus)
print(result.psi)       # integration index [0, 1]
print(result.free_energy)  # prediction error magnitude
```

### Multi-Kernel Organism (Darwinian Brain)

Multiple kernels compete for a shared `GlobalWorkspace` via proposal strength. Energy-managed spawning, merging, and death create a Darwinian selection process at the kernel level.

```python
from zeta_life.kernel import ConsciousOrganism

organism = ConsciousOrganism(n_kernels=5, state_dim=32)
result = organism.step(stimulus)
print(result.psi)           # organism-level integration
print(result.active_kernels) # surviving kernel count
```

### Hierarchical Integration (Cells -> Clusters -> Organism)

A separate simulation layer models multi-level integration with bidirectional information flow:

```
Organism Level   (global integration, top-down modulation)
       ^v
Cluster Level    (local coherence, inter-cluster binding)
       ^v
Cell Level       (individual micro-psyches with archetypes)
```

The formal equations predict **corruption thresholds** (how much damage before collapse) and **critical mass** (minimum units needed for integration to emerge).

### Zeta Function as Temporal Binding

The Riemann zeta function's non-trivial zeros provide the temporal structure:

$$K_\sigma(t) = 2 \sum_n \exp\!\bigl(-\sigma\,|\gamma_n|\bigr)\,\cos(\gamma_n\, t)$$

where $\gamma_n$ are the imaginary parts of zeta zeros (14.134, 21.022, 25.011, ...). These frequencies drive memory consolidation in the DreamEngine and temporal binding in the ZetaRNN.

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
kernel = ConsciousKernel(state_dim=32)
for i in range(100):
    stimulus = torch.randn(32)
    result = kernel.step(stimulus)

print(f"Psi: {result.psi:.4f}")
print(f"Free energy: {result.free_energy:.4f}")
print(f"Dreamed: {result.dreamed}")
```

### Run Experiments

```bash
# Phase transition analysis with real datasets
python experiments/datasets/exp_phase_transitions.py

# Kernel validation (compositionality, grounding, emergence)
python experiments/kernel/exp_conscious_kernel_validation.py

# Multi-kernel organism dynamics
python experiments/kernel/exp_organism_emergence.py
```

### Run Tests

```bash
pytest tests/ -v   # 766 tests
```

---

## Project Structure

```
zeta-life/
|-- src/zeta_life/
|   |-- kernel/          # Active Inference kernel + organism
|   |-- integration/     # Hierarchical integration (formal equations)
|   |-- core/            # Zeta constants, RNN, resonance, tetrahedral space
|   |-- organism/        # Multi-agent emergent intelligence (Fi-Mi dynamics)
|   |-- evolution/       # Evolutionary parameter optimization (IPUESA)
|   |-- psyche/          # Legacy archetype system (experimental)
|   |-- datasets/        # Real-world dataset adapters
|   +-- utils/           # Shared utilities
|
|-- experiments/
|   |-- kernel/          # Kernel validation experiments
|   +-- datasets/        # Phase transition + real data experiments
|
|-- tests/               # 766 unit tests
+-- docs/                # Documentation and plans
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

These are implemented as pure functions in `integration/formal_equations.py` and used by both the hierarchical simulation and the kernel's `_compute_psi()`.

---

## Theoretical Foundations

- **Active Inference / Free Energy Principle** (Friston) -- the kernel minimizes prediction error
- **Complementary Learning Systems** (McClelland et al.) -- fast episodic + slow semantic memory
- **Darwinian Brain** -- multi-kernel competition via Global Workspace
- **Integrated Information Theory** (Tononi) -- Phi as integration measure
- **Riemann zeta zeros** -- temporal binding frequencies for memory consolidation

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
