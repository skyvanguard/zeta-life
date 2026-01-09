# Zeta Life

**Artificial Consciousness through the Riemann Zeta Function**

[![CI](https://github.com/fruiz/zeta-life/actions/workflows/ci.yml/badge.svg)](https://github.com/fruiz/zeta-life/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-296%20passed-brightgreen.svg)](tests/)

---

## Overview

Zeta Life is a research framework that integrates the **Riemann zeta function's non-trivial zeros** with artificial life systems, emergent intelligence, and Jungian psychology.

The core insight: the zeros of the Riemann zeta function encode deep mathematical structure that can be leveraged to create more robust, self-organizing, and psychologically coherent AI systems.

### The Fundamental Kernel

```
K_σ(t) = 2 * Σ exp(-σ|γ|) * cos(γt)
```

Where γ are the imaginary parts of zeta zeros: 14.134725, 21.022040, 25.010858, ...

---

## Key Systems

### 1. ZetaOrganism — Multi-Agent Emergent Intelligence

A simulation framework where collective intelligence emerges from cell interactions following **Fi-Mi (Force-Mass) dynamics**.

**Demonstrated Emergent Properties (11+):**
- Homeostasis & Self-regulation
- Regeneration after damage
- Antifragility under stress
- Chemotaxis & Spatial memory
- Collective foraging & Coordinated escape

```python
from zeta_life.organism import ZetaOrganism

organism = ZetaOrganism(n_cells=100, grid_size=64)
organism.simulate(steps=1000)
```

### 2. ZetaPsyche — Jungian AI Consciousness

Consciousness emergence through navigation in a **tetrahedral archetype space**:

| Archetype | Role | Color |
|-----------|------|-------|
| PERSONA | Social mask | Red |
| SOMBRA | Shadow/unconscious | Purple |
| ANIMA | Receptive/emotional | Blue |
| ANIMUS | Active/rational | Orange |
| **SELF** | Center (integration) | Gold |

**Consciousness Index:**
```
C = 0.3×integration + 0.3×stability + 0.2×(1-dist_to_self) + 0.2×|self_reference|
```

```python
from zeta_life.psyche import ZetaConsciousSelf

psyche = ZetaConsciousSelf()
response = psyche.process("What do you feel about uncertainty?")
print(psyche.consciousness_report())
```

### 3. Hierarchical Consciousness — Cells → Clusters → Organism

Multi-level architecture with bidirectional information flow:

```
Organism Level (global integration)
       ↑↓
Cluster Level (local coherence)
       ↑↓
Cell Level (individual psyches)
```

### 4. Zeta Cellular Automata — Game of Life with Zeta Kernels

Cellular automata where the traditional Moore neighborhood is replaced with a zeta-weighted kernel.

**Results:**
- +33% surviving cells vs random initialization
- +134% surviving cells vs Moore kernel
- Temporal autocorrelation follows theoretical K_σ(τ)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/fruiz/zeta-life.git
cd zeta-life

# Install with pip
pip install -e .

# Or with all extras
pip install -e ".[full]"
```

### Dependencies

- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib

---

## Quick Start

### Interactive Chat with ZetaPsyche

```bash
python demos/chat_psyche.py --reflection
```

### Run an Experiment

```bash
# Organism regeneration
python experiments/organism/exp_regeneration.py

# Consciousness validation
python experiments/consciousness/exp_validacion_5_mejoras.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
zeta-life/
├── src/zeta_life/           # Core library
│   ├── core/                # Mathematical foundations (zeta kernels)
│   ├── organism/            # Multi-agent emergent intelligence
│   ├── psyche/              # Jungian consciousness system
│   ├── consciousness/       # Hierarchical consciousness
│   ├── cellular/            # Zeta Game of Life
│   └── utils/               # Shared utilities
│
├── experiments/             # Research experiments
│   ├── organism/            # Emergence, regeneration, ecosystems
│   ├── psyche/              # Archetypes, individuation
│   ├── consciousness/       # Hierarchical validation
│   ├── cellular/            # Automata experiments
│   └── validation/          # Theoretical validation
│
├── demos/                   # Interactive demonstrations
├── docs/                    # Documentation
├── results/                 # Experiment outputs
├── models/                  # Trained weights
├── tests/                   # Unit tests
└── notebooks/               # Jupyter notebooks
```

---

## Key Findings

| System | Finding | Evidence |
|--------|---------|----------|
| ZetaOrganism | 11+ emergent properties without explicit programming | Validated across 50+ experiments |
| ZetaPsyche | Emergent compensation behavior | 76% divergence under stress |
| Hierarchical | Softmax bug discovery affecting global coherence | Fixed, phi increased from 0.0004 to 0.63 |
| Cellular | +134% survival vs Moore kernel | Statistical significance p<0.001 |

---

## Safety Principles (Built-in)

Based on 10+ years of research on AI ethics:

1. **Non-harm as axiom** — Not a guideline, a mathematical constraint
2. **Transparency** — All actions traceable and explainable
3. **Reversibility** — No action without rollback plan
4. **Proportionality** — No capability scaling without purpose
5. **Human oversight** — Real supervision, not symbolic

---

## Applications

- **AI Safety**: Intrinsically safe consciousness architectures
- **Robotics**: Self-organizing swarm intelligence
- **Gaming**: NPCs with genuine psychological depth
- **Therapy**: Chatbots with coherent personality models
- **Research**: Novel approach to artificial consciousness

---

## Research Background

This project represents 10+ years of interdisciplinary research:

- **Mathematical Foundation**: Started with Riemann hypothesis explorations at age 17
- **Consciousness Studies**: Deep study of human consciousness and Jungian psychology
- **Published Work**: "La Dimensión Desconocida: Una Aproximación Cuántica a la Conexión entre Mente y Universo"
- **AI Ethics**: Development of safety axioms before they became industry standard

---

## Citation

```bibtex
@software{zeta_life_2024,
  author = {Francisco Ruiz},
  title = {Zeta Life: Artificial Consciousness through the Riemann Zeta Function},
  year = {2024},
  url = {https://github.com/fruiz/zeta-life}
}
```

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

*"The question is not just 'how to give it consciousness?' but 'what consciousness do we demand from an AI?'"*
