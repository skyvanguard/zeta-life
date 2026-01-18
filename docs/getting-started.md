# Getting Started

Welcome to Zeta-Life! This guide will help you get up and running quickly.

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/skyvanguard/zeta-life.git
cd zeta-life

# Install with all dependencies
pip install -e ".[full]"
```

### Minimal Install

```bash
# Core only (no visualization)
pip install -e .
```

### Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- PyTorch (for neural components)
- Optional: mpmath (for exact zeta zeros)

## Quick Start Examples

### 1. Zeta Game of Life

Run a cellular automaton with zeta-weighted kernels:

```python
from zeta_life.cellular import ZetaGameOfLife

# Create 64x64 grid with zeta kernel
gol = ZetaGameOfLife(grid_size=64, M=15, sigma=0.05)

# Run 100 steps
for _ in range(100):
    gol.step()

# Visualize
gol.plot()
```

### 2. ZetaOrganism

Create an emergent multi-agent system:

```python
from zeta_life.organism import ZetaOrganism

# Create organism with 100 cells
organism = ZetaOrganism(n_cells=100, grid_size=48)

# Run simulation
for _ in range(200):
    organism.step()

# Check emergent properties
print(f"Clusters: {organism.count_clusters()}")
print(f"Coherence: {organism.measure_coherence():.3f}")
```

### 3. Interactive Psyche Chat

```bash
# Start interactive session with self-reflection
python demos/chat_psyche.py --reflection
```

## Running Experiments

### Reproduce Paper Results

```bash
# Generate all paper figures
python scripts/reproduce_paper.py

# Run specific experiments
python experiments/organism/exp_organism.py
python experiments/consciousness/exp_ipuesa_synth_v2_consolidation.py
```

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_organism.py -v
```

## Project Structure

```
zeta-life/
├── src/zeta_life/      # Core library
│   ├── core/           # Zeta mathematics
│   ├── organism/       # Multi-agent emergence
│   ├── psyche/         # Jungian archetypes
│   ├── consciousness/  # Hierarchical consciousness
│   └── cellular/       # Game of Life variants
├── experiments/        # Research experiments
├── demos/              # Interactive demos
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Key Concepts

### The Zeta Kernel

At the heart of Zeta-Life is the kernel derived from Riemann zeta zeros:

$$K_\sigma(t) = 2 \sum_{n=1}^{M} e^{-\sigma |\gamma_n|} \cos(\gamma_n t)$$

Where $\gamma_n$ are the imaginary parts of non-trivial zeros (14.134725, 21.022040, ...).

### Abstract Vertices (V0-V3)

Instead of biased Jungian archetypes, we use a neutral tetrahedron:

| Vertex | Description |
|--------|-------------|
| V0 | First principal component |
| V1 | Second principal component |
| V2 | Third principal component |
| V3 | Fourth principal component |

### Goldilocks Zone

Functional identity exists only within a narrow parameter window (±5%). Too low = chaos, too high = rigidity.

## Next Steps

- Read the [main paper](papers/zeta-life-framework-paper.md) for theoretical foundations
- Explore the [API reference](api/modules.rst) for detailed documentation
- Check [experiments](EXPERIMENTS.md) for research scripts
- See [troubleshooting](TROUBLESHOOTING.md) if you encounter issues

## Citation

```bibtex
@article{ruiz2026zetalife,
  title={Zeta-Life: A Unified Framework Connecting Riemann Zeta Mathematics,
         Multi-Agent Dynamics, and Computational Functional Identity},
  author={Ruiz, Francisco},
  journal={arXiv preprint},
  year={2026}
}
```
