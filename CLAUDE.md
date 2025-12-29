# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Zeta Game of Life** project - an implementation of cellular automata using kernels derived from the Riemann zeta function's non-trivial zeros. The project explores the integration of number theory (specifically the Riemann Hypothesis) with computational biology/artificial life.

**Theoretical Foundation**: The system uses the kernel `K_σ(t) = 2 * Σ exp(-σ|γ|) * cos(γt)` where γ are the imaginary parts of zeta zeros (14.134725, 21.022040, 25.010858, ...).

## Commands

```bash
# Run Phase 1: Structured initialization with zeta noise
python zeta_game_of_life.py

# Run Phase 2: Weighted neighborhood kernel
python zeta_gol_fase2.py

# Run Phase 3: Complete system with temporal memory
python zeta_gol_fase3.py

# Run Neural CA evolution (requires PyTorch)
python zeta_neural_ca.py

# Run ZetaLSTM experiment (RNN with zeta memory)
python zeta_rnn.py

# Run ZetaRNN tests
python -m pytest tests/test_zeta_rnn.py -v
```

### Dependencies

```bash
pip install numpy matplotlib scipy
pip install mpmath        # Optional: for exact zeta zeros
pip install torch         # Required only for zeta_neural_ca.py
```

## Architecture

### Phase Progression

The project builds incrementally across four phases:

| Phase | File | Key Contribution |
|-------|------|------------------|
| 1 | `zeta_game_of_life.py` | Zeta-structured initialization instead of random |
| 2 | `zeta_gol_fase2.py` | Zeta-weighted kernel replacing Moore neighborhood |
| 3 | `zeta_gol_fase3.py` | Full system with temporal memory via L_zeros |
| NCA | `zeta_neural_ca.py` | Differentiable Neural CA with zeta perception |
| RNN | `zeta_rnn.py` | LSTM enriched with zeta temporal memory layer |

### Core Components

**ZetaKernel** (all phases): Evaluates the kernel at spatial/temporal points using the first M zeta zeros with Abel regularization parameter σ.

**ZetaLaplaceOperator** (Phase 3): Implements bilateral Laplace transform for temporal memory filtering: `x_filtered(t) = Σ_τ K_σ(τ) * history[t-τ]`

**ZetaSpectralFilter** (Phase 3): FFT-based filter with transfer function `H(ω) = Σ exp(-σ|γ|) / (1 + ((ω-γ)/σ)²)` - creates resonance peaks at zeta frequencies.

**ZetaNCA** (Neural CA): PyTorch module with non-learnable zeta perception kernel + learnable update network. Uses Sobel filters alongside zeta convolution for gradient detection.

**ZetaMemoryLayer** (RNN): Computes temporal memory term `m_t = (1/M) * Σ phi(γ_j) * h * cos(γ_j * t)` using zeta zeros as oscillator frequencies.

**ZetaLSTMCell** (RNN): Standard LSTMCell enhanced with additive zeta memory: `h'_t = h_t + α * m_t`

**ZetaLSTM** (RNN): Full sequence processing layer, drop-in replacement for `nn.LSTM` with zeta temporal memory.

### Key Parameters

- `M`: Number of zeta zeros (15-30 typical, more = finer structure)
- `sigma`: Abel regularization, controls decay (0.05-0.1 optimal)
- `R`: Kernel radius in cells (2-3 typical)
- `alpha`: Memory weight in evolution equation (0.05-0.15)
- `beta`: Spectral filter weight (0.02-0.08)
- `birth_range` / `survive_range`: Continuous thresholds replacing B3/S23
- `zeta_weight`: RNN memory scaling factor (0.3-0.5 for best results)

### Evolution Equation (Phase 3)

```
x(t+1) = GoL(x(t)) + α*Memory(history) + β*Spectral(x(t))
```

The binarization uses adaptive threshold: `mean(combined) + 0.1*std(combined)`

### RNN Enhancement Equation (zeta_rnn.py)

```
h'_t = h_t + zeta_weight * m_t
m_t = (1/M) * Σ_j exp(-σ|γ_j|) * h_{t-1} * cos(γ_j * t)
```

Based on paper Section 6.2: "IA Adaptativa a traves de la Hipotesis de Riemann"

## Key Findings

### Cellular Automata (Phases 1-3, NCA)
- Zeta initialization produces +33% more surviving cells vs random
- Zeta kernel produces +134% more surviving cells vs Moore kernel
- Spatial correlations show characteristic zeta oscillations that persist through evolution
- Temporal autocorrelation follows the theoretical kernel K_σ(τ)

### RNN Experiments (zeta_rnn.py)
- ZetaLSTM shows **+5.5% improvement** over vanilla LSTM on zeta-structured noise filtering
- No improvement on generic sequence tasks (both converge similarly)
- Best results when data has temporal correlations matching zeta zero frequencies
- 12 unit tests passing, full documentation in `docs/zeta-lstm-hallazgos.md`
