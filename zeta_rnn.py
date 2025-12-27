"""
ZetaRNN: RNN layers enriched with Riemann zeta kernel memory.

Based on: "IA Adaptativa a traves de la Hipotesis de Riemann" by Francisco Ruiz

Core formula: h'_t = h_t + m_t
Where: m_t = (1/N) * sum_j(phi(gamma_j) * h_{t-1} * cos(gamma_j * t))

The zeta zeros gamma_j create oscillators that capture long-range temporal dependencies.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# Reuse existing zeta zeros function
try:
    from mpmath import zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def get_zeta_zeros(M: int) -> List[float]:
    """Get first M non-trivial zeros of Riemann zeta function."""
    if HAS_MPMATH:
        return [float(zetazero(k).imag) for k in range(1, M + 1)]
    else:
        known = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
                 52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
                 67.079811, 69.546402, 72.067158, 75.704691, 77.144840]
        if M <= len(known):
            return known[:M]
        return known + [2 * np.pi * n / np.log(n + 2) for n in range(len(known) + 1, M + 1)]


class ZetaMemoryLayer(nn.Module):
    """
    Zeta-based temporal memory layer.

    Computes m_t = (1/N) * sum_j(phi(gamma_j) * h * cos(gamma_j * t))

    Where:
    - gamma_j: j-th non-trivial zero of Riemann zeta function
    - phi(gamma_j) = exp(-sigma * |gamma_j|): Abel regularization weight
    - t: current timestep
    - h: hidden state

    Args:
        hidden_size: Dimension of hidden state
        M: Number of zeta zeros to use (default: 15)
        sigma: Abel regularization parameter (default: 0.1)
        learnable_phi: If True, phi weights are learnable (default: False)
    """

    def __init__(
        self,
        hidden_size: int,
        M: int = 15,
        sigma: float = 0.1,
        learnable_phi: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.M = M
        self.sigma = sigma

        # Get zeta zeros
        gammas = get_zeta_zeros(M)
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))

        # Compute phi weights: phi(gamma) = exp(-sigma * |gamma|)
        phi_init = np.array([np.exp(-sigma * abs(g)) for g in gammas])

        if learnable_phi:
            self.phi = nn.Parameter(torch.tensor(phi_init, dtype=torch.float32))
        else:
            self.register_buffer('phi', torch.tensor(phi_init, dtype=torch.float32))

    def forward(self, h: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute zeta memory term m_t.

        Args:
            h: Hidden state (batch, hidden_size)
            t: Current timestep (integer)

        Returns:
            m_t: Memory contribution (batch, hidden_size)
        """
        # Compute cos(gamma_j * t) for all zeros
        # Shape: (M,)
        cos_terms = torch.cos(self.gammas * t)

        # Weighted sum: sum_j(phi_j * cos(gamma_j * t))
        # Shape: scalar
        weights = (self.phi * cos_terms).sum() / self.M

        # Apply to hidden state
        # m_t = weights * h
        m_t = weights * h

        return m_t
