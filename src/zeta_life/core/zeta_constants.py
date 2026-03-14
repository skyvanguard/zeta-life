"""Zeta function constants and utilities.

Provides the non-trivial zeros of the Riemann zeta function,
used as temporal binding frequencies across the system.
"""
from __future__ import annotations

import numpy as np

try:
    from mpmath import zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

KNOWN_ZETA_ZEROS: list[float] = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
]


def get_zeta_zeros(M: int = 15) -> list[float]:
    """Return the first M non-trivial zeros of the Riemann zeta function.

    Uses mpmath for exact computation if available, otherwise falls back
    to hardcoded values and asymptotic approximation.
    """
    if HAS_MPMATH:
        return [float(zetazero(k).imag) for k in range(1, M + 1)]
    if M <= len(KNOWN_ZETA_ZEROS):
        return KNOWN_ZETA_ZEROS[:M]
    extra = [
        2 * np.pi * n / np.log(n + 2)
        for n in range(len(KNOWN_ZETA_ZEROS) + 1, M + 1)
    ]
    return KNOWN_ZETA_ZEROS + extra
