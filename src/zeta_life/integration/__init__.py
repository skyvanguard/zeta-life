"""
Formal equations for the integration index Psi.

Psi = B^3 + Phi (cubic) or a bounded Hill variant; critical threshold
Phi_c = F_i / (alpha - C). These pure functions are consumed by the
ConsciousKernel as its consciousness/integration signal.

(The former hierarchical Cells->Clusters->Organism simulation and the IPUESA
resilience stack were archived in the 2026-06 refocus; see the
legacy/pre-refocus-snapshot branch.)
"""

from .formal_equations import (
    compute_B,
    compute_corruption_threshold,
    compute_M_c,
    compute_phi_c,
    compute_psi,
    compute_psi_hill,
    predict_system_stability,
)
