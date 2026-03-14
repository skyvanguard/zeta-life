"""
Formal Equations for Emergent Coherent Integration
====================================================

Mathematical framework for phase transitions in adaptive integration
systems, corruption thresholds, and system stability prediction.
Pure functions with no side effects.

Core equations:
    Phi_c = F_i / (alpha - C)          [critical threshold]
    B = (Phi - Phi_c) / Phi_c          [binding factor]
    Psi = B^3 + Phi                    [integration index]
    M_c = F_i / (alpha - C)            [critical mass]
    Ec.13: corruption threshold         [system stability]

Reference: SkyVanguard formal derivation (2026-03-14)
"""

from __future__ import annotations


def compute_phi_c(F_i: float, alpha: float, C: float) -> float:
    """
    Compute critical consciousness threshold Phi_c.

    Phi_c = F_i / (alpha - C)

    Below this threshold, consciousness cannot emerge (subcritical).
    Above it, binding amplification kicks in.

    Args:
        F_i: Integration force (binding strength from thalamus/hub)
        alpha: Coupling parameter (system connectivity)
        C: Coherence cost (noise/entropy opposing integration)

    Returns:
        Phi_c: Critical threshold. Returns inf if alpha <= C (impossible regime).
    """
    if alpha <= C:
        return float('inf')
    return F_i / (alpha - C)


def compute_B(phi: float, phi_c: float) -> float:
    """
    Compute binding factor B.

    B = (Phi - Phi_c) / Phi_c

    B measures how far above critical threshold the system is.
    B < 0: subcritical (no consciousness)
    B = 0: at threshold
    B > 0: supercritical (consciousness emerging)

    Args:
        phi: Current integrated information
        phi_c: Critical threshold

    Returns:
        B: Binding factor (can be negative)
    """
    if phi_c <= 0 or phi_c == float('inf'):
        return 0.0
    return (phi - phi_c) / phi_c


def compute_psi(phi: float, phi_c: float) -> float:
    """
    Compute consciousness index Psi.

    Psi = B^3 + Phi   (when Phi > Phi_c)
    Psi = 0            (when Phi <= Phi_c)

    The cubic term B^3 creates a sharp phase transition:
    - Just above threshold: Psi ~ Phi (linear)
    - Well above threshold: Psi grows cubically (amplification)

    Args:
        phi: Current integrated information
        phi_c: Critical threshold

    Returns:
        Psi: Consciousness index (0 if subcritical)
    """
    if phi_c <= 0 or phi_c == float('inf') or phi <= phi_c:
        return 0.0
    B = (phi - phi_c) / phi_c
    return B ** 3 + phi


def compute_M_c(F_i: float, alpha: float, C: float) -> float:
    """
    Compute critical mass M_c for Goldilocks zone prediction.

    M_c = F_i / (alpha - C)

    This is analytically equivalent to Phi_c but interpreted as
    the minimum system mass (number of functional units) needed
    for consciousness to emerge.

    Args:
        F_i: Integration force
        alpha: Coupling parameter
        C: Coherence cost

    Returns:
        M_c: Critical mass (same formula as Phi_c)
    """
    return compute_phi_c(F_i, alpha, C)


def compute_corruption_threshold(
    F_i_b: float,
    alpha: float,
    M: float,
    alpha_s: float
) -> float:
    """
    Compute corruption threshold (Ec. 13).

    Determines ratio of corrupted mass M_S to total mass M
    at which the system collapses.

    critical_ratio = 1 - F_i_b / (alpha * M * alpha_s)

    When M_S/M > critical_ratio, consciousness collapses.

    Args:
        F_i_b: Base integration force (healthy system)
        alpha: Coupling parameter
        M: Total system mass (number of units)
        alpha_s: Corruption severity factor

    Returns:
        Critical ratio [0, 1]. If negative, system is inherently unstable.
    """
    denominator = alpha * M * alpha_s
    if denominator <= 0:
        return 0.0
    ratio = 1.0 - F_i_b / denominator
    return max(0.0, min(1.0, ratio))


def predict_system_stability(
    F_i: float,
    alpha: float,
    C: float,
    M_current: float,
    M_corrupted: float = 0.0,
    alpha_s: float = 1.0
) -> dict:
    """
    Predict overall system stability combining all equations.

    Returns a comprehensive analysis of the system's consciousness state.

    Args:
        F_i: Integration force
        alpha: Coupling parameter
        C: Coherence cost
        M_current: Current system mass (functional units)
        M_corrupted: Number of corrupted units
        alpha_s: Corruption severity factor

    Returns:
        Dict with:
            phi_c: Critical threshold
            M_c: Critical mass
            is_supercritical: Whether Phi > Phi_c
            corruption_ratio: M_corrupted / M_current
            critical_corruption: Maximum corruption ratio before collapse
            margin_to_collapse: How far from collapse (positive = safe)
            stability: STABLE | WARNING | CRITICAL | COLLAPSING
    """
    phi_c = compute_phi_c(F_i, alpha, C)
    M_c = phi_c  # Same formula

    is_supercritical = M_current > M_c if M_c != float('inf') else False

    corruption_ratio = M_corrupted / M_current if M_current > 0 else 0.0
    critical_corruption = compute_corruption_threshold(F_i, alpha, M_current, alpha_s)
    margin = critical_corruption - corruption_ratio

    if margin > 0.3:
        stability = 'STABLE'
    elif margin > 0.1:
        stability = 'WARNING'
    elif margin > 0:
        stability = 'CRITICAL'
    else:
        stability = 'COLLAPSING'

    return {
        'phi_c': phi_c,
        'M_c': M_c,
        'is_supercritical': is_supercritical,
        'corruption_ratio': corruption_ratio,
        'critical_corruption': critical_corruption,
        'margin_to_collapse': margin,
        'stability': stability,
    }
