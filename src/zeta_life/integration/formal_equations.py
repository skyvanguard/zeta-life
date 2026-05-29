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

# Minimum critical threshold below which the binding factor B = (Phi-Phi_c)/Phi_c
# would blow up numerically. Thresholds at or below this are treated as the
# degenerate (no-binding) regime.
_PHI_C_MIN = 1e-6


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
    # Guard the pole at phi_c -> 0+: a vanishing threshold makes B blow up
    # (e.g. phi_c=1e-9 -> B~1e9), which then explodes any B**n downstream.
    # Treat a near-zero threshold as the degenerate (no-binding) case.
    if phi_c <= _PHI_C_MIN or phi_c == float('inf'):
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


def compute_psi_hill(phi: float, phi_c: float, n: float = 4.0, K: float = 0.1) -> float:
    """Compute consciousness index Psi via a Hill (cooperative) function.

    Psi = B^n / (K^n + B^n)   (when Phi > Phi_c)
    Psi = 0                    (when Phi <= Phi_c)
    with B = (Phi - Phi_c) / Phi_c.

    This is a bounded, continuous alternative to :func:`compute_psi` (B^3 + Phi).
    It fixes three issues of the cubic form found during calibration:

    1. **Saturation.** ``B^3 + Phi`` is unbounded; once B > 1 it explodes and any
       downstream clamp collapses every supercritical state to 1.0, so Psi stops
       discriminating *degrees* of integration. The Hill form is naturally bounded
       in ``[0, 1)`` — no clamp needed, no saturation.
    2. **Jump discontinuity.** ``B^3 + Phi`` jumps from 0 to Phi_c at the threshold
       (a first-order transition). The Hill form is continuous: Psi -> 0 as B -> 0.
    3. **Dimensional consistency.** ``B^3 + Phi`` adds a dimensionless ratio to an
       information quantity. The Hill form is a pure ratio (dimensionless).

    The exponent ``n`` controls how sharp the phase transition is (larger n ->
    steeper, closer to a step). ``K`` is the half-activation point (Psi = 0.5 when
    B = K). Both should be calibrated per system; the defaults are a reasonable
    starting point for the kernel's signal scale.

    Args:
        phi: Current integrated information.
        phi_c: Critical threshold.
        n: Hill coefficient (transition sharpness). Must be > 0.
        K: Half-activation point on B. Must be > 0.

    Returns:
        Psi in ``[0, 1)`` (0 if subcritical).
    """
    if phi_c <= 0 or phi_c == float('inf') or phi <= phi_c:
        return 0.0
    B = (phi - phi_c) / phi_c
    Bn = B ** n
    return Bn / (K ** n + Bn)


def compute_M_c(F_i: float, alpha: float, C: float) -> float:
    """
    Compute critical mass M_c for Goldilocks zone prediction.

    M_c = F_i / (alpha - C)

    This reuses the Phi_c formula but is interpreted as the minimum system mass
    (number of functional units) needed for consciousness to emerge.

    DIMENSIONAL CAVEAT: M_c shares Phi_c's formula F_i/(alpha - C), whose output
    has the units of integrated information, yet ``predict_system_stability``
    compares it against ``M_current`` (a unit count). This is only valid under
    the modelling assumption that F_i is expressed in units of mass, so that
    F_i/(alpha - C) yields a count directly. If that assumption does not hold,
    M_c needs its own scaling to mass units before the ``M_current > M_c``
    comparison is meaningful. Flagged for theoretical review.

    Args:
        F_i: Integration force
        alpha: Coupling parameter
        C: Coherence cost

    Returns:
        M_c: Critical mass (same formula as Phi_c; see dimensional caveat)
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
        Critical ratio, upper-bounded by 1.0. A NEGATIVE value means the system
        is inherently unstable (base integration force exceeds coupled capacity)
        and is returned as-is rather than clamped to 0 — clamping would hide the
        degree of instability and contradict the documented semantics.

    Raises:
        ValueError: if any of alpha, M, alpha_s is negative (invalid inputs).
    """
    if alpha < 0 or M < 0 or alpha_s < 0:
        raise ValueError(
            f"alpha, M and alpha_s must be non-negative; got "
            f"alpha={alpha}, M={M}, alpha_s={alpha_s}"
        )
    denominator = alpha * M * alpha_s
    if denominator == 0:
        return 0.0
    ratio = 1.0 - F_i_b / denominator
    # Only clamp the upper bound; keep negatives to signal structural instability.
    return min(1.0, ratio)


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
    M_c = compute_M_c(F_i, alpha, C)

    is_supercritical = M_current > M_c if M_c != float('inf') else False

    corruption_ratio = M_corrupted / M_current if M_current > 0 else 0.0
    critical_corruption = compute_corruption_threshold(F_i, alpha, M_current, alpha_s)
    margin = critical_corruption - corruption_ratio

    # A subcritical system (Phi cannot exceed the threshold, e.g. alpha <= C so
    # phi_c -> inf) cannot sustain integration regardless of corruption margin.
    # Report that directly instead of mislabelling it STABLE on the corruption axis.
    if not is_supercritical:
        stability = 'SUBCRITICAL'
    elif margin > 0.3:
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
