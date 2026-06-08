"""Edge-case tests for the formal equations (Phase 1 fixes).

Covers the pole/clamp/régimen-imposible issues found in the audit:
- compute_B: pole at phi_c -> 0+
- predict_system_stability: subcritical regime no longer mislabelled STABLE
- compute_corruption_threshold: keeps negatives, validates inputs
- compute_M_c: dimensional caveat documented (behavioural check only)
"""
from __future__ import annotations

import pytest

from zeta_life.integration.formal_equations import (
    compute_B,
    compute_corruption_threshold,
    predict_system_stability,
)


# --- #7 compute_B: pole guarded -----------------------------------------
def test_compute_B_guards_tiny_phi_c():
    # A vanishing threshold must not blow up to ~1e9.
    assert compute_B(1.0, 1e-9) == 0.0
    assert compute_B(1.0, 0.0) == 0.0
    assert compute_B(1.0, float("inf")) == 0.0


def test_compute_B_normal_regime_unchanged():
    # Above the guard it still computes (phi - phi_c)/phi_c.
    assert compute_B(1.0, 0.5) == pytest.approx(1.0)


# --- #8 predict_system_stability: subcritical not STABLE -----------------
def test_subcritical_regime_reported_as_subcritical():
    # alpha <= C -> phi_c = inf -> cannot integrate. Must NOT be STABLE.
    out = predict_system_stability(F_i=1.0, alpha=0.5, C=0.5, M_current=100)
    assert out["phi_c"] == float("inf")
    assert out["is_supercritical"] is False
    assert out["stability"] == "SUBCRITICAL"


def test_supercritical_healthy_is_stable():
    out = predict_system_stability(F_i=1.0, alpha=2.0, C=0.1, M_current=100)
    assert out["is_supercritical"] is True
    assert out["stability"] == "STABLE"


# --- #9 compute_corruption_threshold: negatives + validation -------------
def test_corruption_threshold_keeps_negative():
    # Base force exceeds coupled capacity -> structurally unstable -> negative,
    # no longer clamped to 0.
    val = compute_corruption_threshold(F_i_b=10.0, alpha=1.0, M=1.0, alpha_s=1.0)
    assert val < 0.0


def test_corruption_threshold_upper_bounded():
    val = compute_corruption_threshold(F_i_b=0.0, alpha=5.0, M=5.0, alpha_s=1.0)
    assert val <= 1.0


def test_corruption_threshold_rejects_negative_inputs():
    with pytest.raises(ValueError):
        compute_corruption_threshold(F_i_b=1.0, alpha=-2.0, M=1.0, alpha_s=1.0)


def test_corruption_threshold_zero_denominator():
    assert compute_corruption_threshold(F_i_b=1.0, alpha=1.0, M=0.0, alpha_s=1.0) == 0.0
