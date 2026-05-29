"""Edge-case tests for the formal equations (Phase 1 fixes).

Covers the pole/clamp/régimen-imposible issues found in the audit:
- compute_B: pole at phi_c -> 0+
- predict_system_stability: subcritical regime no longer mislabelled STABLE
- compute_corruption_threshold: keeps negatives, validates inputs
- compute_M_c: dimensional caveat documented (behavioural check only)
- MicroPsyche.compute_surprise: L1 normalised to [0, 1]
"""
from __future__ import annotations

import pytest
import torch

from zeta_life.integration.formal_equations import (
    compute_B,
    compute_corruption_threshold,
    predict_system_stability,
)
from zeta_life.integration.micro_psyche import MicroPsyche


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


# --- #11 compute_surprise: normalised to [0, 1] --------------------------
def _psyche_with_states(a, b):
    mp = MicroPsyche.create_random()
    mp.recent_states.clear()
    mp.recent_states.append(torch.tensor(a))
    mp.recent_states.append(torch.tensor(b))
    return mp


def test_surprise_normalised_max_is_one():
    # Two opposite one-hot distributions: L1 = 2.0 -> normalised to 1.0.
    mp = _psyche_with_states([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
    assert mp.compute_surprise() == pytest.approx(1.0, abs=1e-6)


def test_surprise_identical_states_zero():
    mp = _psyche_with_states([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])
    assert mp.compute_surprise() == pytest.approx(0.0, abs=1e-6)


def test_surprise_partial_in_unit_interval():
    mp = _psyche_with_states([0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2])
    s = mp.compute_surprise()
    assert 0.0 < s < 1.0
