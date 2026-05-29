"""
Tests for formal consciousness equations.

Tests the mathematical framework:
    Phi_c = F_i / (alpha - C)
    B = (Phi - Phi_c) / Phi_c
    Psi = B^3 + Phi
    M_c = F_i / (alpha - C)
    Ec.13 corruption threshold
"""

import pytest

from zeta_life.integration.formal_equations import (
    compute_B,
    compute_corruption_threshold,
    compute_M_c,
    compute_phi_c,
    compute_psi,
    predict_system_stability,
)
from zeta_life.psyche.zeta_conscious_self import ConsciousnessIndex


class TestPhiC:
    """Tests for critical threshold computation."""

    def test_basic_computation(self):
        # Phi_c = 2.5 / (1.0 - 0.3) = 2.5 / 0.7 ≈ 3.571
        phi_c = compute_phi_c(F_i=2.5, alpha=1.0, C=0.3)
        assert pytest.approx(phi_c, rel=1e-3) == 2.5 / 0.7

    def test_zero_F_i(self):
        phi_c = compute_phi_c(F_i=0.0, alpha=1.0, C=0.3)
        assert phi_c == 0.0

    def test_alpha_equals_C(self):
        # Impossible regime: alpha == C → division by zero → inf
        phi_c = compute_phi_c(F_i=2.5, alpha=0.3, C=0.3)
        assert phi_c == float('inf')

    def test_alpha_less_than_C(self):
        phi_c = compute_phi_c(F_i=2.5, alpha=0.2, C=0.5)
        assert phi_c == float('inf')

    def test_high_F_i_raises_threshold(self):
        low = compute_phi_c(F_i=1.0, alpha=1.0, C=0.3)
        high = compute_phi_c(F_i=5.0, alpha=1.0, C=0.3)
        assert high > low


class TestBindingFactor:
    """Tests for binding factor B."""

    def test_subcritical(self):
        B = compute_B(phi=1.0, phi_c=3.0)
        assert B < 0

    def test_at_threshold(self):
        B = compute_B(phi=3.0, phi_c=3.0)
        assert B == 0.0

    def test_supercritical(self):
        B = compute_B(phi=6.0, phi_c=3.0)
        assert B == 1.0  # (6-3)/3 = 1

    def test_zero_phi_c(self):
        B = compute_B(phi=1.0, phi_c=0.0)
        assert B == 0.0

    def test_inf_phi_c(self):
        B = compute_B(phi=1.0, phi_c=float('inf'))
        assert B == 0.0


class TestPsi:
    """Tests for consciousness Psi = B^3 + Phi."""

    def test_subcritical_returns_zero(self):
        psi = compute_psi(phi=1.0, phi_c=3.0)
        assert psi == 0.0

    def test_at_threshold_returns_zero(self):
        psi = compute_psi(phi=3.0, phi_c=3.0)
        assert psi == 0.0

    def test_supercritical(self):
        # phi=6, phi_c=3: B=1, Psi = 1^3 + 6 = 7
        psi = compute_psi(phi=6.0, phi_c=3.0)
        assert pytest.approx(psi) == 7.0

    def test_cubic_amplification(self):
        # phi=9, phi_c=3: B=2, Psi = 8 + 9 = 17
        psi = compute_psi(phi=9.0, phi_c=3.0)
        assert pytest.approx(psi) == 17.0

    def test_sharp_transition(self):
        # Just below threshold → 0
        psi_below = compute_psi(phi=2.99, phi_c=3.0)
        # Just above threshold → small positive
        psi_above = compute_psi(phi=3.01, phi_c=3.0)
        assert psi_below == 0.0
        assert psi_above > 0.0

    def test_inf_phi_c(self):
        psi = compute_psi(phi=100.0, phi_c=float('inf'))
        assert psi == 0.0


class TestMc:
    """Tests for critical mass (same formula as Phi_c)."""

    def test_same_as_phi_c(self):
        mc = compute_M_c(F_i=2.5, alpha=1.0, C=0.3)
        phi_c = compute_phi_c(F_i=2.5, alpha=1.0, C=0.3)
        assert mc == phi_c


class TestCorruptionThreshold:
    """Tests for Ec. 13 corruption threshold."""

    def test_basic(self):
        # critical_ratio = 1 - F_i_b / (alpha * M * alpha_s)
        # = 1 - 2.5 / (1.0 * 100 * 1.0) = 1 - 0.025 = 0.975
        ratio = compute_corruption_threshold(
            F_i_b=2.5, alpha=1.0, M=100.0, alpha_s=1.0
        )
        assert pytest.approx(ratio, rel=1e-3) == 0.975

    def test_small_system(self):
        # Small M means lower tolerance for corruption
        ratio = compute_corruption_threshold(
            F_i_b=2.5, alpha=1.0, M=3.0, alpha_s=1.0
        )
        # 1 - 2.5/3 = 0.167
        assert pytest.approx(ratio, rel=1e-2) == 1.0 - 2.5/3.0

    def test_zero_M(self):
        ratio = compute_corruption_threshold(
            F_i_b=2.5, alpha=1.0, M=0.0, alpha_s=1.0
        )
        assert ratio == 0.0

    def test_high_F_i_b_returns_negative(self):
        # Very high F_i_b → base integration force exceeds coupled capacity →
        # structurally unstable. The negative value is now preserved (not clamped
        # to 0), matching the documented "If negative, inherently unstable".
        ratio = compute_corruption_threshold(
            F_i_b=500.0, alpha=1.0, M=10.0, alpha_s=1.0
        )
        assert ratio < 0.0
        assert ratio <= 1.0  # upper bound still enforced


class TestSystemStability:
    """Tests for predict_system_stability."""

    def test_healthy_system(self):
        result = predict_system_stability(
            F_i=2.5, alpha=1.0, C=0.3,
            M_current=100.0, M_corrupted=0.0
        )
        assert result['stability'] == 'STABLE'
        assert result['corruption_ratio'] == 0.0
        assert result['margin_to_collapse'] > 0.3

    def test_heavily_corrupted(self):
        result = predict_system_stability(
            F_i=2.5, alpha=1.0, C=0.3,
            M_current=100.0, M_corrupted=99.0
        )
        assert result['stability'] == 'COLLAPSING'
        assert result['margin_to_collapse'] <= 0

    def test_supercritical_detection(self):
        result = predict_system_stability(
            F_i=2.5, alpha=1.0, C=0.3,
            M_current=100.0
        )
        assert result['is_supercritical'] is True
        assert result['M_c'] < 100.0


class TestConsciousnessIndexIntegration:
    """Tests that ConsciousnessIndex works with formal equations."""

    def test_backward_compatible_default(self):
        """F_i=0 (default) → compute_total uses original weighted sum."""
        ci = ConsciousnessIndex(
            predictive=0.8, attention=0.7, integration=0.6,
            self_luminosity=0.5, stability=0.4, meta_awareness=0.3
        )
        total = ci.compute_total()
        expected = (
            0.20 * 0.8 + 0.20 * 0.7 + 0.25 * 0.6 +
            0.15 * 0.5 + 0.10 * 0.4 + 0.10 * 0.3
        )
        assert pytest.approx(total, rel=1e-3) == expected

    def test_formal_psi_active(self):
        """F_i > 0 → compute_total uses psi (clamped)."""
        ci = ConsciousnessIndex(
            predictive=0.8, attention=0.7, integration=0.6,
            self_luminosity=0.5, stability=0.4, meta_awareness=0.3,
            F_i=0.1, C=0.0, alpha=1.0
        )
        # phi_c = 0.1 / (1.0 - 0.0) = 0.1
        # phi = weighted sum ~ 0.6
        # B = (0.6 - 0.1) / 0.1 = 5.0
        # psi = 125 + 0.6 = 125.6
        # clamped to 1.0
        total = ci.compute_total()
        assert total == 1.0
        assert ci.compute_psi_raw() > 1.0

    def test_subcritical_psi(self):
        """When phi < phi_c, psi = 0."""
        ci = ConsciousnessIndex(
            predictive=0.1, attention=0.1, integration=0.1,
            self_luminosity=0.1, stability=0.1, meta_awareness=0.1,
            F_i=5.0, C=0.0, alpha=1.0
        )
        # phi_c = 5.0, phi ~ 0.1 → subcritical
        assert ci.psi == 0.0
        assert ci.B < 0

    def test_to_dict_includes_psi_when_active(self):
        ci = ConsciousnessIndex(F_i=1.0, alpha=1.0, C=0.0)
        d = ci.to_dict()
        assert 'psi' in d
        assert 'phi_c' in d
        assert 'B' in d

    def test_to_dict_excludes_psi_when_inactive(self):
        ci = ConsciousnessIndex()
        d = ci.to_dict()
        assert 'psi' not in d
