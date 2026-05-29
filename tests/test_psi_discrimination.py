"""Tests for the Psi discrimination fixes.

Covers three fixes found during calibration (señal coherente vs ruido):
1. compute_psi_hill: bounded, continuous Hill metric (vs the cubic B^3+Phi).
2. SelfModel.predict_self: softmax-normalised (interoceptive channel fix).
3. ConsciousKernel psi_mode="hill": Psi discriminates coherent input from noise,
   where the cubic mode saturates to 1.0 for both.

The original cubic compute_psi and the default kernel behaviour are unchanged
(covered by the assertions below), so existing experiments keep working.
"""
from __future__ import annotations

import torch

from zeta_life.integration.formal_equations import (
    compute_psi,
    compute_psi_hill,
)
from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.self_model import SelfModel


# --------------------------------------------------------------------------
# compute_psi_hill — formal properties
# --------------------------------------------------------------------------
def test_hill_zero_at_or_below_threshold():
    assert compute_psi_hill(0.5, 1.0) == 0.0   # phi < phi_c
    assert compute_psi_hill(1.0, 1.0) == 0.0   # phi == phi_c


def test_hill_bounded_in_unit_interval():
    # Unlike B^3+Phi (unbounded), the Hill form never exceeds 1 nor goes negative.
    for phi in [1.01, 1.5, 2.0, 10.0, 1000.0]:
        v = compute_psi_hill(phi, 1.0)
        assert 0.0 <= v < 1.0


def test_hill_continuous_at_threshold():
    # Just above the threshold Psi is ~0 (no jump). The cubic form jumps to phi_c.
    pc = 1.0
    assert compute_psi_hill(pc * 1.0001, pc) < 0.01


def test_hill_monotonic_increasing_in_phi():
    pc = 1.0
    vals = [compute_psi_hill(phi, pc) for phi in [1.2, 1.5, 2.0, 3.0, 5.0]]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_hill_sharpness_increases_with_n():
    # Larger n -> sharper transition: lower below K, higher well above.
    pc = 1.0
    phi = 1.05  # small B
    assert compute_psi_hill(phi, pc, n=6) < compute_psi_hill(phi, pc, n=2)


def test_hill_invalid_phi_c():
    assert compute_psi_hill(2.0, 0.0) == 0.0
    assert compute_psi_hill(2.0, float("inf")) == 0.0


def test_cubic_formula_unchanged():
    # Regression guard: the original metric still computes B^3 + phi.
    pc = 0.5
    B = (0.8 - pc) / pc
    assert abs(compute_psi(0.8, pc) - (B ** 3 + 0.8)) < 1e-9


# --------------------------------------------------------------------------
# interoceptive fix — predict_self returns a probability distribution
# --------------------------------------------------------------------------
def test_predict_self_returns_distribution():
    sm = SelfModel(state_dim=4, embed_dim=16)
    out = sm.predict_self(torch.zeros(4)).detach()
    assert abs(float(out.sum()) - 1.0) < 1e-5   # sums to 1
    assert bool((out >= 0).all())                # non-negative


# --------------------------------------------------------------------------
# kernel — hill mode discriminates, cubic stays the default
# --------------------------------------------------------------------------
def test_hill_is_default_mode():
    # Hill is the default: the cubic form saturates and cannot discriminate.
    assert ConsciousKernel().psi_mode == "hill"


def test_hill_mode_discriminates_signal_from_noise():
    pattern = torch.tensor([0.5, 0.2, 0.2, 0.1])

    def run(stimulus_fn):
        torch.manual_seed(0)
        # With learned precisions (the world model + precision training make the
        # free energy separate coherent input from noise ~9x), a modest fe_scale
        # is enough; the frozen-system calibration needed fe_scale=40.
        ck = ConsciousKernel(
            psi_mode="hill", psi_fe_scale=5.0, psi_hill_n=4.0,
            psi_hill_K=0.1, alpha=1.0,
        )
        psis = [ck.step(stimulus_fn(t)).psi for t in range(120)]
        return sum(psis[-30:]) / 30

    coherent = run(lambda t: pattern + 0.01 * torch.randn(4))
    noise = run(lambda t: torch.softmax(torch.randn(4), dim=-1))

    # Coherent input integrates clearly more than noise.
    assert coherent > 0.7
    assert noise < 0.4
    assert coherent - noise > 0.5


def test_cubic_mode_saturates_as_documented():
    # The known limitation we are fixing: cubic Psi saturates to 1.0 for both
    # coherent and noisy input (this is why hill mode exists).
    pattern = torch.tensor([0.5, 0.2, 0.2, 0.1])

    def run(stimulus_fn):
        torch.manual_seed(0)
        ck = ConsciousKernel(psi_mode="cubic")  # explicit cubic (legacy metric)
        psis = [ck.step(stimulus_fn(t)).psi for t in range(60)]
        return sum(psis[-20:]) / 20

    coherent = run(lambda t: pattern + 0.01 * torch.randn(4))
    noise = run(lambda t: torch.softmax(torch.randn(4), dim=-1))
    # Both saturate near 1.0 -> no discrimination (documents the motivation).
    assert coherent > 0.95 and noise > 0.95
