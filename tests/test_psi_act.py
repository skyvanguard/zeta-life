"""Tests for the Psi_act candidate integration metrics (introspection 'north')."""

import torch

from zeta_life.introspection import psi_act_all
from zeta_life.introspection.psi_act import (
    ALL_METRICS,
    interlayer_coherence,
    participation_ratio,
    phi_proxy,
    trajectory_predictability,
)


def _H(L=12, T=20, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(L, T, D, generator=g)


class TestShapesAndRange:
    def test_all_metrics_in_unit_range(self):
        H = _H()
        for k, v in psi_act_all(H).items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_keys_match_all_metrics(self):
        assert set(psi_act_all(_H()).keys()) == set(ALL_METRICS)

    def test_rejects_wrong_rank(self):
        try:
            participation_ratio(torch.randn(10, 10))
        except ValueError:
            return
        raise AssertionError("expected ValueError for non-3D input")


class TestParticipationRatio:
    def test_low_for_rank_one_cloud(self):
        # All tokens along one direction -> effective dim ~1/D -> low PR.
        L, T, D = 4, 30, 64
        direction = torch.randn(D)
        coeffs = torch.randn(T, 1)
        X = coeffs * direction  # [T, D] rank-1
        H = X.unsqueeze(0).repeat(L, 1, 1)
        assert participation_ratio(H) < 0.1

    def test_higher_for_isotropic_cloud(self):
        # Isotropic random cloud uses many dimensions -> higher PR than rank-1.
        H = _H(L=4, T=200, D=64)
        rank1 = (torch.randn(200, 1) * torch.randn(64)).unsqueeze(0).repeat(4, 1, 1)
        assert participation_ratio(H) > participation_ratio(rank1)


class TestInterlayerCoherence:
    def test_identical_layers_max_coherence(self):
        base = torch.randn(20, 64)
        H = base.unsqueeze(0).repeat(8, 1, 1)  # every layer identical
        assert interlayer_coherence(H) > 0.99

    def test_alternating_layers_low_coherence(self):
        v = torch.randn(64)
        T = 20
        a = v.unsqueeze(0).repeat(T, 1)
        b = (-v).unsqueeze(0).repeat(T, 1)
        H = torch.stack([a, b, a, b, a, b])  # flips each layer -> cos ~ -1
        assert interlayer_coherence(H) < 0.1


class TestTrajectoryPredictability:
    def test_smooth_trajectory_high(self):
        # Slowly drifting trajectory: consecutive states nearly identical.
        D = 64
        steps = torch.cumsum(0.01 * torch.randn(40, D), dim=0)
        base = torch.randn(D)
        X = base + steps  # smooth drift around base
        H = X.unsqueeze(0).repeat(6, 1, 1)
        assert trajectory_predictability(H) > 0.8

    def test_random_trajectory_midrange(self):
        # i.i.d. random states -> consecutive cosine ~0 -> ~0.5 after mapping.
        H = _H(L=6, T=80, D=64)
        assert 0.3 < trajectory_predictability(H) < 0.7


class TestPhiProxy:
    def test_coupled_halves_higher_than_independent(self):
        T, D = 200, 64
        half = D // 2
        shared = torch.randn(T, half)
        coupled = torch.cat([shared, shared + 0.05 * torch.randn(T, half)], dim=1)
        indep = torch.randn(T, D)
        Hc = coupled.unsqueeze(0).repeat(4, 1, 1)
        Hi = indep.unsqueeze(0).repeat(4, 1, 1)
        assert phi_proxy(Hc) > phi_proxy(Hi)
