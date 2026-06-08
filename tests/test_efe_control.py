"""Tests for continuous EFE action selection (the 'controla' work)."""

from __future__ import annotations

import statistics as st

import pytest
import torch

from zeta_life.kernel import ConsciousKernel

TARGET = torch.tensor([0.7, 0.1, 0.1, 0.1])
TARGET = TARGET / TARGET.sum()


def _defaults():
    return ConsciousKernel()


class TestDefaults:
    def test_efe_defaults_unchanged(self):
        ck = _defaults()
        assert ck.efe_n_samples == 0
        assert ck.efe_horizon == 1
        assert ck.efe_discount == 1.0
        assert ck.efe_obs_norm == "softmax"
        assert ck.efe_cem_iters == 0


class TestPlannerValidity:
    def test_continuous_candidates_return_valid_action(self):
        ck = ConsciousKernel(action_mode="efe", preference=TARGET,
                             efe_n_samples=16, efe_obs_norm="l1")
        result = ck.step(torch.rand(4))
        assert result.action.shape == (4,)
        assert float(result.action.sum()) == pytest.approx(1.0, abs=1e-4)

    def test_horizon_returns_valid_action(self):
        ck = ConsciousKernel(action_mode="efe", preference=TARGET,
                             efe_n_samples=8, efe_horizon=4, efe_obs_norm="l1")
        for _ in range(5):
            result = ck.step(torch.rand(4))
        assert result.action.shape == (4,)


def _control_dist(arm_kwargs, n_steps=250, warmup=120, seed=0):
    """Run the inertial control task; return mean ||state - C|| over the tail."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    state = torch.rand(4, generator=g)
    state = state / state.sum()
    ck = ConsciousKernel(preference=TARGET, **arm_kwargs)
    obs = state.clone()
    dists = []
    for t in range(n_steps):
        result = ck.step(obs)
        if t < warmup:
            a = torch.rand(4)
            a = a / a.sum()
            ck.last_action = a.detach()
        else:
            a = result.action
        state = ((1 - 0.3) * state + 0.3 * a.detach()).clamp(min=1e-6)
        state = state / state.sum()
        obs = state
        dists.append(float(torch.linalg.vector_norm(state - TARGET)))
    return st.mean(dists[-50:])


class TestControl:
    def test_continuous_reaches_nonvertex_target(self):
        """Continuous EFE drives the state onto a non-vertex target."""
        dist = _control_dist(
            dict(action_mode="efe", efe_n_samples=48, efe_obs_norm="l1"))
        assert dist < 0.12, f"continuous EFE did not reach the target (dist={dist:.3f})"

    def test_continuous_beats_discrete_onehots(self):
        """The one-hot planner cannot represent a non-vertex target; continuous can."""
        cont = _control_dist(
            dict(action_mode="efe", efe_n_samples=48, efe_obs_norm="l1"))
        disc = _control_dist(
            dict(action_mode="efe", efe_n_samples=0, efe_obs_norm="softmax"))
        assert cont < disc * 0.7, f"continuous ({cont:.3f}) should beat discrete ({disc:.3f})"

    def test_cem_also_reaches_target(self):
        """CEM is a valid alternative search and also reaches the target."""
        dist = _control_dist(
            dict(action_mode="efe", efe_n_samples=16, efe_cem_iters=3, efe_obs_norm="l1"))
        assert dist < 0.12, f"CEM did not reach the target (dist={dist:.3f})"


class TestEFEFixes:
    """Locks in the correctness fixes from the adversarial review."""

    def test_to_simplex_handles_negatives(self):
        # H5: l1 projection floors negatives and returns a valid distribution.
        ck = ConsciousKernel(action_mode="efe", preference=TARGET, efe_obs_norm="l1")
        proj = ck._to_simplex(torch.tensor([2.0, -1.0, 0.5, -3.0]))
        assert abs(float(proj.sum()) - 1.0) < 1e-5
        assert (proj >= 0).all()
        assert proj[1] == 0.0 and proj[3] == 0.0  # negatives floored

    def test_to_simplex_uniform_fallback(self):
        ck = ConsciousKernel(action_mode="efe", preference=TARGET, efe_obs_norm="l1")
        proj = ck._to_simplex(torch.tensor([-1.0, -2.0, -3.0, -4.0]))
        assert torch.allclose(proj, torch.full((4,), 0.25), atol=1e-6)

    def test_horizon_features_are_anticipatory(self):
        # H4: with a temporal bank, the horizon uses DISTINCT future codes
        # phi(t+1..t+H), not a frozen present code.
        from zeta_life.kernel.temporal_features import OscillatorBank
        ck = ConsciousKernel(action_mode="efe", preference=TARGET,
                             temporal_features=OscillatorBank.zeta(M=8))
        ck.t = 5
        feats = ck._horizon_features(3)
        assert feats is not None and len(feats) == 3
        assert not torch.allclose(feats[0], feats[1])  # distinct future steps

    def test_horizon_features_none_without_bank(self):
        ck = ConsciousKernel(action_mode="efe", preference=TARGET)
        assert ck._horizon_features(3) is None

    def test_efe_temporal_horizon_runs(self):
        from zeta_life.kernel.temporal_features import OscillatorBank
        ck = ConsciousKernel(action_mode="efe", preference=TARGET, efe_obs_norm="l1",
                             efe_n_samples=8, efe_horizon=3,
                             temporal_features=OscillatorBank.zeta(M=8))
        for _ in range(5):
            result = ck.step(torch.rand(4))
        assert result.action.shape == (4,)


class TestModelBasedControl:
    """The kernel learns unknown (permuted) action dynamics and inverts them."""

    def _run(self, arm, n_steps=300, warmup=150, seed=0):
        torch.manual_seed(seed)
        perm = torch.tensor([2, 0, 3, 1])
        g = torch.Generator().manual_seed(seed)
        state = torch.rand(4, generator=g)
        state = state / state.sum()
        ck = None
        if arm == "kernel":
            ck = ConsciousKernel(action_mode="efe", preference=TARGET,
                                 efe_n_samples=48, efe_obs_norm="l1",
                                 reflect_interval=10**9, dream_interval=10**9)
        obs = state.clone()
        dists = []
        for t in range(n_steps):
            if arm == "naive":
                a = TARGET.clone()
            else:
                result = ck.step(obs)
                if t < warmup:
                    a = torch.rand(4); a = a / a.sum()
                    ck.last_action = a.detach()
                else:
                    a = result.action
            permuted = a.detach()[perm]
            state = ((1 - 0.4) * state + 0.4 * permuted).clamp(min=1e-6)
            state = state / state.sum()
            obs = state
            dists.append(float(torch.linalg.vector_norm(state - TARGET)))
        return st.mean(dists[-60:])

    def test_kernel_learns_permuted_dynamics(self):
        kernel = self._run("kernel")
        naive = self._run("naive")
        assert kernel < 0.15, f"kernel did not reach target under permuted dynamics ({kernel:.3f})"
        assert kernel < naive * 0.6, f"kernel ({kernel:.3f}) should beat naive model-free ({naive:.3f})"
