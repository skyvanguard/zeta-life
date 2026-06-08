"""Tests for the world-model disagreement ensemble (epistemic signal)."""

from __future__ import annotations

import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.world_model import WorldModel
from zeta_life.kernel.dynamics_ensemble import DynamicsEnsemble


class TestWorldModelDisagreement:
    def test_no_heads_zero_disagreement(self):
        wm = WorldModel()  # disagreement_heads=0
        assert wm.disagreement_heads == 0
        assert wm.disagreement(torch.tensor([1.0, 0, 0, 0])) == 0.0

    def test_ensemble_constructed(self):
        wm = WorldModel(disagreement_heads=5)
        assert len(wm.heads) == 5
        assert hasattr(wm, "head_optimizer")

    def test_disagreement_positive_before_training(self):
        torch.manual_seed(0)
        wm = WorldModel(disagreement_heads=5)
        wm.predict(torch.tensor([1.0, 0, 0, 0]))
        assert wm.disagreement(torch.tensor([1.0, 0, 0, 0])) > 0.0

    def test_disagreement_decreases_after_training(self):
        torch.manual_seed(0)
        wm = WorldModel(disagreement_heads=5)
        a = torch.tensor([1.0, 0, 0, 0])
        pattern = torch.tensor([0.5, 0.2, 0.2, 0.1])
        wm.predict(a)
        wm.observe(pattern)
        before = wm.disagreement(a)
        for _ in range(100):
            wm.predict(a)
            wm.observe(pattern)
        after = wm.disagreement(a)
        assert after < before, f"disagreement did not drop with training ({before:.4f} -> {after:.4f})"

    def test_default_predictor_unchanged(self):
        # disagreement_heads=0 keeps the single-predictor path (param naming etc.)
        wm = WorldModel()
        assert not hasattr(wm, "heads")
        pred, latent = wm.predict(torch.zeros(4))
        assert pred.shape == (4,)


class TestDynamicsEnsemble:
    """Independent one-step dynamics models (Plan2Explore-faithful signal)."""

    def test_disagreement_is_nonneg_float(self):
        ens = DynamicsEnsemble(latent_dim=32, action_in_dim=4, obs_dim=4, n_members=5)
        d = ens.disagreement(torch.zeros(32), torch.tensor([1.0, 0, 0, 0]))
        assert isinstance(d, float) and d >= 0.0

    def test_train_step_updates_members(self):
        torch.manual_seed(0)
        ens = DynamicsEnsemble(latent_dim=8, action_in_dim=4, obs_dim=4, n_members=5)
        before = [p.detach().clone() for p in ens.members.parameters()]
        for _ in range(20):
            ens.train_step(torch.randn(8), torch.tensor([1.0, 0, 0, 0]),
                           torch.tensor([0.5, 0.2, 0.2, 0.1]))
        after = list(ens.members.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_worldmodel_ensemble_higher_disagreement_in_novel_region(self):
        """After training on one action, a novel action should disagree more."""
        torch.manual_seed(0)
        wm = WorldModel(dynamics_ensemble=5)
        home = torch.tensor([0.7, 0.1, 0.1, 0.1])
        target = torch.tensor([0.7, 0.1, 0.1, 0.1])
        for _ in range(150):
            wm.predict(home)
            wm.observe(target)
        d_home = wm.disagreement(home)
        d_novel = wm.disagreement(torch.tensor([0.1, 0.1, 0.1, 0.7]))
        assert d_novel > d_home

    def test_worldmodel_default_has_no_ensemble(self):
        wm = WorldModel()
        assert wm.ensemble is None

    def test_kernel_dynamics_ensemble_runs(self):
        ck = ConsciousKernel(
            action_mode="efe", preference=torch.tensor([0.7, 0.1, 0.1, 0.1]),
            dynamics_ensemble=5, efe_epistemic_mode="disagreement",
            efe_epistemic_weight=100.0, efe_n_samples=16, efe_obs_norm="l1",
        )
        result = None
        for _ in range(10):
            result = ck.step(torch.rand(4))
        assert result.action.shape == (4,)


class TestKernelEpistemic:
    def test_default_epistemic_mode_entropy(self):
        ck = ConsciousKernel()
        assert ck.efe_epistemic_mode == "entropy"
        assert ck.world_model.disagreement_heads == 0

    def test_disagreement_mode_steps(self):
        ck = ConsciousKernel(
            action_mode="efe", preference=torch.tensor([0.7, 0.1, 0.1, 0.1]),
            wm_disagreement_heads=5, efe_epistemic_mode="disagreement",
            efe_epistemic_weight=10.0, efe_n_samples=16, efe_obs_norm="l1",
        )
        result = None
        for _ in range(10):
            result = ck.step(torch.rand(4))
        assert result.action.shape == (4,)
        assert abs(float(result.action.sum()) - 1.0) < 1e-4
