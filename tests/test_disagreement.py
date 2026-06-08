"""Tests for the world-model disagreement ensemble (epistemic signal)."""

from __future__ import annotations

import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.world_model import WorldModel


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
