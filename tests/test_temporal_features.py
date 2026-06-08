"""Tests for temporal feature banks and their wiring into the control path.

Covers:
- OscillatorBank: shape, normalisation, determinism, factory parameter-matching.
- WorldModel: temporal augmentation is backward compatible (temporal_dim=0) and
  changes the GRU input size / requires a feature when temporal_dim>0.
- ConsciousKernel: accepts a bank, stays a valid step loop, and the learned
  bank's frequencies actually move under training.
"""

from __future__ import annotations

import pytest
import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.world_model import WorldModel
from zeta_life.kernel.temporal_features import (
    OscillatorBank,
    generate_spacing_frequencies,
    spacing_stats,
)


# ---------------------------------------------------------------------------
# OscillatorBank
# ---------------------------------------------------------------------------

class TestOscillatorBank:
    def test_dim_is_twice_n_freqs(self):
        bank = OscillatorBank.zeta(M=15)
        assert bank.dim == 30

    def test_output_shape(self):
        bank = OscillatorBank.zeta(M=10)
        feat = bank(5.0)
        assert feat.shape == (20,)

    def test_output_is_l2_normalised(self):
        bank = OscillatorBank.zeta(M=15)
        for t in (0.0, 1.0, 7.0, 42.0):
            feat = bank(t)
            assert torch.linalg.vector_norm(feat).item() == pytest.approx(1.0, abs=1e-5)

    def test_deterministic_for_same_t(self):
        bank = OscillatorBank.zeta(M=15)
        assert torch.allclose(bank(13.0), bank(13.0))

    def test_zeta_and_random_share_dim_and_envelope(self):
        # Parameter-matched: identical output dimensionality.
        z = OscillatorBank.zeta(M=15)
        r = OscillatorBank.random(M=15, seed=1)
        assert z.dim == r.dim

    def test_zeta_uses_actual_zeros(self):
        bank = OscillatorBank.zeta(M=3)
        assert bank.frequencies[0].item() == pytest.approx(14.134725, abs=1e-3)

    def test_random_differs_from_zeta(self):
        z = OscillatorBank.zeta(M=15)
        r = OscillatorBank.random(M=15, seed=1)
        assert not torch.allclose(z.frequencies, r.frequencies)

    def test_frozen_banks_have_no_trainable_params(self):
        for bank in (OscillatorBank.zeta(M=15), OscillatorBank.random(M=15)):
            assert [p for p in bank.parameters() if p.requires_grad] == []

    def test_learned_bank_has_trainable_frequencies(self):
        bank = OscillatorBank.learned(M=15, seed=2)
        trainable = [p for p in bank.parameters() if p.requires_grad]
        assert len(trainable) == 1
        assert trainable[0].shape == (15,)


# ---------------------------------------------------------------------------
# Spacing-statistics factories (the decisive zeta-vs-baseline test)
# ---------------------------------------------------------------------------

class TestSpacingStatistics:
    RANGE = (5.0, 80.0)

    def test_all_statistics_share_range_and_count(self):
        for stat in ("zeta", "gue", "poisson", "uniform"):
            f = generate_spacing_frequencies(stat, M=40, freq_range=self.RANGE, seed=0)
            assert f.shape == (40,)
            # Sorted endpoints land on the shared band (same range & density).
            s = torch.sort(f).values
            assert s[0].item() == pytest.approx(self.RANGE[0], abs=1e-3)
            assert s[-1].item() == pytest.approx(self.RANGE[1], abs=1e-3)

    def test_uniform_has_near_zero_gap_variance(self):
        f = generate_spacing_frequencies("uniform", M=40, freq_range=self.RANGE)
        assert spacing_stats(f)["gap_cv"] < 1e-4

    def test_poisson_clusters_more_than_gue(self):
        # Level repulsion (GUE) suppresses small gaps; Poisson does not.
        gue = spacing_stats(generate_spacing_frequencies("gue", M=60, seed=1))
        poi = spacing_stats(generate_spacing_frequencies("poisson", M=60, seed=1))
        assert poi["frac_small"] > gue["frac_small"]

    def test_zeta_repels_like_gue_not_poisson(self):
        # The zeta zeros' gaps should look GUE-like (few small gaps), unlike Poisson.
        zeta = spacing_stats(generate_spacing_frequencies("zeta", M=60))
        poi = spacing_stats(generate_spacing_frequencies("poisson", M=60, seed=0))
        assert zeta["frac_small"] < poi["frac_small"]

    def test_by_spacing_builds_flat_bank(self):
        bank = OscillatorBank.by_spacing("gue", M=40, sigma=0.0, seed=0)
        assert bank.dim == 80
        # sigma=0 -> flat amplitudes -> unit-norm feature is well defined
        feat = bank(3.0)
        assert feat.shape == (80,)

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError):
            generate_spacing_frequencies("notastat", M=10)


# ---------------------------------------------------------------------------
# Principled fixed bases (the recommended replacements for zeta)
# ---------------------------------------------------------------------------

class TestPrincipledBases:
    def test_fourier_is_equispaced(self):
        bank = OscillatorBank.fourier(M=40, freq_range=(5.0, 80.0))
        assert bank.dim == 80
        # Equispaced lattice -> zero gap variance.
        assert bank.spacing_stats()["gap_cv"] < 1e-4
        s = torch.sort(bank.frequencies).values
        assert s[0].item() == pytest.approx(5.0, abs=1e-4)
        assert s[-1].item() == pytest.approx(80.0, abs=1e-4)

    def test_log_spaced_has_constant_ratio(self):
        bank = OscillatorBank.log_spaced(M=20, freq_range=(1.0, 64.0))
        f = torch.sort(bank.frequencies).values
        ratios = f[1:] / f[:-1]
        # Geometric spacing -> constant successive ratio.
        assert torch.allclose(ratios, ratios.mean().expand_as(ratios), rtol=1e-4)

    def test_log_spaced_rejects_nonpositive_lo(self):
        with pytest.raises(ValueError):
            OscillatorBank.log_spaced(M=10, freq_range=(0.0, 80.0))

    def test_fixed_bases_drive_the_kernel(self):
        for bank in (OscillatorBank.fourier(M=20), OscillatorBank.log_spaced(M=20)):
            ck = ConsciousKernel(temporal_features=bank)
            assert ck.world_model.temporal_dim == 40
            for _ in range(5):
                result = ck.step(torch.randn(4))
            assert result.action.shape == (4,)


# ---------------------------------------------------------------------------
# WorldModel temporal augmentation
# ---------------------------------------------------------------------------

class TestWorldModelTemporal:
    def test_default_is_action_only(self):
        wm = WorldModel()
        assert wm.temporal_dim == 0
        # predict works with no temporal feature (original behaviour)
        pred, latent = wm.predict(torch.zeros(4))
        assert pred.shape == (4,)

    def test_gru_input_size_grows_with_temporal_dim(self):
        wm = WorldModel(temporal_dim=30)
        assert wm.transition.input_size == 4 + 30

    def test_predict_requires_feature_when_temporal(self):
        wm = WorldModel(temporal_dim=8)
        try:
            wm.predict(torch.zeros(4))
            assert False, "expected ValueError when temporal_feat missing"
        except ValueError:
            pass

    def test_predict_with_feature_runs(self):
        wm = WorldModel(temporal_dim=8)
        pred, latent = wm.predict(torch.zeros(4), torch.randn(8))
        assert pred.shape == (4,)

    def test_imagine_broadcasts_single_feature(self):
        wm = WorldModel(temporal_dim=8)
        feat = torch.randn(8)
        preds = wm.imagine([torch.zeros(4), torch.zeros(4)], feat)
        assert len(preds) == 2


# ---------------------------------------------------------------------------
# ConsciousKernel wiring
# ---------------------------------------------------------------------------

class TestKernelTemporal:
    def test_default_kernel_has_no_temporal_features(self):
        ck = ConsciousKernel()
        assert ck.temporal_features is None
        assert ck.world_model.temporal_dim == 0

    def test_kernel_with_zeta_bank_steps(self):
        bank = OscillatorBank.zeta(M=15)
        ck = ConsciousKernel(temporal_features=bank)
        assert ck.world_model.temporal_dim == 30
        for _ in range(10):
            result = ck.step(torch.randn(4))
        assert ck.t == 10
        assert result.action.shape == (4,)

    def test_kernel_with_efe_and_temporal_features(self):
        bank = OscillatorBank.zeta(M=15)
        ck = ConsciousKernel(
            action_mode="efe",
            preference=torch.tensor([0.7, 0.1, 0.1, 0.1]),
            temporal_features=bank,
        )
        for _ in range(10):
            result = ck.step(torch.rand(4))
        assert result.action.shape == (4,)

    def test_learned_frequencies_move_under_training(self):
        torch.manual_seed(0)
        bank = OscillatorBank.learned(M=15, seed=3)
        before = bank.frequencies.detach().clone()
        ck = ConsciousKernel(temporal_features=bank)
        # A time-varying stimulus gives the prior loss a gradient w.r.t. the
        # temporal code, so the learnable frequencies should drift.
        for t in range(60):
            stim = torch.tensor([float((t % 4) == i) for i in range(4)])
            ck.step(stim)
        after = bank.frequencies.detach()
        assert not torch.allclose(before, after), \
            "learned frequencies did not change under training"
