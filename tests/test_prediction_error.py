"""Tests for PredictionErrorEngine — multi-channel prediction error system.

Covers:
- Initialization with default 4 channels (perceptual, interoceptive, temporal, epistemic)
- precisions property returns positive values (via softplus on log_precisions)
- compute_errors(predictions, observations) returns dict with all channels,
  each having {raw, weighted, precision, magnitude}
- Zero error when prediction equals observation
- free_energy(errors) is zero for perfect prediction, increases with error
- recent_errors() returns tensor of shape (n_channels,), initially zeros, updates after compute
"""

import torch
import pytest

from zeta_life.kernel.prediction_error import PredictionErrorEngine


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestPredictionErrorEngineInit:
    """Verify construction with default and custom channel counts."""

    def test_default_channels(self):
        engine = PredictionErrorEngine()
        assert engine.channels == list(PredictionErrorEngine.CHANNEL_NAMES)

    def test_default_has_4_channels(self):
        engine = PredictionErrorEngine()
        assert len(engine.channels) == 4

    def test_channel_names(self):
        engine = PredictionErrorEngine()
        expected = ['perceptual', 'interoceptive', 'temporal', 'epistemic']
        assert engine.channels == expected

    def test_custom_n_channels(self):
        engine = PredictionErrorEngine(n_channels=6)
        assert len(engine.channels) == 6

    def test_custom_channels_named_generically(self):
        """Channels beyond 4 should get generic names."""
        engine = PredictionErrorEngine(n_channels=6)
        assert engine.channels[4] == 'channel_4'
        assert engine.channels[5] == 'channel_5'

    def test_fewer_channels(self):
        engine = PredictionErrorEngine(n_channels=2)
        assert len(engine.channels) == 2
        assert engine.channels == ['perceptual', 'interoceptive']

    def test_log_precisions_shape(self):
        engine = PredictionErrorEngine()
        assert engine.log_precisions.shape == (4,)

    def test_log_precisions_is_parameter(self):
        engine = PredictionErrorEngine()
        assert isinstance(engine.log_precisions, torch.nn.Parameter)

    def test_log_precisions_initialized_to_zeros(self):
        engine = PredictionErrorEngine()
        assert torch.all(engine.log_precisions.data == 0.0)

    def test_error_history_initially_empty(self):
        engine = PredictionErrorEngine()
        assert len(engine._error_history) == 0


# ---------------------------------------------------------------------------
# precisions property
# ---------------------------------------------------------------------------

class TestPrecisions:
    """precisions should be softplus(log_precisions), always positive."""

    def test_precisions_are_positive(self):
        engine = PredictionErrorEngine()
        precs = engine.precisions
        assert torch.all(precs > 0)

    def test_precisions_shape(self):
        engine = PredictionErrorEngine()
        precs = engine.precisions
        assert precs.shape == (4,)

    def test_precisions_shape_custom(self):
        engine = PredictionErrorEngine(n_channels=6)
        precs = engine.precisions
        assert precs.shape == (6,)

    def test_precisions_positive_with_negative_log_precisions(self):
        engine = PredictionErrorEngine()
        engine.log_precisions.data = torch.tensor([-5.0, -3.0, -1.0, -10.0])
        precs = engine.precisions
        assert torch.all(precs > 0)

    def test_precisions_increase_with_log_precisions(self):
        engine = PredictionErrorEngine()
        engine.log_precisions.data = torch.tensor([0.0, 1.0, 2.0, 3.0])
        precs = engine.precisions
        for i in range(3):
            assert precs[i + 1] > precs[i]

    def test_precisions_are_softplus(self):
        engine = PredictionErrorEngine()
        engine.log_precisions.data = torch.tensor([1.0, -1.0, 0.5, 2.0])
        expected = torch.nn.functional.softplus(engine.log_precisions.data)
        actual = engine.precisions
        assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# compute_errors
# ---------------------------------------------------------------------------

class TestComputeErrors:
    """compute_errors should return dict with all channels."""

    def test_returns_dict(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.randn(4) for ch in engine.channels}
        obs = {ch: torch.randn(4) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        assert isinstance(errors, dict)

    def test_contains_all_channels(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.randn(4) for ch in engine.channels}
        obs = {ch: torch.randn(4) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        for ch in engine.channels:
            assert ch in errors

    def test_each_channel_has_required_keys(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.randn(4) for ch in engine.channels}
        obs = {ch: torch.randn(4) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        required_keys = {'raw', 'weighted', 'precision', 'magnitude'}
        for ch in engine.channels:
            assert set(errors[ch].keys()) == required_keys

    def test_raw_is_prediction_minus_observation(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.tensor([1.0, 2.0, 3.0]) for ch in engine.channels}
        obs = {ch: torch.tensor([0.5, 1.0, 2.0]) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        expected_raw = torch.tensor([0.5, 1.0, 1.0])
        for ch in engine.channels:
            assert torch.allclose(errors[ch]['raw'], expected_raw)

    def test_weighted_is_precision_times_raw(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.tensor([1.0, 2.0]) for ch in engine.channels}
        obs = {ch: torch.tensor([0.0, 0.0]) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        precs = engine.precisions
        for i, ch in enumerate(engine.channels):
            expected_weighted = precs[i] * errors[ch]['raw']
            assert torch.allclose(errors[ch]['weighted'], expected_weighted)

    def test_precision_matches_engine_precisions(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.randn(4) for ch in engine.channels}
        obs = {ch: torch.randn(4) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        precs = engine.precisions
        for i, ch in enumerate(engine.channels):
            assert errors[ch]['precision'].item() == pytest.approx(
                precs[i].item(), abs=1e-6
            )

    def test_magnitude_is_norm_of_raw(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.tensor([3.0, 4.0]) for ch in engine.channels}
        obs = {ch: torch.tensor([0.0, 0.0]) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        for ch in engine.channels:
            assert errors[ch]['magnitude'].item() == pytest.approx(5.0, abs=1e-5)

    def test_zero_error_when_equal(self):
        engine = PredictionErrorEngine()
        data = {ch: torch.tensor([1.0, 2.0, 3.0]) for ch in engine.channels}
        errors = engine.compute_errors(data, data)
        for ch in engine.channels:
            assert torch.allclose(errors[ch]['raw'], torch.zeros(3))
            assert torch.allclose(errors[ch]['weighted'], torch.zeros(3))
            assert errors[ch]['magnitude'].item() == pytest.approx(0.0, abs=1e-7)

    def test_updates_error_history(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.randn(4) for ch in engine.channels}
        obs = {ch: torch.randn(4) for ch in engine.channels}
        engine.compute_errors(preds, obs)
        assert len(engine._error_history) == 1

    def test_error_history_accumulates(self):
        engine = PredictionErrorEngine()
        for _ in range(5):
            preds = {ch: torch.randn(4) for ch in engine.channels}
            obs = {ch: torch.randn(4) for ch in engine.channels}
            engine.compute_errors(preds, obs)
        assert len(engine._error_history) == 5


# ---------------------------------------------------------------------------
# free_energy
# ---------------------------------------------------------------------------

class TestFreeEnergy:
    """free_energy should be zero for perfect predictions, positive otherwise."""

    def test_zero_for_perfect_prediction(self):
        engine = PredictionErrorEngine()
        data = {ch: torch.tensor([1.0, 2.0, 3.0]) for ch in engine.channels}
        errors = engine.compute_errors(data, data)
        fe = engine.free_energy(errors)
        assert fe.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_imperfect_prediction(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.tensor([1.0, 0.0]) for ch in engine.channels}
        obs = {ch: torch.tensor([0.0, 0.0]) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        fe = engine.free_energy(errors)
        assert fe.item() > 0.0

    def test_increases_with_larger_errors(self):
        engine = PredictionErrorEngine()

        # Small error
        preds_s = {ch: torch.tensor([0.1, 0.0]) for ch in engine.channels}
        obs = {ch: torch.tensor([0.0, 0.0]) for ch in engine.channels}
        errors_small = engine.compute_errors(preds_s, obs)
        fe_small = engine.free_energy(errors_small)

        # Large error
        preds_l = {ch: torch.tensor([5.0, 5.0]) for ch in engine.channels}
        errors_large = engine.compute_errors(preds_l, obs)
        fe_large = engine.free_energy(errors_large)

        assert fe_large.item() > fe_small.item()

    def test_returns_tensor(self):
        engine = PredictionErrorEngine()
        data = {ch: torch.randn(4) for ch in engine.channels}
        errors = engine.compute_errors(data, data)
        fe = engine.free_energy(errors)
        assert isinstance(fe, torch.Tensor)

    def test_free_energy_is_scalar(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.randn(4) for ch in engine.channels}
        obs = {ch: torch.randn(4) for ch in engine.channels}
        errors = engine.compute_errors(preds, obs)
        fe = engine.free_energy(errors)
        assert fe.dim() == 0  # scalar tensor


# ---------------------------------------------------------------------------
# recent_errors
# ---------------------------------------------------------------------------

class TestRecentErrors:
    """recent_errors should return mean magnitudes from error history."""

    def test_initially_zeros(self):
        engine = PredictionErrorEngine()
        recent = engine.recent_errors()
        assert recent.shape == (4,)
        assert torch.all(recent == 0.0)

    def test_shape_custom_channels(self):
        engine = PredictionErrorEngine(n_channels=6)
        recent = engine.recent_errors()
        assert recent.shape == (6,)

    def test_updates_after_compute(self):
        engine = PredictionErrorEngine()
        preds = {ch: torch.tensor([3.0, 4.0]) for ch in engine.channels}
        obs = {ch: torch.tensor([0.0, 0.0]) for ch in engine.channels}
        engine.compute_errors(preds, obs)
        recent = engine.recent_errors()
        # All channels had magnitude 5.0
        assert torch.allclose(recent, torch.tensor([5.0, 5.0, 5.0, 5.0]))

    def test_averages_multiple_calls(self):
        engine = PredictionErrorEngine()
        # First: magnitude 5.0 for all channels
        preds1 = {ch: torch.tensor([3.0, 4.0]) for ch in engine.channels}
        obs = {ch: torch.tensor([0.0, 0.0]) for ch in engine.channels}
        engine.compute_errors(preds1, obs)
        # Second: magnitude 0.0 for all channels
        engine.compute_errors(obs, obs)
        recent = engine.recent_errors()
        # Average of 5.0 and 0.0 = 2.5
        assert torch.allclose(recent, torch.tensor([2.5, 2.5, 2.5, 2.5]))

    def test_returns_tensor(self):
        engine = PredictionErrorEngine()
        recent = engine.recent_errors()
        assert isinstance(recent, torch.Tensor)
