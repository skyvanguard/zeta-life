"""Tests for ConsciousKernel -- main orchestrator for the Conscious Kernel.

Covers:
- Init: all components initialized (world_model, self_model, error_engine,
  precision_controller, fast_memory, slow_memory, dream_engine, t=0)
- step(stimulus): returns StepResult with free_energy >= 0, increments t,
  errors dict has channel names
- Learning: free_energy decreases on repeated pattern
  (avg first 10 > avg last 10 after 30 steps)
- Dream integration: doesn't crash when dream_interval is reached
- Save/restore: save state, create fresh kernel, load, verify t matches
  and self_embedding matches
"""

import tempfile

import torch
import pytest

from zeta_life.kernel.conscious_kernel import ConsciousKernel, StepResult
from zeta_life.kernel.world_model import WorldModel
from zeta_life.kernel.self_model import SelfModel
from zeta_life.kernel.prediction_error import PredictionErrorEngine
from zeta_life.kernel.precision_controller import PrecisionController
from zeta_life.kernel.complementary_memory import FastMemory, SlowMemory
from zeta_life.kernel.dream_engine import DreamEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kernel(**kwargs) -> ConsciousKernel:
    """Create a ConsciousKernel with optional overrides."""
    defaults = dict(obs_dim=4, latent_dim=32, embed_dim=16,
                    reflect_interval=5, dream_interval=50, save_interval=100)
    defaults.update(kwargs)
    return ConsciousKernel(**defaults)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """ConsciousKernel.__init__ should create all components."""

    def test_world_model_initialized(self):
        ck = _make_kernel()
        assert isinstance(ck.world_model, WorldModel)

    def test_self_model_initialized(self):
        ck = _make_kernel()
        assert isinstance(ck.self_model, SelfModel)

    def test_error_engine_initialized(self):
        ck = _make_kernel()
        assert isinstance(ck.error_engine, PredictionErrorEngine)

    def test_precision_controller_initialized(self):
        ck = _make_kernel()
        assert isinstance(ck.precision_controller, PrecisionController)

    def test_fast_memory_initialized(self):
        ck = _make_kernel()
        assert isinstance(ck.fast_memory, FastMemory)

    def test_slow_memory_initialized(self):
        ck = _make_kernel()
        assert isinstance(ck.slow_memory, SlowMemory)

    def test_dream_engine_initialized(self):
        ck = _make_kernel()
        assert isinstance(ck.dream_engine, DreamEngine)

    def test_t_starts_at_zero(self):
        ck = _make_kernel()
        assert ck.t == 0


# ---------------------------------------------------------------------------
# step(stimulus)
# ---------------------------------------------------------------------------

class TestStep:
    """step(stimulus) should return a StepResult and advance internal state."""

    def test_returns_step_result(self):
        ck = _make_kernel()
        stimulus = torch.randn(4)
        result = ck.step(stimulus)
        assert isinstance(result, StepResult)

    def test_free_energy_non_negative(self):
        ck = _make_kernel()
        result = ck.step(torch.randn(4))
        assert result.free_energy >= 0.0

    def test_increments_t(self):
        ck = _make_kernel()
        assert ck.t == 0
        ck.step(torch.randn(4))
        assert ck.t == 1
        ck.step(torch.randn(4))
        assert ck.t == 2

    def test_errors_dict_has_channel_names(self):
        ck = _make_kernel()
        result = ck.step(torch.randn(4))
        assert isinstance(result.errors, dict)
        expected_channels = {'perceptual', 'interoceptive', 'temporal', 'epistemic'}
        assert set(result.errors.keys()) == expected_channels

    def test_errors_values_are_floats(self):
        ck = _make_kernel()
        result = ck.step(torch.randn(4))
        for ch, mag in result.errors.items():
            assert isinstance(mag, float), f"Channel {ch} magnitude is not float: {type(mag)}"

    def test_action_is_tensor(self):
        ck = _make_kernel()
        result = ck.step(torch.randn(4))
        assert isinstance(result.action, torch.Tensor)

    def test_action_has_correct_dim(self):
        ck = _make_kernel()
        result = ck.step(torch.randn(4))
        assert result.action.shape == (4,)

    def test_consciousness_is_float(self):
        ck = _make_kernel()
        result = ck.step(torch.randn(4))
        assert isinstance(result.consciousness, float)

    def test_reflected_flag_off_when_not_interval(self):
        ck = _make_kernel(reflect_interval=5)
        # Step 1 is not a multiple of 5
        result = ck.step(torch.randn(4))
        assert result.reflected is False

    def test_reflected_flag_on_at_interval(self):
        ck = _make_kernel(reflect_interval=5)
        # Run to step 5 (t=5 after 5 steps)
        for _ in range(4):
            ck.step(torch.randn(4))
        result = ck.step(torch.randn(4))
        assert result.reflected is True

    def test_multiple_steps_without_crash(self):
        ck = _make_kernel()
        for _ in range(20):
            result = ck.step(torch.randn(4))
        assert ck.t == 20


# ---------------------------------------------------------------------------
# Learning: free_energy decreases on repeated pattern
# ---------------------------------------------------------------------------

class TestLearning:
    """Free energy should decrease on repeated stimulus (avg first 10 > avg last 10)."""

    def test_free_energy_decreases_on_repeated_pattern(self):
        ck = _make_kernel(reflect_interval=100, dream_interval=1000)
        # Use a consistent repeating stimulus
        pattern = torch.tensor([1.0, 0.0, 0.5, 0.2])

        energies = []
        for _ in range(30):
            result = ck.step(pattern)
            energies.append(result.free_energy)

        avg_first_10 = sum(energies[:10]) / 10
        avg_last_10 = sum(energies[-10:]) / 10

        assert avg_first_10 > avg_last_10, (
            f"Free energy did not decrease: first_10={avg_first_10:.4f}, "
            f"last_10={avg_last_10:.4f}"
        )


# ---------------------------------------------------------------------------
# Dream integration
# ---------------------------------------------------------------------------

class TestDreamIntegration:
    """Dream cycle should run when dream_interval is reached without crash."""

    def test_dream_triggers_at_interval(self):
        ck = _make_kernel(dream_interval=10, reflect_interval=100)
        # Use high-surprise stimuli so fast_memory has items
        results = []
        for i in range(10):
            stimulus = torch.randn(4) * 3.0  # high variance for surprise
            result = ck.step(stimulus)
            results.append(result)

        # The 10th step should have triggered a dream
        assert results[-1].dreamed is True

    def test_dream_doesnt_crash_empty_memory(self):
        """Dream with empty fast memory should not crash."""
        ck = _make_kernel(dream_interval=5, reflect_interval=100)
        # Use very small stimuli to keep surprise below threshold
        for _ in range(5):
            ck.step(torch.zeros(4))

    def test_dream_doesnt_crash_with_memories(self):
        """Dream cycle should work when fast memory has episodes."""
        ck = _make_kernel(dream_interval=5, reflect_interval=100)
        # Lower the surprise threshold to ensure episodes get stored
        ck.fast_memory.surprise_threshold = 0.0
        for _ in range(5):
            ck.step(torch.randn(4))
        # Should not crash


# ---------------------------------------------------------------------------
# Save / Restore
# ---------------------------------------------------------------------------

class TestSaveRestore:
    """Save state, create fresh kernel, load, verify t and self_embedding match."""

    def test_save_and_load_t_matches(self):
        ck = _make_kernel()
        for _ in range(15):
            ck.step(torch.randn(4))
        assert ck.t == 15

        with tempfile.TemporaryDirectory() as tmp:
            ck.save(tmp, identity_name='test')

            ck2 = _make_kernel()
            assert ck2.t == 0
            ck2.load(tmp, identity_name='test')
            assert ck2.t == 15

    def test_save_and_load_self_embedding_matches(self):
        ck = _make_kernel()
        for _ in range(10):
            ck.step(torch.randn(4))

        original_embedding = ck.self_model.self_embedding.data.clone()

        with tempfile.TemporaryDirectory() as tmp:
            ck.save(tmp, identity_name='test')

            ck2 = _make_kernel()
            ck2.load(tmp, identity_name='test')

            restored_embedding = ck2.self_model.self_embedding.data

            # After load(), a wake-up reflection runs (depth=2, EMA decay=0.95)
            # which slightly shifts the embedding. We verify the loaded values
            # are close to the original (within EMA drift tolerance).
            assert torch.allclose(original_embedding, restored_embedding, atol=0.05), (
                "Self embedding should be close after save/load (within EMA wake-up drift)"
            )

    def test_save_and_load_world_model_state(self):
        ck = _make_kernel()
        for _ in range(10):
            ck.step(torch.randn(4))

        # Capture a world model parameter before save
        original_param = next(ck.world_model.parameters()).data.clone()

        with tempfile.TemporaryDirectory() as tmp:
            ck.save(tmp, identity_name='test')

            ck2 = _make_kernel()
            ck2.load(tmp, identity_name='test')

            restored_param = next(ck2.world_model.parameters()).data
            assert torch.allclose(original_param, restored_param, atol=1e-5)
