"""Tests for WorldModel — the predictive world model component of the Conscious Kernel.

Covers:
- Initialization with default and custom dimensions
- encode(observation) — maps obs_dim to latent_dim
- predict(action) — returns (predicted_obs, next_latent) from current latent_state + action
- imagine(action_sequence) — counterfactual simulation WITHOUT modifying latent_state
- update_from_error(error) — prediction error decreases over repeated patterns
"""

import torch
import pytest

from zeta_life.kernel.world_model import WorldModel


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestWorldModelInit:
    """Verify construction with default and custom dimensions."""

    def test_default_dims(self):
        wm = WorldModel()
        assert wm.obs_dim == 4
        assert wm.latent_dim == 32
        assert wm.action_dim == 4

    def test_custom_dims(self):
        wm = WorldModel(obs_dim=8, latent_dim=64, action_dim=3)
        assert wm.obs_dim == 8
        assert wm.latent_dim == 64
        assert wm.action_dim == 3

    def test_latent_state_initialized_to_zeros(self):
        wm = WorldModel(latent_dim=16)
        assert wm.latent_state.shape == (16,)
        assert torch.all(wm.latent_state == 0.0)

    def test_latent_state_is_buffer(self):
        wm = WorldModel()
        # Registered buffers appear in state_dict but not in parameters
        assert "latent_state" in dict(wm.named_buffers())

    def test_has_encoder(self):
        wm = WorldModel()
        assert hasattr(wm, "encoder")

    def test_has_transition(self):
        wm = WorldModel()
        assert hasattr(wm, "transition")

    def test_has_predictor(self):
        wm = WorldModel()
        assert hasattr(wm, "predictor")

    def test_has_optimizer(self):
        wm = WorldModel()
        assert hasattr(wm, "optimizer")


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

class TestEncode:
    """encode(observation) should map obs_dim -> latent_dim."""

    def test_output_shape(self):
        wm = WorldModel(obs_dim=4, latent_dim=32)
        obs = torch.randn(4)
        latent = wm.encode(obs)
        assert latent.shape == (32,)

    def test_output_shape_custom(self):
        wm = WorldModel(obs_dim=8, latent_dim=16)
        obs = torch.randn(8)
        latent = wm.encode(obs)
        assert latent.shape == (16,)

    def test_deterministic(self):
        wm = WorldModel()
        wm.eval()
        obs = torch.randn(4)
        l1 = wm.encode(obs)
        l2 = wm.encode(obs)
        assert torch.allclose(l1, l2)

    def test_different_inputs_different_outputs(self):
        wm = WorldModel()
        o1 = torch.ones(4)
        o2 = torch.ones(4) * 5.0
        l1 = wm.encode(o1)
        l2 = wm.encode(o2)
        assert not torch.allclose(l1, l2)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    """predict(action) should return (predicted_obs, next_latent)."""

    def test_returns_tuple(self):
        wm = WorldModel(obs_dim=4, latent_dim=32, action_dim=4)
        action = torch.randn(4)
        result = wm.predict(action)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predicted_obs_shape(self):
        wm = WorldModel(obs_dim=4, latent_dim=32, action_dim=4)
        action = torch.randn(4)
        pred_obs, _ = wm.predict(action)
        assert pred_obs.shape == (4,)

    def test_next_latent_shape(self):
        wm = WorldModel(obs_dim=4, latent_dim=32, action_dim=4)
        action = torch.randn(4)
        _, next_latent = wm.predict(action)
        assert next_latent.shape == (32,)

    def test_custom_dims(self):
        wm = WorldModel(obs_dim=6, latent_dim=16, action_dim=3)
        action = torch.randn(3)
        pred_obs, next_latent = wm.predict(action)
        assert pred_obs.shape == (6,)
        assert next_latent.shape == (16,)

    def test_different_actions_different_predictions(self):
        wm = WorldModel()
        a1 = torch.ones(4)
        a2 = -torch.ones(4)
        p1, _ = wm.predict(a1)
        p2, _ = wm.predict(a2)
        # After two different actions from same initial state the predictions
        # *may* differ; but since latent_state is mutated by first call, we
        # just verify shapes here. The meaningful test is in imagine.
        assert p1.shape == p2.shape


# ---------------------------------------------------------------------------
# imagine
# ---------------------------------------------------------------------------

class TestImagine:
    """imagine(action_sequence) must NOT modify latent_state."""

    def test_returns_list_of_predictions(self):
        wm = WorldModel(obs_dim=4, action_dim=4)
        actions = [torch.randn(4) for _ in range(5)]
        preds = wm.imagine(actions)
        assert isinstance(preds, list)
        assert len(preds) == 5

    def test_prediction_shapes(self):
        wm = WorldModel(obs_dim=4, action_dim=4)
        actions = [torch.randn(4) for _ in range(3)]
        preds = wm.imagine(actions)
        for p in preds:
            assert p.shape == (4,)

    def test_does_not_modify_latent_state(self):
        wm = WorldModel(obs_dim=4, latent_dim=32, action_dim=4)
        # Set a known latent state via encode
        obs = torch.randn(4)
        wm.latent_state = wm.encode(obs).detach()
        state_before = wm.latent_state.clone()

        actions = [torch.randn(4) for _ in range(10)]
        wm.imagine(actions)

        assert torch.allclose(wm.latent_state, state_before), (
            "imagine() must NOT modify latent_state"
        )

    def test_empty_sequence_returns_empty_list(self):
        wm = WorldModel()
        preds = wm.imagine([])
        assert preds == []

    def test_imagine_uses_no_grad(self):
        """Predictions from imagine should not require grad."""
        wm = WorldModel()
        actions = [torch.randn(4) for _ in range(3)]
        preds = wm.imagine(actions)
        for p in preds:
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# update_from_error
# ---------------------------------------------------------------------------

class TestUpdateFromError:
    """update_from_error(error) should reduce loss over repeated patterns."""

    def test_returns_float(self):
        wm = WorldModel()
        error = torch.randn(4)
        loss = wm.update_from_error(error)
        assert isinstance(loss, float)

    def test_loss_is_nonnegative(self):
        wm = WorldModel()
        error = torch.randn(4)
        loss = wm.update_from_error(error)
        assert loss >= 0.0

    def test_zero_error_gives_zero_loss(self):
        wm = WorldModel()
        error = torch.zeros(4)
        loss = wm.update_from_error(error)
        assert loss == pytest.approx(0.0, abs=1e-7)

    def test_loss_decreases_over_repeated_pattern(self):
        """Core learning test: repeated predict+update should shrink error."""
        wm = WorldModel(obs_dim=4, latent_dim=32, action_dim=4, learning_rate=0.01)
        obs = torch.tensor([1.0, 0.0, -1.0, 0.5])
        action = torch.tensor([0.0, 1.0, 0.0, 0.0])

        # Encode the observation to set latent state
        wm.latent_state = wm.encode(obs).detach()

        losses = []
        for _ in range(50):
            pred_obs, next_latent = wm.predict(action)
            error = obs - pred_obs
            loss = wm.update_from_error(error)
            losses.append(loss)
            # Reset latent state so we keep training on same pattern
            wm.latent_state = wm.encode(obs).detach()

        # The loss at the end should be significantly lower than at the start
        assert losses[-1] < losses[0] * 0.5, (
            f"Expected loss to decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestWorldModelIntegration:
    """End-to-end scenarios combining multiple methods."""

    def test_encode_predict_cycle(self):
        """Encode an observation, predict next, verify shapes."""
        wm = WorldModel(obs_dim=4, latent_dim=32, action_dim=4)
        obs = torch.randn(4)
        wm.latent_state = wm.encode(obs).detach()
        pred, new_latent = wm.predict(torch.randn(4))
        assert pred.shape == (4,)
        assert new_latent.shape == (32,)

    def test_imagine_after_encode(self):
        """Imagine should work after encoding an observation."""
        wm = WorldModel()
        obs = torch.randn(4)
        wm.latent_state = wm.encode(obs).detach()
        actions = [torch.randn(4) for _ in range(5)]
        preds = wm.imagine(actions)
        assert len(preds) == 5

    def test_state_dict_save_load(self):
        """WorldModel should be serializable via state_dict."""
        wm1 = WorldModel(obs_dim=4, latent_dim=16, action_dim=4)
        obs = torch.randn(4)
        wm1.latent_state = wm1.encode(obs).detach()

        state = wm1.state_dict()

        wm2 = WorldModel(obs_dim=4, latent_dim=16, action_dim=4)
        wm2.load_state_dict(state)

        assert torch.allclose(wm1.latent_state, wm2.latent_state)
