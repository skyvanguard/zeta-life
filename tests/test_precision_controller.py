"""Tests for PrecisionController — learned hyper-model for precision weights.

Covers:
- Creates with defaults (state_dim=4, n_channels=4, hidden_dim=32)
- Output always positive (via Softplus activation)
- Output shape = (n_channels,)
- Different states give different precisions
- High errors change precisions
- state_dict roundtrip (save/load preserves weights)
"""

import torch
import pytest

from zeta_life.kernel.precision_controller import PrecisionController


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestPrecisionControllerInit:
    """Verify construction with default and custom parameters."""

    def test_creates_with_defaults(self):
        pc = PrecisionController()
        assert pc is not None

    def test_default_state_dim(self):
        pc = PrecisionController()
        # First linear layer input = state_dim + n_channels = 4 + 4 = 8
        first_layer = pc.net[0]
        assert first_layer.in_features == 8

    def test_default_n_channels(self):
        pc = PrecisionController()
        # Last linear layer (before Softplus) output = n_channels = 4
        linear_layers = [m for m in pc.net if isinstance(m, torch.nn.Linear)]
        assert linear_layers[-1].out_features == 4

    def test_default_hidden_dim(self):
        pc = PrecisionController()
        first_layer = pc.net[0]
        assert first_layer.out_features == 32

    def test_custom_state_dim(self):
        pc = PrecisionController(state_dim=8)
        first_layer = pc.net[0]
        # input = state_dim + n_channels = 8 + 4 = 12
        assert first_layer.in_features == 12

    def test_custom_n_channels(self):
        pc = PrecisionController(n_channels=6)
        linear_layers = [m for m in pc.net if isinstance(m, torch.nn.Linear)]
        assert linear_layers[-1].out_features == 6

    def test_custom_hidden_dim(self):
        pc = PrecisionController(hidden_dim=64)
        first_layer = pc.net[0]
        assert first_layer.out_features == 64

    def test_is_nn_module(self):
        pc = PrecisionController()
        assert isinstance(pc, torch.nn.Module)

    def test_has_net_attribute(self):
        pc = PrecisionController()
        assert hasattr(pc, 'net')

    def test_net_is_sequential(self):
        pc = PrecisionController()
        assert isinstance(pc.net, torch.nn.Sequential)


# ---------------------------------------------------------------------------
# Output always positive (Softplus)
# ---------------------------------------------------------------------------

class TestOutputAlwaysPositive:
    """Output must always be strictly positive due to Softplus."""

    def test_positive_with_zeros(self):
        pc = PrecisionController()
        state = torch.zeros(4)
        errors = torch.zeros(4)
        output = pc(state, errors)
        assert torch.all(output > 0)

    def test_positive_with_random_inputs(self):
        torch.manual_seed(42)
        pc = PrecisionController()
        state = torch.randn(4)
        errors = torch.randn(4)
        output = pc(state, errors)
        assert torch.all(output > 0)

    def test_positive_with_negative_inputs(self):
        pc = PrecisionController()
        state = torch.tensor([-10.0, -5.0, -1.0, -0.1])
        errors = torch.tensor([-10.0, -5.0, -1.0, -0.1])
        output = pc(state, errors)
        assert torch.all(output > 0)

    def test_positive_with_large_inputs(self):
        pc = PrecisionController()
        state = torch.tensor([100.0, 200.0, 300.0, 400.0])
        errors = torch.tensor([100.0, 200.0, 300.0, 400.0])
        output = pc(state, errors)
        assert torch.all(output > 0)

    def test_positive_many_random_trials(self):
        """Fuzz test: 50 random inputs all produce positive output."""
        torch.manual_seed(123)
        pc = PrecisionController()
        for _ in range(50):
            state = torch.randn(4)
            errors = torch.randn(4)
            output = pc(state, errors)
            assert torch.all(output > 0)


# ---------------------------------------------------------------------------
# Output shape = (n_channels,)
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Output must have shape (n_channels,)."""

    def test_default_shape(self):
        pc = PrecisionController()
        state = torch.zeros(4)
        errors = torch.zeros(4)
        output = pc(state, errors)
        assert output.shape == (4,)

    def test_custom_n_channels_shape(self):
        pc = PrecisionController(n_channels=6)
        state = torch.zeros(4)
        errors = torch.zeros(6)
        output = pc(state, errors)
        assert output.shape == (6,)

    def test_custom_state_dim_shape(self):
        pc = PrecisionController(state_dim=8, n_channels=4)
        state = torch.zeros(8)
        errors = torch.zeros(4)
        output = pc(state, errors)
        assert output.shape == (4,)

    def test_output_is_1d(self):
        pc = PrecisionController()
        state = torch.zeros(4)
        errors = torch.zeros(4)
        output = pc(state, errors)
        assert output.dim() == 1

    def test_output_is_tensor(self):
        pc = PrecisionController()
        state = torch.zeros(4)
        errors = torch.zeros(4)
        output = pc(state, errors)
        assert isinstance(output, torch.Tensor)


# ---------------------------------------------------------------------------
# Different states give different precisions
# ---------------------------------------------------------------------------

class TestDifferentStatesGiveDifferentPrecisions:
    """Different input states should produce different precision outputs."""

    def test_different_states_different_output(self):
        torch.manual_seed(7)
        pc = PrecisionController()
        errors = torch.zeros(4)
        state_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        state_b = torch.tensor([0.0, 0.0, 0.0, 1.0])
        output_a = pc(state_a, errors)
        output_b = pc(state_b, errors)
        assert not torch.allclose(output_a, output_b)

    def test_different_errors_different_output(self):
        torch.manual_seed(7)
        pc = PrecisionController()
        state = torch.zeros(4)
        errors_a = torch.tensor([0.0, 0.0, 0.0, 0.0])
        errors_b = torch.tensor([5.0, 5.0, 5.0, 5.0])
        output_a = pc(state, errors_a)
        output_b = pc(state, errors_b)
        assert not torch.allclose(output_a, output_b)

    def test_sensitivity_to_state_variation(self):
        """Small changes in state should produce some change in output."""
        torch.manual_seed(7)
        pc = PrecisionController()
        errors = torch.zeros(4)
        state_base = torch.tensor([0.5, 0.5, 0.5, 0.5])
        state_perturbed = torch.tensor([0.5, 0.5, 0.5, 1.5])
        output_base = pc(state_base, errors)
        output_perturbed = pc(state_perturbed, errors)
        diff = (output_base - output_perturbed).abs().sum().item()
        assert diff > 0.0


# ---------------------------------------------------------------------------
# High errors change precisions
# ---------------------------------------------------------------------------

class TestHighErrorsChangePrecisions:
    """Model should respond differently to high vs low error magnitudes."""

    def test_zero_vs_high_errors(self):
        torch.manual_seed(42)
        pc = PrecisionController()
        state = torch.tensor([0.5, 0.5, 0.5, 0.5])
        low_errors = torch.zeros(4)
        high_errors = torch.tensor([10.0, 10.0, 10.0, 10.0])
        output_low = pc(state, low_errors)
        output_high = pc(state, high_errors)
        assert not torch.allclose(output_low, output_high)

    def test_increasing_errors_change_output(self):
        """Monotonically increasing error magnitudes should change output."""
        torch.manual_seed(42)
        pc = PrecisionController()
        state = torch.tensor([0.5, 0.5, 0.5, 0.5])
        outputs = []
        for scale in [0.0, 1.0, 5.0, 10.0]:
            errors = torch.full((4,), scale)
            outputs.append(pc(state, errors))
        # At least some pairs should differ
        differences = 0
        for i in range(len(outputs) - 1):
            if not torch.allclose(outputs[i], outputs[i + 1]):
                differences += 1
        assert differences >= 2

    def test_gradients_flow_through_errors(self):
        """Ensure the network can be trained to adjust to errors."""
        pc = PrecisionController()
        state = torch.zeros(4)
        errors = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False)
        output = pc(state, errors)
        loss = output.sum()
        loss.backward()
        # Check that gradients exist in the network parameters
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in pc.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# state_dict roundtrip
# ---------------------------------------------------------------------------

class TestStateDictRoundtrip:
    """Saving and loading state_dict should preserve behavior exactly."""

    def test_state_dict_keys_exist(self):
        pc = PrecisionController()
        sd = pc.state_dict()
        assert len(sd) > 0

    def test_state_dict_contains_net_weights(self):
        pc = PrecisionController()
        sd = pc.state_dict()
        key_prefixes = {k.split('.')[0] for k in sd}
        assert 'net' in key_prefixes

    def test_roundtrip_preserves_output(self):
        torch.manual_seed(99)
        pc1 = PrecisionController()
        state = torch.randn(4)
        errors = torch.randn(4)
        output1 = pc1(state, errors)

        # Save and load into a new instance
        sd = pc1.state_dict()
        pc2 = PrecisionController()
        pc2.load_state_dict(sd)
        output2 = pc2(state, errors)

        assert torch.allclose(output1, output2, atol=1e-7)

    def test_roundtrip_with_custom_dims(self):
        torch.manual_seed(99)
        pc1 = PrecisionController(state_dim=8, n_channels=6, hidden_dim=64)
        state = torch.randn(8)
        errors = torch.randn(6)
        output1 = pc1(state, errors)

        sd = pc1.state_dict()
        pc2 = PrecisionController(state_dim=8, n_channels=6, hidden_dim=64)
        pc2.load_state_dict(sd)
        output2 = pc2(state, errors)

        assert torch.allclose(output1, output2, atol=1e-7)

    def test_fresh_instance_differs_from_loaded(self):
        """A freshly initialized model should differ from a loaded one."""
        torch.manual_seed(99)
        pc1 = PrecisionController()

        sd = pc1.state_dict()

        torch.manual_seed(77)  # Different seed
        pc_fresh = PrecisionController()

        state = torch.randn(4)
        errors = torch.randn(4)
        output_original = pc1(state, errors)
        output_fresh = pc_fresh(state, errors)

        # They should (almost certainly) differ
        assert not torch.allclose(output_original, output_fresh)

        # But after loading, they should match
        pc_fresh.load_state_dict(sd)
        output_loaded = pc_fresh(state, errors)
        assert torch.allclose(output_original, output_loaded, atol=1e-7)
