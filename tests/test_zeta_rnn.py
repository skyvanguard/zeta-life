# tests/test_zeta_rnn.py
import pytest
import torch
import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

def test_zeta_memory_layer_output_shape():
    """ZetaMemoryLayer should output same shape as input hidden state."""
    from zeta_life.core import ZetaMemoryLayer

    layer = ZetaMemoryLayer(hidden_size=64, M=15, sigma=0.1)
    h = torch.randn(8, 64)  # (batch, hidden)
    t = 5  # timestep

    m_t = layer(h, t)

    assert m_t.shape == h.shape, f"Expected {h.shape}, got {m_t.shape}"


def test_zeta_memory_oscillates_with_time():
    """Memory contribution should oscillate as t varies (zeta characteristic)."""
    from zeta_life.core import ZetaMemoryLayer

    layer = ZetaMemoryLayer(hidden_size=32, M=15, sigma=0.1)
    h = torch.ones(1, 32)  # constant input

    # Collect memory values at different timesteps
    values = [layer(h, t)[0, 0].item() for t in range(20)]

    # Should not be constant (oscillates)
    assert len(set(values)) > 1, "Memory should oscillate with time"

    # Should have sign changes (characteristic of zeta kernel)
    signs = [v > 0 for v in values]
    has_sign_change = any(signs[i] != signs[i+1] for i in range(len(signs)-1))
    assert has_sign_change, "Memory should have sign changes (zeta oscillation)"


def test_zeta_memory_bounded():
    """Memory contribution should be bounded by hidden state magnitude."""
    from zeta_life.core import ZetaMemoryLayer

    layer = ZetaMemoryLayer(hidden_size=64, M=15, sigma=0.1)
    h = torch.randn(16, 64)

    for t in range(50):
        m_t = layer(h, t)
        # m_t should be bounded relative to h
        assert m_t.abs().max() <= h.abs().max() * 2, f"Memory unbounded at t={t}"


def test_zeta_memory_learnable_phi():
    """When learnable_phi=True, phi should be a Parameter."""
    from zeta_life.core import ZetaMemoryLayer

    layer = ZetaMemoryLayer(hidden_size=32, M=10, learnable_phi=True)

    # phi should be a parameter
    param_names = [name for name, _ in layer.named_parameters()]
    assert 'phi' in param_names, "phi should be learnable parameter"


def test_zeta_lstm_cell_output_shape():
    """ZetaLSTMCell should output (h', c') with correct shapes."""
    from zeta_life.core import ZetaLSTMCell

    cell = ZetaLSTMCell(input_size=32, hidden_size=64, M=15)

    x = torch.randn(8, 32)  # (batch, input)
    h = torch.randn(8, 64)  # (batch, hidden)
    c = torch.randn(8, 64)  # (batch, cell)

    h_new, c_new = cell(x, (h, c), t=5)

    assert h_new.shape == (8, 64), f"h shape: {h_new.shape}"
    assert c_new.shape == (8, 64), f"c shape: {c_new.shape}"


def test_zeta_lstm_cell_initial_state():
    """ZetaLSTMCell should work with None initial state."""
    from zeta_life.core import ZetaLSTMCell

    cell = ZetaLSTMCell(input_size=16, hidden_size=32)
    x = torch.randn(4, 16)

    h, c = cell(x, None, t=0)

    assert h.shape == (4, 32)
    assert c.shape == (4, 32)


def test_zeta_lstm_sequence():
    """ZetaLSTM should process full sequences."""
    from zeta_life.core import ZetaLSTM

    lstm = ZetaLSTM(input_size=16, hidden_size=32, num_layers=1)

    x = torch.randn(8, 20, 16)  # (batch, seq_len, input)

    output, (h_n, c_n) = lstm(x)

    assert output.shape == (8, 20, 32), f"output: {output.shape}"
    assert h_n.shape == (1, 8, 32), f"h_n: {h_n.shape}"
    assert c_n.shape == (1, 8, 32), f"c_n: {c_n.shape}"


def test_zeta_lstm_multi_layer():
    """ZetaLSTM should support multiple layers."""
    from zeta_life.core import ZetaLSTM

    lstm = ZetaLSTM(input_size=16, hidden_size=32, num_layers=2)
    x = torch.randn(4, 10, 16)

    output, (h_n, c_n) = lstm(x)

    assert h_n.shape == (2, 4, 32), "Should have states for 2 layers"


def test_zeta_sequence_generator():
    """Generator should create sequences with zeta-correlated patterns."""
    from zeta_life.core import ZetaSequenceGenerator

    gen = ZetaSequenceGenerator(seq_length=100, feature_dim=8, M=10)
    x, y = gen.generate_batch(batch_size=16)

    assert x.shape == (16, 100, 8), f"x shape: {x.shape}"
    assert y.shape == (16, 100, 1), f"y shape: {y.shape}"


def test_zeta_sequence_has_long_range_dependency():
    """Sequences should have long-range temporal structure."""
    from zeta_life.core import ZetaSequenceGenerator
    import numpy as np

    gen = ZetaSequenceGenerator(seq_length=200, feature_dim=4, M=15)
    x, y = gen.generate_batch(batch_size=1)

    y_np = y[0, :, 0].numpy()

    # Compute autocorrelation at lag 50 (should be non-zero for zeta sequences)
    y_centered = y_np - y_np.mean()
    autocorr_lag50 = np.correlate(y_centered[:-50], y_centered[50:], mode='valid')[0]
    autocorr_lag0 = np.correlate(y_centered, y_centered, mode='valid')[0]

    normalized_autocorr = autocorr_lag50 / (autocorr_lag0 + 1e-8)

    # Zeta sequences should maintain some correlation at long lags
    assert abs(normalized_autocorr) > 0.01, "Should have long-range correlation"


def test_comparison_experiment():
    """Experiment should run and produce metrics."""
    from zeta_life.core import ZetaLSTMExperiment

    exp = ZetaLSTMExperiment(
        input_size=4,
        hidden_size=16,
        seq_length=50,
        M=5
    )

    results = exp.run(epochs=2, batch_size=8)

    assert 'vanilla_loss' in results
    assert 'zeta_loss' in results
    assert len(results['vanilla_loss']) == 2


def test_full_pipeline_integration():
    """Test complete pipeline from generation to training."""
    from zeta_life.core import ZetaLSTM, ZetaSequenceGenerator

    # Generate data
    gen = ZetaSequenceGenerator(seq_length=50, feature_dim=8, M=10)
    x, y = gen.generate_batch(batch_size=16)

    # Create model
    model = ZetaLSTM(input_size=8, hidden_size=32, M=10)
    output_layer = torch.nn.Linear(32, 1)

    # Forward pass
    output, (h_n, c_n) = model(x)
    pred = output_layer(output)

    # Compute loss
    loss = torch.nn.functional.mse_loss(pred, y)

    # Backward pass
    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print(f"Integration test passed. Loss: {loss.item():.4f}")
