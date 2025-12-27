# tests/test_zeta_rnn.py
import pytest
import torch
import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

def test_zeta_memory_layer_output_shape():
    """ZetaMemoryLayer should output same shape as input hidden state."""
    from zeta_rnn import ZetaMemoryLayer

    layer = ZetaMemoryLayer(hidden_size=64, M=15, sigma=0.1)
    h = torch.randn(8, 64)  # (batch, hidden)
    t = 5  # timestep

    m_t = layer(h, t)

    assert m_t.shape == h.shape, f"Expected {h.shape}, got {m_t.shape}"
