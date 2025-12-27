# tests/test_behavior_engine.py
"""Tests for BehaviorEngine: A<->B transformation algorithm."""
import pytest
import torch
from behavior_engine import BehaviorEngine


def test_bidirectional_influence():
    """A <-> B: influencia bidireccional."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)
    neighbor_states = torch.randn(8, 32)  # 8 vecinos

    influence_out, influence_in = engine.bidirectional_influence(
        cell_state, neighbor_states
    )

    assert influence_out.shape == (8,)  # A -> cada vecino
    assert influence_in.shape == ()     # suma de vecinos -> A


def test_self_similarity():
    """A = AAA*A: patron auto-similar."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)

    recursive = engine.self_similarity(cell_state)

    assert recursive.shape == cell_state.shape


def test_transformation_with_potential():
    """A^3 + V -> B^3 + A: transformacion con potencial vital."""
    engine = BehaviorEngine()

    local_cube = torch.randn(3, 3, 32)  # Celula + vecinos cercanos
    potential = 0.5
    alpha = 0.3  # Peso de continuidad

    new_cube = engine.transform_with_potential(local_cube, potential, alpha)

    assert new_cube.shape == local_cube.shape


def test_net_role():
    """B = AA* - A*A: rol neto."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)

    net = engine.net_role(cell_state)

    # Escalar indicando si es mas Fi o Mi
    assert net.shape == ()


def test_step_integration():
    """Test complete step integrates all components."""
    engine = BehaviorEngine()

    cell = torch.randn(32)
    neighbors = torch.randn(8, 32)
    potential = 0.5

    new_state, role_value = engine.step(cell, neighbors, potential)

    assert new_state.shape == cell.shape
    assert role_value.shape == ()


def test_self_similarity_preserves_norm():
    """Self-similarity should preserve approximate norm."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)
    original_norm = cell_state.norm()

    recursive = engine.self_similarity(cell_state)
    recursive_norm = recursive.norm()

    # Should be approximately the same magnitude
    assert torch.isclose(original_norm, recursive_norm, rtol=0.5)


def test_net_role_symmetry():
    """For real vectors, AA* = A*A, so net_role should be ~0."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)
    net = engine.net_role(cell_state)

    # For real-valued states, the difference should be close to 0
    # since AA* and A*A are equivalent for real numbers
    assert torch.abs(net) < 1.0  # Reasonable bound
