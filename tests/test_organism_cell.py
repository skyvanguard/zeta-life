# tests/test_organism_cell.py
"""Tests for OrganismCell with gated zeta memory."""
import pytest
import torch
from organism_cell import OrganismCell


def test_cell_creation():
    """Celula se crea con componentes NCA y Resonant."""
    cell = OrganismCell(state_dim=32, hidden_dim=64)
    assert hasattr(cell, 'resonant')
    assert hasattr(cell, 'role_detector')


def test_perception():
    """Celula percibe su entorno."""
    cell = OrganismCell(state_dim=32)

    state = torch.randn(1, 32)
    neighbors = torch.randn(1, 8, 32)
    field = torch.randn(1, 1, 16, 16)

    perception = cell.perceive(state, neighbors, field, position=(8, 8))

    assert perception.shape[-1] == 32  # state_dim


def test_gated_memory():
    """Memoria se aplica condicionalmente."""
    cell = OrganismCell(state_dim=32)

    perception = torch.randn(1, 32)

    memory, gate = cell.get_memory(perception)

    assert memory.shape == perception.shape
    assert 0 <= gate <= 1  # Gate es probabilidad


def test_role_detection():
    """Celula detecta su rol."""
    cell = OrganismCell(state_dim=32)

    state = torch.randn(1, 32)

    role_probs = cell.detect_role(state)

    assert role_probs.shape == (1, 3)  # MASS, FORCE, CORRUPT
    assert role_probs.sum().item() == pytest.approx(1.0)


def test_forward_pass():
    """Paso completo produce actualizacion y rol."""
    cell = OrganismCell(state_dim=32)

    state = torch.randn(1, 32)
    neighbors = torch.randn(1, 8, 32)
    field = torch.randn(1, 1, 16, 16)

    new_state, role = cell(state, neighbors, field, position=(8, 8))

    assert new_state.shape == state.shape
    assert role.shape == (1, 3)
