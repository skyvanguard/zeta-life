# tests/test_cell_state.py
import pytest
import torch
from cell_state import CellState, CellRole

def test_cell_state_creation():
    """Celula se crea con estado inicial valido."""
    cell = CellState()
    assert cell.alive == True
    assert cell.mass >= 0
    assert cell.energy >= 0
    assert cell.role == CellRole.MASS

def test_role_transition_to_force():
    """Celula con alta energia y seguidores se convierte en Fi."""
    cell = CellState(energy=0.8, controlled_mass=5)
    cell.update_role(fi_threshold=0.7, min_followers=3)
    assert cell.role == CellRole.FORCE

def test_role_transition_to_corrupt():
    """Fi detecta rival cercano y se vuelve corrupto."""
    cell = CellState(role=CellRole.FORCE)
    rival_nearby = True
    cell.update_role(rival_detected=rival_nearby)
    assert cell.role == CellRole.CORRUPT

def test_equilibrium_scaling():
    """Fi efectiva escala con sqrt de masa controlada."""
    cell = CellState(role=CellRole.FORCE, fi_base=1.0, controlled_mass=4)
    assert cell.effective_fi() == pytest.approx(2.0)  # 1.0 * sqrt(4)
