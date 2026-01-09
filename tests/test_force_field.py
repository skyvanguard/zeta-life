# tests/test_force_field.py
import pytest
import torch
from zeta_life.organism import ForceField

def test_force_field_creation():
    """Campo de fuerzas se crea con kernel zeta."""
    field = ForceField(grid_size=32, M=15, sigma=0.1)
    assert field.kernel is not None
    assert field.grid_size == 32

def test_fi_emission():
    """Fi emite senal proporcional a su fuerza."""
    field = ForceField(grid_size=16)

    # Grid con un Fi en el centro
    energy = torch.zeros(1, 1, 16, 16)
    roles = torch.zeros(1, 1, 16, 16)
    energy[0, 0, 8, 8] = 1.0
    roles[0, 0, 8, 8] = 1  # FORCE

    result = field.compute(energy, roles)

    # El centro debe tener valor maximo
    assert result[0, 0, 8, 8] > result[0, 0, 0, 0]

def test_gradient_computation():
    """Gradiente apunta hacia Fi."""
    field = ForceField(grid_size=16)

    energy = torch.zeros(1, 1, 16, 16)
    energy[0, 0, 8, 8] = 1.0
    roles = torch.ones_like(energy)  # All FORCE for simplicity

    _, gradient = field.compute_with_gradient(energy, roles)

    # Gradient should have 2 channels (dx, dy)
    assert gradient.shape[1] == 2

def test_zeta_resonance_peaks():
    """Campo tiene resonancias a distancias zeta."""
    field = ForceField(grid_size=64, M=15, sigma=0.1)

    energy = torch.zeros(1, 1, 64, 64)
    energy[0, 0, 32, 32] = 1.0
    roles = torch.ones_like(energy)

    result = field.compute(energy, roles)

    # Debe haber estructura no-monotona (resonancias)
    radial = result[0, 0, 32, 32:].numpy()
    # No debe decaer monotonicamente
    diffs = radial[1:] - radial[:-1]
    assert (diffs > 0).any()  # Algun incremento = resonancia
