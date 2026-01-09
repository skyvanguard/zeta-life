# tests/test_zeta_organism.py
import pytest
import torch
import numpy as np
from zeta_life.organism import ZetaOrganism

def test_organism_creation():
    """Organismo se crea con grid y células."""
    org = ZetaOrganism(grid_size=32, n_cells=50)
    assert org.grid_size == 32
    assert len(org.cells) == 50

def test_initialization():
    """Organismo inicializa con Fi semilla."""
    org = ZetaOrganism(grid_size=32, n_cells=50)
    org.initialize(seed_fi=True)

    # Debe haber al menos un Fi
    fi_count = sum(1 for c in org.cells if c.role.argmax().item() == 1)
    assert fi_count >= 1

def test_step():
    """Un paso de simulación actualiza el estado."""
    torch.manual_seed(42)
    np.random.seed(42)
    org = ZetaOrganism(grid_size=32, n_cells=50)
    org.initialize()

    old_states = [c.state.clone() for c in org.cells[:5]]
    org.step()
    new_states = [c.state for c in org.cells[:5]]

    # Estados deben cambiar
    changed = sum(1 for o, n in zip(old_states, new_states)
                  if not torch.allclose(o, n))
    assert changed > 0

def test_metrics():
    """Organismo reporta métricas de inteligencia."""
    torch.manual_seed(42)
    np.random.seed(42)
    org = ZetaOrganism(grid_size=32, n_cells=50)
    org.initialize()

    for _ in range(10):
        org.step()

    metrics = org.get_metrics()

    assert 'n_fi' in metrics
    assert 'n_mass' in metrics
    assert 'coordination' in metrics
    assert 'stability' in metrics
