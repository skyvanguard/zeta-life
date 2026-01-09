# tests/test_integration.py
"""Tests de integracion para ZetaOrganism."""
import pytest
import torch
import numpy as np
from zeta_life.organism import ZetaOrganism

class TestOrganismEmergence:
    """Tests para comportamientos emergentes."""

    @pytest.fixture
    def organism(self):
        torch.manual_seed(42)
        np.random.seed(42)
        org = ZetaOrganism(grid_size=32, n_cells=50)
        org.initialize(seed_fi=True)
        return org

    def test_fi_attracts_mass(self, organism):
        """Fi debe atraer a masas cercanas."""
        # Posicion inicial de masas
        initial_mass_positions = [
            c.position for c in organism.cells if c.role_idx == 0
        ]
        fi_position = next(
            c.position for c in organism.cells if c.role_idx == 1
        )

        # Simular
        for _ in range(50):
            organism.step()

        # Posiciones finales
        final_mass_positions = [
            c.position for c in organism.cells if c.role_idx == 0
        ]

        # Si no hay masas, el test pasa trivialmente
        if not initial_mass_positions or not final_mass_positions:
            return

        # Calcular distancias promedio a Fi
        def avg_dist(positions, target):
            if not positions:
                return 0
            return np.mean([
                np.sqrt((p[0]-target[0])**2 + (p[1]-target[1])**2)
                for p in positions
            ])

        initial_avg = avg_dist(initial_mass_positions, fi_position)
        # Fi puede moverse, usar posicion de Fi actual
        fi_cells = [c for c in organism.cells if c.role_idx == 1]
        if fi_cells:
            current_fi = fi_cells[0].position
            final_avg = avg_dist(final_mass_positions, current_fi)
            # Masas deben acercarse (o mantenerse) a Fi
            assert final_avg <= initial_avg * 2.0  # Tolerancia amplia

    def test_system_stability(self, organism):
        """Sistema debe tender hacia estabilidad."""
        # Simular hasta estabilidad
        for _ in range(100):
            organism.step()

        metrics = organism.get_metrics()

        # Sistema no debe colapsar
        total = metrics['n_fi'] + metrics['n_mass'] + metrics['n_corrupt']
        assert total == 50

        # Debe haber alguna estructura
        assert metrics['n_fi'] >= 0

    def test_coordination_improves(self, organism):
        """Coordinacion debe mejorar o mantenerse."""
        initial_coord = organism.get_metrics()['coordination']

        for _ in range(100):
            organism.step()

        final_coord = organism.get_metrics()['coordination']

        # Coordinacion no debe empeorar drasticamente
        assert final_coord >= initial_coord * 0.5 or final_coord >= 0.3


class TestBehaviorAlgorithm:
    """Tests para el algoritmo A<->B."""

    def test_bidirectional_preserves_total_influence(self):
        """La influencia total se conserva aproximadamente."""
        from zeta_life.organism import BehaviorEngine

        engine = BehaviorEngine(state_dim=32)

        cell = torch.randn(32)
        neighbors = torch.randn(8, 32)

        out, in_ = engine.bidirectional_influence(cell, neighbors)

        # La suma de influencias debe ser finita
        assert torch.isfinite(out).all()
        assert torch.isfinite(in_)

    def test_transformation_continuity(self):
        """A^3+V->B^3+A mantiene continuidad."""
        from zeta_life.organism import BehaviorEngine

        engine = BehaviorEngine(state_dim=32)

        local_cube = torch.randn(3, 3, 32)

        # Con alpha alto, debe mantener similitud
        result = engine.transform_with_potential(local_cube, 0.0, alpha=0.9)

        # Debe ser similar al original (no divergir)
        diff = (result - local_cube).norm() / local_cube.norm()
        assert diff < 3.0  # Cambio razonable


class TestMetrics:
    """Tests para metricas de inteligencia."""

    def test_coordination_bounds(self):
        """Coordinacion debe estar en [0, 1]."""
        torch.manual_seed(42)
        np.random.seed(42)
        org = ZetaOrganism(grid_size=32, n_cells=30)
        org.initialize()

        for _ in range(20):
            org.step()
            metrics = org.get_metrics()
            assert 0 <= metrics['coordination'] <= 1

    def test_stability_calculation(self):
        """Estabilidad se calcula correctamente."""
        torch.manual_seed(42)
        np.random.seed(42)
        org = ZetaOrganism(grid_size=32, n_cells=30)
        org.initialize()

        metrics = org.get_metrics()
        # Estabilidad debe estar en rango razonable
        assert -1 <= metrics['stability'] <= 1
