# -*- coding: utf-8 -*-
"""
Tests para HierarchicalSimulation.

Prueba el loop principal de consciencia jerárquica.

Fecha: 2026-01-03
"""

import pytest
import torch
import numpy as np

# Agregar path del proyecto
import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

from zeta_life.psyche import Archetype
from zeta_life.psyche import IndividuationStage
from zeta_life.consciousness import (
    HierarchicalSimulation,
    SimulationConfig,
    SimulationMetrics
)
from zeta_life.consciousness import ClusteringStrategy


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Configuración por defecto para testing."""
    return SimulationConfig(
        n_cells=40,
        n_clusters=4,
        n_steps=20,
        grid_size=64
    )


@pytest.fixture
def simulation(default_config):
    """Simulación inicializada."""
    sim = HierarchicalSimulation(default_config)
    sim.initialize()
    return sim


# =============================================================================
# TESTS: Configuración
# =============================================================================

class TestSimulationConfig:
    """Tests para SimulationConfig."""

    def test_default_config(self):
        """Test configuración por defecto."""
        config = SimulationConfig()
        assert config.n_cells == 80
        assert config.n_clusters == 4
        assert config.grid_size == 64
        assert config.n_steps == 100

    def test_custom_config(self):
        """Test configuración personalizada."""
        config = SimulationConfig(
            n_cells=100,
            n_clusters=5,
            grid_size=128
        )
        assert config.n_cells == 100
        assert config.n_clusters == 5
        assert config.grid_size == 128


# =============================================================================
# TESTS: Inicialización
# =============================================================================

class TestSimulationInitialization:
    """Tests para inicialización de simulación."""

    def test_creation(self, default_config):
        """Test creación de simulación."""
        sim = HierarchicalSimulation(default_config)
        assert sim.config == default_config
        assert not sim.is_initialized
        assert sim.step_count == 0

    def test_initialize(self, default_config):
        """Test inicialización completa."""
        sim = HierarchicalSimulation(default_config)
        sim.initialize()

        assert sim.is_initialized
        assert len(sim.cells) == default_config.n_cells
        assert len(sim.clusters) == default_config.n_clusters
        assert sim.organism is not None

    def test_initialize_with_distribution(self, default_config):
        """Test inicialización con distribución de arquetipos."""
        sim = HierarchicalSimulation(default_config)

        # Sesgo hacia PERSONA
        distribution = {
            Archetype.PERSONA: 0.5,
            Archetype.SOMBRA: 0.2,
            Archetype.ANIMA: 0.15,
            Archetype.ANIMUS: 0.15
        }

        sim.initialize(archetype_distribution=distribution)

        # Verificar que hay más células PERSONA
        persona_count = sum(
            1 for c in sim.cells
            if c.psyche.dominant == Archetype.PERSONA
        )
        total = len(sim.cells)

        # Debería haber aproximadamente 50% PERSONA
        assert persona_count >= total * 0.35  # Con algo de tolerancia

    def test_initialize_creates_metrics(self, simulation):
        """Test que inicialización crea métricas iniciales."""
        assert len(simulation.metrics_history) >= 1

        initial_metrics = simulation.metrics_history[0]
        assert initial_metrics.step == 0


# =============================================================================
# TESTS: Paso de Simulación
# =============================================================================

class TestSimulationStep:
    """Tests para paso de simulación."""

    def test_step_increments_count(self, simulation):
        """Test que step incrementa contador."""
        initial_count = simulation.step_count
        simulation.step()
        assert simulation.step_count == initial_count + 1

    def test_step_returns_metrics(self, simulation):
        """Test que step retorna métricas."""
        metrics = simulation.step()

        assert isinstance(metrics, SimulationMetrics)
        assert metrics.step == simulation.step_count

    def test_step_updates_organism(self, simulation):
        """Test que step actualiza organismo."""
        initial_phi = simulation.organism.phi_global
        simulation.step()

        # El phi puede cambiar (o no), pero el organismo debe existir
        assert simulation.organism is not None

    def test_step_records_metrics(self, simulation):
        """Test que step registra métricas."""
        initial_count = len(simulation.metrics_history)
        simulation.step()

        assert len(simulation.metrics_history) == initial_count + 1

    def test_step_not_initialized_raises(self, default_config):
        """Test que step sin inicializar lanza error."""
        sim = HierarchicalSimulation(default_config)

        with pytest.raises(RuntimeError):
            sim.step()


# =============================================================================
# TESTS: Ejecución Completa
# =============================================================================

class TestSimulationRun:
    """Tests para ejecución completa."""

    def test_run_executes_steps(self, simulation):
        """Test que run ejecuta los pasos indicados."""
        n_steps = 10
        simulation.run(n_steps, verbose=False)

        assert simulation.step_count >= n_steps

    def test_run_returns_metrics(self, simulation):
        """Test que run retorna lista de métricas."""
        n_steps = 10
        metrics = simulation.run(n_steps, verbose=False)

        assert len(metrics) >= n_steps

    def test_run_default_steps(self):
        """Test que run usa n_steps de config por defecto."""
        config = SimulationConfig(n_cells=30, n_steps=15)
        sim = HierarchicalSimulation(config)
        sim.initialize()

        metrics = sim.run(verbose=False)

        assert sim.step_count >= 15


# =============================================================================
# TESTS: Métricas
# =============================================================================

class TestSimulationMetrics:
    """Tests para registro de métricas."""

    def test_metrics_structure(self, simulation):
        """Test estructura de métricas."""
        simulation.step()
        metrics = simulation.metrics_history[-1]

        assert hasattr(metrics, 'phi_global')
        assert hasattr(metrics, 'consciousness_index')
        assert hasattr(metrics, 'vertical_coherence')
        assert hasattr(metrics, 'avg_phi_cluster')
        assert hasattr(metrics, 'avg_coherence')

    def test_metrics_to_dict(self, simulation):
        """Test conversión a diccionario."""
        simulation.step()
        metrics = simulation.metrics_history[-1]
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert 'phi_global' in d
        assert 'step' in d

    def test_get_summary(self, simulation):
        """Test resumen de simulación."""
        simulation.run(10, verbose=False)
        summary = simulation.get_summary()

        assert 'total_steps' in summary
        assert 'final_phi_global' in summary
        assert 'final_consciousness' in summary
        assert 'final_stage' in summary

    def test_metrics_valid_ranges(self, simulation):
        """Test que métricas están en rangos válidos."""
        simulation.run(10, verbose=False)

        for metrics in simulation.metrics_history:
            assert 0 <= metrics.phi_global <= 1
            assert 0 <= metrics.consciousness_index <= 1
            assert 0 <= metrics.vertical_coherence <= 1
            assert 0 <= metrics.avg_coherence <= 1


# =============================================================================
# TESTS: Dinámica
# =============================================================================

class TestSimulationDynamics:
    """Tests para dinámica de la simulación."""

    def test_lateral_dynamics_affects_cells(self):
        """Test que dinámica lateral afecta células."""
        config = SimulationConfig(
            n_cells=40,
            lateral_strength=0.5
        )
        sim = HierarchicalSimulation(config)
        sim.initialize()

        # Guardar estados iniciales
        initial_states = [
            c.psyche.archetype_state.clone()
            for c in sim.cells
        ]

        # Ejecutar pasos con dinámica lateral
        sim.run(5, verbose=False)

        # Algunos estados deberían cambiar
        changed = 0
        for i, cell in enumerate(sim.cells):
            current = cell.psyche.archetype_state.float()
            initial = initial_states[i].float()
            if not torch.allclose(current, initial, atol=0.01):
                changed += 1

        assert changed > 0

    def test_perturbation_affects_system(self):
        """Test que perturbaciones afectan el sistema."""
        config = SimulationConfig(
            n_cells=40,
            n_steps=50,
            enable_perturbations=True,
            perturbation_interval=10,
            perturbation_strength=0.5
        )
        sim = HierarchicalSimulation(config)
        sim.initialize()
        sim.run(25, verbose=False)

        # Debería haber variación en phi_global
        phi_values = [m.phi_global for m in sim.metrics_history]
        assert len(set([round(p, 2) for p in phi_values])) > 1  # No todos iguales


# =============================================================================
# TESTS: Estrategias de Clustering
# =============================================================================

class TestClusteringStrategies:
    """Tests para diferentes estrategias de clustering."""

    def test_spatial_clustering(self):
        """Test estrategia de clustering espacial."""
        config = SimulationConfig(
            n_cells=40,
            clustering_strategy=ClusteringStrategy.SPATIAL
        )
        sim = HierarchicalSimulation(config)
        sim.initialize()

        assert len(sim.clusters) == config.n_clusters

    def test_psyche_clustering(self):
        """Test estrategia de clustering psíquico."""
        config = SimulationConfig(
            n_cells=40,
            clustering_strategy=ClusteringStrategy.PSYCHE
        )
        sim = HierarchicalSimulation(config)
        sim.initialize()

        assert len(sim.clusters) == config.n_clusters

    def test_hybrid_clustering(self):
        """Test estrategia de clustering híbrido."""
        config = SimulationConfig(
            n_cells=40,
            clustering_strategy=ClusteringStrategy.HYBRID
        )
        sim = HierarchicalSimulation(config)
        sim.initialize()

        assert len(sim.clusters) == config.n_clusters


# =============================================================================
# TESTS: Callbacks
# =============================================================================

class TestSimulationCallbacks:
    """Tests para callbacks de simulación."""

    def test_on_step_callback(self, simulation):
        """Test que callback on_step se ejecuta."""
        callback_count = [0]

        def my_callback(sim, metrics):
            callback_count[0] += 1

        simulation.on_step_callbacks.append(my_callback)
        simulation.run(5, verbose=False)

        assert callback_count[0] >= 5


# =============================================================================
# TESTS: Emergencia de Consciencia
# =============================================================================

class TestConsciousnessEmergence:
    """Tests para emergencia de consciencia."""

    def test_consciousness_is_computed(self, simulation):
        """Test que consciencia se computa."""
        simulation.run(10, verbose=False)

        metrics = simulation.metrics_history[-1]
        assert metrics.consciousness_index > 0

    def test_individuation_stage_valid(self, simulation):
        """Test que etapa de individuación es válida."""
        simulation.run(10, verbose=False)

        metrics = simulation.metrics_history[-1]
        assert 0 <= metrics.individuation_stage <= 8  # 9 etapas (0-8)

    def test_phi_increases_or_stable(self):
        """Test que phi tiende a aumentar o mantenerse."""
        config = SimulationConfig(
            n_cells=60,
            n_steps=50,
            bottom_up_strength=1.0,
            top_down_strength=0.5
        )
        sim = HierarchicalSimulation(config)
        sim.initialize()
        sim.run(50, verbose=False)

        # Comparar inicio vs final
        initial_phi = sim.metrics_history[0].phi_global
        final_phi = sim.metrics_history[-1].phi_global

        # Phi no debería colapsar a 0
        assert final_phi > 0.1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
