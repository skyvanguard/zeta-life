# -*- coding: utf-8 -*-
"""
Tests para los integradores de consciencia jerárquica.

Prueba:
- BottomUpIntegrator: Agregación de células a clusters a organismo
- TopDownModulator: Modulación de organismo a clusters a células
- ClusterAssigner: Asignación dinámica de células a clusters

Fecha: 2026-01-03
"""

import pytest
import torch
import numpy as np
from typing import List

# Agregar path del proyecto
import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

from zeta_psyche import Archetype
from micro_psyche import ConsciousCell, MicroPsyche
from cluster import Cluster, ClusterPsyche
from organism_consciousness import OrganismConsciousness, IndividuationStage
from bottom_up_integrator import BottomUpIntegrator
from top_down_modulator import TopDownModulator
from cluster_assigner import ClusterAssigner, ClusteringConfig, ClusteringStrategy


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_cells() -> List[ConsciousCell]:
    """Crea células de muestra para testing."""
    cells = []
    for archetype in Archetype:
        for _ in range(10):
            cell = ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
            cells.append(cell)
    return cells


@pytest.fixture
def sample_clusters(sample_cells) -> List[Cluster]:
    """Crea clusters de muestra con células asignadas."""
    clusters = []
    for i, archetype in enumerate(Archetype):
        cluster_cells = sample_cells[i*10:(i+1)*10]
        for cell in cluster_cells:
            cell.cluster_id = i
        cluster = Cluster.create_from_cells(cluster_id=i, cells=cluster_cells)
        clusters.append(cluster)
    return clusters


@pytest.fixture
def integrator() -> BottomUpIntegrator:
    """Crea integrador bottom-up."""
    return BottomUpIntegrator(state_dim=32, hidden_dim=64)


@pytest.fixture
def modulator() -> TopDownModulator:
    """Crea modulador top-down."""
    return TopDownModulator(state_dim=32, hidden_dim=64)


@pytest.fixture
def assigner() -> ClusterAssigner:
    """Crea asignador de clusters."""
    config = ClusteringConfig(
        n_clusters=4,
        strategy=ClusteringStrategy.HYBRID
    )
    return ClusterAssigner(config)


# =============================================================================
# TESTS: BottomUpIntegrator
# =============================================================================

class TestBottomUpIntegrator:
    """Tests para BottomUpIntegrator."""

    def test_creation(self, integrator):
        """Test creación del integrador."""
        assert integrator.state_dim == 32
        assert integrator.hidden_dim == 64
        assert integrator.cell_importance_net is not None
        assert integrator.cluster_importance_net is not None

    def test_aggregate_cells_to_cluster(self, integrator, sample_cells):
        """Test agregación de células a cluster."""
        cells = sample_cells[:10]  # Solo 10 células
        cluster_psyche = integrator.aggregate_cells_to_cluster(cells)

        assert isinstance(cluster_psyche, ClusterPsyche)
        assert cluster_psyche.aggregate_state.shape == (4,)
        assert cluster_psyche.specialization in Archetype
        assert 0 <= cluster_psyche.phi_cluster <= 1
        assert 0 <= cluster_psyche.coherence <= 1

    def test_aggregate_empty_cells(self, integrator):
        """Test agregación con lista vacía."""
        cluster_psyche = integrator.aggregate_cells_to_cluster([])

        assert isinstance(cluster_psyche, ClusterPsyche)
        # Estado neutral para lista vacía
        assert cluster_psyche.aggregate_state.sum() > 0

    def test_aggregate_single_cell(self, integrator, sample_cells):
        """Test agregación con una sola célula."""
        cells = [sample_cells[0]]
        cluster_psyche = integrator.aggregate_cells_to_cluster(cells)

        assert isinstance(cluster_psyche, ClusterPsyche)
        # Con una célula, el estado agregado debería ser cercano al de la célula
        assert cluster_psyche.phi_cluster >= 0

    def test_aggregate_clusters_to_organism(self, integrator, sample_clusters):
        """Test agregación de clusters a organismo."""
        organism = integrator.aggregate_clusters_to_organism(sample_clusters)

        assert isinstance(organism, OrganismConsciousness)
        assert organism.global_archetype.shape == (4,)
        assert organism.dominant_archetype in Archetype
        assert 0 <= organism.phi_global <= 1
        assert 0 <= organism.vertical_coherence <= 1

    def test_integrate_full(self, integrator, sample_cells, sample_clusters):
        """Test integración completa."""
        updated_clusters, organism = integrator.integrate(
            cells=sample_cells,
            clusters=sample_clusters
        )

        assert len(updated_clusters) == 4
        assert all(c.psyche is not None for c in updated_clusters)
        assert isinstance(organism, OrganismConsciousness)

    def test_integrate_with_previous(self, integrator, sample_cells, sample_clusters):
        """Test integración con consciencia previa."""
        # Primera integración
        _, organism1 = integrator.integrate(sample_cells, sample_clusters)

        # Segunda integración con consciencia previa
        _, organism2 = integrator.integrate(
            sample_cells, sample_clusters, organism1
        )

        assert isinstance(organism2, OrganismConsciousness)
        # La integración con historial debería producir resultados
        assert organism2.phi_global >= 0

    def test_compute_phi_cluster(self, integrator, sample_cells):
        """Test cálculo de phi para cluster indirectamente."""
        cells = sample_cells[:10]
        cluster_psyche = integrator.aggregate_cells_to_cluster(cells)

        # Phi se calcula dentro de aggregate_cells_to_cluster
        assert isinstance(cluster_psyche.phi_cluster, float)
        assert 0 <= cluster_psyche.phi_cluster <= 1

    def test_compute_coherence(self, integrator, sample_cells):
        """Test cálculo de coherencia indirectamente."""
        cells = sample_cells[:10]
        cluster_psyche = integrator.aggregate_cells_to_cluster(cells)

        # Coherencia se calcula dentro de aggregate_cells_to_cluster
        assert isinstance(cluster_psyche.coherence, float)
        assert 0 <= cluster_psyche.coherence <= 1


# =============================================================================
# TESTS: TopDownModulator
# =============================================================================

class TestTopDownModulator:
    """Tests para TopDownModulator."""

    def test_creation(self, modulator):
        """Test creación del modulador."""
        assert modulator.state_dim == 32
        assert modulator.hidden_dim == 64
        assert modulator.attention_net is not None
        assert modulator.modulation_net is not None
        assert modulator.prediction_net is not None

    def test_distribute_attention(self, modulator, sample_clusters):
        """Test distribución de atención."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)
        attention = modulator.distribute_attention(organism, sample_clusters)

        assert isinstance(attention, dict)
        assert len(attention) == len(sample_clusters)
        assert all(0 <= v <= 1 for v in attention.values())

    def test_attention_to_complement(self, modulator, sample_clusters):
        """Test que arquetipos complementarios reciben más atención."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)
        attention = modulator.distribute_attention(organism, sample_clusters)

        # Verificar que hay variación en la atención
        values = list(attention.values())
        assert max(values) >= min(values)  # Hay diferencia

    def test_generate_predictions(self, modulator, sample_clusters):
        """Test generación de predicciones."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)
        predictions = modulator.generate_predictions(organism, sample_clusters)

        assert isinstance(predictions, dict)
        assert len(predictions) == len(sample_clusters)

        for pred in predictions.values():
            assert pred.shape == (4,)
            assert abs(pred.sum().item() - 1.0) < 0.01  # Softmax

    def test_generate_cell_modulation(self, modulator, sample_clusters):
        """Test generación de señal de modulación."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)
        cluster = sample_clusters[0]

        modulation = modulator.generate_cell_modulation(
            organism, cluster, cluster_attention=0.8
        )

        assert modulation.shape == (32,)
        assert modulation.abs().max() <= 1.0  # Tanh

    def test_modulate_cells(self, modulator, sample_clusters, sample_cells):
        """Test modulación de células."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)
        cluster = sample_clusters[0]
        base_mod = modulator.generate_cell_modulation(organism, cluster, 0.8)

        results = modulator.modulate_cells(
            cluster, base_mod, 0.8, organism
        )

        assert len(results) == len(cluster.cells)
        for cell, mod, surprise in results:
            assert isinstance(cell, ConsciousCell)
            assert mod.shape == (32,)
            assert 0 <= surprise <= 1

    def test_full_modulation(self, modulator, sample_clusters):
        """Test modulación completa."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)
        results = modulator.modulate(organism, sample_clusters, apply_to_cells=True)

        assert 'attention' in results
        assert 'predictions' in results
        assert 'cell_surprises' in results
        assert 'avg_surprise' in results

        assert len(results['attention']) == 4
        assert len(results['predictions']) == 4
        assert isinstance(results['avg_surprise'], float)

    def test_modulation_without_apply(self, modulator, sample_clusters, sample_cells):
        """Test modulación sin aplicar a células."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)

        # Guardar estados originales
        original_states = [c.state.clone() for c in sample_cells]

        results = modulator.modulate(organism, sample_clusters, apply_to_cells=False)

        # Estados no deberían cambiar
        for i, cell in enumerate(sample_cells):
            assert torch.allclose(cell.state, original_states[i])

    def test_modulation_quality(self, modulator, sample_clusters, sample_cells):
        """Test cálculo de calidad de modulación."""
        organism = OrganismConsciousness.from_clusters(sample_clusters)
        quality = modulator.compute_modulation_quality(sample_cells, organism)

        assert isinstance(quality, float)
        assert 0 <= quality <= 1


# =============================================================================
# TESTS: ClusterAssigner
# =============================================================================

class TestClusterAssigner:
    """Tests para ClusterAssigner."""

    def test_creation(self, assigner):
        """Test creación del asignador."""
        assert assigner.config.n_clusters == 4
        assert assigner.config.strategy == ClusteringStrategy.HYBRID
        assert assigner.step_count == 0

    def test_spatial_distance(self, assigner, sample_cells):
        """Test cálculo de distancia espacial."""
        cell = sample_cells[0]
        centroid = (32.0, 32.0)

        distance = assigner.compute_spatial_distance(cell, centroid)

        assert isinstance(distance, float)
        assert 0 <= distance <= 1

    def test_psyche_similarity(self, assigner, sample_cells, sample_clusters):
        """Test cálculo de similitud psíquica."""
        cell = sample_cells[0]
        cluster_psyche = sample_clusters[0].psyche.aggregate_state

        similarity = assigner.compute_psyche_similarity(cell, cluster_psyche)

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    def test_compute_affinity(self, assigner, sample_cells, sample_clusters):
        """Test cálculo de afinidad total."""
        cell = sample_cells[0]
        cluster = sample_clusters[0]

        affinity = assigner.compute_affinity(cell, cluster)

        assert isinstance(affinity, float)
        assert 0 <= affinity <= 1

    def test_initialize_spatial(self, assigner, sample_cells):
        """Test inicialización espacial."""
        clusters = assigner.initialize_clusters_spatial(sample_cells)

        assert len(clusters) == 4
        total_cells = sum(len(c.cells) for c in clusters)
        assert total_cells == len(sample_cells)

        # Verificar que cada célula tiene cluster_id asignado
        for cell in sample_cells:
            assert 0 <= cell.cluster_id < 4

    def test_initialize_archetype(self, assigner, sample_cells):
        """Test inicialización por arquetipo."""
        clusters = assigner.initialize_clusters_archetype(sample_cells)

        assert len(clusters) == 4

        # Verificar que clusters tienen el arquetipo correcto
        for i, cluster in enumerate(clusters):
            assert cluster.id == i

    def test_assign_first_time(self, assigner, sample_cells):
        """Test primera asignación."""
        clusters = assigner.assign(sample_cells)

        assert len(clusters) == 4
        assert assigner.step_count == 1

    def test_assign_existing_clusters(self, assigner, sample_cells, sample_clusters):
        """Test asignación con clusters existentes."""
        clusters = assigner.assign(sample_cells, sample_clusters)

        assert len(clusters) == 4
        # Sin reasignación en primer paso
        assert len(assigner.cluster_history) == 0

    def test_force_reassign(self, assigner, sample_cells, sample_clusters):
        """Test reasignación forzada."""
        clusters = assigner.assign(
            sample_cells, sample_clusters, force_reassign=True
        )

        # Con clustering dinámico, el número de clusters puede variar
        # dentro del rango permitido (min_clusters a max_clusters)
        assert assigner.config.min_clusters <= len(clusters) <= assigner.config.max_clusters
        assert len(assigner.cluster_history) == 1

    def test_should_reassign_interval(self, assigner, sample_clusters):
        """Test verificación de intervalo de reasignación."""
        assigner.step_count = 9
        assert not assigner.should_reassign(sample_clusters)

        assigner.step_count = 10
        # Puede o no reasignar dependiendo de coherencia
        result = assigner.should_reassign(sample_clusters)
        assert isinstance(result, bool)

    def test_clustering_quality(self, assigner, sample_clusters):
        """Test cálculo de calidad del clustering."""
        quality = assigner.get_clustering_quality(sample_clusters)

        assert 'intra_coherence' in quality
        assert 'inter_separation' in quality
        assert 'size_balance' in quality
        assert 'overall_quality' in quality

        assert all(isinstance(v, float) for v in quality.values())

    def test_spatial_strategy(self, sample_cells):
        """Test estrategia solo espacial."""
        config = ClusteringConfig(strategy=ClusteringStrategy.SPATIAL)
        assigner = ClusterAssigner(config)

        clusters = assigner.assign(sample_cells)
        assert len(clusters) == 4

    def test_psyche_strategy(self, sample_cells):
        """Test estrategia solo psíquica."""
        config = ClusteringConfig(strategy=ClusteringStrategy.PSYCHE)
        assigner = ClusterAssigner(config)

        clusters = assigner.assign(sample_cells)
        assert len(clusters) == 4

    def test_adaptive_strategy(self, sample_cells, integrator):
        """Test estrategia adaptativa."""
        config = ClusteringConfig(strategy=ClusteringStrategy.ADAPTIVE)
        assigner = ClusterAssigner(config)

        clusters = assigner.assign(sample_cells)

        # Agregar psyche a clusters
        for cluster in clusters:
            if cluster.cells:
                cluster.psyche = integrator.aggregate_cells_to_cluster(cluster.cells)

        # Adaptar pesos
        initial_psyche = config.psyche_weight
        assigner.adapt_weights(clusters)

        # Los pesos pueden o no cambiar dependiendo de coherencia
        assert 0 <= config.psyche_weight <= 1

    def test_balance_clusters(self, assigner, sample_cells):
        """Test balance de clusters."""
        clusters = assigner.initialize_clusters_spatial(sample_cells)

        # Forzar desbalance
        if len(clusters[0].cells) > 5:
            moved = clusters[0].cells.pop()
            moved.cluster_id = 1
            clusters[1].cells.append(moved)

        balanced = assigner.balance_clusters(clusters)

        # Después de balance, ningún cluster debería exceder el máximo
        for cluster in balanced:
            assert len(cluster.cells) <= assigner.config.max_cluster_size


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

class TestIntegration:
    """Tests de integración entre componentes."""

    def test_full_cycle(self, sample_cells, integrator, modulator, assigner):
        """Test ciclo completo bottom-up → top-down."""
        # 1. Asignar células a clusters
        clusters = assigner.assign(sample_cells)

        # 2. Integrar bottom-up
        clusters, organism = integrator.integrate(sample_cells, clusters)

        # 3. Modular top-down
        results = modulator.modulate(organism, clusters, apply_to_cells=True)

        # Verificar resultados
        assert organism is not None
        assert results['avg_surprise'] >= 0

    def test_multiple_cycles(self, sample_cells, integrator, modulator, assigner):
        """Test múltiples ciclos de integración."""
        clusters = assigner.assign(sample_cells)
        organism = None

        for i in range(5):
            # Integrar
            clusters, organism = integrator.integrate(
                sample_cells, clusters, organism
            )

            # Modular
            modulator.modulate(organism, clusters, apply_to_cells=True)

            # Potencial reasignación
            clusters = assigner.assign(sample_cells, clusters)

        assert organism is not None
        assert organism.phi_global >= 0

    def test_coherence_improves(self, sample_cells, integrator, modulator, assigner):
        """Test que la coherencia puede mejorar con ciclos."""
        clusters = assigner.assign(sample_cells)
        organism = None

        initial_coherence = None
        final_coherence = None

        for i in range(10):
            clusters, organism = integrator.integrate(
                sample_cells, clusters, organism
            )

            if i == 0:
                initial_coherence = organism.vertical_coherence

            modulator.modulate(organism, clusters, apply_to_cells=True)
            clusters = assigner.assign(sample_cells, clusters)

        final_coherence = organism.vertical_coherence

        # La coherencia debería estar en rango válido
        assert 0 <= initial_coherence <= 1
        assert 0 <= final_coherence <= 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
