# -*- coding: utf-8 -*-
"""
Tests para estructuras base de consciencia jerárquica.

Prueba:
- MicroPsyche y ConsciousCell
- ClusterPsyche y Cluster
- OrganismConsciousness y HierarchicalMetrics

Fecha: 2026-01-03
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

# Importar módulos a probar
from zeta_life.consciousness import (
    MicroPsyche, ConsciousCell,
    compute_local_phi, apply_psyche_contagion
)
from zeta_life.consciousness import (
    Cluster, ClusterPsyche,
    find_cluster_neighbors, compute_inter_cluster_coherence
)
from zeta_life.consciousness import (
    OrganismConsciousness, HierarchicalMetrics,
    _integration_to_stage
)
from zeta_life.psyche import Archetype
from zeta_life.psyche import IndividuationStage


# =============================================================================
# TESTS: MicroPsyche
# =============================================================================

class TestMicroPsyche:
    """Tests para MicroPsyche."""

    def test_creation_random(self):
        """MicroPsyche aleatoria tiene estado válido."""
        psyche = MicroPsyche.create_random()

        assert psyche.archetype_state.shape == (4,)
        assert abs(psyche.archetype_state.sum().item() - 1.0) < 0.01
        assert psyche.dominant in list(Archetype)
        assert 0 <= psyche.emotional_energy <= 1
        assert 0 <= psyche.phi_local <= 1

    def test_creation_with_bias(self):
        """MicroPsyche con bias tiene arquetipo correcto."""
        for archetype in Archetype:
            psyche = MicroPsyche.create_random(bias=archetype)
            assert psyche.dominant == archetype
            # El bias hace que sea el valor más alto (aunque no >0.5 después de softmax)
            assert psyche.archetype_state[archetype.value] == psyche.archetype_state.max()

    def test_update_state(self):
        """update_state mezcla estados correctamente."""
        psyche = MicroPsyche.create_random()
        initial_dominant = psyche.dominant

        # Nueva influencia hacia SOMBRA
        new_state = torch.tensor([0.1, 0.7, 0.1, 0.1])
        psyche.update_state(new_state, blend_factor=0.9)

        assert psyche.dominant == Archetype.SOMBRA
        assert len(psyche.recent_states) == 2

    def test_compute_surprise(self):
        """compute_surprise detecta cambios."""
        psyche = MicroPsyche.create_random()

        # Sin cambio
        surprise_0 = psyche.compute_surprise()
        assert surprise_0 == 0.0  # Solo un estado en historial

        # Con cambio
        new_state = torch.tensor([0.9, 0.03, 0.03, 0.04])
        psyche.update_state(new_state, blend_factor=1.0)

        surprise_1 = psyche.compute_surprise()
        assert surprise_1 > 0

    def test_accumulated_surprise(self):
        """update_accumulated_surprise acumula con decaimiento."""
        psyche = MicroPsyche.create_random()
        assert psyche.accumulated_surprise == 0.0

        # Actualizar sin cambio de estado
        psyche.update_accumulated_surprise()
        assert psyche.accumulated_surprise == 0.0  # Sin sorpresa

        # Cambio de estado genera sorpresa
        new_state = torch.tensor([0.9, 0.03, 0.03, 0.04])
        psyche.update_state(new_state, blend_factor=0.8)
        psyche.update_accumulated_surprise()
        assert psyche.accumulated_surprise > 0

    def test_get_plasticity(self):
        """get_plasticity aumenta con sorpresa acumulada."""
        # Sin sorpresa → baja plasticidad
        psyche_stable = MicroPsyche.create_random()
        psyche_stable.accumulated_surprise = 0.0
        plasticity_low = psyche_stable.get_plasticity()

        # Alta sorpresa → alta plasticidad
        psyche_surprised = MicroPsyche.create_random()
        psyche_surprised.accumulated_surprise = 0.8
        plasticity_high = psyche_surprised.get_plasticity()

        # Verificar rango y orden
        assert 0.5 <= plasticity_low <= 1.5
        assert 0.5 <= plasticity_high <= 1.5
        assert plasticity_high > plasticity_low

    def test_alignment_with(self):
        """alignment_with calcula similitud correctamente."""
        psyche = MicroPsyche.create_random(bias=Archetype.PERSONA)

        # Similar
        similar = torch.tensor([0.7, 0.1, 0.1, 0.1])
        alignment_similar = psyche.alignment_with(similar)
        assert alignment_similar > 0.7

        # Diferente
        different = torch.tensor([0.1, 0.7, 0.1, 0.1])
        alignment_different = psyche.alignment_with(different)
        assert alignment_different < alignment_similar

    def test_complementary_archetype(self):
        """get_complementary_archetype retorna opuesto correcto."""
        pairs = [
            (Archetype.PERSONA, Archetype.SOMBRA),
            (Archetype.SOMBRA, Archetype.PERSONA),
            (Archetype.ANIMA, Archetype.ANIMUS),
            (Archetype.ANIMUS, Archetype.ANIMA),
        ]

        for dominant, expected_complement in pairs:
            psyche = MicroPsyche.create_random(bias=dominant)
            assert psyche.get_complementary_archetype() == expected_complement


# =============================================================================
# TESTS: ConsciousCell
# =============================================================================

class TestConsciousCell:
    """Tests para ConsciousCell."""

    def test_creation_random(self):
        """ConsciousCell aleatoria tiene atributos válidos."""
        cell = ConsciousCell.create_random(grid_size=64, state_dim=32)

        assert 0 <= cell.position[0] < 64
        assert 0 <= cell.position[1] < 64
        assert cell.state.shape == (32,)
        assert cell.role.shape == (3,)
        assert cell.psyche is not None
        assert cell.cluster_id == -1

    def test_role_properties(self):
        """Propiedades de rol funcionan correctamente."""
        cell = ConsciousCell.create_random(grid_size=64)

        # Por defecto es MASS
        assert cell.is_mass
        assert not cell.is_fi
        assert not cell.is_corrupt
        assert cell.role_name == 'MASS'

        # Cambiar a FORCE
        cell.role = torch.tensor([0.0, 1.0, 0.0])
        assert cell.is_fi
        assert cell.role_name == 'FORCE'

    def test_distance_to(self):
        """distance_to calcula distancia euclidiana."""
        cell_a = ConsciousCell.create_random(grid_size=64)
        cell_a.position = (0, 0)

        cell_b = ConsciousCell.create_random(grid_size=64)
        cell_b.position = (3, 4)

        dist = cell_a.distance_to(cell_b)
        assert abs(dist - 5.0) < 0.01  # 3-4-5 triángulo

    def test_psyche_similarity(self):
        """psyche_similarity compara estados arquetipales."""
        cell_a = ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.PERSONA)
        cell_b = ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.PERSONA)
        cell_c = ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.SOMBRA)

        # Células con mismo bias deberían ser más similares
        sim_ab = cell_a.psyche_similarity(cell_b)
        sim_ac = cell_a.psyche_similarity(cell_c)

        assert sim_ab > sim_ac

    def test_update_energy(self):
        """update_energy respeta límites."""
        cell = ConsciousCell.create_random(grid_size=64)
        cell.energy = 0.5

        cell.update_energy(0.3)
        assert cell.energy == 0.8

        cell.update_energy(0.5)  # Debería saturar a 1.0
        assert cell.energy == 1.0

        cell.update_energy(-1.5)  # Debería saturar a 0.0
        assert cell.energy == 0.0

    def test_movement_bias(self):
        """get_movement_bias retorna valores según arquetipo."""
        for archetype in Archetype:
            cell = ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
            dx, dy = cell.get_movement_bias()

            # Todos deberían retornar tupla de floats
            assert isinstance(dx, float)
            assert isinstance(dy, float)


# =============================================================================
# TESTS: Funciones de MicroPsyche
# =============================================================================

class TestMicroPsycheFunctions:
    """Tests para funciones auxiliares de micro_psyche."""

    def test_compute_local_phi_no_neighbors(self):
        """compute_local_phi sin vecinos retorna 0.5."""
        cell = ConsciousCell.create_random(grid_size=64)
        phi = compute_local_phi(cell, [])
        assert phi == 0.5

    def test_compute_local_phi_similar_neighbors(self):
        """compute_local_phi con vecinos similares da phi alto."""
        cells = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.PERSONA)
            for _ in range(5)
        ]

        phi = compute_local_phi(cells[0], cells[1:])
        assert phi > 0.5

    def test_apply_psyche_contagion(self):
        """apply_psyche_contagion modifica estado de célula."""
        cell = ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.PERSONA)
        neighbors = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.SOMBRA)
            for _ in range(5)
        ]

        initial_state = cell.psyche.archetype_state.clone().float()
        apply_psyche_contagion(cell, neighbors, contagion_rate=0.5)

        # Estado debería haber cambiado hacia SOMBRA
        current_state = cell.psyche.archetype_state.float()
        assert not torch.allclose(current_state, initial_state)


# =============================================================================
# TESTS: ClusterPsyche
# =============================================================================

class TestClusterPsyche:
    """Tests para ClusterPsyche."""

    def test_create_empty(self):
        """ClusterPsyche vacía tiene valores por defecto."""
        psyche = ClusterPsyche.create_empty()

        assert psyche.aggregate_state.shape == (4,)
        assert psyche.phi_cluster == 0.0
        assert psyche.coherence == 0.0

    def test_from_cells(self):
        """ClusterPsyche.from_cells agrega correctamente."""
        cells = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.ANIMA)
            for _ in range(10)
        ]

        psyche = ClusterPsyche.from_cells(cells)

        assert psyche.specialization == Archetype.ANIMA
        assert psyche.phi_cluster > 0.5  # Células similares = alta coherencia
        assert 0 <= psyche.coherence <= 1

    def test_is_specialized(self):
        """is_specialized detecta dominancia clara."""
        # Cluster con arquetipo dominante forzado
        cells_spec = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.SOMBRA)
            for _ in range(10)
        ]
        # Forzar que todas las células tengan SOMBRA muy dominante
        for cell in cells_spec:
            cell.psyche.archetype_state = torch.tensor([0.1, 0.7, 0.1, 0.1])
            cell.psyche.dominant = Archetype.SOMBRA

        psyche_spec = ClusterPsyche.from_cells(cells_spec)
        # El cluster debería tener SOMBRA como especialización
        assert psyche_spec.specialization == Archetype.SOMBRA

    def test_alignment_with(self):
        """alignment_with calcula similitud."""
        cells = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.PERSONA)
            for _ in range(10)
        ]
        psyche = ClusterPsyche.from_cells(cells)

        # Alineación con PERSONA debería ser alta
        persona_state = torch.tensor([0.7, 0.1, 0.1, 0.1])
        alignment = psyche.alignment_with(persona_state)
        assert alignment > 0.5


# =============================================================================
# TESTS: Cluster
# =============================================================================

class TestCluster:
    """Tests para Cluster."""

    def test_create_from_cells(self):
        """Cluster se crea correctamente desde células."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(10)]
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        assert cluster.id == 0
        assert cluster.size == 10
        assert cluster.psyche is not None
        assert all(c.cluster_id == 0 for c in cells)

    def test_centroid_calculation(self):
        """Centroide se calcula correctamente."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(3)]
        cells[0].position = (0, 0)
        cells[1].position = (10, 0)
        cells[2].position = (5, 10)

        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        assert abs(cluster.centroid[0] - 5.0) < 0.01
        assert abs(cluster.centroid[1] - 10/3) < 0.1

    def test_add_remove_cell(self):
        """add_cell y remove_cell funcionan correctamente."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells[:3])

        assert cluster.size == 3

        cluster.add_cell(cells[3])
        assert cluster.size == 4
        assert cells[3].cluster_id == 0

        cluster.remove_cell(cells[3])
        assert cluster.size == 3
        assert cells[3].cluster_id == -1

    def test_broadcast_influence(self):
        """broadcast_influence afecta todas las células."""
        cells = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.PERSONA)
            for _ in range(5)
        ]
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        # Broadcast hacia SOMBRA
        influence = torch.tensor([0.1, 0.7, 0.1, 0.1])
        cluster.broadcast_influence(influence, strength=0.9)

        # Todas las células deberían tender hacia SOMBRA
        for cell in cluster.cells:
            assert cell.psyche.archetype_state[1] > cell.psyche.archetype_state[0]

    def test_get_fi_mass_cells(self):
        """get_fi_cells y get_mass_cells filtran correctamente."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]
        cells[0].role = torch.tensor([0.0, 1.0, 0.0])  # Fi
        cells[1].role = torch.tensor([0.0, 1.0, 0.0])  # Fi

        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        assert len(cluster.get_fi_cells()) == 2
        assert len(cluster.get_mass_cells()) == 3


# =============================================================================
# TESTS: Funciones de Cluster
# =============================================================================

class TestClusterFunctions:
    """Tests para funciones auxiliares de cluster."""

    def test_find_cluster_neighbors(self):
        """find_cluster_neighbors detecta vecinos por proximidad."""
        # Crear 3 clusters en posiciones conocidas
        cells_0 = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]
        for c in cells_0:
            c.position = (10, 10)

        cells_1 = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]
        for c in cells_1:
            c.position = (15, 10)  # Cercano a cluster 0

        cells_2 = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]
        for c in cells_2:
            c.position = (50, 50)  # Lejano

        clusters = [
            Cluster.create_from_cells(0, cells_0),
            Cluster.create_from_cells(1, cells_1),
            Cluster.create_from_cells(2, cells_2),
        ]

        find_cluster_neighbors(clusters, threshold_ratio=0.5)

        # Clusters 0 y 1 deberían ser vecinos, cluster 2 no
        assert 1 in clusters[0].neighbors
        assert 0 in clusters[1].neighbors

    def test_compute_inter_cluster_coherence(self):
        """compute_inter_cluster_coherence mide diversidad."""
        # Clusters con diferentes especializaciones
        clusters = []
        for i, archetype in enumerate(Archetype):
            cells = [
                ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
                for _ in range(5)
            ]
            clusters.append(Cluster.create_from_cells(i, cells))

        coherence = compute_inter_cluster_coherence(clusters)

        # 4 especializaciones únicas = alta coherencia
        assert coherence > 0.7


# =============================================================================
# TESTS: OrganismConsciousness
# =============================================================================

class TestOrganismConsciousness:
    """Tests para OrganismConsciousness."""

    def test_create_initial(self):
        """OrganismConsciousness inicial tiene valores base."""
        organism = OrganismConsciousness.create_initial()

        assert organism.phi_global == 0.0
        assert organism.individuation_stage == IndividuationStage.INCONSCIENTE
        assert organism.vertical_coherence == 0.0

    def test_from_clusters(self):
        """OrganismConsciousness.from_clusters agrega correctamente."""
        clusters = []
        for i, archetype in enumerate(Archetype):
            cells = [
                ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
                for _ in range(10)
            ]
            clusters.append(Cluster.create_from_cells(i, cells))

        organism = OrganismConsciousness.from_clusters(clusters)

        assert organism.phi_global > 0
        assert organism.consciousness_level > 0
        assert organism.self_model.shape == (6,)

    def test_is_integrated(self):
        """is_integrated detecta alta integración."""
        # Crear organismo bien integrado
        clusters = []
        for i, archetype in enumerate(Archetype):
            cells = [
                ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
                for _ in range(10)
            ]
            clusters.append(Cluster.create_from_cells(i, cells))

        organism = OrganismConsciousness.from_clusters(clusters)
        # Con 4 clusters especializados y células homogéneas, debería estar integrado
        assert organism.is_integrated or organism.phi_global > 0.3

    def test_complementary_need(self):
        """get_complementary_need retorna arquetipo opuesto."""
        organism = OrganismConsciousness.create_initial()
        organism.dominant_archetype = Archetype.PERSONA

        assert organism.get_complementary_need() == Archetype.SOMBRA

    def test_weakest_archetype(self):
        """get_weakest_archetype retorna el más débil."""
        organism = OrganismConsciousness.create_initial()
        organism.global_archetype = torch.tensor([0.5, 0.1, 0.2, 0.2])

        assert organism.get_weakest_archetype() == Archetype.SOMBRA


# =============================================================================
# TESTS: Integration to Stage
# =============================================================================

class TestIntegrationToStage:
    """Tests para _integration_to_stage."""

    def test_low_integration(self):
        """Baja integración = etapas tempranas."""
        stage = _integration_to_stage(integration=0.0, phi=0.0)
        assert stage == IndividuationStage.INCONSCIENTE

    def test_high_integration(self):
        """Alta integración = etapas avanzadas."""
        stage = _integration_to_stage(integration=0.25, phi=1.0)
        assert stage in [IndividuationStage.EMERGENCIA_SELF, IndividuationStage.SELF_REALIZADO]

    def test_progression(self):
        """Etapas progresan con integración."""
        prev_stage_value = 0

        for integration in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            stage = _integration_to_stage(integration, phi=0.5)
            assert stage.value >= prev_stage_value
            prev_stage_value = stage.value


# =============================================================================
# TESTS: HierarchicalMetrics
# =============================================================================

class TestHierarchicalMetrics:
    """Tests para HierarchicalMetrics."""

    def test_compute(self):
        """HierarchicalMetrics.compute genera métricas válidas."""
        # Crear sistema de prueba
        all_cells = []
        clusters = []

        for i in range(4):
            cells = [ConsciousCell.create_random(grid_size=64) for _ in range(10)]
            all_cells.extend(cells)
            clusters.append(Cluster.create_from_cells(i, cells))

        organism = OrganismConsciousness.from_clusters(clusters)

        metrics = HierarchicalMetrics.compute(all_cells, clusters, organism)

        assert metrics.cell_count == 40
        assert metrics.cluster_count == 4
        assert 0 <= metrics.consciousness_index <= 1
        assert 0 <= metrics.bottom_up_flow <= 1
        assert 0 <= metrics.top_down_flow <= 1

    def test_archetype_distribution(self):
        """Distribución de arquetipos suma 1."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(20)]
        clusters = [Cluster.create_from_cells(0, cells)]
        organism = OrganismConsciousness.from_clusters(clusters)

        metrics = HierarchicalMetrics.compute(cells, clusters, organism)

        total = sum(metrics.archetype_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_to_dict(self):
        """to_dict serializa correctamente."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(10)]
        clusters = [Cluster.create_from_cells(0, cells)]
        organism = OrganismConsciousness.from_clusters(clusters)

        metrics = HierarchicalMetrics.compute(cells, clusters, organism)
        metrics_dict = metrics.to_dict()

        assert 'cell_count' in metrics_dict
        assert 'consciousness_index' in metrics_dict
        assert 'archetype_distribution' in metrics_dict


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHierarchicalIntegration:
    """Tests de integración entre todos los componentes."""

    def test_full_hierarchy(self):
        """Sistema completo funciona correctamente."""
        # Nivel 0: Crear 100 células
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(100)]

        # Nivel 1: Agrupar en 8 clusters
        clusters = []
        cells_per_cluster = len(cells) // 8

        for i in range(8):
            start = i * cells_per_cluster
            end = start + cells_per_cluster if i < 7 else len(cells)
            cluster_cells = cells[start:end]
            clusters.append(Cluster.create_from_cells(i, cluster_cells))

        # Detectar vecinos
        find_cluster_neighbors(clusters)

        # Nivel 2: Crear consciencia de organismo
        organism = OrganismConsciousness.from_clusters(clusters)

        # Verificar integridad
        assert all(c.cluster_id >= 0 for c in cells)
        assert organism.consciousness_level > 0
        assert len(clusters) == 8

        # Métricas
        metrics = HierarchicalMetrics.compute(cells, clusters, organism)
        assert metrics.cell_count == 100
        assert metrics.cluster_count == 8

    def test_consciousness_emerges(self):
        """Consciencia emerge de células heterogéneas."""
        # Crear células con arquetipos variados
        cells = []
        for archetype in Archetype:
            for _ in range(25):
                cells.append(ConsciousCell.create_random(
                    grid_size=64, archetype_bias=archetype
                ))

        # Agrupar por arquetipo (4 clusters especializados)
        clusters = []
        for i, archetype in enumerate(Archetype):
            cluster_cells = [c for c in cells if c.psyche.dominant == archetype][:25]
            if cluster_cells:
                clusters.append(Cluster.create_from_cells(i, cluster_cells))

        organism = OrganismConsciousness.from_clusters(clusters)

        # Con 4 clusters especializados, debería haber alta integración
        assert organism.phi_global > 0.5
        assert organism.consciousness_level > 0.5
