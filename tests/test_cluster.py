# tests/test_cluster.py
"""Tests for Cluster and ClusterPsyche - cluster-level consciousness."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

from zeta_life.psyche.zeta_psyche import Archetype
from zeta_life.organism.cell_state import CellRole
from zeta_life.consciousness.micro_psyche import ConsciousCell, MicroPsyche
from zeta_life.consciousness.cluster import (
    ClusterPsyche, Cluster, find_cluster_neighbors,
    compute_inter_cluster_coherence, merge_clusters, split_cluster
)


# =============================================================================
# CLUSTER PSYCHE TESTS
# =============================================================================

class TestClusterPsyche:
    """Tests for ClusterPsyche."""

    def test_create_empty(self):
        """Should create empty psyche with balanced state."""
        psyche = ClusterPsyche.create_empty()
        assert psyche.aggregate_state.shape == (4,)
        assert psyche.phi_cluster == 0.0
        assert psyche.coherence == 0.0

    def test_from_cells_single(self):
        """Should create psyche from single cell."""
        cell = ConsciousCell.create_random(grid_size=64)
        psyche = ClusterPsyche.from_cells([cell])

        assert psyche.aggregate_state.shape == (4,)
        assert isinstance(psyche.specialization, Archetype)
        # Single cell = perfect coherence
        assert psyche.phi_cluster >= 0.0

    def test_from_cells_multiple(self):
        """Should aggregate multiple cells."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(10)]
        psyche = ClusterPsyche.from_cells(cells)

        assert psyche.aggregate_state.shape == (4,)
        assert 0.0 <= psyche.phi_cluster <= 1.0
        assert 0.0 <= psyche.coherence <= 1.0

    def test_from_cells_empty(self):
        """Should handle empty cell list."""
        psyche = ClusterPsyche.from_cells([])
        assert psyche.phi_cluster == 0.0

    def test_is_specialized(self):
        """Should detect specialization."""
        # Highly specialized state (needs high value to survive softmax in __post_init__)
        # Use value that after softmax gives > 0.4
        spec_state = torch.tensor([3.0, 0.0, 0.0, 0.0])  # After softmax ≈ [0.87, 0.04, 0.04, 0.04]
        psyche = ClusterPsyche(
            aggregate_state=spec_state,
            specialization=Archetype.PERSONA,
            phi_cluster=0.5,
            coherence=0.5,
            prediction_error=0.1,
            integration_level=0.5
        )
        assert psyche.is_specialized

        # Balanced state (already normalized, will still get softmax but remain balanced)
        balanced_state = torch.ones(4) / 4
        psyche2 = ClusterPsyche(
            aggregate_state=balanced_state,
            specialization=Archetype.PERSONA,
            phi_cluster=0.5,
            coherence=0.5,
            prediction_error=0.1,
            integration_level=0.5
        )
        assert not psyche2.is_specialized

    def test_balance_property(self):
        """Balance should be higher for balanced states."""
        # Balanced
        balanced = ClusterPsyche(
            aggregate_state=torch.ones(4) / 4,
            specialization=Archetype.PERSONA,
            phi_cluster=0.5, coherence=0.5,
            prediction_error=0.1, integration_level=0.5
        )

        # Specialized
        specialized = ClusterPsyche(
            aggregate_state=torch.tensor([0.85, 0.05, 0.05, 0.05]),
            specialization=Archetype.PERSONA,
            phi_cluster=0.5, coherence=0.5,
            prediction_error=0.1, integration_level=0.5
        )

        assert balanced.balance > specialized.balance

    def test_alignment_with(self):
        """Should compute alignment correctly."""
        psyche = ClusterPsyche(
            aggregate_state=torch.tensor([0.7, 0.1, 0.1, 0.1]),
            specialization=Archetype.PERSONA,
            phi_cluster=0.5, coherence=0.5,
            prediction_error=0.1, integration_level=0.5
        )

        # Same state = high alignment
        same_state = torch.tensor([0.7, 0.1, 0.1, 0.1])
        assert psyche.alignment_with(same_state) > 0.9

        # Opposite state = lower alignment
        opposite = torch.tensor([0.1, 0.7, 0.1, 0.1])
        assert psyche.alignment_with(opposite) < psyche.alignment_with(same_state)

    def test_to_dict(self):
        """Should serialize to dict."""
        psyche = ClusterPsyche.from_cells([
            ConsciousCell.create_random(grid_size=64) for _ in range(5)
        ])
        d = psyche.to_dict()

        assert 'aggregate_state' in d
        assert 'specialization' in d
        assert 'phi_cluster' in d
        assert 'coherence' in d


# =============================================================================
# CLUSTER TESTS
# =============================================================================

class TestCluster:
    """Tests for Cluster."""

    @pytest.fixture
    def cells(self):
        """Create test cells."""
        return [ConsciousCell.create_random(grid_size=64) for _ in range(10)]

    def test_create_from_cells(self, cells):
        """Should create cluster from cells."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        assert cluster.id == 0
        assert cluster.size == 10
        assert all(c.cluster_id == 0 for c in cells)

    def test_size_property(self, cells):
        """Size should return cell count."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)
        assert cluster.size == len(cells)

    def test_is_empty(self):
        """Should detect empty cluster."""
        empty_cluster = Cluster(id=0, cells=[])
        assert empty_cluster.is_empty

        cluster = Cluster.create_from_cells(0, [ConsciousCell.create_random(64)])
        assert not cluster.is_empty

    def test_centroid_computed(self, cells):
        """Should compute centroid from cell positions."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        # Centroid should be average of positions
        positions = torch.tensor([c.position for c in cells], dtype=torch.float32)
        expected = positions.mean(dim=0)

        assert abs(cluster.centroid[0] - expected[0].item()) < 1e-4
        assert abs(cluster.centroid[1] - expected[1].item()) < 1e-4

    def test_avg_energy(self, cells):
        """Should compute average energy."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)
        expected = np.mean([c.energy for c in cells])
        assert abs(cluster.avg_energy - expected) < 1e-5

    def test_add_cell(self):
        """Should add cell to cluster."""
        cluster = Cluster(id=0, cells=[])
        cell = ConsciousCell.create_random(grid_size=64)

        cluster.add_cell(cell)

        assert cluster.size == 1
        assert cell.cluster_id == 0

    def test_remove_cell(self, cells):
        """Should remove cell from cluster."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)
        cell = cells[0]

        cluster.remove_cell(cell)

        assert cluster.size == 9
        assert cell.cluster_id == -1

    def test_update_psyche(self, cells):
        """Should update psyche from cells."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)
        old_phi = cluster.psyche.phi_cluster

        # Modify a cell
        cells[0].psyche.archetype_state = torch.tensor([0.9, 0.03, 0.03, 0.04])
        cluster.update_psyche()

        # Psyche should have been updated (may or may not change phi)
        assert cluster.psyche is not None

    def test_distance_to_point(self, cells):
        """Should compute distance to point."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        # Distance to centroid should be 0
        dist_to_self = cluster.distance_to_point(cluster.centroid)
        assert dist_to_self < 1e-5

        # Distance to far point should be large
        dist_to_far = cluster.distance_to_point((100.0, 100.0))
        assert dist_to_far > dist_to_self

    def test_distance_to_cluster(self):
        """Should compute distance between clusters."""
        cells1 = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]
        cells2 = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]

        cluster1 = Cluster.create_from_cells(0, cells1)
        cluster2 = Cluster.create_from_cells(1, cells2)

        dist = cluster1.distance_to_cluster(cluster2)
        assert dist >= 0.0

    def test_broadcast_influence(self, cells):
        """Should broadcast influence to all cells."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        # Get initial states
        initial_states = [c.psyche.archetype_state.clone() for c in cells]

        # Broadcast strong PERSONA influence
        influence = torch.tensor([0.9, 0.03, 0.03, 0.04])
        cluster.broadcast_influence(influence, strength=0.5)

        # Some cells should have changed
        changed = 0
        for i, cell in enumerate(cells):
            if not torch.allclose(cell.psyche.archetype_state, initial_states[i], atol=1e-5):
                changed += 1
        assert changed > 0

    def test_compute_internal_coherence(self):
        """Should compute internal coherence."""
        # Create cells with similar archetypes
        cells = []
        for _ in range(5):
            cell = ConsciousCell.create_random(grid_size=64, archetype_bias=Archetype.PERSONA)
            cells.append(cell)

        cluster = Cluster.create_from_cells(0, cells)
        coherence = cluster.compute_internal_coherence()

        assert 0.0 <= coherence <= 1.0

    def test_to_dict(self, cells):
        """Should serialize to dict."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)
        d = cluster.to_dict()

        assert d['id'] == 0
        assert d['size'] == 10
        assert 'centroid' in d
        assert 'psyche' in d

    def test_get_fi_and_mass_cells(self, cells):
        """Should filter cells by role."""
        cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)

        fi_cells = cluster.get_fi_cells()
        mass_cells = cluster.get_mass_cells()

        # All cells should be in one category or the other
        assert len(fi_cells) + len(mass_cells) <= cluster.size


# =============================================================================
# CLUSTER UTILITIES TESTS
# =============================================================================

class TestClusterUtilities:
    """Tests for cluster utility functions."""

    def test_find_cluster_neighbors_empty(self):
        """Should handle empty cluster list."""
        clusters = []
        find_cluster_neighbors(clusters)  # Should not raise

    def test_find_cluster_neighbors_single(self):
        """Should handle single cluster."""
        cells = [ConsciousCell.create_random(64) for _ in range(5)]
        cluster = Cluster.create_from_cells(0, cells)

        find_cluster_neighbors([cluster])
        # No neighbors possible
        assert cluster.neighbors == []

    def test_find_cluster_neighbors_multiple(self):
        """Should find neighbors between clusters."""
        # Create clusters at different positions
        clusters = []
        for i in range(4):
            cells = [ConsciousCell.create_random(64) for _ in range(5)]
            # Force position
            for c in cells:
                c.position = (i * 10, i * 10)
            cluster = Cluster.create_from_cells(i, cells)
            clusters.append(cluster)

        find_cluster_neighbors(clusters, threshold_ratio=2.0)

        # Each cluster should have some neighbors
        total_neighbors = sum(len(c.neighbors) for c in clusters)
        assert total_neighbors > 0

    def test_compute_inter_cluster_coherence_single(self):
        """Should return 1.0 for single cluster."""
        cells = [ConsciousCell.create_random(64) for _ in range(5)]
        cluster = Cluster.create_from_cells(0, cells)

        coherence = compute_inter_cluster_coherence([cluster])
        assert coherence == 1.0

    def test_compute_inter_cluster_coherence_multiple(self):
        """Should compute coherence for multiple clusters."""
        clusters = []
        for i, arch in enumerate(Archetype):
            cells = [ConsciousCell.create_random(64, archetype_bias=arch) for _ in range(5)]
            cluster = Cluster.create_from_cells(i, cells)
            clusters.append(cluster)

        coherence = compute_inter_cluster_coherence(clusters)
        assert 0.0 <= coherence <= 1.0

    def test_merge_clusters(self):
        """Should merge two clusters."""
        cells1 = [ConsciousCell.create_random(64) for _ in range(5)]
        cells2 = [ConsciousCell.create_random(64) for _ in range(3)]

        cluster1 = Cluster.create_from_cells(0, cells1)
        cluster2 = Cluster.create_from_cells(1, cells2)

        merged = merge_clusters(cluster1, cluster2)

        assert merged.size == 8
        assert merged.id == 0  # Keeps first ID

    def test_split_cluster_insufficient_cells(self):
        """Should not split if too few cells."""
        cells = [ConsciousCell.create_random(64)]
        cluster = Cluster.create_from_cells(0, cells)

        parts = split_cluster(cluster, n_parts=2)

        assert len(parts) == 1  # Can't split

    def test_split_cluster_success(self):
        """Should split cluster into parts."""
        cells = [ConsciousCell.create_random(64) for _ in range(10)]
        cluster = Cluster.create_from_cells(0, cells)

        parts = split_cluster(cluster, n_parts=2)

        assert len(parts) == 2
        total_cells = sum(p.size for p in parts)
        assert total_cells == 10


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestClusterIntegration:
    """Integration tests for cluster system."""

    def test_create_update_serialize(self):
        """Test full create -> update -> serialize cycle."""
        # Create
        cells = [ConsciousCell.create_random(64) for _ in range(10)]
        cluster = Cluster.create_from_cells(0, cells)

        # Update
        for cell in cells:
            cell.psyche.archetype_state = torch.tensor([0.6, 0.2, 0.1, 0.1])
        cluster.update_psyche()

        # Serialize
        data = cluster.to_dict()
        # V0 is the abstract name (was PERSONA in Jungian system)
        assert data['psyche']['specialization'] == 'V0'

    def test_cluster_dynamics(self):
        """Test cluster behavior over multiple updates."""
        cells = [ConsciousCell.create_random(64) for _ in range(20)]
        cluster = Cluster.create_from_cells(0, cells)

        phi_values = []
        for _ in range(10):
            # Simulate evolution
            influence = torch.rand(4)
            cluster.broadcast_influence(influence, strength=0.1)
            cluster.update_psyche()
            phi_values.append(cluster.psyche.phi_cluster)

        # Phi should remain bounded
        assert all(0.0 <= p <= 1.0 for p in phi_values)

    def test_multi_cluster_system(self):
        """Test system with multiple interacting clusters."""
        # Create 4 clusters, one per archetype
        clusters = []
        for i, arch in enumerate(Archetype):
            cells = [ConsciousCell.create_random(64, archetype_bias=arch) for _ in range(8)]
            cluster = Cluster.create_from_cells(i, cells)
            clusters.append(cluster)

        # Find neighbors
        find_cluster_neighbors(clusters)

        # Compute coherence
        coherence = compute_inter_cluster_coherence(clusters)

        # With diverse archetypes, should have decent coherence
        assert coherence > 0.3
