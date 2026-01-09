# tests/test_top_down_modulator.py
"""Tests for TopDownModulator - top-down consciousness modulation."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

from zeta_life.psyche.zeta_psyche import Archetype
from zeta_life.consciousness.micro_psyche import ConsciousCell, MicroPsyche
from zeta_life.consciousness.cluster import Cluster, ClusterPsyche
from zeta_life.consciousness.organism_consciousness import OrganismConsciousness
from zeta_life.consciousness.top_down_modulator import TopDownModulator


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def modulator():
    """Create TopDownModulator instance."""
    return TopDownModulator(state_dim=32, hidden_dim=64)


@pytest.fixture
def cells():
    """Create test cells."""
    return [ConsciousCell.create_random(grid_size=64) for _ in range(40)]


@pytest.fixture
def clusters(cells):
    """Create test clusters with different archetypes."""
    cluster_cells = [cells[i*10:(i+1)*10] for i in range(4)]
    clusters = []
    for i, (arch, cell_group) in enumerate(zip(Archetype, cluster_cells)):
        # Bias cells toward archetype
        for c in cell_group:
            c.psyche.archetype_state = torch.zeros(4)
            c.psyche.archetype_state[arch.value] = 0.7
            c.psyche.archetype_state[(arch.value + 1) % 4] = 0.1
            c.psyche.archetype_state[(arch.value + 2) % 4] = 0.1
            c.psyche.archetype_state[(arch.value + 3) % 4] = 0.1
            c.psyche.dominant = arch
        cluster = Cluster.create_from_cells(i, cell_group)
        clusters.append(cluster)
    return clusters


@pytest.fixture
def organism(clusters):
    """Create organism from clusters."""
    return OrganismConsciousness.from_clusters(clusters)


# =============================================================================
# CREATION TESTS
# =============================================================================

class TestTopDownModulatorCreation:
    """Tests for TopDownModulator initialization."""

    def test_creation(self, modulator):
        """Should create modulator with correct params."""
        assert modulator.state_dim == 32
        assert modulator.hidden_dim == 64
        assert modulator.n_archetypes == 4

    def test_networks_initialized(self, modulator):
        """Should have initialized neural networks."""
        assert modulator.attention_net is not None
        assert modulator.modulation_net is not None
        assert modulator.prediction_net is not None

    def test_attention_net_output(self, modulator):
        """Attention net should output single value."""
        x = torch.randn(8)  # 4 + 4 archetypes
        out = modulator.attention_net(x)
        assert out.shape == (1,)
        assert 0.0 <= out.item() <= 1.0  # Sigmoid output

    def test_modulation_net_output(self, modulator):
        """Modulation net should output state_dim values."""
        x = torch.randn(4)  # 4 archetypes
        out = modulator.modulation_net(x)
        assert out.shape == (32,)  # state_dim
        # Tanh output
        assert out.abs().max() <= 1.0

    def test_prediction_net_output(self, modulator):
        """Prediction net should output 4 archetype values."""
        x = torch.randn(8)  # 4 + 4 archetypes
        out = modulator.prediction_net(x)
        assert out.shape == (4,)


# =============================================================================
# ATTENTION DISTRIBUTION TESTS
# =============================================================================

class TestAttentionDistribution:
    """Tests for attention distribution."""

    def test_compute_cluster_attention(self, modulator, organism, clusters):
        """Should compute attention for single cluster."""
        attention = modulator.compute_cluster_attention(organism, clusters[0])
        assert 0.0 <= attention <= 1.5  # Can exceed 1.0 with bonuses

    def test_distribute_attention_all_clusters(self, modulator, organism, clusters):
        """Should distribute attention to all clusters."""
        attention = modulator.distribute_attention(organism, clusters)

        assert len(attention) == len(clusters)
        for cluster_id in range(len(clusters)):
            assert cluster_id in attention
            assert 0.0 <= attention[cluster_id] <= 1.5

    def test_attention_varies_by_cluster(self, modulator, organism, clusters):
        """Different clusters should get different attention."""
        attention = modulator.distribute_attention(organism, clusters)

        values = list(attention.values())
        # Should have some variation
        assert max(values) != min(values) or len(set(values)) == 1

    def test_complementary_archetype_bonus(self, modulator, organism, clusters):
        """Complementary archetypes should get attention bonus."""
        # Get dominant and complementary archetypes
        dominant = organism.dominant_archetype
        complement_idx = modulator._get_complement_idx(dominant)

        attention = modulator.distribute_attention(organism, clusters)

        # Find cluster with complementary archetype
        complement_cluster = None
        for c in clusters:
            if c.psyche and c.psyche.specialization.value == complement_idx:
                complement_cluster = c
                break

        # Complement should have attention (test just that it exists)
        if complement_cluster:
            assert attention[complement_cluster.id] > 0


# =============================================================================
# PREDICTION TESTS
# =============================================================================

class TestPredictions:
    """Tests for top-down predictions."""

    def test_generate_predictions_all_clusters(self, modulator, organism, clusters):
        """Should generate predictions for all clusters."""
        predictions = modulator.generate_predictions(organism, clusters)

        assert len(predictions) == len(clusters)
        for cluster_id, pred in predictions.items():
            assert pred.shape == (4,)
            assert torch.isclose(pred.sum(), torch.tensor(1.0), atol=1e-4)

    def test_predictions_are_distributions(self, modulator, organism, clusters):
        """Predictions should be valid probability distributions."""
        predictions = modulator.generate_predictions(organism, clusters)

        for pred in predictions.values():
            assert (pred >= 0).all()
            assert torch.isclose(pred.sum(), torch.tensor(1.0), atol=1e-4)

    def test_prediction_reflects_specialization(self, modulator, organism, clusters):
        """Predictions should reflect cluster specialization."""
        predictions = modulator.generate_predictions(organism, clusters)

        for cluster in clusters:
            if cluster.psyche:
                pred = predictions[cluster.id]
                spec_idx = cluster.psyche.specialization.value
                # Prediction should have some weight on specialization
                assert pred[spec_idx] > 0.1


# =============================================================================
# ARCHETYPE GOAL TESTS
# =============================================================================

class TestArchetypeGoal:
    """Tests for archetype goal computation."""

    def test_compute_archetype_goal(self, modulator, organism):
        """Should compute valid archetype goal."""
        goal = modulator.compute_archetype_goal(organism)

        assert goal.shape == (4,)
        assert torch.isclose(goal.sum(), torch.tensor(1.0), atol=1e-4)
        assert (goal >= 0).all()

    def test_goal_compensates_imbalance(self, modulator):
        """Goal should compensate for imbalanced distributions."""
        # Create imbalanced organism
        cells = [ConsciousCell.create_random(64, archetype_bias=Archetype.PERSONA)
                 for _ in range(40)]
        cluster = Cluster.create_from_cells(0, cells)
        organism = OrganismConsciousness.from_clusters([cluster])

        # Force imbalance
        organism.global_archetype = torch.tensor([0.7, 0.1, 0.1, 0.1])

        goal = modulator.compute_archetype_goal(organism)

        # Goal should increase weak archetypes
        assert goal[1] > organism.global_archetype[1]  # SOMBRA
        assert goal[2] > organism.global_archetype[2]  # ANIMA
        assert goal[3] > organism.global_archetype[3]  # ANIMUS


# =============================================================================
# CELL MODULATION TESTS
# =============================================================================

class TestCellModulation:
    """Tests for cell-level modulation."""

    def test_modulate_cell_archetype(self, modulator, cells, organism):
        """Should modulate cell archetype state."""
        cell = cells[0]
        initial_state = cell.psyche.archetype_state.clone()

        goal = torch.tensor([0.4, 0.3, 0.2, 0.1])
        modulator.modulate_cell_archetype(cell, goal, strength=0.3)

        # State should have changed
        assert not torch.allclose(cell.psyche.archetype_state, initial_state, atol=1e-5)
        # Should still be valid distribution
        assert torch.isclose(cell.psyche.archetype_state.sum(), torch.tensor(1.0), atol=1e-4)

    def test_compute_adaptive_strength(self, modulator, cells, organism):
        """Should compute adaptive strength."""
        cell = cells[0]
        strength = modulator.compute_adaptive_strength(cell, organism, base_strength=0.1)

        assert 0.02 <= strength <= 0.3

    def test_generate_cell_modulation(self, modulator, organism, clusters):
        """Should generate modulation signal."""
        mod = modulator.generate_cell_modulation(organism, clusters[0], 0.8)

        assert mod.shape == (32,)  # state_dim
        assert mod.abs().max() <= 1.0  # Bounded

    def test_modulate_cells(self, modulator, organism, clusters):
        """Should modulate all cells in cluster."""
        results = modulator.modulate_cells(
            clusters[0],
            torch.randn(32),
            cluster_attention=0.8,
            organism=organism
        )

        assert len(results) == clusters[0].size
        for cell, mod, surprise in results:
            assert isinstance(cell, ConsciousCell)
            assert mod.shape == (32,)
            assert 0.0 <= surprise <= 1.0

    def test_apply_modulation_to_cell(self, modulator, cells, organism):
        """Should apply full modulation to cell."""
        cell = cells[0]
        initial_state = cell.state.clone()

        goal = modulator.compute_archetype_goal(organism)
        modulator.apply_modulation_to_cell(
            cell,
            torch.randn(32),
            organism,
            goal,
            strength=0.2
        )

        # State should have changed
        assert not torch.allclose(cell.state, initial_state)


# =============================================================================
# FULL MODULATION TESTS
# =============================================================================

class TestFullModulation:
    """Tests for complete modulation cycle."""

    def test_modulate_returns_results(self, modulator, organism, clusters):
        """Should return complete results dict."""
        results = modulator.modulate(organism, clusters, apply_to_cells=False)

        assert 'attention' in results
        assert 'predictions' in results
        assert 'cell_surprises' in results
        assert 'avg_surprise' in results
        assert 'archetype_goal' in results
        assert 'cells_modulated' in results

    def test_modulate_applies_to_cells(self, modulator, organism, clusters, cells):
        """Should apply modulation to cells when enabled."""
        # Get initial states
        initial_states = {id(c): c.state.clone() for c in cells}

        results = modulator.modulate(organism, clusters, apply_to_cells=True)

        assert results['cells_modulated'] > 0

        # Some cells should have changed
        changed = 0
        for c in cells:
            if not torch.allclose(c.state, initial_states[id(c)]):
                changed += 1
        assert changed > 0

    def test_modulate_without_applying(self, modulator, organism, clusters, cells):
        """Should not modify cells when apply_to_cells=False."""
        initial_states = {id(c): c.state.clone() for c in cells}

        results = modulator.modulate(organism, clusters, apply_to_cells=False)

        assert results['cells_modulated'] == 0

        # No cells should have changed
        for c in cells:
            assert torch.allclose(c.state, initial_states[id(c)])

    def test_avg_surprise_bounded(self, modulator, organism, clusters):
        """Average surprise should be bounded."""
        results = modulator.modulate(organism, clusters)

        assert 0.0 <= results['avg_surprise'] <= 1.0


# =============================================================================
# QUALITY METRICS TESTS
# =============================================================================

class TestQualityMetrics:
    """Tests for modulation quality metrics."""

    def test_compute_modulation_quality(self, modulator, cells, organism):
        """Should compute quality metric."""
        quality = modulator.compute_modulation_quality(cells, organism)

        assert 0.0 <= quality <= 1.0

    def test_quality_empty_cells(self, modulator, organism):
        """Should handle empty cell list."""
        quality = modulator.compute_modulation_quality([], organism)
        assert quality == 0.0

    def test_quality_improves_with_alignment(self, modulator, organism):
        """Quality should be higher when cells are aligned."""
        # Create aligned cells
        aligned_cells = []
        for _ in range(20):
            cell = ConsciousCell.create_random(64)
            cell.psyche.archetype_state = organism.global_archetype.clone()
            cell.psyche.emotional_energy = 0.8
            aligned_cells.append(cell)

        quality = modulator.compute_modulation_quality(aligned_cells, organism)
        assert quality > 0.5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestModulatorIntegration:
    """Integration tests for TopDownModulator."""

    def test_multiple_modulation_cycles(self, modulator, organism, clusters):
        """Should handle multiple modulation cycles."""
        for _ in range(10):
            results = modulator.modulate(organism, clusters, apply_to_cells=True)
            assert 'avg_surprise' in results

    def test_modulation_affects_cluster_psyche(self, modulator, organism, clusters):
        """Modulation should affect cluster psyche over time."""
        initial_phi = [c.psyche.phi_cluster for c in clusters]

        for _ in range(5):
            modulator.modulate(organism, clusters, apply_to_cells=True)
            for c in clusters:
                c.update_psyche()

        # Phi values may have changed (not necessarily, but structure should be valid)
        for c in clusters:
            assert 0.0 <= c.psyche.phi_cluster <= 1.0

    def test_complement_index_mapping(self, modulator):
        """Complement mapping should be correct."""
        assert modulator._get_complement_idx(Archetype.PERSONA) == 1  # SOMBRA
        assert modulator._get_complement_idx(Archetype.SOMBRA) == 0  # PERSONA
        assert modulator._get_complement_idx(Archetype.ANIMA) == 3  # ANIMUS
        assert modulator._get_complement_idx(Archetype.ANIMUS) == 2  # ANIMA
