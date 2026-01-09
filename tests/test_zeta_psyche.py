# tests/test_zeta_psyche.py
"""Tests for ZetaPsyche - core archetype-based consciousness system."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\admin\\Documents\\life')

from zeta_life.psyche.zeta_psyche import (
    Archetype, TetrahedralSpace, ZetaModulator, PsychicCell,
    ZetaPsyche, SymbolSystem, PsycheInterface, get_zeta_zeros
)


# =============================================================================
# ARCHETYPE TESTS
# =============================================================================

class TestArchetype:
    """Tests for Archetype enum."""

    def test_archetype_values(self):
        """Archetypes should have correct integer values."""
        assert Archetype.PERSONA.value == 0
        assert Archetype.SOMBRA.value == 1
        assert Archetype.ANIMA.value == 2
        assert Archetype.ANIMUS.value == 3

    def test_archetype_count(self):
        """Should have exactly 4 archetypes."""
        assert len(Archetype) == 4


# =============================================================================
# TETRAHEDRAL SPACE TESTS
# =============================================================================

class TestTetrahedralSpace:
    """Tests for TetrahedralSpace."""

    @pytest.fixture
    def space(self):
        return TetrahedralSpace()

    def test_vertices_shape(self, space):
        """Vertices should be 4x3 tensor."""
        assert space.vertices.shape == (4, 3)

    def test_vertices_normalized(self, space):
        """Vertices should be on unit sphere."""
        norms = torch.norm(space.vertices, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_center_is_mean(self, space):
        """Center should be mean of vertices."""
        expected = space.vertices.mean(dim=0)
        assert torch.allclose(space.center, expected)

    def test_barycentric_to_3d_identity(self, space):
        """Pure archetype weights should map near vertices (after softmax)."""
        for i in range(4):
            weights = torch.zeros(4)
            weights[i] = 100.0  # Large value to dominate after softmax
            pos = space.barycentric_to_3d(weights)
            # After softmax, the dominant weight is close to 1.0
            # so position should be close to vertex
            assert torch.allclose(pos, space.vertices[i], atol=0.1)

    def test_barycentric_to_3d_center(self, space):
        """Equal weights should map to center."""
        weights = torch.ones(4) / 4
        pos = space.barycentric_to_3d(weights)
        assert torch.allclose(pos, space.center, atol=1e-4)

    def test_get_dominant_archetype(self, space):
        """Should return archetype with highest weight."""
        weights = torch.tensor([0.1, 0.5, 0.3, 0.1])
        assert space.get_dominant_archetype(weights) == Archetype.SOMBRA

    def test_integration_score_range(self, space):
        """Integration score should be in [0, 1]."""
        for _ in range(10):
            weights = torch.rand(4)
            score = space.integration_score(weights)
            assert 0.0 <= score <= 1.0

    def test_integration_score_max_at_center(self, space):
        """Integration score should be maximal at center."""
        center_weights = torch.ones(4) / 4
        center_score = space.integration_score(center_weights)

        # Compare with vertex (extreme position)
        vertex_weights = torch.tensor([0.97, 0.01, 0.01, 0.01])
        vertex_score = space.integration_score(vertex_weights)

        assert center_score > vertex_score

    def test_distance_to_center(self, space):
        """Distance to center should be 0 at center."""
        center_weights = torch.ones(4) / 4
        dist = space.distance_to_center(center_weights)
        assert dist < 0.01  # Very close to 0


# =============================================================================
# ZETA MODULATOR TESTS
# =============================================================================

class TestZetaModulator:
    """Tests for ZetaModulator."""

    def test_creation(self):
        """Should create modulator with correct params."""
        mod = ZetaModulator(M=10, sigma=0.1)
        assert mod.M == 10
        assert mod.sigma == 0.1
        assert len(mod.gammas) == 10

    def test_gammas_are_zeta_zeros(self):
        """Gammas should be first M zeta zeros."""
        mod = ZetaModulator(M=5)
        expected = torch.tensor([14.134725, 21.022040, 25.010858, 30.424876, 32.935062])
        assert torch.allclose(mod.gammas, expected, atol=1e-4)

    def test_phi_weights_normalized(self):
        """Phi weights should sum to 1."""
        mod = ZetaModulator(M=15)
        assert torch.isclose(mod.phi.sum(), torch.tensor(1.0), atol=1e-5)

    def test_forward_modulates_input(self):
        """Forward should modulate input tensor."""
        mod = ZetaModulator(M=10)
        x = torch.ones(5, 10)
        y = mod(x)
        assert y.shape == x.shape
        # Should be close to original but not identical (modulation applied)
        assert not torch.allclose(y, x)

    def test_forward_increments_time(self):
        """Forward should increment internal time."""
        mod = ZetaModulator(M=10)
        assert mod.t == 0
        mod(torch.randn(2, 4))
        assert mod.t == 1
        mod(torch.randn(2, 4))
        assert mod.t == 2

    def test_get_resonance_shape(self):
        """Resonance should have M elements."""
        mod = ZetaModulator(M=15)
        mod(torch.randn(2, 4))  # Advance time
        res = mod.get_resonance()
        assert res.shape == (15,)


# =============================================================================
# PSYCHIC CELL TESTS
# =============================================================================

class TestPsychicCell:
    """Tests for PsychicCell."""

    def test_creation(self):
        """Should create cell with correct initial state."""
        pos = torch.tensor([0.25, 0.25, 0.25, 0.25])
        cell = PsychicCell(position=pos, energy=0.5)
        assert torch.allclose(cell.position, pos)
        assert cell.energy == 0.5
        assert cell.age == 0

    def test_memory_initialized(self):
        """Memory should be initialized to zeros."""
        pos = torch.rand(4)
        cell = PsychicCell(position=pos)
        assert cell.memory.shape == (10, 4)
        assert torch.allclose(cell.memory, torch.zeros(10, 4))

    def test_update_memory(self):
        """Update memory should store current position."""
        pos = torch.tensor([0.7, 0.1, 0.1, 0.1])
        cell = PsychicCell(position=pos)
        cell.update_memory()

        assert torch.allclose(cell.memory[0], pos)
        assert cell.age == 1

    def test_update_memory_rolls(self):
        """Memory should roll when updated."""
        cell = PsychicCell(position=torch.tensor([1.0, 0.0, 0.0, 0.0]))
        cell.update_memory()

        cell.position = torch.tensor([0.0, 1.0, 0.0, 0.0])
        cell.update_memory()

        # Most recent should be at index 0
        assert cell.memory[0, 1] == 1.0
        # Previous should be at index 1
        assert cell.memory[1, 0] == 1.0

    def test_get_trajectory(self):
        """Trajectory should return recent positions."""
        cell = PsychicCell(position=torch.rand(4))
        for _ in range(5):
            cell.update_memory()

        traj = cell.get_trajectory()
        assert len(traj) == 5


# =============================================================================
# ZETA PSYCHE TESTS
# =============================================================================

class TestZetaPsyche:
    """Tests for ZetaPsyche main class."""

    @pytest.fixture
    def psyche(self):
        return ZetaPsyche(n_cells=50, hidden_dim=32, M=10)

    def test_creation(self, psyche):
        """Should create psyche with correct params."""
        assert psyche.n_cells == 50
        assert len(psyche.cells) == 50
        assert psyche.t == 0

    def test_cells_have_valid_positions(self, psyche):
        """All cells should have valid barycentric positions."""
        for cell in psyche.cells:
            # Should sum to 1
            assert torch.isclose(cell.position.sum(), torch.tensor(1.0), atol=1e-5)
            # All should be non-negative
            assert (cell.position >= 0).all()

    def test_global_state_is_mean(self, psyche):
        """Global state should be mean of cell positions."""
        positions = torch.stack([c.position for c in psyche.cells])
        expected = F.softmax(positions.mean(dim=0), dim=-1)
        assert torch.allclose(psyche.global_state, expected, atol=1e-4)

    def test_observe_self_returns_metrics(self, psyche):
        """observe_self should return required metrics."""
        obs = psyche.observe_self()

        required_keys = [
            'integration', 'dominant', 'blend', 'dist_to_self',
            'stability', 'self_reference', 'consciousness_index',
            'global_state', 'population_distribution'
        ]
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"

    def test_consciousness_index_range(self, psyche):
        """Consciousness index should be in [0, 1]."""
        for _ in range(10):
            obs = psyche.step()
            assert 0.0 <= obs['consciousness_index'] <= 1.0

    def test_step_advances_time(self, psyche):
        """Step should advance internal time."""
        assert psyche.t == 0
        psyche.step()
        assert psyche.t == 1

    def test_step_updates_history(self, psyche):
        """Step should update consciousness history."""
        assert len(psyche.consciousness_history) == 0
        psyche.step()
        assert len(psyche.consciousness_history) == 1

    def test_receive_stimulus_affects_cells(self, psyche):
        """Stimulus should affect cell positions."""
        initial_positions = [c.position.clone() for c in psyche.cells]

        # Strong stimulus toward PERSONA
        stimulus = torch.tensor([1.0, 0.0, 0.0, 0.0])
        psyche.receive_stimulus(stimulus)

        # At least some cells should have moved
        moved_count = sum(
            1 for i, c in enumerate(psyche.cells)
            if not torch.allclose(c.position, initial_positions[i], atol=1e-5)
        )
        assert moved_count > 0

    def test_communicate_returns_state(self, psyche):
        """Communicate should return global state tensor."""
        input_weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
        output = psyche.communicate(input_weights)

        assert output.shape == (4,)
        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5)

    def test_get_response_returns_archetype(self, psyche):
        """get_response should return dominant archetype and blend."""
        dominant, blend = psyche.get_response()

        assert isinstance(dominant, Archetype)
        assert len(blend) == 4
        for arch in Archetype:
            assert arch in blend

    def test_population_distribution_sums_to_one(self, psyche):
        """Population distribution should sum to 1."""
        dist = psyche.get_population_distribution()
        assert torch.isclose(dist.sum(), torch.tensor(1.0), atol=1e-5)

    def test_consciousness_trend_initial(self, psyche):
        """Trend should be 0 with insufficient history."""
        trend = psyche.get_consciousness_trend(window=50)
        assert trend == 0.0

    def test_consciousness_trend_with_history(self, psyche):
        """Trend should be computable with enough history."""
        for _ in range(100):
            psyche.step()

        trend = psyche.get_consciousness_trend(window=25)
        assert isinstance(trend, float)


# =============================================================================
# SYMBOL SYSTEM TESTS
# =============================================================================

class TestSymbolSystem:
    """Tests for SymbolSystem."""

    @pytest.fixture
    def symbols(self):
        return SymbolSystem()

    def test_has_archetype_symbols(self, symbols):
        """Should have symbols for each archetype."""
        archetype_symbols = ['☉', '☽', '♀', '♂']
        for sym in archetype_symbols:
            assert sym in symbols.symbol_to_weights

    def test_encode_pure_archetypes(self, symbols):
        """Pure archetype weights should encode to archetype symbols."""
        expected = {
            0: '☉',  # PERSONA
            1: '☽',  # SOMBRA
            2: '♀',  # ANIMA
            3: '♂',  # ANIMUS
        }
        for idx, expected_sym in expected.items():
            weights = torch.zeros(4)
            weights[idx] = 1.0
            sym = symbols.encode(weights)
            assert sym == expected_sym, f"Expected {expected_sym} for archetype {idx}, got {sym}"

    def test_encode_balanced_returns_self(self, symbols):
        """Balanced weights should encode to Self symbol."""
        weights = torch.ones(4) / 4
        sym = symbols.encode(weights)
        assert sym == '✧'

    def test_decode_returns_weights(self, symbols):
        """Decode should return valid weights."""
        for sym, expected in symbols.symbols:
            decoded = symbols.decode(sym)
            assert torch.allclose(decoded, expected)

    def test_decode_unknown_returns_center(self, symbols):
        """Unknown symbol should decode to center."""
        decoded = symbols.decode('X')
        expected = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert torch.allclose(decoded, expected)

    def test_encode_sequence(self, symbols):
        """Should encode trajectory to symbol string."""
        trajectory = torch.tensor([
            [0.9, 0.03, 0.03, 0.04],  # PERSONA
            [0.03, 0.9, 0.03, 0.04],  # SOMBRA
            [0.03, 0.03, 0.9, 0.04],  # ANIMA
        ])
        seq = symbols.encode_sequence(trajectory)
        assert len(seq) == 3
        assert seq == '☉☽♀'

    def test_decode_sequence(self, symbols):
        """Should decode symbol string to trajectory."""
        seq = '☉☽'
        decoded = symbols.decode_sequence(seq)
        assert decoded.shape == (2, 4)


# =============================================================================
# PSYCHE INTERFACE TESTS
# =============================================================================

class TestPsycheInterface:
    """Tests for PsycheInterface."""

    @pytest.fixture
    def interface(self):
        psyche = ZetaPsyche(n_cells=30, hidden_dim=16, M=5)
        return PsycheInterface(psyche)

    def test_process_input_returns_dict(self, interface):
        """process_input should return response dict."""
        resp = interface.process_input("hola", n_steps=5)

        assert 'symbol' in resp
        assert 'dominant' in resp
        assert 'blend' in resp
        assert 'consciousness' in resp

    def test_process_input_symbol_valid(self, interface):
        """Response symbol should be valid."""
        resp = interface.process_input("amor", n_steps=5)
        valid_symbols = ['☉', '☽', '♀', '♂', '◈', '◇', '◆', '●', '○', '◐', '✧']
        assert resp['symbol'] in valid_symbols

    def test_process_different_words(self, interface):
        """Different emotional words should produce different states."""
        resp_love = interface.process_input("amor", n_steps=10)
        resp_fear = interface.process_input("miedo", n_steps=10)

        # States should be different
        assert resp_love['dominant'] != resp_fear['dominant'] or \
               resp_love['blend'] != resp_fear['blend']


# =============================================================================
# GET ZETA ZEROS TESTS
# =============================================================================

class TestGetZetaZeros:
    """Tests for get_zeta_zeros function."""

    def test_returns_tensor(self):
        """Should return tensor."""
        zeros = get_zeta_zeros(5)
        assert isinstance(zeros, torch.Tensor)

    def test_correct_count(self):
        """Should return M zeros."""
        for m in [5, 10, 15, 20]:
            zeros = get_zeta_zeros(m)
            assert len(zeros) == m

    def test_first_zero_correct(self):
        """First zero should be ~14.134725."""
        zeros = get_zeta_zeros(1)
        assert torch.isclose(zeros[0], torch.tensor(14.134725), atol=1e-4)

    def test_zeros_increasing(self):
        """Zeros should be strictly increasing."""
        zeros = get_zeta_zeros(20)
        for i in range(len(zeros) - 1):
            assert zeros[i] < zeros[i + 1]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for ZetaPsyche system."""

    def test_full_session(self):
        """Test a full session with multiple stimuli."""
        psyche = ZetaPsyche(n_cells=50, hidden_dim=32, M=10)

        stimuli = [
            torch.tensor([0.7, 0.1, 0.1, 0.1]),  # PERSONA
            torch.tensor([0.1, 0.7, 0.1, 0.1]),  # SOMBRA
            torch.tensor([0.1, 0.1, 0.7, 0.1]),  # ANIMA
            torch.tensor([0.1, 0.1, 0.1, 0.7]),  # ANIMUS
        ]

        for stimulus in stimuli:
            for _ in range(10):
                obs = psyche.step(stimulus)
                assert 'consciousness_index' in obs

        # Should have history
        assert len(psyche.consciousness_history) == 40

    def test_symbols_encode_trajectory(self):
        """Test that psyche trajectory can be encoded."""
        psyche = ZetaPsyche(n_cells=30, hidden_dim=16, M=5)
        symbols = SymbolSystem()

        trajectory = []
        for _ in range(20):
            obs = psyche.step()
            trajectory.append(obs['global_state'])

        trajectory_tensor = torch.stack(trajectory)
        encoded = symbols.encode_sequence(trajectory_tensor)

        assert len(encoded) == 20
        assert all(c in '☉☽♀♂◈◇◆●○◐✧' for c in encoded)
