"""Tests for OrganismState."""
import torch
import pytest
from collections import deque
from unittest.mock import MagicMock

from zeta_life.kernel.organism_state import OrganismState


def _mock_kernel(embed: torch.Tensor, action: torch.Tensor, energy: float = 5.0):
    k = MagicMock()
    k.self_model = MagicMock()
    k.self_model.self_embedding = MagicMock()
    k.self_model.self_embedding.data = embed
    k.last_action = action
    k.energy = energy
    return k


def _mock_gw(history: list[int]):
    gw = MagicMock()
    gw.history = deque(history, maxlen=100)
    gw.spotlight_owner = history[-1] if history else None
    return gw


class TestInit:
    def test_creation(self):
        state = OrganismState()
        assert state.integration_index == 0.0


class TestDiversity:
    def test_identical_kernels_zero_diversity(self):
        state = OrganismState()
        embed = torch.tensor([1.0, 0.0, 0.0, 0.0])
        kernels = {
            0: _mock_kernel(embed.clone(), torch.zeros(4)),
            1: _mock_kernel(embed.clone(), torch.zeros(4)),
        }
        state.update(kernels, _mock_gw([0]))
        assert state.diversity < 0.05

    def test_different_kernels_high_diversity(self):
        state = OrganismState()
        kernels = {
            0: _mock_kernel(torch.tensor([1.0, 0., 0., 0.]), torch.zeros(4)),
            1: _mock_kernel(torch.tensor([0., 0., 0., 1.0]), torch.zeros(4)),
        }
        state.update(kernels, _mock_gw([0]))
        assert state.diversity > 0.3


class TestCoherence:
    def test_same_actions_high_coherence(self):
        state = OrganismState()
        action = torch.tensor([0.5, 0.2, 0.2, 0.1])
        kernels = {
            0: _mock_kernel(torch.randn(4), action.clone()),
            1: _mock_kernel(torch.randn(4), action.clone()),
        }
        state.update(kernels, _mock_gw([0]))
        assert state.coherence > 0.9

    def test_different_actions_low_coherence(self):
        state = OrganismState()
        kernels = {
            0: _mock_kernel(torch.randn(4), torch.tensor([1., 0., 0., 0.])),
            1: _mock_kernel(torch.randn(4), torch.tensor([0., 0., 0., 1.])),
        }
        state.update(kernels, _mock_gw([0]))
        assert state.coherence < 0.5


class TestTurnover:
    def test_monopoly_low_turnover(self):
        state = OrganismState()
        kernels = {0: _mock_kernel(torch.randn(4), torch.randn(4)),
                   1: _mock_kernel(torch.randn(4), torch.randn(4))}
        gw = _mock_gw([0] * 20)  # same winner 20 times
        state.update(kernels, gw)
        assert state.turnover < 0.2

    def test_balanced_high_turnover(self):
        state = OrganismState()
        kernels = {0: _mock_kernel(torch.randn(4), torch.randn(4)),
                   1: _mock_kernel(torch.randn(4), torch.randn(4))}
        gw = _mock_gw([0, 1] * 10)  # alternating
        state.update(kernels, gw)
        assert state.turnover > 0.5


class TestConsciousnessIndex:
    def test_in_zero_one_range(self):
        state = OrganismState()
        kernels = {0: _mock_kernel(torch.randn(4), torch.randn(4)),
                   1: _mock_kernel(torch.randn(4), torch.randn(4))}
        state.update(kernels, _mock_gw([0, 1, 0]))
        assert 0.0 <= state.integration_index <= 1.0
