"""Tests for EnergyPool."""
import pytest
from unittest.mock import MagicMock

from zeta_life.kernel.energy_pool import EnergyPool


def _mock_kernel(energy: float = 5.0, free_energy: float = 0.3,
                 fast_memory_len: int = 10):
    k = MagicMock()
    k.energy = energy
    k._last_result = MagicMock()
    k._last_result.free_energy = free_energy
    k._last_result.dreamed = False
    k.fast_memory = MagicMock()
    k.fast_memory.__len__ = MagicMock(return_value=fast_memory_len)
    return k


class TestInit:
    def test_total_energy(self):
        ep = EnergyPool(total_energy=10.0)
        assert ep.total_energy == 10.0

    def test_default_params(self):
        ep = EnergyPool()
        assert ep.metabolic_cost == 0.01
        assert ep.memory_cost == 0.005


class TestRewardWinner:
    def test_winner_gains_energy(self):
        ep = EnergyPool(total_energy=10.0)
        kernels = {0: _mock_kernel(5.0, 0.2), 1: _mock_kernel(5.0, 0.4)}
        old_energy_0 = kernels[0].energy
        ep.reward_winner(0, kernels)
        assert kernels[0].energy > old_energy_0

    def test_losers_lose_energy(self):
        ep = EnergyPool(total_energy=10.0)
        kernels = {0: _mock_kernel(5.0, 0.2), 1: _mock_kernel(5.0, 0.4)}
        old_energy_1 = kernels[1].energy
        ep.reward_winner(0, kernels)
        assert kernels[1].energy < old_energy_1

    def test_conservation_after_reward(self):
        ep = EnergyPool(total_energy=10.0)
        kernels = {0: _mock_kernel(5.0, 0.2), 1: _mock_kernel(5.0, 0.4)}
        ep.reward_winner(0, kernels)
        total = sum(k.energy for k in kernels.values())
        assert abs(total - 10.0) < 0.01


class TestDecayAll:
    def test_all_lose_metabolic_cost(self):
        ep = EnergyPool(total_energy=10.0)
        kernels = {0: _mock_kernel(5.0), 1: _mock_kernel(5.0)}
        ep.decay_all(kernels)
        for k in kernels.values():
            assert k.energy < 5.0

    def test_memory_cost_proportional(self):
        ep = EnergyPool(total_energy=10.0)
        k_few = _mock_kernel(5.0, fast_memory_len=10)
        k_many = _mock_kernel(5.0, fast_memory_len=100)
        kernels = {0: k_few, 1: k_many}
        ep.decay_all(kernels)
        assert k_many.energy < k_few.energy

    def test_dream_bonus(self):
        ep = EnergyPool(total_energy=10.0)
        k = _mock_kernel(5.0)
        k._last_result.dreamed = True
        kernels = {0: k}
        ep.decay_all(kernels)
        # Dream bonus partially offsets metabolic cost
        assert k.energy > 5.0 - ep.metabolic_cost


class TestNormalize:
    def test_enforces_conservation(self):
        ep = EnergyPool(total_energy=10.0)
        kernels = {0: _mock_kernel(7.0), 1: _mock_kernel(8.0)}
        ep.normalize(kernels)
        total = sum(k.energy for k in kernels.values())
        assert abs(total - 10.0) < 1e-6

    def test_preserves_ratios(self):
        ep = EnergyPool(total_energy=10.0)
        kernels = {0: _mock_kernel(6.0), 1: _mock_kernel(4.0)}
        ep.normalize(kernels)
        # Ratio should be preserved: 6:4 = 0.6:0.4
        assert abs(kernels[0].energy / 10.0 - 0.6) < 0.01
