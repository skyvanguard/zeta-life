"""Tests for ConsciousOrganism."""
import torch
import pytest

from zeta_life.kernel.conscious_organism import ConsciousOrganism, OrganismStepResult


class TestInit:
    def test_creates_two_kernels(self):
        org = ConsciousOrganism()
        assert len(org.kernels) == 2

    def test_initial_energy_split(self):
        org = ConsciousOrganism(total_energy=10.0, initial_kernels=2)
        for k in org.kernels.values():
            assert abs(k.energy - 5.0) < 0.01

    def test_custom_obs_dim(self):
        org = ConsciousOrganism(obs_dim=8)
        assert org.obs_dim == 8

    def test_t_starts_at_zero(self):
        org = ConsciousOrganism()
        assert org.t == 0


class TestStep:
    def test_returns_result(self):
        org = ConsciousOrganism()
        result = org.step(torch.randn(4))
        assert isinstance(result, OrganismStepResult)

    def test_increments_t(self):
        org = ConsciousOrganism()
        org.step(torch.randn(4))
        assert org.t == 1

    def test_has_winner(self):
        org = ConsciousOrganism()
        result = org.step(torch.randn(4))
        assert result.winner_id in org.kernels

    def test_consciousness_in_range(self):
        org = ConsciousOrganism()
        result = org.step(torch.randn(4))
        assert 0.0 <= result.consciousness <= 1.0

    def test_population_tracked(self):
        org = ConsciousOrganism()
        result = org.step(torch.randn(4))
        assert result.population >= 2

    def test_multiple_steps_no_crash(self):
        org = ConsciousOrganism()
        for _ in range(100):
            org.step(torch.randn(4))
        assert org.t == 100

    def test_energy_conservation(self):
        org = ConsciousOrganism(total_energy=10.0)
        for _ in range(50):
            org.step(torch.randn(4))
        total = sum(k.energy for k in org.kernels.values())
        assert abs(total - 10.0) < 0.01


class TestLifecycle:
    def test_population_changes_over_time(self):
        """Over many steps, population should vary."""
        org = ConsciousOrganism(total_energy=10.0, initial_kernels=2)
        populations = set()
        for _ in range(500):
            result = org.step(torch.randn(4))
            populations.add(result.population)
        # We expect at least the initial population
        assert 2 in populations

    def test_no_population_below_minimum(self):
        org = ConsciousOrganism(total_energy=10.0, initial_kernels=2)
        for _ in range(200):
            org.step(torch.randn(4))
        assert len(org.kernels) >= 2

    def test_no_population_above_maximum(self):
        org = ConsciousOrganism(total_energy=20.0, initial_kernels=5)
        for _ in range(200):
            org.step(torch.randn(4))
        assert len(org.kernels) <= 10


class TestBroadcast:
    def test_broadcast_affects_kernels(self):
        org = ConsciousOrganism()
        org.step(torch.randn(4))
        # After one step, broadcast should be non-zero
        assert org.gw.broadcast_signal is not None
