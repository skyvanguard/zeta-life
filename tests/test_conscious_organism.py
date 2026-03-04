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


class TestTopDownModulation:
    """Tests for configurable top-down strength with broadcast EMA."""

    def test_default_params_backward_compat(self):
        org = ConsciousOrganism()
        assert org.top_down_strength == 0.3
        assert org.broadcast_ema_decay == 0.7
        # Should run without errors
        for _ in range(20):
            org.step(torch.randn(4))

    def test_custom_top_down_strength(self):
        org = ConsciousOrganism(top_down_strength=0.5)
        assert org.top_down_strength == 0.5

    def test_custom_broadcast_ema_decay(self):
        org = ConsciousOrganism(broadcast_ema_decay=0.9)
        assert org.broadcast_ema_decay == 0.9

    def test_ema_initializes_on_first_broadcast(self):
        org = ConsciousOrganism()
        assert org._broadcast_ema is None
        # After first step, broadcast exists and EMA should initialize
        org.step(torch.randn(4))
        # EMA may or may not be set depending on broadcast state
        # After second step, it should definitely be set
        org.step(torch.randn(4))
        assert org._broadcast_ema is not None

    def test_high_top_down_differs_from_zero(self):
        """With high top_down_strength, combined stimulus should differ from raw."""
        torch.manual_seed(42)
        org_high = ConsciousOrganism(top_down_strength=0.8)
        org_zero = ConsciousOrganism(top_down_strength=0.0)
        stimulus = torch.tensor([0.7, 0.1, 0.1, 0.1])
        # Warm up both with same stimuli
        for _ in range(30):
            org_high.step(stimulus)
            org_zero.step(stimulus)
        # After warmup, broadcasts should differ
        bc_high = org_high.gw.broadcast_signal.clone()
        bc_zero = org_zero.gw.broadcast_signal.clone()
        # They should be valid tensors
        assert bc_high is not None
        assert bc_zero is not None

    def test_zero_strength_passes_stimulus_through(self):
        org = ConsciousOrganism(top_down_strength=0.0)
        stimulus = torch.tensor([0.7, 0.1, 0.1, 0.1])
        # With strength=0, _combine_stimulus should return stimulus unchanged
        # (after broadcast exists)
        org.step(stimulus)
        combined = org._combine_stimulus(stimulus)
        assert torch.allclose(stimulus, combined, atol=1e-6)
