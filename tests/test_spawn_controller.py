"""Tests for SpawnController."""
import torch
import pytest
from unittest.mock import MagicMock

from zeta_life.kernel.spawn_controller import (
    SpawnController, LifecycleEvent, SpawnEvent, MergeEvent, DeathEvent,
)


def _mock_kernel(energy: float = 5.0, t: int = 200, embed_dim: int = 16):
    k = MagicMock()
    k.energy = energy
    k.t = t
    k.self_model = MagicMock()
    k.self_model.self_embedding = MagicMock()
    k.self_model.self_embedding.data = torch.randn(embed_dim)
    k.obs_dim = 4
    return k


class TestInit:
    def test_defaults(self):
        sc = SpawnController()
        assert sc.min_kernels == 2
        assert sc.max_kernels == 10


class TestEvaluateSpawn:
    def test_spawn_when_energy_high(self):
        sc = SpawnController(spawn_energy=7.0, min_age=100)
        kernels = {0: _mock_kernel(8.0, t=200), 1: _mock_kernel(4.0)}
        events = sc.evaluate(kernels)
        spawn_events = [e for e in events if isinstance(e, SpawnEvent)]
        assert len(spawn_events) == 1
        assert spawn_events[0].parent_id == 0

    def test_no_spawn_if_immature(self):
        sc = SpawnController(spawn_energy=7.0, min_age=100)
        kernels = {0: _mock_kernel(8.0, t=50), 1: _mock_kernel(4.0)}
        events = sc.evaluate(kernels)
        spawn_events = [e for e in events if isinstance(e, SpawnEvent)]
        assert len(spawn_events) == 0

    def test_no_spawn_at_max_population(self):
        sc = SpawnController(max_kernels=2)
        kernels = {0: _mock_kernel(8.0), 1: _mock_kernel(8.0)}
        events = sc.evaluate(kernels)
        spawn_events = [e for e in events if isinstance(e, SpawnEvent)]
        assert len(spawn_events) == 0


class TestEvaluateDeath:
    def test_death_when_energy_low(self):
        sc = SpawnController(death_energy=1.0, min_kernels=2)
        kernels = {0: _mock_kernel(0.5), 1: _mock_kernel(5.0), 2: _mock_kernel(5.0)}
        events = sc.evaluate(kernels)
        death_events = [e for e in events if isinstance(e, DeathEvent)]
        assert len(death_events) == 1
        assert death_events[0].kernel_id == 0

    def test_no_death_at_min_population(self):
        sc = SpawnController(death_energy=1.0, min_kernels=2)
        kernels = {0: _mock_kernel(0.5), 1: _mock_kernel(5.0)}
        events = sc.evaluate(kernels)
        death_events = [e for e in events if isinstance(e, DeathEvent)]
        assert len(death_events) == 0


class TestEvaluateMerge:
    def test_merge_when_similar(self):
        sc = SpawnController(merge_similarity=0.95)
        k0 = _mock_kernel(2.5)
        k1 = _mock_kernel(2.5)
        # Make embeddings nearly identical
        embed = torch.randn(16)
        k0.self_model.self_embedding.data = embed.clone()
        k1.self_model.self_embedding.data = embed.clone() + torch.randn(16) * 0.01
        kernels = {0: k0, 1: k1, 2: _mock_kernel(5.0)}
        events = sc.evaluate(kernels)
        merge_events = [e for e in events if isinstance(e, MergeEvent)]
        assert len(merge_events) >= 1

    def test_no_merge_when_different(self):
        sc = SpawnController(merge_similarity=0.95)
        kernels = {0: _mock_kernel(2.5), 1: _mock_kernel(2.5)}
        # Random embeddings are unlikely to be >0.95 similar
        events = sc.evaluate(kernels)
        merge_events = [e for e in events if isinstance(e, MergeEvent)]
        assert len(merge_events) == 0
