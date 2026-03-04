"""Tests for ComplementaryMemory -- dual-speed memory system for the Conscious Kernel.

Covers:
- Episode: creation, to_dict/from_dict roundtrip
- CompressedEpisode: creation, to_dict/from_dict, consolidated flag
- FastMemory: stores surprising (above threshold), ignores unsurprising,
  capacity limit, recall_by_similarity (cosine on archetype_state),
  serialize/restore roundtrip
- SlowMemory: generalize returns correct shape, learns pattern over 200+
  integrations (error < 0.3), state_dict save/load roundtrip
"""

import torch
import pytest

from zeta_life.kernel.complementary_memory import (
    Episode,
    CompressedEpisode,
    FastMemory,
    SlowMemory,
)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

class TestEpisode:
    """Episode dataclass: creation and serialization."""

    def test_creation(self):
        ep = Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=torch.randn(4),
            surprise=0.75,
            dominant="V0",
            timestamp=42,
        )
        assert ep.surprise == 0.75
        assert ep.dominant == "V0"
        assert ep.timestamp == 42

    def test_default_prediction_errors(self):
        ep = Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=torch.randn(4),
            surprise=0.5,
            dominant="V1",
            timestamp=0,
        )
        assert ep.prediction_errors is None

    def test_with_prediction_errors(self):
        ep = Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=torch.randn(4),
            surprise=0.5,
            dominant="V1",
            timestamp=0,
            prediction_errors={"perceptual": 0.3, "temporal": 0.1},
        )
        assert ep.prediction_errors["perceptual"] == 0.3

    def test_tensor_shapes(self):
        stim = torch.randn(8)
        obs = torch.randn(8)
        arch = torch.randn(4)
        ep = Episode(
            stimulus=stim,
            observation=obs,
            archetype_state=arch,
            surprise=0.1,
            dominant="V2",
            timestamp=1,
        )
        assert ep.stimulus.shape == (8,)
        assert ep.observation.shape == (8,)
        assert ep.archetype_state.shape == (4,)

    def test_to_dict(self):
        ep = Episode(
            stimulus=torch.tensor([1.0, 2.0]),
            observation=torch.tensor([3.0, 4.0]),
            archetype_state=torch.tensor([0.5, 0.5]),
            surprise=0.8,
            dominant="V3",
            timestamp=10,
        )
        d = ep.to_dict()
        assert isinstance(d, dict)
        assert "stimulus" in d
        assert "observation" in d
        assert "archetype_state" in d
        assert d["surprise"] == 0.8
        assert d["dominant"] == "V3"
        assert d["timestamp"] == 10

    def test_from_dict(self):
        original = Episode(
            stimulus=torch.tensor([1.0, 2.0]),
            observation=torch.tensor([3.0, 4.0]),
            archetype_state=torch.tensor([0.5, 0.5]),
            surprise=0.65,
            dominant="V0",
            timestamp=7,
        )
        d = original.to_dict()
        restored = Episode.from_dict(d)
        assert isinstance(restored, Episode)
        assert torch.allclose(restored.stimulus, original.stimulus)
        assert torch.allclose(restored.observation, original.observation)
        assert torch.allclose(restored.archetype_state, original.archetype_state)
        assert restored.surprise == original.surprise
        assert restored.dominant == original.dominant
        assert restored.timestamp == original.timestamp

    def test_roundtrip_with_prediction_errors(self):
        original = Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=torch.randn(4),
            surprise=0.9,
            dominant="V1",
            timestamp=99,
            prediction_errors={"perceptual": 0.4, "epistemic": 0.2},
        )
        d = original.to_dict()
        restored = Episode.from_dict(d)
        assert restored.prediction_errors is not None
        assert restored.prediction_errors["perceptual"] == pytest.approx(0.4)
        assert restored.prediction_errors["epistemic"] == pytest.approx(0.2)

    def test_roundtrip_without_prediction_errors(self):
        original = Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=torch.randn(4),
            surprise=0.3,
            dominant="V2",
            timestamp=5,
        )
        d = original.to_dict()
        restored = Episode.from_dict(d)
        assert restored.prediction_errors is None


# ---------------------------------------------------------------------------
# CompressedEpisode
# ---------------------------------------------------------------------------

class TestCompressedEpisode:
    """CompressedEpisode dataclass: creation and serialization."""

    def test_creation(self):
        ce = CompressedEpisode(
            archetype_state=torch.randn(4),
            surprise=0.6,
            dominant="V0",
            timestamp=5,
        )
        assert ce.surprise == 0.6
        assert ce.dominant == "V0"
        assert ce.timestamp == 5

    def test_default_consolidated_false(self):
        ce = CompressedEpisode(
            archetype_state=torch.randn(4),
            surprise=0.5,
            dominant="V1",
            timestamp=0,
        )
        assert ce.consolidated is False

    def test_consolidated_flag(self):
        ce = CompressedEpisode(
            archetype_state=torch.randn(4),
            surprise=0.5,
            dominant="V1",
            timestamp=0,
            consolidated=True,
        )
        assert ce.consolidated is True

    def test_to_dict(self):
        ce = CompressedEpisode(
            archetype_state=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            surprise=0.7,
            dominant="V2",
            timestamp=15,
            consolidated=True,
        )
        d = ce.to_dict()
        assert isinstance(d, dict)
        assert "archetype_state" in d
        assert d["surprise"] == 0.7
        assert d["dominant"] == "V2"
        assert d["timestamp"] == 15
        assert d["consolidated"] is True

    def test_from_dict(self):
        original = CompressedEpisode(
            archetype_state=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            surprise=0.55,
            dominant="V3",
            timestamp=20,
            consolidated=True,
        )
        d = original.to_dict()
        restored = CompressedEpisode.from_dict(d)
        assert isinstance(restored, CompressedEpisode)
        assert torch.allclose(restored.archetype_state, original.archetype_state)
        assert restored.surprise == original.surprise
        assert restored.dominant == original.dominant
        assert restored.timestamp == original.timestamp
        assert restored.consolidated == original.consolidated

    def test_roundtrip_not_consolidated(self):
        original = CompressedEpisode(
            archetype_state=torch.randn(4),
            surprise=0.2,
            dominant="V0",
            timestamp=3,
            consolidated=False,
        )
        d = original.to_dict()
        restored = CompressedEpisode.from_dict(d)
        assert restored.consolidated is False


# ---------------------------------------------------------------------------
# FastMemory
# ---------------------------------------------------------------------------

class TestFastMemoryInit:
    """Verify FastMemory construction."""

    def test_default_capacity(self):
        fm = FastMemory()
        assert fm.capacity == 100

    def test_custom_capacity(self):
        fm = FastMemory(capacity=50)
        assert fm.capacity == 50

    def test_default_threshold(self):
        fm = FastMemory()
        assert fm.surprise_threshold == 0.5

    def test_custom_threshold(self):
        fm = FastMemory(surprise_threshold=0.3)
        assert fm.surprise_threshold == 0.3

    def test_initially_empty(self):
        fm = FastMemory()
        assert len(fm) == 0


class TestFastMemoryStore:
    """store(episode) should only keep surprising episodes."""

    def _make_episode(self, surprise: float, arch_state=None) -> Episode:
        return Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=arch_state if arch_state is not None else torch.randn(4),
            surprise=surprise,
            dominant="V0",
            timestamp=0,
        )

    def test_stores_surprising_episode(self):
        fm = FastMemory(surprise_threshold=0.5)
        ep = self._make_episode(surprise=0.8)
        fm.store(ep)
        assert len(fm) == 1

    def test_ignores_unsurprising_episode(self):
        fm = FastMemory(surprise_threshold=0.5)
        ep = self._make_episode(surprise=0.2)
        fm.store(ep)
        assert len(fm) == 0

    def test_stores_at_threshold(self):
        fm = FastMemory(surprise_threshold=0.5)
        ep = self._make_episode(surprise=0.5)
        fm.store(ep)
        assert len(fm) == 1

    def test_stores_above_threshold(self):
        fm = FastMemory(surprise_threshold=0.5)
        ep = self._make_episode(surprise=0.51)
        fm.store(ep)
        assert len(fm) == 1

    def test_does_not_store_below_threshold(self):
        fm = FastMemory(surprise_threshold=0.5)
        ep = self._make_episode(surprise=0.49)
        fm.store(ep)
        assert len(fm) == 0

    def test_capacity_limit(self):
        fm = FastMemory(capacity=5, surprise_threshold=0.0)
        for i in range(10):
            ep = self._make_episode(surprise=0.8)
            fm.store(ep)
        assert len(fm) == 5

    def test_capacity_fifo(self):
        """When capacity is exceeded, oldest episodes are dropped (FIFO)."""
        fm = FastMemory(capacity=3, surprise_threshold=0.0)
        for i in range(5):
            ep = Episode(
                stimulus=torch.randn(4),
                observation=torch.randn(4),
                archetype_state=torch.randn(4),
                surprise=1.0,
                dominant=f"V{i % 4}",
                timestamp=i,
            )
            fm.store(ep)
        # Only the last 3 should remain
        assert len(fm) == 3

    def test_multiple_stores(self):
        fm = FastMemory(capacity=100, surprise_threshold=0.5)
        for i in range(20):
            ep = self._make_episode(surprise=0.6 + i * 0.01)
            fm.store(ep)
        assert len(fm) == 20


class TestFastMemoryRecall:
    """recall_by_similarity should find episodes closest to query state."""

    def _make_episode(self, arch_state: torch.Tensor, surprise: float = 0.8) -> Episode:
        return Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=arch_state,
            surprise=surprise,
            dominant="V0",
            timestamp=0,
        )

    def test_returns_list(self):
        fm = FastMemory(surprise_threshold=0.0)
        fm.store(self._make_episode(torch.tensor([1.0, 0.0, 0.0, 0.0])))
        results = fm.recall_by_similarity(torch.tensor([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert isinstance(results, list)

    def test_returns_correct_k(self):
        fm = FastMemory(surprise_threshold=0.0)
        for _ in range(10):
            fm.store(self._make_episode(torch.randn(4)))
        results = fm.recall_by_similarity(torch.randn(4), top_k=3)
        assert len(results) == 3

    def test_returns_fewer_when_not_enough(self):
        fm = FastMemory(surprise_threshold=0.0)
        fm.store(self._make_episode(torch.randn(4)))
        results = fm.recall_by_similarity(torch.randn(4), top_k=5)
        assert len(results) == 1

    def test_empty_memory_returns_empty(self):
        fm = FastMemory()
        results = fm.recall_by_similarity(torch.randn(4), top_k=3)
        assert results == []

    def test_most_similar_first(self):
        fm = FastMemory(surprise_threshold=0.0)

        # Store three episodes with known archetype states
        target = torch.tensor([1.0, 0.0, 0.0, 0.0])
        close = torch.tensor([0.9, 0.1, 0.0, 0.0])
        far = torch.tensor([0.0, 0.0, 0.0, 1.0])

        fm.store(self._make_episode(far))
        fm.store(self._make_episode(close))

        results = fm.recall_by_similarity(target, top_k=2)
        # The closest one should be first (higher cosine similarity)
        sim_first = torch.nn.functional.cosine_similarity(
            results[0].archetype_state.unsqueeze(0), target.unsqueeze(0)
        )
        sim_second = torch.nn.functional.cosine_similarity(
            results[1].archetype_state.unsqueeze(0), target.unsqueeze(0)
        )
        assert sim_first >= sim_second

    def test_returns_compressed_episodes(self):
        fm = FastMemory(surprise_threshold=0.0)
        fm.store(self._make_episode(torch.randn(4)))
        results = fm.recall_by_similarity(torch.randn(4), top_k=1)
        assert isinstance(results[0], CompressedEpisode)


class TestFastMemorySerialize:
    """serialize/restore roundtrip should preserve memory contents."""

    def _make_episode(self, surprise: float = 0.8) -> Episode:
        return Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=torch.randn(4),
            surprise=surprise,
            dominant="V0",
            timestamp=0,
        )

    def test_serialize_returns_dict(self):
        fm = FastMemory(surprise_threshold=0.0)
        fm.store(self._make_episode())
        data = fm.serialize()
        assert isinstance(data, dict)

    def test_restore_recovers_episodes(self):
        fm = FastMemory(capacity=10, surprise_threshold=0.3)
        for _ in range(5):
            fm.store(self._make_episode(surprise=0.9))
        data = fm.serialize()

        fm2 = FastMemory.restore(data)
        assert len(fm2) == len(fm)

    def test_roundtrip_preserves_config(self):
        fm = FastMemory(capacity=42, surprise_threshold=0.7)
        data = fm.serialize()
        fm2 = FastMemory.restore(data)
        assert fm2.capacity == 42
        assert fm2.surprise_threshold == 0.7

    def test_roundtrip_preserves_archetype_states(self):
        fm = FastMemory(surprise_threshold=0.0)
        arch = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ep = Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=arch,
            surprise=0.9,
            dominant="V0",
            timestamp=0,
        )
        fm.store(ep)
        data = fm.serialize()
        fm2 = FastMemory.restore(data)

        # Recall should return episode with same archetype state
        results = fm2.recall_by_similarity(arch, top_k=1)
        assert len(results) == 1
        assert torch.allclose(results[0].archetype_state, arch)

    def test_empty_serialize_restore(self):
        fm = FastMemory()
        data = fm.serialize()
        fm2 = FastMemory.restore(data)
        assert len(fm2) == 0


# ---------------------------------------------------------------------------
# SlowMemory
# ---------------------------------------------------------------------------

class TestSlowMemoryInit:
    """Verify SlowMemory construction."""

    def test_creation(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        assert isinstance(sm, torch.nn.Module)

    def test_has_knowledge_network(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        assert hasattr(sm, "knowledge")

    def test_has_optimizer(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        assert hasattr(sm, "optimizer")

    def test_default_learning_rate(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        assert sm.lr == 0.0001


class TestSlowMemoryGeneralize:
    """generalize(query) should return tensor of outcome_dim shape."""

    def test_returns_tensor(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        query = torch.randn(8)
        result = sm.generalize(query)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        query = torch.randn(8)
        result = sm.generalize(query)
        assert result.shape == (4,)

    def test_custom_dims(self):
        sm = SlowMemory(context_dim=16, outcome_dim=8)
        query = torch.randn(16)
        result = sm.generalize(query)
        assert result.shape == (8,)

    def test_no_grad(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        query = torch.randn(8)
        result = sm.generalize(query)
        assert not result.requires_grad

    def test_deterministic(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        sm.eval()
        query = torch.randn(8)
        r1 = sm.generalize(query)
        r2 = sm.generalize(query)
        assert torch.allclose(r1, r2)


class TestSlowMemoryIntegrate:
    """integrate(context, outcome) should learn from examples."""

    def test_returns_float_loss(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        context = torch.randn(8)
        outcome = torch.randn(4)
        loss = sm.integrate(context, outcome)
        assert isinstance(loss, float)

    def test_loss_is_nonnegative(self):
        sm = SlowMemory(context_dim=8, outcome_dim=4)
        loss = sm.integrate(torch.randn(8), torch.randn(4))
        assert loss >= 0.0

    def test_learning_rate_scale(self):
        """With scale=0, no learning should occur."""
        sm = SlowMemory(context_dim=4, outcome_dim=4)
        params_before = {
            name: p.data.clone() for name, p in sm.knowledge.named_parameters()
        }
        sm.integrate(torch.randn(4), torch.randn(4), learning_rate_scale=0.0)
        for name, p in sm.knowledge.named_parameters():
            assert torch.allclose(p.data, params_before[name]), (
                f"Parameter {name} should not change with scale=0"
            )

    def test_learns_pattern_over_200_integrations(self):
        """After 200+ integrations on the same pattern, error should be < 0.3."""
        torch.manual_seed(42)
        sm = SlowMemory(context_dim=4, outcome_dim=4, learning_rate=0.005)

        # Fixed pattern: context -> outcome
        context = torch.tensor([1.0, 0.0, -1.0, 0.5])
        outcome = torch.tensor([0.5, 0.5, 0.0, -0.5])

        for _ in range(300):
            sm.integrate(context, outcome)

        # After training, generalization should be close to outcome
        predicted = sm.generalize(context)
        error = torch.norm(predicted - outcome).item()
        assert error < 0.3, (
            f"After 300 integrations, error should be < 0.3, got {error:.4f}"
        )


class TestSlowMemoryStateDictRoundtrip:
    """state_dict save/load should preserve learned knowledge."""

    def test_state_dict_save_load(self):
        torch.manual_seed(42)
        sm1 = SlowMemory(context_dim=4, outcome_dim=4, learning_rate=0.001)

        # Train on a pattern
        context = torch.tensor([1.0, 0.0, -1.0, 0.5])
        outcome = torch.tensor([0.5, 0.5, 0.0, -0.5])
        for _ in range(100):
            sm1.integrate(context, outcome)

        # Save and restore
        state = sm1.state_dict()
        sm2 = SlowMemory(context_dim=4, outcome_dim=4, learning_rate=0.001)
        sm2.load_state_dict(state)

        # Both should produce the same output
        pred1 = sm1.generalize(context)
        pred2 = sm2.generalize(context)
        assert torch.allclose(pred1, pred2, atol=1e-6)

    def test_state_dict_is_dict(self):
        sm = SlowMemory(context_dim=4, outcome_dim=4)
        state = sm.state_dict()
        assert isinstance(state, dict)
        assert len(state) > 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestComplementaryMemoryIntegration:
    """End-to-end scenarios combining FastMemory and SlowMemory."""

    def test_fast_stores_slow_learns(self):
        """Surprising episodes go to fast memory; slow memory learns patterns."""
        fast = FastMemory(capacity=50, surprise_threshold=0.5)
        slow = SlowMemory(context_dim=4, outcome_dim=4, learning_rate=0.001)

        for i in range(100):
            stim = torch.randn(4)
            obs = torch.randn(4)
            arch = torch.randn(4)
            surprise = 0.3 + 0.5 * (i % 2)  # alternating 0.3 and 0.8

            ep = Episode(
                stimulus=stim,
                observation=obs,
                archetype_state=arch,
                surprise=surprise,
                dominant="V0",
                timestamp=i,
            )
            fast.store(ep)
            slow.integrate(stim, obs)

        # Fast memory should have ~50 episodes (the surprising ones)
        assert len(fast) == 50

    def test_fast_recall_feeds_slow(self):
        """Recalled episodes from fast memory can be used as slow memory input."""
        fast = FastMemory(capacity=50, surprise_threshold=0.0)
        slow = SlowMemory(context_dim=4, outcome_dim=4, learning_rate=0.001)

        # Store episodes
        target_arch = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for i in range(10):
            ep = Episode(
                stimulus=torch.randn(4),
                observation=torch.randn(4),
                archetype_state=target_arch + torch.randn(4) * 0.1,
                surprise=0.8,
                dominant="V0",
                timestamp=i,
            )
            fast.store(ep)

        # Recall similar episodes
        recalled = fast.recall_by_similarity(target_arch, top_k=5)
        assert len(recalled) == 5

        # Use recalled archetype states as context for slow memory
        for ce in recalled:
            slow.integrate(ce.archetype_state, target_arch)

        # Slow memory should not crash and should produce valid output
        result = slow.generalize(target_arch)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()
