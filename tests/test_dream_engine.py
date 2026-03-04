"""Tests for DreamEngine -- zeta-driven sleep consolidation for the Conscious Kernel.

Covers:
- zeta_kernel(t): returns float, varies with time, positive at t=0
- phase_from_kernel(t): returns one of ('slow_oscillation', 'spindle', 'ripple')
- dream_cycle(duration): returns DreamReport with transfers/replays/selections > 0
- dream_cycle with empty memory: returns report with zeros
- select_for_replay(): sorts by surprise descending, skips consolidated
"""

import torch
import pytest

from zeta_life.kernel.dream_engine import DreamEngine, DreamReport
from zeta_life.kernel.complementary_memory import (
    Episode,
    FastMemory,
    SlowMemory,
)
from zeta_life.kernel.self_model import SelfModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(
    surprise: float = 0.8,
    arch_state: torch.Tensor | None = None,
    timestamp: int = 0,
    consolidated: bool = False,
) -> Episode:
    """Create a test Episode with controllable surprise and archetype state."""
    return Episode(
        stimulus=torch.randn(4),
        observation=torch.randn(4),
        archetype_state=arch_state if arch_state is not None else torch.randn(4),
        surprise=surprise,
        dominant="V0",
        timestamp=timestamp,
    )


def _populated_fast_memory(n: int = 10) -> FastMemory:
    """Return a FastMemory pre-loaded with *n* episodes of varying surprise."""
    fm = FastMemory(capacity=100, surprise_threshold=0.0)
    for i in range(n):
        fm.store(_make_episode(surprise=0.1 * (i + 1), timestamp=i))
    return fm


def _build_engine(n_episodes: int = 10) -> DreamEngine:
    """Build a DreamEngine with populated fast memory for convenience."""
    fast = _populated_fast_memory(n_episodes)
    slow = SlowMemory(context_dim=4, outcome_dim=4)
    self_model = SelfModel(state_dim=4, embed_dim=16)
    return DreamEngine(
        fast_memory=fast,
        slow_memory=slow,
        self_model=self_model,
        sigma=0.1,
        M=15,
    )


def _build_empty_engine() -> DreamEngine:
    """Build a DreamEngine with completely empty fast memory."""
    fast = FastMemory(capacity=100, surprise_threshold=0.5)
    slow = SlowMemory(context_dim=4, outcome_dim=4)
    self_model = SelfModel(state_dim=4, embed_dim=16)
    return DreamEngine(
        fast_memory=fast,
        slow_memory=slow,
        self_model=self_model,
        sigma=0.1,
        M=15,
    )


# ---------------------------------------------------------------------------
# DreamReport dataclass
# ---------------------------------------------------------------------------

class TestDreamReport:
    """DreamReport should hold all dream cycle statistics."""

    def test_creation(self):
        dr = DreamReport(
            duration=50,
            selections=5,
            transfers=3,
            replays=2,
            total_loss=0.42,
            identity_updated=True,
            phases_visited={"slow_oscillation": 10, "spindle": 20, "ripple": 20},
        )
        assert dr.duration == 50
        assert dr.selections == 5
        assert dr.transfers == 3
        assert dr.replays == 2
        assert dr.total_loss == pytest.approx(0.42)
        assert dr.identity_updated is True
        assert "slow_oscillation" in dr.phases_visited

    def test_default_phases_empty(self):
        dr = DreamReport(
            duration=10,
            selections=0,
            transfers=0,
            replays=0,
            total_loss=0.0,
            identity_updated=False,
            phases_visited={},
        )
        assert dr.phases_visited == {}


# ---------------------------------------------------------------------------
# zeta_kernel
# ---------------------------------------------------------------------------

class TestZetaKernel:
    """zeta_kernel(t) = 2 * sum(exp(-sigma*|g|) * cos(g*t))."""

    def test_returns_float(self):
        engine = _build_engine()
        result = engine.zeta_kernel(0.0)
        assert isinstance(result, float)

    def test_positive_at_t_zero(self):
        """At t=0 all cos terms are 1, so kernel must be positive."""
        engine = _build_engine()
        k0 = engine.zeta_kernel(0.0)
        assert k0 > 0.0

    def test_varies_with_time(self):
        """Kernel should not be constant -- different times yield different values."""
        engine = _build_engine()
        k0 = engine.zeta_kernel(0.0)
        k1 = engine.zeta_kernel(1.0)
        k2 = engine.zeta_kernel(2.0)
        # At least one must differ
        assert not (k0 == k1 == k2), "Kernel should vary over time"

    def test_finite_values(self):
        """Kernel should be finite for a range of time values."""
        engine = _build_engine()
        for t in range(100):
            k = engine.zeta_kernel(float(t))
            assert isinstance(k, float)
            assert abs(k) < 1e6, f"Kernel blew up at t={t}: {k}"

    def test_different_sigma(self):
        """Different sigma should produce different kernel magnitudes."""
        fast = _populated_fast_memory(5)
        slow = SlowMemory(context_dim=4, outcome_dim=4)
        sm = SelfModel(state_dim=4, embed_dim=16)

        eng1 = DreamEngine(fast, slow, sm, sigma=0.05, M=15)
        eng2 = DreamEngine(fast, slow, sm, sigma=0.5, M=15)

        k1 = eng1.zeta_kernel(0.0)
        k2 = eng2.zeta_kernel(0.0)
        # Higher sigma -> more decay -> smaller kernel at t=0
        assert k1 > k2


# ---------------------------------------------------------------------------
# phase_from_kernel
# ---------------------------------------------------------------------------

class TestPhaseFromKernel:
    """phase_from_kernel(t) should map kernel value to a sleep phase."""

    VALID_PHASES = {"slow_oscillation", "spindle", "ripple"}

    def test_returns_valid_phase(self):
        engine = _build_engine()
        for t in range(50):
            phase = engine.phase_from_kernel(float(t))
            assert phase in self.VALID_PHASES, f"Invalid phase: {phase}"

    def test_slow_oscillation_at_t_zero(self):
        """At t=0, kernel is large and positive -> slow_oscillation."""
        engine = _build_engine()
        phase = engine.phase_from_kernel(0.0)
        assert phase == "slow_oscillation"

    def test_all_three_phases_reachable(self):
        """Over enough time, all three phases should appear."""
        engine = _build_engine()
        phases_seen: set[str] = set()
        for t in range(200):
            phase = engine.phase_from_kernel(float(t) * 0.1)
            phases_seen.add(phase)
        assert phases_seen == self.VALID_PHASES, (
            f"Not all phases reached, only saw: {phases_seen}"
        )


# ---------------------------------------------------------------------------
# dream_cycle
# ---------------------------------------------------------------------------

class TestDreamCycle:
    """dream_cycle(duration) should replay memories and consolidate."""

    def test_returns_dream_report(self):
        engine = _build_engine()
        report = engine.dream_cycle(duration=30)
        assert isinstance(report, DreamReport)

    def test_report_duration_matches(self):
        engine = _build_engine()
        report = engine.dream_cycle(duration=42)
        assert report.duration == 42

    def test_nonempty_memory_has_positive_counters(self):
        """With memories to replay, we expect selections/transfers/replays > 0."""
        engine = _build_engine(n_episodes=10)
        report = engine.dream_cycle(duration=50)
        assert report.selections > 0
        assert report.transfers > 0
        assert report.replays > 0

    def test_phases_visited_dict(self):
        engine = _build_engine()
        report = engine.dream_cycle(duration=50)
        assert isinstance(report.phases_visited, dict)
        assert sum(report.phases_visited.values()) == 50

    def test_total_loss_nonnegative(self):
        engine = _build_engine()
        report = engine.dream_cycle(duration=30)
        assert report.total_loss >= 0.0

    def test_increments_total_dreams(self):
        engine = _build_engine()
        assert engine.total_dreams == 0
        engine.dream_cycle(duration=10)
        assert engine.total_dreams == 1
        engine.dream_cycle(duration=10)
        assert engine.total_dreams == 2


class TestDreamCycleEmpty:
    """dream_cycle with empty memory should return a report with zeros."""

    def test_empty_memory_returns_zeros(self):
        engine = _build_empty_engine()
        report = engine.dream_cycle(duration=30)
        assert report.selections == 0
        assert report.transfers == 0
        assert report.replays == 0

    def test_empty_memory_total_loss_zero(self):
        engine = _build_empty_engine()
        report = engine.dream_cycle(duration=20)
        assert report.total_loss == pytest.approx(0.0)

    def test_empty_memory_identity_not_updated(self):
        engine = _build_empty_engine()
        report = engine.dream_cycle(duration=20)
        assert report.identity_updated is False

    def test_empty_memory_phases_still_tracked(self):
        engine = _build_empty_engine()
        report = engine.dream_cycle(duration=20)
        assert isinstance(report.phases_visited, dict)
        assert sum(report.phases_visited.values()) == 20


# ---------------------------------------------------------------------------
# select_for_replay
# ---------------------------------------------------------------------------

class TestSelectForReplay:
    """select_for_replay() sorts by surprise descending, skips consolidated."""

    def test_returns_list(self):
        engine = _build_engine()
        selected = engine.select_for_replay()
        assert isinstance(selected, list)

    def test_sorted_by_surprise_descending(self):
        engine = _build_engine(n_episodes=10)
        selected = engine.select_for_replay()
        surprises = [ep.surprise for ep in selected]
        for i in range(len(surprises) - 1):
            assert surprises[i] >= surprises[i + 1], (
                f"Not sorted: {surprises[i]} < {surprises[i + 1]}"
            )

    def test_skips_consolidated(self):
        """Consolidated episodes should NOT appear in the selection."""
        fast = FastMemory(capacity=100, surprise_threshold=0.0)
        # Store episodes: mark some as needing consolidation tracking
        for i in range(5):
            fast.store(_make_episode(surprise=0.8, timestamp=i))

        slow = SlowMemory(context_dim=4, outcome_dim=4)
        sm = SelfModel(state_dim=4, embed_dim=16)
        engine = DreamEngine(fast, slow, sm, sigma=0.1, M=15)

        # Run a dream cycle to consolidate some memories
        engine.dream_cycle(duration=50)

        # After consolidation, select_for_replay should skip consolidated ones
        selected = engine.select_for_replay()
        for ep in selected:
            assert not ep.consolidated, "Consolidated episode should be skipped"

    def test_empty_memory_returns_empty(self):
        engine = _build_empty_engine()
        selected = engine.select_for_replay()
        assert selected == []

    def test_top_surprise_first(self):
        """The first element should have the highest surprise."""
        fast = FastMemory(capacity=100, surprise_threshold=0.0)
        fast.store(_make_episode(surprise=0.3, timestamp=0))
        fast.store(_make_episode(surprise=0.9, timestamp=1))
        fast.store(_make_episode(surprise=0.5, timestamp=2))

        slow = SlowMemory(context_dim=4, outcome_dim=4)
        sm = SelfModel(state_dim=4, embed_dim=16)
        engine = DreamEngine(fast, slow, sm, sigma=0.1, M=15)

        selected = engine.select_for_replay()
        assert len(selected) > 0
        assert selected[0].surprise == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Attractor memory integration
# ---------------------------------------------------------------------------

class TestDreamCycleWithAttractors:
    """DreamEngine should optionally update identity from attractor memory."""

    def test_with_attractor_memory(self):
        fast = _populated_fast_memory(5)
        slow = SlowMemory(context_dim=4, outcome_dim=4)
        sm = SelfModel(state_dim=4, embed_dim=16)

        # Simple mock attractor memory
        class MockAttractorMemory:
            def __init__(self):
                self.attractors = [
                    {"state": torch.tensor([1.0, 0.0, 0.0, 0.0]), "strength": 5.0},
                    {"state": torch.tensor([0.0, 1.0, 0.0, 0.0]), "strength": 3.0},
                ]

        attractor_mem = MockAttractorMemory()
        engine = DreamEngine(fast, slow, sm, attractor_memory=attractor_mem)
        report = engine.dream_cycle(duration=30)

        assert isinstance(report, DreamReport)
        assert report.identity_updated is True

    def test_without_attractor_memory(self):
        engine = _build_engine()
        report = engine.dream_cycle(duration=30)
        assert isinstance(report, DreamReport)
        # identity_updated depends on whether there are memories to consolidate
        # but should not crash without attractor memory
