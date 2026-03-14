"""DreamEngine -- zeta-driven sleep consolidation for the Conscious Kernel.

Implements a three-phase dream cycle inspired by the neuroscience of sleep
memory consolidation, using the Riemann zeta kernel to determine phase
transitions:

1. **Slow oscillation** (K > 0.5): Selection of surprising memories for replay.
2. **Spindle** (K > -0.2): Transfer from fast (hippocampal) to slow
   (neocortical) memory via self-supervised reconstruction.
3. **Ripple** (K <= -0.2): Deep self-reflection replay that updates the
   identity embedding.

The zeta kernel ``K_sigma(t) = 2 * sum(exp(-sigma*|gamma|) * cos(gamma*t))``
produces a quasi-periodic signal whose natural rhythm drives the phase
schedule, connecting the mathematical structure of the Riemann zeta function
to the biological rhythm of sleep.

Based on:
- Complementary Learning Systems (McClelland et al., 1995)
- Active sleep consolidation (Diekelmann & Born, 2010)
- Riemann zeta zeros as temporal binding (project-specific)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from zeta_life.core.zeta_constants import get_zeta_zeros
from zeta_life.kernel.complementary_memory import (
    CompressedEpisode,
    FastMemory,
    SlowMemory,
)
from zeta_life.kernel.self_model import SelfModel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DreamReport:
    """Summary statistics produced by a single dream cycle.

    Parameters
    ----------
    duration : int
        Number of time steps in the dream cycle.
    selections : int
        Number of memories selected for replay (slow_oscillation phase).
    transfers : int
        Number of fast->slow memory transfers (spindle phase).
    replays : int
        Number of self-reflection replays (ripple phase).
    total_loss : float
        Accumulated MSE loss from slow-memory integrations.
    identity_updated : bool
        Whether the self-model identity embedding was updated.
    phases_visited : dict[str, int]
        Count of time steps spent in each phase.
    """

    duration: int
    selections: int
    transfers: int
    replays: int
    total_loss: float
    identity_updated: bool
    phases_visited: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DreamEngine
# ---------------------------------------------------------------------------

class DreamEngine:
    """Zeta-driven sleep consolidation engine.

    Uses the zeta kernel to determine a three-phase dream rhythm
    (slow_oscillation / spindle / ripple) and performs memory consolidation
    operations accordingly.

    Parameters
    ----------
    fast_memory : FastMemory
        Episodic hippocampal buffer.
    slow_memory : SlowMemory
        Semantic neocortical network.
    self_model : SelfModel
        Recursive self-model with identity embedding.
    attractor_memory : object | None
        Optional attractor memory with ``attractors`` list for identity
        blending after consolidation.
    sigma : float
        Abel regularization parameter for the zeta kernel decay.
    M : int
        Number of non-trivial zeta zeros to use.
    """

    def __init__(
        self,
        fast_memory: FastMemory,
        slow_memory: SlowMemory,
        self_model: SelfModel,
        attractor_memory: object | None = None,
        sigma: float = 0.1,
        M: int = 15,
    ) -> None:
        self.fast_memory = fast_memory
        self.slow_memory = slow_memory
        self.self_model = self_model
        self.attractor_memory = attractor_memory
        self.sigma = sigma
        self.M = M

        self.gammas: list[float] = get_zeta_zeros(M)
        self.total_dreams: int = 0

    # ------------------------------------------------------------------
    # Zeta kernel
    # ------------------------------------------------------------------

    def zeta_kernel(self, t: float) -> float:
        """Evaluate the Abel-regularized zeta kernel at time *t*.

        .. math::

            K_{\\sigma}(t) = 2 \\sum_{n=1}^{M}
                \\exp(-\\sigma |\\gamma_n|) \\cos(\\gamma_n t)

        Parameters
        ----------
        t : float
            Continuous time parameter.

        Returns
        -------
        float
            Kernel value (can be positive or negative).
        """
        total = 0.0
        for g in self.gammas:
            total += math.exp(-self.sigma * abs(g)) * math.cos(g * t)
        return 2.0 * total

    # ------------------------------------------------------------------
    # Phase mapping
    # ------------------------------------------------------------------

    def phase_from_kernel(self, t: float) -> str:
        """Map a time point to a sleep phase based on the kernel value.

        Phase thresholds:
        - ``k > 0.5``  -> ``'slow_oscillation'``
        - ``k > -0.2`` -> ``'spindle'``
        - otherwise     -> ``'ripple'``

        Parameters
        ----------
        t : float
            Continuous time parameter.

        Returns
        -------
        str
            One of ``'slow_oscillation'``, ``'spindle'``, ``'ripple'``.
        """
        k = self.zeta_kernel(t)
        if k > 0.5:
            return "slow_oscillation"
        if k > -0.2:
            return "spindle"
        return "ripple"

    # ------------------------------------------------------------------
    # Memory selection
    # ------------------------------------------------------------------

    def select_for_replay(self) -> list[CompressedEpisode]:
        """Select unconsolidated memories sorted by surprise (descending).

        Iterates all episodes in fast memory, converts them to
        :class:`CompressedEpisode` instances, filters out those already
        marked as consolidated, and returns the remainder sorted by
        surprise in descending order.

        Returns
        -------
        list[CompressedEpisode]
            Unconsolidated compressed episodes, highest surprise first.
        """
        episodes = self.fast_memory._episodes
        if len(episodes) == 0:
            return []

        compressed: list[CompressedEpisode] = []
        for ep in episodes:
            ce = CompressedEpisode(
                archetype_state=ep.archetype_state.clone(),
                surprise=ep.surprise,
                dominant=ep.dominant,
                timestamp=ep.timestamp,
                consolidated=getattr(ep, "_consolidated", False),
            )
            if not ce.consolidated:
                compressed.append(ce)

        # Sort by surprise descending
        compressed.sort(key=lambda x: x.surprise, reverse=True)
        return compressed

    # ------------------------------------------------------------------
    # Dream cycle
    # ------------------------------------------------------------------

    def dream_cycle(self, duration: int = 50) -> DreamReport:
        """Run one full dream cycle of *duration* time steps.

        At each step the zeta kernel determines the current phase:

        - **slow_oscillation**: Select the most surprising unconsolidated
          memory for upcoming replay.
        - **spindle**: Transfer the selected memory to slow memory via
          self-supervised reconstruction (context = outcome = archetype_state).
          The ``learning_rate_scale`` is boosted by ``1 + binding_weight``
          where *binding_weight* is the absolute kernel value.
        - **ripple**: Replay the memory through the self-model's reflect
          method at depth 2 for identity integration.

        After the cycle:
        - Top replayed memories are marked as consolidated.
        - If attractor memory is present, identity embedding is updated.

        Parameters
        ----------
        duration : int
            Number of discrete time steps.

        Returns
        -------
        DreamReport
            Statistics for this dream cycle.
        """
        phases_visited: dict[str, int] = {
            "slow_oscillation": 0,
            "spindle": 0,
            "ripple": 0,
        }
        selections = 0
        transfers = 0
        replays = 0
        total_loss = 0.0

        # Pre-select memories for replay
        replay_queue = self.select_for_replay()
        current_memory: CompressedEpisode | None = None
        replay_idx = 0

        transferred_indices: list[int] = []

        for step in range(duration):
            t = float(step)
            phase = self.phase_from_kernel(t)
            phases_visited[phase] += 1

            if phase == "slow_oscillation":
                # Select next memory from the queue
                if replay_idx < len(replay_queue):
                    current_memory = replay_queue[replay_idx]
                    replay_idx += 1
                    selections += 1

            elif phase == "spindle":
                # Transfer: integrate memory into slow memory
                if current_memory is not None:
                    arch_state = current_memory.archetype_state
                    binding_weight = abs(self.zeta_kernel(t))
                    lr_scale = 1.0 + binding_weight
                    loss = self.slow_memory.integrate(
                        arch_state,
                        arch_state,
                        learning_rate_scale=lr_scale,
                    )
                    total_loss += loss
                    transfers += 1
                    # Track which index was transferred for consolidation
                    if (replay_idx - 1) not in transferred_indices:
                        transferred_indices.append(replay_idx - 1)

            elif phase == "ripple":
                # Deep replay: self-reflection
                if current_memory is not None:
                    self.self_model.reflect(
                        current_memory.archetype_state,
                        depth=2,
                    )
                    replays += 1

        # Mark transferred memories as consolidated in the original episodes
        consolidated_count = 0
        if transferred_indices and len(replay_queue) > 0:
            # Map replay_queue entries back to fast memory episodes by timestamp
            transferred_timestamps = {
                replay_queue[idx].timestamp
                for idx in transferred_indices
                if idx < len(replay_queue)
            }
            for ep in self.fast_memory._episodes:
                if ep.timestamp in transferred_timestamps:
                    ep._consolidated = True  # type: ignore[attr-defined]
                    consolidated_count += 1

        # Optionally update identity from attractor memory
        identity_updated = False
        if self.attractor_memory is not None and consolidated_count > 0:
            self.self_model.update_embedding_from_attractors(self.attractor_memory)
            identity_updated = True

        self.total_dreams += 1

        return DreamReport(
            duration=duration,
            selections=selections,
            transfers=transfers,
            replays=replays,
            total_loss=total_loss,
            identity_updated=identity_updated,
            phases_visited=phases_visited,
        )
