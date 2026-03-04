"""GlobalWorkspace -- winner-takes-all competitive bottleneck.

Implements Global Workspace Theory (Baars, 1988) where multiple
ConsciousKernels compete for a limited-capacity broadcast channel.
The winner's state is broadcast to all kernels as top-down influence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Proposal:
    """A kernel's bid for the Global Workspace spotlight."""

    kernel_id: int
    state: Tensor       # self_model.self_embedding
    free_energy: float  # prediction quality (lower = better)
    energy: float       # current energy level
    action: Tensor      # proposed action vector
    salience: float     # self-assessed importance

    def signal_strength(self, novelty_bonus: float = 1.0) -> float:
        """Compute competitive signal strength.

        signal = (1 / free_energy) * energy * novelty_bonus
        """
        return (1.0 / max(self.free_energy, 1e-6)) * self.energy * novelty_bonus


class GlobalWorkspace:
    """Winner-takes-all competitive bottleneck for consciousness.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of action/observation space.
    anti_monopoly_threshold : int
        Consecutive wins before penalty kicks in.
    penalty_factor : float
        Multiplier for monopolist (< 1.0 = penalty).
    boost_factor : float
        Multiplier for non-monopolists (> 1.0 = boost).
    """

    def __init__(
        self,
        obs_dim: int = 4,
        anti_monopoly_threshold: int = 3,
        penalty_factor: float = 0.5,
        boost_factor: float = 1.5,
    ) -> None:
        self.obs_dim = obs_dim
        self.anti_monopoly_threshold = anti_monopoly_threshold
        self.penalty_factor = penalty_factor
        self.boost_factor = boost_factor

        self.spotlight: Tensor | None = None
        self.spotlight_owner: int | None = None
        self.broadcast_signal: Tensor = torch.zeros(obs_dim)
        self.consecutive_wins: dict[int, int] = {}
        self.history: deque[int] = deque(maxlen=100)

    def compete(self, proposals: dict[int, Proposal]) -> int:
        """Select the winning kernel via winner-takes-all.

        Returns the kernel_id of the winner.
        """
        best_id = -1
        best_signal = -1.0

        for kid, prop in proposals.items():
            wins = self.consecutive_wins.get(kid, 0)
            if wins >= self.anti_monopoly_threshold:
                bonus = self.penalty_factor
            elif any(
                v >= self.anti_monopoly_threshold
                for k, v in self.consecutive_wins.items()
                if k != kid
            ):
                bonus = self.boost_factor
            else:
                bonus = 1.0

            signal = prop.signal_strength(bonus)
            if signal > best_signal:
                best_signal = signal
                best_id = kid

        # Update consecutive wins
        for kid in proposals:
            if kid == best_id:
                self.consecutive_wins[kid] = self.consecutive_wins.get(kid, 0) + 1
            else:
                self.consecutive_wins[kid] = 0

        return best_id

    def broadcast(self, proposal: Proposal) -> None:
        """Broadcast the winning proposal to all kernels."""
        self.spotlight = proposal.state.clone().detach()
        self.spotlight_owner = proposal.kernel_id
        self.broadcast_signal = proposal.action.clone().detach()
        self.history.append(proposal.kernel_id)
