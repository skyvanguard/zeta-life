"""Actor and Critic for Dreamer-style behaviour learning in imagination.

The actor is an amortized policy mapping a world-model latent to a simplex action
(deterministic: ``softmax(logits)``, so value gradients flow straight through it).
The critic estimates the expected cumulative reward (here, negative expected free
energy) from a latent. Both are trained inside imagined rollouts of the world
model — see ``ConsciousKernel._train_behavior``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Actor(nn.Module):
    """Amortized policy: latent -> action on the simplex (deterministic)."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Return a simplex action (sums to 1) for ``latent`` (batched or not)."""
        return F.softmax(self.net(latent), dim=-1)


class Critic(nn.Module):
    """State-value estimate V(latent)."""

    def __init__(self, latent_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Return value of shape ``(B,)`` (or scalar) for ``latent``."""
        return self.net(latent).squeeze(-1)
