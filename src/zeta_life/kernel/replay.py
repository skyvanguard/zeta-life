"""ReplayBuffer — transition replay for DreamerV3-style behaviour learning.

Stores raw environment transitions ``(obs, action, next_obs)`` and samples
uniform batches. The Dreamer learner re-encodes the sampled observations with the
*current* world model and imagines from them, so behaviour is trained from a
diverse set of visited states (including rare ones retained in the buffer) rather
than only the policy's recent online states — the fix for the oscillating
CartPole curve, where an online latent buffer let the actor forget how to recover
from rare (tilted) states.

Storing observations (cheap, re-encoded on demand) rather than latents also avoids
staleness: an old latent reflects an old world model, but an old observation
re-encoded by the current model is always consistent. A dedicated RNG keeps
sampling from perturbing the global torch stream the action sampler draws from.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, seed: int = 20260608) -> None:
        self._obs: deque[Tensor] = deque(maxlen=capacity)
        self._act: deque[Tensor] = deque(maxlen=capacity)
        self._next: deque[Tensor] = deque(maxlen=capacity)
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def add(self, obs: Tensor, action: Tensor, next_obs: Tensor) -> None:
        self._obs.append(obs.detach().clone())
        self._act.append(action.detach().clone())
        self._next.append(next_obs.detach().clone())

    def __len__(self) -> int:
        return len(self._obs)

    def sample(self, n: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (obs, action, next_obs) batches of shape (n, ...)."""
        idx = torch.randint(0, len(self._obs), (n,), generator=self._rng)
        obs = torch.stack([self._obs[int(i)] for i in idx])
        act = torch.stack([self._act[int(i)] for i in idx])
        nxt = torch.stack([self._next[int(i)] for i in idx])
        return obs, act, nxt
