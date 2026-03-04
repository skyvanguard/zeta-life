"""ComplementaryMemory -- dual-speed memory system for the Conscious Kernel.

Implements a Complementary Learning Systems (CLS) architecture with two
memory subsystems:

- **FastMemory**: Episodic buffer that stores surprising experiences in a
  bounded deque.  Supports similarity-based recall using cosine distance
  on archetype states.  Analogous to the hippocampus.

- **SlowMemory**: Semantic knowledge network (nn.Module) that gradually
  extracts statistical regularities from experience via slow SGD.
  Analogous to the neocortex.

Data structures:
- **Episode**: Full episodic record (stimulus, observation, archetype_state,
  surprise, dominant vertex, timestamp, optional prediction errors).
- **CompressedEpisode**: Lightweight summary for recall results and
  consolidation.

Based on:
- Complementary Learning Systems (McClelland, McNaughton & O'Reilly, 1995)
- Active Inference episodic memory (Fountas et al., 2020)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """Full episodic memory record.

    Parameters
    ----------
    stimulus : Tensor
        Input stimulus vector.
    observation : Tensor
        Observed outcome vector.
    archetype_state : Tensor
        Archetype / vertex distribution at time of encoding.
    surprise : float
        Scalar surprise value (higher = more unexpected).
    dominant : str
        Name of the dominant vertex (e.g., ``"V0"``).
    timestamp : int
        Discrete time step when the episode occurred.
    prediction_errors : dict[str, float] | None
        Optional per-channel prediction errors.
    """

    stimulus: Tensor
    observation: Tensor
    archetype_state: Tensor
    surprise: float
    dominant: str
    timestamp: int
    prediction_errors: dict[str, float] | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict (tensors become lists)."""
        d: dict = {
            "stimulus": self.stimulus.tolist(),
            "observation": self.observation.tolist(),
            "archetype_state": self.archetype_state.tolist(),
            "surprise": self.surprise,
            "dominant": self.dominant,
            "timestamp": self.timestamp,
            "prediction_errors": self.prediction_errors,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        """Deserialize from a plain dict."""
        return cls(
            stimulus=torch.tensor(d["stimulus"]),
            observation=torch.tensor(d["observation"]),
            archetype_state=torch.tensor(d["archetype_state"]),
            surprise=d["surprise"],
            dominant=d["dominant"],
            timestamp=d["timestamp"],
            prediction_errors=d.get("prediction_errors"),
        )


@dataclass
class CompressedEpisode:
    """Lightweight episode summary for recall and consolidation.

    Parameters
    ----------
    archetype_state : Tensor
        Archetype / vertex distribution.
    surprise : float
        Scalar surprise value.
    dominant : str
        Dominant vertex name.
    timestamp : int
        Discrete time step.
    consolidated : bool
        Whether this episode has been consolidated into slow memory.
    """

    archetype_state: Tensor
    surprise: float
    dominant: str
    timestamp: int
    consolidated: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "archetype_state": self.archetype_state.tolist(),
            "surprise": self.surprise,
            "dominant": self.dominant,
            "timestamp": self.timestamp,
            "consolidated": self.consolidated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompressedEpisode:
        """Deserialize from a plain dict."""
        return cls(
            archetype_state=torch.tensor(d["archetype_state"]),
            surprise=d["surprise"],
            dominant=d["dominant"],
            timestamp=d["timestamp"],
            consolidated=d.get("consolidated", False),
        )


# ---------------------------------------------------------------------------
# FastMemory — episodic buffer (hippocampus analogy)
# ---------------------------------------------------------------------------

class FastMemory:
    """Episodic memory buffer that stores surprising experiences.

    Only episodes whose ``surprise`` meets or exceeds ``surprise_threshold``
    are stored.  The buffer is a bounded FIFO deque — oldest episodes are
    evicted when capacity is reached.

    Recall is performed via cosine similarity on ``archetype_state``.

    Parameters
    ----------
    capacity : int
        Maximum number of episodes to retain.
    surprise_threshold : float
        Minimum surprise required for an episode to be stored.
    """

    def __init__(
        self,
        capacity: int = 100,
        surprise_threshold: float = 0.5,
    ) -> None:
        self.capacity = capacity
        self.surprise_threshold = surprise_threshold
        self._episodes: deque[Episode] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._episodes)

    def store(self, episode: Episode) -> None:
        """Store an episode if its surprise meets the threshold.

        Parameters
        ----------
        episode : Episode
            The episode to potentially store.
        """
        if episode.surprise >= self.surprise_threshold:
            self._episodes.append(episode)

    def recall_by_similarity(
        self,
        query_state: Tensor,
        top_k: int = 5,
    ) -> list[CompressedEpisode]:
        """Retrieve the most similar episodes by cosine similarity on archetype_state.

        Parameters
        ----------
        query_state : Tensor
            Query archetype state vector.
        top_k : int
            Maximum number of episodes to return.

        Returns
        -------
        list[CompressedEpisode]
            Up to ``top_k`` compressed episodes, sorted by descending similarity.
        """
        if len(self._episodes) == 0:
            return []

        # Compute cosine similarities
        query = query_state.unsqueeze(0)  # (1, dim)
        similarities: list[tuple[float, Episode]] = []

        for ep in self._episodes:
            ep_state = ep.archetype_state.unsqueeze(0)  # (1, dim)
            sim = F.cosine_similarity(query, ep_state, dim=1).item()
            similarities.append((sim, ep))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Take top_k and convert to CompressedEpisode
        results: list[CompressedEpisode] = []
        for sim, ep in similarities[:top_k]:
            results.append(
                CompressedEpisode(
                    archetype_state=ep.archetype_state.clone(),
                    surprise=ep.surprise,
                    dominant=ep.dominant,
                    timestamp=ep.timestamp,
                )
            )

        return results

    def serialize(self) -> dict:
        """Serialize the entire fast memory to a plain dict.

        Returns
        -------
        dict
            Contains ``capacity``, ``surprise_threshold``, and ``episodes``.
        """
        return {
            "capacity": self.capacity,
            "surprise_threshold": self.surprise_threshold,
            "episodes": [ep.to_dict() for ep in self._episodes],
        }

    @classmethod
    def restore(cls, data: dict) -> FastMemory:
        """Restore a FastMemory instance from a serialized dict.

        Parameters
        ----------
        data : dict
            Output of :meth:`serialize`.

        Returns
        -------
        FastMemory
            Restored instance with same capacity, threshold, and episodes.
        """
        fm = cls(
            capacity=data["capacity"],
            surprise_threshold=data["surprise_threshold"],
        )
        for ep_dict in data.get("episodes", []):
            ep = Episode.from_dict(ep_dict)
            # Bypass threshold check — these were already accepted
            fm._episodes.append(ep)
        return fm


# ---------------------------------------------------------------------------
# SlowMemory — semantic knowledge network (neocortex analogy)
# ---------------------------------------------------------------------------

class SlowMemory(nn.Module):
    """Semantic knowledge network that gradually learns statistical regularities.

    Uses a small feedforward network trained with SGD at a very low learning
    rate, mirroring the slow interleaved learning of the neocortex.

    Architecture:
        knowledge: Linear(context_dim, 64) -> ReLU -> Linear(64, 64) -> ReLU
                   -> Linear(64, outcome_dim)

    Parameters
    ----------
    context_dim : int
        Dimensionality of the context (input) space.
    outcome_dim : int
        Dimensionality of the outcome (output) space.
    learning_rate : float
        Base learning rate for SGD optimizer.
    """

    def __init__(
        self,
        context_dim: int = 4,
        outcome_dim: int = 4,
        learning_rate: float = 0.0001,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.outcome_dim = outcome_dim
        self.lr = learning_rate

        # Knowledge network: context -> outcome
        self.knowledge = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, outcome_dim),
        )

        # Very slow optimizer — mirrors neocortical learning
        self.optimizer = torch.optim.SGD(
            self.knowledge.parameters(),
            lr=learning_rate,
        )

    def generalize(self, query: Tensor) -> Tensor:
        """Produce a generalized prediction from the learned knowledge.

        Parameters
        ----------
        query : Tensor
            Context vector of shape ``(context_dim,)``.

        Returns
        -------
        Tensor
            Predicted outcome of shape ``(outcome_dim,)``, detached.
        """
        with torch.no_grad():
            return self.knowledge(query)

    def integrate(
        self,
        context: Tensor,
        outcome: Tensor,
        learning_rate_scale: float = 1.0,
    ) -> float:
        """Learn from a single context-outcome pair.

        Computes MSE loss and performs one gradient step with optionally
        scaled learning rate.

        Parameters
        ----------
        context : Tensor
            Context vector of shape ``(context_dim,)``.
        outcome : Tensor
            Target outcome of shape ``(outcome_dim,)``.
        learning_rate_scale : float
            Multiplier for the base learning rate (0.0 = no learning).

        Returns
        -------
        float
            The scalar loss value.
        """
        # Adjust learning rate for this step
        effective_lr = self.lr * learning_rate_scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = effective_lr

        # Forward pass
        prediction = self.knowledge(context)
        loss = F.mse_loss(prediction, outcome)

        # Backward pass (skip if scale is zero to avoid unnecessary work)
        if learning_rate_scale > 0.0:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Restore base learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

        return loss.item()
