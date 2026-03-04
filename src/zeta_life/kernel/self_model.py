"""SelfModel -- recursive self-modeling with Strange Loop for the Conscious Kernel.

Implements a self-referential model that maintains a persistent identity
embedding and uses recursive reflection to observe and update its own state.
This is the core mechanism for the Strange Loop: the system models itself
as part of the world it models.

Architecture:
    state_to_embed:      Linear(state_dim, embed_dim)
    reflection_net:      GRUCell(embed_dim, embed_dim)
    embed_to_prediction: Linear(embed_dim, state_dim)
    action_to_embed:     Linear(state_dim, embed_dim)
    self_embedding:      nn.Parameter(embed_dim) -- persistent identity

Strange Loop:
    State -> Encode -> Combine with self_embedding -> GRU reflection ->
    Prediction error vs self_embedding -> Update self_embedding (slow EMA) ->
    ... LOOP ...
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn


class SelfModel(nn.Module):
    """Recursive self-model with persistent identity embedding.

    The model maintains a ``self_embedding`` that represents the system's
    current sense of identity.  The :meth:`reflect` method implements a
    recursive Strange Loop: at each depth level, the current embedding is
    combined with the identity embedding and processed through a GRU cell,
    producing a prediction error that measures how far the reflected state
    is from the identity.

    After reflection, the identity embedding is updated via exponential
    moving average (EMA), ensuring slow, stable identity drift rather than
    abrupt changes.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the external state space.
    embed_dim : int
        Dimensionality of the internal embedding / identity space.
    ema_decay : float
        Decay factor for the EMA update of ``self_embedding``.
        Higher values (closer to 1.0) mean slower identity updates.
    """

    def __init__(
        self,
        state_dim: int = 4,
        embed_dim: int = 16,
        ema_decay: float = 0.95,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay

        # Persistent identity embedding (learnable parameter)
        self.self_embedding = nn.Parameter(torch.randn(embed_dim) * 0.1)

        # State -> embedding space
        self.state_to_embed = nn.Linear(state_dim, embed_dim)

        # Recursive reflection via GRU cell
        self.reflection_net = nn.GRUCell(input_size=embed_dim, hidden_size=embed_dim)

        # Embedding -> state prediction
        self.embed_to_prediction = nn.Linear(embed_dim, state_dim)

        # Action -> embedding space (for predict_self)
        self.action_to_embed = nn.Linear(state_dim, embed_dim)

        # History of reflection results
        self.reflection_history: deque[list[dict]] = deque(maxlen=100)

    def reflect(self, current_state: Tensor, depth: int = 3) -> list[dict]:
        """Perform recursive self-reflection at multiple depth levels.

        At each depth level the current embedding is combined with the
        identity embedding (self-as-variable), processed through the GRU
        reflection network, and the prediction error relative to the
        identity is recorded.

        After all depth levels, the identity embedding is updated via
        slow EMA toward the final reflected state.

        Parameters
        ----------
        current_state : Tensor
            External state vector of shape ``(state_dim,)``.
        depth : int
            Number of recursive reflection levels.

        Returns
        -------
        list[dict]
            One dict per depth level with keys:
            ``{depth, state, prediction_error}``.
        """
        # Encode the external state into embedding space
        embed = self.state_to_embed(current_state)

        reflections: list[dict] = []

        for d in range(1, depth + 1):
            # Self-as-variable: combine current embedding with identity
            combined = embed + self.self_embedding

            # GRU reflection step (expects batch dimension)
            embed = self.reflection_net(
                combined.unsqueeze(0),
                embed.unsqueeze(0),
            ).squeeze(0)

            # Prediction error: distance between reflected state and identity
            pe = torch.norm(embed - self.self_embedding).item()

            reflections.append({
                'depth': d,
                'state': embed.detach().clone(),
                'prediction_error': pe,
            })

        # Slow EMA update of self_embedding (no grad — identity drifts slowly)
        with torch.no_grad():
            self.self_embedding.data = (
                self.ema_decay * self.self_embedding.data
                + (1.0 - self.ema_decay) * embed.detach()
            )

        # Record in history
        self.reflection_history.append(reflections)

        return reflections

    def predict_self(self, future_action: Tensor) -> Tensor:
        """Predict the system's future state given an action.

        Projects the action into embedding space, runs it through the
        reflection network conditioned on the current identity, and
        decodes back to state space.

        Parameters
        ----------
        future_action : Tensor
            Action vector of shape ``(state_dim,)``.

        Returns
        -------
        Tensor
            Predicted future state of shape ``(state_dim,)``.
        """
        action_embed = self.action_to_embed(future_action)

        # Project through reflection net using identity as hidden state
        projected = self.reflection_net(
            action_embed.unsqueeze(0),
            self.self_embedding.unsqueeze(0),
        ).squeeze(0)

        return self.embed_to_prediction(projected)

    def identity_distance(self, state: Tensor) -> float:
        """Measure how far a state is from the current identity.

        Encodes the state into embedding space and computes the L2 norm
        of the difference with ``self_embedding``.

        Parameters
        ----------
        state : Tensor
            State vector of shape ``(state_dim,)``.

        Returns
        -------
        float
            Non-negative distance in embedding space.
        """
        embed = self.state_to_embed(state)
        return torch.norm(embed - self.self_embedding).item()

    def update_embedding_from_attractors(self, attractor_memory: object) -> None:
        """Blend information from attractor memory into the identity embedding.

        Computes a strength-weighted average of attractor states, projects
        it into embedding space, and blends it into ``self_embedding`` with
        a conservative ratio (0.98 / 0.02) to preserve identity stability.

        Parameters
        ----------
        attractor_memory : object
            An object with an ``attractors`` attribute — a list of dicts,
            each containing ``'state'`` (Tensor) and ``'strength'`` (float).
        """
        attractors = attractor_memory.attractors

        if not attractors:
            return

        # Compute strength-weighted average of attractor states
        total_strength = sum(a['strength'] for a in attractors)
        if total_strength <= 0.0:
            return

        weighted_state = torch.zeros(self.state_dim)
        for a in attractors:
            weight = a['strength'] / total_strength
            weighted_state = weighted_state + weight * a['state']

        # Project to embedding space
        with torch.no_grad():
            attractor_embed = self.state_to_embed(weighted_state)

            # Conservative blend: 0.98 identity preservation, 0.02 attractor influence
            self.self_embedding.data = (
                0.98 * self.self_embedding.data
                + 0.02 * attractor_embed.detach()
            )
