"""WorldModel — predictive world model for the Conscious Kernel.

Implements a learned internal model of the environment that supports:
- Bottom-up encoding of observations into latent space
- Top-down prediction of future observations given actions
- Counterfactual imagination without modifying internal state
- Online learning from prediction errors

Architecture:
    encoder:    Linear(obs_dim, 64) -> ReLU -> Linear(64, latent_dim)
    transition: GRUCell(action_dim, latent_dim)
    predictor:  Linear(latent_dim, obs_dim)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WorldModel(nn.Module):
    """Predictive world model that learns environment dynamics.

    The model maintains a latent state that summarizes past observations
    and uses a GRU-based transition to predict future states given actions.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation space.
    latent_dim : int
        Dimensionality of the internal latent representation.
    action_dim : int
        Dimensionality of the action space.
    learning_rate : float
        Learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        latent_dim: int = 32,
        action_dim: int = 4,
        learning_rate: float = 0.005,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Bottom-up encoder: observation -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # Transition model: latent + action -> next latent
        self.transition = nn.GRUCell(input_size=action_dim, hidden_size=latent_dim)

        # Top-down predictor: latent -> predicted observation
        self.predictor = nn.Linear(latent_dim, obs_dim)

        # Persistent latent state (registered as buffer so it's in state_dict
        # but not a parameter)
        self.register_buffer("latent_state", torch.zeros(latent_dim))

        # Optimizer for online learning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, observation: Tensor) -> Tensor:
        """Encode an observation into latent space (bottom-up).

        Parameters
        ----------
        observation : Tensor
            Observation vector of shape ``(obs_dim,)``.

        Returns
        -------
        Tensor
            Latent representation of shape ``(latent_dim,)``.
        """
        return self.encoder(observation)

    def predict(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Predict the next observation given an action (top-down).

        Uses the current ``latent_state`` and the provided action to compute
        the next latent state via the GRU transition, then decodes it into
        a predicted observation.  The internal ``latent_state`` is updated
        to the new latent.

        Parameters
        ----------
        action : Tensor
            Action vector of shape ``(action_dim,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(predicted_obs, next_latent)`` where ``predicted_obs`` has shape
            ``(obs_dim,)`` and ``next_latent`` has shape ``(latent_dim,)``.
        """
        # GRUCell expects (batch, features) — add and remove batch dim
        action_batched = action.unsqueeze(0)
        latent_batched = self.latent_state.unsqueeze(0)

        next_latent_batched = self.transition(action_batched, latent_batched)
        next_latent = next_latent_batched.squeeze(0)

        predicted_obs = self.predictor(next_latent)

        # Update internal state
        self.latent_state = next_latent.detach()

        return predicted_obs, next_latent

    def imagine(self, action_sequence: list[Tensor]) -> list[Tensor]:
        """Run a counterfactual simulation without modifying internal state.

        Rolls out the transition model over the given action sequence starting
        from the current ``latent_state``, collecting predicted observations.
        The internal ``latent_state`` is **not** modified.

        Parameters
        ----------
        action_sequence : list[Tensor]
            Sequence of action vectors, each of shape ``(action_dim,)``.

        Returns
        -------
        list[Tensor]
            Predicted observations for each step, each of shape ``(obs_dim,)``.
        """
        if not action_sequence:
            return []

        predictions: list[Tensor] = []
        # Clone the latent state so we don't modify the real one
        imagined_latent = self.latent_state.clone()

        with torch.no_grad():
            for action in action_sequence:
                action_batched = action.unsqueeze(0)
                latent_batched = imagined_latent.unsqueeze(0)

                next_latent_batched = self.transition(action_batched, latent_batched)
                imagined_latent = next_latent_batched.squeeze(0)

                predicted_obs = self.predictor(imagined_latent)
                predictions.append(predicted_obs)

        return predictions

    def update_from_error(self, error: Tensor) -> float:
        """Update the model from a prediction error signal.

        Computes the sum-of-squares loss from the error and performs a
        single gradient step.

        Parameters
        ----------
        error : Tensor
            Prediction error vector of shape ``(obs_dim,)``.

        Returns
        -------
        float
            The scalar loss value.
        """
        loss = torch.sum(error ** 2)

        # Only backpropagate when the error is connected to the computation
        # graph (i.e., it came from a predict() call).  Standalone error
        # tensors (e.g., torch.randn) have no grad_fn, so backward() would
        # raise.  We still report the loss value in that case.
        if loss.requires_grad:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
