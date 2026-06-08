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
        posterior_blend: float = 0.5,
        temporal_dim: int = 0,
        disagreement_heads: int = 0,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        # Dimensionality of the optional temporal feature vector concatenated to
        # the action before the GRU transition. 0 (the default) reproduces the
        # original action-only world model byte-for-byte; >0 lets the model see a
        # time code (e.g. zeta oscillators) and anticipate time-structured
        # dynamics instead of only reacting to them.
        self.temporal_dim = temporal_dim
        # Weight of the prior (transition) vs the encoded observation when
        # forming the posterior latent in observe(). 1.0 = pure recurrence,
        # 0.0 = pure observation. 0.5 balances memory and perception.
        self.posterior_blend = posterior_blend
        # Last prior latent produced by predict(), kept (with grad) so the
        # prior-loss backward in update_from_error can reach the transition.
        self._prior_latent: Tensor | None = None

        # Bottom-up encoder: observation -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # Transition model: (action [+ temporal feature]) -> next latent
        self.transition = nn.GRUCell(
            input_size=action_dim + temporal_dim, hidden_size=latent_dim
        )

        # Top-down predictor: latent -> predicted observation
        self.predictor = nn.Linear(latent_dim, obs_dim)

        # Persistent latent state (registered as buffer so it's in state_dict
        # but not a parameter)
        self.register_buffer("latent_state", torch.zeros(latent_dim))

        # Optimizer for online learning (encoder + transition + predictor).
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Optional ensemble of predictor heads for an epistemic (disagreement)
        # signal: where the heads disagree, the model is uncertain / has not
        # learned that region. Trained on a SEPARATE optimizer with per-head
        # bootstrap masking so they diverge where data is sparse. The main
        # prediction path is unchanged. disagreement_heads=0 (default) -> no
        # ensemble, byte-identical to the point world model.
        self.disagreement_heads = disagreement_heads
        if disagreement_heads > 0:
            self.heads = nn.ModuleList(
                [nn.Linear(latent_dim, obs_dim) for _ in range(disagreement_heads)]
            )
            self.head_optimizer = torch.optim.Adam(
                self.heads.parameters(), lr=learning_rate
            )
            # Dedicated RNG for bootstrap masking so head training does NOT perturb
            # the global torch stream (which the EFE action sampler draws from).
            # Without this, enabling the ensemble silently shifts downstream action
            # randomness, making "ensemble on vs off" an uncontrolled comparison.
            self._head_rng = torch.Generator()
            self._head_rng.manual_seed(20260608)

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

    def _transition_input(self, action: Tensor, temporal_feat: Tensor | None) -> Tensor:
        """Build the GRU input from the action and the optional temporal feature.

        When ``temporal_dim == 0`` the input is the action unchanged (original
        behaviour). Otherwise the temporal feature is concatenated; its gradient
        is preserved so a trainable feature bank can learn from the prior loss.
        """
        if self.temporal_dim == 0:
            return action
        if temporal_feat is None:
            raise ValueError(
                "temporal_feat is required when temporal_dim > 0 "
                f"(got temporal_dim={self.temporal_dim})"
            )
        return torch.cat([action, temporal_feat])

    def predict(
        self, action: Tensor, temporal_feat: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Predict the next observation given an action (top-down).

        Uses the current ``latent_state`` and the provided action to compute
        the next latent state via the GRU transition, then decodes it into
        a predicted observation.  The internal ``latent_state`` is updated
        to the new latent.

        Parameters
        ----------
        action : Tensor
            Action vector of shape ``(action_dim,)``.
        temporal_feat : Tensor | None
            Temporal feature of shape ``(temporal_dim,)``. Required when the
            model was built with ``temporal_dim > 0``; ignored otherwise.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(predicted_obs, next_latent)`` where ``predicted_obs`` has shape
            ``(obs_dim,)`` and ``next_latent`` has shape ``(latent_dim,)``.
        """
        # GRUCell expects (batch, features) — add and remove batch dim
        action_batched = self._transition_input(action, temporal_feat).unsqueeze(0)
        latent_batched = self.latent_state.unsqueeze(0)

        next_latent_batched = self.transition(action_batched, latent_batched)
        next_latent = next_latent_batched.squeeze(0)

        predicted_obs = self.predictor(next_latent)

        # Keep the prior latent (with grad) for the posterior step in observe().
        # The buffer is updated to a detached copy so reads before observe() are
        # safe; observe() overwrites it with the posterior.
        self._prior_latent = next_latent
        self.latent_state = next_latent.detach()

        return predicted_obs, next_latent

    def observe(self, observation: Tensor) -> float:
        """Incorporate an observation into the latent state (posterior step).

        Forms the posterior latent as a blend of the prior (from the last
        :meth:`predict`) and the bottom-up encoding of the observation, then
        takes a gradient step on the reconstruction loss
        ``||predictor(posterior) - observation||^2``.

        This is what makes the **encoder learn** (the gradient flows
        encoder -> posterior -> prediction -> loss) and gives the latent a real
        **posterior correction** from perception, instead of the previous code
        that overwrote the recurrent latent with a detached ``encode(obs)``
        (which froze the encoder and discarded the transition's memory).

        The prior is detached here so this step trains only the encoder and
        predictor; the transition is trained by the prior loss in
        :meth:`update_from_error`. This keeps the two backward passes on
        separate graphs (no double-backward).

        Parameters
        ----------
        observation : Tensor
            Observation vector of shape ``(obs_dim,)``.

        Returns
        -------
        float
            The scalar reconstruction loss.
        """
        prior = (
            self._prior_latent.detach()
            if self._prior_latent is not None
            else self.latent_state
        )
        encoded = self.encode(observation)
        posterior = self.posterior_blend * prior + (1.0 - self.posterior_blend) * encoded

        loss = torch.sum((self.predictor(posterior) - observation) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Recompute the posterior detached for the persistent latent (the graph
        # above was freed by backward).
        with torch.no_grad():
            encoded_d = self.encode(observation)
            posterior_d = (
                self.posterior_blend * prior + (1.0 - self.posterior_blend) * encoded_d
            )
            self.latent_state = posterior_d
        # Train the disagreement ensemble on the detached posterior -> observation.
        self._train_heads(posterior_d, observation.detach())
        return loss.item()

    def imagine(
        self,
        action_sequence: list[Tensor],
        temporal_feats: list[Tensor] | Tensor | None = None,
    ) -> list[Tensor]:
        """Run a counterfactual simulation without modifying internal state.

        Rolls out the transition model over the given action sequence starting
        from the current ``latent_state``, collecting predicted observations.
        The internal ``latent_state`` is **not** modified.

        Parameters
        ----------
        action_sequence : list[Tensor]
            Sequence of action vectors, each of shape ``(action_dim,)``.
        temporal_feats : list[Tensor] | Tensor | None
            Temporal feature(s) for each imagined step. A single tensor is
            broadcast to every step (the common case: planning one step ahead at
            the current time). Required when ``temporal_dim > 0``.

        Returns
        -------
        list[Tensor]
            Predicted observations for each step, each of shape ``(obs_dim,)``.
        """
        if not action_sequence:
            return []

        if self.temporal_dim > 0 and isinstance(temporal_feats, Tensor):
            temporal_feats = [temporal_feats] * len(action_sequence)

        predictions: list[Tensor] = []
        # Clone the latent state so we don't modify the real one
        imagined_latent = self.latent_state.clone()

        with torch.no_grad():
            for i, action in enumerate(action_sequence):
                feat = temporal_feats[i] if temporal_feats is not None else None
                action_batched = self._transition_input(action, feat).unsqueeze(0)
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

    # ------------------------------------------------------------------
    # Epistemic (disagreement) ensemble
    # ------------------------------------------------------------------

    def _train_heads(self, latent: Tensor, target: Tensor) -> None:
        """Train the disagreement heads on (detached latent -> observation).

        Per-head bootstrap masking (each head included with prob 0.5 per step)
        makes the heads see different data subsets, so they diverge where data is
        sparse. No-op when no ensemble is configured.
        """
        if self.disagreement_heads == 0:
            return
        loss = torch.zeros((), dtype=latent.dtype)
        used = False
        for head in self.heads:
            if float(torch.rand(1, generator=self._head_rng).item()) < 0.5:
                loss = loss + torch.sum((head(latent) - target) ** 2)
                used = True
        if used:
            self.head_optimizer.zero_grad()
            loss.backward()
            self.head_optimizer.step()

    def disagreement(self, action: Tensor, temporal_feat: Tensor | None = None) -> float:
        """Variance across ensemble heads of the imagined next observation.

        A read-only epistemic signal: high where the heads disagree (a region the
        model has not learned). Returns 0.0 if no ensemble is configured. Does not
        mutate internal state.
        """
        if self.disagreement_heads == 0:
            return 0.0
        with torch.no_grad():
            gru_in = self._transition_input(action, temporal_feat).unsqueeze(0)
            latent = self.latent_state.unsqueeze(0)
            next_latent = self.transition(gru_in, latent).squeeze(0)
            preds = torch.stack([head(next_latent) for head in self.heads])
            return float(preds.var(dim=0).mean().item())
