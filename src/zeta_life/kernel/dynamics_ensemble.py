"""DynamicsEnsemble — independent one-step dynamics models for an epistemic signal.

This is the Plan2Explore-faithful disagreement source. The earlier
``WorldModel.heads`` ensemble shares the transition GRU and differs only in a
final linear readout over an *identical* next-latent, so its members converge
quickly and its disagreement is near-flat (the controlled curiosity experiment
found no effect). Here each member is its OWN MLP mapping ``(latent, action) ->
next observation`` — they model the dynamics independently, so they disagree
where the dynamics are unlearned, which is the signal exploration needs.

Bootstrap masking (each member trains on a random ~half of steps) plus
independent initialisation make the members diverge in novel regions and agree
where data is dense. Training uses a DEDICATED RNG so it does not perturb the
global torch stream that downstream action sampling draws from (the C1 lesson).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicsEnsemble(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_in_dim: int,
        obs_dim: int,
        n_members: int = 5,
        hidden: int = 64,
        learning_rate: float = 0.005,
        seed: int = 20260608,
    ) -> None:
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim + action_in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, obs_dim),
                )
                for _ in range(n_members)
            ]
        )
        self.optimizer = torch.optim.Adam(self.members.parameters(), lr=learning_rate)
        # Dedicated RNG for bootstrap masking (does not touch the global stream).
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def train_step(self, latent: Tensor, action_in: Tensor, target: Tensor) -> None:
        """One bootstrap-masked gradient step toward ``(latent, action) -> target``."""
        x = torch.cat([latent.detach(), action_in.detach()])
        loss = torch.zeros((), dtype=x.dtype)
        used = False
        for member in self.members:
            if float(torch.rand(1, generator=self._rng).item()) < 0.5:
                loss = loss + torch.sum((member(x) - target) ** 2)
                used = True
        if used:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def disagreement(self, latent: Tensor, action_in: Tensor) -> float:
        """Mean variance across members of the predicted next observation.

        High where the members disagree (unlearned dynamics) -> the epistemic
        signal. Read-only; does not mutate state.
        """
        with torch.no_grad():
            x = torch.cat([latent, action_in])
            preds = torch.stack([member(x) for member in self.members])  # (n, obs)
            return float(preds.var(dim=0).mean().item())
