"""PrecisionController — learned hyper-model for precision weights.

A small neural network that maps the current global state and recent
prediction errors to a vector of per-channel precision weights.
The final Softplus activation guarantees that all outputs are strictly
positive, satisfying the mathematical requirement for precision
(inverse variance) in the active inference framework.

Usage::

    pc = PrecisionController(state_dim=4, n_channels=4, hidden_dim=32)
    precisions = pc(global_state, recent_errors)  # shape (n_channels,)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PrecisionController(nn.Module):
    """Hyper-model that outputs learned precision weights.

    Parameters
    ----------
    state_dim:
        Dimensionality of the global state vector.
    n_channels:
        Number of prediction-error channels (output size).
    hidden_dim:
        Width of the hidden layer.
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_channels: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels),
            nn.Softplus(),  # always positive
        )

    def forward(self, global_state: Tensor, recent_errors: Tensor) -> Tensor:
        """Compute precision weights from state and recent errors.

        Both inputs are detached so that gradients do not flow back
        into the upstream modules that produced them — the controller
        is trained with its own objective.

        Parameters
        ----------
        global_state:
            Current global state vector, shape ``(state_dim,)``.
        recent_errors:
            Recent mean error magnitudes per channel, shape ``(n_channels,)``.

        Returns
        -------
        Tensor
            Positive precision weights, shape ``(n_channels,)``.
        """
        context = torch.cat([global_state.detach(), recent_errors.detach()])
        return self.net(context)
