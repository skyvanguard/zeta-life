"""PredictionErrorEngine — multi-channel prediction error for the Conscious Kernel.

Implements precision-weighted prediction errors across multiple channels
(perceptual, interoceptive, temporal, epistemic) following the Active
Inference framework.  Each channel maintains a learnable precision
(inverse variance) that modulates error signals.

Key concepts:
- **Precision**: learnable confidence per channel (softplus of log_precisions)
- **Weighted error**: precision * raw error — amplifies reliable channels
- **Free energy**: sum of precision-weighted squared errors
- **Error history**: rolling window of magnitudes for trend monitoring
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PredictionErrorEngine(nn.Module):
    """Multi-channel prediction error engine with learnable precisions.

    Computes precision-weighted prediction errors across named channels
    and tracks error magnitudes over time for monitoring.

    Parameters
    ----------
    n_channels : int
        Number of error channels.  The first four channels use the
        canonical names from ``CHANNEL_NAMES``; additional channels
        receive generic names (``channel_4``, ``channel_5``, ...).
    """

    CHANNEL_NAMES = ('perceptual', 'interoceptive', 'temporal', 'epistemic')

    def __init__(self, n_channels: int = 4) -> None:
        super().__init__()

        # Build channel name list
        self.channels: list[str] = []
        for i in range(n_channels):
            if i < len(self.CHANNEL_NAMES):
                self.channels.append(self.CHANNEL_NAMES[i])
            else:
                self.channels.append(f'channel_{i}')

        # Learnable log-precisions (one per channel)
        self.log_precisions = nn.Parameter(torch.zeros(n_channels))

        # Rolling history of per-channel error magnitudes
        self._error_history: deque[Tensor] = deque(maxlen=50)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def precisions(self) -> Tensor:
        """Return positive precisions via softplus transform.

        Returns
        -------
        Tensor
            Precision values of shape ``(n_channels,)``, guaranteed > 0.
        """
        return F.softplus(self.log_precisions)

    # -----------------------------------------------------------------
    # Core methods
    # -----------------------------------------------------------------

    def compute_errors(
        self,
        predictions: dict[str, Tensor],
        observations: dict[str, Tensor],
    ) -> dict[str, dict[str, Tensor]]:
        """Compute precision-weighted prediction errors for every channel.

        Parameters
        ----------
        predictions : dict[str, Tensor]
            Predicted values keyed by channel name.
        observations : dict[str, Tensor]
            Observed values keyed by channel name.

        Returns
        -------
        dict[str, dict[str, Tensor]]
            For each channel: ``{raw, weighted, precision, magnitude}``.
        """
        precs = self.precisions
        errors: dict[str, dict[str, Tensor]] = {}
        magnitudes: list[Tensor] = []

        for i, ch in enumerate(self.channels):
            raw = predictions[ch] - observations[ch]
            precision = precs[i]
            weighted = precision * raw
            magnitude = torch.norm(raw)

            errors[ch] = {
                'raw': raw,
                'weighted': weighted,
                'precision': precision,
                'magnitude': magnitude,
            }
            magnitudes.append(magnitude.detach())

        # Store magnitudes in history for recent_errors()
        self._error_history.append(torch.stack(magnitudes))

        return errors

    def free_energy(self, errors: dict[str, dict[str, Tensor]]) -> Tensor:
        """Compute variational free energy from prediction errors.

        Free energy is the sum of precision-weighted squared raw errors
        across all channels: ``F = sum_i precision_i * ||raw_i||^2``.

        Parameters
        ----------
        errors : dict[str, dict[str, Tensor]]
            Output of :meth:`compute_errors`.

        Returns
        -------
        Tensor
            Scalar free energy value (>= 0).
        """
        fe = torch.tensor(0.0)
        for ch in self.channels:
            raw = errors[ch]['raw']
            precision = errors[ch]['precision']
            fe = fe + precision * torch.sum(raw ** 2)
        return fe

    def recent_errors(self) -> Tensor:
        """Return mean error magnitudes from the rolling history.

        Returns
        -------
        Tensor
            Per-channel mean magnitudes of shape ``(n_channels,)``.
            Returns zeros if no errors have been computed yet.
        """
        if len(self._error_history) == 0:
            return torch.zeros(len(self.channels))

        stacked = torch.stack(list(self._error_history))  # (T, n_channels)
        return stacked.mean(dim=0)
