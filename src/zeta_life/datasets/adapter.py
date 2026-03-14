"""DatasetAdapter bridges external data sources to consciousness interfaces.

Wraps any DataSource + Projector into the StimulusEnvironment pattern
used by ConsciousKernel.step() and HierarchicalSimulation.step().
"""

from __future__ import annotations

import numpy as np
import torch

from .projector import Projector
from .sources import DataSource


PHASE_LABELS = ("early", "mid", "late", "final")


class DatasetAdapter:
    """Adapt a DataSource into stimulus compatible with consciousness systems.

    Parameters
    ----------
    source : DataSource
        Any object implementing the DataSource protocol.
    obs_dim : int
        Target observation dimensionality (must match kernel's obs_dim).
    projection : str
        Projection method: 'pca' or 'identity'.
    normalize : str
        Normalization strategy for output stimuli:
        - 'positive': z-score then shift to ensure all values > 0.
        - 'raw': no normalization after projection.
    loop : bool
        If True, cycle through data when exhausted. If False, raise StopIteration.
    """

    def __init__(
        self,
        source: DataSource,
        obs_dim: int = 4,
        projection: str = "pca",
        normalize: str = "positive",
        loop: bool = True,
    ) -> None:
        self._source = source
        self._obs_dim = obs_dim
        self._normalize_mode = normalize
        self._loop = loop

        # Load and project data
        raw = source.load()
        self._projector = Projector(target_dim=obs_dim, method=projection)
        self._projector.fit(raw)
        self._projected = self._projector.transform(raw)  # (n_samples, obs_dim)

        self._idx = 0

    def __len__(self) -> int:
        return self._projected.shape[0]

    @property
    def phase(self) -> str:
        """Current phase based on position in dataset (quarter-based)."""
        n = len(self)
        if n == 0:
            return PHASE_LABELS[0]
        quarter = int(self._idx / n * 4)
        quarter = min(quarter, 3)
        return PHASE_LABELS[quarter]

    @property
    def source_name(self) -> str:
        return self._source.name

    def reset(self) -> None:
        """Reset iteration to the beginning."""
        self._idx = 0

    def get_stimulus(self) -> tuple[torch.Tensor, str]:
        """Get next stimulus as torch Tensor + phase label.

        Compatible with StimulusEnvironment.get_stimulus() interface.

        Returns
        -------
        tuple[torch.Tensor, str]
            (stimulus of shape (obs_dim,), phase label)
        """
        sample = self._next_sample()
        normalized = self._apply_normalize(sample)
        tensor = torch.tensor(normalized, dtype=torch.float32)
        return tensor, self.phase

    def get_hierarchical_stimulus(self) -> np.ndarray:
        """Get next stimulus as numpy array for HierarchicalSimulation.

        Returns
        -------
        np.ndarray
            Stimulus of shape (obs_dim,).
        """
        sample = self._next_sample()
        return self._apply_normalize(sample)

    def _next_sample(self) -> np.ndarray:
        """Advance index and return next projected sample."""
        if self._idx >= len(self):
            if self._loop:
                self._idx = 0
            else:
                raise StopIteration("Dataset exhausted (loop=False)")
        sample = self._projected[self._idx]
        self._idx += 1
        return sample

    def _apply_normalize(self, sample: np.ndarray) -> np.ndarray:
        """Normalize a single sample according to the selected mode."""
        if self._normalize_mode == "raw":
            return sample.copy()

        if self._normalize_mode == "positive":
            # Z-score across projected dataset, then shift to positive
            mean = self._projected.mean(axis=0)
            std = self._projected.std(axis=0)
            std[std == 0] = 1.0
            z = (sample - mean) / std
            # abs + small shift to ensure strictly positive
            return np.abs(z) + 0.01

        raise ValueError(f"Unknown normalize mode: {self._normalize_mode}")
