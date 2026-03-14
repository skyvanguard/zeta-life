"""Data source protocol and registry for Zeta Life datasets."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DataSource(Protocol):
    """Protocol for all data sources feeding into consciousness systems."""

    @property
    def name(self) -> str: ...

    @property
    def n_features(self) -> int: ...

    @property
    def n_samples(self) -> int: ...

    @property
    def feature_names(self) -> list[str]: ...

    def load(self) -> np.ndarray:
        """Load data as array of shape (n_samples, n_features)."""
        ...
