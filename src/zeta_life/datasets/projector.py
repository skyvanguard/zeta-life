"""Dimensionality projection for mapping source features to obs_dim.

Uses SVD-based PCA implemented with numpy (no sklearn dependency).
"""

from __future__ import annotations

import numpy as np


class Projector:
    """Project data from source dimensionality to target_dim.

    Parameters
    ----------
    target_dim : int
        Output dimensionality (typically obs_dim=4).
    method : str
        'pca' for SVD-based principal component projection,
        'identity' for passthrough (requires n_features == target_dim).
    """

    def __init__(self, target_dim: int = 4, method: str = "pca") -> None:
        if method not in ("pca", "identity"):
            raise ValueError(f"method must be 'pca' or 'identity', got '{method}'")
        self.target_dim = target_dim
        self.method = method
        self._components: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._explained_variance_ratio: np.ndarray | None = None

    @property
    def explained_variance_ratio(self) -> np.ndarray | None:
        """Fraction of variance explained by each component (after fit)."""
        return self._explained_variance_ratio

    def fit(self, data: np.ndarray) -> Projector:
        """Fit projection from data of shape (n_samples, n_features).

        Returns self for chaining.
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        n_features = data.shape[1]

        if self.method == "identity":
            if n_features != self.target_dim:
                raise ValueError(
                    f"Identity projection requires n_features == target_dim, "
                    f"got {n_features} != {self.target_dim}"
                )
            self._components = np.eye(self.target_dim)
            self._mean = np.zeros(n_features)
            self._explained_variance_ratio = np.ones(self.target_dim) / self.target_dim
            return self

        # PCA via SVD
        self._mean = data.mean(axis=0)
        centered = data - self._mean
        # Economy SVD
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        # Keep top target_dim components
        k = min(self.target_dim, n_features, len(s))
        self._components = vt[:k]  # shape (k, n_features)
        # Explained variance
        variance = (s**2) / (data.shape[0] - 1)
        total_var = variance.sum()
        if total_var > 0:
            self._explained_variance_ratio = variance[:k] / total_var
        else:
            self._explained_variance_ratio = np.zeros(k)

        # Pad if n_features < target_dim
        if k < self.target_dim:
            pad = np.zeros((self.target_dim - k, n_features))
            self._components = np.vstack([self._components, pad])
            self._explained_variance_ratio = np.concatenate(
                [self._explained_variance_ratio, np.zeros(self.target_dim - k)]
            )

        return self

    def transform(self, sample: np.ndarray) -> np.ndarray:
        """Project a single sample to target_dim.

        Parameters
        ----------
        sample : np.ndarray
            Shape (n_features,) or (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Shape (target_dim,) or (n_samples, target_dim).
        """
        if self._components is None:
            raise RuntimeError("Projector must be fit() before transform()")

        if sample.ndim == 1:
            centered = sample - self._mean
            return centered @ self._components.T
        elif sample.ndim == 2:
            centered = sample - self._mean
            return centered @ self._components.T
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {sample.shape}")
