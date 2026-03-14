"""Colored noise generator via FFT spectral shaping (1/f^beta).

No external dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np


class ColoredNoiseSource:
    """Generate colored noise signals with controlled spectral properties.

    Parameters
    ----------
    n_samples : int
        Number of time steps to generate.
    n_channels : int
        Number of independent noise channels.
    noise_type : str
        One of 'white' (beta=0), 'pink' (beta=1), 'brown' (beta=2), or
        'mixed' (each channel gets a different beta).
    seed : int | None
        Random seed for reproducibility.
    """

    BETA_MAP = {"white": 0.0, "pink": 1.0, "brown": 2.0}

    def __init__(
        self,
        n_samples: int = 5000,
        n_channels: int = 4,
        noise_type: str = "pink",
        seed: int | None = 42,
    ) -> None:
        if noise_type not in (*self.BETA_MAP, "mixed"):
            raise ValueError(
                f"noise_type must be one of {list(self.BETA_MAP) + ['mixed']}, "
                f"got '{noise_type}'"
            )
        self._n_samples = n_samples
        self._n_channels = n_channels
        self._noise_type = noise_type
        self._seed = seed

    # -- DataSource protocol --------------------------------------------------

    @property
    def name(self) -> str:
        return f"colored_noise_{self._noise_type}"

    @property
    def n_features(self) -> int:
        return self._n_channels

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def feature_names(self) -> list[str]:
        return [f"ch_{i}" for i in range(self._n_channels)]

    def load(self) -> np.ndarray:
        """Generate colored noise array of shape (n_samples, n_channels)."""
        rng = np.random.default_rng(self._seed)
        betas = self._resolve_betas()
        data = np.empty((self._n_samples, self._n_channels), dtype=np.float64)
        for ch, beta in enumerate(betas):
            data[:, ch] = self._generate_channel(beta, rng)
        return data

    # -- internals ------------------------------------------------------------

    def _resolve_betas(self) -> list[float]:
        if self._noise_type == "mixed":
            # Spread betas evenly across channels: white → brown
            return [
                2.0 * i / max(self._n_channels - 1, 1)
                for i in range(self._n_channels)
            ]
        return [self.BETA_MAP[self._noise_type]] * self._n_channels

    def _generate_channel(self, beta: float, rng: np.random.Generator) -> np.ndarray:
        """FFT-based 1/f^beta noise generation."""
        n = self._n_samples
        # White noise in frequency domain
        white = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        freqs = np.fft.fftfreq(n)
        # Avoid division by zero at DC
        freqs[0] = 1.0
        # Shape spectrum: amplitude ~ 1/f^(beta/2)
        amplitudes = np.abs(freqs) ** (-beta / 2.0)
        amplitudes[0] = 0.0  # zero DC component
        shaped = white * amplitudes
        signal = np.fft.ifft(shaped).real
        # Normalize to zero mean, unit variance
        std = signal.std()
        if std > 0:
            signal = (signal - signal.mean()) / std
        return signal
