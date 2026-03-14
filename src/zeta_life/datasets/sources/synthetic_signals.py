"""Synthetic AM/FM modulated signals for consciousness experiments.

Generates complex temporal patterns with regime transitions,
useful for testing how consciousness systems adapt to changing stimuli.
"""

from __future__ import annotations

import numpy as np


class SyntheticSignalSource:
    """Generate synthetic signals with amplitude/frequency modulation.

    Parameters
    ----------
    n_samples : int
        Number of time steps.
    n_channels : int
        Number of signal channels.
    pattern : str
        'am' (amplitude modulation), 'fm' (frequency modulation),
        or 'mixed' (different per channel).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_channels: int = 4,
        pattern: str = "mixed",
        seed: int | None = 42,
    ) -> None:
        if pattern not in ("am", "fm", "mixed"):
            raise ValueError(f"pattern must be 'am', 'fm', or 'mixed', got '{pattern}'")
        self._n_samples = n_samples
        self._n_channels = n_channels
        self._pattern = pattern
        self._seed = seed

    @property
    def name(self) -> str:
        return f"synthetic_signal_{self._pattern}"

    @property
    def n_features(self) -> int:
        return self._n_channels

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def feature_names(self) -> list[str]:
        return [f"sig_{i}" for i in range(self._n_channels)]

    def load(self) -> np.ndarray:
        """Generate modulated signals of shape (n_samples, n_channels)."""
        rng = np.random.default_rng(self._seed)
        t = np.linspace(0, 4 * np.pi, self._n_samples)
        data = np.empty((self._n_samples, self._n_channels), dtype=np.float64)

        for ch in range(self._n_channels):
            mode = self._channel_mode(ch)
            carrier_freq = 1.0 + rng.uniform(0.5, 3.0)
            mod_freq = rng.uniform(0.1, 0.5)

            if mode == "am":
                data[:, ch] = self._am_signal(t, carrier_freq, mod_freq, rng)
            else:
                data[:, ch] = self._fm_signal(t, carrier_freq, mod_freq, rng)

            # Add regime transition: abrupt change at midpoint
            mid = self._n_samples // 2
            data[mid:, ch] *= rng.uniform(0.5, 2.0)
            data[mid:, ch] += rng.uniform(-0.5, 0.5)

        return data

    def _channel_mode(self, ch: int) -> str:
        if self._pattern == "mixed":
            return "am" if ch % 2 == 0 else "fm"
        return self._pattern

    @staticmethod
    def _am_signal(
        t: np.ndarray,
        carrier_freq: float,
        mod_freq: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Amplitude-modulated sinusoid."""
        carrier = np.sin(carrier_freq * t)
        modulator = 1.0 + 0.7 * np.sin(mod_freq * t)
        noise = rng.standard_normal(len(t)) * 0.05
        return carrier * modulator + noise

    @staticmethod
    def _fm_signal(
        t: np.ndarray,
        carrier_freq: float,
        mod_freq: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Frequency-modulated sinusoid."""
        mod_index = 2.0
        phase = carrier_freq * t + mod_index * np.sin(mod_freq * t)
        noise = rng.standard_normal(len(t)) * 0.05
        return np.sin(phase) + noise
