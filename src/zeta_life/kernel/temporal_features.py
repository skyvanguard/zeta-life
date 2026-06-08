"""Temporal feature banks for the Conscious Kernel's control loop.

This module wires the Riemann-zeta frequencies *into the control path* of the
kernel (the world model / planner), instead of leaving them confined to the
dream-consolidation rhythm. It exposes an :class:`OscillatorBank` that turns a
discrete time step ``t`` into a vector of sinusoidal features

.. math::

    \\phi(t) = \\frac{1}{\\lVert\\cdot\\rVert}
        \\big[\\, a_1\\cos(\\gamma_1 t),\\, a_1\\sin(\\gamma_1 t),\\, \\dots,\\,
                  a_M\\cos(\\gamma_M t),\\, a_M\\sin(\\gamma_M t) \\,\\big]

with Abel amplitudes ``a_k = exp(-sigma * |gamma_k|)`` -- exactly the spectral
decomposition of the project's kernel ``K_sigma(t) = 2 * sum a_k cos(gamma_k t)``.
The bank is fed as extra input to the world model's transition, so the model can
*anticipate* time-structured dynamics rather than only react to them.

Three factory configurations make the kernel's central research question a fair,
controlled experiment:

- :meth:`OscillatorBank.zeta`   -- frequencies = non-trivial zeta zeros (hypothesis)
- :meth:`OscillatorBank.random` -- frequencies drawn uniformly from the same range
  (perfectly parameter-matched to ``zeta``; isolates "are the *specific*
  frequencies special?")
- :meth:`OscillatorBank.learned` -- frequencies are trainable parameters
  (the parameter-matched "RNN equivalent" that must learn its own temporal code)

All banks L2-normalise their output per step, so ``zeta`` and ``random`` differ
only in the *structure* (phase relationships) of the features, never in scale --
the cleanest possible control.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from zeta_life.core.zeta_constants import get_zeta_zeros


class OscillatorBank(nn.Module):
    """A bank of sinusoidal oscillators that maps time ``t`` to a feature vector.

    Parameters
    ----------
    frequencies : list[float]
        Angular frequencies ``gamma_k`` for the oscillators.
    sigma : float
        Abel regularisation; sets the amplitude envelope ``a_k = exp(-sigma|g|)``.
    learnable : bool
        If ``True`` the frequencies are trainable :class:`~torch.nn.Parameter`
        objects (the model learns its own temporal code); otherwise they are a
        frozen buffer.
    """

    def __init__(
        self,
        frequencies: list[float],
        sigma: float = 0.1,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.learnable = learnable
        self.n_freqs = len(frequencies)

        freq_tensor = torch.tensor(frequencies, dtype=torch.float32)
        if learnable:
            self.frequencies = nn.Parameter(freq_tensor)
        else:
            self.register_buffer("frequencies", freq_tensor)

    @property
    def dim(self) -> int:
        """Output dimensionality of the feature vector (``2 * n_freqs``)."""
        return 2 * self.n_freqs

    def forward(self, t: float) -> Tensor:
        """Compute the L2-normalised oscillator features at time ``t``.

        Parameters
        ----------
        t : float
            Discrete (or continuous) time step.

        Returns
        -------
        Tensor
            Feature vector of shape ``(2 * n_freqs,)``.
        """
        freqs = self.frequencies
        angles = freqs * float(t)
        amps = torch.exp(-self.sigma * freqs.abs())
        feats = torch.cat([amps * torch.cos(angles), amps * torch.sin(angles)])
        norm = torch.linalg.vector_norm(feats)
        return feats / (norm + 1e-8)

    # ------------------------------------------------------------------
    # Factory configurations (the three experimental arms)
    # ------------------------------------------------------------------

    @classmethod
    def zeta(cls, M: int = 15, sigma: float = 0.1) -> "OscillatorBank":
        """Bank whose frequencies are the first ``M`` non-trivial zeta zeros."""
        return cls(get_zeta_zeros(M), sigma=sigma, learnable=False)

    @classmethod
    def random(
        cls,
        M: int = 15,
        sigma: float = 0.1,
        seed: int = 0,
        freq_range: tuple[float, float] | None = None,
    ) -> "OscillatorBank":
        """Bank with ``M`` frozen random frequencies in the zeta-zero range.

        Parameter-matched to :meth:`zeta`: same count, same Abel envelope, same
        output dimension -- only the frequency *values* are arbitrary. Drawing
        from the same range keeps the amplitude distribution comparable, so any
        performance gap is attributable to the structure of the zeta spectrum,
        not to a scale or capacity difference.
        """
        if freq_range is None:
            zeros = get_zeta_zeros(M)
            freq_range = (min(zeros), max(zeros))
        lo, hi = freq_range
        g = torch.Generator().manual_seed(seed)
        freqs = (lo + (hi - lo) * torch.rand(M, generator=g)).tolist()
        return cls(freqs, sigma=sigma, learnable=False)

    @classmethod
    def learned(
        cls,
        M: int = 15,
        sigma: float = 0.1,
        seed: int = 0,
        freq_range: tuple[float, float] | None = None,
    ) -> "OscillatorBank":
        """Bank with ``M`` *trainable* frequencies (random init).

        The parameter-matched "RNN equivalent": same network shape as the zeta
        and random banks, but the temporal frequencies are learned end-to-end
        from the control task instead of fixed to the zeta zeros. If ``zeta``
        cannot beat ``learned``, the specific zeta frequencies add nothing the
        network could not discover on its own.
        """
        if freq_range is None:
            zeros = get_zeta_zeros(M)
            freq_range = (min(zeros), max(zeros))
        lo, hi = freq_range
        g = torch.Generator().manual_seed(seed)
        freqs = (lo + (hi - lo) * torch.rand(M, generator=g)).tolist()
        return cls(freqs, sigma=sigma, learnable=True)

    # ------------------------------------------------------------------
    # Principled fixed bases (recommended over zeta for a fixed temporal code)
    # ------------------------------------------------------------------
    #
    # The spacing-statistics experiment showed an equispaced lattice gives the
    # best basis quality (smallest covering radius, best-conditioned temporal
    # code) and that zeta's GUE spacing buys nothing functional. So for a FIXED
    # basis these are the principled defaults; for an ADAPTIVE basis use
    # :meth:`learned`. The zeta/random/gue/poisson factories remain for
    # reproducing the comparison studies.

    @classmethod
    def fourier(
        cls,
        M: int = 40,
        sigma: float = 0.0,
        freq_range: tuple[float, float] = (5.0, 80.0),
    ) -> "OscillatorBank":
        """Equispaced (linear) frequency lattice -- the classic Fourier basis.

        Winner of the spacing-statistics experiment (best covering radius and
        best-conditioned temporal code). The recommended fixed temporal basis.
        ``sigma=0`` keeps amplitudes flat so every frequency contributes.
        """
        lo, hi = freq_range
        return cls(torch.linspace(lo, hi, M).tolist(), sigma=sigma, learnable=False)

    @classmethod
    def log_spaced(
        cls,
        M: int = 40,
        sigma: float = 0.0,
        freq_range: tuple[float, float] = (1.0, 80.0),
    ) -> "OscillatorBank":
        """Geometrically (log) spaced frequencies -- a multi-scale basis.

        The spacing used by Transformer sinusoidal positional encodings: dense
        at low frequencies, sparse at high ones. Prefer this over :meth:`fourier`
        when the signal has structure across many time-scales (slow trends plus
        fast detail) rather than a flat band. Requires ``lo > 0``.
        """
        lo, hi = freq_range
        if lo <= 0.0:
            raise ValueError("log_spaced requires freq_range lo > 0")
        freqs = torch.logspace(math.log10(lo), math.log10(hi), M).tolist()
        return cls(freqs, sigma=sigma, learnable=False)

    # ------------------------------------------------------------------
    # Spacing-statistics configurations (the decisive zeta-vs-baseline test)
    # ------------------------------------------------------------------
    #
    # The genuinely distinctive property of the Riemann-zeta zeros is not their
    # arithmetic values but the *statistics of their gaps*: consecutive zeros
    # show GUE level repulsion (Montgomery-Odlyzko), i.e. small gaps are
    # suppressed. To test whether THAT is what (if anything) makes zeta useful,
    # :meth:`by_spacing` builds banks with the SAME range and SAME mean density
    # but a chosen gap statistic, so only the spacing distribution differs.

    @classmethod
    def by_spacing(
        cls,
        statistic: str,
        M: int = 40,
        sigma: float = 0.0,
        freq_range: tuple[float, float] = (5.0, 80.0),
        seed: int = 0,
    ) -> "OscillatorBank":
        """Bank whose frequencies follow a chosen spacing *statistic*.

        Parameters
        ----------
        statistic : str
            One of ``'zeta'``, ``'gue'``, ``'poisson'``, ``'uniform'``.
        M : int
            Number of frequencies (all statistics share this count and range, so
            mean density is identical -- only the gap distribution differs).
        sigma : float
            Abel envelope. **Default 0.0 (flat amplitudes)** so every frequency
            contributes equally; with the usual sigma~0.1 only ~4 oscillators
            survive and the spacing statistic becomes irrelevant.
        freq_range : tuple[float, float]
            Common ``(lo, hi)`` band shared by all statistics.
        seed : int
            Seed for the stochastic statistics (gue, poisson).
        """
        freqs = generate_spacing_frequencies(statistic, M, freq_range, seed)
        return cls(freqs.tolist(), sigma=sigma, learnable=False)

    def spacing_stats(self) -> dict:
        """Summarise the normalised nearest-neighbour gap distribution.

        Returns ``gap_cv`` (coefficient of variation of the gaps; 0 for a rigid
        lattice, ~1 for Poisson) and ``frac_small`` (fraction of normalised gaps
        below 0.5; high for Poisson clustering, ~0 under level repulsion).
        """
        return spacing_stats(self.frequencies)


# ---------------------------------------------------------------------------
# Spacing-statistics helpers (module-level so experiments can reuse them)
# ---------------------------------------------------------------------------

def _rescale(x: Tensor, lo: float, hi: float) -> Tensor:
    """Affinely map ``x`` onto ``[lo, hi]`` (preserves relative gap structure)."""
    xmin, xmax = x.min(), x.max()
    span = (xmax - xmin).clamp(min=1e-12)
    return lo + (x - xmin) * (hi - lo) / span


def generate_spacing_frequencies(
    statistic: str,
    M: int = 40,
    freq_range: tuple[float, float] = (5.0, 80.0),
    seed: int = 0,
) -> Tensor:
    """Generate ``M`` frequencies in ``freq_range`` with a given gap statistic.

    - ``uniform``  : equispaced lattice (zero gap variance, maximal rigidity).
    - ``poisson``  : sorted uniform points (exponential gaps, no repulsion).
    - ``gue``      : eigenvalues of a random complex-Hermitian (GUE) matrix,
      which exhibit genuine level repulsion -- the same statistic as the zeta
      zeros, but *not* the zeros themselves.
    - ``zeta``     : the actual non-trivial zeta zeros, rescaled to the band so
      range and density match the others (their relative GUE spacing survives
      the affine rescale).
    """
    lo, hi = freq_range
    g = torch.Generator().manual_seed(seed)
    if statistic == "uniform":
        return torch.linspace(lo, hi, M)
    if statistic == "poisson":
        u = torch.sort(torch.rand(M, generator=g)).values
        return _rescale(u, lo, hi)
    if statistic == "gue":
        re = torch.randn(M, M, generator=g)
        im = torch.randn(M, M, generator=g)
        herm = (torch.complex(re, im) + torch.complex(re, im).conj().T) / 2
        ev = torch.linalg.eigvalsh(herm)  # real, ascending
        return _rescale(ev, lo, hi)
    if statistic == "zeta":
        ev = torch.tensor(get_zeta_zeros(M), dtype=torch.float32)
        return _rescale(ev, lo, hi)
    raise ValueError(f"unknown spacing statistic: {statistic!r}")


def spacing_stats(frequencies: Tensor) -> dict:
    """Coefficient of variation of gaps and fraction of small normalised gaps."""
    s = torch.sort(frequencies).values
    gaps = s[1:] - s[:-1]
    mean_gap = gaps.mean().clamp(min=1e-12)
    norm = gaps / mean_gap
    return {
        "gap_cv": float((gaps.std(unbiased=False) / mean_gap).item()),
        "frac_small": float((norm < 0.5).float().mean().item()),
    }
