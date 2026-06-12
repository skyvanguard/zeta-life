"""Instrumentation — paired per-tick logging for the science pipeline.

The :class:`TickLogger` records one JSON object per tick (scores, Psi,
free energy, the second-order precision error, the workspace winner, ...),
append-only, so an experiment or the Yvyra bridge can later analyse the
coupling between signals without polluting the kernel itself.
"""

from .tick_logger import TickLogger, load_ticks

__all__ = ['TickLogger', 'load_ticks']
