"""Narrative layer for visualization and demos.

This module provides optional mappings from abstract vertices (V0-V3)
to human-readable names for visualization purposes. It never affects
calculations - only display.
"""

from .mapper import NarrativeMapper

__all__ = ['NarrativeMapper']
