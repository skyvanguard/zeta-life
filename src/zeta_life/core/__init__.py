"""Core abstractions for Zeta Life.

Shared primitives: the zeta constants and the abstract vertex / tetrahedral
geometry. (The heavy zeta_memory / zeta_rnn / zeta_resonance modules were
archived in the 2026-06 refocus; see the legacy/pre-refocus-snapshot branch.)
"""

from .tetrahedral_space import TetrahedralSpace, get_tetrahedral_space
from .vertex import BehaviorVector, Vertex, VertexBehaviors
from .zeta_constants import KNOWN_ZETA_ZEROS, get_zeta_zeros

__all__ = [
    # Zeta constants
    'KNOWN_ZETA_ZEROS',
    'get_zeta_zeros',
    # Vertex system
    'Vertex',
    'BehaviorVector',
    'VertexBehaviors',
    'TetrahedralSpace',
    'get_tetrahedral_space',
]
