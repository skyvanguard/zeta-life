"""Core abstractions for Zeta Life.

This module contains the abstract vertex system that replaces
Jungian archetypes with semantically-neutral geometric vertices.
"""

from .vertex import Vertex, BehaviorVector, VertexBehaviors
from .tetrahedral_space import TetrahedralSpace, get_tetrahedral_space

__all__ = [
    'Vertex',
    'BehaviorVector',
    'VertexBehaviors',
    'TetrahedralSpace',
    'get_tetrahedral_space',
]
