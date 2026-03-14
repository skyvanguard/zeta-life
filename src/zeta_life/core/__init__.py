"""Core abstractions for Zeta Life.

This module contains the abstract vertex system that replaces
Jungian archetypes with semantically-neutral geometric vertices,
plus the memory system for long-term state persistence.
"""

from .tetrahedral_space import TetrahedralSpace, get_tetrahedral_space
from .vertex import BehaviorVector, Vertex, VertexBehaviors
from .zeta_constants import KNOWN_ZETA_ZEROS, get_zeta_zeros
from .zeta_memory import (
    EpisodicMemory,
    MemoryAwarePsyche,
    ProceduralMemory,
    SemanticMemory,
    ZetaMemorySystem,
)
from .zeta_rnn import (
    ZetaLSTM,
    ZetaLSTMCell,
    ZetaLSTMExperiment,
    ZetaMemoryLayer,
    ZetaSequenceGenerator,
)

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
    # Memory system
    'EpisodicMemory',
    'SemanticMemory',
    'ProceduralMemory',
    'ZetaMemorySystem',
    'MemoryAwarePsyche',
    # RNN system
    'ZetaMemoryLayer',
    'ZetaLSTMCell',
    'ZetaLSTM',
    'ZetaSequenceGenerator',
    'ZetaLSTMExperiment',
]
