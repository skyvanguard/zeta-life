"""Abstract vertex system for Zeta Life.

This module defines semantically-neutral vertices (V0-V3) with parametric
behaviors, replacing the Jungian archetype system. The abstraction preserves
all emergent dynamics while eliminating human psychological bias.

Correspondence Map:
    V0 (index 0) ←→ LEADER ←→ PERSONA
    V1 (index 1) ←→ DISRUPTOR ←→ SOMBRA
    V2 (index 2) ←→ FOLLOWER ←→ ANIMA
    V3 (index 3) ←→ EXPLORER ←→ ANIMUS
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import json


class Vertex(Enum):
    """Abstract vertices of the tetrahedral state space.

    These replace Jungian archetypes with semantically-neutral identifiers.
    All dynamics and calculations use only these indices.

    For backwards compatibility, the old archetype names are available
    as aliases: PERSONA=V0, SOMBRA=V1, ANIMA=V2, ANIMUS=V3
    """
    V0 = 0
    V1 = 1
    V2 = 2
    V3 = 3

    # Backwards compatibility aliases (deprecated, use V0-V3)
    PERSONA = 0
    SOMBRA = 1
    ANIMA = 2
    ANIMUS = 3


@dataclass
class BehaviorVector:
    """Parametric behavior specification for a vertex.
    
    This replaces hardcoded archetype-specific behaviors with configurable
    parameters. The default values (all 1.0 or 0.0) represent neutral behavior.
    
    Attributes:
        field_response: Response multiplier to force field.
            >1.0 = stronger following (leader-like)
            <1.0 = weaker following
            Default: 1.0
        attraction: Proximity seeking multiplier.
            >1.0 = more social/cohesive
            <1.0 = more isolated
            Default: 1.0
        exploration: Probability of random movement [0-1].
            Higher = more exploratory behavior
            Default: 0.0
        opposition: Probability of opposing field direction [0-1].
            Higher = more disruptive behavior
            Default: 0.0
    """
    field_response: float = 1.0
    attraction: float = 1.0
    exploration: float = 0.0
    opposition: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'field_response': self.field_response,
            'attraction': self.attraction,
            'exploration': self.exploration,
            'opposition': self.opposition,
        }
    
    def to_list(self) -> list:
        """Convert to list [field_response, attraction, exploration, opposition]."""
        return [self.field_response, self.attraction, self.exploration, self.opposition]
    
    @classmethod
    def from_dict(cls, d: dict) -> 'BehaviorVector':
        """Create from dictionary."""
        return cls(
            field_response=d.get('field_response', 1.0),
            attraction=d.get('attraction', 1.0),
            exploration=d.get('exploration', 0.0),
            opposition=d.get('opposition', 0.0),
        )
    
    @classmethod
    def from_list(cls, lst: list) -> 'BehaviorVector':
        """Create from list [field_response, attraction, exploration, opposition]."""
        return cls(
            field_response=lst[0] if len(lst) > 0 else 1.0,
            attraction=lst[1] if len(lst) > 1 else 1.0,
            exploration=lst[2] if len(lst) > 2 else 0.0,
            opposition=lst[3] if len(lst) > 3 else 0.0,
        )


@dataclass
class VertexBehaviors:
    """Complete behavior configuration for all vertices.
    
    This class manages the parametric behaviors for all four vertices,
    allowing different configurations for experiments and controls.
    """
    behaviors: Dict[Vertex, BehaviorVector] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure all vertices have behaviors."""
        for v in Vertex:
            if v not in self.behaviors:
                self.behaviors[v] = BehaviorVector()
    
    @classmethod
    def default(cls) -> 'VertexBehaviors':
        """Default configuration preserving current dynamics.
        
        This configuration produces behavior equivalent to the original
        Jungian archetype system:
            V0 (PERSONA): Strong field following (leader)
            V1 (SOMBRA): Occasional opposition (disruptor)
            V2 (ANIMA): Enhanced attraction (follower)
            V3 (ANIMUS): Occasional exploration (explorer)
        """
        return cls({
            Vertex.V0: BehaviorVector(
                field_response=1.3, 
                attraction=1.0, 
                exploration=0.0, 
                opposition=0.0
            ),
            Vertex.V1: BehaviorVector(
                field_response=1.0, 
                attraction=1.0, 
                exploration=0.0, 
                opposition=0.3
            ),
            Vertex.V2: BehaviorVector(
                field_response=1.0, 
                attraction=1.1, 
                exploration=0.0, 
                opposition=0.0
            ),
            Vertex.V3: BehaviorVector(
                field_response=1.0, 
                attraction=1.0, 
                exploration=0.2, 
                opposition=0.0
            ),
        })
    
    @classmethod
    def uniform(cls) -> 'VertexBehaviors':
        """All vertices behave identically.
        
        This is the null hypothesis / control condition for experiments.
        All vertices have neutral behavior vectors.
        """
        return cls({v: BehaviorVector() for v in Vertex})
    
    @classmethod
    def from_json(cls, path: str) -> 'VertexBehaviors':
        """Load behavior configuration from JSON file.
        
        Expected format:
        {
            "behaviors": {
                "V0": {"field_response": 1.3, "attraction": 1.0, ...},
                "V1": {...},
                ...
            }
        }
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        behaviors = {}
        for k, v in data.get('behaviors', {}).items():
            vertex = Vertex[k] if isinstance(k, str) else Vertex(k)
            behaviors[vertex] = BehaviorVector.from_dict(v)
        
        return cls(behaviors)
    
    def to_json(self, path: str) -> None:
        """Save behavior configuration to JSON file."""
        data = {
            'behaviors': {
                v.name: self.behaviors[v].to_dict()
                for v in Vertex
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def get(self, vertex: Vertex) -> BehaviorVector:
        """Get behavior vector for a vertex."""
        return self.behaviors.get(vertex, BehaviorVector())
    
    def __getitem__(self, vertex: Vertex) -> BehaviorVector:
        """Allow dict-like access: behaviors[Vertex.V0]."""
        return self.get(vertex)


# Backwards compatibility alias (deprecated)
# This allows gradual migration from Archetype to Vertex
Archetype = Vertex
