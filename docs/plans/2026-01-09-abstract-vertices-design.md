# Abstract Vertices Design

**Date**: 2026-01-09
**Status**: Approved
**Author**: Co-designed with Claude

## Overview

This document describes the abstraction of Jungian archetypes (PERSONA, SOMBRA, ANIMA, ANIMUS) into a quantifiable, reproducible system of abstract vertices (V0, V1, V2, V3) with parametric behaviors.

### Goals

1. Eliminate semantic/psychological bias from calculations
2. Preserve emergent dynamics (attractors, compensation, vertical coherence)
3. Enable rigorous experiments on normative self without human-loaded concepts
4. Maintain optional narrative layer for visualization/demos

### Non-Goals

- Changing the tetrahedral geometry
- Altering the zeta kernel mathematics
- Breaking backwards compatibility immediately

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Complementarity | Geometric opposition | Derived from max-distance vertex pairs, not hardcoded |
| Movement behaviors | Parametric vectors | Configurable data, not switch statements |
| Naming convention | V0-V3 in core | Zero semantic loading in calculations |
| Narrative layer | JSON configuration | Separates display from logic, versionable |

---

## Core Data Structures

### 1. Vertex Enum

Replaces `Archetype` enum.

```python
from enum import Enum

class Vertex(Enum):
    """Abstract vertices of the tetrahedral state space."""
    V0 = 0
    V1 = 1
    V2 = 2
    V3 = 3
```

### 2. BehaviorVector

Parametric behavior specification for each vertex.

```python
from dataclasses import dataclass

@dataclass
class BehaviorVector:
    """Parametric behavior for each vertex.

    Attributes:
        field_response: Response multiplier to force field (>1 = stronger following)
        attraction: Proximity seeking multiplier (>1 = more social)
        exploration: Probability of random movement [0-1]
        opposition: Probability of opposing field direction [0-1]
    """
    field_response: float = 1.0
    attraction: float = 1.0
    exploration: float = 0.0
    opposition: float = 0.0

    def to_dict(self) -> dict:
        return {
            'field_response': self.field_response,
            'attraction': self.attraction,
            'exploration': self.exploration,
            'opposition': self.opposition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BehaviorVector':
        return cls(**d)
```

### 3. VertexBehaviors

Complete behavior configuration for all vertices.

```python
from typing import Dict
import json

@dataclass
class VertexBehaviors:
    """Complete behavior configuration for all vertices."""
    behaviors: Dict[Vertex, BehaviorVector]

    @classmethod
    def default(cls) -> 'VertexBehaviors':
        """Default configuration preserving current Jungian-equivalent dynamics."""
        return cls({
            Vertex.V0: BehaviorVector(field_response=1.3, attraction=1.0, exploration=0.0, opposition=0.0),
            Vertex.V1: BehaviorVector(field_response=1.0, attraction=1.0, exploration=0.0, opposition=0.3),
            Vertex.V2: BehaviorVector(field_response=1.0, attraction=1.1, exploration=0.0, opposition=0.0),
            Vertex.V3: BehaviorVector(field_response=1.0, attraction=1.0, exploration=0.2, opposition=0.0),
        })

    @classmethod
    def uniform(cls) -> 'VertexBehaviors':
        """All vertices behave identically (control condition for experiments)."""
        return cls({v: BehaviorVector() for v in Vertex})

    @classmethod
    def from_json(cls, path: str) -> 'VertexBehaviors':
        """Load behavior configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        behaviors = {
            Vertex[k]: BehaviorVector.from_dict(v)
            for k, v in data['behaviors'].items()
        }
        return cls(behaviors)

    def get(self, vertex: Vertex) -> BehaviorVector:
        """Get behavior vector for a vertex."""
        return self.behaviors[vertex]
```

### 4. TetrahedralSpace (updated)

Geometric complementarity computed from vertex distances.

```python
class TetrahedralSpace:
    def __init__(self) -> None:
        # Vertices in 3D (semantic-agnostic)
        self.vertices = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=torch.float32)
        self.vertices = F.normalize(self.vertices, dim=1)
        self.center = self.vertices.mean(dim=0)

        # Compute geometric complements once
        self._complement_map = self._compute_geometric_complements()

    def get_complement(self, vertex: Vertex) -> Vertex:
        """Returns geometrically opposite vertex (max distance)."""
        return self._complement_map[vertex]

    def _compute_geometric_complements(self) -> Dict[Vertex, Vertex]:
        """Compute max-distance pairs from geometry."""
        complements = {}
        for i, v in enumerate(Vertex):
            distances = torch.norm(self.vertices - self.vertices[i], dim=1)
            max_idx = distances.argmax().item()
            complements[v] = Vertex(max_idx)
        return complements

    # ... rest of existing methods unchanged
```

---

## Narrative Layer

### JSON Configuration Structure

File: `src/zeta_life/narrative/configs/jungian.json`

```json
{
  "name": "jungian",
  "description": "Classical Jungian archetype mapping for visualization",
  "vertices": {
    "V0": {
      "functional": "LEADER",
      "narrative": "PERSONA",
      "description": "La máscara que mostramos al mundo",
      "color": "#E53E3E",
      "symbol": "☉"
    },
    "V1": {
      "functional": "DISRUPTOR",
      "narrative": "SOMBRA",
      "description": "Lo reprimido, el lado oscuro",
      "color": "#553C9A",
      "symbol": "☽"
    },
    "V2": {
      "functional": "FOLLOWER",
      "narrative": "ANIMA",
      "description": "El lado emocional, receptivo",
      "color": "#3182CE",
      "symbol": "♀"
    },
    "V3": {
      "functional": "EXPLORER",
      "narrative": "ANIMUS",
      "description": "El lado racional, activo",
      "color": "#DD6B20",
      "symbol": "♂"
    }
  },
  "center": {
    "narrative": "SELF",
    "description": "Integración total",
    "color": "#D69E2E",
    "symbol": "✧"
  },
  "complement_pairs": [
    ["V0", "V1"],
    ["V2", "V3"]
  ]
}
```

### NarrativeMapper Class

File: `src/zeta_life/narrative/mapper.py`

```python
import json
from typing import Optional
from ..core.vertex import Vertex

class NarrativeMapper:
    """Optional layer for visualization/demo. Never used in calculations."""

    def __init__(self, config_path: Optional[str] = None):
        self.mapping = self._load(config_path) if config_path else {}
        self.name = self.mapping.get("name", "neutral")

    def _load(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_name(self, vertex: Vertex, layer: str = "functional") -> str:
        """Get display name for a vertex.

        Args:
            vertex: The vertex to get name for
            layer: 'functional' | 'narrative' | 'raw'

        Returns:
            Display name string
        """
        if not self.mapping or layer == "raw":
            return vertex.name  # V0, V1, V2, V3

        vertex_data = self.mapping.get("vertices", {}).get(vertex.name, {})
        return vertex_data.get(layer, vertex.name)

    def get_color(self, vertex: Vertex) -> str:
        """Get display color for a vertex."""
        if not self.mapping:
            return "#888888"
        return self.mapping.get("vertices", {}).get(vertex.name, {}).get("color", "#888888")

    def get_symbol(self, vertex: Vertex) -> str:
        """Get display symbol for a vertex."""
        if not self.mapping:
            return "●"
        return self.mapping.get("vertices", {}).get(vertex.name, {}).get("symbol", "●")

    def get_description(self, vertex: Vertex) -> str:
        """Get description for a vertex."""
        if not self.mapping:
            return f"Vertex {vertex.name}"
        return self.mapping.get("vertices", {}).get(vertex.name, {}).get("description", "")
```

---

## Correspondence Map

Reference table for mapping between systems:

| Vertex | Index | Functional | Jungian | Behavior Vector | Geometric Complement |
|--------|-------|------------|---------|-----------------|---------------------|
| V0 | 0 | LEADER | PERSONA | [1.3, 1.0, 0.0, 0.0] | V1 |
| V1 | 1 | DISRUPTOR | SOMBRA | [1.0, 1.0, 0.0, 0.3] | V0 |
| V2 | 2 | FOLLOWER | ANIMA | [1.0, 1.1, 0.0, 0.0] | V3 |
| V3 | 3 | EXPLORER | ANIMUS | [1.0, 1.0, 0.2, 0.0] | V2 |

### Behavior Vector Format

`[field_response, attraction, exploration, opposition]`

- **field_response**: Multiplier for force field following (default 1.0)
- **attraction**: Multiplier for proximity seeking (default 1.0)
- **exploration**: Probability of random movement (default 0.0)
- **opposition**: Probability of opposing field (default 0.0)

---

## Files to Modify

### New Files

```
src/zeta_life/
├── core/
│   ├── __init__.py
│   ├── vertex.py              # Vertex, BehaviorVector, VertexBehaviors
│   └── tetrahedral_space.py   # Moved from psyche, with geometric complements
│
└── narrative/
    ├── __init__.py
    ├── mapper.py              # NarrativeMapper class
    └── configs/
        ├── jungian.json       # Jung mapping
        ├── functional.json    # Functional roles only
        └── neutral.json       # V0-V3 only (no narrative)
```

### Modified Files

| File | Changes |
|------|---------|
| `psyche/zeta_psyche.py` | Replace `Archetype` with `Vertex`, remove hardcoded colors/descriptions |
| `consciousness/micro_psyche.py` | Use `VertexBehaviors` for movement, geometric `get_complement()` |
| `consciousness/top_down_modulator.py` | Use `TetrahedralSpace.get_complement()` |
| `consciousness/organism_consciousness.py` | Same complement refactor |
| `visualization/*.py` | Use `NarrativeMapper` for colors/names |

### Backwards Compatibility (temporary)

```python
# In zeta_psyche.py during transition period
from ..core.vertex import Vertex

# Deprecated alias - will be removed in v2.0
Archetype = Vertex

import warnings
def _archetype_deprecation_warning():
    warnings.warn(
        "Archetype is deprecated, use Vertex instead",
        DeprecationWarning,
        stacklevel=3
    )
```

---

## Experiments Enabled

### Why Abstraction Eliminates Bias

**Before (Jungian):**
```python
if dominant == Archetype.PERSONA:
    # "PERSONA follows group" ← semantic bias
    return (1.3, 1.3)
```

**After (Abstract):**
```python
behavior = self.behaviors.get(self.dominant)  # V0, V1...
return (behavior.field_response, behavior.attraction)
```

The behavior is now **configurable data**, not a psychological interpretation.

### Enabled Experiments

| Experiment | How Abstraction Helps |
|------------|----------------------|
| **IPUESA** (identity preference) | Test P(A) > P(B) without "A = PERSONA" influencing interpretation |
| **Permutation test** | Swap behavior vectors between vertices, measure if dynamics change |
| **Null hypothesis** | Use `VertexBehaviors.uniform()` as control (all identical) |
| **Ablation studies** | Systematically disable components (exploration=0, opposition=0) |
| **Cross-narrative** | Same dynamics, different narrative mappings, verify identical results |

---

## Summary

> The V0-V3 abstraction with parametric behavior vectors preserves emergent dynamics (attractors, compensation, vertical coherence) while eliminating human semantic loading. Behaviors are configurable data, not psychological interpretations. This enables normative self experiments where attractor preference is measured without names like "PERSONA" or "SOMBRA" introducing analysis bias. The optional narrative layer allows mapping back to Jung for visualization without affecting calculations.

---

## Implementation Plan

1. **Phase 1**: Create `core/vertex.py` with new data structures
2. **Phase 2**: Update `TetrahedralSpace` with geometric complements
3. **Phase 3**: Create `narrative/` module with mapper and configs
4. **Phase 4**: Refactor `micro_psyche.py` to use parametric behaviors
5. **Phase 5**: Refactor `top_down_modulator.py` and `organism_consciousness.py`
6. **Phase 6**: Update visualization to use `NarrativeMapper`
7. **Phase 7**: Run existing tests, verify dynamics unchanged
8. **Phase 8**: Run IPUESA experiment with abstract system
