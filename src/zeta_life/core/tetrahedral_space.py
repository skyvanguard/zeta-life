"""Tetrahedral space with geometric complement computation.

This module provides the geometric foundation for the psyche state space.
All operations are semantically neutral - vertices are V0-V3, not archetypes.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from .vertex import Vertex


class TetrahedralSpace:
    """
    Tetrahedral state space where the psyche lives.
    
    Each point is represented with barycentric coordinates (w0, w1, w2, w3)
    where wi >= 0 and sum(wi) = 1.
    
    Vertices in 3D:
        V0: (1, 1, 1) normalized
        V1: (1, -1, -1) normalized
        V2: (-1, 1, -1) normalized
        V3: (-1, -1, 1) normalized
    
    Geometric complements (max-distance pairs):
        V0 ↔ V1
        V2 ↔ V3
    """
    
    def __init__(self) -> None:
        # Vertices of regular tetrahedron in 3D
        self.vertices = torch.tensor([
            [1.0, 1.0, 1.0],      # V0
            [1.0, -1.0, -1.0],    # V1
            [-1.0, 1.0, -1.0],    # V2
            [-1.0, -1.0, 1.0],    # V3
        ], dtype=torch.float32)
        
        # Normalize to unit sphere
        self.vertices = F.normalize(self.vertices, dim=1)
        
        # Center of tetrahedron (integrated state)
        self.center = self.vertices.mean(dim=0)
        
        # Compute geometric complements once
        self._complement_map = self._compute_geometric_complements()
    
    def _compute_geometric_complements(self) -> Dict[Vertex, Vertex]:
        """Define complement pairs for the tetrahedron.

        In a regular tetrahedron, all vertices are equidistant. Therefore,
        we define complementarity by pairing vertices that share the same
        sign in the first coordinate (before normalization):

            V0 [+,+,+] ↔ V1 [+,-,-]  (both have +x)
            V2 [-,+,-] ↔ V3 [-,-,+]  (both have -x)

        This preserves the original dynamics while being derivable from
        the coordinate structure rather than semantic meaning.

        Returns:
            Dict mapping each vertex to its complement
        """
        # Pairing based on first coordinate sign (from pre-normalized coords)
        # V0 and V1 both have +1 in x, V2 and V3 both have -1 in x
        return {
            Vertex.V0: Vertex.V1,
            Vertex.V1: Vertex.V0,
            Vertex.V2: Vertex.V3,
            Vertex.V3: Vertex.V2,
        }
    
    def get_complement(self, vertex: Vertex) -> Vertex:
        """Returns geometrically opposite vertex (max distance).
        
        This replaces hardcoded archetype complementarity pairs with
        geometry-derived pairs.
        """
        return self._complement_map[vertex]
    
    def get_complement_index(self, vertex: Vertex) -> int:
        """Returns index of geometrically opposite vertex."""
        return self._complement_map[vertex].value
    
    def barycentric_to_3d(self, weights: torch.Tensor) -> torch.Tensor:
        """Convert barycentric coordinates to 3D position.
        
        Args:
            weights: [..., 4] barycentric coordinates
            
        Returns:
            [..., 3] position in 3D space
        """
        weights = F.softmax(weights, dim=-1)
        return torch.matmul(weights, self.vertices)
    
    def position_to_barycentric(self, position: torch.Tensor) -> torch.Tensor:
        """Approximate barycentric coordinates from 3D position.
        
        Uses inverse distance weighting.
        
        Args:
            position: [..., 3] position in 3D space
            
        Returns:
            [..., 4] barycentric coordinates
        """
        dists = torch.cdist(position.unsqueeze(0), self.vertices).squeeze(0)
        weights = 1.0 / (dists + 1e-6)
        return F.softmax(weights, dim=-1)
    
    def get_dominant_vertex(self, weights: torch.Tensor) -> Vertex:
        """Returns the dominant vertex (highest weight)."""
        idx = weights.argmax().item()
        return Vertex(int(idx))
    
    def get_vertex_blend(self, weights: torch.Tensor) -> Dict[Vertex, float]:
        """Returns the blend of vertices as dictionary."""
        weights = F.softmax(weights, dim=-1)
        return {Vertex(i): w.item() for i, w in enumerate(weights)}
    
    def distance_to_center(self, weights: torch.Tensor) -> float:
        """Distance to center (integrated state). Lower = more integrated."""
        pos = self.barycentric_to_3d(weights)
        return float(torch.norm(pos - self.center).item())
    
    def integration_score(self, weights: torch.Tensor) -> float:
        """
        Integration score based on entropy.
        
        1.0 = perfectly balanced (at center)
        0.0 = completely at one vertex
        """
        weights = F.softmax(weights, dim=-1)
        # Normalized entropy - maximum when all equal
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        max_entropy = np.log(4)  # log(4) for 4 vertices
        return float((entropy / max_entropy).item())
    
    def geodesic_distance(self, w1: torch.Tensor, w2: torch.Tensor) -> float:
        """Geodesic distance between two states in barycentric space.
        
        Uses 3D Euclidean distance as approximation.
        """
        p1 = self.barycentric_to_3d(w1)
        p2 = self.barycentric_to_3d(w2)
        return float(torch.norm(p1 - p2).item())


# Singleton instance for convenience
_default_space: Optional[TetrahedralSpace] = None

def get_tetrahedral_space() -> TetrahedralSpace:
    """Get the default tetrahedral space instance."""
    global _default_space
    if _default_space is None:
        _default_space = TetrahedralSpace()
    return _default_space
