"""Narrative mapper for visualization.

This module provides optional mappings from abstract vertices to
human-readable names. It is NEVER used in calculations.
"""

import json
import os
from typing import Any, Dict, Optional

from ..core.vertex import Vertex


class NarrativeMapper:
    """Optional layer for visualization/demo. Never used in calculations.
    
    This class maps abstract vertices (V0-V3) to display names, colors,
    and symbols. It supports multiple narrative layers:
    - 'raw': V0, V1, V2, V3 (no mapping)
    - 'functional': LEADER, DISRUPTOR, FOLLOWER, EXPLORER
    - 'narrative': PERSONA, SOMBRA, ANIMA, ANIMUS (Jungian)
    
    Usage:
        mapper = NarrativeMapper.from_preset('jungian')
        display_name = mapper.get_name(Vertex.V0, layer='narrative')
        color = mapper.get_color(Vertex.V0)
    """

    # Default configs directory
    _CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'configs')

    def __init__(self, config_path: str | None = None):
        """Initialize with optional config file.
        
        Args:
            config_path: Path to JSON config file. If None, uses raw vertex names.
        """
        self.mapping: dict[str, Any] = {}
        self.name = 'neutral'

        if config_path:
            self._load(config_path)

    def _load(self, path: str) -> None:
        """Load configuration from JSON file."""
        with open(path, encoding='utf-8') as f:
            self.mapping = json.load(f)
        self.name = self.mapping.get('name', 'custom')

    @classmethod
    def from_preset(cls, preset: str) -> 'NarrativeMapper':
        """Load a preset narrative configuration.
        
        Args:
            preset: One of 'jungian', 'functional', 'neutral'
            
        Returns:
            NarrativeMapper instance
        """
        config_path = os.path.join(cls._CONFIGS_DIR, f'{preset}.json')
        if not os.path.exists(config_path):
            # Return empty mapper if preset not found
            return cls()
        return cls(config_path)

    @classmethod
    def jungian(cls) -> 'NarrativeMapper':
        """Shortcut for Jungian narrative."""
        return cls.from_preset('jungian')

    @classmethod
    def functional(cls) -> 'NarrativeMapper':
        """Shortcut for functional narrative."""
        return cls.from_preset('functional')

    def get_name(self, vertex: Vertex, layer: str = 'functional') -> str:
        """Get display name for a vertex.
        
        Args:
            vertex: The vertex to get name for
            layer: 'raw' | 'functional' | 'narrative'
            
        Returns:
            Display name string
        """
        if not self.mapping or layer == 'raw':
            return vertex.name  # V0, V1, V2, V3

        vertex_data = self.mapping.get('vertices', {}).get(vertex.name, {})
        return vertex_data.get(layer, vertex.name)

    def get_color(self, vertex: Vertex) -> str:
        """Get display color for a vertex."""
        if not self.mapping:
            return '#888888'
        return self.mapping.get('vertices', {}).get(vertex.name, {}).get('color', '#888888')

    def get_symbol(self, vertex: Vertex) -> str:
        """Get display symbol for a vertex."""
        if not self.mapping:
            return '●'
        return self.mapping.get('vertices', {}).get(vertex.name, {}).get('symbol', '●')

    def get_description(self, vertex: Vertex) -> str:
        """Get description for a vertex."""
        if not self.mapping:
            return f'Vertex {vertex.name}'
        return self.mapping.get('vertices', {}).get(vertex.name, {}).get('description', '')

    def get_center_name(self, layer: str = 'narrative') -> str:
        """Get display name for the center (integrated state)."""
        if not self.mapping:
            return 'CENTER'
        center = self.mapping.get('center', {})
        return center.get(layer, center.get('narrative', 'CENTER'))

    def get_center_color(self) -> str:
        """Get display color for the center."""
        if not self.mapping:
            return '#D69E2E'
        return self.mapping.get('center', {}).get('color', '#D69E2E')

    def all_colors(self) -> dict[Vertex, str]:
        """Get all vertex colors as a dictionary."""
        return {v: self.get_color(v) for v in Vertex}

    def all_names(self, layer: str = 'functional') -> dict[Vertex, str]:
        """Get all vertex names as a dictionary."""
        return {v: self.get_name(v, layer) for v in Vertex}
