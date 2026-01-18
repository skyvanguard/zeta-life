"""
Resilience Components for Hierarchical Consciousness

Integrates IPUESA mechanisms (gradual damage, micro-modules, temporal anticipation)
into the consciousness system. Based on evolved parameters from 50-generation
optimization (fitness=0.9993, 8/8 self-evidence criteria).

Components:
- CellResilience: Resilience state for individual cells
- MicroModule: Emergent protective modules (8 types)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Degradation state thresholds
DEGRADATION_THRESHOLDS = {
    'OPTIMAL': 0.2,
    'STRESSED': 0.4,
    'IMPAIRED': 0.6,
    'CRITICAL': 0.8,
    'COLLAPSED': 1.0,
}

# Available module types and their roles
MODULE_TYPES = [
    'pattern_detector',       # Recognize threat patterns
    'threat_filter',          # Reduce incoming damage
    'recovery_accelerator',   # Speed up recovery
    'exploration_dampener',   # Reduce exploration under stress
    'embedding_protector',    # Preserve embedding integrity
    'cascade_breaker',        # Prevent damage cascades
    'residual_cleaner',       # Clear accumulated residual damage
    'anticipation_enhancer',  # Improve threat prediction
]

@dataclass
class MicroModule:
    """
    Emergent protective module.

    Modules are created under stress when vulnerability is high,
    consolidate through successful use, and can spread to neighboring
    cells within the same cluster.

    Attributes:
        module_type: One of 8 module types (see MODULE_TYPES)
        strength: Current strength [0, 1], affects effect magnitude
        activations: Number of times module has been activated
        contribution: Net contribution to survival (positive = helpful)
    """

    module_type: str
    strength: float = 0.5
    activations: int = 0
    contribution: float = 0.0

    def __post_init__(self):
        if self.module_type not in MODULE_TYPES:
            raise ValueError(f"Unknown module type: {self.module_type}. "
                           f"Must be one of {MODULE_TYPES}")

    def apply(self, effects_config: dict) -> float:
        """
        Apply module effect and return effect magnitude.

        Args:
            effects_config: Dict mapping module_type to base effect value

        Returns:
            Effect magnitude (base_effect * strength)
        """
        base_effect = effects_config.get(self.module_type, 0.1)
        self.activations += 1
        return base_effect * self.strength

    def reinforce(self, amount: float = 0.1):
        """Reinforce module strength after successful use."""
        self.strength = min(1.0, self.strength + amount)
        self.contribution += amount * self.strength

    def decay(self, rate: float = 0.95):
        """Decay strength from disuse."""
        self.strength *= rate

    def is_consolidated(self, min_activations: int = 3) -> bool:
        """
        Check if module is consolidated (eligible for spreading).

        A module is consolidated when it has been activated enough times
        and has made a positive contribution to survival.
        """
        return self.activations >= min_activations and self.contribution > 0

    def copy_weakened(self, strength_factor: float = 0.5) -> 'MicroModule':
        """Create a weakened copy for spreading to neighbors."""
        return MicroModule(
            module_type=self.module_type,
            strength=self.strength * strength_factor,
            activations=0,
            contribution=0.0,
        )

@dataclass
class CellResilience:
    """
    Resilience state for a single cell.

    Tracks degradation level, accumulated damage, protective modules,
    and temporal anticipation state. Integrates with ConsciousCell
    to modulate plasticity and functional status.

    Attributes:
        degradation_level: Current degradation [0, 1], 0=optimal, 1=collapsed
        residual_damage: Accumulated damage that persists across recovery
        modules: List of active MicroModules
        threat_buffer: EMA of recent damage (for temporal anticipation)
        anticipated_damage: Predicted future damage
        protective_stance: Proactive protection level
        embedding: Optional holographic embedding vector
        embedding_strength: Integrity of embedding [0, 1]
    """

    # Core degradation state
    degradation_level: float = 0.0
    residual_damage: float = 0.0

    # Micro-modules
    modules: list[MicroModule] = field(default_factory=list)

    # Temporal anticipation (TAE)
    threat_buffer: float = 0.0
    anticipated_damage: float = 0.0
    protective_stance: float = 0.0

    # Optional holographic embedding
    embedding: np.ndarray | None = None
    embedding_strength: float = 1.0

    @property
    def state(self) -> str:
        """
        Get degradation state category.

        States: OPTIMAL → STRESSED → IMPAIRED → CRITICAL → COLLAPSED
        """
        if self.degradation_level < DEGRADATION_THRESHOLDS['OPTIMAL']:
            return 'OPTIMAL'
        elif self.degradation_level < DEGRADATION_THRESHOLDS['STRESSED']:
            return 'STRESSED'
        elif self.degradation_level < DEGRADATION_THRESHOLDS['IMPAIRED']:
            return 'IMPAIRED'
        elif self.degradation_level < DEGRADATION_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        else:
            return 'COLLAPSED'

    @property
    def is_functional(self) -> bool:
        """Cell is functional if not collapsed (degradation < 0.8)."""
        return self.degradation_level < DEGRADATION_THRESHOLDS['CRITICAL']

    @property
    def vulnerability(self) -> float:
        """
        Current vulnerability level [0, 1].

        Higher when: high degradation, few modules, low embedding strength.
        """
        # Base vulnerability from degradation
        base = self.degradation_level

        # Module protection (more modules = lower vulnerability)
        module_protection = min(len(self.modules) * 0.1, 0.3)

        # Embedding protection
        embedding_protection = self.embedding_strength * 0.2 if self.embedding is not None else 0

        return max(0.0, min(1.0, base - module_protection - embedding_protection))

    def get_modules_by_type(self, module_type: str) -> list[MicroModule]:
        """Get all modules of a specific type."""
        return [m for m in self.modules if m.module_type == module_type]

    def has_module_type(self, module_type: str) -> bool:
        """Check if cell has at least one module of given type."""
        return any(m.module_type == module_type for m in self.modules)

    def add_module(self, module_type: str, strength: float = 0.5,
                   max_modules: int = 6) -> bool:
        """
        Add a new module if under capacity.

        Args:
            module_type: Type of module to create
            strength: Initial strength
            max_modules: Maximum modules per cell

        Returns:
            True if module was added, False if at capacity
        """
        if len(self.modules) >= max_modules:
            return False

        self.modules.append(MicroModule(
            module_type=module_type,
            strength=strength,
        ))
        return True

    def remove_weak_modules(self, threshold: float = 0.1):
        """Remove modules with strength below threshold."""
        self.modules = [m for m in self.modules if m.strength >= threshold]

    def decay_all_modules(self, rate: float = 0.95):
        """Apply decay to all modules."""
        for module in self.modules:
            module.decay(rate)

    def get_consolidated_modules(self, min_activations: int = 3) -> list[MicroModule]:
        """Get all consolidated modules (eligible for spreading)."""
        return [m for m in self.modules if m.is_consolidated(min_activations)]

    def update_anticipation(self, recent_damage: float, alpha: float = 0.3,
                           momentum: float = 1.2):
        """
        Update temporal anticipation state.

        Args:
            recent_damage: Damage received this step
            alpha: EMA decay factor for threat_buffer
            momentum: Factor for anticipation extrapolation
        """
        # Update threat buffer (EMA)
        self.threat_buffer = alpha * recent_damage + (1 - alpha) * self.threat_buffer

        # Anticipate future damage based on trend
        self.anticipated_damage = self.threat_buffer * momentum

        # Adjust protective stance based on anticipation
        if self.anticipated_damage > 0.3:
            self.protective_stance = min(1.0, self.protective_stance + 0.1)
        else:
            self.protective_stance = max(0.0, self.protective_stance - 0.05)

    def reset(self):
        """Reset resilience to initial state."""
        self.degradation_level = 0.0
        self.residual_damage = 0.0
        self.modules.clear()
        self.threat_buffer = 0.0
        self.anticipated_damage = 0.0
        self.protective_stance = 0.0
        self.embedding_strength = 1.0
