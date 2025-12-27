# cell_state.py
"""Estado de celula para ZetaOrganism."""
import torch
from enum import Enum
from dataclasses import dataclass, field
import math

class CellRole(Enum):
    MASS = 0      # Mi - sigue a Fi
    FORCE = 1     # Fi - atrae masas
    CORRUPT = 2   # Compite con Fi existente

@dataclass
class CellState:
    """Estado multidimensional de una celula."""
    alive: bool = True
    mass: float = 1.0
    energy: float = 0.0
    role: CellRole = CellRole.MASS
    fi_base: float = 1.0
    controlled_mass: float = 0.0
    memory: torch.Tensor = field(default_factory=lambda: torch.zeros(32))
    resonance_gate: float = 0.0

    def effective_fi(self) -> float:
        """Fi efectiva escalada por masa controlada."""
        if self.role != CellRole.FORCE:
            return 0.0
        return self.fi_base * math.sqrt(max(1.0, self.controlled_mass))

    def update_role(self, fi_threshold: float = 0.7,
                    min_followers: int = 3,
                    rival_detected: bool = False):
        """Actualiza rol segun condiciones."""
        if self.role == CellRole.MASS:
            # MASS -> FORCE: alta energia + seguidores
            if self.energy > fi_threshold and self.controlled_mass >= min_followers:
                self.role = CellRole.FORCE
        elif self.role == CellRole.FORCE:
            # FORCE -> CORRUPT: rival detectado
            if rival_detected:
                self.role = CellRole.CORRUPT
        elif self.role == CellRole.CORRUPT:
            # CORRUPT -> MASS: perdio competencia
            if self.energy < fi_threshold * 0.5:
                self.role = CellRole.MASS

    def net_role_value(self, influence_out: float, influence_in: float) -> float:
        """B = AA* - A*A: rol neto basado en balance de influencia."""
        return influence_out - influence_in
