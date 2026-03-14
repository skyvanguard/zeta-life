"""
ZetaOrganism - Multi-agent emergent intelligence system.

Simulates organisms where intelligence emerges from cell interactions
following Fi-Mi (Force-Mass) dynamics.

Demonstrated Emergent Properties (11+):
- Homeostasis, Regeneration, Antifragility
- Quimiotaxis, Spatial memory, Auto-segregation
- Competitive exclusion, Niche partition
- Collective panic, Coordinated escape, Collective foraging
"""

from .behavior_engine import BehaviorEngine
from .cell_state import CellRole, CellState
from .force_field import ForceField
from .organism_cell import OrganismCell
from .zeta_organism import CellEntity, ZetaOrganism
