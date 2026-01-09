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

from .zeta_organism import ZetaOrganism, CellEntity
from .cell_state import CellState, CellRole
from .force_field import ForceField
from .behavior_engine import BehaviorEngine
from .organism_cell import OrganismCell
from .organism_cell_lstm import OrganismCellLSTM
from .zeta_organism_lstm import ZetaOrganismLSTM
