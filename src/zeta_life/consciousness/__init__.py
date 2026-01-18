"""
Hierarchical Consciousness System.

Multi-level consciousness architecture: Cells → Clusters → Organism.

Components:
- MicroPsyche: Cell-level psyche with archetypes
- Cluster: Cluster aggregation and dynamics
- OrganismConsciousness: Organism-level integration
- BottomUpIntegrator: Cell→Cluster→Organism flow
- TopDownModulator: Organism→Cluster→Cell influence

Resilience Components (IPUESA integration):
- CellResilience: Resilience state for individual cells
- MicroModule: Emergent protective modules (8 types)
- DamageSystem: Damage/recovery management
- resilience_config: Configuration mapping from evolved params
"""

from .bottom_up_integrator import BottomUpIntegrator
from .cluster import Cluster, ClusterPsyche, compute_inter_cluster_coherence, find_cluster_neighbors
from .cluster_assigner import ClusterAssigner, ClusteringConfig, ClusteringStrategy
from .damage_system import DamageSystem
from .hierarchical_simulation import HierarchicalSimulation, SimulationConfig, SimulationMetrics
from .micro_psyche import (
    ConsciousCell,
    MicroPsyche,
    apply_psyche_contagion,
    compute_local_phi,
    unbiased_argmax,
)
from .organism_consciousness import (
    HierarchicalMetrics,
    OrganismConsciousness,
    _integration_to_stage,
)

# Resilience components (IPUESA integration)
from .resilience import DEGRADATION_THRESHOLDS, MODULE_TYPES, CellResilience, MicroModule
from .resilience_config import (
    PRESETS,
    create_hierarchical_config,
    get_preset_config,
    get_preset_info,
    list_presets,
)
from .top_down_modulator import TopDownModulator
from .zeta_consciousness import ConsciousnessState, ZetaConsciousness
