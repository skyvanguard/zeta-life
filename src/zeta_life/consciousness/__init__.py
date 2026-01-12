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

from .zeta_consciousness import ZetaConsciousness, ConsciousnessState
from .micro_psyche import MicroPsyche, ConsciousCell, compute_local_phi, unbiased_argmax, apply_psyche_contagion
from .cluster import Cluster, ClusterPsyche, find_cluster_neighbors, compute_inter_cluster_coherence
from .organism_consciousness import OrganismConsciousness, HierarchicalMetrics, _integration_to_stage
from .hierarchical_simulation import HierarchicalSimulation, SimulationConfig, SimulationMetrics
from .bottom_up_integrator import BottomUpIntegrator
from .top_down_modulator import TopDownModulator
from .cluster_assigner import ClusterAssigner, ClusteringConfig, ClusteringStrategy

# Resilience components (IPUESA integration)
from .resilience import CellResilience, MicroModule, MODULE_TYPES, DEGRADATION_THRESHOLDS
from .damage_system import DamageSystem
from .resilience_config import (
    create_hierarchical_config,
    get_preset_config,
    list_presets,
    get_preset_info,
    PRESETS,
)
