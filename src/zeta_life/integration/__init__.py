"""
Hierarchical Integration System.

Multi-level adaptive integration: Cells → Clusters → Organism.

Formal equations (Psi = B^3 + Phi) predict phase transitions from
fragmented processing to coherent integration. The critical threshold
Phi_c = F_i / (alpha - C) determines when the system transitions.

Components:
- MicroPsyche: Cell-level psyche with archetypes
- Cluster: Cluster aggregation and dynamics
- OrganismIntegration: Organism-level integration
- BottomUpIntegrator: Cell→Cluster→Organism flow
- TopDownModulator: Organism→Cluster→Cell influence
- formal_equations: Psi, Phi_c, B, corruption threshold

Resilience Components (IPUESA integration):
- CellResilience: Resilience state for individual cells
- MicroModule: Emergent protective modules (8 types)
- DamageSystem: Damage/recovery management
- resilience_config: Configuration mapping from evolved params
"""

from .bottom_up_integrator import BottomUpIntegrator
from .formal_equations import (
    compute_B,
    compute_corruption_threshold,
    compute_M_c,
    compute_phi_c,
    compute_psi,
    predict_system_stability,
)
from .cluster import Cluster, ClusterPsyche, compute_inter_cluster_coherence, find_cluster_neighbors
from .cluster_assigner import ClusterAssigner, ClusteringConfig, ClusteringStrategy
from .damage_system import DamageSystem
from .hierarchical_simulation import HierarchicalSimulation, SimulationConfig, SimulationMetrics
from .micro_psyche import (
    IntegrationCell,
    MicroPsyche,
    apply_psyche_contagion,
    compute_local_phi,
    unbiased_argmax,
)
from .organism_integration import (
    HierarchicalMetrics,
    OrganismIntegration,
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
