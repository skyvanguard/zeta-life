"""
Conscious Kernel — Active Inference architecture for Zeta Life.

Integrates world modeling, self-modeling, multi-channel prediction errors,
complementary memory systems, dream consolidation, and identity persistence
into a unified consciousness architecture.

Usage:
    from zeta_life.kernel import ConsciousKernel

    ck = ConsciousKernel()
    result = ck.step(stimulus)
    ck.save('~/.zeta_life/', identity_name='my_identity')
    ck.load('~/.zeta_life/', identity_name='my_identity')

Based on:
- "A Beautiful Loop" (Laukkonen, Friston & Chandaria, 2025)
- Complementary Learning Systems (McClelland et al.)
- Riemann zeta zeros as temporal binding mechanism
"""

from .complementary_memory import CompressedEpisode, Episode, FastMemory, SlowMemory
from .conscious_kernel import ConsciousKernel, StepResult
from .dream_engine import DreamEngine, DreamReport
from .persistence import PersistenceLayer
from .precision_controller import PrecisionController
from .prediction_error import PredictionErrorEngine
from .self_model import SelfModel
from .world_model import WorldModel
from .global_workspace import GlobalWorkspace, Proposal
from .energy_pool import EnergyPool
from .spawn_controller import SpawnController, SpawnEvent, MergeEvent, DeathEvent
from .organism_state import OrganismState
from .conscious_organism import ConsciousOrganism, OrganismStepResult

__all__ = [
    'ConsciousKernel',
    'StepResult',
    'WorldModel',
    'SelfModel',
    'PredictionErrorEngine',
    'PrecisionController',
    'FastMemory',
    'SlowMemory',
    'Episode',
    'CompressedEpisode',
    'DreamEngine',
    'DreamReport',
    'PersistenceLayer',
    'GlobalWorkspace',
    'Proposal',
    'EnergyPool',
    'SpawnController',
    'SpawnEvent',
    'MergeEvent',
    'DeathEvent',
    'OrganismState',
    'ConsciousOrganism',
    'OrganismStepResult',
]
