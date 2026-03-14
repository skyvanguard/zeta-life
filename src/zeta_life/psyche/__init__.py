"""
ZetaPsyche — Legacy Jungian archetype-based integration system.

NOTE: This is the original psyche layer. The active computational core
is in ``zeta_life.kernel`` (Active Inference architecture). This module
is maintained for backward compatibility and experimental exploration
of the tetrahedral archetype space.

Archetype space:
- PERSONA: Social mask (red)
- SOMBRA: Shadow/unconscious (purple)
- ANIMA: Receptive/emotional (blue)
- ANIMUS: Active/rational (orange)
- Center = Self (full integration)

Active modules: zeta_psyche, zeta_dreams, zeta_individuation,
                zeta_attention, zeta_conscious_self, zeta_psyche_voice
"""

from .zeta_attention import AttentionOutput, ZetaAttentionSystem
from .zeta_conscious_self import AttractorMemory, ZetaConsciousSelf
from .zeta_dream_consolidation import ConsolidationReport, DreamMemory
from .zeta_dreams import DreamFragment, DreamReport, DreamType
from .zeta_individuation import IndividuationStage, IntegrationMetrics
from .zeta_predictive import ZetaPredictivePsyche
from .zeta_psyche import Archetype, ZetaModulator, ZetaPsyche
from .zeta_psyche_voice import ConversationalPsyche, OrganicVoice
