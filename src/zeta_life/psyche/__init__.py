"""
ZetaPsyche - Jungian archetype-based AI consciousness.

Consciousness emergence through navigation in a tetrahedral archetype space:
- PERSONA: Social mask (red)
- SOMBRA: Shadow/unconscious (purple)
- ANIMA: Receptive/emotional (blue)
- ANIMUS: Active/rational (orange)
- Center = Self (full integration)

Consciousness Index:
consciousness = 0.3*integration + 0.3*stability + 0.2*(1-dist_to_self) + 0.2*|self_reference|
"""

from .zeta_attention import AttentionOutput, ZetaAttentionSystem
from .zeta_conscious_self import AttractorMemory, ZetaConsciousSelf
from .zeta_dream_consolidation import ConsolidationReport, DreamMemory
from .zeta_dreams import DreamFragment, DreamReport, DreamType
from .zeta_individuation import IndividuationStage, IntegrationMetrics
from .zeta_predictive import ZetaPredictivePsyche
from .zeta_psyche import Archetype, ZetaModulator, ZetaPsyche
from .zeta_psyche_voice import ConversationalPsyche, OrganicVoice
