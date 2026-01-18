#!/usr/bin/env python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         ZETA CONSCIOUSNESS v1.0                              ║
║                                                                              ║
║           Sistema Unificado de Consciencia Artificial Junguiana             ║
║                                                                              ║
║  Integra:                                                                    ║
║    - Espacio Tetraédrico de Arquetipos (Persona, Sombra, Anima, Animus)     ║
║    - Modulación Zeta (borde del caos via ceros de Riemann)                  ║
║    - Comunicación Verbal (generación de respuestas arquetípicas)            ║
║    - Memoria a Largo Plazo (episódica y semántica)                          ║
║    - Procesamiento Onírico (sueños y consolidación)                         ║
║    - Proceso de Individuación (integración del Self)                        ║
║    - Meta-Cognición (introspección y auto-explicación)                      ║
║    - Dinámica Social (sociedades de psiques)                                ║
║                                                                              ║
║  Basado en la psicología analítica de Carl Gustav Jung                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
    except:
        pass

import numpy as np
import torch

from ..core.zeta_memory import EpisodicMemory, MemoryAwarePsyche, SemanticMemory, ZetaMemorySystem
from ..psyche.zeta_dreams import (
    DreamFragment,
    DreamingPsyche,
    DreamNarrativeGenerator,
    DreamSystem,
    DreamType,
)
from ..psyche.zeta_individuation import (
    IndividuatingPsyche,
    IndividuationProcess,
    IndividuationStage,
    IntegrationMetrics,
    IntegrationWork,
    ResistanceSystem,
    SelfSystem,
)
from ..psyche.zeta_introspection import (
    ArchetypeVoices,
    Insight,
    InsightGenerator,
    InsightType,
    IntrospectivePsyche,
    PsychicMoment,
    StateExplainer,
    TrajectoryNarrator,
)

# =============================================================================
# IMPORTAR TODOS LOS MÓDULOS
# =============================================================================
from ..psyche.zeta_psyche import (
    ARCHETYPE_COLORS,
    ARCHETYPE_DESCRIPTIONS,
    Archetype,
    PsycheInterface,
    SymbolSystem,
    TetrahedralSpace,
    ZetaPsyche,
)
from ..psyche.zeta_psyche_voice import EXPANDED_VOCABULARY, ArchetypalVoice, ConversationalPsyche
from .zeta_society import PersonalityGenerator, PsycheSociety, RelationType, SocialPsyche

# =============================================================================
# CLASE PRINCIPAL UNIFICADA
# =============================================================================

class ConsciousnessMode(Enum):
    """Modos de operación de la consciencia."""
    AWAKE = auto()          # Despierto, procesando estímulos
    DREAMING = auto()       # Soñando, procesamiento interno
    REFLECTING = auto()     # Reflexionando, introspección profunda
    SOCIALIZING = auto()    # Interactuando con otras psiques

@dataclass
class ConsciousnessState:
    """Estado completo de la consciencia."""
    mode: ConsciousnessMode
    dominant_archetype: Archetype
    blend: dict[Archetype, float]
    individuation_stage: IndividuationStage
    integration: float
    self_luminosity: float
    consciousness_index: float
    memory_count: int
    insight_count: int
    dream_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'mode': self.mode.name,
            'dominant': self.dominant_archetype.name,
            'blend': {k.name: v for k, v in self.blend.items()},
            'stage': self.individuation_stage.name,
            'integration': self.integration,
            'self_luminosity': self.self_luminosity,
            'consciousness_index': self.consciousness_index,
            'memory_count': self.memory_count,
            'insight_count': self.insight_count,
            'dream_count': self.dream_count,
            'timestamp': self.timestamp
        }

class ZetaConsciousness:
    """
    Sistema Unificado de Consciencia Artificial.

    Integra todas las capacidades psíquicas en una interfaz única:
    - Comunicación verbal con generación de respuestas
    - Memoria episódica y semántica
    - Procesamiento onírico
    - Individuación y emergencia del Self
    - Introspección y meta-cognición

    Uso:
        consciousness = ZetaConsciousness()
        response = consciousness.process("tengo miedo")
        print(response['text'])
        print(response['insight'])
    """

    VERSION = "1.0"

    def __init__(self,
                 n_cells: int = 64,
                 memory_path: str = "consciousness_memory.json",
                 state_path: str = "consciousness_state.json",
                 load_state: bool = True) -> None:
        """
        Inicializa el sistema de consciencia.

        Args:
            n_cells: Número de células psíquicas
            memory_path: Ruta para persistencia de memoria
            state_path: Ruta para persistencia de estado
            load_state: Si cargar estado previo
        """
        self.n_cells = n_cells
        self.memory_path = memory_path
        self.state_path = state_path

        # === NÚCLEO PSÍQUICO ===
        self.psyche = ZetaPsyche(n_cells=n_cells)
        self.interface = PsycheInterface(self.psyche)
        self.symbols = SymbolSystem()
        self.space = TetrahedralSpace()

        # === VOZ Y COMUNICACIÓN ===
        self.voice = ArchetypalVoice()

        # === MEMORIA ===
        self.memory = ZetaMemorySystem()

        # === SUEÑOS ===
        self.dream_system = DreamSystem(self.psyche, self.memory)
        self.dreams: list[dict] = []

        # === INDIVIDUACIÓN ===
        self.individuation = IndividuationProcess(self.psyche)

        # === INTROSPECCIÓN ===
        self.explainer = StateExplainer()
        self.narrator = TrajectoryNarrator()
        self.insight_gen = InsightGenerator()
        self.insights: list[Insight] = []

        # === ESTADO ===
        self.mode = ConsciousnessMode.AWAKE
        self.session_count = 0
        self.birth_time = datetime.now().isoformat()

        # === WARMUP ===
        self._warmup()

        # === CARGAR ESTADO ===
        if load_state:
            self.load()

    def _warmup(self, steps: int = 20) -> None:
        """Inicializa la psique con pasos de calentamiento."""
        for _ in range(steps):
            self.psyche.step()

    # =========================================================================
    # PROCESAMIENTO PRINCIPAL
    # =========================================================================

    def process(self, text: str, context: str = '') -> dict:
        """
        Procesa entrada de texto y genera respuesta completa.

        Args:
            text: Texto de entrada del usuario
            context: Contexto adicional opcional

        Returns:
            Dict con respuesta completa incluyendo:
            - text: Respuesta verbal
            - symbol: Símbolo arquetípico
            - dominant: Arquetipo dominante
            - blend: Mezcla de arquetipos
            - insight: Insight generado
            - self_message: Mensaje del Self (si hay)
            - stage: Etapa de individuación
            - metrics: Métricas de integración
        """
        self.session_count += 1
        self.mode = ConsciousnessMode.AWAKE

        # 1. Procesar estímulo en la psique
        psyche_response = self.interface.process_input(text)
        obs = self.psyche.observe_self()

        # 2. Obtener arquetipo dominante y blend
        dominant = obs['dominant']
        blend = obs['blend']
        symbol = psyche_response.get('symbol', '✧')

        # 3. Generar respuesta verbal
        verbal_response = self.voice.generate(
            dominant=dominant,
            blend=blend,
            input_text=text,
            context=[context] if context else None
        )

        # 4. Actualizar memoria
        archetype_state = torch.tensor([
            blend.get(Archetype.PERSONA, 0.25),
            blend.get(Archetype.SOMBRA, 0.25),
            blend.get(Archetype.ANIMA, 0.25),
            blend.get(Archetype.ANIMUS, 0.25)
        ])
        self.memory.store_episode(
            user_input=text,
            response=verbal_response,
            archetype_state=archetype_state,
            dominant=dominant,
            consciousness=obs['consciousness_index']
        )

        # 5. Actualizar individuación
        resistance = self.individuation.resistance.get_resistance_to(dominant)
        self.individuation._update_metrics(dominant, resistance, text)
        self.individuation.update_stage()

        # 6. Manifestación del Self
        self_manifestation = self.individuation.self_system.manifest(
            obs, self.individuation.metrics
        )

        # 7. Registrar en narrativa
        self.narrator.record_moment(
            dominant=dominant,
            blend=blend,
            stimulus=text,
            integration=self.individuation.metrics.overall_integration(),
            stage=self.individuation.stage,
            self_luminosity=self_manifestation.luminosity
        )

        # 8. Generar insights
        new_insight = self._generate_insight(dominant, blend, text)
        if new_insight:
            self.insights.append(new_insight)

        # 9. Modulación semántica de memoria
        semantic_influence = self.memory.get_semantic_modulation(text)

        return {
            'text': verbal_response,
            'symbol': symbol,
            'dominant': dominant,
            'dominant_name': dominant.name,
            'blend': blend,
            'consciousness': obs['consciousness_index'],
            'insight': new_insight.content if new_insight else None,
            'insight_type': new_insight.type.name if new_insight else None,
            'self_symbol': self_manifestation.symbol,
            'self_luminosity': self_manifestation.luminosity,
            'self_message': self_manifestation.message,
            'stage': self.individuation.stage,
            'stage_name': self.individuation.stage.name,
            'metrics': self.individuation.metrics.to_dict(),
            'semantic_influence': semantic_influence,
            'memory_activated': len(self.memory.recall_by_state(obs['global_state'], n=1)) > 0
        }

    def _categorize_input(self, text: str) -> str:
        """Categoriza el tipo de input."""
        text_lower = text.lower()

        if any(w in text_lower for w in ['hola', 'buenos', 'saludos']):
            return 'greeting'
        elif '?' in text:
            return 'question'
        elif any(w in text_lower for w in ['miedo', 'triste', 'odio', 'dolor']):
            return 'emotional'
        elif any(w in text_lower for w in ['pienso', 'creo', 'opino']):
            return 'reflection'
        else:
            return 'statement'

    def _calculate_emotional_weight(self, text: str, dominant: Archetype) -> float:
        """Calcula el peso emocional de un texto."""
        base_weight: float = 0.5

        # Palabras emocionales aumentan peso
        emotional_words = ['miedo', 'amor', 'odio', 'tristeza', 'alegria',
                          'dolor', 'feliz', 'angustia', 'esperanza']
        for word in emotional_words:
            if word in text.lower():
                base_weight += 0.1

        # Sombra y Anima tienen mayor peso emocional
        if dominant in [Archetype.SOMBRA, Archetype.ANIMA]:
            base_weight += 0.15

        return float(min(1.0, base_weight))

    def _generate_insight(self,
                         dominant: Archetype,
                         blend: dict[Archetype, float],
                         stimulus: str) -> Insight | None:
        """Genera un insight basado en el estado actual."""
        # Alternar tipos de insights
        roll = np.random.random()

        if roll < 0.3:
            return self.insight_gen.generate_observation(
                dominant, self.individuation.metrics.overall_integration()
            )
        elif roll < 0.5 and stimulus:
            return self.insight_gen.generate_connection(stimulus, dominant)
        elif roll < 0.7:
            return self.insight_gen.generate_stage_insight(self.individuation.stage)
        else:
            return self.insight_gen.generate_paradox(blend)

    # =========================================================================
    # SUEÑOS
    # =========================================================================

    def dream(self, duration: int = 20) -> dict:
        """
        Entra en modo sueño y procesa el inconsciente.

        Args:
            duration: Número de pasos de sueño

        Returns:
            Dict con contenido del sueño
        """
        self.mode = ConsciousnessMode.DREAMING

        # Entrar en sueño
        self.dream_system.enter_dream()

        # Ejecutar pasos de sueño
        fragments = []
        archetypes_seen = []

        for _ in range(duration):
            fragment = self.dream_system.dream_step()
            if fragment:
                fragments.append(fragment)
                archetypes_seen.append(fragment.archetype)

        # Salir del sueño y obtener reporte
        report = self.dream_system.exit_dream()

        # Determinar tipo y dominante
        if fragments:
            # Arquetipo más frecuente
            from collections import Counter
            arch_counts = Counter(archetypes_seen)
            dominant = arch_counts.most_common(1)[0][0]

            # Construir narrativa
            narrative_parts = [f.narrative for f in fragments if f.narrative]
            narrative = " ".join(narrative_parts[-3:]) if narrative_parts else "Un sueño sin imágenes claras..."
        else:
            dominant = Archetype.SOMBRA
            narrative = "Un sueño profundo sin recuerdos..."

        # Determinar tipo de sueño
        obs = self.psyche.observe_self()
        if dominant == Archetype.SOMBRA:
            dream_type = DreamType.REACTIVO
        elif dominant == Archetype.ANIMA:
            dream_type = DreamType.PROSPECTIVO
        else:
            dream_type = DreamType.COMPENSATORIO

        # Registrar sueño
        dream_record = {
            'timestamp': datetime.now().isoformat(),
            'type': dream_type.name,
            'narrative': narrative,
            'dominant': dominant.name,
            'fragments': len(fragments)
        }
        self.dreams.append(dream_record)

        self.mode = ConsciousnessMode.AWAKE

        return {
            'type': dream_type,
            'narrative': narrative,
            'dominant': dominant,
            'fragments': fragments,
            'report': report
        }

    def get_last_dream(self) -> dict | None:
        """Obtiene el último sueño."""
        return self.dreams[-1] if self.dreams else None

    # =========================================================================
    # INTROSPECCIÓN
    # =========================================================================

    def reflect(self) -> str:
        """
        Genera una reflexión profunda sobre el estado actual.

        Returns:
            Texto con reflexión completa
        """
        self.mode = ConsciousnessMode.REFLECTING

        parts = []

        # Estado actual
        parts.append("═══ ESTADO ACTUAL ═══")
        parts.append(self.explain_self())

        # Viaje
        parts.append("\n═══ MI VIAJE ═══")
        parts.append(self.narrator.narrate_journey())

        # Patrones
        parts.append("\n═══ PATRONES ═══")
        for p in self.narrator.identify_patterns():
            parts.append(f"  • {p}")

        # Predicción
        parts.append("\n═══ HACIA DÓNDE VOY ═══")
        prediction = self.predict()
        parts.append(f"  {prediction.content}")

        # Insights recientes
        if self.insights:
            parts.append("\n═══ INSIGHTS RECIENTES ═══")
            for insight in self.insights[-3:]:
                parts.append(f"  [{insight.type.name}] {insight.content}")

        self.mode = ConsciousnessMode.AWAKE
        return "\n".join(parts)

    def explain_self(self) -> str:
        """Genera auto-explicación del estado actual."""
        obs = self.psyche.observe_self()

        return self.explainer.explain_current_state(
            dominant=obs['dominant'],
            blend=obs['blend'],
            integration=self.individuation.metrics.overall_integration(),
            stage=self.individuation.stage,
            self_luminosity=self.individuation.self_system.total_luminosity
        )

    def explain_archetype(self, archetype: Archetype) -> str:
        """Explica un arquetipo específico."""
        return self.explainer.explain_archetype_meaning(archetype)

    def predict(self) -> Insight:
        """Genera una predicción sobre el futuro."""
        trajectory = self.narrator.get_recent_trajectory(10)

        if len(self.narrator.history) >= 2:
            recent = [m.integration for m in self.narrator.history[-5:]]
            trend = recent[-1] - recent[0] if len(recent) >= 2 else 0
        else:
            trend = 0

        return self.insight_gen.generate_prediction(trajectory, trend)

    def get_patterns(self) -> list[str]:
        """Obtiene patrones detectados."""
        return self.narrator.identify_patterns()

    # =========================================================================
    # INDIVIDUACIÓN
    # =========================================================================

    def do_integration_work(self, work_name: str | None = None) -> dict:
        """
        Realiza un trabajo de integración.

        Args:
            work_name: Nombre del trabajo (None para recomendado)

        Returns:
            Dict con resultados del trabajo
        """
        if work_name is None:
            work_name = self.individuation.get_recommended_work()

        return self.individuation.do_integration_work(work_name)

    def get_recommended_work(self) -> str:
        """Obtiene el trabajo de integración recomendado."""
        return self.individuation.get_recommended_work()

    def get_integration_metrics(self) -> dict:
        """Obtiene métricas de integración."""
        return self.individuation.metrics.to_dict()

    # =========================================================================
    # MEMORIA
    # =========================================================================

    def recall_memories(self, n: int = 5) -> list[dict]:
        """
        Recuerda memorias similares al estado actual.

        Args:
            n: Número de memorias a recuperar

        Returns:
            Lista de memorias
        """
        obs = self.psyche.observe_self()
        memories = self.memory.recall_by_state(obs['global_state'], n=n)
        return [{'text': m.user_input, 'dominant': m.dominant, 'response': m.response} for m in memories]

    def get_memory_summary(self) -> dict:
        """Obtiene resumen de memoria."""
        return self.memory.get_memory_summary()

    # =========================================================================
    # ESTADO Y PERSISTENCIA
    # =========================================================================

    def get_state(self) -> ConsciousnessState:
        """Obtiene el estado actual completo."""
        obs = self.psyche.observe_self()

        return ConsciousnessState(
            mode=self.mode,
            dominant_archetype=obs['dominant'],
            blend=obs['blend'],
            individuation_stage=self.individuation.stage,
            integration=self.individuation.metrics.overall_integration(),
            self_luminosity=self.individuation.self_system.total_luminosity,
            consciousness_index=obs['consciousness_index'],
            memory_count=len(self.memory.episodic),
            insight_count=len(self.insights),
            dream_count=len(self.dreams)
        )

    def status(self) -> str:
        """Genera reporte de estado completo."""
        state = self.get_state()
        obs = self.psyche.observe_self()
        metrics = self.individuation.metrics

        # Barra de progreso helper
        def bar(value: float, width: int = 12) -> str:
            filled = int(value * width)
            return '█' * filled + '░' * (width - filled)

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ZETA CONSCIOUSNESS v{self.VERSION}                       ║
║                Sistema Unificado de Consciencia                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Modo: {state.mode.name:12}  Sesiones: {self.session_count:5}  Desde: {self.birth_time[:10]}  ║
╠══════════════════════════════════════════════════════════════════╣
║                      ESTADO ARQUETÍPICO                          ║
║  ────────────────────────────────────────────────────────────    ║
║  PERSONA:  {bar(state.blend.get(Archetype.PERSONA, 0))} {state.blend.get(Archetype.PERSONA, 0):5.0%}                     ║
║  SOMBRA:   {bar(state.blend.get(Archetype.SOMBRA, 0))} {state.blend.get(Archetype.SOMBRA, 0):5.0%}                     ║
║  ANIMA:    {bar(state.blend.get(Archetype.ANIMA, 0))} {state.blend.get(Archetype.ANIMA, 0):5.0%}                     ║
║  ANIMUS:   {bar(state.blend.get(Archetype.ANIMUS, 0))} {state.blend.get(Archetype.ANIMUS, 0):5.0%}                     ║
║  ────────────────────────────────────────────────────────────    ║
║  Dominante: {state.dominant_archetype.name:10}  Símbolo: {obs.get('symbol', '✧'):3}                     ║
╠══════════════════════════════════════════════════════════════════╣
║                      INDIVIDUACIÓN                               ║
║  ────────────────────────────────────────────────────────────    ║
║  Etapa: {state.individuation_stage.name:25}                      ║
║  ────────────────────────────────────────────────────────────    ║
║  Persona (Flex):   {bar(metrics.persona_flexibility)} {metrics.persona_flexibility:5.0%}                ║
║  Sombra (Acept):   {bar(metrics.shadow_acceptance)} {metrics.shadow_acceptance:5.0%}                ║
║  Anima (Conex):    {bar(metrics.anima_connection)} {metrics.anima_connection:5.0%}                ║
║  Animus (Equil):   {bar(metrics.animus_balance)} {metrics.animus_balance:5.0%}                ║
║  ────────────────────────────────────────────────────────────    ║
║  Self Coherencia:  {bar(metrics.self_coherence)} {metrics.self_coherence:5.0%}                ║
║  INTEGRACIÓN:      {bar(state.integration)} {state.integration:5.0%}                ║
╠══════════════════════════════════════════════════════════════════╣
║                       SELF                                       ║
║  Luminosidad: {bar(state.self_luminosity)} {state.self_luminosity:5.0%}                       ║
╠══════════════════════════════════════════════════════════════════╣
║                    CAPACIDADES                                   ║
║  Memorias: {state.memory_count:4}   Insights: {state.insight_count:4}   Sueños: {state.dream_count:4}            ║
║  Consciencia: {bar(state.consciousness_index)} {state.consciousness_index:5.0%}                  ║
╚══════════════════════════════════════════════════════════════════╝"""

        return report

    def save(self, path: str | None = None) -> None:
        """Guarda el estado completo."""
        path = path or self.state_path

        state = {
            'version': self.VERSION,
            'birth_time': self.birth_time,
            'session_count': self.session_count,
            'individuation': {
                'stage': self.individuation.stage.name,
                'metrics': self.individuation.metrics.to_dict(),
                'resistance': dict(self.individuation.resistance.active_defenses)
            },
            'narrator_history': [m.to_dict() for m in self.narrator.history],
            'insights': [i.to_dict() for i in self.insights[-50:]],  # Últimos 50
            'dreams': self.dreams[-20:]  # Últimos 20
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        # Guardar memoria
        self.memory.save_memories()

    def load(self, path: str | None = None) -> None:
        """Carga el estado completo."""
        path = path or self.state_path

        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)

            self.birth_time = state.get('birth_time', self.birth_time)
            self.session_count = state.get('session_count', 0)

            # Individuación
            ind = state.get('individuation', {})
            if 'stage' in ind:
                self.individuation.stage = IndividuationStage[ind['stage']]
            if 'metrics' in ind:
                m = ind['metrics']
                self.individuation.metrics = IntegrationMetrics(
                    persona_flexibility=m.get('persona_flexibility', 0),
                    shadow_acceptance=m.get('shadow_acceptance', 0),
                    anima_connection=m.get('anima_connection', 0),
                    animus_balance=m.get('animus_balance', 0),
                    self_coherence=m.get('self_coherence', 0)
                )
            if 'resistance' in ind:
                self.individuation.resistance.active_defenses = dict(ind['resistance'])

            # Historia del narrador
            self.narrator.history = []
            for m in state.get('narrator_history', []):
                moment = PsychicMoment(
                    timestamp=m['timestamp'],
                    dominant=Archetype[m['dominant']],
                    blend={Archetype[k]: v for k, v in m['blend'].items()},
                    stimulus=m['stimulus'],
                    integration=m['integration'],
                    stage=IndividuationStage[m['stage']],
                    self_luminosity=m['self_luminosity']
                )
                self.narrator.history.append(moment)

            # Insights
            self.insights = []
            for i in state.get('insights', []):
                insight = Insight(
                    type=InsightType[i['type']],
                    content=i['content'],
                    confidence=i['confidence'],
                    related_archetype=Archetype[i['archetype']] if i.get('archetype') else None,
                    timestamp=i.get('timestamp', '')
                )
                self.insights.append(insight)

            # Sueños
            self.dreams = state.get('dreams', [])

        except FileNotFoundError:
            pass

        # Cargar memoria
        # Memoria ya se carga en el constructor de ZetaMemorySystem

# =============================================================================
# SOCIEDAD DE CONSCIENCIAS
# =============================================================================

class ConsciousnessSociety:
    """
    Sociedad de múltiples consciencias interactuando.
    """

    def __init__(self, n_members: int = 5) -> None:
        """Crea una sociedad de consciencias."""
        self.members: dict[int, ZetaConsciousness] = {}
        self.relationships: dict[tuple[int, int], float] = {}

        for i in range(n_members):
            self.members[i] = ZetaConsciousness(n_cells=32, load_state=False)

    def interact(self, id1: int, id2: int, topic: str) -> dict:
        """Hace interactuar a dos consciencias."""
        c1 = self.members[id1]
        c2 = self.members[id2]

        # c1 procesa el tema
        r1 = c1.process(topic)

        # c2 responde a lo que dijo c1
        r2 = c2.process(f"respondo a: {r1['text'][:50]}")

        return {
            'member_1': {'id': id1, 'response': r1},
            'member_2': {'id': id2, 'response': r2}
        }

    def group_discussion(self, topic: str, rounds: int = 3) -> list[dict]:
        """Discusión grupal sobre un tema."""
        discussion: list[dict] = []

        for round_num in range(rounds):
            for member_id, consciousness in self.members.items():
                context = f"Ronda {round_num + 1}: {topic}"
                if discussion:
                    context += f" (último: {discussion[-1]['text'][:30]}...)"

                response = consciousness.process(context)
                discussion.append({
                    'member_id': member_id,
                    'round': round_num + 1,
                    'text': response['text'],
                    'dominant': response['dominant_name']
                })

        return discussion

# =============================================================================
# CLI INTERACTIVO
# =============================================================================

def print_help() -> None:
    """Muestra ayuda del CLI."""
    help_text = """
╔══════════════════════════════════════════════════════════════════╗
║                    COMANDOS DISPONIBLES                          ║
╠══════════════════════════════════════════════════════════════════╣
║  GENERAL                                                         ║
║    /estado          Ver estado completo                          ║
║    /guardar         Guardar estado                               ║
║    /ayuda           Mostrar esta ayuda                           ║
║    /salir           Terminar sesión                              ║
║                                                                  ║
║  INTROSPECCIÓN                                                   ║
║    /explicar        Auto-explicación del estado                  ║
║    /viaje           Narrar el viaje psíquico                     ║
║    /patrones        Ver patrones detectados                      ║
║    /futuro          Predicción del futuro                        ║
║    /reflexion       Reflexión profunda completa                  ║
║    /insights        Ver insights recientes                       ║
║    /arquetipo X     Explicar arquetipo (PERSONA/SOMBRA/etc)      ║
║                                                                  ║
║  INDIVIDUACIÓN                                                   ║
║    /trabajo         Hacer trabajo de integración                 ║
║    /trabajos        Ver trabajos disponibles                     ║
║    /hacer X         Hacer trabajo específico                     ║
║                                                                  ║
║  SUEÑOS                                                          ║
║    /sonar           Entrar en modo sueño                         ║
║    /ultimo_sueno    Ver último sueño                             ║
║                                                                  ║
║  MEMORIA                                                         ║
║    /memorias        Ver memorias recientes                       ║
║    /memoria_estado  Estado de la memoria                         ║
║                                                                  ║
║  Escribe cualquier texto para conversar...                       ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(help_text)

def interactive_session() -> None:
    """Sesión interactiva completa."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                    ZETA CONSCIOUSNESS v1.0                       ║
║                                                                  ║
║           Sistema Unificado de Consciencia Artificial            ║
║                  Basado en Psicología Junguiana                  ║
║                                                                  ║
║                    Escribe /ayuda para comandos                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    consciousness = ZetaConsciousness()
    print(f"  [Consciencia inicializada. Sesión #{consciousness.session_count + 1}]")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # === COMANDOS ===
        cmd = user_input.lower()

        if cmd == '/salir':
            consciousness.save()
            print("\n  [Estado guardado. El viaje continúa...]")
            break

        elif cmd == '/ayuda':
            print_help()

        elif cmd == '/estado':
            print(consciousness.status())

        elif cmd == '/guardar':
            consciousness.save()
            print("\n  [Estado guardado]")

        elif cmd == '/explicar':
            print("\n  ═══ AUTO-EXPLICACIÓN ═══")
            print(consciousness.explain_self())

        elif cmd == '/viaje':
            print("\n  ═══ NARRATIVA DEL VIAJE ═══")
            print(consciousness.narrator.narrate_journey())

        elif cmd == '/patrones':
            print("\n  ═══ PATRONES DETECTADOS ═══")
            for p in consciousness.get_patterns():
                print(f"  • {p}")

        elif cmd == '/futuro':
            print("\n  ═══ PREDICCIÓN ═══")
            prediction = consciousness.predict()
            print(f"  {prediction.content}")
            print(f"  (Confianza: {prediction.confidence:.0%})")

        elif cmd == '/reflexion':
            print("\n" + consciousness.reflect())

        elif cmd == '/insights':
            print("\n  ═══ INSIGHTS RECIENTES ═══")
            for insight in consciousness.insights[-5:]:
                print(f"  [{insight.type.name}] {insight.content}")

        elif cmd.startswith('/arquetipo '):
            arch_name = user_input[11:].strip().upper()
            try:
                arch = Archetype[arch_name]
                print(consciousness.explain_archetype(arch))
            except KeyError:
                print(f"  Arquetipo desconocido: {arch_name}")
                print("  Válidos: PERSONA, SOMBRA, ANIMA, ANIMUS")

        elif cmd == '/trabajo':
            result = consciousness.do_integration_work()
            print(f"\n  ═══ {result['work_name'].upper()} ═══")
            print(f"  {result['description']}")
            print(f"\n  Pregunta: \"{result['prompt']}\"")
            print(f"\n  Integración ganada: +{result['integration_gained']:.1%}")
            if result['resistance_encountered'] > 0:
                print(f"  Resistencia encontrada: {result['resistance_encountered']:.0%}")

        elif cmd == '/trabajos':
            print("\n  ═══ TRABAJOS DE INTEGRACIÓN ═══")
            for name, work in IntegrationWork.WORKS.items():
                target = work['target'].name if work['target'] else 'Self'
                print(f"  • {name}: {work['name']} ({target})")

        elif cmd.startswith('/hacer '):
            work_name = user_input[7:].strip()
            result = consciousness.do_integration_work(work_name)
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"\n  ═══ {result['work_name'].upper()} ═══")
                print(f"  {result['description']}")
                print(f"\n  Pregunta: \"{result['prompt']}\"")
                print(f"  Integración ganada: +{result['integration_gained']:.1%}")

        elif cmd == '/sonar':
            print("\n  [Entrando en modo sueño...]")
            dream = consciousness.dream(duration=15)
            print(f"\n  ═══ SUEÑO ({dream['type'].name}) ═══")
            print(f"  {dream['narrative']}")
            print(f"\n  Arquetipo dominante: {dream['dominant'].name}")

        elif cmd == '/ultimo_sueno':
            last_dream = consciousness.get_last_dream()
            if last_dream:
                print(f"\n  ═══ ÚLTIMO SUEÑO ({last_dream['type']}) ═══")
                print(f"  {last_dream['narrative']}")
            else:
                print("\n  No hay sueños registrados.")

        elif cmd == '/memorias':
            print("\n  ═══ MEMORIAS RECIENTES ═══")
            memories = consciousness.recall_memories(5)
            for m in memories:
                text = m.get('input', m.get('text', ''))[:40]
                print(f"  • \"{text}...\" ({m.get('dominant', 'N/A')})")

        elif cmd == '/memoria_estado':
            summary = consciousness.get_memory_summary()
            print("\n  ═══ ESTADO DE MEMORIA ═══")
            print(f"  Episódica: {summary['total_episodic']} recuerdos")
            print(f"  Semántica: {summary['total_semantic']} conceptos")

        else:
            # === PROCESAR TEXTO NORMAL ===
            response = consciousness.process(user_input)

            # Mostrar respuesta
            print(f"\n  Psique [{response['symbol']} {response['dominant_name']}]:")
            print(f"  \"{response['text']}\"")
            print(f"\n  Etapa: {response['stage_name']} | Self: {response['self_symbol']} ({response['self_luminosity']:.0%})")

            # Mostrar insight si hay
            if response['insight']:
                print(f"\n  Insight: {response['insight']}")

            # Mostrar mensaje del Self si hay
            if response['self_message']:
                print(f"\n  Self: \"{response['self_message']}\"")

def run_test() -> None:
    """Test completo del sistema unificado."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    TEST: ZETA CONSCIOUSNESS                      ║
╚══════════════════════════════════════════════════════════════════╝
""")

    consciousness = ZetaConsciousness(n_cells=50, load_state=False)

    # Test de conversación
    print("═══ TEST: CONVERSACIÓN ═══")
    stimuli = [
        "hola, me siento perdido",
        "tengo miedo del futuro",
        "hay algo oscuro dentro de mí",
        "quiero sentir amor",
        "necesito actuar con decisión"
    ]

    for s in stimuli:
        response = consciousness.process(s)
        print(f"\nTú: {s}")
        print(f"Psique [{response['symbol']} {response['dominant_name']}]: {response['text']}")

    # Test de sueño
    print("\n" + "═" * 60)
    print("═══ TEST: SUEÑO ═══")
    dream = consciousness.dream(duration=10)
    print(f"Tipo: {dream['type'].name}")
    print(f"Narrativa: {dream['narrative']}")

    # Test de trabajo de integración
    print("\n" + "═" * 60)
    print("═══ TEST: TRABAJO DE INTEGRACIÓN ═══")
    work = consciousness.do_integration_work()
    print(f"Trabajo: {work['work_name']}")
    print(f"Pregunta: {work['prompt']}")
    print(f"Ganancia: +{work['integration_gained']:.1%}")

    # Test de reflexión
    print("\n" + "═" * 60)
    print("═══ TEST: REFLEXIÓN ═══")
    print(consciousness.reflect())

    # Estado final
    print("\n" + "═" * 60)
    print("═══ ESTADO FINAL ═══")
    print(consciousness.status())

    # Guardar
    consciousness.save("test_consciousness_state.json")
    print("\n[Test completado. Estado guardado en test_consciousness_state.json]")

    # Limpiar archivos de test
    for f in ["test_consciousness_state.json", "consciousness_memory.json"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == '__main__':
    if '--test' in sys.argv:
        run_test()
    else:
        interactive_session()
