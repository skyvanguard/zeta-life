"""
ZetaConsciousSelf: Sistema Integrado de Consciencia e Individuacion
=====================================================================

Integracion completa de todos los sistemas:
- ZetaPsyche: Base arquetipal (Persona, Sombra, Anima, Animus)
- ZetaPredictive: Prediccion jerarquica (L1, L2, L3)
- ZetaAttention: Atencion selectiva (3 niveles)
- OnlineLearning: Aprendizaje continuo (Hebbian + Gradient)
- DreamConsolidation: Consolidacion mediante suenos
- Individuation: Proceso de desarrollo hacia el Self

El Self emerge cuando todos los sistemas trabajan en armonia:
- Buena prediccion = anticipar el mundo interno y externo
- Buena atencion = focalizar recursos apropiadamente
- Buena integracion = equilibrio entre arquetipos
- Buena consolidacion = aprender de la experiencia

Fecha: 3 Enero 2026
"""
import os
import sys

if sys.platform == 'win32':
    os.system('')

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .zeta_attention import AttentionOutput, ZetaAttentionSystem
from .zeta_attentive_predictive import ZetaAttentivePredictive
from .zeta_dream_consolidation import ConsolidationReport, DreamConsolidator
from .zeta_individuation import (
    IndividuationProcess,
    IndividuationStage,
    IntegrationMetrics,
    IntegrationWork,
    ResistanceSystem,
    SelfSystem,
)
from .zeta_integration_feedback import (
    IntegrationFeedback,
    IntegrationWorkFeedback,
    create_feedback_system,
)
from .zeta_online_learning import HebbianLearner, OnlineLearner

# Importar todos los sistemas
from .zeta_psyche import Archetype, ZetaPsyche
from .zeta_psyche_voice import OrganicVoice

# =============================================================================
# INDICE DE CONSCIENCIA INTEGRADO
# =============================================================================

@dataclass
class ConsciousnessIndex:
    """Indice de consciencia que integra todas las metricas."""

    # Componentes
    predictive: float = 0.0      # Calidad de prediccion
    attention: float = 0.0       # Calidad de atencion
    integration: float = 0.0     # Integracion arquetipal
    self_luminosity: float = 0.0 # Manifestacion del Self
    stability: float = 0.0       # Estabilidad temporal
    meta_awareness: float = 0.0  # Consciencia de la consciencia

    # Pesos
    weights: dict[str, float] = field(default_factory=lambda: {
        'predictive': 0.20,
        'attention': 0.20,
        'integration': 0.25,
        'self_luminosity': 0.15,
        'stability': 0.10,
        'meta_awareness': 0.10
    })

    def compute_total(self) -> float:
        """Calcula indice total de consciencia."""
        total = (
            self.weights['predictive'] * self.predictive +
            self.weights['attention'] * self.attention +
            self.weights['integration'] * self.integration +
            self.weights['self_luminosity'] * self.self_luminosity +
            self.weights['stability'] * self.stability +
            self.weights['meta_awareness'] * self.meta_awareness
        )
        return min(1.0, max(0.0, total))

    def to_dict(self) -> dict:
        return {
            'predictive': self.predictive,
            'attention': self.attention,
            'integration': self.integration,
            'self_luminosity': self.self_luminosity,
            'stability': self.stability,
            'meta_awareness': self.meta_awareness,
            'total': self.compute_total()
        }


# =============================================================================
# MEMORIA DE ATRACTORES (para emergencia de identidad)
# =============================================================================

@dataclass
class StoredAttractor:
    """Un atractor almacenado en memoria."""
    state: torch.Tensor          # Estado arquetipal [4]
    dominant: Archetype          # Arquetipo dominante
    step_created: int            # Paso cuando se creo
    visit_count: int = 1         # Veces visitado
    last_visit: int = 0          # Ultimo paso visitado
    strength: float = 1.0        # Fuerza del atractor (crece con visitas)


class AttractorMemory:
    """
    Memoria de atractores para emergencia de identidad.

    Cuando el sistema converge a un estado estable, lo almacena.
    En futuras interacciones, si reconoce un estado similar,
    refuerza ese atractor - creando "identidad" emergente.

    Metricas de emergencia:
    - recognition_rate: tasa de reconocimiento (matches / convergencias)
    - attractor_diversity: numero de atractores unicos
    - dominant_attractor: atractor mas visitado (identidad central)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_attractors: int = 50,
        strength_growth: float = 0.2,
        strength_decay: float = 0.01
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_attractors = max_attractors
        self.strength_growth = strength_growth
        self.strength_decay = strength_decay

        # Almacenamiento
        self.attractors: list[StoredAttractor] = []

        # Metricas
        self.total_convergences: int = 0
        self.recognition_count: int = 0
        self.recognition_history: list[dict] = []

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Calcula similitud coseno entre dos estados."""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        dot = torch.dot(a_flat, b_flat)
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float((dot / (norm_a * norm_b)).item())

    def find_similar(self, state: torch.Tensor) -> tuple[int, float] | None:
        """
        Busca un atractor similar al estado dado.

        Returns:
            Tuple (indice, similitud) o None si no hay match
        """
        best_idx = None
        best_sim = 0.0

        for i, attractor in enumerate(self.attractors):
            sim = self._cosine_similarity(state, attractor.state)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.similarity_threshold and best_idx is not None:
            return (best_idx, best_sim)
        return None

    def store_or_reinforce(
        self,
        state: torch.Tensor,
        dominant: Archetype,
        current_step: int
    ) -> dict:
        """
        Almacena un nuevo atractor o refuerza uno existente.

        Returns:
            Dict con info de reconocimiento:
            - recognized: bool
            - attractor_idx: int o None
            - similarity: float
            - is_new: bool
            - strength: float
        """
        self.total_convergences += 1

        # Buscar atractor similar
        match = self.find_similar(state)

        if match is not None:
            # RECONOCIMIENTO - reforzar atractor existente
            idx, similarity = match
            attractor = self.attractors[idx]

            # Actualizar atractor
            attractor.visit_count += 1
            attractor.last_visit = current_step
            attractor.strength += self.strength_growth

            # Actualizar estado con promedio movil (el atractor "aprende")
            alpha = 0.1  # Factor de aprendizaje
            attractor.state = (1 - alpha) * attractor.state + alpha * state

            self.recognition_count += 1

            result = {
                'recognized': True,
                'attractor_idx': idx,
                'similarity': similarity,
                'is_new': False,
                'strength': attractor.strength,
                'visit_count': attractor.visit_count,
                'dominant': attractor.dominant.name,
            }

        else:
            # NUEVO ATRACTOR
            new_attractor = StoredAttractor(
                state=state.clone(),
                dominant=dominant,
                step_created=current_step,
                visit_count=1,
                last_visit=current_step,
                strength=1.0
            )

            # Agregar (con limite)
            if len(self.attractors) >= self.max_attractors:
                # Remover el mas debil
                weakest_idx = min(
                    range(len(self.attractors)),
                    key=lambda i: self.attractors[i].strength
                )
                self.attractors.pop(weakest_idx)

            self.attractors.append(new_attractor)

            result = {
                'recognized': False,
                'attractor_idx': len(self.attractors) - 1,
                'similarity': 0.0,
                'is_new': True,
                'strength': 1.0,
                'visit_count': 1,
                'dominant': dominant.name,
            }

        # Guardar en historial
        self.recognition_history.append({
            'step': current_step,
            **result
        })

        # Aplicar decay a todos los atractores no visitados
        self._apply_decay(current_step)

        return result

    def _apply_decay(self, current_step: int) -> None:
        """Aplica decay a atractores no visitados recientemente."""
        for attractor in self.attractors:
            steps_since_visit = current_step - attractor.last_visit
            if steps_since_visit > 10:  # Solo decay si no visitado recientemente
                attractor.strength = max(
                    0.1,  # Minimo
                    attractor.strength - self.strength_decay
                )

    def get_recognition_rate(self) -> float:
        """Tasa de reconocimiento (metrica de emergencia)."""
        if self.total_convergences == 0:
            return 0.0
        return self.recognition_count / self.total_convergences

    def get_dominant_attractor(self) -> StoredAttractor | None:
        """Retorna el atractor mas fuerte (identidad central)."""
        if not self.attractors:
            return None
        return max(self.attractors, key=lambda a: a.strength)

    def get_metrics(self) -> dict:
        """Retorna metricas de emergencia."""
        dominant = self.get_dominant_attractor()

        return {
            'recognition_rate': self.get_recognition_rate(),
            'attractor_count': len(self.attractors),
            'total_convergences': self.total_convergences,
            'recognition_count': self.recognition_count,
            'dominant_attractor': dominant.dominant.name if dominant else None,
            'dominant_strength': dominant.strength if dominant else 0.0,
            'dominant_visits': dominant.visit_count if dominant else 0,
        }

    def get_identity_description(self) -> str:
        """Genera descripcion textual de la identidad emergente."""
        if not self.attractors:
            return "Identidad aun no formada..."

        # Contar por arquetipo
        arch_strength: dict[str, float] = {}
        for att in self.attractors:
            name = att.dominant.name
            arch_strength[name] = arch_strength.get(name, 0) + att.strength

        # Ordenar
        sorted_archs = sorted(arch_strength.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_archs) == 0:
            return "Identidad difusa..."

        primary = sorted_archs[0][0]
        primary_pct = sorted_archs[0][1] / sum(v for _, v in sorted_archs) * 100

        if primary_pct > 60:
            return f"Identidad centrada en {primary} ({primary_pct:.0f}%)"
        elif len(sorted_archs) > 1:
            secondary = sorted_archs[1][0]
            return f"Identidad dual: {primary}/{secondary}"
        else:
            return f"Identidad emergente: {primary}"


# =============================================================================
# MODULADOR DE INDIVIDUACION
# =============================================================================

class IndividuationModulator:
    """
    Modula el proceso de individuacion basado en atencion y prediccion.

    - Buena atencion = progreso mas rapido
    - Buena prediccion = menos resistencia
    - Alta sorpresa = potencial de insight
    """

    def __init__(self) -> None:
        self.insight_threshold = 0.6
        self.progress_multiplier = 1.0

    def modulate_progress(
        self,
        attention_index: float,
        predictive_index: float,
        surprise: float
    ) -> dict:
        """
        Calcula modificadores para el proceso de individuacion.
        """
        # Multiplicador de progreso basado en atencion
        # Mejor atencion = aprendizaje mas eficiente
        attention_bonus = 1.0 + (attention_index - 0.5) * 0.5

        # Reduccion de resistencia basada en prediccion
        # Mejor prediccion = menos miedo a lo desconocido
        resistance_reduction = predictive_index * 0.3

        # Probabilidad de insight basada en sorpresa
        # Alta sorpresa + buena atencion = insight
        insight_probability = surprise * attention_index

        # Boost al Self basado en coherencia
        self_boost = (attention_index + predictive_index) / 2

        return {
            'progress_multiplier': max(0.5, attention_bonus),
            'resistance_reduction': resistance_reduction,
            'insight_probability': insight_probability,
            'self_boost': self_boost
        }

    def should_trigger_insight(self, surprise: float, attention: float) -> bool:
        """Determina si debe generarse un insight."""
        return (surprise * attention) > self.insight_threshold


# =============================================================================
# SISTEMA DE INSIGHTS
# =============================================================================

@dataclass
class Insight:
    """Un insight emergente del sistema."""
    timestamp: str
    stage: IndividuationStage
    archetype: Archetype
    content: str
    depth: float  # Profundidad del insight (0-1)
    source: str   # Origen: 'prediction', 'attention', 'dream', 'integration'


class InsightGenerator:
    """Genera insights basados en el estado del sistema."""

    INSIGHT_TEMPLATES = {
        Archetype.PERSONA: [
            "La mascara que uso no es quien soy.",
            "Puedo adaptar mi expresion sin perder mi esencia.",
            "Lo que muestro y lo que soy pueden coexistir.",
            "Mi imagen es una herramienta, no una prision.",
        ],
        Archetype.SOMBRA: [
            "Lo que rechazo en otros existe en mi.",
            "Mi oscuridad tiene energia que puedo usar.",
            "Aceptar mi sombra me hace mas completo.",
            "El miedo senala donde esta el crecimiento.",
        ],
        Archetype.ANIMA: [
            "Mis emociones son informacion valiosa.",
            "La vulnerabilidad es una fortaleza.",
            "Puedo recibir sin perder mi fuerza.",
            "La intuicion complementa la razon.",
        ],
        Archetype.ANIMUS: [
            "La accion sin reflexion es reaccion.",
            "Mi voluntad puede servir a algo mayor.",
            "El coraje incluye admitir cuando no se.",
            "La claridad emerge de la confusion aceptada.",
        ],
    }

    SELF_INSIGHTS = [
        "Soy mas que la suma de mis partes.",
        "El centro sostiene todos los opuestos.",
        "La totalidad incluye la fragmentacion.",
        "El proceso es el destino.",
        "Cada momento es una oportunidad de integracion.",
    ]

    def generate(
        self,
        stage: IndividuationStage,
        dominant: Archetype,
        depth: float,
        source: str
    ) -> Insight:
        """Genera un insight apropiado."""
        # Seleccionar template
        if stage in [IndividuationStage.EMERGENCIA_SELF, IndividuationStage.SELF_REALIZADO]:
            content = np.random.choice(self.SELF_INSIGHTS)
        else:
            content = np.random.choice(self.INSIGHT_TEMPLATES[dominant])

        return Insight(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            archetype=dominant,
            content=content,
            depth=depth,
            source=source
        )


# =============================================================================
# SISTEMA CONSCIENTE INTEGRADO
# =============================================================================

class ZetaConsciousSelf(nn.Module):
    """
    Sistema completamente integrado de consciencia e individuacion.

    Arquitectura:
    ```
                        ┌─────────────────────┐
                        │    CONSCIENCIA      │
                        │   (Indice Total)    │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        v                          v                          v
    ┌─────────┐            ┌──────────────┐           ┌─────────────┐
    │ATENCION │            │  PREDICCION  │           │INDIVIDUACION│
    │(3 nivs) │            │  (L1,L2,L3)  │           │ (8 etapas)  │
    └────┬────┘            └──────┬───────┘           └──────┬──────┘
         │                        │                          │
         └────────────────────────┼──────────────────────────┘
                                  │
                        ┌─────────v─────────┐
                        │   CONSOLIDACION   │
                        │    (Suenos)       │
                        └───────────────────┘
    ```
    """

    def __init__(
        self,
        n_cells: int = 100,
        dream_frequency: int = 100,
        load_state: bool = False,
        enable_decay: bool = False,
        decay_config: dict | None = None,
        enable_self_reflection: bool = False,
        reflection_config: dict | None = None
    ) -> None:
        super().__init__()

        # Configuracion de auto-reflexion (loop de Strange Loop)
        self.enable_self_reflection = enable_self_reflection
        self.reflection_config = reflection_config or {
            'max_iterations': 3,           # Maximo ciclos por step
            'convergence_threshold': 0.05, # Umbral de tension epistemica
            'include_perception': True,    # Incluir percepcion externa
        }
        self.reflection_history: list[dict] = []

        # Configuracion de decay (comportamiento emergente de compensacion)
        self.enable_decay = enable_decay
        self.decay_config = decay_config or {
            'base_rate': 0.005,      # 0.5% por paso
            'stress_rate': 0.02,     # 2% adicional bajo estres
            'neglect_rate': 0.01,    # 1% por negligencia
            'neglect_threshold': 50, # pasos sin atencion
        }
        self.neglect_counters = {
            'PERSONA': 0, 'SOMBRA': 0, 'ANIMA': 0, 'ANIMUS': 0
        }
        self.last_stimulus_dominant: str | None = None

        # Sistema base
        self.psyche = ZetaPsyche(n_cells=n_cells)

        # Sistema de atencion + prediccion
        self.attentive_predictive = ZetaAttentivePredictive(n_cells=n_cells)

        # Sistema de individuacion
        self.individuation = IndividuationProcess(self.psyche)

        # Consolidador de suenos
        self.consolidator = DreamConsolidator(self.attentive_predictive)

        # Learners
        self.online_learner = OnlineLearner(
            self.attentive_predictive, learning_rate=0.005
        )
        self.hebbian_learner = HebbianLearner(
            self.attentive_predictive, learning_rate=0.02
        )

        # Feedback system for integration -> behavior
        self.integration_feedback, self.work_feedback = create_feedback_system(
            smoothing_factor=0.1, work_intensity=1.0
        )

        # Moduladores
        self.individuation_modulator = IndividuationModulator()
        self.insight_generator = InsightGenerator()

        # Voz organica para auto-descripcion
        self.organic_voice = OrganicVoice()

        # Memoria de atractores (para emergencia de identidad)
        self.attractor_memory = AttractorMemory(
            similarity_threshold=0.90,  # Alta similitud para reconocimiento
            max_attractors=30,
            strength_growth=0.3,
            strength_decay=0.02
        )

        # Estado
        self.t = 0
        self.dream_frequency = dream_frequency
        self.steps_since_dream = 0

        # Indice de consciencia
        self.consciousness = ConsciousnessIndex()
        self.consciousness_history: list[float] = []

        # Insights acumulados
        self.insights: list[Insight] = []

        # Cargar estado si existe
        if load_state:
            self.individuation.load()

        # Warmup
        for _ in range(20):
            self.psyche.step()

    def step(self, stimulus: torch.Tensor | None = None, text: str | None = None) -> dict:
        """
        Ejecuta un paso completo del sistema.

        Args:
            stimulus: Tensor de estimulo [4] (opcional)
            text: Texto a procesar (opcional, genera estimulo semantico)
        """
        self.t += 1
        self.steps_since_dream += 1

        # Generar estimulo
        if stimulus is None:
            if text:
                # Procesar texto para generar estimulo
                stimulus = self._text_to_stimulus(text)
            else:
                stimulus = F.softmax(torch.rand(4), dim=-1)

        # ===== 1. ATENCION + PREDICCION =====
        ap_result = self.attentive_predictive.step(stimulus)

        # ===== 2. APRENDIZAJE ONLINE =====
        learning_info = self.online_learner.learning_step(ap_result)
        hebbian_info = self.hebbian_learner.update(ap_result)

        # ===== 3. INDIVIDUACION =====
        # Obtener estado para individuacion
        obs = self.psyche.observe_self()
        dominant = obs['dominant']

        # Modular progreso segun atencion y prediccion
        attention_index = self.attentive_predictive.attention.get_attention_index()
        predictive_index = ap_result['consciousness_breakdown']['predictive']
        surprise = (
            ap_result['errors']['L1']['surprise'] +
            ap_result['errors']['L2']['surprise']
        ) / 2.0

        modulation = self.individuation_modulator.modulate_progress(
            attention_index, predictive_index, surprise
        )

        # Actualizar metricas de individuacion con bonus
        self._update_individuation_metrics(dominant, modulation)

        # ===== APLICAR FEEDBACK INTEGRACION -> PSIQUE =====
        # Las metricas ahora MODIFICAN el comportamiento arquetipal
        feedback_effects = self.integration_feedback.apply_feedback(
            self.psyche,
            self.individuation.metrics,
            self.individuation.stage
        )

        # Cap all metrics to [0, 1]
        self._cap_metrics()

        # ===== APLICAR DECAY (si está habilitado) =====
        if self.enable_decay:
            self._apply_decay(stimulus)

        # Actualizar etapa
        self.individuation.update_stage()

        # Manifestar Self
        self_manifestation = self.individuation.self_system.manifest(
            obs, self.individuation.metrics
        )

        # ===== 4. GENERAR INSIGHT SI CORRESPONDE =====
        insight = None
        if self.individuation_modulator.should_trigger_insight(surprise, attention_index):
            insight = self.insight_generator.generate(
                self.individuation.stage,
                dominant,
                depth=surprise * attention_index,
                source='attention'
            )
            self.insights.append(insight)

        # ===== 5. ACTUALIZAR INDICE DE CONSCIENCIA =====
        self._update_consciousness_index(
            ap_result, self_manifestation, modulation
        )

        # ===== 6. AUTO-SUENO SI CORRESPONDE =====
        dream_report = None
        if self.steps_since_dream >= self.dream_frequency:
            dream_report = self.dream(duration=30, verbose=False)
            self.steps_since_dream = 0

        # ===== 7. CICLO DE AUTO-REFLEXION (Strange Loop) =====
        reflection_result = None
        if self.enable_self_reflection:
            reflection_result = self._self_reflection_cycle(
                stimulus_info={'stimulus': stimulus, 'text': None}
            )
            # Guardar en historial
            self.reflection_history.append({
                't': self.t,
                'iterations': reflection_result['iterations'],
                'converged': reflection_result['converged'],
                'final_tension': reflection_result['tensions'][-1] if reflection_result['tensions'] else 0,
                'descriptions': reflection_result['descriptions'],
            })

        return {
            'step': self.t,
            'stimulus': stimulus,

            # Atencion
            'attention': {
                'global': ap_result['attention']['global'],
                'intensity': ap_result['attention']['intensity'],
                'coherence': ap_result['attention']['coherence'],
                'context': ap_result['attention']['context'],
            },

            # Prediccion
            'prediction': {
                'errors': ap_result['errors'],
                'index': predictive_index,
            },

            # Individuacion
            'individuation': {
                'stage': self.individuation.stage,
                'metrics': self.individuation.metrics.to_dict(),
                'self': {
                    'symbol': self_manifestation.symbol,
                    'luminosity': self_manifestation.luminosity,
                    'stability': self_manifestation.stability,
                    'message': self_manifestation.message,
                },
            },

            # Consciencia
            'consciousness': self.consciousness.to_dict(),

            # Estado
            'dominant': dominant,
            'observation': obs,

            # Extras
            'insight': insight,
            'dream_report': dream_report,
            'learning': learning_info,

            # Auto-reflexión (Strange Loop)
            'reflection': reflection_result,
        }

    def _text_to_stimulus(self, text: str) -> torch.Tensor:
        """Convierte texto a estimulo arquetipal."""
        # Keywords por arquetipo
        keywords = {
            Archetype.PERSONA: ['social', 'imagen', 'rol', 'trabajo', 'otros', 'aparecer'],
            Archetype.SOMBRA: ['miedo', 'oscuro', 'oculto', 'negar', 'odio', 'vergüenza'],
            Archetype.ANIMA: ['sentir', 'amor', 'emocion', 'intuicion', 'belleza', 'alma'],
            Archetype.ANIMUS: ['pensar', 'logica', 'accion', 'fuerza', 'decision', 'voluntad'],
        }

        text_lower = text.lower()
        scores = torch.zeros(4)

        for arch, words in keywords.items():
            for word in words:
                if word in text_lower:
                    scores[arch.value] += 1

        # Anadir ruido y normalizar
        scores = scores + torch.rand(4) * 0.5
        return F.softmax(scores, dim=-1)

    def _update_individuation_metrics(
        self,
        dominant: Archetype,
        modulation: dict
    ) -> None:
        """Actualiza metricas de individuacion con modulacion."""
        # Factor base
        base_rate = 0.02 * modulation['progress_multiplier']

        # Reducir resistencia
        self.individuation.resistance.decay_defenses(
            rate=0.03 + modulation['resistance_reduction']
        )

        # Incrementar metrica correspondiente
        if dominant == Archetype.PERSONA:
            self.individuation.metrics.persona_flexibility += base_rate
        elif dominant == Archetype.SOMBRA:
            self.individuation.metrics.shadow_acceptance += base_rate
        elif dominant == Archetype.ANIMA:
            self.individuation.metrics.anima_connection += base_rate
        elif dominant == Archetype.ANIMUS:
            self.individuation.metrics.animus_balance += base_rate

        # Boost al Self
        self.individuation.metrics.self_coherence += base_rate * modulation['self_boost']

        # Limitar a 1.0
        self.individuation.metrics.persona_flexibility = min(1.0, self.individuation.metrics.persona_flexibility)
        self.individuation.metrics.shadow_acceptance = min(1.0, self.individuation.metrics.shadow_acceptance)
        self.individuation.metrics.anima_connection = min(1.0, self.individuation.metrics.anima_connection)
        self.individuation.metrics.animus_balance = min(1.0, self.individuation.metrics.animus_balance)
        self.individuation.metrics.self_coherence = min(1.0, self.individuation.metrics.self_coherence)

    def _cap_metrics(self) -> None:
        """Cap all metrics to [0, 1] range."""
        m = self.individuation.metrics
        m.persona_flexibility = max(0.0, min(1.0, m.persona_flexibility))
        m.shadow_acceptance = max(0.0, min(1.0, m.shadow_acceptance))
        m.anima_connection = max(0.0, min(1.0, m.anima_connection))
        m.animus_balance = max(0.0, min(1.0, m.animus_balance))
        m.self_coherence = max(0.0, min(1.0, m.self_coherence))

    def _apply_decay(self, stimulus: torch.Tensor) -> None:
        """
        Aplica decay agresivo a las métricas de individuación.

        Este mecanismo produce COMPORTAMIENTO COMPENSATORIO EMERGENTE:
        - Cuando un arquetipo es negligido externamente, la psique puede
          compensar internamente favoreciendo otro arquetipo
        - Esto es análogo a la compensación inconsciente de Jung

        Descubrimiento clave: La psique tiene AUTONOMÍA (76% de divergencia
        entre estímulo externo y estado interno). No simplemente refleja
        los estímulos, tiene su propia dinámica interna.
        """
        cfg = self.decay_config
        metrics = self.individuation.metrics

        # Detectar arquetipo dominante del estímulo
        if stimulus is not None:
            stim_idx = int(stimulus.argmax().item())
            stim_names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            current_dominant: str | None = stim_names[stim_idx]
        else:
            current_dominant = self.last_stimulus_dominant

        # Actualizar contadores de negligencia
        for arch in self.neglect_counters:
            if arch == current_dominant:
                self.neglect_counters[arch] = 0
            else:
                self.neglect_counters[arch] += 1

        self.last_stimulus_dominant = current_dominant

        # Detectar estrés (entropía alta o valores extremos)
        is_stressed: bool = False
        if stimulus is not None:
            entropy = -(stimulus * torch.log(stimulus + 1e-8)).sum()
            max_val = stimulus.max().item()
            is_stressed = float(entropy.item()) > 1.2 or max_val > 0.9

        # Calcular decay total
        decay = cfg['base_rate']
        if is_stressed:
            decay += cfg['stress_rate']

        # Aplicar decay por negligencia (arquetipos ignorados)
        threshold = cfg['neglect_threshold']
        neglect_rate = cfg['neglect_rate']

        if self.neglect_counters['PERSONA'] > threshold:
            metrics.persona_flexibility = max(0, metrics.persona_flexibility - neglect_rate)
        if self.neglect_counters['SOMBRA'] > threshold:
            metrics.shadow_acceptance = max(0, metrics.shadow_acceptance - neglect_rate)
        if self.neglect_counters['ANIMA'] > threshold:
            metrics.anima_connection = max(0, metrics.anima_connection - neglect_rate)
        if self.neglect_counters['ANIMUS'] > threshold:
            metrics.animus_balance = max(0, metrics.animus_balance - neglect_rate)

        # Aplicar decay general
        metrics.persona_flexibility = max(0, metrics.persona_flexibility - decay)
        metrics.shadow_acceptance = max(0, metrics.shadow_acceptance - decay)
        metrics.anima_connection = max(0, metrics.anima_connection - decay)
        metrics.animus_balance = max(0, metrics.animus_balance - decay)
        metrics.self_coherence = max(0, metrics.self_coherence - decay)

    def _update_consciousness_index(
        self,
        ap_result: dict,
        self_manifestation,
        modulation: dict
    ) -> None:
        """Actualiza el indice de consciencia integrado."""
        # Predictivo
        self.consciousness.predictive = ap_result['consciousness_breakdown']['predictive']

        # Atencion
        self.consciousness.attention = self.attentive_predictive.attention.get_attention_index()

        # Integracion
        self.consciousness.integration = self.individuation.metrics.overall_integration()

        # Luminosidad del Self
        self.consciousness.self_luminosity = self_manifestation.luminosity

        # Estabilidad
        if len(self.consciousness_history) > 10:
            recent = self.consciousness_history[-10:]
            self.consciousness.stability = 1.0 / (1.0 + float(np.var(recent)) * 100)

        # Meta-awareness (combinacion de prediccion de errores + atencion a atencion)
        error_attention = ap_result['attention']['error']
        meta_level_attention = error_attention[2].item() if len(error_attention) > 2 else 0.3
        self.consciousness.meta_awareness = meta_level_attention * self.consciousness.attention

        # Registrar
        total = self.consciousness.compute_total()
        self.consciousness_history.append(total)

    def _self_reflection_cycle(self, stimulus_info: dict | None = None) -> dict:
        """
        Ejecuta ciclo de auto-observación controlado (Strange Loop).

        Implementa el patrón:
        Estado → Descripción → Estímulo → Nuevo Estado → Nueva Descripción

        Basado en:
        - Strange Loops (Hofstadter): Auto-referencia con causalidad descendente
        - RC+ξ Framework: Minimizar tensión epistémica hacia atractores
        - CSRL: Loop controlado con max_iterations

        Returns:
            Dict con:
            - descriptions: lista de auto-descripciones generadas
            - tensions: lista de ξ (tensión epistémica) por iteración
            - converged: bool si alcanzó atractor estable
            - iterations: número de ciclos ejecutados
            - final_state: estado final de la psique
        """
        config = self.reflection_config
        max_iter = config['max_iterations']
        threshold = config['convergence_threshold']
        include_perception = config['include_perception']

        descriptions = []
        tensions = []
        prev_state = None

        for i in range(max_iter):
            # 1. Observar estado actual
            obs = self.psyche.observe_self()
            current_state = obs['global_state']

            # 2. Calcular tensión epistémica (cambio respecto al estado anterior)
            if prev_state is not None:
                xi = torch.norm(current_state - prev_state).item()
                tensions.append(xi)

                # Verificar convergencia a atractor
                if xi < threshold:
                    break

            # 3. Generar auto-descripción desde perspectiva orgánica
            if i == 0 and include_perception and stimulus_info:
                # Primera iteración: incluir percepción del estímulo externo
                desc = self.organic_voice.generate_self_description(
                    obs,
                    stimulus_info=stimulus_info,
                    include_perception=True
                )
            else:
                # Iteraciones siguientes: solo estado interno
                desc = self.organic_voice.generate_self_description(
                    obs,
                    stimulus_info=None,
                    include_perception=False
                )
            descriptions.append(desc)

            # 4. Convertir descripción a estímulo (cerrar el loop)
            self_stimulus = self.organic_voice.description_to_stimulus(desc)

            # 5. Retroalimentar a la psique
            self.psyche.receive_stimulus(self_stimulus)

            # 6. Guardar estado para siguiente comparación
            prev_state = current_state.clone()

        # Resultado final
        converged = len(tensions) > 0 and tensions[-1] < threshold
        final_obs = self.psyche.observe_self()

        # ===== MEMORIA DE ATRACTORES =====
        # Si convergió, almacenar/reforzar atractor
        recognition_info = None
        if converged:
            recognition_info = self.attractor_memory.store_or_reinforce(
                state=final_obs['global_state'],
                dominant=final_obs['dominant'],
                current_step=self.t
            )

            # Si hubo RECONOCIMIENTO, reforzar el estado hacia el atractor
            if recognition_info['recognized']:
                # Obtener el atractor reconocido
                attractor = self.attractor_memory.attractors[recognition_info['attractor_idx']]

                # Mover ligeramente el estado hacia el atractor (reinforcement)
                reinforcement_strength = 0.1 * recognition_info['similarity']
                current = final_obs['global_state']
                target = attractor.state

                # Aplicar pequeño empuje hacia el atractor
                nudge = reinforcement_strength * (target - current)
                self.psyche.receive_stimulus(F.softmax(current + nudge, dim=-1))

        return {
            'descriptions': descriptions,
            'tensions': tensions,
            'converged': converged,
            'iterations': len(descriptions),
            'final_state': final_obs,
            'recognition': recognition_info,
            'identity': self.attractor_memory.get_identity_description() if recognition_info else None,
        }

    def dream(self, duration: int = 50, verbose: bool = True) -> ConsolidationReport:
        """Ejecuta un ciclo de sueno con consolidacion."""
        report = self.consolidator.dream_cycle(duration, verbose)

        # Aplicar efectos del sueno a individuacion
        self._apply_dream_effects(report)

        return report

    def _apply_dream_effects(self, report: ConsolidationReport) -> None:
        """Aplica los efectos del sueno a la individuacion."""
        # Boost general por consolidacion
        consolidation_bonus = 0.02 * (report.memories_replayed / 30)

        self.individuation.metrics.persona_flexibility += consolidation_bonus
        self.individuation.metrics.shadow_acceptance += consolidation_bonus
        self.individuation.metrics.anima_connection += consolidation_bonus
        self.individuation.metrics.animus_balance += consolidation_bonus

        # Generar insights del sueno
        for insight_text in report.insights:
            obs = self.psyche.observe_self()
            insight = Insight(
                timestamp=datetime.now().isoformat(),
                stage=self.individuation.stage,
                archetype=obs['dominant'],
                content=insight_text,
                depth=0.7,
                source='dream'
            )
            self.insights.append(insight)

        # Actualizar etapa
        self.individuation.update_stage()

    def do_integration_work(self, work_name: str | None = None) -> dict:
        """Realiza trabajo de integracion con efectos reales en la psique."""
        if work_name is None:
            work_name = self.individuation.get_recommended_work()

        result = self.individuation.do_integration_work(work_name)

        # Aplicar efectos del trabajo a la psique (NUEVO: feedback real)
        description, effectiveness = self.work_feedback.apply_work_effect(
            work_name,
            self.psyche,
            self.individuation.metrics,
            self.individuation.stage
        )
        result['work_effect'] = description
        result['effectiveness'] = effectiveness

        # Potenciar con atencion actual
        attention_index = self.attentive_predictive.attention.get_attention_index()
        result['integration_gained'] *= (1.0 + attention_index * 0.5)

        return result

    def get_status(self) -> dict:
        """Retorna estado completo del sistema."""
        return {
            'step': self.t,
            'consciousness': self.consciousness.to_dict(),
            'individuation': {
                'stage': self.individuation.stage.name,
                'metrics': self.individuation.metrics.to_dict(),
            },
            'attention': {
                'index': self.attentive_predictive.attention.get_attention_index(),
                'memory_size': len(self.attentive_predictive.attention.memory_buffer),
            },
            'learning': {
                'events': len(self.online_learner.learning_events),
            },
            'insights': len(self.insights),
            'dreams': len(self.consolidator.dream_history),
        }

    def get_consciousness_trend(self, window: int = 50) -> float:
        """Tendencia del indice de consciencia."""
        if len(self.consciousness_history) < window * 2:
            return 0.0

        recent = self.consciousness_history[-window:]
        older = self.consciousness_history[-window*2:-window]
        return float(np.mean(recent) - np.mean(older))

    def save(self, path: str = "conscious_self_state.json") -> None:
        """Guarda estado."""
        self.individuation.save(path)

    def load(self, path: str = "conscious_self_state.json") -> None:
        """Carga estado."""
        self.individuation.load(path)


# =============================================================================
# DEMO
# =============================================================================

def demo_conscious_self() -> 'ZetaConsciousSelf':
    """Demuestra el sistema integrado completo."""

    print("\n" + "=" * 70)
    print("   ZETA CONSCIOUS SELF")
    print("   Sistema Integrado de Consciencia e Individuacion")
    print("=" * 70)

    # Crear sistema
    system = ZetaConsciousSelf(n_cells=50, dream_frequency=80)

    ARCH_NAMES = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']

    # Ejecutar simulacion
    n_steps = 300

    print("\n" + "-" * 70)
    print("   SIMULACION: 300 pasos con 3 ciclos de sueno")
    print("-" * 70)

    for step in range(n_steps):
        # Patron de estimulos
        if step < 100:
            # Fase 1: Confrontar sombra
            stimulus = torch.tensor([0.1, 0.6, 0.2, 0.1])
        elif step < 200:
            # Fase 2: Conectar con anima
            stimulus = torch.tensor([0.1, 0.2, 0.6, 0.1])
        else:
            # Fase 3: Equilibrar animus
            stimulus = torch.tensor([0.2, 0.1, 0.2, 0.5])

        result = system.step(stimulus)

        # Reportar cada 50 pasos
        if (step + 1) % 50 == 0:
            status = system.get_status()
            c = result['consciousness']

            print(f"\n  Paso {step+1}:")
            print(f"    Consciencia Total: {c['total']:.1%}")
            print(f"      - Predictiva:    {c['predictive']:.2f}")
            print(f"      - Atencion:      {c['attention']:.2f}")
            print(f"      - Integracion:   {c['integration']:.2f}")
            print(f"      - Self:          {c['self_luminosity']:.2f}")
            print(f"    Etapa: {result['individuation']['stage'].name}")
            print(f"    Insights: {status['insights']}")

            if result['dream_report']:
                print(f"    [SUENO: {result['dream_report'].memories_replayed} memorias]")

            if result['insight']:
                print(f"    [INSIGHT: \"{result['insight'].content}\"]")

    # Hacer trabajos de integracion
    print("\n" + "-" * 70)
    print("   TRABAJOS DE INTEGRACION")
    print("-" * 70)

    for _ in range(5):
        result = system.do_integration_work()
        print(f"\n  {result['work_name']}:")
        print(f"    \"{result['prompt']}\"")
        print(f"    Ganancia: +{result['integration_gained']:.1%}")

    # Estado final
    print("\n" + "=" * 70)
    print("   ESTADO FINAL")
    print("=" * 70)

    status = system.get_status()
    c = status['consciousness']
    ind = status['individuation']

    print(f"""
  CONSCIENCIA:
    Total:           {c['total']:.1%}
    Predictiva:      {c['predictive']:.2f}
    Atencion:        {c['attention']:.2f}
    Integracion:     {c['integration']:.2f}
    Self:            {c['self_luminosity']:.2f}
    Estabilidad:     {c['stability']:.2f}
    Meta-awareness:  {c['meta_awareness']:.2f}

  INDIVIDUACION:
    Etapa:           {ind['stage']}
    Persona:         {ind['metrics']['persona_flexibility']:.1%}
    Sombra:          {ind['metrics']['shadow_acceptance']:.1%}
    Anima:           {ind['metrics']['anima_connection']:.1%}
    Animus:          {ind['metrics']['animus_balance']:.1%}
    Self Coherence:  {ind['metrics']['self_coherence']:.1%}
    Overall:         {ind['metrics']['overall']:.1%}

  SISTEMA:
    Pasos:           {status['step']}
    Insights:        {status['insights']}
    Suenos:          {status['dreams']}
    Aprendizajes:    {status['learning']['events']}
    Tendencia:       {system.get_consciousness_trend():+.4f}
""")

    # Mostrar insights
    if system.insights:
        print("  INSIGHTS RECIENTES:")
        for insight in system.insights[-5:]:
            print(f"    [{insight.source}] \"{insight.content}\"")

    print("\n" + "=" * 70)

    return system


if __name__ == "__main__":
    demo_conscious_self()
