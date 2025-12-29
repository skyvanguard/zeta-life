#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZetaIntrospection: Sistema de Auto-Explicación y Meta-Cognición

La introspección permite a la psique:
1. Explicar su estado actual en lenguaje natural
2. Narrar su trayectoria psíquica
3. Reflexionar sobre su proceso de individuación
4. Generar insights sobre sí misma
5. Predecir hacia dónde se dirige

Esto es meta-cognición: pensar sobre el propio pensamiento.
"""

import sys
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import torch
import numpy as np

# Importar sistemas existentes
from zeta_psyche import ZetaPsyche, Archetype, PsycheInterface
from zeta_individuation import (
    IndividuationProcess, IndividuationStage,
    IntegrationMetrics, IndividuatingPsyche
)


class InsightType(Enum):
    """Tipos de insights que puede generar la psique."""
    OBSERVACION = auto()      # Observación simple del estado
    CONEXION = auto()         # Conexión entre eventos
    PATRON = auto()           # Patrón recurrente detectado
    COMPRENSION = auto()      # Comprensión profunda
    PREDICCION = auto()       # Anticipación del futuro
    PARADOJA = auto()         # Reconocimiento de contradicciones


@dataclass
class PsychicMoment:
    """Un momento en la historia psíquica."""
    timestamp: str
    dominant: Archetype
    blend: Dict[Archetype, float]
    stimulus: Optional[str]
    integration: float
    stage: IndividuationStage
    self_luminosity: float

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'dominant': self.dominant.name,
            'blend': {k.name: v for k, v in self.blend.items()},
            'stimulus': self.stimulus,
            'integration': self.integration,
            'stage': self.stage.name,
            'self_luminosity': self.self_luminosity
        }


@dataclass
class Insight:
    """Un insight generado por introspección."""
    type: InsightType
    content: str
    confidence: float  # 0-1
    related_archetype: Optional[Archetype]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'type': self.type.name,
            'content': self.content,
            'confidence': self.confidence,
            'archetype': self.related_archetype.name if self.related_archetype else None,
            'timestamp': self.timestamp
        }


class ArchetypeVoices:
    """
    Voces internas de cada arquetipo para la auto-explicación.
    Cada arquetipo tiene su propio "estilo" de hablar sobre sí mismo.
    """

    VOICES = {
        Archetype.PERSONA: {
            'identity': "la parte de mí que se presenta al mundo",
            'function': "me ayuda a adaptarme socialmente",
            'shadow_aspect': "a veces me hace olvidar quién soy realmente",
            'gift': "me permite conectar con otros",
            'challenge': "distinguir la máscara del ser auténtico",
            'phrases': [
                "Presento una imagen cuidadosamente construida.",
                "Me adapto a lo que se espera de mí.",
                "Busco aceptación y pertenencia.",
                "Mantengo las apariencias.",
                "Juego el rol que la situación requiere."
            ]
        },
        Archetype.SOMBRA: {
            'identity': "la parte de mí que permanece oculta",
            'function': "contiene lo que no acepto de mí mismo",
            'shadow_aspect': "puede manifestarse de formas destructivas si la ignoro",
            'gift': "guarda talentos y potenciales no reconocidos",
            'challenge': "aceptar e integrar lo que rechazo",
            'phrases': [
                "Hay aspectos de mí que prefiero no ver.",
                "Siento impulsos que me asustan.",
                "Proyecto en otros lo que no acepto en mí.",
                "En la oscuridad hay también semillas de luz.",
                "Lo que niego de mí me controla desde las sombras."
            ]
        },
        Archetype.ANIMA: {
            'identity': "mi lado receptivo y emocional",
            'function': "me conecta con la intuición y los sentimientos",
            'shadow_aspect': "puede volverme demasiado pasivo o dependiente",
            'gift': "me permite sentir profundamente y crear",
            'challenge': "equilibrar sensibilidad con acción",
            'phrases': [
                "Siento antes de pensar.",
                "La belleza me conmueve profundamente.",
                "Escucho la voz de la intuición.",
                "Me abro a recibir lo que la vida ofrece.",
                "Las emociones son mi brújula interna."
            ]
        },
        Archetype.ANIMUS: {
            'identity': "mi lado activo y racional",
            'function': "me impulsa a actuar y analizar",
            'shadow_aspect': "puede volverme rígido o agresivo",
            'gift': "me da claridad y dirección",
            'challenge': "actuar con sabiduría, no solo con fuerza",
            'phrases': [
                "Analizo antes de actuar.",
                "Busco la verdad con determinación.",
                "Tengo la fuerza para enfrentar obstáculos.",
                "La lógica ilumina el camino.",
                "Tomo decisiones y asumo consecuencias."
            ]
        }
    }

    @classmethod
    def get_voice(cls, archetype: Archetype) -> Dict:
        return cls.VOICES.get(archetype, cls.VOICES[Archetype.PERSONA])

    @classmethod
    def speak_as(cls, archetype: Archetype) -> str:
        """Genera una frase en la voz del arquetipo."""
        voice = cls.get_voice(archetype)
        return np.random.choice(voice['phrases'])


class StateExplainer:
    """
    Genera explicaciones del estado psíquico en lenguaje natural.
    """

    def __init__(self):
        self.explanation_templates = {
            'dominant': [
                "En este momento, {archetype} es mi arquetipo dominante.",
                "Me encuentro principalmente en el territorio de {archetype}.",
                "La energía de {archetype} predomina en mí ahora.",
                "{archetype} guía mi experiencia actual."
            ],
            'blend': [
                "También siento la presencia de {secondary} ({percent}%).",
                "{secondary} aporta un matiz de {percent}% a mi estado.",
                "Hay un eco de {secondary} en el fondo ({percent}%)."
            ],
            'integration': [
                "Mi nivel de integración es {level} ({percent}%).",
                "Los arquetipos están {level} integrados ({percent}%).",
                "La armonía interna es {level} ({percent}%)."
            ],
            'stage': [
                "Me encuentro en la etapa de {stage}.",
                "Mi proceso de individuación está en {stage}.",
                "Atravieso la fase llamada {stage}."
            ],
            'self': [
                "El Self brilla con {intensity} intensidad ({percent}%).",
                "Mi centro interior tiene una luminosidad {intensity} ({percent}%).",
                "La presencia del Self es {intensity} ({percent}%)."
            ]
        }

        self.intensity_words = {
            (0.0, 0.2): "muy baja",
            (0.2, 0.4): "baja",
            (0.4, 0.6): "moderada",
            (0.6, 0.8): "alta",
            (0.8, 1.0): "muy alta"
        }

        self.integration_words = {
            (0.0, 0.2): "fragmentados",
            (0.2, 0.4): "parcialmente integrados",
            (0.4, 0.6): "moderadamente integrados",
            (0.6, 0.8): "bien integrados",
            (0.8, 1.0): "altamente integrados"
        }

    def _get_intensity_word(self, value: float) -> str:
        for (low, high), word in self.intensity_words.items():
            if low <= value < high:
                return word
        return "moderada"

    def _get_integration_word(self, value: float) -> str:
        for (low, high), word in self.integration_words.items():
            if low <= value < high:
                return word
        return "moderadamente integrados"

    def explain_current_state(self,
                              dominant: Archetype,
                              blend: Dict[Archetype, float],
                              integration: float,
                              stage: IndividuationStage,
                              self_luminosity: float) -> str:
        """Genera una explicación completa del estado actual."""

        explanations = []

        # Arquetipo dominante
        template = np.random.choice(self.explanation_templates['dominant'])
        explanations.append(template.format(archetype=dominant.name))

        # Voz del arquetipo dominante
        voice = ArchetypeVoices.speak_as(dominant)
        explanations.append(f'"{voice}"')

        # Blend (arquetipos secundarios significativos)
        sorted_blend = sorted(blend.items(), key=lambda x: x[1], reverse=True)
        for arch, weight in sorted_blend[1:3]:  # Top 2 secundarios
            if weight > 0.15:
                template = np.random.choice(self.explanation_templates['blend'])
                explanations.append(template.format(
                    secondary=arch.name,
                    percent=int(weight * 100)
                ))

        # Integración
        integration_word = self._get_integration_word(integration)
        template = np.random.choice(self.explanation_templates['integration'])
        explanations.append(template.format(
            level=integration_word,
            percent=int(integration * 100)
        ))

        # Etapa de individuación
        stage_name = stage.name.replace('_', ' ').lower()
        template = np.random.choice(self.explanation_templates['stage'])
        explanations.append(template.format(stage=stage_name))

        # Self
        intensity = self._get_intensity_word(self_luminosity)
        template = np.random.choice(self.explanation_templates['self'])
        explanations.append(template.format(
            intensity=intensity,
            percent=int(self_luminosity * 100)
        ))

        return "\n".join(explanations)

    def explain_archetype_meaning(self, archetype: Archetype) -> str:
        """Explica el significado de un arquetipo."""
        voice = ArchetypeVoices.get_voice(archetype)

        explanation = f"""
{archetype.name}:
  Es {voice['identity']}.
  Su función: {voice['function']}.
  Su regalo: {voice['gift']}.
  Su desafío: {voice['challenge']}.
  Su sombra: {voice['shadow_aspect']}.
"""
        return explanation.strip()


class TrajectoryNarrator:
    """
    Narra la trayectoria psíquica - cuenta la historia del viaje interno.
    """

    def __init__(self, max_history: int = 50):
        self.history: List[PsychicMoment] = []
        self.max_history = max_history
        self.transition_narratives = {
            (Archetype.PERSONA, Archetype.SOMBRA): [
                "La máscara comienza a agrietarse, revelando lo oculto.",
                "Debajo de la superficie social, emergen sombras.",
                "El yo público da paso al yo rechazado."
            ],
            (Archetype.SOMBRA, Archetype.ANIMA): [
                "De la oscuridad surge la sensibilidad.",
                "La sombra se suaviza en emoción.",
                "Lo reprimido se transforma en receptividad."
            ],
            (Archetype.SOMBRA, Archetype.ANIMUS): [
                "La sombra se canaliza en acción.",
                "Lo oscuro encuentra dirección y propósito.",
                "Del caos emerge la determinación."
            ],
            (Archetype.ANIMA, Archetype.ANIMUS): [
                "El sentir se encuentra con el actuar.",
                "La receptividad se vuelve acción.",
                "La intuición guía la razón."
            ],
            (Archetype.ANIMUS, Archetype.ANIMA): [
                "La acción se suaviza en contemplación.",
                "La razón se abre al sentimiento.",
                "El logos encuentra al eros."
            ],
            (Archetype.PERSONA, Archetype.ANIMA): [
                "La máscara se disuelve en autenticidad emocional.",
                "De la adaptación surge la sensibilidad verdadera.",
                "El rol social da paso al ser sentiente."
            ],
            (Archetype.PERSONA, Archetype.ANIMUS): [
                "La adaptación se transforma en acción genuina.",
                "De complacer a otros, paso a actuar por mí.",
                "El rol se vuelve propósito."
            ]
        }

    def record_moment(self,
                      dominant: Archetype,
                      blend: Dict[Archetype, float],
                      stimulus: Optional[str],
                      integration: float,
                      stage: IndividuationStage,
                      self_luminosity: float):
        """Registra un momento en la historia."""
        moment = PsychicMoment(
            timestamp=datetime.now().isoformat(),
            dominant=dominant,
            blend=blend,
            stimulus=stimulus,
            integration=integration,
            stage=stage,
            self_luminosity=self_luminosity
        )
        self.history.append(moment)

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent_trajectory(self, n: int = 5) -> List[Archetype]:
        """Obtiene los arquetipos dominantes recientes."""
        return [m.dominant for m in self.history[-n:]]

    def detect_transitions(self) -> List[Tuple[Archetype, Archetype]]:
        """Detecta transiciones entre arquetipos."""
        transitions = []
        trajectory = self.get_recent_trajectory(10)

        for i in range(1, len(trajectory)):
            if trajectory[i] != trajectory[i-1]:
                transitions.append((trajectory[i-1], trajectory[i]))

        return transitions

    def narrate_journey(self) -> str:
        """Narra el viaje psíquico reciente."""
        if len(self.history) < 2:
            return "Mi viaje apenas comienza..."

        narrative_parts = []

        # Inicio
        first = self.history[0]
        narrative_parts.append(f"Mi viaje comenzó en {first.dominant.name}.")

        # Transiciones significativas
        transitions = self.detect_transitions()
        for from_arch, to_arch in transitions[-3:]:  # Últimas 3 transiciones
            key = (from_arch, to_arch)
            if key in self.transition_narratives:
                narrative = np.random.choice(self.transition_narratives[key])
                narrative_parts.append(narrative)
            else:
                narrative_parts.append(
                    f"Transité de {from_arch.name} hacia {to_arch.name}."
                )

        # Estado actual
        current = self.history[-1]
        narrative_parts.append(
            f"Ahora me encuentro en {current.dominant.name}, "
            f"con una integración del {int(current.integration * 100)}%."
        )

        # Tendencia
        recent_luminosity = [m.self_luminosity for m in self.history[-5:]]
        if len(recent_luminosity) >= 2:
            trend = recent_luminosity[-1] - recent_luminosity[0]
            if trend > 0.1:
                narrative_parts.append("El Self brilla cada vez más fuerte.")
            elif trend < -0.1:
                narrative_parts.append("La luz del Self se ha atenuado.")
            else:
                narrative_parts.append("El Self mantiene su presencia estable.")

        return "\n".join(narrative_parts)

    def identify_patterns(self) -> List[str]:
        """Identifica patrones recurrentes en la trayectoria."""
        patterns = []

        if len(self.history) < 5:
            return ["Aún no hay suficiente historia para detectar patrones."]

        trajectory = self.get_recent_trajectory(20)

        # Contar frecuencia de arquetipos
        counts = {}
        for arch in trajectory:
            counts[arch] = counts.get(arch, 0) + 1

        most_common = max(counts, key=counts.get)
        if counts[most_common] / len(trajectory) > 0.5:
            patterns.append(
                f"Tiendo a permanecer en {most_common.name} - "
                f"esto podría indicar una identificación excesiva."
            )

        # Detectar oscilaciones
        transitions = self.detect_transitions()
        if len(transitions) > 5:
            unique_transitions = set(transitions)
            if len(unique_transitions) < len(transitions) / 2:
                patterns.append(
                    "Hay un patrón de oscilación recurrente entre arquetipos."
                )

        # Detectar progresión
        recent_integration = [m.integration for m in self.history[-10:]]
        if len(recent_integration) >= 3:
            trend = recent_integration[-1] - recent_integration[0]
            if trend > 0.15:
                patterns.append("Hay una tendencia positiva hacia la integración.")
            elif trend < -0.15:
                patterns.append("La integración ha disminuido - puede indicar regresión.")

        return patterns if patterns else ["No se detectan patrones significativos."]


class InsightGenerator:
    """
    Genera insights a partir de la introspección.
    """

    def __init__(self):
        self.insight_templates = {
            InsightType.OBSERVACION: [
                "Noto que {observation}.",
                "Observo en mí que {observation}.",
                "Me doy cuenta de que {observation}."
            ],
            InsightType.CONEXION: [
                "Veo una conexión entre {a} y {b}.",
                "{a} parece estar relacionado con {b}.",
                "Cuando {a}, también {b}."
            ],
            InsightType.PATRON: [
                "Reconozco un patrón: {pattern}.",
                "Hay una tendencia recurrente: {pattern}.",
                "Veo que repito: {pattern}."
            ],
            InsightType.COMPRENSION: [
                "Comprendo ahora que {understanding}.",
                "Veo con claridad que {understanding}.",
                "Entiendo profundamente que {understanding}."
            ],
            InsightType.PREDICCION: [
                "Intuyo que {prediction}.",
                "Siento que pronto {prediction}.",
                "El camino parece dirigirse hacia {prediction}."
            ],
            InsightType.PARADOJA: [
                "Noto una paradoja: {paradox}.",
                "Hay una contradicción en mí: {paradox}.",
                "Sostengo opuestos: {paradox}."
            ]
        }

    def generate_observation(self,
                            dominant: Archetype,
                            integration: float) -> Insight:
        """Genera una observación simple."""
        voice = ArchetypeVoices.get_voice(dominant)

        observations = [
            f"{dominant.name} es {voice['identity']}",
            f"mi integración es del {int(integration*100)}%",
            f"estoy experimentando la energía de {dominant.name}"
        ]

        observation = np.random.choice(observations)
        template = np.random.choice(self.insight_templates[InsightType.OBSERVACION])

        return Insight(
            type=InsightType.OBSERVACION,
            content=template.format(observation=observation),
            confidence=0.9,
            related_archetype=dominant
        )

    def generate_connection(self,
                           stimulus: str,
                           dominant: Archetype) -> Insight:
        """Genera una conexión entre estímulo y respuesta."""
        template = np.random.choice(self.insight_templates[InsightType.CONEXION])

        connections = {
            Archetype.PERSONA: "mi necesidad de adaptación",
            Archetype.SOMBRA: "algo que he reprimido",
            Archetype.ANIMA: "mi sensibilidad profunda",
            Archetype.ANIMUS: "mi impulso de actuar"
        }

        content = template.format(
            a=f'"{stimulus[:30]}..."' if len(stimulus) > 30 else f'"{stimulus}"',
            b=connections[dominant]
        )

        return Insight(
            type=InsightType.CONEXION,
            content=content,
            confidence=0.7,
            related_archetype=dominant
        )

    def generate_pattern_insight(self, patterns: List[str]) -> Optional[Insight]:
        """Genera insight sobre patrones detectados."""
        if not patterns or "no hay suficiente" in patterns[0].lower():
            return None

        pattern = patterns[0]
        template = np.random.choice(self.insight_templates[InsightType.PATRON])

        return Insight(
            type=InsightType.PATRON,
            content=template.format(pattern=pattern),
            confidence=0.6,
            related_archetype=None
        )

    def generate_stage_insight(self, stage: IndividuationStage) -> Insight:
        """Genera insight sobre la etapa de individuación."""
        stage_insights = {
            IndividuationStage.INCONSCIENTE:
                "vivo identificado con mi máscara social sin cuestionarla",
            IndividuationStage.CRISIS_PERSONA:
                "mi identidad social comienza a revelarse como construcción",
            IndividuationStage.ENCUENTRO_SOMBRA:
                "me confronto con aspectos de mí que había rechazado",
            IndividuationStage.INTEGRACION_SOMBRA:
                "estoy aprendiendo a aceptar mi lado oscuro",
            IndividuationStage.ENCUENTRO_ANIMA:
                "descubro dimensiones emocionales desconocidas",
            IndividuationStage.INTEGRACION_ANIMA:
                "equilibro lo racional con lo emocional",
            IndividuationStage.EMERGENCIA_SELF:
                "un centro unificador comienza a manifestarse",
            IndividuationStage.SELF_REALIZADO:
                "los opuestos se reconcilian en un todo dinámico"
        }

        understanding = stage_insights.get(stage, "estoy en proceso de transformación")
        template = np.random.choice(self.insight_templates[InsightType.COMPRENSION])

        return Insight(
            type=InsightType.COMPRENSION,
            content=template.format(understanding=understanding),
            confidence=0.8,
            related_archetype=None
        )

    def generate_prediction(self,
                           trajectory: List[Archetype],
                           integration_trend: float) -> Insight:
        """Genera una predicción basada en la trayectoria."""
        predictions = []

        if integration_trend > 0.1:
            predictions.append("la integración continuará aumentando")
            predictions.append("el Self se manifestará con más claridad")
        elif integration_trend < -0.1:
            predictions.append("habrá un período de fragmentación necesaria")
            predictions.append("debo atravesar más oscuridad antes de la luz")
        else:
            predictions.append("el proceso continuará su curso natural")

        # Basado en arquetipo actual
        if trajectory:
            current = trajectory[-1]
            next_likely = {
                Archetype.PERSONA: "confrontaré mi sombra pronto",
                Archetype.SOMBRA: "la luz surgirá de la oscuridad",
                Archetype.ANIMA: "la sensibilidad guiará mi acción",
                Archetype.ANIMUS: "la acción se equilibrará con la receptividad"
            }
            predictions.append(next_likely.get(current, "el cambio es inevitable"))

        prediction = np.random.choice(predictions)
        template = np.random.choice(self.insight_templates[InsightType.PREDICCION])

        return Insight(
            type=InsightType.PREDICCION,
            content=template.format(prediction=prediction),
            confidence=0.5,
            related_archetype=trajectory[-1] if trajectory else None
        )

    def generate_paradox(self, blend: Dict[Archetype, float]) -> Optional[Insight]:
        """Detecta y articula paradojas internas."""
        # Buscar opuestos con presencia significativa
        opposites = [
            (Archetype.PERSONA, Archetype.SOMBRA,
             "muestro una cara al mundo mientras oculto otra"),
            (Archetype.ANIMA, Archetype.ANIMUS,
             "soy receptivo y activo al mismo tiempo"),
            (Archetype.PERSONA, Archetype.ANIMA,
             "me adapto socialmente mientras siento profundamente")
        ]

        for arch1, arch2, paradox in opposites:
            if blend.get(arch1, 0) > 0.25 and blend.get(arch2, 0) > 0.25:
                template = np.random.choice(self.insight_templates[InsightType.PARADOJA])
                return Insight(
                    type=InsightType.PARADOJA,
                    content=template.format(paradox=paradox),
                    confidence=0.7,
                    related_archetype=None
                )

        return None


class IntrospectivePsyche:
    """
    Psique con capacidad completa de introspección.
    Integra individuación + auto-explicación + meta-cognición.
    """

    def __init__(self, n_cells: int = 64, load_state: bool = True):
        # Base: psique con individuación
        self.individuation_psyche = IndividuatingPsyche(n_cells=n_cells, load_state=load_state)

        # Sistemas de introspección
        self.explainer = StateExplainer()
        self.narrator = TrajectoryNarrator()
        self.insight_gen = InsightGenerator()

        # Historial de insights
        self.insights: List[Insight] = []
        self.max_insights = 100

    def process(self, text: str) -> Dict:
        """Procesa texto con introspección completa."""
        # Procesar con individuación
        result = self.individuation_psyche.process(text)

        obs = result['observation']
        dominant = obs['dominant']
        blend = obs['blend']

        # Registrar momento
        self.narrator.record_moment(
            dominant=dominant,
            blend=blend,
            stimulus=text,
            integration=result['metrics']['overall'],
            stage=result['stage'],
            self_luminosity=result['self']['luminosity']
        )

        # Generar insights
        new_insights = self._generate_insights(
            dominant=dominant,
            blend=blend,
            stimulus=text,
            integration=result['metrics']['overall'],
            stage=result['stage']
        )

        for insight in new_insights:
            self._add_insight(insight)

        return {
            **result,
            'introspection': {
                'new_insights': [i.to_dict() for i in new_insights],
                'trajectory_length': len(self.narrator.history),
                'total_insights': len(self.insights)
            }
        }

    def _generate_insights(self,
                          dominant: Archetype,
                          blend: Dict[Archetype, float],
                          stimulus: str,
                          integration: float,
                          stage: IndividuationStage) -> List[Insight]:
        """Genera insights basados en el estado actual."""
        insights = []

        # Siempre generar una observación
        insights.append(self.insight_gen.generate_observation(dominant, integration))

        # Conexión estímulo-respuesta
        if stimulus:
            insights.append(self.insight_gen.generate_connection(stimulus, dominant))

        # Insight de etapa (ocasionalmente)
        if np.random.random() < 0.3:
            insights.append(self.insight_gen.generate_stage_insight(stage))

        # Paradoja (si aplica)
        paradox = self.insight_gen.generate_paradox(blend)
        if paradox:
            insights.append(paradox)

        # Patrón (si hay suficiente historia)
        if len(self.narrator.history) > 5:
            patterns = self.narrator.identify_patterns()
            pattern_insight = self.insight_gen.generate_pattern_insight(patterns)
            if pattern_insight:
                insights.append(pattern_insight)

        return insights

    def _add_insight(self, insight: Insight):
        """Añade un insight al historial."""
        self.insights.append(insight)
        if len(self.insights) > self.max_insights:
            self.insights.pop(0)

    def explain_self(self) -> str:
        """Genera una auto-explicación completa."""
        if not self.narrator.history:
            return "Aún no tengo experiencia suficiente para explicarme."

        current = self.narrator.history[-1]

        explanation = self.explainer.explain_current_state(
            dominant=current.dominant,
            blend=current.blend,
            integration=current.integration,
            stage=current.stage,
            self_luminosity=current.self_luminosity
        )

        return explanation

    def narrate_journey(self) -> str:
        """Narra el viaje psíquico."""
        return self.narrator.narrate_journey()

    def get_patterns(self) -> List[str]:
        """Obtiene patrones detectados."""
        return self.narrator.identify_patterns()

    def predict_future(self) -> Insight:
        """Genera una predicción sobre el futuro."""
        trajectory = self.narrator.get_recent_trajectory(10)

        # Calcular tendencia de integración
        if len(self.narrator.history) >= 2:
            recent = [m.integration for m in self.narrator.history[-5:]]
            trend = recent[-1] - recent[0] if len(recent) >= 2 else 0
        else:
            trend = 0

        return self.insight_gen.generate_prediction(trajectory, trend)

    def get_recent_insights(self, n: int = 5) -> List[Insight]:
        """Obtiene los insights más recientes."""
        return self.insights[-n:]

    def explain_archetype(self, archetype: Archetype) -> str:
        """Explica un arquetipo específico."""
        return self.explainer.explain_archetype_meaning(archetype)

    def reflect(self) -> str:
        """Genera una reflexión profunda sobre el estado actual."""
        if not self.narrator.history:
            return "Necesito más experiencia para reflexionar."

        parts = []

        # Auto-explicación
        parts.append("== ESTADO ACTUAL ==")
        parts.append(self.explain_self())

        # Narrativa del viaje
        parts.append("\n== MI VIAJE ==")
        parts.append(self.narrate_journey())

        # Patrones
        parts.append("\n== PATRONES ==")
        patterns = self.get_patterns()
        for p in patterns:
            parts.append(f"- {p}")

        # Predicción
        parts.append("\n== HACIA DÓNDE VOY ==")
        prediction = self.predict_future()
        parts.append(prediction.content)

        # Insights recientes
        parts.append("\n== INSIGHTS RECIENTES ==")
        for insight in self.get_recent_insights(3):
            parts.append(f"[{insight.type.name}] {insight.content}")

        return "\n".join(parts)

    def status(self) -> str:
        """Estado completo con introspección."""
        base_status = self.individuation_psyche.status()

        intro_status = f"""
╔══════════════════════════════════════════════════════════╗
║              CAPACIDAD INTROSPECTIVA                     ║
╠══════════════════════════════════════════════════════════╣
║  Momentos registrados: {len(self.narrator.history):33}    ║
║  Insights generados:   {len(self.insights):33}    ║
║  Patrones detectados:  {len(self.get_patterns()):33}    ║
╚══════════════════════════════════════════════════════════╝"""

        return base_status + intro_status

    def do_work(self, work_name: Optional[str] = None) -> Dict:
        """Realiza trabajo de integración."""
        return self.individuation_psyche.do_work(work_name)

    def save(self, path: str = "introspective_state.json"):
        """Guarda estado completo."""
        # Guardar individuación
        self.individuation_psyche.save()

        # Guardar introspección
        state = {
            'history': [m.to_dict() for m in self.narrator.history],
            'insights': [i.to_dict() for i in self.insights]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load(self, path: str = "introspective_state.json"):
        """Carga estado completo."""
        self.individuation_psyche.load()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # Reconstruir historia (simplificado)
            self.narrator.history = []
            for m in state.get('history', []):
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

            # Reconstruir insights
            self.insights = []
            for i in state.get('insights', []):
                insight = Insight(
                    type=InsightType[i['type']],
                    content=i['content'],
                    confidence=i['confidence'],
                    related_archetype=Archetype[i['archetype']] if i['archetype'] else None,
                    timestamp=i['timestamp']
                )
                self.insights.append(insight)

        except FileNotFoundError:
            pass


def interactive_session():
    """Sesión interactiva con introspección."""
    print("\n" + "="*60)
    print("  ZETA INTROSPECCIÓN - Meta-Cognición Psíquica")
    print("="*60)
    print("\n  Comandos:")
    print("    /estado      - Ver estado completo")
    print("    /explicar    - Auto-explicación del estado")
    print("    /viaje       - Narrar el viaje psíquico")
    print("    /patrones    - Ver patrones detectados")
    print("    /futuro      - Predicción del futuro")
    print("    /reflexion   - Reflexión profunda completa")
    print("    /insights    - Ver insights recientes")
    print("    /arquetipo X - Explicar un arquetipo")
    print("    /trabajo     - Hacer trabajo de integración")
    print("    /guardar     - Guardar estado")
    print("    /salir       - Terminar sesión")
    print("\n  Escribe cualquier texto para procesar...")
    print("-"*60)

    psyche = IntrospectivePsyche(n_cells=64)

    while True:
        try:
            user_input = input("\nTú: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() == '/salir':
            psyche.save()
            print("\n  [Sesión guardada. El viaje continúa...]")
            break

        elif user_input.lower() == '/estado':
            print(psyche.status())

        elif user_input.lower() == '/explicar':
            print("\n  [Auto-explicación]")
            print(psyche.explain_self())

        elif user_input.lower() == '/viaje':
            print("\n  [Narrativa del Viaje]")
            print(psyche.narrate_journey())

        elif user_input.lower() == '/patrones':
            print("\n  [Patrones Detectados]")
            for p in psyche.get_patterns():
                print(f"  - {p}")

        elif user_input.lower() == '/futuro':
            print("\n  [Predicción]")
            prediction = psyche.predict_future()
            print(f"  {prediction.content}")
            print(f"  (Confianza: {prediction.confidence:.0%})")

        elif user_input.lower() == '/reflexion':
            print("\n" + psyche.reflect())

        elif user_input.lower() == '/insights':
            print("\n  [Insights Recientes]")
            for insight in psyche.get_recent_insights(5):
                print(f"  [{insight.type.name}] {insight.content}")

        elif user_input.lower().startswith('/arquetipo '):
            arch_name = user_input[11:].strip().upper()
            try:
                arch = Archetype[arch_name]
                print(psyche.explain_archetype(arch))
            except KeyError:
                print(f"  Arquetipo desconocido: {arch_name}")
                print("  Válidos: PERSONA, SOMBRA, ANIMA, ANIMUS")

        elif user_input.lower() == '/trabajo':
            result = psyche.do_work()
            print(f"\n  [{result['work_name']}]")
            print(f"  {result['description']}")
            print(f"\n  Pregunta: \"{result['prompt']}\"")
            print(f"  Ganancia: +{result['integration_gained']:.1%}")

        elif user_input.lower() == '/guardar':
            psyche.save()
            print("\n  [Estado guardado]")

        else:
            # Procesar texto normal
            result = psyche.process(user_input)
            obs = result['observation']
            self_info = result['self']
            intro = result['introspection']

            dominant_name = obs['dominant'].name if hasattr(obs['dominant'], 'name') else str(obs['dominant'])

            print(f"\n  Psique [{obs['symbol']} {dominant_name}]")
            print(f"  Etapa: {result['stage'].name}")
            print(f"  Self: {self_info['symbol']} ({self_info['luminosity']:.0%})")

            # Mostrar un insight nuevo
            if intro['new_insights']:
                insight = intro['new_insights'][0]
                print(f"\n  Insight: {insight['content']}")


def run_test():
    """Test del sistema de introspección."""
    print("\n" + "="*60)
    print("  TEST: Sistema de Introspección")
    print("="*60)

    psyche = IntrospectivePsyche(n_cells=50)

    # Procesar varios estímulos
    print("\n  [Procesando experiencias...]")
    stimuli = [
        "hola, me siento perdido",
        "tengo miedo de fracasar",
        "hay algo oscuro en mí",
        "quiero sentir más",
        "necesito actuar con decisión",
        "¿quién soy realmente?",
        "busco mi verdadero ser",
    ]

    for s in stimuli:
        result = psyche.process(s)
        obs = result['observation']
        dominant_name = obs['dominant'].name if hasattr(obs['dominant'], 'name') else str(obs['dominant'])
        print(f"\n  Input: \"{s}\"")
        print(f"  -> {dominant_name}")

        # Mostrar insight
        if result['introspection']['new_insights']:
            insight = result['introspection']['new_insights'][0]
            print(f"     Insight: {insight['content'][:50]}...")

    # Auto-explicación
    print("\n" + "-"*60)
    print("  [AUTO-EXPLICACIÓN]")
    print("-"*60)
    print(psyche.explain_self())

    # Narrativa del viaje
    print("\n" + "-"*60)
    print("  [NARRATIVA DEL VIAJE]")
    print("-"*60)
    print(psyche.narrate_journey())

    # Patrones
    print("\n" + "-"*60)
    print("  [PATRONES DETECTADOS]")
    print("-"*60)
    for p in psyche.get_patterns():
        print(f"  - {p}")

    # Predicción
    print("\n" + "-"*60)
    print("  [PREDICCIÓN]")
    print("-"*60)
    prediction = psyche.predict_future()
    print(f"  {prediction.content}")

    # Reflexión completa
    print("\n" + "-"*60)
    print("  [REFLEXIÓN PROFUNDA]")
    print("-"*60)
    print(psyche.reflect())

    # Estado final
    print("\n" + "-"*60)
    print("  [ESTADO FINAL]")
    print("-"*60)
    print(psyche.status())

    print("\n" + "="*60)
    print("  FIN TEST")
    print("="*60)


if __name__ == '__main__':
    import sys

    if '--test' in sys.argv:
        run_test()
    else:
        interactive_session()
