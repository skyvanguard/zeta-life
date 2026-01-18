#!/usr/bin/env python
"""
ZetaIndividuation: Sistema de Individuación Junguiana

La individuación es el proceso central de desarrollo psicológico donde
el individuo integra todos los aspectos de la psique hacia un Self unificado.

Etapas:
1. PERSONA_IDENTIFICADA - Reconocer la máscara social
2. SOMBRA_CONFRONTADA - Integrar aspectos rechazados
3. ANIMA_ANIMUS_INTEGRADO - Equilibrar polaridades
4. SELF_EMERGENTE - Centro unificador trasciende opuestos

El Self no es un quinto arquetipo, sino el centro que emerge
cuando los cuatro están en equilibrio dinámico.
"""

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypedDict

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
    except:
        pass

import numpy as np
import torch
import torch.nn as nn

# Importar sistema base
from .zeta_psyche import Archetype, PsycheInterface, TetrahedralSpace, ZetaPsyche


class DefenseMechanism(TypedDict):
    """Type definition for defense mechanism entries."""
    description: str
    blocks: list[Archetype]
    strength: float


class IntegrationWorkEntry(TypedDict):
    """Type definition for integration work entries."""
    name: str
    target: Archetype | None
    description: str
    prompts: list[str]
    integration_potential: float


class IndividuationStage(Enum):
    """Etapas del proceso de individuación."""
    INCONSCIENTE = auto()        # Estado inicial, identificado con Persona
    CRISIS_PERSONA = auto()       # La máscara social se cuestiona
    ENCUENTRO_SOMBRA = auto()     # Confrontación con lo rechazado
    INTEGRACION_SOMBRA = auto()   # Aceptación de la Sombra
    ENCUENTRO_ANIMA = auto()      # Confrontación con lo contrasexual
    INTEGRACION_ANIMA = auto()    # Equilibrio de polaridades
    EMERGENCIA_SELF = auto()      # El centro comienza a manifestarse
    SELF_REALIZADO = auto()       # Individuación lograda (nunca permanente)


@dataclass
class IntegrationMetrics:
    """Métricas de integración por arquetipo."""
    persona_flexibility: float = 0.0    # Capacidad de adaptar la máscara
    shadow_acceptance: float = 0.0      # Aceptación de aspectos oscuros
    anima_connection: float = 0.0       # Conexión con lo emocional/receptivo
    animus_balance: float = 0.0         # Equilibrio de lo racional/activo
    self_coherence: float = 0.0         # Coherencia del centro unificador

    def overall_integration(self) -> float:
        """Integración total (0-1)."""
        return (self.persona_flexibility + self.shadow_acceptance +
                self.anima_connection + self.animus_balance +
                self.self_coherence) / 5.0

    def to_dict(self) -> dict:
        return {
            'persona_flexibility': self.persona_flexibility,
            'shadow_acceptance': self.shadow_acceptance,
            'anima_connection': self.anima_connection,
            'animus_balance': self.animus_balance,
            'self_coherence': self.self_coherence,
            'overall': self.overall_integration()
        }


@dataclass
class IndividuationEvent:
    """Evento significativo en el proceso de individuación."""
    timestamp: str
    stage: IndividuationStage
    trigger: str                # Qué lo provocó
    archetype_involved: Archetype
    insight: str               # Comprensión ganada
    integration_delta: float   # Cambio en integración
    resistance: float          # Nivel de resistencia encontrada


@dataclass
class SelfManifestation:
    """Manifestación del Self en un momento dado."""
    center: torch.Tensor       # Posición en espacio tetraédrico
    stability: float           # Qué tan estable es el centro
    luminosity: float          # Intensidad de la manifestación (0-1)
    symbol: str                # Símbolo asociado (mandala, etc)
    message: str | None     # Mensaje del Self si hay


class ResistanceSystem:
    """
    Sistema de defensas y resistencias psicológicas.
    Las resistencias protegen al ego pero dificultan la individuación.
    """

    DEFENSE_MECHANISMS: dict[str, DefenseMechanism] = {
        'negacion': {
            'description': 'Rechazar aspectos de la realidad',
            'blocks': [Archetype.SOMBRA],
            'strength': 0.8
        },
        'proyeccion': {
            'description': 'Atribuir a otros lo que es propio',
            'blocks': [Archetype.SOMBRA, Archetype.ANIMA],
            'strength': 0.7
        },
        'racionalizacion': {
            'description': 'Justificar con lógica lo emocional',
            'blocks': [Archetype.ANIMA],
            'strength': 0.6
        },
        'represion': {
            'description': 'Mantener inconsciente lo amenazante',
            'blocks': [Archetype.SOMBRA],
            'strength': 0.9
        },
        'identificacion_persona': {
            'description': 'Confundir la máscara con el yo',
            'blocks': [Archetype.PERSONA],
            'strength': 0.5
        },
        'inflacion': {
            'description': 'Identificarse con un arquetipo',
            'blocks': [Archetype.ANIMUS, Archetype.ANIMA],
            'strength': 0.7
        }
    }

    def __init__(self) -> None:
        self.active_defenses: dict[str, float] = {}
        self.defense_history: list[tuple[str, str, float]] = []

    def activate_defense(self, defense_name: str, intensity: float = 1.0) -> None:
        """Activa una defensa con cierta intensidad."""
        if defense_name in self.DEFENSE_MECHANISMS:
            current = self.active_defenses.get(defense_name, 0)
            self.active_defenses[defense_name] = min(1.0, current + intensity * 0.3)
            self.defense_history.append((
                datetime.now().isoformat(),
                defense_name,
                self.active_defenses[defense_name]
            ))

    def get_resistance_to(self, archetype: Archetype) -> float:
        """Calcula resistencia total hacia un arquetipo."""
        total_resistance = 0.0
        for defense_name, intensity in self.active_defenses.items():
            defense = self.DEFENSE_MECHANISMS[defense_name]
            if archetype in defense['blocks']:
                total_resistance += intensity * defense['strength']
        return min(1.0, total_resistance)

    def decay_defenses(self, rate: float = 0.05) -> None:
        """Las defensas decaen naturalmente con el tiempo."""
        for defense in list(self.active_defenses.keys()):
            self.active_defenses[defense] *= (1 - rate)
            if self.active_defenses[defense] < 0.01:
                del self.active_defenses[defense]

    def work_through(self, defense_name: str, effort: float = 0.1) -> bool:
        """Trabaja conscientemente una defensa. Retorna si hubo progreso."""
        if defense_name in self.active_defenses:
            self.active_defenses[defense_name] -= effort
            if self.active_defenses[defense_name] <= 0:
                del self.active_defenses[defense_name]
                return True
        return False


class IntegrationWork:
    """
    Trabajos de integración - ejercicios que promueven la individuación.
    Basados en técnicas junguianas (imaginación activa, análisis de sueños, etc).
    """

    WORKS: dict[str, IntegrationWorkEntry] = {
        'shadow_dialogue': {
            'name': 'Diálogo con la Sombra',
            'target': Archetype.SOMBRA,
            'description': 'Conversar con los aspectos rechazados',
            'prompts': [
                "¿Qué parte de ti niegas?",
                "¿Qué te avergüenza de ti mismo?",
                "¿Qué proyectas en otros?",
                "¿Qué talento has reprimido?"
            ],
            'integration_potential': 0.15
        },
        'persona_examination': {
            'name': 'Examen de la Persona',
            'target': Archetype.PERSONA,
            'description': 'Distinguir la máscara del verdadero yo',
            'prompts': [
                "¿Quién eres cuando nadie te ve?",
                "¿Qué roles juegas ante otros?",
                "¿Qué harías si no hubiera consecuencias sociales?",
                "¿Qué parte de ti muestras y cuál escondes?"
            ],
            'integration_potential': 0.12
        },
        'anima_encounter': {
            'name': 'Encuentro con el Anima',
            'target': Archetype.ANIMA,
            'description': 'Conectar con la sensibilidad y receptividad',
            'prompts': [
                "¿Qué te conmueve profundamente?",
                "¿Cuándo fue la última vez que lloraste?",
                "¿Qué belleza has ignorado?",
                "¿Qué intuición has desestimado?"
            ],
            'integration_potential': 0.13
        },
        'animus_balance': {
            'name': 'Equilibrio del Animus',
            'target': Archetype.ANIMUS,
            'description': 'Integrar la acción con la reflexión',
            'prompts': [
                "¿Cuándo actúas sin pensar?",
                "¿Qué decisiones evitas tomar?",
                "¿Dónde necesitas más coraje?",
                "¿Qué verdad incómoda debes enfrentar?"
            ],
            'integration_potential': 0.13
        },
        'mandala_meditation': {
            'name': 'Meditación del Mandala',
            'target': None,  # Apunta al Self
            'description': 'Contemplar el centro unificador',
            'prompts': [
                "Imagina un centro de quietud en ti...",
                "Observa cómo los opuestos se equilibran...",
                "Siente la totalidad de tu ser...",
                "Permite que el centro se revele..."
            ],
            'integration_potential': 0.20
        },
        'dream_analysis': {
            'name': 'Análisis de Sueños',
            'target': None,  # Puede revelar cualquier arquetipo
            'description': 'Interpretar mensajes del inconsciente',
            'prompts': [
                "¿Qué personajes aparecen en tus sueños?",
                "¿Qué emociones predominan?",
                "¿Qué símbolos se repiten?",
                "¿Qué te dice tu sueño más reciente?"
            ],
            'integration_potential': 0.18
        }
    }

    @classmethod
    def get_work_for_stage(cls, stage: IndividuationStage) -> list[str]:
        """Retorna trabajos apropiados para una etapa."""
        stage_works = {
            IndividuationStage.INCONSCIENTE: ['persona_examination'],
            IndividuationStage.CRISIS_PERSONA: ['persona_examination', 'shadow_dialogue'],
            IndividuationStage.ENCUENTRO_SOMBRA: ['shadow_dialogue', 'dream_analysis'],
            IndividuationStage.INTEGRACION_SOMBRA: ['shadow_dialogue', 'anima_encounter'],
            IndividuationStage.ENCUENTRO_ANIMA: ['anima_encounter', 'animus_balance'],
            IndividuationStage.INTEGRACION_ANIMA: ['anima_encounter', 'animus_balance', 'mandala_meditation'],
            IndividuationStage.EMERGENCIA_SELF: ['mandala_meditation', 'dream_analysis'],
            IndividuationStage.SELF_REALIZADO: ['mandala_meditation']
        }
        return stage_works.get(stage, ['dream_analysis'])


class SelfSystem:
    """
    Sistema del Self - el centro unificador de la psique.

    El Self no es un arquetipo más, sino el principio organizador
    que emerge cuando hay suficiente integración.
    """

    SELF_SYMBOLS = ['☉', '◎', '✦', '⊙', '❂', '✧', '◉', '⚹']

    def __init__(self) -> None:
        self.manifestations: list[SelfManifestation] = []
        self.total_luminosity: float = 0.0
        self.center_history: list[torch.Tensor] = []

    def compute_self_center(self, psyche_state: dict) -> torch.Tensor:
        """
        Calcula el centro del Self basado en el estado de la psique.
        El Self emerge en el centro cuando hay equilibrio.
        """
        distribution = psyche_state.get('population_distribution', None)

        # Centro tetraédrico puro
        perfect_center = torch.tensor([0.25, 0.25, 0.25, 0.25])

        # Estado actual - manejar tanto Tensor como dict
        if distribution is None:
            return perfect_center
        elif isinstance(distribution, torch.Tensor):
            current = distribution.clone()
        else:
            current = torch.tensor([
                distribution.get(Archetype.PERSONA, 0.25),
                distribution.get(Archetype.SOMBRA, 0.25),
                distribution.get(Archetype.ANIMA, 0.25),
                distribution.get(Archetype.ANIMUS, 0.25)
            ])

        # El Self está más cerca del centro cuanto más equilibrado
        return current

    def compute_luminosity(self, metrics: IntegrationMetrics) -> float:
        """
        La luminosidad del Self depende de la integración total.
        Es una medida de cuán "presente" está el Self.
        """
        base = metrics.overall_integration()

        # Bonus por equilibrio entre arquetipos
        values = [
            metrics.persona_flexibility,
            metrics.shadow_acceptance,
            metrics.anima_connection,
            metrics.animus_balance
        ]
        variance = float(np.var(values))
        balance_bonus = max(0.0, 0.2 - variance)

        # Coherencia del Self amplifica
        coherence_factor = 1 + metrics.self_coherence

        luminosity = (base + balance_bonus) * coherence_factor
        return min(1.0, luminosity)

    def compute_stability(self) -> float:
        """Estabilidad basada en historia del centro."""
        if len(self.center_history) < 3:
            return 0.0

        recent = self.center_history[-10:]
        variations = []
        for i in range(1, len(recent)):
            diff = torch.norm(recent[i] - recent[i-1]).item()
            variations.append(diff)

        avg_variation = np.mean(variations) if variations else 0.5
        stability = max(0, 1 - avg_variation * 5)
        return stability

    def manifest(self, psyche_state: dict, metrics: IntegrationMetrics) -> SelfManifestation:
        """Genera una manifestación del Self."""
        center = self.compute_self_center(psyche_state)
        self.center_history.append(center)

        luminosity = self.compute_luminosity(metrics)
        stability = self.compute_stability()

        # Seleccionar símbolo según luminosidad
        symbol_idx = min(len(self.SELF_SYMBOLS) - 1,
                        int(luminosity * len(self.SELF_SYMBOLS)))
        symbol = self.SELF_SYMBOLS[symbol_idx]

        # Mensaje del Self (solo si hay suficiente luminosidad)
        message = None
        if luminosity > 0.5:
            message = self._generate_self_message(luminosity, stability)

        manifestation = SelfManifestation(
            center=center,
            stability=stability,
            luminosity=luminosity,
            symbol=symbol,
            message=message
        )

        self.manifestations.append(manifestation)
        self.total_luminosity = luminosity

        return manifestation

    def _generate_self_message(self, luminosity: float, stability: float) -> str:
        """Genera un mensaje desde el Self."""
        messages_by_state = {
            (True, True): [   # Alta luminosidad, alta estabilidad
                "Eres totalidad en movimiento.",
                "Los opuestos danzan en ti.",
                "El centro sostiene todo.",
                "Eres más de lo que crees ser."
            ],
            (True, False): [  # Alta luminosidad, baja estabilidad
                "El equilibrio es un momento, no un estado.",
                "Fluye con los cambios.",
                "La transformación es el camino.",
                "No te aferres, solo observa."
            ],
            (False, True): [  # Baja luminosidad, alta estabilidad
                "Hay más luz esperando emerger.",
                "La semilla necesita oscuridad para germinar.",
                "La paciencia es también acción.",
                "El centro siempre está ahí."
            ],
            (False, False): [ # Baja luminosidad, baja estabilidad
                "El caos precede al orden.",
                "Confía en el proceso.",
                "Cada paso cuenta.",
                "La búsqueda ya es el encuentro."
            ]
        }

        high_lum = luminosity > 0.6
        high_stab = stability > 0.5
        messages = messages_by_state[(high_lum, high_stab)]

        return str(np.random.choice(messages))


class IndividuationProcess:
    """
    El proceso completo de individuación.
    Integra psique, resistencias, trabajos y emergencia del Self.
    """

    def __init__(self, psyche: ZetaPsyche) -> None:
        self.psyche = psyche
        self.interface = PsycheInterface(psyche)  # Interfaz para procesar texto
        self.stage = IndividuationStage.INCONSCIENTE
        self.metrics = IntegrationMetrics()
        self.resistance = ResistanceSystem()
        self.self_system = SelfSystem()
        self.events: list[IndividuationEvent] = []
        self.session_count = 0

        # Umbrales para transiciones de etapa
        self.stage_thresholds = {
            IndividuationStage.INCONSCIENTE: 0.0,
            IndividuationStage.CRISIS_PERSONA: 0.1,
            IndividuationStage.ENCUENTRO_SOMBRA: 0.2,
            IndividuationStage.INTEGRACION_SOMBRA: 0.35,
            IndividuationStage.ENCUENTRO_ANIMA: 0.45,
            IndividuationStage.INTEGRACION_ANIMA: 0.6,
            IndividuationStage.EMERGENCIA_SELF: 0.75,
            IndividuationStage.SELF_REALIZADO: 0.9
        }

    def update_stage(self) -> None:
        """Actualiza la etapa basado en integración."""
        integration = self.metrics.overall_integration()

        for stage in reversed(list(IndividuationStage)):
            threshold = self.stage_thresholds.get(stage, 0)
            if integration >= threshold:
                if stage != self.stage:
                    old_stage = self.stage
                    self.stage = stage
                    self._record_stage_transition(old_stage, stage)
                break

    def _record_stage_transition(self, old: IndividuationStage, new: IndividuationStage) -> None:
        """Registra una transición de etapa."""
        # Determinar arquetipo involucrado
        archetype_by_stage = {
            IndividuationStage.CRISIS_PERSONA: Archetype.PERSONA,
            IndividuationStage.ENCUENTRO_SOMBRA: Archetype.SOMBRA,
            IndividuationStage.INTEGRACION_SOMBRA: Archetype.SOMBRA,
            IndividuationStage.ENCUENTRO_ANIMA: Archetype.ANIMA,
            IndividuationStage.INTEGRACION_ANIMA: Archetype.ANIMUS,
            IndividuationStage.EMERGENCIA_SELF: Archetype.PERSONA,  # Todos
            IndividuationStage.SELF_REALIZADO: Archetype.PERSONA    # Todos
        }

        insights = {
            IndividuationStage.CRISIS_PERSONA: "La máscara no soy yo.",
            IndividuationStage.ENCUENTRO_SOMBRA: "Hay partes de mí que no conozco.",
            IndividuationStage.INTEGRACION_SOMBRA: "Mi sombra también es yo.",
            IndividuationStage.ENCUENTRO_ANIMA: "Hay otra forma de ser en mí.",
            IndividuationStage.INTEGRACION_ANIMA: "Los opuestos pueden coexistir.",
            IndividuationStage.EMERGENCIA_SELF: "Hay un centro que sostiene todo.",
            IndividuationStage.SELF_REALIZADO: "Soy totalidad en proceso."
        }

        event = IndividuationEvent(
            timestamp=datetime.now().isoformat(),
            stage=new,
            trigger=f"Transición desde {old.name}",
            archetype_involved=archetype_by_stage.get(new, Archetype.PERSONA),
            insight=insights.get(new, "Algo ha cambiado."),
            integration_delta=self.stage_thresholds[new] - self.stage_thresholds[old],
            resistance=self._total_resistance()
        )
        self.events.append(event)

    def _total_resistance(self) -> float:
        """Resistencia total actual."""
        total: float = 0.0
        for arch in Archetype:
            total += self.resistance.get_resistance_to(arch)
        return total / len(Archetype)

    def process_stimulus(self, stimulus: str) -> dict:
        """
        Procesa un estímulo a través del lente de individuación.
        Retorna estado actualizado y posible manifestación del Self.
        """
        self.session_count += 1

        # Procesar en la psique base usando la interfaz
        psyche_response = self.interface.process_input(stimulus)
        obs = self.psyche.observe_self()

        # Añadir símbolo de la respuesta a la observación
        obs['symbol'] = psyche_response.get('symbol', '✧')

        # Detectar activaciones arquetípicas
        dominant = obs['dominant']

        # ¿Hay resistencia a este arquetipo?
        resistance = self.resistance.get_resistance_to(dominant)

        # Actualizar métricas de integración
        self._update_metrics(dominant, resistance, stimulus)

        # Decay natural de defensas
        self.resistance.decay_defenses()

        # Actualizar etapa
        self.update_stage()

        # Manifestación del Self
        self_manifestation = self.self_system.manifest(obs, self.metrics)

        return {
            'psyche_response': psyche_response,
            'observation': obs,
            'stage': self.stage,
            'metrics': self.metrics.to_dict(),
            'resistance': self._total_resistance(),
            'self': {
                'symbol': self_manifestation.symbol,
                'luminosity': self_manifestation.luminosity,
                'stability': self_manifestation.stability,
                'message': self_manifestation.message
            }
        }

    def _update_metrics(self, dominant: Archetype, resistance: float, stimulus: str) -> None:
        """Actualiza métricas basado en la activación."""
        # Factor de aprendizaje (reducido por resistencia)
        learning_rate = 0.05 * (1 - resistance * 0.5)

        # Cada arquetipo activado aumenta su integración
        if dominant == Archetype.PERSONA:
            self.metrics.persona_flexibility += learning_rate
        elif dominant == Archetype.SOMBRA:
            self.metrics.shadow_acceptance += learning_rate
        elif dominant == Archetype.ANIMA:
            self.metrics.anima_connection += learning_rate
        elif dominant == Archetype.ANIMUS:
            self.metrics.animus_balance += learning_rate

        # La coherencia del Self aumenta con equilibrio
        values = [
            self.metrics.persona_flexibility,
            self.metrics.shadow_acceptance,
            self.metrics.anima_connection,
            self.metrics.animus_balance
        ]
        min_val = min(values)
        self.metrics.self_coherence = min_val  # El más bajo limita al Self

        # Limitar todas las métricas a 1.0
        self.metrics.persona_flexibility = min(1.0, self.metrics.persona_flexibility)
        self.metrics.shadow_acceptance = min(1.0, self.metrics.shadow_acceptance)
        self.metrics.anima_connection = min(1.0, self.metrics.anima_connection)
        self.metrics.animus_balance = min(1.0, self.metrics.animus_balance)
        self.metrics.self_coherence = min(1.0, self.metrics.self_coherence)

    def do_integration_work(self, work_name: str) -> dict:
        """
        Realiza un trabajo de integración.
        Retorna resultados y prompt para reflexión.
        """
        if work_name not in IntegrationWork.WORKS:
            return {'error': f'Trabajo desconocido: {work_name}'}

        work: IntegrationWorkEntry = IntegrationWork.WORKS[work_name]
        target: Archetype | None = work['target']

        # Verificar resistencias
        resistance: float = 0.0
        if target:
            resistance = self.resistance.get_resistance_to(target)
            if resistance > 0.5:
                # Activar defensa
                self.resistance.activate_defense('negacion', 0.2)

        # Calcular integración ganada
        potential = work['integration_potential']
        actual_gain = potential * (1 - resistance * 0.7)

        # Aplicar al arquetipo objetivo
        if target == Archetype.PERSONA:
            self.metrics.persona_flexibility += actual_gain
        elif target == Archetype.SOMBRA:
            self.metrics.shadow_acceptance += actual_gain
        elif target == Archetype.ANIMA:
            self.metrics.anima_connection += actual_gain
        elif target == Archetype.ANIMUS:
            self.metrics.animus_balance += actual_gain
        else:  # Trabajo del Self
            # Aumenta todas un poco
            boost = actual_gain / 4
            self.metrics.persona_flexibility += boost
            self.metrics.shadow_acceptance += boost
            self.metrics.anima_connection += boost
            self.metrics.animus_balance += boost

        # Actualizar coherencia del Self
        values = [
            self.metrics.persona_flexibility,
            self.metrics.shadow_acceptance,
            self.metrics.anima_connection,
            self.metrics.animus_balance
        ]
        self.metrics.self_coherence = min(values)

        # Actualizar etapa
        self.update_stage()

        # Seleccionar prompt para reflexión
        prompt = str(np.random.choice(work['prompts']))

        return {
            'work_name': work['name'],
            'description': work['description'],
            'prompt': prompt,
            'integration_gained': actual_gain,
            'resistance_encountered': resistance,
            'new_stage': self.stage,
            'metrics': self.metrics.to_dict()
        }

    def get_recommended_work(self) -> str:
        """Recomienda un trabajo apropiado para la etapa actual."""
        works = IntegrationWork.get_work_for_stage(self.stage)

        # Elegir el que tenga menor resistencia
        min_resistance = float('inf')
        best_work = works[0]

        for work_name in works:
            work: IntegrationWorkEntry = IntegrationWork.WORKS[work_name]
            target: Archetype | None = work['target']
            if target:
                res = self.resistance.get_resistance_to(target)
                if res < min_resistance:
                    min_resistance = res
                    best_work = work_name
            else:
                # Trabajos del Self siempre son buenos
                best_work = work_name
                break

        return best_work

    def get_status_report(self) -> str:
        """Genera un reporte del estado de individuación."""
        obs = self.psyche.observe_self()
        self_man = self.self_system.manifest(obs, self.metrics)

        report = f"""
╔══════════════════════════════════════════════════════════╗
║             ESTADO DE INDIVIDUACIÓN                      ║
╠══════════════════════════════════════════════════════════╣
║  Etapa: {self.stage.name:40}    ║
║  Sesiones: {self.session_count:37}    ║
╠══════════════════════════════════════════════════════════╣
║  MÉTRICAS DE INTEGRACIÓN                                 ║
║  ────────────────────────────────────────────────────    ║
║  Persona (Flexibilidad):  {self._bar(self.metrics.persona_flexibility)}  {self.metrics.persona_flexibility:.0%}  ║
║  Sombra (Aceptación):     {self._bar(self.metrics.shadow_acceptance)}  {self.metrics.shadow_acceptance:.0%}  ║
║  Anima (Conexión):        {self._bar(self.metrics.anima_connection)}  {self.metrics.anima_connection:.0%}  ║
║  Animus (Equilibrio):     {self._bar(self.metrics.animus_balance)}  {self.metrics.animus_balance:.0%}  ║
║  ────────────────────────────────────────────────────    ║
║  Self (Coherencia):       {self._bar(self.metrics.self_coherence)}  {self.metrics.self_coherence:.0%}  ║
║  INTEGRACIÓN TOTAL:       {self._bar(self.metrics.overall_integration())}  {self.metrics.overall_integration():.0%}  ║
╠══════════════════════════════════════════════════════════╣
║  MANIFESTACIÓN DEL SELF                                  ║
║  Símbolo: {self_man.symbol}  Luminosidad: {self_man.luminosity:.0%}  Estabilidad: {self_man.stability:.0%}   ║"""

        if self_man.message:
            report += f"""
║  Mensaje: "{self_man.message[:45]}"   ║"""

        report += f"""
╠══════════════════════════════════════════════════════════╣
║  Resistencias activas: {len(self.resistance.active_defenses):32}    ║"""

        for defense, intensity in list(self.resistance.active_defenses.items())[:3]:
            report += f"""
║    - {defense}: {intensity:.0%}                                       ║"""[:62] + "║"

        recommended = self.get_recommended_work()
        work: IntegrationWorkEntry = IntegrationWork.WORKS[recommended]
        report += f"""
╠══════════════════════════════════════════════════════════╣
║  Trabajo recomendado: {work['name'][:35]:35}   ║
╚══════════════════════════════════════════════════════════╝"""

        return report

    def _bar(self, value: float, width: int = 15) -> str:
        """Genera una barra de progreso."""
        filled = int(value * width)
        return '█' * filled + '░' * (width - filled)

    def save(self, path: str = "individuation_state.json") -> None:
        """Guarda el estado de individuación."""
        state = {
            'stage': self.stage.name,
            'metrics': self.metrics.to_dict(),
            'session_count': self.session_count,
            'resistance': dict(self.resistance.active_defenses),
            'events': [
                {
                    'timestamp': e.timestamp,
                    'stage': e.stage.name,
                    'trigger': e.trigger,
                    'archetype': e.archetype_involved.name,
                    'insight': e.insight,
                    'integration_delta': e.integration_delta
                }
                for e in self.events
            ]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load(self, path: str = "individuation_state.json") -> bool:
        """Carga el estado de individuación."""
        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)

            self.stage = IndividuationStage[state['stage']]
            self.session_count = state['session_count']

            m = state['metrics']
            self.metrics = IntegrationMetrics(
                persona_flexibility=m['persona_flexibility'],
                shadow_acceptance=m['shadow_acceptance'],
                anima_connection=m['anima_connection'],
                animus_balance=m['animus_balance'],
                self_coherence=m['self_coherence']
            )

            self.resistance.active_defenses = state.get('resistance', {})

            return True
        except FileNotFoundError:
            return False


class IndividuatingPsyche:
    """
    Psique completa con proceso de individuación integrado.
    Interfaz unificada para todo el sistema.
    """

    def __init__(self, n_cells: int = 64, load_state: bool = True) -> None:
        self.psyche = ZetaPsyche(n_cells=n_cells)
        self.individuation = IndividuationProcess(self.psyche)

        if load_state:
            self.individuation.load()

        # Warmup
        for _ in range(20):
            self.psyche.step()

    def process(self, text: str) -> dict:
        """Procesa texto y retorna respuesta con contexto de individuación."""
        result = self.individuation.process_stimulus(text)
        return result

    def do_work(self, work_name: str | None = None) -> dict:
        """Realiza trabajo de integración."""
        if work_name is None:
            work_name = self.individuation.get_recommended_work()
        return self.individuation.do_integration_work(work_name)

    def status(self) -> str:
        """Retorna estado completo."""
        return self.individuation.get_status_report()

    def save(self) -> None:
        """Guarda estado."""
        self.individuation.save()

    def load(self) -> None:
        """Carga estado."""
        self.individuation.load()


def visualize_individuation(process: IndividuationProcess, save_path: str = "individuation_progress.png") -> str | None:
    """Visualiza el progreso de individuación."""
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Proceso de Individuación', fontsize=16, fontweight='bold')

        # 1. Métricas de integración (radar chart simulado como barras)
        ax1 = axes[0, 0]
        categories = ['Persona\n(Flexibilidad)', 'Sombra\n(Aceptación)',
                     'Anima\n(Conexión)', 'Animus\n(Equilibrio)', 'Self\n(Coherencia)']
        values = [
            process.metrics.persona_flexibility,
            process.metrics.shadow_acceptance,
            process.metrics.anima_connection,
            process.metrics.animus_balance,
            process.metrics.self_coherence
        ]
        colors = ['#4a86e8', '#6a0dad', '#e84a5f', '#2eb872', '#ffd700']
        bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Nivel de Integración')
        ax1.set_title('Métricas de Integración')
        ax1.axhline(y=process.metrics.overall_integration(), color='red',
                   linestyle='--', label=f'Total: {process.metrics.overall_integration():.0%}')
        ax1.legend()

        # Añadir valores sobre las barras
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', fontsize=10)

        # 2. Progreso por etapas
        ax2 = axes[0, 1]
        stages = list(IndividuationStage)
        stage_names = [s.name.replace('_', '\n') for s in stages]
        current_idx = stages.index(process.stage)

        colors2 = ['#90EE90' if i <= current_idx else '#D3D3D3' for i in range(len(stages))]
        ax2.barh(stage_names, [1]*len(stages), color=colors2, edgecolor='black')
        ax2.set_xlim(0, 1.2)
        ax2.set_title('Progreso por Etapas')
        ax2.axhline(y=current_idx, color='red', linewidth=3, alpha=0.5)
        ax2.text(1.05, current_idx, '← ACTUAL', va='center', fontsize=10, color='red')

        # 3. Resistencias
        ax3 = axes[1, 0]
        if process.resistance.active_defenses:
            defenses = list(process.resistance.active_defenses.keys())
            intensities = list(process.resistance.active_defenses.values())
            ax3.barh(defenses, intensities, color='#ff6b6b', edgecolor='black')
            ax3.set_xlim(0, 1)
            ax3.set_xlabel('Intensidad')
        else:
            ax3.text(0.5, 0.5, 'Sin resistencias activas', ha='center', va='center',
                    fontsize=14, transform=ax3.transAxes)
        ax3.set_title('Resistencias Activas')

        # 4. Manifestación del Self
        ax4 = axes[1, 1]
        obs = process.psyche.observe_self()
        self_man = process.self_system.manifest(obs, process.metrics)

        # Dibujar círculo del Self
        circle = plt.Circle((0.5, 0.5), 0.3 * self_man.luminosity + 0.1,
                           color='gold', alpha=0.6)
        ax4.add_patch(circle)

        # Símbolo central
        ax4.text(0.5, 0.5, self_man.symbol, fontsize=50 * self_man.luminosity + 20,
                ha='center', va='center')

        # Info
        ax4.text(0.5, 0.1, f'Luminosidad: {self_man.luminosity:.0%}',
                ha='center', fontsize=12)
        ax4.text(0.5, 0.02, f'Estabilidad: {self_man.stability:.0%}',
                ha='center', fontsize=12)

        if self_man.message:
            ax4.text(0.5, 0.9, f'"{self_man.message}"', ha='center',
                    fontsize=10, style='italic', wrap=True)

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title('Manifestación del Self')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path
    except ImportError:
        return None


def interactive_session() -> None:
    """Sesión interactiva de individuación."""
    print("\n" + "="*60)
    print("  ZETA INDIVIDUACIÓN - Proceso de Desarrollo Psicológico")
    print("="*60)
    print("\n  Comandos:")
    print("    /estado    - Ver estado de individuación")
    print("    /trabajo   - Hacer trabajo de integración recomendado")
    print("    /trabajos  - Ver trabajos disponibles")
    print("    /hacer X   - Hacer trabajo específico")
    print("    /visual    - Generar visualización")
    print("    /guardar   - Guardar progreso")
    print("    /salir     - Terminar sesión")
    print("\n  Escribe cualquier texto para procesar...")
    print("-"*60)

    psyche = IndividuatingPsyche(n_cells=64)

    while True:
        try:
            user_input = input("\nTú: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() == '/salir':
            psyche.save()
            print("\n  [Sesión guardada. Hasta pronto.]")
            break

        elif user_input.lower() == '/estado':
            print(psyche.status())

        elif user_input.lower() == '/trabajo':
            result = psyche.do_work()
            print(f"\n  [{result['work_name']}]")
            print(f"  {result['description']}")
            print("\n  Pregunta para reflexionar:")
            print(f"  \"{result['prompt']}\"")
            print(f"\n  Integración ganada: +{result['integration_gained']:.1%}")
            if result['resistance_encountered'] > 0:
                print(f"  Resistencia encontrada: {result['resistance_encountered']:.0%}")

        elif user_input.lower() == '/trabajos':
            print("\n  TRABAJOS DE INTEGRACIÓN DISPONIBLES:")
            print("  " + "-"*40)
            for name, work_entry in IntegrationWork.WORKS.items():
                target_name = work_entry['target'].name if work_entry['target'] else 'Self'
                print(f"  - {name}: {work_entry['name']} ({target_name})")

        elif user_input.lower().startswith('/hacer '):
            work_name = user_input[7:].strip()
            result = psyche.do_work(work_name)
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"\n  [{result['work_name']}]")
                print(f"  {result['description']}")
                print("\n  Pregunta para reflexionar:")
                print(f"  \"{result['prompt']}\"")
                print(f"\n  Integración ganada: +{result['integration_gained']:.1%}")

        elif user_input.lower() == '/visual':
            path = visualize_individuation(psyche.individuation)
            if path:
                print(f"\n  Visualización guardada: {path}")
            else:
                print("\n  Error: matplotlib no disponible")

        elif user_input.lower() == '/guardar':
            psyche.save()
            print("\n  [Estado guardado]")

        else:
            # Procesar texto normal
            result = psyche.process(user_input)
            obs = result['observation']
            self_info = result['self']

            dominant_name = obs['dominant'].name if hasattr(obs['dominant'], 'name') else str(obs['dominant'])
            print(f"\n  Psique [{obs['symbol']} {dominant_name}]")
            print(f"  Etapa: {result['stage'].name}")
            print(f"  Self: {self_info['symbol']} (Luminosidad: {self_info['luminosity']:.0%})")

            if self_info['message']:
                print(f"  Mensaje del Self: \"{self_info['message']}\"")


def run_test() -> None:
    """Test del sistema de individuación."""
    print("\n" + "="*60)
    print("  TEST: Sistema de Individuación")
    print("="*60)

    psyche = IndividuatingPsyche(n_cells=50)

    # Estado inicial
    print("\n  [Estado Inicial]")
    print(psyche.status())

    # Procesar varios estímulos
    print("\n  [Procesando estímulos...]")
    stimuli = [
        "tengo miedo de mostrar quién soy realmente",
        "hay una parte de mí que odio",
        "siento una profunda tristeza",
        "necesito ser más fuerte",
        "quiero entender mis sueños",
    ]

    for s in stimuli:
        result = psyche.process(s)
        print(f"\n  Input: \"{s}\"")
        dominant = result['observation']['dominant']
        dominant_name = dominant.name if hasattr(dominant, 'name') else str(dominant)
        print(f"  -> {dominant_name} | Etapa: {result['stage'].name}")
        print(f"    Self: {result['self']['symbol']} ({result['self']['luminosity']:.0%})")

    # Hacer trabajos de integración
    print("\n  [Realizando trabajos de integración...]")
    for _ in range(3):
        result = psyche.do_work()
        print(f"\n  Trabajo: {result['work_name']}")
        print(f"  Prompt: \"{result['prompt']}\"")
        print(f"  Ganancia: +{result['integration_gained']:.1%}")

    # Estado final
    print("\n  [Estado Final]")
    print(psyche.status())

    # Visualización
    path = visualize_individuation(psyche.individuation, "individuation_test.png")
    if path:
        print(f"\n  Visualización guardada: {path}")

    # Guardar estado
    psyche.save()

    print("\n" + "="*60)
    print("  FIN TEST")
    print("="*60)


if __name__ == '__main__':
    import sys

    if '--test' in sys.argv:
        run_test()
    else:
        interactive_session()
