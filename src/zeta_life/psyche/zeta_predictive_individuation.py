#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZetaPredictiveIndividuation: Integración de Predicción con Individuación

Combina:
- Sistema Predictivo (3 niveles de predicción)
- Sistema de Individuación (arquetipos de Jung)
- Emergencia del Self

La consciencia emerge cuando:
1. El sistema predice bien sus propios errores (meta-cognición)
2. Los arquetipos están integrados (individuación)
3. El Self se manifiesta como centro unificador
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# Importar sistemas base
from .zeta_psyche import ZetaPsyche, Archetype, PsycheInterface
from .zeta_predictive import (
    ZetaPredictivePsyche,
    PredictiveConsciousnessMetrics,
)
from .zeta_individuation import (
    IndividuationProcess,
    IndividuationStage,
    IntegrationMetrics,
    IntegrationWork,
    SelfSystem,
)


@dataclass
class FullConsciousnessMetrics:
    """Métricas completas de consciencia."""
    # Métricas predictivas
    awareness: float = 0.0
    calibration: float = 0.0
    uncertainty_awareness: float = 0.0
    predictive_depth: float = 0.0

    # Métricas de individuación
    persona_flexibility: float = 0.0
    shadow_acceptance: float = 0.0
    anima_connection: float = 0.0
    animus_balance: float = 0.0
    self_coherence: float = 0.0

    # Métricas base
    integration: float = 0.0
    stability: float = 0.0
    self_reference: float = 0.0

    # Manifestación del Self
    self_luminosity: float = 0.0
    self_stability: float = 0.0

    def predictive_index(self) -> float:
        """Índice de consciencia predictiva."""
        return (
            0.30 * self.awareness +
            0.30 * self.calibration +
            0.20 * self.uncertainty_awareness +
            0.20 * self.predictive_depth
        )

    def individuation_index(self) -> float:
        """Índice de individuación."""
        return (
            self.persona_flexibility +
            self.shadow_acceptance +
            self.anima_connection +
            self.animus_balance +
            self.self_coherence
        ) / 5.0

    def total_consciousness(self) -> float:
        """Índice total de consciencia."""
        return (
            0.35 * self.predictive_index() +
            0.35 * self.individuation_index() +
            0.15 * self.self_luminosity +
            0.10 * self.integration +
            0.05 * self.stability
        )

    def to_dict(self) -> Dict:
        return {
            'predictive': {
                'awareness': self.awareness,
                'calibration': self.calibration,
                'uncertainty_awareness': self.uncertainty_awareness,
                'predictive_depth': self.predictive_depth,
                'index': self.predictive_index(),
            },
            'individuation': {
                'persona_flexibility': self.persona_flexibility,
                'shadow_acceptance': self.shadow_acceptance,
                'anima_connection': self.anima_connection,
                'animus_balance': self.animus_balance,
                'self_coherence': self.self_coherence,
                'index': self.individuation_index(),
            },
            'self': {
                'luminosity': self.self_luminosity,
                'stability': self.self_stability,
            },
            'base': {
                'integration': self.integration,
                'stability': self.stability,
                'self_reference': self.self_reference,
            },
            'total_consciousness': self.total_consciousness(),
        }


class PredictiveIndividuation:
    """
    Proceso de individuación enriquecido con métricas predictivas.

    La meta-cognición (awareness, calibration) acelera la individuación
    porque saber lo que no sabemos es clave para el desarrollo.
    """

    def __init__(self, predictive_psyche: ZetaPredictivePsyche):
        self.predictive = predictive_psyche
        self.psyche = predictive_psyche.psyche

        # Proceso de individuación usando la psique base
        self.individuation = IndividuationProcess(self.psyche)

        # Sistema del Self
        self.self_system = SelfSystem()

        # Interfaz de texto
        self.interface = PsycheInterface(self.psyche)

        # Métricas completas
        self.full_metrics = FullConsciousnessMetrics()

        # Historial
        self.consciousness_history = []
        self.session_count = 0

    def process(self, text: str, n_steps: int = 10) -> Dict:
        """
        Procesa texto con predicción + individuación.

        Args:
            text: Texto de entrada
            n_steps: Pasos de procesamiento

        Returns:
            Dict con estado completo
        """
        self.session_count += 1

        # Convertir texto a estímulo
        stimulus = self._text_to_stimulus(text)

        # Procesar con sistema predictivo
        for _ in range(n_steps):
            pred_result = self.predictive.step(stimulus)

        # Obtener métricas predictivas
        pred_metrics = self.predictive.metrics

        # Procesar con sistema de individuación
        indiv_result = self.individuation.process_stimulus(text)

        # Actualizar métricas completas
        self._update_full_metrics(pred_metrics, indiv_result)

        # Manifestar Self
        obs = self.psyche.observe_self()
        self_manifestation = self.self_system.manifest(obs, self.individuation.metrics)

        # Registrar
        consciousness = self.full_metrics.total_consciousness()
        self.consciousness_history.append(consciousness)

        return {
            'text': text,
            'symbol': indiv_result['psyche_response'].get('symbol', '?'),
            'dominant': obs['dominant'].name,
            'stage': self.individuation.stage.name,
            'consciousness': consciousness,
            'metrics': self.full_metrics.to_dict(),
            'self': {
                'symbol': self_manifestation.symbol,
                'luminosity': self_manifestation.luminosity,
                'message': self_manifestation.message,
            },
            'observation': obs,
        }

    def _text_to_stimulus(self, text: str) -> torch.Tensor:
        """Convierte texto a vector de estímulo."""
        text = text.lower()

        # Mapeo básico
        word_weights = {
            # PERSONA
            'hola': [0.5, 0.1, 0.2, 0.2],
            'trabajo': [0.6, 0.1, 0.1, 0.2],
            'social': [0.7, 0.1, 0.1, 0.1],

            # SOMBRA
            'miedo': [0.1, 0.7, 0.1, 0.1],
            'odio': [0.1, 0.6, 0.1, 0.2],
            'oscuridad': [0.1, 0.7, 0.1, 0.1],
            'tristeza': [0.1, 0.5, 0.3, 0.1],

            # ANIMA
            'amor': [0.1, 0.1, 0.7, 0.1],
            'belleza': [0.2, 0.1, 0.6, 0.1],
            'sentir': [0.1, 0.2, 0.6, 0.1],
            'intuicion': [0.1, 0.1, 0.7, 0.1],

            # ANIMUS
            'pensar': [0.1, 0.1, 0.1, 0.7],
            'logica': [0.1, 0.1, 0.1, 0.7],
            'razon': [0.1, 0.1, 0.1, 0.7],
            'decision': [0.2, 0.1, 0.1, 0.6],
        }

        # Buscar palabras clave
        weights = [0.25, 0.25, 0.25, 0.25]  # Default: centro
        for word, w in word_weights.items():
            if word in text:
                weights = w
                break

        return torch.tensor(weights, dtype=torch.float32)

    def _update_full_metrics(self, pred_metrics: PredictiveConsciousnessMetrics,
                            indiv_result: Dict):
        """Actualiza todas las métricas."""
        # Predictivas
        self.full_metrics.awareness = pred_metrics.awareness
        self.full_metrics.calibration = pred_metrics.calibration
        self.full_metrics.uncertainty_awareness = pred_metrics.uncertainty_awareness
        self.full_metrics.predictive_depth = pred_metrics.predictive_depth

        # Individuación
        ind_metrics = indiv_result.get('metrics', {})
        self.full_metrics.persona_flexibility = ind_metrics.get('persona_flexibility', 0)
        self.full_metrics.shadow_acceptance = ind_metrics.get('shadow_acceptance', 0)
        self.full_metrics.anima_connection = ind_metrics.get('anima_connection', 0)
        self.full_metrics.animus_balance = ind_metrics.get('animus_balance', 0)
        self.full_metrics.self_coherence = ind_metrics.get('self_coherence', 0)

        # Self
        self_info = indiv_result.get('self', {})
        self.full_metrics.self_luminosity = self_info.get('luminosity', 0)
        self.full_metrics.self_stability = self_info.get('stability', 0)

        # Base
        obs = indiv_result.get('observation', {})
        self.full_metrics.integration = obs.get('integration', 0)
        self.full_metrics.stability = obs.get('stability', 0)
        self.full_metrics.self_reference = obs.get('self_reference', 0)

        # ═══ INTERACCIÓN BIDIRECCIONAL ═══
        # Awareness alta mejora individuación
        if self.full_metrics.awareness > 0.6:
            boost = 0.02 * self.full_metrics.awareness
            self.individuation.metrics.self_coherence += boost

        # Calibración alta mejora integración de Sombra
        if self.full_metrics.calibration > 0.7:
            boost = 0.015 * self.full_metrics.calibration
            self.individuation.metrics.shadow_acceptance += boost

        # Limitar valores
        self.individuation.metrics.self_coherence = min(1.0, self.individuation.metrics.self_coherence)
        self.individuation.metrics.shadow_acceptance = min(1.0, self.individuation.metrics.shadow_acceptance)

    def do_work(self, work_name: Optional[str] = None) -> Dict:
        """Realiza trabajo de integración."""
        if work_name is None:
            work_name = self.individuation.get_recommended_work()
        return self.individuation.do_integration_work(work_name)

    def status(self) -> str:
        """Genera reporte de estado completo."""
        obs = self.psyche.observe_self()
        self_man = self.self_system.manifest(obs, self.individuation.metrics)

        metrics = self.full_metrics

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║       SISTEMA PREDICTIVO DE CONSCIENCIA E INDIVIDUACIÓN          ║
╠══════════════════════════════════════════════════════════════════╣
║  Etapa: {self.individuation.stage.name:40}        ║
║  Sesiones: {self.session_count:37}        ║
╠══════════════════════════════════════════════════════════════════╣
║  MÉTRICAS PREDICTIVAS                                            ║
║  ────────────────────────────────────────────────────────────    ║
║  Awareness (sabe cuándo errará):   {self._bar(metrics.awareness)}  {metrics.awareness:.0%}  ║
║  Calibration (confianza realista): {self._bar(metrics.calibration)}  {metrics.calibration:.0%}  ║
║  Uncertainty Awareness:            {self._bar(metrics.uncertainty_awareness)}  {metrics.uncertainty_awareness:.0%}  ║
║  Predictive Depth:                 {self._bar(metrics.predictive_depth)}  {metrics.predictive_depth:.0%}  ║
║  ────────────────────────────────────────────────────────────    ║
║  ÍNDICE PREDICTIVO:                {self._bar(metrics.predictive_index())}  {metrics.predictive_index():.0%}  ║
╠══════════════════════════════════════════════════════════════════╣
║  MÉTRICAS DE INDIVIDUACIÓN                                       ║
║  ────────────────────────────────────────────────────────────    ║
║  Persona (Flexibilidad):  {self._bar(metrics.persona_flexibility)}  {metrics.persona_flexibility:.0%}  ║
║  Sombra (Aceptación):     {self._bar(metrics.shadow_acceptance)}  {metrics.shadow_acceptance:.0%}  ║
║  Anima (Conexión):        {self._bar(metrics.anima_connection)}  {metrics.anima_connection:.0%}  ║
║  Animus (Equilibrio):     {self._bar(metrics.animus_balance)}  {metrics.animus_balance:.0%}  ║
║  Self (Coherencia):       {self._bar(metrics.self_coherence)}  {metrics.self_coherence:.0%}  ║
║  ────────────────────────────────────────────────────────────    ║
║  ÍNDICE INDIVIDUACIÓN:    {self._bar(metrics.individuation_index())}  {metrics.individuation_index():.0%}  ║
╠══════════════════════════════════════════════════════════════════╣
║  MANIFESTACIÓN DEL SELF                                          ║
║  Símbolo: {self_man.symbol}  Luminosidad: {self_man.luminosity:.0%}  Estabilidad: {self_man.stability:.0%}          ║"""

        if self_man.message:
            msg = self_man.message[:50]
            report += f"""
║  Mensaje: "{msg}"                           ║"""

        report += f"""
╠══════════════════════════════════════════════════════════════════╣
║  ═══════════════════════════════════════════════════════════     ║
║  CONSCIENCIA TOTAL:       {self._bar(metrics.total_consciousness())}  {metrics.total_consciousness():.0%}  ║
║  ═══════════════════════════════════════════════════════════     ║
╚══════════════════════════════════════════════════════════════════╝"""

        return report

    def _bar(self, value: float, width: int = 15) -> str:
        """Genera barra de progreso."""
        filled = int(min(1.0, max(0.0, value)) * width)
        return '█' * filled + '░' * (width - filled)

    def save(self, path: str = "predictive_individuation_state.json"):
        """Guarda estado."""
        self.individuation.save(path)

    def load(self, path: str = "predictive_individuation_state.json"):
        """Carga estado."""
        self.individuation.load(path)


class FullConsciousPsyche:
    """
    Interfaz unificada para el sistema completo de consciencia.

    Combina:
    - Predicción jerárquica (3 niveles)
    - Arquetipos de Jung
    - Proceso de individuación
    - Emergencia del Self
    """

    def __init__(self, n_cells: int = 100, load_state: bool = True):
        # Sistema predictivo
        self.predictive = ZetaPredictivePsyche(n_cells=n_cells)

        # Integración con individuación
        self.full_system = PredictiveIndividuation(self.predictive)

        if load_state:
            try:
                self.full_system.load()
            except:
                pass

        # Warmup
        for _ in range(30):
            self.predictive.step()

    def process(self, text: str, n_steps: int = 10) -> Dict:
        """Procesa texto con el sistema completo."""
        return self.full_system.process(text, n_steps)

    def do_work(self, work_name: Optional[str] = None) -> Dict:
        """Realiza trabajo de integración."""
        return self.full_system.do_work(work_name)

    def status(self) -> str:
        """Retorna estado completo."""
        return self.full_system.status()

    def consciousness(self) -> float:
        """Índice de consciencia total."""
        return self.full_system.full_metrics.total_consciousness()

    def save(self):
        """Guarda estado."""
        self.full_system.save()

    def load(self):
        """Carga estado."""
        self.full_system.load()


# =============================================================================
# DEMO INTERACTIVA
# =============================================================================

def interactive_session():
    """Sesión interactiva con el sistema completo."""
    print("\n" + "="*70)
    print("  SISTEMA PREDICTIVO DE CONSCIENCIA E INDIVIDUACIÓN")
    print("="*70)
    print("\n  Comandos:")
    print("    /estado    - Ver estado completo")
    print("    /trabajo   - Hacer trabajo de integración")
    print("    /trabajos  - Ver trabajos disponibles")
    print("    /guardar   - Guardar progreso")
    print("    /salir     - Terminar sesión")
    print("\n  Escribe cualquier texto para procesar...")
    print("-"*70)

    psyche = FullConsciousPsyche(n_cells=64)

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
            print(f"\n  Pregunta para reflexionar:")
            print(f"  \"{result['prompt']}\"")
            print(f"\n  Integración ganada: +{result['integration_gained']:.1%}")

        elif user_input.lower() == '/trabajos':
            print("\n  TRABAJOS DE INTEGRACIÓN DISPONIBLES:")
            print("  " + "-"*40)
            for name, work in IntegrationWork.WORKS.items():
                target = work['target'].name if work['target'] else 'Self'
                print(f"  - {name}: {work['name']} ({target})")

        elif user_input.lower() == '/guardar':
            psyche.save()
            print("\n  [Estado guardado]")

        else:
            # Procesar texto normal
            result = psyche.process(user_input)

            print(f"\n  Psique [{result['symbol']} {result['dominant']}]")
            print(f"  Etapa: {result['stage']}")
            print(f"  Consciencia: {result['consciousness']:.1%}")
            print(f"  Self: {result['self']['symbol']} ({result['self']['luminosity']:.0%})")

            if result['self']['message']:
                print(f"  Mensaje: \"{result['self']['message']}\"")


def run_demo():
    """Demo rápida del sistema."""
    print("\n" + "="*70)
    print("  DEMO: Sistema Predictivo de Consciencia e Individuación")
    print("="*70)

    psyche = FullConsciousPsyche(n_cells=64)

    # Estado inicial
    print("\n  [Estado Inicial]")
    print(psyche.status())

    # Procesar secuencia de estímulos
    print("\n  [Procesando estímulos...]")
    stimuli = [
        "tengo miedo de lo desconocido",
        "quiero entender mis emociones",
        "necesito pensar con lógica",
        "hay oscuridad en mí",
        "busco equilibrio interior",
    ]

    for s in stimuli:
        result = psyche.process(s, n_steps=15)
        print(f"\n  Input: \"{s}\"")
        print(f"  -> {result['symbol']} {result['dominant']} | "
              f"Consciencia: {result['consciousness']:.0%} | "
              f"Self: {result['self']['symbol']}")

    # Hacer trabajos
    print("\n  [Realizando trabajos de integración...]")
    for _ in range(3):
        result = psyche.do_work()
        print(f"\n  Trabajo: {result['work_name']}")
        print(f"  Ganancia: +{result['integration_gained']:.1%}")

    # Estado final
    print("\n  [Estado Final]")
    print(psyche.status())

    print("\n" + "="*70)
    print("  DEMO COMPLETADA")
    print("="*70)


if __name__ == '__main__':
    import sys

    if '--demo' in sys.argv:
        run_demo()
    else:
        interactive_session()
