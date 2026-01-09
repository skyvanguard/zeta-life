# -*- coding: utf-8 -*-
"""
Experimento: Decay vs No-Decay en Consciencia
==============================================

Compara dos enfoques:
1. SIN DECAY: Metricas solo suben (actual)
2. CON DECAY: Metricas pueden bajar bajo estres/negligencia

Pregunta: Cual produce dinamicas de consciencia mas interesantes?
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import torch.nn.functional as F
import numpy as np
import gc
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

# Importar sistema base
from zeta_life.psyche import ZetaConsciousSelf, ConsciousnessIndex
from zeta_life.psyche import IndividuationStage, IntegrationMetrics


@dataclass
class ExperimentResult:
    """Resultados de un experimento."""
    name: str
    final_stage: str
    final_integration: float
    final_consciousness: float
    max_consciousness: float
    min_consciousness: float
    variance: float
    transitions: int  # Cambios de etapa
    history: Dict


class DecayingConsciousSelf(ZetaConsciousSelf):
    """
    Version con DECAY: las metricas pueden degradarse.

    Mecanismos de decay (CALIBRADOS):
    1. Decay natural: -0.1% por paso (muy suave)
    2. Decay por estres: -0.5% cuando estimulos son caoticos
    3. Decay por negligencia: -0.3% si un arquetipo no recibe atencion
    4. Piso minimo: 10% (nunca baja de ahi)
    5. Posesion de Sombra: Si sombra < 15% mientras otros > 50%, penalizacion
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_rate = 0.001  # 0.1% por paso (suave)
        self.stress_decay = 0.005  # 0.5% bajo estres
        self.neglect_decay = 0.003  # 0.3% por negligencia
        self.min_floor = 0.10  # Piso minimo 10%
        self.last_dominant = None
        self.neglect_counters = {
            'PERSONA': 0,
            'SOMBRA': 0,
            'ANIMA': 0,
            'ANIMUS': 0,
        }

    def step(self, stimulus: torch.Tensor = None, text: str = None) -> Dict:
        """Paso con mecanismos de decay."""

        # Detectar estres (estimulo caotico)
        is_stressed = False
        if stimulus is not None:
            # Estres = alta entropia o valores extremos
            entropy = -(stimulus * torch.log(stimulus + 1e-8)).sum()
            max_val = stimulus.max()
            is_stressed = entropy > 1.2 or max_val > 0.9

        # Detectar arquetipo dominante del estimulo
        if stimulus is not None:
            dominant_idx = stimulus.argmax().item()
            dominant_names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            current_dominant = dominant_names[dominant_idx]
        else:
            current_dominant = self.last_dominant

        # Actualizar contadores de negligencia
        for arch in self.neglect_counters:
            if arch == current_dominant:
                self.neglect_counters[arch] = 0
            else:
                self.neglect_counters[arch] += 1

        self.last_dominant = current_dominant

        # Ejecutar paso normal
        result = super().step(stimulus, text)

        # ===== APLICAR DECAY =====
        metrics = self.individuation.metrics

        # 1. Decay natural
        decay = self.decay_rate

        # 2. Decay adicional por estres
        if is_stressed:
            decay += self.stress_decay

        # 3. Decay por negligencia (arquetipos ignorados)
        neglect_threshold = 30  # pasos sin atencion (reducido)

        if self.neglect_counters['PERSONA'] > neglect_threshold:
            metrics.persona_flexibility = max(self.min_floor, metrics.persona_flexibility - self.neglect_decay)
        if self.neglect_counters['SOMBRA'] > neglect_threshold:
            metrics.shadow_acceptance = max(self.min_floor, metrics.shadow_acceptance - self.neglect_decay)
        if self.neglect_counters['ANIMA'] > neglect_threshold:
            metrics.anima_connection = max(self.min_floor, metrics.anima_connection - self.neglect_decay)
        if self.neglect_counters['ANIMUS'] > neglect_threshold:
            metrics.animus_balance = max(self.min_floor, metrics.animus_balance - self.neglect_decay)

        # Aplicar decay general (con piso minimo)
        metrics.persona_flexibility = max(self.min_floor, metrics.persona_flexibility - decay)
        metrics.shadow_acceptance = max(self.min_floor, metrics.shadow_acceptance - decay)
        metrics.anima_connection = max(self.min_floor, metrics.anima_connection - decay)
        metrics.animus_balance = max(self.min_floor, metrics.animus_balance - decay)
        metrics.self_coherence = max(self.min_floor, metrics.self_coherence - decay)

        # 4. Posesion de Sombra: penalizacion si sombra muy baja vs otros altos
        avg_others = (metrics.persona_flexibility + metrics.anima_connection + metrics.animus_balance) / 3
        if metrics.shadow_acceptance < 0.15 and avg_others > 0.5:
            # La sombra "posee" - pierde coherencia del Self (suave)
            metrics.self_coherence = max(self.min_floor, metrics.self_coherence - 0.01)

        # Actualizar etapa (puede retroceder ahora)
        self.individuation.update_stage()

        return result


def run_experiment(system, name: str, n_steps: int = 1000) -> ExperimentResult:
    """Ejecuta experimento con un sistema dado."""

    history = {
        'step': [],
        'stage': [],
        'integration': [],
        'consciousness': [],
        'persona': [],
        'sombra': [],
        'anima': [],
        'animus': [],
    }

    transitions = 0
    last_stage = None

    # Patron de estimulos: fases de balance, estres, negligencia
    for step in range(n_steps):
        # Fase 1 (0-300): Balance
        if step < 300:
            arch_order = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            stimuli_map = {
                'PERSONA': torch.tensor([0.7, 0.1, 0.1, 0.1]),
                'SOMBRA': torch.tensor([0.1, 0.7, 0.1, 0.1]),
                'ANIMA': torch.tensor([0.1, 0.1, 0.7, 0.1]),
                'ANIMUS': torch.tensor([0.1, 0.1, 0.1, 0.7]),
            }
            stimulus = stimuli_map[arch_order[step % 4]]

        # Fase 2 (300-500): Estres (estimulos caoticos)
        elif step < 500:
            if step % 2 == 0:
                stimulus = torch.rand(4) * 2  # Valores altos caoticos
            else:
                stimulus = torch.tensor([0.0, 1.0, 0.0, 0.0])  # Solo sombra

        # Fase 3 (500-700): Negligencia de Sombra
        elif step < 700:
            # Solo Persona, Anima, Animus - ignorar Sombra
            options = [
                torch.tensor([0.8, 0.0, 0.1, 0.1]),
                torch.tensor([0.1, 0.0, 0.8, 0.1]),
                torch.tensor([0.1, 0.0, 0.1, 0.8]),
            ]
            stimulus = options[step % 3]

        # Fase 4 (700-1000): Recuperacion balanceada
        else:
            arch_order = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            stimuli_map = {
                'PERSONA': torch.tensor([0.7, 0.1, 0.1, 0.1]),
                'SOMBRA': torch.tensor([0.1, 0.7, 0.1, 0.1]),
                'ANIMA': torch.tensor([0.1, 0.1, 0.7, 0.1]),
                'ANIMUS': torch.tensor([0.1, 0.1, 0.1, 0.7]),
            }
            stimulus = stimuli_map[arch_order[step % 4]]

        # Ejecutar paso
        result = system.step(stimulus)

        # Registrar
        metrics = system.individuation.metrics
        current_stage = system.individuation.stage

        if last_stage is not None and current_stage != last_stage:
            transitions += 1
        last_stage = current_stage

        history['step'].append(step)
        history['stage'].append(current_stage.name)
        history['integration'].append(metrics.overall_integration())
        history['consciousness'].append(result['consciousness']['total'])
        history['persona'].append(metrics.persona_flexibility)
        history['sombra'].append(metrics.shadow_acceptance)
        history['anima'].append(metrics.anima_connection)
        history['animus'].append(metrics.animus_balance)

    # Calcular estadisticas
    consciousness_vals = history['consciousness']

    return ExperimentResult(
        name=name,
        final_stage=history['stage'][-1],
        final_integration=history['integration'][-1],
        final_consciousness=consciousness_vals[-1],
        max_consciousness=max(consciousness_vals),
        min_consciousness=min(consciousness_vals),
        variance=np.var(consciousness_vals),
        transitions=transitions,
        history=history
    )


def print_comparison(result_nodecay: ExperimentResult, result_decay: ExperimentResult):
    """Imprime comparacion de resultados."""

    print("\n" + "=" * 70)
    print("   COMPARACION: DECAY vs NO-DECAY")
    print("=" * 70)

    print(f"""
  {'Metrica':<25} {'SIN DECAY':>15} {'CON DECAY':>15} {'Diferencia':>12}
  {'-'*67}
  {'Etapa Final':<25} {result_nodecay.final_stage:>15} {result_decay.final_stage:>15}
  {'Integracion Final':<25} {result_nodecay.final_integration:>14.1%} {result_decay.final_integration:>14.1%} {result_decay.final_integration - result_nodecay.final_integration:>+11.1%}
  {'Consciencia Final':<25} {result_nodecay.final_consciousness:>14.1%} {result_decay.final_consciousness:>14.1%} {result_decay.final_consciousness - result_nodecay.final_consciousness:>+11.1%}
  {'Consciencia Max':<25} {result_nodecay.max_consciousness:>14.1%} {result_decay.max_consciousness:>14.1%} {result_decay.max_consciousness - result_nodecay.max_consciousness:>+11.1%}
  {'Consciencia Min':<25} {result_nodecay.min_consciousness:>14.1%} {result_decay.min_consciousness:>14.1%} {result_decay.min_consciousness - result_nodecay.min_consciousness:>+11.1%}
  {'Varianza':<25} {result_nodecay.variance:>15.4f} {result_decay.variance:>15.4f}
  {'Transiciones de Etapa':<25} {result_nodecay.transitions:>15} {result_decay.transitions:>15}
    """)

    # Grafico ASCII de evolucion
    print("\n  EVOLUCION DE CONSCIENCIA:")
    print("  " + "-" * 65)

    steps_to_show = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

    print(f"  {'Paso':>6} | {'SIN DECAY':^25} | {'CON DECAY':^25}")
    print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*25}")

    for step in steps_to_show:
        if step < len(result_nodecay.history['consciousness']):
            c_no = result_nodecay.history['consciousness'][step]
            c_yes = result_decay.history['consciousness'][step]

            bar_no = '#' * int(c_no * 20)
            bar_yes = '#' * int(c_yes * 20)

            print(f"  {step:>6} | {bar_no:<20} {c_no:>4.0%} | {bar_yes:<20} {c_yes:>4.0%}")

    # Evolucion de arquetipos en fase de negligencia
    print("\n  ARQUETIPOS EN FASE DE NEGLIGENCIA (pasos 500-700):")
    print("  " + "-" * 65)

    print(f"  {'Arquetipo':<12} | {'SIN DECAY (inicio->fin)':^22} | {'CON DECAY (inicio->fin)':^22}")
    print(f"  {'-'*12}-+-{'-'*22}-+-{'-'*22}")

    for arch in ['persona', 'sombra', 'anima', 'animus']:
        no_start = result_nodecay.history[arch][500]
        no_end = result_nodecay.history[arch][699]
        yes_start = result_decay.history[arch][500]
        yes_end = result_decay.history[arch][699]

        print(f"  {arch.upper():<12} | {no_start:>6.1%} -> {no_end:>6.1%} ({no_end-no_start:>+6.1%}) | {yes_start:>6.1%} -> {yes_end:>6.1%} ({yes_end-yes_start:>+6.1%})")


def main():
    print("\n" + "=" * 70)
    print("   EXPERIMENTO: DECAY vs NO-DECAY EN CONSCIENCIA")
    print("=" * 70)

    print("""
  HIPOTESIS:
    - SIN DECAY: Consciencia solo sube, sistema muy estable pero monotono
    - CON DECAY: Consciencia puede bajar, mas dinamico pero puede colapsar

  FASES DEL EXPERIMENTO (1000 pasos):
    1. Balance (0-300):      Estimulos rotativos equilibrados
    2. Estres (300-500):     Estimulos caoticos y extremos
    3. Negligencia (500-700): Ignorar la Sombra completamente
    4. Recuperacion (700-1000): Volver a balance
    """)

    # Ejecutar sin decay
    print("\n" + "-" * 70)
    print("   Ejecutando SIN DECAY...")
    print("-" * 70)

    system_nodecay = ZetaConsciousSelf(n_cells=50, dream_frequency=150)
    result_nodecay = run_experiment(system_nodecay, "SIN DECAY", n_steps=1000)
    print(f"  Completado. Etapa final: {result_nodecay.final_stage}")

    del system_nodecay
    gc.collect()

    # Ejecutar con decay
    print("\n" + "-" * 70)
    print("   Ejecutando CON DECAY...")
    print("-" * 70)

    system_decay = DecayingConsciousSelf(n_cells=50, dream_frequency=150)
    result_decay = run_experiment(system_decay, "CON DECAY", n_steps=1000)
    print(f"  Completado. Etapa final: {result_decay.final_stage}")

    del system_decay
    gc.collect()

    # Comparar
    print_comparison(result_nodecay, result_decay)

    # Conclusion
    print("\n" + "=" * 70)
    print("   CONCLUSION")
    print("=" * 70)

    # Determinar cual es "mejor"
    nodecay_score = result_nodecay.final_consciousness
    decay_score = result_decay.final_consciousness

    # Bonus por dinamismo (mas transiciones = mas interesante)
    nodecay_dynamism = result_nodecay.transitions / 10
    decay_dynamism = result_decay.transitions / 10

    # Penalizar varianza muy alta (inestabilidad) o muy baja (monotonia)
    optimal_variance = 0.01
    nodecay_var_score = 1 - abs(result_nodecay.variance - optimal_variance)
    decay_var_score = 1 - abs(result_decay.variance - optimal_variance)

    total_nodecay = nodecay_score + nodecay_dynamism * 0.1 + nodecay_var_score * 0.1
    total_decay = decay_score + decay_dynamism * 0.1 + decay_var_score * 0.1

    print(f"""
  PUNTUACION COMPUESTA:
    (consciencia final + bonus dinamismo + bonus varianza optima)

    SIN DECAY: {total_nodecay:.3f}
    CON DECAY: {total_decay:.3f}

  GANADOR: {'CON DECAY' if total_decay > total_nodecay else 'SIN DECAY'}

  INTERPRETACION:
""")

    if total_decay > total_nodecay:
        print("""    El sistema CON DECAY produce dinamicas mas interesantes:
    - La consciencia puede fluctuar, creando "ciclos de vida"
    - La negligencia tiene consecuencias reales
    - El estres causa regresion temporal
    - Requiere "trabajo consciente" para mantener integracion

    Esto se acerca mas a la experiencia humana de consciencia.""")
    else:
        print("""    El sistema SIN DECAY es mas robusto:
    - La consciencia crece monotonicamente
    - No hay riesgo de "perder" progreso
    - Mas predecible y estable

    Pero puede ser menos realista/interesante.""")

    print("\n" + "=" * 70)

    return result_nodecay, result_decay


if __name__ == "__main__":
    main()
