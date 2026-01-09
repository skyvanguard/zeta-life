# -*- coding: utf-8 -*-
"""
Experimento: Camino hacia SELF_REALIZADO
=========================================

Para alcanzar SELF_REALIZADO se necesita:
- 90% de integracion total
- TODOS los arquetipos deben estar balanceados (el minimo limita al Self)

Estrategia: Estimulos rotativos + trabajos de integracion dirigidos
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import numpy as np
from zeta_life.psyche import ZetaConsciousSelf
from zeta_life.psyche import IndividuationStage, IntegrationWork


def experimento_self_realizado():
    print("\n" + "=" * 70)
    print("   EXPERIMENTO: CAMINO HACIA SELF_REALIZADO")
    print("=" * 70)

    print("""
  OBJETIVO: Alcanzar el estado SELF_REALIZADO (90% integracion)

  REQUISITOS:
    - Persona (Flexibilidad) >= 90%
    - Sombra (Aceptacion)    >= 90%
    - Anima (Conexion)       >= 90%
    - Animus (Equilibrio)    >= 90%
    - Self Coherence = min(arriba) >= 90%

  ESTRATEGIA:
    1. Estimulos rotativos para balancear arquetipos
    2. Trabajos de integracion dirigidos al mas bajo
    3. Ciclos de sueno para consolidar
    """)

    # Crear sistema
    system = ZetaConsciousSelf(n_cells=100, dream_frequency=50)

    # Definir estimulos por arquetipo
    stimuli = {
        'PERSONA': torch.tensor([0.8, 0.1, 0.05, 0.05]),   # Oportunidad social
        'SOMBRA': torch.tensor([0.1, 0.8, 0.05, 0.05]),    # Amenaza/oscuro
        'ANIMA': torch.tensor([0.05, 0.1, 0.8, 0.05]),     # Emocional
        'ANIMUS': torch.tensor([0.05, 0.05, 0.1, 0.8]),    # Cognitivo/accion
    }

    # Mapeo de trabajos por arquetipo
    work_by_arch = {
        'PERSONA': 'persona_examination',
        'SOMBRA': 'shadow_dialogue',
        'ANIMA': 'anima_encounter',
        'ANIMUS': 'animus_balance',
    }

    max_steps = 2000
    work_interval = 50
    report_interval = 200

    history = {
        'consciousness': [],
        'integration': [],
        'stage': [],
        'persona': [],
        'sombra': [],
        'anima': [],
        'animus': [],
        'self_coherence': [],
    }

    print("\n" + "-" * 70)
    print("   SIMULACION")
    print("-" * 70)

    reached_self = False
    final_step = 0

    for step in range(max_steps):
        # Obtener metricas actuales para balancear
        metrics = system.individuation.metrics
        arch_values = {
            'PERSONA': metrics.persona_flexibility,
            'SOMBRA': metrics.shadow_acceptance,
            'ANIMA': metrics.anima_connection,
            'ANIMUS': metrics.animus_balance,
        }

        # Encontrar el arquetipo mas bajo
        lowest_arch = min(arch_values, key=arch_values.get)

        # 70% del tiempo: estimular el mas bajo
        # 30% del tiempo: rotacion para mantener balance
        if np.random.random() < 0.7:
            stimulus = stimuli[lowest_arch]
        else:
            # Rotacion ciclica
            arch_order = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            current_arch = arch_order[step % 4]
            stimulus = stimuli[current_arch]

        # Paso del sistema
        result = system.step(stimulus)

        # Guardar historial
        history['consciousness'].append(result['consciousness']['total'])
        history['integration'].append(metrics.overall_integration())
        history['stage'].append(system.individuation.stage.value)
        history['persona'].append(metrics.persona_flexibility)
        history['sombra'].append(metrics.shadow_acceptance)
        history['anima'].append(metrics.anima_connection)
        history['animus'].append(metrics.animus_balance)
        history['self_coherence'].append(metrics.self_coherence)

        # Trabajo de integracion cada N pasos
        if (step + 1) % work_interval == 0:
            # Hacer trabajo del arquetipo mas bajo
            work_name = work_by_arch[lowest_arch]
            work_result = system.individuation.do_integration_work(work_name)

        # Reporte cada N pasos
        if (step + 1) % report_interval == 0:
            stage = system.individuation.stage
            print(f"\n  Paso {step + 1}:")
            print(f"    Etapa: {stage.name}")
            print(f"    Integracion Total: {metrics.overall_integration():.1%}")
            print(f"    Arquetipos:")
            print(f"      Persona: {metrics.persona_flexibility:.1%}")
            print(f"      Sombra:  {metrics.shadow_acceptance:.1%}")
            print(f"      Anima:   {metrics.anima_connection:.1%}")
            print(f"      Animus:  {metrics.animus_balance:.1%}")
            print(f"    Self Coherence: {metrics.self_coherence:.1%}")
            print(f"    Consciencia: {result['consciousness']['total']:.1%}")

            # Verificar si alcanzamos SELF_REALIZADO
            if stage == IndividuationStage.SELF_REALIZADO:
                print("\n  " + "*" * 50)
                print("  ***  SELF_REALIZADO ALCANZADO!  ***")
                print("  " + "*" * 50)
                reached_self = True
                final_step = step + 1
                break

    if not reached_self:
        final_step = max_steps
        print(f"\n  Simulacion completada ({max_steps} pasos)")
        print(f"  Etapa final: {system.individuation.stage.name}")

    # Analisis final
    print("\n" + "=" * 70)
    print("   RESULTADOS FINALES")
    print("=" * 70)

    metrics = system.individuation.metrics
    stage = system.individuation.stage

    print(f"\n  ETAPA ALCANZADA: {stage.name}")
    print(f"  Pasos necesarios: {final_step}")

    print(f"\n  METRICAS DE INTEGRACION:")
    print(f"    Persona (Flexibilidad): {metrics.persona_flexibility:.1%}")
    print(f"    Sombra (Aceptacion):    {metrics.shadow_acceptance:.1%}")
    print(f"    Anima (Conexion):       {metrics.anima_connection:.1%}")
    print(f"    Animus (Equilibrio):    {metrics.animus_balance:.1%}")
    print(f"    Self Coherence:         {metrics.self_coherence:.1%}")
    print(f"    -" * 25)
    print(f"    TOTAL:                  {metrics.overall_integration():.1%}")

    # Distancia al objetivo
    if not reached_self:
        needed = 0.9  # 90% para SELF_REALIZADO
        current = metrics.overall_integration()
        gap = needed - current

        print(f"\n  DISTANCIA AL OBJETIVO:")
        print(f"    Necesario: 90%")
        print(f"    Actual:    {current:.1%}")
        print(f"    Falta:     {gap:.1%}")

        # Que falta por arquetipo
        print(f"\n  POR ARQUETIPO (necesario ~90% cada uno):")
        for name, val in [('Persona', metrics.persona_flexibility),
                          ('Sombra', metrics.shadow_acceptance),
                          ('Anima', metrics.anima_connection),
                          ('Animus', metrics.animus_balance)]:
            if val < 0.9:
                print(f"    {name}: falta {(0.9 - val):.1%}")

    # Evolucion grafica ASCII
    print("\n  EVOLUCION DE INTEGRACION:")
    print("  " + "-" * 60)

    n_points = 10
    step_size = len(history['integration']) // n_points

    for i in range(n_points):
        start = i * step_size
        end = (i + 1) * step_size
        avg = np.mean(history['integration'][start:end])
        bar = '#' * int(avg * 50)
        print(f"  {(i+1)*step_size:4d} | {bar:<50} | {avg:.1%}")

    # Evolucion por etapas
    print("\n  TRANSICIONES DE ETAPA:")
    print("  " + "-" * 60)

    prev_stage = history['stage'][0]
    transitions = [(0, prev_stage)]
    for i, s in enumerate(history['stage']):
        if s != prev_stage:
            transitions.append((i, s))
            prev_stage = s

    stage_names = {s.value: s.name for s in IndividuationStage}
    for step, stage_val in transitions:
        name = stage_names.get(stage_val, f"Stage_{stage_val}")
        print(f"    Paso {step:4d}: {name}")

    # Insights del Self
    print("\n  MENSAJES DEL SELF:")
    print("  " + "-" * 60)

    self_manifest = system.individuation.self_system.manifest(
        system.psyche.observe_self(),
        system.individuation.metrics
    )

    print(f"    Simbolo: {self_manifest.symbol}")
    print(f"    Luminosidad: {self_manifest.luminosity:.1%}")
    print(f"    Estabilidad: {self_manifest.stability:.1%}")
    if self_manifest.message:
        print(f"    Mensaje: \"{self_manifest.message}\"")

    print("\n" + "=" * 70)

    return {
        'reached': reached_self,
        'final_step': final_step,
        'final_stage': stage,
        'metrics': metrics,
        'history': history
    }


if __name__ == "__main__":
    experimento_self_realizado()
