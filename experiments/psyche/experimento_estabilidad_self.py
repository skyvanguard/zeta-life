# -*- coding: utf-8 -*-
"""
Experimento: Estabilidad de SELF_REALIZADO
==========================================

Pregunta: Una vez alcanzado SELF_REALIZADO, se mantiene o retrocede?

Jung decia: "La individuacion nunca es un estado permanente,
sino un equilibrio dinamico que debe mantenerse activamente."

Fases del experimento:
1. Alcanzar SELF_REALIZADO (estimulos balanceados)
2. Fase de estres (estimulos caoticos/unilaterales)
3. Observar regresion
4. Fase de recuperacion
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import numpy as np
import gc
from zeta_life.psyche import ZetaConsciousSelf
from zeta_life.psyche import IndividuationStage


def experimento_estabilidad():
    print("\n" + "=" * 70)
    print("   EXPERIMENTO: ESTABILIDAD DE SELF_REALIZADO")
    print("=" * 70)

    print("""
  PREGUNTA: Una vez alcanzado, puede perderse SELF_REALIZADO?

  FASES:
    1. ALCANZAR   - Llegar a SELF_REALIZADO (800 pasos balanceados)
    2. ESTRES     - Estimulos caoticos y unilaterales (400 pasos)
    3. OBSERVAR   - Ver si hay regresion
    4. RECUPERAR  - Volver a estimulos balanceados (400 pasos)
    """)

    # Crear sistema (reducido para evitar errores de memoria)
    system = ZetaConsciousSelf(n_cells=50, dream_frequency=100)

    stimuli = {
        'PERSONA': torch.tensor([0.8, 0.1, 0.05, 0.05]),
        'SOMBRA': torch.tensor([0.1, 0.8, 0.05, 0.05]),
        'ANIMA': torch.tensor([0.05, 0.1, 0.8, 0.05]),
        'ANIMUS': torch.tensor([0.05, 0.05, 0.1, 0.8]),
    }

    work_by_arch = {
        'PERSONA': 'persona_examination',
        'SOMBRA': 'shadow_dialogue',
        'ANIMA': 'anima_encounter',
        'ANIMUS': 'animus_balance',
    }

    history = {
        'step': [],
        'phase': [],
        'stage': [],
        'integration': [],
        'self_coherence': [],
        'consciousness': [],
    }

    def record(step, phase):
        metrics = system.individuation.metrics
        result = system.step(torch.rand(4) * 0.1)  # paso minimo para obtener estado
        history['step'].append(step)
        history['phase'].append(phase)
        history['stage'].append(system.individuation.stage.name)
        history['integration'].append(metrics.overall_integration())
        history['self_coherence'].append(metrics.self_coherence)
        history['consciousness'].append(result['consciousness']['total'])

    # ===== FASE 1: ALCANZAR SELF_REALIZADO =====
    print("\n" + "-" * 70)
    print("   FASE 1: ALCANZAR SELF_REALIZADO")
    print("-" * 70)

    phase1_steps = 800
    for step in range(phase1_steps):
        metrics = system.individuation.metrics
        arch_values = {
            'PERSONA': metrics.persona_flexibility,
            'SOMBRA': metrics.shadow_acceptance,
            'ANIMA': metrics.anima_connection,
            'ANIMUS': metrics.animus_balance,
        }
        lowest_arch = min(arch_values, key=arch_values.get)

        if np.random.random() < 0.7:
            stimulus = stimuli[lowest_arch]
        else:
            arch_order = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            stimulus = stimuli[arch_order[step % 4]]

        result = system.step(stimulus)

        if (step + 1) % 50 == 0:
            work_name = work_by_arch[lowest_arch]
            system.individuation.do_integration_work(work_name)

        if (step + 1) % 200 == 0:
            record(step + 1, 'ALCANZAR')
            stage = system.individuation.stage
            print(f"  Paso {step+1}: {stage.name} | Integracion: {metrics.overall_integration():.1%}")

    print(f"\n  >> Estado al final de Fase 1: {system.individuation.stage.name}")
    gc.collect()  # Liberar memoria

    # ===== FASE 2: ESTRES (estimulos caoticos) =====
    print("\n" + "-" * 70)
    print("   FASE 2: ESTRES (estimulos caoticos y unilaterales)")
    print("-" * 70)

    print("  Aplicando: ruido aleatorio, estimulos extremos, sin trabajos de integracion...")

    phase2_steps = 400
    for step in range(phase2_steps):
        # Estimulos estresantes
        if step % 3 == 0:
            # Ruido caotico
            stimulus = torch.rand(4) * 2.0
        elif step % 3 == 1:
            # Solo SOMBRA (trauma)
            stimulus = torch.tensor([0.0, 1.0, 0.0, 0.0])
        else:
            # Extremos alternantes
            stimulus = torch.tensor([1.0, 1.0, 0.0, 0.0]) if step % 2 == 0 else torch.tensor([0.0, 0.0, 1.0, 1.0])

        result = system.step(stimulus)

        # Sin trabajos de integracion - solo estres

        if (step + 1) % 100 == 0:
            record(phase1_steps + step + 1, 'ESTRES')
            metrics = system.individuation.metrics
            stage = system.individuation.stage
            print(f"  Paso {phase1_steps + step + 1}: {stage.name} | Integracion: {metrics.overall_integration():.1%}")

    print(f"\n  >> Estado al final de Fase 2: {system.individuation.stage.name}")

    # Verificar regresion
    final_stage_stress = system.individuation.stage
    regressed = final_stage_stress != IndividuationStage.SELF_REALIZADO

    if regressed:
        print(f"  ** REGRESION DETECTADA: {final_stage_stress.name} **")
    else:
        print("  ** SELF_REALIZADO MANTENIDO bajo estres **")
    gc.collect()  # Liberar memoria

    # ===== FASE 3: RECUPERACION =====
    print("\n" + "-" * 70)
    print("   FASE 3: RECUPERACION (volver a estimulos balanceados)")
    print("-" * 70)

    phase3_steps = 400
    for step in range(phase3_steps):
        metrics = system.individuation.metrics
        arch_values = {
            'PERSONA': metrics.persona_flexibility,
            'SOMBRA': metrics.shadow_acceptance,
            'ANIMA': metrics.anima_connection,
            'ANIMUS': metrics.animus_balance,
        }
        lowest_arch = min(arch_values, key=arch_values.get)

        if np.random.random() < 0.7:
            stimulus = stimuli[lowest_arch]
        else:
            arch_order = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            stimulus = stimuli[arch_order[step % 4]]

        result = system.step(stimulus)

        if (step + 1) % 50 == 0:
            work_name = work_by_arch[lowest_arch]
            system.individuation.do_integration_work(work_name)

        if (step + 1) % 100 == 0:
            record(phase1_steps + phase2_steps + step + 1, 'RECUPERAR')
            stage = system.individuation.stage
            print(f"  Paso {phase1_steps + phase2_steps + step + 1}: {stage.name} | Integracion: {metrics.overall_integration():.1%}")

    print(f"\n  >> Estado al final de Fase 3: {system.individuation.stage.name}")

    # ===== ANALISIS FINAL =====
    print("\n" + "=" * 70)
    print("   RESULTADOS")
    print("=" * 70)

    # Grafico ASCII de la evolucion
    print("\n  EVOLUCION DE INTEGRACION POR FASE:")
    print("  " + "-" * 60)
    print("  Fase      | Paso | Integracion | Etapa")
    print("  " + "-" * 60)

    for i in range(len(history['step'])):
        phase = history['phase'][i]
        step = history['step'][i]
        integ = history['integration'][i]
        stage = history['stage'][i]
        bar = '#' * int(integ * 30)
        print(f"  {phase:10} | {step:4} | {bar:<30} | {stage}")

    # Analisis de estabilidad
    print("\n  ANALISIS DE ESTABILIDAD:")
    print("  " + "-" * 60)

    # Encontrar puntos clave
    pre_stress_idx = len([h for h in history['phase'] if h == 'ALCANZAR']) - 1
    post_stress_idx = pre_stress_idx + len([h for h in history['phase'] if h == 'ESTRES'])
    final_idx = len(history['step']) - 1

    if pre_stress_idx >= 0 and post_stress_idx < len(history['integration']):
        pre_stress_integ = history['integration'][pre_stress_idx]
        post_stress_integ = history['integration'][post_stress_idx]
        final_integ = history['integration'][final_idx]

        print(f"    Antes del estres:   {pre_stress_integ:.1%} ({history['stage'][pre_stress_idx]})")
        print(f"    Despues del estres: {post_stress_integ:.1%} ({history['stage'][post_stress_idx]})")
        print(f"    Tras recuperacion:  {final_integ:.1%} ({history['stage'][final_idx]})")

        loss = pre_stress_integ - post_stress_integ
        recovery = final_integ - post_stress_integ

        print(f"\n    Perdida por estres: {loss:+.1%}")
        print(f"    Recuperacion:       {recovery:+.1%}")

        if final_integ >= pre_stress_integ:
            print("\n    >> RECUPERACION COMPLETA")
        elif final_integ >= 0.9:
            print("\n    >> SELF_REALIZADO RECUPERADO")
        else:
            print(f"\n    >> Recuperacion parcial ({final_integ/pre_stress_integ:.0%} del estado original)")

    # Conclusion
    print("\n" + "=" * 70)
    print("   CONCLUSION")
    print("=" * 70)

    final_stage = system.individuation.stage
    metrics = system.individuation.metrics

    print(f"""
  El estado SELF_REALIZADO es {'DINAMICO' if regressed else 'ROBUSTO'}:

  - {'Puede perderse bajo estres prolongado' if regressed else 'Se mantuvo bajo estres'}
  - {'Puede recuperarse con trabajo consciente' if final_stage == IndividuationStage.SELF_REALIZADO else 'Requiere mas trabajo para recuperar'}

  Esto confirma la vision de Jung:
  "La individuacion no es un destino, sino un PROCESO continuo."

  Estado final: {final_stage.name}
  Integracion:  {metrics.overall_integration():.1%}
  Self:         {metrics.self_coherence:.1%}
    """)

    # Mensaje del Self
    self_manifest = system.individuation.self_system.manifest(
        system.psyche.observe_self(),
        system.individuation.metrics
    )
    if self_manifest.message:
        print(f"  Mensaje del Self: \"{self_manifest.message}\"")

    print("\n" + "=" * 70)

    return history


if __name__ == "__main__":
    experimento_estabilidad()
