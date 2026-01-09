# -*- coding: utf-8 -*-
"""
Experimento: Efecto de multiples ciclos de sueno
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import numpy as np
from zeta_life.psyche import ConsciousSystemWithDreams

def experimento_ciclos_sueno():
    print("\n" + "=" * 70)
    print("   EXPERIMENTO: EFECTO DE MULTIPLES CICLOS DE SUENO")
    print("=" * 70)

    # Sistema A: Con suenos
    print("\n  Creando Sistema A (CON suenos)...")
    system_dreams = ConsciousSystemWithDreams(n_cells=50, dream_frequency=100)

    # Sistema B: Sin suenos (control)
    print("  Creando Sistema B (SIN suenos - control)...")
    system_control = ConsciousSystemWithDreams(n_cells=50, dream_frequency=10000)

    # Ejecutar 500 pasos con patrones variados
    n_steps = 500
    dream_interval = 100

    consciousness_dreams = []
    consciousness_control = []
    dream_reports = []

    print("\n" + "-" * 70)
    print("   EJECUTANDO SIMULACION")
    print("-" * 70)

    for step in range(n_steps):
        # Generar estimulo con patron
        phase = step % 60
        if phase < 20:
            stimulus = torch.tensor([0.1, 0.7, 0.1, 0.1])  # Amenaza
        elif phase < 40:
            stimulus = torch.tensor([0.7, 0.1, 0.1, 0.1])  # Oportunidad
        else:
            stimulus = torch.rand(4)  # Aleatorio

        # Sistema con suenos
        result_dreams = system_dreams.step(stimulus, auto_dream=False)
        consciousness_dreams.append(result_dreams['consciousness'])

        # Sistema control
        result_control = system_control.step(stimulus, auto_dream=False)
        consciousness_control.append(result_control['consciousness'])

        # Ciclo de sueno cada 100 pasos
        if (step + 1) % dream_interval == 0:
            print(f"\n  --- Paso {step+1}: Ciclo de sueno ---")
            report = system_dreams.dream(duration=30, verbose=False)
            dream_reports.append(report)

            avg_dreams = np.mean(consciousness_dreams[-50:])
            avg_control = np.mean(consciousness_control[-50:])

            print(f"  Sistema CON suenos: {avg_dreams:.2%}")
            print(f"  Sistema SIN suenos: {avg_control:.2%}")
            print(f"  Diferencia: {avg_dreams - avg_control:+.2%}")
            print(f"  Loss reducido: {report.loss_reduction:+.4f}")

    # Analisis final
    print("\n" + "=" * 70)
    print("   RESULTADOS FINALES")
    print("=" * 70)

    # Por fases
    n_phases = 5
    phase_size = n_steps // n_phases

    print("\n  Evolucion por fases:")
    print(f"  {'Fase':<10} | {'Con Suenos':<12} | {'Sin Suenos':<12} | {'Diferencia':<12}")
    print("  " + "-" * 52)

    for i in range(n_phases):
        start = i * phase_size
        end = (i + 1) * phase_size

        avg_dreams = np.mean(consciousness_dreams[start:end])
        avg_control = np.mean(consciousness_control[start:end])
        diff = avg_dreams - avg_control

        print(f"  {start:3d}-{end:3d}   | {avg_dreams:.2%}       | {avg_control:.2%}       | {diff:+.2%}")

    # Totales
    print("\n  Promedios totales:")
    total_dreams = np.mean(consciousness_dreams)
    total_control = np.mean(consciousness_control)
    print(f"    Con suenos:  {total_dreams:.2%}")
    print(f"    Sin suenos:  {total_control:.2%}")
    print(f"    Diferencia:  {total_dreams - total_control:+.2%}")

    # Estabilidad
    var_dreams = np.var(consciousness_dreams)
    var_control = np.var(consciousness_control)
    print(f"\n  Estabilidad (menor varianza = mejor):")
    print(f"    Con suenos:  {var_dreams:.6f}")
    print(f"    Sin suenos:  {var_control:.6f}")

    # Consolidacion acumulada
    print(f"\n  Consolidacion acumulada ({len(dream_reports)} ciclos de sueno):")
    total_replays = sum(r.memories_replayed for r in dream_reports)
    total_learning = sum(r.learning_events for r in dream_reports)
    total_loss_reduction = sum(r.loss_reduction for r in dream_reports)

    print(f"    Memorias replayeadas: {total_replays}")
    print(f"    Eventos de aprendizaje: {total_learning}")
    print(f"    Reduccion total de loss: {total_loss_reduction:+.4f}")

    # Insights acumulados
    all_insights = []
    for r in dream_reports:
        all_insights.extend(r.insights)

    print(f"\n  Insights generados ({len(all_insights)} total):")
    from collections import Counter
    insight_counts = Counter(all_insights)
    for insight, count in insight_counts.most_common(5):
        print(f"    [{count}x] {insight[:50]}...")

    # Grafico ASCII
    print("\n  Evolucion de consciencia:")
    print("  " + "-" * 60)

    # Simplificar a 10 puntos
    n_points = 10
    step_size = n_steps // n_points

    print(f"  {'':5} | Con Suenos                    | Sin Suenos")
    for i in range(n_points):
        start = i * step_size
        end = (i + 1) * step_size

        avg_d = np.mean(consciousness_dreams[start:end])
        avg_c = np.mean(consciousness_control[start:end])

        bar_d = '#' * int(avg_d * 30)
        bar_c = '#' * int(avg_c * 30)

        print(f"  {end:4d}  | {bar_d:<30} | {bar_c:<30}")

    print("\n" + "=" * 70)

    return {
        'dreams': consciousness_dreams,
        'control': consciousness_control,
        'reports': dream_reports
    }


if __name__ == "__main__":
    experimento_ciclos_sueno()
