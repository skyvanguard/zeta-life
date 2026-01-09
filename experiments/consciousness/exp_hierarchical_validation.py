# -*- coding: utf-8 -*-
"""
Experimento de Validación: Consciencia Jerárquica.

Valida las hipótesis del sistema de consciencia jerárquica:
1. La consciencia emerge de la integración bottom-up
2. La modulación top-down mejora la coherencia
3. El sistema es resiliente ante perturbaciones
4. Los arquetipos complementarios reciben más atención

Fecha: 2026-01-03
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
import json

from zeta_life.psyche import Archetype
from zeta_life.consciousness import (
    HierarchicalSimulation,
    SimulationConfig,
    run_emergence_experiment,
    run_perturbation_experiment,
    run_archetype_bias_experiment
)
from cluster_assigner import ClusteringStrategy


def experiment_emergence_baseline(n_runs: int = 3) -> Dict:
    """
    Experimento 1: Emergencia de consciencia desde estado random.

    Hipótesis: El sistema debería mostrar aumento progresivo de
    consciencia desde un estado inicial aleatorio.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO 1: Emergencia de Consciencia (Baseline)")
    print("=" * 60)

    results = []

    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}...")
        result = run_emergence_experiment(n_steps=100, verbose=False)
        summary = result['summary']
        results.append(summary)

        print(f"    φ final: {summary['final_phi_global']:.3f}")
        print(f"    Consciencia final: {summary['final_consciousness']:.3f}")
        print(f"    Etapa: {summary['final_stage']}")

    # Estadísticas
    phi_values = [r['final_phi_global'] for r in results]
    consciousness_values = [r['final_consciousness'] for r in results]

    print(f"\n  Estadísticas ({n_runs} runs):")
    print(f"    φ global: {np.mean(phi_values):.3f} ± {np.std(phi_values):.3f}")
    print(f"    Consciencia: {np.mean(consciousness_values):.3f} ± {np.std(consciousness_values):.3f}")

    return {
        'results': results,
        'mean_phi': np.mean(phi_values),
        'std_phi': np.std(phi_values),
        'mean_consciousness': np.mean(consciousness_values),
        'std_consciousness': np.std(consciousness_values)
    }


def experiment_top_down_vs_no_top_down(n_steps: int = 100) -> Dict:
    """
    Experimento 2: Comparar con y sin modulación top-down.

    Hipótesis: La modulación top-down debería mejorar la coherencia
    del sistema comparado con solo bottom-up.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO 2: Con vs Sin Modulación Top-Down")
    print("=" * 60)

    # Con top-down (baseline)
    print("\n  Ejecutando CON modulación top-down...")
    config_with = SimulationConfig(
        n_cells=80,
        n_steps=n_steps,
        top_down_strength=0.5,
        bottom_up_strength=1.0
    )
    sim_with = HierarchicalSimulation(config_with)
    sim_with.initialize()
    sim_with.run(n_steps, verbose=False)
    summary_with = sim_with.get_summary()

    # Sin top-down
    print("  Ejecutando SIN modulación top-down...")
    config_without = SimulationConfig(
        n_cells=80,
        n_steps=n_steps,
        top_down_strength=0.0,  # Sin modulación
        bottom_up_strength=1.0
    )
    sim_without = HierarchicalSimulation(config_without)
    sim_without.initialize()
    sim_without.run(n_steps, verbose=False)
    summary_without = sim_without.get_summary()

    # Comparar
    print("\n  Resultados:")
    print(f"    CON top-down:")
    print(f"      φ global: {summary_with['final_phi_global']:.3f}")
    print(f"      Consciencia: {summary_with['final_consciousness']:.3f}")
    print(f"      Coherencia prom: {summary_with['avg_coherence']:.3f}")

    print(f"    SIN top-down:")
    print(f"      φ global: {summary_without['final_phi_global']:.3f}")
    print(f"      Consciencia: {summary_without['final_consciousness']:.3f}")
    print(f"      Coherencia prom: {summary_without['avg_coherence']:.3f}")

    improvement_phi = summary_with['final_phi_global'] - summary_without['final_phi_global']
    improvement_consciousness = summary_with['final_consciousness'] - summary_without['final_consciousness']
    improvement_coherence = summary_with['avg_coherence'] - summary_without['avg_coherence']

    print(f"\n  Diferencia (con - sin):")
    print(f"    φ global: {improvement_phi:+.3f}")
    print(f"    Consciencia: {improvement_consciousness:+.3f}")
    print(f"    Coherencia: {improvement_coherence:+.3f}")

    return {
        'with_top_down': summary_with,
        'without_top_down': summary_without,
        'improvement_phi': improvement_phi,
        'improvement_consciousness': improvement_consciousness,
        'improvement_coherence': improvement_coherence
    }


def experiment_clustering_strategies(n_steps: int = 100) -> Dict:
    """
    Experimento 3: Comparar estrategias de clustering.

    Hipótesis: El clustering híbrido (espacial + psíquico) debería
    producir mejor coherencia que solo espacial o solo psíquico.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO 3: Estrategias de Clustering")
    print("=" * 60)

    strategies = [
        ('SPATIAL', ClusteringStrategy.SPATIAL),
        ('PSYCHE', ClusteringStrategy.PSYCHE),
        ('HYBRID', ClusteringStrategy.HYBRID),
        ('ADAPTIVE', ClusteringStrategy.ADAPTIVE)
    ]

    results = {}

    for name, strategy in strategies:
        print(f"\n  Ejecutando estrategia {name}...")
        config = SimulationConfig(
            n_cells=80,
            n_steps=n_steps,
            clustering_strategy=strategy
        )
        sim = HierarchicalSimulation(config)
        sim.initialize()
        sim.run(n_steps, verbose=False)
        summary = sim.get_summary()
        results[name] = summary

        print(f"    φ global: {summary['final_phi_global']:.3f}")
        print(f"    Consciencia: {summary['final_consciousness']:.3f}")
        print(f"    Coherencia: {summary['avg_coherence']:.3f}")

    # Determinar mejor estrategia
    best_strategy = max(results.keys(), key=lambda k: results[k]['avg_coherence'])
    print(f"\n  Mejor estrategia por coherencia: {best_strategy}")

    return results


def experiment_perturbation_recovery(n_steps: int = 200) -> Dict:
    """
    Experimento 4: Resiliencia ante perturbaciones.

    Hipótesis: El sistema debería recuperarse después de perturbaciones,
    demostrando homeostasis.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO 4: Resiliencia ante Perturbaciones")
    print("=" * 60)

    result = run_perturbation_experiment(n_steps=n_steps, verbose=True)

    sim = result['simulation']
    metrics = sim.metrics_history
    interval = sim.config.perturbation_interval

    # Calcular pasos de perturbación dinámicamente
    perturbation_steps = [i for i in range(interval, n_steps, interval)]
    recovery_window = min(25, interval - 5)  # Ventana para medir recuperación
    recoveries = []

    for p_step in perturbation_steps:
        if p_step >= len(metrics) or p_step + recovery_window >= len(metrics):
            continue

        # Valores antes, durante y después
        before = metrics[p_step - 1].phi_global
        at = metrics[p_step].phi_global
        after = metrics[p_step + recovery_window].phi_global

        drop = before - at
        recovery = after - at
        recovery_percent = (recovery / drop * 100) if drop > 0 else 0

        recoveries.append({
            'step': p_step,
            'drop': drop,
            'recovery': recovery,
            'recovery_percent': recovery_percent
        })

    print("\n  Análisis de recuperación:")
    for rec in recoveries:
        print(f"    Paso {rec['step']}: "
              f"caída={rec['drop']:.3f}, "
              f"recuperación={rec['recovery']:.3f} "
              f"({rec['recovery_percent']:.1f}%)")

    avg_recovery = np.mean([r['recovery_percent'] for r in recoveries]) if recoveries else 0
    print(f"\n  Recuperación promedio: {avg_recovery:.1f}%")

    return {
        'simulation': sim,
        'recoveries': recoveries,
        'avg_recovery_percent': avg_recovery
    }


def experiment_archetype_influence(n_steps: int = 100) -> Dict:
    """
    Experimento 5: Influencia del sesgo arquetípico inicial.

    Hipótesis: Un sesgo hacia un arquetipo debería reflejarse en el
    arquetipo dominante del organismo.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO 5: Influencia del Sesgo Arquetípico")
    print("=" * 60)

    results = {}

    for archetype in Archetype:
        print(f"\n  Probando sesgo hacia {archetype.name}...")
        result = run_archetype_bias_experiment(
            dominant_archetype=archetype,
            bias_strength=0.5,
            n_steps=n_steps,
            verbose=False
        )

        matches = result['bias_matches_dominant']
        final_dominant = result['final_dominant']

        results[archetype.name] = {
            'bias_matches': matches,
            'final_dominant': final_dominant.name
        }

        status = "✓" if matches else "✗"
        print(f"    Final: {final_dominant.name} {status}")

    # Calcular tasa de correspondencia
    match_rate = sum(1 for r in results.values() if r['bias_matches']) / len(results)
    print(f"\n  Tasa de correspondencia sesgo→dominante: {match_rate*100:.1f}%")

    return {
        'results': results,
        'match_rate': match_rate
    }


def run_all_experiments() -> Dict:
    """Ejecuta todos los experimentos de validación."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("\n" + "=" * 70)
    print("  VALIDACIÓN COMPLETA: CONSCIENCIA JERÁRQUICA")
    print("=" * 70)
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    all_results = {}

    # Experimento 1: Emergencia baseline
    all_results['emergence'] = experiment_emergence_baseline(n_runs=3)

    # Experimento 2: Top-down vs no top-down
    all_results['top_down'] = experiment_top_down_vs_no_top_down(n_steps=100)

    # Experimento 3: Estrategias de clustering
    all_results['clustering'] = experiment_clustering_strategies(n_steps=100)

    # Experimento 4: Resiliencia (200 pasos para permitir recuperación con intervalo=50)
    all_results['perturbation'] = experiment_perturbation_recovery(n_steps=200)

    # Experimento 5: Influencia arquetípica
    all_results['archetype'] = experiment_archetype_influence(n_steps=100)

    # Resumen
    print("\n" + "=" * 70)
    print("  RESUMEN DE VALIDACIÓN")
    print("=" * 70)

    print("\n  1. Emergencia de Consciencia:")
    print(f"     φ promedio: {all_results['emergence']['mean_phi']:.3f}")
    print(f"     Consciencia promedio: {all_results['emergence']['mean_consciousness']:.3f}")

    print("\n  2. Efecto Top-Down:")
    td = all_results['top_down']
    print(f"     Mejora en coherencia: {td['improvement_coherence']:+.3f}")
    verdict = "CONFIRMADO" if td['improvement_coherence'] > 0 else "NO CONFIRMADO"
    print(f"     Hipótesis: {verdict}")

    print("\n  3. Mejor Estrategia de Clustering:")
    clustering = all_results['clustering']
    best = max(clustering.keys(), key=lambda k: clustering[k]['avg_coherence'])
    print(f"     {best} (coherencia: {clustering[best]['avg_coherence']:.3f})")

    print("\n  4. Resiliencia:")
    print(f"     Recuperación promedio: {all_results['perturbation']['avg_recovery_percent']:.1f}%")
    verdict = "RESILIENTE" if all_results['perturbation']['avg_recovery_percent'] > 50 else "FRÁGIL"
    print(f"     Sistema: {verdict}")

    print("\n  5. Influencia Arquetípica:")
    print(f"     Tasa de correspondencia: {all_results['archetype']['match_rate']*100:.1f}%")

    # Guardar resultados
    # Filtrar datos no serializables
    serializable_results = {
        'timestamp': timestamp,
        'emergence': {
            'mean_phi': all_results['emergence']['mean_phi'],
            'std_phi': all_results['emergence']['std_phi'],
            'mean_consciousness': all_results['emergence']['mean_consciousness'],
            'std_consciousness': all_results['emergence']['std_consciousness']
        },
        'top_down': {
            'improvement_phi': all_results['top_down']['improvement_phi'],
            'improvement_consciousness': all_results['top_down']['improvement_consciousness'],
            'improvement_coherence': all_results['top_down']['improvement_coherence']
        },
        'clustering': {k: v['avg_coherence'] for k, v in all_results['clustering'].items()},
        'perturbation': {
            'avg_recovery_percent': all_results['perturbation']['avg_recovery_percent']
        },
        'archetype': {
            'match_rate': all_results['archetype']['match_rate']
        }
    }

    with open(f'validation_results_{timestamp}.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n  Resultados guardados en: validation_results_{timestamp}.json")

    return all_results


def plot_validation_summary(results: Dict, save_path: str = None):
    """Genera gráfica resumen de la validación."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Emergencia de consciencia
    ax1 = axes[0, 0]
    emergence = results['emergence']['results']
    phi_values = [r['final_phi_global'] for r in emergence]
    consciousness_values = [r['final_consciousness'] for r in emergence]

    x = range(len(emergence))
    ax1.bar([i-0.15 for i in x], phi_values, 0.3, label='φ global', color='blue')
    ax1.bar([i+0.15 for i in x], consciousness_values, 0.3, label='Consciencia', color='red')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Valor')
    ax1.set_title('Emergencia de Consciencia')
    ax1.legend()
    ax1.set_xticks(x)

    # 2. Top-down vs No top-down
    ax2 = axes[0, 1]
    td = results['top_down']
    categories = ['φ global', 'Consciencia', 'Coherencia']
    with_td = [td['with_top_down']['final_phi_global'],
               td['with_top_down']['final_consciousness'],
               td['with_top_down']['avg_coherence']]
    without_td = [td['without_top_down']['final_phi_global'],
                  td['without_top_down']['final_consciousness'],
                  td['without_top_down']['avg_coherence']]

    x = np.arange(len(categories))
    ax2.bar(x - 0.15, with_td, 0.3, label='Con Top-Down', color='green')
    ax2.bar(x + 0.15, without_td, 0.3, label='Sin Top-Down', color='gray')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Valor')
    ax2.set_title('Efecto de Modulación Top-Down')
    ax2.legend()

    # 3. Estrategias de clustering
    ax3 = axes[1, 0]
    clustering = results['clustering']
    strategies = list(clustering.keys())
    coherences = [clustering[s]['avg_coherence'] for s in strategies]

    colors = ['royalblue', 'forestgreen', 'darkorange', 'purple']
    ax3.bar(strategies, coherences, color=colors)
    ax3.set_ylabel('Coherencia Promedio')
    ax3.set_title('Comparación de Estrategias de Clustering')

    # 4. Influencia arquetípica
    ax4 = axes[1, 1]
    archetype = results['archetype']['results']
    archetypes = list(archetype.keys())
    matches = [1 if archetype[a]['bias_matches'] else 0 for a in archetypes]

    colors = ['green' if m else 'red' for m in matches]
    ax4.bar(archetypes, [1]*len(archetypes), color=colors, alpha=0.7)
    ax4.set_ylabel('Correspondencia')
    ax4.set_title('Influencia del Sesgo Arquetípico\n(Verde=Match, Rojo=No match)')
    ax4.set_ylim(0, 1.2)

    for i, (a, m) in enumerate(zip(archetypes, matches)):
        symbol = "✓" if m else "✗"
        ax4.text(i, 0.5, symbol, ha='center', va='center', fontsize=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Ejecutar todos los experimentos
    results = run_all_experiments()

    # Generar visualización
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plot_validation_summary(
        results,
        save_path=f"validation_summary_{timestamp}.png"
    )

    print("\n" + "=" * 70)
    print("  VALIDACIÓN COMPLETADA")
    print("=" * 70)
