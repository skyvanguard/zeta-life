#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento de Escalabilidad: ZetaOrganism con diferentes cantidades de agentes.

Objetivo: Evaluar como escala el sistema con 100, 200, 500 y 1000 agentes.
Metricas: Emergencia de Fi, coordinacion, tiempo de computo, comportamiento colectivo.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sys

# Importar ZetaOrganism
from zeta_life.organism import ZetaOrganism, CellEntity
from zeta_life.organism import CellRole


@dataclass
class ScalabilityMetrics:
    """Metricas de un experimento de escalabilidad."""
    n_cells: int
    grid_size: int
    n_steps: int

    # Metricas de emergencia
    fi_inicial: int
    fi_final: int
    fi_max: int
    fi_ratio: float  # fi_final / n_cells

    # Metricas de coordinacion
    coord_inicial: float
    coord_final: float
    coord_promedio: float

    # Metricas de rendimiento
    tiempo_total: float
    tiempo_por_step: float
    steps_por_segundo: float

    # Metricas de densidad
    densidad: float  # n_cells / grid_size^2

    # Historial
    fi_history: List[int] = None
    coord_history: List[float] = None


def calcular_coordinacion(org: ZetaOrganism) -> float:
    """Calcula metrica de coordinacion del organismo."""
    if len(org.cells) == 0:
        return 0.0

    fi_cells = [c for c in org.cells if c.role_idx == 1]
    mass_cells = [c for c in org.cells if c.role_idx == 0]

    if len(fi_cells) == 0 or len(mass_cells) == 0:
        return 0.0

    # Coordinacion basada en distancia promedio Mass-Fi
    total_coord = 0.0
    for mass in mass_cells:
        min_dist = float('inf')
        for fi in fi_cells:
            dist = np.sqrt(
                (mass.position[0] - fi.position[0])**2 +
                (mass.position[1] - fi.position[1])**2
            )
            min_dist = min(min_dist, dist)
        # Normalizar por el tamaño del grid
        total_coord += 1.0 - (min_dist / (org.grid_size * np.sqrt(2)))

    return total_coord / len(mass_cells)


def contar_fi(org: ZetaOrganism) -> int:
    """Cuenta celulas Fi."""
    return sum(1 for c in org.cells if c.role_idx == 1)


def run_scalability_experiment(
    n_cells: int,
    grid_size: int = None,
    n_steps: int = 200,
    verbose: bool = True
) -> ScalabilityMetrics:
    """
    Ejecuta experimento con n_cells agentes.

    Args:
        n_cells: Numero de celulas/agentes
        grid_size: Tamaño del grid (auto-calculado si None)
        n_steps: Pasos de simulacion
        verbose: Mostrar progreso

    Returns:
        ScalabilityMetrics con resultados
    """
    # Auto-calcular grid_size basado en densidad objetivo (~0.02-0.03)
    if grid_size is None:
        # Densidad objetivo: 0.025 (2.5% del grid ocupado)
        grid_size = int(np.ceil(np.sqrt(n_cells / 0.025)))
        grid_size = max(64, min(256, grid_size))  # Entre 64 y 256

    densidad = n_cells / (grid_size ** 2)

    if verbose:
        print(f'\n{"="*60}')
        print(f'EXPERIMENTO: {n_cells} agentes')
        print(f'{"="*60}')
        print(f'  Grid: {grid_size}x{grid_size}')
        print(f'  Densidad: {densidad:.4f} ({densidad*100:.2f}%)')
        print(f'  Steps: {n_steps}')

    # Crear organismo
    org = ZetaOrganism(
        grid_size=grid_size,
        n_cells=n_cells,
        state_dim=32,
        hidden_dim=64,
        M=15,
        sigma=0.1
    )
    org.initialize(seed_fi=True)

    # Metricas iniciales
    fi_inicial = contar_fi(org)
    coord_inicial = calcular_coordinacion(org)

    # Historial
    fi_history = [fi_inicial]
    coord_history = [coord_inicial]
    fi_max = fi_inicial

    # Ejecutar simulacion
    start_time = time()

    for step in range(n_steps):
        org.step()

        fi = contar_fi(org)
        coord = calcular_coordinacion(org)

        fi_history.append(fi)
        coord_history.append(coord)
        fi_max = max(fi_max, fi)

        if verbose and (step + 1) % 50 == 0:
            elapsed = time() - start_time
            rate = (step + 1) / elapsed
            print(f'  Step {step+1}/{n_steps}: Fi={fi}, Coord={coord:.3f}, Rate={rate:.1f} steps/s')

    tiempo_total = time() - start_time

    # Metricas finales
    fi_final = contar_fi(org)
    coord_final = calcular_coordinacion(org)
    coord_promedio = np.mean(coord_history)

    metrics = ScalabilityMetrics(
        n_cells=n_cells,
        grid_size=grid_size,
        n_steps=n_steps,
        fi_inicial=fi_inicial,
        fi_final=fi_final,
        fi_max=fi_max,
        fi_ratio=fi_final / n_cells,
        coord_inicial=coord_inicial,
        coord_final=coord_final,
        coord_promedio=coord_promedio,
        tiempo_total=tiempo_total,
        tiempo_por_step=tiempo_total / n_steps,
        steps_por_segundo=n_steps / tiempo_total,
        densidad=densidad,
        fi_history=fi_history,
        coord_history=coord_history
    )

    if verbose:
        print(f'\nResultados:')
        print(f'  Fi: {fi_inicial} -> {fi_final} (max: {fi_max})')
        print(f'  Fi ratio: {metrics.fi_ratio:.3f} ({metrics.fi_ratio*100:.1f}%)')
        print(f'  Coordinacion: {coord_inicial:.3f} -> {coord_final:.3f}')
        print(f'  Tiempo total: {tiempo_total:.2f}s')
        print(f'  Velocidad: {metrics.steps_por_segundo:.1f} steps/s')

    return metrics, org


def run_full_scalability_study(
    agent_counts: List[int] = [100, 200, 500, 1000],
    n_steps: int = 200
) -> Dict[int, ScalabilityMetrics]:
    """
    Ejecuta estudio completo de escalabilidad.
    """
    print('\n' + '='*70)
    print('ESTUDIO DE ESCALABILIDAD - ZetaOrganism')
    print('='*70)
    print(f'Configuraciones: {agent_counts}')
    print(f'Steps por experimento: {n_steps}')

    results = {}
    orgs = {}

    for n_cells in agent_counts:
        metrics, org = run_scalability_experiment(
            n_cells=n_cells,
            n_steps=n_steps,
            verbose=True
        )
        results[n_cells] = metrics
        orgs[n_cells] = org

    return results, orgs


def plot_scalability_results(
    results: Dict[int, ScalabilityMetrics],
    orgs: Dict[int, 'ZetaOrganism'],
    save_path: str = 'zeta_organism_escalabilidad.png'
):
    """Genera visualizacion de resultados."""

    fig = plt.figure(figsize=(16, 12))

    agent_counts = sorted(results.keys())

    # 1. Emergencia de Fi vs numero de agentes
    ax1 = fig.add_subplot(2, 3, 1)
    fi_finals = [results[n].fi_final for n in agent_counts]
    fi_ratios = [results[n].fi_ratio * 100 for n in agent_counts]

    ax1_twin = ax1.twinx()
    bars = ax1.bar(range(len(agent_counts)), fi_finals, color='steelblue', alpha=0.7, label='Fi final')
    line, = ax1_twin.plot(range(len(agent_counts)), fi_ratios, 'ro-', linewidth=2, markersize=8, label='% Fi')

    ax1.set_xticks(range(len(agent_counts)))
    ax1.set_xticklabels(agent_counts)
    ax1.set_xlabel('Numero de agentes')
    ax1.set_ylabel('Fi final (absoluto)', color='steelblue')
    ax1_twin.set_ylabel('% de agentes como Fi', color='red')
    ax1.set_title('Emergencia de Liderazgo')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # 2. Coordinacion
    ax2 = fig.add_subplot(2, 3, 2)
    coord_finals = [results[n].coord_final for n in agent_counts]
    coord_proms = [results[n].coord_promedio for n in agent_counts]

    x = np.arange(len(agent_counts))
    width = 0.35
    ax2.bar(x - width/2, coord_finals, width, label='Final', color='green', alpha=0.7)
    ax2.bar(x + width/2, coord_proms, width, label='Promedio', color='lightgreen', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(agent_counts)
    ax2.set_xlabel('Numero de agentes')
    ax2.set_ylabel('Coordinacion')
    ax2.set_title('Coordinacion Colectiva')
    ax2.legend()
    ax2.set_ylim(0, 1)

    # 3. Rendimiento (tiempo)
    ax3 = fig.add_subplot(2, 3, 3)
    tiempos = [results[n].tiempo_total for n in agent_counts]
    velocidades = [results[n].steps_por_segundo for n in agent_counts]

    ax3_twin = ax3.twinx()
    ax3.bar(range(len(agent_counts)), tiempos, color='orange', alpha=0.7, label='Tiempo total')
    ax3_twin.plot(range(len(agent_counts)), velocidades, 'bs-', linewidth=2, markersize=8, label='Steps/s')

    ax3.set_xticks(range(len(agent_counts)))
    ax3.set_xticklabels(agent_counts)
    ax3.set_xlabel('Numero de agentes')
    ax3.set_ylabel('Tiempo total (s)', color='orange')
    ax3_twin.set_ylabel('Steps por segundo', color='blue')
    ax3.set_title('Rendimiento Computacional')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    # 4-6. Estados finales de cada configuracion (seleccionar 3)
    configs_to_plot = agent_counts[:3] if len(agent_counts) >= 3 else agent_counts

    for idx, n_cells in enumerate(configs_to_plot):
        ax = fig.add_subplot(2, 3, 4 + idx)
        org = orgs[n_cells]
        metrics = results[n_cells]

        # Fondo del grid
        ax.set_xlim(0, org.grid_size)
        ax.set_ylim(0, org.grid_size)

        # Dibujar celulas
        for cell in org.cells:
            x, y = cell.position
            role_idx = cell.role_idx

            if role_idx == 1:  # Fi
                color = 'red'
                size = 80
                marker = '*'
            elif role_idx == 2:  # Corrupt
                color = 'purple'
                size = 60
                marker = 'x'
            else:  # Mass
                color = 'blue'
                size = 20 + cell.energy * 40
                marker = 'o'

            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.6)

        ax.set_title(f'{n_cells} agentes (Fi={metrics.fi_final}, Coord={metrics.coord_final:.2f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nVisualizacion guardada en: {save_path}')
    plt.close()


def plot_evolution_comparison(
    results: Dict[int, ScalabilityMetrics],
    save_path: str = 'zeta_organism_escalabilidad_evolucion.png'
):
    """Compara evolucion temporal de Fi y coordinacion."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results)))

    # Fi evolution
    ax1 = axes[0]
    for (n_cells, metrics), color in zip(sorted(results.items()), colors):
        steps = range(len(metrics.fi_history))
        ax1.plot(steps, metrics.fi_history, color=color, linewidth=2,
                label=f'{n_cells} agentes')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Numero de Fi')
    ax1.set_title('Evolucion de Liderazgo (Fi)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Coordination evolution
    ax2 = axes[1]
    for (n_cells, metrics), color in zip(sorted(results.items()), colors):
        steps = range(len(metrics.coord_history))
        ax2.plot(steps, metrics.coord_history, color=color, linewidth=2,
                label=f'{n_cells} agentes')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Coordinacion')
    ax2.set_title('Evolucion de Coordinacion')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Evolucion guardada en: {save_path}')
    plt.close()


def print_summary_table(results: Dict[int, ScalabilityMetrics]):
    """Imprime tabla resumen de resultados."""

    print('\n' + '='*90)
    print('RESUMEN DE ESCALABILIDAD')
    print('='*90)

    headers = ['Agentes', 'Grid', 'Densidad', 'Fi Final', '% Fi', 'Coord', 'Tiempo(s)', 'Steps/s']
    print(f'{headers[0]:>10} {headers[1]:>8} {headers[2]:>10} {headers[3]:>10} {headers[4]:>8} {headers[5]:>8} {headers[6]:>10} {headers[7]:>10}')
    print('-'*90)

    for n_cells in sorted(results.keys()):
        m = results[n_cells]
        print(f'{m.n_cells:>10} {m.grid_size:>8} {m.densidad:>10.4f} {m.fi_final:>10} {m.fi_ratio*100:>7.1f}% {m.coord_final:>8.3f} {m.tiempo_total:>10.2f} {m.steps_por_segundo:>10.1f}')

    print('-'*90)

    # Analisis de escalabilidad
    agent_counts = sorted(results.keys())
    if len(agent_counts) >= 2:
        # Escalabilidad de Fi
        fi_scale = results[agent_counts[-1]].fi_final / results[agent_counts[0]].fi_final
        agent_scale = agent_counts[-1] / agent_counts[0]
        fi_efficiency = fi_scale / agent_scale

        # Escalabilidad de tiempo
        time_scale = results[agent_counts[-1]].tiempo_total / results[agent_counts[0]].tiempo_total

        print(f'\nANALISIS DE ESCALABILIDAD:')
        print(f'  Aumento de agentes: {agent_counts[0]} -> {agent_counts[-1]} ({agent_scale:.1f}x)')
        print(f'  Aumento de Fi: {results[agent_counts[0]].fi_final} -> {results[agent_counts[-1]].fi_final} ({fi_scale:.1f}x)')
        print(f'  Eficiencia de Fi: {fi_efficiency:.2f} (1.0 = lineal)')
        print(f'  Aumento de tiempo: {time_scale:.1f}x')
        print(f'  Escalabilidad temporal: {"Sub-lineal (bueno)" if time_scale < agent_scale else "Super-lineal (malo)"}')


if __name__ == '__main__':
    # Configuracion del experimento
    # Usar menos agentes para prueba rapida, o mas para estudio completo

    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Prueba rapida
        agent_counts = [100, 200]
        n_steps = 100
    elif len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Estudio completo
        agent_counts = [100, 200, 500, 1000, 2000]
        n_steps = 300
    else:
        # Default
        agent_counts = [100, 200, 500, 1000]
        n_steps = 200

    # Ejecutar estudio
    results, orgs = run_full_scalability_study(
        agent_counts=agent_counts,
        n_steps=n_steps
    )

    # Imprimir resumen
    print_summary_table(results)

    # Generar visualizaciones
    plot_scalability_results(results, orgs)
    plot_evolution_comparison(results)

    print('\nExperimento completado!')
