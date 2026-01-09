#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento de Estres Masivo: Escenarios desafiantes con 500-1000 agentes.

Escenarios:
1. Dano severo repetido (eliminar 80% de Fi multiples veces)
2. Escasez extrema de energia
3. Migracion forzada con gradientes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple
import sys

from zeta_life.organism import ZetaOrganism, CellEntity
from zeta_life.organism import CellRole


def contar_fi(org: ZetaOrganism) -> int:
    return sum(1 for c in org.cells if c.role_idx == 1)


def calcular_coordinacion(org: ZetaOrganism) -> float:
    if len(org.cells) == 0:
        return 0.0
    fi_cells = [c for c in org.cells if c.role_idx == 1]
    mass_cells = [c for c in org.cells if c.role_idx == 0]
    if len(fi_cells) == 0 or len(mass_cells) == 0:
        return 0.0
    total_coord = 0.0
    for mass in mass_cells:
        min_dist = float('inf')
        for fi in fi_cells:
            dist = np.sqrt(
                (mass.position[0] - fi.position[0])**2 +
                (mass.position[1] - fi.position[1])**2
            )
            min_dist = min(min_dist, dist)
        total_coord += 1.0 - (min_dist / (org.grid_size * np.sqrt(2)))
    return total_coord / len(mass_cells)


def calcular_centroide(org: ZetaOrganism) -> Tuple[float, float]:
    """Calcula centroide del organismo."""
    if len(org.cells) == 0:
        return (org.grid_size/2, org.grid_size/2)
    x_sum = sum(c.position[0] for c in org.cells)
    y_sum = sum(c.position[1] for c in org.cells)
    return (x_sum / len(org.cells), y_sum / len(org.cells))


# ============================================================
# ESCENARIO 1: DANO SEVERO REPETIDO
# ============================================================

def escenario_dano_severo(n_cells: int = 500, grid_size: int = None, n_rondas: int = 5):
    """
    Dano severo repetido: Eliminar 80% de Fi en multiples rondas.
    Evalua capacidad de regeneracion bajo estres continuo.
    """
    print(f'\n{"="*70}')
    print(f'ESCENARIO 1: DANO SEVERO REPETIDO ({n_cells} agentes)')
    print(f'{"="*70}')

    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(n_cells / 0.025)))
        grid_size = max(64, min(256, grid_size))

    # Crear organismo
    org = ZetaOrganism(grid_size=grid_size, n_cells=n_cells)
    org.initialize(seed_fi=True)

    # Estabilizacion inicial
    print('\n[FASE 0] Estabilizacion inicial (100 steps)...')
    for _ in range(100):
        org.step()

    fi_baseline = contar_fi(org)
    coord_baseline = calcular_coordinacion(org)
    print(f'  Baseline: Fi={fi_baseline}, Coord={coord_baseline:.3f}')

    # Historial
    fi_history = [fi_baseline]
    coord_history = [coord_baseline]
    damage_points = []
    recovery_rates = []

    # Rondas de dano
    for ronda in range(n_rondas):
        print(f'\n[RONDA {ronda+1}/{n_rondas}] Dano 80%...')

        # Estado pre-dano
        fi_pre = contar_fi(org)
        coord_pre = calcular_coordinacion(org)

        # Aplicar dano: eliminar 80% de Fi
        fi_cells = [c for c in org.cells if c.role_idx == 1]
        n_to_remove = int(len(fi_cells) * 0.8)

        if n_to_remove > 0:
            to_remove = np.random.choice(len(fi_cells), n_to_remove, replace=False)
            for idx in sorted(to_remove, reverse=True):
                fi_cells[idx].role = torch.tensor([1.0, 0.0, 0.0])  # Convertir a MASS
                fi_cells[idx].energy *= 0.5

        fi_post = contar_fi(org)
        damage_points.append(len(fi_history))
        print(f'  Pre-dano: Fi={fi_pre}, Post-dano: Fi={fi_post}')

        # Recuperacion (100 steps)
        for step in range(100):
            org.step()
            fi_history.append(contar_fi(org))
            coord_history.append(calcular_coordinacion(org))

        fi_recovered = contar_fi(org)
        recovery_rate = fi_recovered / max(1, fi_pre) * 100
        recovery_rates.append(recovery_rate)
        print(f'  Recuperado: Fi={fi_recovered} ({recovery_rate:.1f}% del pre-dano)')

    # Resultados
    results = {
        'n_cells': n_cells,
        'fi_baseline': fi_baseline,
        'fi_final': contar_fi(org),
        'coord_final': calcular_coordinacion(org),
        'recovery_rates': recovery_rates,
        'avg_recovery': np.mean(recovery_rates),
        'fi_history': fi_history,
        'coord_history': coord_history,
        'damage_points': damage_points
    }

    print(f'\n[RESUMEN]')
    print(f'  Fi baseline: {fi_baseline}')
    print(f'  Fi final: {results["fi_final"]}')
    print(f'  Recuperacion promedio: {results["avg_recovery"]:.1f}%')
    print(f'  Coord final: {results["coord_final"]:.3f}')

    return results, org


# ============================================================
# ESCENARIO 2: ESCASEZ EXTREMA DE ENERGIA
# ============================================================

def escenario_escasez_extrema(n_cells: int = 500, grid_size: int = None):
    """
    Escasez extrema: Reducir energia gradualmente hasta colapso,
    luego restaurar y medir recuperacion.
    """
    print(f'\n{"="*70}')
    print(f'ESCENARIO 2: ESCASEZ EXTREMA ({n_cells} agentes)')
    print(f'{"="*70}')

    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(n_cells / 0.025)))
        grid_size = max(64, min(256, grid_size))

    # Crear organismo
    org = ZetaOrganism(grid_size=grid_size, n_cells=n_cells)
    org.initialize(seed_fi=True)

    # Estabilizacion
    print('\n[FASE 0] Estabilizacion (100 steps)...')
    for _ in range(100):
        org.step()

    fi_baseline = contar_fi(org)
    print(f'  Baseline: Fi={fi_baseline}')

    # Historial
    fi_history = [fi_baseline]
    energy_history = [1.0]

    # Niveles de escasez
    scarcity_levels = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05]

    for scarcity in scarcity_levels:
        print(f'\n[ESCASEZ {scarcity*100:.0f}%]')

        # Aplicar escasez
        for cell in org.cells:
            cell.energy = min(cell.energy, scarcity)

        # Evolucionar 50 steps
        for _ in range(50):
            org.step()
            # Mantener cap de energia
            for cell in org.cells:
                cell.energy = min(cell.energy, scarcity)

            fi_history.append(contar_fi(org))
            energy_history.append(scarcity)

        fi_current = contar_fi(org)
        print(f'  Fi actual: {fi_current}')

        if fi_current == 0:
            print('  *** COLAPSO TOTAL ***')
            break

    # Recuperacion
    print('\n[RECUPERACION] Restaurando energia...')
    for cell in org.cells:
        cell.energy = np.random.uniform(0.5, 1.0)

    for step in range(100):
        org.step()
        fi_history.append(contar_fi(org))
        energy_history.append(1.0)

    fi_recovered = contar_fi(org)
    antifragility = (fi_recovered - fi_baseline) / max(1, fi_baseline) * 100

    results = {
        'n_cells': n_cells,
        'fi_baseline': fi_baseline,
        'fi_final': fi_recovered,
        'antifragility': antifragility,
        'fi_history': fi_history,
        'energy_history': energy_history
    }

    print(f'\n[RESUMEN]')
    print(f'  Fi baseline: {fi_baseline}')
    print(f'  Fi recuperado: {fi_recovered}')
    print(f'  Antifragilidad: {antifragility:+.1f}%')

    return results, org


# ============================================================
# ESCENARIO 3: MIGRACION FORZADA
# ============================================================

def escenario_migracion_forzada(n_cells: int = 500, grid_size: int = None):
    """
    Migracion forzada: Aplicar gradientes de energia para forzar
    movimiento colectivo del organismo.
    """
    print(f'\n{"="*70}')
    print(f'ESCENARIO 3: MIGRACION FORZADA ({n_cells} agentes)')
    print(f'{"="*70}')

    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(n_cells / 0.025)))
        grid_size = max(64, min(256, grid_size))

    # Crear organismo
    org = ZetaOrganism(grid_size=grid_size, n_cells=n_cells)
    org.initialize(seed_fi=True)

    # Estabilizacion
    print('\n[FASE 0] Estabilizacion (100 steps)...')
    for _ in range(100):
        org.step()

    centroide_inicial = calcular_centroide(org)
    fi_baseline = contar_fi(org)
    print(f'  Centroide inicial: ({centroide_inicial[0]:.1f}, {centroide_inicial[1]:.1f})')
    print(f'  Fi baseline: {fi_baseline}')

    # Historial
    centroide_history = [centroide_inicial]
    fi_history = [fi_baseline]

    # FASE 1: Gradiente hacia la derecha
    print('\n[FASE 1] Gradiente hacia DERECHA (150 steps)...')
    for step in range(150):
        # Aplicar gradiente: mas energia a la derecha
        for cell in org.cells:
            x, y = cell.position
            gradient_bonus = (x / grid_size) * 0.3
            cell.energy = min(1.0, cell.energy + gradient_bonus * 0.1)

        org.step()
        centroide_history.append(calcular_centroide(org))
        fi_history.append(contar_fi(org))

    centroide_fase1 = calcular_centroide(org)
    desplazamiento1 = centroide_fase1[0] - centroide_inicial[0]
    print(f'  Centroide: ({centroide_fase1[0]:.1f}, {centroide_fase1[1]:.1f})')
    print(f'  Desplazamiento X: {desplazamiento1:+.1f} celdas')

    # FASE 2: Gradiente hacia el centro
    print('\n[FASE 2] Gradiente RADIAL hacia centro (150 steps)...')
    center = grid_size / 2
    for step in range(150):
        for cell in org.cells:
            x, y = cell.position
            dist_to_center = np.sqrt((x - center)**2 + (y - center)**2)
            max_dist = center * np.sqrt(2)
            gradient_bonus = (1 - dist_to_center / max_dist) * 0.3
            cell.energy = min(1.0, cell.energy + gradient_bonus * 0.1)

        org.step()
        centroide_history.append(calcular_centroide(org))
        fi_history.append(contar_fi(org))

    centroide_fase2 = calcular_centroide(org)
    dist_to_center = np.sqrt(
        (centroide_fase2[0] - center)**2 +
        (centroide_fase2[1] - center)**2
    )
    print(f'  Centroide: ({centroide_fase2[0]:.1f}, {centroide_fase2[1]:.1f})')
    print(f'  Distancia al centro: {dist_to_center:.1f} celdas')

    # Calcular trayectoria total
    total_distance = 0
    for i in range(1, len(centroide_history)):
        dx = centroide_history[i][0] - centroide_history[i-1][0]
        dy = centroide_history[i][1] - centroide_history[i-1][1]
        total_distance += np.sqrt(dx**2 + dy**2)

    results = {
        'n_cells': n_cells,
        'centroide_inicial': centroide_inicial,
        'centroide_final': centroide_fase2,
        'desplazamiento_fase1': desplazamiento1,
        'dist_to_center': dist_to_center,
        'total_distance': total_distance,
        'fi_final': contar_fi(org),
        'centroide_history': centroide_history,
        'fi_history': fi_history
    }

    print(f'\n[RESUMEN]')
    print(f'  Distancia total recorrida: {total_distance:.1f} celdas')
    print(f'  Fi final: {results["fi_final"]}')

    return results, org


# ============================================================
# VISUALIZACION
# ============================================================

def plot_estres_results(
    dano_results: dict,
    escasez_results: dict,
    migracion_results: dict,
    save_path: str = 'zeta_organism_estres_masivo.png'
):
    """Genera visualizacion de todos los escenarios."""

    fig = plt.figure(figsize=(16, 10))

    # 1. Dano severo - Fi history
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(dano_results['fi_history'], 'b-', linewidth=1.5)
    for dp in dano_results['damage_points']:
        ax1.axvline(x=dp, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Numero de Fi')
    ax1.set_title(f'Dano Severo Repetido ({dano_results["n_cells"]} agentes)\n'
                  f'Recuperacion promedio: {dano_results["avg_recovery"]:.1f}%')
    ax1.grid(True, alpha=0.3)

    # 2. Dano severo - Tasas de recuperacion
    ax2 = fig.add_subplot(2, 3, 2)
    rondas = range(1, len(dano_results['recovery_rates']) + 1)
    colors = ['green' if r >= 100 else 'orange' if r >= 75 else 'red'
              for r in dano_results['recovery_rates']]
    ax2.bar(rondas, dano_results['recovery_rates'], color=colors, alpha=0.7)
    ax2.axhline(y=100, color='black', linestyle='--', label='100% recuperacion')
    ax2.set_xlabel('Ronda de dano')
    ax2.set_ylabel('% Recuperacion')
    ax2.set_title('Tasa de Recuperacion por Ronda')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Escasez - Fi vs Energia
    ax3 = fig.add_subplot(2, 3, 3)
    ax3_twin = ax3.twinx()
    steps = range(len(escasez_results['fi_history']))
    ax3.plot(steps, escasez_results['fi_history'], 'b-', linewidth=1.5, label='Fi')
    ax3_twin.plot(steps, escasez_results['energy_history'], 'r--', linewidth=1, label='Energia max', alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Numero de Fi', color='blue')
    ax3_twin.set_ylabel('Nivel de energia', color='red')
    ax3.set_title(f'Escasez Extrema ({escasez_results["n_cells"]} agentes)\n'
                  f'Antifragilidad: {escasez_results["antifragility"]:+.1f}%')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    # 4. Migracion - Trayectoria
    ax4 = fig.add_subplot(2, 3, 4)
    cx = [c[0] for c in migracion_results['centroide_history']]
    cy = [c[1] for c in migracion_results['centroide_history']]
    n_points = len(cx)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    for i in range(n_points - 1):
        ax4.plot(cx[i:i+2], cy[i:i+2], color=colors[i], linewidth=2)
    ax4.scatter(cx[0], cy[0], c='green', s=100, marker='o', label='Inicio', zorder=5)
    ax4.scatter(cx[-1], cy[-1], c='red', s=100, marker='*', label='Final', zorder=5)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title(f'Trayectoria de Migracion\nDistancia total: {migracion_results["total_distance"]:.1f} celdas')
    ax4.legend()
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # 5. Migracion - Fi durante migracion
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(migracion_results['fi_history'], 'b-', linewidth=1.5)
    ax5.axvline(x=150, color='orange', linestyle='--', label='Cambio de gradiente')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Numero de Fi')
    ax5.set_title(f'Fi Durante Migracion ({migracion_results["n_cells"]} agentes)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Resumen comparativo
    ax6 = fig.add_subplot(2, 3, 6)
    scenarios = ['Dano\nSevero', 'Escasez\nExtrema', 'Migracion\nForzada']
    metrics = [
        dano_results['avg_recovery'],
        100 + escasez_results['antifragility'],  # Normalizado a 100% base
        migracion_results['fi_final'] / migracion_results['n_cells'] * 500  # Escalado
    ]
    colors = ['steelblue', 'green', 'orange']
    bars = ax6.bar(scenarios, metrics, color=colors, alpha=0.7)
    ax6.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax6.set_ylabel('Metrica de Resiliencia (%)')
    ax6.set_title('Comparacion de Resiliencia')

    # Anotar valores
    for bar, val in zip(bars, metrics):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nVisualizacion guardada en: {save_path}')
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Configuracion
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        n_cells = 200
        n_rondas_dano = 3
    elif len(sys.argv) > 1 and sys.argv[1] == '--large':
        n_cells = 1000
        n_rondas_dano = 5
    else:
        n_cells = 500
        n_rondas_dano = 5

    print(f'\nEXPERIMENTO DE ESTRES MASIVO')
    print(f'Agentes: {n_cells}')
    print(f'='*70)

    start_total = time()

    # Ejecutar escenarios
    dano_results, _ = escenario_dano_severo(n_cells=n_cells, n_rondas=n_rondas_dano)
    escasez_results, _ = escenario_escasez_extrema(n_cells=n_cells)
    migracion_results, _ = escenario_migracion_forzada(n_cells=n_cells)

    tiempo_total = time() - start_total

    # Visualizar
    plot_estres_results(dano_results, escasez_results, migracion_results)

    # Resumen final
    print(f'\n{"="*70}')
    print(f'RESUMEN FINAL - ESTRES MASIVO ({n_cells} agentes)')
    print(f'{"="*70}')
    print(f'\n1. DANO SEVERO REPETIDO:')
    print(f'   Recuperacion promedio: {dano_results["avg_recovery"]:.1f}%')
    print(f'   Fi final vs baseline: {dano_results["fi_final"]}/{dano_results["fi_baseline"]}')

    print(f'\n2. ESCASEZ EXTREMA:')
    print(f'   Antifragilidad: {escasez_results["antifragility"]:+.1f}%')
    print(f'   Fi final vs baseline: {escasez_results["fi_final"]}/{escasez_results["fi_baseline"]}')

    print(f'\n3. MIGRACION FORZADA:')
    print(f'   Distancia total: {migracion_results["total_distance"]:.1f} celdas')
    print(f'   Fi mantenidos: {migracion_results["fi_final"]}')

    print(f'\nTiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)')
