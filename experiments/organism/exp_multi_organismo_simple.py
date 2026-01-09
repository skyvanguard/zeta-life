#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento Multi-Organismo Simple: Usando ZetaOrganism existente.
2-3 organismos independientes con interaccion basica.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple
import sys

from zeta_life.organism import ZetaOrganism, CellEntity


def contar_fi(org: ZetaOrganism) -> int:
    return sum(1 for c in org.cells if c.role_idx == 1)


def calcular_centroide(org: ZetaOrganism) -> Tuple[float, float]:
    if len(org.cells) == 0:
        return (org.grid_size/2, org.grid_size/2)
    x_sum = sum(c.position[0] for c in org.cells)
    y_sum = sum(c.position[1] for c in org.cells)
    return (x_sum / len(org.cells), y_sum / len(org.cells))


def calcular_shannon(populations: List[int]) -> float:
    """Calcula indice de Shannon para diversidad."""
    total = sum(populations)
    if total == 0:
        return 0
    probs = [p/total for p in populations if p > 0]
    return -sum(p * np.log(p) for p in probs) if probs else 0


class MultiOrganismSimple:
    """Sistema multi-organismo usando ZetaOrganism independientes."""

    def __init__(self, n_orgs: int, n_cells_per_org: int, grid_size: int):
        self.n_orgs = n_orgs
        self.n_cells_per_org = n_cells_per_org
        self.grid_size = grid_size

        # Crear organismos
        self.organisms: List[ZetaOrganism] = []
        for _ in range(n_orgs):
            org = ZetaOrganism(
                grid_size=grid_size,
                n_cells=n_cells_per_org,
                state_dim=32,
                hidden_dim=64
            )
            org.initialize(seed_fi=True)
            self.organisms.append(org)

        # Reubicar organismos en diferentes posiciones
        self._distribute_organisms()

    def _distribute_organisms(self):
        """Distribuye organismos en diferentes regiones del grid."""
        if self.n_orgs == 2:
            positions = [
                (self.grid_size // 4, self.grid_size // 2),
                (3 * self.grid_size // 4, self.grid_size // 2)
            ]
        elif self.n_orgs == 3:
            positions = [
                (self.grid_size // 2, self.grid_size // 4),
                (self.grid_size // 4, 3 * self.grid_size // 4),
                (3 * self.grid_size // 4, 3 * self.grid_size // 4)
            ]
        else:
            # Distribucion uniforme
            angle_step = 2 * np.pi / self.n_orgs
            center = self.grid_size // 2
            radius = self.grid_size // 3
            positions = [
                (int(center + radius * np.cos(i * angle_step)),
                 int(center + radius * np.sin(i * angle_step)))
                for i in range(self.n_orgs)
            ]

        # Reubicar celulas de cada organismo
        for org_id, (cx, cy) in enumerate(positions):
            org = self.organisms[org_id]
            for cell in org.cells:
                # Nueva posicion cercana al centroide asignado
                new_x = int(cx + np.random.randn() * 8)
                new_y = int(cy + np.random.randn() * 8)
                new_x = max(0, min(self.grid_size - 1, new_x))
                new_y = max(0, min(self.grid_size - 1, new_y))
                cell.position = (new_x, new_y)
            org._update_grids()

    def step(self):
        """Ejecuta un paso de simulacion."""
        # Cada organismo evoluciona independientemente
        for org in self.organisms:
            org.step()

        # Interaccion entre organismos: competencia por territorio
        self._apply_competition()

    def _apply_competition(self):
        """Aplica competencia entre organismos cercanos."""
        # Para cada par de organismos
        for i in range(self.n_orgs):
            for j in range(i + 1, self.n_orgs):
                org_i = self.organisms[i]
                org_j = self.organisms[j]

                # Buscar celulas cercanas
                for cell_i in org_i.cells:
                    for cell_j in org_j.cells:
                        dist = np.sqrt(
                            (cell_i.position[0] - cell_j.position[0])**2 +
                            (cell_i.position[1] - cell_j.position[1])**2
                        )

                        if dist < 5:
                            # Competencia: ambos pierden energia
                            cell_i.energy *= 0.98
                            cell_j.energy *= 0.98

                            # Conversion ocasional (Fi convierte Mass enemigo)
                            if cell_i.role_idx == 1 and cell_j.role_idx == 0:
                                if dist < 3 and np.random.random() < 0.02:
                                    # Transferir celula
                                    org_j.cells.remove(cell_j)
                                    cell_j.energy *= 0.7
                                    org_i.cells.append(cell_j)
                                    break
                            elif cell_j.role_idx == 1 and cell_i.role_idx == 0:
                                if dist < 3 and np.random.random() < 0.02:
                                    org_i.cells.remove(cell_i)
                                    cell_i.energy *= 0.7
                                    org_j.cells.append(cell_i)
                                    break

    def get_metrics(self) -> dict:
        """Obtiene metricas del sistema."""
        populations = [len(org.cells) for org in self.organisms]
        fi_counts = [contar_fi(org) for org in self.organisms]
        centroides = [calcular_centroide(org) for org in self.organisms]

        # Separacion promedio
        dists = []
        for i in range(len(centroides)):
            for j in range(i+1, len(centroides)):
                d = np.sqrt(
                    (centroides[i][0] - centroides[j][0])**2 +
                    (centroides[i][1] - centroides[j][1])**2
                )
                dists.append(d)
        separation = np.mean(dists) if dists else 0

        return {
            'populations': populations,
            'fi_counts': fi_counts,
            'centroides': centroides,
            'shannon': calcular_shannon(populations),
            'separation': separation,
            'total_cells': sum(populations)
        }


def run_multi_experiment(
    n_orgs: int,
    n_cells_per_org: int,
    n_steps: int = 200,
    grid_size: int = None
) -> Tuple[dict, MultiOrganismSimple]:
    """Ejecuta experimento multi-organismo."""

    print(f'\n{"="*70}')
    print(f'EXPERIMENTO: {n_orgs} ORGANISMOS ({n_cells_per_org} agentes c/u)')
    print(f'{"="*70}')

    if grid_size is None:
        grid_size = int(np.sqrt(n_orgs * n_cells_per_org / 0.02))
        grid_size = max(80, min(256, grid_size))

    print(f'Grid: {grid_size}x{grid_size}')
    print(f'Total agentes: {n_orgs * n_cells_per_org}')

    # Crear sistema
    system = MultiOrganismSimple(n_orgs, n_cells_per_org, grid_size)

    # Historial
    pop_history = [[] for _ in range(n_orgs)]
    fi_history = [[] for _ in range(n_orgs)]
    shannon_history = []

    # Ejecutar
    start_time = time()
    for step in range(n_steps):
        system.step()
        metrics = system.get_metrics()

        for org_id in range(n_orgs):
            pop_history[org_id].append(metrics['populations'][org_id])
            fi_history[org_id].append(metrics['fi_counts'][org_id])
        shannon_history.append(metrics['shannon'])

        if (step + 1) % 50 == 0:
            elapsed = time() - start_time
            rate = (step + 1) / elapsed
            print(f'  Step {step+1}/{n_steps}: Pop={metrics["populations"]}, '
                  f'Fi={metrics["fi_counts"]}, Rate={rate:.1f} steps/s')

    tiempo = time() - start_time
    final_metrics = system.get_metrics()

    results = {
        'n_orgs': n_orgs,
        'n_cells_per_org': n_cells_per_org,
        'grid_size': grid_size,
        'n_steps': n_steps,
        'pop_history': pop_history,
        'fi_history': fi_history,
        'shannon_history': shannon_history,
        'final': final_metrics,
        'tiempo': tiempo
    }

    print(f'\n[RESULTADOS]')
    print(f'  Poblaciones: {final_metrics["populations"]}')
    print(f'  Fi: {final_metrics["fi_counts"]}')
    print(f'  Shannon: {final_metrics["shannon"]:.3f}')
    print(f'  Tiempo: {tiempo:.1f}s')

    return results, system


def plot_results(
    results_2: dict,
    results_3: dict,
    system_2: MultiOrganismSimple,
    system_3: MultiOrganismSimple,
    save_path: str = 'zeta_organism_multi_simple.png'
):
    """Visualiza resultados."""
    fig = plt.figure(figsize=(16, 10))
    colors = ['blue', 'red', 'green']

    # 1. Poblacion 2 orgs
    ax1 = fig.add_subplot(2, 3, 1)
    for i in range(2):
        ax1.plot(results_2['pop_history'][i], color=colors[i], lw=1.5, label=f'Org {i}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Poblacion')
    ax1.set_title(f'2 Organismos ({results_2["n_cells_per_org"]} c/u)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Poblacion 3 orgs
    ax2 = fig.add_subplot(2, 3, 2)
    for i in range(3):
        ax2.plot(results_3['pop_history'][i], color=colors[i], lw=1.5, label=f'Org {i}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Poblacion')
    ax2.set_title(f'3 Organismos ({results_3["n_cells_per_org"]} c/u)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Shannon index
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(results_2['shannon_history'], 'b-', lw=1.5, label='2 orgs')
    ax3.plot(results_3['shannon_history'], 'r-', lw=1.5, label='3 orgs')
    ax3.axhline(y=np.log(2), color='b', ls='--', alpha=0.5, label='Max 2')
    ax3.axhline(y=np.log(3), color='r', ls='--', alpha=0.5, label='Max 3')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Shannon Index')
    ax3.set_title('Diversidad del Sistema')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Estado final 2 orgs
    ax4 = fig.add_subplot(2, 3, 4)
    for org_id, org in enumerate(system_2.organisms):
        for cell in org.cells:
            x, y = cell.position
            if cell.role_idx == 1:
                marker = '*'
                size = 60
            else:
                marker = 'o'
                size = 15 + cell.energy * 30
            ax4.scatter(x, y, c=colors[org_id], s=size, marker=marker, alpha=0.5)
    ax4.set_xlim(0, system_2.grid_size)
    ax4.set_ylim(0, system_2.grid_size)
    ax4.set_title(f'Estado Final 2 Orgs\nPop={results_2["final"]["populations"]}')
    ax4.set_aspect('equal')
    ax4.grid(alpha=0.3)

    # 5. Estado final 3 orgs
    ax5 = fig.add_subplot(2, 3, 5)
    for org_id, org in enumerate(system_3.organisms):
        for cell in org.cells:
            x, y = cell.position
            if cell.role_idx == 1:
                marker = '*'
                size = 60
            else:
                marker = 'o'
                size = 15 + cell.energy * 30
            ax5.scatter(x, y, c=colors[org_id], s=size, marker=marker, alpha=0.5)
    ax5.set_xlim(0, system_3.grid_size)
    ax5.set_ylim(0, system_3.grid_size)
    ax5.set_title(f'Estado Final 3 Orgs\nPop={results_3["final"]["populations"]}')
    ax5.set_aspect('equal')
    ax5.grid(alpha=0.3)

    # 6. Comparacion Fi
    ax6 = fig.add_subplot(2, 3, 6)
    fi_2 = results_2['final']['fi_counts']
    fi_3 = results_3['final']['fi_counts']

    x = np.arange(max(len(fi_2), len(fi_3)))
    width = 0.35

    bars_2 = ax6.bar(x[:len(fi_2)] - width/2, fi_2, width, label='2 orgs', color='steelblue')
    bars_3 = ax6.bar(x[:len(fi_3)] + width/2, fi_3, width, label='3 orgs', color='coral')

    ax6.set_xlabel('Organismo')
    ax6.set_ylabel('Numero de Fi')
    ax6.set_title('Fi por Organismo')
    ax6.set_xticks(x[:max(len(fi_2), len(fi_3))])
    ax6.legend()
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nVisualizacion guardada: {save_path}')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        n_cells_2 = 100
        n_cells_3 = 80
        n_steps = 100
    elif len(sys.argv) > 1 and sys.argv[1] == '--large':
        n_cells_2 = 400
        n_cells_3 = 300
        n_steps = 300
    else:
        n_cells_2 = 200
        n_cells_3 = 150
        n_steps = 200

    print('\nEXPERIMENTO MULTI-ORGANISMO SIMPLE')
    print('='*70)

    start = time()

    results_2, system_2 = run_multi_experiment(
        n_orgs=2,
        n_cells_per_org=n_cells_2,
        n_steps=n_steps
    )

    results_3, system_3 = run_multi_experiment(
        n_orgs=3,
        n_cells_per_org=n_cells_3,
        n_steps=n_steps
    )

    total_time = time() - start

    plot_results(results_2, results_3, system_2, system_3)

    print(f'\n{"="*70}')
    print(f'RESUMEN MULTI-ORGANISMO')
    print(f'{"="*70}')
    print(f'\n2 ORGANISMOS:')
    print(f'  Pop final: {results_2["final"]["populations"]}')
    print(f'  Fi: {results_2["final"]["fi_counts"]}')
    print(f'  Shannon: {results_2["final"]["shannon"]:.3f}/{np.log(2):.3f}')

    print(f'\n3 ORGANISMOS:')
    print(f'  Pop final: {results_3["final"]["populations"]}')
    print(f'  Fi: {results_3["final"]["fi_counts"]}')
    print(f'  Shannon: {results_3["final"]["shannon"]:.3f}/{np.log(3):.3f}')

    print(f'\nTiempo total: {total_time:.1f}s')
