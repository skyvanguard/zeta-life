#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento Multi-Organismo Grande: 2-3 organismos con 300+ agentes cada uno.

Escenarios:
1. Dos organismos grandes en competencia
2. Tres organismos con recursos distribuidos
3. Ecosistema dinamico con muchos agentes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys

from zeta_life.organism import ZetaOrganism, CellEntity
from zeta_life.organism import ForceField
from zeta_life.organism import BehaviorEngine
from zeta_life.organism import OrganismCell


@dataclass
class MultiOrgMetrics:
    """Metricas de sistema multi-organismo."""
    n_orgs: int
    n_cells_per_org: int
    total_cells: int
    grid_size: int

    # Por organismo
    fi_counts: List[int]
    populations: List[int]
    energies: List[float]

    # Globales
    shannon_index: float
    coordination: float
    separation: float  # Distancia promedio entre centroides


class MultiOrganismSystem:
    """Sistema con multiples organismos compitiendo/cooperando."""

    def __init__(self, grid_size: int, n_orgs: int, n_cells_per_org: int,
                 state_dim: int = 32, hidden_dim: int = 64):
        self.grid_size = grid_size
        self.n_orgs = n_orgs
        self.n_cells_per_org = n_cells_per_org
        self.state_dim = state_dim

        # Componentes compartidos
        self.force_field = ForceField(grid_size, M=15, sigma=0.1)
        self.behavior = BehaviorEngine(state_dim, hidden_dim)
        self.cell_module = OrganismCell(state_dim, hidden_dim, M=15, sigma=0.1)

        # Celulas por organismo
        self.cells_by_org: List[List[CellEntity]] = [[] for _ in range(n_orgs)]

        # Energia grid
        self.energy_grid = torch.zeros(1, 1, grid_size, grid_size)

        # Historial
        self.history = []

    def initialize(self, positions: List[Tuple[int, int]] = None):
        """
        Inicializa organismos.
        positions: Lista de (x, y) para centroide de cada organismo.
        """
        if positions is None:
            # Distribuir equitativamente
            if self.n_orgs == 2:
                positions = [
                    (self.grid_size // 4, self.grid_size // 2),
                    (3 * self.grid_size // 4, self.grid_size // 2)
                ]
            elif self.n_orgs == 3:
                # Triangulo
                positions = [
                    (self.grid_size // 2, self.grid_size // 4),
                    (self.grid_size // 4, 3 * self.grid_size // 4),
                    (3 * self.grid_size // 4, 3 * self.grid_size // 4)
                ]
            else:
                # Aleatorio
                positions = [
                    (np.random.randint(20, self.grid_size - 20),
                     np.random.randint(20, self.grid_size - 20))
                    for _ in range(self.n_orgs)
                ]

        # Crear celulas para cada organismo
        for org_id in range(self.n_orgs):
            cx, cy = positions[org_id]
            self.cells_by_org[org_id] = []

            for i in range(self.n_cells_per_org):
                # Posicion cerca del centroide con ruido
                x = int(cx + np.random.randn() * 10)
                y = int(cy + np.random.randn() * 10)
                x = max(0, min(self.grid_size - 1, x))
                y = max(0, min(self.grid_size - 1, y))

                state = torch.randn(self.state_dim) * 0.1

                # Primera celula es Fi semilla
                if i == 0:
                    role = torch.tensor([0.0, 1.0, 0.0])
                    energy = 0.9
                else:
                    role = torch.tensor([1.0, 0.0, 0.0])
                    energy = np.random.uniform(0.3, 0.7)

                cell = CellEntity(
                    position=(x, y),
                    state=state,
                    role=role,
                    energy=energy
                )
                self.cells_by_org[org_id].append(cell)

    def step(self):
        """Ejecuta un paso de simulacion."""
        # Actualizar grid de energia
        self._update_energy_grid()

        # Procesar cada organismo
        for org_id in range(self.n_orgs):
            self._step_organism(org_id)

        # Interacciones entre organismos (competencia por territorio)
        self._inter_organism_interactions()

    def _update_energy_grid(self):
        """Actualiza grid de energia con todas las celulas."""
        self.energy_grid.zero_()
        for org_id in range(self.n_orgs):
            for cell in self.cells_by_org[org_id]:
                x, y = cell.position
                self.energy_grid[0, 0, y, x] += cell.energy

    def _step_organism(self, org_id: int):
        """Procesa un paso para un organismo."""
        cells = self.cells_by_org[org_id]
        if len(cells) == 0:
            return

        # Obtener Fi del organismo
        fi_cells = [c for c in cells if c.role_idx == 1]
        mass_cells = [c for c in cells if c.role_idx == 0]

        # Campo de fuerzas local (simplificado - no usamos el campo completo aqui)

        for cell in cells:
            # Actualizar estado con red neural
            x, y = cell.position

            # Percepcion local
            r = 3
            x_min = max(0, x - r)
            x_max = min(self.grid_size, x + r + 1)
            y_min = max(0, y - r)
            y_max = min(self.grid_size, y + r + 1)

            local_energy = self.energy_grid[0, 0, y_min:y_max, x_min:x_max].mean()

            # Vecinos del mismo organismo
            neighbors = self._get_neighbors(cell, cells, radius=5)
            neighbor_energy = np.mean([n.energy for n in neighbors]) if neighbors else 0

            # Actualizar celula
            new_state, gate_output = self.cell_module(
                cell.state.unsqueeze(0),
                torch.tensor([local_energy, neighbor_energy, len(neighbors) / 10]),
                t=len(self.history)
            )
            cell.state = new_state.squeeze(0)

            # Actualizar energia
            cell.energy = min(1.0, cell.energy + 0.01 * local_energy.item())

            # Transiciones de rol
            self._update_role(cell, cells)

            # Movimiento
            self._move_cell(cell, fi_cells)

    def _get_neighbors(self, cell: CellEntity, cells: List[CellEntity],
                       radius: int = 5) -> List[CellEntity]:
        """Obtiene vecinos de una celula."""
        neighbors = []
        cx, cy = cell.position
        for other in cells:
            if other is cell:
                continue
            ox, oy = other.position
            dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
            if dist <= radius:
                neighbors.append(other)
        return neighbors

    def _update_role(self, cell: CellEntity, cells: List[CellEntity]):
        """Actualiza rol de celula basado en reglas Fi-Mi."""
        fi_cells = [c for c in cells if c.role_idx == 1]
        mass_cells = [c for c in cells if c.role_idx == 0]

        # Transicion Mass -> Fi
        if cell.role_idx == 0:  # Es Mass
            neighbors = self._get_neighbors(cell, cells)
            mass_neighbors = [n for n in neighbors if n.role_idx == 0]
            fi_neighbors = [n for n in neighbors if n.role_idx == 1]

            # Puede convertirse en Fi si tiene energia alta y seguidores potenciales
            if cell.energy > 0.7 and len(mass_neighbors) >= 2 and len(fi_neighbors) == 0:
                cell.role = torch.tensor([0.0, 1.0, 0.0])

        # Transicion Fi -> Mass
        elif cell.role_idx == 1:  # Es Fi
            neighbors = self._get_neighbors(cell, cells)
            mass_neighbors = [n for n in neighbors if n.role_idx == 0]

            # Pierde rol Fi si no tiene seguidores o energia baja
            if len(mass_neighbors) == 0 or cell.energy < 0.2:
                cell.role = torch.tensor([1.0, 0.0, 0.0])

    def _move_cell(self, cell: CellEntity, fi_cells: List[CellEntity]):
        """Mueve celula segun campo de fuerzas."""
        if cell.role_idx == 1:  # Fi no se mueve tanto
            return

        x, y = cell.position

        # Encontrar Fi mas cercano
        if fi_cells:
            min_dist = float('inf')
            target_fi = None
            for fi in fi_cells:
                dist = np.sqrt((x - fi.position[0])**2 + (y - fi.position[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    target_fi = fi

            if target_fi and min_dist > 2:
                # Moverse hacia Fi
                dx = np.sign(target_fi.position[0] - x)
                dy = np.sign(target_fi.position[1] - y)

                # Con probabilidad
                if np.random.random() < 0.3:
                    new_x = max(0, min(self.grid_size - 1, x + dx))
                    new_y = max(0, min(self.grid_size - 1, y + dy))
                    cell.position = (new_x, new_y)

    def _inter_organism_interactions(self):
        """Maneja interacciones entre organismos."""
        # Competencia: celulas cercanas de diferentes organismos pierden energia
        for org_id in range(self.n_orgs):
            for cell in self.cells_by_org[org_id]:
                for other_org_id in range(self.n_orgs):
                    if other_org_id == org_id:
                        continue

                    for other_cell in self.cells_by_org[other_org_id]:
                        dist = np.sqrt(
                            (cell.position[0] - other_cell.position[0])**2 +
                            (cell.position[1] - other_cell.position[1])**2
                        )
                        if dist < 3:
                            # Competencia: ambos pierden energia
                            cell.energy *= 0.99
                            other_cell.energy *= 0.99

        # Conversion: Fi puede convertir Mass enemigos cercanos
        for org_id in range(self.n_orgs):
            fi_cells = [c for c in self.cells_by_org[org_id] if c.role_idx == 1]

            for fi in fi_cells:
                for other_org_id in range(self.n_orgs):
                    if other_org_id == org_id:
                        continue

                    cells_to_convert = []
                    for other_cell in self.cells_by_org[other_org_id]:
                        if other_cell.role_idx != 0:  # Solo Mass
                            continue

                        dist = np.sqrt(
                            (fi.position[0] - other_cell.position[0])**2 +
                            (fi.position[1] - other_cell.position[1])**2
                        )

                        if dist < 4 and np.random.random() < 0.05:
                            cells_to_convert.append(other_cell)

                    # Convertir
                    for cell in cells_to_convert:
                        self.cells_by_org[other_org_id].remove(cell)
                        cell.energy *= 0.7  # Penalizacion por conversion
                        self.cells_by_org[org_id].append(cell)

    def get_metrics(self) -> MultiOrgMetrics:
        """Obtiene metricas actuales del sistema."""
        fi_counts = []
        populations = []
        energies = []
        centroides = []

        for org_id in range(self.n_orgs):
            cells = self.cells_by_org[org_id]
            fi_counts.append(sum(1 for c in cells if c.role_idx == 1))
            populations.append(len(cells))
            energies.append(np.mean([c.energy for c in cells]) if cells else 0)

            if cells:
                cx = np.mean([c.position[0] for c in cells])
                cy = np.mean([c.position[1] for c in cells])
                centroides.append((cx, cy))
            else:
                centroides.append((self.grid_size/2, self.grid_size/2))

        # Shannon index
        total = sum(populations)
        if total > 0:
            probs = [p/total for p in populations if p > 0]
            shannon = -sum(p * np.log(p) for p in probs) if probs else 0
        else:
            shannon = 0

        # Separacion promedio
        if len(centroides) >= 2:
            dists = []
            for i in range(len(centroides)):
                for j in range(i+1, len(centroides)):
                    d = np.sqrt(
                        (centroides[i][0] - centroides[j][0])**2 +
                        (centroides[i][1] - centroides[j][1])**2
                    )
                    dists.append(d)
            separation = np.mean(dists)
        else:
            separation = 0

        return MultiOrgMetrics(
            n_orgs=self.n_orgs,
            n_cells_per_org=self.n_cells_per_org,
            total_cells=sum(populations),
            grid_size=self.grid_size,
            fi_counts=fi_counts,
            populations=populations,
            energies=energies,
            shannon_index=shannon,
            coordination=0.0,  # Placeholder
            separation=separation
        )


# ============================================================
# EXPERIMENTOS
# ============================================================

def experimento_dos_organismos_grandes(
    n_cells_per_org: int = 300,
    n_steps: int = 300,
    grid_size: int = None
):
    """Dos organismos grandes compitiendo."""
    print(f'\n{"="*70}')
    print(f'EXPERIMENTO: 2 ORGANISMOS GRANDES ({n_cells_per_org} agentes c/u)')
    print(f'{"="*70}')

    if grid_size is None:
        grid_size = int(np.sqrt(n_cells_per_org * 2 / 0.02))
        grid_size = max(100, min(300, grid_size))

    print(f'Grid: {grid_size}x{grid_size}')
    print(f'Total agentes: {n_cells_per_org * 2}')

    # Crear sistema
    system = MultiOrganismSystem(
        grid_size=grid_size,
        n_orgs=2,
        n_cells_per_org=n_cells_per_org
    )
    system.initialize()

    # Historial
    pop_history = [[], []]
    fi_history = [[], []]
    separation_history = []

    # Ejecutar
    start_time = time()
    for step in range(n_steps):
        system.step()

        metrics = system.get_metrics()
        for org_id in range(2):
            pop_history[org_id].append(metrics.populations[org_id])
            fi_history[org_id].append(metrics.fi_counts[org_id])
        separation_history.append(metrics.separation)

        if (step + 1) % 50 == 0:
            elapsed = time() - start_time
            rate = (step + 1) / elapsed
            print(f'  Step {step+1}/{n_steps}: Pop={metrics.populations}, '
                  f'Fi={metrics.fi_counts}, Rate={rate:.1f} steps/s')

    tiempo_total = time() - start_time

    # Resultados finales
    final_metrics = system.get_metrics()

    results = {
        'n_cells_per_org': n_cells_per_org,
        'grid_size': grid_size,
        'n_steps': n_steps,
        'pop_history': pop_history,
        'fi_history': fi_history,
        'separation_history': separation_history,
        'final_populations': final_metrics.populations,
        'final_fi': final_metrics.fi_counts,
        'shannon_index': final_metrics.shannon_index,
        'tiempo': tiempo_total
    }

    print(f'\n[RESULTADOS]')
    print(f'  Poblaciones finales: {final_metrics.populations}')
    print(f'  Fi finales: {final_metrics.fi_counts}')
    print(f'  Shannon index: {final_metrics.shannon_index:.3f}')
    print(f'  Tiempo: {tiempo_total:.1f}s')

    return results, system


def experimento_tres_organismos_grandes(
    n_cells_per_org: int = 250,
    n_steps: int = 300,
    grid_size: int = None
):
    """Tres organismos grandes con recursos distribuidos."""
    print(f'\n{"="*70}')
    print(f'EXPERIMENTO: 3 ORGANISMOS GRANDES ({n_cells_per_org} agentes c/u)')
    print(f'{"="*70}')

    if grid_size is None:
        grid_size = int(np.sqrt(n_cells_per_org * 3 / 0.02))
        grid_size = max(120, min(350, grid_size))

    print(f'Grid: {grid_size}x{grid_size}')
    print(f'Total agentes: {n_cells_per_org * 3}')

    # Crear sistema
    system = MultiOrganismSystem(
        grid_size=grid_size,
        n_orgs=3,
        n_cells_per_org=n_cells_per_org
    )
    system.initialize()

    # Historial
    pop_history = [[], [], []]
    fi_history = [[], [], []]

    # Ejecutar
    start_time = time()
    for step in range(n_steps):
        system.step()

        metrics = system.get_metrics()
        for org_id in range(3):
            pop_history[org_id].append(metrics.populations[org_id])
            fi_history[org_id].append(metrics.fi_counts[org_id])

        if (step + 1) % 50 == 0:
            elapsed = time() - start_time
            rate = (step + 1) / elapsed
            print(f'  Step {step+1}/{n_steps}: Pop={metrics.populations}, '
                  f'Fi={metrics.fi_counts}, Rate={rate:.1f} steps/s')

    tiempo_total = time() - start_time
    final_metrics = system.get_metrics()

    results = {
        'n_cells_per_org': n_cells_per_org,
        'grid_size': grid_size,
        'n_steps': n_steps,
        'pop_history': pop_history,
        'fi_history': fi_history,
        'final_populations': final_metrics.populations,
        'final_fi': final_metrics.fi_counts,
        'shannon_index': final_metrics.shannon_index,
        'tiempo': tiempo_total
    }

    print(f'\n[RESULTADOS]')
    print(f'  Poblaciones finales: {final_metrics.populations}')
    print(f'  Fi finales: {final_metrics.fi_counts}')
    print(f'  Shannon index: {final_metrics.shannon_index:.3f}')
    print(f'  Tiempo: {tiempo_total:.1f}s')

    return results, system


# ============================================================
# VISUALIZACION
# ============================================================

def plot_multi_org_results(
    dos_results: dict,
    tres_results: dict,
    dos_system: MultiOrganismSystem,
    tres_system: MultiOrganismSystem,
    save_path: str = 'zeta_organism_multi_grande.png'
):
    """Visualiza resultados de multi-organismo."""

    fig = plt.figure(figsize=(16, 10))
    colors = ['blue', 'red', 'green']

    # 1. Poblacion - 2 organismos
    ax1 = fig.add_subplot(2, 3, 1)
    for org_id in range(2):
        ax1.plot(dos_results['pop_history'][org_id],
                color=colors[org_id], linewidth=1.5, label=f'Org {org_id}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Poblacion')
    ax1.set_title(f'2 Organismos ({dos_results["n_cells_per_org"]} c/u)\n'
                  f'Final: {dos_results["final_populations"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Poblacion - 3 organismos
    ax2 = fig.add_subplot(2, 3, 2)
    for org_id in range(3):
        ax2.plot(tres_results['pop_history'][org_id],
                color=colors[org_id], linewidth=1.5, label=f'Org {org_id}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Poblacion')
    ax2.set_title(f'3 Organismos ({tres_results["n_cells_per_org"]} c/u)\n'
                  f'Final: {tres_results["final_populations"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Fi - 2 organismos
    ax3 = fig.add_subplot(2, 3, 3)
    for org_id in range(2):
        ax3.plot(dos_results['fi_history'][org_id],
                color=colors[org_id], linewidth=1.5, label=f'Org {org_id}')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Numero de Fi')
    ax3.set_title(f'Emergencia de Fi (2 orgs)\nFinal: {dos_results["final_fi"]}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Estado final - 2 organismos
    ax4 = fig.add_subplot(2, 3, 4)
    for org_id in range(2):
        cells = dos_system.cells_by_org[org_id]
        for cell in cells:
            x, y = cell.position
            if cell.role_idx == 1:
                marker = '*'
                size = 60
            else:
                marker = 'o'
                size = 15 + cell.energy * 30
            ax4.scatter(x, y, c=colors[org_id], s=size, marker=marker, alpha=0.5)
    ax4.set_xlim(0, dos_system.grid_size)
    ax4.set_ylim(0, dos_system.grid_size)
    ax4.set_title('Estado Final (2 orgs)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # 5. Estado final - 3 organismos
    ax5 = fig.add_subplot(2, 3, 5)
    for org_id in range(3):
        cells = tres_system.cells_by_org[org_id]
        for cell in cells:
            x, y = cell.position
            if cell.role_idx == 1:
                marker = '*'
                size = 60
            else:
                marker = 'o'
                size = 15 + cell.energy * 30
            ax5.scatter(x, y, c=colors[org_id], s=size, marker=marker, alpha=0.5)
    ax5.set_xlim(0, tres_system.grid_size)
    ax5.set_ylim(0, tres_system.grid_size)
    ax5.set_title('Estado Final (3 orgs)')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # 6. Metricas comparativas
    ax6 = fig.add_subplot(2, 3, 6)
    labels = ['2 Orgs', '3 Orgs']
    shannon = [dos_results['shannon_index'], tres_results['shannon_index']]
    max_shannon = [np.log(2), np.log(3)]

    x = np.arange(len(labels))
    width = 0.35
    ax6.bar(x - width/2, shannon, width, label='Shannon Index', color='steelblue')
    ax6.bar(x + width/2, max_shannon, width, label='Max posible', color='lightblue')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.set_ylabel('Shannon Index (diversidad)')
    ax6.set_title('Diversidad del Sistema')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nVisualizacion guardada en: {save_path}')
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        n_cells_2 = 150
        n_cells_3 = 100
        n_steps = 150
    elif len(sys.argv) > 1 and sys.argv[1] == '--large':
        n_cells_2 = 500
        n_cells_3 = 400
        n_steps = 400
    else:
        n_cells_2 = 300
        n_cells_3 = 250
        n_steps = 300

    print(f'\nEXPERIMENTO MULTI-ORGANISMO GRANDE')
    print(f'='*70)

    start_total = time()

    # Ejecutar experimentos
    dos_results, dos_system = experimento_dos_organismos_grandes(
        n_cells_per_org=n_cells_2,
        n_steps=n_steps
    )

    tres_results, tres_system = experimento_tres_organismos_grandes(
        n_cells_per_org=n_cells_3,
        n_steps=n_steps
    )

    tiempo_total = time() - start_total

    # Visualizar
    plot_multi_org_results(dos_results, tres_results, dos_system, tres_system)

    # Resumen
    print(f'\n{"="*70}')
    print(f'RESUMEN FINAL - MULTI-ORGANISMO GRANDE')
    print(f'{"="*70}')

    print(f'\n2 ORGANISMOS ({n_cells_2} c/u, total {n_cells_2*2}):')
    print(f'  Poblaciones: {dos_results["final_populations"]}')
    print(f'  Fi: {dos_results["final_fi"]}')
    print(f'  Shannon: {dos_results["shannon_index"]:.3f} (max: {np.log(2):.3f})')

    print(f'\n3 ORGANISMOS ({n_cells_3} c/u, total {n_cells_3*3}):')
    print(f'  Poblaciones: {tres_results["final_populations"]}')
    print(f'  Fi: {tres_results["final_fi"]}')
    print(f'  Shannon: {tres_results["shannon_index"]:.3f} (max: {np.log(3):.3f})')

    print(f'\nTiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)')
