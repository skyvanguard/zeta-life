# exp_tres_organismos.py
"""Experimento: Tres organismos compitiendo por recursos limitados.

Configuracion:
- 3 organismos en formacion triangular
- Recurso de energia concentrado en el centro
- Energia limitada (suma constante)
- Observar: dominacion, alianzas, extincion, coexistencia
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict

from zeta_life.organism import ForceField
from zeta_life.organism import BehaviorEngine
from zeta_life.organism import OrganismCell


@dataclass
class TriCellEntity:
    """Celula que pertenece a uno de tres organismos."""
    position: tuple
    state: torch.Tensor
    role: torch.Tensor
    energy: float = 0.0
    organism_id: int = 0  # 0, 1, o 2
    controlled_mass: float = 0.0

    @property
    def role_idx(self) -> int:
        return self.role.argmax().item()


class TripleOrganism(nn.Module):
    """Tres organismos compitiendo por recursos."""

    def __init__(self, grid_size: int = 64, n_cells_per_org: int = 30,
                 state_dim: int = 32, hidden_dim: int = 64,
                 fi_threshold: float = 0.5, total_energy: float = 60.0):
        super().__init__()

        self.grid_size = grid_size
        self.n_cells_per_org = n_cells_per_org
        self.state_dim = state_dim
        self.fi_threshold = fi_threshold
        self.total_energy = total_energy  # Energia total fija en el sistema

        # Componentes
        self.force_field = ForceField(grid_size, M=15, sigma=0.1)

        # Cada organismo tiene su BehaviorEngine
        self.behaviors = nn.ModuleList([
            BehaviorEngine(state_dim, hidden_dim) for _ in range(3)
        ])

        self.cell_module = OrganismCell(state_dim, hidden_dim)

        # Estado
        self.cells: List[TriCellEntity] = []
        self.energy_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.role_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.resource_grid = np.zeros((grid_size, grid_size))
        self.history = []

    def initialize(self):
        """Inicializa tres organismos en formacion triangular."""
        self.cells = []
        center = self.grid_size // 2
        radius = self.grid_size // 3

        # Posiciones de los tres organismos (triangulo)
        positions = [
            (center, center - radius),           # Arriba (Org 0 - Azul)
            (center - int(radius * 0.866), center + radius // 2),  # Abajo-izq (Org 1 - Rojo)
            (center + int(radius * 0.866), center + radius // 2),  # Abajo-der (Org 2 - Verde)
        ]

        for org_id in range(3):
            base_x, base_y = positions[org_id]
            for i in range(self.n_cells_per_org):
                x = np.clip(base_x + np.random.randint(-8, 9), 2, self.grid_size - 3)
                y = np.clip(base_y + np.random.randint(-8, 9), 2, self.grid_size - 3)

                state = torch.randn(self.state_dim) * 0.1

                if i == 0:
                    role = torch.tensor([0.0, 1.0, 0.0])
                    energy = 0.8
                else:
                    role = torch.tensor([1.0, 0.0, 0.0])
                    energy = np.random.uniform(0.2, 0.4)

                cell = TriCellEntity(
                    position=(x, y),
                    state=state,
                    role=role,
                    energy=energy,
                    organism_id=org_id
                )
                self.cells.append(cell)

        # Inicializar recursos en el centro
        self._init_resources()
        self._update_grids()
        self._normalize_energy()

    def _init_resources(self):
        """Crea distribucion de recursos con concentracion central."""
        center = self.grid_size // 2
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                max_dist = np.sqrt(2) * self.grid_size / 2
                self.resource_grid[y, x] = max(0, 1 - dist / max_dist)

    def _normalize_energy(self):
        """Normaliza energia total del sistema."""
        current_total = sum(c.energy for c in self.cells)
        if current_total > 0:
            factor = self.total_energy / current_total
            for cell in self.cells:
                cell.energy *= factor

    def _update_grids(self):
        """Actualiza grids."""
        self.energy_grid.zero_()
        self.role_grid.zero_()

        for cell in self.cells:
            x, y = cell.position
            self.energy_grid[0, 0, y, x] = cell.energy
            self.role_grid[0, 0, y, x] = cell.role_idx

    def _get_neighbors(self, cell: TriCellEntity, radius: int = 5,
                       same_org_only: bool = False) -> List[TriCellEntity]:
        """Obtiene vecinos."""
        neighbors = []
        cx, cy = cell.position

        for other in self.cells:
            if other is cell:
                continue
            if same_org_only and other.organism_id != cell.organism_id:
                continue

            ox, oy = other.position
            dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
            if dist <= radius:
                neighbors.append(other)

        return neighbors

    def step(self):
        """Un paso de simulacion con competencia por recursos."""
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        new_cells = []

        for cell in self.cells:
            x, y = cell.position

            # Vecinos por organismo
            all_neighbors = self._get_neighbors(cell, radius=5)
            same_org = [n for n in all_neighbors if n.organism_id == cell.organism_id]
            other_orgs = [n for n in all_neighbors if n.organism_id != cell.organism_id]

            same_mass = sum(1 for n in same_org if n.role_idx == 0)
            same_fi = sum(1 for n in same_org if n.role_idx == 1)

            # Contar enemigos por organismo
            enemy_fi = {}
            for org_id in range(3):
                if org_id != cell.organism_id:
                    enemy_fi[org_id] = sum(1 for n in other_orgs
                                          if n.organism_id == org_id and n.role_idx == 1)

            total_enemy_fi = sum(enemy_fi.values())

            # Recursos locales
            local_resource = self.resource_grid[y, x]

            # Comportamiento neural
            behavior = self.behaviors[cell.organism_id]
            if same_org:
                neighbor_states = torch.stack([n.state for n in same_org])
                influence_out, influence_in = behavior.bidirectional_influence(
                    cell.state, neighbor_states
                )
                net_influence = (influence_out.mean() - influence_in).item()
                self_pattern = behavior.self_similarity(cell.state)
                new_state = cell.state + 0.1 * self_pattern
            else:
                net_influence = 0.0
                new_state = cell.state.clone()

            # === DINAMICA DE ENERGIA CON RECURSOS LIMITADOS ===
            if cell.role_idx == 1:  # Fi
                # Gana de seguidores propios
                energy_gain = 0.03 * same_mass
                # Gana de recursos locales
                energy_gain += 0.02 * local_resource
                # Pierde por competencia con Fi enemigos
                energy_loss = 0.04 * total_enemy_fi
                # Costo de mantenimiento
                energy_loss += 0.01
                new_energy = cell.energy + energy_gain - energy_loss
            else:  # Mass
                # Decae naturalmente
                new_energy = cell.energy * 0.98
                # Gana si hay Fi propio cerca
                if same_fi > 0:
                    new_energy += 0.02
                # Gana de recursos locales (menos que Fi)
                new_energy += 0.01 * local_resource
                # Pierde si hay Fi enemigo cerca
                new_energy -= 0.02 * total_enemy_fi

            new_energy = np.clip(new_energy, 0, 1)

            # === TRANSICION DE ROL ===
            influence_score = net_influence + 0.5

            if cell.role_idx == 0:  # Mass
                can_become_fi = (
                    new_energy > self.fi_threshold and
                    same_mass >= 2 and
                    same_fi == 0 and
                    total_enemy_fi == 0
                )
                new_role = torch.tensor([0.0, 1.0, 0.0]) if can_become_fi else torch.tensor([1.0, 0.0, 0.0])

            elif cell.role_idx == 1:  # Fi
                loses_fi = (
                    same_mass < 1 or
                    new_energy < 0.15 or
                    total_enemy_fi > same_fi + 1
                )
                new_role = torch.tensor([1.0, 0.0, 0.0]) if loses_fi else torch.tensor([0.0, 1.0, 0.0])
            else:
                new_role = cell.role.clone()

            # === MOVIMIENTO ===
            new_role_idx = new_role.argmax().item()

            if new_role_idx == 0:  # Mass sigue a Fi propio
                same_fi_cells = [c for c in self.cells
                                if c.organism_id == cell.organism_id and c.role_idx == 1]
                if same_fi_cells:
                    nearest = min(same_fi_cells, key=lambda f:
                        (f.position[0] - x)**2 + (f.position[1] - y)**2)
                    dx = int(np.sign(nearest.position[0] - x))
                    dy = int(np.sign(nearest.position[1] - y))
                    x = np.clip(x + dx, 0, self.grid_size - 1)
                    y = np.clip(y + dy, 0, self.grid_size - 1)

            elif new_role_idx == 1:  # Fi se mueve hacia recursos
                # Buscar direccion de mayor recurso
                best_resource = local_resource
                best_dx, best_dy = 0, 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.resource_grid[ny, nx] > best_resource:
                            best_resource = self.resource_grid[ny, nx]
                            best_dx, best_dy = dx, dy

                if np.random.random() < 0.3:  # 30% probabilidad de moverse
                    x = np.clip(x + best_dx, 0, self.grid_size - 1)
                    y = np.clip(y + best_dy, 0, self.grid_size - 1)

            # === CONVERSION (rara) ===
            new_org_id = cell.organism_id
            if cell.role_idx == 0 and same_fi == 0 and total_enemy_fi >= 2:
                # Buscar organismo dominante cercano
                dominant_org = max(enemy_fi.keys(), key=lambda k: enemy_fi[k])
                if enemy_fi[dominant_org] >= 2 and np.random.random() < 0.15:
                    new_org_id = dominant_org

            new_cell = TriCellEntity(
                position=(x, y),
                state=new_state.detach(),
                role=new_role,
                energy=new_energy,
                organism_id=new_org_id,
                controlled_mass=same_mass
            )
            new_cells.append(new_cell)

        self.cells = new_cells
        self._update_grids()
        self._normalize_energy()  # Mantener energia total constante
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Metricas por organismo."""
        metrics = {}

        for org_id in range(3):
            org_cells = [c for c in self.cells if c.organism_id == org_id]
            key = f'org_{org_id}'

            metrics[key] = {
                'n_fi': sum(1 for c in org_cells if c.role_idx == 1),
                'n_mass': sum(1 for c in org_cells if c.role_idx == 0),
                'n_total': len(org_cells),
                'total_energy': sum(c.energy for c in org_cells),
                'avg_energy': np.mean([c.energy for c in org_cells]) if org_cells else 0,
                'centroid': (
                    np.mean([c.position[0] for c in org_cells]) if org_cells else 0,
                    np.mean([c.position[1] for c in org_cells]) if org_cells else 0
                )
            }

        # Distancia al centro (recursos)
        center = self.grid_size // 2
        for org_id in range(3):
            key = f'org_{org_id}'
            cx, cy = metrics[key]['centroid']
            metrics[key]['dist_to_center'] = np.sqrt((cx - center)**2 + (cy - center)**2)

        return metrics


def run_competition_experiment():
    """Ejecuta experimento de competencia."""
    print('='*70)
    print('EXPERIMENTO: TRES ORGANISMOS COMPITIENDO POR RECURSOS')
    print('='*70)

    torch.manual_seed(42)
    np.random.seed(42)

    tri = TripleOrganism(
        grid_size=64,
        n_cells_per_org=30,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        total_energy=60.0
    )

    # Cargar pesos
    try:
        weights = torch.load('zeta_organism_weights.pt')
        for b in tri.behaviors:
            b.load_state_dict(weights['behavior_state'])
        print('Pesos cargados!')
    except:
        print('Sin pesos entrenados')

    tri.initialize()

    colors = ['Azul', 'Rojo', 'Verde']
    print(f'\nConfiguracion:')
    print(f'  Grid: {tri.grid_size}x{tri.grid_size}')
    print(f'  Celulas por organismo: {tri.n_cells_per_org}')
    print(f'  Energia total fija: {tri.total_energy}')

    initial = tri.get_metrics()
    print(f'\nEstado inicial:')
    for i in range(3):
        m = initial[f'org_{i}']
        print(f'  Org {i} ({colors[i]}): Fi={m["n_fi"]}, Total={m["n_total"]}, '
              f'Energia={m["total_energy"]:.1f}')

    # Guardar posiciones iniciales
    initial_positions = [(c.position[0], c.position[1], c.organism_id, c.role_idx)
                        for c in tri.cells]

    # Simulacion
    print(f'\n[SIMULACION] 400 steps...')
    for step in range(400):
        tri.step()

        if (step + 1) % 100 == 0:
            m = tri.get_metrics()
            print(f'\n  Step {step+1}:')
            for i in range(3):
                org = m[f'org_{i}']
                print(f'    Org {i} ({colors[i]}): Total={org["n_total"]}, Fi={org["n_fi"]}, '
                      f'Energia={org["total_energy"]:.1f}, Dist centro={org["dist_to_center"]:.1f}')

    final = tri.get_metrics()

    # Analisis
    print('\n' + '='*70)
    print('ANALISIS DE COMPETENCIA')
    print('='*70)

    print(f'\n{"Organismo":<15} {"Inicial":<10} {"Final":<10} {"Cambio":<10} {"Energia":<10}')
    print('-'*55)
    for i in range(3):
        init_total = tri.n_cells_per_org
        final_total = final[f'org_{i}']['n_total']
        change = final_total - init_total
        energy = final[f'org_{i}']['total_energy']
        print(f'Org {i} ({colors[i]:<5}) {init_total:<10} {final_total:<10} {change:+d}{"":<6} {energy:<10.1f}')

    # Determinar resultado
    totals = [(i, final[f'org_{i}']['n_total']) for i in range(3)]
    totals.sort(key=lambda x: -x[1])

    print('\n*** RESULTADO ***')

    winner, win_count = totals[0]
    second, second_count = totals[1]
    third, third_count = totals[2]

    if third_count == 0:
        print(f'  EXTINCION: Organismo {third} ({colors[third]}) fue eliminado')

    if win_count > second_count * 1.5:
        print(f'  DOMINACION: Organismo {winner} ({colors[winner]}) domina con {win_count} celulas')
    elif abs(win_count - second_count) < 5:
        print(f'  EQUILIBRIO: Organismos {winner} y {second} en empate aproximado')
    else:
        print(f'  JERARQUIA: {colors[winner]} > {colors[second]} > {colors[third]}')

    # Quien controla el centro?
    center_dist = [(i, final[f'org_{i}']['dist_to_center']) for i in range(3)]
    center_dist.sort(key=lambda x: x[1])
    closest = center_dist[0][0]
    print(f'  CONTROL DE RECURSOS: Organismo {closest} ({colors[closest]}) mas cerca del centro')

    # Visualizacion
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    org_colors = ['royalblue', 'crimson', 'forestgreen']

    # 1. Estado inicial
    ax = axes[0, 0]
    for x, y, org_id, role_idx in initial_positions:
        color = org_colors[org_id]
        marker = 's' if role_idx == 1 else 'o'
        size = 80 if role_idx == 1 else 25
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
    # Mostrar recursos
    ax.imshow(tri.resource_grid, extent=[0, 64, 0, 64], origin='lower',
              cmap='YlOrRd', alpha=0.2)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Inicial + Recursos')
    ax.set_aspect('equal')

    # 2. Estado final
    ax = axes[0, 1]
    ax.imshow(tri.resource_grid, extent=[0, 64, 0, 64], origin='lower',
              cmap='YlOrRd', alpha=0.2)
    for cell in tri.cells:
        x, y = cell.position
        color = org_colors[cell.organism_id]
        marker = 's' if cell.role_idx == 1 else 'o'
        size = 80 if cell.role_idx == 1 else 25
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.8)
    ax.scatter(32, 32, c='gold', s=200, marker='*', zorder=10, label='Centro')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Final')
    ax.set_aspect('equal')
    ax.legend()

    # 3. Evolucion de poblacion
    ax = axes[0, 2]
    steps = range(len(tri.history))
    for i in range(3):
        totals = [h[f'org_{i}']['n_total'] for h in tri.history]
        ax.plot(steps, totals, color=org_colors[i], linewidth=2, label=f'Org {i}')
    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Poblacion')
    ax.set_title('Evolucion de Poblacion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Evolucion de energia
    ax = axes[1, 0]
    for i in range(3):
        energies = [h[f'org_{i}']['total_energy'] for h in tri.history]
        ax.plot(steps, energies, color=org_colors[i], linewidth=2, label=f'Org {i}')
    ax.axhline(y=20, color='gray', linestyle=':', alpha=0.5, label='Equitativo')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Energia Total')
    ax.set_title('Distribucion de Energia')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Distancia al centro (recursos)
    ax = axes[1, 1]
    for i in range(3):
        dists = [h[f'org_{i}']['dist_to_center'] for h in tri.history]
        ax.plot(steps, dists, color=org_colors[i], linewidth=2, label=f'Org {i}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Distancia al Centro')
    ax.set_title('Acceso a Recursos')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Cantidad de Fi
    ax = axes[1, 2]
    for i in range(3):
        fis = [h[f'org_{i}']['n_fi'] for h in tri.history]
        ax.plot(steps, fis, color=org_colors[i], linewidth=2, label=f'Org {i}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Evolucion de Lideres')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_organism_triple.png', dpi=150)
    print('\nGuardado: zeta_organism_triple.png')

    return tri


if __name__ == '__main__':
    run_competition_experiment()
