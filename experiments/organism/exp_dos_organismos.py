# exp_dos_organismos.py
"""Experimento: Dos organismos interactuando en el mismo espacio.

Hipotesis posibles:
1. Fusion: Los organismos se fusionan en uno solo
2. Competencia: Un organismo domina y elimina al otro
3. Coexistencia: Forman territorios separados
4. Simbiosis: Cooperan y se benefician mutuamente
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict


# Importar componentes base
from zeta_life.organism import ForceField
from zeta_life.organism import BehaviorEngine
from zeta_life.organism import OrganismCell


@dataclass
class DualCellEntity:
    """Celula que pertenece a uno de dos organismos."""
    position: tuple
    state: torch.Tensor
    role: torch.Tensor  # [MASS, FORCE, CORRUPT]
    energy: float = 0.0
    organism_id: int = 0  # 0 o 1
    controlled_mass: float = 0.0

    @property
    def role_idx(self) -> int:
        return self.role.argmax().item()


class DualOrganism(nn.Module):
    """Dos organismos compartiendo el mismo espacio."""

    def __init__(self, grid_size: int = 64, n_cells_per_org: int = 40,
                 state_dim: int = 32, hidden_dim: int = 64,
                 fi_threshold: float = 0.5):
        super().__init__()

        self.grid_size = grid_size
        self.n_cells_per_org = n_cells_per_org
        self.state_dim = state_dim
        self.fi_threshold = fi_threshold

        # Componentes compartidos
        self.force_field = ForceField(grid_size, M=15, sigma=0.1)

        # Cada organismo tiene su propio BehaviorEngine
        self.behavior_0 = BehaviorEngine(state_dim, hidden_dim)
        self.behavior_1 = BehaviorEngine(state_dim, hidden_dim)

        # Modulo de celula compartido
        self.cell_module = OrganismCell(state_dim, hidden_dim)

        # Estado
        self.cells: List[DualCellEntity] = []
        self.energy_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.role_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.history = []

    def initialize(self, separation: str = 'horizontal'):
        """Inicializa dos organismos en regiones separadas."""
        self.cells = []

        for org_id in range(2):
            for i in range(self.n_cells_per_org):
                # Posicionar segun separacion
                if separation == 'horizontal':
                    # Org 0 a la izquierda, Org 1 a la derecha
                    if org_id == 0:
                        x = np.random.randint(5, self.grid_size // 2 - 5)
                    else:
                        x = np.random.randint(self.grid_size // 2 + 5, self.grid_size - 5)
                    y = np.random.randint(5, self.grid_size - 5)
                elif separation == 'vertical':
                    x = np.random.randint(5, self.grid_size - 5)
                    if org_id == 0:
                        y = np.random.randint(5, self.grid_size // 2 - 5)
                    else:
                        y = np.random.randint(self.grid_size // 2 + 5, self.grid_size - 5)
                elif separation == 'corners':
                    if org_id == 0:
                        x = np.random.randint(5, self.grid_size // 3)
                        y = np.random.randint(5, self.grid_size // 3)
                    else:
                        x = np.random.randint(2 * self.grid_size // 3, self.grid_size - 5)
                        y = np.random.randint(2 * self.grid_size // 3, self.grid_size - 5)

                state = torch.randn(self.state_dim) * 0.1

                # Primera celula de cada org es Fi semilla
                if i == 0:
                    role = torch.tensor([0.0, 1.0, 0.0])
                    energy = 0.9
                else:
                    role = torch.tensor([1.0, 0.0, 0.0])
                    energy = np.random.uniform(0.3, 0.6)

                cell = DualCellEntity(
                    position=(x, y),
                    state=state,
                    role=role,
                    energy=energy,
                    organism_id=org_id
                )
                self.cells.append(cell)

        self._update_grids()

    def _update_grids(self):
        """Actualiza grids de energia y roles."""
        self.energy_grid.zero_()
        self.role_grid.zero_()

        for cell in self.cells:
            x, y = cell.position
            self.energy_grid[0, 0, y, x] = cell.energy
            self.role_grid[0, 0, y, x] = cell.role_idx

    def _get_neighbors(self, cell: DualCellEntity, radius: int = 5,
                       same_org_only: bool = False) -> List[DualCellEntity]:
        """Obtiene vecinos, opcionalmente solo del mismo organismo."""
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
        """Un paso de simulacion con interaccion entre organismos."""
        # Calcular campo de fuerzas (todos los Fi contribuyen)
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        new_cells = []

        for cell in self.cells:
            # Obtener vecinos (todos y del mismo organismo)
            all_neighbors = self._get_neighbors(cell, radius=5, same_org_only=False)
            same_org_neighbors = self._get_neighbors(cell, radius=5, same_org_only=True)
            other_org_neighbors = [n for n in all_neighbors if n.organism_id != cell.organism_id]

            # Contar por tipo y organismo
            same_mass = sum(1 for n in same_org_neighbors if n.role_idx == 0)
            same_fi = sum(1 for n in same_org_neighbors if n.role_idx == 1)
            other_mass = sum(1 for n in other_org_neighbors if n.role_idx == 0)
            other_fi = sum(1 for n in other_org_neighbors if n.role_idx == 1)

            potential = field[0, 0, cell.position[1], cell.position[0]].item()

            # Seleccionar BehaviorEngine segun organismo
            behavior = self.behavior_0 if cell.organism_id == 0 else self.behavior_1

            # Componente neural
            if same_org_neighbors:
                neighbor_states = torch.stack([n.state for n in same_org_neighbors])
                influence_out, influence_in = behavior.bidirectional_influence(
                    cell.state, neighbor_states
                )
                net_influence = (influence_out.mean() - influence_in).item()
                self_pattern = behavior.self_similarity(cell.state)
                cell_enriched = cell.state + 0.1 * self_pattern
                v_input = torch.cat([cell_enriched, torch.tensor([potential])])
                transformed = behavior.transform_net(v_input)
                new_state = transformed + 0.3 * cell.state
            else:
                net_influence = 0.0
                new_state = cell.state.clone()

            # === DINAMICA DE ENERGIA CON INTERACCION ===
            if cell.role_idx == 1:  # Fi
                # Gana de seguidores propios
                energy_gain = 0.02 * same_mass
                # Pierde si hay Fi enemigos cerca (competencia)
                energy_loss = 0.03 * other_fi
                new_energy = cell.energy + energy_gain - energy_loss
            else:  # Mass
                new_energy = cell.energy * 0.995 + 0.05 * max(0, potential)
                # Bonus si hay Fi del mismo organismo cerca
                if same_fi > 0:
                    new_energy += 0.02
                # Penalizacion si hay Fi enemigo cerca (confusion)
                if other_fi > 0:
                    new_energy -= 0.01

            new_energy += 0.02 * max(0, net_influence + 0.5)
            new_energy = np.clip(new_energy, 0, 1)

            # === TRANSICION DE ROL ===
            current_role_idx = cell.role_idx
            influence_score = net_influence + 0.5

            if current_role_idx == 0:  # MASS
                # Solo puede convertirse en Fi si tiene seguidores del mismo org
                # y no hay Fi enemigos dominantes cerca
                can_become_fi = (
                    new_energy > self.fi_threshold and
                    same_mass >= 2 and
                    (same_fi == 0 or influence_score > 0.3) and
                    other_fi == 0  # No convertirse si hay competencia
                )
                if can_become_fi:
                    new_role = torch.tensor([0.0, 1.0, 0.0])
                else:
                    new_role = torch.tensor([1.0, 0.0, 0.0])

            elif current_role_idx == 1:  # FORCE
                # Pierde si no tiene seguidores o hay competencia fuerte
                loses_fi = (
                    same_mass < 1 or
                    new_energy < 0.2 or
                    (other_fi > 0 and other_fi >= same_fi)  # Competencia
                )
                if loses_fi:
                    new_role = torch.tensor([1.0, 0.0, 0.0])
                else:
                    new_role = torch.tensor([0.0, 1.0, 0.0])
            else:
                new_role = cell.role.clone()

            # === MOVIMIENTO ===
            x, y = cell.position
            new_role_idx = new_role.argmax().item()

            if new_role_idx == 0:  # Mass sigue a Fi de su organismo
                # Buscar Fi mas cercano del mismo organismo
                same_fi_cells = [c for c in self.cells
                                if c.organism_id == cell.organism_id and c.role_idx == 1]
                if same_fi_cells:
                    nearest_fi = min(same_fi_cells, key=lambda f:
                        (f.position[0] - x)**2 + (f.position[1] - y)**2)
                    dx = int(np.sign(nearest_fi.position[0] - x))
                    dy = int(np.sign(nearest_fi.position[1] - y))
                    x = np.clip(x + dx, 0, self.grid_size - 1)
                    y = np.clip(y + dy, 0, self.grid_size - 1)

            # === CONVERSION DE ORGANISMO (raro) ===
            # Si una Mass esta rodeada de Fi enemigos y sin Fi propios, puede cambiar de bando
            new_org_id = cell.organism_id
            if current_role_idx == 0 and same_fi == 0 and other_fi >= 2:
                if np.random.random() < 0.1:  # 10% probabilidad
                    new_org_id = 1 - cell.organism_id  # Cambiar de bando

            new_cell = DualCellEntity(
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
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Metricas por organismo y globales."""
        metrics = {
            'org_0': {'n_fi': 0, 'n_mass': 0, 'avg_energy': 0, 'centroid': (0, 0)},
            'org_1': {'n_fi': 0, 'n_mass': 0, 'avg_energy': 0, 'centroid': (0, 0)},
            'total_cells': len(self.cells),
            'boundary_contacts': 0
        }

        for org_id in [0, 1]:
            org_cells = [c for c in self.cells if c.organism_id == org_id]
            key = f'org_{org_id}'

            metrics[key]['n_fi'] = sum(1 for c in org_cells if c.role_idx == 1)
            metrics[key]['n_mass'] = sum(1 for c in org_cells if c.role_idx == 0)
            metrics[key]['n_total'] = len(org_cells)

            if org_cells:
                metrics[key]['avg_energy'] = np.mean([c.energy for c in org_cells])
                cx = np.mean([c.position[0] for c in org_cells])
                cy = np.mean([c.position[1] for c in org_cells])
                metrics[key]['centroid'] = (cx, cy)

        # Contar contactos en frontera (celulas de diff org cerca)
        for cell in self.cells:
            other_org = [c for c in self.cells
                        if c.organism_id != cell.organism_id and
                        np.sqrt((c.position[0] - cell.position[0])**2 +
                               (c.position[1] - cell.position[1])**2) < 3]
            if other_org:
                metrics['boundary_contacts'] += 1

        metrics['boundary_contacts'] //= 2  # Evitar doble conteo

        return metrics


def run_dual_organism_experiment():
    """Ejecuta experimento de dos organismos."""
    print('='*70)
    print('EXPERIMENTO: DOS ORGANISMOS INTERACTUANDO')
    print('='*70)

    torch.manual_seed(42)
    np.random.seed(42)

    dual = DualOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )

    # Cargar pesos si existen
    try:
        weights = torch.load('zeta_organism_weights.pt')
        dual.behavior_0.load_state_dict(weights['behavior_state'])
        dual.behavior_1.load_state_dict(weights['behavior_state'])
        print('Pesos cargados para ambos organismos!')
    except:
        print('Sin pesos entrenados')

    dual.initialize(separation='horizontal')

    print(f'\nConfiguracion:')
    print(f'  Grid: {dual.grid_size}x{dual.grid_size}')
    print(f'  Celulas por organismo: {dual.n_cells_per_org}')
    print(f'  Total celulas: {dual.n_cells_per_org * 2}')

    initial = dual.get_metrics()
    print(f'\nEstado inicial:')
    print(f'  Org 0 (azul): Fi={initial["org_0"]["n_fi"]}, Mass={initial["org_0"]["n_mass"]}')
    print(f'  Org 1 (rojo): Fi={initial["org_1"]["n_fi"]}, Mass={initial["org_1"]["n_mass"]}')

    # Guardar posiciones iniciales
    initial_positions = [(c.position[0], c.position[1], c.organism_id, c.role_idx)
                        for c in dual.cells]

    # Simulacion
    print('\n[SIMULACION] 300 steps...')
    for step in range(300):
        dual.step()

        if (step + 1) % 50 == 0:
            m = dual.get_metrics()
            print(f'\n  Step {step+1}:')
            print(f'    Org 0: Fi={m["org_0"]["n_fi"]}, Total={m["org_0"]["n_total"]}, '
                  f'Centro=({m["org_0"]["centroid"][0]:.1f}, {m["org_0"]["centroid"][1]:.1f})')
            print(f'    Org 1: Fi={m["org_1"]["n_fi"]}, Total={m["org_1"]["n_total"]}, '
                  f'Centro=({m["org_1"]["centroid"][0]:.1f}, {m["org_1"]["centroid"][1]:.1f})')
            print(f'    Contactos frontera: {m["boundary_contacts"]}')

    final = dual.get_metrics()

    # Analisis
    print('\n' + '='*70)
    print('ANALISIS DE INTERACCION')
    print('='*70)

    print(f'\n{"Metrica":<20} {"Org 0 (Azul)":<15} {"Org 1 (Rojo)":<15}')
    print('-'*50)
    print(f'{"Fi":<20} {final["org_0"]["n_fi"]:<15} {final["org_1"]["n_fi"]:<15}')
    print(f'{"Mass":<20} {final["org_0"]["n_mass"]:<15} {final["org_1"]["n_mass"]:<15}')
    print(f'{"Total":<20} {final["org_0"]["n_total"]:<15} {final["org_1"]["n_total"]:<15}')
    print(f'{"Energia prom":<20} {final["org_0"]["avg_energy"]:<15.3f} {final["org_1"]["avg_energy"]:<15.3f}')

    # Determinar resultado
    print('\n*** RESULTADO ***')
    org0_total = final["org_0"]["n_total"]
    org1_total = final["org_1"]["n_total"]
    initial_per_org = dual.n_cells_per_org

    if org0_total == 0 or org1_total == 0:
        winner = 0 if org0_total > 0 else 1
        print(f'  DOMINACION: Organismo {winner} elimino al otro')
    elif abs(org0_total - org1_total) > initial_per_org * 0.5:
        winner = 0 if org0_total > org1_total else 1
        print(f'  DOMINACION PARCIAL: Organismo {winner} absorbio parte del otro')
    elif final["boundary_contacts"] > 10:
        print('  FUSION/MEZCLA: Los organismos se mezclaron')
    else:
        print('  COEXISTENCIA: Los organismos mantienen territorios separados')

    # Calcular distancia entre centroides
    c0 = final["org_0"]["centroid"]
    c1 = final["org_1"]["centroid"]
    centroid_dist = np.sqrt((c0[0] - c1[0])**2 + (c0[1] - c1[1])**2)
    print(f'  Distancia entre centroides: {centroid_dist:.1f} celdas')

    # Visualizacion
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Estado inicial
    ax = axes[0, 0]
    for x, y, org_id, role_idx in initial_positions:
        color = 'royalblue' if org_id == 0 else 'crimson'
        marker = 's' if role_idx == 1 else 'o'
        size = 80 if role_idx == 1 else 30
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
    ax.axvline(x=32, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Inicial')
    ax.set_aspect('equal')

    # 2. Estado final
    ax = axes[0, 1]
    for cell in dual.cells:
        x, y = cell.position
        color = 'royalblue' if cell.organism_id == 0 else 'crimson'
        marker = 's' if cell.role_idx == 1 else 'o'
        size = 80 if cell.role_idx == 1 else 30
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7,
                  edgecolors='white' if cell.role_idx == 1 else 'none', linewidths=1)
    ax.axvline(x=32, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Final (cuadrado=Fi)')
    ax.set_aspect('equal')

    # 3. Evolucion de poblacion
    ax = axes[0, 2]
    steps = range(len(dual.history))
    org0_total = [h['org_0']['n_total'] for h in dual.history]
    org1_total = [h['org_1']['n_total'] for h in dual.history]
    ax.plot(steps, org0_total, 'b-', linewidth=2, label='Org 0 (azul)')
    ax.plot(steps, org1_total, 'r-', linewidth=2, label='Org 1 (rojo)')
    ax.axhline(y=40, color='gray', linestyle=':', alpha=0.5, label='Inicial')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Total celulas')
    ax.set_title('Evolucion de Poblacion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Evolucion de Fi
    ax = axes[1, 0]
    org0_fi = [h['org_0']['n_fi'] for h in dual.history]
    org1_fi = [h['org_1']['n_fi'] for h in dual.history]
    ax.plot(steps, org0_fi, 'b-', linewidth=2, label='Org 0')
    ax.plot(steps, org1_fi, 'r-', linewidth=2, label='Org 1')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Evolucion de Lideres (Fi)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Contactos en frontera
    ax = axes[1, 1]
    boundary = [h['boundary_contacts'] for h in dual.history]
    ax.plot(steps, boundary, 'g-', linewidth=2)
    ax.fill_between(steps, boundary, alpha=0.3, color='green')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Contactos')
    ax.set_title('Contactos en Frontera')
    ax.grid(True, alpha=0.3)

    # 6. Trayectoria de centroides
    ax = axes[1, 2]
    c0_x = [h['org_0']['centroid'][0] for h in dual.history]
    c0_y = [h['org_0']['centroid'][1] for h in dual.history]
    c1_x = [h['org_1']['centroid'][0] for h in dual.history]
    c1_y = [h['org_1']['centroid'][1] for h in dual.history]

    ax.plot(c0_x, c0_y, 'b-', linewidth=1, alpha=0.5)
    ax.plot(c1_x, c1_y, 'r-', linewidth=1, alpha=0.5)
    ax.scatter(c0_x[0], c0_y[0], c='blue', s=100, marker='o', label='Org 0 inicio')
    ax.scatter(c0_x[-1], c0_y[-1], c='blue', s=100, marker='s', label='Org 0 final')
    ax.scatter(c1_x[0], c1_y[0], c='red', s=100, marker='o', label='Org 1 inicio')
    ax.scatter(c1_x[-1], c1_y[-1], c='red', s=100, marker='s', label='Org 1 final')
    ax.axvline(x=32, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Trayectoria de Centroides')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('zeta_organism_dual.png', dpi=150)
    print('\nGuardado: zeta_organism_dual.png')

    return dual


if __name__ == '__main__':
    run_dual_organism_experiment()
