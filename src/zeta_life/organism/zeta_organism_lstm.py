# zeta_organism_lstm.py
"""ZetaOrganismLSTM: Organismo con memoria temporal LSTM integrada.

Evolucion de ZetaOrganism: usa OrganismCellLSTMPool para que cada celula
mantenga su propio estado LSTM con memoria zeta.
"""
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from .cell_state import CellState, CellRole
from .force_field import ForceField
from .behavior_engine import BehaviorEngine
from .organism_cell_lstm import OrganismCellLSTMPool


@dataclass
class CellEntityLSTM:
    """Entidad celula con ID para tracking de estado LSTM."""
    id: int
    position: tuple
    state: torch.Tensor
    role: torch.Tensor
    energy: float = 0.0
    controlled_mass: float = 0.0
    memory_info: dict = None

    @property
    def role_idx(self) -> int:
        return self.role.argmax().item()


class ZetaOrganismLSTM(nn.Module):
    """Organismo con memoria LSTM-zeta por celula.

    Diferencias vs ZetaOrganism:
    - Cada celula tiene estado LSTM persistente (h, c)
    - La memoria evoluciona con el tiempo usando ZetaLSTMCell
    - Mejor capacidad de anticipacion y memoria temporal
    """

    def __init__(self, grid_size: int = 64, n_cells: int = 80,
                 state_dim: int = 32, hidden_dim: int = 64,
                 M: int = 15, sigma: float = 0.1,
                 zeta_weight: float = 0.2,
                 fi_threshold: float = 0.7,
                 equilibrium_factor: float = 0.5):
        super().__init__()

        self.grid_size = grid_size
        self.n_cells = n_cells
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.fi_threshold = fi_threshold
        self.equilibrium_factor = equilibrium_factor

        # Componentes
        self.force_field = ForceField(grid_size, M, sigma)
        self.behavior = BehaviorEngine(state_dim, hidden_dim)

        # Pool de celdas LSTM (reemplaza OrganismCell)
        self.cell_pool = OrganismCellLSTMPool(
            n_cells=n_cells,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            M=M,
            sigma=sigma,
            zeta_weight=zeta_weight
        )

        # Estado
        self.cells: List[CellEntityLSTM] = []
        self.energy_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.role_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.history = []
        self.next_cell_id = 0

    def _create_cell_id(self) -> int:
        """Genera ID unico para celula."""
        cell_id = self.next_cell_id
        self.next_cell_id += 1
        return cell_id

    def _create_cells(self, seed_fi: bool = False):
        """Crea celulas en posiciones aleatorias."""
        self.cells = []
        self.cell_pool.reset()

        for i in range(self.n_cells):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            state = torch.randn(self.state_dim) * 0.1

            if seed_fi and i == 0:
                role = torch.tensor([0.0, 1.0, 0.0])  # FORCE
                energy = 0.9
            else:
                role = torch.tensor([1.0, 0.0, 0.0])  # MASS
                energy = np.random.uniform(0.1, 0.5)

            cell = CellEntityLSTM(
                id=self._create_cell_id(),
                position=(x, y),
                state=state,
                role=role,
                energy=energy
            )
            self.cells.append(cell)

        self._update_grids()

    def initialize(self, seed_fi: bool = True):
        """Inicializa organismo."""
        self._create_cells(seed_fi=seed_fi)

    def _update_grids(self):
        """Actualiza grids de energia y roles."""
        self.energy_grid.zero_()
        self.role_grid.zero_()

        for cell in self.cells:
            x, y = cell.position
            self.energy_grid[0, 0, y, x] = cell.energy
            self.role_grid[0, 0, y, x] = cell.role_idx

    def _get_neighbors(self, cell: CellEntityLSTM, radius: int = 3) -> List[CellEntityLSTM]:
        """Obtiene celulas vecinas."""
        neighbors = []
        cx, cy = cell.position

        for other in self.cells:
            if other.id == cell.id:
                continue
            ox, oy = other.position
            dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
            if dist <= radius:
                neighbors.append(other)

        return neighbors

    def _count_followers(self, fi_cell: CellEntityLSTM) -> int:
        """Cuenta seguidores de un Fi."""
        neighbors = self._get_neighbors(fi_cell, radius=5)
        return sum(1 for n in neighbors if n.role_idx == 0)  # MASS

    def step(self):
        """Un paso de simulacion con memoria LSTM."""
        # 1. Calcular campo de fuerzas
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        # 2. Procesar cada celula con su memoria LSTM
        updates = []

        for cell in self.cells:
            # Obtener vecinos
            neighbors = self._get_neighbors(cell, radius=3)

            # Preparar tensores
            state = cell.state.unsqueeze(0)  # [1, state_dim]

            if neighbors:
                neighbor_states = torch.stack([n.state for n in neighbors])
                neighbor_states = neighbor_states.unsqueeze(0)  # [1, N, state_dim]
            else:
                neighbor_states = torch.zeros(1, 1, self.state_dim)

            # Forward con memoria LSTM persistente
            new_state, role_probs, memory_info = self.cell_pool.forward(
                cell.id, state, neighbor_states, field, cell.position
            )

            updates.append({
                'cell': cell,
                'new_state': new_state.squeeze(0),
                'role_probs': role_probs.squeeze(0),
                'memory_info': memory_info
            })

        # 3. Aplicar actualizaciones
        for upd in updates:
            cell = upd['cell']
            cell.state = upd['new_state'].detach()
            cell.memory_info = upd['memory_info']

        # 4. Actualizar roles y energia (logica Fi-Mi)
        self._update_roles_and_energy(field, gradient)

        # 5. Movimiento
        self._move_cells(gradient)

        # 6. Avanzar timestep global
        self.cell_pool.step_time()

        # 7. Actualizar grids
        self._update_grids()

        # 8. Guardar historia
        self.history.append(self.get_metrics())

    def _update_roles_and_energy(self, field, gradient):
        """Actualiza roles y energia usando logica Fi-Mi."""
        fi_cells = [c for c in self.cells if c.role_idx == 1]
        mass_cells = [c for c in self.cells if c.role_idx == 0]

        # Calcular cuantos Fi deberia haber
        total_mass = len(self.cells)
        expected_fi = max(1, int(np.sqrt(total_mass) * self.equilibrium_factor))

        # Promocion: MASS -> FI
        if len(fi_cells) < expected_fi:
            candidates = sorted(mass_cells, key=lambda c: -c.energy)
            for c in candidates[:expected_fi - len(fi_cells)]:
                if c.energy > self.fi_threshold:
                    c.role = torch.tensor([0.0, 1.0, 0.0])

        # Degradacion: FI -> MASS
        for fi in fi_cells:
            followers = self._count_followers(fi)
            if followers < 2 or fi.energy < 0.3:
                fi.role = torch.tensor([1.0, 0.0, 0.0])

        # Transferencia de energia
        for fi in [c for c in self.cells if c.role_idx == 1]:
            followers = self._get_neighbors(fi, radius=5)
            followers = [f for f in followers if f.role_idx == 0]

            if followers:
                total_transfer = min(0.1, fi.energy * 0.2)
                per_follower = total_transfer / len(followers)
                fi.energy -= total_transfer

                for follower in followers:
                    follower.energy = min(1.0, follower.energy + per_follower)

    def _move_cells(self, gradient):
        """Mueve celulas siguiendo gradiente del campo."""
        for cell in self.cells:
            x, y = cell.position

            if cell.role_idx == 0:  # MASS sigue gradiente
                gx = gradient[0, 0, y, x].item()
                gy = gradient[0, 1, y, x].item()

                if abs(gx) > 0.01 or abs(gy) > 0.01:
                    dx = int(np.sign(gx)) if np.random.random() < 0.3 else 0
                    dy = int(np.sign(gy)) if np.random.random() < 0.3 else 0

                    new_x = np.clip(x + dx, 0, self.grid_size - 1)
                    new_y = np.clip(y + dy, 0, self.grid_size - 1)
                    cell.position = (new_x, new_y)

    def get_metrics(self) -> dict:
        """Obtiene metricas del estado actual."""
        fi_cells = [c for c in self.cells if c.role_idx == 1]
        mass_cells = [c for c in self.cells if c.role_idx == 0]

        # Calcular norma promedio de memoria
        h_norms = [c.memory_info['h_norm'] for c in self.cells if c.memory_info]
        avg_h_norm = np.mean(h_norms) if h_norms else 0

        return {
            'n_fi': len(fi_cells),
            'n_mass': len(mass_cells),
            'n_total': len(self.cells),
            'avg_energy': np.mean([c.energy for c in self.cells]),
            'fi_energy': np.mean([c.energy for c in fi_cells]) if fi_cells else 0,
            'mass_energy': np.mean([c.energy for c in mass_cells]) if mass_cells else 0,
            'avg_h_norm': avg_h_norm,
            'timestep': self.cell_pool.t
        }

    def damage(self, region: tuple, intensity: float = 0.5):
        """Aplica dano a una region."""
        x1, y1, x2, y2 = region
        damaged = []

        for cell in self.cells:
            cx, cy = cell.position
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if np.random.random() < intensity:
                    damaged.append(cell)

        for cell in damaged:
            self.cells.remove(cell)
            # Limpiar estado LSTM de la celula eliminada
            if cell.id in self.cell_pool.h_states:
                del self.cell_pool.h_states[cell.id]
                del self.cell_pool.c_states[cell.id]

        self._update_grids()
        return len(damaged)


if __name__ == '__main__':
    print('=' * 70)
    print('ZetaOrganismLSTM - Test')
    print('=' * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ZetaOrganismLSTM(
        grid_size=64,
        n_cells=50,
        state_dim=32,
        hidden_dim=64,
        zeta_weight=0.2
    )

    org.initialize(seed_fi=True)

    print(f"\nEstado inicial:")
    m = org.get_metrics()
    print(f"  Celulas: {m['n_total']} (Fi={m['n_fi']}, Mass={m['n_mass']})")
    print(f"  Energia promedio: {m['avg_energy']:.3f}")

    print(f"\nSimulando 20 steps...")
    for step in range(20):
        org.step()

        if (step + 1) % 5 == 0:
            m = org.get_metrics()
            print(f"  Step {step+1}: Fi={m['n_fi']}, h_norm={m['avg_h_norm']:.4f}")

    print(f"\nEstado final:")
    m = org.get_metrics()
    print(f"  Celulas: {m['n_total']} (Fi={m['n_fi']}, Mass={m['n_mass']})")
    print(f"  Energia promedio: {m['avg_energy']:.3f}")
    print(f"  h_norm promedio: {m['avg_h_norm']:.4f}")
    print(f"  Timestep: {m['timestep']}")

    # Test de dano
    print(f"\nAplicando dano...")
    damaged = org.damage((20, 20, 40, 40), intensity=0.8)
    print(f"  Celulas eliminadas: {damaged}")

    m = org.get_metrics()
    print(f"  Celulas restantes: {m['n_total']}")

    print('\n[OK] All tests passed!')
