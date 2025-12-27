# zeta_organism.py
"""ZetaOrganism: Organismo artificial con inteligencia colectiva."""
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from cell_state import CellState, CellRole
from force_field import ForceField
from behavior_engine import BehaviorEngine
from organism_cell import OrganismCell

@dataclass
class CellEntity:
    """Entidad célula en el grid."""
    position: tuple
    state: torch.Tensor
    role: torch.Tensor  # probabilities [MASS, FORCE, CORRUPT]
    energy: float = 0.0
    controlled_mass: float = 0.0

    @property
    def role_idx(self) -> int:
        return self.role.argmax().item()


class ZetaOrganism(nn.Module):
    """Organismo artificial distribuido."""

    def __init__(self, grid_size: int = 64, n_cells: int = 100,
                 state_dim: int = 32, hidden_dim: int = 64,
                 M: int = 15, sigma: float = 0.1,
                 fi_threshold: float = 0.7,
                 equilibrium_factor: float = 0.5):
        super().__init__()

        self.grid_size = grid_size
        self.n_cells = n_cells
        self.state_dim = state_dim
        self.fi_threshold = fi_threshold
        self.equilibrium_factor = equilibrium_factor

        # Componentes
        self.force_field = ForceField(grid_size, M, sigma)
        self.behavior = BehaviorEngine(state_dim, hidden_dim)
        self.cell_module = OrganismCell(state_dim, hidden_dim, M, sigma)

        # Estado
        self.cells: List[CellEntity] = []
        self.energy_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.role_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.history = []

        # Crear células iniciales (sin Fi semilla)
        self._create_cells(seed_fi=False)

    def _create_cells(self, seed_fi: bool = False):
        """Crea células en posiciones aleatorias."""
        self.cells = []

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

            cell = CellEntity(
                position=(x, y),
                state=state,
                role=role,
                energy=energy
            )
            self.cells.append(cell)

        self._update_grids()

    def initialize(self, seed_fi: bool = True):
        """Inicializa organismo con células aleatorias (con Fi semilla opcional)."""
        self._create_cells(seed_fi=seed_fi)

    def _update_grids(self):
        """Actualiza grids de energía y roles."""
        self.energy_grid.zero_()
        self.role_grid.zero_()

        for cell in self.cells:
            x, y = cell.position
            self.energy_grid[0, 0, y, x] = cell.energy
            self.role_grid[0, 0, y, x] = cell.role_idx

    def _get_neighbors(self, cell: CellEntity, radius: int = 3) -> List[CellEntity]:
        """Obtiene células vecinas."""
        neighbors = []
        cx, cy = cell.position

        for other in self.cells:
            if other is cell:
                continue
            ox, oy = other.position
            dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
            if dist <= radius:
                neighbors.append(other)

        return neighbors

    def step(self):
        """Un paso de simulación."""
        # 1. Calcular campo de fuerzas
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        # 2. Actualizar cada célula
        new_cells = []
        for cell in self.cells:
            neighbors = self._get_neighbors(cell)

            state = cell.state.unsqueeze(0)
            if neighbors:
                neighbor_states = torch.stack([n.state for n in neighbors]).unsqueeze(0)
            else:
                neighbor_states = torch.zeros(1, 1, self.state_dim)

            new_state, role_probs = self.cell_module(
                state, neighbor_states, field, cell.position
            )

            potential = field[0, 0, cell.position[1], cell.position[0]].item()
            new_energy = cell.energy + 0.1 * potential
            new_energy = np.clip(new_energy, 0, 1)

            if role_probs[0, 1] > 0.5:  # Fi
                controlled = sum(1 for n in neighbors if n.role_idx == 0)
            else:
                controlled = 0

            x, y = cell.position
            if role_probs[0, 0] > 0.5:  # MASS sigue gradiente
                grad = gradient[0, :, y, x]
                dx = int(np.sign(grad[0].item()))
                dy = int(np.sign(grad[1].item()))
                x = np.clip(x + dx, 0, self.grid_size - 1)
                y = np.clip(y + dy, 0, self.grid_size - 1)

            new_cell = CellEntity(
                position=(x, y),
                state=new_state.squeeze(0).detach(),
                role=role_probs.squeeze(0).detach(),
                energy=new_energy,
                controlled_mass=controlled
            )
            new_cells.append(new_cell)

        self.cells = new_cells
        self._update_grids()
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Métricas de inteligencia colectiva."""
        n_fi = sum(1 for c in self.cells if c.role_idx == 1)
        n_mass = sum(1 for c in self.cells if c.role_idx == 0)
        n_corrupt = sum(1 for c in self.cells if c.role_idx == 2)

        if n_fi > 0:
            fi_cells = [c for c in self.cells if c.role_idx == 1]
            mass_cells = [c for c in self.cells if c.role_idx == 0]

            total_dist = 0
            for m in mass_cells:
                min_dist = min(
                    np.sqrt((m.position[0] - f.position[0])**2 +
                           (m.position[1] - f.position[1])**2)
                    for f in fi_cells
                ) if fi_cells else self.grid_size
                total_dist += min_dist

            avg_dist = total_dist / max(len(mass_cells), 1)
            coordination = 1.0 - (avg_dist / self.grid_size)
        else:
            coordination = 0.0

        energies = [c.energy for c in self.cells]
        stability = 1.0 - np.std(energies) if energies else 0.0

        return {
            'n_fi': n_fi,
            'n_mass': n_mass,
            'n_corrupt': n_corrupt,
            'coordination': coordination,
            'stability': stability,
            'avg_energy': np.mean(energies) if energies else 0.0
        }
