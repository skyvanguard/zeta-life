# zeta_organism.py
"""ZetaOrganism: Organismo artificial con inteligencia colectiva."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .behavior_engine import BehaviorEngine
from .cell_state import CellRole, CellState
from .force_field import ForceField
from .organism_cell import OrganismCell

@dataclass
class CellEntity:
    """Entidad célula en el grid."""
    position: tuple[int, int]
    state: torch.Tensor
    role: torch.Tensor  # probabilities [MASS, FORCE, CORRUPT]
    energy: float = 0.0
    controlled_mass: float = 0.0

    @property
    def role_idx(self) -> int:
        return int(self.role.argmax().item())

class ZetaOrganism(nn.Module):
    """Organismo artificial distribuido."""

    def __init__(self, grid_size: int = 64, n_cells: int = 100,
                 state_dim: int = 32, hidden_dim: int = 64,
                 M: int = 15, sigma: float = 0.1,
                 fi_threshold: float = 0.7,
                 equilibrium_factor: float = 0.5) -> None:
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
        self.cells: list[CellEntity] = []
        self.energy_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.role_grid = torch.zeros(1, 1, grid_size, grid_size)
        self.history: list[dict] = []

        # Crear células iniciales (sin Fi semilla)
        self._create_cells(seed_fi=False)

    def _create_cells(self, seed_fi: bool = False) -> None:
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

    def initialize(self, seed_fi: bool = True) -> None:
        """Inicializa organismo con células aleatorias (con Fi semilla opcional)."""
        self._create_cells(seed_fi=seed_fi)

    def _update_grids(self) -> None:
        """Actualiza grids de energía y roles."""
        self.energy_grid.zero_()
        self.role_grid.zero_()

        for cell in self.cells:
            x, y = cell.position
            self.energy_grid[0, 0, y, x] = cell.energy
            self.role_grid[0, 0, y, x] = cell.role_idx

    def _get_neighbors(self, cell: CellEntity, radius: int = 3) -> list[CellEntity]:
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

    def step(self) -> None:
        """Un paso de simulación HÍBRIDO: reglas Fi-Mi + BehaviorEngine neural."""
        # 1. Calcular campo de fuerzas (solo Fi emiten)
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        # 2. Pre-calcular vecinos para cada célula
        neighbor_map = {}
        for cell in self.cells:
            neighbors = self._get_neighbors(cell, radius=5)
            mass_neighbors = sum(1 for n in neighbors if n.role_idx == 0)
            fi_neighbors = sum(1 for n in neighbors if n.role_idx == 1)
            neighbor_map[id(cell)] = (neighbors, mass_neighbors, fi_neighbors)

        # 3. Actualizar cada célula con HÍBRIDO
        new_cells = []
        for cell in self.cells:
            neighbors, mass_neighbors, fi_neighbors = neighbor_map[id(cell)]
            potential = field[0, 0, cell.position[1], cell.position[0]].item()

            # ========== PARTE NEURAL: BehaviorEngine ==========
            if neighbors:
                neighbor_states = torch.stack([n.state for n in neighbors])

                # A ↔ B: Influencia bidireccional
                influence_out, influence_in = self.behavior.bidirectional_influence(
                    cell.state, neighbor_states
                )

                # B = AA* - A*A: Rol neto basado en influencia
                # Positivo = emite más (Fi), Negativo = recibe más (Mi)
                net_influence = (influence_out.mean() - influence_in).item()

                # A = AAA*A: Patrón auto-similar
                self_pattern = self.behavior.self_similarity(cell.state)

                # A³ + V → B³ + A: Transformación con potencial vital
                cell_enriched = cell.state + 0.1 * self_pattern
                v_input = torch.cat([cell_enriched, torch.tensor([potential])])
                transformed = self.behavior.transform_net(v_input)
                new_state = transformed + 0.3 * cell.state  # α = 0.3
            else:
                net_influence = 0.0
                new_state = cell.state.clone()

            # ========== PARTE REGLAS: Dinámica Fi-Mi ==========

            # Energía: Fi gana de seguidores, Mass puede acumular del campo
            if cell.role_idx == 1:  # Es Fi
                energy_gain = 0.02 * mass_neighbors
                new_energy = cell.energy + energy_gain
            else:  # Es Mass
                # Mass acumula energía del campo (más lento decay, más ganancia)
                new_energy = cell.energy * 0.995 + 0.05 * max(0, potential)
                # Bonus si tiene muchos vecinos mass (potencial líder)
                if mass_neighbors >= 3:
                    new_energy += 0.02

            # Bonus de energía por influencia neta positiva (neural)
            new_energy += 0.02 * max(0, net_influence)
            new_energy = np.clip(new_energy, 0, 1)

            # ========== HÍBRIDO: Transición de rol ==========
            # Combina reglas físicas con señal neural (net_influence)

            current_role_idx = cell.role_idx

            if current_role_idx == 0:  # MASS
                # MASS → FORCE: energía alta + seguidores
                # Neural: net_influence más alto que promedio = potencial líder
                # (red no entrenada da valores negativos, usamos comparación relativa)
                influence_score = net_influence + 0.5  # Normalizar a ~0
                can_become_fi = (
                    new_energy > self.fi_threshold and
                    mass_neighbors >= 2 and
                    (fi_neighbors == 0 or influence_score > 0.2)  # Sin Fi cerca O influencia destacada
                )
                if can_become_fi:
                    new_role = torch.tensor([0.0, 1.0, 0.0])  # → FORCE
                else:
                    new_role = torch.tensor([1.0, 0.0, 0.0])  # Sigue MASS

            elif current_role_idx == 1:  # FORCE
                # FORCE → MASS: pierde si no tiene seguidores O energía muy baja
                influence_score = net_influence + 0.5
                loses_fi = (
                    mass_neighbors < 1 or
                    new_energy < 0.2 or
                    influence_score < -0.3  # Neural: influencia muy por debajo del promedio
                )
                if loses_fi:
                    new_role = torch.tensor([1.0, 0.0, 0.0])  # → MASS
                else:
                    new_role = torch.tensor([0.0, 1.0, 0.0])  # Sigue FORCE

            else:  # CORRUPT
                # CORRUPT compite con Fi existentes
                if fi_neighbors > 0 and net_influence > 0.5:
                    new_role = torch.tensor([0.0, 1.0, 0.0])  # → FORCE (gana)
                elif new_energy < 0.2:
                    new_role = torch.tensor([1.0, 0.0, 0.0])  # → MASS (pierde)
                else:
                    new_role = torch.tensor([0.0, 0.0, 1.0])  # Sigue CORRUPT

            # ========== MOVIMIENTO ==========
            x, y = cell.position
            new_role_idx = int(new_role.argmax().item())

            if new_role_idx == 0:  # MASS sigue gradiente hacia Fi
                grad = gradient[0, :, y, x]
                if abs(grad[0].item()) > 0.01 or abs(grad[1].item()) > 0.01:
                    dx = int(np.sign(grad[0].item()))
                    dy = int(np.sign(grad[1].item()))
                    x = np.clip(x + dx, 0, self.grid_size - 1)
                    y = np.clip(y + dy, 0, self.grid_size - 1)
            elif new_role_idx == 2:  # CORRUPT se mueve hacia Fi para competir
                if fi_neighbors > 0:
                    # Buscar Fi más cercano
                    fi_cells = [n for n in neighbors if n.role_idx == 1]
                    if fi_cells:
                        target = min(fi_cells, key=lambda f:
                            (f.position[0]-x)**2 + (f.position[1]-y)**2)
                        dx = int(np.sign(target.position[0] - x))
                        dy = int(np.sign(target.position[1] - y))
                        x = np.clip(x + dx, 0, self.grid_size - 1)
                        y = np.clip(y + dy, 0, self.grid_size - 1)

            new_cell = CellEntity(
                position=(x, y),
                state=new_state.detach(),
                role=new_role,
                energy=new_energy,
                controlled_mass=mass_neighbors
            )
            new_cells.append(new_cell)

        self.cells = new_cells
        self._update_grids()
        self.history.append(self.get_metrics())

    def get_metrics(self) -> dict:
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
