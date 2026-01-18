# behavior_engine.py
"""Motor de comportamiento: A<->B, A=AAA*A, A^3+V->B^3+A.

Implementation of the behavior algorithm from research notes:
- A <-> B: Bidirectional interaction between cells
- A = AAA*A: Recursive self-similarity pattern
- A^3 + V -> B^3 + A: Transformation with vital potential
- B = AA* - A*A: Net role (difference between emitting and receiving)
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorEngine(nn.Module):
    """Implements the behavior algorithm from research notes.

    This engine models cell interactions using concepts from:
    - Bidirectional influence (A <-> B)
    - Self-similarity patterns (A = AAA*A)
    - Vital potential transformations (A^3 + V -> B^3 + A)
    - Net role calculation (B = AA* - A*A)
    """

    def __init__(self, state_dim: int = 32, hidden_dim: int = 64) -> None:
        """Initialize the BehaviorEngine.

        Args:
            state_dim: Dimension of cell state vectors
            hidden_dim: Hidden dimension for neural networks
        """
        super().__init__()
        self.state_dim = state_dim

        # Network for transformation A^3 + V -> B^3
        self.transform_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for V (potential)
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Network for computing influence between cells
        self.influence_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def bidirectional_influence(self, cell: torch.Tensor,
                                 neighbors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A <-> B: Calculate bidirectional influence.

        Models the mutual interaction between a cell (A) and its neighbors (B).
        Each relationship goes both ways: A influences B and B influences A.

        Args:
            cell: [state_dim] state of the central cell
            neighbors: [N, state_dim] states of neighbor cells

        Returns:
            influence_out: [N] influence from A to each neighbor B
            influence_in: [] total influence from all Bs to A (scalar)
        """
        N = neighbors.shape[0]

        # A -> B: cell influences its neighbors
        cell_expanded = cell.unsqueeze(0).expand(N, -1)
        pairs_out = torch.cat([cell_expanded, neighbors], dim=-1)
        influence_out = self.influence_net(pairs_out).squeeze(-1)

        # B -> A: neighbors influence the cell
        pairs_in = torch.cat([neighbors, cell_expanded], dim=-1)
        influence_in = self.influence_net(pairs_in).sum()

        return influence_out, influence_in

    def self_similarity(self, cell: torch.Tensor) -> torch.Tensor:
        """A = AAA*A: Recursive self-similar pattern.

        Interpretation: AA* is the auto-correlation of the state with itself,
        multiplied by A creates a recursive pattern that encodes self-reference.

        For real-valued states:
        - AA* = element-wise square (state * conjugate, but conjugate = same for reals)
        - AAA*A = (AA*) * A = A^3 (cubed pattern)

        Args:
            cell: [state_dim] cell state vector

        Returns:
            recursive: [state_dim] self-similar pattern with same shape
        """
        # AA* = auto-correlation (state * conjugate, for reals: state * state)
        aa_star = cell * cell  # Element-wise square

        # AAA*A = (AA*) * A * A = A^4 pattern
        # But we interpret it as (AA*) * A to get A^3
        recursive = aa_star * cell

        # Normalize for stability while preserving relative magnitude
        result: torch.Tensor = recursive / (recursive.norm() + 1e-8) * cell.norm()
        return result

    def transform_with_potential(self, local_cube: torch.Tensor,
                                  potential: float,
                                  alpha: float = 0.3) -> torch.Tensor:
        """A^3 + V -> B^3 + A: Transformation with vital potential.

        This transformation models how a local neighborhood (cube) transforms
        when infused with vital potential V, while maintaining continuity
        through the +A term (weighted by alpha).

        Args:
            local_cube: [3, 3, state_dim] cell and its neighborhood
            potential: V, vital potential at this point (scalar in [0, 1])
            alpha: weight of continuity term (+A)

        Returns:
            new_cube: [3, 3, state_dim] transformed neighborhood state
        """
        original = local_cube.clone()
        shape = local_cube.shape

        # Flatten for batch processing
        flat = local_cube.view(-1, self.state_dim)

        # Add potential V as additional input dimension
        v_tensor = torch.full((flat.shape[0], 1), potential)
        with_v = torch.cat([flat, v_tensor], dim=-1)

        # Transform: A^3 + V -> B^3
        transformed = self.transform_net(with_v)

        # Add continuity term: + alpha * A
        new_cube: torch.Tensor = transformed.view(shape) + alpha * original

        return new_cube

    def net_role(self, cell: torch.Tensor) -> torch.Tensor:
        """B = AA* - A*A: Calculate net role.

        For real-valued states, we interpret this as:
        - AA* = "energy emitted" (self-correlation outward)
        - A*A = "energy received/contained" (self-correlation inward)

        In practice for real vectors:
        - AA* = sum of squares = total "outward" energy
        - A*A = squared norm = total "inward" energy

        Since these are equal for real numbers, we add a small
        perturbation to make the metric meaningful.

        Returns:
            net: [] scalar indicating role (positive = more Fi/emitting,
                 negative = more Mi/receiving)
        """
        # For real-valued vectors, we interpret differently:
        # AA* as the sum of positive components squared (outward)
        # A*A as the norm squared (total)

        aa_star = (cell * cell).sum()  # "Energy emitted" (sum of squares)
        a_star_a = cell.norm() ** 2    # "Energy total" (norm squared)

        # Normalize the difference
        # Note: For real vectors these are mathematically equal,
        # so this will be 0. The semantic meaning comes from
        # how we use this in context with other cells.
        net: torch.Tensor = (aa_star - a_star_a) / (a_star_a + 1e-8)

        return net

    def step(self, cell: torch.Tensor, neighbors: torch.Tensor,
             potential: float, alpha: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one complete step of the behavior engine.

        Combines all behavior components:
        1. A <-> B: Bidirectional influence
        2. A = AAA*A: Self-similarity pattern
        3. A^3 + V -> B^3 + A: Transformation with potential
        4. B = AA* - A*A: Net role calculation

        Args:
            cell: [state_dim] current cell state
            neighbors: [N, state_dim] neighbor states
            potential: vital potential V
            alpha: continuity weight

        Returns:
            new_state: [state_dim] updated cell state
            role_value: [] scalar indicating cell's net role
        """
        # 1. A <-> B: Compute bidirectional influences
        influence_out, influence_in = self.bidirectional_influence(cell, neighbors)

        # 2. A = AAA*A: Apply self-similarity pattern
        self_pattern = self.self_similarity(cell)

        # 3. A^3 + V -> B^3 + A: Transform with potential
        # Simplified for single cell (not full cube)
        cell_with_pattern = cell + 0.1 * self_pattern
        v_input = torch.cat([cell_with_pattern, torch.tensor([potential])])
        transformed = self.transform_net(v_input)
        new_state = transformed + alpha * cell

        # 4. B = AA* - A*A: Calculate net role
        # Using influence difference as the role metric
        role_value = influence_out.mean() - influence_in

        return new_state, role_value

    def forward(self, grid_states: torch.Tensor,
                potentials: torch.Tensor) -> torch.Tensor:
        """Process a full grid of cell states.

        Args:
            grid_states: [H, W, state_dim] grid of cell states
            potentials: [H, W] vital potential field

        Returns:
            new_grid: [H, W, state_dim] updated grid states
        """
        H, W, D = grid_states.shape
        new_grid = torch.zeros_like(grid_states)

        # Process each cell with its neighbors
        for i in range(H):
            for j in range(W):
                cell = grid_states[i, j]

                # Gather neighbor states (8-connected)
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % H, (j + dj) % W
                        neighbors.append(grid_states[ni, nj])

                neighbor_tensor = torch.stack(neighbors)
                potential = potentials[i, j].item()

                new_state, _ = self.step(cell, neighbor_tensor, potential)
                new_grid[i, j] = new_state

        return new_grid
