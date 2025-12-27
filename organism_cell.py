# organism_cell.py
"""Celula del organismo con NCA + Resonant."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import ZetaMemoryGatedSimple from zeta_resonance
try:
    from zeta_resonance import ZetaMemoryGatedSimple
except ImportError:
    # Fallback: define locally if import fails
    from force_field import get_zeta_zeros

    class ZetaMemoryGatedSimple(nn.Module):
        """Memoria zeta con gate aprendido."""

        def __init__(self, input_dim: int, hidden_dim: int,
                     M: int = 15, sigma: float = 0.1):
            super().__init__()
            self.input_dim = input_dim
            gammas = get_zeta_zeros(M)
            weights = [np.exp(-sigma * abs(g)) for g in gammas]
            self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))
            self.register_buffer('phi', torch.tensor(weights, dtype=torch.float32))
            self.gate_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            self.memory_net = nn.Linear(input_dim, input_dim)
            self.t = 0

        def forward(self, x: torch.Tensor) -> tuple:
            self.t += 1
            oscillation = (self.phi * torch.cos(self.gammas * self.t)).sum()
            zeta_mod = self.memory_net(x) * oscillation
            gate = self.gate_net(x)
            memory = gate * zeta_mod
            return memory, gate


class OrganismCell(nn.Module):
    """Celula individual del organismo artificial.

    Combina:
    - Percepcion del entorno (estado + vecinos + campo)
    - Memoria temporal gateada (ZetaResonant)
    - Deteccion de rol (MASS/FORCE/CORRUPT)
    """

    def __init__(self, state_dim: int = 32, hidden_dim: int = 64,
                 M: int = 15, sigma: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Percepcion: combina estado, vecinos, campo
        # Input: state_dim (state) + state_dim (neighbors_agg) + 2 (field_grad)
        self.perception_net = nn.Sequential(
            nn.Linear(state_dim + state_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Memoria temporal gateada
        self.resonant = ZetaMemoryGatedSimple(state_dim, hidden_dim, M=M, sigma=sigma)

        # Detector de rol: MASS (0), FORCE (1), CORRUPT (2)
        self.role_detector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Red de actualizacion
        self.update_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def perceive(self, state: torch.Tensor, neighbors: torch.Tensor,
                 field: torch.Tensor, position: tuple) -> torch.Tensor:
        """Percibe el entorno combinando estado, vecinos y campo.

        Args:
            state: [B, state_dim] - estado actual de la celula
            neighbors: [B, 8, state_dim] - estados de los 8 vecinos
            field: [B, 1, H, W] - campo de fuerzas
            position: (x, y) - posicion en el grid

        Returns:
            perception: [B, state_dim] - percepcion integrada
        """
        B = state.shape[0]

        # Agregar vecinos (promedio)
        neighbors_agg = neighbors.mean(dim=1)  # [B, state_dim]

        # Extraer gradiente del campo en la posicion
        x, y = position
        if field is not None and x > 0 and y > 0:
            H, W = field.shape[2], field.shape[3]
            # Gradiente en x
            x_next = min(x + 1, W - 1)
            x_prev = max(x - 1, 0)
            grad_x = field[:, 0, y, x_next] - field[:, 0, y, x_prev]
            # Gradiente en y
            y_next = min(y + 1, H - 1)
            y_prev = max(y - 1, 0)
            grad_y = field[:, 0, y_next, x] - field[:, 0, y_prev, x]
            field_grad = torch.stack([grad_x, grad_y], dim=-1)  # [B, 2]
        else:
            field_grad = torch.zeros(B, 2, device=state.device)

        # Combinar todo
        combined = torch.cat([state, neighbors_agg, field_grad], dim=-1)  # [B, state_dim*2 + 2]

        return self.perception_net(combined)

    def get_memory(self, perception: torch.Tensor) -> tuple:
        """Obtiene memoria gateada basada en percepcion.

        Args:
            perception: [B, state_dim] - percepcion del entorno

        Returns:
            memory: [B, state_dim] - memoria modulada
            gate: float - valor del gate (0-1)
        """
        memory, gate = self.resonant(perception)
        return memory, gate.mean().item()

    def detect_role(self, state: torch.Tensor) -> torch.Tensor:
        """Detecta el rol de la celula.

        Args:
            state: [B, state_dim] - estado de la celula

        Returns:
            role_probs: [B, 3] - probabilidades de cada rol (MASS, FORCE, CORRUPT)
        """
        logits = self.role_detector(state)
        return F.softmax(logits, dim=-1)

    def forward(self, state: torch.Tensor, neighbors: torch.Tensor,
                field: torch.Tensor, position: tuple) -> tuple:
        """Forward pass completo de la celula.

        Args:
            state: [B, state_dim] - estado actual
            neighbors: [B, 8, state_dim] - estados de vecinos
            field: [B, 1, H, W] - campo de fuerzas
            position: (x, y) - posicion en grid

        Returns:
            new_state: [B, state_dim] - nuevo estado
            role: [B, 3] - probabilidades de rol
        """
        # 1. Percibir entorno
        perception = self.perceive(state, neighbors, field, position)

        # 2. Obtener memoria gateada
        memory, gate = self.get_memory(perception)

        # 3. Combinar para actualizacion
        if gate > 0.5:
            # Aplicar memoria zeta
            update_input = torch.cat([perception, memory], dim=-1)
        else:
            # Sin memoria, usar percepcion duplicada para mantener dimension
            update_input = torch.cat([perception, perception], dim=-1)

        # 4. Calcular delta de estado
        delta = self.update_net(update_input)

        # 5. Actualizar estado con residual suave
        new_state = state + 0.1 * delta

        # 6. Detectar rol
        role = self.detect_role(new_state)

        return new_state, role


if __name__ == '__main__':
    print('=' * 70)
    print('OrganismCell - Test')
    print('=' * 70)

    # Test basico
    cell = OrganismCell(state_dim=32, hidden_dim=64)

    state = torch.randn(1, 32)
    neighbors = torch.randn(1, 8, 32)
    field = torch.randn(1, 1, 16, 16)

    new_state, role = cell(state, neighbors, field, position=(8, 8))

    print(f'State shape: {state.shape}')
    print(f'New state shape: {new_state.shape}')
    print(f'Role shape: {role.shape}')
    print(f'Role probs: {role}')
    print(f'Role sum: {role.sum().item():.4f}')

    # Test percepcion
    perception = cell.perceive(state, neighbors, field, position=(8, 8))
    print(f'\nPerception shape: {perception.shape}')

    # Test memoria
    memory, gate = cell.get_memory(perception)
    print(f'Memory shape: {memory.shape}')
    print(f'Gate value: {gate:.4f}')

    # Contar parametros
    params = sum(p.numel() for p in cell.parameters())
    print(f'\nTotal parameters: {params}')

    print('\n[OK] All tests passed!')
