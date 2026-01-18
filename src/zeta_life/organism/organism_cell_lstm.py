# organism_cell_lstm.py
"""Celula del organismo con memoria ZetaLSTM integrada.

Evolucion de OrganismCell: reemplaza ZetaMemoryGatedSimple por ZetaLSTMCell
para capturar dependencias temporales mas complejas.
"""
from collections.abc import Iterator
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.zeta_rnn import ZetaLSTMCell, ZetaMemoryLayer


class OrganismCellLSTM(nn.Module):
    """Celula con memoria LSTM enriquecida con kernel zeta.

    Diferencias vs OrganismCell original:
    - Usa ZetaLSTMCell en lugar de ZetaMemoryGatedSimple
    - Mantiene estado (h, c) persistente entre timesteps
    - La memoria zeta modula el hidden state del LSTM

    Arquitectura:
        perception -> ZetaLSTMCell -> update -> new_state
                         ^
                         |
                    (h_t, c_t) persistentes
    """

    def __init__(self, state_dim: int = 32, hidden_dim: int = 64,
                 M: int = 15, sigma: float = 0.1, zeta_weight: float = 0.2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Percepcion: combina estado, vecinos, campo
        # Input: state_dim (state) + state_dim (neighbors_agg) + 2 (field_grad)
        perception_input_dim = state_dim + state_dim + 2
        self.perception_net = nn.Sequential(
            nn.Linear(perception_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Memoria temporal: ZetaLSTMCell (reemplaza ZetaMemoryGatedSimple)
        self.lstm_cell = ZetaLSTMCell(
            input_size=state_dim,
            hidden_size=hidden_dim,
            M=M,
            sigma=sigma,
            zeta_weight=zeta_weight
        )

        # Estado LSTM persistente (inicializado en reset())
        self.h: torch.Tensor | None = None
        self.c: torch.Tensor | None = None
        self.t: int = 0  # timestep para ZetaLSTMCell

        # Detector de rol: MASS (0), FORCE (1), CORRUPT (2)
        self.role_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Proyeccion de hidden a state
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, state_dim)
        )

    def reset_memory(self, batch_size: int = 1, device=None) -> None:
        """Resetea el estado LSTM."""
        if device is None:
            device = next(self.parameters()).device
        self.h = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.c = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.t = 0

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
        device = state.device

        # Agregar vecinos (promedio)
        if neighbors.shape[1] > 0:
            neighbors_agg = neighbors.mean(dim=1)  # [B, state_dim]
        else:
            neighbors_agg = torch.zeros(B, self.state_dim, device=device)

        # Extraer gradiente del campo en la posicion
        x, y = position
        if field is not None and field.numel() > 0:
            H, W = field.shape[2], field.shape[3]
            x = min(max(x, 1), W - 2)
            y = min(max(y, 1), H - 2)
            # Gradiente en x
            grad_x = field[:, 0, y, min(x + 1, W-1)] - field[:, 0, y, max(x - 1, 0)]
            # Gradiente en y
            grad_y = field[:, 0, min(y + 1, H-1), x] - field[:, 0, max(y - 1, 0), x]
            field_grad = torch.stack([grad_x, grad_y], dim=-1)  # [B, 2]
        else:
            field_grad = torch.zeros(B, 2, device=device)

        # Combinar todo
        combined = torch.cat([state, neighbors_agg, field_grad], dim=-1)

        result: torch.Tensor = self.perception_net(combined)
        return result

    def detect_role(self, hidden: torch.Tensor) -> torch.Tensor:
        """Detecta el rol basado en el hidden state del LSTM.

        Args:
            hidden: [B, hidden_dim] - hidden state del LSTM

        Returns:
            role_probs: [B, 3] - probabilidades de cada rol
        """
        logits = self.role_detector(hidden)
        return F.softmax(logits, dim=-1)

    def forward(self, state: torch.Tensor, neighbors: torch.Tensor,
                field: torch.Tensor, position: tuple) -> tuple:
        """Forward pass con memoria LSTM persistente.

        Args:
            state: [B, state_dim] - estado actual
            neighbors: [B, N, state_dim] - estados de vecinos
            field: [B, 1, H, W] - campo de fuerzas
            position: (x, y) - posicion en grid

        Returns:
            new_state: [B, state_dim] - nuevo estado
            role: [B, 3] - probabilidades de rol
            memory_info: dict con info de memoria
        """
        B = state.shape[0]
        device = state.device

        # Inicializar memoria si es necesario
        if self.h is None or self.h.shape[0] != B:
            self.reset_memory(B, device)

        # 1. Percibir entorno
        perception = self.perceive(state, neighbors, field, position)

        # 2. Pasar por ZetaLSTMCell (con memoria temporal)
        # After reset_memory, h and c are guaranteed to be Tensor
        assert self.h is not None and self.c is not None
        self.h, self.c = self.lstm_cell(perception, (self.h, self.c), t=self.t)
        self.t += 1

        # 3. Proyectar hidden a state
        delta = self.output_net(self.h)

        # 4. Actualizar estado con residual
        new_state = state + 0.1 * delta

        # 5. Detectar rol basado en hidden state
        role = self.detect_role(self.h)

        # Info de memoria for debugging
        # h and c are guaranteed to be Tensor after lstm_cell call
        memory_info = {
            'h_norm': self.h.norm().item(),
            'c_norm': self.c.norm().item(),
            't': self.t
        }

        return new_state, role, memory_info


class OrganismCellLSTMPool:
    """Pool de celdas LSTM para el organismo.

    Maneja multiples celdas, cada una con su propio estado LSTM.
    """

    def __init__(self, n_cells: int, state_dim: int = 32, hidden_dim: int = 64,
                 M: int = 15, sigma: float = 0.1, zeta_weight: float = 0.2):
        self.n_cells = n_cells
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Una celda compartida (pesos) pero estados separados
        self.cell = OrganismCellLSTM(state_dim, hidden_dim, M, sigma, zeta_weight)

        # Estados LSTM individuales por celula
        self.h_states: dict[int, torch.Tensor] = {}  # cell_id -> h tensor
        self.c_states: dict[int, torch.Tensor] = {}  # cell_id -> c tensor
        self.t: int = 0

    def reset(self, device=None) -> None:
        """Resetea todos los estados."""
        if device is None:
            device = next(self.cell.parameters()).device
        self.h_states = {}
        self.c_states = {}
        self.t = 0

    def get_state(self, cell_id: int, device) -> tuple:
        """Obtiene o crea estado LSTM para una celula."""
        if cell_id not in self.h_states:
            self.h_states[cell_id] = torch.zeros(1, self.hidden_dim, device=device)
            self.c_states[cell_id] = torch.zeros(1, self.hidden_dim, device=device)
        return self.h_states[cell_id], self.c_states[cell_id]

    def set_state(self, cell_id: int, h: torch.Tensor, c: torch.Tensor) -> None:
        """Guarda estado LSTM de una celula."""
        self.h_states[cell_id] = h
        self.c_states[cell_id] = c

    def forward(self, cell_id: int, state: torch.Tensor, neighbors: torch.Tensor,
                field: torch.Tensor, position: tuple) -> tuple:
        """Forward para una celula especifica."""
        device = state.device

        # Obtener estado LSTM de esta celula
        h, c = self.get_state(cell_id, device)

        # Configurar celda con este estado
        self.cell.h = h
        self.cell.c = c
        self.cell.t = self.t

        # Forward
        new_state, role, memory_info = self.cell(state, neighbors, field, position)

        # Guardar estado actualizado
        # After forward, h and c are guaranteed to be Tensor
        assert self.cell.h is not None and self.cell.c is not None
        self.set_state(cell_id, self.cell.h, self.cell.c)

        return new_state, role, memory_info

    def step_time(self) -> None:
        """Avanza timestep global."""
        self.t += 1

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Retorna parametros de la celda compartida."""
        return self.cell.parameters()

    def state_dict(self) -> dict:
        """Retorna state dict de la celda."""
        return self.cell.state_dict()

    def load_state_dict(self, state_dict) -> None:
        """Carga state dict."""
        self.cell.load_state_dict(state_dict)


if __name__ == '__main__':
    print('=' * 70)
    print('OrganismCellLSTM - Test')
    print('=' * 70)

    # Test celula individual
    cell = OrganismCellLSTM(state_dim=32, hidden_dim=64, M=15, zeta_weight=0.2)

    state = torch.randn(1, 32)
    neighbors = torch.randn(1, 8, 32)
    field = torch.randn(1, 1, 16, 16)

    # Multiples pasos para ver evolucion de memoria
    print("\nEvolucion de memoria LSTM:")
    for t in range(5):
        new_state, role, mem_info = cell(state, neighbors, field, position=(8, 8))
        print(f"  t={t}: h_norm={mem_info['h_norm']:.4f}, c_norm={mem_info['c_norm']:.4f}")
        state = new_state

    print(f'\nState shape: {state.shape}')
    print(f'Role shape: {role.shape}')
    print(f'Role probs: {role}')

    # Test pool
    print("\n" + "=" * 70)
    print("OrganismCellLSTMPool - Test")
    print("=" * 70)

    pool = OrganismCellLSTMPool(n_cells=10, state_dim=32, hidden_dim=64)

    # Simular 3 celulas diferentes
    for cell_id in [0, 5, 9]:
        state = torch.randn(1, 32)
        neighbors = torch.randn(1, 8, 32)

        new_state, role, mem_info = pool.forward(
            cell_id, state, neighbors, field, position=(cell_id, cell_id)
        )
        print(f"Cell {cell_id}: h_norm={mem_info['h_norm']:.4f}")

    pool.step_time()
    print(f"\nGlobal timestep: {pool.t}")

    # Contar parametros
    params = sum(p.numel() for p in pool.parameters())
    print(f'\nTotal parameters: {params}')

    print('\n[OK] All tests passed!')
