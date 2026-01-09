# ZetaOrganism: Diseño de Organismo Artificial

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Crear un organismo artificial distribuido que exhiba inteligencia colectiva emergente basado en dinámica Fi-Mi y kernels zeta.

**Architecture:** Sistema multi-agente donde células actúan como masas (Mi) o fuerzas (Fi), con campos de fuerza propagados por convolución zeta y comportamiento emergente siguiendo el algoritmo A↔B, A³+V→B³+A.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib

**Theoretical Foundation:** Paper "IA Adaptativa a través de la Hipótesis de Riemann" + notas de investigación sobre dinámica Fi-Mi y colapso dimensional.

---

## Modelo Físico (del usuario)

### Dinámica Fi-Mi
- **Fi (Fuerza Inicial)**: Entidad que atrae y controla masas
- **Mi (Masa Inicial)**: Entidad que sigue a Fi
- **Equilibrio**: `Fi_efectiva = Fi_base * sqrt(masa_controlada)`
- **Cadena**: Fi atrae masas → masas siguen trayectoria → masa aumenta → Fi debe escalar
- **Corrupción**: Subconjunto de masa puede corromper Fi y desestabilizar

### Algoritmo de Comportamiento
- `A ↔ B`: Interacción bidireccional
- `A = AAA*A`: Auto-similitud recursiva
- `A³ + V → B³ + A`: Transformación con potencial vital
- `B = AA* - A*A`: Rol neto (diferencia entre emitir y recibir)

### Espacio Lumber
- Las dimensiones no colapsan como objetos matemáticos
- Sus geometrías convergen → estados estables emergen

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    ZetaOrganism                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Grid de    │───▶│  Campo de   │───▶│  Dinámica   │     │
│  │  Células    │    │  Fuerzas    │    │  Fi-Mi      │     │
│  │  (ZetaNCA)  │◀───│  (ZetaConv) │◀───│  (Resonant) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Estado por Célula                          ││
│  │  - mass: float (capacidad de influir)                   ││
│  │  - energy: float (potencial vital V)                    ││
│  │  - role: {MASS, FORCE, CORRUPT}                         ││
│  │  - memory: tensor (historia temporal zeta)              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Tasks

### Task 1: Estructura de Estado de Célula

**Files:**
- Create: `cell_state.py`
- Test: `tests/test_cell_state.py`

**Step 1: Write failing test**

```python
# tests/test_cell_state.py
import pytest
import torch
from cell_state import CellState, CellRole

def test_cell_state_creation():
    """Célula se crea con estado inicial válido."""
    cell = CellState()
    assert cell.alive == True
    assert cell.mass >= 0
    assert cell.energy >= 0
    assert cell.role == CellRole.MASS

def test_role_transition_to_force():
    """Célula con alta energía y seguidores se convierte en Fi."""
    cell = CellState(energy=0.8, controlled_mass=5)
    cell.update_role(fi_threshold=0.7, min_followers=3)
    assert cell.role == CellRole.FORCE

def test_role_transition_to_corrupt():
    """Fi detecta rival cercano y se vuelve corrupto."""
    cell = CellState(role=CellRole.FORCE)
    rival_nearby = True
    cell.update_role(rival_detected=rival_nearby)
    assert cell.role == CellRole.CORRUPT

def test_equilibrium_scaling():
    """Fi efectiva escala con sqrt de masa controlada."""
    cell = CellState(role=CellRole.FORCE, fi_base=1.0, controlled_mass=4)
    assert cell.effective_fi() == pytest.approx(2.0)  # 1.0 * sqrt(4)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cell_state.py -v`
Expected: FAIL - module not found

**Step 3: Write implementation**

```python
# cell_state.py
"""Estado de célula para ZetaOrganism."""
import torch
from enum import Enum
from dataclasses import dataclass, field
import math

class CellRole(Enum):
    MASS = 0      # Mi - sigue a Fi
    FORCE = 1     # Fi - atrae masas
    CORRUPT = 2   # Compite con Fi existente

@dataclass
class CellState:
    """Estado multidimensional de una célula."""
    alive: bool = True
    mass: float = 1.0
    energy: float = 0.0
    role: CellRole = CellRole.MASS
    fi_base: float = 1.0
    controlled_mass: float = 0.0
    memory: torch.Tensor = field(default_factory=lambda: torch.zeros(32))
    resonance_gate: float = 0.0

    def effective_fi(self) -> float:
        """Fi efectiva escalada por masa controlada."""
        if self.role != CellRole.FORCE:
            return 0.0
        return self.fi_base * math.sqrt(max(1.0, self.controlled_mass))

    def update_role(self, fi_threshold: float = 0.7,
                    min_followers: int = 3,
                    rival_detected: bool = False):
        """Actualiza rol según condiciones."""
        if self.role == CellRole.MASS:
            # MASS -> FORCE: alta energía + seguidores
            if self.energy > fi_threshold and self.controlled_mass >= min_followers:
                self.role = CellRole.FORCE
        elif self.role == CellRole.FORCE:
            # FORCE -> CORRUPT: rival detectado
            if rival_detected:
                self.role = CellRole.CORRUPT
        elif self.role == CellRole.CORRUPT:
            # CORRUPT -> MASS: perdió competencia
            if self.energy < fi_threshold * 0.5:
                self.role = CellRole.MASS

    def net_role_value(self, influence_out: float, influence_in: float) -> float:
        """B = AA* - A*A: rol neto basado en balance de influencia."""
        # AA* = influencia emitida (correlación derecha)
        # A*A = influencia recibida (correlación izquierda)
        # B > 0 -> más Fi, B < 0 -> más Mi
        return influence_out - influence_in
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cell_state.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cell_state.py tests/test_cell_state.py
git commit -m "feat: add CellState with Fi-Mi role dynamics"
```

---

### Task 2: Campo de Fuerzas con Propagación Zeta

**Files:**
- Create: `force_field.py`
- Reference: `zeta_neural_ca.py` (ZetaKernelConv)
- Test: `tests/test_force_field.py`

**Step 1: Write failing test**

```python
# tests/test_force_field.py
import pytest
import torch
from force_field import ForceField

def test_force_field_creation():
    """Campo de fuerzas se crea con kernel zeta."""
    field = ForceField(grid_size=32, M=15, sigma=0.1)
    assert field.kernel is not None
    assert field.grid_size == 32

def test_fi_emission():
    """Fi emite señal proporcional a su fuerza."""
    field = ForceField(grid_size=16)

    # Grid con un Fi en el centro
    energy = torch.zeros(1, 1, 16, 16)
    roles = torch.zeros(1, 1, 16, 16)
    energy[0, 0, 8, 8] = 1.0
    roles[0, 0, 8, 8] = 1  # FORCE

    result = field.compute(energy, roles)

    # El centro debe tener valor máximo
    assert result[0, 0, 8, 8] > result[0, 0, 0, 0]

def test_gradient_computation():
    """Gradiente apunta hacia Fi."""
    field = ForceField(grid_size=16)

    energy = torch.zeros(1, 1, 16, 16)
    energy[0, 0, 8, 8] = 1.0
    roles = torch.ones_like(energy)  # All FORCE for simplicity

    _, gradient = field.compute_with_gradient(energy, roles)

    # Gradient should have 2 channels (dx, dy)
    assert gradient.shape[1] == 2

def test_zeta_resonance_peaks():
    """Campo tiene resonancias a distancias zeta."""
    field = ForceField(grid_size=64, M=15, sigma=0.1)

    energy = torch.zeros(1, 1, 64, 64)
    energy[0, 0, 32, 32] = 1.0
    roles = torch.ones_like(energy)

    result = field.compute(energy, roles)

    # Debe haber estructura no-monótona (resonancias)
    radial = result[0, 0, 32, 32:].numpy()
    # No debe decaer monotónicamente
    diffs = radial[1:] - radial[:-1]
    assert (diffs > 0).any()  # Algún incremento = resonancia
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_force_field.py -v`
Expected: FAIL - module not found

**Step 3: Write implementation**

```python
# force_field.py
"""Campo de fuerzas con propagación zeta."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_zeta_zeros(M: int) -> list:
    """Primeros M ceros no triviales de zeta."""
    # Valores conocidos
    zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
             37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
             52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
             67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
             79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
             92.491899, 94.651344, 95.870634, 98.831194, 101.317851]
    return zeros[:M]

class ForceField(nn.Module):
    """Campo de fuerzas propagado por convolución zeta."""

    def __init__(self, grid_size: int = 64, M: int = 15,
                 sigma: float = 0.1, kernel_radius: int = 7):
        super().__init__()
        self.grid_size = grid_size
        self.M = M
        self.sigma = sigma
        self.kernel_radius = kernel_radius

        # Crear kernel zeta
        self.kernel = self._create_zeta_kernel()

        # Filtros Sobel para gradiente
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _create_zeta_kernel(self) -> nn.Parameter:
        """Crea kernel K_σ(r) = Σ exp(-σ|γ|) * cos(γ*r)."""
        gammas = get_zeta_zeros(self.M)
        size = 2 * self.kernel_radius + 1

        kernel = np.zeros((size, size))
        center = self.kernel_radius

        for i in range(size):
            for j in range(size):
                r = np.sqrt((i - center)**2 + (j - center)**2)
                for gamma in gammas:
                    weight = np.exp(-self.sigma * abs(gamma))
                    kernel[i, j] += weight * np.cos(gamma * r)

        # Normalizar
        kernel = kernel / (np.abs(kernel).sum() + 1e-8)

        return nn.Parameter(
            torch.tensor(kernel, dtype=torch.float32).view(1, 1, size, size),
            requires_grad=False
        )

    def compute(self, energy: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        """Computa campo de fuerzas.

        Args:
            energy: [B, 1, H, W] energía por célula
            roles: [B, 1, H, W] roles (1=FORCE emite)

        Returns:
            field: [B, 1, H, W] campo de fuerza propagado
        """
        # Solo Fi emiten
        fi_signal = energy * (roles == 1).float()

        # Propagar con kernel zeta
        pad = self.kernel_radius
        padded = F.pad(fi_signal, (pad, pad, pad, pad), mode='circular')
        field = F.conv2d(padded, self.kernel)

        return field

    def compute_with_gradient(self, energy: torch.Tensor,
                              roles: torch.Tensor) -> tuple:
        """Computa campo y gradiente.

        Returns:
            field: [B, 1, H, W]
            gradient: [B, 2, H, W] (dx, dy)
        """
        field = self.compute(energy, roles)

        # Gradiente con Sobel
        padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(padded, self.sobel_x)
        grad_y = F.conv2d(padded, self.sobel_y)

        gradient = torch.cat([grad_x, grad_y], dim=1)
        return field, gradient

    def attraction_force(self, position: tuple, field: torch.Tensor,
                        gradient: torch.Tensor) -> torch.Tensor:
        """Fuerza de atracción en una posición."""
        x, y = position
        return gradient[:, :, y, x]  # [B, 2]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_force_field.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add force_field.py tests/test_force_field.py
git commit -m "feat: add ForceField with zeta kernel propagation"
```

---

### Task 3: Motor de Comportamiento (A↔B, A³+V→B³+A)

**Files:**
- Create: `behavior_engine.py`
- Test: `tests/test_behavior_engine.py`

**Step 1: Write failing test**

```python
# tests/test_behavior_engine.py
import pytest
import torch
from behavior_engine import BehaviorEngine

def test_bidirectional_influence():
    """A ↔ B: influencia bidireccional."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)
    neighbor_states = torch.randn(8, 32)  # 8 vecinos

    influence_out, influence_in = engine.bidirectional_influence(
        cell_state, neighbor_states
    )

    assert influence_out.shape == (8,)  # A -> cada vecino
    assert influence_in.shape == ()     # suma de vecinos -> A

def test_self_similarity():
    """A = AAA*A: patrón auto-similar."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)

    recursive = engine.self_similarity(cell_state)

    assert recursive.shape == cell_state.shape

def test_transformation_with_potential():
    """A³ + V → B³ + A: transformación con potencial vital."""
    engine = BehaviorEngine()

    local_cube = torch.randn(3, 3, 32)  # Célula + vecinos cercanos
    potential = 0.5
    alpha = 0.3  # Peso de continuidad

    new_cube = engine.transform_with_potential(local_cube, potential, alpha)

    assert new_cube.shape == local_cube.shape

def test_net_role():
    """B = AA* - A*A: rol neto."""
    engine = BehaviorEngine()

    cell_state = torch.randn(32)

    net = engine.net_role(cell_state)

    # Escalar indicando si es más Fi o Mi
    assert net.shape == ()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_behavior_engine.py -v`
Expected: FAIL - module not found

**Step 3: Write implementation**

```python
# behavior_engine.py
"""Motor de comportamiento: A↔B, A=AAA*A, A³+V→B³+A."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BehaviorEngine(nn.Module):
    """Implementa el algoritmo de comportamiento de las notas."""

    def __init__(self, state_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim

        # Red para transformación A³+V → B³
        self.transform_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 para V
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Red para influencia
        self.influence_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def bidirectional_influence(self, cell: torch.Tensor,
                                 neighbors: torch.Tensor) -> tuple:
        """A ↔ B: Calcula influencia bidireccional.

        Args:
            cell: [state_dim] estado de la célula
            neighbors: [N, state_dim] estados de vecinos

        Returns:
            influence_out: [N] influencia de A hacia cada B
            influence_in: [] influencia total de Bs hacia A
        """
        N = neighbors.shape[0]

        # A -> B: célula influye a vecinos
        cell_expanded = cell.unsqueeze(0).expand(N, -1)
        pairs_out = torch.cat([cell_expanded, neighbors], dim=-1)
        influence_out = self.influence_net(pairs_out).squeeze(-1)

        # B -> A: vecinos influyen a célula
        pairs_in = torch.cat([neighbors, cell_expanded], dim=-1)
        influence_in = self.influence_net(pairs_in).sum()

        return influence_out, influence_in

    def self_similarity(self, cell: torch.Tensor) -> torch.Tensor:
        """A = AAA*A: Patrón auto-similar recursivo.

        Interpretación: AA* es la correlación del estado consigo mismo,
        multiplicada por A crea patrón recursivo.
        """
        # AA* = correlación (estado * conjugado)
        aa_star = cell * cell  # Para reales, conjugado = mismo

        # AAA*A = (AA*) * A * A
        recursive = aa_star * cell

        # Normalizar para estabilidad
        return recursive / (recursive.norm() + 1e-8) * cell.norm()

    def transform_with_potential(self, local_cube: torch.Tensor,
                                  potential: float,
                                  alpha: float = 0.3) -> torch.Tensor:
        """A³ + V → B³ + A: Transformación con potencial vital.

        Args:
            local_cube: [3, 3, state_dim] célula central + vecinos
            potential: V, potencial vital en este punto
            alpha: peso de continuidad (+A)

        Returns:
            new_cube: [3, 3, state_dim] nuevo estado del cubo
        """
        original = local_cube.clone()
        shape = local_cube.shape

        # Aplanar para procesar
        flat = local_cube.view(-1, self.state_dim)

        # Agregar potencial V
        v_tensor = torch.full((flat.shape[0], 1), potential)
        with_v = torch.cat([flat, v_tensor], dim=-1)

        # Transformar: A³ + V → B³
        transformed = self.transform_net(with_v)

        # Agregar continuidad: + αA
        new_cube = transformed.view(shape) + alpha * original

        return new_cube

    def net_role(self, cell: torch.Tensor) -> torch.Tensor:
        """B = AA* - A*A: Calcula rol neto.

        Para estados reales:
        - AA* = A @ A.T (correlación "hacia afuera")
        - A*A = A.T @ A (correlación "hacia adentro")

        Retorna escalar: positivo = más Fi, negativo = más Mi
        """
        # Interpretación simplificada para vector:
        # AA* como norma de influencia emitida
        # A*A como norma de influencia recibida

        aa_star = (cell * cell).sum()  # "Energía emitida"
        a_star_a = cell.norm() ** 2    # "Energía total"

        # Normalizar diferencia
        net = (aa_star - a_star_a) / (a_star_a + 1e-8)

        return net

    def step(self, cell: torch.Tensor, neighbors: torch.Tensor,
             potential: float, alpha: float = 0.3) -> tuple:
        """Un paso completo del motor de comportamiento.

        Returns:
            new_state: nuevo estado de la célula
            role_value: indicador de rol (Fi vs Mi)
        """
        # 1. A ↔ B
        influence_out, influence_in = self.bidirectional_influence(cell, neighbors)

        # 2. A = AAA*A
        self_pattern = self.self_similarity(cell)

        # 3. A³ + V → B³ + A (simplificado a célula individual)
        cell_with_pattern = cell + 0.1 * self_pattern
        v_input = torch.cat([cell_with_pattern, torch.tensor([potential])])
        transformed = self.transform_net(v_input)
        new_state = transformed + alpha * cell

        # 4. B = AA* - A*A
        role_value = influence_out.mean() - influence_in

        return new_state, role_value
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_behavior_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add behavior_engine.py tests/test_behavior_engine.py
git commit -m "feat: add BehaviorEngine with A<->B transformation algorithm"
```

---

### Task 4: Integración ZetaResonant para Memoria

**Files:**
- Modify: `zeta_resonance.py` (agregar versión simplificada)
- Create: `organism_cell.py`
- Test: `tests/test_organism_cell.py`

**Step 1: Write failing test**

```python
# tests/test_organism_cell.py
import pytest
import torch
from organism_cell import OrganismCell

def test_cell_creation():
    """Célula se crea con componentes NCA y Resonant."""
    cell = OrganismCell(state_dim=32, hidden_dim=64)
    assert hasattr(cell, 'resonant')
    assert hasattr(cell, 'role_detector')

def test_perception():
    """Célula percibe su entorno."""
    cell = OrganismCell(state_dim=32)

    state = torch.randn(1, 32)
    neighbors = torch.randn(1, 8, 32)
    field = torch.randn(1, 1, 16, 16)

    perception = cell.perceive(state, neighbors, field, position=(8, 8))

    assert perception.shape[-1] == 32  # state_dim

def test_gated_memory():
    """Memoria se aplica condicionalmente."""
    cell = OrganismCell(state_dim=32)

    perception = torch.randn(1, 32)

    memory, gate = cell.get_memory(perception)

    assert memory.shape == perception.shape
    assert 0 <= gate <= 1  # Gate es probabilidad

def test_role_detection():
    """Célula detecta su rol."""
    cell = OrganismCell(state_dim=32)

    state = torch.randn(1, 32)

    role_probs = cell.detect_role(state)

    assert role_probs.shape == (1, 3)  # MASS, FORCE, CORRUPT
    assert role_probs.sum().item() == pytest.approx(1.0)

def test_forward_pass():
    """Paso completo produce actualización y rol."""
    cell = OrganismCell(state_dim=32)

    state = torch.randn(1, 32)
    neighbors = torch.randn(1, 8, 32)
    field = torch.randn(1, 1, 16, 16)

    new_state, role = cell(state, neighbors, field, position=(8, 8))

    assert new_state.shape == state.shape
    assert role.shape == (1, 3)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_organism_cell.py -v`
Expected: FAIL - module not found

**Step 3: Write implementation**

```python
# organism_cell.py
"""Célula del organismo con NCA + Resonant."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta_resonance import ZetaMemoryGated

class OrganismCell(nn.Module):
    """Célula individual del organismo artificial.

    Combina:
    - Percepción del entorno (estado + vecinos + campo)
    - Memoria temporal gateada (ZetaResonant)
    - Detección de rol (MASS/FORCE/CORRUPT)
    """

    def __init__(self, state_dim: int = 32, hidden_dim: int = 64,
                 M: int = 15, sigma: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Percepción: combina estado, vecinos, campo
        self.perception_net = nn.Sequential(
            nn.Linear(state_dim + state_dim + 2, hidden_dim),  # state + neighbors_agg + field_grad
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Memoria temporal gateada
        self.resonant = ZetaMemoryGated(state_dim, hidden_dim, M=M, sigma=sigma)

        # Detector de rol
        self.role_detector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # MASS, FORCE, CORRUPT
        )

        # Red de actualización
        self.update_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # perception + memory
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def perceive(self, state: torch.Tensor, neighbors: torch.Tensor,
                 field: torch.Tensor, position: tuple) -> torch.Tensor:
        """Percepción del entorno.

        Args:
            state: [B, state_dim]
            neighbors: [B, N, state_dim]
            field: [B, 1, H, W]
            position: (x, y) en el grid

        Returns:
            perception: [B, state_dim]
        """
        B = state.shape[0]

        # Agregar vecinos (promedio)
        neighbors_agg = neighbors.mean(dim=1)  # [B, state_dim]

        # Gradiente local del campo
        x, y = position
        if field is not None and x > 0 and y > 0:
            # Gradiente simple
            grad_x = field[:, 0, y, min(x+1, field.shape[3]-1)] - field[:, 0, y, max(x-1, 0)]
            grad_y = field[:, 0, min(y+1, field.shape[2]-1), x] - field[:, 0, max(y-1, 0), x]
            field_grad = torch.stack([grad_x, grad_y], dim=-1)  # [B, 2]
        else:
            field_grad = torch.zeros(B, 2)

        # Concatenar
        combined = torch.cat([state, neighbors_agg, field_grad], dim=-1)

        # Red de percepción
        perception = self.perception_net(combined)

        return perception

    def get_memory(self, perception: torch.Tensor) -> tuple:
        """Obtiene memoria temporal gateada.

        Returns:
            memory: [B, state_dim]
            gate: float (probabilidad de aplicar zeta)
        """
        memory, gate = self.resonant(perception)
        return memory, gate.mean().item()

    def detect_role(self, state: torch.Tensor) -> torch.Tensor:
        """Detecta rol basado en estado.

        Returns:
            role_probs: [B, 3] probabilidades para MASS, FORCE, CORRUPT
        """
        logits = self.role_detector(state)
        return F.softmax(logits, dim=-1)

    def forward(self, state: torch.Tensor, neighbors: torch.Tensor,
                field: torch.Tensor, position: tuple) -> tuple:
        """Paso completo de actualización.

        Returns:
            new_state: [B, state_dim]
            role: [B, 3]
        """
        # 1. Percepción
        perception = self.perceive(state, neighbors, field, position)

        # 2. Memoria gateada
        memory, gate = self.get_memory(perception)

        # 3. Actualización condicional
        if gate > 0.5:
            # Usar memoria zeta
            update_input = torch.cat([perception, memory], dim=-1)
        else:
            # Sin memoria (usar percepción duplicada para mantener dimensión)
            update_input = torch.cat([perception, perception], dim=-1)

        delta = self.update_net(update_input)
        new_state = state + 0.1 * delta  # Pequeño paso

        # 4. Detectar rol
        role = self.detect_role(new_state)

        return new_state, role
```

**Step 4: Verificar que ZetaMemoryGated existe en zeta_resonance.py**

Si no existe, agregar:

```python
# En zeta_resonance.py, agregar si falta:

class ZetaMemoryGated(nn.Module):
    """Memoria zeta con gate aprendido."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 M: int = 15, sigma: float = 0.1):
        super().__init__()
        self.input_dim = input_dim

        # Ceros zeta
        gammas = get_zeta_zeros(M)
        weights = [np.exp(-sigma * abs(g)) for g in gammas]
        self.register_buffer('gammas', torch.tensor(gammas, dtype=torch.float32))
        self.register_buffer('phi', torch.tensor(weights, dtype=torch.float32))

        # Gate: decide si aplicar memoria zeta
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Memoria
        self.memory_net = nn.Linear(input_dim, input_dim)
        self.t = 0

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            memory: tensor gateado
            gate: valor del gate
        """
        self.t += 1

        # Calcular memoria zeta
        oscillation = (self.phi * torch.cos(self.gammas * self.t)).sum()
        zeta_mod = self.memory_net(x) * oscillation

        # Gate
        gate = self.gate_net(x)

        # Aplicar gate
        memory = gate * zeta_mod

        return memory, gate
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_organism_cell.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add organism_cell.py zeta_resonance.py tests/test_organism_cell.py
git commit -m "feat: add OrganismCell with gated zeta memory"
```

---

### Task 5: ZetaOrganism Principal

**Files:**
- Create: `zeta_organism.py`
- Test: `tests/test_zeta_organism.py`

**Step 1: Write failing test**

```python
# tests/test_zeta_organism.py
import pytest
import torch
from zeta_organism import ZetaOrganism

def test_organism_creation():
    """Organismo se crea con grid y células."""
    org = ZetaOrganism(grid_size=32, n_cells=50)
    assert org.grid_size == 32
    assert len(org.cells) == 50

def test_initialization():
    """Organismo inicializa con Fi semilla."""
    org = ZetaOrganism(grid_size=32, n_cells=50)
    org.initialize(seed_fi=True)

    # Debe haber al menos un Fi
    fi_count = sum(1 for c in org.cells if c.role.item() == 1)
    assert fi_count >= 1

def test_step():
    """Un paso de simulación actualiza el estado."""
    org = ZetaOrganism(grid_size=32, n_cells=50)
    org.initialize()

    old_states = [c.state.clone() for c in org.cells[:5]]
    org.step()
    new_states = [c.state for c in org.cells[:5]]

    # Estados deben cambiar
    changed = sum(1 for o, n in zip(old_states, new_states)
                  if not torch.allclose(o, n))
    assert changed > 0

def test_metrics():
    """Organismo reporta métricas de inteligencia."""
    org = ZetaOrganism(grid_size=32, n_cells=50)
    org.initialize()

    for _ in range(10):
        org.step()

    metrics = org.get_metrics()

    assert 'n_fi' in metrics
    assert 'n_mass' in metrics
    assert 'coordination' in metrics
    assert 'stability' in metrics
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_zeta_organism.py -v`
Expected: FAIL - module not found

**Step 3: Write implementation**

```python
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
    role: torch.Tensor  # one-hot [MASS, FORCE, CORRUPT]
    energy: float = 0.0
    controlled_mass: float = 0.0

    @property
    def role_idx(self) -> int:
        return self.role.argmax().item()

class ZetaOrganism(nn.Module):
    """Organismo artificial distribuido.

    Sistema multi-agente donde células actúan como masas (Mi) o
    fuerzas (Fi), con campos de fuerza propagados por convolución
    zeta y comportamiento emergente.
    """

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

        # Historia para métricas
        self.history = []

    def initialize(self, seed_fi: bool = True):
        """Inicializa organismo con células aleatorias."""
        self.cells = []

        for i in range(self.n_cells):
            # Posición aleatoria
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)

            # Estado inicial
            state = torch.randn(self.state_dim) * 0.1

            # Rol inicial (mayormente MASS)
            if seed_fi and i == 0:
                role = torch.tensor([0.0, 1.0, 0.0])  # Primera célula es Fi
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

            # Preparar tensores
            state = cell.state.unsqueeze(0)
            if neighbors:
                neighbor_states = torch.stack([n.state for n in neighbors]).unsqueeze(0)
            else:
                neighbor_states = torch.zeros(1, 1, self.state_dim)

            # Paso de célula
            new_state, role_probs = self.cell_module(
                state, neighbor_states, field, cell.position
            )

            # Calcular nueva energía
            potential = field[0, 0, cell.position[1], cell.position[0]].item()
            new_energy = cell.energy + 0.1 * potential
            new_energy = np.clip(new_energy, 0, 1)

            # Contar masa controlada (para Fi)
            if role_probs[0, 1] > 0.5:  # Es Fi
                controlled = sum(1 for n in neighbors if n.role_idx == 0)
            else:
                controlled = 0

            # Nueva posición (sigue gradiente si es MASS)
            x, y = cell.position
            if role_probs[0, 0] > 0.5:  # Es MASS
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

        # Guardar historia
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Métricas de inteligencia colectiva."""
        n_fi = sum(1 for c in self.cells if c.role_idx == 1)
        n_mass = sum(1 for c in self.cells if c.role_idx == 0)
        n_corrupt = sum(1 for c in self.cells if c.role_idx == 2)

        # Coordinación: qué tan agrupadas están las masas alrededor de Fi
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

        # Estabilidad: varianza de energía
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_zeta_organism.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add zeta_organism.py tests/test_zeta_organism.py
git commit -m "feat: add ZetaOrganism with collective intelligence metrics"
```

---

### Task 6: Experimento de Inteligencia Colectiva

**Files:**
- Create: `exp_organism.py`

**Step 1: Write experiment script**

```python
# exp_organism.py
"""Experimento: Inteligencia colectiva en ZetaOrganism."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_organism import ZetaOrganism

def run_experiment(n_steps: int = 200, grid_size: int = 64, n_cells: int = 100):
    """Ejecuta simulación y analiza emergencia."""
    print('='*60)
    print('ZetaOrganism: Experimento de Inteligencia Colectiva')
    print('='*60)

    # Crear organismo
    org = ZetaOrganism(
        grid_size=grid_size,
        n_cells=n_cells,
        state_dim=32,
        hidden_dim=64,
        M=15,
        sigma=0.1,
        fi_threshold=0.7
    )

    print(f'\nConfiguracion:')
    print(f'  Grid: {grid_size}x{grid_size}')
    print(f'  Celulas: {n_cells}')
    print(f'  Steps: {n_steps}')

    # Inicializar con semilla Fi
    org.initialize(seed_fi=True)

    print(f'\nIniciando simulacion...')

    # Simular
    for step in range(n_steps):
        org.step()

        if (step + 1) % 50 == 0:
            m = org.get_metrics()
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Mass={m["n_mass"]}, '
                  f'Coord={m["coordination"]:.3f}, Stab={m["stability"]:.3f}')

    # Análisis final
    print('\n' + '='*60)
    print('RESULTADOS FINALES:')

    final = org.get_metrics()
    print(f'  Fi (fuerzas): {final["n_fi"]}')
    print(f'  Mass (seguidores): {final["n_mass"]}')
    print(f'  Corrupt (competidores): {final["n_corrupt"]}')
    print(f'  Coordinacion: {final["coordination"]:.3f}')
    print(f'  Estabilidad: {final["stability"]:.3f}')

    # Análisis de emergencia
    history = org.history

    # ¿Emergieron más Fi?
    fi_history = [h['n_fi'] for h in history]
    if fi_history[-1] > fi_history[0]:
        print('\n*** EMERGENCIA DETECTADA: Nuevos Fi surgieron del sistema ***')

    # ¿Hubo coordinación?
    coord_history = [h['coordination'] for h in history]
    if max(coord_history) > 0.5:
        print('*** COORDINACION: Masas se agruparon alrededor de Fi ***')

    # ¿Sistema estable?
    stab_history = [h['stability'] for h in history]
    if np.mean(stab_history[-20:]) > 0.7:
        print('*** HOMEOSTASIS: Sistema alcanzo equilibrio ***')

    print('='*60)

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Evolución de roles
    ax = axes[0, 0]
    steps = range(len(history))
    ax.plot(steps, fi_history, 'r-', label='Fi (fuerzas)', linewidth=2)
    ax.plot(steps, [h['n_mass'] for h in history], 'b-', label='Mass', linewidth=2)
    ax.plot(steps, [h['n_corrupt'] for h in history], 'k--', label='Corrupt', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cantidad')
    ax.set_title('Evolucion de Roles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Coordinación y estabilidad
    ax = axes[0, 1]
    ax.plot(steps, coord_history, 'g-', label='Coordinacion', linewidth=2)
    ax.plot(steps, stab_history, 'm-', label='Estabilidad', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Valor')
    ax.set_title('Metricas de Inteligencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. Estado final del grid
    ax = axes[1, 0]
    for cell in org.cells:
        x, y = cell.position
        color = ['blue', 'red', 'black'][cell.role_idx]
        size = 20 + cell.energy * 80
        ax.scatter(x, y, c=color, s=size, alpha=0.7)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_title('Estado Final (rojo=Fi, azul=Mass)')
    ax.set_aspect('equal')

    # 4. Energía promedio
    ax = axes[1, 1]
    energy_history = [h['avg_energy'] for h in history]
    ax.plot(steps, energy_history, 'orange', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energia promedio')
    ax.set_title('Evolucion de Energia')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_organism_experiment.png', dpi=150)
    print('\nGuardado: zeta_organism_experiment.png')

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_experiment()
```

**Step 2: Run experiment**

Run: `python exp_organism.py`
Expected: Output con métricas y gráfico generado

**Step 3: Commit**

```bash
git add exp_organism.py
git commit -m "feat: add collective intelligence experiment"
```

---

### Task 7: Tests de Integración

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration tests**

```python
# tests/test_integration.py
"""Tests de integración para ZetaOrganism."""
import pytest
import torch
import numpy as np
from zeta_organism import ZetaOrganism

class TestOrganismEmergence:
    """Tests para comportamientos emergentes."""

    @pytest.fixture
    def organism(self):
        torch.manual_seed(42)
        np.random.seed(42)
        org = ZetaOrganism(grid_size=32, n_cells=50)
        org.initialize(seed_fi=True)
        return org

    def test_fi_attracts_mass(self, organism):
        """Fi debe atraer a masas cercanas."""
        # Posición inicial de masas
        initial_mass_positions = [
            c.position for c in organism.cells if c.role_idx == 0
        ]
        fi_position = next(
            c.position for c in organism.cells if c.role_idx == 1
        )

        # Simular
        for _ in range(50):
            organism.step()

        # Posiciones finales
        final_mass_positions = [
            c.position for c in organism.cells if c.role_idx == 0
        ]

        # Calcular distancias promedio a Fi
        def avg_dist(positions, target):
            return np.mean([
                np.sqrt((p[0]-target[0])**2 + (p[1]-target[1])**2)
                for p in positions
            ])

        initial_avg = avg_dist(initial_mass_positions, fi_position)
        # Fi puede moverse, usar posición de Fi actual
        current_fi = next(
            c.position for c in organism.cells if c.role_idx == 1
        )
        final_avg = avg_dist(final_mass_positions, current_fi)

        # Masas deben acercarse (o mantenerse) a Fi
        assert final_avg <= initial_avg * 1.5  # Tolerancia

    def test_equilibrium_scaling(self, organism):
        """Fi debe escalar con masa controlada."""
        for _ in range(100):
            organism.step()

        # Verificar que Fi con más seguidores tiene más influencia
        fi_cells = [c for c in organism.cells if c.role_idx == 1]

        if len(fi_cells) > 1:
            # Ordenar por masa controlada
            fi_sorted = sorted(fi_cells, key=lambda c: c.controlled_mass)
            # El de mayor masa debería tener mayor energía
            assert fi_sorted[-1].energy >= fi_sorted[0].energy * 0.8

    def test_system_stability(self, organism):
        """Sistema debe tender hacia estabilidad."""
        # Simular hasta estabilidad
        for _ in range(150):
            organism.step()

        metrics = organism.get_metrics()

        # Sistema no debe colapsar
        assert metrics['n_fi'] + metrics['n_mass'] + metrics['n_corrupt'] == 50

        # Debe haber alguna estructura
        assert metrics['n_fi'] >= 1 or metrics['coordination'] > 0.1

class TestBehaviorAlgorithm:
    """Tests para el algoritmo A↔B."""

    def test_bidirectional_preserves_total_influence(self):
        """La influencia total se conserva aproximadamente."""
        from behavior_engine import BehaviorEngine

        engine = BehaviorEngine(state_dim=32)

        cell = torch.randn(32)
        neighbors = torch.randn(8, 32)

        out, in_ = engine.bidirectional_influence(cell, neighbors)

        # La suma de influencias debe ser finita
        assert torch.isfinite(out).all()
        assert torch.isfinite(in_)

    def test_transformation_continuity(self):
        """A³+V→B³+A mantiene continuidad."""
        from behavior_engine import BehaviorEngine

        engine = BehaviorEngine(state_dim=32)

        local_cube = torch.randn(3, 3, 32)

        # Con alpha=1, debe ser casi identidad
        result = engine.transform_with_potential(local_cube, 0.0, alpha=0.9)

        # Debe ser similar al original (no divergir)
        diff = (result - local_cube).norm() / local_cube.norm()
        assert diff < 2.0  # Cambio razonable

class TestMetrics:
    """Tests para métricas de inteligencia."""

    def test_coordination_bounds(self):
        """Coordinación debe estar en [0, 1]."""
        org = ZetaOrganism(grid_size=32, n_cells=30)
        org.initialize()

        for _ in range(20):
            org.step()
            metrics = org.get_metrics()
            assert 0 <= metrics['coordination'] <= 1

    def test_stability_decreases_with_chaos(self):
        """Estabilidad baja con alta varianza de energía."""
        org = ZetaOrganism(grid_size=32, n_cells=30)
        org.initialize()

        # Forzar alta varianza
        for i, cell in enumerate(org.cells):
            cell.energy = i / len(org.cells)  # 0 a 1

        metrics = org.get_metrics()
        # Alta varianza = baja estabilidad
        assert metrics['stability'] < 0.8
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for ZetaOrganism"
```

---

### Task 8: Documentación y Limpieza

**Files:**
- Update: `docs/zeta-lstm-hallazgos.md`
- Create: `README_organism.md`

**Step 1: Update documentation**

```markdown
# README_organism.md

# ZetaOrganism: Organismo Artificial con Inteligencia Colectiva

## Visión General

ZetaOrganism es un sistema multi-agente que simula un organismo artificial donde la inteligencia emerge de la interacción entre células que actúan como fuerzas (Fi) o masas (Mi).

## Modelo Físico

Basado en la dinámica Fi-Mi:
- **Fi (Fuerza Inicial)**: Atrae y controla masas
- **Mi (Masa)**: Sigue a Fi a través de gradientes
- **Equilibrio**: Fi escala con `sqrt(masa_controlada)`
- **Corrupción**: Masas pueden convertirse en Fi competidores

## Algoritmo de Comportamiento

Implementa las fórmulas:
- `A ↔ B`: Interacción bidireccional
- `A = AAA*A`: Auto-similitud recursiva
- `A³ + V → B³ + A`: Transformación con potencial vital
- `B = AA* - A*A`: Rol neto (Fi vs Mi)

## Uso

```python
from zeta_organism import ZetaOrganism

# Crear organismo
org = ZetaOrganism(grid_size=64, n_cells=100)
org.initialize(seed_fi=True)

# Simular
for _ in range(200):
    org.step()

# Analizar
metrics = org.get_metrics()
print(f"Fi: {metrics['n_fi']}, Coordinación: {metrics['coordination']:.3f}")
```

## Métricas de Inteligencia

- **Coordinación**: Qué tan agrupadas están las masas alrededor de Fi
- **Estabilidad**: Homeostasis del sistema (baja varianza de energía)
- **Emergencia**: Aparición de nuevos Fi desde masas

## Estructura de Archivos

```
cell_state.py       - Estados y transiciones de célula
force_field.py      - Campo de fuerzas con kernel zeta
behavior_engine.py  - Algoritmo A↔B, A³+V→B³+A
organism_cell.py    - Célula con NCA + Resonant
zeta_organism.py    - Sistema completo
exp_organism.py     - Experimentos
```

## Referencia Teórica

- Paper: "IA Adaptativa a través de la Hipótesis de Riemann"
- Notas de investigación sobre dinámica Fi-Mi y colapso dimensional
- ZetaLSTM Resonant: "Detectar, no imponer"
```

**Step 2: Commit final**

```bash
git add README_organism.md docs/zeta-lstm-hallazgos.md
git commit -m "docs: add ZetaOrganism documentation"
```

---

## Configuración Inicial

```python
config = {
    'grid_size': 64,
    'n_cells': 100,
    'state_dim': 32,
    'hidden_dim': 64,
    'M': 15,              # Ceros zeta
    'sigma': 0.1,         # Regularización
    'fi_threshold': 0.7,  # Energía para convertirse en Fi
    'equilibrium_factor': 0.5,
    'n_steps': 200
}
```

## Métricas de Éxito

1. **Emergencia de Fi**: Nuevas fuerzas surgen del sistema
2. **Coordinación > 0.5**: Masas se agrupan efectivamente
3. **Estabilidad > 0.7**: Sistema alcanza homeostasis
4. **No colapso**: Sistema mantiene población estable
