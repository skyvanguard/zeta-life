# -*- coding: utf-8 -*-
"""
MicroPsyche: Psique simplificada para células individuales.

Cada célula del ZetaOrganism tiene una micro-psique que representa
su estado arquetipal interno. La consciencia del organismo emerge
de la interacción de estas micro-psiques.

Fecha: 2026-01-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple
from collections import deque
from enum import Enum

# Importar del sistema existente
from zeta_psyche import Archetype
from cell_state import CellRole


# =============================================================================
# UTILIDADES
# =============================================================================

def unbiased_argmax(tensor: torch.Tensor, tolerance: float = 0.01) -> int:
    """
    Argmax sin sesgo hacia índices bajos en caso de empate.

    Cuando hay valores casi iguales (dentro de tolerance del máximo),
    selecciona aleatoriamente entre ellos en lugar de siempre
    retornar el índice más bajo.

    Args:
        tensor: Tensor 1D de valores
        tolerance: Margen para considerar valores como "empatados"

    Returns:
        Índice del máximo (aleatorio si hay empate)
    """
    max_val = tensor.max().item()
    # Encontrar todos los índices con valores cercanos al máximo
    candidates = (tensor >= max_val - tolerance).nonzero(as_tuple=True)[0]

    if len(candidates) == 1:
        return candidates[0].item()
    else:
        # Selección aleatoria entre candidatos empatados
        return candidates[np.random.randint(len(candidates))].item()


# =============================================================================
# MICRO-PSIQUE
# =============================================================================

@dataclass
class MicroPsyche:
    """
    Psique simplificada para una célula individual.

    A diferencia de ZetaPsyche completa (que tiene múltiples células internas),
    MicroPsyche es un estado único que representa la "personalidad" de la célula.

    Attributes:
        archetype_state: Tensor[4] con pesos normalizados para cada arquetipo
        dominant: El arquetipo dominante actual
        emotional_energy: Intensidad emocional (afecta interacciones)
        recent_states: Historial para predicción local
        phi_local: Integración con vecinos (mini-Φ)
    """

    archetype_state: torch.Tensor  # [PERSONA, SOMBRA, ANIMA, ANIMUS]
    dominant: Archetype
    emotional_energy: float  # 0-1
    recent_states: Deque[torch.Tensor] = field(default_factory=lambda: deque(maxlen=5))
    phi_local: float = 0.5
    accumulated_surprise: float = 0.0  # Sorpresa acumulada para plasticidad

    def __post_init__(self):
        """Asegurar que archetype_state está normalizado."""
        # Solo normalizar si no es distribución de probabilidad válida
        # (evitar doble softmax que uniformiza estados)
        state_sum = self.archetype_state.sum().item()
        if abs(state_sum - 1.0) > 0.01:  # No es distribución válida
            self.archetype_state = F.softmax(self.archetype_state, dim=0)
        if len(self.recent_states) == 0:
            self.recent_states.append(self.archetype_state.clone())

    def update_state(
        self,
        new_state: torch.Tensor,
        blend_factor: float = 0.1,
        noise_scale: float = 0.02
    ):
        """
        Actualiza el estado arquetipal con mezcla suave y ruido estocástico.

        El ruido preserva diversidad y evita sincronización total.

        Args:
            new_state: Nuevo estado propuesto [4]
            blend_factor: Cuánto del nuevo estado incorporar (0-1)
            noise_scale: Escala del ruido estocástico (default 0.02)
        """
        # Mezcla suave
        blended = (1 - blend_factor) * self.archetype_state + blend_factor * new_state

        # Añadir ruido estocástico para preservar diversidad
        # Esto evita la convergencia total a estados idénticos
        if noise_scale > 0:
            noise = torch.randn(4) * noise_scale
            blended = blended + noise

        self.archetype_state = F.softmax(blended, dim=0)

        # Actualizar dominante
        self.dominant = Archetype(unbiased_argmax(self.archetype_state))

        # Guardar en historial
        self.recent_states.append(self.archetype_state.clone())

    def compute_surprise(self) -> float:
        """
        Calcula sorpresa como diferencia con estado anterior.

        Returns:
            Sorpresa normalizada [0, 1]
        """
        if len(self.recent_states) < 2:
            return 0.0

        prev = self.recent_states[-2]
        curr = self.recent_states[-1]

        diff = (curr - prev).abs().sum().item()
        return min(1.0, diff)

    def update_accumulated_surprise(self, decay: float = 0.9) -> None:
        """
        Actualiza la sorpresa acumulada con decaimiento exponencial.

        Mantiene un promedio móvil de la sorpresa reciente, que se usa
        para determinar la plasticidad de la célula.

        Args:
            decay: Factor de decaimiento (0-1). Valores altos = memoria más larga.
        """
        current_surprise = self.compute_surprise()
        self.accumulated_surprise = decay * self.accumulated_surprise + (1 - decay) * current_surprise

    def get_plasticity(self) -> float:
        """
        Calcula plasticidad basada en sorpresa acumulada.

        Células con alta sorpresa son más plásticas (más abiertas al cambio).
        Esto implementa el principio de "free energy minimization":
        sistemas sorprendidos buscan activamente nuevos modelos.

        Returns:
            Factor de plasticidad [0.5, 1.5]
            - 0.5: Muy estable (baja sorpresa)
            - 1.0: Normal
            - 1.5: Muy plástico (alta sorpresa)
        """
        # Escala no-lineal: baja sorpresa → menos plástico, alta → más
        # Usamos función sigmoide suavizada centrada en sorpresa=0.3
        surprise_centered = self.accumulated_surprise - 0.3
        plasticity = 1.0 + 0.5 * np.tanh(surprise_centered * 3)
        return float(plasticity)

    def alignment_with(self, other_state: torch.Tensor) -> float:
        """
        Calcula alineación con otro estado arquetipal.

        Args:
            other_state: Estado a comparar [4]

        Returns:
            Similitud coseno normalizada a [0, 1]
        """
        sim = F.cosine_similarity(
            self.archetype_state.unsqueeze(0),
            other_state.unsqueeze(0)
        ).item()
        return (sim + 1) / 2  # Normalizar de [-1,1] a [0,1]

    def get_complementary_archetype(self) -> Archetype:
        """
        Retorna el arquetipo complementario (opuesto en el tetraedro).

        PERSONA <-> SOMBRA
        ANIMA <-> ANIMUS
        """
        complements = {
            Archetype.PERSONA: Archetype.SOMBRA,
            Archetype.SOMBRA: Archetype.PERSONA,
            Archetype.ANIMA: Archetype.ANIMUS,
            Archetype.ANIMUS: Archetype.ANIMA,
        }
        return complements[self.dominant]

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            'archetype_state': self.archetype_state.tolist(),
            'dominant': self.dominant.name,
            'emotional_energy': self.emotional_energy,
            'phi_local': self.phi_local,
            'surprise': self.compute_surprise()
        }

    @classmethod
    def create_random(cls, bias: Optional[Archetype] = None) -> 'MicroPsyche':
        """
        Crea una MicroPsyche con estado aleatorio.

        Args:
            bias: Arquetipo hacia el cual sesgar (opcional)
        """
        if bias is not None:
            # Estado sesgado hacia un arquetipo CON VARIACIÓN ALEATORIA
            # Base: sesgo fuerte hacia el arquetipo dominante
            # Variación: ruido aleatorio para diversidad
            base_dominant = 2.0 + np.random.uniform(-0.3, 0.3)  # 1.7 a 2.3
            base_others = 0.1 + np.random.uniform(-0.05, 0.15)  # 0.05 a 0.25

            state = torch.ones(4) * base_others
            # Añadir variación individual a cada componente
            state = state + torch.rand(4) * 0.2  # Variación 0-0.2
            state[bias.value] = base_dominant
        else:
            # Estado completamente aleatorio
            state = torch.rand(4)

        state = F.softmax(state, dim=0)

        return cls(
            archetype_state=state,
            dominant=Archetype(unbiased_argmax(state)),
            emotional_energy=np.random.uniform(0.3, 0.7),
            recent_states=deque([state.clone()], maxlen=5),
            phi_local=0.5
        )


# =============================================================================
# CONSCIOUS CELL
# =============================================================================

@dataclass
class ConsciousCell:
    """
    Célula consciente que combina física (CellEntity) con psique (MicroPsyche).

    Extiende el concepto de CellEntity agregando una micro-psique interna.
    La célula ahora tiene "personalidad" que afecta su comportamiento físico.

    Attributes:
        position: (x, y) en el grid
        state: Estado interno de la red neural [state_dim]
        role: Probabilidades de rol [MASS, FORCE, CORRUPT]
        energy: Nivel de energía (0-1)
        controlled_mass: Masa controlada (para Fi)
        psyche: Micro-psique de la célula
        cluster_id: ID del cluster al que pertenece
        cluster_weight: Peso/importancia en su cluster
    """

    # Atributos físicos (de CellEntity original)
    position: Tuple[int, int]
    state: torch.Tensor  # [state_dim]
    role: torch.Tensor   # [3] probabilidades MASS, FORCE, CORRUPT
    energy: float = 0.5
    controlled_mass: float = 0.0

    # Atributos de consciencia (nuevos)
    psyche: MicroPsyche = None
    cluster_id: int = -1
    cluster_weight: float = 1.0

    def __post_init__(self):
        """Inicializar psique si no se proporcionó."""
        if self.psyche is None:
            self.psyche = MicroPsyche.create_random()

    @property
    def role_idx(self) -> int:
        """Índice del rol dominante."""
        return self.role.argmax().item()

    @property
    def role_name(self) -> str:
        """Nombre del rol dominante."""
        return ['MASS', 'FORCE', 'CORRUPT'][self.role_idx]

    @property
    def is_fi(self) -> bool:
        """¿Es esta célula un Fi (líder)?"""
        return self.role_idx == 1

    @property
    def is_mass(self) -> bool:
        """¿Es esta célula Mass (seguidor)?"""
        return self.role_idx == 0

    @property
    def is_corrupt(self) -> bool:
        """¿Es esta célula Corrupt?"""
        return self.role_idx == 2

    def distance_to(self, other: 'ConsciousCell') -> float:
        """Calcula distancia euclidiana a otra célula."""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return np.sqrt(dx**2 + dy**2)

    def psyche_similarity(self, other: 'ConsciousCell') -> float:
        """Calcula similitud psíquica con otra célula."""
        return self.psyche.alignment_with(other.psyche.archetype_state)

    def update_energy(self, delta: float, min_val: float = 0.0, max_val: float = 1.0):
        """Actualiza energía con límites."""
        self.energy = max(min_val, min(max_val, self.energy + delta))

    def apply_archetype_influence(self, influence: torch.Tensor, strength: float = 0.1):
        """
        Aplica influencia arquetipal externa.

        Args:
            influence: Tensor [4] de influencia arquetipal
            strength: Fuerza de la influencia
        """
        self.psyche.update_state(influence, blend_factor=strength)

    def get_movement_bias(self) -> Tuple[float, float]:
        """
        Calcula sesgo de movimiento basado en arquetipo dominante.

        Returns:
            (dx_bias, dy_bias) multiplicadores para movimiento
        """
        dominant = self.psyche.dominant

        if dominant == Archetype.PERSONA:
            # Persona: sigue al grupo (mayor respuesta al campo)
            return (1.3, 1.3)
        elif dominant == Archetype.SOMBRA:
            # Sombra: errático, puede ir contra el campo
            if np.random.random() < 0.3:
                return (-1.0, -1.0)
            return (1.0, 1.0)
        elif dominant == Archetype.ANIMA:
            # Anima: busca cercanía (atracción extra)
            return (1.1, 1.1)
        elif dominant == Archetype.ANIMUS:
            # Animus: explorador (movimiento ocasional aleatorio)
            if np.random.random() < 0.2:
                return (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            return (1.0, 1.0)

        return (1.0, 1.0)

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            'position': self.position,
            'role': self.role_name,
            'energy': self.energy,
            'cluster_id': self.cluster_id,
            'psyche': self.psyche.to_dict()
        }

    @classmethod
    def create_random(
        cls,
        grid_size: int,
        state_dim: int = 32,
        archetype_bias: Optional[Archetype] = None
    ) -> 'ConsciousCell':
        """
        Crea una célula consciente con posición y estado aleatorios.

        Args:
            grid_size: Tamaño del grid para posición
            state_dim: Dimensión del estado interno
            archetype_bias: Arquetipo hacia el cual sesgar
        """
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)

        return cls(
            position=(x, y),
            state=torch.randn(state_dim) * 0.1,
            role=torch.tensor([1.0, 0.0, 0.0]),  # MASS por defecto
            energy=np.random.uniform(0.3, 0.7),
            controlled_mass=0.0,
            psyche=MicroPsyche.create_random(bias=archetype_bias),
            cluster_id=-1,
            cluster_weight=1.0
        )


# =============================================================================
# UTILIDADES
# =============================================================================

def compute_local_phi(cell: ConsciousCell, neighbors: list) -> float:
    """
    Calcula Φ local (integración con vecinos).

    Args:
        cell: Célula central
        neighbors: Lista de ConsciousCell vecinas

    Returns:
        phi_local entre 0 y 1
    """
    if not neighbors:
        return 0.5  # Sin vecinos, integración neutral

    # Obtener estados de vecinos
    neighbor_states = torch.stack([n.psyche.archetype_state for n in neighbors])

    # Varianza entre vecinos
    variance = neighbor_states.var(dim=0).mean().item()

    # Similitud de la célula con promedio de vecinos
    avg_neighbor = neighbor_states.mean(dim=0)
    similarity = cell.psyche.alignment_with(avg_neighbor)

    # Φ = alta similitud + baja varianza
    phi = similarity * (1.0 - min(1.0, variance * 2))

    return max(0.0, min(1.0, phi))


def apply_psyche_contagion(
    cell: ConsciousCell,
    neighbors: list,
    contagion_rate: float = 0.1,
    similarity_threshold: float = 0.85,
    friction_factor: float = 0.2
):
    """
    Aplica contagio psíquico de vecinos con fricción adaptativa.

    Las células cercanas tienden a alinearse arquetipalmente,
    pero la fricción evita sincronización total.

    Args:
        cell: Célula a actualizar
        neighbors: Lista de vecinos
        contagion_rate: Velocidad de contagio base
        similarity_threshold: Umbral de similitud para aplicar fricción
        friction_factor: Factor de reducción cuando similitud > threshold
    """
    if not neighbors:
        return

    # Promedio ponderado por cercanía
    weights = []
    states = []

    for neighbor in neighbors:
        dist = cell.distance_to(neighbor)
        weight = 1.0 / (1.0 + dist)  # Más cerca = más influencia
        weights.append(weight)
        states.append(neighbor.psyche.archetype_state)

    weights = torch.tensor(weights)
    weights = weights / weights.sum()

    states = torch.stack(states)
    weighted_avg = (weights.unsqueeze(1) * states).sum(dim=0)

    # Calcular similitud actual con el promedio de vecinos
    current_similarity = cell.psyche.alignment_with(weighted_avg)

    # Aplicar fricción si la similitud ya es alta
    # Esto evita la convergencia total
    effective_rate = contagion_rate
    if current_similarity > similarity_threshold:
        # Reducir drásticamente el contagio cuando ya son muy similares
        # Cuanto más similar, menos contagio
        excess = (current_similarity - similarity_threshold) / (1.0 - similarity_threshold)
        effective_rate = contagion_rate * (friction_factor + (1 - friction_factor) * (1 - excess))

    # Aplicar influencia con tasa efectiva
    cell.psyche.update_state(weighted_avg, blend_factor=effective_rate)


# =============================================================================
# TESTS BÁSICOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: MicroPsyche y ConsciousCell")
    print("=" * 60)

    # Test MicroPsyche
    print("\n1. Crear MicroPsyche aleatoria:")
    psyche = MicroPsyche.create_random()
    print(f"   Estado: {psyche.archetype_state.tolist()}")
    print(f"   Dominante: {psyche.dominant.name}")
    print(f"   Energía emocional: {psyche.emotional_energy:.2f}")

    # Test MicroPsyche con bias
    print("\n2. Crear MicroPsyche con bias SOMBRA:")
    psyche_shadow = MicroPsyche.create_random(bias=Archetype.SOMBRA)
    print(f"   Estado: {psyche_shadow.archetype_state.tolist()}")
    print(f"   Dominante: {psyche_shadow.dominant.name}")

    # Test actualización
    print("\n3. Actualizar estado:")
    old_state = psyche.archetype_state.clone()
    new_influence = torch.tensor([0.1, 0.7, 0.1, 0.1])
    psyche.update_state(new_influence, blend_factor=0.3)
    print(f"   Antes: {old_state.tolist()}")
    print(f"   Después: {psyche.archetype_state.tolist()}")
    print(f"   Nuevo dominante: {psyche.dominant.name}")

    # Test ConsciousCell
    print("\n4. Crear ConsciousCell:")
    cell = ConsciousCell.create_random(grid_size=64, state_dim=32)
    print(f"   Posición: {cell.position}")
    print(f"   Rol: {cell.role_name}")
    print(f"   Arquetipo: {cell.psyche.dominant.name}")

    # Test múltiples células y contagio
    print("\n5. Crear 5 células y probar contagio:")
    cells = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]

    print("   Antes del contagio:")
    for i, c in enumerate(cells):
        print(f"     Cell {i}: {c.psyche.dominant.name}")

    # Aplicar contagio a la primera célula
    apply_psyche_contagion(cells[0], cells[1:], contagion_rate=0.3)

    print("   Después del contagio (cell 0):")
    print(f"     Cell 0: {cells[0].psyche.dominant.name}")

    # Test Φ local
    print("\n6. Calcular Φ local:")
    phi = compute_local_phi(cells[0], cells[1:])
    print(f"   Φ local de cell 0: {phi:.3f}")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETADOS")
    print("=" * 60)
