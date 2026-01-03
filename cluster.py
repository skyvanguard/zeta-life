# -*- coding: utf-8 -*-
"""
Cluster: Grupo de células conscientes con psique emergente.

Un cluster es un grupo de ConsciousCells que comparten proximidad
espacial y/o afinidad arquetipal. Cada cluster desarrolla una
psique emergente (ClusterPsyche) que representa el "mood" colectivo.

Fecha: 2026-01-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

# Importar del sistema existente
from zeta_psyche import Archetype
from cell_state import CellRole
from micro_psyche import ConsciousCell, MicroPsyche, compute_local_phi, unbiased_argmax


# =============================================================================
# CLUSTER PSYCHE
# =============================================================================

@dataclass
class ClusterPsyche:
    """
    Psique emergente de un cluster de células.

    Representa el estado colectivo del cluster, que emerge de
    la agregación de las micro-psiques individuales.

    Attributes:
        aggregate_state: Estado arquetipal agregado [4]
        specialization: Arquetipo dominante del cluster
        phi_cluster: Integración intra-cluster (coherencia)
        coherence: Acuerdo entre células del cluster
        prediction_error: Error de predicción promedio
        integration_level: Nivel de integración general
    """

    aggregate_state: torch.Tensor  # [4] arquetipos
    specialization: Archetype
    phi_cluster: float
    coherence: float
    prediction_error: float
    integration_level: float

    def __post_init__(self):
        """Normalizar estado agregado."""
        if self.aggregate_state.sum() > 0:
            self.aggregate_state = F.softmax(self.aggregate_state, dim=0)

    @property
    def is_specialized(self) -> bool:
        """¿El cluster tiene especialización clara?"""
        return self.aggregate_state.max().item() > 0.4

    @property
    def balance(self) -> float:
        """Qué tan balanceado está el cluster (0=dominante único, 1=equilibrio)."""
        return self.aggregate_state.min().item() * 4  # Normalizado a [0, 1]

    def alignment_with(self, other_state: torch.Tensor) -> float:
        """Calcula alineación con otro estado."""
        sim = F.cosine_similarity(
            self.aggregate_state.unsqueeze(0),
            other_state.unsqueeze(0)
        ).item()
        return (sim + 1) / 2

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            'aggregate_state': self.aggregate_state.tolist(),
            'specialization': self.specialization.name,
            'phi_cluster': self.phi_cluster,
            'coherence': self.coherence,
            'prediction_error': self.prediction_error,
            'integration_level': self.integration_level,
            'is_specialized': self.is_specialized,
            'balance': self.balance
        }

    @classmethod
    def create_empty(cls) -> 'ClusterPsyche':
        """Crea una psique de cluster vacía."""
        # Seleccionar arquetipo aleatorio para evitar sesgo hacia PERSONA
        random_archetype = Archetype(np.random.randint(4))
        return cls(
            aggregate_state=torch.ones(4) / 4,
            specialization=random_archetype,
            phi_cluster=0.0,
            coherence=0.0,
            prediction_error=0.0,
            integration_level=0.0
        )

    @classmethod
    def from_cells(cls, cells: List[ConsciousCell]) -> 'ClusterPsyche':
        """
        Crea ClusterPsyche agregando estados de células.

        Args:
            cells: Lista de ConsciousCell del cluster

        Returns:
            ClusterPsyche con estado agregado
        """
        if not cells:
            return cls.create_empty()

        # Obtener estados de todas las células
        states = torch.stack([c.psyche.archetype_state for c in cells])

        # Calcular sorpresas individuales
        surprises = [c.psyche.compute_surprise() for c in cells]

        # Pesos basados en energía, phi_local, y plasticidad (de sorpresa)
        # Células sorprendidas contribuyen más al agregado porque
        # detectan información nueva relevante para el cluster
        weights = torch.tensor([
            c.energy * c.psyche.phi_local * c.psyche.get_plasticity()
            for c in cells
        ])
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(len(cells)) / len(cells)

        # Agregación ponderada
        aggregate = (weights.unsqueeze(1) * states).sum(dim=0)
        aggregate = F.softmax(aggregate, dim=0)

        # Especialización = arquetipo dominante
        specialization = Archetype(unbiased_argmax(aggregate))

        # Φ cluster = 1 - varianza normalizada
        variance = states.var(dim=0).mean().item()
        phi_cluster = 1.0 - min(1.0, variance * 2)

        # Coherencia = similitud promedio entre células
        coherence = 1.0 - variance

        # Error de predicción (basado en sorpresa ya calculada arriba)
        prediction_error = np.mean(surprises) if surprises else 0.0

        # Nivel de integración
        integration_level = phi_cluster * (1 - prediction_error)

        return cls(
            aggregate_state=aggregate,
            specialization=specialization,
            phi_cluster=phi_cluster,
            coherence=coherence,
            prediction_error=prediction_error,
            integration_level=integration_level
        )


# =============================================================================
# CLUSTER
# =============================================================================

@dataclass
class Cluster:
    """
    Un cluster de células conscientes.

    Agrupa células por proximidad espacial y afinidad arquetipal.
    Cada cluster desarrolla comportamiento colectivo emergente.

    Attributes:
        id: Identificador único del cluster
        cells: Lista de células del cluster
        psyche: Psique emergente del cluster
        centroid: Centro espacial del cluster
        neighbors: IDs de clusters vecinos
        collective_role: Rol dominante del cluster
    """

    id: int
    cells: List[ConsciousCell] = field(default_factory=list)
    psyche: Optional[ClusterPsyche] = None
    centroid: Tuple[float, float] = (0.0, 0.0)
    neighbors: List[int] = field(default_factory=list)
    collective_role: CellRole = CellRole.MASS

    def __post_init__(self):
        """Inicializar psique y calcular centroide."""
        if self.cells:
            self._update_centroid()
            if self.psyche is None:
                self.psyche = ClusterPsyche.from_cells(self.cells)
            self._update_collective_role()

    @property
    def size(self) -> int:
        """Número de células en el cluster."""
        return len(self.cells)

    @property
    def is_empty(self) -> bool:
        """¿Está vacío el cluster?"""
        return len(self.cells) == 0

    @property
    def avg_energy(self) -> float:
        """Energía promedio de células."""
        if not self.cells:
            return 0.0
        return np.mean([c.energy for c in self.cells])

    @property
    def dominant_archetype(self) -> Archetype:
        """Arquetipo dominante del cluster."""
        if self.psyche:
            return self.psyche.specialization
        # Fallback aleatorio para evitar sesgo hacia PERSONA
        return Archetype(np.random.randint(4))

    def _update_centroid(self):
        """Actualiza el centroide espacial."""
        if not self.cells:
            return

        positions = torch.tensor(
            [c.position for c in self.cells],
            dtype=torch.float32
        )
        center = positions.mean(dim=0)
        self.centroid = (center[0].item(), center[1].item())

    def _update_collective_role(self):
        """Determina el rol colectivo del cluster."""
        if not self.cells:
            self.collective_role = CellRole.MASS
            return

        role_counts = [0, 0, 0]  # MASS, FORCE, CORRUPT
        for cell in self.cells:
            role_idx = cell.role.argmax().item()
            role_counts[role_idx] += 1

        self.collective_role = CellRole(np.argmax(role_counts))

    def add_cell(self, cell: ConsciousCell):
        """Agrega una célula al cluster."""
        cell.cluster_id = self.id
        self.cells.append(cell)
        self._update_centroid()

    def remove_cell(self, cell: ConsciousCell):
        """Remueve una célula del cluster."""
        if cell in self.cells:
            self.cells.remove(cell)
            cell.cluster_id = -1
            self._update_centroid()

    def update_psyche(self):
        """Actualiza la psique del cluster desde sus células."""
        self.psyche = ClusterPsyche.from_cells(self.cells)
        self._update_collective_role()

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Distancia del centroide a un punto."""
        dx = self.centroid[0] - point[0]
        dy = self.centroid[1] - point[1]
        return np.sqrt(dx**2 + dy**2)

    def distance_to_cluster(self, other: 'Cluster') -> float:
        """Distancia entre centroides de clusters."""
        return self.distance_to_point(other.centroid)

    def get_fi_cells(self) -> List[ConsciousCell]:
        """Retorna células con rol Fi (líderes)."""
        return [c for c in self.cells if c.is_fi]

    def get_mass_cells(self) -> List[ConsciousCell]:
        """Retorna células con rol Mass."""
        return [c for c in self.cells if c.is_mass]

    def broadcast_influence(self, influence: torch.Tensor, strength: float = 0.1):
        """
        Aplica influencia arquetipal a todas las células del cluster.

        Args:
            influence: Tensor [4] de influencia
            strength: Fuerza de la influencia
        """
        for cell in self.cells:
            cell.apply_archetype_influence(influence, strength)

    def compute_internal_coherence(self) -> float:
        """
        Calcula coherencia interna (similitud entre células).

        Returns:
            Coherencia promedio [0, 1]
        """
        if len(self.cells) < 2:
            return 1.0

        similarities = []
        for i, cell_a in enumerate(self.cells):
            for cell_b in self.cells[i+1:]:
                sim = cell_a.psyche_similarity(cell_b)
                similarities.append(sim)

        return np.mean(similarities) if similarities else 1.0

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            'id': self.id,
            'size': self.size,
            'centroid': self.centroid,
            'collective_role': self.collective_role.name,
            'neighbors': self.neighbors,
            'psyche': self.psyche.to_dict() if self.psyche else None,
            'avg_energy': self.avg_energy
        }

    @classmethod
    def create_from_cells(
        cls,
        cluster_id: int,
        cells: List[ConsciousCell]
    ) -> 'Cluster':
        """
        Crea un cluster desde una lista de células.

        Args:
            cluster_id: ID del cluster
            cells: Lista de células
        """
        cluster = cls(id=cluster_id, cells=cells)

        # Asignar cluster_id a cada célula
        for cell in cells:
            cell.cluster_id = cluster_id

        return cluster


# =============================================================================
# UTILIDADES DE CLUSTERING
# =============================================================================

def find_cluster_neighbors(
    clusters: List[Cluster],
    threshold_ratio: float = 1.5
) -> None:
    """
    Detecta clusters vecinos por proximidad espacial.

    Modifica clusters in-place, actualizando sus listas de neighbors.

    Args:
        clusters: Lista de clusters
        threshold_ratio: Multiplicador para umbral de distancia
    """
    if len(clusters) < 2:
        return

    # Calcular distancia promedio entre clusters
    distances = []
    for i, cluster_a in enumerate(clusters):
        for cluster_b in clusters[i+1:]:
            dist = cluster_a.distance_to_cluster(cluster_b)
            distances.append(dist)

    if not distances:
        return

    avg_distance = np.mean(distances)
    threshold = avg_distance * threshold_ratio

    # Asignar vecinos
    for cluster in clusters:
        cluster.neighbors = []
        for other in clusters:
            if cluster.id != other.id:
                dist = cluster.distance_to_cluster(other)
                if dist < threshold:
                    cluster.neighbors.append(other.id)


def compute_inter_cluster_coherence(clusters: List[Cluster]) -> float:
    """
    Calcula coherencia entre clusters (diversidad de especializaciones).

    Alta coherencia = cada cluster tiene especialización diferente.

    Args:
        clusters: Lista de clusters

    Returns:
        Coherencia inter-cluster [0, 1]
    """
    if len(clusters) < 2:
        return 1.0

    # Contar especializaciones únicas
    specializations = [c.psyche.specialization for c in clusters if c.psyche]
    unique_specs = len(set(specializations))

    # Bonus por tener los 4 arquetipos representados
    diversity_score = unique_specs / 4.0

    # Promedio de phi de clusters
    avg_phi = np.mean([c.psyche.phi_cluster for c in clusters if c.psyche])

    return diversity_score * 0.5 + avg_phi * 0.5


def merge_clusters(cluster_a: Cluster, cluster_b: Cluster) -> Cluster:
    """
    Fusiona dos clusters en uno.

    Args:
        cluster_a: Primer cluster
        cluster_b: Segundo cluster

    Returns:
        Nuevo cluster fusionado
    """
    merged_cells = cluster_a.cells + cluster_b.cells
    return Cluster.create_from_cells(
        cluster_id=cluster_a.id,  # Mantiene ID del primero
        cells=merged_cells
    )


def split_cluster(
    cluster: Cluster,
    n_parts: int = 2
) -> List[Cluster]:
    """
    Divide un cluster en partes basándose en posición espacial.

    Args:
        cluster: Cluster a dividir
        n_parts: Número de partes

    Returns:
        Lista de nuevos clusters
    """
    if cluster.size < n_parts:
        return [cluster]

    # Ordenar células por posición X
    sorted_cells = sorted(cluster.cells, key=lambda c: c.position[0])

    # Dividir equitativamente
    cells_per_part = len(sorted_cells) // n_parts
    new_clusters = []

    for i in range(n_parts):
        start_idx = i * cells_per_part
        if i == n_parts - 1:
            # Último grupo toma el resto
            part_cells = sorted_cells[start_idx:]
        else:
            part_cells = sorted_cells[start_idx:start_idx + cells_per_part]

        new_cluster = Cluster.create_from_cells(
            cluster_id=cluster.id * 10 + i,  # Nuevo ID derivado
            cells=part_cells
        )
        new_clusters.append(new_cluster)

    return new_clusters


# =============================================================================
# TESTS BÁSICOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: Cluster y ClusterPsyche")
    print("=" * 60)

    # Crear células de prueba
    print("\n1. Crear 10 células aleatorias:")
    cells = [ConsciousCell.create_random(grid_size=64) for _ in range(10)]
    for i, c in enumerate(cells):
        print(f"   Cell {i}: pos={c.position}, arch={c.psyche.dominant.name}")

    # Crear cluster
    print("\n2. Crear cluster desde células:")
    cluster = Cluster.create_from_cells(cluster_id=0, cells=cells)
    print(f"   ID: {cluster.id}")
    print(f"   Tamaño: {cluster.size}")
    print(f"   Centroide: ({cluster.centroid[0]:.1f}, {cluster.centroid[1]:.1f})")
    print(f"   Rol colectivo: {cluster.collective_role.name}")

    # ClusterPsyche
    print("\n3. Psique del cluster:")
    print(f"   Especialización: {cluster.psyche.specialization.name}")
    print(f"   Estado: {cluster.psyche.aggregate_state.tolist()}")
    print(f"   Φ cluster: {cluster.psyche.phi_cluster:.3f}")
    print(f"   Coherencia: {cluster.psyche.coherence:.3f}")

    # Crear segundo cluster
    print("\n4. Crear segundo cluster:")
    cells2 = [ConsciousCell.create_random(grid_size=64) for _ in range(8)]
    cluster2 = Cluster.create_from_cells(cluster_id=1, cells=cells2)
    print(f"   Cluster 1 centroid: {cluster2.centroid}")
    print(f"   Especialización: {cluster2.psyche.specialization.name}")

    # Distancia entre clusters
    dist = cluster.distance_to_cluster(cluster2)
    print(f"\n5. Distancia entre clusters: {dist:.2f}")

    # Encontrar vecinos
    print("\n6. Detectar vecinos:")
    clusters = [cluster, cluster2]
    find_cluster_neighbors(clusters)
    print(f"   Cluster 0 vecinos: {cluster.neighbors}")
    print(f"   Cluster 1 vecinos: {cluster2.neighbors}")

    # Coherencia inter-cluster
    print("\n7. Coherencia inter-cluster:")
    coherence = compute_inter_cluster_coherence(clusters)
    print(f"   Coherencia: {coherence:.3f}")

    # Broadcast influence
    print("\n8. Broadcast influence al cluster 0:")
    print(f"   Antes - Cell 0 arch: {cluster.cells[0].psyche.dominant.name}")
    influence = torch.tensor([0.7, 0.1, 0.1, 0.1])  # PERSONA dominante
    cluster.broadcast_influence(influence, strength=0.5)
    print(f"   Después - Cell 0 arch: {cluster.cells[0].psyche.dominant.name}")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETADOS")
    print("=" * 60)
