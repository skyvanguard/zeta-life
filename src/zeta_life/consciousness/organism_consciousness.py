"""
OrganismConsciousness: Estado de consciencia del organismo completo.

Representa el nivel más alto de consciencia en el sistema jerárquico.
Emerge de la integración de todos los clusters.

Fecha: 2026-01-03
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.tetrahedral_space import get_tetrahedral_space

# Importar del sistema existente
from ..core.vertex import Vertex
from ..psyche.zeta_conscious_self import ConsciousnessIndex
from ..psyche.zeta_individuation import IndividuationStage, IntegrationMetrics

# Backwards compatibility alias
Archetype = Vertex

# Importar de módulos nuevos
from .cluster import Cluster, ClusterPsyche
from .micro_psyche import ConsciousCell, MicroPsyche, unbiased_argmax

# =============================================================================
# ORGANISM CONSCIOUSNESS
# =============================================================================

@dataclass
class OrganismConsciousness:
    """
    Estado de consciencia del organismo completo.

    Integra la información de todos los clusters para formar
    una consciencia unificada. El Self emerge aquí.

    Attributes:
        consciousness_index: Índice compuesto de consciencia
        phi_global: Φ global (integración entre clusters)
        global_archetype: Estado arquetipal agregado [4]
        dominant_archetype: Arquetipo dominante del organismo
        individuation_stage: Etapa de individuación actual
        self_model: Representación del "yo" del organismo
        vertical_coherence: Coherencia entre niveles
    """

    consciousness_index: ConsciousnessIndex
    phi_global: float
    global_archetype: torch.Tensor  # [4]
    dominant_archetype: Archetype
    individuation_stage: IndividuationStage
    self_model: torch.Tensor  # embedding del organismo
    vertical_coherence: float

    def __post_init__(self):
        """Normalizar estado global."""
        if self.global_archetype.sum() > 0:
            self.global_archetype = F.softmax(self.global_archetype, dim=0)

    @property
    def consciousness_level(self) -> float:
        """Nivel de consciencia total [0, 1]."""
        return self.consciousness_index.compute_total()

    @property
    def is_integrated(self) -> bool:
        """¿El organismo tiene alta integración?"""
        return self.phi_global > 0.5 and self.vertical_coherence > 0.5

    @property
    def is_individuated(self) -> bool:
        """¿El organismo ha alcanzado etapas avanzadas?"""
        advanced_stages = [
            IndividuationStage.EMERGENCIA_SELF,
            IndividuationStage.SELF_REALIZADO
        ]
        return self.individuation_stage in advanced_stages

    @property
    def balance(self) -> float:
        """Balance arquetipal (0=dominante único, 1=equilibrio perfecto)."""
        return self.global_archetype.min().item() * 4

    def get_complementary_need(self) -> Vertex:
        """
        Retorna el vertice que mas necesita atencion.

        Es el opuesto geometrico del dominante, para promover equilibrio.
        """
        space = get_tetrahedral_space()
        return space.get_complement(self.dominant_archetype)

    def get_weakest_archetype(self) -> Archetype:
        """Retorna el arquetipo con menor peso."""
        min_idx = self.global_archetype.argmin().item()
        return Archetype(min_idx)

    def alignment_with(self, other_state: torch.Tensor) -> float:
        """Calcula alineación con otro estado."""
        sim = F.cosine_similarity(
            self.global_archetype.unsqueeze(0),
            other_state.unsqueeze(0)
        ).item()
        return (sim + 1) / 2

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            'consciousness_level': self.consciousness_level,
            'phi_global': self.phi_global,
            'global_archetype': self.global_archetype.tolist(),
            'dominant_archetype': self.dominant_archetype.name,
            'individuation_stage': self.individuation_stage.name,
            'vertical_coherence': self.vertical_coherence,
            'is_integrated': self.is_integrated,
            'is_individuated': self.is_individuated,
            'balance': self.balance,
            'consciousness_index': self.consciousness_index.to_dict()
        }

    @classmethod
    def create_initial(cls) -> 'OrganismConsciousness':
        """Crea consciencia inicial (estado base)."""
        # Seleccionar arquetipo aleatorio para evitar sesgo hacia PERSONA
        random_archetype = Archetype(np.random.randint(4))
        return cls(
            consciousness_index=ConsciousnessIndex(),
            phi_global=0.0,
            global_archetype=torch.ones(4) / 4,
            dominant_archetype=random_archetype,
            individuation_stage=IndividuationStage.INCONSCIENTE,
            self_model=torch.zeros(6),  # archetype[4] + phi + coherence
            vertical_coherence=0.0
        )

    @classmethod
    def from_clusters(
        cls,
        clusters: list[Cluster],
        prev_consciousness: Optional['OrganismConsciousness'] = None
    ) -> 'OrganismConsciousness':
        """
        Crea OrganismConsciousness agregando clusters.

        Args:
            clusters: Lista de clusters del organismo
            prev_consciousness: Consciencia anterior (para continuidad)

        Returns:
            Nueva OrganismConsciousness
        """
        if not clusters:
            return cls.create_initial()

        # Filtrar clusters con psique válida
        valid_clusters = [c for c in clusters if c.psyche is not None]
        if not valid_clusters:
            return cls.create_initial()

        # 1. Calcular pesos de cada cluster
        # Peso = phi_cluster × tamaño (proporcional, sin softmax para preservar ratios)
        weight_list: list[float] = []
        for cluster in valid_clusters:
            # cluster.psyche is guaranteed to be non-None due to filtering above
            assert cluster.psyche is not None
            weight = cluster.psyche.phi_cluster * cluster.size
            weight_list.append(weight)

        weights = torch.tensor(weight_list)
        if weights.sum() > 0:
            # Normalización simple (no softmax) para preservar proporciones
            weights = weights / weights.sum()
        else:
            weights = torch.ones(len(valid_clusters)) / len(valid_clusters)

        # 2. Agregación de estados arquetipales
        # All clusters have non-None psyche due to filtering
        states = torch.stack([c.psyche.aggregate_state for c in valid_clusters if c.psyche is not None])
        global_archetype = (weights.unsqueeze(1) * states).sum(dim=0)
        # NO aplicar softmax adicional - la agregación ya produce distribución válida

        # 3. Arquetipo dominante
        dominant = Archetype(unbiased_argmax(global_archetype))

        # 4. Φ global (coherencia inter-cluster + diversidad)
        # Queremos: alta coherencia interna + diversidad de especializaciones
        specializations = [c.psyche.specialization.value for c in valid_clusters if c.psyche is not None]
        unique_specs = len(set(specializations))
        diversity_bonus = unique_specs / 4.0

        phi_values = [c.psyche.phi_cluster for c in valid_clusters if c.psyche is not None]
        avg_phi_cluster = float(np.mean(phi_values))
        phi_global: float = avg_phi_cluster * (0.5 + 0.5 * diversity_bonus)

        # 5. Coherencia vertical (calculada externamente, usar placeholder)
        vertical_coherence: float = avg_phi_cluster  # Placeholder

        # 6. Etapa de individuación
        integration = float(global_archetype.min().item())  # Arquetipo más débil
        stage = _integration_to_stage(integration, phi_global)

        # 7. Self-model
        self_model = torch.cat([
            global_archetype,
            torch.tensor([phi_global, vertical_coherence])
        ])

        # 8. Índice de consciencia
        consciousness_index = _compute_consciousness_index(
            clusters=valid_clusters,
            phi_global=phi_global,
            vertical_coherence=vertical_coherence,
            prev_index=prev_consciousness.consciousness_index if prev_consciousness else None
        )

        return cls(
            consciousness_index=consciousness_index,
            phi_global=phi_global,
            global_archetype=global_archetype,
            dominant_archetype=dominant,
            individuation_stage=stage,
            self_model=self_model,
            vertical_coherence=vertical_coherence
        )

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _integration_to_stage(integration: float, phi: float) -> IndividuationStage:
    """
    Mapea nivel de integración a etapa de individuación.

    Args:
        integration: Valor del arquetipo más débil [0, 0.25]
        phi: Φ global [0, 1]

    Returns:
        Etapa de individuación correspondiente
    """
    # Score combinado
    score = integration * 4 * 0.6 + phi * 0.4  # integration normalizado a [0,1]

    if score < 0.1:
        return IndividuationStage.INCONSCIENTE
    elif score < 0.2:
        return IndividuationStage.CRISIS_PERSONA
    elif score < 0.3:
        return IndividuationStage.ENCUENTRO_SOMBRA
    elif score < 0.45:
        return IndividuationStage.INTEGRACION_SOMBRA
    elif score < 0.6:
        return IndividuationStage.ENCUENTRO_ANIMA
    elif score < 0.75:
        return IndividuationStage.INTEGRACION_ANIMA
    elif score < 0.9:
        return IndividuationStage.EMERGENCIA_SELF
    else:
        return IndividuationStage.SELF_REALIZADO

def _compute_consciousness_index(
    clusters: list[Cluster],
    phi_global: float,
    vertical_coherence: float,
    prev_index: ConsciousnessIndex | None = None
) -> ConsciousnessIndex:
    """
    Calcula el índice de consciencia desde los clusters.

    Args:
        clusters: Lista de clusters
        phi_global: Φ global
        vertical_coherence: Coherencia vertical
        prev_index: Índice anterior (para estabilidad)

    Returns:
        Nuevo ConsciousnessIndex
    """
    # Predictive: inverso del error de predicción promedio
    pred_errors = [c.psyche.prediction_error for c in clusters if c.psyche is not None]
    avg_error = float(np.mean(pred_errors)) if pred_errors else 0.5
    predictive = 1.0 - avg_error

    # Attention: basado en coherencia de clusters
    coherences = [c.psyche.coherence for c in clusters if c.psyche is not None]
    attention = float(np.mean(coherences)) if coherences else 0.5

    # Integration: basado en phi_global
    integration = phi_global

    # Self luminosity: diversidad de especializaciones
    specializations = [c.psyche.specialization.value for c in clusters if c.psyche is not None]
    unique_specs = len(set(specializations))
    self_luminosity = unique_specs / 4.0

    # Stability: comparar con índice anterior
    if prev_index:
        stability = 1.0 - abs(prev_index.compute_total() -
                             (predictive + attention + integration) / 3)
    else:
        stability = 0.5

    # Meta-awareness: basado en vertical_coherence
    meta_awareness = vertical_coherence

    return ConsciousnessIndex(
        predictive=predictive,
        attention=attention,
        integration=integration,
        self_luminosity=self_luminosity,
        stability=stability,
        meta_awareness=meta_awareness
    )

# =============================================================================
# HIERARCHICAL METRICS
# =============================================================================

@dataclass
class HierarchicalMetrics:
    """Métricas completas del sistema jerárquico."""

    # Nivel 0: Células
    cell_count: int
    avg_cell_energy: float
    avg_cell_phi_local: float
    archetype_distribution: dict[str, float]
    role_distribution: dict[str, float]

    # Nivel 1: Clusters
    cluster_count: int
    avg_cluster_size: float
    avg_cluster_phi: float
    avg_cluster_coherence: float
    cluster_specializations: dict[str, int]

    # Nivel 2: Organismo
    consciousness_index: float
    phi_global: float
    vertical_coherence: float
    individuation_stage: str
    dominant_archetype: str

    # Flujos de integración
    bottom_up_flow: float
    top_down_flow: float
    horizontal_flow: float

    # Temporales
    stability: float
    adaptability: float
    insight_rate: float

    def to_dict(self) -> dict:
        """Serializa a diccionario."""
        return {
            'cell_count': self.cell_count,
            'avg_cell_energy': self.avg_cell_energy,
            'avg_cell_phi_local': self.avg_cell_phi_local,
            'archetype_distribution': self.archetype_distribution,
            'role_distribution': self.role_distribution,
            'cluster_count': self.cluster_count,
            'avg_cluster_size': self.avg_cluster_size,
            'avg_cluster_phi': self.avg_cluster_phi,
            'avg_cluster_coherence': self.avg_cluster_coherence,
            'cluster_specializations': self.cluster_specializations,
            'consciousness_index': self.consciousness_index,
            'phi_global': self.phi_global,
            'vertical_coherence': self.vertical_coherence,
            'individuation_stage': self.individuation_stage,
            'dominant_archetype': self.dominant_archetype,
            'bottom_up_flow': self.bottom_up_flow,
            'top_down_flow': self.top_down_flow,
            'horizontal_flow': self.horizontal_flow,
            'stability': self.stability,
            'adaptability': self.adaptability,
            'insight_rate': self.insight_rate
        }

    @classmethod
    def compute(
        cls,
        cells: list[ConsciousCell],
        clusters: list[Cluster],
        organism: OrganismConsciousness,
        prev_metrics: Optional['HierarchicalMetrics'] = None
    ) -> 'HierarchicalMetrics':
        """
        Calcula métricas completas del sistema.

        Args:
            cells: Lista de todas las células
            clusters: Lista de clusters
            organism: Consciencia del organismo
            prev_metrics: Métricas anteriores (para estabilidad)
        """
        # === NIVEL 0: CÉLULAS ===
        avg_energy = np.mean([c.energy for c in cells]) if cells else 0
        avg_phi_local = np.mean([c.psyche.phi_local for c in cells]) if cells else 0

        # Distribución de arquetipos
        arch_counts = {a.name: 0 for a in Archetype}
        for cell in cells:
            arch_counts[cell.psyche.dominant.name] += 1
        arch_dist = {k: v / max(1, len(cells)) for k, v in arch_counts.items()}

        # Distribución de roles
        role_names = ['MASS', 'FORCE', 'CORRUPT']
        role_counts: dict[str, int] = {'MASS': 0, 'FORCE': 0, 'CORRUPT': 0}
        for cell in cells:
            role_idx = int(cell.role.argmax().item())
            role_name = role_names[role_idx]
            role_counts[role_name] += 1
        role_dist = {k: v / max(1, len(cells)) for k, v in role_counts.items()}

        # === NIVEL 1: CLUSTERS ===
        valid_clusters = [c for c in clusters if c.psyche is not None]
        cluster_sizes = [c.size for c in clusters]
        avg_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0

        cluster_phis = [c.psyche.phi_cluster for c in valid_clusters if c.psyche is not None]
        avg_phi = float(np.mean(cluster_phis)) if cluster_phis else 0.0

        cluster_coherences = [c.psyche.coherence for c in valid_clusters if c.psyche is not None]
        avg_coherence = float(np.mean(cluster_coherences)) if cluster_coherences else 0.0

        spec_counts: dict[str, int] = {a.name: 0 for a in Archetype}
        for cluster in valid_clusters:
            if cluster.psyche is not None:
                spec_counts[cluster.psyche.specialization.name] += 1

        # === FLUJOS ===
        # Bottom-up: qué tan bien clusters representan células
        bottom_up = _compute_bottom_up_flow(cells, clusters)

        # Top-down: alineación de células con organismo
        top_down = _compute_top_down_flow(cells, organism)

        # Horizontal: interacción entre clusters
        horizontal = _compute_horizontal_flow(clusters)

        # === TEMPORALES ===
        if prev_metrics:
            stability = 1.0 - abs(
                prev_metrics.consciousness_index - organism.consciousness_level
            )
            # Adaptabilidad: cambio en etapa
            if prev_metrics.individuation_stage != organism.individuation_stage.name:
                adaptability = 0.8
            else:
                adaptability = 0.5
        else:
            stability = 0.5
            adaptability = 0.5

        return cls(
            cell_count=len(cells),
            avg_cell_energy=float(avg_energy),
            avg_cell_phi_local=float(avg_phi_local),
            archetype_distribution=arch_dist,
            role_distribution=role_dist,
            cluster_count=len(clusters),
            avg_cluster_size=float(avg_size),
            avg_cluster_phi=float(avg_phi),
            avg_cluster_coherence=avg_coherence,
            cluster_specializations=spec_counts,
            consciousness_index=organism.consciousness_level,
            phi_global=organism.phi_global,
            vertical_coherence=organism.vertical_coherence,
            individuation_stage=organism.individuation_stage.name,
            dominant_archetype=organism.dominant_archetype.name,
            bottom_up_flow=bottom_up,
            top_down_flow=top_down,
            horizontal_flow=horizontal,
            stability=stability,
            adaptability=adaptability,
            insight_rate=0.0  # Calculado externamente
        )

def _compute_bottom_up_flow(cells: list[ConsciousCell], clusters: list[Cluster]) -> float:
    """Calcula calidad del flujo bottom-up."""
    if not cells or not clusters:
        return 0.0

    total_rep = 0.0
    for cluster in clusters:
        if cluster.psyche and cluster.cells:
            for cell in cluster.cells:
                sim = cell.psyche.alignment_with(cluster.psyche.aggregate_state)
                total_rep += sim

    return total_rep / len(cells)

def _compute_top_down_flow(cells: list[ConsciousCell], organism: OrganismConsciousness) -> float:
    """Calcula calidad del flujo top-down."""
    if not cells:
        return 0.0

    # Células con alta energía emocional están más "conectadas"
    responsive = [c for c in cells if c.psyche.emotional_energy > 0.5]
    if not responsive:
        return 0.3

    alignments = [c.psyche.alignment_with(organism.global_archetype) for c in responsive]
    return float(np.mean(alignments))

def _compute_horizontal_flow(clusters: list[Cluster]) -> float:
    """Calcula flujo horizontal entre clusters."""
    if len(clusters) < 2:
        return 0.0

    interactions = 0.0
    total_pairs = 0

    for cluster in clusters:
        if not cluster.psyche:
            continue
        for neighbor_id in cluster.neighbors:
            neighbor = next((c for c in clusters if c.id == neighbor_id), None)
            if neighbor and neighbor.psyche:
                sim = cluster.psyche.alignment_with(neighbor.psyche.aggregate_state)
                interactions += sim
                total_pairs += 1

    return interactions / max(1, total_pairs)

# =============================================================================
# TESTS BÁSICOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: OrganismConsciousness")
    print("=" * 60)

    # Crear células y clusters de prueba
    print("\n1. Crear células y clusters:")
    all_cells = []
    clusters = []

    for cluster_id in range(4):
        # Crear 10 células por cluster con bias hacia un arquetipo
        archetype = Archetype(cluster_id)
        cells = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
            for _ in range(10)
        ]
        all_cells.extend(cells)

        cluster = Cluster.create_from_cells(cluster_id=cluster_id, cells=cells)
        clusters.append(cluster)

        specialization_name = cluster.psyche.specialization.name if cluster.psyche else "None"
        print(f"   Cluster {cluster_id}: {cluster.size} células, "
              f"especialización={specialization_name}")

    # Crear OrganismConsciousness
    print("\n2. Crear OrganismConsciousness:")
    organism = OrganismConsciousness.from_clusters(clusters)

    print(f"   Consciencia: {organism.consciousness_level:.3f}")
    print(f"   Φ global: {organism.phi_global:.3f}")
    print(f"   Dominante: {organism.dominant_archetype.name}")
    print(f"   Etapa: {organism.individuation_stage.name}")
    print(f"   Balance: {organism.balance:.3f}")
    print(f"   Integrado: {organism.is_integrated}")

    # Self-model
    print("\n3. Self-model:")
    print(f"   Dimensión: {organism.self_model.shape}")
    print(f"   Valores: {organism.self_model.tolist()}")

    # Calcular métricas
    print("\n4. Métricas jerárquicas:")
    metrics = HierarchicalMetrics.compute(all_cells, clusters, organism)

    print(f"   Células: {metrics.cell_count}")
    print(f"   Clusters: {metrics.cluster_count}")
    print(f"   Bottom-up flow: {metrics.bottom_up_flow:.3f}")
    print(f"   Top-down flow: {metrics.top_down_flow:.3f}")
    print(f"   Horizontal flow: {metrics.horizontal_flow:.3f}")

    # Distribución de arquetipos
    print("\n5. Distribución de arquetipos:")
    for arch, pct in metrics.archetype_distribution.items():
        print(f"   {arch}: {pct:.1%}")

    # Especializaciones de clusters
    print("\n6. Especializaciones de clusters:")
    for arch, count in metrics.cluster_specializations.items():
        print(f"   {arch}: {count} clusters")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETADOS")
    print("=" * 60)
