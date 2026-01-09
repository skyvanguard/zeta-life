# -*- coding: utf-8 -*-
"""
BottomUpIntegrator: Agregación de información de niveles inferiores a superiores.

Implementa el flujo bottom-up en la consciencia jerárquica:
- Células → Clusters (agregación de micro-psiques)
- Clusters → Organismo (emergencia de consciencia global)

La consciencia emerge de este proceso de agregación.

Fecha: 2026-01-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Importar del sistema existente
from ..psyche.zeta_psyche import Archetype
from ..psyche.zeta_individuation import IndividuationStage
from ..psyche.zeta_conscious_self import ConsciousnessIndex

# Importar de módulos nuevos
from .micro_psyche import ConsciousCell, MicroPsyche, unbiased_argmax
from .cluster import Cluster, ClusterPsyche
from .organism_consciousness import OrganismConsciousness


# =============================================================================
# BOTTOM-UP INTEGRATOR
# =============================================================================

class BottomUpIntegrator(nn.Module):
    """
    Agrega información de niveles inferiores hacia superiores.

    Flujo:
    1. Células → Cluster: promedio ponderado de micro-psiques
    2. Clusters → Organismo: votación arquetipal + integración

    La consciencia emerge de la calidad de esta integración.
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        n_archetypes: int = 4
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_archetypes = n_archetypes

        # Red para calcular importancia de cada célula en su cluster
        # Input: state[32] + archetype[4] = 36
        self.cell_importance_net = nn.Sequential(
            nn.Linear(state_dim + n_archetypes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Red para calcular importancia de cada cluster en organismo
        # Input: archetype[4] + metrics[3] = 7
        self.cluster_importance_net = nn.Sequential(
            nn.Linear(n_archetypes + 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Inicializar pesos
        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # =========================================================================
    # NIVEL 0 → NIVEL 1: Células a Clusters
    # =========================================================================

    def compute_cell_importance(self, cell: ConsciousCell) -> float:
        """
        Calcula la importancia de una célula para su cluster.

        Factores:
        - Estado interno
        - Estado arquetipal
        - Energía
        - Φ local

        Returns:
            Importancia normalizada [0, 1]
        """
        # Concatenar features (asegurar float32)
        features = torch.cat([
            cell.state.float(),
            cell.psyche.archetype_state.float()
        ])

        # Pasar por red
        with torch.no_grad():
            base_importance = self.cell_importance_net(features).item()

        # Modular por energía y phi_local
        importance = base_importance * cell.energy * cell.psyche.phi_local

        return float(max(0.01, min(1.0, importance)))

    def aggregate_cells_to_cluster(
        self,
        cells: List[ConsciousCell]
    ) -> ClusterPsyche:
        """
        Agrega micro-psiques de células en psique de cluster.

        Algoritmo:
        1. Calcular peso de cada célula (energía × coherencia × importancia)
        2. Promedio ponderado de estados arquetipales
        3. Calcular Φ₁ (integración intra-cluster)
        4. Detectar especialización emergente

        Args:
            cells: Lista de células del cluster

        Returns:
            ClusterPsyche agregada
        """
        if not cells:
            return ClusterPsyche.create_empty()

        # 1. Calcular pesos de células
        weight_list: List[float] = []
        for cell in cells:
            importance = self.compute_cell_importance(cell)
            weight = importance * cell.energy * cell.psyche.phi_local
            weight_list.append(weight)

        weights = torch.tensor(weight_list, dtype=torch.float32)

        # Normalizar pesos (simple, no softmax para preservar proporciones)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(len(cells)) / len(cells)

        # 2. Agregación ponderada de arquetipos
        archetype_states = torch.stack([c.psyche.archetype_state for c in cells])
        aggregate = (weights.unsqueeze(1) * archetype_states).sum(dim=0)
        # NO aplicar softmax - el agregado ya es distribución válida

        # 3. Calcular Φ cluster (coherencia intra-cluster)
        # Φ = 1 - varianza_normalizada
        # Nota: varianza requiere al menos 2 elementos para ser significativa
        n_cells = archetype_states.shape[0]
        if n_cells >= 2:
            variance = archetype_states.var(dim=0).mean().item()
        else:
            # Con 1 célula, no hay variabilidad grupal - asumimos coherencia perfecta
            variance = 0.0
        phi_cluster = 1.0 - min(1.0, variance * 2)

        # 4. Especialización = arquetipo con mayor peso
        specialization = Archetype(unbiased_argmax(aggregate))

        # 5. Coherencia = acuerdo entre células
        coherence = 1.0 - variance

        # 6. Error de predicción agregado (sorpresa promedio)
        surprises = [c.psyche.compute_surprise() for c in cells]
        prediction_error = float(np.mean(surprises)) if surprises else 0.0

        # 7. Nivel de integración
        integration_level = float(phi_cluster * (1 - prediction_error))

        return ClusterPsyche(
            aggregate_state=aggregate,
            specialization=specialization,
            phi_cluster=phi_cluster,
            coherence=coherence,
            prediction_error=prediction_error,
            integration_level=integration_level
        )

    def update_all_cluster_psyches(self, clusters: List[Cluster]) -> None:
        """
        Actualiza la psique de todos los clusters.

        Args:
            clusters: Lista de clusters a actualizar
        """
        for cluster in clusters:
            if cluster.cells:
                cluster.psyche = self.aggregate_cells_to_cluster(cluster.cells)

    # =========================================================================
    # NIVEL 1 → NIVEL 2: Clusters a Organismo
    # =========================================================================

    def compute_cluster_importance(self, cluster: Cluster) -> float:
        """
        Calcula la importancia de un cluster para el organismo.

        Factores:
        - Estado arquetipal
        - Φ del cluster
        - Coherencia
        - Tamaño

        Returns:
            Importancia normalizada [0, 1]
        """
        if not cluster.psyche:
            return 0.1

        # Features del cluster (asegurar float32)
        features = torch.cat([
            cluster.psyche.aggregate_state.float(),
            torch.tensor([
                cluster.psyche.phi_cluster,
                cluster.psyche.coherence,
                min(1.0, cluster.size / 20.0)  # Tamaño normalizado
            ], dtype=torch.float32)
        ])

        # Pasar por red
        with torch.no_grad():
            base_importance = self.cluster_importance_net(features).item()

        # Modular por phi
        importance = base_importance * cluster.psyche.phi_cluster

        return float(max(0.01, min(1.0, importance)))

    def aggregate_clusters_to_organism(
        self,
        clusters: List[Cluster],
        prev_consciousness: Optional[OrganismConsciousness] = None
    ) -> OrganismConsciousness:
        """
        Agrega psiques de clusters en consciencia de organismo.

        Algoritmo:
        1. Calcular peso de cada cluster
        2. Votación ponderada de arquetipos
        3. Calcular Φ₂ (integración inter-cluster)
        4. Determinar etapa de individuación
        5. Construir self-model

        Args:
            clusters: Lista de clusters del organismo
            prev_consciousness: Consciencia anterior (para continuidad)

        Returns:
            OrganismConsciousness integrada
        """
        # Filtrar clusters válidos
        valid_clusters = [c for c in clusters if c.psyche is not None and c.cells]

        if not valid_clusters:
            return OrganismConsciousness.create_initial()

        # 1. Calcular pesos de clusters (basado en importancia × tamaño)
        weight_list: List[float] = []
        for cluster in valid_clusters:
            importance = self.compute_cluster_importance(cluster)
            # Peso proporcional a tamaño para preservar distribución
            weight = importance * cluster.size
            weight_list.append(weight)

        weights = torch.tensor(weight_list, dtype=torch.float32)
        # Normalización simple (no softmax) para preservar proporciones
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(len(valid_clusters)) / len(valid_clusters)

        # 2. Agregación de arquetipos (votación ponderada)
        # Note: valid_clusters already filtered to have non-None psyche
        cluster_states = torch.stack([c.psyche.aggregate_state for c in valid_clusters if c.psyche is not None])
        global_archetype = (weights.unsqueeze(1) * cluster_states).sum(dim=0)
        # NO aplicar softmax - el agregado ya es distribución válida

        # 3. Arquetipo dominante
        dominant = Archetype(unbiased_argmax(global_archetype))

        # 4. Φ global (coherencia inter-cluster + diversidad)
        phi_global = self._compute_phi_global(valid_clusters)

        # 5. Coherencia vertical real (células→clusters→organismo)
        vertical_coherence = self._compute_vertical_coherence(
            valid_clusters, global_archetype, dominant
        )

        # 6. Etapa de individuación
        integration = global_archetype.min().item()  # Arquetipo más débil
        stage = self._integration_to_stage(integration, phi_global)

        # 7. Self-model
        self_model = torch.cat([
            global_archetype,
            torch.tensor([phi_global, vertical_coherence])
        ])

        # 8. Índice de consciencia
        consciousness_index = self._compute_consciousness_index(
            valid_clusters, phi_global, vertical_coherence,
            prev_consciousness.consciousness_index if prev_consciousness else None
        )

        return OrganismConsciousness(
            consciousness_index=consciousness_index,
            phi_global=phi_global,
            global_archetype=global_archetype,
            dominant_archetype=dominant,
            individuation_stage=stage,
            self_model=self_model,
            vertical_coherence=vertical_coherence
        )

    def _compute_inter_cluster_integration(self, clusters: List[Cluster]) -> float:
        """
        Calcula integración real entre clusters.

        Mide qué tan conectados están los clusters entre sí usando
        similitud coseno de sus estados agregados.

        Integración óptima requiere:
        - Clusters suficientemente diferentes (no idénticos)
        - Pero no completamente desconectados

        Returns:
            Valor de integración [0, 1]
        """
        if len(clusters) < 2:
            return 0.7  # Valor alto con un solo cluster (trivialmente integrado)

        # Calcular similitud coseno entre todos los pares de clusters
        similarities = []
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if i < j and c1.psyche and c2.psyche:
                    sim = F.cosine_similarity(
                        c1.psyche.aggregate_state.unsqueeze(0).float(),
                        c2.psyche.aggregate_state.unsqueeze(0).float()
                    ).item()
                    # Normalizar de [-1, 1] a [0, 1]
                    similarities.append((sim + 1) / 2)

        if not similarities:
            return 0.7

        avg_sim = float(np.mean(similarities))

        # Función de integración más suave:
        # - sim=0.0 (opuestos): integration=0.6 (conectados pero muy diferentes)
        # - sim=0.5 (moderado): integration=1.0 (óptimo)
        # - sim=1.0 (idénticos): integration=0.5 (sincronización, no integración)
        #
        # Usamos una parábola invertida centrada en 0.5, pero más suave
        # integration = 1.0 - (avg_sim - 0.5)^2 * 2
        deviation = abs(avg_sim - 0.5)
        integration = 1.0 - deviation * deviation * 2

        # Asegurar mínimo razonable
        integration = max(0.4, min(1.0, integration))

        return integration

    def _compute_phi_global(self, clusters: List[Cluster]) -> float:
        """
        Calcula Φ global del organismo.

        Combina tres factores:
        1. Coherencia intra-cluster: qué tan coherentes son los clusters internamente
        2. Integración inter-cluster: qué tan conectados están entre sí
        3. Diversidad arquetipal: qué tan diversos son los arquetipos

        Φ = intra × inter × (0.6 + 0.4 × diversidad)
        """
        if not clusters:
            return 0.0

        valid_clusters = [c for c in clusters if c.psyche is not None]
        if not valid_clusters:
            return 0.0

        # 1. Coherencia intra-cluster promedio
        intra_coherences = [c.psyche.phi_cluster for c in valid_clusters if c.psyche is not None]
        avg_intra = float(np.mean(intra_coherences)) if intra_coherences else 0.0

        # 2. Integración inter-cluster (similitud real entre clusters)
        inter_integration = self._compute_inter_cluster_integration(valid_clusters)

        # 3. Diversidad de especializaciones
        specializations = [c.psyche.specialization.value for c in valid_clusters if c.psyche is not None]
        unique_specs = len(set(specializations))
        diversity_bonus = unique_specs / 4.0  # 1.0 si tiene los 4 arquetipos

        # Φ global = promedio geométrico de intra e inter, modulado por diversidad
        # Esto requiere que AMBOS sean altos para tener phi alto
        phi_base = float(np.sqrt(avg_intra * inter_integration))
        phi_global = phi_base * (0.6 + 0.4 * diversity_bonus)

        return float(min(1.0, phi_global))

    def _compute_vertical_coherence(
        self,
        clusters: List[Cluster],
        organism_state: torch.Tensor,
        organism_dominant: Archetype
    ) -> float:
        """
        Calcula coherencia vertical real entre niveles jerárquicos.

        Mide la consistencia de información a través de:
        - Nivel 0 (Células) → Nivel 1 (Clusters) → Nivel 2 (Organismo)

        Componentes:
        1. Cell-to-Cluster: Alineación de células con su cluster
        2. Cluster-to-Organism: Alineación de clusters con el organismo
        3. Hierarchy Consistency: Consistencia del arquetipo dominante

        Args:
            clusters: Lista de clusters con psyche
            organism_state: Estado arquetipal global del organismo
            organism_dominant: Arquetipo dominante del organismo

        Returns:
            Coherencia vertical [0, 1]
        """
        if not clusters:
            return 0.0

        valid_clusters = [c for c in clusters if c.psyche is not None and c.cells]
        if not valid_clusters:
            return 0.0

        # =====================================================================
        # 1. CELL-TO-CLUSTER ALIGNMENT
        # ¿Las células se alinean con su cluster?
        # =====================================================================
        cell_to_cluster_scores: List[float] = []
        for cluster in valid_clusters:
            if cluster.psyche is None:
                continue
            cluster_state = cluster.psyche.aggregate_state
            for cell in cluster.cells:
                # Similitud coseno entre célula y su cluster
                sim = F.cosine_similarity(
                    cell.psyche.archetype_state.unsqueeze(0).float(),
                    cluster_state.unsqueeze(0).float()
                ).item()
                # Normalizar a [0, 1]
                cell_to_cluster_scores.append((sim + 1) / 2)

        avg_cell_to_cluster = float(np.mean(cell_to_cluster_scores)) if cell_to_cluster_scores else 0.5

        # =====================================================================
        # 2. CLUSTER-TO-ORGANISM ALIGNMENT
        # ¿Los clusters se alinean con el organismo?
        # =====================================================================
        cluster_to_org_scores: List[float] = []
        for cluster in valid_clusters:
            if cluster.psyche is None:
                continue
            sim = F.cosine_similarity(
                cluster.psyche.aggregate_state.unsqueeze(0).float(),
                organism_state.unsqueeze(0).float()
            ).item()
            cluster_to_org_scores.append((sim + 1) / 2)

        avg_cluster_to_org = float(np.mean(cluster_to_org_scores)) if cluster_to_org_scores else 0.5

        # =====================================================================
        # 3. HIERARCHY CONSISTENCY
        # ¿El arquetipo dominante es consistente entre niveles?
        # =====================================================================
        # Contar cuántos clusters comparten el dominante del organismo
        clusters_aligned = sum(
            1 for c in valid_clusters
            if c.psyche is not None and c.psyche.specialization == organism_dominant
        )
        cluster_consistency = clusters_aligned / len(valid_clusters)

        # Contar células que comparten el dominante de su cluster
        cells_aligned = 0
        total_cells = 0
        for cluster in valid_clusters:
            if cluster.psyche is None:
                continue
            cluster_dominant = cluster.psyche.specialization
            for cell in cluster.cells:
                total_cells += 1
                if cell.psyche.dominant == cluster_dominant:
                    cells_aligned += 1

        cell_consistency = cells_aligned / total_cells if total_cells > 0 else 0.5

        # =====================================================================
        # COHERENCIA VERTICAL FINAL
        # =====================================================================
        # Promedio ponderado de las tres componentes:
        # - Cell-to-Cluster: 40% (base de la jerarquía)
        # - Cluster-to-Organism: 30% (nivel intermedio)
        # - Hierarchy Consistency: 30% (consistencia global)
        vertical_coherence = (
            0.4 * avg_cell_to_cluster +
            0.3 * avg_cluster_to_org +
            0.15 * cluster_consistency +
            0.15 * cell_consistency
        )

        return float(min(1.0, max(0.0, vertical_coherence)))

    def _integration_to_stage(self, integration: float, phi: float) -> IndividuationStage:
        """
        Mapea nivel de integración a etapa de individuación.

        Args:
            integration: Valor del arquetipo más débil [0, 0.25]
            phi: Φ global [0, 1]
        """
        # Score combinado (integration normalizado a [0,1])
        score = integration * 4 * 0.6 + phi * 0.4

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
        self,
        clusters: List[Cluster],
        phi_global: float,
        vertical_coherence: float,
        prev_index: Optional[ConsciousnessIndex] = None
    ) -> ConsciousnessIndex:
        """
        Calcula el índice de consciencia desde los clusters.
        """
        # Filter valid clusters
        valid_clusters = [c for c in clusters if c.psyche is not None]

        # Predictive: inverso del error de predicción promedio
        pred_errors = [c.psyche.prediction_error for c in valid_clusters if c.psyche is not None]
        avg_error = float(np.mean(pred_errors)) if pred_errors else 0.5
        predictive = 1.0 - avg_error

        # Attention: basado en coherencia de clusters
        coherences = [c.psyche.coherence for c in valid_clusters if c.psyche is not None]
        attention = float(np.mean(coherences)) if coherences else 0.5

        # Integration: basado en phi_global
        integration = phi_global

        # Self luminosity: diversidad de especializaciones
        specializations = [c.psyche.specialization.value for c in valid_clusters if c.psyche is not None]
        unique_specs = len(set(specializations))
        self_luminosity = unique_specs / 4.0

        # Stability: comparar con índice anterior
        if prev_index:
            prev_total = prev_index.compute_total()
            current_estimate = (predictive + attention + integration) / 3
            stability = 1.0 - min(1.0, abs(prev_total - current_estimate) * 2)
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

    # =========================================================================
    # INTEGRACIÓN COMPLETA
    # =========================================================================

    def integrate(
        self,
        cells: List[ConsciousCell],
        clusters: List[Cluster],
        prev_consciousness: Optional[OrganismConsciousness] = None
    ) -> Tuple[List[Cluster], OrganismConsciousness]:
        """
        Realiza integración bottom-up completa.

        1. Actualiza psiques de todos los clusters
        2. Agrega clusters en consciencia de organismo

        Args:
            cells: Todas las células
            clusters: Todos los clusters
            prev_consciousness: Consciencia anterior

        Returns:
            (clusters_actualizados, nueva_consciencia)
        """
        # Paso 1: Actualizar psiques de clusters
        self.update_all_cluster_psyches(clusters)

        # Paso 2: Agregar en organismo
        organism = self.aggregate_clusters_to_organism(clusters, prev_consciousness)

        return clusters, organism

    def compute_integration_quality(
        self,
        cells: List[ConsciousCell],
        clusters: List[Cluster],
        organism: OrganismConsciousness
    ) -> float:
        """
        Calcula la calidad de la integración bottom-up.

        Mide qué tan bien los niveles superiores representan a los inferiores.

        Returns:
            Calidad de integración [0, 1]
        """
        if not cells or not clusters:
            return 0.0

        total_representation = 0.0

        # Para cada cluster, verificar que representa bien a sus células
        for cluster in clusters:
            if not cluster.psyche or not cluster.cells:
                continue

            for cell in cluster.cells:
                # Similitud célula-cluster
                sim = F.cosine_similarity(
                    cell.psyche.archetype_state.unsqueeze(0),
                    cluster.psyche.aggregate_state.unsqueeze(0)
                ).item()
                total_representation += (sim + 1) / 2  # Normalizar a [0, 1]

        return total_representation / len(cells) if cells else 0.0


# =============================================================================
# TESTS BÁSICOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: BottomUpIntegrator")
    print("=" * 60)

    # Crear integrador
    integrator = BottomUpIntegrator(state_dim=32, hidden_dim=64)

    # Crear células de prueba
    print("\n1. Crear células con arquetipos variados:")
    all_cells = []
    for archetype in Archetype:
        for _ in range(10):
            cell = ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
            all_cells.append(cell)
    print(f"   Total: {len(all_cells)} células")

    # Crear clusters
    print("\n2. Crear clusters y agregar células:")
    clusters = []
    for i, archetype in enumerate(Archetype):
        # Filtrar células de este arquetipo
        arch_cells = [c for c in all_cells if c.psyche.dominant == archetype][:10]
        cluster = Cluster.create_from_cells(cluster_id=i, cells=arch_cells)
        clusters.append(cluster)
        print(f"   Cluster {i}: {cluster.size} células")

    # Test agregación células → cluster
    print("\n3. Agregar células a cluster (bottom-up nivel 1):")
    for cluster in clusters:
        psyche = integrator.aggregate_cells_to_cluster(cluster.cells)
        print(f"   Cluster {cluster.id}: spec={psyche.specialization.name}, "
              f"φ={psyche.phi_cluster:.3f}")

    # Test agregación clusters → organismo
    print("\n4. Agregar clusters a organismo (bottom-up nivel 2):")
    organism = integrator.aggregate_clusters_to_organism(clusters)
    print(f"   Consciencia: {organism.consciousness_level:.3f}")
    print(f"   Φ global: {organism.phi_global:.3f}")
    print(f"   Dominante: {organism.dominant_archetype.name}")
    print(f"   Etapa: {organism.individuation_stage.name}")

    # Test integración completa
    print("\n5. Integración completa:")
    clusters, organism = integrator.integrate(all_cells, clusters)
    print(f"   Consciencia final: {organism.consciousness_level:.3f}")

    # Calidad de integración
    print("\n6. Calidad de integración:")
    quality = integrator.compute_integration_quality(all_cells, clusters, organism)
    print(f"   Calidad: {quality:.3f}")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETADOS")
    print("=" * 60)
