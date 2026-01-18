"""
ClusterAssigner: Asignación dinámica de células a clusters.

Implementa clustering jerárquico basado en:
- Proximidad espacial (posición en la grilla)
- Similitud psíquica (arquetipos compatibles)
- Coherencia del cluster (maximizar phi)

El clustering es dinámico y se recalcula cada N pasos.

Fecha: 2026-01-03
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..organism.cell_state import CellRole

# Importar del sistema existente
from ..psyche.zeta_psyche import Archetype
from .cluster import Cluster, ClusterPsyche

# Importar de módulos nuevos
from .micro_psyche import ConsciousCell, MicroPsyche

# =============================================================================
# ESTRATEGIAS DE CLUSTERING
# =============================================================================

class ClusteringStrategy(Enum):
    """Estrategias de asignación de clusters."""
    SPATIAL = 0      # Solo por proximidad espacial
    PSYCHE = 1       # Solo por similitud psíquica
    HYBRID = 2       # Combinación espacial + psíquica
    ADAPTIVE = 3     # Adapta pesos según coherencia

@dataclass
class ClusteringConfig:
    """Configuración para el asignador de clusters."""
    # Número de clusters (dinámico)
    n_clusters: int = 4               # Número inicial de clusters
    min_clusters: int = 2             # Mínimo de clusters permitidos
    max_clusters: int = 8             # Máximo de clusters permitidos
    dynamic_clusters: bool = True     # Habilitar clustering dinámico

    # Estrategia
    strategy: ClusteringStrategy = ClusteringStrategy.HYBRID
    spatial_weight: float = 0.5       # Peso de proximidad espacial
    psyche_weight: float = 0.5        # Peso de similitud psíquica

    # Tamaño de clusters
    min_cluster_size: int = 3         # Mínimo de células por cluster
    max_cluster_size: int = 50        # Máximo de células por cluster

    # Reasignación
    reassign_interval: int = 10       # Cada cuántos pasos reasignar
    coherence_threshold: float = 0.3  # Umbral para reasignación forzada

    # Umbrales para split/merge (clustering dinámico)
    split_coherence_threshold: float = 0.4   # Dividir si coherencia < umbral
    merge_similarity_threshold: float = 0.85  # Fusionar si similitud > umbral
    split_min_size: int = 8           # Mínimo de células para poder dividir

# =============================================================================
# CLUSTER ASSIGNER
# =============================================================================

class ClusterAssigner:
    """
    Asigna células a clusters de forma dinámica.

    Algoritmo híbrido que considera:
    1. Proximidad espacial (k-means style)
    2. Similitud psíquica (arquetipos compatibles)
    3. Balance de tamaño (evitar clusters muy grandes/pequeños)
    """

    def __init__(self, config: ClusteringConfig | None = None):
        self.config = config or ClusteringConfig()
        self.step_count = 0
        self.cluster_history: list[dict[int, int]] = []  # cell_id -> cluster_id

    # =========================================================================
    # MÉTRICAS DE SIMILITUD
    # =========================================================================

    def compute_spatial_distance(
        self,
        cell: ConsciousCell,
        centroid: tuple[float, float]
    ) -> float:
        """
        Distancia espacial entre célula y centroide.

        Returns:
            Distancia euclidiana normalizada [0, 1]
        """
        dx = cell.position[0] - centroid[0]
        dy = cell.position[1] - centroid[1]
        dist = np.sqrt(dx**2 + dy**2)

        # Normalizar por diagonal de la grilla (asumiendo 64x64)
        max_dist = np.sqrt(64**2 + 64**2)
        return float(min(1.0, dist / max_dist))

    def compute_psyche_similarity(
        self,
        cell: ConsciousCell,
        cluster_psyche: torch.Tensor
    ) -> float:
        """
        Similitud psíquica entre célula y cluster.

        Args:
            cell: Célula a evaluar
            cluster_psyche: Estado arquetípico agregado del cluster

        Returns:
            Similitud [0, 1] (1 = muy similar)
        """
        similarity = F.cosine_similarity(
            cell.psyche.archetype_state.unsqueeze(0),
            cluster_psyche.unsqueeze(0)
        ).item()

        # Convertir de [-1, 1] a [0, 1]
        return (similarity + 1) / 2

    def compute_affinity(
        self,
        cell: ConsciousCell,
        cluster: Cluster
    ) -> float:
        """
        Afinidad total entre célula y cluster.

        Combina distancia espacial y similitud psíquica.

        Returns:
            Afinidad [0, 1] (1 = alta afinidad)
        """
        # Distancia espacial (invertir: menor distancia = mayor afinidad)
        spatial_dist = self.compute_spatial_distance(cell, cluster.centroid)
        spatial_affinity = 1.0 - spatial_dist

        # Similitud psíquica
        if cluster.psyche:
            psyche_sim = self.compute_psyche_similarity(
                cell, cluster.psyche.aggregate_state
            )
        else:
            psyche_sim = 0.5  # Neutral si no hay psyche

        # Combinar según estrategia
        if self.config.strategy == ClusteringStrategy.SPATIAL:
            return spatial_affinity
        elif self.config.strategy == ClusteringStrategy.PSYCHE:
            return psyche_sim
        else:  # HYBRID o ADAPTIVE
            return (
                self.config.spatial_weight * spatial_affinity +
                self.config.psyche_weight * psyche_sim
            )

    # =========================================================================
    # INICIALIZACIÓN DE CLUSTERS
    # =========================================================================

    def initialize_clusters_spatial(
        self,
        cells: list[ConsciousCell],
        grid_size: int = 64
    ) -> list[Cluster]:
        """
        Inicializa clusters dividiendo el espacio en cuadrantes.

        Args:
            cells: Lista de células
            grid_size: Tamaño de la grilla

        Returns:
            Lista de clusters iniciales
        """
        n = self.config.n_clusters

        # Calcular divisiones de la grilla
        sqrt_n = int(np.ceil(np.sqrt(n)))
        cell_width = grid_size / sqrt_n
        cell_height = grid_size / sqrt_n

        # Crear clusters vacíos con centroides
        clusters = []
        for i in range(n):
            row = i // sqrt_n
            col = i % sqrt_n
            centroid = (
                (col + 0.5) * cell_width,
                (row + 0.5) * cell_height
            )
            cluster = Cluster(
                id=i,
                cells=[],
                psyche=None,
                centroid=centroid,
                neighbors=[],
                collective_role=CellRole.MASS
            )
            clusters.append(cluster)

        # Asignar células al cluster más cercano
        for cell in cells:
            best_cluster = min(
                clusters,
                key=lambda c: self.compute_spatial_distance(cell, c.centroid)
            )
            best_cluster.cells.append(cell)
            cell.cluster_id = best_cluster.id

        # Actualizar vecinos
        self._update_neighbors(clusters)

        return clusters

    def initialize_clusters_archetype(
        self,
        cells: list[ConsciousCell]
    ) -> list[Cluster]:
        """
        Inicializa clusters por arquetipo dominante.

        Crea un cluster por cada arquetipo y asigna células
        según su arquetipo dominante.

        Returns:
            Lista de 4 clusters (uno por arquetipo)
        """
        # Crear cluster por arquetipo
        clusters = []
        for archetype in Archetype:
            cluster = Cluster(
                id=archetype.value,
                cells=[],
                psyche=None,
                centroid=(32.0, 32.0),  # Se actualizará
                neighbors=[],
                collective_role=CellRole.MASS
            )
            clusters.append(cluster)

        # Asignar células por arquetipo dominante
        for cell in cells:
            cluster_id = cell.psyche.dominant.value
            clusters[cluster_id].cells.append(cell)
            cell.cluster_id = cluster_id

        # Actualizar centroides
        for cluster in clusters:
            cluster._update_centroid()

        # Actualizar vecinos
        self._update_neighbors(clusters)

        return clusters

    def _update_neighbors(self, clusters: list[Cluster]) -> None:
        """Actualiza lista de clusters vecinos basado en proximidad."""
        for cluster in clusters:
            # Ordenar otros clusters por distancia al centroide
            others = [c for c in clusters if c.id != cluster.id]
            others.sort(key=lambda c: np.sqrt(
                (c.centroid[0] - cluster.centroid[0])**2 +
                (c.centroid[1] - cluster.centroid[1])**2
            ))
            # Los 2 más cercanos son vecinos
            cluster.neighbors = [c.id for c in others[:2]]

    # =========================================================================
    # REASIGNACIÓN DE CLUSTERS
    # =========================================================================

    def should_reassign(
        self,
        clusters: list[Cluster],
        force: bool = False
    ) -> bool:
        """
        Determina si es necesario reasignar células.

        Condiciones:
        1. Ha pasado el intervalo de reasignación
        2. Coherencia promedio está por debajo del umbral
        3. Hay clusters vacíos o demasiado grandes
        """
        if force:
            return True

        # Verificar intervalo
        if self.step_count % self.config.reassign_interval != 0:
            return False

        # Verificar coherencia
        coherences = [
            c.psyche.coherence if c.psyche else 0.0
            for c in clusters
        ]
        avg_coherence = np.mean(coherences) if coherences else 0.0
        if avg_coherence < self.config.coherence_threshold:
            return True

        # Verificar balance de tamaño
        sizes = [len(c.cells) for c in clusters]
        if min(sizes) < self.config.min_cluster_size:
            return True
        if max(sizes) > self.config.max_cluster_size:
            return True

        return False

    def reassign_cells(
        self,
        cells: list[ConsciousCell],
        clusters: list[Cluster]
    ) -> list[Cluster]:
        """
        Reasigna células a clusters optimizando afinidad.

        Algoritmo:
        1. Calcular afinidad de cada célula a cada cluster
        2. Asignar greedily maximizando afinidad total
        3. Balancear tamaños si es necesario

        Returns:
            Clusters actualizados con nuevas asignaciones
        """
        # Vaciar clusters
        for cluster in clusters:
            cluster.cells = []

        # Calcular matriz de afinidades
        affinities = np.zeros((len(cells), len(clusters)))
        for i, cell in enumerate(cells):
            for j, cluster in enumerate(clusters):
                affinities[i, j] = self.compute_affinity(cell, cluster)

        # Asignación greedy
        assigned = set()
        for _ in range(len(cells)):
            # Encontrar la mejor asignación no realizada
            best_score = -1
            best_cell = -1
            best_cluster = -1

            for i in range(len(cells)):
                if i in assigned:
                    continue
                for j in range(len(clusters)):
                    # Penalizar clusters que ya tienen muchas células
                    size_penalty = len(clusters[j].cells) / self.config.max_cluster_size
                    adjusted_score = affinities[i, j] * (1 - 0.5 * size_penalty)

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_cell = i
                        best_cluster = j

            if best_cell >= 0:
                cells[best_cell].cluster_id = best_cluster
                clusters[best_cluster].cells.append(cells[best_cell])
                assigned.add(best_cell)

        # Actualizar centroides
        for cluster in clusters:
            cluster._update_centroid()

        # Actualizar vecinos
        self._update_neighbors(clusters)

        # Guardar en historial
        self.cluster_history.append({
            id(cell): cell.cluster_id for cell in cells
        })

        return clusters

    # =========================================================================
    # BALANCE Y OPTIMIZACIÓN
    # =========================================================================

    def balance_clusters(
        self,
        clusters: list[Cluster]
    ) -> list[Cluster]:
        """
        Balancea tamaños de clusters moviendo células de borde.

        Mueve células con baja afinidad de clusters grandes
        a clusters pequeños vecinos.
        """
        sizes = [len(c.cells) for c in clusters]
        avg_size = np.mean(sizes)

        for cluster in clusters:
            # Si está muy grande, mover algunas células
            while len(cluster.cells) > self.config.max_cluster_size:
                # Encontrar célula con menor afinidad
                if not cluster.cells:
                    break

                worst_cell = min(
                    cluster.cells,
                    key=lambda c: self.compute_affinity(c, cluster)
                )

                # Mover a cluster vecino más pequeño
                neighbors = [
                    c for c in clusters
                    if c.id in cluster.neighbors and len(c.cells) < avg_size
                ]

                if neighbors:
                    target = min(neighbors, key=lambda c: len(c.cells))
                    cluster.cells.remove(worst_cell)
                    target.cells.append(worst_cell)
                    worst_cell.cluster_id = target.id
                else:
                    break  # No hay vecinos disponibles

        # Actualizar centroides
        for cluster in clusters:
            cluster._update_centroid()

        return clusters

    def adapt_weights(
        self,
        clusters: list[Cluster]
    ) -> None:
        """
        Adapta pesos espacial/psíquico según coherencia.

        Si la coherencia es baja, aumentar peso psíquico.
        Si la coherencia es alta, mantener balance.
        """
        if self.config.strategy != ClusteringStrategy.ADAPTIVE:
            return

        coherences = [
            c.psyche.coherence if c.psyche else 0.0
            for c in clusters
        ]
        avg_coherence = np.mean(coherences) if coherences else 0.5

        if avg_coherence < 0.4:
            # Baja coherencia: priorizar similitud psíquica
            self.config.psyche_weight = min(0.8, self.config.psyche_weight + 0.05)
            self.config.spatial_weight = 1.0 - self.config.psyche_weight
        elif avg_coherence > 0.7:
            # Alta coherencia: balance
            self.config.psyche_weight = 0.5
            self.config.spatial_weight = 0.5

    # =========================================================================
    # CLUSTERING DINÁMICO: SPLIT, MERGE, CLEANUP
    # =========================================================================

    def _compute_cluster_coherence(self, cluster: Cluster) -> float:
        """
        Calcula coherencia interna de un cluster.

        Returns:
            Coherencia [0, 1] (1 = muy coherente)
        """
        if len(cluster.cells) < 2:
            return 1.0  # Trivialmente coherente

        # Similitud promedio entre pares de células
        sims = []
        for i, c1 in enumerate(cluster.cells):
            for c2 in cluster.cells[i+1:]:
                sim = F.cosine_similarity(
                    c1.psyche.archetype_state.unsqueeze(0),
                    c2.psyche.archetype_state.unsqueeze(0)
                ).item()
                sims.append((sim + 1) / 2)

        return float(np.mean(sims)) if sims else 1.0

    def _compute_cluster_similarity(
        self,
        cluster1: Cluster,
        cluster2: Cluster
    ) -> float:
        """
        Calcula similitud entre dos clusters.

        Returns:
            Similitud [0, 1] (1 = muy similares)
        """
        if not cluster1.psyche or not cluster2.psyche:
            return 0.5

        sim = F.cosine_similarity(
            cluster1.psyche.aggregate_state.unsqueeze(0),
            cluster2.psyche.aggregate_state.unsqueeze(0)
        ).item()

        return (sim + 1) / 2

    def should_split_cluster(self, cluster: Cluster) -> bool:
        """
        Determina si un cluster debe dividirse.

        Condiciones:
        1. Tiene suficientes células
        2. Coherencia interna es baja
        3. No excedería max_clusters
        """
        if not self.config.dynamic_clusters:
            return False

        # Verificar tamaño mínimo para split
        if len(cluster.cells) < self.config.split_min_size:
            return False

        # Verificar coherencia
        coherence = self._compute_cluster_coherence(cluster)
        if coherence >= self.config.split_coherence_threshold:
            return False

        return True

    def split_cluster(
        self,
        cluster: Cluster,
        all_clusters: list[Cluster]
    ) -> tuple[Cluster, Cluster] | None:
        """
        Divide un cluster heterogéneo en dos.

        Algoritmo:
        1. Identificar las dos "facciones" principales por arquetipo
        2. Asignar células a cada facción
        3. Crear dos nuevos clusters

        Returns:
            Tupla de dos nuevos clusters, o None si no se puede dividir
        """
        if len(all_clusters) >= self.config.max_clusters:
            return None

        if len(cluster.cells) < self.config.split_min_size:
            return None

        # Encontrar los dos arquetipos más comunes
        archetype_counts: dict[Archetype, int] = {}
        for cell in cluster.cells:
            arch = cell.psyche.dominant
            archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

        if len(archetype_counts) < 2:
            return None  # No hay diversidad para dividir

        # Ordenar por frecuencia
        sorted_archetypes = sorted(
            archetype_counts.keys(),
            key=lambda a: archetype_counts[a],
            reverse=True
        )

        primary_arch = sorted_archetypes[0]
        secondary_arch = sorted_archetypes[1]

        # Dividir células
        cells_primary = []
        cells_secondary = []

        for cell in cluster.cells:
            if cell.psyche.dominant == primary_arch:
                cells_primary.append(cell)
            elif cell.psyche.dominant == secondary_arch:
                cells_secondary.append(cell)
            else:
                # Asignar al grupo más cercano
                primary_state = torch.zeros(4)
                primary_state[primary_arch.value] = 1.0
                secondary_state = torch.zeros(4)
                secondary_state[secondary_arch.value] = 1.0

                sim_primary = F.cosine_similarity(
                    cell.psyche.archetype_state.unsqueeze(0),
                    primary_state.unsqueeze(0)
                ).item()
                sim_secondary = F.cosine_similarity(
                    cell.psyche.archetype_state.unsqueeze(0),
                    secondary_state.unsqueeze(0)
                ).item()

                if sim_primary >= sim_secondary:
                    cells_primary.append(cell)
                else:
                    cells_secondary.append(cell)

        # Verificar tamaños mínimos
        if len(cells_primary) < self.config.min_cluster_size:
            return None
        if len(cells_secondary) < self.config.min_cluster_size:
            return None

        # Generar nuevo ID
        max_id = max(c.id for c in all_clusters) if all_clusters else -1
        new_id = max_id + 1

        # Crear nuevos clusters
        cluster1 = Cluster(
            id=cluster.id,  # Mantener ID original
            cells=cells_primary,
            psyche=None,
            centroid=cluster.centroid,
            neighbors=[],
            collective_role=CellRole.MASS
        )
        cluster1._update_centroid()

        cluster2 = Cluster(
            id=new_id,
            cells=cells_secondary,
            psyche=None,
            centroid=cluster.centroid,
            neighbors=[],
            collective_role=CellRole.MASS
        )
        cluster2._update_centroid()

        # Actualizar cluster_id de células
        for cell in cells_primary:
            cell.cluster_id = cluster1.id
        for cell in cells_secondary:
            cell.cluster_id = cluster2.id

        return (cluster1, cluster2)

    def should_merge_clusters(
        self,
        cluster1: Cluster,
        cluster2: Cluster
    ) -> bool:
        """
        Determina si dos clusters deben fusionarse.

        Condiciones:
        1. Son muy similares (alta similitud)
        2. Resultado no excedería max_cluster_size
        3. No reduciría por debajo de min_clusters
        """
        if not self.config.dynamic_clusters:
            return False

        # Verificar tamaño combinado
        combined_size = len(cluster1.cells) + len(cluster2.cells)
        if combined_size > self.config.max_cluster_size:
            return False

        # Verificar similitud
        similarity = self._compute_cluster_similarity(cluster1, cluster2)
        if similarity < self.config.merge_similarity_threshold:
            return False

        return True

    def merge_clusters(
        self,
        cluster1: Cluster,
        cluster2: Cluster
    ) -> Cluster:
        """
        Fusiona dos clusters en uno.

        Args:
            cluster1: Primer cluster (mantiene su ID)
            cluster2: Segundo cluster (será absorbido)

        Returns:
            Cluster fusionado
        """
        # Combinar células
        merged_cells = cluster1.cells + cluster2.cells

        # Actualizar cluster_id de células del cluster2
        for cell in cluster2.cells:
            cell.cluster_id = cluster1.id

        # Crear cluster fusionado
        merged = Cluster(
            id=cluster1.id,
            cells=merged_cells,
            psyche=None,  # Se recalculará
            centroid=cluster1.centroid,
            neighbors=list(set(cluster1.neighbors + cluster2.neighbors)),
            collective_role=CellRole.MASS
        )
        merged._update_centroid()

        return merged

    def cleanup_clusters(
        self,
        clusters: list[Cluster]
    ) -> list[Cluster]:
        """
        Elimina clusters vacíos o demasiado pequeños.

        Reasigna células de clusters pequeños al cluster más cercano.

        Returns:
            Lista de clusters limpia
        """
        valid_clusters = []
        orphan_cells = []

        for cluster in clusters:
            if len(cluster.cells) >= self.config.min_cluster_size:
                valid_clusters.append(cluster)
            else:
                # Recolectar células huérfanas
                orphan_cells.extend(cluster.cells)

        # Reasignar huérfanos al cluster más cercano
        for cell in orphan_cells:
            if not valid_clusters:
                break

            best_cluster = max(
                valid_clusters,
                key=lambda c: self.compute_affinity(cell, c)
            )
            best_cluster.cells.append(cell)
            cell.cluster_id = best_cluster.id

        # Actualizar centroides
        for cluster in valid_clusters:
            cluster._update_centroid()

        return valid_clusters

    def apply_dynamic_clustering(
        self,
        clusters: list[Cluster]
    ) -> list[Cluster]:
        """
        Aplica operaciones de clustering dinámico.

        Orden de operaciones:
        1. Merge clusters muy similares
        2. Split clusters heterogéneos
        3. Cleanup clusters vacíos/pequeños

        Returns:
            Lista de clusters actualizada
        """
        if not self.config.dynamic_clusters:
            return clusters

        # 1. MERGE: Fusionar clusters similares
        merged_any = True
        while merged_any and len(clusters) > self.config.min_clusters:
            merged_any = False
            for i, c1 in enumerate(clusters):
                for j, c2 in enumerate(clusters):
                    if i >= j:
                        continue
                    if self.should_merge_clusters(c1, c2):
                        merged = self.merge_clusters(c1, c2)
                        clusters = [c for c in clusters if c.id not in (c1.id, c2.id)]
                        clusters.append(merged)
                        merged_any = True
                        break
                if merged_any:
                    break

        # 2. SPLIT: Dividir clusters heterogéneos
        new_clusters: list[Cluster] = []
        for cluster in clusters:
            if self.should_split_cluster(cluster) and len(clusters) + len(new_clusters) < self.config.max_clusters:
                result = self.split_cluster(cluster, clusters + new_clusters)
                if result:
                    c1, c2 = result
                    new_clusters.extend([c1, c2])
                else:
                    new_clusters.append(cluster)
            else:
                new_clusters.append(cluster)
        clusters = new_clusters

        # 3. CLEANUP: Eliminar clusters vacíos
        clusters = self.cleanup_clusters(clusters)

        # Actualizar vecinos
        self._update_neighbors(clusters)

        return clusters

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def assign(
        self,
        cells: list[ConsciousCell],
        existing_clusters: list[Cluster] | None = None,
        force_reassign: bool = False
    ) -> list[Cluster]:
        """
        Asigna células a clusters.

        Si no hay clusters existentes, inicializa nuevos.
        Si hay clusters y es momento de reasignar, lo hace.

        Args:
            cells: Lista de células a asignar
            existing_clusters: Clusters existentes (opcional)
            force_reassign: Forzar reasignación

        Returns:
            Lista de clusters con células asignadas
        """
        self.step_count += 1

        if existing_clusters is None:
            # Primera asignación
            if self.config.strategy == ClusteringStrategy.PSYCHE:
                clusters = self.initialize_clusters_archetype(cells)
            else:
                clusters = self.initialize_clusters_spatial(cells)
        else:
            clusters = existing_clusters

            if self.should_reassign(clusters, force_reassign):
                # Adaptar pesos si es estrategia adaptativa
                self.adapt_weights(clusters)

                # Reasignar
                clusters = self.reassign_cells(cells, clusters)

                # Balancear
                clusters = self.balance_clusters(clusters)

                # Aplicar clustering dinámico (split/merge/cleanup)
                if self.config.dynamic_clusters:
                    clusters = self.apply_dynamic_clustering(clusters)

        return clusters

    def get_clustering_quality(
        self,
        clusters: list[Cluster]
    ) -> dict[str, float]:
        """
        Calcula métricas de calidad del clustering.

        Returns:
            Dict con métricas de calidad
        """
        # Coherencia intra-cluster
        coherences = []
        for cluster in clusters:
            if len(cluster.cells) < 2:
                continue

            # Similitud promedio entre pares de células
            sims = []
            for i, c1 in enumerate(cluster.cells):
                for c2 in cluster.cells[i+1:]:
                    sim = F.cosine_similarity(
                        c1.psyche.archetype_state.unsqueeze(0),
                        c2.psyche.archetype_state.unsqueeze(0)
                    ).item()
                    sims.append((sim + 1) / 2)

            if sims:
                coherences.append(float(np.mean(sims)))

        avg_coherence = float(np.mean(coherences)) if coherences else 0.0

        # Separación inter-cluster
        separations: list[float] = []
        for i, cluster_a in enumerate(clusters):
            for cluster_b in clusters[i+1:]:
                if cluster_a.psyche and cluster_b.psyche:
                    sep = F.cosine_similarity(
                        cluster_a.psyche.aggregate_state.unsqueeze(0),
                        cluster_b.psyche.aggregate_state.unsqueeze(0)
                    ).item()
                    separations.append(1 - (sep + 1) / 2)  # Invertir

        avg_separation = float(np.mean(separations)) if separations else 0.0

        # Balance de tamaño
        sizes = [len(c.cells) for c in clusters]
        size_std = float(np.std(sizes)) / (float(np.mean(sizes)) + 1e-6)
        balance = 1.0 / (1.0 + size_std)

        return {
            'intra_coherence': avg_coherence,
            'inter_separation': avg_separation,
            'size_balance': balance,
            'overall_quality': (avg_coherence + avg_separation + balance) / 3
        }

# =============================================================================
# TESTS BÁSICOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: ClusterAssigner")
    print("=" * 60)

    # Crear células de prueba
    print("\n1. Crear células de prueba:")
    cells = []
    for i, archetype in enumerate(Archetype):
        for j in range(15):
            # Posición en cuadrante según arquetipo
            base_x = 16 + (i % 2) * 32
            base_y = 16 + (i // 2) * 32
            cell = ConsciousCell.create_random(
                grid_size=64,
                archetype_bias=archetype
            )
            # Sobrescribir posición para test espacial
            cell.position = (
                base_x + np.random.randint(-10, 10),
                base_y + np.random.randint(-10, 10)
            )
            cells.append(cell)

    print(f"   {len(cells)} células creadas")

    # Test inicialización espacial
    print("\n2. Inicialización espacial:")
    config = ClusteringConfig(
        n_clusters=4,
        strategy=ClusteringStrategy.SPATIAL
    )
    assigner = ClusterAssigner(config)
    clusters = assigner.assign(cells)

    for cluster in clusters:
        print(f"   Cluster {cluster.id}: {len(cluster.cells)} células, "
              f"centroide={cluster.centroid[0]:.1f}, {cluster.centroid[1]:.1f}")

    # Test inicialización por arquetipo
    print("\n3. Inicialización por arquetipo:")
    config2 = ClusteringConfig(
        n_clusters=4,
        strategy=ClusteringStrategy.PSYCHE
    )
    assigner2 = ClusterAssigner(config2)
    clusters2 = assigner2.initialize_clusters_archetype(cells)

    for cluster in clusters2:
        archetype_name = Archetype(cluster.id).name
        print(f"   Cluster {cluster.id} ({archetype_name}): {len(cluster.cells)} células")

    # Test clustering híbrido
    print("\n4. Clustering híbrido:")
    config3 = ClusteringConfig(
        n_clusters=4,
        strategy=ClusteringStrategy.HYBRID,
        spatial_weight=0.5,
        psyche_weight=0.5
    )
    assigner3 = ClusterAssigner(config3)
    clusters3 = assigner3.assign(cells)

    for cluster in clusters3:
        print(f"   Cluster {cluster.id}: {len(cluster.cells)} células")

    # Test reasignación
    print("\n5. Test de reasignación:")
    # Simular pasos para activar reasignación
    for _ in range(15):
        clusters3 = assigner3.assign(cells, clusters3)

    print(f"   Pasos simulados: {assigner3.step_count}")
    print(f"   Reasignaciones en historial: {len(assigner3.cluster_history)}")

    # Test calidad del clustering
    print("\n6. Calidad del clustering:")

    # Primero crear ClusterPsyche para cada cluster
    from bottom_up_integrator import BottomUpIntegrator
    integrator = BottomUpIntegrator()

    for cluster in clusters3:
        if cluster.cells:
            cluster.psyche = integrator.aggregate_cells_to_cluster(cluster.cells)

    quality = assigner3.get_clustering_quality(clusters3)
    print(f"   Coherencia intra-cluster: {quality['intra_coherence']:.3f}")
    print(f"   Separación inter-cluster: {quality['inter_separation']:.3f}")
    print(f"   Balance de tamaño: {quality['size_balance']:.3f}")
    print(f"   Calidad general: {quality['overall_quality']:.3f}")

    # Test estrategia adaptativa
    print("\n7. Estrategia adaptativa:")
    config4 = ClusteringConfig(
        n_clusters=4,
        strategy=ClusteringStrategy.ADAPTIVE
    )
    assigner4 = ClusterAssigner(config4)
    print(f"   Pesos iniciales: spatial={config4.spatial_weight}, psyche={config4.psyche_weight}")

    clusters4 = assigner4.assign(cells)
    for cluster in clusters4:
        if cluster.cells:
            cluster.psyche = integrator.aggregate_cells_to_cluster(cluster.cells)

    assigner4.adapt_weights(clusters4)
    print(f"   Pesos adaptados: spatial={assigner4.config.spatial_weight:.2f}, "
          f"psyche={assigner4.config.psyche_weight:.2f}")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETADOS")
    print("=" * 60)
