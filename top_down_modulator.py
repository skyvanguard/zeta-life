# -*- coding: utf-8 -*-
"""
TopDownModulator: Modulación de niveles superiores a inferiores.

Implementa el flujo top-down en la consciencia jerárquica:
- Organismo → Clusters (distribución de atención)
- Clusters → Células (señales de modulación)

El organismo influye en sus partes, creando coherencia vertical.

Fecha: 2026-01-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Importar del sistema existente
from zeta_psyche import Archetype

# Importar de módulos nuevos
from micro_psyche import ConsciousCell, MicroPsyche, unbiased_argmax
from cluster import Cluster, ClusterPsyche
from organism_consciousness import OrganismConsciousness


# =============================================================================
# TOP-DOWN MODULATOR
# =============================================================================

class TopDownModulator(nn.Module):
    """
    Modula niveles inferiores desde consciencia superior.

    Flujo:
    1. Organismo → Clusters: distribución de atención
    2. Clusters → Células: señales de modulación

    Permite control ejecutivo y coherencia vertical.
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

        # Red de atención: qué clusters priorizar
        # Input: global_archetype[4] + cluster_archetype[4] = 8
        self.attention_net = nn.Sequential(
            nn.Linear(n_archetypes * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Red de modulación: cómo afectar células
        # Input: global_archetype[4] → modulación[state_dim]
        self.modulation_net = nn.Sequential(
            nn.Linear(n_archetypes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()  # Modulación en [-1, 1]
        )

        # Red de predicción: expectativas para clusters
        # Input: global_archetype[4] + specialization[4] → prediction[4]
        self.prediction_net = nn.Sequential(
            nn.Linear(n_archetypes * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_archetypes),
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
    # NIVEL 2 → NIVEL 1: Organismo modula Clusters
    # =========================================================================

    def compute_cluster_attention(
        self,
        organism: OrganismConsciousness,
        cluster: Cluster
    ) -> float:
        """
        Calcula cuánta atención dar a un cluster.

        Factores:
        - Relevancia del cluster para el estado global
        - Complementariedad (arquetipos opuestos reciben más atención)

        Args:
            organism: Consciencia del organismo
            cluster: Cluster a evaluar

        Returns:
            Atención normalizada [0, 1]
        """
        if not cluster.psyche:
            return 0.1

        # Features para la red (asegurar float32)
        features = torch.cat([
            organism.global_archetype.float(),
            cluster.psyche.aggregate_state.float()
        ])

        # Atención base de la red
        with torch.no_grad():
            base_attention = self.attention_net(features).item()

        # Bonus por complementariedad
        # Si el cluster tiene el arquetipo complementario al dominante, boost
        complement_idx = self._get_complement_idx(organism.dominant_archetype)
        if cluster.psyche.specialization.value == complement_idx:
            base_attention *= 1.3  # 30% más atención a arquetipos complementarios

        # Bonus si el cluster tiene el arquetipo más débil del organismo
        weakest_idx = organism.global_archetype.argmin().item()
        if cluster.psyche.specialization.value == weakest_idx:
            base_attention *= 1.2  # 20% más atención a arquetipos débiles

        return min(1.0, base_attention)

    def _get_complement_idx(self, archetype: Archetype) -> int:
        """Retorna índice del arquetipo complementario."""
        complements = {
            Archetype.PERSONA: 1,   # SOMBRA
            Archetype.SOMBRA: 0,    # PERSONA
            Archetype.ANIMA: 3,     # ANIMUS
            Archetype.ANIMUS: 2,    # ANIMA
        }
        return complements[archetype]

    def distribute_attention(
        self,
        organism: OrganismConsciousness,
        clusters: List[Cluster]
    ) -> Dict[int, float]:
        """
        Distribuye atención del organismo entre clusters.

        Args:
            organism: Consciencia del organismo
            clusters: Lista de clusters

        Returns:
            Dict[cluster_id, attention_weight]
        """
        attention_weights = {}

        for cluster in clusters:
            attention = self.compute_cluster_attention(organism, cluster)
            attention_weights[cluster.id] = attention

        # Normalizar (softmax-like pero manteniendo proporciones)
        total = sum(attention_weights.values())
        if total > 0:
            # Escalar para que el máximo sea ~1.0
            max_att = max(attention_weights.values())
            if max_att > 0:
                attention_weights = {
                    k: v / max_att for k, v in attention_weights.items()
                }

        return attention_weights

    def generate_predictions(
        self,
        organism: OrganismConsciousness,
        clusters: List[Cluster]
    ) -> Dict[int, torch.Tensor]:
        """
        Genera predicciones top-down para cada cluster.

        El organismo "espera" cierto comportamiento de sus partes.
        La discrepancia entre predicción y realidad = sorpresa.

        Args:
            organism: Consciencia del organismo
            clusters: Lista de clusters

        Returns:
            Dict[cluster_id, prediction_tensor[4]]
        """
        predictions = {}

        for cluster in clusters:
            if not cluster.psyche:
                # Predicción neutral
                predictions[cluster.id] = torch.ones(4) / 4
                continue

            # Features para predicción
            # One-hot encoding de especialización
            spec_onehot = torch.zeros(4)
            spec_onehot[cluster.psyche.specialization.value] = 1.0

            features = torch.cat([
                organism.global_archetype.float(),
                spec_onehot.float()
            ])

            # Generar predicción
            with torch.no_grad():
                raw_pred = self.prediction_net(features)
                prediction = F.softmax(raw_pred, dim=0)

            # Mezclar con expectativa de especialización
            # El cluster debería tender a su especialización, modulado por necesidades globales
            target = torch.zeros(4)
            target[cluster.psyche.specialization.value] = 0.6

            # 70% especialización esperada + 30% necesidades globales
            final_pred = 0.7 * target + 0.3 * prediction
            final_pred = F.softmax(final_pred, dim=0)

            predictions[cluster.id] = final_pred

        return predictions

    # =========================================================================
    # NIVEL 1 → NIVEL 0: Clusters modulan Células
    # =========================================================================

    def generate_cell_modulation(
        self,
        organism: OrganismConsciousness,
        cluster: Cluster,
        cluster_attention: float
    ) -> torch.Tensor:
        """
        Genera señal de modulación base para células de un cluster.

        Args:
            organism: Consciencia del organismo
            cluster: Cluster objetivo
            cluster_attention: Atención asignada al cluster

        Returns:
            Señal de modulación [state_dim]
        """
        # Modulación base desde arquetipo global (asegurar float32)
        with torch.no_grad():
            base_signal = self.modulation_net(organism.global_archetype.float())

        # Escalar por atención del cluster
        modulation = base_signal * cluster_attention

        return modulation

    def modulate_cells(
        self,
        cluster: Cluster,
        base_modulation: torch.Tensor,
        cluster_attention: float,
        organism: OrganismConsciousness
    ) -> List[Tuple[ConsciousCell, torch.Tensor, float]]:
        """
        Genera modulaciones específicas para cada célula de un cluster.

        Args:
            cluster: Cluster objetivo
            base_modulation: Modulación base del cluster
            cluster_attention: Atención del cluster
            organism: Consciencia del organismo

        Returns:
            Lista de (cell, modulation_signal, surprise)
        """
        results = []

        for cell in cluster.cells:
            # Modulación específica para esta célula
            cell_mod = base_modulation.clone()

            # Ajustar según alineación con organismo
            alignment = F.cosine_similarity(
                cell.psyche.archetype_state.unsqueeze(0),
                organism.global_archetype.unsqueeze(0)
            ).item()

            # Alta alineación → reforzar estado actual
            # Baja alineación → sugerir cambio hacia global
            if alignment > 0.7:
                cell_mod *= 1.2  # Refuerzo
            elif alignment < 0.3:
                cell_mod *= 0.8  # Suavizar para permitir divergencia

            # Modular por phi_local de la célula
            cell_mod *= cell.psyche.phi_local

            # Calcular sorpresa (discrepancia con expectativa)
            surprise = 1.0 - alignment

            results.append((cell, cell_mod, surprise))

        return results

    def apply_modulation_to_cell(
        self,
        cell: ConsciousCell,
        modulation: torch.Tensor,
        strength: float = 0.05
    ) -> None:
        """
        Aplica señal de modulación a una célula.

        Args:
            cell: Célula a modular
            modulation: Señal de modulación
            strength: Fuerza de la modulación
        """
        # Aplicar al estado físico
        cell.state = cell.state + strength * modulation

        # La modulación también puede afectar energía emocional
        mod_magnitude = modulation.abs().mean().item()
        if mod_magnitude > 0.5:
            # Alta modulación aumenta energía emocional
            cell.psyche.emotional_energy = min(
                1.0,
                cell.psyche.emotional_energy + 0.05
            )

    # =========================================================================
    # MODULACIÓN COMPLETA
    # =========================================================================

    def modulate(
        self,
        organism: OrganismConsciousness,
        clusters: List[Cluster],
        apply_to_cells: bool = True
    ) -> Dict:
        """
        Realiza modulación top-down completa.

        1. Distribuye atención entre clusters
        2. Genera predicciones para cada cluster
        3. Modula células de cada cluster

        Args:
            organism: Consciencia del organismo
            clusters: Lista de clusters
            apply_to_cells: Si aplicar modulación a células

        Returns:
            Dict con resultados de modulación
        """
        results = {
            'attention': {},
            'predictions': {},
            'cell_surprises': [],
            'avg_surprise': 0.0
        }

        # 1. Distribuir atención
        attention = self.distribute_attention(organism, clusters)
        results['attention'] = attention

        # 2. Generar predicciones
        predictions = self.generate_predictions(organism, clusters)
        results['predictions'] = predictions

        # 3. Modular células
        all_surprises = []

        for cluster in clusters:
            cluster_attention = attention.get(cluster.id, 0.5)

            # Generar modulación base
            base_mod = self.generate_cell_modulation(
                organism, cluster, cluster_attention
            )

            # Modular cada célula
            cell_results = self.modulate_cells(
                cluster, base_mod, cluster_attention, organism
            )

            for cell, modulation, surprise in cell_results:
                all_surprises.append(surprise)

                if apply_to_cells:
                    self.apply_modulation_to_cell(cell, modulation)

        results['cell_surprises'] = all_surprises
        results['avg_surprise'] = np.mean(all_surprises) if all_surprises else 0.0

        return results

    def compute_modulation_quality(
        self,
        cells: List[ConsciousCell],
        organism: OrganismConsciousness
    ) -> float:
        """
        Calcula la calidad de la modulación top-down.

        Mide qué tan bien las células responden al organismo.

        Returns:
            Calidad de modulación [0, 1]
        """
        if not cells:
            return 0.0

        # Células con alta energía emocional están más "conectadas"
        responsive = [c for c in cells if c.psyche.emotional_energy > 0.5]

        if not responsive:
            return 0.3  # Baseline bajo

        # Alineación con arquetipo global
        alignments = []
        for cell in responsive:
            alignment = F.cosine_similarity(
                cell.psyche.archetype_state.unsqueeze(0),
                organism.global_archetype.unsqueeze(0)
            ).item()
            alignments.append((alignment + 1) / 2)  # Normalizar a [0, 1]

        return np.mean(alignments)


# =============================================================================
# TESTS BÁSICOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: TopDownModulator")
    print("=" * 60)

    # Crear modulator
    modulator = TopDownModulator(state_dim=32, hidden_dim=64)

    # Crear sistema de prueba
    print("\n1. Crear sistema de prueba:")
    all_cells = []
    clusters = []

    for i, archetype in enumerate(Archetype):
        cells = [
            ConsciousCell.create_random(grid_size=64, archetype_bias=archetype)
            for _ in range(10)
        ]
        all_cells.extend(cells)
        cluster = Cluster.create_from_cells(cluster_id=i, cells=cells)
        clusters.append(cluster)

    print(f"   {len(all_cells)} células en {len(clusters)} clusters")

    # Crear organismo
    organism = OrganismConsciousness.from_clusters(clusters)
    print(f"   Organismo: dominante={organism.dominant_archetype.name}")

    # Test distribución de atención
    print("\n2. Distribución de atención:")
    attention = modulator.distribute_attention(organism, clusters)
    for cluster_id, att in attention.items():
        cluster = clusters[cluster_id]
        print(f"   Cluster {cluster_id} ({cluster.psyche.specialization.name}): "
              f"atención={att:.3f}")

    # Test predicciones
    print("\n3. Predicciones top-down:")
    predictions = modulator.generate_predictions(organism, clusters)
    for cluster_id, pred in predictions.items():
        cluster = clusters[cluster_id]
        dominant_pred = Archetype(unbiased_argmax(pred)).name
        print(f"   Cluster {cluster_id}: predicción dominante={dominant_pred}")

    # Test modulación completa
    print("\n4. Modulación completa:")
    results = modulator.modulate(organism, clusters, apply_to_cells=True)
    print(f"   Sorpresa promedio: {results['avg_surprise']:.3f}")

    # Calidad de modulación
    print("\n5. Calidad de modulación:")
    quality = modulator.compute_modulation_quality(all_cells, organism)
    print(f"   Calidad: {quality:.3f}")

    # Verificar que células fueron afectadas
    print("\n6. Estado de células después de modulación:")
    sample_cell = all_cells[0]
    print(f"   Cell 0: energía_emocional={sample_cell.psyche.emotional_energy:.3f}")

    print("\n" + "=" * 60)
    print("  TESTS COMPLETADOS")
    print("=" * 60)
