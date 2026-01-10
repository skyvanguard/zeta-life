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
from ..core.vertex import Vertex
from ..core.tetrahedral_space import get_tetrahedral_space

# Backwards compatibility alias
Archetype = Vertex

# Importar de módulos nuevos
from .micro_psyche import ConsciousCell, MicroPsyche, unbiased_argmax
from .cluster import Cluster, ClusterPsyche
from .organism_consciousness import OrganismConsciousness


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

        return float(min(1.0, base_attention))

    def _get_complement_idx(self, vertex: Vertex) -> int:
        """Retorna indice del vertice complementario (geometrico)."""
        space = get_tetrahedral_space()
        return space.get_complement(vertex).value

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

    def compute_archetype_goal(
        self,
        organism: OrganismConsciousness
    ) -> torch.Tensor:
        """
        Calcula la distribución arquetipal objetivo que el organismo necesita.

        Estrategia compensatoria: impulsar arquetipos débiles para lograr
        equilibrio e integración.

        Args:
            organism: Consciencia del organismo

        Returns:
            Tensor[4] con la distribución objetivo normalizada
        """
        current = organism.global_archetype

        # Identificar desequilibrios
        # Objetivo: distribución equilibrada pero manteniendo especialización
        ideal_balanced = torch.ones(4) / 4  # [0.25, 0.25, 0.25, 0.25]

        # Cuánto necesitamos compensar
        deficit = ideal_balanced - current  # Positivo = necesita más

        # Crear objetivo compensatorio
        # Más peso a arquetipos con déficit
        compensation_strength = 0.3  # Qué tan agresiva es la compensación
        goal = current + compensation_strength * deficit

        # Normalizar
        goal = F.softmax(goal, dim=0)

        return goal

    def modulate_cell_archetype(
        self,
        cell: ConsciousCell,
        archetype_goal: torch.Tensor,
        strength: float = 0.1
    ) -> None:
        """
        Modula directamente el estado arquetipal de una célula.

        Esta es la clave de la modulación top-down efectiva:
        el organismo puede influir en la distribución de arquetipos
        de sus células.

        Args:
            cell: Célula a modular
            archetype_goal: Distribución arquetipal objetivo
            strength: Fuerza de la modulación (0-1)
        """
        # Calcular dirección de cambio
        current = cell.psyche.archetype_state
        delta = archetype_goal - current

        # Aplicar cambio suave (en espacio de logits para mantener proporciones)
        # Convertir a logits, modificar, reconvertir
        epsilon = 1e-6
        current_logits = torch.log(current + epsilon)
        goal_logits = torch.log(archetype_goal + epsilon)

        # Interpolar en espacio de logits
        new_logits = current_logits + strength * (goal_logits - current_logits)

        # Reconvertir a probabilidades
        new_state = F.softmax(new_logits, dim=0)

        # Actualizar estado (sin aplicar softmax adicional)
        cell.psyche.archetype_state = new_state
        cell.psyche.dominant = Archetype(unbiased_argmax(new_state))

        # Añadir al historial
        cell.psyche.recent_states.append(new_state.clone())

    def compute_adaptive_strength(
        self,
        cell: ConsciousCell,
        organism: OrganismConsciousness,
        base_strength: float = 0.1
    ) -> float:
        """
        Calcula fuerza adaptativa de modulación para una célula.

        Factores:
        - Células desalineadas reciben más modulación
        - Células con baja energía emocional reciben menos
        - Células con alto phi_local reciben más (más integradas)

        Args:
            cell: Célula objetivo
            organism: Consciencia del organismo
            base_strength: Fuerza base

        Returns:
            Fuerza adaptativa [0, 0.3]
        """
        # 1. Factor de alineación (inverso)
        alignment = F.cosine_similarity(
            cell.psyche.archetype_state.unsqueeze(0).float(),
            organism.global_archetype.unsqueeze(0).float()
        ).item()
        alignment_factor = 1.0 + (1.0 - alignment)  # 1.0-2.0

        # 2. Factor de energía emocional
        energy_factor = 0.5 + 0.5 * cell.psyche.emotional_energy  # 0.5-1.0

        # 3. Factor de integración local
        phi_factor = 0.7 + 0.3 * cell.psyche.phi_local  # 0.7-1.0

        # 4. Factor de plasticidad (basado en sorpresa acumulada)
        # Células sorprendidas son más receptivas a modulación top-down
        plasticity = cell.psyche.get_plasticity()  # 0.5-1.5

        # Combinar factores
        adaptive_strength = base_strength * alignment_factor * energy_factor * phi_factor * plasticity

        # Limitar a rango razonable
        return min(0.3, max(0.02, adaptive_strength))

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
        modulation: torch.Tensor = base_signal * cluster_attention

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
        organism: OrganismConsciousness,
        archetype_goal: torch.Tensor,
        strength: float = 0.1
    ) -> None:
        """
        Aplica señal de modulación completa a una célula.

        Incluye:
        1. Modulación del estado físico (cell.state)
        2. Modulación del estado arquetipal (cell.psyche.archetype_state)
        3. Actualización de energía emocional

        Args:
            cell: Célula a modular
            modulation: Señal de modulación física
            organism: Consciencia del organismo
            archetype_goal: Distribución arquetipal objetivo
            strength: Fuerza base de la modulación
        """
        # Calcular fuerza adaptativa
        adaptive_strength = self.compute_adaptive_strength(cell, organism, strength)

        # 1. Aplicar al estado físico
        cell.state = cell.state + adaptive_strength * modulation

        # 2. Aplicar modulación arquetipal (NUEVO - clave para top-down efectivo)
        archetype_strength = adaptive_strength * 0.5  # Más suave para arquetipos
        self.modulate_cell_archetype(cell, archetype_goal, archetype_strength)

        # 3. Actualizar energía emocional
        mod_magnitude = modulation.abs().mean().item()
        if mod_magnitude > 0.3:
            # Modulación aumenta energía emocional (más conexión con organismo)
            cell.psyche.emotional_energy = min(
                1.0,
                cell.psyche.emotional_energy + 0.03 * adaptive_strength
            )

        # 4. Actualizar sorpresa acumulada para plasticidad futura
        cell.psyche.update_accumulated_surprise()

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
            'avg_surprise': 0.0,
            'archetype_goal': None,
            'cells_modulated': 0
        }

        # 0. Calcular objetivo arquetipal (compensatorio)
        archetype_goal = self.compute_archetype_goal(organism)
        results['archetype_goal'] = archetype_goal

        # 1. Distribuir atención
        attention = self.distribute_attention(organism, clusters)
        results['attention'] = attention

        # 2. Generar predicciones
        predictions = self.generate_predictions(organism, clusters)
        results['predictions'] = predictions

        # 3. Modular células
        all_surprises = []
        cells_modulated = 0

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
                    # Usar la nueva firma con modulación arquetipal
                    self.apply_modulation_to_cell(
                        cell, modulation, organism, archetype_goal
                    )
                    cells_modulated += 1

        results['cell_surprises'] = all_surprises
        results['avg_surprise'] = np.mean(all_surprises) if all_surprises else 0.0
        results['cells_modulated'] = cells_modulated

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

        return float(np.mean(alignments))


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
        specialization_name = cluster.psyche.specialization.name if cluster.psyche else "None"
        print(f"   Cluster {cluster_id} ({specialization_name}): "
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
