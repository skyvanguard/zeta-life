# -*- coding: utf-8 -*-
"""
HierarchicalSimulation: Loop principal de consciencia jerárquica.

Orquesta el sistema completo:
1. Inicialización de células, clusters y organismo
2. Ciclo principal: bottom-up → top-down → reasignación
3. Registro de métricas y visualización
4. Experimentos de validación

Fecha: 2026-01-03
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import json
from datetime import datetime

# Importar del sistema existente
from zeta_psyche import Archetype

# Importar módulos de consciencia jerárquica
from micro_psyche import ConsciousCell, MicroPsyche, compute_local_phi, apply_psyche_contagion, unbiased_argmax
from cluster import Cluster, ClusterPsyche, find_cluster_neighbors
from organism_consciousness import (
    OrganismConsciousness, HierarchicalMetrics,
    IndividuationStage
)
from bottom_up_integrator import BottomUpIntegrator
from top_down_modulator import TopDownModulator
from cluster_assigner import ClusterAssigner, ClusteringConfig, ClusteringStrategy


# =============================================================================
# CONFIGURACIÓN DE SIMULACIÓN
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuración para la simulación jerárquica."""
    # Dimensiones
    grid_size: int = 64
    n_cells: int = 80
    n_clusters: int = 4
    state_dim: int = 32
    hidden_dim: int = 64

    # Ciclo de simulación
    n_steps: int = 100
    reassign_interval: int = 10
    metrics_interval: int = 5

    # Pesos de integración
    bottom_up_strength: float = 1.0
    top_down_strength: float = 0.5
    lateral_strength: float = 0.3  # Contagio entre células

    # Clustering
    clustering_strategy: ClusteringStrategy = ClusteringStrategy.HYBRID
    spatial_weight: float = 0.5
    psyche_weight: float = 0.5

    # Dinámica
    energy_decay: float = 0.01
    phi_threshold: float = 0.3  # Umbral para consciencia
    base_energy_recovery: float = 0.005  # Recuperación base independiente de phi

    # Ambiente
    enable_perturbations: bool = False
    perturbation_interval: int = 50  # Más tiempo entre perturbaciones (era 20)
    perturbation_strength: float = 0.2  # Perturbaciones más suaves (era 0.3)
    post_perturbation_contagion: float = 0.5  # Contagio extra post-perturbación


# =============================================================================
# REGISTRO DE MÉTRICAS
# =============================================================================

@dataclass
class SimulationMetrics:
    """Registro de métricas durante la simulación."""
    step: int = 0

    # Métricas de organismo
    phi_global: float = 0.0
    consciousness_index: float = 0.0
    vertical_coherence: float = 0.0
    individuation_stage: int = 0
    dominant_archetype: int = 0

    # Métricas de clusters
    avg_phi_cluster: float = 0.0
    avg_coherence: float = 0.0
    cluster_sizes: List[int] = field(default_factory=list)

    # Métricas de células
    avg_phi_local: float = 0.0
    avg_energy: float = 0.0
    n_fi_cells: int = 0
    n_mi_cells: int = 0

    # Métricas de modulación
    avg_surprise: float = 0.0
    modulation_quality: float = 0.0

    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'step': self.step,
            'phi_global': self.phi_global,
            'consciousness_index': self.consciousness_index,
            'vertical_coherence': self.vertical_coherence,
            'individuation_stage': self.individuation_stage,
            'dominant_archetype': self.dominant_archetype,
            'avg_phi_cluster': self.avg_phi_cluster,
            'avg_coherence': self.avg_coherence,
            'cluster_sizes': self.cluster_sizes,
            'avg_phi_local': self.avg_phi_local,
            'avg_energy': self.avg_energy,
            'n_fi_cells': self.n_fi_cells,
            'n_mi_cells': self.n_mi_cells,
            'avg_surprise': self.avg_surprise,
            'modulation_quality': self.modulation_quality
        }


# =============================================================================
# SIMULACIÓN JERÁRQUICA
# =============================================================================

class HierarchicalSimulation:
    """
    Simulación principal de consciencia jerárquica.

    Orquesta:
    - Nivel 0: Células (MicroPsyche)
    - Nivel 1: Clusters (ClusterPsyche)
    - Nivel 2: Organismo (OrganismConsciousness)

    Flujos:
    - Bottom-up: Células → Clusters → Organismo
    - Top-down: Organismo → Clusters → Células
    - Lateral: Contagio psíquico entre células
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()

        # Componentes del sistema
        self.cells: List[ConsciousCell] = []
        self.clusters: List[Cluster] = []
        self.organism: Optional[OrganismConsciousness] = None

        # Integradores
        self.integrator = BottomUpIntegrator(
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim
        )
        self.modulator = TopDownModulator(
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim
        )

        # Asignador de clusters
        cluster_config = ClusteringConfig(
            n_clusters=self.config.n_clusters,
            strategy=self.config.clustering_strategy,
            spatial_weight=self.config.spatial_weight,
            psyche_weight=self.config.psyche_weight,
            reassign_interval=self.config.reassign_interval
        )
        self.assigner = ClusterAssigner(cluster_config)

        # Estado de simulación
        self.step_count = 0
        self.metrics_history: List[SimulationMetrics] = []
        self.is_initialized = False

        # Callbacks opcionales
        self.on_step_callbacks: List[Callable] = []

    # =========================================================================
    # INICIALIZACIÓN
    # =========================================================================

    def initialize(
        self,
        archetype_distribution: Optional[Dict[Archetype, float]] = None
    ) -> None:
        """
        Inicializa el sistema completo.

        Args:
            archetype_distribution: Distribución inicial de arquetipos
                                   (default: uniforme)
        """
        # Distribución por defecto: uniforme
        if archetype_distribution is None:
            archetype_distribution = {a: 0.25 for a in Archetype}

        # Crear células
        self.cells = self._create_cells(archetype_distribution)

        # Asignar clusters iniciales
        self.clusters = self.assigner.assign(self.cells)

        # Integrar para crear consciencia inicial
        self.clusters, self.organism = self.integrator.integrate(
            self.cells, self.clusters
        )

        self.is_initialized = True
        self.step_count = 0
        self.metrics_history = []

        # Registrar métricas iniciales
        self._record_metrics()

    def _create_cells(
        self,
        distribution: Dict[Archetype, float]
    ) -> List[ConsciousCell]:
        """Crea células según distribución de arquetipos."""
        cells = []

        # Normalizar distribución
        total = sum(distribution.values())
        normalized = {k: v/total for k, v in distribution.items()}

        # Calcular cantidad por arquetipo
        counts = {}
        remaining = self.config.n_cells
        for archetype in Archetype:
            count = int(normalized.get(archetype, 0.25) * self.config.n_cells)
            counts[archetype] = count
            remaining -= count

        # Distribuir resto
        for archetype in Archetype:
            if remaining <= 0:
                break
            counts[archetype] += 1
            remaining -= 1

        # Crear células
        for archetype, count in counts.items():
            for _ in range(count):
                cell = ConsciousCell.create_random(
                    grid_size=self.config.grid_size,
                    archetype_bias=archetype
                )
                cells.append(cell)

        return cells

    # =========================================================================
    # CICLO PRINCIPAL
    # =========================================================================

    def step(self) -> SimulationMetrics:
        """
        Ejecuta un paso de simulación.

        Flujo:
        1. Aplicar dinámica lateral (contagio)
        2. Integración bottom-up
        3. Modulación top-down
        4. Actualizar energía y estados
        5. Potencial reasignación de clusters

        Returns:
            Métricas del paso actual
        """
        if not self.is_initialized:
            raise RuntimeError("Simulación no inicializada. Llamar initialize() primero.")

        self.step_count += 1

        # 1. Dinámica lateral: contagio psíquico entre células
        self._apply_lateral_dynamics()

        # 2. Integración bottom-up
        self.clusters, self.organism = self.integrator.integrate(
            self.cells, self.clusters, self.organism
        )

        # 3. Modulación top-down
        mod_results = self.modulator.modulate(
            self.organism,
            self.clusters,
            apply_to_cells=True
        )

        # 4. Actualizar energía y estados de células
        self._update_cell_dynamics()

        # 5. Potencial reasignación de clusters
        self.clusters = self.assigner.assign(self.cells, self.clusters)

        # 6. Perturbaciones opcionales
        if self.config.enable_perturbations:
            if self.step_count % self.config.perturbation_interval == 0:
                self._apply_perturbation()

        # Registrar métricas
        metrics = self._record_metrics(mod_results)

        # Ejecutar callbacks
        for callback in self.on_step_callbacks:
            callback(self, metrics)

        return metrics

    def run(
        self,
        n_steps: Optional[int] = None,
        verbose: bool = True
    ) -> List[SimulationMetrics]:
        """
        Ejecuta múltiples pasos de simulación.

        Args:
            n_steps: Número de pasos (default: config.n_steps)
            verbose: Mostrar progreso

        Returns:
            Lista de métricas por paso
        """
        n_steps = n_steps or self.config.n_steps

        if verbose:
            print(f"Ejecutando {n_steps} pasos de simulación...")

        for i in range(n_steps):
            metrics = self.step()

            if verbose and (i + 1) % 10 == 0:
                print(f"  Paso {i+1}/{n_steps}: "
                      f"φ_global={metrics.phi_global:.3f}, "
                      f"consciousness={metrics.consciousness_index:.3f}, "
                      f"stage={IndividuationStage(metrics.individuation_stage).name}")

        if verbose:
            print("Simulación completada.")

        return self.metrics_history

    # =========================================================================
    # DINÁMICA INTERNA
    # =========================================================================

    def _apply_lateral_dynamics(self) -> None:
        """Aplica contagio psíquico entre células vecinas."""
        if self.config.lateral_strength <= 0:
            return

        # Para cada célula, encontrar vecinos y aplicar contagio
        for cluster in self.clusters:
            if len(cluster.cells) < 2:
                continue

            for cell in cluster.cells:
                # Encontrar vecinos en el mismo cluster
                neighbors = [
                    c for c in cluster.cells
                    if c is not cell and cell.distance_to(c) < 15.0
                ]

                if neighbors:
                    apply_psyche_contagion(
                        cell,
                        neighbors,
                        contagion_rate=self.config.lateral_strength
                    )

    def _update_cell_dynamics(self) -> None:
        """Actualiza energía y phi local de células."""
        for cell in self.cells:
            # Decay de energía
            cell.energy = max(0.1, cell.energy - self.config.energy_decay)

            # Encontrar vecinos para phi local
            neighbors = [
                c for c in self.cells
                if c is not cell and cell.distance_to(c) < 20.0
            ]

            # Actualizar phi local
            cell.psyche.phi_local = compute_local_phi(cell, neighbors)

            # Recuperación base independiente de phi (permite recuperarse de perturbaciones)
            cell.energy = min(1.0, cell.energy + self.config.base_energy_recovery)

            # Bonus adicional si phi es alto (coherencia con vecinos)
            if cell.psyche.phi_local > self.config.phi_threshold:
                cell.energy = min(1.0, cell.energy + 0.015)

    def _apply_perturbation(self) -> None:
        """Aplica perturbación aleatoria al sistema."""
        n_affected = max(1, int(len(self.cells) * 0.2))
        affected = list(np.random.choice(self.cells, n_affected, replace=False))

        for cell in affected:
            # Perturbar estado arquetípico (más suave con strength=0.2)
            noise = torch.randn(4) * self.config.perturbation_strength
            cell.psyche.archetype_state = F.softmax(
                cell.psyche.archetype_state + noise, dim=0
            )

            # Actualizar dominante
            cell.psyche.dominant = Archetype(
                unbiased_argmax(cell.psyche.archetype_state)
            )

            # Reducir energía (menos agresivo: 0.85 en lugar de 0.8)
            cell.energy *= 0.85

        # Re-sincronización post-perturbación: contagio extra para células afectadas
        affected_ids = {id(c) for c in affected}
        for cell in affected:
            neighbors = [
                c for c in self.cells
                if id(c) not in affected_ids and cell.distance_to(c) < 25.0
            ]
            if neighbors:
                # Contagio más fuerte desde vecinos NO afectados
                apply_psyche_contagion(
                    cell,
                    neighbors,
                    contagion_rate=self.config.post_perturbation_contagion
                )

    # =========================================================================
    # MÉTRICAS Y REGISTRO
    # =========================================================================

    def _record_metrics(
        self,
        mod_results: Optional[Dict] = None
    ) -> SimulationMetrics:
        """Registra métricas del estado actual."""
        metrics = SimulationMetrics(step=self.step_count)

        # Métricas de organismo
        if self.organism:
            metrics.phi_global = self.organism.phi_global
            metrics.consciousness_index = self.organism.consciousness_index.compute_total()
            metrics.vertical_coherence = self.organism.vertical_coherence
            metrics.individuation_stage = self.organism.individuation_stage.value
            metrics.dominant_archetype = self.organism.dominant_archetype.value

        # Métricas de clusters
        if self.clusters:
            phi_clusters = [
                c.psyche.phi_cluster if c.psyche else 0.0
                for c in self.clusters
            ]
            coherences = [
                c.psyche.coherence if c.psyche else 0.0
                for c in self.clusters
            ]

            metrics.avg_phi_cluster = np.mean(phi_clusters)
            metrics.avg_coherence = np.mean(coherences)
            metrics.cluster_sizes = [len(c.cells) for c in self.clusters]

        # Métricas de células
        if self.cells:
            metrics.avg_phi_local = np.mean([c.psyche.phi_local for c in self.cells])
            metrics.avg_energy = np.mean([c.energy for c in self.cells])
            metrics.n_fi_cells = sum(1 for c in self.cells if c.is_fi)
            metrics.n_mi_cells = sum(1 for c in self.cells if c.is_mass)

        # Métricas de modulación
        if mod_results:
            metrics.avg_surprise = mod_results.get('avg_surprise', 0.0)
            metrics.modulation_quality = self.modulator.compute_modulation_quality(
                self.cells, self.organism
            )

        self.metrics_history.append(metrics)
        return metrics

    def get_summary(self) -> Dict:
        """Retorna resumen de la simulación."""
        if not self.metrics_history:
            return {}

        final = self.metrics_history[-1]
        initial = self.metrics_history[0]

        return {
            'total_steps': self.step_count,
            'final_phi_global': final.phi_global,
            'final_consciousness': final.consciousness_index,
            'final_stage': IndividuationStage(final.individuation_stage).name,
            'phi_improvement': final.phi_global - initial.phi_global,
            'consciousness_improvement': final.consciousness_index - initial.consciousness_index,
            'avg_coherence': np.mean([m.avg_coherence for m in self.metrics_history]),
            'avg_modulation_quality': np.mean([m.modulation_quality for m in self.metrics_history]),
        }

    # =========================================================================
    # VISUALIZACIÓN
    # =========================================================================

    def plot_metrics(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Grafica evolución de métricas."""
        if not self.metrics_history:
            print("No hay métricas para graficar.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = [m.step for m in self.metrics_history]

        # 1. Phi y consciencia global
        ax1 = axes[0, 0]
        ax1.plot(steps, [m.phi_global for m in self.metrics_history],
                 'b-', label='φ global', linewidth=2)
        ax1.plot(steps, [m.consciousness_index for m in self.metrics_history],
                 'r--', label='Consciencia', linewidth=2)
        ax1.set_xlabel('Paso')
        ax1.set_ylabel('Valor')
        ax1.set_title('Evolución de Consciencia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Coherencia vertical y de clusters
        ax2 = axes[0, 1]
        ax2.plot(steps, [m.vertical_coherence for m in self.metrics_history],
                 'g-', label='Coherencia vertical', linewidth=2)
        ax2.plot(steps, [m.avg_coherence for m in self.metrics_history],
                 'm--', label='Coherencia clusters', linewidth=2)
        ax2.set_xlabel('Paso')
        ax2.set_ylabel('Coherencia')
        ax2.set_title('Coherencia Jerárquica')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Phi por nivel
        ax3 = axes[1, 0]
        ax3.plot(steps, [m.phi_global for m in self.metrics_history],
                 'b-', label='φ organismo', linewidth=2)
        ax3.plot(steps, [m.avg_phi_cluster for m in self.metrics_history],
                 'g--', label='φ clusters (prom)', linewidth=2)
        ax3.plot(steps, [m.avg_phi_local for m in self.metrics_history],
                 'r:', label='φ células (prom)', linewidth=2)
        ax3.set_xlabel('Paso')
        ax3.set_ylabel('φ (Phi)')
        ax3.set_title('Información Integrada por Nivel')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Modulación y sorpresa
        ax4 = axes[1, 1]
        ax4.plot(steps, [m.avg_surprise for m in self.metrics_history],
                 'orange', label='Sorpresa promedio', linewidth=2)
        ax4.plot(steps, [m.modulation_quality for m in self.metrics_history],
                 'purple', label='Calidad modulación', linewidth=2)
        ax4.set_xlabel('Paso')
        ax4.set_ylabel('Valor')
        ax4.set_title('Dinámica Top-Down')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_archetype_distribution(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Grafica distribución de arquetipos en el sistema."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        colors = ['red', 'purple', 'blue', 'orange']
        archetype_names = [a.name for a in Archetype]

        # 1. Distribución en células
        ax1 = axes[0]
        cell_counts = [0, 0, 0, 0]
        for cell in self.cells:
            cell_counts[cell.psyche.dominant.value] += 1
        ax1.bar(archetype_names, cell_counts, color=colors)
        ax1.set_title('Células por Arquetipo')
        ax1.set_ylabel('Cantidad')

        # 2. Distribución en clusters
        ax2 = axes[1]
        if self.clusters:
            cluster_specs = [0, 0, 0, 0]
            for cluster in self.clusters:
                if cluster.psyche:
                    cluster_specs[cluster.psyche.specialization.value] += 1
            ax2.bar(archetype_names, cluster_specs, color=colors)
        ax2.set_title('Clusters por Especialización')
        ax2.set_ylabel('Cantidad')

        # 3. Estado global del organismo
        ax3 = axes[2]
        if self.organism:
            global_state = self.organism.global_archetype.detach().numpy()
            ax3.bar(archetype_names, global_state, color=colors)
            ax3.axhline(y=0.25, color='gray', linestyle='--',
                       label='Equilibrio', alpha=0.5)
        ax3.set_title('Estado Global del Organismo')
        ax3.set_ylabel('Activación')
        ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_spatial_distribution(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Grafica distribución espacial de células y clusters."""
        fig, ax = plt.subplots(figsize=(10, 10))

        colors = ['red', 'purple', 'blue', 'orange']

        # Dibujar células
        for cell in self.cells:
            color = colors[cell.psyche.dominant.value]
            size = 30 + cell.energy * 50
            alpha = 0.3 + cell.psyche.phi_local * 0.7
            ax.scatter(cell.position[0], cell.position[1],
                      c=color, s=size, alpha=alpha)

        # Dibujar centroides de clusters
        for cluster in self.clusters:
            if cluster.psyche:
                color = colors[cluster.psyche.specialization.value]
                ax.scatter(cluster.centroid[0], cluster.centroid[1],
                          c=color, s=300, marker='*',
                          edgecolors='black', linewidths=2)
                ax.annotate(f'C{cluster.id}',
                           (cluster.centroid[0], cluster.centroid[1]),
                           fontsize=12, ha='center', va='bottom')

        ax.set_xlim(0, self.config.grid_size)
        ax.set_ylim(0, self.config.grid_size)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Distribución Espacial\n(Tamaño=energía, Opacidad=φ_local)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=c, markersize=10, label=n)
            for c, n in zip(colors, [a.name for a in Archetype])
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    # =========================================================================
    # PERSISTENCIA
    # =========================================================================

    def save_metrics(self, filepath: str) -> None:
        """Guarda métricas en archivo JSON."""
        data = {
            'config': {
                'grid_size': self.config.grid_size,
                'n_cells': self.config.n_cells,
                'n_clusters': self.config.n_clusters,
                'n_steps': self.config.n_steps
            },
            'summary': self.get_summary(),
            'metrics': [m.to_dict() for m in self.metrics_history]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Métricas guardadas en: {filepath}")


# =============================================================================
# EXPERIMENTOS DE VALIDACIÓN
# =============================================================================

def run_emergence_experiment(
    n_steps: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Experimento: Emergencia de consciencia desde células random.

    Hipótesis: El sistema debería mostrar aumento progresivo de:
    - φ_global (información integrada)
    - Consciencia (índice compuesto)
    - Coherencia vertical
    """
    config = SimulationConfig(
        n_cells=80,
        n_clusters=4,
        n_steps=n_steps,
        bottom_up_strength=1.0,
        top_down_strength=0.5,
        lateral_strength=0.3
    )

    sim = HierarchicalSimulation(config)
    sim.initialize()

    if verbose:
        print("=" * 60)
        print("  EXPERIMENTO: Emergencia de Consciencia")
        print("=" * 60)

    sim.run(n_steps, verbose=verbose)

    summary = sim.get_summary()

    if verbose:
        print("\nResultados:")
        print(f"  φ_global: {summary['final_phi_global']:.3f} "
              f"(mejora: {summary['phi_improvement']:+.3f})")
        print(f"  Consciencia: {summary['final_consciousness']:.3f} "
              f"(mejora: {summary['consciousness_improvement']:+.3f})")
        print(f"  Etapa final: {summary['final_stage']}")

    return {
        'simulation': sim,
        'summary': summary
    }


def run_perturbation_experiment(
    n_steps: int = 200,
    verbose: bool = True
) -> Dict:
    """
    Experimento: Resiliencia ante perturbaciones.

    Hipótesis: El sistema debería recuperarse después de perturbaciones.
    """
    # Usar defaults mejorados: interval=50, strength=0.2
    config = SimulationConfig(
        n_cells=80,
        n_clusters=4,
        n_steps=n_steps,
        enable_perturbations=True
        # Usa defaults: perturbation_interval=50, perturbation_strength=0.2
    )

    sim = HierarchicalSimulation(config)
    sim.initialize()

    if verbose:
        print("=" * 60)
        print("  EXPERIMENTO: Resiliencia ante Perturbaciones")
        print("=" * 60)
        print(f"  Intervalo: {config.perturbation_interval}, Fuerza: {config.perturbation_strength}")

    sim.run(n_steps, verbose=verbose)

    # Analizar recuperación
    metrics = sim.metrics_history

    # Calcular pasos de perturbación dinámicamente
    interval = config.perturbation_interval
    perturbation_steps = [i for i in range(interval, n_steps, interval)]
    recoveries = []

    for p_step in perturbation_steps:
        if p_step >= len(metrics):
            continue

        # Valor en perturbación y 20 pasos después (más tiempo para recuperar)
        at_perturbation = metrics[p_step].phi_global
        recovery_window = min(20, interval - 5)  # Ventana de recuperación
        if p_step + recovery_window < len(metrics):
            after_recovery = metrics[p_step + recovery_window].phi_global
            recovery = after_recovery - at_perturbation
            recoveries.append(recovery)

    avg_recovery = np.mean(recoveries) if recoveries else 0.0

    if verbose:
        print(f"\nRecuperación promedio: {avg_recovery:+.3f}")

    return {
        'simulation': sim,
        'avg_recovery': avg_recovery
    }


def run_archetype_bias_experiment(
    dominant_archetype: Archetype = Archetype.PERSONA,
    bias_strength: float = 0.5,
    n_steps: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Experimento: Efecto del sesgo arquetípico inicial.

    Hipótesis: Un sesgo inicial hacia un arquetipo debería:
    - Reflejarse en el arquetipo dominante del organismo
    - Producir menor coherencia que distribución uniforme
    """
    config = SimulationConfig(
        n_cells=80,
        n_clusters=4,
        n_steps=n_steps
    )

    sim = HierarchicalSimulation(config)

    # Distribución sesgada
    distribution = {a: 0.1 for a in Archetype}
    distribution[dominant_archetype] = bias_strength + 0.1

    # Normalizar
    total = sum(distribution.values())
    distribution = {k: v/total for k, v in distribution.items()}

    sim.initialize(archetype_distribution=distribution)

    if verbose:
        print("=" * 60)
        print(f"  EXPERIMENTO: Sesgo hacia {dominant_archetype.name}")
        print("=" * 60)
        print(f"  Distribución: {distribution}")

    sim.run(n_steps, verbose=verbose)

    summary = sim.get_summary()

    # Verificar si el dominante coincide
    final_dominant = Archetype(sim.metrics_history[-1].dominant_archetype)
    matches = final_dominant == dominant_archetype

    if verbose:
        print(f"\nArquetipo dominante final: {final_dominant.name}")
        print(f"Coincide con sesgo: {matches}")

    return {
        'simulation': sim,
        'bias_matches_dominant': matches,
        'final_dominant': final_dominant
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  HIERARCHICAL CONSCIOUSNESS SIMULATION")
    print("=" * 60)

    # Experimento básico de emergencia
    result = run_emergence_experiment(n_steps=50, verbose=True)
    sim = result['simulation']

    # Visualizar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("\nGenerando visualizaciones...")
    sim.plot_metrics(
        save_path=f"hierarchical_consciousness_{timestamp}.png",
        show=False
    )

    sim.plot_archetype_distribution(
        save_path=f"archetype_distribution_{timestamp}.png",
        show=False
    )

    sim.plot_spatial_distribution(
        save_path=f"spatial_distribution_{timestamp}.png",
        show=False
    )

    # Guardar métricas
    sim.save_metrics(f"hierarchical_metrics_{timestamp}.json")

    print("\n" + "=" * 60)
    print("  SIMULACIÓN COMPLETADA")
    print("=" * 60)
