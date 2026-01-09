# -*- coding: utf-8 -*-
"""
Validacion Teorica: Entropia de Shannon

Hipotesis: Los ceros zeta producen entropia INTERMEDIA (borde del caos)
- UNIFORM (orden): Entropia BAJA (predecible)
- RANDOM (caos): Entropia ALTA (impredecible)
- ZETA (criticidad): Entropia INTERMEDIA (complejo pero estructurado)

Metricas:
1. Entropia de distribucion arquetipal
2. Entropia espacial (distribucion en grid)
3. Entropia temporal (predictibilidad de transiciones)

Fecha: 2026-01-03
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
from collections import Counter
import scipy.stats as stats

# Frecuencias
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
])

np.random.seed(42)
torch.manual_seed(42)


@dataclass
class EntropyResult:
    """Resultado del analisis de entropia."""
    modulation_type: str
    system_name: str

    # Entropias promedio
    archetype_entropy: float      # H de distribucion arquetipal
    spatial_entropy: float        # H de distribucion espacial
    temporal_entropy: float       # H de transiciones

    # Series temporales
    archetype_entropy_series: List[float]
    spatial_entropy_series: List[float]

    # Estadisticas
    entropy_mean: float
    entropy_std: float
    entropy_range: Tuple[float, float]


def generate_frequencies(freq_type: str, n: int = 15) -> np.ndarray:
    """Genera frecuencias segun el tipo."""
    if freq_type == "ZETA":
        return ZETA_ZEROS[:n]
    elif freq_type == "RANDOM":
        return np.sort(np.random.uniform(10, 70, n))
    elif freq_type == "UNIFORM":
        return np.linspace(14, 65, n)
    else:
        raise ValueError(f"Tipo desconocido: {freq_type}")


def zeta_kernel(t: float, frequencies: np.ndarray, sigma: float = 0.1) -> float:
    """Kernel K_sigma(t)."""
    weights = np.exp(-sigma * np.abs(frequencies))
    oscillations = np.cos(frequencies * t)
    return 2.0 * np.sum(weights * oscillations)


def shannon_entropy(probs: np.ndarray) -> float:
    """Calcula entropia de Shannon: H = -sum(p * log2(p))"""
    probs = np.array(probs).flatten()
    probs = probs[probs > 1e-10]  # Evitar log(0)
    probs = probs / probs.sum()   # Normalizar
    return -np.sum(probs * np.log2(probs))


def discretize_positions(positions: np.ndarray, grid_size: int,
                         n_bins: int = 8) -> np.ndarray:
    """Discretiza posiciones en bins para calcular entropia espacial."""
    bin_size = grid_size / n_bins
    binned = (positions / bin_size).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    return binned


def spatial_entropy(positions: np.ndarray, grid_size: int, n_bins: int = 8) -> float:
    """Calcula entropia de la distribucion espacial."""
    binned = discretize_positions(positions, grid_size, n_bins)

    # Contar celulas por bin
    if len(binned.shape) == 2:
        # 2D positions
        bin_counts = Counter(tuple(b) for b in binned)
    else:
        bin_counts = Counter(binned.tolist())

    counts = np.array(list(bin_counts.values()))
    probs = counts / counts.sum()

    return shannon_entropy(probs)


def transition_entropy(states_history: List[np.ndarray], n_bins: int = 4) -> float:
    """Calcula entropia de transiciones (predictibilidad temporal)."""
    if len(states_history) < 2:
        return 0.0

    # Discretizar estados
    transitions = []
    for i in range(len(states_history) - 1):
        s1 = states_history[i]
        s2 = states_history[i + 1]

        # Discretizar por arquetipo dominante
        d1 = np.argmax(s1.mean(axis=0) if len(s1.shape) > 1 else s1)
        d2 = np.argmax(s2.mean(axis=0) if len(s2.shape) > 1 else s2)

        transitions.append((d1, d2))

    # Contar transiciones
    trans_counts = Counter(transitions)
    counts = np.array(list(trans_counts.values()))
    probs = counts / counts.sum()

    return shannon_entropy(probs)


# =============================================================================
# SISTEMAS SIMPLIFICADOS
# =============================================================================

class EntropyHierarchicalSystem:
    """Sistema jerarquico para analisis de entropia."""

    def __init__(self, n_cells: int = 30, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05
        self.time = 0.0
        self.grid_size = 32

        # Estados: arquetipos + posiciones
        self.archetypes = self._init_archetypes()
        self.positions = np.random.rand(n_cells, 2) * self.grid_size
        self.history = []

    def _init_archetypes(self) -> np.ndarray:
        """Inicializa distribuciones arquetipales."""
        archs = np.random.rand(self.n_cells, 4)
        archs = archs / archs.sum(axis=1, keepdims=True)
        return archs

    def step(self, dt: float = 0.1):
        self.time += dt

        # Modulacion zeta
        mod = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_norm = 0.5 + 0.5 * np.tanh(mod / 5.0)

        # Dinamica arquetipal
        global_arch = self.archetypes.mean(axis=0)

        # Top-down: convergencia modulada
        for i in range(self.n_cells):
            diff = global_arch - self.archetypes[i]
            self.archetypes[i] += 0.2 * mod_norm * diff

        # Lateral: difusion inversa
        for i in range(self.n_cells):
            j = np.random.randint(0, self.n_cells)
            if i != j:
                diff = self.archetypes[j] - self.archetypes[i]
                self.archetypes[i] += 0.1 * (1 - mod_norm) * diff

        # Exploracion dependiente de modulacion
        noise_scale = 0.02 * (1.0 + 0.5 * (1 - mod_norm))
        self.archetypes += np.random.randn(self.n_cells, 4) * noise_scale

        # Normalizar
        self.archetypes = np.clip(self.archetypes, 0.01, None)
        row_sums = self.archetypes.sum(axis=1, keepdims=True)
        self.archetypes = self.archetypes / row_sums

        # Movimiento espacial
        center = self.positions.mean(axis=0)
        for i in range(self.n_cells):
            # Atraccion al centro modulada
            to_center = center - self.positions[i]
            self.positions[i] += 0.05 * mod_norm * to_center
            # Ruido
            self.positions[i] += np.random.randn(2) * 0.5

        self.positions = np.clip(self.positions, 0, self.grid_size)

        # Guardar historia
        self.history.append(self.archetypes.copy())

    def get_archetype_entropy(self) -> float:
        """Entropia de la distribucion arquetipal global."""
        global_arch = self.archetypes.mean(axis=0)
        return shannon_entropy(global_arch)

    def get_spatial_entropy(self) -> float:
        """Entropia de la distribucion espacial."""
        return spatial_entropy(self.positions, self.grid_size)


class EntropyZetaOrganism:
    """ZetaOrganism para analisis de entropia."""

    def __init__(self, n_cells: int = 30, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.grid_size = 32
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05
        self.time = 0.0

        # Estado
        self.positions = np.random.rand(n_cells, 2) * self.grid_size
        self.energies = np.random.rand(n_cells) * 0.5 + 0.1
        self.roles = np.zeros(n_cells)  # 0=Mass, 1=Force
        self.roles[0] = 1  # Un lider

        self.history = []

    def step(self, dt: float = 0.1):
        self.time += dt

        # Modulacion
        mod = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_norm = 0.5 + 0.5 * np.tanh(mod / 5.0)

        # Encontrar lideres
        leaders = np.where(self.roles > 0.5)[0]

        # Movimiento de masas hacia lideres
        for i in range(self.n_cells):
            if self.roles[i] < 0.5:  # Es masa
                force = np.zeros(2)
                for j in leaders:
                    diff = self.positions[j] - self.positions[i]
                    dist = np.linalg.norm(diff) + 0.1

                    # Fuerza con kernel
                    k = zeta_kernel(dist / 5.0, self.frequencies, self.sigma)
                    attraction = self.energies[j] * (0.5 + 0.5 * k) / dist
                    force += attraction * diff / dist

                self.positions[i] += force * self.energies[i] * mod_norm * 0.3

        # Ruido dependiente de modulacion
        noise_scale = 0.3 * (1.0 + 0.5 * (1 - mod_norm))
        self.positions += np.random.randn(self.n_cells, 2) * noise_scale

        # Energia decay
        self.energies *= 0.99
        self.energies += np.random.rand(self.n_cells) * 0.01
        self.energies = np.clip(self.energies, 0.01, 1.0)

        # Limites
        self.positions = np.clip(self.positions, 0, self.grid_size)

        # Historia (guardar distribucion de energia como proxy de arquetipos)
        energy_bins = np.digitize(self.energies, bins=[0.25, 0.5, 0.75, 1.0])
        self.history.append(energy_bins.copy())

    def get_archetype_entropy(self) -> float:
        """Entropia de la distribucion de energia (proxy)."""
        energy_hist, _ = np.histogram(self.energies, bins=4, range=(0, 1))
        if energy_hist.sum() > 0:
            probs = energy_hist / energy_hist.sum()
            return shannon_entropy(probs)
        return 0.0

    def get_spatial_entropy(self) -> float:
        """Entropia espacial."""
        return spatial_entropy(self.positions, self.grid_size)


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_entropy_experiment():
    """Ejecuta experimento de entropia."""

    print("=" * 70)
    print("VALIDACION TEORICA: ENTROPIA DE SHANNON")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Hipotesis: ZETA produce entropia INTERMEDIA (borde del caos)")
    print("  UNIFORM: Entropia BAJA (orden)")
    print("  RANDOM:  Entropia ALTA (caos)")
    print("  ZETA:    Entropia INTERMEDIA (criticidad)")
    print()

    modulation_types = ["ZETA", "RANDOM", "UNIFORM"]
    n_steps = 50  # Reducido para velocidad
    n_trials = 3  # Reducido para velocidad

    systems = [
        ("Hierarchical", EntropyHierarchicalSystem, {"n_cells": 30}),
        ("ZetaOrganism", EntropyZetaOrganism, {"n_cells": 30})
    ]

    all_results = {}

    for sys_name, sys_class, kwargs in systems:
        print(f"\n{'='*60}")
        print(f"SISTEMA: {sys_name}")
        print(f"{'='*60}")

        all_results[sys_name] = {}

        for mod_type in modulation_types:
            print(f"\n  Modulacion: {mod_type}")

            arch_entropies = []
            spatial_entropies = []
            arch_series_all = []
            spatial_series_all = []

            for trial in range(n_trials):
                np.random.seed(42 + trial)

                freqs = generate_frequencies(mod_type)
                system = sys_class(frequencies=freqs, **kwargs)

                arch_series = []
                spatial_series = []

                for step in range(n_steps):
                    system.step()

                    h_arch = system.get_archetype_entropy()
                    h_spatial = system.get_spatial_entropy()

                    arch_series.append(h_arch)
                    spatial_series.append(h_spatial)

                # Promedios de este trial
                arch_entropies.append(np.mean(arch_series[-50:]))  # Segunda mitad
                spatial_entropies.append(np.mean(spatial_series[-50:]))
                arch_series_all.append(arch_series)
                spatial_series_all.append(spatial_series)

            # Calcular entropia temporal
            if hasattr(system, 'history') and len(system.history) > 10:
                h_temporal = transition_entropy(system.history)
            else:
                h_temporal = 0.0

            # Promediar series
            avg_arch_series = np.mean(arch_series_all, axis=0).tolist()
            avg_spatial_series = np.mean(spatial_series_all, axis=0).tolist()

            # Crear resultado
            result = EntropyResult(
                modulation_type=mod_type,
                system_name=sys_name,
                archetype_entropy=np.mean(arch_entropies),
                spatial_entropy=np.mean(spatial_entropies),
                temporal_entropy=h_temporal,
                archetype_entropy_series=avg_arch_series,
                spatial_entropy_series=avg_spatial_series,
                entropy_mean=np.mean(arch_entropies),
                entropy_std=np.std(arch_entropies),
                entropy_range=(min(arch_entropies), max(arch_entropies))
            )

            all_results[sys_name][mod_type] = result

            print(f"    H_arquetipal: {result.archetype_entropy:.4f}")
            print(f"    H_espacial:   {result.spatial_entropy:.4f}")
            print(f"    H_temporal:   {result.temporal_entropy:.4f}")

    # =========================================================================
    # ANALISIS
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALISIS DE RESULTADOS")
    print("=" * 70)

    edge_of_chaos_systems = 0
    total_systems = 0

    for sys_name in all_results:
        print(f"\n{sys_name}:")
        print("-" * 50)

        # Obtener entropias
        h_zeta = all_results[sys_name]["ZETA"].archetype_entropy
        h_random = all_results[sys_name]["RANDOM"].archetype_entropy
        h_uniform = all_results[sys_name]["UNIFORM"].archetype_entropy

        # Ordenar
        entropies = [("ZETA", h_zeta), ("RANDOM", h_random), ("UNIFORM", h_uniform)]
        sorted_h = sorted(entropies, key=lambda x: x[1])

        print("  Ranking de entropia arquetipal:")
        for i, (name, h) in enumerate(sorted_h):
            label = ["(menor)", "(medio)", "(mayor)"][i]
            print(f"    {i+1}. {name:8s}: H = {h:.4f} {label}")

        # Verificar hipotesis: ZETA debe estar en el medio
        zeta_rank = [i for i, (n, _) in enumerate(sorted_h) if n == "ZETA"][0]

        total_systems += 1

        if zeta_rank == 1:  # Posicion media
            print(f"\n  [OK] HIPOTESIS CONFIRMADA: ZETA tiene entropia INTERMEDIA")
            print(f"       ZETA esta entre {sorted_h[0][0]} y {sorted_h[2][0]}")
            edge_of_chaos_systems += 1
        else:
            # Verificar si ZETA esta mas cerca del medio que en un extremo
            h_range = sorted_h[2][1] - sorted_h[0][1]
            h_mid = (sorted_h[0][1] + sorted_h[2][1]) / 2
            zeta_dist_to_mid = abs(h_zeta - h_mid)

            if h_range > 0.01:  # Hay variacion significativa
                relative_pos = (h_zeta - sorted_h[0][1]) / h_range
                if 0.3 < relative_pos < 0.7:
                    print(f"\n  [~] ZETA en zona INTERMEDIA (posicion relativa: {relative_pos:.2f})")
                    edge_of_chaos_systems += 1
                else:
                    print(f"\n  [?] ZETA no esta en posicion intermedia (pos: {relative_pos:.2f})")
            else:
                print(f"\n  [?] Variacion insuficiente entre modulaciones")

    # =========================================================================
    # RESUMEN
    # =========================================================================

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(f"\n  Sistemas donde ZETA tiene entropia intermedia: {edge_of_chaos_systems}/{total_systems}")

    if edge_of_chaos_systems == total_systems:
        print("\n  *** HIPOTESIS DEL BORDE DEL CAOS VALIDADA ***")
        print("  Los ceros de Riemann producen complejidad intermedia")
    elif edge_of_chaos_systems > 0:
        print("\n  ** HIPOTESIS PARCIALMENTE VALIDADA **")
    else:
        print("\n  X HIPOTESIS NO VALIDADA")

    # =========================================================================
    # VISUALIZACION
    # =========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Validacion: Entropia de Shannon\n' +
                 'Hipotesis: ZETA produce entropia intermedia (borde del caos)',
                 fontsize=13, fontweight='bold')

    colors = {"ZETA": "blue", "RANDOM": "red", "UNIFORM": "gray"}

    # Fila 1: Series temporales por sistema
    for idx, sys_name in enumerate(all_results):
        ax = axes[0, idx]

        for mod_type in modulation_types:
            result = all_results[sys_name][mod_type]
            series = result.archetype_entropy_series
            ax.plot(series, color=colors[mod_type], linewidth=2, alpha=0.8,
                   label=f'{mod_type} (H={result.archetype_entropy:.3f})')

        ax.set_title(f'{sys_name}: Entropia Arquetipal')
        ax.set_xlabel('Paso')
        ax.set_ylabel('Entropia (bits)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Fila 1, Col 3: Comparacion de entropias finales
    ax_comp = axes[0, 2]

    x = np.arange(len(all_results))
    width = 0.25

    for i, mod_type in enumerate(modulation_types):
        vals = [all_results[sys][mod_type].archetype_entropy for sys in all_results]
        ax_comp.bar(x + i*width, vals, width, label=mod_type,
                   color=colors[mod_type], alpha=0.8)

    ax_comp.set_ylabel('Entropia (bits)')
    ax_comp.set_title('Comparacion: Entropia Arquetipal')
    ax_comp.set_xticks(x + width)
    ax_comp.set_xticklabels(list(all_results.keys()))
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3, axis='y')

    # Fila 2: Entropia espacial
    for idx, sys_name in enumerate(all_results):
        ax = axes[1, idx]

        for mod_type in modulation_types:
            result = all_results[sys_name][mod_type]
            series = result.spatial_entropy_series
            ax.plot(series, color=colors[mod_type], linewidth=2, alpha=0.8,
                   label=f'{mod_type} (H={result.spatial_entropy:.3f})')

        ax.set_title(f'{sys_name}: Entropia Espacial')
        ax.set_xlabel('Paso')
        ax.set_ylabel('Entropia (bits)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Fila 2, Col 3: Posicion relativa de ZETA
    ax_pos = axes[1, 2]

    for idx, sys_name in enumerate(all_results):
        h_zeta = all_results[sys_name]["ZETA"].archetype_entropy
        h_random = all_results[sys_name]["RANDOM"].archetype_entropy
        h_uniform = all_results[sys_name]["UNIFORM"].archetype_entropy

        h_min = min(h_zeta, h_random, h_uniform)
        h_max = max(h_zeta, h_random, h_uniform)
        h_range = h_max - h_min

        if h_range > 0.001:
            # Normalizar posiciones a [0, 1]
            pos_zeta = (h_zeta - h_min) / h_range
            pos_random = (h_random - h_min) / h_range
            pos_uniform = (h_uniform - h_min) / h_range

            y = idx
            ax_pos.scatter([pos_zeta], [y], c='blue', s=200, marker='o',
                          label='ZETA' if idx == 0 else '', zorder=3)
            ax_pos.scatter([pos_random], [y], c='red', s=150, marker='^',
                          label='RANDOM' if idx == 0 else '', zorder=3)
            ax_pos.scatter([pos_uniform], [y], c='gray', s=150, marker='s',
                          label='UNIFORM' if idx == 0 else '', zorder=3)

            # Linea de referencia
            ax_pos.plot([0, 1], [y, y], 'k-', alpha=0.3, linewidth=2)

    ax_pos.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                  label='Centro (ideal)')
    ax_pos.axvspan(0.3, 0.7, alpha=0.1, color='green', label='Zona intermedia')

    ax_pos.set_xlim(-0.1, 1.1)
    ax_pos.set_yticks(range(len(all_results)))
    ax_pos.set_yticklabels(list(all_results.keys()))
    ax_pos.set_xlabel('Posicion relativa (0=min, 1=max)')
    ax_pos.set_title('Posicion de ZETA en el espectro\nde entropia')
    ax_pos.legend(loc='upper right', fontsize=8)
    ax_pos.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Guardar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'entropy_validation_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nGrafico guardado: {filename}")
    plt.close()

    return all_results


if __name__ == "__main__":
    results = run_entropy_experiment()
