# -*- coding: utf-8 -*-
"""
Validacion Teorica: Dimension de Correlacion del Atractor

Hipotesis: ZETA produce atractores de dimension INTERMEDIA
- UNIFORM (orden): Dimension BAJA (atractor simple, ciclo limite)
- RANDOM (caos): Dimension ALTA (atractor extrano)
- ZETA (criticidad): Dimension INTERMEDIA (complejidad optima)

Metodo: Algoritmo de Grassberger-Procaccia
D2 = lim(r->0) log(C(r)) / log(r)

Donde C(r) = fraccion de pares de puntos a distancia < r

Fecha: 2026-01-03
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress

# Frecuencias
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
])

np.random.seed(42)


@dataclass
class CorrelationDimensionResult:
    """Resultado del analisis de dimension de correlacion."""
    modulation_type: str
    system_name: str
    correlation_dimension: float
    dimension_std: float
    scaling_range: Tuple[float, float]
    r_values: List[float]
    c_values: List[float]
    embedding_dim: int


def generate_frequencies(freq_type: str, n: int = 15) -> np.ndarray:
    """Genera frecuencias segun el tipo."""
    if freq_type == "ZETA":
        return ZETA_ZEROS[:n]
    elif freq_type == "RANDOM":
        return np.sort(np.random.uniform(10, 70, n))
    elif freq_type == "UNIFORM":
        return np.linspace(14, 65, n)
    else:
        raise ValueError(f"Unknown type: {freq_type}")


def zeta_kernel(t: float, frequencies: np.ndarray, sigma: float = 0.1) -> float:
    """Kernel K_sigma(t)."""
    weights = np.exp(-sigma * np.abs(frequencies))
    oscillations = np.cos(frequencies * t)
    return 2.0 * np.sum(weights * oscillations)


def time_delay_embedding(time_series: np.ndarray, embedding_dim: int,
                         delay: int = 1) -> np.ndarray:
    """
    Reconstruccion del atractor usando embedding de Takens.

    Args:
        time_series: Serie temporal 1D o multidimensional
        embedding_dim: Dimension del embedding
        delay: Retardo temporal

    Returns:
        Matriz de puntos en el espacio de embedding
    """
    if len(time_series.shape) == 1:
        time_series = time_series.reshape(-1, 1)

    n_samples, n_features = time_series.shape
    n_points = n_samples - (embedding_dim - 1) * delay

    if n_points <= 0:
        raise ValueError("Serie muy corta para el embedding")

    embedded = np.zeros((n_points, embedding_dim * n_features))

    for i in range(embedding_dim):
        start = i * delay
        end = start + n_points
        embedded[:, i*n_features:(i+1)*n_features] = time_series[start:end]

    return embedded


def correlation_sum(points: np.ndarray, r: float) -> float:
    """
    Calcula la suma de correlacion C(r).
    C(r) = (2 / N(N-1)) * sum_{i<j} H(r - |x_i - x_j|)
    """
    n = len(points)
    if n < 2:
        return 0.0

    # Calcular todas las distancias
    distances = pdist(points)

    # Contar pares dentro de radio r
    count = np.sum(distances < r)

    # Normalizar
    total_pairs = n * (n - 1) / 2
    return count / total_pairs


def estimate_correlation_dimension(points: np.ndarray,
                                    r_min: Optional[float] = None,
                                    r_max: Optional[float] = None,
                                    n_r: int = 20) -> Tuple[float, float, List, List]:
    """
    Estima la dimension de correlacion D2.

    Returns:
        (D2, std_error, r_values, C_values)
    """
    # Calcular rango de r
    distances = pdist(points)

    if len(distances) == 0:
        return 0.0, 0.0, [], []

    if r_min is None:
        r_min = np.percentile(distances, 1)
    if r_max is None:
        r_max = np.percentile(distances, 50)

    if r_min <= 0:
        r_min = 1e-6
    if r_max <= r_min:
        r_max = r_min * 10

    # Valores de r en escala logaritmica
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_r)

    # Calcular C(r) para cada r
    c_values = []
    for r in r_values:
        c = correlation_sum(points, r)
        c_values.append(c)

    c_values = np.array(c_values)

    # Filtrar valores validos para regresion log-log
    valid = (c_values > 1e-10) & (r_values > 0)

    if np.sum(valid) < 3:
        return 0.0, 0.0, r_values.tolist(), c_values.tolist()

    log_r = np.log(r_values[valid])
    log_c = np.log(c_values[valid])

    # Regresion lineal para estimar pendiente (D2)
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_c)

    return slope, std_err, r_values.tolist(), c_values.tolist()


# =============================================================================
# SISTEMAS SIMPLIFICADOS
# =============================================================================

class CorrelationHierarchicalSystem:
    """Sistema jerarquico para analisis de dimension."""

    def __init__(self, n_cells: int = 25, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05
        self.time = 0.0

        # Estado
        self.archetypes = np.random.rand(n_cells, 4)
        self.archetypes = self.archetypes / self.archetypes.sum(axis=1, keepdims=True)

        self.trajectory = []

    def step(self, dt: float = 0.1):
        self.time += dt

        # Modulacion zeta
        mod = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_norm = 0.5 + 0.5 * np.tanh(mod / 5.0)

        # Global archetype
        global_arch = self.archetypes.mean(axis=0)

        # Top-down
        for i in range(self.n_cells):
            diff = global_arch - self.archetypes[i]
            self.archetypes[i] += 0.2 * mod_norm * diff

        # Lateral
        for i in range(self.n_cells):
            j = np.random.randint(0, self.n_cells)
            if i != j:
                diff = self.archetypes[j] - self.archetypes[i]
                self.archetypes[i] += 0.1 * (1 - mod_norm) * diff

        # Ruido controlado
        noise = np.random.randn(self.n_cells, 4) * 0.02 * (1 + 0.5 * (1 - mod_norm))
        self.archetypes += noise

        # Normalizar
        self.archetypes = np.clip(self.archetypes, 0.01, None)
        self.archetypes = self.archetypes / self.archetypes.sum(axis=1, keepdims=True)

        # Guardar estado global
        self.trajectory.append(global_arch.copy())

    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)


class CorrelationZetaOrganism:
    """ZetaOrganism para analisis de dimension."""

    def __init__(self, n_cells: int = 25, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.grid_size = 32
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05
        self.time = 0.0

        # Estado
        self.positions = np.random.rand(n_cells, 2) * self.grid_size
        self.energies = np.random.rand(n_cells) * 0.5 + 0.1
        self.roles = np.zeros(n_cells)
        self.roles[0] = 1

        self.trajectory = []

    def step(self, dt: float = 0.1):
        self.time += dt

        mod = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_norm = 0.5 + 0.5 * np.tanh(mod / 5.0)

        # Movimiento
        leaders = np.where(self.roles > 0.5)[0]

        for i in range(self.n_cells):
            if self.roles[i] < 0.5:
                force = np.zeros(2)
                for j in leaders:
                    diff = self.positions[j] - self.positions[i]
                    dist = np.linalg.norm(diff) + 0.1
                    k = zeta_kernel(dist / 5.0, self.frequencies, self.sigma)
                    attraction = self.energies[j] * (0.5 + 0.5 * k) / dist
                    force += attraction * diff / dist

                self.positions[i] += force * self.energies[i] * mod_norm * 0.3

        # Ruido
        noise = np.random.randn(self.n_cells, 2) * 0.3 * (1 + 0.3 * (1 - mod_norm))
        self.positions += noise

        # Energia
        self.energies *= 0.99
        self.energies += np.random.rand(self.n_cells) * 0.01
        self.energies = np.clip(self.energies, 0.01, 1.0)

        # Limites
        self.positions = np.clip(self.positions, 0, self.grid_size)

        # Guardar estado: centroide + dispersion + energia media
        centroid = self.positions.mean(axis=0)
        dispersion = self.positions.std()
        energy_mean = self.energies.mean()

        state = np.concatenate([centroid, [dispersion, energy_mean]])
        self.trajectory.append(state)

    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_correlation_dimension_experiment():
    """Ejecuta experimento de dimension de correlacion."""

    print("=" * 70)
    print("VALIDACION TEORICA: DIMENSION DE CORRELACION DEL ATRACTOR")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Hipotesis: ZETA produce atractores de dimension INTERMEDIA")
    print("  UNIFORM: Dimension BAJA (sistema ordenado)")
    print("  RANDOM:  Dimension ALTA (sistema caotico)")
    print("  ZETA:    Dimension INTERMEDIA (borde del caos)")
    print()

    modulation_types = ["ZETA", "RANDOM", "UNIFORM"]
    n_steps = 500  # Necesitamos trayectoria larga
    embedding_dims = [3, 4, 5, 6]  # Probar varios
    n_trials = 3

    systems = [
        ("Hierarchical", CorrelationHierarchicalSystem, {"n_cells": 25}),
        ("ZetaOrganism", CorrelationZetaOrganism, {"n_cells": 25})
    ]

    all_results = {}

    for sys_name, sys_class, kwargs in systems:
        print(f"\n{'='*60}")
        print(f"SISTEMA: {sys_name}")
        print(f"{'='*60}")

        all_results[sys_name] = {}

        for mod_type in modulation_types:
            print(f"\n  Modulacion: {mod_type}")

            dimensions_per_embedding = {d: [] for d in embedding_dims}
            best_r_values = None
            best_c_values = None

            for trial in range(n_trials):
                np.random.seed(42 + trial * 100)

                freqs = generate_frequencies(mod_type)
                system = sys_class(frequencies=freqs, **kwargs)

                # Generar trayectoria
                for _ in range(n_steps):
                    system.step()

                trajectory = system.get_trajectory()

                # Calcular dimension para cada embedding
                for emb_dim in embedding_dims:
                    try:
                        embedded = time_delay_embedding(trajectory, emb_dim, delay=2)

                        # Submuestrear si hay muchos puntos
                        if len(embedded) > 500:
                            indices = np.random.choice(len(embedded), 500, replace=False)
                            embedded = embedded[indices]

                        d2, std, r_vals, c_vals = estimate_correlation_dimension(embedded)

                        if d2 > 0 and not np.isnan(d2):
                            dimensions_per_embedding[emb_dim].append(d2)

                            if best_r_values is None and len(r_vals) > 0:
                                best_r_values = r_vals
                                best_c_values = c_vals

                    except Exception as e:
                        pass

            # Calcular dimension promedio (usar embedding mas alto con datos validos)
            best_dim = 0
            best_std = 0
            best_emb = 0

            for emb_dim in reversed(embedding_dims):
                dims = dimensions_per_embedding[emb_dim]
                if len(dims) >= 2:
                    best_dim = np.mean(dims)
                    best_std = np.std(dims)
                    best_emb = emb_dim
                    break

            result = CorrelationDimensionResult(
                modulation_type=mod_type,
                system_name=sys_name,
                correlation_dimension=best_dim,
                dimension_std=best_std,
                scaling_range=(0, 0),
                r_values=best_r_values if best_r_values else [],
                c_values=best_c_values if best_c_values else [],
                embedding_dim=best_emb
            )

            all_results[sys_name][mod_type] = result

            print(f"    D2 = {result.correlation_dimension:.3f} +/- {result.dimension_std:.3f}")
            print(f"    (embedding dim = {result.embedding_dim})")

    # =========================================================================
    # ANALISIS
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALISIS DE RESULTADOS")
    print("=" * 70)

    edge_of_chaos_count = 0
    total_count = 0

    for sys_name in all_results:
        print(f"\n{sys_name}:")
        print("-" * 50)

        dims = {}
        for mod in modulation_types:
            dims[mod] = all_results[sys_name][mod].correlation_dimension

        # Ordenar por dimension
        sorted_dims = sorted(dims.items(), key=lambda x: x[1])

        print("  Ranking de dimension de correlacion:")
        labels = ["(menor)", "(medio)", "(mayor)"]
        for i, (name, d) in enumerate(sorted_dims):
            print(f"    {i+1}. {name:8s}: D2 = {d:.3f} {labels[i]}")

        # Verificar hipotesis
        total_count += 1
        zeta_rank = [i for i, (n, _) in enumerate(sorted_dims) if n == "ZETA"][0]

        if zeta_rank == 1:  # Posicion media
            print(f"\n  [OK] HIPOTESIS CONFIRMADA: ZETA tiene dimension INTERMEDIA")
            edge_of_chaos_count += 1
        else:
            # Verificar posicion relativa
            d_min = sorted_dims[0][1]
            d_max = sorted_dims[2][1]
            d_zeta = dims["ZETA"]

            if d_max - d_min > 0.01:
                rel_pos = (d_zeta - d_min) / (d_max - d_min)
                if 0.2 < rel_pos < 0.8:
                    print(f"\n  [~] ZETA en zona intermedia (posicion: {rel_pos:.2f})")
                    edge_of_chaos_count += 1
                else:
                    print(f"\n  [?] ZETA no en posicion intermedia (pos: {rel_pos:.2f})")
            else:
                print(f"\n  [?] Variacion insuficiente entre modulaciones")

    # =========================================================================
    # RESUMEN
    # =========================================================================

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(f"\n  Sistemas con ZETA en dimension intermedia: {edge_of_chaos_count}/{total_count}")

    if edge_of_chaos_count == total_count:
        print("\n  *** HIPOTESIS VALIDADA: DIMENSION INTERMEDIA ***")
        print("  Los ceros zeta producen atractores de complejidad optima")
    elif edge_of_chaos_count > 0:
        print("\n  ** HIPOTESIS PARCIALMENTE VALIDADA **")
    else:
        print("\n  X HIPOTESIS NO VALIDADA")

    # =========================================================================
    # VISUALIZACION
    # =========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dimension de Correlacion del Atractor\n' +
                 'Hipotesis: ZETA produce dimension intermedia (borde del caos)',
                 fontsize=13, fontweight='bold')

    colors = {"ZETA": "blue", "RANDOM": "red", "UNIFORM": "gray"}

    # Plot 1 & 2: Scaling log-log por sistema
    for idx, sys_name in enumerate(all_results):
        ax = axes[0, idx]

        for mod_type in modulation_types:
            result = all_results[sys_name][mod_type]
            if result.r_values and result.c_values:
                r = np.array(result.r_values)
                c = np.array(result.c_values)
                valid = c > 1e-10
                if np.sum(valid) > 1:
                    ax.loglog(r[valid], c[valid], 'o-', color=colors[mod_type],
                             label=f'{mod_type} (D2={result.correlation_dimension:.2f})',
                             linewidth=2, markersize=4, alpha=0.8)

        ax.set_title(f'{sys_name}: C(r) vs r')
        ax.set_xlabel('r (distancia)')
        ax.set_ylabel('C(r) (suma de correlacion)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Comparacion de dimensiones
    ax3 = axes[1, 0]

    x = np.arange(len(all_results))
    width = 0.25

    for i, mod_type in enumerate(modulation_types):
        vals = [all_results[sys][mod_type].correlation_dimension for sys in all_results]
        errs = [all_results[sys][mod_type].dimension_std for sys in all_results]
        ax3.bar(x + i*width, vals, width, label=mod_type,
               color=colors[mod_type], alpha=0.8, yerr=errs, capsize=3)

    ax3.set_ylabel('Dimension de correlacion D2')
    ax3.set_title('Comparacion: Dimension del Atractor')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(list(all_results.keys()))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Posicion relativa de ZETA
    ax4 = axes[1, 1]

    for idx, sys_name in enumerate(all_results):
        dims = {mod: all_results[sys_name][mod].correlation_dimension
                for mod in modulation_types}

        d_min = min(dims.values())
        d_max = max(dims.values())
        d_range = d_max - d_min

        if d_range > 0.001:
            for mod_type in modulation_types:
                pos = (dims[mod_type] - d_min) / d_range
                marker = 'o' if mod_type == "ZETA" else ('^' if mod_type == "RANDOM" else 's')
                size = 200 if mod_type == "ZETA" else 120
                ax4.scatter([pos], [idx], c=colors[mod_type], s=size,
                           marker=marker, zorder=3,
                           label=mod_type if idx == 0 else '')

            ax4.plot([0, 1], [idx, idx], 'k-', alpha=0.3, linewidth=2)

    ax4.axvline(x=0.5, color='green', linestyle='--', linewidth=2)
    ax4.axvspan(0.3, 0.7, alpha=0.1, color='green')

    ax4.set_xlim(-0.1, 1.1)
    ax4.set_yticks(range(len(all_results)))
    ax4.set_yticklabels(list(all_results.keys()))
    ax4.set_xlabel('Posicion relativa (0=min D2, 1=max D2)')
    ax4.set_title('Posicion de ZETA en el espectro\nde dimension')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Guardar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'correlation_dimension_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nGrafico guardado: {filename}")
    plt.close()

    return all_results


if __name__ == "__main__":
    results = run_correlation_dimension_experiment()
