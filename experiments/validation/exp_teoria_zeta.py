#!/usr/bin/env python
"""
EXPERIMENTO: Fundamentos Matemáticos del Kernel Zeta

Pregunta central:
    ¿Son los ceros de la función zeta ESPECIALES para producir
    comportamientos emergentes, o cualquier frecuencia funcionaría igual?

Comparamos 4 tipos de frecuencias:
1. ZETA: Ceros reales de la función zeta (γ₁, γ₂, ...)
2. RANDOM: Frecuencias aleatorias en el mismo rango
3. UNIFORM: Frecuencias uniformemente espaciadas
4. GUE: Frecuencias con espaciado de matrices aleatorias (sin ser ceros zeta)

Métricas:
- Complejidad de patrones emergentes (entropía)
- Estabilidad del sistema (varianza temporal)
- Velocidad de convergencia a equilibrio
- Riqueza de comportamientos (número de estados distintos)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DEFINICIÓN DE FRECUENCIAS
# =============================================================================

# Ceros zeta reales (primeros 30)
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
])

def generate_random_frequencies(n: int, min_val: float, max_val: float, seed: int = 42) -> np.ndarray:
    """Genera frecuencias aleatorias uniformes."""
    np.random.seed(seed)
    return np.sort(np.random.uniform(min_val, max_val, n))

def generate_uniform_frequencies(n: int, min_val: float, max_val: float) -> np.ndarray:
    """Genera frecuencias uniformemente espaciadas."""
    return np.linspace(min_val, max_val, n)

def generate_gue_frequencies(n: int, min_val: float, max_val: float, seed: int = 42) -> np.ndarray:
    """
    Genera frecuencias con espaciado GUE (Gaussian Unitary Ensemble).

    Los ceros zeta siguen estadísticas GUE. Generamos frecuencias
    que tienen el mismo tipo de espaciado pero NO son los ceros reales.
    """
    np.random.seed(seed)
    # Generar matriz hermitiana aleatoria
    size = n + 10  # un poco más para tener margen
    H = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    H = (H + H.conj().T) / 2  # Hacer hermitiana

    # Eigenvalores siguen distribución GUE
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(H)))

    # Escalar al rango deseado
    eigenvalues = eigenvalues[:n]
    eigenvalues = (eigenvalues - eigenvalues.min()) / (eigenvalues.max() - eigenvalues.min())
    eigenvalues = eigenvalues * (max_val - min_val) + min_val

    return eigenvalues

# =============================================================================
# KERNEL Y SISTEMA
# =============================================================================

@dataclass
class FrequencyKernel:
    """Kernel basado en un conjunto de frecuencias."""
    frequencies: np.ndarray
    sigma: float = 0.1
    name: str = "unknown"

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evalúa K(t) = Σ exp(-σ|γ|) * cos(γ*t)"""
        result = np.zeros_like(t, dtype=float)
        for gamma in self.frequencies:
            weight = np.exp(-self.sigma * abs(gamma))
            result += weight * np.cos(gamma * t)
        return result / len(self.frequencies)

    def evaluate_2d(self, size: int, radius: int = 3) -> np.ndarray:
        """Genera kernel 2D para convolución."""
        kernel = np.zeros((2*radius+1, 2*radius+1))
        center = radius
        for i in range(2*radius+1):
            for j in range(2*radius+1):
                r = np.sqrt((i-center)**2 + (j-center)**2)
                if r > 0:
                    kernel[i,j] = self.evaluate(np.array([r]))[0]
        kernel[center, center] = 0  # No auto-interacción
        return kernel / (kernel.sum() + 1e-10)

class CellularAutomaton:
    """Autómata celular con kernel configurable."""

    def __init__(self, size: int, kernel: FrequencyKernel):
        self.size = size
        self.kernel = kernel
        self.kernel_2d = kernel.evaluate_2d(size, radius=3)
        self.grid = None
        self.history = []

    def initialize(self, seed: int = 42):
        """Inicializa con ruido estructurado."""
        np.random.seed(seed)
        # Mezcla de ruido + estructura
        noise = np.random.random((self.size, self.size))
        # Añadir algo de estructura inicial
        x, y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        structure = 0.5 + 0.3 * np.sin(x * 0.2) * np.cos(y * 0.2)
        self.grid = (noise * 0.5 + structure * 0.5 > 0.5).astype(float)
        self.history = [self.grid.copy()]

    def step(self):
        """Un paso de evolución."""
        # Convolución con kernel
        from scipy.signal import convolve2d
        neighbor_sum = convolve2d(self.grid, self.kernel_2d, mode='same', boundary='wrap')

        # Regla continua suavizada (similar a Game of Life pero continuo)
        # Nacimiento: vecinos en [0.2, 0.35]
        # Supervivencia: vecinos en [0.15, 0.45]
        birth = (neighbor_sum > 0.2) & (neighbor_sum < 0.35) & (self.grid < 0.5)
        survive = (neighbor_sum > 0.15) & (neighbor_sum < 0.45) & (self.grid >= 0.5)

        new_grid = np.zeros_like(self.grid)
        new_grid[birth | survive] = 1.0

        # Suavizado para evitar oscilaciones bruscas
        self.grid = 0.7 * new_grid + 0.3 * self.grid
        self.grid = np.clip(self.grid, 0, 1)

        self.history.append(self.grid.copy())

    def run(self, steps: int):
        """Ejecuta múltiples pasos."""
        for _ in range(steps):
            self.step()

# =============================================================================
# MÉTRICAS DE ANÁLISIS
# =============================================================================

def compute_spatial_entropy(grid: np.ndarray, bins: int = 10) -> float:
    """Entropía espacial del patrón."""
    hist, _ = np.histogram(grid.flatten(), bins=bins, range=(0, 1))
    hist = hist / hist.sum()
    return entropy(hist + 1e-10)

def compute_temporal_stability(history: List[np.ndarray], window: int = 20) -> float:
    """Estabilidad temporal (inverso de varianza)."""
    if len(history) < window:
        return 0.0
    recent = history[-window:]
    # Varianza entre frames consecutivos
    diffs = [np.mean(np.abs(recent[i+1] - recent[i])) for i in range(len(recent)-1)]
    return 1.0 / (np.mean(diffs) + 0.01)

def compute_pattern_complexity(grid: np.ndarray) -> float:
    """Complejidad del patrón (basada en gradientes)."""
    gx = np.abs(np.diff(grid, axis=0))
    gy = np.abs(np.diff(grid, axis=1))
    return np.mean(gx) + np.mean(gy)

def compute_autocorrelation_decay(history: List[np.ndarray], max_lag: int = 50) -> float:
    """Velocidad de decaimiento de autocorrelación temporal."""
    if len(history) < max_lag + 10:
        return 0.0

    # Tomar serie temporal del centro
    center = len(history[0]) // 2
    series = np.array([h[center, center] for h in history])

    # Autocorrelación
    mean = np.mean(series)
    var = np.var(series)
    if var < 1e-10:
        return 0.0

    autocorr = []
    for lag in range(1, min(max_lag, len(series)//2)):
        c = np.mean((series[:-lag] - mean) * (series[lag:] - mean)) / var
        autocorr.append(c)

    # Encontrar dónde cae a 1/e
    autocorr = np.array(autocorr)
    decay_idx = np.where(autocorr < 1/np.e)[0]
    if len(decay_idx) > 0:
        return 1.0 / (decay_idx[0] + 1)
    return 1.0 / max_lag

def count_distinct_states(history: List[np.ndarray], threshold: float = 0.1) -> int:
    """Cuenta estados cualitativamente distintos."""
    if len(history) < 2:
        return 1

    # Discretizar estados
    states = []
    for grid in history[::5]:  # Cada 5 frames
        # Hash simple: suma por regiones
        regions = []
        step = len(grid) // 4
        for i in range(4):
            for j in range(4):
                region_sum = grid[i*step:(i+1)*step, j*step:(j+1)*step].mean()
                regions.append(int(region_sum * 10))
        states.append(tuple(regions))

    return len(set(states))

# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_comparison_experiment(n_frequencies: int = 20,
                              grid_size: int = 64,
                              n_steps: int = 200,
                              n_trials: int = 3) -> Dict:
    """
    Compara los 4 tipos de frecuencias.
    """
    print("="*70)
    print("EXPERIMENTO: COMPARACIÓN DE FRECUENCIAS")
    print("="*70)
    print(f"Frecuencias: {n_frequencies}")
    print(f"Grid: {grid_size}x{grid_size}")
    print(f"Steps: {n_steps}")
    print(f"Trials: {n_trials}")
    print()

    # Rango de frecuencias (basado en ceros zeta)
    min_freq = ZETA_ZEROS[0]
    max_freq = ZETA_ZEROS[n_frequencies-1]

    # Crear los 4 tipos de frecuencias
    freq_types = {
        'ZETA': ZETA_ZEROS[:n_frequencies],
        'RANDOM': generate_random_frequencies(n_frequencies, min_freq, max_freq),
        'UNIFORM': generate_uniform_frequencies(n_frequencies, min_freq, max_freq),
        'GUE': generate_gue_frequencies(n_frequencies, min_freq, max_freq),
    }

    # Mostrar frecuencias
    print("Primeras 5 frecuencias de cada tipo:")
    for name, freqs in freq_types.items():
        print(f"  {name:8s}: {freqs[:5].round(2)}")
    print()

    # Métricas por tipo
    results = {name: {
        'entropy': [], 'stability': [], 'complexity': [],
        'autocorr_decay': [], 'distinct_states': []
    } for name in freq_types}

    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}...")

        for name, frequencies in freq_types.items():
            # Crear kernel y sistema
            kernel = FrequencyKernel(frequencies, sigma=0.1, name=name)
            ca = CellularAutomaton(grid_size, kernel)
            ca.initialize(seed=42 + trial)

            # Ejecutar
            ca.run(n_steps)

            # Calcular métricas
            final_grid = ca.grid
            results[name]['entropy'].append(compute_spatial_entropy(final_grid))
            results[name]['stability'].append(compute_temporal_stability(ca.history))
            results[name]['complexity'].append(compute_pattern_complexity(final_grid))
            results[name]['autocorr_decay'].append(compute_autocorrelation_decay(ca.history))
            results[name]['distinct_states'].append(count_distinct_states(ca.history))

    # Promediar resultados
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)

    summary = {}
    metrics = ['entropy', 'stability', 'complexity', 'autocorr_decay', 'distinct_states']

    print(f"\n{'Tipo':<10} {'Entropía':<12} {'Estabilidad':<12} {'Complejidad':<12} {'Decorr':<12} {'Estados':<10}")
    print("-"*70)

    for name in freq_types:
        summary[name] = {}
        row = f"{name:<10}"
        for metric in metrics:
            mean_val = np.mean(results[name][metric])
            std_val = np.std(results[name][metric])
            summary[name][metric] = (mean_val, std_val)
            row += f" {mean_val:>8.3f}±{std_val:.2f}"
        print(row)

    # Análisis estadístico
    print("\n" + "="*70)
    print("ANÁLISIS: ¿Son los ceros zeta especiales?")
    print("="*70)

    zeta_scores = []
    for metric in metrics:
        zeta_val = summary['ZETA'][metric][0]
        other_vals = [summary[name][metric][0] for name in ['RANDOM', 'UNIFORM', 'GUE']]

        # ¿Zeta es el mejor o peor?
        if metric in ['entropy', 'complexity', 'distinct_states']:
            # Mayor es mejor
            rank = sum(1 for v in other_vals if zeta_val > v)
        else:
            # Para estabilidad, mayor es mejor
            rank = sum(1 for v in other_vals if zeta_val > v)

        zeta_scores.append(rank)
        status = "[MEJOR]" if rank == 3 else ("[MEDIO]" if rank >= 1 else "[PEOR]")
        print(f"  {metric:<20}: ZETA rank {rank+1}/4 {status}")

    overall = np.mean(zeta_scores)
    print(f"\n  Ranking promedio ZETA: {overall+1:.1f}/4")

    if overall >= 2:
        print("\n  [HIPÓTESIS SOPORTADA] Los ceros zeta producen mejores comportamientos")
    elif overall >= 1:
        print("\n  [RESULTADO MIXTO] Ceros zeta comparables a otras frecuencias")
    else:
        print("\n  [HIPÓTESIS RECHAZADA] Ceros zeta no son especiales")

    return {'freq_types': freq_types, 'results': results, 'summary': summary}


def analyze_spacing_distribution(results: Dict):
    """Analiza la distribución de espaciado entre frecuencias."""
    print("\n" + "="*70)
    print("ANÁLISIS: DISTRIBUCIÓN DE ESPACIADO")
    print("="*70)

    freq_types = results['freq_types']

    print(f"\n{'Tipo':<10} {'Media':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Ratio max/min':<12}")
    print("-"*62)

    for name, freqs in freq_types.items():
        spacings = np.diff(freqs)
        mean_s = np.mean(spacings)
        std_s = np.std(spacings)
        min_s = np.min(spacings)
        max_s = np.max(spacings)
        ratio = max_s / (min_s + 0.01)

        print(f"{name:<10} {mean_s:<10.3f} {std_s:<10.3f} {min_s:<10.3f} {max_s:<10.3f} {ratio:<12.2f}")

    print("\nInterpretacion:")
    print("  - UNIFORM: ratio = 1 (espaciado constante)")
    print("  - RANDOM: ratio alto (espaciado caotico)")
    print("  - GUE: ratio medio (repulsion de eigenvalores)")
    print("  - ZETA: ratio especifico (estructura de primos)")


def visualize_kernels(freq_types: Dict, save_path: str = "zeta_kernel_comparison.png"):
    """Visualiza los kernels 1D de cada tipo."""
    t = np.linspace(0, 10, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, freqs) in enumerate(freq_types.items()):
        kernel = FrequencyKernel(freqs, sigma=0.1, name=name)
        y = kernel.evaluate(t)

        axes[idx].plot(t, y, 'b-', linewidth=1)
        axes[idx].set_title(f'{name} Kernel (n={len(freqs)})')
        axes[idx].set_xlabel('t')
        axes[idx].set_ylabel('K(t)')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nGuardado: {save_path}")


def visualize_final_patterns(freq_types: Dict, grid_size: int = 64,
                             n_steps: int = 200, save_path: str = "zeta_patterns_comparison.png"):
    """Visualiza los patrones finales de cada tipo."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, freqs) in enumerate(freq_types.items()):
        kernel = FrequencyKernel(freqs, sigma=0.1, name=name)
        ca = CellularAutomaton(grid_size, kernel)
        ca.initialize(seed=42)
        ca.run(n_steps)

        axes[idx].imshow(ca.grid, cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(f'{name} (step {n_steps})')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Guardado: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Ejecutar experimento principal
    results = run_comparison_experiment(
        n_frequencies=20,
        grid_size=64,
        n_steps=200,
        n_trials=5
    )

    # Análisis de espaciado
    analyze_spacing_distribution(results)

    # Visualizaciones
    visualize_kernels(results['freq_types'])
    visualize_final_patterns(results['freq_types'])

    print("\n" + "="*70)
    print("EXPERIMENTO COMPLETO")
    print("="*70)
