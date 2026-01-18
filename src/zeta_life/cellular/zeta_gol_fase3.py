"""
Zeta Game of Life - Fase 3: Sistema Completo con Memoria Temporal
Basado en el framework de kernels zeta de Francisco Ruiz

Implementación completa que incluye:
1. Inicialización zeta estructurada (Fase 1)
2. Kernel de vecindario ponderado (Fase 2)
3. Memoria temporal via L_zeros (transformada de Laplace bilateral)
4. Filtrado espectral basado en ceros de zeta

Este es el sistema más cercano al paper original, implementando
el operador de Laplace bilateral y la memoria de largo alcance.
"""

import matplotlib

matplotlib.use('Agg')

import warnings
from collections import deque
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.fft import fft2, fftfreq, ifft2
from scipy.signal import convolve2d

try:
    from mpmath import zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

def get_zeta_zeros(M: int) -> list[float]:
    """Obtiene los primeros M ceros de ζ(s)."""
    if HAS_MPMATH:
        return [float(zetazero(k).imag) for k in range(1, M + 1)]
    else:
        known = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
                 52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
                 67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
                 79.337375, 82.910381, 84.735493, 87.425275, 88.809111]
        if M <= len(known):
            return known[:M]
        zeros = known.copy()
        for k in range(len(known), M):
            n = k + 1
            zeros.append(2 * np.pi * n / np.log(n + 2))
        return zeros

class ZetaLaplaceOperator:
    """
    Operador de Laplace bilateral basado en ceros de zeta.

    Implementa L_zeros(s) = Σ_ρ (1/(s - ρ) + 1/(s - ρ̄))

    Para convolución temporal, usamos la representación del kernel:
    K_σ(t) = Σ_ρ exp(-σ|γ|) * 2*cos(γt)

    La memoria se implementa como un filtro sobre el historial:
    x_filtered(t) = Σ_τ K_σ(t - τ) * x(τ)
    """

    def __init__(self, M: int = 30, sigma: float = 0.1):
        self.M = M
        self.sigma = sigma
        self.gammas = get_zeta_zeros(M)
        self.weights = np.array([np.exp(-sigma * abs(g)) for g in self.gammas])

    def kernel_temporal(self, t: float) -> float:
        """
        Evalúa el kernel temporal K_σ(t).

        El decay es O(1/log|t|) para t grande, como muestra tu paper.
        """
        result = 0.0
        for gamma, w in zip(self.gammas, self.weights):
            result += w * np.cos(gamma * t)
        return 2 * result

    def apply_memory_filter(
        self,
        history: list[np.ndarray],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Aplica el filtro de memoria zeta sobre el historial.

        Esto implementa la "memoria de largo alcance" del paper:
        output(x,y) = Σ_τ K_σ(τ) * history[t-τ](x,y)
        """
        if len(history) == 0:
            raise ValueError("Historial vacío")

        T = len(history)
        rows, cols = history[0].shape
        result = np.zeros((rows, cols))
        total_weight = 0.0

        for tau, grid in enumerate(history):
            # τ = 0 es el estado más reciente, τ = T-1 es el más antiguo
            t = tau  # Tiempo relativo desde el presente
            weight = self.kernel_temporal(t)
            result += weight * grid
            total_weight += abs(weight)

        if normalize and total_weight > 0:
            result /= total_weight

        return result

class ZetaSpectralFilter:
    """
    Filtro espectral basado en los ceros de zeta.

    La función de transferencia se construye a partir de los ceros:
    H(ω) = Σ_ρ exp(-σ|γ|) / (1 + (ω - γ)²/σ²)

    Esto crea picos de resonancia en las frecuencias γ_k.
    """

    def __init__(self, rows: int, cols: int, M: int = 30, sigma: float = 0.1):
        self.rows = rows
        self.cols = cols
        self.M = M
        self.sigma = sigma
        self.gammas = get_zeta_zeros(M)
        self.transfer_function = self._build_transfer_function()

    def _build_transfer_function(self) -> np.ndarray:
        """
        Construye la función de transferencia 2D.

        H(kx, ky) basada en frecuencias radiales ω = √(kx² + ky²)
        """
        kx = fftfreq(self.cols) * 2 * np.pi
        ky = fftfreq(self.rows) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)

        # Frecuencia radial normalizada
        omega = np.sqrt(KX**2 + KY**2)

        # Función de transferencia con picos en los ceros
        H = np.zeros_like(omega)
        for gamma in self.gammas:
            w = np.exp(-self.sigma * abs(gamma))
            # Lorentziana centrada en γ
            H += w / (1 + ((omega - gamma * 0.1) / self.sigma)**2)

        # Normalizar
        H = H / np.max(H)

        # Añadir componente DC para estabilidad
        H[0, 0] = 1.0

        return H  # type: ignore[return-value,no-any-return]

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro espectral al grid.

        Proceso:
        1. FFT del grid
        2. Multiplicar por función de transferencia
        3. IFFT para volver al dominio espacial
        """
        spectrum = fft2(grid)
        filtered_spectrum = spectrum * self.transfer_function
        filtered: np.ndarray = np.real(ifft2(filtered_spectrum))
        return filtered

class ZetaFullSystem:
    """
    Sistema completo de Game of Life con kernel zeta.

    Combina:
    - Inicialización estructurada
    - Kernel de vecindario ponderado
    - Memoria temporal via L_zeros
    - Filtrado espectral opcional

    La evolución combina:
    x(t+1) = GoL(x(t)) + α*Memory(history) + β*Spectral(x(t))
    """

    def __init__(
        self,
        rows: int = 100,
        cols: int = 100,
        M: int = 30,
        sigma: float = 0.1,
        memory_depth: int = 20,
        alpha: float = 0.1,  # Peso de memoria
        beta: float = 0.05,  # Peso de filtro espectral
        birth_range: tuple[float, float] = (0.8, 1.5),
        survive_range: tuple[float, float] = (0.3, 2.0),
        seed: int | None = None
    ):
        self.rows = rows
        self.cols = cols
        self.M = M
        self.sigma = sigma
        self.memory_depth = memory_depth
        self.alpha = alpha
        self.beta = beta
        self.birth_range = birth_range
        self.survive_range = survive_range
        self.generation = 0

        # Operadores zeta
        self.laplace_op = ZetaLaplaceOperator(M, sigma)
        self.spectral_filter = ZetaSpectralFilter(rows, cols, M, sigma)

        # Kernel espacial
        self.spatial_kernel = self._build_spatial_kernel()

        # Historial para memoria
        self.history: deque[np.ndarray] = deque(maxlen=memory_depth)

        # Inicialización
        if seed is not None:
            np.random.seed(seed)
        self.grid = self._zeta_init()
        self.history.append(self.grid.copy())

        # Métricas
        self.metrics_history: list[dict[str, int | float]] = []

    def _build_spatial_kernel(self, R: int = 2) -> np.ndarray:
        """Construye kernel espacial zeta (de Fase 2)."""
        gammas = get_zeta_zeros(self.M)
        size = 2 * R + 1
        K = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                x, y = i - R, j - R
                if x == 0 and y == 0:
                    continue
                r = np.sqrt(x**2 + y**2)
                for gamma in gammas:
                    K[i, j] += np.exp(-self.sigma * abs(gamma)) * np.cos(gamma * r)

        total = np.sum(np.abs(K))
        if total > 0:
            K = K / total * 8
        return K

    def _zeta_init(self) -> np.ndarray:
        """Inicialización con ruido zeta estructurado."""
        gammas = self.laplace_op.gammas
        M = len(gammas)
        field = np.zeros((self.rows, self.cols), dtype=complex)
        phases = np.random.uniform(0, 2 * np.pi, M)

        for i in range(self.rows):
            for j in range(self.cols):
                x = 2 * np.pi * i / self.rows
                y = 2 * np.pi * j / self.cols

                for k, gamma in enumerate(gammas):
                    gamma_y = gammas[(k + 17) % M]
                    w = np.exp(-self.sigma * abs(gamma))
                    phase = gamma * x + gamma_y * y + phases[k]
                    field[i, j] += w * np.exp(1j * phase)

        real_field: np.ndarray = np.real(field)
        return (real_field > np.mean(real_field)).astype(float)  # type: ignore[return-value,no-any-return]

    def weighted_neighbors(self) -> np.ndarray:
        """Calcula vecinos ponderados con kernel zeta."""
        result: np.ndarray = convolve2d(self.grid, self.spatial_kernel, mode='same', boundary='wrap')
        return result

    def apply_gol_rules(self, neighbors: np.ndarray) -> np.ndarray:
        """Aplica reglas tipo GoL con umbrales continuos."""
        birth = (self.grid == 0) & \
                (neighbors >= self.birth_range[0]) & \
                (neighbors <= self.birth_range[1])

        survive = (self.grid == 1) & \
                  (neighbors >= self.survive_range[0]) & \
                  (neighbors <= self.survive_range[1])

        result: np.ndarray = (birth | survive).astype(float)
        return result

    def step(self) -> np.ndarray:
        """
        Ejecuta un paso completo del sistema.

        x(t+1) = GoL(x(t)) + α*Memory(history) + β*Spectral(x(t))
        """
        # 1. Evolución GoL con kernel zeta
        neighbors = self.weighted_neighbors()
        gol_next = self.apply_gol_rules(neighbors)

        # 2. Componente de memoria temporal
        if len(self.history) > 1:
            memory_contrib = self.laplace_op.apply_memory_filter(
                list(self.history)
            )
        else:
            memory_contrib = np.zeros_like(self.grid)

        # 3. Componente espectral
        spectral_contrib = self.spectral_filter.apply(self.grid)

        # 4. Combinar componentes
        combined = gol_next + self.alpha * memory_contrib + self.beta * spectral_contrib

        # 5. Binarizar con umbral adaptativo
        threshold = np.mean(combined) + 0.1 * np.std(combined)
        self.grid = (combined > threshold).astype(float)

        # Actualizar historial
        self.history.append(self.grid.copy())
        self.generation += 1

        # Guardar métricas
        self.metrics_history.append(self.get_statistics())

        return self.grid

    def run(self, steps: int) -> np.ndarray:
        """Ejecuta múltiples pasos."""
        for _ in range(steps):
            self.step()
        return self.grid

    def get_statistics(self) -> dict[str, int | float]:
        """Calcula estadísticas del estado actual."""
        alive = np.sum(self.grid)
        return {
            'generation': self.generation,
            'alive_cells': int(alive),
            'density': alive / (self.rows * self.cols)
        }

    def analyze_correlations(self, max_distance: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Analiza correlaciones espaciales."""
        distances = np.arange(1, max_distance + 1)
        correlations = []

        grid_centered = self.grid - np.mean(self.grid)
        variance = np.var(self.grid)

        if variance == 0:
            return distances, np.zeros_like(distances, dtype=float)

        for d in distances:
            corr = 0
            count = 0

            if d < self.cols:
                corr += np.mean(grid_centered[:, :-d] * grid_centered[:, d:])
                count += 1
            if d < self.rows:
                corr += np.mean(grid_centered[:-d, :] * grid_centered[d:, :])
                count += 1

            correlations.append(corr / (count * variance) if count > 0 else 0)

        return distances, np.array(correlations)

    def analyze_temporal_memory(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Analiza el efecto de la memoria temporal.

        Compara el estado actual con estados históricos
        para medir la persistencia de información.
        """
        if len(self.history) < 2:
            return np.array([0]), np.array([0])

        history_list = list(self.history)
        current = history_list[-1]
        lags = np.arange(1, min(len(history_list), 20))
        temporal_corr = []

        current_centered = current - np.mean(current)
        var_current = np.var(current)

        if var_current == 0:
            return lags, np.zeros_like(lags, dtype=float)

        for lag in lags:
            past = history_list[-(lag + 1)]
            past_centered = past - np.mean(past)
            corr = np.mean(current_centered * past_centered) / var_current
            temporal_corr.append(corr)

        return lags, np.array(temporal_corr)

def visualize_full_system(game: ZetaFullSystem):
    """Visualización completa del sistema."""
    fig = plt.figure(figsize=(18, 14))

    # Layout: 3 filas
    # Fila 1: Estados en diferentes generaciones
    # Fila 2: Análisis de componentes
    # Fila 3: Métricas temporales

    # === FILA 1: Evolución ===
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    cmap = LinearSegmentedColormap.from_list('zeta', ['#0a0a0a', '#00ff88'])

    # Estado inicial (reiniciar para mostrar evolución)
    game_viz = ZetaFullSystem(
        rows=game.rows, cols=game.cols,
        M=game.M, sigma=game.sigma,
        memory_depth=game.memory_depth,
        alpha=game.alpha, beta=game.beta,
        seed=42
    )

    generations = [0, 20, 50, 100]
    for idx, gen in enumerate(generations):
        ax = fig.add_subplot(gs[0, idx])
        if gen > 0:
            game_viz.run(gen - game_viz.generation)
        ax.imshow(game_viz.grid, cmap=cmap, interpolation='nearest')
        stats = game_viz.get_statistics()
        ax.set_title(f'Gen {gen} | Densidad: {stats["density"]:.3f}')
        ax.axis('off')

    # === FILA 2: Análisis de componentes ===

    # Kernel espacial
    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(game.spatial_kernel, cmap='RdBu_r', origin='lower')
    ax1.set_title('Kernel Espacial Zeta')
    plt.colorbar(im1, ax=ax1)

    # Función de transferencia espectral
    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.imshow(np.log1p(np.abs(np.fft.fftshift(game.spectral_filter.transfer_function))),
                     cmap='viridis', origin='lower')
    ax2.set_title('Función de Transferencia (log)')
    plt.colorbar(im2, ax=ax2)

    # Kernel temporal
    ax3 = fig.add_subplot(gs[1, 2])
    t_vals = np.linspace(0, 20, 100)
    k_vals = [game.laplace_op.kernel_temporal(t) for t in t_vals]
    ax3.plot(t_vals, k_vals, 'b-', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('τ (lag temporal)')
    ax3.set_ylabel('K_σ(τ)')
    ax3.set_title('Kernel Temporal L_zeros')
    ax3.grid(True, alpha=0.3)

    # Correlaciones espaciales
    ax4 = fig.add_subplot(gs[1, 3])
    dist, corr = game_viz.analyze_correlations()
    ax4.plot(dist, corr, 'go-', label='Espacial')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Distancia')
    ax4.set_ylabel('Correlación')
    ax4.set_title('Correlaciones Espaciales')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # === FILA 3: Métricas temporales ===

    # Densidad vs tiempo
    ax5 = fig.add_subplot(gs[2, 0:2])
    if len(game_viz.metrics_history) > 0:
        gens = [m['generation'] for m in game_viz.metrics_history]
        densities = [m['density'] for m in game_viz.metrics_history]
        ax5.plot(gens, densities, 'b-', linewidth=2)
        ax5.set_xlabel('Generación')
        ax5.set_ylabel('Densidad')
        ax5.set_title('Evolución de Densidad')
        ax5.grid(True, alpha=0.3)

    # Correlación temporal
    ax6 = fig.add_subplot(gs[2, 2:4])
    lags, temp_corr = game_viz.analyze_temporal_memory()
    if len(lags) > 1:
        ax6.plot(lags, temp_corr, 'r-o', linewidth=2, markersize=4)

        # Añadir curva teórica del kernel
        theory = np.array([game.laplace_op.kernel_temporal(l) for l in lags])
        if np.max(np.abs(theory)) > 0:
            theory = theory / np.max(np.abs(theory)) * np.max(np.abs(temp_corr))
        ax6.plot(lags, theory, 'b--', alpha=0.7, label='Kernel teórico (escalado)')

    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Lag temporal')
    ax6.set_ylabel('Autocorrelación')
    ax6.set_title('Memoria Temporal (Autocorrelación)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Sistema Completo Zeta GoL | α={game.alpha}, β={game.beta}, M={game.M}', fontsize=14)

    return fig

def compare_memory_effects(rows=80, cols=80, steps=100):
    """
    Compara el sistema con y sin memoria temporal.
    """
    print("Comparando efectos de memoria temporal...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap = LinearSegmentedColormap.from_list('zeta', ['#0a0a0a', '#00ff88'])

    # Sin memoria (α=0)
    game_no_memory = ZetaFullSystem(
        rows=rows, cols=cols, M=20, sigma=0.1,
        memory_depth=20, alpha=0.0, beta=0.0, seed=42
    )

    # Con memoria
    game_memory = ZetaFullSystem(
        rows=rows, cols=cols, M=20, sigma=0.1,
        memory_depth=20, alpha=0.15, beta=0.05, seed=42
    )

    generations = [0, 30, 60, 100]

    for idx, gen in enumerate(generations):
        if gen > 0:
            while game_no_memory.generation < gen:
                game_no_memory.step()
            while game_memory.generation < gen:
                game_memory.step()

        axes[0, idx].imshow(game_no_memory.grid, cmap=cmap)
        axes[0, idx].set_title(f'Sin Memoria - Gen {gen}')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(game_memory.grid, cmap=cmap)
        axes[1, idx].set_title(f'Con Memoria - Gen {gen}')
        axes[1, idx].axis('off')

    # Estadísticas finales
    stats_no = game_no_memory.get_statistics()
    stats_mem = game_memory.get_statistics()

    plt.suptitle(
        f'Efecto de Memoria Temporal | '
        f'Sin: {stats_no["density"]:.3f} | '
        f'Con: {stats_mem["density"]:.3f}',
        fontsize=14
    )

    plt.tight_layout()
    return fig

def analyze_alpha_beta_space(rows=60, cols=60, steps=50):
    """
    Explora el espacio de parámetros (alpha, beta).
    """
    print("Explorando espacio de parametros alpha-beta...")

    alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
    betas = [0.0, 0.02, 0.05, 0.08, 0.1]

    density_map = np.zeros((len(betas), len(alphas)))

    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            game = ZetaFullSystem(
                rows=rows, cols=cols, M=15, sigma=0.1,
                memory_depth=15, alpha=alpha, beta=beta, seed=42
            )
            game.run(steps)
            density_map[i, j] = game.get_statistics()['density']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(density_map, cmap='viridis', origin='lower',
                   extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
                   aspect='auto')
    ax.set_xlabel('α (peso memoria)')
    ax.set_ylabel('β (peso espectral)')
    ax.set_title(f'Densidad Final vs Parámetros (después de {steps} generaciones)')
    plt.colorbar(im, ax=ax, label='Densidad')

    # Marcar óptimo
    max_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
    ax.plot(alphas[max_idx[1]], betas[max_idx[0]], 'r*', markersize=15, label='Máximo')
    ax.legend()

    return fig

def demo_fase3():
    """Demostración completa de Fase 3."""
    print("=" * 70)
    print("ZETA GAME OF LIFE - Fase 3: Sistema Completo con Memoria Temporal")
    print("=" * 70)

    # 1. Crear sistema completo
    print("\n1. Creando sistema completo...")
    game = ZetaFullSystem(
        rows=100, cols=100,
        M=25, sigma=0.1,
        memory_depth=20,
        alpha=0.12,
        beta=0.04,
        seed=42
    )
    print(f"   Grid: {game.rows}x{game.cols}")
    print(f"   Ceros de zeta: {game.M}")
    print(f"   Memoria temporal: {game.memory_depth} generaciones")
    print(f"   alpha (memoria): {game.alpha}, beta (espectral): {game.beta}")

    # 2. Ejecutar evolución
    print("\n2. Ejecutando 100 generaciones...")
    game.run(100)
    stats = game.get_statistics()
    print(f"   Células vivas: {stats['alive_cells']}")
    print(f"   Densidad final: {stats['density']:.3f}")

    # 3. Visualización completa
    print("\n3. Generando visualización completa...")
    fig1 = visualize_full_system(game)
    fig1.savefig('zeta_full_system.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_full_system.png")

    # 4. Comparar con/sin memoria
    print("\n4. Comparando con/sin memoria temporal...")
    fig2 = compare_memory_effects()
    fig2.savefig('zeta_memory_comparison.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_memory_comparison.png")

    # 5. Explorar espacio de parámetros
    print("\n5. Explorando espacio alpha-beta...")
    fig3 = analyze_alpha_beta_space()
    fig3.savefig('zeta_alpha_beta_space.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_alpha_beta_space.png")

    # 6. Estado final detallado
    print("\n6. Generando estado final de alta resolución...")
    game_hr = ZetaFullSystem(
        rows=150, cols=150,
        M=30, sigma=0.08,
        memory_depth=25,
        alpha=0.12, beta=0.04,
        seed=123
    )
    game_hr.run(150)

    fig4, ax = plt.subplots(figsize=(14, 14))
    cmap = LinearSegmentedColormap.from_list('zeta', ['#0a0a0a', '#00ff88'])
    ax.imshow(game_hr.grid, cmap=cmap, interpolation='nearest')
    stats_hr = game_hr.get_statistics()
    ax.set_title(f'Zeta GoL Completo - Gen {stats_hr["generation"]} | Densidad: {stats_hr["density"]:.3f}')
    ax.axis('off')
    fig4.savefig('zeta_full_system_highres.png', dpi=200, bbox_inches='tight')
    print("   Guardado: zeta_full_system_highres.png")

    print("\n" + "=" * 70)
    print("Fase 3 completada - Sistema completo implementado")
    print("=" * 70)

    return game

if __name__ == "__main__":
    demo_fase3()
