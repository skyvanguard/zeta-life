"""
Zeta Game of Life - Fase 1: Inicialización con Ruido Estructurado
Basado en el framework de kernels zeta de Francisco Ruiz

El kernel zeta usa los ceros no triviales de ζ(s) para crear
correlaciones estructuradas en la inicialización del autómata.
"""

import matplotlib

matplotlib.use('Agg')  # Backend no interactivo

import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# Intentar importar mpmath para ceros exactos de zeta
try:
    from mpmath import zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
    warnings.warn("mpmath no disponible. Usando aproximaciones de ceros de zeta.")


def get_zeta_zeros(M: int) -> list[float]:
    """
    Obtiene los primeros M ceros no triviales de ζ(s).
    Los ceros están en s = 1/2 + iγ, retornamos las partes imaginarias γ.
    """
    if HAS_MPMATH:
        return [float(zetazero(k).imag) for k in range(1, M + 1)]
    else:
        # Aproximaciones conocidas de los primeros ceros
        known_zeros = [
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
            67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
            79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
            92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
            103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
            114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
            124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
            134.756509, 138.116042, 139.736209, 141.123707, 143.111846
        ]
        if M <= len(known_zeros):
            return known_zeros[:M]
        else:
            # Extrapolar usando la densidad asintótica de ceros
            zeros = known_zeros.copy()
            for k in range(len(known_zeros), M):
                # Aproximación: γ_n ≈ 2πn / ln(n)
                n = k + 1
                zeros.append(2 * np.pi * n / np.log(n + 2))
            return zeros


class ZetaKernel:
    """
    Kernel basado en ceros de la función zeta de Riemann.

    K_σ(t) = Σ_ρ exp(-σ|γ|) * (exp(iγt) + exp(-iγt))
           = 2 * Σ_ρ exp(-σ|γ|) * cos(γt)
    """

    def __init__(self, M: int = 50, sigma: float = 0.1) -> None:
        """
        Args:
            M: Número de ceros a usar
            sigma: Parámetro de regularización (decay exponencial)
        """
        self.M = M
        self.sigma = sigma
        self.gammas = get_zeta_zeros(M)
        self.weights = np.array([np.exp(-sigma * abs(g)) for g in self.gammas])

    def evaluate(self, t: float) -> float:
        """Evalúa el kernel en un punto t."""
        result = 0.0
        for gamma, w in zip(self.gammas, self.weights):
            result += w * np.cos(gamma * t)
        return 2 * result

    def evaluate_2d(self, x: float, y: float, mode: str = 'radial') -> float:
        """
        Evalúa el kernel en 2D.

        Args:
            x, y: Coordenadas
            mode: 'radial' (distancia), 'separable' (producto), 'sum' (suma)
        """
        if mode == 'radial':
            r = np.sqrt(x**2 + y**2)
            return self.evaluate(r)
        elif mode == 'separable':
            return self.evaluate(x) * self.evaluate(y)
        elif mode == 'sum':
            return self.evaluate(x) + self.evaluate(y)
        else:
            raise ValueError(f"Modo desconocido: {mode}")


class ZetaGameOfLife:
    """
    Game of Life con inicialización basada en kernel zeta.
    """

    def __init__(
        self,
        rows: int = 100,
        cols: int = 100,
        M: int = 50,
        sigma: float = 0.1,
        threshold: float = 0.0,
        seed: int | None = None
    ):
        """
        Args:
            rows, cols: Dimensiones del grid
            M: Número de ceros de zeta a usar
            sigma: Regularización del kernel
            threshold: Umbral para binarización (en desviaciones estándar)
            seed: Semilla para reproducibilidad
        """
        self.rows = rows
        self.cols = cols
        self.kernel = ZetaKernel(M, sigma)
        self.threshold = threshold
        self.seed = seed
        self.generation = 0
        self.history: list[np.ndarray] = []

        # Inicializar grid con ruido zeta estructurado
        self.grid = self._initialize_zeta_noise()

    def _initialize_zeta_noise(self) -> np.ndarray:
        """
        Genera el estado inicial usando superposición de ondas
        basadas en los ceros de zeta.

        La correlación espacial viene de:
        C(Δx, Δy) ~ Σ exp(-σ|γ|) * cos(γ·r)

        donde r = √(Δx² + Δy²)
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        gammas = self.kernel.gammas
        weights = self.kernel.weights
        M = len(gammas)

        # Campo complejo para capturar interferencia
        field = np.zeros((self.rows, self.cols), dtype=complex)

        # Fase aleatoria para cada modo (rompe simetría)
        phases = np.random.uniform(0, 2 * np.pi, M)

        for i in range(self.rows):
            for j in range(self.cols):
                # Normalizar coordenadas a [0, 2π]
                x = 2 * np.pi * i / self.rows
                y = 2 * np.pi * j / self.cols

                for k, (gamma, w) in enumerate(zip(gammas, weights)):
                    # Usar un segundo gamma para la dirección y
                    gamma_y = gammas[(k + 17) % M]  # Offset primo para evitar patrones

                    # Onda con peso y fase
                    phase = gamma * x + gamma_y * y + phases[k]
                    field[i, j] += w * np.exp(1j * phase)

        # Parte real del campo
        real_field = np.real(field)

        # Normalizar y binarizar
        mean = np.mean(real_field)
        std = np.std(real_field)

        # Umbral: valores > threshold*std se convierten en células vivas
        binary: np.ndarray = (real_field > mean + self.threshold * std).astype(int)

        return binary

    def count_neighbors(self, grid: np.ndarray) -> np.ndarray:
        """Cuenta vecinos vivos usando convolución (Moore neighborhood)."""
        # Kernel de Moore: todos los 8 vecinos
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])

        # Condiciones de frontera periódicas
        padded = np.pad(grid, 1, mode='wrap')

        neighbors = np.zeros_like(grid)
        for di in range(3):
            for dj in range(3):
                if di == 1 and dj == 1:
                    continue
                neighbors += padded[di:di+self.rows, dj:dj+self.cols]

        return neighbors

    def step(self) -> np.ndarray:
        """
        Ejecuta un paso del Game of Life clásico (B3/S23).

        Reglas:
        - Célula viva con 2-3 vecinos: sobrevive
        - Célula muerta con exactamente 3 vecinos: nace
        - Cualquier otro caso: muere/permanece muerta
        """
        neighbors = self.count_neighbors(self.grid)

        # B3/S23
        birth = (self.grid == 0) & (neighbors == 3)
        survive = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))

        self.grid = (birth | survive).astype(int)
        self.generation += 1

        return self.grid

    def run(self, steps: int, save_history: bool = False) -> np.ndarray:
        """Ejecuta múltiples pasos."""
        if save_history:
            self.history = [self.grid.copy()]

        for _ in range(steps):
            self.step()
            if save_history:
                self.history.append(self.grid.copy())

        return self.grid

    def get_statistics(self) -> dict:
        """Calcula estadísticas del estado actual."""
        alive = np.sum(self.grid)
        total = self.rows * self.cols
        density = alive / total

        return {
            'generation': self.generation,
            'alive_cells': int(alive),
            'total_cells': total,
            'density': density
        }

    def analyze_correlations(self, max_distance: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """
        Analiza la función de correlación espacial.
        Esto nos permite verificar si las correlaciones zeta persisten.
        """
        distances = np.arange(1, max_distance + 1)
        correlations = []

        grid_centered = self.grid - np.mean(self.grid)
        variance = np.var(self.grid)

        if variance == 0:
            return distances, np.zeros_like(distances, dtype=float)

        for d in distances:
            # Correlación a distancia d (promedio sobre direcciones)
            corr = 0
            count = 0

            # Horizontal
            if d < self.cols:
                corr += np.mean(grid_centered[:, :-d] * grid_centered[:, d:])
                count += 1

            # Vertical
            if d < self.rows:
                corr += np.mean(grid_centered[:-d, :] * grid_centered[d:, :])
                count += 1

            # Diagonal
            if d < min(self.rows, self.cols):
                corr += np.mean(grid_centered[:-d, :-d] * grid_centered[d:, d:])
                count += 1

            correlations.append(corr / (count * variance) if count > 0 else 0)

        return distances, np.array(correlations)


class ZetaVisualizer:
    """Visualización del Game of Life con kernel zeta."""

    def __init__(self, game: ZetaGameOfLife) -> None:
        self.game = game

        # Colormap personalizado
        colors = ['#0a0a0a', '#00ff88']  # Negro -> Verde neón
        self.cmap = LinearSegmentedColormap.from_list('zeta', colors)

    def plot_state(self, title: str | None = None, ax: plt.Axes | None = None) -> plt.Axes:
        """Muestra el estado actual del grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(self.game.grid, cmap=self.cmap, interpolation='nearest')

        stats = self.game.get_statistics()
        if title is None:
            title = f"Generación {stats['generation']} | Densidad: {stats['density']:.3f}"
        ax.set_title(title)
        ax.axis('off')

        return ax

    def plot_kernel(self, size: int = 50) -> plt.Figure:
        """Visualiza el kernel zeta en 2D."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)

        modes = ['radial', 'separable', 'sum']
        titles = ['Radial: K(√(x²+y²))', 'Separable: K(x)·K(y)', 'Suma: K(x)+K(y)']

        for ax, mode, title in zip(axes, modes, titles):
            Z = np.zeros_like(X)
            for i in range(size):
                for j in range(size):
                    Z[i, j] = self.game.kernel.evaluate_2d(X[i, j], Y[i, j], mode)

            im = ax.imshow(Z, extent=[-5, 5, -5, 5], cmap='RdBu_r', origin='lower')
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)

        plt.suptitle(f'Kernel Zeta (M={self.game.kernel.M}, σ={self.game.kernel.sigma})')
        plt.tight_layout()
        return fig

    def plot_correlations(self, max_distance: int = 30) -> plt.Figure:
        """Compara correlaciones: inicial vs evolucionado vs aleatorio."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Correlaciones del estado actual
        distances, corr = self.game.analyze_correlations(max_distance)
        ax.plot(distances, corr, 'o-', label=f'Gen {self.game.generation}', linewidth=2)

        # Referencia: ruido aleatorio (correlación ~0)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Ruido aleatorio')

        # Curva teórica del kernel zeta (normalizada)
        theory = np.array([self.game.kernel.evaluate(d) for d in distances])
        theory = theory / theory[0] if theory[0] != 0 else theory
        ax.plot(distances, theory, 's--', alpha=0.7, label='Kernel teórico (normalizado)')

        ax.set_xlabel('Distancia (celdas)')
        ax.set_ylabel('Correlación')
        ax.set_title('Función de Correlación Espacial')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def animate(self, frames: int = 100, interval: int = 100, save_path: str | None = None) -> FuncAnimation:
        """Crea una animación de la evolución."""
        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(self.game.grid, cmap=self.cmap, interpolation='nearest')
        ax.axis('off')
        title = ax.set_title('')

        def update(frame) -> list:
            self.game.step()
            im.set_array(self.game.grid)
            stats = self.game.get_statistics()
            title.set_text(f"Generación {stats['generation']} | Vivas: {stats['alive_cells']} | Densidad: {stats['density']:.3f}")
            return [im, title]

        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animación guardada en: {save_path}")

        plt.show()
        return anim


def compare_random_vs_zeta(rows: int = 100, cols: int = 100, steps: int = 100) -> plt.Figure:
    """
    Compara la evolución de Game of Life con inicialización aleatoria
    vs inicialización con kernel zeta.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Inicialización aleatoria
    np.random.seed(42)
    random_grid = (np.random.random((rows, cols)) > 0.5).astype(int)

    # Inicialización zeta
    zeta_game = ZetaGameOfLife(rows, cols, M=50, sigma=0.1, threshold=0.0, seed=42)
    zeta_grid_init = zeta_game.grid.copy()

    # Mostrar estados iniciales
    axes[0, 0].imshow(random_grid, cmap='binary')
    axes[0, 0].set_title('Aleatorio - Gen 0')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(zeta_grid_init, cmap='binary')
    axes[1, 0].set_title('Zeta - Gen 0')
    axes[1, 0].axis('off')

    # Evolucionar y mostrar en diferentes generaciones
    generations = [10, 50, 100]

    # Evolución aleatoria
    current_random = random_grid.copy()
    gen = 0
    for idx, target_gen in enumerate(generations):
        while gen < target_gen:
            neighbors = np.zeros_like(current_random)
            padded = np.pad(current_random, 1, mode='wrap')
            for di in range(3):
                for dj in range(3):
                    if di == 1 and dj == 1:
                        continue
                    neighbors += padded[di:di+rows, dj:dj+cols]

            birth = (current_random == 0) & (neighbors == 3)
            survive = (current_random == 1) & ((neighbors == 2) | (neighbors == 3))
            current_random = (birth | survive).astype(int)
            gen += 1

        axes[0, idx + 1].imshow(current_random, cmap='binary')
        axes[0, idx + 1].set_title(f'Aleatorio - Gen {target_gen}')
        axes[0, idx + 1].axis('off')

    # Evolución zeta
    for idx, target_gen in enumerate(generations):
        while zeta_game.generation < target_gen:
            zeta_game.step()

        axes[1, idx + 1].imshow(zeta_game.grid, cmap='binary')
        axes[1, idx + 1].set_title(f'Zeta - Gen {target_gen}')
        axes[1, idx + 1].axis('off')

    # Estadísticas finales
    random_alive = np.sum(current_random)
    zeta_alive = np.sum(zeta_game.grid)

    plt.suptitle(f'Comparación: Aleatorio ({random_alive} vivas) vs Zeta ({zeta_alive} vivas) después de {steps} generaciones')
    plt.tight_layout()

    return fig


def demo() -> None:
    """Demostración completa del sistema."""
    print("=" * 60)
    print("ZETA GAME OF LIFE - Fase 1: Ruido Estructurado")
    print("Basado en el framework de kernels de la función zeta")
    print("=" * 60)

    # Crear juego con parámetros del paper
    print("\n1. Inicializando Game of Life con kernel zeta...")
    game = ZetaGameOfLife(
        rows=100,
        cols=100,
        M=50,           # 50 ceros de zeta
        sigma=0.1,      # Regularización
        threshold=0.0,  # Umbral en la media
        seed=42         # Reproducibilidad
    )

    stats = game.get_statistics()
    print(f"   Grid: {game.rows}x{game.cols}")
    print(f"   Ceros usados: {game.kernel.M}")
    print(f"   Células vivas iniciales: {stats['alive_cells']}")
    print(f"   Densidad inicial: {stats['density']:.3f}")

    # Visualizar
    print("\n2. Generando visualizaciones...")
    viz = ZetaVisualizer(game)

    # Estado inicial
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    viz.plot_state("Estado Inicial con Ruido Zeta Estructurado", ax1)
    fig1.savefig('zeta_gol_inicial.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_gol_inicial.png")

    # Kernel
    fig2 = viz.plot_kernel()
    fig2.savefig('zeta_kernel_2d.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_kernel_2d.png")

    # Evolucionar
    print("\n3. Evolucionando 100 generaciones...")
    game.run(100)

    stats = game.get_statistics()
    print(f"   Células vivas después: {stats['alive_cells']}")
    print(f"   Densidad final: {stats['density']:.3f}")

    # Estado final
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    viz.plot_state(f"Generación {stats['generation']}", ax3)
    fig3.savefig('zeta_gol_gen100.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_gol_gen100.png")

    # Correlaciones
    fig4 = viz.plot_correlations()
    fig4.savefig('zeta_correlations.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_correlations.png")

    # Comparación
    print("\n4. Comparando con inicialización aleatoria...")
    fig5 = compare_random_vs_zeta()
    fig5.savefig('zeta_vs_random_comparison.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_vs_random_comparison.png")

    print("\n" + "=" * 60)
    print("Demostración completada.")
    print("Archivos generados en el directorio actual.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
