"""
Zeta Game of Life - Fase 2: Kernel de Vecindario Ponderado
Basado en el framework de kernels zeta de Francisco Ruiz

En lugar del conteo binario de vecinos (Moore), usamos un kernel
ponderado por los ceros de la función zeta de Riemann.

La diferencia clave con Fase 1:
- Fase 1: Inicialización estructurada, evolución clásica
- Fase 2: Evolución con pesos zeta, la estructura persiste
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
from typing import List, Tuple, Optional
import warnings

try:
    from mpmath import zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
    warnings.warn("mpmath no disponible. Usando aproximaciones.")


def get_zeta_zeros(M: int) -> List[float]:
    """Obtiene los primeros M ceros de ζ(s)."""
    if HAS_MPMATH:
        return [float(zetazero(k).imag) for k in range(1, M + 1)]
    else:
        known = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
                 52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
                 67.079811, 69.546402, 72.067158, 75.704691, 77.144840]
        if M <= len(known):
            return known[:M]
        zeros = known.copy()
        for k in range(len(known), M):
            n = k + 1
            zeros.append(2 * np.pi * n / np.log(n + 2))
        return zeros


class ZetaNeighborhoodKernel:
    """
    Kernel de vecindario basado en ceros de zeta.

    En lugar del kernel de Moore estándar:
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]]

    Usamos:
        K(x,y) = Σ_ρ exp(-σ|γ|) * cos(γ * √(x² + y²))

    Esto introduce:
    - Pesos no uniformes basados en distancia
    - Correlaciones de largo alcance (R > 1)
    - Oscilaciones características de los ceros
    """

    def __init__(self, R: int = 3, M: int = 30, sigma: float = 0.1):
        """
        Args:
            R: Radio del vecindario (R=1 es Moore clásico)
            M: Número de ceros de zeta
            sigma: Parámetro de regularización
        """
        self.R = R
        self.M = M
        self.sigma = sigma
        self.gammas = get_zeta_zeros(M)
        self.kernel = self._build_kernel()

    def _build_kernel(self) -> np.ndarray:
        """Construye el kernel 2D discreto."""
        size = 2 * self.R + 1
        K = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                x = i - self.R
                y = j - self.R

                # Centro tiene peso 0 (no se cuenta a sí mismo)
                if x == 0 and y == 0:
                    continue

                # Distancia radial
                r = np.sqrt(x**2 + y**2)

                # Suma sobre ceros de zeta
                for gamma in self.gammas:
                    K[i, j] += np.exp(-self.sigma * abs(gamma)) * np.cos(gamma * r)

        # Normalizar para que sea comparable con Moore (suma ~8)
        total = np.sum(np.abs(K))
        if total > 0:
            K = K / total * 8

        return K

    def visualize(self) -> plt.Figure:
        """Visualiza el kernel."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Kernel zeta
        im0 = axes[0].imshow(self.kernel, cmap='RdBu_r', origin='lower')
        axes[0].set_title(f'Kernel Zeta (R={self.R}, M={self.M})')
        plt.colorbar(im0, ax=axes[0])

        # Kernel de Moore para comparación
        moore = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        im1 = axes[1].imshow(moore, cmap='RdBu_r', origin='lower')
        axes[1].set_title('Kernel Moore Clásico')
        plt.colorbar(im1, ax=axes[1])

        # Diferencia radial
        r_values = np.linspace(0, self.R * 1.5, 100)
        kernel_1d = np.zeros_like(r_values)
        for i, r in enumerate(r_values):
            if r > 0:
                for gamma in self.gammas:
                    kernel_1d[i] += np.exp(-self.sigma * abs(gamma)) * np.cos(gamma * r)

        axes[2].plot(r_values, kernel_1d, 'b-', linewidth=2)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2].axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Moore (uniforme)')
        axes[2].set_xlabel('Distancia r')
        axes[2].set_ylabel('Peso K(r)')
        axes[2].set_title('Perfil Radial del Kernel')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class ZetaWeightedGoL:
    """
    Game of Life con kernel de vecindario zeta-ponderado.

    Las reglas se modifican:
    - En lugar de contar vecinos (0-8 enteros)
    - Calculamos suma ponderada (valor continuo)
    - Los umbrales se ajustan proporcionalmente
    """

    def __init__(
        self,
        rows: int = 100,
        cols: int = 100,
        R: int = 3,
        M: int = 30,
        sigma: float = 0.1,
        birth_range: Tuple[float, float] = (2.5, 3.5),
        survive_range: Tuple[float, float] = (1.5, 3.5),
        seed: Optional[int] = None
    ):
        """
        Args:
            rows, cols: Dimensiones del grid
            R: Radio del kernel
            M: Número de ceros de zeta
            sigma: Regularización
            birth_range: Rango (min, max) para nacimiento
            survive_range: Rango (min, max) para supervivencia
            seed: Semilla para reproducibilidad
        """
        self.rows = rows
        self.cols = cols
        self.kernel_obj = ZetaNeighborhoodKernel(R, M, sigma)
        self.kernel = self.kernel_obj.kernel
        self.birth_range = birth_range
        self.survive_range = survive_range
        self.generation = 0
        self.history: List[np.ndarray] = []

        # Inicialización
        if seed is not None:
            np.random.seed(seed)

        # Usar inicialización zeta de Fase 1
        self.grid = self._zeta_init()

    def _zeta_init(self) -> np.ndarray:
        """Inicialización con ruido zeta estructurado (de Fase 1)."""
        gammas = self.kernel_obj.gammas
        M = len(gammas)

        field = np.zeros((self.rows, self.cols), dtype=complex)
        phases = np.random.uniform(0, 2 * np.pi, M)

        for i in range(self.rows):
            for j in range(self.cols):
                x = 2 * np.pi * i / self.rows
                y = 2 * np.pi * j / self.cols

                for k, gamma in enumerate(gammas):
                    gamma_y = gammas[(k + 17) % M]
                    w = np.exp(-self.kernel_obj.sigma * abs(gamma))
                    phase = gamma * x + gamma_y * y + phases[k]
                    field[i, j] += w * np.exp(1j * phase)

        real_field: np.ndarray = np.real(field)
        return (real_field > np.mean(real_field)).astype(float)  # type: ignore[return-value,no-any-return]

    def weighted_neighbors(self) -> np.ndarray:
        """
        Calcula la suma ponderada de vecinos usando convolución.

        En lugar de contar vecinos (entero 0-8),
        obtenemos un valor continuo basado en el kernel zeta.
        """
        result: np.ndarray = convolve2d(self.grid, self.kernel, mode='same', boundary='wrap')
        return result

    def step(self) -> np.ndarray:
        """
        Ejecuta un paso con reglas ponderadas.

        Reglas modificadas (análogas a B3/S23):
        - Célula muerta con peso en birth_range: nace
        - Célula viva con peso en survive_range: sobrevive
        """
        neighbors = self.weighted_neighbors()

        # Nacimiento: célula muerta con vecindario en rango de nacimiento
        birth = (self.grid == 0) & \
                (neighbors >= self.birth_range[0]) & \
                (neighbors <= self.birth_range[1])

        # Supervivencia: célula viva con vecindario en rango de supervivencia
        survive = (self.grid == 1) & \
                  (neighbors >= self.survive_range[0]) & \
                  (neighbors <= self.survive_range[1])

        self.grid = (birth | survive).astype(float)
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
        return {
            'generation': self.generation,
            'alive_cells': int(alive),
            'density': alive / (self.rows * self.cols)
        }

    def analyze_correlations(self, max_distance: int = 20) -> Tuple[np.ndarray, np.ndarray]:
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


def compare_moore_vs_zeta(rows=100, cols=100, steps=100, seed=42):
    """
    Compara evolución con kernel Moore clásico vs kernel zeta.
    """
    print("Comparando Moore clásico vs Zeta ponderado...")

    # Crear figura
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Configuración común
    np.random.seed(seed)
    init_grid = (np.random.random((rows, cols)) > 0.5).astype(float)

    # === FILA 1: Moore Clásico ===
    moore_grid = init_grid.copy()
    moore_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    axes[0, 0].imshow(moore_grid, cmap='binary')
    axes[0, 0].set_title('Moore - Gen 0')
    axes[0, 0].axis('off')

    # Evolucionar con Moore
    generations_show = [10, 50, 100]
    gen = 0

    for idx, target in enumerate(generations_show):
        while gen < target:
            neighbors = convolve2d(moore_grid, moore_kernel, mode='same', boundary='wrap')
            birth = (moore_grid == 0) & (neighbors == 3)
            survive = (moore_grid == 1) & ((neighbors == 2) | (neighbors == 3))
            moore_grid = (birth | survive).astype(float)
            gen += 1

        axes[0, idx + 1].imshow(moore_grid, cmap='binary')
        axes[0, idx + 1].set_title(f'Moore - Gen {target}')
        axes[0, idx + 1].axis('off')

    moore_final_alive = int(np.sum(moore_grid))

    # === FILA 2: Zeta Kernel con umbrales adaptativos ===
    # Calibrar umbrales basados en la distribución del kernel
    zeta_game = ZetaWeightedGoL(
        rows=rows, cols=cols,
        R=2, M=20, sigma=0.05,  # Parámetros más suaves
        birth_range=(0.8, 1.5),  # Umbrales ajustados para kernel zeta
        survive_range=(0.3, 2.0),
        seed=seed
    )

    axes[1, 0].imshow(zeta_game.grid, cmap='binary')
    axes[1, 0].set_title('Zeta - Gen 0')
    axes[1, 0].axis('off')

    for idx, target in enumerate(generations_show):
        while zeta_game.generation < target:
            zeta_game.step()

        axes[1, idx + 1].imshow(zeta_game.grid, cmap='binary')
        axes[1, idx + 1].set_title(f'Zeta - Gen {target}')
        axes[1, idx + 1].axis('off')

    zeta_final_alive = int(np.sum(zeta_game.grid))

    # === FILA 3: Análisis ===
    # Kernel visualization
    axes[2, 0].imshow(zeta_game.kernel_obj.kernel, cmap='RdBu_r', origin='lower')
    axes[2, 0].set_title('Kernel Zeta 2D')
    axes[2, 0].axis('off')

    # Histograma de pesos de vecinos
    neighbors_zeta = zeta_game.weighted_neighbors()
    axes[2, 1].hist(neighbors_zeta.flatten(), bins=50, alpha=0.7, color='blue', label='Zeta')
    axes[2, 1].axvline(x=2.5, color='green', linestyle='--', label='Birth min')
    axes[2, 1].axvline(x=3.5, color='red', linestyle='--', label='Birth max')
    axes[2, 1].set_title('Distribución de Pesos')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].set_xlabel('Suma ponderada')

    # Correlaciones
    dist, corr_zeta = zeta_game.analyze_correlations(30)
    axes[2, 2].plot(dist, corr_zeta, 'b-o', label='Zeta Gen 100')
    axes[2, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2, 2].set_title('Correlaciones Espaciales')
    axes[2, 2].set_xlabel('Distancia')
    axes[2, 2].set_ylabel('Correlación')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # Texto resumen
    axes[2, 3].axis('off')
    summary = f"""
    RESUMEN FASE 2
    ═══════════════════════════

    Moore Clásico:
    • Kernel: 3x3 uniforme
    • Células vivas: {moore_final_alive}
    • Densidad: {moore_final_alive/(rows*cols):.3f}

    Zeta Ponderado:
    • Kernel: 7x7 (R=3)
    • M = 30 ceros de ζ(s)
    • σ = 0.1
    • Células vivas: {zeta_final_alive}
    • Densidad: {zeta_final_alive/(rows*cols):.3f}

    Diferencia: {((zeta_final_alive - moore_final_alive) / moore_final_alive * 100):+.1f}%
    """
    axes[2, 3].text(0.1, 0.5, summary, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[2, 3].transAxes)

    plt.suptitle(f'Fase 2: Moore vs Zeta Kernel ({steps} generaciones)', fontsize=14)
    plt.tight_layout()

    return fig, zeta_game


def analyze_parameter_sensitivity(rows=80, cols=80, steps=50):
    """
    Analiza cómo los parámetros del kernel zeta afectan la evolución.
    """
    print("Analizando sensibilidad a parámetros...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Umbrales adaptativos para el kernel zeta
    birth_range = (0.8, 1.5)
    survive_range = (0.3, 2.0)

    # Variar sigma (regularización)
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.5]
    densities_sigma = []

    for sigma in sigmas:
        game = ZetaWeightedGoL(rows=rows, cols=cols, R=2, M=20, sigma=sigma,
                               birth_range=birth_range, survive_range=survive_range, seed=42)
        game.run(steps)
        densities_sigma.append(game.get_statistics()['density'])

    axes[0, 0].plot(sigmas, densities_sigma, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('σ (regularización)')
    axes[0, 0].set_ylabel('Densidad final')
    axes[0, 0].set_title('Efecto de σ')
    axes[0, 0].grid(True, alpha=0.3)

    # Variar M (número de ceros)
    Ms = [5, 10, 20, 30, 50]
    densities_M = []

    for M in Ms:
        game = ZetaWeightedGoL(rows=rows, cols=cols, R=2, M=M, sigma=0.05,
                               birth_range=birth_range, survive_range=survive_range, seed=42)
        game.run(steps)
        densities_M.append(game.get_statistics()['density'])

    axes[0, 1].plot(Ms, densities_M, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('M (número de ceros)')
    axes[0, 1].set_ylabel('Densidad final')
    axes[0, 1].set_title('Efecto de M')
    axes[0, 1].grid(True, alpha=0.3)

    # Variar R (radio)
    Rs = [1, 2, 3, 4, 5]
    densities_R = []

    for R in Rs:
        game = ZetaWeightedGoL(rows=rows, cols=cols, R=R, M=20, sigma=0.05,
                               birth_range=birth_range, survive_range=survive_range, seed=42)
        game.run(steps)
        densities_R.append(game.get_statistics()['density'])

    axes[0, 2].plot(Rs, densities_R, 'ro-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('R (radio del kernel)')
    axes[0, 2].set_ylabel('Densidad final')
    axes[0, 2].set_title('Efecto de R')
    axes[0, 2].grid(True, alpha=0.3)

    # Visualizar kernels con diferentes parámetros
    params = [
        (2, 10, 0.1, 'R=2, M=10'),
        (3, 30, 0.1, 'R=3, M=30'),
        (5, 50, 0.05, 'R=5, M=50')
    ]

    for ax, (R, M, sigma, title) in zip(axes[1], params):
        kernel_obj = ZetaNeighborhoodKernel(R=R, M=M, sigma=sigma)
        im = ax.imshow(kernel_obj.kernel, cmap='RdBu_r', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle('Sensibilidad a Parámetros del Kernel Zeta', fontsize=14)
    plt.tight_layout()

    return fig


def demo_fase2():
    """Demostración completa de Fase 2."""
    print("=" * 60)
    print("ZETA GAME OF LIFE - Fase 2: Kernel de Vecindario Ponderado")
    print("=" * 60)

    # 1. Visualizar el kernel
    print("\n1. Visualizando kernel zeta...")
    kernel_obj = ZetaNeighborhoodKernel(R=3, M=30, sigma=0.1)
    fig1 = kernel_obj.visualize()
    fig1.savefig('zeta_kernel_fase2.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_kernel_fase2.png")

    # 2. Comparar Moore vs Zeta
    print("\n2. Comparando Moore clásico vs Zeta ponderado...")
    fig2, game = compare_moore_vs_zeta(rows=100, cols=100, steps=100, seed=42)
    fig2.savefig('zeta_vs_moore_comparison.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_vs_moore_comparison.png")

    # 3. Análisis de sensibilidad
    print("\n3. Analizando sensibilidad a parámetros...")
    fig3 = analyze_parameter_sensitivity()
    fig3.savefig('zeta_parameter_sensitivity.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_parameter_sensitivity.png")

    # 4. Estado final detallado
    print("\n4. Generando estado final detallado...")
    fig4, ax = plt.subplots(figsize=(12, 12))

    # Reiniciar y correr más generaciones
    game_long = ZetaWeightedGoL(
        rows=150, cols=150, R=2, M=20, sigma=0.05,
        birth_range=(0.8, 1.5), survive_range=(0.3, 2.0), seed=123
    )
    game_long.run(200)

    cmap = LinearSegmentedColormap.from_list('zeta', ['#0a0a0a', '#00ff88'])
    ax.imshow(game_long.grid, cmap=cmap, interpolation='nearest')
    stats = game_long.get_statistics()
    ax.set_title(f'Zeta GoL - Gen {stats["generation"]} | Densidad: {stats["density"]:.3f}')
    ax.axis('off')

    fig4.savefig('zeta_gol_fase2_final.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_gol_fase2_final.png")

    print("\n" + "=" * 60)
    print("Fase 2 completada.")
    print("=" * 60)

    return game


if __name__ == "__main__":
    demo_fase2()
