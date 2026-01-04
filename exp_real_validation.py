# -*- coding: utf-8 -*-
"""
Validacion con sistemas REALES (no simplificados).

Compara ZETA vs RANDOM vs UNIFORM usando:
- HierarchicalSimulation real
- ZetaOrganism real (con ForceField modificable)

Metricas:
- Entropia de Shannon de distribucion arquetipal
- Dimension de correlacion del atractor
- Entropia espectral

Fecha: 2026-01-03
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from scipy.signal import welch
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Importar sistemas REALES
from hierarchical_simulation import HierarchicalSimulation, SimulationConfig
from zeta_organism import ZetaOrganism
from force_field import ForceField, get_zeta_zeros
from zeta_psyche import Archetype


# =============================================================================
# FRECUENCIAS DE PRUEBA
# =============================================================================

ZETA_ZEROS = np.array(get_zeta_zeros(15))
RANDOM_FREQS = np.random.RandomState(42).uniform(10, 100, 15)
UNIFORM_FREQS = np.linspace(14, 100, 15)


# =============================================================================
# FORCEFIELDS CON DIFERENTES FRECUENCIAS
# =============================================================================

class ModifiableForceField(ForceField):
    """ForceField con frecuencias configurables."""

    def __init__(self, grid_size: int = 64, frequencies: np.ndarray = None,
                 sigma: float = 0.1, kernel_radius: int = 7):
        # Guardar frecuencias antes de llamar a super
        self._custom_frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.M = len(self._custom_frequencies)

        # Llamar super pero con M correcto
        super(ForceField, self).__init__()  # Saltar ForceField.__init__

        self.grid_size = grid_size
        self.sigma = sigma
        self.kernel_radius = kernel_radius

        # Crear kernel con frecuencias custom
        self.kernel = self._create_custom_kernel()

        # Filtros Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _create_custom_kernel(self):
        """Crea kernel con frecuencias custom."""
        gammas = self._custom_frequencies
        size = 2 * self.kernel_radius + 1

        kernel = np.zeros((size, size))
        center = self.kernel_radius

        for i in range(size):
            for j in range(size):
                r = np.sqrt((i - center)**2 + (j - center)**2)
                for gamma in gammas:
                    weight = np.exp(-self.sigma * abs(gamma))
                    kernel[i, j] += weight * np.cos(gamma * r)

        # Normalizar
        kernel = kernel / (np.abs(kernel).sum() + 1e-8)

        return torch.nn.Parameter(
            torch.tensor(kernel, dtype=torch.float32).view(1, 1, size, size),
            requires_grad=False
        )


class ModifiableZetaOrganism(ZetaOrganism):
    """ZetaOrganism con frecuencias configurables."""

    def __init__(self, frequencies: np.ndarray = None, **kwargs):
        # Inicializar sin crear ForceField
        super(ZetaOrganism, self).__init__()

        self.grid_size = kwargs.get('grid_size', 64)
        self.n_cells = kwargs.get('n_cells', 100)
        self.state_dim = kwargs.get('state_dim', 32)
        self.fi_threshold = kwargs.get('fi_threshold', 0.7)
        self.equilibrium_factor = kwargs.get('equilibrium_factor', 0.5)

        # Componentes con frecuencias custom
        freqs = frequencies if frequencies is not None else ZETA_ZEROS
        self.force_field = ModifiableForceField(
            self.grid_size,
            frequencies=freqs,
            sigma=kwargs.get('sigma', 0.1)
        )

        # Importar BehaviorEngine y OrganismCell
        from behavior_engine import BehaviorEngine
        from organism_cell import OrganismCell

        hidden_dim = kwargs.get('hidden_dim', 64)
        M = len(freqs)
        sigma = kwargs.get('sigma', 0.1)

        self.behavior = BehaviorEngine(self.state_dim, hidden_dim)
        self.cell_module = OrganismCell(self.state_dim, hidden_dim, M, sigma)

        # Estado
        self.cells = []
        self.energy_grid = torch.zeros(1, 1, self.grid_size, self.grid_size)
        self.role_grid = torch.zeros(1, 1, self.grid_size, self.grid_size)
        self.history = []

        # Crear celulas
        self._create_cells(seed_fi=False)


# =============================================================================
# HIERARCHICAL CON MODULACION TEMPORAL
# =============================================================================

class ModulatedHierarchicalSimulation(HierarchicalSimulation):
    """HierarchicalSimulation con modulacion temporal externa."""

    def __init__(self, frequencies: np.ndarray = None, sigma: float = 0.05,
                 config=None):
        super().__init__(config)
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = sigma
        self.time = 0.0

    def _compute_modulation(self) -> float:
        """Computa factor de modulacion basado en frecuencias."""
        mod = 0.0
        for gamma in self.frequencies:
            weight = np.exp(-self.sigma * abs(gamma))
            mod += weight * np.cos(gamma * self.time)
        # Normalizar a [0, 1]
        return 0.5 + 0.5 * np.tanh(mod / 5.0)

    def step(self):
        """Step con modulacion temporal aplicada."""
        self.time += 0.1

        # Obtener factor de modulacion
        mod_factor = self._compute_modulation()

        # Aplicar modulacion a la fuerza top-down
        original_strength = self.config.top_down_strength
        self.config.top_down_strength = original_strength * mod_factor

        # Step normal
        metrics = super().step()

        # Restaurar
        self.config.top_down_strength = original_strength

        return metrics


# =============================================================================
# METRICAS
# =============================================================================

def shannon_entropy(probs: np.ndarray) -> float:
    """Entropia de Shannon."""
    probs = np.array(probs).flatten()
    probs = probs[probs > 1e-10]
    if len(probs) == 0:
        return 0.0
    probs = probs / probs.sum()
    return -np.sum(probs * np.log2(probs))


def correlation_dimension(points: np.ndarray, r_min: float = 0.01,
                          r_max: float = 1.0, n_r: int = 15) -> float:
    """Dimension de correlacion usando Grassberger-Procaccia."""
    if len(points) < 10:
        return 0.0

    # Normalizar puntos
    points = (points - points.min()) / (points.max() - points.min() + 1e-10)

    distances = pdist(points)
    if len(distances) == 0:
        return 0.0

    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    n = len(points)
    n_pairs = n * (n - 1) / 2

    log_r, log_c = [], []
    for r in r_values:
        count = np.sum(distances < r)
        if count > 0:
            c = count / n_pairs
            log_r.append(np.log(r))
            log_c.append(np.log(c))

    if len(log_r) < 5:
        return 0.0

    slope, _, _, _, _ = linregress(log_r, log_c)
    return slope


def spectral_entropy(series: np.ndarray) -> float:
    """Entropia espectral normalizada."""
    if len(series) < 10:
        return 0.0

    freqs, psd = welch(series, fs=10, nperseg=min(len(series), 64))

    if psd.sum() < 1e-10:
        return 0.0

    psd_norm = psd / psd.sum()
    psd_norm = psd_norm[psd_norm > 1e-10]

    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return entropy / np.log2(len(psd_norm)) if len(psd_norm) > 1 else 0.0


# =============================================================================
# EXPERIMENTOS
# =============================================================================

@dataclass
class ValidationResult:
    """Resultado de validacion."""
    system: str
    modulation: str
    entropy_mean: float
    entropy_std: float
    corr_dim: float
    spectral_ent: float


def run_hierarchical_experiment(frequencies: np.ndarray,
                                 modulation_name: str,
                                 n_steps: int = 100,
                                 n_trials: int = 3) -> ValidationResult:
    """Ejecuta experimento con HierarchicalSimulation real."""

    all_entropies = []  # Entropias a lo largo del tiempo
    all_trajectories = []  # Trayectorias arquetipales

    for trial in range(n_trials):
        config = SimulationConfig(
            n_cells=50,
            n_clusters=4,
            n_steps=n_steps,
            grid_size=32
        )

        sim = ModulatedHierarchicalSimulation(
            frequencies=frequencies,
            config=config
        )
        sim.initialize()

        trial_entropies = []
        trial_archetypes = []

        # Correr simulacion y recolectar datos en cada paso
        for step in range(n_steps):
            sim.step()

            # Recolectar distribucion arquetipal de CELULAS (no solo organismo)
            cell_dominants = [c.psyche.dominant.value for c in sim.cells]
            cell_counts = np.bincount(cell_dominants, minlength=4)
            cell_probs = cell_counts / cell_counts.sum()

            entropy = shannon_entropy(cell_probs)
            trial_entropies.append(entropy)
            trial_archetypes.append(cell_probs.copy())

        all_entropies.extend(trial_entropies)
        all_trajectories.extend(trial_archetypes)

    # Calcular metricas
    entropy_mean = np.mean(all_entropies) if all_entropies else 0.0
    entropy_std = np.std(all_entropies) if all_entropies else 0.0

    # Dimension de correlacion del atractor
    if len(all_trajectories) >= 20:
        trajectory = np.array(all_trajectories)
        corr_dim = correlation_dimension(trajectory)
    else:
        corr_dim = 0.0

    # Entropia espectral (de la entropia a lo largo del tiempo)
    if len(all_entropies) >= 20:
        spec_ent = spectral_entropy(np.array(all_entropies))
    else:
        spec_ent = 0.0

    return ValidationResult(
        system="Hierarchical",
        modulation=modulation_name,
        entropy_mean=entropy_mean,
        entropy_std=entropy_std,
        corr_dim=corr_dim,
        spectral_ent=spec_ent
    )


def run_organism_experiment(frequencies: np.ndarray,
                            modulation_name: str,
                            n_steps: int = 100,
                            n_trials: int = 3) -> ValidationResult:
    """Ejecuta experimento con ZetaOrganism real."""

    all_entropies = []
    all_states = []

    for trial in range(n_trials):
        organism = ModifiableZetaOrganism(
            frequencies=frequencies,
            grid_size=32,
            n_cells=50,
            state_dim=16
        )
        organism.initialize(seed_fi=True)

        trial_entropies = []
        trial_states = []

        # Correr simulacion y recolectar datos en cada paso
        for step in range(n_steps):
            organism.step()

            # Extraer distribucion de roles y energia como proxy
            energies = np.array([c.energy for c in organism.cells])

            # Crear distribucion "arquetipal" basada en energia
            energy_hist, _ = np.histogram(energies, bins=4, range=(0, 1))
            if energy_hist.sum() > 0:
                probs = energy_hist / energy_hist.sum()
                trial_entropies.append(shannon_entropy(probs))
                trial_states.append(probs.copy())

        all_entropies.extend(trial_entropies)
        all_states.extend(trial_states)

    # Calcular metricas
    entropy_mean = np.mean(all_entropies) if all_entropies else 0.0
    entropy_std = np.std(all_entropies) if all_entropies else 0.0

    # Dimension de correlacion
    if len(all_states) >= 20:
        trajectory = np.array(all_states)
        corr_dim = correlation_dimension(trajectory)
    else:
        corr_dim = 0.0

    # Entropia espectral
    if len(all_entropies) >= 20:
        spec_ent = spectral_entropy(np.array(all_entropies))
    else:
        spec_ent = 0.0

    return ValidationResult(
        system="ZetaOrganism",
        modulation=modulation_name,
        entropy_mean=entropy_mean,
        entropy_std=entropy_std,
        corr_dim=corr_dim,
        spectral_ent=spec_ent
    )


def run_full_validation():
    """Ejecuta validacion completa con sistemas reales."""

    print("=" * 70)
    print("  VALIDACION CON SISTEMAS REALES")
    print("=" * 70)

    modulations = [
        ("ZETA", ZETA_ZEROS),
        ("RANDOM", RANDOM_FREQS),
        ("UNIFORM", UNIFORM_FREQS)
    ]

    results: List[ValidationResult] = []

    # Hierarchical
    print("\n--- HierarchicalSimulation ---")
    for name, freqs in modulations:
        print(f"  Probando {name}...")
        result = run_hierarchical_experiment(freqs, name, n_steps=80, n_trials=3)
        results.append(result)
        print(f"    Entropia: {result.entropy_mean:.4f} +/- {result.entropy_std:.4f}")

    # ZetaOrganism
    print("\n--- ZetaOrganism ---")
    for name, freqs in modulations:
        print(f"  Probando {name}...")
        result = run_organism_experiment(freqs, name, n_steps=80, n_trials=3)
        results.append(result)
        print(f"    Entropia: {result.entropy_mean:.4f} +/- {result.entropy_std:.4f}")

    # Mostrar resultados
    print("\n" + "=" * 70)
    print("  RESULTADOS FINALES")
    print("=" * 70)

    print("\n### Entropia de Shannon ###")
    print(f"{'Sistema':<15} {'ZETA':>10} {'RANDOM':>10} {'UNIFORM':>10} {'Orden':<25}")
    print("-" * 70)

    for system in ["Hierarchical", "ZetaOrganism"]:
        sys_results = [r for r in results if r.system == system]
        zeta = next((r.entropy_mean for r in sys_results if r.modulation == "ZETA"), 0)
        rand = next((r.entropy_mean for r in sys_results if r.modulation == "RANDOM"), 0)
        unif = next((r.entropy_mean for r in sys_results if r.modulation == "UNIFORM"), 0)

        # Determinar orden
        vals = [("ZETA", zeta), ("RANDOM", rand), ("UNIFORM", unif)]
        sorted_vals = sorted(vals, key=lambda x: x[1])
        order = " < ".join([v[0] for v in sorted_vals])

        # Verificar si ZETA esta en el medio
        if sorted_vals[1][0] == "ZETA":
            order += " [OK]"
        else:
            order += " [--]"

        print(f"{system:<15} {zeta:>10.4f} {rand:>10.4f} {unif:>10.4f} {order:<25}")

    print("\n### Dimension de Correlacion ###")
    print(f"{'Sistema':<15} {'ZETA':>10} {'RANDOM':>10} {'UNIFORM':>10}")
    print("-" * 55)

    for system in ["Hierarchical", "ZetaOrganism"]:
        sys_results = [r for r in results if r.system == system]
        zeta = next((r.corr_dim for r in sys_results if r.modulation == "ZETA"), 0)
        rand = next((r.corr_dim for r in sys_results if r.modulation == "RANDOM"), 0)
        unif = next((r.corr_dim for r in sys_results if r.modulation == "UNIFORM"), 0)
        print(f"{system:<15} {zeta:>10.4f} {rand:>10.4f} {unif:>10.4f}")

    print("\n### Entropia Espectral ###")
    print(f"{'Sistema':<15} {'ZETA':>10} {'RANDOM':>10} {'UNIFORM':>10}")
    print("-" * 55)

    for system in ["Hierarchical", "ZetaOrganism"]:
        sys_results = [r for r in results if r.system == system]
        zeta = next((r.spectral_ent for r in sys_results if r.modulation == "ZETA"), 0)
        rand = next((r.spectral_ent for r in sys_results if r.modulation == "RANDOM"), 0)
        unif = next((r.spectral_ent for r in sys_results if r.modulation == "UNIFORM"), 0)
        print(f"{system:<15} {zeta:>10.4f} {rand:>10.4f} {unif:>10.4f}")

    # Guardar grafico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    systems = ["Hierarchical", "ZetaOrganism"]
    mods = ["ZETA", "RANDOM", "UNIFORM"]
    colors = {'ZETA': 'blue', 'RANDOM': 'red', 'UNIFORM': 'green'}

    # Entropia
    ax = axes[0]
    x = np.arange(len(systems))
    width = 0.25
    for i, mod in enumerate(mods):
        vals = [next((r.entropy_mean for r in results
                     if r.system == sys and r.modulation == mod), 0)
                for sys in systems]
        ax.bar(x + i*width, vals, width, label=mod, color=colors[mod], alpha=0.7)
    ax.set_ylabel('Entropia Shannon')
    ax.set_title('Entropia (mayor = mas caotico)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(systems)
    ax.legend()

    # Correlacion
    ax = axes[1]
    for i, mod in enumerate(mods):
        vals = [next((r.corr_dim for r in results
                     if r.system == sys and r.modulation == mod), 0)
                for sys in systems]
        ax.bar(x + i*width, vals, width, label=mod, color=colors[mod], alpha=0.7)
    ax.set_ylabel('Dimension Correlacion')
    ax.set_title('Dim. Correlacion (mayor = mas complejo)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(systems)
    ax.legend()

    # Espectral
    ax = axes[2]
    for i, mod in enumerate(mods):
        vals = [next((r.spectral_ent for r in results
                     if r.system == sys and r.modulation == mod), 0)
                for sys in systems]
        ax.bar(x + i*width, vals, width, label=mod, color=colors[mod], alpha=0.7)
    ax.set_ylabel('Entropia Espectral')
    ax.set_title('Entropia Espectral (intermedio = borde)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(systems)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"real_validation_{timestamp}.png", dpi=150)
    plt.close()

    print(f"\nGrafico guardado: real_validation_{timestamp}.png")

    return results


if __name__ == "__main__":
    run_full_validation()
