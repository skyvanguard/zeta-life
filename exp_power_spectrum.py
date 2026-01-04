# -*- coding: utf-8 -*-
"""
Validacion Teorica: Espectro de Potencia de Oscilaciones

Hipotesis: ZETA produce espectro de potencia INTERMEDIO
- UNIFORM: Picos agudos (periodicidad), entropia espectral BAJA
- RANDOM:  Espectro plano (ruido), entropia espectral ALTA
- ZETA:    Estructura compleja, entropia espectral INTERMEDIA

Metricas:
1. Entropia espectral (distribucion de potencia)
2. Pendiente espectral (1/f^beta)
3. Prominencia de picos

Fecha: 2026-01-03
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from scipy import signal
from scipy.stats import linregress

# Frecuencias
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
])

np.random.seed(42)


@dataclass
class PowerSpectrumResult:
    """Resultado del analisis de espectro de potencia."""
    modulation_type: str
    system_name: str
    spectral_entropy: float       # Entropia del espectro
    spectral_slope: float         # Pendiente 1/f^beta
    peak_prominence: float        # Prominencia promedio de picos
    n_significant_peaks: int      # Numero de picos significativos
    frequencies: List[float]
    power_spectrum: List[float]


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


def compute_power_spectrum(time_series: np.ndarray, fs: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el espectro de potencia usando el metodo de Welch.

    Returns:
        (frequencies, power_spectrum)
    """
    # Remover tendencia
    detrended = signal.detrend(time_series)

    # Welch para estimacion espectral robusta
    freqs, psd = signal.welch(detrended, fs=fs, nperseg=min(256, len(detrended)//2))

    return freqs, psd


def spectral_entropy(psd: np.ndarray) -> float:
    """
    Calcula la entropia espectral.
    H = -sum(p * log(p)) donde p = psd_normalizado

    Baja entropia = potencia concentrada (periodico)
    Alta entropia = potencia dispersa (ruido)
    """
    # Normalizar a distribucion de probabilidad
    psd_norm = psd / (psd.sum() + 1e-10)
    psd_norm = psd_norm[psd_norm > 1e-10]

    # Entropia
    entropy = -np.sum(psd_norm * np.log2(psd_norm))

    # Normalizar por entropia maxima (log2(N))
    max_entropy = np.log2(len(psd_norm))

    return entropy / max_entropy if max_entropy > 0 else 0


def spectral_slope(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Calcula la pendiente espectral (beta en 1/f^beta).

    beta ~ 0: ruido blanco
    beta ~ 1: ruido rosa (1/f)
    beta ~ 2: ruido browniano (1/f^2)
    """
    # Filtrar frecuencias validas (evitar f=0)
    valid = (freqs > 0.1) & (psd > 1e-10)

    if np.sum(valid) < 3:
        return 0.0

    log_f = np.log10(freqs[valid])
    log_psd = np.log10(psd[valid])

    # Regresion lineal
    slope, _, _, _, _ = linregress(log_f, log_psd)

    return -slope  # Negativo porque P ~ 1/f^beta


def find_peaks_prominence(psd: np.ndarray) -> Tuple[float, int]:
    """
    Encuentra picos y calcula prominencia promedio.

    Returns:
        (prominencia_promedio, n_picos)
    """
    # Encontrar picos
    peaks, properties = signal.find_peaks(psd, prominence=psd.std() * 0.5)

    if len(peaks) == 0:
        return 0.0, 0

    prominences = properties['prominences']

    # Normalizar por potencia media
    mean_prominence = np.mean(prominences) / (psd.mean() + 1e-10)

    return mean_prominence, len(peaks)


# =============================================================================
# SISTEMAS SIMPLIFICADOS
# =============================================================================

class SpectrumHierarchicalSystem:
    """Sistema jerarquico para analisis espectral."""

    def __init__(self, n_cells: int = 25, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05
        self.time = 0.0

        self.archetypes = np.random.rand(n_cells, 4)
        self.archetypes = self.archetypes / self.archetypes.sum(axis=1, keepdims=True)

        self.trajectory = []

    def step(self, dt: float = 0.1):
        self.time += dt

        mod = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_norm = 0.5 + 0.5 * np.tanh(mod / 5.0)

        global_arch = self.archetypes.mean(axis=0)

        for i in range(self.n_cells):
            diff = global_arch - self.archetypes[i]
            self.archetypes[i] += 0.2 * mod_norm * diff

        for i in range(self.n_cells):
            j = np.random.randint(0, self.n_cells)
            if i != j:
                diff = self.archetypes[j] - self.archetypes[i]
                self.archetypes[i] += 0.1 * (1 - mod_norm) * diff

        noise = np.random.randn(self.n_cells, 4) * 0.02 * (1 + 0.5 * (1 - mod_norm))
        self.archetypes += noise

        self.archetypes = np.clip(self.archetypes, 0.01, None)
        self.archetypes = self.archetypes / self.archetypes.sum(axis=1, keepdims=True)

        # Guardar observable: primer arquetipo global
        self.trajectory.append(global_arch[0])

    def get_time_series(self) -> np.ndarray:
        return np.array(self.trajectory)


class SpectrumZetaOrganism:
    """ZetaOrganism para analisis espectral."""

    def __init__(self, n_cells: int = 25, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.grid_size = 32
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05
        self.time = 0.0

        self.positions = np.random.rand(n_cells, 2) * self.grid_size
        self.energies = np.random.rand(n_cells) * 0.5 + 0.1
        self.roles = np.zeros(n_cells)
        self.roles[0] = 1

        self.trajectory = []

    def step(self, dt: float = 0.1):
        self.time += dt

        mod = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_norm = 0.5 + 0.5 * np.tanh(mod / 5.0)

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

        noise = np.random.randn(self.n_cells, 2) * 0.3 * (1 + 0.3 * (1 - mod_norm))
        self.positions += noise

        self.energies *= 0.99
        self.energies += np.random.rand(self.n_cells) * 0.01
        self.energies = np.clip(self.energies, 0.01, 1.0)

        self.positions = np.clip(self.positions, 0, self.grid_size)

        # Guardar observable: dispersion espacial
        dispersion = self.positions.std()
        self.trajectory.append(dispersion)

    def get_time_series(self) -> np.ndarray:
        return np.array(self.trajectory)


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_power_spectrum_experiment():
    """Ejecuta experimento de espectro de potencia."""

    print("=" * 70)
    print("VALIDACION TEORICA: ESPECTRO DE POTENCIA")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Hipotesis: ZETA produce entropia espectral INTERMEDIA")
    print("  UNIFORM: Picos agudos, entropia BAJA (periodico)")
    print("  RANDOM:  Espectro plano, entropia ALTA (ruido)")
    print("  ZETA:    Estructura compleja, entropia INTERMEDIA")
    print()

    modulation_types = ["ZETA", "RANDOM", "UNIFORM"]
    n_steps = 1000  # Serie larga para buen espectro
    n_trials = 3
    fs = 10.0  # Frecuencia de muestreo

    systems = [
        ("Hierarchical", SpectrumHierarchicalSystem, {"n_cells": 25}),
        ("ZetaOrganism", SpectrumZetaOrganism, {"n_cells": 25})
    ]

    all_results = {}

    for sys_name, sys_class, kwargs in systems:
        print(f"\n{'='*60}")
        print(f"SISTEMA: {sys_name}")
        print(f"{'='*60}")

        all_results[sys_name] = {}

        for mod_type in modulation_types:
            print(f"\n  Modulacion: {mod_type}")

            entropies = []
            slopes = []
            prominences = []
            n_peaks_list = []
            best_freqs = None
            best_psd = None

            for trial in range(n_trials):
                np.random.seed(42 + trial * 100)

                freqs_mod = generate_frequencies(mod_type)
                system = sys_class(frequencies=freqs_mod, **kwargs)

                # Generar serie temporal
                for _ in range(n_steps):
                    system.step(dt=1.0/fs)

                time_series = system.get_time_series()

                # Calcular espectro
                spec_freqs, psd = compute_power_spectrum(time_series, fs=fs)

                # Metricas
                h_spec = spectral_entropy(psd)
                beta = spectral_slope(spec_freqs, psd)
                prom, n_peaks = find_peaks_prominence(psd)

                entropies.append(h_spec)
                slopes.append(beta)
                prominences.append(prom)
                n_peaks_list.append(n_peaks)

                if best_freqs is None:
                    best_freqs = spec_freqs
                    best_psd = psd

            result = PowerSpectrumResult(
                modulation_type=mod_type,
                system_name=sys_name,
                spectral_entropy=np.mean(entropies),
                spectral_slope=np.mean(slopes),
                peak_prominence=np.mean(prominences),
                n_significant_peaks=int(np.mean(n_peaks_list)),
                frequencies=best_freqs.tolist() if best_freqs is not None else [],
                power_spectrum=best_psd.tolist() if best_psd is not None else []
            )

            all_results[sys_name][mod_type] = result

            print(f"    H_espectral: {result.spectral_entropy:.4f}")
            print(f"    Pendiente:   {result.spectral_slope:.3f} (1/f^beta)")
            print(f"    Prominencia: {result.peak_prominence:.3f}")
            print(f"    N picos:     {result.n_significant_peaks}")

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

        entropies = {}
        for mod in modulation_types:
            entropies[mod] = all_results[sys_name][mod].spectral_entropy

        # Ordenar por entropia
        sorted_h = sorted(entropies.items(), key=lambda x: x[1])

        print("  Ranking de entropia espectral:")
        labels = ["(menor/ordenado)", "(medio/critico)", "(mayor/caotico)"]
        for i, (name, h) in enumerate(sorted_h):
            print(f"    {i+1}. {name:8s}: H = {h:.4f} {labels[i]}")

        # Verificar hipotesis
        total_count += 1
        zeta_rank = [i for i, (n, _) in enumerate(sorted_h) if n == "ZETA"][0]

        if zeta_rank == 1:
            print(f"\n  [OK] HIPOTESIS CONFIRMADA: ZETA tiene entropia espectral INTERMEDIA")
            edge_of_chaos_count += 1
        else:
            h_min = sorted_h[0][1]
            h_max = sorted_h[2][1]
            h_zeta = entropies["ZETA"]

            if h_max - h_min > 0.01:
                rel_pos = (h_zeta - h_min) / (h_max - h_min)
                if 0.2 < rel_pos < 0.8:
                    print(f"\n  [~] ZETA en zona intermedia (posicion: {rel_pos:.2f})")
                    edge_of_chaos_count += 1
                else:
                    print(f"\n  [?] ZETA no en posicion intermedia (pos: {rel_pos:.2f})")
            else:
                print(f"\n  [?] Variacion insuficiente")

    # =========================================================================
    # ANALISIS DE PENDIENTE ESPECTRAL
    # =========================================================================

    print("\n" + "-" * 50)
    print("ANALISIS DE PENDIENTE ESPECTRAL (1/f^beta)")
    print("-" * 50)
    print("  beta ~ 0: ruido blanco")
    print("  beta ~ 1: ruido rosa (1/f) - criticidad")
    print("  beta ~ 2: ruido browniano")

    for sys_name in all_results:
        print(f"\n  {sys_name}:")
        for mod in modulation_types:
            beta = all_results[sys_name][mod].spectral_slope
            if beta < 0.5:
                tipo = "blanco"
            elif beta < 1.5:
                tipo = "rosa (critico)"
            else:
                tipo = "browniano"
            print(f"    {mod:8s}: beta = {beta:.2f} ({tipo})")

    # =========================================================================
    # RESUMEN
    # =========================================================================

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(f"\n  Sistemas con ZETA en entropia espectral intermedia: {edge_of_chaos_count}/{total_count}")

    if edge_of_chaos_count == total_count:
        print("\n  *** HIPOTESIS VALIDADA: ENTROPIA ESPECTRAL INTERMEDIA ***")
    elif edge_of_chaos_count > 0:
        print("\n  ** HIPOTESIS PARCIALMENTE VALIDADA **")
    else:
        print("\n  X HIPOTESIS NO VALIDADA")

    # =========================================================================
    # VISUALIZACION
    # =========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Espectro de Potencia de Oscilaciones\n' +
                 'Hipotesis: ZETA produce entropia espectral intermedia',
                 fontsize=13, fontweight='bold')

    colors = {"ZETA": "blue", "RANDOM": "red", "UNIFORM": "gray"}

    # Fila 1: Espectros de potencia
    for idx, sys_name in enumerate(all_results):
        ax = axes[0, idx]

        for mod_type in modulation_types:
            result = all_results[sys_name][mod_type]
            if result.frequencies and result.power_spectrum:
                freqs = np.array(result.frequencies)
                psd = np.array(result.power_spectrum)
                valid = psd > 0
                ax.semilogy(freqs[valid], psd[valid], color=colors[mod_type],
                           label=f'{mod_type} (H={result.spectral_entropy:.3f})',
                           linewidth=1.5, alpha=0.8)

        ax.set_title(f'{sys_name}: Espectro de Potencia')
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Densidad espectral')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Comparacion de entropia espectral
    ax_ent = axes[0, 2]
    x = np.arange(len(all_results))
    width = 0.25

    for i, mod_type in enumerate(modulation_types):
        vals = [all_results[sys][mod_type].spectral_entropy for sys in all_results]
        ax_ent.bar(x + i*width, vals, width, label=mod_type,
                  color=colors[mod_type], alpha=0.8)

    ax_ent.set_ylabel('Entropia espectral (normalizada)')
    ax_ent.set_title('Comparacion: Entropia Espectral')
    ax_ent.set_xticks(x + width)
    ax_ent.set_xticklabels(list(all_results.keys()))
    ax_ent.legend()
    ax_ent.grid(True, alpha=0.3, axis='y')

    # Fila 2: Pendiente y prominencia
    ax_slope = axes[1, 0]
    for i, mod_type in enumerate(modulation_types):
        vals = [all_results[sys][mod_type].spectral_slope for sys in all_results]
        ax_slope.bar(x + i*width, vals, width, label=mod_type,
                    color=colors[mod_type], alpha=0.8)

    ax_slope.axhline(y=1.0, color='green', linestyle='--', label='1/f (critico)')
    ax_slope.set_ylabel('Pendiente beta')
    ax_slope.set_title('Pendiente Espectral (1/f^beta)')
    ax_slope.set_xticks(x + width)
    ax_slope.set_xticklabels(list(all_results.keys()))
    ax_slope.legend(fontsize=8)
    ax_slope.grid(True, alpha=0.3, axis='y')

    # Prominencia de picos
    ax_prom = axes[1, 1]
    for i, mod_type in enumerate(modulation_types):
        vals = [all_results[sys][mod_type].peak_prominence for sys in all_results]
        ax_prom.bar(x + i*width, vals, width, label=mod_type,
                   color=colors[mod_type], alpha=0.8)

    ax_prom.set_ylabel('Prominencia promedio')
    ax_prom.set_title('Prominencia de Picos')
    ax_prom.set_xticks(x + width)
    ax_prom.set_xticklabels(list(all_results.keys()))
    ax_prom.legend()
    ax_prom.grid(True, alpha=0.3, axis='y')

    # Posicion relativa
    ax_pos = axes[1, 2]
    for idx, sys_name in enumerate(all_results):
        entropies = {mod: all_results[sys_name][mod].spectral_entropy
                    for mod in modulation_types}

        h_min = min(entropies.values())
        h_max = max(entropies.values())
        h_range = h_max - h_min

        if h_range > 0.001:
            for mod_type in modulation_types:
                pos = (entropies[mod_type] - h_min) / h_range
                marker = 'o' if mod_type == "ZETA" else ('^' if mod_type == "RANDOM" else 's')
                size = 200 if mod_type == "ZETA" else 120
                ax_pos.scatter([pos], [idx], c=colors[mod_type], s=size,
                              marker=marker, zorder=3,
                              label=mod_type if idx == 0 else '')

            ax_pos.plot([0, 1], [idx, idx], 'k-', alpha=0.3, linewidth=2)

    ax_pos.axvline(x=0.5, color='green', linestyle='--', linewidth=2)
    ax_pos.axvspan(0.3, 0.7, alpha=0.1, color='green')

    ax_pos.set_xlim(-0.1, 1.1)
    ax_pos.set_yticks(range(len(all_results)))
    ax_pos.set_yticklabels(list(all_results.keys()))
    ax_pos.set_xlabel('Posicion relativa (0=ordenado, 1=caotico)')
    ax_pos.set_title('Posicion de ZETA en\nel espectro')
    ax_pos.legend(loc='upper right', fontsize=8)
    ax_pos.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'power_spectrum_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nGrafico guardado: {filename}")
    plt.close()

    return all_results


if __name__ == "__main__":
    results = run_power_spectrum_experiment()
