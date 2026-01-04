# -*- coding: utf-8 -*-
"""
Experimento de Validación Teórica: Exponente de Lyapunov

Valida la conjetura central de la teoría:
"Los ceros zeta minimizan |L| para sistemas con kernel K_σ(t)"

Donde:
- L > 0: Caos (divergencia exponencial)
- L < 0: Orden (convergencia exponencial)
- L ~ 0: Criticidad (borde del caos)

Método: Benettin (perturbación de condiciones iniciales)

Fecha: 2026-01-03
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime
import copy

# Frecuencias para comparación
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
])

np.random.seed(42)
torch.manual_seed(42)


@dataclass
class LyapunovResult:
    """Resultado del cálculo de Lyapunov."""
    modulation_type: str
    lyapunov_exponent: float
    std_error: float
    divergence_history: List[float]
    system_name: str


def generate_frequencies(freq_type: str, n: int = 15) -> np.ndarray:
    """Genera frecuencias según el tipo de modulación."""
    if freq_type == "ZETA":
        return ZETA_ZEROS[:n]
    elif freq_type == "RANDOM":
        # Frecuencias aleatorias en el mismo rango
        return np.random.uniform(10, 70, n)
    elif freq_type == "UNIFORM":
        # Frecuencias uniformemente espaciadas
        return np.linspace(14, 65, n)
    else:
        raise ValueError(f"Tipo desconocido: {freq_type}")


def zeta_kernel(t: float, frequencies: np.ndarray, sigma: float = 0.1) -> float:
    """Calcula el kernel K_σ(t) = Σ exp(-σ|γ|) * cos(γt)."""
    weights = np.exp(-sigma * np.abs(frequencies))
    oscillations = np.cos(frequencies * t)
    return 2.0 * np.sum(weights * oscillations)


# =============================================================================
# SISTEMA 1: MODELO SIMPLIFICADO DE CONSCIENCIA JERÁRQUICA
# =============================================================================

class SimplifiedHierarchicalSystem:
    """
    Versión simplificada del sistema jerárquico para cálculo de Lyapunov.
    Captura la esencia: células con arquetipos, modulación zeta, y agregación.
    """

    def __init__(self, n_cells: int = 20, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.1
        self.time = 0.0

        # Estado: cada célula tiene distribución de 4 arquetipos
        self.states = self._initialize_states()

    def _initialize_states(self) -> torch.Tensor:
        """Inicializa estados de células (n_cells x 4 arquetipos)."""
        # Distribución aleatoria normalizada
        states = torch.rand(self.n_cells, 4)
        states = states / states.sum(dim=1, keepdim=True)
        return states

    def get_state_vector(self) -> torch.Tensor:
        """Retorna estado como vector plano para cálculo de divergencia."""
        return self.states.flatten()

    def set_state_vector(self, vec: torch.Tensor):
        """Establece estado desde vector plano."""
        self.states = vec.reshape(self.n_cells, 4)
        # Renormalizar para mantener distribuciones válidas
        self.states = torch.clamp(self.states, min=0.01)
        self.states = self.states / self.states.sum(dim=1, keepdim=True)

    def step(self, dt: float = 0.1):
        """Un paso de evolución con modulación zeta."""
        self.time += dt

        # 1. Modulación por kernel zeta
        modulation = zeta_kernel(self.time, self.frequencies, self.sigma)
        modulation_normalized = 0.5 + 0.5 * np.tanh(modulation / 10.0)

        # 2. Calcular "global archetype" (agregación bottom-up)
        global_archetype = self.states.mean(dim=0)

        # 3. Top-down: células se acercan al global con fuerza modulada
        top_down_strength = 0.1 * modulation_normalized
        for i in range(self.n_cells):
            # Diferencia con global
            diff = global_archetype - self.states[i]
            # Actualizar con modulación
            self.states[i] = self.states[i] + top_down_strength * diff

        # 4. Contagio lateral (vecinos aleatorios)
        lateral_strength = 0.05 * (1 - modulation_normalized)
        for i in range(self.n_cells):
            # Elegir vecino aleatorio
            j = np.random.randint(0, self.n_cells)
            if i != j:
                diff = self.states[j] - self.states[i]
                self.states[i] = self.states[i] + lateral_strength * diff

        # 5. Ruido exploratorio
        noise = torch.randn_like(self.states) * 0.01
        self.states = self.states + noise

        # 6. Renormalizar
        self.states = torch.clamp(self.states, min=0.01)
        self.states = self.states / self.states.sum(dim=1, keepdim=True)


# =============================================================================
# SISTEMA 2: MODELO SIMPLIFICADO DE ZETA ORGANISM
# =============================================================================

class SimplifiedZetaOrganism:
    """
    Versión simplificada del ZetaOrganism para cálculo de Lyapunov.
    Captura la esencia: células con posiciones, roles Fi/Mi, campo de fuerzas.
    """

    def __init__(self, n_cells: int = 20, grid_size: int = 32,
                 frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.grid_size = grid_size
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.1
        self.time = 0.0

        # Estado: posiciones (n_cells x 2) + energías (n_cells)
        self.positions = self._initialize_positions()
        self.energies = torch.rand(n_cells) * 0.5 + 0.1
        self.roles = torch.zeros(n_cells)  # 0=Mass, 1=Force
        self.roles[0] = 1  # Un líder inicial

    def _initialize_positions(self) -> torch.Tensor:
        """Inicializa posiciones aleatorias."""
        return torch.rand(self.n_cells, 2) * self.grid_size

    def get_state_vector(self) -> torch.Tensor:
        """Retorna estado como vector plano."""
        return torch.cat([self.positions.flatten(), self.energies])

    def set_state_vector(self, vec: torch.Tensor):
        """Establece estado desde vector plano."""
        n_pos = self.n_cells * 2
        self.positions = vec[:n_pos].reshape(self.n_cells, 2)
        self.energies = vec[n_pos:]
        # Clamp para mantener en rango válido
        self.positions = torch.clamp(self.positions, 0, self.grid_size)
        self.energies = torch.clamp(self.energies, 0.01, 1.0)

    def _compute_force_field(self, pos: torch.Tensor) -> torch.Tensor:
        """Calcula fuerza en una posición basado en líderes (Fi)."""
        force = torch.zeros(2)

        for i in range(self.n_cells):
            if self.roles[i] > 0.5:  # Es Fi (líder)
                diff = self.positions[i] - pos
                dist = torch.norm(diff) + 1e-6

                # Kernel zeta para la fuerza
                kernel_val = zeta_kernel(dist.item() / 10.0,
                                        self.frequencies, self.sigma)

                # Fuerza atractiva modulada por kernel
                attraction = self.energies[i] * kernel_val / (dist + 1)
                force = force + attraction * diff / dist

        return force

    def step(self, dt: float = 0.1):
        """Un paso de evolución."""
        self.time += dt

        # 1. Modulación temporal global
        modulation = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_factor = 0.5 + 0.5 * np.tanh(modulation / 10.0)

        # 2. Mover células según campo de fuerzas
        for i in range(self.n_cells):
            if self.roles[i] < 0.5:  # Solo Mi (masa) se mueve
                force = self._compute_force_field(self.positions[i])
                # Movimiento proporcional a energía y modulación
                movement = force * self.energies[i] * mod_factor * dt
                self.positions[i] = self.positions[i] + movement

        # 3. Actualizar energías (decay + recuperación)
        self.energies = self.energies * 0.99  # Decay

        # Fi ganan energía de Mi cercanos
        for i in range(self.n_cells):
            if self.roles[i] > 0.5:  # Fi
                for j in range(self.n_cells):
                    if self.roles[j] < 0.5:  # Mi
                        dist = torch.norm(self.positions[i] - self.positions[j])
                        if dist < 5:
                            transfer = 0.01 * self.energies[j]
                            self.energies[i] = self.energies[i] + transfer
                            self.energies[j] = self.energies[j] - transfer

        # 4. Ruido
        self.positions = self.positions + torch.randn_like(self.positions) * 0.1
        self.energies = self.energies + torch.randn(self.n_cells) * 0.01

        # 5. Clamp
        self.positions = torch.clamp(self.positions, 0, self.grid_size)
        self.energies = torch.clamp(self.energies, 0.01, 1.0)


# =============================================================================
# CÁLCULO DE LYAPUNOV (BENETTIN)
# =============================================================================

def compute_lyapunov_benettin(system_class, frequencies: np.ndarray,
                               n_steps: int = 100,
                               perturbation: float = 1e-6,
                               n_trials: int = 5,
                               **system_kwargs) -> LyapunovResult:
    """
    Calcula exponente de Lyapunov usando método de Benettin.

    L = lim(T→∞) (1/T) * Σ ln(|δx(t)|/|δx(0)|)
    """
    lyapunov_estimates = []
    all_divergences = []

    for trial in range(n_trials):
        # Crear sistema original
        np.random.seed(42 + trial)
        torch.manual_seed(42 + trial)

        system1 = system_class(frequencies=frequencies, **system_kwargs)

        # Crear copia perturbada
        np.random.seed(42 + trial)
        torch.manual_seed(42 + trial)

        system2 = system_class(frequencies=frequencies, **system_kwargs)

        # Aplicar perturbación
        state1 = system1.get_state_vector()
        perturbation_vec = torch.randn_like(state1) * perturbation
        perturbation_vec = perturbation_vec / torch.norm(perturbation_vec) * perturbation

        state2 = state1 + perturbation_vec
        system2.set_state_vector(state2)

        # Evolucionar y medir divergencia
        divergences = []
        lyapunov_sum = 0.0
        d0 = perturbation

        for step in range(n_steps):
            system1.step()
            system2.step()

            # Calcular divergencia
            s1 = system1.get_state_vector()
            s2 = system2.get_state_vector()
            d = torch.norm(s1 - s2).item()

            if d > 1e-10:
                divergences.append(d)
                lyapunov_sum += np.log(d / d0)

                # Renormalizar perturbación (Benettin)
                diff = s2 - s1
                diff = diff / torch.norm(diff) * perturbation
                system2.set_state_vector(s1 + diff)
                d0 = perturbation

        # Calcular L para este trial
        if len(divergences) > 0:
            lyapunov = lyapunov_sum / len(divergences)
            lyapunov_estimates.append(lyapunov)
            all_divergences.append(divergences)

    # Promediar resultados
    mean_lyapunov = np.mean(lyapunov_estimates)
    std_lyapunov = np.std(lyapunov_estimates) / np.sqrt(len(lyapunov_estimates))

    # Promedio de divergencias para visualización
    max_len = max(len(d) for d in all_divergences)
    avg_divergence = []
    for i in range(max_len):
        vals = [d[i] for d in all_divergences if len(d) > i]
        avg_divergence.append(np.mean(vals))

    return LyapunovResult(
        modulation_type="",  # Se llenará después
        lyapunov_exponent=mean_lyapunov,
        std_error=std_lyapunov,
        divergence_history=avg_divergence,
        system_name=""  # Se llenará después
    )


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_lyapunov_experiment():
    """Ejecuta experimento completo de Lyapunov."""

    print("=" * 70)
    print("VALIDACIÓN TEÓRICA: EXPONENTE DE LYAPUNOV")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Conjetura: Los ceros zeta minimizan |L|  (L = Lyapunov)")
    print("  L > 0: Caos (divergencia)")
    print("  L < 0: Orden (convergencia)")
    print("  L = 0: Criticidad (borde del caos)")
    print()

    modulation_types = ["ZETA", "RANDOM", "UNIFORM"]
    systems = [
        ("Hierarchical Consciousness", SimplifiedHierarchicalSystem, {"n_cells": 20}),
        ("ZetaOrganism", SimplifiedZetaOrganism, {"n_cells": 20, "grid_size": 32})
    ]

    results = {}

    for system_name, system_class, kwargs in systems:
        print(f"\n{'='*60}")
        print(f"SISTEMA: {system_name}")
        print(f"{'='*60}")

        results[system_name] = {}

        for mod_type in modulation_types:
            print(f"\n  Modulación: {mod_type}...", end=" ", flush=True)

            frequencies = generate_frequencies(mod_type)

            result = compute_lyapunov_benettin(
                system_class,
                frequencies,
                n_steps=50,  # Rápido para PoC
                perturbation=1e-6,
                n_trials=3,  # Menos trials para PoC
                **kwargs
            )

            result.modulation_type = mod_type
            result.system_name = system_name
            results[system_name][mod_type] = result

            print(f"L = {result.lyapunov_exponent:+.4f} +/- {result.std_error:.4f}")

    # ==========================================================================
    # ANÁLISIS DE RESULTADOS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("ANÁLISIS DE RESULTADOS")
    print("=" * 70)

    for system_name in results:
        print(f"\n{system_name}:")
        print("-" * 40)

        lyapunovs = {mod: results[system_name][mod].lyapunov_exponent
                     for mod in modulation_types}

        # Ordenar por |L| (cercanía a criticidad)
        sorted_mods = sorted(lyapunovs.keys(), key=lambda m: abs(lyapunovs[m]))

        for mod in sorted_mods:
            L = lyapunovs[mod]
            dist_to_critical = abs(L)

            if L > 0.1:
                regime = "CAÓTICO"
            elif L < -0.1:
                regime = "ORDENADO"
            else:
                regime = "CRÍTICO"

            print(f"  {mod:8s}: L = {L:+.4f}  |L| = {dist_to_critical:.4f}  [{regime}]")

        # Verificar hipótesis
        zeta_abs = abs(lyapunovs["ZETA"])
        random_abs = abs(lyapunovs["RANDOM"])
        uniform_abs = abs(lyapunovs["UNIFORM"])

        if zeta_abs < random_abs and zeta_abs < uniform_abs:
            print(f"\n  OK HIPÓTESIS CONFIRMADA: ZETA tiene |L| mínimo")
            print(f"    ZETA más cercano a criticidad que RANDOM ({random_abs/zeta_abs:.2f}x)")
            print(f"    ZETA más cercano a criticidad que UNIFORM ({uniform_abs/zeta_abs:.2f}x)")
        else:
            print(f"\n  ? HIPÓTESIS NO CONFIRMADA en este sistema")
            print(f"    |L_ZETA|={zeta_abs:.4f}, |L_RANDOM|={random_abs:.4f}, |L_UNIFORM|={uniform_abs:.4f}")

    # ==========================================================================
    # RESUMEN FINAL
    # ==========================================================================

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    confirmations = 0
    for system_name in results:
        lyapunovs = {mod: results[system_name][mod].lyapunov_exponent
                     for mod in modulation_types}
        zeta_abs = abs(lyapunovs["ZETA"])
        if zeta_abs <= min(abs(lyapunovs["RANDOM"]), abs(lyapunovs["UNIFORM"])):
            confirmations += 1

    print(f"\n  Sistemas donde ZETA minimiza |L|: {confirmations}/{len(results)}")

    if confirmations == len(results):
        print("\n  *** CONJETURA VALIDADA EN TODOS LOS SISTEMAS ***")
        print("  Los ceros de Riemann producen dinámica en el borde del caos")
    elif confirmations > 0:
        print("\n  ** CONJETURA PARCIALMENTE VALIDADA **")
    else:
        print("\n  X CONJETURA NO VALIDADA (revisar implementación)")

    # ==========================================================================
    # VISUALIZACIÓN
    # ==========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Validación Teórica: Exponente de Lyapunov\n' +
                 'Conjetura: Los ceros zeta minimizan |L| (borde del caos)',
                 fontsize=14, fontweight='bold')

    colors = {"ZETA": "blue", "RANDOM": "red", "UNIFORM": "gray"}

    # Plot 1 & 2: Divergencia temporal por sistema
    for idx, system_name in enumerate(results):
        ax = axes[0, idx]

        for mod_type in modulation_types:
            result = results[system_name][mod_type]
            divergence = result.divergence_history
            ax.semilogy(divergence, color=colors[mod_type],
                       label=f'{mod_type} (L={result.lyapunov_exponent:+.3f})',
                       linewidth=2, alpha=0.8)

        ax.set_title(f'{system_name}')
        ax.set_xlabel('Paso')
        ax.set_ylabel('Divergencia |δx|')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Comparación de |L|
    ax3 = axes[1, 0]

    x_positions = np.arange(len(results))
    width = 0.25

    for i, mod_type in enumerate(modulation_types):
        values = [abs(results[sys][mod_type].lyapunov_exponent) for sys in results]
        errors = [results[sys][mod_type].std_error for sys in results]
        ax3.bar(x_positions + i*width, values, width,
               label=mod_type, color=colors[mod_type], alpha=0.8,
               yerr=errors, capsize=3)

    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.5,
               label='Criticidad (L=0)')
    ax3.set_xlabel('Sistema')
    ax3.set_ylabel('|L| (distancia a criticidad)')
    ax3.set_title('Comparación: Distancia al Borde del Caos')
    ax3.set_xticks(x_positions + width)
    ax3.set_xticklabels(list(results.keys()), rotation=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: L con signo (régimen)
    ax4 = axes[1, 1]

    for i, mod_type in enumerate(modulation_types):
        values = [results[sys][mod_type].lyapunov_exponent for sys in results]
        errors = [results[sys][mod_type].std_error for sys in results]
        ax4.bar(x_positions + i*width, values, width,
               label=mod_type, color=colors[mod_type], alpha=0.8,
               yerr=errors, capsize=3)

    ax4.axhline(y=0, color='green', linestyle='-', linewidth=2,
               label='Criticidad (L=0)')
    ax4.axhspan(-0.1, 0.1, alpha=0.1, color='green', label='Zona crítica')
    ax4.set_xlabel('Sistema')
    ax4.set_ylabel('L (exponente de Lyapunov)')
    ax4.set_title('Régimen Dinámico (L>0: caos, L<0: orden)')
    ax4.set_xticks(x_positions + width)
    ax4.set_xticklabels(list(results.keys()), rotation=15)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Guardar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'lyapunov_validation_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nGráfico guardado: {filename}")

    plt.show()

    return results


if __name__ == "__main__":
    results = run_lyapunov_experiment()
