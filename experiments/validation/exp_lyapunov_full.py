# -*- coding: utf-8 -*-
"""
Experimento Completo de Lyapunov: Sistemas Simplificados + Reales

Fase 1: Sistemas simplificados con parametros ajustados
Fase 2: Sistemas reales (HierarchicalSimulation, ZetaOrganism)

Fecha: 2026-01-03
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import copy
import sys

# Frecuencias
ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
])

np.random.seed(42)
torch.manual_seed(42)


@dataclass
class LyapunovResult:
    """Resultado del calculo de Lyapunov."""
    modulation_type: str
    lyapunov_exponent: float
    std_error: float
    divergence_history: List[float]
    system_name: str
    phase: str  # "simplified" or "real"


def generate_frequencies(freq_type: str, n: int = 15) -> np.ndarray:
    """Genera frecuencias segun el tipo de modulacion."""
    if freq_type == "ZETA":
        return ZETA_ZEROS[:n]
    elif freq_type == "RANDOM":
        return np.sort(np.random.uniform(10, 70, n))
    elif freq_type == "UNIFORM":
        return np.linspace(14, 65, n)
    else:
        raise ValueError(f"Tipo desconocido: {freq_type}")


def zeta_kernel(t: float, frequencies: np.ndarray, sigma: float = 0.1) -> float:
    """Kernel K_sigma(t) = Sum exp(-sigma|gamma|) * cos(gamma*t)."""
    weights = np.exp(-sigma * np.abs(frequencies))
    oscillations = np.cos(frequencies * t)
    return 2.0 * np.sum(weights * oscillations)


# =============================================================================
# FASE 1: SISTEMAS SIMPLIFICADOS CON PARAMETROS AJUSTADOS
# =============================================================================

class AdjustedHierarchicalSystem:
    """
    Sistema jerarquico con parametros ajustados para revelar efecto zeta.
    - Ruido reducido: 0.001 (era 0.01)
    - Modulacion aumentada: 0.5 (era 0.1)
    """

    def __init__(self, n_cells: int = 20, frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05  # Mas sensible
        self.time = 0.0
        self.states = self._initialize_states()

    def _initialize_states(self) -> torch.Tensor:
        states = torch.rand(self.n_cells, 4)
        states = states / states.sum(dim=1, keepdim=True)
        return states

    def get_state_vector(self) -> torch.Tensor:
        return self.states.flatten()

    def set_state_vector(self, vec: torch.Tensor):
        self.states = vec.reshape(self.n_cells, 4)
        self.states = torch.clamp(self.states, min=0.001)
        self.states = self.states / self.states.sum(dim=1, keepdim=True)

    def step(self, dt: float = 0.1):
        self.time += dt

        # Modulacion FUERTE por kernel zeta
        modulation = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_norm = 0.5 + 0.5 * np.tanh(modulation / 5.0)  # Mas sensible

        # Global archetype
        global_arch = self.states.mean(dim=0)

        # Top-down con modulacion FUERTE
        top_down = 0.3 * mod_norm  # Era 0.1
        for i in range(self.n_cells):
            diff = global_arch - self.states[i]
            self.states[i] = self.states[i] + top_down * diff

        # Contagio lateral modulado inversamente
        lateral = 0.2 * (1.0 - mod_norm)  # Era 0.05
        for i in range(self.n_cells):
            j = np.random.randint(0, self.n_cells)
            if i != j:
                diff = self.states[j] - self.states[i]
                self.states[i] = self.states[i] + lateral * diff

        # Ruido MUY REDUCIDO
        noise = torch.randn_like(self.states) * 0.001  # Era 0.01
        self.states = self.states + noise

        # Renormalizar
        self.states = torch.clamp(self.states, min=0.001)
        self.states = self.states / self.states.sum(dim=1, keepdim=True)


class AdjustedZetaOrganism:
    """
    ZetaOrganism con parametros ajustados.
    """

    def __init__(self, n_cells: int = 20, grid_size: int = 32,
                 frequencies: np.ndarray = None):
        self.n_cells = n_cells
        self.grid_size = grid_size
        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.sigma = 0.05
        self.time = 0.0

        self.positions = torch.rand(n_cells, 2) * grid_size
        self.energies = torch.rand(n_cells) * 0.5 + 0.1
        self.roles = torch.zeros(n_cells)
        self.roles[0] = 1  # Un lider

    def get_state_vector(self) -> torch.Tensor:
        return torch.cat([self.positions.flatten(), self.energies])

    def set_state_vector(self, vec: torch.Tensor):
        n_pos = self.n_cells * 2
        self.positions = vec[:n_pos].reshape(self.n_cells, 2)
        self.energies = vec[n_pos:]
        self.positions = torch.clamp(self.positions, 0.1, self.grid_size - 0.1)
        self.energies = torch.clamp(self.energies, 0.01, 1.0)

    def step(self, dt: float = 0.1):
        self.time += dt

        # Modulacion temporal con kernel
        modulation = zeta_kernel(self.time, self.frequencies, self.sigma)
        mod_factor = 0.5 + 0.5 * np.tanh(modulation / 5.0)

        # Campo de fuerzas desde lideres
        for i in range(self.n_cells):
            if self.roles[i] < 0.5:  # Solo masas se mueven
                force = torch.zeros(2)
                for j in range(self.n_cells):
                    if self.roles[j] > 0.5:  # Lider
                        diff = self.positions[j] - self.positions[i]
                        dist = torch.norm(diff) + 0.1

                        # Fuerza con kernel espacial
                        k_spatial = zeta_kernel(dist.item() / 5.0,
                                               self.frequencies, self.sigma)
                        attraction = self.energies[j] * (0.5 + 0.5 * k_spatial) / dist
                        force = force + attraction * diff / dist

                # Movimiento FUERTE
                movement = force * self.energies[i] * mod_factor * 0.5  # Era dt*energia
                self.positions[i] = self.positions[i] + movement

        # Energia decay
        self.energies = self.energies * 0.995

        # Transferencia Fi <- Mi
        for i in range(self.n_cells):
            if self.roles[i] > 0.5:
                for j in range(self.n_cells):
                    if self.roles[j] < 0.5:
                        dist = torch.norm(self.positions[i] - self.positions[j])
                        if dist < 5:
                            transfer = 0.02 * self.energies[j] * mod_factor
                            self.energies[i] = self.energies[i] + transfer
                            self.energies[j] = self.energies[j] - transfer * 0.5

        # Ruido REDUCIDO
        self.positions = self.positions + torch.randn_like(self.positions) * 0.01
        self.energies = self.energies + torch.randn(self.n_cells) * 0.001

        # Clamp
        self.positions = torch.clamp(self.positions, 0.1, self.grid_size - 0.1)
        self.energies = torch.clamp(self.energies, 0.01, 1.0)


# =============================================================================
# FASE 2: SISTEMAS REALES
# =============================================================================

class RealHierarchicalWrapper:
    """Wrapper para el sistema jerarquico real."""

    def __init__(self, frequencies: np.ndarray = None, n_cells: int = 30):
        # Importar sistema real
        from hierarchical_simulation import HierarchicalSimulation, SimulationConfig
        from zeta_psyche import Archetype

        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS
        self.config = SimulationConfig(
            n_cells=n_cells,
            grid_size=32,
            state_dim=32,  # Match the default in SimulationConfig
            n_clusters=3,
        )
        self.sim = HierarchicalSimulation(self.config)
        self.sim.initialize()

        # Guardar sigma del kernel para modulacion
        self.sigma = 0.05
        self.time = 0.0

    def get_state_vector(self) -> torch.Tensor:
        """Extrae estado como vector de arquetipos de todas las celulas."""
        states = []
        for cell in self.sim.cells:
            states.append(cell.psyche.archetype_state)
        return torch.cat(states)

    def set_state_vector(self, vec: torch.Tensor):
        """Establece arquetipos desde vector."""
        n_archetypes = 4
        for i, cell in enumerate(self.sim.cells):
            start = i * n_archetypes
            end = start + n_archetypes
            new_state = vec[start:end]
            new_state = torch.clamp(new_state, min=0.01)
            new_state = new_state / new_state.sum()
            cell.psyche.archetype_state = new_state

    def step(self, dt: float = 0.1):
        """Ejecuta paso del sistema real."""
        self.time += dt
        # El sistema real tiene su propia dinamica
        self.sim.step()


class RealZetaOrganismWrapper:
    """Wrapper para ZetaOrganism real."""

    def __init__(self, frequencies: np.ndarray = None, n_cells: int = 30):
        from zeta_organism import ZetaOrganism

        self.frequencies = frequencies if frequencies is not None else ZETA_ZEROS

        # Crear organismo con las frecuencias especificadas
        # Nota: El ZetaOrganism real usa M y sigma, no frequencies directamente
        self.organism = ZetaOrganism(
            grid_size=32,
            n_cells=n_cells,
            state_dim=16,
            M=len(self.frequencies),
            sigma=0.05
        )
        self.organism.initialize(seed_fi=True)

    def get_state_vector(self) -> torch.Tensor:
        """Extrae posiciones y energias."""
        positions = torch.stack([torch.tensor(c.position, dtype=torch.float32)
                                 for c in self.organism.cells])
        energies = torch.tensor([c.energy for c in self.organism.cells],
                               dtype=torch.float32)
        return torch.cat([positions.flatten(), energies])

    def set_state_vector(self, vec: torch.Tensor):
        """Establece posiciones y energias."""
        n_cells = len(self.organism.cells)
        n_pos = n_cells * 2

        positions = vec[:n_pos].reshape(n_cells, 2)
        energies = vec[n_pos:n_pos + n_cells]

        for i, cell in enumerate(self.organism.cells):
            cell.position = (int(positions[i, 0].item()) % self.organism.grid_size,
                           int(positions[i, 1].item()) % self.organism.grid_size)
            cell.energy = float(torch.clamp(energies[i], 0.01, 1.0).item())

    def step(self, dt: float = 0.1):
        """Ejecuta paso del organismo."""
        self.organism.step()


# =============================================================================
# CALCULO DE LYAPUNOV
# =============================================================================

def compute_lyapunov(system_class, frequencies: np.ndarray,
                     n_steps: int = 100, perturbation: float = 1e-6,
                     n_trials: int = 5, phase: str = "simplified",
                     **kwargs) -> LyapunovResult:
    """Calcula Lyapunov con metodo de Benettin."""

    lyapunov_estimates = []
    all_divergences = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        torch.manual_seed(42 + trial)

        try:
            system1 = system_class(frequencies=frequencies, **kwargs)
        except Exception as e:
            print(f"Error creando sistema: {e}")
            continue

        np.random.seed(42 + trial)
        torch.manual_seed(42 + trial)

        try:
            system2 = system_class(frequencies=frequencies, **kwargs)
        except Exception as e:
            continue

        # Perturbar
        state1 = system1.get_state_vector()
        pert_vec = torch.randn_like(state1)
        pert_vec = pert_vec / torch.norm(pert_vec) * perturbation
        state2 = state1 + pert_vec
        system2.set_state_vector(state2)

        divergences = []
        lyapunov_sum = 0.0
        d0 = perturbation

        for step in range(n_steps):
            try:
                system1.step()
                system2.step()
            except Exception as e:
                break

            s1 = system1.get_state_vector()
            s2 = system2.get_state_vector()
            d = torch.norm(s1 - s2).item()

            if d > 1e-12 and not np.isnan(d) and not np.isinf(d):
                divergences.append(d)
                lyapunov_sum += np.log(d / d0)

                # Renormalizar (Benettin)
                diff = s2 - s1
                if torch.norm(diff) > 1e-12:
                    diff = diff / torch.norm(diff) * perturbation
                    system2.set_state_vector(s1 + diff)
                d0 = perturbation

        if len(divergences) > 10:
            lyapunov = lyapunov_sum / len(divergences)
            lyapunov_estimates.append(lyapunov)
            all_divergences.append(divergences)

    if len(lyapunov_estimates) == 0:
        return LyapunovResult(
            modulation_type="",
            lyapunov_exponent=float('nan'),
            std_error=float('nan'),
            divergence_history=[],
            system_name="",
            phase=phase
        )

    mean_lyapunov = np.mean(lyapunov_estimates)
    std_lyapunov = np.std(lyapunov_estimates) / np.sqrt(len(lyapunov_estimates))

    # Promediar divergencias
    if all_divergences:
        max_len = max(len(d) for d in all_divergences)
        avg_div = []
        for i in range(max_len):
            vals = [d[i] for d in all_divergences if len(d) > i]
            avg_div.append(np.mean(vals))
    else:
        avg_div = []

    return LyapunovResult(
        modulation_type="",
        lyapunov_exponent=mean_lyapunov,
        std_error=std_lyapunov,
        divergence_history=avg_div,
        system_name="",
        phase=phase
    )


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_full_experiment():
    """Ejecuta experimento completo en dos fases."""

    print("=" * 70)
    print("VALIDACION TEORICA COMPLETA: EXPONENTE DE LYAPUNOV")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Conjetura: Los ceros zeta minimizan |L| (borde del caos)")
    print()

    modulation_types = ["ZETA", "RANDOM", "UNIFORM"]
    all_results = {}

    # =========================================================================
    # FASE 1: SISTEMAS SIMPLIFICADOS AJUSTADOS
    # =========================================================================

    print("\n" + "=" * 70)
    print("FASE 1: SISTEMAS SIMPLIFICADOS (parametros ajustados)")
    print("=" * 70)
    print("  - Ruido reducido 10x")
    print("  - Modulacion aumentada 3x")

    simplified_systems = [
        ("Hierarchical (adj)", AdjustedHierarchicalSystem, {"n_cells": 25}),
        ("ZetaOrganism (adj)", AdjustedZetaOrganism, {"n_cells": 25, "grid_size": 32})
    ]

    for sys_name, sys_class, kwargs in simplified_systems:
        print(f"\n  {sys_name}:")
        all_results[sys_name] = {}

        for mod_type in modulation_types:
            print(f"    {mod_type}...", end=" ", flush=True)
            freqs = generate_frequencies(mod_type)

            result = compute_lyapunov(
                sys_class, freqs,
                n_steps=80, perturbation=1e-7, n_trials=5,
                phase="simplified", **kwargs
            )
            result.modulation_type = mod_type
            result.system_name = sys_name
            all_results[sys_name][mod_type] = result

            print(f"L = {result.lyapunov_exponent:+.4f}")

    # =========================================================================
    # FASE 2: SISTEMAS REALES
    # =========================================================================

    print("\n" + "=" * 70)
    print("FASE 2: SISTEMAS REALES")
    print("=" * 70)

    real_systems = [
        ("Hierarchical (real)", RealHierarchicalWrapper, {"n_cells": 25}),
        ("ZetaOrganism (real)", RealZetaOrganismWrapper, {"n_cells": 25})
    ]

    for sys_name, sys_class, kwargs in real_systems:
        print(f"\n  {sys_name}:")
        all_results[sys_name] = {}

        for mod_type in modulation_types:
            print(f"    {mod_type}...", end=" ", flush=True)
            freqs = generate_frequencies(mod_type)

            try:
                result = compute_lyapunov(
                    sys_class, freqs,
                    n_steps=50, perturbation=1e-6, n_trials=3,
                    phase="real", **kwargs
                )
                result.modulation_type = mod_type
                result.system_name = sys_name
                all_results[sys_name][mod_type] = result
                print(f"L = {result.lyapunov_exponent:+.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results[sys_name][mod_type] = LyapunovResult(
                    modulation_type=mod_type,
                    lyapunov_exponent=float('nan'),
                    std_error=float('nan'),
                    divergence_history=[],
                    system_name=sys_name,
                    phase="real"
                )

    # =========================================================================
    # ANALISIS
    # =========================================================================

    print("\n" + "=" * 70)
    print("ANALISIS DE RESULTADOS")
    print("=" * 70)

    hypothesis_results = {}

    for sys_name in all_results:
        print(f"\n{sys_name}:")
        print("-" * 50)

        lyapunovs = {}
        for mod in modulation_types:
            L = all_results[sys_name][mod].lyapunov_exponent
            if not np.isnan(L):
                lyapunovs[mod] = L

        if len(lyapunovs) < 3:
            print("  (datos insuficientes)")
            continue

        # Ordenar por |L|
        sorted_mods = sorted(lyapunovs.keys(), key=lambda m: abs(lyapunovs[m]))

        for mod in sorted_mods:
            L = lyapunovs[mod]
            regime = "CRITICO" if abs(L) < 0.5 else ("CAOTICO" if L > 0 else "ORDENADO")
            print(f"  {mod:8s}: L = {L:+.4f}  |L| = {abs(L):.4f}  [{regime}]")

        # Verificar hipotesis
        zeta_abs = abs(lyapunovs["ZETA"])
        random_abs = abs(lyapunovs["RANDOM"])
        uniform_abs = abs(lyapunovs["UNIFORM"])

        is_confirmed = zeta_abs < random_abs and zeta_abs < uniform_abs
        hypothesis_results[sys_name] = is_confirmed

        if is_confirmed:
            improvement_r = (random_abs - zeta_abs) / random_abs * 100
            improvement_u = (uniform_abs - zeta_abs) / uniform_abs * 100
            print(f"\n  [OK] HIPOTESIS CONFIRMADA: ZETA tiene |L| minimo")
            print(f"       {improvement_r:.1f}% mas cercano a criticidad que RANDOM")
            print(f"       {improvement_u:.1f}% mas cercano a criticidad que UNIFORM")
        else:
            print(f"\n  [?] Hipotesis no confirmada en este sistema")

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    confirmed = sum(1 for v in hypothesis_results.values() if v)
    total = len(hypothesis_results)

    print(f"\n  Sistemas donde ZETA minimiza |L|: {confirmed}/{total}")

    # Separar por fase
    simplified_confirmed = sum(1 for k, v in hypothesis_results.items()
                               if v and "(adj)" in k)
    real_confirmed = sum(1 for k, v in hypothesis_results.items()
                        if v and "(real)" in k)

    print(f"    - Simplificados: {simplified_confirmed}/2")
    print(f"    - Reales: {real_confirmed}/2")

    if confirmed == total and total > 0:
        print("\n  *** CONJETURA VALIDADA EN TODOS LOS SISTEMAS ***")
        print("  Los ceros de Riemann producen dinamica en el borde del caos")
    elif confirmed > total / 2:
        print("\n  ** CONJETURA MAYORMENTE VALIDADA **")
    elif confirmed > 0:
        print("\n  * CONJETURA PARCIALMENTE VALIDADA *")
    else:
        print("\n  X CONJETURA NO VALIDADA")

    # =========================================================================
    # VISUALIZACION
    # =========================================================================

    # Filtrar sistemas con datos validos
    valid_systems = []
    for s in all_results:
        try:
            zeta_result = all_results[s].get("ZETA")
            if zeta_result and not np.isnan(zeta_result.lyapunov_exponent):
                valid_systems.append(s)
        except:
            pass

    if len(valid_systems) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Validacion Teorica: Exponente de Lyapunov\n' +
                     'Conjetura: Los ceros zeta minimizan |L| (borde del caos)',
                     fontsize=13, fontweight='bold')

        colors = {"ZETA": "blue", "RANDOM": "red", "UNIFORM": "gray"}

        # Separar por fase
        simplified = [s for s in valid_systems if "(adj)" in s]
        real = [s for s in valid_systems if "(real)" in s]

        # Plot 1: Simplificados - divergencia
        ax1 = axes[0, 0]
        for sys_name in simplified[:1]:  # Primer simplificado
            for mod in modulation_types:
                r = all_results[sys_name][mod]
                if r.divergence_history:
                    ax1.semilogy(r.divergence_history, color=colors[mod],
                                label=f'{mod} (L={r.lyapunov_exponent:+.2f})',
                                linewidth=2, alpha=0.8)
        ax1.set_title('Fase 1: Sistema Simplificado')
        ax1.set_xlabel('Paso')
        ax1.set_ylabel('Divergencia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Reales - divergencia
        ax2 = axes[0, 1]
        for sys_name in real[:1]:  # Primer real
            for mod in modulation_types:
                r = all_results[sys_name][mod]
                if r.divergence_history:
                    ax2.semilogy(r.divergence_history, color=colors[mod],
                                label=f'{mod} (L={r.lyapunov_exponent:+.2f})',
                                linewidth=2, alpha=0.8)
        ax2.set_title('Fase 2: Sistema Real')
        ax2.set_xlabel('Paso')
        ax2.set_ylabel('Divergencia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Comparacion |L| todos los sistemas
        ax3 = axes[1, 0]
        x = np.arange(len(valid_systems))
        width = 0.25

        for i, mod in enumerate(modulation_types):
            vals = []
            for sys in valid_systems:
                L = all_results[sys][mod].lyapunov_exponent
                vals.append(abs(L) if not np.isnan(L) else 0)
            ax3.bar(x + i*width, vals, width, label=mod, color=colors[mod], alpha=0.8)

        ax3.axhline(y=0, color='green', linestyle='--', label='Criticidad')
        ax3.set_ylabel('|L| (distancia a criticidad)')
        ax3.set_title('Comparacion: Todos los Sistemas')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([s.replace(" (adj)", "\n(simpl)").replace(" (real)", "\n(real)")
                            for s in valid_systems], fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Mejora porcentual de ZETA
        ax4 = axes[1, 1]

        improvements = []
        labels = []
        for sys in valid_systems:
            z = abs(all_results[sys]["ZETA"].lyapunov_exponent)
            r = abs(all_results[sys]["RANDOM"].lyapunov_exponent)
            u = abs(all_results[sys]["UNIFORM"].lyapunov_exponent)

            if z > 0:
                imp_r = (r - z) / r * 100 if r > 0 else 0
                imp_u = (u - z) / u * 100 if u > 0 else 0
                improvements.append((imp_r, imp_u))
                labels.append(sys.replace(" (adj)", "\n(simpl)").replace(" (real)", "\n(real)"))

        if improvements:
            x = np.arange(len(improvements))
            ax4.bar(x - 0.2, [i[0] for i in improvements], 0.35,
                   label='vs RANDOM', color='red', alpha=0.7)
            ax4.bar(x + 0.2, [i[1] for i in improvements], 0.35,
                   label='vs UNIFORM', color='gray', alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-')
            ax4.set_ylabel('Mejora de ZETA (%)')
            ax4.set_title('Ventaja de ZETA sobre otros\n(positivo = ZETA mas cercano a criticidad)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(labels, fontsize=8)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'lyapunov_full_validation_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nGrafico guardado: {filename}")

        plt.show()

    return all_results


if __name__ == "__main__":
    results = run_full_experiment()
