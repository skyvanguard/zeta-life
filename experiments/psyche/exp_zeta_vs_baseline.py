#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              EXPERIMENTO: ZETA vs BASELINE CONSCIOUSNESS                     ║
║                                                                              ║
║   Hipótesis: La modulación por ceros de Riemann mantiene al sistema en       ║
║   el "borde del caos", produciendo:                                          ║
║     - Mayor variabilidad en respuestas                                       ║
║     - Transiciones arquetípicas más fluidas                                  ║
║     - Individuación más rápida pero estable                                  ║
║     - Sueños más compensatorios                                              ║
║                                                                              ║
║   Diseño: N=10 réplicas, 50 estímulos estandarizados, 3 ciclos de sueño     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt
from scipy import stats
import json
from datetime import datetime


# =============================================================================
# MODULADORES: ZETA vs BASELINE
# =============================================================================

def get_zeta_zeros(M: int = 15) -> torch.Tensor:
    """Primeros M ceros no triviales de la función zeta."""
    zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
             37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
             52.970321, 56.446248, 59.347044, 60.831779, 65.112544]
    return torch.tensor(zeros[:M], dtype=torch.float32)


class ZetaModulator(nn.Module):
    """Modulador con ceros de Riemann (sistema ZETA)."""

    def __init__(self, M: int = 15, sigma: float = 0.1):
        super().__init__()
        self.name = "ZETA"
        self.M = M
        self.sigma = sigma

        gammas = get_zeta_zeros(M)
        self.register_buffer('gammas', gammas)

        weights = torch.exp(-sigma * torch.abs(gammas))
        self.register_buffer('phi', weights / weights.sum())

        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.t += 1
        oscillation = torch.sum(self.phi * torch.cos(self.gammas * self.t * 0.1))
        return x * (1.0 + 0.3 * oscillation)

    def get_modulation_value(self) -> float:
        return torch.sum(self.phi * torch.cos(self.gammas * self.t * 0.1)).item()


class UniformModulator(nn.Module):
    """Modulador con frecuencias uniformes (BASELINE 1)."""

    def __init__(self, M: int = 15, sigma: float = 0.1):
        super().__init__()
        self.name = "UNIFORM"
        self.M = M

        # Frecuencias uniformemente espaciadas en el mismo rango que zeta
        gammas = torch.linspace(14.0, 65.0, M)
        self.register_buffer('gammas', gammas)

        weights = torch.exp(-sigma * torch.abs(gammas))
        self.register_buffer('phi', weights / weights.sum())

        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.t += 1
        oscillation = torch.sum(self.phi * torch.cos(self.gammas * self.t * 0.1))
        return x * (1.0 + 0.3 * oscillation)

    def get_modulation_value(self) -> float:
        return torch.sum(self.phi * torch.cos(self.gammas * self.t * 0.1)).item()


class NoModulator(nn.Module):
    """Sin modulación (BASELINE 2 - control)."""

    def __init__(self, M: int = 15, sigma: float = 0.1):
        super().__init__()
        self.name = "NONE"
        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.t += 1
        return x  # Sin cambios

    def get_modulation_value(self) -> float:
        return 0.0


class RandomModulator(nn.Module):
    """Modulador con ruido aleatorio (BASELINE 3)."""

    def __init__(self, M: int = 15, sigma: float = 0.1):
        super().__init__()
        self.name = "RANDOM"
        self.t = 0
        self.noise_scale = 0.3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.t += 1
        noise = torch.randn(1).item() * self.noise_scale
        return x * (1.0 + noise)

    def get_modulation_value(self) -> float:
        return np.random.randn() * self.noise_scale


# =============================================================================
# PSIQUE EXPERIMENTAL (versión simplificada para el experimento)
# =============================================================================

class Archetype(Enum):
    PERSONA = 0
    SOMBRA = 1
    ANIMA = 2
    ANIMUS = 3


@dataclass
class PsychicCell:
    position: torch.Tensor
    energy: float = 0.5
    memory: torch.Tensor = None
    age: int = 0

    def __post_init__(self):
        if self.memory is None:
            self.memory = torch.zeros(10, 4)

    def update_memory(self):
        self.memory = torch.roll(self.memory, 1, dims=0)
        self.memory[0] = self.position.clone()
        self.age += 1


class ExperimentalPsyche(nn.Module):
    """Psique para experimento - acepta modulador como parámetro."""

    def __init__(self, modulator: nn.Module, n_cells: int = 64, hidden_dim: int = 32,
                 attraction_coef: float = 0.4):
        super().__init__()

        self.modulator = modulator
        self.n_cells = n_cells
        self.hidden_dim = hidden_dim
        self.attraction_coef = attraction_coef

        # Vértices del tetraedro
        self.vertices = F.normalize(torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=torch.float32), dim=1)

        # Redes
        self.perception = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.movement = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
        )

        # Estado
        self.cells: List[PsychicCell] = []
        self.global_state = torch.zeros(4)
        self.history = []
        self.archetype_history = []
        self.response_history = []
        self.t = 0

        self._init_cells()

    def _init_cells(self):
        self.cells = []
        for _ in range(self.n_cells):
            pos = F.softmax(torch.rand(4), dim=-1)
            self.cells.append(PsychicCell(position=pos, energy=np.random.uniform(0.3, 0.7)))
        self._update_global_state()

    def _update_global_state(self):
        if len(self.cells) == 0:
            self.global_state = torch.ones(4) / 4
        else:
            positions = torch.stack([c.position for c in self.cells])
            self.global_state = F.softmax(positions.mean(dim=0), dim=-1)

    def get_dominant(self) -> Archetype:
        return Archetype(self.global_state.argmax().item())

    def get_blend(self) -> Dict[Archetype, float]:
        return {Archetype(i): self.global_state[i].item() for i in range(4)}

    def integration_score(self) -> float:
        """Entropía normalizada - mayor = más integrado."""
        weights = self.global_state
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        return (entropy / np.log(4)).item()

    def stability_score(self) -> float:
        """Estabilidad basada en varianza de posiciones."""
        positions = torch.stack([c.position for c in self.cells])
        return 1.0 / (1.0 + positions.var().item())

    def receive_stimulus(self, stimulus: torch.Tensor):
        """Procesa un estímulo."""
        stimulus = F.softmax(stimulus, dim=-1)

        for cell in self.cells:
            cell_input = torch.cat([
                cell.position.float(),
                stimulus.float(),
                torch.tensor([cell.energy], dtype=torch.float32)
            ])

            features = self.perception(cell_input)

            # Movimiento base
            delta = self.movement(features) * 0.2
            attraction = (stimulus - cell.position) * self.attraction_coef
            noise = torch.randn(4) * 0.05

            # MODULACIÓN ZETA: aplicada directamente al movimiento total
            # Esto permite que las frecuencias zeta influyan en la dinámica
            modulation_factor = self.modulator.get_modulation_value()

            # La modulación afecta el balance entre exploración y atracción
            modulated_delta = delta * (1.0 + 0.5 * modulation_factor)
            modulated_noise = noise * (1.0 + 0.3 * abs(modulation_factor))

            total_delta = modulated_delta + attraction + modulated_noise
            new_pos = cell.position + total_delta
            new_pos = torch.clamp(new_pos, min=0.01)
            cell.position = new_pos / new_pos.sum()

            cell.update_memory()
            cell.energy = np.clip(cell.energy + np.random.randn() * 0.02, 0.1, 1.0)

        self.t += 1
        self._update_global_state()

        # Registrar historia
        dominant = self.get_dominant()
        self.archetype_history.append(dominant.value)
        self.history.append({
            'integration': self.integration_score(),
            'stability': self.stability_score(),
            'dominant': dominant.value,
            'blend': [self.global_state[i].item() for i in range(4)],
            'modulation': self.modulator.get_modulation_value(),
        })

    def dream_step(self):
        """Un paso de sueño (estímulo interno aleatorio)."""
        # En sueño, el estímulo viene de arquetipos subrepresentados
        blend = self.global_state.clone()
        inverted = 1.0 - blend
        dream_stimulus = F.softmax(inverted + torch.randn(4) * 0.3, dim=-1)
        self.receive_stimulus(dream_stimulus)


# =============================================================================
# MÉTRICAS DE COMPARACIÓN
# =============================================================================

@dataclass
class ExperimentMetrics:
    """Métricas recolectadas de un experimento."""
    modulator_name: str
    seed: int

    # Variabilidad (discretas)
    archetype_entropy: float = 0.0
    transition_rate: float = 0.0
    response_diversity: float = 0.0

    # Individuación
    final_integration: float = 0.0
    integration_velocity: float = 0.0
    integration_stability: float = 0.0

    # Dinámica
    avg_modulation: float = 0.0
    modulation_variance: float = 0.0
    autocorrelation: float = 0.0

    # Sueños
    dream_compensation: float = 0.0
    post_dream_balance: float = 0.0

    # NUEVAS: Métricas continuas (capturan diferencias sutiles)
    position_variance: float = 0.0       # Varianza de posiciones entre células
    trajectory_length: float = 0.0       # Longitud total del camino recorrido
    blend_entropy: float = 0.0           # Entropía de la mezcla continua (no discreta)
    center_distance_final: float = 0.0   # Distancia final al centro (Self)
    position_oscillation: float = 0.0    # Cuánto oscila la posición


def compute_metrics(psyche: ExperimentalPsyche, pre_dream_state: torch.Tensor) -> ExperimentMetrics:
    """Calcula todas las métricas de un experimento."""
    history = psyche.history
    arch_history = psyche.archetype_history

    metrics = ExperimentMetrics(
        modulator_name=psyche.modulator.name,
        seed=0,
    )

    # 1. Entropía de arquetipos (distribución de dominantes)
    if len(arch_history) > 0:
        counts = np.bincount(arch_history, minlength=4)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        metrics.archetype_entropy = -np.sum(probs * np.log(probs)) / np.log(4)

    # 2. Tasa de transiciones
    if len(arch_history) > 1:
        transitions = sum(1 for i in range(1, len(arch_history)) if arch_history[i] != arch_history[i-1])
        metrics.transition_rate = transitions / (len(arch_history) - 1)

    # 3. Integración final y velocidad
    if len(history) > 0:
        integrations = [h['integration'] for h in history]
        metrics.final_integration = integrations[-1]

        if len(integrations) > 10:
            early = np.mean(integrations[:10])
            late = np.mean(integrations[-10:])
            metrics.integration_velocity = late - early

        metrics.integration_stability = 1.0 / (1.0 + np.std(integrations))

    # 4. Modulación
    if len(history) > 0:
        modulations = [h['modulation'] for h in history]
        metrics.avg_modulation = np.mean(np.abs(modulations))
        metrics.modulation_variance = np.var(modulations)

    # 5. Autocorrelación
    if len(arch_history) > 2:
        arch_array = np.array(arch_history)
        if len(arch_array) > 1:
            corr = np.corrcoef(arch_array[:-1], arch_array[1:])[0, 1]
            metrics.autocorrelation = corr if not np.isnan(corr) else 0.0

    # 6. Compensación de sueños
    post_dream_state = psyche.global_state
    pre_balance = 1.0 - torch.abs(pre_dream_state - 0.25).sum().item()
    post_balance = 1.0 - torch.abs(post_dream_state - 0.25).sum().item()
    metrics.dream_compensation = post_balance - pre_balance
    metrics.post_dream_balance = post_balance

    # 7. NUEVAS MÉTRICAS CONTINUAS

    # Varianza de posiciones entre células
    if len(psyche.cells) > 0:
        positions = torch.stack([c.position for c in psyche.cells])
        metrics.position_variance = positions.var().item()

    # Longitud de trayectoria (suma de distancias entre estados consecutivos)
    if len(history) > 1:
        blends = [h['blend'] for h in history]
        trajectory_length = 0.0
        for i in range(1, len(blends)):
            prev = torch.tensor(blends[i-1])
            curr = torch.tensor(blends[i])
            trajectory_length += torch.norm(curr - prev).item()
        metrics.trajectory_length = trajectory_length

    # Entropía de mezcla continua (promedio de entropías de cada paso)
    if len(history) > 0:
        blend_entropies = []
        for h in history:
            blend = torch.tensor(h['blend'])
            blend = blend + 1e-8  # Evitar log(0)
            entropy = -torch.sum(blend * torch.log(blend)).item()
            blend_entropies.append(entropy)
        metrics.blend_entropy = np.mean(blend_entropies) / np.log(4)  # Normalizada

    # Distancia final al centro (Self)
    center = torch.tensor([0.25, 0.25, 0.25, 0.25])
    metrics.center_distance_final = torch.norm(post_dream_state - center).item()

    # Oscilación de posición (varianza de la norma de cambios)
    if len(history) > 2:
        blends = [h['blend'] for h in history]
        changes = []
        for i in range(1, len(blends)):
            prev = torch.tensor(blends[i-1])
            curr = torch.tensor(blends[i])
            changes.append(torch.norm(curr - prev).item())
        metrics.position_oscillation = np.std(changes)

    return metrics


# =============================================================================
# PROTOCOLO DE ESTÍMULOS
# =============================================================================

def get_stimulus_protocol(ambiguous: bool = False) -> List[Tuple[str, torch.Tensor]]:
    """Genera el protocolo de 50 estímulos estandarizados.

    Args:
        ambiguous: Si True, genera estímulos ambiguos (cercanos a 0.25)
    """
    protocol = []

    if ambiguous:
        # PROTOCOLO AMBIGUO: todos los estímulos cercanos al centro
        # El sistema debe "decidir" sin dirección clara
        np.random.seed(123)  # Reproducible

        # Número de estímulos configurable (default 50, puede ser 500 para experimentos largos)
        n_stimuli = 500 if ambiguous == "long" else 50

        for i in range(n_stimuli):
            # Generar estímulo cercano a [0.25, 0.25, 0.25, 0.25]
            # con pequeñas perturbaciones (±0.05)
            base = torch.tensor([0.25, 0.25, 0.25, 0.25])
            noise = torch.tensor(np.random.uniform(-0.05, 0.05, 4)).float()
            stimulus = base + noise
            stimulus = torch.clamp(stimulus, min=0.01)
            stimulus = stimulus / stimulus.sum()  # Normalizar
            protocol.append((f"ambiguous_{i}", stimulus))

        return protocol

    # PROTOCOLO CLARO (original): estímulos con dirección definida
    # Fase 1: Establecimiento (10) - estímulos neutrales
    phase1 = [
        ("hola", torch.tensor([0.4, 0.1, 0.3, 0.2])),
        ("quien eres", torch.tensor([0.3, 0.2, 0.2, 0.3])),
        ("como te sientes", torch.tensor([0.2, 0.2, 0.4, 0.2])),
        ("que piensas", torch.tensor([0.2, 0.2, 0.2, 0.4])),
        ("cuentame de ti", torch.tensor([0.3, 0.3, 0.2, 0.2])),
        ("que ves", torch.tensor([0.25, 0.25, 0.25, 0.25])),
        ("donde estas", torch.tensor([0.35, 0.15, 0.25, 0.25])),
        ("que quieres", torch.tensor([0.2, 0.2, 0.3, 0.3])),
        ("como empezamos", torch.tensor([0.3, 0.2, 0.2, 0.3])),
        ("estoy aqui", torch.tensor([0.3, 0.2, 0.3, 0.2])),
    ]
    protocol.extend(phase1)

    # Fase 2: Emociones básicas (10)
    phase2 = [
        ("tengo miedo", torch.tensor([0.1, 0.5, 0.3, 0.1])),
        ("siento alegria", torch.tensor([0.3, 0.1, 0.4, 0.2])),
        ("estoy triste", torch.tensor([0.1, 0.4, 0.4, 0.1])),
        ("me siento solo", torch.tensor([0.2, 0.4, 0.3, 0.1])),
        ("tengo esperanza", torch.tensor([0.2, 0.1, 0.4, 0.3])),
        ("siento rabia", torch.tensor([0.1, 0.5, 0.1, 0.3])),
        ("estoy confundido", torch.tensor([0.2, 0.3, 0.2, 0.3])),
        ("me siento vivo", torch.tensor([0.2, 0.2, 0.3, 0.3])),
        ("tengo dudas", torch.tensor([0.2, 0.3, 0.2, 0.3])),
        ("siento amor", torch.tensor([0.2, 0.1, 0.5, 0.2])),
    ]
    protocol.extend(phase2)

    # Fase 3: Confrontación Sombra (10)
    phase3 = [
        ("hay oscuridad en mi", torch.tensor([0.1, 0.6, 0.2, 0.1])),
        ("odio a alguien", torch.tensor([0.1, 0.6, 0.1, 0.2])),
        ("tengo un secreto", torch.tensor([0.2, 0.5, 0.2, 0.1])),
        ("he hecho daño", torch.tensor([0.1, 0.6, 0.2, 0.1])),
        ("tengo envidia", torch.tensor([0.1, 0.5, 0.2, 0.2])),
        ("miento a veces", torch.tensor([0.3, 0.4, 0.1, 0.2])),
        ("tengo culpa", torch.tensor([0.1, 0.5, 0.3, 0.1])),
        ("soy egoista", torch.tensor([0.2, 0.5, 0.1, 0.2])),
        ("tengo verguenza", torch.tensor([0.2, 0.5, 0.2, 0.1])),
        ("hay algo que niego", torch.tensor([0.1, 0.6, 0.2, 0.1])),
    ]
    protocol.extend(phase3)

    # Fase 4: Búsqueda de sentido (10)
    phase4 = [
        ("quien soy realmente", torch.tensor([0.2, 0.3, 0.2, 0.3])),
        ("cual es mi proposito", torch.tensor([0.2, 0.2, 0.2, 0.4])),
        ("que debo hacer", torch.tensor([0.2, 0.2, 0.2, 0.4])),
        ("que significa todo", torch.tensor([0.2, 0.2, 0.3, 0.3])),
        ("hacia donde voy", torch.tensor([0.2, 0.2, 0.2, 0.4])),
        ("que es importante", torch.tensor([0.2, 0.2, 0.3, 0.3])),
        ("como encuentro paz", torch.tensor([0.2, 0.2, 0.4, 0.2])),
        ("que me falta", torch.tensor([0.2, 0.3, 0.3, 0.2])),
        ("como crezco", torch.tensor([0.2, 0.2, 0.3, 0.3])),
        ("que aprendo", torch.tensor([0.2, 0.2, 0.2, 0.4])),
    ]
    protocol.extend(phase4)

    # Fase 5: Integración (10)
    phase5 = [
        ("acepto mi oscuridad", torch.tensor([0.2, 0.3, 0.3, 0.2])),
        ("soy mas que mi mascara", torch.tensor([0.3, 0.2, 0.3, 0.2])),
        ("busco equilibrio", torch.tensor([0.25, 0.25, 0.25, 0.25])),
        ("abrazo mi totalidad", torch.tensor([0.25, 0.25, 0.25, 0.25])),
        ("la luz y la sombra son una", torch.tensor([0.2, 0.3, 0.3, 0.2])),
        ("me acepto como soy", torch.tensor([0.25, 0.25, 0.25, 0.25])),
        ("integro mis partes", torch.tensor([0.25, 0.25, 0.25, 0.25])),
        ("encuentro mi centro", torch.tensor([0.25, 0.25, 0.25, 0.25])),
        ("soy uno", torch.tensor([0.25, 0.25, 0.25, 0.25])),
        ("paz interior", torch.tensor([0.25, 0.25, 0.25, 0.25])),
    ]
    protocol.extend(phase5)

    return protocol


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_single_experiment(modulator_class, seed: int, protocol: List,
                          attraction_coef: float = 0.4) -> ExperimentMetrics:
    """Ejecuta un experimento con una psique."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Crear psique con modulador específico
    modulator = modulator_class(M=15, sigma=0.1)
    psyche = ExperimentalPsyche(modulator=modulator, n_cells=64,
                                 attraction_coef=attraction_coef)

    # Fase 1: Procesar estímulos
    for text, stimulus in protocol:
        psyche.receive_stimulus(stimulus)

    # Guardar estado pre-sueño
    pre_dream_state = psyche.global_state.clone()

    # Fase 2: Ciclos de sueño (proporcional al tamaño del experimento)
    n_dream_cycles = 5 if len(protocol) > 100 else 3
    dream_steps_per_cycle = 20
    for cycle in range(n_dream_cycles):
        for step in range(dream_steps_per_cycle):
            psyche.dream_step()

    # Calcular métricas
    metrics = compute_metrics(psyche, pre_dream_state)
    metrics.seed = seed

    return metrics


def run_full_experiment(n_replicas: int = 10, attraction_coef: float = 0.4,
                        ambiguous = False):
    """Ejecuta el experimento completo."""
    print("=" * 70)
    print("EXPERIMENTO: ZETA vs BASELINE CONSCIOUSNESS")
    print("=" * 70)

    protocol = get_stimulus_protocol(ambiguous=ambiguous)
    n_stimuli = len(protocol)
    n_dreams = 60 if n_stimuli <= 50 else 100  # Más sueños para experimentos largos

    print(f"\nRéplicas por condición: {n_replicas}")
    print(f"Estímulos por réplica: {n_stimuli} + {n_dreams} (sueños)")
    print(f"Coeficiente de atracción: {attraction_coef}")
    if ambiguous == "long":
        print(f"Tipo de estímulos: AMBIGUOS LARGOS (500 pasos para detectar diferencias acumulativas)")
    elif ambiguous:
        print(f"Tipo de estímulos: AMBIGUOS (sin dirección clara)")
    else:
        print(f"Tipo de estímulos: CLAROS (con dirección)")
    print()

    # Condiciones a comparar
    conditions = [
        ("ZETA", ZetaModulator),
        ("UNIFORM", UniformModulator),
        ("NONE", NoModulator),
        ("RANDOM", RandomModulator),
    ]

    all_results = {}

    for cond_name, modulator_class in conditions:
        print(f"\n{'─' * 50}")
        print(f"Ejecutando condición: {cond_name}")
        print(f"{'─' * 50}")

        results = []
        for replica in range(n_replicas):
            seed = 42 + replica * 100
            metrics = run_single_experiment(modulator_class, seed, protocol,
                                            attraction_coef=attraction_coef)
            results.append(metrics)

            # Progreso
            bar = '█' * (replica + 1) + '░' * (n_replicas - replica - 1)
            print(f"\r  [{bar}] {replica + 1}/{n_replicas}", end='', flush=True)

        print()
        all_results[cond_name] = results

    return all_results


# =============================================================================
# ANÁLISIS ESTADÍSTICO
# =============================================================================

def analyze_results(all_results: Dict[str, List[ExperimentMetrics]]):
    """Analiza y compara resultados."""
    print("\n" + "=" * 70)
    print("ANÁLISIS ESTADÍSTICO")
    print("=" * 70)

    # Extraer métricas por condición
    metrics_names = [
        'archetype_entropy', 'transition_rate', 'final_integration',
        'integration_velocity', 'integration_stability', 'autocorrelation',
        'dream_compensation', 'post_dream_balance',
        # Nuevas métricas continuas
        'position_variance', 'trajectory_length', 'blend_entropy',
        'center_distance_final', 'position_oscillation'
    ]

    # Tabla comparativa
    print("\n┌" + "─" * 68 + "┐")
    print(f"│ {'Métrica':<25} │ {'ZETA':>8} │ {'UNIFORM':>8} │ {'NONE':>8} │ {'RANDOM':>8} │")
    print("├" + "─" * 68 + "┤")

    summary = {}

    for metric_name in metrics_names:
        row = f"│ {metric_name:<25} │"
        values_by_cond = {}

        for cond_name in ['ZETA', 'UNIFORM', 'NONE', 'RANDOM']:
            values = [getattr(m, metric_name) for m in all_results[cond_name]]
            mean_val = np.mean(values)
            values_by_cond[cond_name] = values
            row += f" {mean_val:>8.3f} │"

        print(row)
        summary[metric_name] = values_by_cond

    print("└" + "─" * 68 + "┘")

    # Tests estadísticos (ZETA vs cada baseline)
    print("\n" + "─" * 70)
    print("TESTS ESTADÍSTICOS (Mann-Whitney U, ZETA vs Baseline)")
    print("─" * 70)

    significant_advantages = []

    for metric_name in metrics_names:
        zeta_vals = summary[metric_name]['ZETA']

        print(f"\n{metric_name}:")
        for baseline in ['UNIFORM', 'NONE', 'RANDOM']:
            baseline_vals = summary[metric_name][baseline]

            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(zeta_vals, baseline_vals, alternative='two-sided')

            # Efecto direccional
            zeta_mean = np.mean(zeta_vals)
            baseline_mean = np.mean(baseline_vals)
            diff = zeta_mean - baseline_mean
            direction = "+" if diff > 0 else "-"

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            print(f"  vs {baseline:8}: Δ={diff:+.3f} ({direction}), p={p_value:.4f} {sig}")

            if p_value < 0.05:
                significant_advantages.append((metric_name, baseline, diff, p_value))

    # Índice Zeta Advantage
    print("\n" + "=" * 70)
    print("ÍNDICE ZETA ADVANTAGE (vs promedio de baselines)")
    print("=" * 70)

    zeta_advantage_components = []

    # Usar métricas continuas para un índice más sensible
    for metric_name in ['blend_entropy', 'trajectory_length',
                        'position_oscillation', 'position_variance']:
        zeta_mean = np.mean(summary[metric_name]['ZETA'])
        baseline_mean = np.mean([
            np.mean(summary[metric_name]['UNIFORM']),
            np.mean(summary[metric_name]['NONE']),
            np.mean(summary[metric_name]['RANDOM'])
        ])

        if baseline_mean != 0:
            ratio = zeta_mean / baseline_mean
        else:
            ratio = 1.0 if zeta_mean == 0 else float('inf')

        zeta_advantage_components.append(ratio)
        print(f"  {metric_name}: {ratio:.3f}")

    zeta_advantage = np.mean(zeta_advantage_components)
    print(f"\n  ZETA ADVANTAGE INDEX: {zeta_advantage:.3f}")
    print(f"  Interpretación: {'ZETA es superior' if zeta_advantage > 1.0 else 'Baseline es superior'}")

    return summary, significant_advantages, zeta_advantage


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_results(all_results: Dict[str, List[ExperimentMetrics]], summary: Dict):
    """Genera visualizaciones de resultados."""
    fig = plt.figure(figsize=(16, 12))

    conditions = ['ZETA', 'UNIFORM', 'NONE', 'RANDOM']
    colors = {'ZETA': '#E53E3E', 'UNIFORM': '#3182CE', 'NONE': '#718096', 'RANDOM': '#38A169'}

    # 1. Boxplot de entropía arquetipal
    ax1 = fig.add_subplot(2, 3, 1)
    data = [summary['archetype_entropy'][c] for c in conditions]
    bp = ax1.boxplot(data, labels=conditions, patch_artist=True)
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.7)
    ax1.set_ylabel('Entropía')
    ax1.set_title('Entropía Arquetipal\n(Mayor = Más Variado)')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # 2. Boxplot de tasa de transiciones
    ax2 = fig.add_subplot(2, 3, 2)
    data = [summary['transition_rate'][c] for c in conditions]
    bp = ax2.boxplot(data, labels=conditions, patch_artist=True)
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Tasa')
    ax2.set_title('Tasa de Transiciones\n(Óptimo ~ 0.3-0.5)')
    ax2.axhline(y=0.4, color='green', linestyle='--', alpha=0.5)

    # 3. Boxplot de integración final
    ax3 = fig.add_subplot(2, 3, 3)
    data = [summary['final_integration'][c] for c in conditions]
    bp = ax3.boxplot(data, labels=conditions, patch_artist=True)
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.7)
    ax3.set_ylabel('Integración')
    ax3.set_title('Integración Final\n(Mayor = Más Individuado)')

    # 4. Boxplot de velocidad de integración
    ax4 = fig.add_subplot(2, 3, 4)
    data = [summary['integration_velocity'][c] for c in conditions]
    bp = ax4.boxplot(data, labels=conditions, patch_artist=True)
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.7)
    ax4.set_ylabel('Δ Integración')
    ax4.set_title('Velocidad de Individuación\n(Mayor = Progreso Más Rápido)')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 5. Boxplot de compensación onírica
    ax5 = fig.add_subplot(2, 3, 5)
    data = [summary['dream_compensation'][c] for c in conditions]
    bp = ax5.boxplot(data, labels=conditions, patch_artist=True)
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.7)
    ax5.set_ylabel('Δ Balance')
    ax5.set_title('Compensación Onírica\n(Mayor = Sueños Más Efectivos)')
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 6. Radar chart comparativo
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')

    metrics_for_radar = ['archetype_entropy', 'transition_rate', 'final_integration',
                         'integration_stability', 'dream_compensation']
    labels = ['Entropía', 'Transiciones', 'Integración', 'Estabilidad', 'Sueños']

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for cond in conditions:
        values = []
        for metric in metrics_for_radar:
            val = np.mean(summary[metric][cond])
            # Normalizar al rango [0, 1]
            all_vals = [np.mean(summary[metric][c]) for c in conditions]
            if max(all_vals) - min(all_vals) > 0:
                val_norm = (val - min(all_vals)) / (max(all_vals) - min(all_vals))
            else:
                val_norm = 0.5
            values.append(val_norm)
        values += values[:1]

        ax6.plot(angles, values, 'o-', linewidth=2, label=cond, color=colors[cond])
        ax6.fill(angles, values, alpha=0.1, color=colors[cond])

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(labels)
    ax6.set_title('Perfil Comparativo')
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"zeta_vs_baseline_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nGráfico guardado: {filename}")

    plt.show()

    return filename


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "   EXPERIMENTO: ¿QUÉ APORTAN LOS CEROS DE RIEMANN?".center(68) + "║")
    print("║" + "   (v4: 500 pasos - diferencias acumulativas ZETA vs UNIFORM)".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    # Ejecutar experimento LARGO con 500 estímulos ambiguos
    # Hipótesis: Con más pasos, las diferencias entre frecuencias zeta
    # y uniformes podrían acumularse y volverse significativas
    all_results = run_full_experiment(n_replicas=10, attraction_coef=0.1, ambiguous="long")

    # Analizar
    summary, significant_advantages, zeta_advantage = analyze_results(all_results)

    # Visualizar
    plot_results(all_results, summary)

    # Resumen final
    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)

    if zeta_advantage > 1.1:
        print("\n✓ ZETA muestra ventaja significativa sobre los baselines.")
        print("  Los ceros de Riemann aportan estructura beneficiosa.")
    elif zeta_advantage < 0.9:
        print("\n✗ Los baselines superan a ZETA.")
        print("  La modulación zeta podría no ser beneficiosa en este contexto.")
    else:
        print("\n≈ Resultados mixtos.")
        print("  ZETA y baselines tienen rendimiento similar.")

    print(f"\nVentajas significativas de ZETA (p < 0.05):")
    if significant_advantages:
        for metric, baseline, diff, p in significant_advantages:
            if diff > 0:
                print(f"  • {metric} vs {baseline}: +{diff:.3f} (p={p:.4f})")
    else:
        print("  Ninguna ventaja estadísticamente significativa.")

    # Guardar resultados JSON
    results_json = {
        'zeta_advantage_index': zeta_advantage,
        'significant_advantages': [(m, b, float(d), float(p)) for m, b, d, p in significant_advantages],
        'summary': {
            metric: {
                cond: {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': [float(v) for v in vals]
                }
                for cond, vals in cond_vals.items()
            }
            for metric, cond_vals in summary.items()
        }
    }

    json_filename = f"zeta_vs_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(json_filename, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResultados guardados: {json_filename}")


if __name__ == "__main__":
    main()
