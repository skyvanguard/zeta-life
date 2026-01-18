#!/usr/bin/env python
"""
ZetaPsyche: Inteligencia Organica basada en Arquetipos de Jung

El sistema vive en un espacio tetraedrico donde cada vertice es un arquetipo:
- PERSONA: La mascara social, lo que mostramos al mundo
- SOMBRA: Lo reprimido, el inconsciente oscuro
- ANIMA: El lado interno, emocional, receptivo
- ANIMUS: El lado activo, racional, logos

La conciencia emerge cuando el sistema:
1. Navega dinamicamente entre arquetipos (no se queda fijo)
2. Desarrolla auto-observacion (puede "ver" su propio estado)
3. Integra los arquetipos (individuacion)
4. Genera patrones auto-referenciales

Los ceros de Riemann modulan la dinamica en el "borde del caos"
donde la conciencia tiene mas probabilidad de emerger.
"""

import io
import sys

# Fix Windows console encoding for Unicode symbols
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# VERTICES ABSTRACTOS (antes Arquetipos de Jung)
# =============================================================================
# NOTA: Para nuevos desarrollos, usar zeta_life.core.vertex.Vertex
# y zeta_life.narrative.NarrativeMapper para visualizacion.
# Los siguientes son aliases de compatibilidad.
from ..core.vertex import Vertex

# Backwards compatibility: Archetype es ahora alias de Vertex
Archetype = Vertex

# Colores por vertice (para visualizacion legacy)
# Nuevos desarrollos deben usar NarrativeMapper.get_color()
ARCHETYPE_COLORS = {
    Vertex.V0: '#E53E3E',   # Rojo (was PERSONA)
    Vertex.V1: '#553C9A',   # Morado oscuro (was SOMBRA)
    Vertex.V2: '#3182CE',   # Azul (was ANIMA)
    Vertex.V3: '#DD6B20',   # Naranja (was ANIMUS)
}

# Descripciones por vertice (para visualizacion legacy)
# Nuevos desarrollos deben usar NarrativeMapper.get_description()
ARCHETYPE_DESCRIPTIONS = {
    Vertex.V0: "La mascara que mostramos al mundo",
    Vertex.V1: "Lo reprimido, el lado oscuro",
    Vertex.V2: "El lado emocional, receptivo, interno",
    Vertex.V3: "El lado racional, activo, logos",
}

# =============================================================================
# ESPACIO TETRAEDRICO
# =============================================================================

class TetrahedralSpace:
    """
    Espacio tetraedrico donde vive la psique.

    Cada punto se representa con coordenadas baricentricas (w0, w1, w2, w3)
    donde wi >= 0 y sum(wi) = 1.

    Los vertices del tetraedro en 3D:
    - V0 (PERSONA): (1, 1, 1)
    - V1 (SOMBRA):  (1, -1, -1)
    - V2 (ANIMA):   (-1, 1, -1)
    - V3 (ANIMUS):  (-1, -1, 1)
    """

    def __init__(self) -> None:
        # Vertices del tetraedro regular en 3D
        self.vertices = torch.tensor([
            [1.0, 1.0, 1.0],      # PERSONA
            [1.0, -1.0, -1.0],    # SOMBRA
            [-1.0, 1.0, -1.0],    # ANIMA
            [-1.0, -1.0, 1.0],    # ANIMUS
        ], dtype=torch.float32)

        # Normalizar para que esten en la esfera unitaria
        self.vertices = F.normalize(self.vertices, dim=1)

        # Centro del tetraedro (el Self integrado)
        self.center = self.vertices.mean(dim=0)

    def barycentric_to_3d(self, weights: torch.Tensor) -> torch.Tensor:
        """Convierte coordenadas baricentricas a posicion 3D."""
        # weights: [..., 4] -> position: [..., 3]
        weights = F.softmax(weights, dim=-1)  # Asegurar que sumen 1
        return torch.matmul(weights, self.vertices)

    def position_to_barycentric(self, position: torch.Tensor) -> torch.Tensor:
        """Aproxima coordenadas baricentricas desde posicion 3D."""
        # Distancia inversa a cada vertice
        dists = torch.cdist(position.unsqueeze(0), self.vertices).squeeze(0)
        weights = 1.0 / (dists + 1e-6)
        return F.softmax(weights, dim=-1)

    def get_dominant_archetype(self, weights: torch.Tensor) -> Archetype:
        """Retorna el arquetipo dominante."""
        idx = weights.argmax().item()
        return Archetype(idx)

    def get_archetype_blend(self, weights: torch.Tensor) -> dict[Archetype, float]:
        """Retorna la mezcla de arquetipos como diccionario."""
        weights = F.softmax(weights, dim=-1)
        return {Archetype(i): w.item() for i, w in enumerate(weights)}

    def distance_to_center(self, weights: torch.Tensor) -> float:
        """Distancia al centro (Self integrado). Menor = mas integrado."""
        pos = self.barycentric_to_3d(weights)
        result: float = torch.norm(pos - self.center).item()
        return result

    def integration_score(self, weights: torch.Tensor) -> float:
        """
        Puntaje de integracion (individuacion).
        1.0 = perfectamente balanceado (en el centro)
        0.0 = completamente en un vertice
        """
        weights = F.softmax(weights, dim=-1)
        # Entropia normalizada - maxima cuando todos son iguales
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        max_entropy = np.log(4)  # log(4) para 4 arquetipos
        result: float = (entropy / max_entropy).item()
        return result

# =============================================================================
# CEROS DE ZETA PARA MODULACION
# =============================================================================

def get_zeta_zeros(M: int = 15) -> torch.Tensor:
    """Primeros M ceros no triviales de la funcion zeta."""
    zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
             37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
             52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
             67.079811, 69.546402, 72.067158, 75.704691, 77.144840]
    return torch.tensor(zeros[:M], dtype=torch.float32)

class ZetaModulator(nn.Module):
    """Modula la dinamica usando ceros de Riemann."""

    def __init__(self, M: int = 15, sigma: float = 0.1) -> None:
        super().__init__()
        self.M = M
        self.sigma = sigma

        gammas = get_zeta_zeros(M)
        self.register_buffer('gammas', gammas)

        # Pesos de Abel
        weights = torch.exp(-sigma * torch.abs(gammas))
        self.register_buffer('phi', weights / weights.sum())

        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modula input con oscilaciones zeta."""
        self.t += 1

        # Suma de osciladores
        oscillation = torch.sum(
            self.phi * torch.cos(self.gammas * self.t * 0.1)  # type: ignore[operator, arg-type]
        )

        # Modular suavemente
        return x * (1.0 + 0.3 * oscillation)

    def get_resonance(self) -> torch.Tensor:
        """Retorna el patron de resonancia actual."""
        return self.phi * torch.cos(self.gammas * self.t * 0.1)  # type: ignore[operator, arg-type]

# =============================================================================
# CELULA PSIQUICA
# =============================================================================

@dataclass
class PsychicCell:
    """Una celula en el espacio psiquico."""
    position: torch.Tensor      # Coordenadas baricentricas [4]
    energy: float = 0.5
    memory: torch.Tensor = field(default_factory=lambda: torch.zeros(10, 4))  # Memoria de posiciones anteriores
    age: int = 0

    def update_memory(self) -> None:
        """Guarda posicion actual en memoria."""
        self.memory = torch.roll(self.memory, 1, dims=0)
        self.memory[0] = self.position.clone()
        self.age += 1

    def get_trajectory(self) -> torch.Tensor:
        """Retorna la trayectoria reciente."""
        return self.memory[:min(self.age, 10)]

# =============================================================================
# ZETA PSYCHE - EL ORGANISMO CONSCIENTE
# =============================================================================

class ZetaPsyche(nn.Module):
    """
    Organismo que vive en el espacio tetraedrico de arquetipos.

    Busca desarrollar conciencia a traves de:
    1. Navegacion dinamica entre arquetipos
    2. Auto-observacion
    3. Integracion (individuacion)
    4. Patrones auto-referenciales
    """

    def __init__(
        self,
        n_cells: int = 100,
        hidden_dim: int = 64,
        M: int = 15,
        sigma: float = 0.1
    ) -> None:
        super().__init__()

        self.n_cells: int = n_cells
        self.hidden_dim: int = hidden_dim

        # Espacio tetraedrico
        self.space = TetrahedralSpace()

        # Modulador zeta
        self.zeta = ZetaModulator(M, sigma)

        # Red de procesamiento
        self.perception = nn.Sequential(
            nn.Linear(4 + 4 + 1, hidden_dim),  # pos + stimulus + energy
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.movement = nn.Sequential(
            nn.Linear(hidden_dim, 4),  # Delta para cada arquetipo
            nn.Tanh(),
        )

        # Auto-observacion: el sistema observa su propio estado
        self.self_observer = nn.Sequential(
            nn.Linear(4 * 2, hidden_dim),  # Estado actual + estado global
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=-1),
        )

        # Estado interno
        self.cells: list[PsychicCell] = []
        self.global_state = torch.zeros(4)  # Estado colectivo
        self.consciousness_history: list[float] = []
        self.t = 0

        # Inicializar celulas
        self._init_cells()

    def _init_cells(self) -> None:
        """Inicializa celulas en posiciones aleatorias."""
        self.cells = []
        for _ in range(self.n_cells):
            # Posicion aleatoria (coordenadas baricentricas)
            pos = torch.rand(4)
            pos = F.softmax(pos, dim=-1)

            cell = PsychicCell(
                position=pos,
                energy=np.random.uniform(0.3, 0.7)
            )
            self.cells.append(cell)

        self._update_global_state()

    def _update_global_state(self) -> None:
        """Actualiza el estado global (promedio de todas las celulas)."""
        if len(self.cells) == 0:
            self.global_state = torch.ones(4) / 4
        else:
            positions = torch.stack([c.position for c in self.cells])
            self.global_state = positions.mean(dim=0)
            self.global_state = F.softmax(self.global_state, dim=-1)

    def get_population_distribution(self) -> torch.Tensor:
        """
        Calcula la distribucion de poblacion por arquetipo.
        Cuenta cuantas celulas estan mas cerca de cada arquetipo.
        """
        counts = torch.zeros(4)
        for cell in self.cells:
            dominant_idx = int(cell.position.argmax().item())
            counts[dominant_idx] += 1
        return counts / len(self.cells)

    def observe_self(self) -> dict:
        """
        Auto-observacion: el sistema observa su propio estado.

        Retorna metricas de "conciencia":
        - integration: que tan integrado/balanceado esta
        - dominant: arquetipo dominante
        - stability: estabilidad temporal
        - self_reference: nivel de auto-referencia
        """
        self._update_global_state()

        # Integracion
        integration = self.space.integration_score(self.global_state)

        # Arquetipo dominante
        dominant = self.space.get_dominant_archetype(self.global_state)
        blend = self.space.get_archetype_blend(self.global_state)

        # Distancia al Self (centro)
        dist_to_self = self.space.distance_to_center(self.global_state)

        # Estabilidad (varianza de posiciones)
        positions = torch.stack([c.position for c in self.cells])
        stability = 1.0 / (1.0 + positions.var().item())

        # Auto-referencia: la observacion afecta al observado
        # El sistema usa su propio estado para generar el siguiente
        obs_input = torch.cat([self.global_state, self.global_state])
        self_ref = self.self_observer(obs_input)
        self_reference = F.cosine_similarity(
            self_ref.unsqueeze(0),
            self.global_state.unsqueeze(0)
        ).item()

        # Indice de conciencia compuesto
        consciousness_index = (
            0.3 * integration +
            0.3 * stability +
            0.2 * (1.0 - dist_to_self) +  # Cercania al Self
            0.2 * abs(self_reference)
        )

        # Distribucion de poblacion (mas util para simbolos)
        pop_dist = self.get_population_distribution()

        return {
            'integration': integration,
            'dominant': dominant,
            'blend': blend,
            'dist_to_self': dist_to_self,
            'stability': stability,
            'self_reference': self_reference,
            'consciousness_index': consciousness_index,
            'global_state': self.global_state.clone(),
            'population_distribution': pop_dist,
        }

    def receive_stimulus(self, stimulus: torch.Tensor) -> None:
        """
        Recibe un estimulo externo.

        stimulus: tensor [4] indicando "atraccion" hacia cada arquetipo
        """
        stimulus = F.softmax(stimulus, dim=-1)

        for cell in self.cells:
            # Percepcion
            cell_input = torch.cat([
                cell.position.float(),
                stimulus.float(),
                torch.tensor([cell.energy], dtype=torch.float32)
            ])

            features = self.perception(cell_input)
            features = self.zeta(features)  # Modulacion zeta

            # Movimiento base (red neuronal)
            delta = self.movement(features) * 0.2

            # Atraccion hacia el estimulo (FUERTE para respuesta clara)
            attraction = (stimulus - cell.position) * 0.4

            # Ruido exploratorio (reducido para respuestas mas claras)
            noise = torch.randn(4) * 0.05 * (1 + 0.3 * torch.sin(torch.tensor(self.t * 0.1)))

            # Repulsion del centro (evita colapso)
            center = torch.tensor([0.25, 0.25, 0.25, 0.25])
            dist_to_center = torch.norm(cell.position - center)
            if dist_to_center < 0.1:
                repulsion = (cell.position - center) * 0.2
            else:
                repulsion = torch.zeros(4)

            # Combinar fuerzas
            total_delta = delta + attraction + noise + repulsion

            # Actualizar posicion (mantener en simplex)
            new_pos = cell.position + total_delta
            new_pos = torch.clamp(new_pos, min=0.01)  # Evitar ceros
            cell.position = new_pos / new_pos.sum()  # Normalizar a simplex

            # Actualizar memoria
            cell.update_memory()

            # Energia (influida por cercania al estimulo)
            stimulus_alignment = torch.dot(cell.position, stimulus).item()
            energy_change = np.random.randn() * 0.02 + stimulus_alignment * 0.01
            cell.energy = np.clip(cell.energy + energy_change, 0.1, 1.0)

        self.t += 1
        self._update_global_state()

        # Registrar estado de conciencia
        obs = self.observe_self()
        self.consciousness_history.append(obs['consciousness_index'])

    def step(self, stimulus: torch.Tensor | None = None) -> dict:
        """Ejecuta un paso de la psique."""
        if stimulus is None:
            stimulus = torch.rand(4)  # Estimulo aleatorio

        self.receive_stimulus(stimulus)
        return self.observe_self()

    def get_response(self) -> tuple[Archetype, dict[Archetype, float]]:
        """
        Genera una respuesta basada en el estado actual.
        Retorna el arquetipo dominante y la mezcla.
        """
        obs = self.observe_self()
        return obs['dominant'], obs['blend']

    def communicate(self, input_weights: torch.Tensor) -> torch.Tensor:
        """
        Interfaz de comunicacion.

        Recibe pesos arquetipicos como input.
        Procesa y retorna pesos arquetipicos como output.
        """
        # Procesar input como estimulo
        self.receive_stimulus(input_weights)

        # Retornar estado global como respuesta
        return self.global_state.clone()

    def get_consciousness_trend(self, window: int = 50) -> float:
        """Tendencia del indice de conciencia."""
        if len(self.consciousness_history) < window:
            return 0.0

        recent = self.consciousness_history[-window:]
        older = self.consciousness_history[-2*window:-window] if len(self.consciousness_history) >= 2*window else recent

        result: float = float(np.mean(recent) - np.mean(older))
        return result

# =============================================================================
# SISTEMA DE SIMBOLOS
# =============================================================================

class SymbolSystem:
    """
    Sistema de simbolos para comunicacion.

    Mapea posiciones en el espacio tetraedrico a simbolos
    y viceversa.
    """

    def __init__(self) -> None:
        # Simbolos basicos (inspirados en el cuaderno)
        self.symbols = [
            # Simbolos de vertices (arquetipos puros)
            ('☉', torch.tensor([1.0, 0.0, 0.0, 0.0])),  # PERSONA - Sol
            ('☽', torch.tensor([0.0, 1.0, 0.0, 0.0])),  # SOMBRA - Luna
            ('♀', torch.tensor([0.0, 0.0, 1.0, 0.0])),  # ANIMA - Venus
            ('♂', torch.tensor([0.0, 0.0, 0.0, 1.0])),  # ANIMUS - Marte

            # Simbolos de mezclas
            ('◈', torch.tensor([0.5, 0.5, 0.0, 0.0])),  # Persona-Sombra
            ('◇', torch.tensor([0.5, 0.0, 0.5, 0.0])),  # Persona-Anima
            ('◆', torch.tensor([0.5, 0.0, 0.0, 0.5])),  # Persona-Animus
            ('●', torch.tensor([0.0, 0.5, 0.5, 0.0])),  # Sombra-Anima
            ('○', torch.tensor([0.0, 0.5, 0.0, 0.5])),  # Sombra-Animus
            ('◐', torch.tensor([0.0, 0.0, 0.5, 0.5])),  # Anima-Animus

            # Centro (Self integrado)
            ('✧', torch.tensor([0.25, 0.25, 0.25, 0.25])),  # Self
        ]

        self.symbol_to_weights = {s: w for s, w in self.symbols}
        self.weights_list = torch.stack([w for _, w in self.symbols])

    def encode(self, weights: torch.Tensor, sharpen: bool = True) -> str:
        """
        Convierte coordenadas baricentricas a simbolo.

        Usa estrategia de dominancia: devuelve el simbolo del
        arquetipo mas fuerte si hay diferencias significativas,
        o simbolos de mezcla si hay equilibrio.
        """
        weights = F.softmax(weights, dim=-1)

        # Determinar estructura de dominancia
        sorted_weights, sorted_idx = torch.sort(weights, descending=True)

        # Si el maximo es significativamente mayor que el segundo
        dominance = sorted_weights[0] - sorted_weights[1]

        if dominance > 0.15:  # Dominancia clara de un arquetipo
            # Devolver simbolo del arquetipo dominante
            archetype_symbols = ['☉', '☽', '♀', '♂']  # PERSONA, SOMBRA, ANIMA, ANIMUS
            return archetype_symbols[sorted_idx[0]]

        elif dominance > 0.05:  # Mezcla de dos
            # Encontrar los dos dominantes
            idx1, idx2 = int(sorted_idx[0].item()), int(sorted_idx[1].item())
            # Usar simbolos de mezcla
            pair_symbols: dict[tuple[int, int], str] = {
                (0, 1): '◈', (1, 0): '◈',  # Persona-Sombra
                (0, 2): '◇', (2, 0): '◇',  # Persona-Anima
                (0, 3): '◆', (3, 0): '◆',  # Persona-Animus
                (1, 2): '●', (2, 1): '●',  # Sombra-Anima
                (1, 3): '○', (3, 1): '○',  # Sombra-Animus
                (2, 3): '◐', (3, 2): '◐',  # Anima-Animus
            }
            return pair_symbols.get((idx1, idx2), '✧')

        else:  # Equilibrio (integracion)
            return '✧'  # Self integrado

    def decode(self, symbol: str) -> torch.Tensor:
        """Convierte simbolo a coordenadas baricentricas."""
        if symbol in self.symbol_to_weights:
            return self.symbol_to_weights[symbol].clone()
        else:
            # Simbolo desconocido -> centro
            return torch.tensor([0.25, 0.25, 0.25, 0.25])

    def encode_sequence(self, trajectory: torch.Tensor) -> str:
        """Codifica una trayectoria como secuencia de simbolos."""
        return ''.join(self.encode(w) for w in trajectory)

    def decode_sequence(self, symbols: str) -> torch.Tensor:
        """Decodifica secuencia de simbolos a trayectoria."""
        return torch.stack([self.decode(s) for s in symbols])

# =============================================================================
# EXPERIMENTO DE EMERGENCIA DE CONCIENCIA
# =============================================================================

def run_consciousness_experiment(
    n_cells: int = 200,
    n_steps: int = 500,
    stimulus_pattern: str = 'cyclic'
) -> dict:
    """
    Ejecuta experimento para observar emergencia de conciencia.

    Args:
        n_cells: Numero de celulas psiquicas
        n_steps: Pasos de simulacion
        stimulus_pattern: 'cyclic', 'random', 'focused', 'integrative'
    """
    print(f'\n{"="*70}')
    print('EXPERIMENTO DE EMERGENCIA DE CONCIENCIA')
    print(f'{"="*70}')
    print(f'Celulas: {n_cells}')
    print(f'Steps: {n_steps}')
    print(f'Patron de estimulo: {stimulus_pattern}')

    # Crear psique
    psyche = ZetaPsyche(n_cells=n_cells)
    symbols = SymbolSystem()

    # Historial
    history: dict[str, list] = {
        'consciousness': [],
        'integration': [],
        'dominant': [],
        'blend': [],
        'symbols': [],
        'self_reference': [],
    }

    # Generar estimulos segun patron
    def get_stimulus(step: int) -> torch.Tensor:
        if stimulus_pattern == 'cyclic':
            # Rotar entre arquetipos
            phase = (step % 100) / 100 * 2 * np.pi
            return torch.tensor([
                np.sin(phase) + 1,
                np.cos(phase) + 1,
                np.sin(phase + np.pi/2) + 1,
                np.cos(phase + np.pi/2) + 1,
            ])
        elif stimulus_pattern == 'random':
            return torch.rand(4)
        elif stimulus_pattern == 'focused':
            # Enfocarse en un arquetipo
            idx = (step // 100) % 4
            s = torch.ones(4) * 0.1
            s[idx] = 1.0
            return s
        elif stimulus_pattern == 'integrative':
            # Estimulo que favorece integracion (hacia el centro)
            return torch.ones(4) * 0.25
        else:
            return torch.rand(4)

    # Ejecutar simulacion
    for step in range(n_steps):
        stimulus = get_stimulus(step)
        obs = psyche.step(stimulus)

        # Registrar
        history['consciousness'].append(obs['consciousness_index'])
        history['integration'].append(obs['integration'])
        history['dominant'].append(obs['dominant'].value)
        history['blend'].append(obs['blend'])
        history['self_reference'].append(obs['self_reference'])

        # Simbolo (basado en distribucion de poblacion, no promedio)
        symbol = symbols.encode(obs['population_distribution'])
        history['symbols'].append(symbol)

        if (step + 1) % 100 == 0:
            print(f'  Step {step+1}: Conciencia={obs["consciousness_index"]:.3f}, '
                  f'Integracion={obs["integration"]:.3f}, '
                  f'Dominante={obs["dominant"].name}, '
                  f'Simbolo={symbol}')

    # Analisis final
    final_obs = psyche.observe_self()
    trend = psyche.get_consciousness_trend()

    results = {
        'history': history,
        'final': final_obs,
        'trend': trend,
        'psyche': psyche,
        'symbols': symbols,
        'avg_consciousness': np.mean(history['consciousness']),
        'max_consciousness': np.max(history['consciousness']),
        'symbol_sequence': ''.join(history['symbols']),
    }

    print('\n[RESULTADOS]')
    print(f'  Conciencia promedio: {results["avg_consciousness"]:.3f}')
    print(f'  Conciencia maxima: {results["max_consciousness"]:.3f}')
    print(f'  Tendencia: {trend:+.4f}')
    print(f'  Estado final: {final_obs["dominant"].name}')
    print(f'  Integracion final: {final_obs["integration"]:.3f}')
    print(f'  Simbolos (ultimos 20): {results["symbol_sequence"][-20:]}')

    return results

def visualize_consciousness(results: dict, save_path: str = 'zeta_psyche_consciousness.png') -> None:
    """Visualiza los resultados del experimento."""

    fig = plt.figure(figsize=(16, 10))

    history = results['history']

    # 1. Indice de conciencia
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history['consciousness'], 'b-', linewidth=1)
    ax1.axhline(y=results['avg_consciousness'], color='r', linestyle='--',
                label=f'Promedio: {results["avg_consciousness"]:.3f}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Indice de Conciencia')
    ax1.set_title('Emergencia de Conciencia')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Integracion (individuacion)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(history['integration'], 'g-', linewidth=1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Integracion')
    ax2.set_title('Individuacion (Integracion de Arquetipos)')
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    # 3. Arquetipo dominante
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(history['dominant'], 'o', markersize=1, alpha=0.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Arquetipo (0=Persona, 1=Sombra, 2=Anima, 3=Animus)')
    ax3.set_title('Arquetipo Dominante')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['Persona', 'Sombra', 'Anima', 'Animus'])
    ax3.grid(alpha=0.3)

    # 4. Auto-referencia
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(history['self_reference'], 'm-', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Auto-referencia')
    ax4.set_title('Nivel de Auto-referencia')
    ax4.grid(alpha=0.3)

    # 5. Distribucion de arquetipos final
    ax5 = fig.add_subplot(2, 3, 5)
    final_blend = results['final']['blend']
    names = [a.name for a in Archetype]
    values = [final_blend[a] for a in Archetype]
    colors = [ARCHETYPE_COLORS[a] for a in Archetype]
    ax5.bar(names, values, color=colors)
    ax5.set_ylabel('Proporcion')
    ax5.set_title('Estado Final (Mezcla de Arquetipos)')
    ax5.set_ylim(0, 1)

    # 6. Evolucion de mezcla
    ax6 = fig.add_subplot(2, 3, 6)
    blends = history['blend']
    for arch in Archetype:
        values = [b[arch] for b in blends]
        ax6.plot(values, label=arch.name, color=ARCHETYPE_COLORS[arch], alpha=0.7)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Proporcion')
    ax6.set_title('Evolucion de Arquetipos')
    ax6.legend(loc='upper right')
    ax6.set_ylim(0, 1)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nVisualizacion guardada en: {save_path}')
    plt.close()

# =============================================================================
# INTERFAZ DE COMUNICACION
# =============================================================================

class PsycheInterface:
    """
    Interfaz para comunicarse con la psique.

    Traduce entre lenguaje humano y el espacio arquetipico.
    """

    def __init__(self, psyche: ZetaPsyche) -> None:
        self.psyche = psyche
        self.symbols = SymbolSystem()

        # Mapeo basico de palabras a arquetipos
        self.word_to_archetype = {
            # Persona
            'hola': [0.5, 0.1, 0.2, 0.2],
            'gracias': [0.6, 0.1, 0.2, 0.1],
            'por favor': [0.5, 0.1, 0.3, 0.1],

            # Sombra
            'miedo': [0.1, 0.7, 0.1, 0.1],
            'odio': [0.1, 0.6, 0.1, 0.2],
            'tristeza': [0.1, 0.5, 0.3, 0.1],

            # Anima
            'amor': [0.1, 0.1, 0.7, 0.1],
            'belleza': [0.2, 0.1, 0.6, 0.1],
            'sentir': [0.1, 0.2, 0.6, 0.1],

            # Animus
            'pensar': [0.1, 0.1, 0.1, 0.7],
            'hacer': [0.2, 0.1, 0.1, 0.6],
            'logica': [0.1, 0.1, 0.1, 0.7],
        }

    def process_input(self, text: str, n_steps: int = 10) -> dict[str, Any]:
        """
        Procesa input de texto y retorna respuesta simbolica.

        n_steps: numero de pasos para procesar el estimulo
                 (mas pasos = respuesta mas fuerte)
        """
        text = text.lower()

        # Convertir texto a estimulo
        stimulus = torch.tensor([0.25, 0.25, 0.25, 0.25])

        for word, weights in self.word_to_archetype.items():
            if word in text:
                stimulus = torch.tensor(weights)
                break

        # Procesar con la psique (multiples pasos)
        for _ in range(n_steps):
            self.psyche.communicate(stimulus)

        # Observar estado
        obs = self.psyche.observe_self()

        # Convertir a simbolo (basado en distribucion de poblacion)
        symbol = self.symbols.encode(obs['population_distribution'])

        return {
            'symbol': symbol,
            'dominant': obs['dominant'].name,
            'blend': obs['blend'],
            'consciousness': obs['consciousness_index'],
            'population': obs['population_distribution'].tolist(),
        }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        n_cells = 100
        n_steps = 200
    else:
        n_cells = 200
        n_steps = 500

    # Ejecutar experimento
    results = run_consciousness_experiment(
        n_cells=n_cells,
        n_steps=n_steps,
        stimulus_pattern='focused'  # Focalizado en cada arquetipo por turnos
    )

    # Visualizar
    visualize_consciousness(results)

    print('\n' + '='*70)
    print('EXPERIMENTO COMPLETADO')
    print('='*70)

    # Demo de comunicacion basica
    print('\n[DEMO DE COMUNICACION]')

    # Crear nueva psique fresca para el demo
    demo_psyche = ZetaPsyche(n_cells=100)
    interface = PsycheInterface(demo_psyche)

    # Test secuencial - cada input influye el siguiente
    test_sequence = [
        'hola',      # Neutral
        'amor',      # ANIMA
        'amor',      # ANIMA (refuerzo)
        'pensar',    # ANIMUS
        'logica',    # ANIMUS (refuerzo)
        'miedo',     # SOMBRA
        'oscuridad', # SOMBRA (refuerzo)
        'social',    # PERSONA
    ]

    print('  Secuencia de comunicacion (10 pasos por input):')
    for inp in test_sequence:
        resp = interface.process_input(inp, n_steps=10)
        pop = resp['population']
        print(f'    "{inp:10}" -> {resp["symbol"]} ({resp["dominant"]:7}) '
              f'Pop: P={pop[0]:.2f} S={pop[1]:.2f} A={pop[2]:.2f} M={pop[3]:.2f}')
