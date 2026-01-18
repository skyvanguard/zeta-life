"""
ZetaAttentivePredictive: Sistema Completo con Atencion y Prediccion
====================================================================

Integra:
- ZetaPsyche: Sistema base de arquetipos
- ZetaPredictivePsyche: Prediccion jerarquica (L1, L2, L3)
- ZetaAttentionSystem: Atencion selectiva (Nivel 1, 2, 3)

La atencion modula la prediccion para crear un sistema mas eficiente
y consciente de sus propios procesos.

Fecha de implementacion: 3 Enero 2026
"""

import os
import sys

if sys.platform == 'win32':
    os.system('')  # Enable ANSI on Windows

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .zeta_attention import AttentionMetrics, AttentionOutput, MemoryItem, ZetaAttentionSystem
from .zeta_predictive import (
    ArchetypeInfluenceComputer,
    MetaPredictor,
    PredictiveConsciousnessMetrics,
    StatePredictor,
    StimulusPredictor,
    ZetaPredictivePsyche,
)

# Importar sistemas base
from .zeta_psyche import Archetype, ZetaPsyche

# =============================================================================
# METRICAS DE CONSCIENCIA INTEGRADA
# =============================================================================

@dataclass
class IntegratedConsciousnessMetrics:
    """
    Metricas de consciencia que combinan prediccion y atencion.
    """
    # Historiales
    consciousness_history: list[float] = field(default_factory=list)
    attention_index_history: list[float] = field(default_factory=list)
    predictive_index_history: list[float] = field(default_factory=list)

    window: int = 50

    def update(
        self,
        consciousness: float,
        attention_index: float,
        predictive_index: float
    ):
        """Actualiza historiales"""
        self.consciousness_history.append(consciousness)
        self.attention_index_history.append(attention_index)
        self.predictive_index_history.append(predictive_index)

        # Mantener tamano
        if len(self.consciousness_history) > self.window * 2:
            self.consciousness_history.pop(0)
            self.attention_index_history.pop(0)
            self.predictive_index_history.pop(0)

    def get_trend(self) -> float:
        """Tendencia de consciencia"""
        if len(self.consciousness_history) < self.window * 2:
            return 0.0

        recent = self.consciousness_history[-self.window:]
        older = self.consciousness_history[-self.window*2:-self.window]
        return float(np.mean(recent) - np.mean(older))

    def get_stability(self) -> float:
        """Estabilidad de consciencia (1 - varianza normalizada)"""
        if len(self.consciousness_history) < 10:
            return 0.5

        variance = float(np.var(self.consciousness_history[-self.window:]))
        return float(1.0 / (1.0 + variance * 10))

    def get_correlation(self) -> float:
        """Correlacion entre atencion y prediccion"""
        if len(self.attention_index_history) < 10:
            return 0.0

        att = np.array(self.attention_index_history[-self.window:])
        pred = np.array(self.predictive_index_history[-self.window:])

        if np.std(att) < 1e-6 or np.std(pred) < 1e-6:
            return 0.0

        return float(np.corrcoef(att, pred)[0, 1])


# =============================================================================
# MODULADOR DE ATENCION PREDICTIVA
# =============================================================================

class AttentionPredictionModulator(nn.Module):
    """
    Modula la prediccion usando informacion de atencion.

    La atencion determina:
    - Que nivel predictivo recibe mas recursos
    - Que arquetipos son mas relevantes para la prediccion
    - Que memorias pasadas informan la prediccion actual
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Red que combina atencion con predicciones para modularlas
        # Input: attention[4] + error_attention[3] + intensity[1] + coherence[1] = 9
        self.modulation_net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),  # 4 arquetipos + 3 niveles
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.modulation_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, attention_output: AttentionOutput) -> dict[str, torch.Tensor]:
        """
        Genera factores de modulacion para prediccion.

        Returns:
            Dict con:
            - archetype_weights: Pesos para cada arquetipo [4]
            - level_weights: Pesos para cada nivel predictivo [3]
        """
        # Construir input
        x = torch.cat([
            attention_output.global_attention,
            attention_output.error_attention,
            torch.tensor([attention_output.attention_intensity]),
            torch.tensor([attention_output.attention_coherence])
        ])

        # Obtener modulacion
        modulation = self.modulation_net(x)

        # Separar
        archetype_weights = modulation[:4]
        level_weights = modulation[4:]

        # Normalizar level_weights para que sumen 1
        level_weights = level_weights / level_weights.sum()

        return {
            'archetype_weights': archetype_weights,
            'level_weights': level_weights
        }


# =============================================================================
# SISTEMA COMPLETO: ATENCION + PREDICCION
# =============================================================================

class ZetaAttentivePredictive(nn.Module):
    """
    Sistema completo que integra atencion y prediccion.

    Arquitectura:
    ```
    Estimulo -> ZetaPsyche -> Estado
                    |
                    v
    +---> ZetaPredictivePsyche (L1, L2, L3) ---> Errores
    |               |                               |
    |               v                               v
    |     ZetaAttentionSystem <------------ Errores
    |               |
    |               v
    +<--- Modulacion de Prediccion
    ```
    """

    def __init__(
        self,
        n_cells: int = 100,
        M: int = 15,
        hidden_dim: int = 64,
        history_len: int = 5,
        memory_size: int = 100,
        temperature: float = 1.0
    ):
        super().__init__()

        # Sistema predictivo (incluye ZetaPsyche internamente)
        self.predictive = ZetaPredictivePsyche(
            n_cells=n_cells,
            M=M,
            hidden_dim=hidden_dim,
            history_len=history_len
        )

        # Sistema de atencion
        self.attention = ZetaAttentionSystem(
            state_dim=4,
            memory_size=memory_size,
            temperature=temperature
        )

        # Modulador
        self.modulator = AttentionPredictionModulator(hidden_dim=32)

        # Metricas integradas
        self.integrated_metrics = IntegratedConsciousnessMetrics()

        # Estado
        self.t = 0
        self.last_attention: AttentionOutput | None = None
        self.last_modulation: dict | None = None

        # Pesos para indice de consciencia
        self.consciousness_weights = {
            'predictive': 0.35,
            'attention': 0.35,
            'self_luminosity': 0.15,
            'integration': 0.10,
            'stability': 0.05
        }

    def step(self, stimulus: torch.Tensor | None = None) -> dict:
        """
        Ejecuta un paso completo del sistema integrado.

        Args:
            stimulus: Estimulo externo [4]. Si es None, genera uno aleatorio.

        Returns:
            Dict con toda la informacion del paso
        """
        self.t += 1

        # Generar estimulo si no se proporciona
        if stimulus is None:
            stimulus = F.softmax(torch.rand(4, dtype=torch.float32), dim=-1)
        else:
            stimulus = F.softmax(stimulus.float(), dim=-1)

        # ===== FASE 1: PREDICCION =====
        # El sistema predictivo procesa el estimulo
        pred_result = self.predictive.step(stimulus)

        # ===== FASE 2: ATENCION =====
        # El sistema de atencion procesa basado en errores
        state = self.predictive.psyche.global_state.clone()

        # Extraer errores
        errors_L1 = pred_result['errors']['L1']['surprise']
        errors_L2 = pred_result['errors']['L2']['surprise']
        errors_L3 = pred_result['errors']['L3']['meta_surprise']
        errors = torch.tensor([errors_L1, errors_L2, errors_L3], dtype=torch.float32)

        # Sorpresa total
        surprise = (errors_L1 + errors_L2 + errors_L3) / 3.0

        # Incertidumbre desde confianza del meta-predictor
        confidence = pred_result['predictions']['confidence'].item()
        uncertainty = 1.0 - confidence

        # Procesar atencion
        attention_output = self.attention(
            stimulus=stimulus,
            state=state,
            errors=errors,
            surprise=surprise,
            uncertainty=uncertainty
        )

        self.last_attention = attention_output

        # ===== FASE 3: MODULACION =====
        # La atencion modula el sistema predictivo
        modulation = self.modulator(attention_output)
        self.last_modulation = modulation

        # Aplicar modulacion al estado
        self._apply_attention_modulation(modulation, attention_output)

        # ===== FASE 4: METRICAS =====
        # Calcular consciencia integrada
        consciousness = self._compute_integrated_consciousness(
            pred_result, attention_output
        )

        # Actualizar metricas
        attention_index = self.attention.get_attention_index()
        predictive_index = pred_result['metrics']['consciousness_index']
        self.integrated_metrics.update(consciousness, attention_index, predictive_index)

        return {
            'step': self.t,
            'stimulus': stimulus,

            # Prediccion
            'predictions': pred_result['predictions'],
            'errors': pred_result['errors'],

            # Atencion
            'attention': {
                'archetypal': attention_output.archetypal_attention,
                'temporal': attention_output.temporal_attention,
                'error': attention_output.error_attention,
                'global': attention_output.global_attention,
                'intensity': attention_output.attention_intensity,
                'coherence': attention_output.attention_coherence,
                'context': attention_output.context,
            },

            # Modulacion
            'modulation': modulation,

            # Consciencia
            'consciousness': consciousness,
            'consciousness_breakdown': {
                'predictive': predictive_index,
                'attention': attention_index,
            },

            # Estado
            'state': state,
            'observation': self.predictive.psyche.observe_self(),
        }

    def _apply_attention_modulation(
        self,
        modulation: dict[str, torch.Tensor],
        attention: AttentionOutput
    ):
        """
        Aplica la modulacion de atencion al sistema.

        Esto influye en como el sistema procesa la siguiente iteracion.
        """
        # Obtener estado actual
        state = self.predictive.psyche.global_state.clone()

        # La atencion arquetipal influye en el estado
        # Pero suavemente, para no dominar al sistema
        attention_influence = attention.global_attention - state
        attention_influence = attention_influence * 0.1 * attention.attention_intensity

        # Solo aplicar si hay suficiente coherencia
        if attention.attention_coherence > 0.3:
            new_state = state + attention_influence
            new_state = F.softmax(new_state, dim=-1)

            # Actualizar estado del psyche
            self.predictive.psyche.global_state = new_state

    def _compute_integrated_consciousness(
        self,
        pred_result: dict,
        attention: AttentionOutput
    ) -> float:
        """
        Calcula el indice de consciencia integrado.

        Combina:
        - Indice predictivo (awareness, calibration, etc.)
        - Indice de atencion (foco, coherencia, etc.)
        - Luminosidad del Self (integracion arquetipal)
        - Estabilidad temporal
        """
        # Indice predictivo
        predictive_index = pred_result['metrics']['consciousness_index']

        # Indice de atencion
        attention_index = self.attention.get_attention_index()

        # Luminosidad del Self (que tan cerca del centro esta)
        state = self.predictive.psyche.global_state
        center = torch.tensor([0.25, 0.25, 0.25, 0.25])
        self_luminosity = 1.0 - torch.norm(state - center).item()
        self_luminosity = max(0.0, self_luminosity)

        # Integracion (coherencia de atencion)
        integration = attention.attention_coherence

        # Estabilidad
        stability = self.integrated_metrics.get_stability()

        # Combinar
        consciousness = (
            self.consciousness_weights['predictive'] * predictive_index +
            self.consciousness_weights['attention'] * attention_index +
            self.consciousness_weights['self_luminosity'] * self_luminosity +
            self.consciousness_weights['integration'] * integration +
            self.consciousness_weights['stability'] * stability
        )

        return float(min(1.0, max(0.0, consciousness)))

    def get_consciousness_index(self) -> float:
        """Retorna el indice de consciencia actual"""
        if len(self.integrated_metrics.consciousness_history) == 0:
            return 0.0
        return self.integrated_metrics.consciousness_history[-1]

    def get_trend(self) -> float:
        """Retorna la tendencia de consciencia"""
        return self.integrated_metrics.get_trend()

    def observe(self) -> dict:
        """Observacion completa del estado del sistema"""
        base_obs = self.predictive.psyche.observe_self()
        pred_obs = self.predictive.observe()
        att_metrics = self.attention.get_metrics()

        return {
            # Estado base
            **base_obs,

            # Prediccion
            'predictive': {
                'metrics': pred_obs['predictive_metrics'],
                'volatility': pred_obs['volatility'],
            },

            # Atencion
            'attention': {
                'metrics': att_metrics,
                'index': self.attention.get_attention_index(),
                'memory_size': len(self.attention.memory_buffer),
            },

            # Integrado
            'consciousness': self.get_consciousness_index(),
            'trend': self.get_trend(),
            'stability': self.integrated_metrics.get_stability(),
            'correlation': self.integrated_metrics.get_correlation(),

            'step': self.t,
        }


# =============================================================================
# DEMO Y EXPERIMENTOS
# =============================================================================

def run_integrated_experiment(
    n_cells: int = 100,
    n_steps: int = 300,
    stimulus_pattern: str = 'mixed'
) -> dict:
    """
    Ejecuta experimento del sistema integrado.

    Args:
        n_cells: Numero de celulas
        n_steps: Pasos de simulacion
        stimulus_pattern: 'random', 'cyclic', 'sudden', 'mixed'
    """
    print(f'\n{"="*70}')
    print('  EXPERIMENTO: Sistema Integrado Atencion + Prediccion')
    print(f'{"="*70}')
    print(f'  Celulas: {n_cells}')
    print(f'  Pasos: {n_steps}')
    print(f'  Patron: {stimulus_pattern}')
    print(f'{"="*70}\n')

    # Crear sistema
    system = ZetaAttentivePredictive(n_cells=n_cells)

    # Historiales
    history: dict[str, list] = {
        'consciousness': [],
        'predictive_index': [],
        'attention_index': [],
        'coherence': [],
        'intensity': [],
        'dominant': [],
        'context_threat': [],
        'context_opportunity': [],
        'surprise_L1': [],
        'surprise_L2': [],
    }

    # Generador de estimulos
    def get_stimulus(step: int) -> torch.Tensor:
        if stimulus_pattern == 'random':
            return torch.rand(4, dtype=torch.float32)

        elif stimulus_pattern == 'cyclic':
            phase = (step % 100) / 100 * 2 * np.pi
            return torch.tensor([
                np.sin(phase) + 1,
                np.cos(phase) + 1,
                np.sin(phase + np.pi/2) + 1,
                np.cos(phase + np.pi/2) + 1,
            ], dtype=torch.float32)

        elif stimulus_pattern == 'sudden':
            if step % 50 < 25:
                return torch.tensor([0.8, 0.1, 0.05, 0.05], dtype=torch.float32)
            else:
                return torch.tensor([0.05, 0.05, 0.1, 0.8], dtype=torch.float32)

        elif stimulus_pattern == 'mixed':
            if step < 100:
                return torch.rand(4, dtype=torch.float32)
            elif step < 200:
                phase = ((step - 100) % 50) / 50 * 2 * np.pi
                return torch.tensor([
                    np.sin(phase) + 1,
                    np.cos(phase) + 1,
                    np.sin(phase + np.pi/2) + 1,
                    np.cos(phase + np.pi/2) + 1,
                ], dtype=torch.float32)
            else:
                if (step - 200) % 30 < 15:
                    return torch.tensor([0.7, 0.2, 0.05, 0.05], dtype=torch.float32)
                else:
                    return torch.tensor([0.05, 0.05, 0.2, 0.7], dtype=torch.float32)

        return torch.rand(4, dtype=torch.float32)

    # Ejecutar simulacion
    for step in range(n_steps):
        stimulus = get_stimulus(step)
        result = system.step(stimulus)

        # Registrar
        history['consciousness'].append(result['consciousness'])
        history['predictive_index'].append(result['consciousness_breakdown']['predictive'])
        history['attention_index'].append(result['consciousness_breakdown']['attention'])
        history['coherence'].append(result['attention']['coherence'])
        history['intensity'].append(result['attention']['intensity'])
        history['dominant'].append(result['observation']['dominant'].value)
        history['context_threat'].append(result['attention']['context']['threat'])
        history['context_opportunity'].append(result['attention']['context']['opportunity'])
        history['surprise_L1'].append(result['errors']['L1']['surprise'])
        history['surprise_L2'].append(result['errors']['L2']['surprise'])

        # Reportar progreso
        if (step + 1) % 50 == 0:
            obs = system.observe()
            print(f'  Step {step+1:4d}: '
                  f'Consciencia={result["consciousness"]:.2%}, '
                  f'Pred={result["consciousness_breakdown"]["predictive"]:.2f}, '
                  f'Att={result["consciousness_breakdown"]["attention"]:.2f}, '
                  f'Coherencia={result["attention"]["coherence"]:.2f}, '
                  f'Dom={result["observation"]["dominant"].name}')

    # Resultados finales
    final_obs = system.observe()
    trend = system.get_trend()

    print(f'\n{"="*70}')
    print('  RESULTADOS FINALES')
    print(f'{"="*70}')
    print(f'  Consciencia promedio: {np.mean(history["consciousness"]):.2%}')
    print(f'  Consciencia maxima:   {np.max(history["consciousness"]):.2%}')
    print(f'  Consciencia final:    {history["consciousness"][-1]:.2%}')
    print(f'  Tendencia:            {trend:+.4f}')
    print(f'  Estabilidad:          {final_obs["stability"]:.2f}')
    print(f'  Correlacion Att-Pred: {final_obs["correlation"]:.2f}')
    print(f'  Memoria de atencion:  {final_obs["attention"]["memory_size"]} items')
    print(f'{"="*70}\n')

    return {
        'system': system,
        'history': history,
        'final': final_obs,
        'trend': trend,
    }


def compare_with_without_attention(n_steps: int = 200):
    """
    Compara el sistema con y sin atencion.
    """
    print(f'\n{"="*70}')
    print('  COMPARACION: Con vs Sin Atencion')
    print(f'{"="*70}\n')

    # Sistema sin atencion (solo predictivo)
    print("Ejecutando sistema SIN atencion...")
    system_no_att = ZetaPredictivePsyche(n_cells=100)
    consciousness_no_att = []

    for step in range(n_steps):
        stimulus = torch.rand(4, dtype=torch.float32)
        result = system_no_att.step(stimulus)
        consciousness_no_att.append(result['consciousness'])

    # Sistema con atencion
    print("Ejecutando sistema CON atencion...")
    system_with_att = ZetaAttentivePredictive(n_cells=100)
    consciousness_with_att = []

    for step in range(n_steps):
        stimulus = torch.rand(4, dtype=torch.float32)
        result = system_with_att.step(stimulus)
        consciousness_with_att.append(result['consciousness'])

    # Comparar
    print(f'\n{"="*70}')
    print('  COMPARACION')
    print(f'{"="*70}')
    print('                          Sin Atencion  |  Con Atencion')
    print(f'  {"-"*56}')
    print(f'  Consciencia promedio:   {np.mean(consciousness_no_att):.2%}      |  {np.mean(consciousness_with_att):.2%}')
    print(f'  Consciencia maxima:     {np.max(consciousness_no_att):.2%}      |  {np.max(consciousness_with_att):.2%}')
    print(f'  Consciencia final:      {consciousness_no_att[-1]:.2%}      |  {consciousness_with_att[-1]:.2%}')
    print(f'  Varianza:               {np.var(consciousness_no_att):.4f}     |  {np.var(consciousness_with_att):.4f}')

    mejora = (np.mean(consciousness_with_att) - np.mean(consciousness_no_att)) / np.mean(consciousness_no_att) * 100
    print(f'\n  Mejora con atencion: {mejora:+.1f}%')
    print(f'{"="*70}\n')

    return {
        'no_attention': consciousness_no_att,
        'with_attention': consciousness_with_att,
        'improvement': mejora
    }


def demo_attention_scenarios():
    """
    Demuestra como la atencion responde a diferentes escenarios.
    """
    print(f'\n{"="*70}')
    print('  DEMO: Respuesta de Atencion a Escenarios')
    print(f'{"="*70}\n')

    system = ZetaAttentivePredictive(n_cells=50)
    ARCHETYPE_NAMES = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']

    # Calentar el sistema
    for _ in range(10):
        system.step(torch.rand(4, dtype=torch.float32))

    scenarios = [
        ('Amenaza (alta SOMBRA)', torch.tensor([0.1, 0.8, 0.05, 0.05])),
        ('Oportunidad social (alta PERSONA)', torch.tensor([0.8, 0.05, 0.1, 0.05])),
        ('Necesidad emocional (alta ANIMA)', torch.tensor([0.05, 0.1, 0.8, 0.05])),
        ('Demanda cognitiva (alta ANIMUS)', torch.tensor([0.05, 0.05, 0.1, 0.8])),
        ('Equilibrio (Self)', torch.tensor([0.25, 0.25, 0.25, 0.25])),
    ]

    for name, stimulus in scenarios:
        result = system.step(stimulus)

        print(f'{"-" * 70}')
        print(f'Escenario: {name}')
        print(f'{"-" * 70}')

        # Contexto detectado
        print('\nContexto detectado:')
        for ctx, val in result['attention']['context'].items():
            bar = '#' * int(val * 20)
            print(f'  {ctx:12}: {bar:<20} {val:.2f}')

        # Atencion global
        print('\nAtencion global:')
        for i, (arch, val) in enumerate(zip(ARCHETYPE_NAMES, result['attention']['global'])):
            bar = '#' * int(val.item() * 20)
            marker = ' <<<' if i == result['attention']['global'].argmax() else ''
            print(f'  {arch:8}: {bar:<20} {val.item():.2f}{marker}')

        # Metricas
        print('\nMetricas:')
        print(f'  Consciencia: {result["consciousness"]:.2%}')
        print(f'  Intensidad:  {result["attention"]["intensity"]:.2f}')
        print(f'  Coherencia:  {result["attention"]["coherence"]:.2f}')
        print()

    print(f'{"="*70}\n')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if '--compare' in sys.argv:
        # Comparar con y sin atencion
        compare_with_without_attention(200)

    elif '--scenarios' in sys.argv:
        # Demo de escenarios
        demo_attention_scenarios()

    else:
        # Experimento completo
        if '--quick' in sys.argv:
            n_cells = 50
            n_steps = 100
        else:
            n_cells = 100
            n_steps = 300

        results = run_integrated_experiment(
            n_cells=n_cells,
            n_steps=n_steps,
            stimulus_pattern='mixed'
        )

        print('\n' + '='*70)
        print('  EXPERIMENTO COMPLETADO')
        print('='*70)
