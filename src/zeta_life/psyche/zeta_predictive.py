#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZetaPredictivePsyche: Sistema de Consciencia con Predicción Jerárquica

Implementa Predictive Processing (teoría de Friston) sobre el sistema de arquetipos:
- Nivel 1: Predice estímulos externos
- Nivel 2: Predice estados internos
- Nivel 3: Predice errores de predicción (meta-cognición)

La consciencia emerge de la capacidad de predecir y de saber cuándo fallará.
"""

import sys
import os

if sys.platform == 'win32':
    os.system('')  # Enable ANSI on Windows

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Deque
from enum import Enum

# Importar sistema base
from .zeta_psyche import ZetaPsyche, Archetype, ZetaModulator, get_zeta_zeros


# =============================================================================
# NIVEL 1: STIMULUS PREDICTOR
# =============================================================================

class StimulusPredictor(nn.Module):
    """
    Nivel 1: Predice estímulos externos.

    Aprende patrones temporales en los estímulos para anticipar
    qué input llegará del mundo externo.
    """

    def __init__(self, history_len: int = 5, hidden_dim: int = 64, M: int = 15) -> None:
        super().__init__()

        self.history_len = history_len
        self.hidden_dim = hidden_dim

        # Historial de estímulos
        self.stimulus_history: Deque[torch.Tensor] = deque(maxlen=history_len)

        # Inicializar historial con ceros
        for _ in range(history_len):
            self.stimulus_history.append(torch.zeros(4))

        # Modulador zeta
        self.zeta = ZetaModulator(M)

        # Red neuronal
        # Input: historial aplanado [history_len * 4] + estado actual [4]
        input_dim = history_len * 4 + 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        # Estadísticas de error
        self.error_history: Deque[float] = deque(maxlen=100)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predice el próximo estímulo basado en historial y estado actual.

        Args:
            state: Estado actual del sistema [4]

        Returns:
            stimulus_predicted: Predicción del próximo estímulo [4]
        """
        # Construir input
        history_flat = torch.cat(list(self.stimulus_history))
        x = torch.cat([history_flat, state])

        # Pasar por red
        features = self.net[:2](x)  # Primera capa + ReLU
        features = self.zeta(features)  # Modulación zeta
        output = self.net[2:](features)  # Resto de la red

        # Softmax para obtener distribución
        return F.softmax(output, dim=-1)

    def compute_error(self, predicted: torch.Tensor, real: torch.Tensor) -> Dict:
        """
        Calcula error entre predicción y realidad.

        Returns:
            Dict con error, surprise, y error por arquetipo
        """
        error = real - predicted
        surprise = torch.norm(error).item() ** 2

        # Error por arquetipo
        error_by_archetype = {
            Archetype.PERSONA: abs(error[0].item()),
            Archetype.SOMBRA: abs(error[1].item()),
            Archetype.ANIMA: abs(error[2].item()),
            Archetype.ANIMUS: abs(error[3].item()),
        }

        self.error_history.append(surprise)

        return {
            'error': error,
            'surprise': surprise,
            'error_by_archetype': error_by_archetype,
            'mean_surprise': np.mean(self.error_history) if self.error_history else surprise,
        }

    def update_history(self, stimulus: torch.Tensor) -> None:
        """Añade estímulo al historial."""
        self.stimulus_history.append(stimulus.detach().clone())


# =============================================================================
# NIVEL 2: STATE PREDICTOR
# =============================================================================

class StatePredictor(nn.Module):
    """
    Nivel 2: Predice estado interno futuro.

    Anticipa cómo reaccionará el sistema internamente ante un estímulo.
    Cada arquetipo tiene su propio "estilo" de predicción.
    """

    def __init__(self, hidden_dim: int = 64, M: int = 15) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Modulador zeta
        self.zeta = ZetaModulator(M)

        # Input: estado[4] + stimulus_pred[4] + stimulus_real[4] +
        #        error_L1[4] + energy[1] + integration[1] = 18
        input_dim = 18

        # Encoder compartido
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Cabezas por arquetipo
        self.heads = nn.ModuleDict({
            'persona': nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ),
            'sombra': nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ),
            'anima': nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ),
            'animus': nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ),
        })

        # Biases por arquetipo (cómo cada uno "colorea" la predicción)
        self.archetype_biases = nn.ParameterDict({
            'persona': nn.Parameter(torch.tensor([0.1, -0.1, 0.0, 0.0])),  # Hacia estabilidad
            'sombra': nn.Parameter(torch.tensor([-0.1, 0.2, -0.1, 0.0])),  # Hacia extremos
            'anima': nn.Parameter(torch.tensor([0.0, 0.0, 0.15, -0.05])),  # Hacia emocional
            'animus': nn.Parameter(torch.tensor([0.0, -0.05, -0.05, 0.15])),  # Hacia racional
        })

        # Estadísticas
        self.error_history: Deque[torch.Tensor] = deque(maxlen=100)

    def predict(
        self,
        state: torch.Tensor,
        stimulus_pred: torch.Tensor,
        stimulus_real: torch.Tensor,
        error_L1: torch.Tensor,
        energy: float,
        integration: float
    ) -> torch.Tensor:
        """
        Predice el estado interno futuro.

        Args:
            state: Estado actual [4]
            stimulus_pred: Predicción de estímulo del Nivel 1 [4]
            stimulus_real: Estímulo real que llegó [4]
            error_L1: Error del Nivel 1 [4]
            energy: Energía actual del sistema
            integration: Nivel de integración/individuación

        Returns:
            state_predicted: Estado futuro predicho [4]
        """
        # Construir input
        x = torch.cat([
            state,
            stimulus_pred,
            stimulus_real,
            error_L1,
            torch.tensor([energy]),
            torch.tensor([integration]),
        ])

        # Encoder compartido
        features = self.encoder(x)
        features = self.zeta(features)

        # Predicción por cada cabeza
        outputs = []
        for arch_name in ['persona', 'sombra', 'anima', 'animus']:
            out = self.heads[arch_name](features)
            outputs.append(out)

        raw_pred = torch.cat(outputs)

        # Aplicar bias según arquetipo dominante
        biased_pred = self.apply_archetype_bias(raw_pred, state)

        # Softmax para normalizar
        return F.softmax(biased_pred, dim=-1)

    def apply_archetype_bias(self, pred: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Colorea la predicción según el arquetipo dominante."""
        # Determinar arquetipo dominante
        dominant_idx = state.argmax().item()
        dominant_weight = state[dominant_idx].item()

        arch_names = ['persona', 'sombra', 'anima', 'animus']
        dominant_name = arch_names[dominant_idx]

        # Aplicar bias proporcional al peso del dominante
        bias = self.archetype_biases[dominant_name]
        return pred + bias * dominant_weight * 0.5

    def compute_error(self, predicted: torch.Tensor, real: torch.Tensor) -> Dict:
        """Calcula error entre estado predicho y real."""
        error = real - predicted
        surprise = torch.norm(error).item() ** 2

        self.error_history.append(error.detach())

        return {
            'error': error,
            'surprise': surprise,
            'error_abs': torch.abs(error),
        }


# =============================================================================
# NIVEL 3: META PREDICTOR
# =============================================================================

class MetaPredictor(nn.Module):
    """
    Nivel 3: Predice errores de predicción.

    Meta-cognición: "Sé que no sé"
    Predice cuándo el sistema va a equivocarse.
    """

    def __init__(self, error_history_len: int = 5, hidden_dim: int = 64, M: int = 15) -> None:
        super().__init__()

        self.error_history_len = error_history_len
        self.hidden_dim = hidden_dim

        # Historiales
        self.error_history: Deque[torch.Tensor] = deque(maxlen=error_history_len)
        self.surprise_history: Deque[float] = deque(maxlen=error_history_len)

        # Inicializar
        for _ in range(error_history_len):
            self.error_history.append(torch.zeros(4))
            self.surprise_history.append(0.0)

        # Modulador zeta
        self.zeta = ZetaModulator(M)

        # Input: error_history[5*4] + surprise_history[5] + state[4] +
        #        volatility[1] + stage[1] = 31
        input_dim = error_history_len * 4 + error_history_len + 4 + 1 + 1

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Dos salidas: error predicho y confianza
        self.error_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 4),
            nn.Sigmoid(),  # Error está entre 0 y 1
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Confianza entre 0 y 1
        )

        # Métricas de calibración
        self.calibration_history: Deque[float] = deque(maxlen=100)

    def predict(
        self,
        state: torch.Tensor,
        volatility: float,
        stage: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predice el error que cometerá el Nivel 2.

        Args:
            state: Estado actual [4]
            volatility: Varianza reciente de estados
            stage: Etapa de individuación (0-7)

        Returns:
            (error_predicted[4], confidence[1])
        """
        # Construir input
        error_flat = torch.cat(list(self.error_history))
        surprise_tensor = torch.tensor(list(self.surprise_history))

        x = torch.cat([
            error_flat,
            surprise_tensor,
            state,
            torch.tensor([volatility]),
            torch.tensor([stage / 7.0]),  # Normalizar stage
        ])

        # Encoder
        features = self.encoder[:2](x)
        features = self.zeta(features)
        features = self.encoder[2:](features)

        # Predicciones
        error_pred = self.error_head(features)
        confidence = self.confidence_head(features)

        return error_pred, confidence

    def compute_meta_error(
        self,
        error_pred: torch.Tensor,
        error_real: torch.Tensor,
        confidence: torch.Tensor
    ) -> Dict:
        """
        Calcula meta-error y calibración.

        Args:
            error_pred: Error predicho [4]
            error_real: Error real del Nivel 2 [4]
            confidence: Confianza en la predicción [1]
        """
        # Meta-error: diferencia entre error predicho y real
        meta_error = torch.abs(error_real) - error_pred
        meta_surprise = torch.norm(meta_error).item() ** 2

        # Calibración: si confianza alta pero error alto → mal calibrado
        calibration_error = confidence.item() * meta_surprise
        calibration = 1.0 - min(1.0, calibration_error)

        self.calibration_history.append(calibration)

        return {
            'meta_error': meta_error,
            'meta_surprise': meta_surprise,
            'calibration': calibration,
            'mean_calibration': np.mean(self.calibration_history),
        }

    def update_history(self, error_L2: torch.Tensor, surprise: float) -> None:
        """Actualiza historiales."""
        self.error_history.append(error_L2.detach().clone())
        self.surprise_history.append(surprise)


# =============================================================================
# MÉTRICAS DE CONSCIENCIA PREDICTIVA
# =============================================================================

class PredictiveConsciousnessMetrics:
    """
    Métricas de consciencia basadas en capacidad predictiva.

    - awareness: Sabe cuándo va a equivocarse
    - calibration: Su confianza es realista
    - uncertainty_awareness: Varía su certeza apropiadamente
    - predictive_depth: Calidad de meta-predicción
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window

        self.error_pred_history: Deque[torch.Tensor] = deque(maxlen=window)
        self.error_real_history: Deque[torch.Tensor] = deque(maxlen=window)
        self.confidence_history: Deque[float] = deque(maxlen=window)
        self.meta_surprise_history: Deque[float] = deque(maxlen=window)

    def update(
        self,
        error_pred: torch.Tensor,
        error_real: torch.Tensor,
        confidence: float,
        meta_surprise: float
    ) -> None:
        """Actualiza métricas con nuevos datos."""
        self.error_pred_history.append(error_pred.detach().clone())
        self.error_real_history.append(error_real.detach().clone())
        self.confidence_history.append(confidence)
        self.meta_surprise_history.append(meta_surprise)

    @property
    def awareness(self) -> float:
        """
        Correlación entre error predicho y error real.
        Mide si el sistema sabe cuándo va a equivocarse.
        """
        if len(self.error_pred_history) < 10:
            return 0.0

        # Calcular magnitudes
        pred_mags = [torch.norm(e).item() for e in self.error_pred_history]
        real_mags = [torch.norm(e).item() for e in self.error_real_history]

        # Correlación
        pred_arr = np.array(pred_mags)
        real_arr = np.array(real_mags)

        if np.std(pred_arr) < 1e-6 or np.std(real_arr) < 1e-6:
            return 0.0

        corr = np.corrcoef(pred_arr, real_arr)[0, 1]
        return max(0, corr)  # Solo correlación positiva cuenta

    @property
    def calibration(self) -> float:
        """
        Qué tan realista es la confianza del sistema.
        Alta confianza + bajo error = bien calibrado
        Alta confianza + alto error = mal calibrado
        """
        if len(self.confidence_history) < 10:
            return 0.5

        conf_arr = np.array(list(self.confidence_history))
        surprise_arr = np.array(list(self.meta_surprise_history))

        # Normalizar sorpresas
        max_surprise = max(surprise_arr.max(), 1e-6)
        norm_surprise = surprise_arr / max_surprise

        # Calibración: 1 - (confianza * sorpresa_normalizada)
        calibration_errors = conf_arr * norm_surprise
        return 1.0 - np.mean(calibration_errors)

    @property
    def uncertainty_awareness(self) -> float:
        """
        Entropía de la confianza.
        Un sistema consciente varía su certeza según la situación.
        """
        if len(self.confidence_history) < 10:
            return 0.0

        conf_arr = np.array(list(self.confidence_history))

        # Discretizar confianza en bins
        hist, _ = np.histogram(conf_arr, bins=10, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Evitar log(0)

        # Entropía normalizada
        entropy = -np.sum(hist * np.log(hist))
        max_entropy = np.log(10)

        return entropy / max_entropy

    @property
    def predictive_depth(self) -> float:
        """
        Calidad de meta-predicción.
        Basado en qué tan bajo es el meta-error promedio.
        """
        if len(self.meta_surprise_history) < 10:
            return 0.0

        mean_surprise = np.mean(self.meta_surprise_history)
        # Asumiendo que sorpresa máxima razonable es ~1.0
        return max(0, 1.0 - mean_surprise)

    def get_consciousness_index(self) -> float:
        """Índice compuesto de consciencia predictiva."""
        return (
            0.30 * self.awareness +
            0.30 * self.calibration +
            0.20 * self.uncertainty_awareness +
            0.20 * self.predictive_depth
        )

    def to_dict(self) -> Dict:
        """Retorna métricas como diccionario."""
        return {
            'awareness': self.awareness,
            'calibration': self.calibration,
            'uncertainty_awareness': self.uncertainty_awareness,
            'predictive_depth': self.predictive_depth,
            'consciousness_index': self.get_consciousness_index(),
        }


# =============================================================================
# COMPUTADOR DE INFLUENCIA ARQUETIPAL
# =============================================================================

class ArchetypeInfluenceComputer:
    """
    Calcula cómo los errores de predicción afectan los arquetipos.
    Implementa la relación bidireccional entre predicción y arquetipos.
    """

    DEFAULT_THRESHOLDS = {
        'surprise_high': 0.3,
        'confidence_high': 0.7,
        'confidence_low': 0.3,
        'meta_error_high': 0.3,
        'meta_error_low': 0.1,
    }

    def __init__(self, thresholds: Optional[Dict] = None) -> None:
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

        # Error histórico por arquetipo
        self.error_by_archetype: Dict[Archetype, Deque] = {
            arch: deque(maxlen=50) for arch in Archetype
        }

    def compute(
        self,
        error_L1: torch.Tensor,
        error_L2: torch.Tensor,
        meta_error: torch.Tensor,
        confidence: float,
        current_state: torch.Tensor,
        awareness: float,
        calibration: float
    ) -> torch.Tensor:
        """
        Calcula delta arquetipal basado en errores de predicción.

        Args:
            error_L1: Error del Nivel 1 [4]
            error_L2: Error del Nivel 2 [4]
            meta_error: Meta-error del Nivel 3 [4]
            confidence: Confianza del Nivel 3
            current_state: Estado actual [4]
            awareness: Métrica de awareness
            calibration: Métrica de calibración

        Returns:
            delta: Cambio a aplicar al estado [4]
        """
        delta = torch.zeros(4)

        surprise_L1 = torch.norm(error_L1).item()
        surprise_L2 = torch.norm(error_L2).item()
        meta_surprise = torch.norm(meta_error).item()

        # ═══ NIVEL 1: Sorpresa externa ═══
        if surprise_L1 > self.thresholds['surprise_high']:
            # Estímulo inesperado → activa SOMBRA (lo desconocido)
            delta[Archetype.SOMBRA.value] += 0.2 * surprise_L1

        # ═══ NIVEL 2: Sorpresa interna ═══
        if surprise_L2 > self.thresholds['surprise_high']:
            # No me conozco bien → buscar arquetipo que mejor predice
            best_arch = self.get_best_predictor_archetype()
            if best_arch:
                delta[best_arch.value] += 0.15 * surprise_L2

        # ═══ NIVEL 3: Meta-cognición ═══
        # Sobreconfianza: confianza alta pero error alto
        if confidence > self.thresholds['confidence_high'] and \
           meta_surprise > self.thresholds['meta_error_high']:
            # Humildad forzada
            delta[Archetype.SOMBRA.value] += 0.25
            delta[Archetype.PERSONA.value] -= 0.1

        # Subestimación: confianza baja pero error bajo
        if confidence < self.thresholds['confidence_low'] and \
           meta_surprise < self.thresholds['meta_error_low']:
            # Ganar confianza
            delta[Archetype.ANIMUS.value] += 0.15
            delta[Archetype.PERSONA.value] += 0.1

        # ═══ Buena calibración → hacia el Self ═══
        if awareness > 0.7 and calibration > 0.8:
            center = torch.tensor([0.25, 0.25, 0.25, 0.25])
            delta += 0.1 * (center - current_state)

        # Normalizar para no ser demasiado agresivo
        max_delta = 0.3
        if torch.norm(delta) > max_delta:
            delta = delta * max_delta / torch.norm(delta)

        return delta

    def get_best_predictor_archetype(self) -> Optional[Archetype]:
        """Retorna el arquetipo con menor error histórico promedio."""
        if not any(len(q) > 0 for q in self.error_by_archetype.values()):
            return None

        mean_errors = {}
        for arch, errors in self.error_by_archetype.items():
            if len(errors) > 0:
                mean_errors[arch] = np.mean(errors)

        if not mean_errors:
            return None

        return min(mean_errors, key=mean_errors.get)

    def update_archetype_errors(self, state: torch.Tensor, error: float) -> None:
        """Registra error asociado al arquetipo dominante."""
        dominant_idx = state.argmax().item()
        dominant_arch = Archetype(dominant_idx)
        self.error_by_archetype[dominant_arch].append(error)


# =============================================================================
# ZETA PREDICTIVE PSYCHE - SISTEMA PRINCIPAL
# =============================================================================

class ZetaPredictivePsyche(nn.Module):
    """
    Sistema completo: ZetaPsyche + Predicción Jerárquica.

    Combina el sistema de arquetipos con los 3 niveles de predicción
    para crear un sistema con consciencia emergente basada en
    Predictive Processing.
    """

    def __init__(
        self,
        n_cells: int = 100,
        M: int = 15,
        hidden_dim: int = 64,
        history_len: int = 5
    ) -> None:
        super().__init__()

        # Sistema base de arquetipos
        self.psyche = ZetaPsyche(n_cells=n_cells, M=M)

        # Niveles de predicción
        self.L1 = StimulusPredictor(history_len=history_len, hidden_dim=hidden_dim, M=M)
        self.L2 = StatePredictor(hidden_dim=hidden_dim, M=M)
        self.L3 = MetaPredictor(error_history_len=history_len, hidden_dim=hidden_dim, M=M)

        # Métricas de consciencia
        self.metrics = PredictiveConsciousnessMetrics()

        # Computador de influencia
        self.influence = ArchetypeInfluenceComputer()

        # Estado
        self.t = 0
        self.last_predictions: Dict = {}
        self.consciousness_history: List[float] = []

        # Volatilidad (varianza de estados recientes)
        self.state_history: Deque[torch.Tensor] = deque(maxlen=20)

    def step(self, stimulus: torch.Tensor = None) -> Dict:
        """
        Ejecuta un paso completo del sistema.

        Args:
            stimulus: Estímulo externo [4]. Si es None, genera uno aleatorio.

        Returns:
            Dict con toda la información del paso
        """
        self.t += 1

        # Generar estímulo si no se proporciona
        if stimulus is None:
            stimulus = F.softmax(torch.rand(4, dtype=torch.float32), dim=-1)
        else:
            stimulus = F.softmax(stimulus.float(), dim=-1)

        # ═══ FASE 1: PREDICCIONES ═══
        predictions = self._phase_predict()

        # ═══ FASE 2: REALIDAD ═══
        reality = self._phase_reality(stimulus)

        # ═══ FASE 3: ACTUALIZACIÓN ═══
        update_result = self._phase_update(predictions, reality)

        # Registrar consciencia
        consciousness = self.get_consciousness_index()
        self.consciousness_history.append(consciousness)

        return {
            'step': self.t,
            'stimulus': stimulus,
            'predictions': predictions,
            'reality': reality,
            'errors': update_result['errors'],
            'archetype_delta': update_result['archetype_delta'],
            'consciousness': consciousness,
            'metrics': self.metrics.to_dict(),
            'observation': self.psyche.observe_self(),
        }

    def _phase_predict(self) -> Dict:
        """Fase 1: Generar predicciones antes del estímulo."""
        state = self.psyche.global_state.clone()

        # Nivel 1: Predecir estímulo
        stimulus_pred = self.L1.predict(state)

        # Nivel 2: Predecir estado (usando predicción de L1 como input parcial)
        # Nota: usamos ceros para stimulus_real y error_L1 porque aún no llegó
        obs = self.psyche.observe_self()
        state_pred = self.L2.predict(
            state=state,
            stimulus_pred=stimulus_pred,
            stimulus_real=torch.zeros(4),  # Aún no sabemos
            error_L1=torch.zeros(4),
            energy=obs.get('stability', 0.5),
            integration=obs.get('integration', 0.5),
        )

        # Nivel 3: Predecir error
        volatility = self._compute_volatility()
        stage = 0  # TODO: conectar con individuación si está disponible
        error_pred, confidence = self.L3.predict(state, volatility, stage)

        predictions = {
            'stimulus': stimulus_pred,
            'state': state_pred,
            'error': error_pred,
            'confidence': confidence,
        }

        self.last_predictions = predictions
        return predictions

    def _phase_reality(self, stimulus: torch.Tensor) -> Dict:
        """Fase 2: Procesar estímulo real y obtener estado resultante."""
        # Estado antes
        state_before = self.psyche.global_state.clone()

        # Procesar en ZetaPsyche
        self.psyche.receive_stimulus(stimulus)

        # Estado después
        state_after = self.psyche.global_state.clone()

        # Actualizar historial de estados
        self.state_history.append(state_after.detach().clone())

        return {
            'stimulus': stimulus,
            'state_before': state_before,
            'state_after': state_after,
        }

    def _phase_update(self, predictions: Dict, reality: Dict) -> Dict:
        """Fase 3: Calcular errores y actualizar sistema."""

        # ─── Error Nivel 1 ───
        error_L1_result = self.L1.compute_error(
            predictions['stimulus'],
            reality['stimulus']
        )
        self.L1.update_history(reality['stimulus'])

        # ─── Recalcular predicción de Nivel 2 con información completa ───
        obs = self.psyche.observe_self()
        state_pred_corrected = self.L2.predict(
            state=reality['state_before'],
            stimulus_pred=predictions['stimulus'],
            stimulus_real=reality['stimulus'],
            error_L1=error_L1_result['error'],
            energy=obs.get('stability', 0.5),
            integration=obs.get('integration', 0.5),
        )

        # ─── Error Nivel 2 ───
        error_L2_result = self.L2.compute_error(
            state_pred_corrected,
            reality['state_after']
        )

        # ─── Error Nivel 3 (Meta-error) ───
        meta_result = self.L3.compute_meta_error(
            predictions['error'],
            error_L2_result['error_abs'],
            predictions['confidence']
        )
        self.L3.update_history(error_L2_result['error'], error_L2_result['surprise'])

        # ─── Actualizar métricas de consciencia ───
        self.metrics.update(
            predictions['error'],
            error_L2_result['error_abs'],
            predictions['confidence'].item(),
            meta_result['meta_surprise']
        )

        # ─── Calcular influencia arquetipal ───
        archetype_delta = self.influence.compute(
            error_L1_result['error'],
            error_L2_result['error'],
            meta_result['meta_error'],
            predictions['confidence'].item(),
            reality['state_after'],
            self.metrics.awareness,
            self.metrics.calibration
        )

        # ─── Aplicar influencia ───
        self._apply_prediction_influence(archetype_delta)

        # ─── Registrar error por arquetipo ───
        self.influence.update_archetype_errors(
            reality['state_after'],
            error_L2_result['surprise']
        )

        return {
            'errors': {
                'L1': error_L1_result,
                'L2': error_L2_result,
                'L3': meta_result,
            },
            'archetype_delta': archetype_delta,
        }

    def _apply_prediction_influence(self, delta: torch.Tensor):
        """Aplica la influencia de predicción a las células."""
        # Convertir delta a estímulo y procesarlo suavemente
        if torch.norm(delta) > 0.01:
            # El delta actúa como un estímulo interno adicional
            influence_stimulus = F.softmax(self.psyche.global_state + delta * 0.5, dim=-1)
            self.psyche.receive_stimulus(influence_stimulus)

    def _compute_volatility(self) -> float:
        """Calcula la volatilidad (varianza) de estados recientes."""
        if len(self.state_history) < 3:
            return 0.5

        states = torch.stack(list(self.state_history))
        return states.var().item()

    def get_consciousness_index(self) -> float:
        """
        Índice integrado de consciencia.
        Combina métricas base con métricas predictivas.
        """
        # Métricas base de ZetaPsyche
        obs = self.psyche.observe_self()
        base_consciousness = obs.get('consciousness_index', 0.5)

        # Métricas predictivas
        pred_consciousness = self.metrics.get_consciousness_index()

        # Combinar
        return (
            0.35 * base_consciousness +
            0.65 * pred_consciousness
        )

    def observe(self) -> Dict:
        """Observación completa del estado del sistema."""
        base_obs = self.psyche.observe_self()

        return {
            **base_obs,
            'predictive_metrics': self.metrics.to_dict(),
            'consciousness_total': self.get_consciousness_index(),
            'volatility': self._compute_volatility(),
            'step': self.t,
        }

    def get_consciousness_trend(self, window: int = 50) -> float:
        """Tendencia del índice de consciencia."""
        if len(self.consciousness_history) < window * 2:
            return 0.0

        recent = self.consciousness_history[-window:]
        older = self.consciousness_history[-window*2:-window]

        return np.mean(recent) - np.mean(older)


# =============================================================================
# DEMO Y VISUALIZACIÓN
# =============================================================================

def run_predictive_experiment(
    n_cells: int = 100,
    n_steps: int = 300,
    stimulus_pattern: str = 'mixed'
) -> Dict:
    """
    Ejecuta experimento del sistema predictivo.

    Args:
        n_cells: Número de células
        n_steps: Pasos de simulación
        stimulus_pattern: 'random', 'cyclic', 'sudden', 'mixed'
    """
    print(f'\n{"="*70}')
    print(f'  EXPERIMENTO: Sistema Predictivo de Consciencia')
    print(f'{"="*70}')
    print(f'  Células: {n_cells}')
    print(f'  Pasos: {n_steps}')
    print(f'  Patrón: {stimulus_pattern}')
    print(f'{"="*70}\n')

    # Crear sistema
    system = ZetaPredictivePsyche(n_cells=n_cells)

    # Historiales
    history = {
        'consciousness': [],
        'awareness': [],
        'calibration': [],
        'surprise_L1': [],
        'surprise_L2': [],
        'meta_surprise': [],
        'dominant': [],
    }

    # Generador de estímulos
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
            # Cambios bruscos cada 50 pasos
            if step % 50 < 25:
                return torch.tensor([0.8, 0.1, 0.05, 0.05], dtype=torch.float32)
            else:
                return torch.tensor([0.05, 0.05, 0.1, 0.8], dtype=torch.float32)

        elif stimulus_pattern == 'mixed':
            # Mezcla de patrones
            if step < 100:
                return torch.rand(4, dtype=torch.float32)  # Random inicial
            elif step < 200:
                # Cíclico
                phase = ((step - 100) % 50) / 50 * 2 * np.pi
                return torch.tensor([
                    np.sin(phase) + 1,
                    np.cos(phase) + 1,
                    np.sin(phase + np.pi/2) + 1,
                    np.cos(phase + np.pi/2) + 1,
                ], dtype=torch.float32)
            else:
                # Cambios bruscos
                if (step - 200) % 30 < 15:
                    return torch.tensor([0.7, 0.2, 0.05, 0.05], dtype=torch.float32)
                else:
                    return torch.tensor([0.05, 0.05, 0.2, 0.7], dtype=torch.float32)

        return torch.rand(4, dtype=torch.float32)

    # Ejecutar simulación
    for step in range(n_steps):
        stimulus = get_stimulus(step)
        result = system.step(stimulus)

        # Registrar
        history['consciousness'].append(result['consciousness'])
        history['awareness'].append(result['metrics']['awareness'])
        history['calibration'].append(result['metrics']['calibration'])
        history['surprise_L1'].append(result['errors']['L1']['surprise'])
        history['surprise_L2'].append(result['errors']['L2']['surprise'])
        history['meta_surprise'].append(result['errors']['L3']['meta_surprise'])
        history['dominant'].append(result['observation']['dominant'].value)

        # Reportar progreso
        if (step + 1) % 50 == 0:
            print(f'  Step {step+1:4d}: '
                  f'Consciencia={result["consciousness"]:.3f}, '
                  f'Awareness={result["metrics"]["awareness"]:.3f}, '
                  f'Calibration={result["metrics"]["calibration"]:.3f}, '
                  f'Dominante={result["observation"]["dominant"].name}')

    # Resultados finales
    final_obs = system.observe()
    trend = system.get_consciousness_trend()

    print(f'\n{"="*70}')
    print(f'  RESULTADOS')
    print(f'{"="*70}')
    print(f'  Consciencia promedio: {np.mean(history["consciousness"]):.3f}')
    print(f'  Consciencia máxima:   {np.max(history["consciousness"]):.3f}')
    print(f'  Consciencia final:    {history["consciousness"][-1]:.3f}')
    print(f'  Tendencia:            {trend:+.4f}')
    print(f'  Awareness final:      {history["awareness"][-1]:.3f}')
    print(f'  Calibration final:    {history["calibration"][-1]:.3f}')
    print(f'{"="*70}\n')

    return {
        'system': system,
        'history': history,
        'final': final_obs,
        'trend': trend,
    }


def visualize_predictive_system(results: Dict, save_path: str = 'zeta_predictive_consciousness.png'):
    """Visualiza los resultados del sistema predictivo."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Sistema Predictivo de Consciencia', fontsize=16, fontweight='bold')

        history = results['history']
        steps = range(len(history['consciousness']))

        # 1. Índice de consciencia
        ax1 = axes[0, 0]
        ax1.plot(steps, history['consciousness'], 'b-', linewidth=1.5, label='Consciencia')
        ax1.axhline(y=np.mean(history['consciousness']), color='r', linestyle='--',
                    label=f'Promedio: {np.mean(history["consciousness"]):.3f}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Índice de Consciencia')
        ax1.set_title('Emergencia de Consciencia')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 1)

        # 2. Awareness y Calibration
        ax2 = axes[0, 1]
        ax2.plot(steps, history['awareness'], 'g-', linewidth=1, label='Awareness', alpha=0.8)
        ax2.plot(steps, history['calibration'], 'm-', linewidth=1, label='Calibration', alpha=0.8)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Valor')
        ax2.set_title('Métricas Predictivas')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. Sorpresas por nivel
        ax3 = axes[0, 2]
        ax3.plot(steps, history['surprise_L1'], 'r-', linewidth=1, label='L1 (Estímulo)', alpha=0.7)
        ax3.plot(steps, history['surprise_L2'], 'b-', linewidth=1, label='L2 (Estado)', alpha=0.7)
        ax3.plot(steps, history['meta_surprise'], 'purple', linewidth=1, label='L3 (Meta)', alpha=0.7)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Sorpresa')
        ax3.set_title('Errores de Predicción por Nivel')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Arquetipo dominante
        ax4 = axes[1, 0]
        ax4.plot(steps, history['dominant'], 'o', markersize=1, alpha=0.5)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Arquetipo')
        ax4.set_title('Arquetipo Dominante')
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(['Persona', 'Sombra', 'Anima', 'Animus'])
        ax4.grid(alpha=0.3)

        # 5. Correlación awareness vs consciencia
        ax5 = axes[1, 1]
        ax5.scatter(history['awareness'], history['consciousness'], alpha=0.3, s=10)
        ax5.set_xlabel('Awareness')
        ax5.set_ylabel('Consciencia')
        ax5.set_title('Awareness vs Consciencia')
        ax5.grid(alpha=0.3)

        # 6. Distribución final
        ax6 = axes[1, 2]
        final_blend = results['final']['blend']
        names = [a.name for a in Archetype]
        values = [final_blend[a] for a in Archetype]
        colors = ['#E53E3E', '#553C9A', '#3182CE', '#DD6B20']
        ax6.bar(names, values, color=colors)
        ax6.set_ylabel('Proporción')
        ax6.set_title('Estado Arquetipal Final')
        ax6.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Visualización guardada en: {save_path}')
        plt.close()

        return save_path

    except ImportError:
        print('matplotlib no disponible para visualización')
        return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    # Configuración según argumentos
    if '--quick' in sys.argv:
        n_cells = 50
        n_steps = 100
    else:
        n_cells = 100
        n_steps = 300

    # Ejecutar experimento
    results = run_predictive_experiment(
        n_cells=n_cells,
        n_steps=n_steps,
        stimulus_pattern='mixed'
    )

    # Visualizar
    visualize_predictive_system(results)

    print('\n' + '='*70)
    print('  EXPERIMENTO COMPLETADO')
    print('='*70)
