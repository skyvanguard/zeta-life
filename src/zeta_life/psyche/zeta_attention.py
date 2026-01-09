# -*- coding: utf-8 -*-
"""
ZetaAttention: Sistema de Atención Selectiva Jerárquica
========================================================

Implementa atención selectiva en 3 niveles basado en Predictive Processing:
- Nivel 3: GlobalArchetypalAttention - ¿Qué arquetipo necesita atención?
- Nivel 2: TemporalAttention - ¿Qué momentos pasados son relevantes?
- Nivel 1: ErrorAttention - ¿Qué nivel predictivo necesita recursos?

Basado en la teoría de Friston (precision-weighted prediction errors).

Fecha de implementación: 3 Enero 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class MemoryItem:
    """Un elemento en el buffer de memoria"""
    state: torch.Tensor           # Estado arquetipal [4]
    stimulus: torch.Tensor        # Estímulo recibido [4]
    errors: torch.Tensor          # Errores de predicción [3] (L1, L2, L3)
    surprise: float               # Sorpresa total
    timestamp: float              # Momento temporal
    archetype_dominant: int       # Índice del arquetipo dominante

    def to_tensor(self) -> torch.Tensor:
        """Convierte a tensor concatenado para atención"""
        return torch.cat([
            self.state,
            self.stimulus,
            self.errors,
            torch.tensor([self.surprise, self.timestamp], dtype=torch.float32)
        ])


@dataclass
class AttentionOutput:
    """Salida del sistema de atención"""
    # Distribuciones de atención
    archetypal_attention: torch.Tensor    # [4] atención por arquetipo
    temporal_attention: torch.Tensor      # [buffer_size] atención temporal
    error_attention: torch.Tensor         # [3] atención por nivel predictivo

    # Atención integrada
    global_attention: torch.Tensor        # [4] atención final combinada
    attention_intensity: float            # Intensidad global [0,1]
    attention_coherence: float            # Coherencia entre niveles [0,1]

    # Contexto detectado
    context: Dict[str, float]             # threat, opportunity, emotional, cognitive

    # Memoria atendida
    attended_memory: Optional[torch.Tensor] = None  # Representación ponderada


# =============================================================================
# NIVEL 3: DETECCIÓN DE CONTEXTO Y ATENCIÓN ARQUETIPAL
# =============================================================================

class ContextDetector(nn.Module):
    """
    Detecta el tipo de contexto actual para guiar la atención arquetipal.

    Señales detectadas:
    - Amenaza → activa SOMBRA
    - Oportunidad → activa PERSONA
    - Necesidad emocional → activa ANIMA
    - Demanda cognitiva → activa ANIMUS
    """

    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()

        # Red para detectar contexto desde estímulo + estado
        self.context_net = nn.Sequential(
            nn.Linear(input_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # [threat, opportunity, emotional, cognitive]
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialización con bias hacia detección balanceada"""
        for layer in self.context_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, stimulus: torch.Tensor, state: torch.Tensor) -> Dict[str, float]:
        """
        Detecta el contexto actual.

        Args:
            stimulus: Estímulo actual [4]
            state: Estado arquetipal actual [4]

        Returns:
            Dict con probabilidades de cada tipo de contexto
        """
        combined = torch.cat([stimulus, state])
        context_probs = self.context_net(combined)

        return {
            'threat': context_probs[0].item(),
            'opportunity': context_probs[1].item(),
            'emotional': context_probs[2].item(),
            'cognitive': context_probs[3].item()
        }


class GlobalArchetypalAttention(nn.Module):
    """
    NIVEL 3: Atención a nivel de arquetipos.

    Decide qué arquetipo debe recibir más recursos basándose en:
    - Contexto detectado
    - Estado actual del sistema
    - Historia reciente de errores

    Arquetipos: [PERSONA, SOMBRA, ANIMA, ANIMUS]
    """

    # Mapeo contexto → arquetipo preferido
    CONTEXT_ARCHETYPE_MAP = {
        'threat': 1,      # SOMBRA
        'opportunity': 0,  # PERSONA
        'emotional': 2,    # ANIMA
        'cognitive': 3     # ANIMUS
    }

    def __init__(self, state_dim: int = 4, temperature: float = 1.0) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.temperature = temperature

        # Red que combina contexto + estado para atención
        self.attention_net = nn.Sequential(
            nn.Linear(state_dim + 4, 16),  # state + context
            nn.ReLU(),
            nn.Linear(16, state_dim)
        )

        # Pesos aprendibles para cada contexto → arquetipo
        self.context_weights = nn.Parameter(torch.eye(4))  # 4 contextos → 4 arquetipos

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.attention_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        state: torch.Tensor,
        context: Dict[str, float],
        uncertainty: float = 0.5
    ) -> torch.Tensor:
        """
        Calcula atención arquetipal.

        Args:
            state: Estado arquetipal actual [4]
            context: Dict con probabilidades de contexto
            uncertainty: Incertidumbre del sistema [0,1]

        Returns:
            attention: Distribución de atención [4]
        """
        # Convertir contexto a tensor
        context_tensor = torch.tensor([
            context['threat'],
            context['opportunity'],
            context['emotional'],
            context['cognitive']
        ], dtype=torch.float32)

        # Contribución directa del contexto
        context_attention = torch.matmul(context_tensor, self.context_weights)

        # Contribución de la red (combina estado + contexto)
        combined = torch.cat([state, context_tensor])
        net_attention = self.attention_net(combined)

        # Combinar ambas contribuciones
        raw_attention = context_attention + net_attention

        # Ajustar temperatura según incertidumbre
        # Alta incertidumbre → atención difusa (temperatura alta)
        # Baja incertidumbre → atención enfocada (temperatura baja)
        effective_temp = self.temperature * (0.5 + uncertainty)

        # Softmax con temperatura
        attention = F.softmax(raw_attention / effective_temp, dim=0)

        return attention


# =============================================================================
# NIVEL 2: MEMORIA Y ATENCIÓN TEMPORAL
# =============================================================================

class MemoryBuffer:
    """
    Buffer de memoria con consolidación y decaimiento.

    Almacena eventos pasados y permite atención selectiva sobre ellos.
    Los eventos con alta sorpresa se consolidan (decaen más lento).
    """

    def __init__(
        self,
        max_size: int = 100,
        consolidation_threshold: float = 0.7,
        base_decay: float = 0.99
    ):
        self.max_size = max_size
        self.consolidation_threshold = consolidation_threshold
        self.base_decay = base_decay

        self.buffer: deque = deque(maxlen=max_size)
        self.importance: List[float] = []  # Importancia de cada item

    def add(self, item: MemoryItem) -> None:
        """Añade un nuevo elemento al buffer"""
        self.buffer.append(item)

        # Importancia inicial basada en sorpresa
        importance = min(1.0, item.surprise)
        self.importance.append(importance)

        # Mantener sincronizado con el buffer
        while len(self.importance) > len(self.buffer):
            self.importance.pop(0)

    def decay(self) -> None:
        """Aplica decaimiento a la importancia de los items"""
        for i in range(len(self.importance)):
            item = self.buffer[i]

            # Items consolidados (alta sorpresa) decaen más lento
            if item.surprise > self.consolidation_threshold:
                decay_rate = self.base_decay ** 0.5  # Raíz cuadrada = más lento
            else:
                decay_rate = self.base_decay

            self.importance[i] *= decay_rate

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna tensores para atención.

        Returns:
            memories: [buffer_size, feature_dim] tensor de memorias
            importance: [buffer_size] tensor de importancia
        """
        if len(self.buffer) == 0:
            return torch.zeros(1, 13), torch.ones(1)  # 13 = 4+4+3+2

        memories = torch.stack([item.to_tensor() for item in self.buffer])
        importance = torch.tensor(self.importance, dtype=torch.float32)

        return memories, importance

    def __len__(self) -> int:
        return len(self.buffer)


class TemporalAttention(nn.Module):
    """
    NIVEL 2: Atención sobre el buffer de memoria.

    Usa scaled dot-product attention con:
    - Bonus por sorpresa (eventos sorprendentes son más relevantes)
    - Decaimiento por recencia (eventos recientes ponderan más)
    - Relevancia arquetipal (atender a memorias del arquetipo activo)
    """

    def __init__(
        self,
        memory_dim: int = 13,  # 4+4+3+2 (state+stim+errors+surprise+time)
        hidden_dim: int = 16,
        n_heads: int = 2
    ) -> None:
        super().__init__()

        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim

        # Proyecciones Query, Key, Value
        self.query_proj = nn.Linear(memory_dim, hidden_dim)
        self.key_proj = nn.Linear(memory_dim, hidden_dim)
        self.value_proj = nn.Linear(memory_dim, hidden_dim)

        # Proyección de salida
        self.output_proj = nn.Linear(hidden_dim, memory_dim)

        # Parámetros de atención temporal
        self.recency_weight = nn.Parameter(torch.tensor(0.1))
        self.surprise_weight = nn.Parameter(torch.tensor(0.3))

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        memories: torch.Tensor,
        importance: torch.Tensor,
        archetypal_attention: torch.Tensor,
        current_time: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula atención temporal sobre memorias.

        Args:
            query: Query actual [memory_dim] (estado actual concatenado)
            memories: Buffer de memorias [N, memory_dim]
            importance: Importancia de cada memoria [N]
            archetypal_attention: Atención arquetipal actual [4]
            current_time: Timestamp actual

        Returns:
            attended: Representación atendida [memory_dim]
            attention_weights: Pesos de atención [N]
        """
        N = memories.shape[0]

        # Proyectar Query, Keys, Values
        Q = self.query_proj(query.unsqueeze(0))  # [1, hidden]
        K = self.key_proj(memories)              # [N, hidden]
        V = self.value_proj(memories)            # [N, hidden]

        # Scaled dot-product attention
        scale = self.hidden_dim ** 0.5
        scores = torch.matmul(Q, K.T) / scale  # [1, N]
        scores = scores.squeeze(0)              # [N]

        # Bonus por sorpresa (columna 11 del tensor de memoria)
        surprise_bonus = memories[:, 11] * self.surprise_weight
        scores = scores + surprise_bonus

        # Decaimiento por recencia (columna 12 es timestamp)
        timestamps = memories[:, 12]
        recency = torch.exp(-self.recency_weight * (current_time - timestamps))
        scores = scores + torch.log(recency + 1e-8)

        # Relevancia arquetipal (qué arquetipo dominaba en cada memoria)
        # Las memorias cuyo arquetipo dominante coincide con el atendido tienen bonus
        archetype_indices = memories[:, :4].argmax(dim=1)  # [N]
        current_archetype = archetypal_attention.argmax()
        archetype_match = (archetype_indices == current_archetype).float() * 0.5
        scores = scores + archetype_match

        # Ponderar por importancia almacenada
        scores = scores + torch.log(importance + 1e-8)

        # Softmax para obtener pesos de atención
        attention_weights = F.softmax(scores, dim=0)  # [N]

        # Calcular valor atendido
        attended = torch.matmul(attention_weights.unsqueeze(0), V)  # [1, hidden]
        attended = self.output_proj(attended.squeeze(0))           # [memory_dim]

        return attended, attention_weights


# =============================================================================
# NIVEL 1: ATENCIÓN SOBRE ERRORES PREDICTIVOS
# =============================================================================

class ErrorAttention(nn.Module):
    """
    NIVEL 1: Atención sobre los niveles de predicción.

    Decide qué nivel del sistema predictivo necesita más recursos:
    - L1 (StimulusPredictor): Predicción de estímulos
    - L2 (StatePredictor): Predicción de estados
    - L3 (MetaPredictor): Meta-predicción de errores

    Usa precision-weighting de Friston:
    precision = 1 / variance
    Los errores con alta precisión (baja varianza) reciben más atención.
    """

    def __init__(self, n_levels: int = 3, history_size: int = 20) -> None:
        super().__init__()

        self.n_levels = n_levels
        self.history_size = history_size

        # Historial de errores para calcular varianza
        self.error_history: List[torch.Tensor] = []

        # Red para aprender pesos de atención
        self.attention_net = nn.Sequential(
            nn.Linear(n_levels * 2, 8),  # errors + precisions
            nn.ReLU(),
            nn.Linear(8, n_levels)
        )

        # Pesos base por nivel (más atención a niveles superiores por defecto)
        self.level_bias = nn.Parameter(torch.tensor([0.0, 0.2, 0.4]))

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.attention_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def update_history(self, errors: torch.Tensor) -> None:
        """Actualiza el historial de errores"""
        self.error_history.append(errors.detach())
        if len(self.error_history) > self.history_size:
            self.error_history.pop(0)

    def compute_precisions(self) -> torch.Tensor:
        """
        Calcula precisiones basadas en varianza histórica.

        precision = 1 / (variance + epsilon)
        """
        if len(self.error_history) < 2:
            return torch.ones(self.n_levels)

        # Stack history y calcular varianza por nivel
        history = torch.stack(self.error_history)  # [history_size, n_levels]
        variances = history.var(dim=0)             # [n_levels]

        # Precisión = 1 / varianza (con epsilon para estabilidad)
        precisions: torch.Tensor = 1.0 / (variances + 1e-4)

        # Normalizar para que sumen 1
        precisions = precisions / precisions.sum()

        return precisions

    def forward(
        self,
        current_errors: torch.Tensor,
        uncertainty: float = 0.5
    ) -> torch.Tensor:
        """
        Calcula atención sobre niveles de predicción.

        Args:
            current_errors: Errores actuales [n_levels]
            uncertainty: Incertidumbre del sistema [0,1]

        Returns:
            attention: Distribución de atención sobre niveles [n_levels]
        """
        # Actualizar historial
        self.update_history(current_errors)

        # Calcular precisiones
        precisions = self.compute_precisions()

        # Combinar errores y precisiones
        combined = torch.cat([current_errors, precisions])

        # Pasar por la red
        raw_attention = self.attention_net(combined) + self.level_bias

        # Precision-weighting: multiplicar por precisión
        raw_attention = raw_attention * precisions

        # Ajustar por incertidumbre
        # Alta incertidumbre → más atención a meta-nivel (L3)
        uncertainty_boost = torch.tensor([0.0, 0.0, uncertainty])
        raw_attention = raw_attention + uncertainty_boost

        # Softmax
        attention = F.softmax(raw_attention, dim=0)

        return attention


# =============================================================================
# INTEGRADOR DE ATENCIÓN
# =============================================================================

class AttentionIntegrator(nn.Module):
    """
    Integra los 3 niveles de atención en una representación unificada.

    Calcula:
    - global_attention: Atención final combinada [4]
    - attention_intensity: Qué tan enfocada está la atención [0,1]
    - attention_coherence: Qué tan alineados están los niveles [0,1]
    """

    def __init__(self, state_dim: int = 4) -> None:
        super().__init__()

        self.state_dim = state_dim

        # Pesos para combinar niveles
        self.level_weights = nn.Parameter(torch.tensor([0.4, 0.35, 0.25]))

        # Red para detectar conflictos entre niveles
        self.coherence_net = nn.Sequential(
            nn.Linear(state_dim + 3, 8),  # archetypal + error attention
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Umbral de coherencia para activar resolución de conflictos
        self.coherence_threshold = 0.3

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.coherence_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        archetypal_attention: torch.Tensor,  # [4]
        temporal_attention: torch.Tensor,     # [buffer_size]
        error_attention: torch.Tensor,        # [3]
        attended_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Integra los niveles de atención.

        Returns:
            global_attention: Atención integrada [4]
            intensity: Intensidad de atención [0,1]
            coherence: Coherencia entre niveles [0,1]
        """
        # La atención global está principalmente guiada por arquetipal
        # pero modulada por los otros niveles

        # Intensidad = qué tan enfocada está la atención arquetipal
        intensity = 1.0 - self._entropy(archetypal_attention)

        # Factor de modulación desde error attention
        # Si L3 (meta) tiene alta atención → más reflexivo → difuminar atención
        meta_attention = error_attention[2].item()
        modulation = 1.0 - (meta_attention * 0.3)  # Hasta 30% de difuminación

        # Aplicar modulación
        modulated_attention = archetypal_attention ** modulation
        modulated_attention = modulated_attention / modulated_attention.sum()

        # Si hay memoria atendida, incorporarla
        if attended_memory is not None:
            # Los primeros 4 elementos de la memoria son el estado
            memory_state = attended_memory[:4]
            memory_state = F.softmax(memory_state, dim=0)

            # Combinar con peso temporal
            temporal_weight = self.level_weights[1]
            global_attention = (
                (1 - temporal_weight) * modulated_attention +
                temporal_weight * memory_state
            )
        else:
            global_attention = modulated_attention

        # Normalizar
        global_attention = global_attention / global_attention.sum()

        # Calcular coherencia
        combined = torch.cat([archetypal_attention, error_attention])
        coherence = self.coherence_net(combined).item()

        return global_attention, intensity, coherence

    def _entropy(self, probs: torch.Tensor) -> float:
        """Calcula entropía normalizada [0,1]"""
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps))
        max_entropy = np.log(len(probs))
        normalized_entropy: float = (entropy / max_entropy).item()
        return normalized_entropy

    def resolve_conflict(
        self,
        global_attention: torch.Tensor,
        coherence: float
    ) -> torch.Tensor:
        """
        Resuelve conflictos cuando la coherencia es baja.

        Estrategias:
        - Muy baja coherencia → empujar hacia el Self (centro)
        - Baja coherencia → explorar (aumentar entropía)
        """
        if coherence > self.coherence_threshold:
            return global_attention

        # Muy baja coherencia → hacia el Self (distribución uniforme)
        if coherence < 0.15:
            self_state = torch.ones(4) / 4
            blend = 0.5  # 50% hacia el Self
            return (1 - blend) * global_attention + blend * self_state

        # Baja coherencia → aumentar exploración
        noise = torch.rand(4) * 0.1
        exploratory = global_attention + noise
        return exploratory / exploratory.sum()


# =============================================================================
# MÉTRICAS DE ATENCIÓN
# =============================================================================

@dataclass
class AttentionMetrics:
    """Métricas para evaluar la calidad de la atención"""

    # Historial para cálculos
    attention_history: List[torch.Tensor] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)
    intensity_history: List[float] = field(default_factory=list)

    history_size: int = 50

    def update(
        self,
        attention: torch.Tensor,
        coherence: float,
        intensity: float
    ) -> None:
        """Actualiza el historial"""
        self.attention_history.append(attention.detach())
        self.coherence_history.append(coherence)
        self.intensity_history.append(intensity)

        # Mantener tamaño
        if len(self.attention_history) > self.history_size:
            self.attention_history.pop(0)
            self.coherence_history.pop(0)
            self.intensity_history.pop(0)

    def compute(self) -> Dict[str, float]:
        """
        Calcula métricas de atención.

        Returns:
            Dict con:
            - entropy: Entropía promedio de la atención
            - stability: Estabilidad temporal de la atención
            - flexibility: Capacidad de cambiar el foco
            - integration: Calidad de integración (coherencia promedio)
        """
        if len(self.attention_history) < 2:
            return {
                'entropy': 0.5,
                'stability': 0.5,
                'flexibility': 0.5,
                'integration': 0.5
            }

        # Entropía promedio
        entropies = []
        for att in self.attention_history:
            eps = 1e-8
            entropy = -torch.sum(att * torch.log(att + eps))
            max_entropy = np.log(len(att))
            entropies.append((entropy / max_entropy).item())
        avg_entropy = np.mean(entropies)

        # Estabilidad (baja varianza = alta estabilidad)
        att_stack = torch.stack(self.attention_history)
        variance = att_stack.var(dim=0).mean().item()
        stability = 1.0 / (1.0 + variance * 10)

        # Flexibilidad (capacidad de cambio)
        changes = []
        for i in range(1, len(self.attention_history)):
            diff = torch.abs(self.attention_history[i] - self.attention_history[i-1])
            changes.append(diff.mean().item())
        flexibility = np.mean(changes) if changes else 0.5

        # Integración (coherencia promedio)
        integration = np.mean(self.coherence_history)

        return {
            'entropy': avg_entropy,
            'stability': stability,
            'flexibility': min(1.0, flexibility * 5),  # Escalar para [0,1]
            'integration': integration
        }


# =============================================================================
# SISTEMA COMPLETO DE ATENCIÓN
# =============================================================================

class ZetaAttentionSystem(nn.Module):
    """
    Sistema completo de atención selectiva jerárquica.

    Integra:
    - Nivel 3: GlobalArchetypalAttention
    - Nivel 2: TemporalAttention + MemoryBuffer
    - Nivel 1: ErrorAttention
    - AttentionIntegrator
    """

    def __init__(
        self,
        state_dim: int = 4,
        memory_size: int = 100,
        temperature: float = 1.0
    ) -> None:
        super().__init__()

        self.state_dim = state_dim

        # Componentes
        self.context_detector = ContextDetector(state_dim)
        self.archetypal_attention = GlobalArchetypalAttention(state_dim, temperature)
        self.memory_buffer = MemoryBuffer(memory_size)
        self.temporal_attention = TemporalAttention()
        self.error_attention = ErrorAttention()
        self.integrator = AttentionIntegrator(state_dim)

        # Métricas
        self.metrics = AttentionMetrics()

        # Estado interno
        self.current_time = 0.0
        self.last_output: Optional[AttentionOutput] = None

    def forward(
        self,
        stimulus: torch.Tensor,
        state: torch.Tensor,
        errors: torch.Tensor,
        surprise: float,
        uncertainty: float = 0.5
    ) -> AttentionOutput:
        """
        Procesa un paso de atención.

        Args:
            stimulus: Estímulo actual [4]
            state: Estado arquetipal actual [4]
            errors: Errores de predicción [3] (L1, L2, L3)
            surprise: Sorpresa total del sistema
            uncertainty: Incertidumbre del meta-predictor

        Returns:
            AttentionOutput con toda la información de atención
        """
        self.current_time += 1.0

        # ===== NIVEL 3: Contexto y Atención Arquetipal =====
        context = self.context_detector(stimulus, state)
        arch_attention = self.archetypal_attention(state, context, uncertainty)

        # ===== NIVEL 1: Atención sobre Errores =====
        err_attention = self.error_attention(errors, uncertainty)

        # ===== Añadir a memoria =====
        memory_item = MemoryItem(
            state=state.detach(),
            stimulus=stimulus.detach(),
            errors=errors.detach(),
            surprise=surprise,
            timestamp=self.current_time,
            archetype_dominant=int(state.argmax().item())
        )
        self.memory_buffer.add(memory_item)
        self.memory_buffer.decay()

        # ===== NIVEL 2: Atención Temporal =====
        memories, importance = self.memory_buffer.get_tensors()

        # Query es el estado actual concatenado
        query = memory_item.to_tensor()

        attended_memory, temp_attention = self.temporal_attention(
            query, memories, importance, arch_attention, self.current_time
        )

        # ===== INTEGRACIÓN =====
        global_attention, intensity, coherence = self.integrator(
            arch_attention, temp_attention, err_attention, attended_memory
        )

        # Resolver conflictos si es necesario
        if coherence < self.integrator.coherence_threshold:
            global_attention = self.integrator.resolve_conflict(
                global_attention, coherence
            )

        # ===== MÉTRICAS =====
        self.metrics.update(global_attention, coherence, intensity)

        # ===== CONSTRUIR SALIDA =====
        output = AttentionOutput(
            archetypal_attention=arch_attention,
            temporal_attention=temp_attention,
            error_attention=err_attention,
            global_attention=global_attention,
            attention_intensity=intensity,
            attention_coherence=coherence,
            context=context,
            attended_memory=attended_memory
        )

        self.last_output = output
        return output

    def get_metrics(self) -> Dict[str, float]:
        """Retorna métricas actuales de atención"""
        return self.metrics.compute()

    def get_attention_index(self) -> float:
        """
        Calcula un índice general de calidad de atención [0,1].

        Combina:
        - Intensidad (foco)
        - Coherencia (integración)
        - Estabilidad temporal
        """
        metrics = self.metrics.compute()

        # Ponderación
        index = (
            0.30 * (1.0 - metrics['entropy']) +  # Foco (baja entropía)
            0.30 * metrics['integration'] +       # Coherencia
            0.20 * metrics['stability'] +         # Estabilidad
            0.20 * metrics['flexibility']         # Flexibilidad
        )

        return min(1.0, max(0.0, index))


# =============================================================================
# DEMO Y PRUEBAS
# =============================================================================

def demo_attention_system() -> None:
    """Demostracion del sistema de atencion"""

    print("=" * 70)
    print("DEMO: Sistema de Atencion Selectiva Jerarquica")
    print("=" * 70)

    # Crear sistema
    system = ZetaAttentionSystem(
        state_dim=4,
        memory_size=50,
        temperature=1.0
    )

    # Nombres de arquetipos
    ARCHETYPE_NAMES = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']

    # Simular diferentes escenarios
    scenarios = [
        {
            'name': 'Amenaza detectada',
            'stimulus': torch.tensor([0.1, 0.8, 0.05, 0.05]),  # Alta SOMBRA
            'state': torch.tensor([0.4, 0.3, 0.15, 0.15]),
            'errors': torch.tensor([0.3, 0.5, 0.2]),
            'surprise': 0.7,
            'uncertainty': 0.6
        },
        {
            'name': 'Oportunidad social',
            'stimulus': torch.tensor([0.7, 0.1, 0.1, 0.1]),  # Alta PERSONA
            'state': torch.tensor([0.5, 0.2, 0.15, 0.15]),
            'errors': torch.tensor([0.2, 0.3, 0.1]),
            'surprise': 0.4,
            'uncertainty': 0.3
        },
        {
            'name': 'Necesidad emocional',
            'stimulus': torch.tensor([0.1, 0.1, 0.7, 0.1]),  # Alta ANIMA
            'state': torch.tensor([0.2, 0.2, 0.4, 0.2]),
            'errors': torch.tensor([0.4, 0.4, 0.3]),
            'surprise': 0.5,
            'uncertainty': 0.5
        },
        {
            'name': 'Demanda cognitiva',
            'stimulus': torch.tensor([0.1, 0.1, 0.1, 0.7]),  # Alta ANIMUS
            'state': torch.tensor([0.2, 0.2, 0.2, 0.4]),
            'errors': torch.tensor([0.5, 0.6, 0.4]),
            'surprise': 0.6,
            'uncertainty': 0.4
        },
        {
            'name': 'Estado balanceado',
            'stimulus': torch.tensor([0.25, 0.25, 0.25, 0.25]),
            'state': torch.tensor([0.25, 0.25, 0.25, 0.25]),
            'errors': torch.tensor([0.3, 0.3, 0.3]),
            'surprise': 0.3,
            'uncertainty': 0.5
        }
    ]

    for i, scenario in enumerate(scenarios):
        print(f"\n{'-' * 70}")
        print(f"Escenario {i+1}: {scenario['name']}")
        print(f"{'-' * 70}")

        # Procesar
        output = system(
            stimulus=scenario['stimulus'],
            state=scenario['state'],
            errors=scenario['errors'],
            surprise=scenario['surprise'],
            uncertainty=scenario['uncertainty']
        )

        # Mostrar contexto detectado
        print(f"\nContexto detectado:")
        for ctx_name, ctx_val in output.context.items():
            bar = '#' * int(ctx_val * 20)
            print(f"  {ctx_name:12}: {bar:<20} {ctx_val:.2f}")

        # Mostrar atencion arquetipal
        print(f"\nAtencion Arquetipal (Nivel 3):")
        for j, (name, val) in enumerate(zip(ARCHETYPE_NAMES, output.archetypal_attention)):
            bar = '#' * int(val.item() * 20)
            marker = ' <<<' if j == output.archetypal_attention.argmax() else ''
            print(f"  {name:8}: {bar:<20} {val.item():.2f}{marker}")

        # Mostrar atencion de errores
        print(f"\nAtencion de Errores (Nivel 1):")
        for j, (name, val) in enumerate(zip(['L1-Stim', 'L2-State', 'L3-Meta'], output.error_attention)):
            bar = '#' * int(val.item() * 20)
            print(f"  {name:8}: {bar:<20} {val.item():.2f}")

        # Mostrar atencion global
        print(f"\nAtencion Global Integrada:")
        for j, (name, val) in enumerate(zip(ARCHETYPE_NAMES, output.global_attention)):
            bar = '#' * int(val.item() * 20)
            marker = ' <<<' if j == output.global_attention.argmax() else ''
            print(f"  {name:8}: {bar:<20} {val.item():.2f}{marker}")

        # Metricas
        print(f"\nMetricas:")
        print(f"  Intensidad:  {output.attention_intensity:.2f}")
        print(f"  Coherencia:  {output.attention_coherence:.2f}")
        print(f"  Memoria:     {len(system.memory_buffer)} items")

    # Metricas finales
    print(f"\n{'=' * 70}")
    print("METRICAS FINALES DEL SISTEMA")
    print(f"{'=' * 70}")

    metrics = system.get_metrics()
    for name, val in metrics.items():
        bar = '#' * int(val * 30)
        print(f"  {name:12}: {bar:<30} {val:.2f}")

    attention_index = system.get_attention_index()
    print(f"\n  INDICE DE ATENCION: {'#' * int(attention_index * 30):<30} {attention_index:.2%}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    demo_attention_system()
