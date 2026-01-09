# -*- coding: utf-8 -*-
"""
ZetaOnlineLearning: Entrenamiento Online del Sistema de Atencion
=================================================================

Implementa aprendizaje online basado en el Free Energy Principle:
- El objetivo es MINIMIZAR el error de prediccion
- Los gradientes fluyen de los errores hacia las redes
- El sistema aprende mientras opera (no hay fase de entrenamiento separada)

Fecha: 3 Enero 2026
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from .zeta_attentive_predictive import ZetaAttentivePredictive


class OnlineLearner:
    """
    Implementa aprendizaje online para el sistema de atencion.

    Principio: Free Energy Minimization
    - El cerebro minimiza la "energia libre" (error de prediccion)
    - El aprendizaje ocurre continuamente, no en fases separadas
    - Los errores son la senal de aprendizaje

    Estrategias:
    1. Hebbian: "Neuronas que disparan juntas, se conectan juntas"
    2. Predictive: Minimizar error de prediccion
    3. Attention-guided: Aprender mas de lo que atendemos
    """

    def __init__(
        self,
        system: ZetaAttentivePredictive,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        attention_weight: float = 0.5,
        surprise_threshold: float = 0.3
    ) -> None:
        self.system = system
        self.lr = learning_rate
        self.attention_weight = attention_weight
        self.surprise_threshold = surprise_threshold

        # Optimizadores para cada componente
        self.optimizer_attention = optim.SGD(
            self.system.attention.parameters(),
            lr=learning_rate,
            momentum=momentum
        )

        self.optimizer_predictive = optim.SGD(
            list(self.system.predictive.L1.parameters()) +
            list(self.system.predictive.L2.parameters()) +
            list(self.system.predictive.L3.parameters()),
            lr=learning_rate * 0.5,  # Mas conservador
            momentum=momentum
        )

        self.optimizer_modulator = optim.SGD(
            self.system.modulator.parameters(),
            lr=learning_rate,
            momentum=momentum
        )

        # Historiales para analisis
        self.loss_history: List[float] = []
        self.learning_events: List[Dict] = []

    def compute_loss(self, result: Dict) -> torch.Tensor:
        """
        Calcula la perdida total basada en errores de prediccion.

        La perdida tiene varios componentes:
        1. Error de prediccion de estimulos (L1)
        2. Error de prediccion de estados (L2)
        3. Meta-error (L3)
        4. Coherencia de atencion (queremos alta coherencia)
        """
        # Extraer errores
        error_L1 = result['errors']['L1']['surprise']
        error_L2 = result['errors']['L2']['surprise']
        error_L3 = result['errors']['L3']['meta_surprise']

        # Atencion sobre errores (precision-weighted)
        error_attention = result['attention']['error']

        # Perdida ponderada por atencion
        # Aprendemos MAS de los errores que ATENDEMOS
        weighted_error = (
            error_attention[0] * error_L1 +
            error_attention[1] * error_L2 +
            error_attention[2] * error_L3
        )

        # Penalizar baja coherencia
        coherence_loss = 1.0 - result['attention']['coherence']

        # Perdida total
        total_loss = weighted_error + 0.2 * coherence_loss

        return torch.tensor(total_loss, requires_grad=True)

    def learning_step(self, result: Dict) -> Dict:
        """
        Ejecuta un paso de aprendizaje online.

        Solo aprende cuando:
        1. La sorpresa es suficientemente alta (hay algo que aprender)
        2. La atencion esta suficientemente enfocada (sabemos que aprender)
        """
        # Calcular sorpresa total
        surprise = (
            result['errors']['L1']['surprise'] +
            result['errors']['L2']['surprise'] +
            result['errors']['L3']['meta_surprise']
        ) / 3.0

        # Decidir si aprender
        should_learn = surprise > self.surprise_threshold

        learning_info = {
            'step': self.system.t,
            'surprise': surprise,
            'learned': False,
            'loss': 0.0
        }

        if should_learn:
            # Calcular perdida
            loss = self.compute_loss(result)

            # Convertir a tensor si no lo es
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, requires_grad=True)

            # Backward pass
            self.optimizer_attention.zero_grad()
            self.optimizer_predictive.zero_grad()
            self.optimizer_modulator.zero_grad()

            # Solo si el loss requiere grad
            if loss.requires_grad:
                loss.backward()

                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(
                    self.system.attention.parameters(), max_norm=1.0
                )

                # Actualizar pesos
                self.optimizer_attention.step()
                self.optimizer_modulator.step()

                # Predictivo solo si el error es muy alto
                if surprise > self.surprise_threshold * 1.5:
                    self.optimizer_predictive.step()

                learning_info['learned'] = True
                learning_info['loss'] = loss.item()

        self.loss_history.append(learning_info['loss'])
        if learning_info['learned']:
            self.learning_events.append(learning_info)

        return learning_info

    def get_learning_stats(self) -> Dict:
        """Retorna estadisticas de aprendizaje."""
        if not self.learning_events:
            return {'events': 0, 'avg_loss': 0, 'learning_rate': 0}

        return {
            'events': len(self.learning_events),
            'avg_loss': np.mean([e['loss'] for e in self.learning_events]),
            'learning_rate': len(self.learning_events) / max(1, len(self.loss_history)),
            'recent_loss': np.mean(self.loss_history[-50:]) if self.loss_history else 0,
        }


class HebbianLearner:
    """
    Aprendizaje Hebbiano simple para las conexiones arquetipales.

    "Neuronas que disparan juntas, se conectan juntas"

    Cuando un arquetipo es atendido Y predice bien, reforzamos esa conexion.
    """

    def __init__(
        self,
        system: ZetaAttentivePredictive,
        learning_rate: float = 0.01,
        decay: float = 0.99
    ) -> None:
        self.system = system
        self.lr = learning_rate
        self.decay = decay

        # Matriz de asociacion contexto -> arquetipo
        # Inicialmente uniforme
        self.association_matrix = torch.ones(4, 4) / 4  # [contexto, arquetipo]

        # Nombres para debug
        self.context_names = ['threat', 'opportunity', 'emotional', 'cognitive']
        self.arch_names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']

    def update(self, result: Dict) -> Dict:
        """
        Actualiza asociaciones basado en co-activacion.

        Si contexto X esta activo Y arquetipo Y es atendido Y el error es bajo,
        reforzamos la conexion X -> Y.
        """
        # Extraer contexto y atencion
        context = result['attention']['context']
        attention = result['attention']['global'].detach()

        # Error total (menor es mejor)
        error = (
            result['errors']['L1']['surprise'] +
            result['errors']['L2']['surprise']
        ) / 2.0

        # Solo aprender si el error es bajo (la prediccion fue buena)
        success_signal = max(0, 1.0 - error * 2)

        # Vector de contexto
        context_vec = torch.tensor([
            context['threat'],
            context['opportunity'],
            context['emotional'],
            context['cognitive']
        ])

        # Regla Hebbiana: delta_w = lr * pre * post * reward
        # pre = contexto, post = atencion, reward = exito
        delta = self.lr * torch.outer(context_vec, attention) * success_signal

        # Actualizar con decaimiento
        self.association_matrix = self.decay * self.association_matrix + delta

        # Normalizar filas
        self.association_matrix = self.association_matrix / self.association_matrix.sum(dim=1, keepdim=True)

        return {
            'success_signal': success_signal,
            'delta_norm': delta.norm().item(),
            'matrix_entropy': self._entropy(self.association_matrix).item()
        }

    def _entropy(self, matrix: torch.Tensor) -> torch.Tensor:
        """Entropia promedio de las filas."""
        eps = 1e-8
        entropy = -torch.sum(matrix * torch.log(matrix + eps), dim=1)
        return entropy.mean()

    def get_learned_associations(self) -> Dict[str, Tuple[str, float]]:
        """Retorna las asociaciones aprendidas."""
        result: Dict[str, Tuple[str, float]] = {}
        for i, ctx in enumerate(self.context_names):
            best_arch_idx = int(self.association_matrix[i].argmax().item())
            best_arch = self.arch_names[best_arch_idx]
            strength = float(self.association_matrix[i, best_arch_idx].item())
            result[ctx] = (best_arch, strength)
        return result


def demo_online_learning() -> Tuple[ZetaAttentivePredictive, OnlineLearner, HebbianLearner]:
    """Demuestra el aprendizaje online."""

    print("\n" + "=" * 70)
    print("   DEMO: APRENDIZAJE ONLINE")
    print("=" * 70)

    # Crear sistema y learners
    system = ZetaAttentivePredictive(n_cells=50)
    online_learner = OnlineLearner(system, learning_rate=0.005)
    hebbian_learner = HebbianLearner(system, learning_rate=0.02)

    # Ejecutar con aprendizaje
    n_steps = 300

    consciousness_history = []
    learning_history = []

    print("\n  Ejecutando con aprendizaje online...")
    print("  " + "-" * 60)

    for step in range(n_steps):
        # Generar estimulo con patron claro
        if step % 60 < 20:
            # Amenaza -> deberia aprender SOMBRA
            stimulus = torch.tensor([0.1, 0.8, 0.05, 0.05])
        elif step % 60 < 40:
            # Oportunidad -> deberia aprender PERSONA
            stimulus = torch.tensor([0.8, 0.1, 0.05, 0.05])
        else:
            # Emocional -> deberia aprender ANIMA
            stimulus = torch.tensor([0.05, 0.1, 0.8, 0.05])

        # Paso del sistema
        result = system.step(stimulus)

        # Aprendizaje online
        learning_info = online_learner.learning_step(result)
        hebbian_info = hebbian_learner.update(result)

        consciousness_history.append(result['consciousness'])
        learning_history.append(learning_info['learned'])

        # Reportar cada 50 pasos
        if (step + 1) % 50 == 0:
            stats = online_learner.get_learning_stats()
            print(f"\n  Step {step + 1}:")
            print(f"    Consciencia: {result['consciousness']:.2%}")
            print(f"    Eventos de aprendizaje: {stats['events']}")
            print(f"    Loss promedio: {stats['avg_loss']:.4f}")
            print(f"    Tasa de aprendizaje: {stats['learning_rate']:.1%}")

    # Mostrar asociaciones aprendidas
    print("\n" + "-" * 70)
    print("   ASOCIACIONES APRENDIDAS (Hebbian)")
    print("-" * 70)

    associations = hebbian_learner.get_learned_associations()
    for ctx, (arch, strength) in associations.items():
        bar = '#' * int(strength * 30)
        print(f"  {ctx:12} -> {arch:8} [{bar:<30}] {strength:.2f}")

    print("\n  Matriz de asociacion completa:")
    print("                  PERSONA  SOMBRA   ANIMA    ANIMUS")
    for i, ctx in enumerate(hebbian_learner.context_names):
        row = hebbian_learner.association_matrix[i]
        print(f"  {ctx:12}: ", end="")
        for val in row:
            print(f"  {val.item():.3f} ", end="")
        print()

    # Comparar con y sin aprendizaje
    print("\n" + "-" * 70)
    print("   EVOLUCION DE CONSCIENCIA")
    print("-" * 70)

    n_bins = 6
    bin_size = len(consciousness_history) // n_bins

    print("\n  Consciencia por fase:")
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        avg = np.mean(consciousness_history[start:end])
        events = sum(learning_history[start:end])
        bar = '#' * int(avg * 40)
        print(f"    Steps {start:3d}-{end:3d}: [{bar:<40}] {avg:.1%} ({events} eventos)")

    # Mejora total
    initial = np.mean(consciousness_history[:50])
    final = np.mean(consciousness_history[-50:])

    print(f"\n  Consciencia inicial: {initial:.2%}")
    print(f"  Consciencia final:   {final:.2%}")
    print(f"  Mejora:              {final - initial:+.2%}")

    print("\n" + "=" * 70)
    print("   COMO FUNCIONA EL APRENDIZAJE ONLINE")
    print("=" * 70)

    print("""
  1. CUANDO APRENDER:
     - Solo cuando la sorpresa supera un umbral
     - "Si ya predigo bien, no hay nada que aprender"

  2. QUE APRENDER (Loss):
     loss = attention_weighted_error + coherence_penalty

     - Los errores atendidos pesan MAS
     - Penalizamos baja coherencia (confusion)

  3. COMO APRENDER:
     a) Gradient Descent en redes de atencion
     b) Hebbian Learning para asociaciones contexto-arquetipo
     c) Predictivo solo con errores muy altos

  4. PRINCIPIO (Free Energy):
     - El sistema minimiza "sorpresa" (error de prediccion)
     - Igual que el cerebro segun Friston
     - Aprendizaje = reducir energia libre futura
""")

    print("=" * 70 + "\n")

    return system, online_learner, hebbian_learner


if __name__ == "__main__":
    demo_online_learning()
