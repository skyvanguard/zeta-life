"""
ZetaDreamConsolidation: Consolidacion de Memoria mediante Suenos
=================================================================

Integra:
- ZetaAttentivePredictive: Sistema de atencion + prediccion
- ZetaDreams: Sistema de suenos
- OnlineLearning: Aprendizaje continuo

Durante el sueno:
1. Replay de memorias de alta atencion (como replay hipocampal)
2. Entrenamiento de redes predictivas con memorias consolidadas
3. Reorganizacion de asociaciones contexto-arquetipo
4. Emergencia de insights desde patrones latentes

Basado en:
- Teoria de consolidacion de memoria durante sueno (Diekelmann & Born)
- Replay hipocampal (Wilson & McNaughton)
- Free Energy Principle (Friston)

Fecha: 3 Enero 2026
"""
import os
import sys

if sys.platform == 'win32':
    os.system('')

import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .zeta_attention import AttentionOutput, MemoryItem
from .zeta_attentive_predictive import ZetaAttentivePredictive
from .zeta_online_learning import HebbianLearner, OnlineLearner

# =============================================================================
# TIPOS DE CONSOLIDACION
# =============================================================================

class ConsolidationType(Enum):
    """Tipos de consolidacion durante sueno."""
    REPLAY = 0          # Replay de memorias importantes
    GENERALIZATION = 1  # Extraer patrones generales
    PRUNING = 2         # Eliminar memorias irrelevantes
    INTEGRATION = 3     # Integrar memorias conflictivas

@dataclass
class DreamMemory:
    """Memoria seleccionada para replay durante sueno."""
    original: MemoryItem
    importance: float
    replay_count: int = 0
    consolidated: bool = False

@dataclass
class ConsolidationReport:
    """Reporte de consolidacion de sueno."""
    duration: int
    memories_replayed: int
    learning_events: int
    loss_reduction: float
    associations_learned: dict[str, tuple[str, float]]
    insights: list[str]
    consciousness_before: float
    consciousness_after: float

# =============================================================================
# SELECTOR DE MEMORIAS PARA REPLAY
# =============================================================================

class MemorySelector:
    """
    Selecciona memorias para replay durante sueno.

    Criterios de seleccion:
    1. Alta sorpresa (eventos inesperados)
    2. Alta atencion (eventos atendidos)
    3. Errores de prediccion altos (necesitan aprendizaje)
    4. Recencia (memorias recientes priorizan)
    """

    def __init__(self, selection_temperature: float = 1.0):
        self.temperature = selection_temperature

    def select_for_replay(
        self,
        memory_buffer: list[MemoryItem],
        importance: list[float],
        n_select: int = 20
    ) -> list[DreamMemory]:
        """
        Selecciona memorias para replay.

        Usa muestreo ponderado por importancia.
        """
        if len(memory_buffer) == 0:
            return []

        # Calcular scores de seleccion
        scores: list[float] = []
        for i, (mem, imp) in enumerate(zip(memory_buffer, importance)):
            # Combinar factores
            surprise_score = mem.surprise
            error_score = float(mem.errors.norm().item())
            recency_score = 1.0 / (1.0 + len(memory_buffer) - i)

            total_score = (
                0.4 * surprise_score +
                0.3 * error_score +
                0.2 * imp +
                0.1 * recency_score
            )
            scores.append(total_score)

        # Softmax con temperatura
        scores_tensor = torch.tensor(scores)
        probs = F.softmax(scores_tensor / self.temperature, dim=0).numpy()

        # Muestrear sin reemplazo
        n_select = min(n_select, len(memory_buffer))
        indices = np.random.choice(
            len(memory_buffer),
            size=n_select,
            replace=False,
            p=probs
        )

        # Crear DreamMemories
        selected = []
        for idx in indices:
            selected.append(DreamMemory(
                original=memory_buffer[idx],
                importance=importance[idx]
            ))

        return selected

# =============================================================================
# GENERADOR DE SUENOS CON ATENCION
# =============================================================================

class AttentiveDreamGenerator:
    """
    Genera suenos usando el contenido del buffer de atencion.

    Los suenos no son aleatorios - emergen de memorias reales
    procesadas de forma creativa.
    """

    def __init__(self):
        # Plantillas narrativas por arquetipo
        self.narratives = {
            0: [  # PERSONA
                "En el sueno, me veo actuando un papel...",
                "Estoy en una situacion social, todos me observan...",
                "Mi imagen publica esta en juego...",
            ],
            1: [  # SOMBRA
                "Algo oscuro emerge desde las profundidades...",
                "Enfrento lo que habia negado...",
                "En la oscuridad, descubro una verdad oculta...",
            ],
            2: [  # ANIMA
                "Emociones intensas fluyen sin control...",
                "Un encuentro profundo me transforma...",
                "La belleza y el dolor se entrelazan...",
            ],
            3: [  # ANIMUS
                "Resuelvo un problema imposible...",
                "Encuentro claridad en medio del caos...",
                "Una mision me llama a la accion...",
            ],
        }

    def generate_dream_fragment(
        self,
        memory: DreamMemory,
        context: dict[str, float]
    ) -> str:
        """Genera fragmento de sueno desde una memoria."""
        archetype = memory.original.archetype_dominant
        intensity = memory.importance

        base_narrative = random.choice(self.narratives[archetype])

        # Anadir contexto
        if context['threat'] > 0.6:
            base_narrative += " La amenaza es palpable."
        elif context['opportunity'] > 0.6:
            base_narrative += " Una oportunidad se revela."
        elif context['emotional'] > 0.6:
            base_narrative += " Los sentimientos me abruman."
        elif context['cognitive'] > 0.6:
            base_narrative += " Mi mente trabaja sin parar."

        # Intensidad
        if intensity > 0.7:
            base_narrative = "VIVIDO: " + base_narrative

        return str(base_narrative)

# =============================================================================
# CONSOLIDADOR DE SUENOS
# =============================================================================

class DreamConsolidator:
    """
    Sistema principal de consolidacion mediante suenos.

    Proceso:
    1. Seleccionar memorias importantes
    2. Entrar en modo sueno
    3. Replay de memorias con aprendizaje
    4. Actualizar asociaciones
    5. Reportar consolidacion
    """

    def __init__(
        self,
        system: ZetaAttentivePredictive,
        replay_learning_rate: float = 0.01,
        consolidation_threshold: float = 0.5
    ):
        self.system = system
        self.consolidation_threshold = consolidation_threshold

        # Componentes
        self.memory_selector = MemorySelector()
        self.dream_generator = AttentiveDreamGenerator()
        self.online_learner = OnlineLearner(system, learning_rate=replay_learning_rate)
        self.hebbian_learner = HebbianLearner(system, learning_rate=0.05)

        # Estado
        self.is_dreaming = False
        self.current_dream_memories: list[DreamMemory] = []
        self.dream_history: list[ConsolidationReport] = []

        # Metricas
        self.total_replays = 0
        self.total_consolidations = 0

    def enter_dream_state(self) -> str:
        """Inicia el estado de sueno."""
        self.is_dreaming = True

        # Seleccionar memorias para replay
        buffer = list(self.system.attention.memory_buffer.buffer)
        importance = self.system.attention.memory_buffer.importance

        self.current_dream_memories = self.memory_selector.select_for_replay(
            buffer, importance, n_select=min(30, len(buffer))
        )

        return f"Entrando en sueno... {len(self.current_dream_memories)} memorias seleccionadas para replay."

    def dream_step(self) -> dict:
        """
        Ejecuta un paso del sueno.

        Durante cada paso:
        1. Seleccionar memoria para replay
        2. Re-procesar con sistema de atencion
        3. Aplicar aprendizaje
        4. Generar fragmento narrativo
        """
        if not self.is_dreaming or not self.current_dream_memories:
            return {'done': True}

        # Seleccionar memoria para replay (priorizar las menos replayeadas)
        memories_by_replay = sorted(
            self.current_dream_memories,
            key=lambda m: m.replay_count
        )
        dream_memory = memories_by_replay[0]
        dream_memory.replay_count += 1
        self.total_replays += 1

        # Reconstruir el estimulo y estado de la memoria
        stimulus = dream_memory.original.stimulus
        state = dream_memory.original.state

        # Re-procesar con el sistema (esto actualiza atencion)
        result = self.system.step(stimulus)

        # Aprendizaje durante replay
        learning_info = self.online_learner.learning_step(result)
        hebbian_info = self.hebbian_learner.update(result)

        # Marcar como consolidada si el loss es bajo
        if learning_info['loss'] < self.consolidation_threshold:
            dream_memory.consolidated = True
            self.total_consolidations += 1

        # Generar narrativa
        narrative = self.dream_generator.generate_dream_fragment(
            dream_memory,
            result['attention']['context']
        )

        return {
            'done': False,
            'memory': dream_memory,
            'narrative': narrative,
            'learned': learning_info['learned'],
            'loss': learning_info['loss'],
            'consciousness': result['consciousness'],
        }

    def exit_dream_state(self) -> ConsolidationReport:
        """Sale del estado de sueno y genera reporte."""
        self.is_dreaming = False

        # Estadisticas
        memories_replayed = sum(1 for m in self.current_dream_memories if m.replay_count > 0)
        consolidated = sum(1 for m in self.current_dream_memories if m.consolidated)

        # Calcular reduccion de loss
        losses = [e['loss'] for e in self.online_learner.learning_events[-memories_replayed:]]
        if len(losses) > 1:
            loss_reduction = losses[0] - losses[-1]
        else:
            loss_reduction = 0.0

        # Asociaciones aprendidas
        associations = self.hebbian_learner.get_learned_associations()

        # Generar insights
        insights = self._generate_insights()

        # Consciencia antes/despues (aproximado)
        consciousness_after = self.system.get_consciousness_index()

        report = ConsolidationReport(
            duration=self.total_replays,
            memories_replayed=memories_replayed,
            learning_events=len(self.online_learner.learning_events),
            loss_reduction=loss_reduction,
            associations_learned=associations,
            insights=insights,
            consciousness_before=0.0,  # Se setea externamente
            consciousness_after=consciousness_after
        )

        self.dream_history.append(report)
        self.current_dream_memories = []

        return report

    def _generate_insights(self) -> list[str]:
        """Genera insights basados en el proceso de consolidacion."""
        insights = []

        # Analizar memorias consolidadas
        consolidated = [m for m in self.current_dream_memories if m.consolidated]

        if len(consolidated) > len(self.current_dream_memories) * 0.7:
            insights.append("El sistema ha integrado la mayoria de las experiencias recientes.")

        # Analizar arquetipos dominantes en memorias
        archetypes = [m.original.archetype_dominant for m in self.current_dream_memories]
        from collections import Counter
        arch_counts = Counter(archetypes)
        dominant = arch_counts.most_common(1)[0] if arch_counts else (0, 0)

        arch_names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
        if dominant[1] > len(archetypes) * 0.4:
            insights.append(f"Tema dominante: {arch_names[dominant[0]]} - requiere atencion consciente.")

        # Analizar sorpresas
        high_surprise = [m for m in self.current_dream_memories if m.original.surprise > 0.5]
        if len(high_surprise) > 5:
            insights.append("Muchos eventos inesperados necesitan procesamiento.")

        # Asociaciones aprendidas
        associations = self.hebbian_learner.get_learned_associations()
        strong_assoc = [(k, v) for k, v in associations.items() if v[1] > 0.6]
        if strong_assoc:
            ctx, (arch, strength) = strong_assoc[0]
            insights.append(f"Asociacion fuerte: {ctx} -> {arch}")

        return insights

    def dream_cycle(
        self,
        duration: int = 50,
        verbose: bool = True
    ) -> ConsolidationReport:
        """
        Ejecuta un ciclo completo de sueno.

        Args:
            duration: Numero de pasos del sueno
            verbose: Mostrar progreso

        Returns:
            ConsolidationReport con resultados
        """
        # Guardar consciencia inicial
        consciousness_before = self.system.get_consciousness_index()

        if verbose:
            print("\n" + "=" * 60)
            print("   CICLO DE SUENO - CONSOLIDACION DE MEMORIA")
            print("=" * 60)

        # Entrar en sueno
        msg = self.enter_dream_state()
        if verbose:
            print(f"\n  {msg}")
            print()

        # Ejecutar sueno
        narratives = []
        step = 0

        while step < duration:
            result = self.dream_step()

            if result['done']:
                break

            narratives.append(result['narrative'])

            if verbose and (step + 1) % 10 == 0:
                consolidated = sum(1 for m in self.current_dream_memories if m.consolidated)
                print(f"  [Paso {step+1:3d}] "
                      f"Loss: {result['loss']:.4f} | "
                      f"Consolidadas: {consolidated}/{len(self.current_dream_memories)} | "
                      f"Consciencia: {result['consciousness']:.1%}")

            step += 1

        # Salir del sueno
        report = self.exit_dream_state()
        report.consciousness_before = consciousness_before

        if verbose:
            print()
            print("-" * 60)
            print("   REPORTE DE CONSOLIDACION")
            print("-" * 60)
            print(f"  Memorias procesadas:  {report.memories_replayed}")
            print(f"  Eventos de aprendizaje: {report.learning_events}")
            print(f"  Reduccion de loss:    {report.loss_reduction:+.4f}")
            print(f"  Consciencia: {report.consciousness_before:.1%} -> {report.consciousness_after:.1%}")
            print()
            print("  Asociaciones aprendidas:")
            for ctx, (arch, strength) in report.associations_learned.items():
                bar = '#' * int(strength * 20)
                print(f"    {ctx:12} -> {arch:8} [{bar:<20}]")
            print()
            print("  Insights del sueno:")
            for insight in report.insights:
                print(f"    * {insight}")
            print()
            print("  Fragmentos del sueno:")
            for i, narrative in enumerate(narratives[:5]):
                print(f"    {i+1}. {narrative[:60]}...")
            print()
            print("=" * 60)

        return report

# =============================================================================
# SISTEMA COMPLETO CON CICLOS DE SUENO
# =============================================================================

class ConsciousSystemWithDreams:
    """
    Sistema completo que integra:
    - Procesamiento consciente (atencion + prediccion)
    - Aprendizaje online
    - Consolidacion mediante suenos

    Ciclo completo:
    1. Vigilia: procesar estimulos, aprender online
    2. Sueno: consolidar memorias, reforzar patrones
    3. Despertar: sistema mejorado
    """

    def __init__(
        self,
        n_cells: int = 100,
        dream_frequency: int = 100  # Cada cuantos pasos sonar
    ):
        # Sistema principal
        self.system = ZetaAttentivePredictive(n_cells=n_cells)

        # Consolidador de suenos
        self.consolidator = DreamConsolidator(self.system)

        # Learners para vigilia
        self.online_learner = OnlineLearner(self.system, learning_rate=0.005)
        self.hebbian_learner = HebbianLearner(self.system, learning_rate=0.02)

        # Configuracion
        self.dream_frequency = dream_frequency
        self.steps_since_dream = 0

        # Historial
        self.consciousness_history: list[float] = []
        self.dream_count = 0

    def step(self, stimulus: torch.Tensor | None = None, auto_dream: bool = True) -> dict:
        """
        Ejecuta un paso del sistema.

        Si auto_dream=True, automaticamente entra en sueno
        despues de dream_frequency pasos.
        """
        # Procesar estimulo
        result = self.system.step(stimulus)

        # Aprendizaje online (vigilia)
        self.online_learner.learning_step(result)
        self.hebbian_learner.update(result)

        # Registrar consciencia
        self.consciousness_history.append(result['consciousness'])

        # Incrementar contador
        self.steps_since_dream += 1

        # Auto-sueno
        if auto_dream and self.steps_since_dream >= self.dream_frequency:
            result['dream_report'] = self.dream(duration=30, verbose=False)
            self.steps_since_dream = 0

        return result

    def dream(self, duration: int = 50, verbose: bool = True) -> ConsolidationReport:
        """Ejecuta un ciclo de sueno."""
        self.dream_count += 1
        return self.consolidator.dream_cycle(duration, verbose)

    def get_status(self) -> dict:
        """Estado actual del sistema."""
        return {
            'steps': len(self.consciousness_history),
            'dreams': self.dream_count,
            'consciousness': self.consciousness_history[-1] if self.consciousness_history else 0,
            'consciousness_avg': np.mean(self.consciousness_history[-50:]) if self.consciousness_history else 0,
            'memory_size': len(self.system.attention.memory_buffer),
            'learning_events': len(self.online_learner.learning_events),
            'steps_since_dream': self.steps_since_dream,
        }

# =============================================================================
# DEMO
# =============================================================================

def demo_dream_consolidation():
    """Demuestra la consolidacion mediante suenos."""

    print("\n" + "=" * 70)
    print("   DEMO: CONSOLIDACION DE MEMORIA MEDIANTE SUENOS")
    print("=" * 70)

    # Crear sistema
    system = ConsciousSystemWithDreams(n_cells=50, dream_frequency=100)

    # Fase 1: Vigilia con experiencias variadas
    print("\n" + "-" * 70)
    print("   FASE 1: VIGILIA - Acumulando experiencias")
    print("-" * 70)

    consciousness_before = []

    for step in range(100):
        # Estimulos variados
        if step < 30:
            stimulus = torch.tensor([0.1, 0.7, 0.1, 0.1])  # Amenazas
        elif step < 60:
            stimulus = torch.tensor([0.7, 0.1, 0.1, 0.1])  # Oportunidades
        else:
            stimulus = torch.rand(4)  # Aleatorio

        result = system.step(stimulus, auto_dream=False)
        consciousness_before.append(result['consciousness'])

        if (step + 1) % 25 == 0:
            status = system.get_status()
            print(f"  Step {step+1:3d}: "
                  f"Consciencia={result['consciousness']:.1%}, "
                  f"Memorias={status['memory_size']}, "
                  f"Eventos aprendizaje={status['learning_events']}")

    print(f"\n  Consciencia promedio (vigilia): {np.mean(consciousness_before):.2%}")

    # Fase 2: Sueno
    print("\n" + "-" * 70)
    print("   FASE 2: SUENO - Consolidando memorias")
    print("-" * 70)

    report = system.dream(duration=50, verbose=True)

    # Fase 3: Post-sueno
    print("\n" + "-" * 70)
    print("   FASE 3: POST-SUENO - Verificando mejora")
    print("-" * 70)

    consciousness_after = []

    for step in range(50):
        # Mismos tipos de estimulos
        if step < 15:
            stimulus = torch.tensor([0.1, 0.7, 0.1, 0.1])
        elif step < 30:
            stimulus = torch.tensor([0.7, 0.1, 0.1, 0.1])
        else:
            stimulus = torch.rand(4)

        result = system.step(stimulus, auto_dream=False)
        consciousness_after.append(result['consciousness'])

    print(f"\n  Consciencia promedio (post-sueno): {np.mean(consciousness_after):.2%}")

    # Comparacion
    mejora = np.mean(consciousness_after) - np.mean(consciousness_before)
    print(f"\n  Mejora por consolidacion: {mejora:+.2%}")

    # Resumen final
    print("\n" + "=" * 70)
    print("   RESUMEN: EFECTO DEL SUENO")
    print("=" * 70)
    print(f"""
  ANTES del sueno:
    - Consciencia promedio: {np.mean(consciousness_before):.2%}
    - Memorias acumuladas: {len(system.system.attention.memory_buffer)}

  DURANTE el sueno:
    - Memorias replayeadas: {report.memories_replayed}
    - Eventos de aprendizaje: {report.learning_events}
    - Reduccion de loss: {report.loss_reduction:+.4f}

  DESPUES del sueno:
    - Consciencia promedio: {np.mean(consciousness_after):.2%}
    - Mejora total: {mejora:+.2%}

  ASOCIACIONES APRENDIDAS:
""")
    for ctx, (arch, strength) in report.associations_learned.items():
        print(f"    {ctx:12} -> {arch:8} (fuerza: {strength:.2f})")

    print("\n" + "=" * 70)
    print("   COMO FUNCIONA LA CONSOLIDACION")
    print("=" * 70)
    print("""
  1. SELECCION DE MEMORIAS
     - Se seleccionan memorias por: sorpresa, atencion, error, recencia
     - Memorias importantes tienen mayor probabilidad de replay

  2. REPLAY DURANTE SUENO
     - Cada memoria se re-procesa por el sistema
     - Los errores de prediccion guian el aprendizaje
     - Las asociaciones se refuerzan (Hebbian)

  3. CONSOLIDACION
     - Memorias con bajo loss se marcan como consolidadas
     - Las redes de atencion mejoran sus predicciones
     - El contexto se asocia correctamente con arquetipos

  4. RESULTADO
     - Mejor prediccion de estimulos futuros
     - Respuestas mas apropiadas al contexto
     - Consciencia mas estable y elevada
""")
    print("=" * 70 + "\n")

    return system, report

if __name__ == "__main__":
    demo_dream_consolidation()
