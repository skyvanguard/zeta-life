#!/usr/bin/env python
"""
ZetaPsyche Memory: Sistema de memoria a largo plazo.

Implementa tres tipos de memoria inspirados en la psicologia cognitiva:
1. Episodica: Eventos especificos con contexto emocional
2. Semantica: Conocimiento y asociaciones aprendidas
3. Procedural: Patrones de respuesta

La memoria usa el estado arquetipico como "color emocional" para
facilitar el recuerdo por similitud afectiva.
"""

import io
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
    except (AttributeError, io.UnsupportedOperation):
        pass

# Use Vertex from local module to avoid circular import with psyche
# Vertex has backwards-compatible aliases: PERSONA, SOMBRA, ANIMA, ANIMUS
from .vertex import Vertex as Archetype

# =============================================================================
# ESTRUCTURAS DE MEMORIA
# =============================================================================

@dataclass
class EpisodicMemory:
    """
    Memoria episodica: un evento especifico en el tiempo.

    Almacena:
    - El input del usuario
    - La respuesta generada
    - El estado arquetipico en ese momento
    - La intensidad emocional
    - Timestamp
    """
    timestamp: str
    user_input: str
    response: str
    archetype_state: list[float]  # [PERSONA, SOMBRA, ANIMA, ANIMUS]
    dominant: str
    emotional_intensity: float  # 0-1, que tan "intenso" fue el momento
    consciousness_level: float
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'EpisodicMemory':
        return cls(**data)

    def similarity_to(self, state: torch.Tensor) -> float:
        """Calcula similitud con un estado arquetipico."""
        my_state = torch.tensor(self.archetype_state)
        return F.cosine_similarity(my_state.unsqueeze(0), state.unsqueeze(0)).item()

@dataclass
class SemanticMemory:
    """
    Memoria semantica: conocimiento aprendido.

    Almacena asociaciones entre conceptos y estados arquetipicos.
    Por ejemplo: "miedo" -> SOMBRA con fuerza 0.8
    """
    concept: str
    archetype_weights: list[float]
    strength: float  # 0-1, que tan fuerte es la asociacion
    frequency: int  # Cuantas veces se ha reforzado
    last_accessed: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'SemanticMemory':
        return cls(**data)

    def reinforce(self, new_weights: list[float], alpha: float = 0.3) -> None:
        """Refuerza la asociacion con nuevos pesos."""
        for i in range(4):
            self.archetype_weights[i] = (
                (1 - alpha) * self.archetype_weights[i] +
                alpha * new_weights[i]
            )
        self.frequency += 1
        self.strength = min(1.0, self.strength + 0.1)
        self.last_accessed = datetime.now().isoformat()

@dataclass
class ProceduralMemory:
    """
    Memoria procedural: patrones de respuesta aprendidos.

    Almacena secuencias de estados que llevaron a resultados positivos.
    """
    trigger_pattern: list[str]  # Palabras clave que activan este patron
    response_style: str  # Tipo de respuesta preferida
    archetype_sequence: list[list[float]]  # Secuencia de estados
    success_rate: float  # Que tan exitoso ha sido este patron
    usage_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ProceduralMemory':
        return cls(**data)

# =============================================================================
# SISTEMA DE MEMORIA COMPLETO
# =============================================================================

class ZetaMemorySystem:
    """
    Sistema de memoria a largo plazo para ZetaPsyche.

    Caracteristicas:
    - Almacenamiento persistente (JSON)
    - Recall por similitud emocional
    - Consolidacion automatica
    - Olvido gradual de memorias no accesadas
    """

    def __init__(self, memory_path: str | None = None) -> None:
        self.memory_path: str = memory_path or "zeta_memories.json"

        # Memorias
        self.episodic: list[EpisodicMemory] = []
        self.semantic: dict[str, SemanticMemory] = {}
        self.procedural: list[ProceduralMemory] = []

        # Parametros
        self.max_episodic = 1000  # Maximo de memorias episodicas
        self.consolidation_threshold = 0.3  # Intensidad minima para consolidar (bajo para recordar mas)
        self.forgetting_rate = 0.01  # Tasa de olvido por sesion

        # Buffer de corto plazo (se consolida periodicamente)
        self.short_term_buffer: list[EpisodicMemory] = []
        self.buffer_size = 10

        # Cargar memorias existentes
        self._load_memories()

    def _load_memories(self) -> None:
        """Carga memorias desde archivo."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, encoding='utf-8') as f:
                    data = json.load(f)

                self.episodic = [
                    EpisodicMemory.from_dict(m)
                    for m in data.get('episodic', [])
                ]
                self.semantic = {
                    k: SemanticMemory.from_dict(v)
                    for k, v in data.get('semantic', {}).items()
                }
                self.procedural = [
                    ProceduralMemory.from_dict(m)
                    for m in data.get('procedural', [])
                ]

                print(f"  [Memorias cargadas: {len(self.episodic)} episodicas, "
                      f"{len(self.semantic)} semanticas]")
            except Exception as e:
                print(f"  [Error cargando memorias: {e}]")

    def save_memories(self) -> None:
        """Guarda memorias a archivo."""
        data = {
            'episodic': [m.to_dict() for m in self.episodic],
            'semantic': {k: v.to_dict() for k, v in self.semantic.items()},
            'procedural': [m.to_dict() for m in self.procedural],
            'metadata': {
                'last_saved': datetime.now().isoformat(),
                'version': '1.0'
            }
        }

        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # ALMACENAMIENTO
    # =========================================================================

    def store_episode(
        self,
        user_input: str,
        response: str,
        archetype_state: torch.Tensor,
        dominant: Archetype,
        consciousness: float,
        tags: list[str] | None = None
    ) -> None:
        """
        Almacena un episodio en memoria de corto plazo.
        Se consolidara si es lo suficientemente intenso.
        """
        # Calcular intensidad emocional
        state_tensor = archetype_state if isinstance(archetype_state, torch.Tensor) else torch.tensor(archetype_state)
        intensity = self._calculate_intensity(state_tensor)

        memory = EpisodicMemory(
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            response=response,
            archetype_state=state_tensor.tolist(),
            dominant=dominant.name if isinstance(dominant, Archetype) else dominant,
            emotional_intensity=intensity,
            consciousness_level=consciousness,
            tags=tags or []
        )

        # Agregar a buffer de corto plazo
        self.short_term_buffer.append(memory)

        # Consolidar si el buffer esta lleno
        if len(self.short_term_buffer) >= self.buffer_size:
            self._consolidate_buffer()

        # Aprender asociaciones semanticas
        self._learn_semantic(user_input, state_tensor)

    def _calculate_intensity(self, state: torch.Tensor) -> float:
        """
        Calcula la intensidad emocional de un estado.
        Estados mas polarizados = mas intensos.
        """
        # Entropia inversa - estados concentrados son mas intensos
        probs = F.softmax(state, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        max_entropy = np.log(4)

        # Intensidad = 1 - entropia normalizada
        intensity = 1.0 - float(entropy / max_entropy)

        return float(intensity)

    def _learn_semantic(self, text: str, state: torch.Tensor) -> None:
        """Aprende asociaciones semanticas del texto."""
        words = text.lower().split()

        for word in words:
            # Filtrar palabras cortas
            if len(word) < 3:
                continue

            # Limpiar puntuacion
            word = ''.join(c for c in word if c.isalnum())

            if not word:
                continue

            if word in self.semantic:
                # Reforzar asociacion existente
                self.semantic[word].reinforce(state.tolist())
            else:
                # Crear nueva asociacion
                self.semantic[word] = SemanticMemory(
                    concept=word,
                    archetype_weights=state.tolist(),
                    strength=0.3,  # Inicial debil
                    frequency=1,
                    last_accessed=datetime.now().isoformat()
                )

    def _consolidate_buffer(self) -> None:
        """
        Consolida memorias del buffer de corto plazo a largo plazo.
        Solo las memorias intensas se consolidan.
        """
        for memory in self.short_term_buffer:
            if memory.emotional_intensity >= self.consolidation_threshold:
                self.episodic.append(memory)

        # Limpiar buffer
        self.short_term_buffer = []

        # Aplicar limite de memorias
        if len(self.episodic) > self.max_episodic:
            # Mantener las mas intensas y recientes
            self.episodic.sort(
                key=lambda m: (m.emotional_intensity, m.timestamp),
                reverse=True
            )
            self.episodic = self.episodic[:self.max_episodic]

    # =========================================================================
    # RECUERDO (RECALL)
    # =========================================================================

    def recall_by_state(
        self,
        state: torch.Tensor,
        n: int = 5,
        min_similarity: float = 0.5
    ) -> list[EpisodicMemory]:
        """
        Recuerda episodios similares al estado actual.
        Busqueda por similitud emocional.
        """
        if not self.episodic:
            return []

        # Calcular similitud con cada memoria
        memories_with_sim = [
            (memory, memory.similarity_to(state))
            for memory in self.episodic
        ]

        # Filtrar por similitud minima
        memories_with_sim = [
            (m, s) for m, s in memories_with_sim if s >= min_similarity
        ]

        # Ordenar por similitud
        memories_with_sim.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in memories_with_sim[:n]]

    def recall_by_concept(self, concept: str) -> SemanticMemory | None:
        """Recuerda asociacion semantica de un concepto."""
        concept = concept.lower()
        return self.semantic.get(concept)

    def recall_by_keywords(
        self,
        text: str,
        n: int = 3
    ) -> list[EpisodicMemory]:
        """
        Recuerda episodios que contengan palabras similares.
        """
        if not self.episodic:
            return []

        words = set(text.lower().split())

        # Puntuar memorias por palabras en comun
        scored_memories = []
        for memory in self.episodic:
            memory_words = set(memory.user_input.lower().split())
            common = len(words & memory_words)
            if common > 0:
                score = common / len(words | memory_words)
                scored_memories.append((memory, score))

        # Ordenar por puntaje
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored_memories[:n]]

    def get_semantic_modulation(self, text: str) -> torch.Tensor:
        """
        Obtiene modulacion semantica basada en memorias aprendidas.

        Combina las asociaciones aprendidas de las palabras en el texto
        para modular el estimulo.
        """
        words = text.lower().split()

        modulations = []
        weights = []

        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if word in self.semantic:
                mem = self.semantic[word]
                modulations.append(torch.tensor(mem.archetype_weights))
                weights.append(mem.strength * mem.frequency)

        if not modulations:
            return torch.tensor([0.25, 0.25, 0.25, 0.25])

        # Promedio ponderado
        modulations_stacked = torch.stack(modulations)
        weights_tensor = torch.tensor(weights)
        weights_tensor = weights_tensor / weights_tensor.sum()

        result = (modulations_stacked * weights_tensor.unsqueeze(1)).sum(dim=0)
        return F.softmax(result, dim=-1)

    # =========================================================================
    # INTROSPECCION
    # =========================================================================

    def get_memory_summary(self) -> dict:
        """Resumen del estado de la memoria."""
        archetype_counts = {a.name: 0 for a in Archetype}

        for memory in self.episodic:
            archetype_counts[memory.dominant] += 1

        return {
            'total_episodic': len(self.episodic),
            'total_semantic': len(self.semantic),
            'total_procedural': len(self.procedural),
            'buffer_size': len(self.short_term_buffer),
            'dominant_memories': archetype_counts,
            'avg_intensity': np.mean([m.emotional_intensity for m in self.episodic]) if self.episodic else 0,
        }

    def get_recent_context(self, n: int = 5) -> list[dict]:
        """Obtiene contexto de conversaciones recientes."""
        recent = sorted(
            self.episodic + self.short_term_buffer,
            key=lambda m: m.timestamp,
            reverse=True
        )[:n]

        return [
            {
                'user': m.user_input,
                'response': m.response,
                'dominant': m.dominant,
                'intensity': m.emotional_intensity
            }
            for m in recent
        ]

    def format_memories_for_context(self, memories: list[EpisodicMemory]) -> str:
        """Formatea memorias como contexto legible."""
        if not memories:
            return ""

        lines = ["[Recuerdos relevantes:]"]
        for m in memories:
            lines.append(f"  - \"{m.user_input}\" -> {m.dominant} (intensidad: {m.emotional_intensity:.2f})")

        return "\n".join(lines)

# =============================================================================
# PSYCHE CON MEMORIA
# =============================================================================

class MemoryAwarePsyche:
    """
    Extension de ConversationalPsyche con memoria a largo plazo.
    """

    def __init__(self, n_cells: int = 100, memory_path: str | None = None) -> None:
        # Importar aqui para evitar circular import
        from zeta_psyche_voice import ConversationalPsyche

        self.psyche = ConversationalPsyche(n_cells=n_cells)
        self.memory = ZetaMemorySystem(memory_path)

        # Configuracion
        self.use_semantic_modulation = True
        self.recall_similar_memories = True
        self.context_window = 3  # Memorias a considerar

    def process(self, user_input: str) -> dict:
        """
        Procesa input con memoria.

        1. Busca memorias relevantes
        2. Modula el estimulo con conocimiento semantico
        3. Procesa con la psique
        4. Almacena el episodio
        """
        # 1. Buscar memorias similares por palabras clave
        similar_memories = []
        if self.recall_similar_memories:
            similar_memories = self.memory.recall_by_keywords(user_input, n=self.context_window)

        # 2. Obtener modulacion semantica
        semantic_mod = torch.tensor([0.25, 0.25, 0.25, 0.25])
        if self.use_semantic_modulation:
            semantic_mod = self.memory.get_semantic_modulation(user_input)

        # 3. Procesar con psique (la modulacion afecta el estado inicial)
        # Primero aplicar modulacion semantica suave
        if self.use_semantic_modulation:
            self._apply_semantic_modulation(semantic_mod)

        # Procesar normalmente
        response = self.psyche.process(user_input)

        # 4. Almacenar episodio
        obs = self.psyche.psyche.observe_self()
        self.memory.store_episode(
            user_input=user_input,
            response=response['text'],
            archetype_state=obs['population_distribution'],
            dominant=obs['dominant'],
            consciousness=obs['consciousness_index'],
            tags=self._extract_tags(user_input)
        )

        # 5. Enriquecer respuesta con contexto de memoria
        response['memories'] = [
            {'input': m.user_input, 'dominant': m.dominant}
            for m in similar_memories
        ]
        response['semantic_influence'] = semantic_mod.tolist()

        result: dict[Any, Any] = response
        return result

    def _apply_semantic_modulation(self, modulation: torch.Tensor) -> None:
        """Aplica modulacion semantica al estado de la psique."""
        # Aplicar como estimulo suave
        self.psyche.psyche.communicate(modulation)

    def _extract_tags(self, text: str) -> list[str]:
        """Extrae tags del texto para categorizar la memoria."""
        tags = []

        # Detectar emociones
        emotions = {
            'miedo': ['miedo', 'terror', 'asustado', 'temo'],
            'amor': ['amor', 'amo', 'quiero', 'carino'],
            'tristeza': ['triste', 'tristeza', 'llorar', 'dolor'],
            'alegria': ['feliz', 'alegria', 'contento', 'bien'],
            'ira': ['rabia', 'ira', 'enojado', 'furioso'],
        }

        text_lower = text.lower()
        for emotion, keywords in emotions.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(emotion)

        # Detectar temas
        themes = {
            'pregunta': ['?', 'que', 'como', 'por que'],
            'personal': ['yo', 'mi', 'me', 'siento'],
            'reflexion': ['pienso', 'creo', 'quizas', 'tal vez'],
        }

        for theme, keywords in themes.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(theme)

        return tags

    def recall_similar(self, n: int = 5) -> list[dict]:
        """Recuerda memorias similares al estado actual."""
        current_state = self.psyche.psyche.observe_self()['population_distribution']
        memories = self.memory.recall_by_state(current_state, n=n)

        return [
            {
                'input': m.user_input,
                'response': m.response,
                'dominant': m.dominant,
                'similarity': m.similarity_to(current_state)
            }
            for m in memories
        ]

    def get_status(self) -> str:
        """Estado completo incluyendo memoria."""
        psyche_status = self.psyche.get_status_bar()
        memory_summary = self.memory.get_memory_summary()

        lines = [
            psyche_status,
            "",
            "  [MEMORIA]",
            f"  Episodica: {memory_summary['total_episodic']} recuerdos",
            f"  Semantica: {memory_summary['total_semantic']} conceptos",
            f"  Buffer: {memory_summary['buffer_size']}/{self.memory.buffer_size}",
            f"  Intensidad promedio: {memory_summary['avg_intensity']:.2f}",
        ]

        return "\n".join(lines)

    def save(self) -> None:
        """Guarda memorias a disco."""
        self.memory.save_memories()

    def forget_session(self) -> None:
        """Olvida la sesion actual (limpia buffer sin consolidar)."""
        self.memory.short_term_buffer = []

# =============================================================================
# CLI CON MEMORIA
# =============================================================================

def run_memory_cli() -> None:
    """CLI interactivo con memoria a largo plazo."""
    print()
    print("=" * 60)
    print("  ZETA PSYCHE - Con Memoria a Largo Plazo")
    print("=" * 60)
    print()
    print("  Comandos especiales:")
    print("    /estado   - Ver estado interno y memoria")
    print("    /recordar - Ver memorias similares al estado actual")
    print("    /historia - Ver conversacion reciente")
    print("    /guardar  - Guardar memorias a disco")
    print("    /olvidar  - Olvidar sesion actual")
    print("    /reset    - Reiniciar psique (mantiene memoria)")
    print("    /salir    - Terminar (guarda automaticamente)")
    print()
    print("-" * 60)

    psyche = MemoryAwarePsyche(n_cells=100)

    # Warmup
    print("\n  [Inicializando consciencia...]")
    for _ in range(20):
        psyche.psyche.psyche.step()
    print("  [Listo. Memorias cargadas.]")
    print()

    try:
        while True:
            user_input = input("Tu: ").strip()

            if not user_input:
                continue

            # Comandos especiales
            if user_input.lower() == '/salir':
                print("\n  [Guardando memorias...]")
                psyche.save()
                print("  [Hasta pronto...]\n")
                break

            elif user_input.lower() == '/estado':
                print()
                print(psyche.get_status())
                print()
                continue

            elif user_input.lower() == '/recordar':
                print("\n  [Memorias similares al estado actual:]")
                memories = psyche.recall_similar(n=5)
                if memories:
                    for m in memories:
                        print(f"    \"{m['input'][:40]}...\" -> {m['dominant']} "
                              f"(sim: {m['similarity']:.2f})")
                else:
                    print("    (sin memorias similares)")
                print()
                continue

            elif user_input.lower() == '/historia':
                print("\n  [Conversacion reciente:]")
                recent = psyche.memory.get_recent_context(n=5)
                for r in recent:
                    print(f"    Tu: {r['user'][:40]}...")
                    print(f"    Psyche [{r['dominant']}]: {r['response'][:40]}...")
                    print()
                continue

            elif user_input.lower() == '/guardar':
                psyche.save()
                print("\n  [Memorias guardadas]\n")
                continue

            elif user_input.lower() == '/olvidar':
                psyche.forget_session()
                print("\n  [Sesion olvidada (buffer limpiado)]\n")
                continue

            elif user_input.lower() == '/reset':
                from zeta_psyche_voice import ConversationalPsyche
                psyche.psyche = ConversationalPsyche(n_cells=100)
                print("\n  [Psique reiniciada (memoria conservada)]\n")
                continue

            # Procesar input normal
            response = psyche.process(user_input)

            # Mostrar respuesta
            symbol = response['symbol']
            dominant = response['dominant']
            text = response['text']

            # Mostrar si hay memorias relevantes
            if response.get('memories'):
                print("\n  [Recuerdo algo similar...]")

            print(f"\nPsyche [{symbol} {dominant}]: {text}\n")

    except KeyboardInterrupt:
        print("\n\n  [Guardando memorias...]")
        psyche.save()
        print("  [Interrumpido]\n")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n[TEST - Sistema de Memoria]")

        # Test basico
        memory = ZetaMemorySystem("test_memories.json")

        # Almacenar algunos episodios
        test_episodes = [
            ("tengo miedo", "El miedo es un maestro", [0.1, 0.8, 0.05, 0.05], "SOMBRA", 0.9),
            ("te amo", "El amor transforma", [0.1, 0.1, 0.7, 0.1], "ANIMA", 0.85),
            ("necesito pensar", "Analicemos esto", [0.1, 0.1, 0.1, 0.7], "ANIMUS", 0.8),
            ("hola amigo", "Bienvenido", [0.7, 0.1, 0.1, 0.1], "PERSONA", 0.6),
        ]

        # Test directo sin buffer
        memory.consolidation_threshold = 0.0  # Aceptar todo

        for inp, resp, state, dom, consc in test_episodes:
            # Calcular intensidad manualmente
            state_t = torch.tensor(state)
            intensity = memory._calculate_intensity(state_t)
            print(f"  Storing: '{inp}' intensity={intensity:.2f}")

            # Agregar directamente a episodic (bypass buffer para test)
            ep = EpisodicMemory(
                timestamp=datetime.now().isoformat(),
                user_input=inp,
                response=resp,
                archetype_state=state,
                dominant=dom,
                emotional_intensity=intensity,
                consciousness_level=consc,
                tags=[]
            )
            memory.episodic.append(ep)

            # Tambien aprender semantico
            memory._learn_semantic(inp, state_t)

        print(f"\nMemoria almacenada: {len(memory.episodic)} episodios")
        print(f"Conceptos aprendidos: {len(memory.semantic)}")

        # Test recall por estado
        print("\n[Recall por estado SOMBRA:]")
        sombra_state = torch.tensor([0.1, 0.8, 0.05, 0.05])
        similar = memory.recall_by_state(sombra_state, n=3)
        for m in similar:
            print(f"  - \"{m.user_input}\" ({m.dominant})")

        # Test recall por keywords
        print("\n[Recall por keyword 'miedo':]")
        by_kw = memory.recall_by_keywords("tengo miedo", n=3)
        for m in by_kw:
            print(f"  - \"{m.user_input}\" ({m.dominant})")

        # Test modulacion semantica
        print("\n[Modulacion semantica para 'miedo oscuro':]")
        mod = memory.get_semantic_modulation("miedo oscuro")
        print(f"  PERSONA: {mod[0]:.2f}, SOMBRA: {mod[1]:.2f}, "
              f"ANIMA: {mod[2]:.2f}, ANIMUS: {mod[3]:.2f}")

        # Guardar
        memory.save_memories()
        print("\n[Memorias guardadas en test_memories.json]")

        # Limpiar archivo de test
        import os
        os.remove("test_memories.json")
        print("[Archivo de test eliminado]")

        print("\n[FIN TEST]")

    else:
        run_memory_cli()
