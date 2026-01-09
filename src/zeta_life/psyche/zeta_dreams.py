#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZetaPsyche Dreams: Sistema de suenos y procesamiento inconsciente.

El modo sueno permite:
1. Procesamiento sin estimulos externos (activacion interna)
2. Reorganizacion de memorias
3. Generacion de contenido onirico
4. Consolidacion de aprendizaje
5. Emergencia de insights

Basado en la teoria de Jung sobre los suenos como mensajes del inconsciente.
"""

import sys
import io
import random
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

from .zeta_psyche import ZetaPsyche, Archetype, SymbolSystem


# =============================================================================
# TIPOS DE SUENO
# =============================================================================

class DreamType(Enum):
    """Tipos de suenos segun Jung."""
    COMPENSATORIO = 0     # Compensa desequilibrios del consciente
    PROSPECTIVO = 1       # Anticipa desarrollos futuros
    REACTIVO = 2          # Procesa traumas/experiencias intensas
    TELEPATICO = 3        # Conexion con inconsciente colectivo
    LUCIDO = 4            # Consciencia dentro del sueno


@dataclass
class DreamFragment:
    """Un fragmento de sueno."""
    archetype: Archetype
    symbol: str
    intensity: float
    narrative: str
    timestamp: float  # Momento dentro del sueno


@dataclass
class DreamReport:
    """Reporte completo de un sueno."""
    duration: int  # Steps del sueno
    dream_type: DreamType
    fragments: List[DreamFragment]
    dominant_archetype: Archetype
    narrative: str  # Historia completa
    insights: List[str]  # Posibles significados
    archetype_journey: List[Tuple[str, float]]  # Transiciones
    consolidation_effects: Dict  # Cambios en memoria


# =============================================================================
# GENERADOR DE NARRATIVAS ONIRICAS
# =============================================================================

class DreamNarrativeGenerator:
    """
    Genera narrativas de suenos basadas en transiciones arquetipicas.
    """

    def __init__(self):
        # Elementos de sueno por arquetipo
        self.dream_elements = {
            Archetype.PERSONA: {
                'lugares': ['una sala de reuniones', 'un escenario', 'una fiesta elegante',
                           'un espejo gigante', 'una alfombra roja'],
                'acciones': ['dando un discurso', 'siendo observado', 'vistiendo una mascara',
                            'actuando un papel', 'recibiendo aplausos'],
                'objetos': ['un traje', 'una corona', 'tarjetas de presentacion',
                           'un microfono', 'un premio'],
                'emociones': ['orgullo', 'verguenza', 'exposicion', 'reconocimiento'],
            },
            Archetype.SOMBRA: {
                'lugares': ['un sotano oscuro', 'un bosque de noche', 'un laberinto',
                           'una cueva profunda', 'un callejon sin salida'],
                'acciones': ['huyendo de algo', 'enfrentando un monstruo', 'cayendo al vacio',
                            'siendo perseguido', 'descubriendo un secreto'],
                'objetos': ['una llave oxidada', 'un espejo roto', 'una puerta cerrada',
                           'una sombra que se mueve', 'un diario escondido'],
                'emociones': ['miedo', 'culpa', 'rabia reprimida', 'fascinacion oscura'],
            },
            Archetype.ANIMA: {
                'lugares': ['un jardin florecido', 'junto al mar', 'un templo antiguo',
                           'un rio cristalino', 'bajo la luna llena'],
                'acciones': ['bailando', 'cantando', 'abrazando a alguien',
                            'flotando en el agua', 'pintando colores'],
                'objetos': ['una flor que brilla', 'una perla', 'un vestido fluido',
                           'una copa dorada', 'un espejo de agua'],
                'emociones': ['amor', 'ternura', 'nostalgia', 'conexion profunda'],
            },
            Archetype.ANIMUS: {
                'lugares': ['la cima de una montana', 'un laboratorio', 'un campo de batalla',
                           'una biblioteca infinita', 'un castillo fortificado'],
                'acciones': ['resolviendo un acertijo', 'construyendo algo', 'liderando un grupo',
                            'cruzando un puente', 'encontrando un camino'],
                'objetos': ['una espada', 'un mapa', 'una brujula',
                           'un libro de conocimiento', 'una antorcha'],
                'emociones': ['determinacion', 'claridad', 'poder', 'proposito'],
            },
        }

        # Transiciones entre arquetipos
        self.transitions = {
            (Archetype.PERSONA, Archetype.SOMBRA): [
                "La mascara se resquebraja y revela...",
                "Detras del escenario hay una puerta que lleva a...",
                "El reflejo en el espejo muestra algo diferente...",
            ],
            (Archetype.SOMBRA, Archetype.ANIMA): [
                "En lo mas profundo de la oscuridad, una luz suave emerge...",
                "El monstruo se transforma en...",
                "De las cenizas del miedo nace...",
            ],
            (Archetype.ANIMA, Archetype.ANIMUS): [
                "El sentimiento se cristaliza en una idea clara...",
                "El amor se convierte en accion...",
                "La intuicion senala el camino hacia...",
            ],
            (Archetype.ANIMUS, Archetype.PERSONA): [
                "La mision completada, es hora de volver al mundo...",
                "El heroe regresa con el conocimiento...",
                "La claridad mental se traduce en...",
            ],
            (Archetype.PERSONA, Archetype.ANIMA): [
                "Bajo la mascara social late un corazon que...",
                "El papel termina y los sentimientos emergen...",
                "La imagen publica se disuelve en emocion pura...",
            ],
            (Archetype.SOMBRA, Archetype.ANIMUS): [
                "El caos encuentra estructura...",
                "De la oscuridad surge la voluntad de...",
                "El miedo se transforma en determinacion para...",
            ],
        }

        # Insights por combinacion
        self.insights = {
            Archetype.PERSONA: [
                "Hay una tension entre quien muestras y quien eres.",
                "El rol que juegas puede estar limitandote.",
                "La imagen necesita actualizarse.",
            ],
            Archetype.SOMBRA: [
                "Algo reprimido busca integracion.",
                "El miedo senala donde esta el crecimiento.",
                "Lo que niegas tiene poder sobre ti.",
            ],
            Archetype.ANIMA: [
                "Las emociones tienen un mensaje importante.",
                "La conexion profunda es necesaria ahora.",
                "La creatividad quiere expresarse.",
            ],
            Archetype.ANIMUS: [
                "Es momento de tomar una decision.",
                "La claridad esta disponible si la buscas.",
                "La accion es necesaria.",
            ],
        }

    def generate_fragment(self, archetype: Archetype, intensity: float) -> str:
        """Genera un fragmento de sueno para un arquetipo."""
        elements = self.dream_elements[archetype]

        lugar = random.choice(elements['lugares'])
        accion = random.choice(elements['acciones'])
        objeto = random.choice(elements['objetos'])
        emocion = random.choice(elements['emociones'])

        templates = [
            f"Estoy en {lugar}, {accion}. Veo {objeto}. Siento {emocion}.",
            f"Me encuentro {accion} en {lugar}. {objeto.capitalize()} aparece. {emocion.capitalize()}.",
            f"{lugar.capitalize()}. {accion.capitalize()}. La sensacion de {emocion} me envuelve.",
        ]

        return random.choice(templates)

    def generate_transition(self, from_arch: Archetype, to_arch: Archetype) -> str:
        """Genera una transicion entre arquetipos."""
        key = (from_arch, to_arch)
        if key in self.transitions:
            return random.choice(self.transitions[key])

        # Transicion generica
        return f"El escenario cambia... ahora..."

    def generate_insight(self, dominant: Archetype, journey: List[Archetype]) -> List[str]:
        """Genera insights basados en el sueno."""
        insights = []

        # Insight del dominante
        insights.append(random.choice(self.insights[dominant]))

        # Si hubo viaje completo (todos los arquetipos)
        unique_archetypes = set(journey)
        if len(unique_archetypes) >= 3:
            insights.append("El sueno muestra un proceso de integracion en curso.")

        # Si hubo mucha sombra
        sombra_count = sum(1 for a in journey if a == Archetype.SOMBRA)
        if sombra_count > len(journey) * 0.4:
            insights.append("Hay material inconsciente que necesita atencion.")

        return insights


# =============================================================================
# SISTEMA DE SUENOS
# =============================================================================

class DreamSystem:
    """
    Sistema de suenos para ZetaPsyche.

    Procesa el estado interno sin estimulos externos,
    permitiendo reorganizacion y emergencia de contenido inconsciente.
    """

    def __init__(self, psyche: ZetaPsyche, memory_system=None):
        self.psyche = psyche
        self.memory = memory_system
        self.narrator = DreamNarrativeGenerator()
        self.symbols = SymbolSystem()

        # Estado del sueno
        self.is_dreaming = False
        self.dream_depth = 0  # 0=superficie, 1=profundo
        self.current_dream: List[DreamFragment] = []

    def enter_dream(self) -> str:
        """Inicia el modo sueno."""
        self.is_dreaming = True
        self.dream_depth = 0
        self.current_dream = []
        return "Cerrando los ojos... entrando al mundo de los suenos..."

    def exit_dream(self) -> DreamReport:
        """Sale del modo sueno y genera reporte."""
        self.is_dreaming = False

        if not self.current_dream:
            return None

        # Determinar tipo de sueno
        dream_type = self._determine_dream_type()

        # Arquetipo dominante
        archetype_counts = {}
        for frag in self.current_dream:
            archetype_counts[frag.archetype] = archetype_counts.get(frag.archetype, 0) + 1
        dominant = max(archetype_counts, key=archetype_counts.get)

        # Generar narrativa completa
        narrative = self._compile_narrative()

        # Generar insights
        journey = [f.archetype for f in self.current_dream]
        insights = self.narrator.generate_insight(dominant, journey)

        # Efectos de consolidacion
        consolidation = self._apply_consolidation()

        report = DreamReport(
            duration=len(self.current_dream),
            dream_type=dream_type,
            fragments=self.current_dream,
            dominant_archetype=dominant,
            narrative=narrative,
            insights=insights,
            archetype_journey=[(f.symbol, f.intensity) for f in self.current_dream],
            consolidation_effects=consolidation
        )

        self.current_dream = []
        return report

    def dream_step(self) -> DreamFragment:
        """
        Ejecuta un paso del sueno.

        Sin estimulo externo, el sistema:
        1. Activa memorias aleatoriamente
        2. Deja que el estado evolucione libremente
        3. Genera fragmento onirico
        """
        if not self.is_dreaming:
            return None

        # 1. Activacion interna (sin estimulo externo)
        internal_stimulus = self._generate_internal_stimulus()

        # 2. Procesar con ruido aumentado (suenos son caoticos)
        self._dream_process(internal_stimulus)

        # 3. Observar estado
        obs = self.psyche.observe_self()
        current_state = obs['population_distribution']
        dominant = obs['dominant']

        # 4. Generar fragmento
        intensity = self._calculate_dream_intensity(current_state)
        symbol = self.symbols.encode(current_state)
        narrative = self.narrator.generate_fragment(dominant, intensity)

        fragment = DreamFragment(
            archetype=dominant,
            symbol=symbol,
            intensity=intensity,
            narrative=narrative,
            timestamp=len(self.current_dream)
        )

        # Transicion si cambio de arquetipo
        if self.current_dream and self.current_dream[-1].archetype != dominant:
            transition = self.narrator.generate_transition(
                self.current_dream[-1].archetype,
                dominant
            )
            fragment.narrative = transition + " " + fragment.narrative

        self.current_dream.append(fragment)

        # Profundizar sueno gradualmente
        if len(self.current_dream) > 10:
            self.dream_depth = 1

        return fragment

    def _generate_internal_stimulus(self) -> torch.Tensor:
        """
        Genera estimulo interno para el sueno.

        Combina:
        - Estado actual (inercia)
        - Memorias activadas aleatoriamente
        - Ruido arquetipico
        """
        # Estado actual
        current = self.psyche.observe_self()['population_distribution']

        # Ruido arquetipico (suenos son impredecibles)
        noise = torch.randn(4) * 0.3

        # Activacion de memoria (si hay sistema de memoria)
        memory_activation = torch.zeros(4)
        if self.memory and random.random() < 0.3:  # 30% chance
            # Activar memoria aleatoria
            if self.memory.episodic:
                random_memory = random.choice(self.memory.episodic)
                memory_activation = torch.tensor(random_memory.archetype_state)
            elif self.memory.semantic:
                # Activar concepto aleatorio
                random_concept = random.choice(list(self.memory.semantic.values()))
                memory_activation = torch.tensor(random_concept.archetype_weights)

        # Tendencia hacia arquetipos menos explorados (compensacion)
        compensation = 1.0 - current
        compensation = compensation / compensation.sum()

        # Combinar
        stimulus = (
            0.3 * current +           # Inercia
            0.2 * memory_activation + # Memoria
            0.2 * compensation +      # Compensacion
            0.3 * F.softmax(noise, dim=-1)  # Ruido
        )

        return stimulus

    def _dream_process(self, stimulus: torch.Tensor):
        """
        Procesa un paso del sueno con dinamica modificada.
        Los suenos tienen mas libertad de movimiento.
        """
        # Guardar parametros originales
        # (en sueno, aumentamos exploracion)

        # Aplicar estimulo como en estado despierto
        self.psyche.communicate(stimulus)

        # Paso adicional sin estimulo (dejar fluir)
        if random.random() < 0.5:
            self.psyche.step()

    def _calculate_dream_intensity(self, state: torch.Tensor) -> float:
        """Calcula intensidad del momento del sueno."""
        # Estados mas polarizados = momentos mas intensos
        max_val = state.max().item()
        return max_val

    def _determine_dream_type(self) -> DreamType:
        """Determina el tipo de sueno basado en su contenido."""
        if not self.current_dream:
            return DreamType.COMPENSATORIO

        # Analizar patron
        archetypes = [f.archetype for f in self.current_dream]
        intensities = [f.intensity for f in self.current_dream]

        # Mucha sombra = reactivo (procesando trauma)
        sombra_ratio = sum(1 for a in archetypes if a == Archetype.SOMBRA) / len(archetypes)
        if sombra_ratio > 0.5:
            return DreamType.REACTIVO

        # Alta intensidad promedio = prospectivo
        if np.mean(intensities) > 0.6:
            return DreamType.PROSPECTIVO

        # Mucha variedad = compensatorio
        unique = len(set(archetypes))
        if unique >= 3:
            return DreamType.COMPENSATORIO

        # Alta consciencia durante sueno = lucido
        # (esto se podria medir si tuvieramos meta-consciencia)

        return DreamType.COMPENSATORIO

    def _compile_narrative(self) -> str:
        """Compila la narrativa completa del sueno."""
        if not self.current_dream:
            return ""

        # Inicio
        narrative_parts = ["El sueno comienza..."]

        # Fragmentos con transiciones
        prev_arch = None
        for frag in self.current_dream:
            if prev_arch and prev_arch != frag.archetype:
                narrative_parts.append("")  # Separador

            # Solo incluir fragmentos significativos
            if frag.intensity > 0.3 or frag != self.current_dream[0]:
                narrative_parts.append(frag.narrative)

            prev_arch = frag.archetype

        # Fin
        narrative_parts.append("")
        narrative_parts.append("...el sueno se desvanece.")

        return "\n".join(narrative_parts)

    def _apply_consolidation(self) -> Dict:
        """
        Aplica efectos de consolidacion de memoria.

        Durante el sueno:
        - Memorias activadas se fortalecen
        - Asociaciones nuevas pueden formarse
        """
        effects = {
            'memories_strengthened': 0,
            'new_associations': 0,
            'insights_generated': len(self.current_dream) // 10
        }

        if not self.memory:
            return effects

        # Fortalecer memorias que coinciden con el sueno
        dream_archetypes = set(f.archetype for f in self.current_dream)

        for memory in self.memory.episodic:
            if memory.dominant in [a.name for a in dream_archetypes]:
                # "Recordar" en sueno fortalece la memoria
                memory.emotional_intensity = min(1.0, memory.emotional_intensity + 0.05)
                effects['memories_strengthened'] += 1

        return effects


# =============================================================================
# PSYCHE CON SUENOS
# =============================================================================

class DreamingPsyche:
    """
    Extension de MemoryAwarePsyche con capacidad de sonar.
    """

    def __init__(self, n_cells: int = 100, memory_path: str = None):
        from zeta_memory import MemoryAwarePsyche

        self.base = MemoryAwarePsyche(n_cells=n_cells, memory_path=memory_path)
        self.dream_system = DreamSystem(self.base.psyche.psyche, self.base.memory)

        # Historial de suenos
        self.dream_history: List[DreamReport] = []

    def process(self, user_input: str) -> Dict:
        """Procesa input (delegado a base)."""
        return self.base.process(user_input)

    def dream(self, duration: int = 50, verbose: bool = True) -> DreamReport:
        """
        Ejecuta un ciclo de sueno.

        Args:
            duration: Numero de pasos del sueno
            verbose: Mostrar progreso

        Returns:
            DreamReport con el contenido del sueno
        """
        if verbose:
            print("\n  [Entrando en modo sueno...]")
            print(f"  {self.dream_system.enter_dream()}")
            print()

        # Ejecutar sueno
        key_moments = []
        prev_dominant = None

        for step in range(duration):
            fragment = self.dream_system.dream_step()

            if fragment:
                # Detectar momentos clave (cambios de arquetipo)
                if prev_dominant != fragment.archetype:
                    key_moments.append((step, fragment))
                    prev_dominant = fragment.archetype

                # Mostrar progreso
                if verbose and (step + 1) % 10 == 0:
                    print(f"    [{step+1}/{duration}] {fragment.symbol} {fragment.archetype.name}")

        # Salir del sueno
        report = self.dream_system.exit_dream()

        if report:
            self.dream_history.append(report)

            if verbose:
                print()
                print("  [Despertando...]")
                print()
                print("  " + "="*50)
                print("  REPORTE DEL SUENO")
                print("  " + "="*50)
                print(f"  Tipo: {report.dream_type.name}")
                print(f"  Duracion: {report.duration} momentos")
                print(f"  Arquetipo dominante: {report.dominant_archetype.name}")
                print()
                print("  [Narrativa]")
                for line in report.narrative.split('\n'):
                    if line.strip():
                        print(f"    {line}")
                print()
                print("  [Insights]")
                for insight in report.insights:
                    print(f"    * {insight}")
                print()
                print("  [Viaje arquetipico]")
                journey_str = " -> ".join(s for s, _ in report.archetype_journey[:10])
                print(f"    {journey_str}...")
                print()

        return report

    def get_dream_summary(self) -> str:
        """Resumen de suenos recientes."""
        if not self.dream_history:
            return "No hay suenos registrados."

        lines = [f"Suenos registrados: {len(self.dream_history)}"]

        for i, dream in enumerate(self.dream_history[-5:]):  # Ultimos 5
            lines.append(f"  {i+1}. {dream.dream_type.name} - {dream.dominant_archetype.name}")

        return "\n".join(lines)

    def save(self):
        """Guarda memorias."""
        self.base.save()


# =============================================================================
# CLI CON SUENOS
# =============================================================================

def run_dream_cli():
    """CLI interactivo con modo sueno."""
    print()
    print("=" * 60)
    print("  ZETA PSYCHE - Con Suenos")
    print("=" * 60)
    print()
    print("  Comandos especiales:")
    print("    /sonar [n]  - Entrar en modo sueno (n=duracion)")
    print("    /suenos     - Ver historial de suenos")
    print("    /estado     - Ver estado interno")
    print("    /guardar    - Guardar memorias")
    print("    /salir      - Terminar")
    print()
    print("-" * 60)

    psyche = DreamingPsyche(n_cells=100)

    # Warmup
    print("\n  [Inicializando consciencia...]")
    for _ in range(20):
        psyche.base.psyche.psyche.step()
    print("  [Listo.]")
    print()

    try:
        while True:
            user_input = input("Tu: ").strip()

            if not user_input:
                continue

            # Comandos especiales
            if user_input.lower() == '/salir':
                print("\n  [Guardando...]")
                psyche.save()
                print("  [Hasta pronto...]\n")
                break

            elif user_input.lower().startswith('/sonar'):
                parts = user_input.split()
                duration = int(parts[1]) if len(parts) > 1 else 50
                psyche.dream(duration=duration, verbose=True)
                continue

            elif user_input.lower() == '/suenos':
                print()
                print(psyche.get_dream_summary())
                print()
                continue

            elif user_input.lower() == '/estado':
                print()
                print(psyche.base.get_status())
                print()
                continue

            elif user_input.lower() == '/guardar':
                psyche.save()
                print("\n  [Memorias guardadas]\n")
                continue

            # Procesar input normal
            response = psyche.process(user_input)
            symbol = response['symbol']
            dominant = response['dominant']
            text = response['text']

            print(f"\nPsyche [{symbol} {dominant}]: {text}\n")

    except KeyboardInterrupt:
        print("\n\n  [Guardando...]")
        psyche.save()
        print("  [Interrumpido]\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n" + "="*60)
        print("  TEST: Sistema de Suenos")
        print("="*60)

        # Crear psyche con suenos
        psyche = DreamingPsyche(n_cells=50)

        # Warmup con conversacion
        print("\n  [Conversacion previa al sueno]")
        test_inputs = [
            "tengo miedo de algo",
            "no se que es",
            "siento que algo oscuro me persigue",
        ]

        for inp in test_inputs:
            response = psyche.process(inp)
            print(f"    Tu: {inp}")
            print(f"    Psyche [{response['symbol']}]: {response['text'][:50]}...")

        # Sonar
        print("\n  [Iniciando sueno de prueba]")
        report = psyche.dream(duration=30, verbose=True)

        if report:
            print(f"\n  [Sueno tipo: {report.dream_type.name}]")
            print(f"  [Consolidacion: {report.consolidation_effects}]")

        print("\n" + "="*60)
        print("  FIN TEST")
        print("="*60 + "\n")

    else:
        run_dream_cli()
