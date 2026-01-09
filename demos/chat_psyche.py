#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZetaPsyche Chat v2.0 - CLI Interactivo Avanzado

Un sistema de chat donde ZetaPsyche responde en lenguaje natural
con personalidad arquetipal, mostrando su estado de consciencia
en tiempo real con barras visuales.

Uso:
    python chat_psyche.py [opciones]

Opciones:
    --simple     Modo simple (sin ZetaConsciousSelf)
    --no-bars    Desactivar barras de estado
    --decay      Activar decay emergente
    --reflection Activar loop de auto-reflexion (Strange Loop)
"""

import sys
import os
import argparse

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('')  # Enable ANSI codes
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, Exception):
        pass

import torch
from typing import Dict, List, Optional
from enum import Enum

from zeta_life.psyche import Archetype
from zeta_life.psyche import ArchetypalVoice, EXPANDED_VOCABULARY


# =============================================================================
# UTILIDADES DE DISPLAY
# =============================================================================

class Colors:
    """ANSI color codes para terminal."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colores para arquetipos
    PERSONA = '\033[91m'   # Rojo
    SOMBRA = '\033[95m'    # Magenta/Purpura
    ANIMA = '\033[94m'     # Azul
    ANIMUS = '\033[93m'    # Amarillo/Naranja

    # Otros
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

    @classmethod
    def for_archetype(cls, arch_name: str) -> str:
        return getattr(cls, arch_name.upper(), cls.WHITE)


def create_bar(value: float, width: int = 10, filled: str = '#', empty: str = '.') -> str:
    """Crea una barra de progreso ASCII."""
    filled_count = int(value * width)
    return filled * filled_count + empty * (width - filled_count)


def format_archetype_bars(population: List[float], show_colors: bool = True) -> str:
    """Formatea las barras de estado de arquetipos."""
    names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
    lines = []

    for i, name in enumerate(names):
        val = population[i]
        bar = create_bar(val)
        pct = val * 100

        if show_colors:
            color = Colors.for_archetype(name)
            line = f"  {color}{name:8s}{Colors.RESET} [{bar}] {pct:5.1f}%"
        else:
            line = f"  {name:8s} [{bar}] {pct:5.1f}%"

        lines.append(line)

    return '\n'.join(lines)


def format_consciousness_bar(consciousness: float, luminosity: float) -> str:
    """Formatea la barra de consciencia y luminosidad."""
    c_bar = create_bar(consciousness)
    l_bar = create_bar(luminosity)

    # Simbolo del Self basado en luminosidad
    if luminosity > 0.7:
        self_symbol = "*"  # Self manifestandose
    elif luminosity > 0.4:
        self_symbol = "o"  # Self emergiendo
    else:
        self_symbol = "."  # Self latente

    return (
        f"  Consciencia: [{c_bar}] {consciousness:.3f}\n"
        f"  Luminosidad: [{l_bar}] {luminosity:.3f} Self: {self_symbol}"
    )


# =============================================================================
# CLI AVANZADO
# =============================================================================

class ZetaPsycheCLI:
    """
    CLI interactivo avanzado para ZetaPsyche.

    Caracteristicas:
    - Respuestas en lenguaje natural con personalidad arquetipal
    - Barras de estado visual actualizandose
    - Comandos especiales para introspección
    - Integracion con ZetaConsciousSelf (opcional)
    """

    def __init__(
        self,
        use_conscious_self: bool = True,
        show_bars: bool = True,
        enable_decay: bool = False,
        enable_reflection: bool = False,
        n_cells: int = 100
    ):
        self.show_bars = show_bars
        self.use_conscious_self = use_conscious_self
        self.enable_reflection = enable_reflection
        self.processing_steps = 15
        self.conversation_history = []

        # Vocabulario para convertir texto a estimulo
        self.vocabulary = EXPANDED_VOCABULARY

        # Generador de voz
        self.voice = ArchetypalVoice()

        # Inicializar sistema
        if use_conscious_self:
            try:
                from zeta_conscious_self import ZetaConsciousSelf
                self.system = ZetaConsciousSelf(
                    n_cells=n_cells,
                    enable_decay=enable_decay,
                    enable_self_reflection=enable_reflection,
                    reflection_config={
                        'max_iterations': 4,
                        'convergence_threshold': 0.01,
                        'include_perception': True,
                    }
                )
                self.system_type = 'conscious_self'
                print("  [Usando ZetaConsciousSelf]")
                if enable_reflection:
                    print("  [Loop de auto-reflexion: ACTIVO]")
            except ImportError as e:
                print(f"  [Warning: No se pudo cargar ZetaConsciousSelf: {e}]")
                print("  [Usando ZetaPsyche basico]")
                from zeta_psyche import ZetaPsyche
                self.system = ZetaPsyche(n_cells=n_cells)
                self.system_type = 'psyche'
        else:
            from zeta_psyche import ZetaPsyche
            self.system = ZetaPsyche(n_cells=n_cells)
            self.system_type = 'psyche'
            print("  [Usando ZetaPsyche basico]")

        # Estado actual
        self.last_state = None

    def _text_to_stimulus(self, text: str) -> torch.Tensor:
        """Convierte texto a estimulo arquetipico."""
        text = text.lower()

        # Buscar palabras clave y promediar
        found_weights = []
        for word, weights in self.vocabulary.items():
            if word in text:
                found_weights.append(torch.tensor(weights))

        if found_weights:
            stimulus = torch.stack(found_weights).mean(dim=0)
        else:
            stimulus = torch.tensor([0.25, 0.25, 0.25, 0.25])

        return stimulus

    def _get_state(self) -> Dict:
        """Obtiene el estado actual del sistema."""
        if self.system_type == 'conscious_self':
            # ZetaConsciousSelf tiene observe_self via psyche
            psyche_obs = self.system.psyche.observe_self()

            # Obtener consciencia del sistema
            if hasattr(self.system, 'consciousness_index'):
                consciousness = self.system.consciousness_index.compute_total()
            else:
                consciousness = psyche_obs['consciousness_index']

            # Obtener stage de individuacion
            if hasattr(self.system, 'individuation'):
                stage = self.system.individuation.stage.name
            else:
                stage = 'INCONSCIENTE'

            return {
                'population': psyche_obs['population_distribution'],
                'dominant': psyche_obs['dominant'],
                'blend': psyche_obs['blend'],
                'consciousness': consciousness,
                'stage': stage
            }
        else:
            # ZetaPsyche basico
            obs = self.system.observe_self()
            return {
                'population': obs['population_distribution'],
                'dominant': obs['dominant'],
                'blend': obs['blend'],
                'consciousness': obs['consciousness_index'],
                'stage': 'N/A'
            }

    def _calculate_luminosity(self, population: torch.Tensor) -> float:
        """Calcula luminosidad (integracion del Self)."""
        center = torch.tensor([0.25, 0.25, 0.25, 0.25])
        dist_to_center = torch.sqrt(((population - center) ** 2).sum()).item()
        max_dist = 0.75
        return 1.0 - min(dist_to_center / max_dist, 1.0)

    def process_input(self, user_input: str) -> Dict:
        """Procesa input del usuario y genera respuesta."""
        # Convertir a estimulo
        stimulus = self._text_to_stimulus(user_input)

        # Guardar ultima reflexion
        last_reflection = None

        # Procesar segun tipo de sistema
        if self.system_type == 'conscious_self':
            for _ in range(self.processing_steps):
                result = self.system.step(stimulus=stimulus)
                # Capturar ultima reflexion si existe
                if result.get('reflection'):
                    last_reflection = result['reflection']
        else:
            for _ in range(self.processing_steps):
                self.system.communicate(stimulus)

        # Obtener estado
        state = self._get_state()
        self.last_state = state

        # Calcular luminosidad
        luminosity = self._calculate_luminosity(state['population'])

        # Extraer contexto
        context = [ex['user'] for ex in self.conversation_history[-3:]]

        # Generar respuesta verbal
        response_text = self.voice.generate(
            dominant=state['dominant'],
            blend=state['blend'],
            input_text=user_input,
            context=context if context else None,
            consciousness=state['consciousness'],
            luminosity=luminosity
        )

        # Guardar en historial
        exchange = {
            'user': user_input,
            'response': response_text,
            'dominant': state['dominant'].name,
            'consciousness': state['consciousness'],
            'luminosity': luminosity,
        }
        self.conversation_history.append(exchange)

        return {
            'text': response_text,
            'dominant': state['dominant'].name,
            'blend': {k.name: v for k, v in state['blend'].items()},
            'population': state['population'].tolist(),
            'consciousness': state['consciousness'],
            'luminosity': luminosity,
            'stage': state['stage'],
            'reflection': last_reflection,
        }

    def display_status_bar(self) -> None:
        """Muestra la barra de estado visual."""
        if not self.show_bars or self.last_state is None:
            return

        pop = self.last_state['population'].tolist()
        consciousness = self.last_state['consciousness']
        luminosity = self._calculate_luminosity(self.last_state['population'])

        print()
        print("+" + "-" * 50 + "+")
        print(format_archetype_bars(pop))
        print()
        print(format_consciousness_bar(consciousness, luminosity))
        print("+" + "-" * 50 + "+")

    def cmd_help(self) -> None:
        """Muestra ayuda."""
        help_text = """
  Comandos disponibles:
  ----------------------
  /ayuda     - Mostrar esta ayuda
  /estado    - Ver estado interno detallado
  /viaje     - Ver narrativa del viaje psicologico
  /sonar     - Entrar en ciclo de sueno (solo ZetaConsciousSelf)
  /reflexion - Forzar ciclo de auto-reflexion (Strange Loop)
  /identidad - Ver metricas de identidad emergente
  /trabajo   - Trabajo de integracion activo
  /barra     - Toggle barras de estado (on/off)
  /reset     - Reiniciar psique
  /salir     - Terminar conversacion

  Escribe cualquier texto para conversar con la psique.
  Las respuestas reflejan el arquetipo dominante.

  Usa --reflection al iniciar para ver ciclos de auto-reflexion.
"""
        print(help_text)

    def cmd_status(self) -> None:
        """Muestra estado detallado."""
        if self.last_state is None:
            # Ejecutar un paso para obtener estado
            if self.system_type == 'conscious_self':
                self.system.step()
            else:
                self.system.step()
            self.last_state = self._get_state()

        state = self.last_state
        pop = state['population'].tolist()
        luminosity = self._calculate_luminosity(state['population'])

        print("\n  === ESTADO INTERNO ===")
        print()
        print(format_archetype_bars(pop))
        print()
        print(format_consciousness_bar(state['consciousness'], luminosity))
        print()
        print(f"  Arquetipo dominante: {state['dominant'].name}")
        print(f"  Etapa individuacion: {state['stage']}")
        print()

        # Mostrar blend
        print("  Mezcla arquetipal:")
        for arch, weight in sorted(state['blend'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {arch.name}: {weight:.2%}")
        print()

    def cmd_journey(self) -> None:
        """Muestra narrativa del viaje psicologico."""
        if self.system_type != 'conscious_self':
            print("  [Requiere ZetaConsciousSelf para narrativa completa]")
            return

        state = self._get_state()
        stage = state['stage']

        narratives = {
            'INCONSCIENTE': "El viajero aun duerme, identificado con su mascara...",
            'CRISIS_PERSONA': "La mascara comienza a agrietarse. Hay inquietud.",
            'ENCUENTRO_SOMBRA': "En las profundidades, algo espera ser visto...",
            'INTEGRACION_SOMBRA': "Lo oscuro encuentra su lugar. Hay aceptacion.",
            'ENCUENTRO_ANIMA': "Lo contrasexual emerge. Nuevas energias despiertan.",
            'INTEGRACION_ANIMA': "Las polaridades danzan juntas. Hay equilibrio.",
            'EMERGENCIA_SELF': "El centro comienza a brillar. El Self se manifiesta.",
            'SELF_REALIZADO': "La totalidad se expresa. Todo tiene su lugar.",
        }

        narrative = narratives.get(stage, "El viaje continua...")

        print("\n  === VIAJE DE INDIVIDUACION ===")
        print()
        print(f"  Etapa actual: {stage}")
        print()
        print(f"  {narrative}")
        print()

        # Mostrar historial emocional
        if self.conversation_history:
            print("  Temas recientes:")
            for ex in self.conversation_history[-5:]:
                dom = ex['dominant']
                print(f"    - [{dom}] \"{ex['user'][:40]}...\"" if len(ex['user']) > 40 else f"    - [{dom}] \"{ex['user']}\"")
        print()

    def cmd_dream(self) -> None:
        """Ejecuta ciclo de sueno."""
        if self.system_type != 'conscious_self':
            print("  [Requiere ZetaConsciousSelf para ciclos de sueno]")
            return

        print("\n  [Entrando en sueno...]")
        print()

        try:
            report = self.system.dream(duration=30, verbose=False)

            print("  === FRAGMENTOS ONIRICOS ===")
            print()

            if hasattr(report, 'fragments') and report.fragments:
                for frag in report.fragments[:3]:
                    print(f"    * {frag}")
            else:
                print("    * Imagenes difusas danzan en la oscuridad...")
                print("    * Simbolos antiguos emergen y se disuelven...")

            print()
            print("  [Despertando...]")

            # Mostrar insight si hay
            if hasattr(report, 'insight') and report.insight:
                print(f"\n  Insight: \"{report.insight}\"")

        except Exception as e:
            print(f"  [Error en sueno: {e}]")
        print()

    def cmd_work(self) -> None:
        """Trabajo de integracion activo."""
        if self.system_type != 'conscious_self':
            print("  [Requiere ZetaConsciousSelf para trabajo de integracion]")
            return

        print("\n  [Iniciando trabajo de integracion...]")

        # Ejecutar pasos de integracion
        for _ in range(20):
            self.system.step()

        state = self._get_state()
        luminosity = self._calculate_luminosity(state['population'])

        print()
        print(f"  Despues del trabajo:")
        print(f"    Consciencia: {state['consciousness']:.3f}")
        print(f"    Luminosidad: {luminosity:.3f}")
        print(f"    Etapa: {state['stage']}")
        print()

    def display_reflection(self, reflection: dict) -> None:
        """Muestra el ciclo de auto-reflexión."""
        if not reflection or not reflection.get('descriptions'):
            return

        print()
        print(f"  {Colors.GRAY}[Auto-reflexion...]{Colors.RESET}")

        for i, desc in enumerate(reflection['descriptions']):
            xi = reflection['tensions'][i] if i < len(reflection['tensions']) else None
            xi_str = f"{xi:.4f}" if xi is not None else "--"

            # Color segun tension (mas verde = mas convergido)
            if xi is not None and xi < 0.02:
                tension_color = Colors.GREEN
            elif xi is not None and xi < 0.05:
                tension_color = Colors.CYAN
            else:
                tension_color = Colors.GRAY

            print(f"  {tension_color}Ciclo {i+1} (xi={xi_str}):{Colors.RESET}")
            print(f"    {Colors.DIM}{desc}{Colors.RESET}")

        # Estado final
        if reflection.get('converged'):
            print(f"  {Colors.GREEN}[Convergencia alcanzada]{Colors.RESET}")
        else:
            print(f"  {Colors.GRAY}[Max iteraciones]{Colors.RESET}")

        # Reconocimiento de atractor
        rec = reflection.get('recognition')
        if rec:
            if rec['recognized']:
                print(f"  {Colors.CYAN}[RECONOCIDO: {rec['dominant']} sim={rec['similarity']:.2f} fuerza={rec['strength']:.1f}]{Colors.RESET}")
            else:
                print(f"  {Colors.GRAY}[Nuevo atractor: {rec['dominant']}]{Colors.RESET}")

        # Identidad emergente
        identity = reflection.get('identity')
        if identity:
            print(f"  {Colors.BOLD}{identity}{Colors.RESET}")

        print()

    def cmd_reflection(self) -> None:
        """Fuerza un ciclo de reflexión manual."""
        if self.system_type != 'conscious_self':
            print("  [Requiere ZetaConsciousSelf para reflexion]")
            return

        if not hasattr(self.system, '_self_reflection_cycle'):
            print("  [Sistema no tiene capacidad de reflexion]")
            return

        print("\n  [Iniciando ciclo de reflexion...]")

        # Forzar ciclo de reflexión
        reflection = self.system._self_reflection_cycle()
        self.display_reflection(reflection)

        # Mostrar estado final
        if reflection.get('final_state'):
            final = reflection['final_state']
            print(f"  Estado final: {final['dominant'].name}")

    def cmd_identity(self) -> None:
        """Muestra metricas de identidad emergente."""
        if self.system_type != 'conscious_self':
            print("  [Requiere ZetaConsciousSelf para identidad]")
            return

        if not hasattr(self.system, 'attractor_memory'):
            print("  [Sistema no tiene memoria de atractores]")
            return

        mem = self.system.attractor_memory
        metrics = mem.get_metrics()

        print("\n  === IDENTIDAD EMERGENTE ===")
        print()
        print(f"  {mem.get_identity_description()}")
        print()
        print(f"  Metricas:")
        print(f"    Recognition rate: {metrics['recognition_rate']:.1%}")
        print(f"    Atractores unicos: {metrics['attractor_count']}")
        print(f"    Convergencias: {metrics['total_convergences']}")
        print(f"    Reconocimientos: {metrics['recognition_count']}")
        print()

        if metrics['dominant_attractor']:
            print(f"  Atractor dominante:")
            print(f"    Arquetipo: {metrics['dominant_attractor']}")
            print(f"    Fuerza: {metrics['dominant_strength']:.1f}")
            print(f"    Visitas: {metrics['dominant_visits']}")
        print()

    def toggle_bars(self) -> None:
        """Toggle barras de estado."""
        self.show_bars = not self.show_bars
        status = "activadas" if self.show_bars else "desactivadas"
        print(f"\n  [Barras de estado {status}]\n")

    def reset(self) -> None:
        """Reinicia la psique."""
        if self.system_type == 'conscious_self':
            from zeta_conscious_self import ZetaConsciousSelf
            self.system = ZetaConsciousSelf(n_cells=100)
        else:
            from zeta_psyche import ZetaPsyche
            self.system = ZetaPsyche(n_cells=100)

        self.conversation_history = []
        self.last_state = None
        print("\n  [Psique reiniciada]\n")

    def run(self) -> None:
        """Ejecuta el CLI interactivo."""
        print()
        print("=" * 60)
        print("  ZETA PSYCHE CHAT v2.0")
        print("  Inteligencia Organica Arquetipal")
        print("=" * 60)
        print()
        print("  Escribe /ayuda para ver comandos disponibles.")
        print("  Escribe cualquier texto para conversar.")
        print()
        print("-" * 60)

        # Warmup inicial
        print("\n  [Inicializando consciencia...]")
        for _ in range(20):
            if self.system_type == 'conscious_self':
                self.system.step()
            else:
                self.system.step()

        self.last_state = self._get_state()
        luminosity = self._calculate_luminosity(self.last_state['population'])
        print(f"  [Consciencia inicial: {self.last_state['consciousness']:.3f}]")
        print(f"  [Luminosidad: {luminosity:.3f}]")
        print()

        while True:
            try:
                # Mostrar barra de estado si esta activa
                if self.show_bars and self.last_state:
                    self.display_status_bar()

                user_input = input("\nTu: ").strip()

                if not user_input:
                    continue

                # Procesar comandos
                cmd = user_input.lower()

                if cmd == '/salir' or cmd == '/exit' or cmd == '/quit':
                    print("\n  [Hasta pronto. El viaje continua...]\n")
                    break

                elif cmd == '/ayuda' or cmd == '/help':
                    self.cmd_help()
                    continue

                elif cmd == '/estado' or cmd == '/status':
                    self.cmd_status()
                    continue

                elif cmd == '/viaje' or cmd == '/journey':
                    self.cmd_journey()
                    continue

                elif cmd == '/sonar' or cmd == '/dream':
                    self.cmd_dream()
                    continue

                elif cmd == '/trabajo' or cmd == '/work':
                    self.cmd_work()
                    continue

                elif cmd == '/reflexion' or cmd == '/reflection':
                    self.cmd_reflection()
                    continue

                elif cmd == '/identidad' or cmd == '/identity':
                    self.cmd_identity()
                    continue

                elif cmd == '/barra' or cmd == '/bars':
                    self.toggle_bars()
                    continue

                elif cmd == '/reset':
                    self.reset()
                    continue

                # Procesar input normal
                response = self.process_input(user_input)

                # Mostrar reflexión primero (si está habilitada)
                if self.enable_reflection and response.get('reflection'):
                    self.display_reflection(response['reflection'])

                # Mostrar respuesta
                dominant = response['dominant']
                text = response['text']
                color = Colors.for_archetype(dominant)

                print()
                print(f"{color}Psyche [{dominant}]:{Colors.RESET} {text}")

                # Mostrar insight ocasional
                if response['luminosity'] > 0.5 and response['consciousness'] > 0.7:
                    if len(self.conversation_history) % 5 == 0:
                        print()
                        print(f"{Colors.DIM}[Insight: La integracion avanza...]{Colors.RESET}")

            except KeyboardInterrupt:
                print("\n\n  [Interrumpido]\n")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"\n  [Error: {e}]\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ZetaPsyche Chat v2.0')
    parser.add_argument('--simple', action='store_true',
                        help='Usar ZetaPsyche basico en lugar de ZetaConsciousSelf')
    parser.add_argument('--no-bars', action='store_true',
                        help='Desactivar barras de estado visual')
    parser.add_argument('--decay', action='store_true',
                        help='Activar modo decay emergente')
    parser.add_argument('--reflection', action='store_true',
                        help='Activar loop de auto-reflexion (Strange Loop)')
    parser.add_argument('--cells', type=int, default=100,
                        help='Numero de celulas (default: 100)')

    args = parser.parse_args()

    cli = ZetaPsycheCLI(
        use_conscious_self=not args.simple,
        show_bars=not args.no_bars,
        enable_decay=args.decay,
        enable_reflection=args.reflection,
        n_cells=args.cells
    )

    cli.run()


if __name__ == '__main__':
    main()
