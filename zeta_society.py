#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZetaPsyche Society: Sistema Multi-Psyche.

Multiples psiques interactuando en una sociedad artificial.
Permite observar:
1. Emergencia de consenso y conflicto
2. Contagio emocional entre psiques
3. Formacion de "opiniones" grupales
4. Dinamicas de influencia
5. Polarizacion vs integracion colectiva

Basado en teoria de sistemas complejos y psicologia social.
"""

import sys
import io
import random
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

from zeta_psyche import ZetaPsyche, Archetype, SymbolSystem
from zeta_psyche_voice import ArchetypalVoice, EXPANDED_VOCABULARY


# =============================================================================
# TIPOS DE RELACION SOCIAL
# =============================================================================

class RelationType(Enum):
    """Tipos de relacion entre psiques."""
    NEUTRAL = 0      # Sin relacion especial
    ALIADO = 1       # Tienden a alinearse
    RIVAL = 2        # Tienden a oponerse
    MENTOR = 3       # Uno influye mas al otro
    ESPEJO = 4       # Reflejan aspectos del otro


# =============================================================================
# INDIVIDUO EN LA SOCIEDAD
# =============================================================================

@dataclass
class SocialPsyche:
    """Una psique con capacidades sociales."""
    id: int
    name: str
    psyche: ZetaPsyche
    personality_bias: torch.Tensor  # Tendencia arquetipica innata
    openness: float  # Apertura a influencia externa (0-1)
    expressiveness: float  # Que tanto expresa su estado (0-1)

    # Estado social
    relationships: Dict[int, RelationType] = field(default_factory=dict)
    interaction_history: List[Tuple[int, torch.Tensor]] = field(default_factory=list)

    def get_state(self) -> torch.Tensor:
        """Obtiene estado arquetipico actual."""
        obs = self.psyche.observe_self()
        return obs['population_distribution']

    def get_expressed_state(self) -> torch.Tensor:
        """Estado que expresa a otros (modulado por expresividad)."""
        state = self.get_state()
        # Mezclar con bias de personalidad segun expresividad
        expressed = self.expressiveness * state + (1 - self.expressiveness) * self.personality_bias
        return F.softmax(expressed, dim=-1)

    def receive_influence(self, other_state: torch.Tensor, strength: float = 0.1):
        """Recibe influencia de otra psique."""
        # Modulado por apertura
        effective_strength = strength * self.openness

        # El estado del otro actua como estimulo
        self.psyche.communicate(other_state * effective_strength +
                                self.get_state() * (1 - effective_strength))


# =============================================================================
# GENERADOR DE PERSONALIDADES
# =============================================================================

class PersonalityGenerator:
    """Genera personalidades unicas para las psiques."""

    # Nombres arquetipicos
    NAMES = {
        Archetype.PERSONA: ['Marcus', 'Diana', 'Victor', 'Elena'],
        Archetype.SOMBRA: ['Umbra', 'Nyx', 'Raven', 'Shade'],
        Archetype.ANIMA: ['Luna', 'Aurora', 'Celeste', 'Harmony'],
        Archetype.ANIMUS: ['Atlas', 'Logos', 'Titan', 'Rex'],
    }

    @classmethod
    def generate(cls, id: int, bias: Archetype = None) -> SocialPsyche:
        """Genera una psique con personalidad unica."""
        # Bias aleatorio si no se especifica
        if bias is None:
            bias = random.choice(list(Archetype))

        # Nombre segun bias
        name = random.choice(cls.NAMES[bias])

        # Crear bias de personalidad
        personality_bias = torch.zeros(4)
        personality_bias[bias.value] = 0.5
        personality_bias += torch.rand(4) * 0.3
        personality_bias = F.softmax(personality_bias, dim=-1)

        # Crear psique base
        psyche = ZetaPsyche(n_cells=50)  # Menos celulas para eficiencia

        # Inicializar con bias
        for _ in range(10):
            psyche.communicate(personality_bias)

        return SocialPsyche(
            id=id,
            name=f"{name}_{id}",
            psyche=psyche,
            personality_bias=personality_bias,
            openness=random.uniform(0.3, 0.8),
            expressiveness=random.uniform(0.4, 0.9),
        )


# =============================================================================
# SOCIEDAD DE PSIQUES
# =============================================================================

class PsycheSociety:
    """
    Una sociedad de multiples psiques interactuando.
    """

    def __init__(self, n_members: int = 5):
        self.members: List[SocialPsyche] = []
        self.symbols = SymbolSystem()
        self.voice = ArchetypalVoice()
        self.vocabulary = EXPANDED_VOCABULARY

        # Historial
        self.collective_history: List[Dict] = []
        self.interaction_log: List[Dict] = []

        # Crear miembros con diferentes biases
        archetypes = list(Archetype)
        for i in range(n_members):
            bias = archetypes[i % len(archetypes)]
            member = PersonalityGenerator.generate(i, bias)
            self.members.append(member)

        # Crear red social inicial
        self._initialize_relationships()

    def _initialize_relationships(self):
        """Crea relaciones iniciales entre miembros."""
        for i, member in enumerate(self.members):
            for j, other in enumerate(self.members):
                if i != j:
                    # Relacion basada en similitud de personalidad
                    similarity = F.cosine_similarity(
                        member.personality_bias.unsqueeze(0),
                        other.personality_bias.unsqueeze(0)
                    ).item()

                    if similarity > 0.7:
                        member.relationships[j] = RelationType.ALIADO
                    elif similarity < 0.3:
                        member.relationships[j] = RelationType.RIVAL
                    else:
                        member.relationships[j] = RelationType.NEUTRAL

    def get_member(self, id: int) -> SocialPsyche:
        """Obtiene miembro por ID."""
        return self.members[id]

    def get_collective_state(self) -> torch.Tensor:
        """Estado colectivo (promedio ponderado por expresividad)."""
        states = []
        weights = []

        for member in self.members:
            states.append(member.get_expressed_state())
            weights.append(member.expressiveness)

        states = torch.stack(states)
        weights = torch.tensor(weights)
        weights = weights / weights.sum()

        collective = (states * weights.unsqueeze(1)).sum(dim=0)
        return collective

    def get_polarization(self) -> float:
        """
        Mide polarizacion del grupo.
        0 = consenso total, 1 = maxima polarizacion.
        """
        states = torch.stack([m.get_state() for m in self.members])

        # Varianza promedio entre miembros
        variance = states.var(dim=0).mean().item()

        # Normalizar a [0, 1]
        return min(1.0, variance * 4)

    def get_dominant_collective(self) -> Archetype:
        """Arquetipo dominante del colectivo."""
        collective = self.get_collective_state()
        return Archetype(collective.argmax().item())

    # =========================================================================
    # INTERACCIONES
    # =========================================================================

    def interact(self, member1_id: int, member2_id: int) -> Dict:
        """
        Interaccion entre dos miembros.
        Se influyen mutuamente segun sus estados y relacion.
        """
        m1 = self.members[member1_id]
        m2 = self.members[member2_id]

        # Estados antes
        state1_before = m1.get_state().clone()
        state2_before = m2.get_state().clone()

        # Determinar fuerza de influencia segun relacion
        relation = m1.relationships.get(member2_id, RelationType.NEUTRAL)

        if relation == RelationType.ALIADO:
            strength = 0.2  # Alta influencia mutua
        elif relation == RelationType.RIVAL:
            strength = 0.05  # Baja influencia, tendencia a oponerse
        elif relation == RelationType.MENTOR:
            strength = 0.3  # m1 influye mas a m2
        else:
            strength = 0.1  # Normal

        # Intercambiar influencia
        expressed1 = m1.get_expressed_state()
        expressed2 = m2.get_expressed_state()

        m1.receive_influence(expressed2, strength)
        m2.receive_influence(expressed1, strength)

        # Estados despues
        state1_after = m1.get_state()
        state2_after = m2.get_state()

        # Calcular cambio
        change1 = (state1_after - state1_before).abs().sum().item()
        change2 = (state2_after - state2_before).abs().sum().item()

        # Registrar interaccion
        interaction = {
            'members': (m1.name, m2.name),
            'relation': relation.name,
            'change': (change1, change2),
            'states_after': (state1_after.tolist(), state2_after.tolist()),
        }
        self.interaction_log.append(interaction)

        return interaction

    def broadcast_stimulus(self, stimulus: torch.Tensor, source_id: int = None):
        """
        Envia un estimulo a todos los miembros.
        Si hay source_id, ese miembro no recibe (es el emisor).
        """
        for member in self.members:
            if source_id is None or member.id != source_id:
                # Influencia reducida por ser broadcast
                member.receive_influence(stimulus, strength=0.1)

    def group_discussion(self, topic: str, rounds: int = 5) -> List[Dict]:
        """
        Simula una discusion grupal sobre un tema.
        """
        # Convertir tema a estimulo
        stimulus = self._topic_to_stimulus(topic)

        discussion = []

        for round_num in range(rounds):
            round_data = {'round': round_num + 1, 'exchanges': []}

            # Cada miembro puede hablar
            for member in self.members:
                # Recibir estimulo del tema
                member.receive_influence(stimulus, strength=0.15)

                # Procesar
                member.psyche.step()

                # Expresar opinion
                state = member.get_expressed_state()
                dominant = Archetype(state.argmax().item())
                symbol = self.symbols.encode(state)

                # Generar respuesta verbal
                response = self.voice.generate(
                    dominant=dominant,
                    blend={Archetype(i): state[i].item() for i in range(4)},
                    category='reflection'
                )

                round_data['exchanges'].append({
                    'speaker': member.name,
                    'symbol': symbol,
                    'dominant': dominant.name,
                    'response': response,
                })

                # Influir a vecinos
                for other in self.members:
                    if other.id != member.id:
                        relation = member.relationships.get(other.id, RelationType.NEUTRAL)
                        if relation != RelationType.RIVAL:
                            other.receive_influence(state, strength=0.05)

            discussion.append(round_data)

        return discussion

    def _topic_to_stimulus(self, topic: str) -> torch.Tensor:
        """Convierte tema a estimulo arquetipico."""
        topic = topic.lower()
        stimulus = torch.tensor([0.25, 0.25, 0.25, 0.25])

        for word, weights in self.vocabulary.items():
            if word in topic:
                stimulus = torch.tensor(weights)
                break

        return stimulus

    # =========================================================================
    # SIMULACION
    # =========================================================================

    def simulate_step(self):
        """
        Un paso de simulacion social.
        - Interacciones aleatorias
        - Evolucion individual
        """
        # Interacciones aleatorias
        n_interactions = len(self.members) // 2
        for _ in range(n_interactions):
            i, j = random.sample(range(len(self.members)), 2)
            self.interact(i, j)

        # Evolucion individual
        for member in self.members:
            member.psyche.step()

        # Registrar estado colectivo
        self.collective_history.append({
            'collective_state': self.get_collective_state().tolist(),
            'polarization': self.get_polarization(),
            'dominant': self.get_dominant_collective().name,
        })

    def simulate(self, steps: int = 50, verbose: bool = True) -> Dict:
        """
        Ejecuta simulacion completa.
        """
        if verbose:
            print(f"\n  Simulando {steps} pasos con {len(self.members)} psiques...")
            print()

        for step in range(steps):
            self.simulate_step()

            if verbose and (step + 1) % 10 == 0:
                state = self.get_collective_state()
                pol = self.get_polarization()
                dom = self.get_dominant_collective()
                print(f"    Step {step+1}: Dominante={dom.name}, Polarizacion={pol:.2f}")

        return {
            'history': self.collective_history,
            'final_polarization': self.get_polarization(),
            'final_dominant': self.get_dominant_collective(),
            'interactions': len(self.interaction_log),
        }

    # =========================================================================
    # VISUALIZACION
    # =========================================================================

    def get_status(self) -> str:
        """Estado actual de la sociedad."""
        lines = ["  SOCIEDAD DE PSIQUES", "  " + "="*40]

        # Estado colectivo
        collective = self.get_collective_state()
        lines.append(f"\n  Estado Colectivo:")
        lines.append(f"    PERSONA: {collective[0]*100:5.1f}%")
        lines.append(f"    SOMBRA:  {collective[1]*100:5.1f}%")
        lines.append(f"    ANIMA:   {collective[2]*100:5.1f}%")
        lines.append(f"    ANIMUS:  {collective[3]*100:5.1f}%")
        lines.append(f"\n  Polarizacion: {self.get_polarization():.2f}")
        lines.append(f"  Dominante: {self.get_dominant_collective().name}")

        # Miembros
        lines.append(f"\n  Miembros ({len(self.members)}):")
        for m in self.members:
            state = m.get_state()
            dom = Archetype(state.argmax().item())
            symbol = self.symbols.encode(state)
            lines.append(f"    {symbol} {m.name}: {dom.name} (open={m.openness:.1f})")

        return "\n".join(lines)

    def visualize(self, save_path: str = 'zeta_society.png'):
        """Visualiza la evolucion de la sociedad."""
        if not self.collective_history:
            print("  [No hay historia para visualizar]")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Evolucion de arquetipos colectivos
        ax1 = axes[0, 0]
        history = self.collective_history
        steps = range(len(history))

        persona = [h['collective_state'][0] for h in history]
        sombra = [h['collective_state'][1] for h in history]
        anima = [h['collective_state'][2] for h in history]
        animus = [h['collective_state'][3] for h in history]

        ax1.plot(steps, persona, 'r-', label='PERSONA', lw=2)
        ax1.plot(steps, sombra, 'm-', label='SOMBRA', lw=2)
        ax1.plot(steps, anima, 'b-', label='ANIMA', lw=2)
        ax1.plot(steps, animus, 'orange', label='ANIMUS', lw=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Proporcion')
        ax1.set_title('Evolucion Colectiva')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Polarizacion
        ax2 = axes[0, 1]
        polarization = [h['polarization'] for h in history]
        ax2.plot(steps, polarization, 'k-', lw=2)
        ax2.axhline(y=0.5, color='r', ls='--', alpha=0.5, label='Umbral')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Polarizacion')
        ax2.set_title('Polarizacion del Grupo')
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)

        # 3. Estado actual de miembros
        ax3 = axes[1, 0]
        member_names = [m.name.split('_')[0] for m in self.members]
        member_states = torch.stack([m.get_state() for m in self.members]).numpy()

        x = np.arange(len(self.members))
        width = 0.2

        ax3.bar(x - 1.5*width, member_states[:, 0], width, label='PERSONA', color='red', alpha=0.7)
        ax3.bar(x - 0.5*width, member_states[:, 1], width, label='SOMBRA', color='purple', alpha=0.7)
        ax3.bar(x + 0.5*width, member_states[:, 2], width, label='ANIMA', color='blue', alpha=0.7)
        ax3.bar(x + 1.5*width, member_states[:, 3], width, label='ANIMUS', color='orange', alpha=0.7)

        ax3.set_xlabel('Miembro')
        ax3.set_ylabel('Estado')
        ax3.set_title('Estado Individual')
        ax3.set_xticks(x)
        ax3.set_xticklabels(member_names, rotation=45)
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Red de relaciones
        ax4 = axes[1, 1]

        # Posiciones en circulo
        n = len(self.members)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        positions = [(np.cos(a), np.sin(a)) for a in angles]

        # Dibujar conexiones
        for i, m in enumerate(self.members):
            for j, rel in m.relationships.items():
                if i < j:  # Solo una direccion
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]

                    if rel == RelationType.ALIADO:
                        color, style = 'green', '-'
                    elif rel == RelationType.RIVAL:
                        color, style = 'red', '--'
                    else:
                        color, style = 'gray', ':'

                    ax4.plot([x1, x2], [y1, y2], color=color, ls=style, alpha=0.5)

        # Dibujar nodos
        for i, m in enumerate(self.members):
            x, y = positions[i]
            state = m.get_state()
            dom = Archetype(state.argmax().item())

            colors = {
                Archetype.PERSONA: 'red',
                Archetype.SOMBRA: 'purple',
                Archetype.ANIMA: 'blue',
                Archetype.ANIMUS: 'orange',
            }

            ax4.scatter(x, y, s=500, c=colors[dom], alpha=0.7, edgecolors='black', linewidth=2)
            ax4.annotate(m.name.split('_')[0], (x, y), ha='center', va='center', fontsize=8)

        ax4.set_xlim(-1.5, 1.5)
        ax4.set_ylim(-1.5, 1.5)
        ax4.set_aspect('equal')
        ax4.set_title('Red Social (verde=aliado, rojo=rival)')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Visualizacion guardada: {save_path}")
        plt.close()


# =============================================================================
# CLI MULTI-PSYCHE
# =============================================================================

def run_society_cli():
    """CLI interactivo para la sociedad de psiques."""
    print()
    print("=" * 60)
    print("  ZETA SOCIETY - Sociedad de Psiques")
    print("=" * 60)
    print()
    print("  Comandos:")
    print("    /estado      - Ver estado de la sociedad")
    print("    /simular [n] - Simular n pasos")
    print("    /discutir    - Discusion grupal")
    print("    /visualizar  - Generar grafico")
    print("    /hablar [id] - Hablar con un miembro")
    print("    /salir       - Terminar")
    print()
    print("-" * 60)

    # Crear sociedad
    print("\n  [Creando sociedad de 5 psiques...]")
    society = PsycheSociety(n_members=5)

    # Warmup
    for _ in range(10):
        society.simulate_step()

    print("  [Sociedad creada]")
    print(society.get_status())
    print()

    try:
        while True:
            user_input = input("Tu: ").strip()

            if not user_input:
                continue

            if user_input.lower() == '/salir':
                print("\n  [Hasta pronto...]\n")
                break

            elif user_input.lower() == '/estado':
                print()
                print(society.get_status())
                print()

            elif user_input.lower().startswith('/simular'):
                parts = user_input.split()
                steps = int(parts[1]) if len(parts) > 1 else 30
                society.simulate(steps=steps, verbose=True)
                print()

            elif user_input.lower().startswith('/discutir'):
                parts = user_input.split(maxsplit=1)
                topic = parts[1] if len(parts) > 1 else "vida"

                print(f"\n  [Discusion sobre: {topic}]")
                discussion = society.group_discussion(topic, rounds=3)

                for round_data in discussion:
                    print(f"\n  --- Ronda {round_data['round']} ---")
                    for ex in round_data['exchanges']:
                        print(f"    {ex['speaker']} [{ex['symbol']}]: {ex['response'][:60]}...")
                print()

            elif user_input.lower() == '/visualizar':
                society.visualize()

            elif user_input.lower().startswith('/hablar'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("  [Uso: /hablar <id>]")
                    continue

                member_id = int(parts[1])
                if member_id >= len(society.members):
                    print(f"  [ID invalido. Usa 0-{len(society.members)-1}]")
                    continue

                member = society.members[member_id]
                print(f"\n  [Hablando con {member.name}]")

                # Mini-conversacion
                while True:
                    msg = input(f"  Tu a {member.name.split('_')[0]}: ").strip()
                    if not msg or msg.lower() == '/fin':
                        break

                    # Procesar
                    stimulus = society._topic_to_stimulus(msg)
                    member.receive_influence(stimulus, strength=0.2)
                    member.psyche.step()

                    state = member.get_expressed_state()
                    dominant = Archetype(state.argmax().item())
                    symbol = society.symbols.encode(state)
                    response = society.voice.generate(
                        dominant=dominant,
                        blend={Archetype(i): state[i].item() for i in range(4)},
                        category=society.voice.categorize_input(msg)
                    )

                    print(f"  {member.name.split('_')[0]} [{symbol}]: {response}")
                print()

            else:
                # Broadcast a toda la sociedad
                print(f"\n  [Broadcast a la sociedad...]")
                stimulus = society._topic_to_stimulus(user_input)
                society.broadcast_stimulus(stimulus)

                # Mostrar reacciones
                for member in society.members[:3]:  # Solo primeros 3
                    state = member.get_expressed_state()
                    symbol = society.symbols.encode(state)
                    dom = Archetype(state.argmax().item())
                    print(f"    {member.name} [{symbol} {dom.name}]")
                print()

    except KeyboardInterrupt:
        print("\n\n  [Interrumpido]\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n" + "="*60)
        print("  TEST: Sociedad de Psiques")
        print("="*60)

        # Crear sociedad
        print("\n  [Creando sociedad de 5 psiques...]")
        society = PsycheSociety(n_members=5)

        print("\n  [Estado inicial]")
        print(society.get_status())

        # Simular
        print("\n  [Simulando 50 pasos...]")
        results = society.simulate(steps=50, verbose=True)

        print(f"\n  [Resultados]")
        print(f"    Polarizacion final: {results['final_polarization']:.2f}")
        print(f"    Dominante final: {results['final_dominant'].name}")
        print(f"    Interacciones: {results['interactions']}")

        # Discusion
        print("\n  [Discusion sobre 'miedo']")
        discussion = society.group_discussion("miedo", rounds=2)
        for round_data in discussion:
            print(f"\n    Ronda {round_data['round']}:")
            for ex in round_data['exchanges'][:2]:
                print(f"      {ex['speaker']}: {ex['response'][:50]}...")

        # Visualizar
        society.visualize('zeta_society_test.png')

        print("\n" + "="*60)
        print("  FIN TEST")
        print("="*60 + "\n")

    else:
        run_society_cli()
