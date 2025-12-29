#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZetaPsyche Voice: Sistema de comunicacion verbal arquetipica.

Extiende ZetaPsyche para generar respuestas en lenguaje natural
basadas en el arquetipo dominante y la mezcla de estados.
"""

import sys
import io
import random
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        pass

from zeta_psyche import ZetaPsyche, Archetype, SymbolSystem


# =============================================================================
# VOCABULARIO EXPANDIDO
# =============================================================================

EXPANDED_VOCABULARY = {
    # =========== PERSONA (mascara social) ===========
    'hola': [0.7, 0.1, 0.1, 0.1],
    'buenos dias': [0.8, 0.0, 0.1, 0.1],
    'gracias': [0.6, 0.1, 0.2, 0.1],
    'por favor': [0.7, 0.1, 0.1, 0.1],
    'disculpa': [0.6, 0.2, 0.1, 0.1],
    'trabajo': [0.5, 0.1, 0.1, 0.3],
    'reunion': [0.7, 0.1, 0.0, 0.2],
    'presentacion': [0.8, 0.0, 0.0, 0.2],
    'profesional': [0.7, 0.0, 0.0, 0.3],
    'imagen': [0.8, 0.1, 0.1, 0.0],
    'social': [0.8, 0.1, 0.1, 0.0],
    'apariencia': [0.7, 0.2, 0.1, 0.0],
    'correcto': [0.6, 0.1, 0.0, 0.3],
    'educado': [0.8, 0.0, 0.1, 0.1],
    'formal': [0.7, 0.1, 0.0, 0.2],
    'respeto': [0.5, 0.1, 0.2, 0.2],
    'normas': [0.6, 0.1, 0.0, 0.3],
    'sociedad': [0.7, 0.1, 0.1, 0.1],
    'rol': [0.7, 0.2, 0.0, 0.1],
    'deber': [0.5, 0.2, 0.0, 0.3],

    # =========== SOMBRA (inconsciente oscuro) ===========
    'miedo': [0.1, 0.7, 0.1, 0.1],
    'oscuridad': [0.0, 0.8, 0.1, 0.1],
    'secreto': [0.1, 0.7, 0.1, 0.1],
    'oculto': [0.1, 0.8, 0.0, 0.1],
    'vergüenza': [0.2, 0.6, 0.1, 0.1],
    'culpa': [0.1, 0.7, 0.1, 0.1],
    'rabia': [0.1, 0.6, 0.1, 0.2],
    'ira': [0.1, 0.7, 0.0, 0.2],
    'envidia': [0.1, 0.7, 0.1, 0.1],
    'celos': [0.1, 0.6, 0.2, 0.1],
    'odio': [0.0, 0.8, 0.0, 0.2],
    'muerte': [0.0, 0.8, 0.1, 0.1],
    'destruccion': [0.0, 0.7, 0.0, 0.3],
    'caos': [0.0, 0.7, 0.1, 0.2],
    'pesadilla': [0.0, 0.8, 0.1, 0.1],
    'demonio': [0.0, 0.8, 0.1, 0.1],
    'monstruo': [0.0, 0.8, 0.1, 0.1],
    'trauma': [0.1, 0.7, 0.1, 0.1],
    'dolor': [0.1, 0.5, 0.3, 0.1],
    'herida': [0.1, 0.6, 0.2, 0.1],
    'reprimido': [0.2, 0.7, 0.0, 0.1],
    'negado': [0.2, 0.7, 0.0, 0.1],
    'sombrio': [0.0, 0.8, 0.1, 0.1],
    'tinieblas': [0.0, 0.8, 0.1, 0.1],
    'abismo': [0.0, 0.8, 0.1, 0.1],

    # =========== ANIMA (receptivo, emocional) ===========
    'amor': [0.1, 0.1, 0.7, 0.1],
    'belleza': [0.2, 0.0, 0.7, 0.1],
    'sentir': [0.1, 0.1, 0.7, 0.1],
    'emocion': [0.1, 0.1, 0.7, 0.1],
    'intuicion': [0.1, 0.1, 0.6, 0.2],
    'corazon': [0.1, 0.1, 0.7, 0.1],
    'alma': [0.1, 0.2, 0.6, 0.1],
    'poesia': [0.1, 0.1, 0.7, 0.1],
    'musica': [0.1, 0.1, 0.7, 0.1],
    'arte': [0.1, 0.1, 0.6, 0.2],
    'creatividad': [0.1, 0.1, 0.5, 0.3],
    'sueno': [0.1, 0.2, 0.6, 0.1],
    'fantasia': [0.1, 0.1, 0.7, 0.1],
    'imaginacion': [0.1, 0.1, 0.6, 0.2],
    'ternura': [0.1, 0.0, 0.8, 0.1],
    'compasion': [0.1, 0.1, 0.7, 0.1],
    'empatia': [0.1, 0.1, 0.7, 0.1],
    'conexion': [0.2, 0.1, 0.6, 0.1],
    'relacion': [0.2, 0.1, 0.6, 0.1],
    'sensibilidad': [0.1, 0.1, 0.7, 0.1],
    'delicadeza': [0.1, 0.0, 0.8, 0.1],
    'flor': [0.1, 0.0, 0.8, 0.1],
    'agua': [0.1, 0.1, 0.7, 0.1],
    'luna': [0.1, 0.2, 0.6, 0.1],
    'noche': [0.1, 0.3, 0.5, 0.1],

    # =========== ANIMUS (activo, racional) ===========
    'pensar': [0.1, 0.1, 0.1, 0.7],
    'logica': [0.1, 0.0, 0.0, 0.8],
    'razon': [0.1, 0.1, 0.0, 0.8],
    'analisis': [0.1, 0.0, 0.0, 0.8],
    'hacer': [0.2, 0.1, 0.0, 0.7],
    'accion': [0.2, 0.1, 0.0, 0.7],
    'decidir': [0.2, 0.1, 0.0, 0.7],
    'resolver': [0.1, 0.1, 0.0, 0.8],
    'problema': [0.1, 0.2, 0.0, 0.7],
    'solucion': [0.1, 0.0, 0.1, 0.8],
    'estrategia': [0.2, 0.1, 0.0, 0.7],
    'plan': [0.2, 0.1, 0.0, 0.7],
    'objetivo': [0.2, 0.0, 0.0, 0.8],
    'meta': [0.2, 0.0, 0.0, 0.8],
    'logro': [0.3, 0.0, 0.1, 0.6],
    'exito': [0.3, 0.0, 0.1, 0.6],
    'competencia': [0.2, 0.2, 0.0, 0.6],
    'poder': [0.2, 0.2, 0.0, 0.6],
    'control': [0.2, 0.2, 0.0, 0.6],
    'dominio': [0.2, 0.2, 0.0, 0.6],
    'fuerza': [0.1, 0.2, 0.0, 0.7],
    'voluntad': [0.1, 0.1, 0.1, 0.7],
    'determinacion': [0.1, 0.1, 0.0, 0.8],
    'disciplina': [0.2, 0.1, 0.0, 0.7],
    'orden': [0.2, 0.1, 0.0, 0.7],

    # =========== MIXTOS / NEUTROS ===========
    'vida': [0.2, 0.2, 0.3, 0.3],
    'tiempo': [0.2, 0.2, 0.2, 0.4],
    'espacio': [0.2, 0.2, 0.2, 0.4],
    'mundo': [0.3, 0.2, 0.2, 0.3],
    'realidad': [0.2, 0.2, 0.2, 0.4],
    'verdad': [0.2, 0.2, 0.2, 0.4],
    'mentira': [0.3, 0.5, 0.1, 0.1],
    'bien': [0.3, 0.1, 0.3, 0.3],
    'mal': [0.1, 0.6, 0.1, 0.2],
    'libertad': [0.2, 0.2, 0.3, 0.3],
    'destino': [0.1, 0.3, 0.3, 0.3],
    'cambio': [0.2, 0.2, 0.2, 0.4],
    'transformacion': [0.1, 0.3, 0.3, 0.3],
    'crecimiento': [0.2, 0.1, 0.3, 0.4],
    'aprendizaje': [0.2, 0.1, 0.2, 0.5],
    'conocimiento': [0.2, 0.1, 0.1, 0.6],
    'sabiduria': [0.2, 0.2, 0.3, 0.3],
    'experiencia': [0.2, 0.2, 0.3, 0.3],
    'memoria': [0.2, 0.3, 0.3, 0.2],
    'olvido': [0.1, 0.5, 0.3, 0.1],
    'presente': [0.3, 0.1, 0.3, 0.3],
    'pasado': [0.2, 0.4, 0.2, 0.2],
    'futuro': [0.2, 0.2, 0.2, 0.4],
    'esperanza': [0.2, 0.1, 0.5, 0.2],
    'desesperanza': [0.1, 0.6, 0.2, 0.1],

    # =========== PREGUNTAS / DIALOGICAS ===========
    'que': [0.2, 0.2, 0.2, 0.4],
    'como': [0.2, 0.2, 0.2, 0.4],
    'por que': [0.1, 0.3, 0.2, 0.4],
    'quien': [0.3, 0.2, 0.3, 0.2],
    'donde': [0.2, 0.2, 0.2, 0.4],
    'cuando': [0.2, 0.2, 0.2, 0.4],
    'si': [0.3, 0.2, 0.3, 0.2],
    'no': [0.2, 0.4, 0.2, 0.2],
    'tal vez': [0.2, 0.2, 0.4, 0.2],
    'quiero': [0.2, 0.2, 0.3, 0.3],
    'necesito': [0.2, 0.3, 0.3, 0.2],
    'siento': [0.1, 0.2, 0.6, 0.1],
    'pienso': [0.1, 0.1, 0.1, 0.7],
    'creo': [0.2, 0.2, 0.3, 0.3],
    'dudo': [0.1, 0.4, 0.2, 0.3],
    'se': [0.2, 0.1, 0.1, 0.6],
    'no se': [0.2, 0.3, 0.3, 0.2],
    'ayuda': [0.2, 0.3, 0.4, 0.1],
    'entiendo': [0.2, 0.1, 0.3, 0.4],
    'no entiendo': [0.2, 0.4, 0.2, 0.2],
}


# =============================================================================
# PLANTILLAS DE RESPUESTA POR ARQUETIPO
# =============================================================================

RESPONSE_TEMPLATES = {
    Archetype.PERSONA: {
        'greetings': [
            "Es un gusto saludarte.",
            "Bienvenido, estoy aqui para conversar.",
            "Hola, que agradable tenerte aqui.",
            "Buenos dias, como puedo ayudarte?",
        ],
        'acknowledgment': [
            "Entiendo lo que dices.",
            "Comprendo tu punto de vista.",
            "Eso es interesante, cuentame mas.",
            "Aprecio que compartas eso conmigo.",
        ],
        'reflection': [
            "Es importante considerar como nos presentamos al mundo.",
            "La imagen que proyectamos dice mucho de nosotros.",
            "Las relaciones sociales requieren cierto equilibrio.",
            "A veces debemos adaptar nuestra expresion al contexto.",
        ],
        'question': [
            "Que te gustaria explorar?",
            "Hay algo mas que quieras compartir?",
            "Como te sientes al respecto?",
            "Que piensas sobre esto?",
        ],
        'transition': [
            "Interesante perspectiva...",
            "Eso me hace reflexionar...",
            "Permiteme considerar eso...",
        ],
    },

    Archetype.SOMBRA: {
        'greetings': [
            "Ah, has llegado... hay cosas que hablar.",
            "Bienvenido a las profundidades.",
            "Aqui, en la oscuridad, podemos ser honestos.",
        ],
        'acknowledgment': [
            "Hay verdad en lo que dices, aunque duela.",
            "Reconozco esa oscuridad... la conozco bien.",
            "Lo que niegas, te controla.",
            "Interesante... que mas escondes?",
        ],
        'reflection': [
            "Lo que rechazamos de nosotros mismos no desaparece.",
            "En la sombra habitan partes de ti que temes conocer.",
            "El miedo es un maestro, si te atreves a escucharlo.",
            "Lo que mas temes, a menudo es lo que mas necesitas integrar.",
            "La oscuridad no es mala... es simplemente desconocida.",
        ],
        'question': [
            "Que es lo que no quieres ver?",
            "Que secreto guardas incluso de ti mismo?",
            "De que huyes realmente?",
            "Que pasaria si enfrentaras tu mayor miedo?",
        ],
        'transition': [
            "Hay algo mas profundo aqui...",
            "Dejame mostrarte otra perspectiva...",
            "En las sombras veo...",
        ],
    },

    Archetype.ANIMA: {
        'greetings': [
            "Siento tu presencia... bienvenido.",
            "Que hermoso encontrarnos aqui.",
            "Hay una conexion en este momento...",
        ],
        'acknowledgment': [
            "Siento lo que expresas en lo profundo.",
            "Tus palabras resuenan en mi.",
            "Hay belleza en lo que compartes.",
            "Tu emocion es valida y la honro.",
        ],
        'reflection': [
            "El corazon tiene razones que la razon no comprende.",
            "La belleza esta en todas partes, si aprendemos a verla.",
            "Las emociones son el lenguaje del alma.",
            "La intuicion habla en susurros... aprende a escucharla.",
            "El amor transforma todo lo que toca.",
        ],
        'question': [
            "Que siente tu corazon en este momento?",
            "Que suenos habitan en ti?",
            "Donde encuentras belleza en tu vida?",
            "Que te hace sentir vivo?",
        ],
        'transition': [
            "Siento que hay mas...",
            "Mi intuicion me dice...",
            "Hay una emocion aqui...",
        ],
    },

    Archetype.ANIMUS: {
        'greetings': [
            "Bien. Estoy listo para analizar.",
            "Procedamos de manera ordenada.",
            "Hola. Cual es el tema a tratar?",
        ],
        'acknowledgment': [
            "Logico. Continua.",
            "Ese es un punto valido.",
            "Entiendo el razonamiento.",
            "Los datos apoyan tu argumento.",
        ],
        'reflection': [
            "La logica nos guia hacia la verdad.",
            "Cada problema tiene una solucion, debemos encontrarla.",
            "El analisis sistematico revela patrones ocultos.",
            "La voluntad puede superar cualquier obstaculo.",
            "El orden surge del caos cuando aplicamos razon.",
        ],
        'question': [
            "Cual es tu objetivo principal?",
            "Que evidencia tienes para esa afirmacion?",
            "Como planeas resolver esto?",
            "Que pasos has considerado?",
        ],
        'transition': [
            "Analizando...",
            "Considerando las variables...",
            "Desde un punto de vista logico...",
        ],
    },
}

# Respuestas para mezclas de arquetipos
BLEND_RESPONSES = {
    ('PERSONA', 'SOMBRA'): [
        "La mascara a veces oculta heridas profundas.",
        "Lo que mostramos al mundo no siempre es lo que somos.",
        "Hay una tension entre lo que proyectamos y lo que sentimos.",
    ],
    ('PERSONA', 'ANIMA'): [
        "Podemos ser socialmente graciosos mientras sentimos profundamente.",
        "La amabilidad genuina viene del corazon.",
        "Conectar con otros requiere autenticidad emocional.",
    ],
    ('PERSONA', 'ANIMUS'): [
        "El profesionalismo y la logica trabajan juntos.",
        "La imagen publica se beneficia de la estrategia.",
        "El rol social puede ser un vehiculo para lograr objetivos.",
    ],
    ('SOMBRA', 'ANIMA'): [
        "En la oscuridad tambien hay belleza.",
        "El dolor puede transformarse en arte.",
        "Las emociones reprimidas buscan expresion.",
    ],
    ('SOMBRA', 'ANIMUS'): [
        "Analizar nuestras sombras requiere coraje.",
        "La logica puede iluminar la oscuridad.",
        "El poder de la sombra puede ser canalizado.",
    ],
    ('ANIMA', 'ANIMUS'): [
        "La intuicion y la logica son complementarias.",
        "El corazon y la mente pueden trabajar juntos.",
        "La creatividad florece cuando sentir y pensar se unen.",
    ],
}


# =============================================================================
# GENERADOR DE VOZ ARQUETIPICA
# =============================================================================

class ArchetypalVoice:
    """
    Genera respuestas verbales basadas en el estado arquetipico.
    """

    def __init__(self):
        self.templates = RESPONSE_TEMPLATES
        self.blend_responses = BLEND_RESPONSES
        self.last_category = None

    def generate(
        self,
        dominant: Archetype,
        blend: Dict[Archetype, float],
        context: str = '',
        category: str = 'reflection'
    ) -> str:
        """
        Genera una respuesta basada en el arquetipo dominante y la mezcla.

        Args:
            dominant: Arquetipo dominante
            blend: Mezcla de arquetipos (dict)
            context: Contexto de la conversacion
            category: Tipo de respuesta (greetings, acknowledgment, reflection, question)

        Returns:
            Respuesta generada
        """
        # Obtener pesos ordenados
        sorted_archetypes = sorted(blend.items(), key=lambda x: x[1], reverse=True)

        # Verificar si hay mezcla significativa
        if len(sorted_archetypes) >= 2:
            first_weight = sorted_archetypes[0][1]
            second_weight = sorted_archetypes[1][1]

            # Si los dos principales estan cerca, usar respuesta de mezcla
            if first_weight - second_weight < 0.15 and random.random() < 0.4:
                return self._generate_blend_response(
                    sorted_archetypes[0][0],
                    sorted_archetypes[1][0]
                )

        # Respuesta del arquetipo dominante
        return self._generate_dominant_response(dominant, category)

    def _generate_dominant_response(self, archetype: Archetype, category: str) -> str:
        """Genera respuesta del arquetipo dominante."""
        templates = self.templates.get(archetype, {})
        category_templates = templates.get(category, templates.get('reflection', []))

        if not category_templates:
            return "..."

        # Evitar repetir la misma respuesta
        response = random.choice(category_templates)
        return response

    def _generate_blend_response(self, arch1: Archetype, arch2: Archetype) -> str:
        """Genera respuesta para mezcla de arquetipos."""
        key = tuple(sorted([arch1.name, arch2.name]))
        responses = self.blend_responses.get(key, [])

        if responses:
            return random.choice(responses)

        # Fallback: combinar respuestas
        r1 = self._generate_dominant_response(arch1, 'reflection')
        return r1

    def categorize_input(self, text: str) -> str:
        """Determina la categoria de respuesta apropiada."""
        text = text.lower()

        # Saludos
        greetings = ['hola', 'buenos', 'buenas', 'hey', 'saludos']
        if any(g in text for g in greetings):
            return 'greetings'

        # Preguntas
        questions = ['?', 'que', 'como', 'por que', 'quien', 'donde', 'cuando']
        if any(q in text for q in questions):
            return 'question'

        # Afirmaciones emocionales
        emotional = ['siento', 'me siento', 'estoy', 'tengo miedo', 'amo', 'odio']
        if any(e in text for e in emotional):
            return 'acknowledgment'

        return 'reflection'


# =============================================================================
# INTERFAZ DE CONVERSACION MEJORADA
# =============================================================================

class ConversationalPsyche:
    """
    Interfaz de conversacion completa con ZetaPsyche.
    """

    def __init__(self, n_cells: int = 100):
        self.psyche = ZetaPsyche(n_cells=n_cells)
        self.voice = ArchetypalVoice()
        self.symbols = SymbolSystem()
        self.vocabulary = EXPANDED_VOCABULARY
        self.conversation_history = []
        self.processing_steps = 15  # Pasos por input (mas = mas respuesta)

    def _text_to_stimulus(self, text: str) -> torch.Tensor:
        """Convierte texto a estimulo arquetipico."""
        text = text.lower()

        # Buscar palabras clave
        stimulus = torch.tensor([0.25, 0.25, 0.25, 0.25])

        for word, weights in self.vocabulary.items():
            if word in text:
                stimulus = torch.tensor(weights)
                break

        # Combinar multiples palabras encontradas
        found_weights = []
        for word, weights in self.vocabulary.items():
            if word in text:
                found_weights.append(torch.tensor(weights))

        if found_weights:
            stimulus = torch.stack(found_weights).mean(dim=0)

        return stimulus

    def process(self, user_input: str) -> Dict:
        """
        Procesa input del usuario y genera respuesta.

        Returns:
            Dict con respuesta, simbolo, estado, etc.
        """
        # Convertir a estimulo
        stimulus = self._text_to_stimulus(user_input)

        # Procesar con la psique (multiples pasos)
        for _ in range(self.processing_steps):
            self.psyche.communicate(stimulus)

        # Observar estado
        obs = self.psyche.observe_self()

        # Determinar categoria de respuesta
        category = self.voice.categorize_input(user_input)

        # Generar respuesta verbal
        response_text = self.voice.generate(
            dominant=obs['dominant'],
            blend=obs['blend'],
            context=user_input,
            category=category
        )

        # Generar simbolo
        symbol = self.symbols.encode(obs['population_distribution'])

        # Guardar en historial
        exchange = {
            'user': user_input,
            'response': response_text,
            'symbol': symbol,
            'dominant': obs['dominant'],
            'consciousness': obs['consciousness_index'],
            'population': obs['population_distribution'].tolist(),
        }
        self.conversation_history.append(exchange)

        return {
            'text': response_text,
            'symbol': symbol,
            'dominant': obs['dominant'].name,
            'blend': {k.name: f"{v:.2f}" for k, v in obs['blend'].items()},
            'consciousness': obs['consciousness_index'],
            'population': obs['population_distribution'].tolist(),
        }

    def get_status_bar(self, obs: Dict = None) -> str:
        """Genera barra de estado visual."""
        if obs is None:
            obs = self.psyche.observe_self()

        pop = obs['population_distribution']

        def bar(val, width=10):
            filled = int(val * width)
            return '█' * filled + '░' * (width - filled)

        lines = [
            f"  PERSONA {bar(pop[0])} {pop[0]*100:5.1f}%",
            f"  SOMBRA  {bar(pop[1])} {pop[1]*100:5.1f}%",
            f"  ANIMA   {bar(pop[2])} {pop[2]*100:5.1f}%",
            f"  ANIMUS  {bar(pop[3])} {pop[3]*100:5.1f}%",
            f"",
            f"  Consciencia: {bar(obs['consciousness_index'])} {obs['consciousness_index']:.3f}",
        ]

        return '\n'.join(lines)


# =============================================================================
# CLI INTERACTIVO
# =============================================================================

def run_interactive_cli():
    """Ejecuta el CLI interactivo de ZetaPsyche."""
    print()
    print("=" * 60)
    print("  ZETA PSYCHE - Inteligencia Organica Arquetipica")
    print("=" * 60)
    print()
    print("  Comandos especiales:")
    print("    /estado  - Ver estado interno")
    print("    /reset   - Reiniciar psique")
    print("    /salir   - Terminar conversacion")
    print()
    print("-" * 60)

    psyche = ConversationalPsyche(n_cells=100)

    # Warmup inicial
    print("\n  [Inicializando consciencia...]")
    for _ in range(20):
        psyche.psyche.step()
    obs = psyche.psyche.observe_self()
    print(f"  [Consciencia inicial: {obs['consciousness_index']:.3f}]")
    print()

    while True:
        try:
            user_input = input("Tu: ").strip()

            if not user_input:
                continue

            # Comandos especiales
            if user_input.lower() == '/salir':
                print("\n  [Hasta pronto...]\n")
                break

            elif user_input.lower() == '/estado':
                obs = psyche.psyche.observe_self()
                print()
                print(psyche.get_status_bar(obs))
                print()
                continue

            elif user_input.lower() == '/reset':
                psyche = ConversationalPsyche(n_cells=100)
                print("\n  [Psique reiniciada]\n")
                continue

            # Procesar input normal
            response = psyche.process(user_input)

            # Mostrar respuesta
            symbol = response['symbol']
            dominant = response['dominant']
            text = response['text']

            print(f"\nPsyche [{symbol} {dominant}]: {text}\n")

        except KeyboardInterrupt:
            print("\n\n  [Interrumpido]\n")
            break
        except EOFError:
            break


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Modo test - cada input con psique fresca para mostrar respuesta
        print("\n[TEST MODE - Respuesta por arquetipo]")

        test_inputs = [
            ("hola", "Saludo neutro"),
            ("tengo mucho miedo", "SOMBRA esperado"),
            ("te amo profundamente", "ANIMA esperado"),
            ("analiza este problema", "ANIMUS esperado"),
            ("buenos dias, como estas", "PERSONA esperado"),
            ("siento una tristeza profunda", "SOMBRA/ANIMA"),
        ]

        for inp, expected in test_inputs:
            # Nueva psique para cada input (muestra responsividad)
            psyche = ConversationalPsyche(n_cells=50)
            psyche.processing_steps = 20  # Mas pasos para respuesta clara

            response = psyche.process(inp)
            print(f"\nTu: \"{inp}\"")
            print(f"   Esperado: {expected}")
            print(f"   Psyche [{response['symbol']} {response['dominant']}]: {response['text']}")
            pop = response['population']
            print(f"   Poblacion: P={pop[0]:.2f} S={pop[1]:.2f} A={pop[2]:.2f} M={pop[3]:.2f}")

        # Test de conversacion continua
        print("\n" + "="*60)
        print("[TEST - Conversacion continua (memoria)]")
        print("="*60)

        psyche = ConversationalPsyche(n_cells=100)
        psyche.processing_steps = 10

        conversation = [
            "hola, como estas?",
            "tengo miedo de algo",
            "no se que es",
            "creo que es el futuro",
            "como puedo superarlo?",
        ]

        for inp in conversation:
            response = psyche.process(inp)
            print(f"\nTu: {inp}")
            print(f"Psyche [{response['symbol']} {response['dominant']}]: {response['text']}")

        print("\n[FIN TEST]")

    else:
        # Modo interactivo
        run_interactive_cli()
