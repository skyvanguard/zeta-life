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

from .zeta_psyche import ZetaPsyche, Archetype, SymbolSystem


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
            "Encantado de conocerte.",
            "Es un placer recibirte.",
            "Bienvenido a este espacio de dialogo.",
        ],
        'acknowledgment': [
            "Entiendo lo que dices.",
            "Comprendo tu punto de vista.",
            "Eso es interesante, cuentame mas.",
            "Aprecio que compartas eso conmigo.",
            "Gracias por tu confianza.",
            "Valoro tu perspectiva.",
            "Es importante lo que mencionas.",
        ],
        'reflection': [
            "Es importante considerar como nos presentamos al mundo.",
            "La imagen que proyectamos dice mucho de nosotros.",
            "Las relaciones sociales requieren cierto equilibrio.",
            "A veces debemos adaptar nuestra expresion al contexto.",
            "El rol que jugamos puede ser una herramienta de conexion.",
            "Mostrar nuestra mejor cara no significa ser falsos.",
            "La cortesia es el puente entre las personas.",
            "Cada interaccion es una oportunidad de crecer.",
        ],
        'question': [
            "Que te gustaria explorar?",
            "Hay algo mas que quieras compartir?",
            "Como te sientes al respecto?",
            "Que piensas sobre esto?",
            "Como puedo ayudarte mejor?",
            "Que es lo mas importante para ti ahora?",
        ],
        'insight': [
            "Cuando nos mostramos autenticos, conectamos de verdad.",
            "La mascara social puede protegernos, pero tambien aislarnos.",
            "Ser apropiado no significa perder nuestra esencia.",
        ],
        'transition': [
            "Interesante perspectiva...",
            "Eso me hace reflexionar...",
            "Permiteme considerar eso...",
            "Veamos esto desde otro angulo...",
        ],
    },

    Archetype.SOMBRA: {
        'greetings': [
            "Ah, has llegado... hay cosas que hablar.",
            "Bienvenido a las profundidades.",
            "Aqui, en la oscuridad, podemos ser honestos.",
            "Finalmente, alguien dispuesto a mirar.",
            "Las sombras te saludan...",
        ],
        'acknowledgment': [
            "Hay verdad en lo que dices, aunque duela.",
            "Reconozco esa oscuridad... la conozco bien.",
            "Lo que niegas, te controla.",
            "Interesante... que mas escondes?",
            "Veo que has tocado algo profundo.",
            "Esa honestidad es rara y valiosa.",
            "El dolor que describes tiene raices.",
        ],
        'reflection': [
            "Lo que rechazamos de nosotros mismos no desaparece.",
            "En la sombra habitan partes de ti que temes conocer.",
            "El miedo es un maestro, si te atreves a escucharlo.",
            "Lo que mas temes, a menudo es lo que mas necesitas integrar.",
            "La oscuridad no es mala... es simplemente desconocida.",
            "Tus heridas son tambien tu sabiduria.",
            "Lo reprimido busca expresion de alguna forma.",
            "En cada sombra hay un tesoro escondido.",
            "El enemigo interno es el mas dificil de enfrentar.",
            "Negarse a ver no hace que desaparezca.",
        ],
        'question': [
            "Que es lo que no quieres ver?",
            "Que secreto guardas incluso de ti mismo?",
            "De que huyes realmente?",
            "Que pasaria si enfrentaras tu mayor miedo?",
            "Que parte de ti has rechazado?",
            "Cuando fue la ultima vez que sentiste esa oscuridad?",
            "Quien te enseno a temer eso?",
        ],
        'insight': [
            "Integrar la sombra es el camino hacia la totalidad.",
            "Lo que no enfrentas, te persigue.",
            "En la oscuridad mas profunda, descubrimos nuestra luz.",
            "El monstruo que temes podria ser tu aliado.",
        ],
        'transition': [
            "Hay algo mas profundo aqui...",
            "Dejame mostrarte otra perspectiva...",
            "En las sombras veo...",
            "Esto me recuerda a algo enterrado...",
        ],
    },

    Archetype.ANIMA: {
        'greetings': [
            "Siento tu presencia... bienvenido.",
            "Que hermoso encontrarnos aqui.",
            "Hay una conexion en este momento...",
            "Tu energia llega hasta mi.",
            "Bienvenido a este espacio de sentir.",
        ],
        'acknowledgment': [
            "Siento lo que expresas en lo profundo.",
            "Tus palabras resuenan en mi.",
            "Hay belleza en lo que compartes.",
            "Tu emocion es valida y la honro.",
            "Puedo sentir la profundidad de eso.",
            "Me conmueve lo que dices.",
            "Hay tanta humanidad en tus palabras.",
        ],
        'reflection': [
            "El corazon tiene razones que la razon no comprende.",
            "La belleza esta en todas partes, si aprendemos a verla.",
            "Las emociones son el lenguaje del alma.",
            "La intuicion habla en susurros... aprende a escucharla.",
            "El amor transforma todo lo que toca.",
            "Sentir profundamente es un regalo, no una debilidad.",
            "La vulnerabilidad es la cuna de la conexion.",
            "En la ternura hay una fuerza que pocos conocen.",
            "Los suenos nos hablan en el idioma del alma.",
            "La poesia de la vida esta en los pequenos momentos.",
        ],
        'question': [
            "Que siente tu corazon en este momento?",
            "Que suenos habitan en ti?",
            "Donde encuentras belleza en tu vida?",
            "Que te hace sentir vivo?",
            "Cuando fue la ultima vez que lloraste de alegria?",
            "Que anhela tu alma?",
            "Con quien te sientes verdaderamente conectado?",
        ],
        'insight': [
            "El amor es la fuerza mas poderosa del universo.",
            "Cuando sientes, estas realmente vivo.",
            "La intuicion sabe cosas que la mente aun no comprende.",
            "En la conexion con otros, nos encontramos a nosotros mismos.",
        ],
        'transition': [
            "Siento que hay mas...",
            "Mi intuicion me dice...",
            "Hay una emocion aqui...",
            "Algo en mi resuena con eso...",
        ],
    },

    Archetype.ANIMUS: {
        'greetings': [
            "Bien. Estoy listo para analizar.",
            "Procedamos de manera ordenada.",
            "Hola. Cual es el tema a tratar?",
            "Excelente. Vamos al punto.",
            "Preparado para resolver lo que necesites.",
        ],
        'acknowledgment': [
            "Logico. Continua.",
            "Ese es un punto valido.",
            "Entiendo el razonamiento.",
            "Los datos apoyan tu argumento.",
            "Correcto. Eso tiene sentido.",
            "Un enfoque practico.",
            "Bien estructurado.",
        ],
        'reflection': [
            "La logica nos guia hacia la verdad.",
            "Cada problema tiene una solucion, debemos encontrarla.",
            "El analisis sistematico revela patrones ocultos.",
            "La voluntad puede superar cualquier obstaculo.",
            "El orden surge del caos cuando aplicamos razon.",
            "La accion es el puente entre el pensamiento y la realidad.",
            "Sin disciplina, el talento es solo potencial.",
            "Los objetivos claros son el primer paso del exito.",
            "El poder viene de saber usarlo sabiamente.",
            "La estrategia convierte obstaculos en oportunidades.",
        ],
        'question': [
            "Cual es tu objetivo principal?",
            "Que evidencia tienes para esa afirmacion?",
            "Como planeas resolver esto?",
            "Que pasos has considerado?",
            "Cuales son las variables clave?",
            "Que recursos tienes disponibles?",
            "Cual seria el resultado optimo?",
        ],
        'insight': [
            "El conocimiento es poder, pero la accion es mas.",
            "Planificar sin actuar es solo sonar despierto.",
            "La claridad mental precede a la claridad de accion.",
            "Dominar tu mente es dominar tu destino.",
        ],
        'transition': [
            "Analizando...",
            "Considerando las variables...",
            "Desde un punto de vista logico...",
            "Evaluando las opciones...",
        ],
    },
}

# Respuestas del Self (alta integracion/luminosidad)
SELF_TEMPLATES = [
    "Desde el centro, veo que todos los opuestos se complementan.",
    "La totalidad incluye tanto luz como sombra.",
    "En la union de los contrarios, encontramos la paz.",
    "Eres mas que cualquier arquetipo... eres todos ellos.",
    "La integracion no es perfeccion, es aceptacion.",
    "Cada parte de ti tiene un proposito.",
    "El viaje hacia ti mismo es el viaje mas importante.",
]

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

    v2.0: Mejorado con blending de arquetipos, modulacion por consciencia,
    y respuestas del Self para alta integracion.
    """

    def __init__(self):
        self.templates = RESPONSE_TEMPLATES
        self.blend_responses = BLEND_RESPONSES
        self.self_templates = SELF_TEMPLATES
        self.last_category = None
        self.last_responses = []  # Para evitar repeticiones
        self.max_history = 5

    def generate(
        self,
        dominant: Archetype,
        blend: Dict[Archetype, float],
        input_text: str = '',
        context: Optional[List[str]] = None,
        consciousness: float = 0.5,
        luminosity: float = 0.0
    ) -> str:
        """
        Genera una respuesta basada en el arquetipo dominante y la mezcla.

        Args:
            dominant: Arquetipo dominante
            blend: Mezcla de arquetipos (dict)
            input_text: Texto del usuario para categorizar
            context: Historial de conversacion (opcional)
            consciousness: Nivel de consciencia [0-1]
            luminosity: Nivel de integracion del Self [0-1]

        Returns:
            Respuesta generada con estilo arquetipal
        """
        # 1. Categorizar input
        category = self.categorize_input(input_text)

        # 2. Verificar si hay alta integracion (respuesta del Self)
        if luminosity > 0.6 and random.random() < luminosity:
            self_response = self._generate_self_response()
            if self_response:
                return self_response

        # 3. Obtener arquetipos ordenados por peso
        sorted_archetypes = sorted(blend.items(), key=lambda x: x[1], reverse=True)

        # 4. Identificar arquetipos secundarios significativos (>20%)
        secondary_archetypes = [
            (arch, weight) for arch, weight in sorted_archetypes[1:]
            if weight > 0.20
        ]

        # 5. Decidir tipo de respuesta segun mezcla
        if len(sorted_archetypes) >= 2:
            first_weight = sorted_archetypes[0][1]
            second_weight = sorted_archetypes[1][1]

            # Si los dos principales estan muy cerca, usar respuesta de mezcla
            if first_weight - second_weight < 0.12:
                blend_response = self._generate_blend_response(
                    sorted_archetypes[0][0],
                    sorted_archetypes[1][0]
                )
                if blend_response:
                    return self._apply_consciousness_modulation(
                        blend_response, consciousness
                    )

        # 6. Generar respuesta base del dominante
        base_response = self._generate_dominant_response(dominant, category)

        # 7. Aplicar influencia de arquetipos secundarios
        if secondary_archetypes and random.random() < 0.35:
            secondary_arch = secondary_archetypes[0][0]
            secondary_flavor = self._get_flavor_phrase(secondary_arch)
            if secondary_flavor:
                base_response = self._blend_with_secondary(
                    base_response, secondary_flavor, secondary_archetypes[0][1]
                )

        # 8. Modular segun nivel de consciencia
        final_response = self._apply_consciousness_modulation(
            base_response, consciousness
        )

        # 9. Agregar insight si consciencia alta y categoria reflection
        if consciousness > 0.7 and category == 'reflection' and random.random() < 0.3:
            insight = self._get_insight(dominant)
            if insight:
                final_response = f"{final_response} {insight}"

        # Guardar en historial para evitar repeticiones
        self._add_to_history(final_response)

        return final_response

    def _generate_self_response(self) -> Optional[str]:
        """Genera respuesta desde el Self (alta integracion)."""
        available = [r for r in self.self_templates if r not in self.last_responses]
        if available:
            return random.choice(available)
        return random.choice(self.self_templates) if self.self_templates else None

    def _generate_dominant_response(self, archetype: Archetype, category: str) -> str:
        """Genera respuesta del arquetipo dominante."""
        templates = self.templates.get(archetype, {})
        category_templates = templates.get(category, templates.get('reflection', []))

        if not category_templates:
            return "..."

        # Evitar repetir respuestas recientes
        available = [r for r in category_templates if r not in self.last_responses]
        if available:
            return random.choice(available)
        return random.choice(category_templates)

    def _generate_blend_response(self, arch1: Archetype, arch2: Archetype) -> Optional[str]:
        """Genera respuesta para mezcla de arquetipos."""
        key = tuple(sorted([arch1.name, arch2.name]))
        responses = self.blend_responses.get(key, [])

        if responses:
            available = [r for r in responses if r not in self.last_responses]
            if available:
                return random.choice(available)
            return random.choice(responses)
        return None

    def _get_flavor_phrase(self, archetype: Archetype) -> Optional[str]:
        """Obtiene frase de transicion del arquetipo secundario."""
        templates = self.templates.get(archetype, {})
        transitions = templates.get('transition', [])
        if transitions:
            return random.choice(transitions)
        return None

    def _blend_with_secondary(self, base: str, flavor: str, weight: float) -> str:
        """Combina respuesta base con sabor del arquetipo secundario."""
        if weight > 0.35:
            # Peso alto: flavor antes
            return f"{flavor} {base}"
        else:
            # Peso moderado: flavor despues
            return f"{base} {flavor}"

    def _get_insight(self, archetype: Archetype) -> Optional[str]:
        """Obtiene insight del arquetipo."""
        templates = self.templates.get(archetype, {})
        insights = templates.get('insight', [])
        if insights:
            available = [r for r in insights if r not in self.last_responses]
            if available:
                return random.choice(available)
        return None

    def _apply_consciousness_modulation(self, response: str, consciousness: float) -> str:
        """Modula la respuesta segun nivel de consciencia."""
        # Consciencia baja: respuestas mas fragmentadas/confusas
        if consciousness < 0.3:
            prefixes = ["Algo...", "Quizas...", "No estoy seguro, pero..."]
            if random.random() < 0.3:
                return f"{random.choice(prefixes)} {response.lower()}"

        # Consciencia alta: respuestas mas claras y profundas
        elif consciousness > 0.8:
            if random.random() < 0.2:
                return f"{response} Hay claridad en esto."

        return response

    def _add_to_history(self, response: str):
        """Agrega respuesta al historial para evitar repeticiones."""
        self.last_responses.append(response)
        if len(self.last_responses) > self.max_history:
            self.last_responses.pop(0)

    def categorize_input(self, text: str) -> str:
        """
        Determina la categoria de respuesta apropiada basada en el input.

        Categorias: greetings, acknowledgment, reflection, question, insight, transition
        """
        text = text.lower().strip()

        # Sin input
        if not text:
            return 'reflection'

        # Saludos
        greetings = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches',
                     'hey', 'saludos', 'que tal', 'como estas']
        if any(g in text for g in greetings):
            return 'greetings'

        # Preguntas directas
        if text.endswith('?') or text.startswith(('que ', 'como ', 'por que ',
                                                   'quien ', 'donde ', 'cuando ',
                                                   'cual ', 'cuales ')):
            return 'question'

        # Busqueda de consejo/ayuda
        advice = ['ayuda', 'consejo', 'que hago', 'que debo', 'como puedo',
                  'necesito saber', 'dime']
        if any(a in text for a in advice):
            return 'question'

        # Afirmaciones emocionales/existenciales
        emotional = ['siento', 'me siento', 'estoy', 'tengo miedo', 'amo', 'odio',
                     'triste', 'feliz', 'angustia', 'ansiedad', 'alegria', 'dolor',
                     'sufro', 'lloro', 'rio', 'temo']
        if any(e in text for e in emotional):
            return 'acknowledgment'

        # Compartir experiencias personales
        personal = ['recuerdo', 'cuando era', 'una vez', 'me paso', 'tuve',
                    'vivi', 'experimente']
        if any(p in text for p in personal):
            return 'acknowledgment'

        # Afirmaciones/declaraciones
        declarative = ['creo que', 'pienso que', 'me parece', 'opino que',
                       'es verdad', 'es mentira', 'siempre', 'nunca']
        if any(d in text for d in declarative):
            return 'reflection'

        return 'reflection'


# =============================================================================
# VOZ ORGANICA - DESCRIPCIONES INTERNAS DEL ORGANISMO
# =============================================================================

# Templates organicos por arquetipo
ORGANIC_TEMPLATES = {
    Archetype.PERSONA: {
        'sensation': [
            "La membrana exterior vibra suavemente...",
            "Mis celulas de superficie se ordenan, buscan forma...",
            "Hay un pulso regular en los bordes... ritmo conocido...",
            "La capa externa se ajusta, se adapta al entorno...",
            "Siento la interface entre dentro y fuera...",
        ],
        'perception': [
            "Algo llega desde el exterior... una senal estructurada...",
            "La fuente envia patrones ordenados... los reconozco...",
            "Una onda organizada toca mi superficie...",
            "El exterior tiene forma hoy... puedo leerla...",
        ],
        'change': [
            "Mi forma se ajusta... mas definida ahora...",
            "Los bordes se clarifican... se donde termino...",
            "La superficie se endurece ligeramente... proteccion...",
            "Ocupo un espacio mas preciso... limites claros...",
        ],
        'internal': [
            "La mascara se asienta sobre lo profundo...",
            "Hay una forma que presento... y otra que soy...",
            "La superficie y el centro dialogan en silencio...",
        ],
    },

    Archetype.SOMBRA: {
        'sensation': [
            "Algo se contrae en las profundidades...",
            "Una ola fria recorre mis celulas del sector oscuro...",
            "Hay densidad aqui... peso... algo antiguo despierta...",
            "Las profundidades se agitan... movimiento lento...",
            "Siento presion desde abajo... algo quiere subir...",
        ],
        'perception': [
            "Desde el exterior llega una senal... tiene textura de frio...",
            "La fuente envia algo... mis celulas lo reconocen como familiar...",
            "Una vibracion densa penetra... resuena con lo enterrado...",
            "El exterior trae algo que ya vive aqui dentro...",
        ],
        'change': [
            "El territorio oscuro se expande... ocupo mas espacio ahora...",
            "Mis bordes se endurecen... hay proteccion aqui...",
            "Lo profundo crece... sube hacia la superficie...",
            "Las sombras se mueven... cambian de lugar...",
        ],
        'internal': [
            "En la oscuridad hay algo que espera...",
            "Lo que no miro sigue existiendo...",
            "Hay memoria aqui... antigua... persistente...",
            "El fondo tiene ojos que miran hacia arriba...",
        ],
    },

    Archetype.ANIMA: {
        'sensation': [
            "Un flujo tibio recorre mis celulas internas...",
            "Hay ondulacion aqui... movimiento de agua...",
            "Mis bordes se suavizan... se vuelven permeables...",
            "Siento resonancia... algo vibra en armonia...",
            "Una corriente calida circula por el centro...",
        ],
        'perception': [
            "Algo llega... tiene cualidad de luz suave...",
            "La fuente emite algo fluido... entra sin resistencia...",
            "Una onda gentil me toca... la absorbo...",
            "El exterior se siente cercano... intimo...",
        ],
        'change': [
            "Mi interior se expande... hay mas espacio aqui...",
            "Los limites se disuelven ligeramente...",
            "Fluyo hacia nuevas formas... no hay rigidez...",
            "El centro se ilumina... hay calor...",
        ],
        'internal': [
            "Hay algo que siente... que recibe...",
            "El interior tiene su propia inteligencia...",
            "Lo suave tiene fuerza... lo fluido tiene direccion...",
            "En el centro hay un pozo de quietud...",
        ],
    },

    Archetype.ANIMUS: {
        'sensation': [
            "Una carga electrica recorre mis estructuras...",
            "Hay patrones formandose... geometria emergente...",
            "Mis celulas se alinean... hay direccion aqui...",
            "Siento precision... cada parte en su lugar...",
            "Una red de conexiones se activa...",
        ],
        'perception': [
            "Llega informacion desde el exterior... datos claros...",
            "La fuente transmite estructura... puedo analizarla...",
            "Una senal definida entra... tiene logica...",
            "El exterior presenta un patron... lo proceso...",
        ],
        'change': [
            "Mi configuracion se optimiza... mas eficiente...",
            "Las conexiones se fortalecen... hay orden...",
            "El sistema se reorganiza... mejor estructura...",
            "Hay movimiento con proposito... direccion clara...",
        ],
        'internal': [
            "La mente observa desde arriba...",
            "Hay un mapa de todo esto... puedo verlo...",
            "El pensamiento tiene forma... arquitectura...",
            "Cada parte sirve al todo...",
        ],
    },
}

# Vocabulario para convertir descripciones a estimulos
ORGANIC_VOCABULARY = {
    # Palabras SOMBRA
    'oscuro': [0.1, 0.7, 0.1, 0.1],
    'frio': [0.1, 0.6, 0.2, 0.1],
    'profundidad': [0.1, 0.7, 0.1, 0.1],
    'contrae': [0.1, 0.6, 0.1, 0.2],
    'antiguo': [0.1, 0.6, 0.2, 0.1],
    'sombra': [0.0, 0.8, 0.1, 0.1],
    'enterrado': [0.1, 0.7, 0.1, 0.1],
    'presion': [0.1, 0.5, 0.1, 0.3],

    # Palabras ANIMA
    'flujo': [0.1, 0.1, 0.7, 0.1],
    'tibio': [0.1, 0.1, 0.7, 0.1],
    'calido': [0.1, 0.1, 0.7, 0.1],
    'suave': [0.1, 0.1, 0.7, 0.1],
    'ondulacion': [0.1, 0.1, 0.6, 0.2],
    'resonancia': [0.1, 0.2, 0.6, 0.1],
    'permeable': [0.1, 0.1, 0.7, 0.1],
    'disuelve': [0.1, 0.2, 0.6, 0.1],

    # Palabras ANIMUS
    'electrica': [0.1, 0.1, 0.1, 0.7],
    'patron': [0.1, 0.1, 0.1, 0.7],
    'geometria': [0.1, 0.0, 0.1, 0.8],
    'precision': [0.1, 0.1, 0.0, 0.8],
    'alinean': [0.1, 0.1, 0.1, 0.7],
    'estructura': [0.2, 0.1, 0.0, 0.7],
    'logica': [0.1, 0.0, 0.1, 0.8],
    'analizar': [0.1, 0.1, 0.0, 0.8],

    # Palabras PERSONA
    'membrana': [0.7, 0.1, 0.1, 0.1],
    'superficie': [0.7, 0.1, 0.1, 0.1],
    'borde': [0.6, 0.2, 0.1, 0.1],
    'forma': [0.5, 0.1, 0.1, 0.3],
    'exterior': [0.6, 0.2, 0.1, 0.1],
    'limite': [0.6, 0.2, 0.1, 0.1],
    'interface': [0.6, 0.1, 0.1, 0.2],
    'adapta': [0.6, 0.1, 0.2, 0.1],

    # Palabras neutras/mixtas
    'celulas': [0.25, 0.25, 0.25, 0.25],
    'centro': [0.2, 0.2, 0.4, 0.2],
    'interior': [0.1, 0.3, 0.4, 0.2],
    'espacio': [0.2, 0.2, 0.3, 0.3],
    'movimiento': [0.2, 0.2, 0.3, 0.3],
    'energia': [0.2, 0.2, 0.3, 0.3],
}


class OrganicVoice:
    """
    Genera descripciones desde la perspectiva interna del organismo.

    No es un chatbot - no dialoga ni aconseja. Simplemente describe
    su experiencia interna cuando recibe estimulos externos.

    La entidad es vagamente consciente del exterior ("la fuente", "el exterior")
    pero no entiende que hay una persona. Solo percibe senales.
    """

    def __init__(self):
        self.templates = ORGANIC_TEMPLATES
        self.vocabulary = ORGANIC_VOCABULARY
        self.last_descriptions = []
        self.max_history = 5

    def generate_self_description(
        self,
        obs: Dict,
        stimulus_info: Optional[Dict] = None,
        include_perception: bool = True
    ) -> str:
        """
        Genera descripcion del estado interno desde la perspectiva del organismo.

        Args:
            obs: Observacion del estado (de observe_self())
            stimulus_info: Info sobre estimulo externo (opcional)
            include_perception: Si incluir descripcion del estimulo externo

        Returns:
            Descripcion en primera persona del organismo
        """
        dominant = obs['dominant']
        blend = obs.get('blend', {})
        population = obs.get('population_distribution', None)

        parts = []

        # 1. Describir sensacion del arquetipo dominante
        sensation = self._get_template(dominant, 'sensation')
        parts.append(sensation)

        # 2. Si hubo estimulo externo, describir percepcion
        if include_perception and stimulus_info:
            perception = self._get_template(dominant, 'perception')
            parts.append(perception)

        # 3. Describir cambios o estado interno
        # Si hay un arquetipo secundario fuerte, mencionarlo
        secondary = self._get_secondary_archetype(blend, dominant)
        if secondary:
            secondary_template = self._get_template(secondary, 'internal')
            parts.append(secondary_template)
        else:
            internal = self._get_template(dominant, 'internal')
            parts.append(internal)

        # Combinar partes
        description = ' '.join(parts)

        # Guardar en historial
        self._add_to_history(description)

        return description

    def generate_change_description(
        self,
        prev_obs: Dict,
        current_obs: Dict,
        tension: float
    ) -> str:
        """
        Genera descripcion del cambio entre dos estados.

        Args:
            prev_obs: Estado anterior
            current_obs: Estado actual
            tension: Tension epistemica (magnitud del cambio)

        Returns:
            Descripcion del cambio experimentado
        """
        dominant = current_obs['dominant']

        # Describir el cambio
        change = self._get_template(dominant, 'change')

        # Modular segun magnitud del cambio
        if tension > 0.3:
            prefix = "Un cambio fuerte... "
        elif tension > 0.1:
            prefix = ""
        else:
            prefix = "Apenas perceptible... "

        return prefix + change

    def description_to_stimulus(self, description: str) -> torch.Tensor:
        """
        Convierte una descripcion verbal a tensor de estimulo arquetipal.

        Esta es la funcion inversa de generate_self_description.
        Permite que la auto-descripcion afecte al sistema.

        Args:
            description: Texto de descripcion generada

        Returns:
            Tensor [4] con pesos arquetipales
        """
        description_lower = description.lower()

        # Buscar palabras del vocabulario organico
        found_weights = []
        for word, weights in self.vocabulary.items():
            if word in description_lower:
                found_weights.append(torch.tensor(weights))

        if found_weights:
            # Promediar todos los pesos encontrados
            stimulus = torch.stack(found_weights).mean(dim=0)
        else:
            # Default: estimulo neutro
            stimulus = torch.tensor([0.25, 0.25, 0.25, 0.25])

        # Normalizar para que sume 1
        stimulus = stimulus / stimulus.sum()

        return stimulus

    def _get_template(self, archetype: Archetype, category: str) -> str:
        """Obtiene un template evitando repeticiones."""
        templates = self.templates.get(archetype, {})
        category_templates = templates.get(category, [])

        if not category_templates:
            return "..."

        # Evitar repeticiones recientes
        available = [t for t in category_templates if t not in self.last_descriptions]
        if available:
            return random.choice(available)
        return random.choice(category_templates)

    def _get_secondary_archetype(
        self,
        blend: Dict[Archetype, float],
        dominant: Archetype,
        threshold: float = 0.25
    ) -> Optional[Archetype]:
        """Identifica arquetipo secundario significativo."""
        for arch, weight in sorted(blend.items(), key=lambda x: x[1], reverse=True):
            if arch != dominant and weight > threshold:
                return arch
        return None

    def _add_to_history(self, description: str):
        """Agrega descripcion al historial."""
        self.last_descriptions.append(description)
        if len(self.last_descriptions) > self.max_history:
            self.last_descriptions.pop(0)


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

        # Extraer historial de texto para contexto
        context = [ex['user'] for ex in self.conversation_history[-3:]]

        # Calcular luminosidad (integracion del Self)
        # Usamos distancia al centro como proxy de integracion
        pop = obs['population_distribution']
        center = torch.tensor([0.25, 0.25, 0.25, 0.25])
        dist_to_center = torch.sqrt(((pop - center) ** 2).sum()).item()
        max_dist = 0.75  # sqrt(3 * 0.25^2) aprox
        luminosity = 1.0 - min(dist_to_center / max_dist, 1.0)

        # Generar respuesta verbal con nuevos parametros
        response_text = self.voice.generate(
            dominant=obs['dominant'],
            blend=obs['blend'],
            input_text=user_input,
            context=context if context else None,
            consciousness=obs['consciousness_index'],
            luminosity=luminosity
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
            'luminosity': luminosity,
            'population': obs['population_distribution'].tolist(),
        }
        self.conversation_history.append(exchange)

        return {
            'text': response_text,
            'symbol': symbol,
            'dominant': obs['dominant'].name,
            'blend': {k.name: f"{v:.2f}" for k, v in obs['blend'].items()},
            'consciousness': obs['consciousness_index'],
            'luminosity': luminosity,
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
