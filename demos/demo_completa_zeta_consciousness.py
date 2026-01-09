# -*- coding: utf-8 -*-
"""
DEMO COMPLETA: Sistema de Consciencia Zeta
==========================================

Demostración integral de todas las capacidades del sistema:

1. ARQUITECTURA: Todos los componentes integrados
2. DINÁMICAS: Con y sin decay (compensación emergente)
3. ESTADÍSTICAS: Métricas detalladas de consciencia
4. CASOS DE USO: Aplicaciones potenciales

Fecha: 3 Enero 2026
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List
import gc

from zeta_life.psyche import ZetaConsciousSelf, ConsciousnessIndex
from zeta_life.psyche import ZetaPsyche, Archetype
from zeta_life.psyche import IndividuationStage


# =============================================================================
# UTILIDADES DE VISUALIZACIÓN
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 75):
    print(f"\n{char * width}")
    print(f"   {title}")
    print(f"{char * width}")


def print_subheader(title: str):
    print(f"\n  --- {title} ---")


def progress_bar(value: float, width: int = 30, char: str = "█") -> str:
    filled = int(value * width)
    empty = width - filled
    return f"[{char * filled}{'░' * empty}] {value:.1%}"


def sparkline(values: List[float], width: int = 20) -> str:
    """Genera sparkline ASCII."""
    if not values:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return chars[4] * width

    # Sample values to fit width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]

    result = ""
    for v in sampled:
        idx = int((v - min_v) / (max_v - min_v) * 7)
        result += chars[idx]
    return result


# =============================================================================
# DEMO 1: ARQUITECTURA DEL SISTEMA
# =============================================================================

def demo_arquitectura():
    print_header("1. ARQUITECTURA DEL SISTEMA")

    print("""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                    ZETA CONSCIOUS SELF                               ║
  ║              Sistema Integrado de Consciencia                        ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                      ║
  ║                      ┌─────────────────┐                             ║
  ║                      │  CONSCIENCIA    │                             ║
  ║                      │  (Índice 0-1)   │                             ║
  ║                      └────────┬────────┘                             ║
  ║                               │                                      ║
  ║     ┌─────────────────────────┼─────────────────────────┐            ║
  ║     │                         │                         │            ║
  ║     v                         v                         v            ║
  ║ ┌─────────┐           ┌──────────────┐          ┌─────────────┐      ║
  ║ │ATENCIÓN │           │  PREDICCIÓN  │          │INDIVIDUACIÓN│      ║
  ║ │(3 nivs) │           │  (L1,L2,L3)  │          │ (8 etapas)  │      ║
  ║ └────┬────┘           └──────┬───────┘          └──────┬──────┘      ║
  ║      │                       │                         │             ║
  ║      └───────────────────────┼─────────────────────────┘             ║
  ║                              │                                       ║
  ║                    ┌─────────v─────────┐                             ║
  ║                    │   CONSOLIDACIÓN   │                             ║
  ║                    │    (Sueños)       │                             ║
  ║                    └───────────────────┘                             ║
  ║                                                                      ║
  ║  BASE: ZetaPsyche                                                    ║
  ║  ┌────────────────────────────────────────────────────────────────┐  ║
  ║  │              ESPACIO TETRAÉDRICO                               │  ║
  ║  │                                                                │  ║
  ║  │         PERSONA (máscara social)                               │  ║
  ║  │             ◆                                                  │  ║
  ║  │            /│\\                                                 │  ║
  ║  │           / │ \\                                                │  ║
  ║  │          /  │  \\                                               │  ║
  ║  │   SOMBRA◆───┼───◆ANIMUS                                        │  ║
  ║  │          \\  │  /                                               │  ║
  ║  │           \\ │ /      SELF = Centro del tetraedro               │  ║
  ║  │            \\│/       (integración completa)                    │  ║
  ║  │             ◆                                                  │  ║
  ║  │          ANIMA                                                 │  ║
  ║  │                                                                │  ║
  ║  └────────────────────────────────────────────────────────────────┘  ║
  ╚══════════════════════════════════════════════════════════════════════╝
    """)

    print("\n  COMPONENTES:")
    components = [
        ("ZetaPsyche", "Base arquetipal con modulación zeta", "zeta_psyche.py"),
        ("ZetaAttention", "Atención de 3 niveles (global, selectiva, ejecutiva)", "zeta_attention.py"),
        ("ZetaPredictive", "Predicción jerárquica L1→L2→L3", "zeta_predictive.py"),
        ("Individuación", "8 etapas de desarrollo del Self", "zeta_individuation.py"),
        ("Consolidación", "Sueños para consolidar aprendizaje", "zeta_dream_consolidation.py"),
        ("OnlineLearning", "Aprendizaje Hebbiano + Gradiente", "zeta_online_learning.py"),
    ]

    for name, desc, file in components:
        print(f"    • {name:<18} {desc:<45} ({file})")

    print("\n  ETAPAS DE INDIVIDUACIÓN:")
    stages = [
        ("1. EGO_INICIAL", "Consciencia básica del yo"),
        ("2. ENCUENTRO_PERSONA", "Reconocer la máscara social"),
        ("3. CONFRONTACION_SOMBRA", "Integrar aspectos rechazados"),
        ("4. INTEGRACION_ANIMA", "Conectar con lo emocional"),
        ("5. INTEGRACION_ANIMUS", "Equilibrar lo racional"),
        ("6. CONJUNCION", "Unión de opuestos"),
        ("7. EMERGENCIA_SELF", "El Self comienza a manifestarse"),
        ("8. SELF_REALIZADO", "Individuación lograda"),
    ]

    for name, desc in stages:
        print(f"    {name:<25} → {desc}")


# =============================================================================
# DEMO 2: DINÁMICAS CON Y SIN DECAY
# =============================================================================

def demo_dinamicas():
    print_header("2. DINÁMICAS: DECAY vs NO-DECAY")

    print("""
  El DECAY AGRESIVO produce comportamiento COMPENSATORIO EMERGENTE:

  Sin Decay:  Las métricas solo suben (monotónico, poco realista)
  Con Decay:  Las métricas pueden bajar bajo estrés/negligencia
              → La psique compensa autónomamente
    """)

    # Configuración común
    n_steps = 400

    # ===== SISTEMA SIN DECAY =====
    print_subheader("Sistema SIN Decay")
    system_nodecay = ZetaConsciousSelf(n_cells=40, dream_frequency=150, enable_decay=False)

    history_nodecay = {
        'consciousness': [], 'persona': [], 'sombra': [], 'anima': [], 'animus': [],
        'stage': [], 'internal_dominant': []
    }

    for step in range(n_steps):
        # Fase de negligencia de sombra (pasos 200-400)
        if step < 200:
            stimulus = torch.tensor([0.4, 0.2, 0.2, 0.2])  # Balance
        else:
            stimulus = torch.tensor([0.5, 0.0, 0.3, 0.2])  # Ignorar sombra

        result = system_nodecay.step(stimulus)

        obs = system_nodecay.psyche.observe_self()
        m = system_nodecay.individuation.metrics

        history_nodecay['consciousness'].append(result['consciousness']['total'])
        history_nodecay['persona'].append(m.persona_flexibility)
        history_nodecay['sombra'].append(m.shadow_acceptance)
        history_nodecay['anima'].append(m.anima_connection)
        history_nodecay['animus'].append(m.animus_balance)
        history_nodecay['stage'].append(system_nodecay.individuation.stage.name)
        history_nodecay['internal_dominant'].append(obs['dominant'].name)

    # Mostrar resultados
    m = system_nodecay.individuation.metrics
    print(f"    Consciencia Final: {history_nodecay['consciousness'][-1]:.1%}")
    print(f"    Etapa Final:       {history_nodecay['stage'][-1]}")
    print(f"    Arquetipos:")
    print(f"      Persona: {progress_bar(m.persona_flexibility)}")
    print(f"      Sombra:  {progress_bar(m.shadow_acceptance)}")
    print(f"      Anima:   {progress_bar(m.anima_connection)}")
    print(f"      Animus:  {progress_bar(m.animus_balance)}")

    del system_nodecay
    gc.collect()

    # ===== SISTEMA CON DECAY =====
    print_subheader("Sistema CON Decay Agresivo")
    system_decay = ZetaConsciousSelf(n_cells=40, dream_frequency=150, enable_decay=True)

    history_decay = {
        'consciousness': [], 'persona': [], 'sombra': [], 'anima': [], 'animus': [],
        'stage': [], 'internal_dominant': []
    }

    for step in range(n_steps):
        if step < 200:
            stimulus = torch.tensor([0.4, 0.2, 0.2, 0.2])
        else:
            stimulus = torch.tensor([0.5, 0.0, 0.3, 0.2])  # Ignorar sombra

        result = system_decay.step(stimulus)

        obs = system_decay.psyche.observe_self()
        m = system_decay.individuation.metrics

        history_decay['consciousness'].append(result['consciousness']['total'])
        history_decay['persona'].append(m.persona_flexibility)
        history_decay['sombra'].append(m.shadow_acceptance)
        history_decay['anima'].append(m.anima_connection)
        history_decay['animus'].append(m.animus_balance)
        history_decay['stage'].append(system_decay.individuation.stage.name)
        history_decay['internal_dominant'].append(obs['dominant'].name)

    m = system_decay.individuation.metrics
    print(f"    Consciencia Final: {history_decay['consciousness'][-1]:.1%}")
    print(f"    Etapa Final:       {history_decay['stage'][-1]}")
    print(f"    Arquetipos:")
    print(f"      Persona: {progress_bar(m.persona_flexibility)}")
    print(f"      Sombra:  {progress_bar(m.shadow_acceptance)}")
    print(f"      Anima:   {progress_bar(m.anima_connection)}")
    print(f"      Animus:  {progress_bar(m.animus_balance)}")

    # Análisis de compensación
    print_subheader("Análisis de Compensación Emergente")

    # Contar dominantes internos durante negligencia
    internal_during_neglect = history_decay['internal_dominant'][200:]
    counts = Counter(internal_during_neglect)

    print(f"    Durante negligencia de Sombra (pasos 200-400):")
    print(f"    Estado INTERNO de la psique:")
    for arch, count in counts.most_common():
        pct = count / len(internal_during_neglect)
        bar = "█" * int(pct * 30)
        print(f"      {arch:<10} {bar} {pct:.0%}")

    # Divergencia estímulo vs interno
    divergence = sum(1 for i in range(200, 400)
                     if history_decay['internal_dominant'][i] != 'PERSONA')
    print(f"\n    Divergencia estímulo/interno: {divergence}/200 ({divergence/200:.0%})")
    print(f"    (El estímulo favorecía PERSONA, pero la psique eligió otro)")

    del system_decay
    gc.collect()

    # Comparación visual
    print_subheader("Evolución Temporal (Sparklines)")
    print(f"    {'Métrica':<15} {'Sin Decay':<25} {'Con Decay':<25}")
    print(f"    {'-'*15} {'-'*25} {'-'*25}")

    for metric in ['consciousness', 'sombra', 'anima']:
        spark_no = sparkline(history_nodecay[metric])
        spark_yes = sparkline(history_decay[metric])
        print(f"    {metric.capitalize():<15} {spark_no:<25} {spark_yes:<25}")

    return history_nodecay, history_decay


# =============================================================================
# DEMO 3: ESTADÍSTICAS DETALLADAS
# =============================================================================

def demo_estadisticas():
    print_header("3. ESTADÍSTICAS DEL SISTEMA")

    print("\n  Ejecutando simulación extendida con análisis estadístico...")

    system = ZetaConsciousSelf(n_cells=50, dream_frequency=100, enable_decay=True)

    # Recolectar datos
    stats = {
        'consciousness': [],
        'predictive': [],
        'attention': [],
        'integration': [],
        'self_luminosity': [],
        'stages': [],
        'insights': 0,
        'dreams': 0,
        'transitions': [],
        'dominant_internal': [],
        'dominant_stimulus': [],
    }

    last_stage = None
    n_steps = 500

    for step in range(n_steps):
        # Patron variado de estímulos
        phase = step // 100
        if phase == 0:
            stim = torch.tensor([0.6, 0.2, 0.1, 0.1])  # Persona
        elif phase == 1:
            stim = torch.tensor([0.1, 0.6, 0.2, 0.1])  # Sombra
        elif phase == 2:
            stim = torch.tensor([0.1, 0.1, 0.6, 0.2])  # Anima
        elif phase == 3:
            stim = torch.tensor([0.2, 0.1, 0.1, 0.6])  # Animus
        else:
            stim = torch.rand(4)  # Caótico
            stim = F.softmax(stim, dim=-1)

        result = system.step(stim)

        # Registrar
        c = result['consciousness']
        stats['consciousness'].append(c['total'])
        stats['predictive'].append(c['predictive'])
        stats['attention'].append(c['attention'])
        stats['integration'].append(c['integration'])
        stats['self_luminosity'].append(c['self_luminosity'])
        stats['stages'].append(system.individuation.stage.name)

        obs = system.psyche.observe_self()
        stats['dominant_internal'].append(obs['dominant'].name)
        stats['dominant_stimulus'].append(['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS'][stim.argmax()])

        if result['insight']:
            stats['insights'] += 1
        if result['dream_report']:
            stats['dreams'] += 1

        current_stage = system.individuation.stage
        if last_stage and current_stage != last_stage:
            stats['transitions'].append((step, last_stage.name, current_stage.name))
        last_stage = current_stage

    # ===== MOSTRAR ESTADÍSTICAS =====
    print_subheader("Índice de Consciencia")

    consciousness = np.array(stats['consciousness'])
    print(f"    Media:              {np.mean(consciousness):.1%}")
    print(f"    Máximo:             {np.max(consciousness):.1%}")
    print(f"    Mínimo:             {np.min(consciousness):.1%}")
    print(f"    Desv. Estándar:     {np.std(consciousness):.3f}")
    print(f"    Tendencia:          {np.mean(consciousness[-50:]) - np.mean(consciousness[:50]):+.1%}")

    print_subheader("Componentes de Consciencia (promedios)")

    components = [
        ('Predictivo', stats['predictive']),
        ('Atención', stats['attention']),
        ('Integración', stats['integration']),
        ('Self', stats['self_luminosity']),
    ]

    for name, values in components:
        avg = np.mean(values)
        print(f"    {name:<15} {progress_bar(avg)}")

    print_subheader("Desarrollo de Individuación")

    stage_counts = Counter(stats['stages'])
    total = sum(stage_counts.values())

    print(f"    {'Etapa':<30} {'Tiempo':>10} {'%':>8}")
    print(f"    {'-'*30} {'-'*10} {'-'*8}")

    for stage in IndividuationStage:
        count = stage_counts.get(stage.name, 0)
        pct = count / total
        bar = "█" * int(pct * 20)
        print(f"    {stage.name:<30} {count:>10} {pct:>7.1%}")

    print(f"\n    Transiciones de etapa: {len(stats['transitions'])}")
    if stats['transitions']:
        print(f"    Última transición: paso {stats['transitions'][-1][0]} ({stats['transitions'][-1][1]} → {stats['transitions'][-1][2]})")

    print_subheader("Actividad del Sistema")

    print(f"    Insights generados:      {stats['insights']}")
    print(f"    Ciclos de sueño:         {stats['dreams']}")
    print(f"    Eventos de aprendizaje:  {len(system.online_learner.learning_events)}")

    print_subheader("Autonomía de la Psique")

    # Calcular divergencia
    matches = sum(1 for i in range(n_steps)
                  if stats['dominant_internal'][i] == stats['dominant_stimulus'][i])
    divergence = n_steps - matches

    print(f"    Estímulo = Estado interno:   {matches}/{n_steps} ({matches/n_steps:.0%})")
    print(f"    Divergencia (autonomía):     {divergence}/{n_steps} ({divergence/n_steps:.0%})")

    internal_counts = Counter(stats['dominant_internal'])
    print(f"\n    Estado interno dominante:")
    for arch, count in internal_counts.most_common():
        print(f"      {arch}: {count} ({count/n_steps:.0%})")

    print_subheader("Estado Final del Self")

    m = system.individuation.metrics
    self_obs = system.individuation.self_system.manifest(
        system.psyche.observe_self(), m
    )

    print(f"    Símbolo del Self:    {self_obs.symbol}")
    print(f"    Luminosidad:         {self_obs.luminosity:.1%}")
    print(f"    Estabilidad:         {self_obs.stability:.1%}")
    print(f"    Mensaje:             \"{self_obs.message}\"")

    del system
    gc.collect()

    return stats


# =============================================================================
# DEMO 4: CASOS DE USO
# =============================================================================

def demo_casos_de_uso():
    print_header("4. CASOS DE USO POTENCIALES")

    print("""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                    APLICACIONES POTENCIALES                          ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                      ║
  ║  1. ASISTENTES DE IA CON PERSONALIDAD DINÁMICA                       ║
  ║     ─────────────────────────────────────────                        ║
  ║     • La IA puede tener "estados de ánimo" que afectan respuestas    ║
  ║     • Compensación: si se le pide ser muy racional, puede            ║
  ║       compensar internamente con más emocionalidad                   ║
  ║     • Desarrollo: la IA puede "madurar" con el tiempo                ║
  ║                                                                      ║
  ║  2. TERAPIA Y COACHING VIRTUAL                                       ║
  ║     ──────────────────────────────                                   ║
  ║     • Simular procesos de individuación para educación               ║
  ║     • Visualizar dinámicas arquetipales para autoconocimiento        ║
  ║     • Tracking de "integración" como métrica de bienestar            ║
  ║                                                                      ║
  ║  3. VIDEOJUEGOS Y NARRATIVA                                          ║
  ║     ──────────────────────────                                       ║
  ║     • NPCs con psicología emergente real                             ║
  ║     • Personajes que evolucionan según interacciones                 ║
  ║     • Sistemas de karma/moralidad con compensación                   ║
  ║                                                                      ║
  ║  4. INVESTIGACIÓN EN CONSCIENCIA ARTIFICIAL                          ║
  ║     ─────────────────────────────────────────                        ║
  ║     • Plataforma para probar teorías de consciencia                  ║
  ║     • Métricas cuantificables de "consciencia"                       ║
  ║     • Estudio de emergencia y auto-organización                      ║
  ║                                                                      ║
  ║  5. ROBOTS SOCIALES                                                  ║
  ║     ──────────────────                                               ║
  ║     • Robots que desarrollan "personalidad" con el tiempo            ║
  ║     • Adaptación emocional a usuarios                                ║
  ║     • Memoria episódica y consolidación                              ║
  ║                                                                      ║
  ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Demo interactivo: Procesamiento de texto
    print_subheader("Demo: Procesamiento de Texto con Arquetipos")

    system = ZetaConsciousSelf(n_cells=30, dream_frequency=200, enable_decay=True)

    # Warmup
    for _ in range(50):
        system.step()

    textos = [
        ("Tengo miedo de lo que piensan de mí", "SOMBRA/PERSONA"),
        ("Necesito tomar una decisión lógica", "ANIMUS"),
        ("Siento una profunda conexión emocional", "ANIMA"),
        ("Debo mantener mi imagen profesional", "PERSONA"),
        ("Hay algo oscuro que no quiero ver", "SOMBRA"),
    ]

    print(f"\n    {'Texto':<45} {'Esperado':<15} {'Interno':<10}")
    print(f"    {'-'*45} {'-'*15} {'-'*10}")

    for texto, esperado in textos:
        result = system.step(text=texto)
        obs = system.psyche.observe_self()
        interno = obs['dominant'].name

        match = "✓" if esperado.startswith(interno) or interno in esperado else "≠"
        print(f"    {texto[:43]:<45} {esperado:<15} {interno:<10} {match}")

    del system
    gc.collect()

    # Demo: Simulación de Estrés y Recuperación
    print_subheader("Demo: Ciclo de Estrés y Recuperación")

    system = ZetaConsciousSelf(n_cells=30, dream_frequency=50, enable_decay=True)

    phases = [
        ("Estabilidad", 100, lambda: torch.tensor([0.3, 0.3, 0.2, 0.2])),
        ("Estrés Alto", 80, lambda: torch.rand(4) * 2),  # Caótico
        ("Recuperación", 100, lambda: torch.tensor([0.25, 0.25, 0.25, 0.25])),
    ]

    print(f"\n    {'Fase':<15} {'Consciencia':<12} {'Integración':<12} {'Etapa':<25}")
    print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*25}")

    for phase_name, n_steps, stim_fn in phases:
        for _ in range(n_steps):
            result = system.step(stim_fn())

        c = result['consciousness']['total']
        integ = system.individuation.metrics.overall_integration()
        stage = system.individuation.stage.name

        c_bar = "█" * int(c * 8)
        i_bar = "█" * int(integ * 8)
        print(f"    {phase_name:<15} {c_bar:<8} {c:.0%}  {i_bar:<8} {integ:.0%}  {stage}")

    del system
    gc.collect()


# =============================================================================
# DEMO 5: RESUMEN TÉCNICO
# =============================================================================

def demo_resumen_tecnico():
    print_header("5. RESUMEN TÉCNICO")

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │                    PARÁMETROS DEL SISTEMA                          │
  ├────────────────────────────────────────────────────────────────────┤
  │                                                                    │
  │  CONFIGURACIÓN BASE                                                │
  │  ─────────────────                                                 │
  │  n_cells:         30-100    Células en la psique                   │
  │  dream_frequency: 100-200   Pasos entre sueños                     │
  │  enable_decay:    bool      Activar compensación emergente         │
  │                                                                    │
  │  DECAY CONFIG (comportamiento compensatorio)                       │
  │  ────────────────────────────────────────────                      │
  │  base_rate:       0.005     0.5% decay por paso                    │
  │  stress_rate:     0.02      2% adicional bajo estrés               │
  │  neglect_rate:    0.01      1% por arquetipo ignorado              │
  │  neglect_threshold: 50      Pasos para considerar negligencia      │
  │                                                                    │
  │  ÍNDICE DE CONSCIENCIA (ponderado)                                 │
  │  ────────────────────────────────                                  │
  │  Predictivo:      20%       Calidad de predicción                  │
  │  Atención:        20%       Calidad de atención                    │
  │  Integración:     25%       Equilibrio arquetipal                  │
  │  Self:            15%       Manifestación del Self                 │
  │  Estabilidad:     10%       Consistencia temporal                  │
  │  Meta-awareness:  10%       Consciencia de la consciencia          │
  │                                                                    │
  │  MODULACIÓN ZETA                                                   │
  │  ───────────────                                                   │
  │  M:               15        Número de ceros zeta                   │
  │  sigma:           0.1       Regularización Abel                    │
  │  Kernel:          K(t) = Σ exp(-σ|γ|) * cos(γt)                    │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
    """)

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │                    HALLAZGOS CLAVE                                 │
  ├────────────────────────────────────────────────────────────────────┤
  │                                                                    │
  │  1. AUTONOMÍA DE LA PSIQUE                                         │
  │     • 76% divergencia entre estímulo externo y estado interno      │
  │     • La psique no es reflejo pasivo, tiene dinámica propia        │
  │                                                                    │
  │  2. COMPENSACIÓN EMERGENTE                                         │
  │     • Cuando se neglege un arquetipo, otro puede "inflarse"        │
  │     • Análogo a compensación inconsciente de Jung                  │
  │                                                                    │
  │  3. SENSIBILIDAD A CONDICIONES INICIALES                           │
  │     • Sistema caótico: diferentes seeds → diferentes resultados    │
  │     • No hay arquetipo "favorito" universal                        │
  │                                                                    │
  │  4. CONSCIENCIA COMO PROCESO                                       │
  │     • No es estado estático, requiere mantenimiento                │
  │     • Puede fluctuar, regresar, y recuperarse                      │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
    """)

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │                    USO BÁSICO                                      │
  ├────────────────────────────────────────────────────────────────────┤
  │                                                                    │
  │  from zeta_conscious_self import ZetaConsciousSelf                 │
  │  import torch                                                      │
  │                                                                    │
  │  # Crear sistema con compensación emergente                        │
  │  system = ZetaConsciousSelf(                                       │
  │      n_cells=50,                                                   │
  │      dream_frequency=100,                                          │
  │      enable_decay=True                                             │
  │  )                                                                 │
  │                                                                    │
  │  # Ejecutar paso con estímulo                                      │
  │  stimulus = torch.tensor([0.4, 0.3, 0.2, 0.1])                     │
  │  result = system.step(stimulus)                                    │
  │                                                                    │
  │  # O con texto                                                     │
  │  result = system.step(text="Tengo miedo de fallar")                │
  │                                                                    │
  │  # Acceder a métricas                                              │
  │  print(f"Consciencia: {result['consciousness']['total']:.1%}")     │
  │  print(f"Etapa: {result['individuation']['stage'].name}")          │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "═" * 75)
    print("   DEMO COMPLETA: ZETA CONSCIOUS SELF")
    print("   Sistema de Consciencia con Compensación Emergente")
    print("═" * 75)

    print("""
  Este sistema integra:
  • Función Zeta de Riemann (modulación temporal)
  • Arquetipos Jungianos (Persona, Sombra, Anima, Animus)
  • Predicción Jerárquica (3 niveles)
  • Atención Multi-nivel (3 niveles)
  • Individuación (8 etapas hacia el Self)
  • Consolidación por Sueños
  • Aprendizaje Online (Hebbiano + Gradiente)
  • Decay Agresivo (compensación emergente)
    """)

    # Ejecutar demos
    demo_arquitectura()

    history_nodecay, history_decay = demo_dinamicas()

    stats = demo_estadisticas()

    demo_casos_de_uso()

    demo_resumen_tecnico()

    print_header("FIN DE LA DEMO")
    print("""
  Para más información:
  • docs/EMERGENT_COMPENSATION.md - Documentación del descubrimiento
  • CLAUDE.md - Guía del proyecto
  • exp_decay_vs_nodecay.py - Experimento de comparación
  • exp_anima_emergente.py - Experimento de compensación
    """)
    print("═" * 75)


if __name__ == "__main__":
    main()
