#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test de conversacion con memoria.
Simula una sesion de chat para demostrar la memoria a largo plazo.
"""

import sys
import io

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

from zeta_life.core import MemoryAwarePsyche

def main():
    print("\n" + "="*60)
    print("  TEST: Conversacion con Memoria")
    print("="*60)

    # Crear psique con memoria (usa archivo temporal)
    psyche = MemoryAwarePsyche(n_cells=50, memory_path="test_session_memories.json")
    psyche.psyche.processing_steps = 15  # Respuesta mas clara

    # Warmup
    print("\n  [Inicializando...]")
    for _ in range(10):
        psyche.psyche.psyche.step()

    # Conversacion simulada
    conversation = [
        "hola, como estas?",
        "tengo mucho miedo ultimamente",
        "no se por que, es algo oscuro",
        "siento que algo malo va a pasar",
        "como puedo superar este miedo?",
        "gracias por escucharme",
    ]

    print("\n" + "-"*60)
    print("  SESION 1: Primera conversacion")
    print("-"*60)

    for user_input in conversation:
        response = psyche.process(user_input)
        print(f"\nTu: {user_input}")
        print(f"Psyche [{response['symbol']} {response['dominant']}]: {response['text']}")

        # Mostrar si hubo influencia de memoria
        if response.get('memories'):
            print(f"  [Recuerdo: {len(response['memories'])} memorias similares]")

    # Guardar memorias
    psyche.save()
    print("\n  [Memorias guardadas]")

    # Mostrar estado de memoria
    summary = psyche.memory.get_memory_summary()
    print(f"\n  Memoria:")
    print(f"    Episodica: {summary['total_episodic']} recuerdos")
    print(f"    Semantica: {summary['total_semantic']} conceptos")

    # Nueva sesion - simular "dia siguiente"
    print("\n" + "-"*60)
    print("  SESION 2: Retomando conversacion (memoria persistente)")
    print("-"*60)

    # Crear nueva psique que carga memorias existentes
    psyche2 = MemoryAwarePsyche(n_cells=50, memory_path="test_session_memories.json")
    psyche2.psyche.processing_steps = 15

    # Warmup
    for _ in range(10):
        psyche2.psyche.psyche.step()

    # Nueva conversacion que deberia activar memorias
    new_conversation = [
        "hola de nuevo",
        "recuerdas que tenia miedo?",
        "hoy me siento mejor",
        "creo que puedo enfrentarlo",
    ]

    for user_input in new_conversation:
        response = psyche2.process(user_input)
        print(f"\nTu: {user_input}")
        print(f"Psyche [{response['symbol']} {response['dominant']}]: {response['text']}")

        # Mostrar modulacion semantica
        sem = response.get('semantic_influence', [0.25]*4)
        if max(sem) > 0.3:  # Si hay modulacion significativa
            archetypes = ['P', 'S', 'A', 'M']
            max_idx = sem.index(max(sem))
            print(f"  [Memoria semantica activa: {archetypes[max_idx]}={max(sem):.2f}]")

    # Ver memorias similares al estado actual
    print("\n  [Memorias relacionadas con el estado actual:]")
    similar = psyche2.recall_similar(n=3)
    for m in similar:
        print(f"    - \"{m['input'][:30]}...\" ({m['dominant']}, sim={m['similarity']:.2f})")

    # Guardar y limpiar
    psyche2.save()

    # Limpiar archivo de test
    import os
    os.remove("test_session_memories.json")
    print("\n  [Test completado, archivo temporal eliminado]")

    print("\n" + "="*60)
    print("  FIN TEST")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
