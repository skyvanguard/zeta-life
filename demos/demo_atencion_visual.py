# -*- coding: utf-8 -*-
"""
Demo Visual del Sistema de Atencion
"""
import sys
import os

# Configurar encoding para Windows
if sys.platform == 'win32':
    os.system('')  # Enable ANSI codes on Windows

import torch
import numpy as np
from zeta_life.psyche import ZetaAttentivePredictive

def print_bar(value, width=30, char='#'):
    """Imprime barra visual"""
    filled = int(value * width)
    return char * filled + '.' * (width - filled)

def run_visual_demo():
    print("\n" + "=" * 70)
    print("   DEMO VISUAL: Sistema de Atencion + Prediccion")
    print("=" * 70)

    system = ZetaAttentivePredictive(n_cells=100)
    ARCHS = ['PERSONA', 'SOMBRA ', 'ANIMA  ', 'ANIMUS ']

    # Fases del experimento
    phases = [
        ("FASE 1: Caos Inicial (Random)", 'random', 50),
        ("FASE 2: Patron Ciclico", 'cyclic', 50),
        ("FASE 3: Amenazas Repetidas", 'threat', 50),
        ("FASE 4: Oportunidades", 'opportunity', 50),
        ("FASE 5: Equilibrio", 'balance', 50),
    ]

    consciousness_history = []
    attention_history = []

    for phase_name, phase_type, n_steps in phases:
        print(f"\n{'-' * 70}")
        print(f"  {phase_name}")
        print(f"{'-' * 70}")

        phase_consciousness = []

        for step in range(n_steps):
            # Generar estimulo segun fase
            if phase_type == 'random':
                stimulus = torch.rand(4)
            elif phase_type == 'cyclic':
                phase = (step % 20) / 20 * 2 * np.pi
                stimulus = torch.tensor([
                    np.sin(phase) + 1,
                    np.cos(phase) + 1,
                    np.sin(phase + np.pi/2) + 1,
                    np.cos(phase + np.pi/2) + 1
                ], dtype=torch.float32)
            elif phase_type == 'threat':
                stimulus = torch.tensor([0.1, 0.7, 0.1, 0.1])
            elif phase_type == 'opportunity':
                stimulus = torch.tensor([0.7, 0.1, 0.1, 0.1])
            elif phase_type == 'balance':
                stimulus = torch.tensor([0.25, 0.25, 0.25, 0.25])

            result = system.step(stimulus)
            phase_consciousness.append(result['consciousness'])
            consciousness_history.append(result['consciousness'])
            attention_history.append(result['attention']['global'].detach().numpy())

        # Mostrar estadisticas de la fase
        avg_c = np.mean(phase_consciousness)
        final_att = result['attention']['global']

        print(f"\n  Consciencia promedio: {avg_c:.1%}")
        print(f"  Coherencia final:     {result['attention']['coherence']:.2f}")

        print(f"\n  Atencion arquetipal final:")
        for i, (name, val) in enumerate(zip(ARCHS, final_att)):
            bar = print_bar(val.item(), 25)
            marker = " <--" if i == final_att.argmax() else ""
            print(f"    {name}: [{bar}] {val.item():.2f}{marker}")

        print(f"\n  Contexto detectado:")
        for ctx, val in result['attention']['context'].items():
            bar = print_bar(val, 20)
            print(f"    {ctx:12}: [{bar}] {val:.2f}")

    # Resumen final
    print("\n" + "=" * 70)
    print("   RESUMEN FINAL")
    print("=" * 70)

    final_obs = system.observe()

    print(f"\n  Total de pasos: {len(consciousness_history)}")
    print(f"  Consciencia promedio: {np.mean(consciousness_history):.1%}")
    print(f"  Consciencia maxima:   {np.max(consciousness_history):.1%}")
    print(f"  Consciencia final:    {consciousness_history[-1]:.1%}")
    print(f"  Tendencia:            {final_obs['trend']:+.4f}")

    # Evolucion de consciencia
    print(f"\n  Evolucion de consciencia:")
    n_bins = 10
    bin_size = len(consciousness_history) // n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        avg = np.mean(consciousness_history[start:end])
        bar = print_bar(avg, 40)
        print(f"    Steps {start:3d}-{end:3d}: [{bar}] {avg:.1%}")

    # Evolucion de atencion por arquetipo
    print(f"\n  Evolucion de atencion por arquetipo:")
    att_array = np.array(attention_history)
    for i, name in enumerate(ARCHS):
        values = att_array[:, i]
        initial = np.mean(values[:50])
        final = np.mean(values[-50:])
        trend = final - initial
        print(f"    {name}: inicial={initial:.2f} -> final={final:.2f} ({trend:+.2f})")

    print("\n" + "=" * 70)
    print("   FIN DE LA DEMO")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_visual_demo()
