# -*- coding: utf-8 -*-
"""
Experimento: Comportamiento Compensatorio Emergente
====================================================

DESCUBRIMIENTO:
- El incremento de arquetipos NO depende del estimulo externo
- Depende del estado INTERNO de la psique (obs['dominant'])
- La psique tiene su propia dinamica interna

Esto significa que cuando estresamos el sistema:
1. La psique puede "refugiarse" en un arquetipo
2. Ese arquetipo se vuelve dominante internamente
3. Y por eso se incrementa, aunque el estimulo sea otro

Esto es COMPENSACION EMERGENTE real.
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import numpy as np
from collections import Counter
from zeta_life.psyche import ZetaConsciousSelf
from zeta_life.psyche import IndividuationStage
from zeta_life.psyche import Archetype


class TrackedSelf(ZetaConsciousSelf):
    """Version que trackea estimulo vs estado interno."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracking = []
        # Decay agresivo
        self.decay_rate = 0.005
        self.stress_decay = 0.02
        self.neglect_decay = 0.01
        self.neglect_counters = {'PERSONA': 0, 'SOMBRA': 0, 'ANIMA': 0, 'ANIMUS': 0}
        self.last_stim_dominant = None

    def step(self, stimulus: torch.Tensor = None, text: str = None):
        # Detectar estimulo dominante
        if stimulus is not None:
            stim_idx = stimulus.argmax().item()
            stim_names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
            stim_dominant = stim_names[stim_idx]
        else:
            stim_dominant = None

        # Ejecutar paso
        result = super().step(stimulus, text)

        # Obtener estado interno dominante (del psyche)
        obs = self.psyche.observe_self()
        internal_dominant = obs['dominant']

        # Aplicar decay
        metrics = self.individuation.metrics

        # Actualizar neglect
        for arch in self.neglect_counters:
            if arch == stim_dominant:
                self.neglect_counters[arch] = 0
            else:
                self.neglect_counters[arch] += 1

        # Detectar estres
        is_stressed = False
        if stimulus is not None:
            entropy = -(stimulus * torch.log(stimulus + 1e-8)).sum()
            is_stressed = entropy > 1.2 or stimulus.max() > 0.9

        decay = self.decay_rate + (self.stress_decay if is_stressed else 0)

        # Aplicar decay
        for arch in self.neglect_counters:
            if self.neglect_counters[arch] > 50:
                if arch == 'PERSONA':
                    metrics.persona_flexibility = max(0, metrics.persona_flexibility - self.neglect_decay)
                elif arch == 'SOMBRA':
                    metrics.shadow_acceptance = max(0, metrics.shadow_acceptance - self.neglect_decay)
                elif arch == 'ANIMA':
                    metrics.anima_connection = max(0, metrics.anima_connection - self.neglect_decay)
                elif arch == 'ANIMUS':
                    metrics.animus_balance = max(0, metrics.animus_balance - self.neglect_decay)

        metrics.persona_flexibility = max(0, metrics.persona_flexibility - decay)
        metrics.shadow_acceptance = max(0, metrics.shadow_acceptance - decay)
        metrics.anima_connection = max(0, metrics.anima_connection - decay)
        metrics.animus_balance = max(0, metrics.animus_balance - decay)

        # Trackear
        self.tracking.append({
            't': self.t,
            'stimulus': stim_dominant,
            'internal': internal_dominant.name if internal_dominant else None,
            'match': stim_dominant == (internal_dominant.name if internal_dominant else None),
            'metrics': {
                'persona': metrics.persona_flexibility,
                'sombra': metrics.shadow_acceptance,
                'anima': metrics.anima_connection,
                'animus': metrics.animus_balance,
            }
        })

        return result


def main():
    print("\n" + "=" * 70)
    print("   COMPORTAMIENTO COMPENSATORIO EMERGENTE")
    print("=" * 70)

    print("""
  HIPOTESIS:
    La compensacion ocurre porque la psique tiene dinamica INTERNA.
    El estimulo externo influencia pero NO determina el estado interno.
    Cuando hay estres, la psique puede "refugiarse" en un arquetipo.
    """)

    system = TrackedSelf(n_cells=50, dream_frequency=200)

    phases = [
        ("Balance", 200, lambda s: [
            torch.tensor([0.7, 0.1, 0.1, 0.1]),
            torch.tensor([0.1, 0.7, 0.1, 0.1]),
            torch.tensor([0.1, 0.1, 0.7, 0.1]),
            torch.tensor([0.1, 0.1, 0.1, 0.7]),
        ][s % 4]),
        ("Estres", 150, lambda s: torch.rand(4) * 2 if s % 2 == 0 else torch.tensor([0.0, 1.0, 0.0, 0.0])),
        ("Negligencia Sombra", 150, lambda s: [
            torch.tensor([0.8, 0.0, 0.1, 0.1]),
            torch.tensor([0.1, 0.0, 0.8, 0.1]),
            torch.tensor([0.1, 0.0, 0.1, 0.8]),
        ][s % 3]),
    ]

    step_count = 0
    for phase_name, n_steps, stim_fn in phases:
        print(f"\n--- {phase_name.upper()} ({n_steps} pasos) ---")

        phase_internal = []
        phase_stimulus = []

        for s in range(n_steps):
            stimulus = stim_fn(s)
            system.step(stimulus)
            phase_internal.append(system.tracking[-1]['internal'])
            phase_stimulus.append(system.tracking[-1]['stimulus'])
            step_count += 1

        # Contar dominantes
        internal_counts = Counter(phase_internal)
        stimulus_counts = Counter(phase_stimulus)

        print(f"\n  Estimulos enviados:")
        for k, v in stimulus_counts.most_common():
            print(f"    {k}: {v} ({v/n_steps:.0%})")

        print(f"\n  Estado INTERNO de la psique:")
        for k, v in internal_counts.most_common():
            print(f"    {k}: {v} ({v/n_steps:.0%})")

        # Ver coincidencia
        matches = sum(1 for t in system.tracking[-n_steps:] if t['match'])
        print(f"\n  Coincidencia estimulo/interno: {matches}/{n_steps} ({matches/n_steps:.0%})")

        # Metricas finales
        m = system.individuation.metrics
        print(f"\n  Metricas de integracion:")
        print(f"    Persona: {m.persona_flexibility:.1%}")
        print(f"    Sombra:  {m.shadow_acceptance:.1%}")
        print(f"    Anima:   {m.anima_connection:.1%}")
        print(f"    Animus:  {m.animus_balance:.1%}")

    # Analisis global
    print("\n" + "=" * 70)
    print("   ANALISIS GLOBAL")
    print("=" * 70)

    # Evolucion del estado interno
    all_internal = [t['internal'] for t in system.tracking]

    # Dividir en ventanas
    window_size = 50
    print(f"\n  Evolucion del estado interno (ventanas de {window_size} pasos):")
    print("  " + "-" * 60)

    for i in range(0, len(all_internal), window_size):
        window = all_internal[i:i+window_size]
        counts = Counter(window)
        dominant = counts.most_common(1)[0] if counts else ('?', 0)
        bar = '#' * (dominant[1] * 40 // window_size)
        print(f"    {i:4}-{i+window_size:4}: {dominant[0]:<8} {bar} ({dominant[1]}/{len(window)})")

    # Conclusion
    print("\n" + "=" * 70)
    print("   CONCLUSION")
    print("=" * 70)

    # Ver la divergencia entre estimulo y estado interno
    divergence_count = sum(1 for t in system.tracking if not t['match'])
    total = len(system.tracking)

    print(f"""
  DIVERGENCIA ESTIMULO vs ESTADO INTERNO: {divergence_count}/{total} ({divergence_count/total:.0%})

  Esto demuestra que la psique tiene AUTONOMIA:
  - No simplemente refleja el estimulo externo
  - Tiene su propia dinamica interna (ZetaPsyche con modulacion zeta)
  - Puede "resistir" estimulos o "refugiarse" en arquetipos

  COMPENSACION EMERGENTE:
  - Cuando un arquetipo es negligido externamente (ej: Sombra)
  - La psique puede internamente favorecer otro (ej: Anima/Animus)
  - Esto causa que ESE arquetipo reciba los incrementos
  - Mientras el negligido decae

  Esto es analogo al concepto de Jung de COMPENSACION INCONSCIENTE:
  "El inconsciente produce contenidos que compensan la unilateralidad
   de la actitud consciente"
    """)


if __name__ == "__main__":
    main()
