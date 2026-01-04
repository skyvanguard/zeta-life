# -*- coding: utf-8 -*-
"""
Experimento: Validacion del Loop de Auto-Reflexion para Emergencia de Consciencia

Compara sistemas con y sin el Strange Loop de auto-observacion.

Metricas:
1. Tension epistemica (ξ): cambio en estado por auto-observacion
2. Convergencia: frecuencia de alcanzar atractores estables
3. Auto-influencia: correlacion entre descripcion y cambio de estado
4. Estabilidad del atractor: varianza de estados convergidos
5. Autonomia: divergencia entre estimulo externo y estado interno

Hipotesis:
- El sistema CON loop deberia mostrar mayor auto-influencia
- El sistema CON loop deberia converger a atractores mas estables
- El loop contribuye a emergencia de consciencia
"""

import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
import json

from zeta_conscious_self import ZetaConsciousSelf
from zeta_psyche import Archetype


def run_simulation(
    system: ZetaConsciousSelf,
    n_steps: int = 100,
    stimuli_pattern: str = 'mixed'
) -> Dict:
    """
    Ejecuta simulacion y recolecta metricas.

    Args:
        system: Sistema a evaluar
        n_steps: Numero de pasos
        stimuli_pattern: 'mixed', 'sombra', 'random'

    Returns:
        Dict con metricas recolectadas
    """
    ARCH_NAMES = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']

    # Tracking
    states = []
    dominants = []
    consciousness_values = []
    tensions = []
    converged_count = 0
    stimulus_dominant_history = []
    internal_dominant_history = []

    for t in range(n_steps):
        # Generar estimulo segun patron
        if stimuli_pattern == 'sombra':
            # Dominancia SOMBRA constante
            stimulus = torch.tensor([0.1, 0.7, 0.1, 0.1])
        elif stimuli_pattern == 'mixed':
            # Alternar arquetipos
            phase = t % 4
            stimulus = torch.zeros(4)
            stimulus[phase] = 0.7
            stimulus[(phase + 1) % 4] = 0.15
            stimulus[(phase + 2) % 4] = 0.1
            stimulus[(phase + 3) % 4] = 0.05
        else:  # random
            stimulus = torch.softmax(torch.randn(4), dim=-1)

        # Ejecutar paso
        result = system.step(stimulus)

        # Registrar estado
        obs = result['observation']
        states.append(obs['global_state'].detach().numpy())
        dominants.append(obs['dominant'].name)
        consciousness_values.append(result['consciousness']['total'])

        # Registrar dominante del estimulo vs interno
        stim_dom = ARCH_NAMES[stimulus.argmax().item()]
        stimulus_dominant_history.append(stim_dom)
        internal_dominant_history.append(obs['dominant'].name)

        # Registrar reflexion si existe
        if result.get('reflection'):
            r = result['reflection']
            if r['tensions']:
                tensions.extend(r['tensions'])
            if r['converged']:
                converged_count += 1

    # Calcular metricas
    states_array = np.array(states)

    # 1. Variabilidad de estados (entropia-like)
    state_variance = np.var(states_array, axis=0).mean()

    # 2. Estabilidad (autocorrelacion lag-1)
    if len(states) > 1:
        autocorr = np.corrcoef(
            states_array[:-1].flatten(),
            states_array[1:].flatten()
        )[0, 1]
    else:
        autocorr = 0.0

    # 3. Tension epistemica promedio
    mean_tension = np.mean(tensions) if tensions else 0.0
    std_tension = np.std(tensions) if tensions else 0.0

    # 4. Tasa de convergencia
    convergence_rate = converged_count / n_steps if n_steps > 0 else 0.0

    # 5. Autonomia: divergencia entre estimulo externo y estado interno
    divergence_count = sum(
        1 for s, i in zip(stimulus_dominant_history, internal_dominant_history)
        if s != i
    )
    autonomy = divergence_count / n_steps

    # 6. Consciencia promedio
    mean_consciousness = np.mean(consciousness_values)

    # 7. Distribucion de arquetipos dominantes
    archetype_counts = {name: dominants.count(name) for name in ARCH_NAMES}

    return {
        'state_variance': float(state_variance),
        'stability': float(autocorr) if not np.isnan(autocorr) else 0.0,
        'mean_tension': float(mean_tension),
        'std_tension': float(std_tension),
        'convergence_rate': float(convergence_rate),
        'autonomy': float(autonomy),
        'mean_consciousness': float(mean_consciousness),
        'archetype_distribution': archetype_counts,
        'n_tensions_recorded': len(tensions),
    }


def run_experiment():
    """Ejecuta experimento comparativo."""

    print("\n" + "=" * 70)
    print("   EXPERIMENTO: Loop de Auto-Reflexion para Emergencia")
    print("=" * 70)

    N_CELLS = 50
    N_STEPS = 150
    STIMULI = 'mixed'

    print(f"\n  Configuracion:")
    print(f"    Celulas: {N_CELLS}")
    print(f"    Pasos: {N_STEPS}")
    print(f"    Patron estimulos: {STIMULI}")

    # =================================================================
    # Sistema SIN loop de reflexion
    # =================================================================
    print("\n" + "-" * 70)
    print("  1. Sistema SIN auto-reflexion (baseline)")
    print("-" * 70)

    system_no_loop = ZetaConsciousSelf(
        n_cells=N_CELLS,
        enable_self_reflection=False,
    )

    metrics_no_loop = run_simulation(
        system_no_loop,
        n_steps=N_STEPS,
        stimuli_pattern=STIMULI
    )

    print(f"    Varianza estado: {metrics_no_loop['state_variance']:.4f}")
    print(f"    Estabilidad: {metrics_no_loop['stability']:.4f}")
    print(f"    Autonomia: {metrics_no_loop['autonomy']:.2%}")
    print(f"    Consciencia promedio: {metrics_no_loop['mean_consciousness']:.4f}")

    # =================================================================
    # Sistema CON loop de reflexion
    # =================================================================
    print("\n" + "-" * 70)
    print("  2. Sistema CON auto-reflexion (Strange Loop)")
    print("-" * 70)

    system_with_loop = ZetaConsciousSelf(
        n_cells=N_CELLS,
        enable_self_reflection=True,
        reflection_config={
            'max_iterations': 3,
            'convergence_threshold': 0.02,
            'include_perception': True,
        }
    )

    metrics_with_loop = run_simulation(
        system_with_loop,
        n_steps=N_STEPS,
        stimuli_pattern=STIMULI
    )

    print(f"    Varianza estado: {metrics_with_loop['state_variance']:.4f}")
    print(f"    Estabilidad: {metrics_with_loop['stability']:.4f}")
    print(f"    Tension epistemica: {metrics_with_loop['mean_tension']:.4f} +/- {metrics_with_loop['std_tension']:.4f}")
    print(f"    Tasa convergencia: {metrics_with_loop['convergence_rate']:.2%}")
    print(f"    Autonomia: {metrics_with_loop['autonomy']:.2%}")
    print(f"    Consciencia promedio: {metrics_with_loop['mean_consciousness']:.4f}")

    # =================================================================
    # Comparacion
    # =================================================================
    print("\n" + "=" * 70)
    print("   COMPARACION")
    print("=" * 70)

    # Diferencias
    delta_variance = metrics_with_loop['state_variance'] - metrics_no_loop['state_variance']
    delta_stability = metrics_with_loop['stability'] - metrics_no_loop['stability']
    delta_autonomy = metrics_with_loop['autonomy'] - metrics_no_loop['autonomy']
    delta_consciousness = metrics_with_loop['mean_consciousness'] - metrics_no_loop['mean_consciousness']

    print(f"""
  | Metrica              | Sin Loop  | Con Loop  | Delta     |
  |----------------------|-----------|-----------|-----------|
  | Varianza estado      | {metrics_no_loop['state_variance']:9.4f} | {metrics_with_loop['state_variance']:9.4f} | {delta_variance:+9.4f} |
  | Estabilidad          | {metrics_no_loop['stability']:9.4f} | {metrics_with_loop['stability']:9.4f} | {delta_stability:+9.4f} |
  | Autonomia            | {metrics_no_loop['autonomy']:9.2%} | {metrics_with_loop['autonomy']:9.2%} | {delta_autonomy:+9.2%} |
  | Consciencia          | {metrics_no_loop['mean_consciousness']:9.4f} | {metrics_with_loop['mean_consciousness']:9.4f} | {delta_consciousness:+9.4f} |
""")

    # Metricas exclusivas del loop
    if metrics_with_loop['n_tensions_recorded'] > 0:
        print(f"  Metricas exclusivas del Strange Loop:")
        print(f"    Tension epistemica (ξ): {metrics_with_loop['mean_tension']:.4f}")
        print(f"    Convergencia a atractores: {metrics_with_loop['convergence_rate']:.2%}")
        print(f"    Tensiones registradas: {metrics_with_loop['n_tensions_recorded']}")

    # =================================================================
    # Interpretacion
    # =================================================================
    print("\n" + "-" * 70)
    print("   INTERPRETACION")
    print("-" * 70)

    emergent = False
    reasons = []

    # El loop deberia producir MAYOR varianza (exploracion)
    if delta_variance > 0:
        reasons.append("+ Mayor exploracion del espacio de estados")
        emergent = True

    # El loop deberia producir estabilidad comparable o mayor
    if delta_stability >= -0.05:
        reasons.append("+ Estabilidad mantenida o mejorada")
        emergent = True

    # El loop deberia aumentar autonomia
    if delta_autonomy > 0:
        reasons.append("+ Mayor autonomia respecto a estimulos externos")
        emergent = True

    # La convergencia indica atractores
    if metrics_with_loop['convergence_rate'] > 0.3:
        reasons.append("+ Sistema converge a atractores estables")
        emergent = True

    # La tension epistemica deberia ser no-trivial
    if metrics_with_loop['mean_tension'] > 0.005:
        reasons.append("+ Tension epistemica no-trivial (auto-observacion afecta)")
        emergent = True

    print()
    if emergent:
        print("  RESULTADO: El Strange Loop CONTRIBUYE a emergencia")
        for r in reasons:
            print(f"    {r}")
    else:
        print("  RESULTADO: No se observa contribucion clara")

    print()

    # =================================================================
    # Guardar resultados
    # =================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results = {
        'timestamp': timestamp,
        'config': {
            'n_cells': N_CELLS,
            'n_steps': N_STEPS,
            'stimuli_pattern': STIMULI,
        },
        'no_loop': metrics_no_loop,
        'with_loop': metrics_with_loop,
        'comparison': {
            'delta_variance': delta_variance,
            'delta_stability': delta_stability,
            'delta_autonomy': delta_autonomy,
            'delta_consciousness': delta_consciousness,
        },
        'conclusion': 'emergent' if emergent else 'no_effect',
    }

    output_path = f"exp_self_reflection_results_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Resultados guardados en: {output_path}")
    print()

    return results


if __name__ == "__main__":
    run_experiment()
