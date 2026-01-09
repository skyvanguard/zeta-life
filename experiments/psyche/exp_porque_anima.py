# -*- coding: utf-8 -*-
"""
Investigacion: ¿Por que ZetaPsyche se estabiliza en ANIMA?
==========================================================

Hipotesis a probar:
1. INICIALIZACION: Las posiciones aleatorias favorecen ANIMA
2. RED NEURONAL: Los pesos no entrenados tienen sesgo hacia ANIMA
3. ZETA MODULATION: Las oscilaciones zeta crean atractor en ANIMA
4. SOFTMAX: La normalizacion crea sesgo numerico
5. GEOMETRIA: El vertice ANIMA tiene propiedades especiales en el tetraedro
"""
import sys
import os
if sys.platform == 'win32':
    os.system('')

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

from zeta_life.psyche import ZetaPsyche, Archetype, TetrahedralSpace, ZetaModulator


def test_inicializacion():
    """Test: ¿La inicializacion favorece ANIMA?"""
    print("\n" + "=" * 60)
    print("TEST 1: INICIALIZACION")
    print("=" * 60)

    # Crear muchas psiques y ver donde empiezan
    n_tests = 20
    initial_dominants = []

    for _ in range(n_tests):
        psyche = ZetaPsyche(n_cells=50)
        obs = psyche.observe_self()
        initial_dominants.append(obs['dominant'].name)

    counts = Counter(initial_dominants)
    print(f"\n  Dominante inicial en {n_tests} psiques:")
    for arch, count in counts.most_common():
        print(f"    {arch}: {count} ({count/n_tests:.0%})")

    # Conclusion
    if counts.most_common(1)[0][0] == 'ANIMA':
        print("\n  >> ANIMA es favorecida en inicializacion")
        return True
    else:
        print(f"\n  >> {counts.most_common(1)[0][0]} es favorecida, no ANIMA")
        return False


def test_random_softmax():
    """Test: ¿El softmax de valores random favorece algun indice?"""
    print("\n" + "=" * 60)
    print("TEST 2: SOFTMAX DE RANDOM")
    print("=" * 60)

    n_samples = 10000
    dominant_counts = [0, 0, 0, 0]

    for _ in range(n_samples):
        x = torch.rand(4)
        y = F.softmax(x, dim=-1)
        dominant = y.argmax().item()
        dominant_counts[dominant] += 1

    archetypes = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
    print(f"\n  Dominante despues de softmax({n_samples} samples):")
    for i, name in enumerate(archetypes):
        print(f"    {name}: {dominant_counts[i]} ({dominant_counts[i]/n_samples:.1%})")

    # Deberia ser ~25% cada uno
    expected = n_samples / 4
    deviations = [abs(c - expected) / expected for c in dominant_counts]
    max_deviation = max(deviations)

    if max_deviation < 0.1:  # Menos de 10% desviacion
        print("\n  >> Softmax es uniforme, no hay sesgo")
        return False
    else:
        biased = archetypes[dominant_counts.index(max(dominant_counts))]
        print(f"\n  >> Hay sesgo hacia {biased}")
        return True


def test_red_neuronal():
    """Test: ¿Los pesos de la red no entrenada favorecen ANIMA?"""
    print("\n" + "=" * 60)
    print("TEST 3: RED NEURONAL NO ENTRENADA")
    print("=" * 60)

    psyche = ZetaPsyche(n_cells=1)  # Solo una celula

    # Probar con estimulos uniformes para cada arquetipo
    archetypes = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
    stimuli = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    ]

    responses = []
    for name, stim in zip(archetypes, stimuli):
        # Reset
        psyche._init_cells()

        # Aplicar estimulo 50 veces
        for _ in range(50):
            psyche.step(stim)

        obs = psyche.observe_self()
        response = obs['dominant'].name
        responses.append((name, response))
        print(f"  Estimulo {name} -> Respuesta {response}")

    # Ver si hay sesgo
    response_counts = Counter([r[1] for r in responses])
    if len(response_counts) == 1:
        biased = list(response_counts.keys())[0]
        print(f"\n  >> La red siempre responde {biased}, independiente del estimulo!")
        return True
    else:
        print("\n  >> La red responde diferente a diferentes estimulos")
        return False


def test_zeta_modulation():
    """Test: ¿La modulacion zeta crea atractor en ANIMA?"""
    print("\n" + "=" * 60)
    print("TEST 4: MODULACION ZETA")
    print("=" * 60)

    zeta = ZetaModulator(M=15, sigma=0.1)

    # Probar modulacion sobre vectores unitarios
    archetypes = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
    vectors = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    ]

    print("\n  Modulacion zeta sobre 100 pasos:")

    for name, vec in zip(archetypes, vectors):
        zeta.t = 0  # Reset time
        modulated_sums = []

        for _ in range(100):
            modulated = zeta(vec.clone())
            modulated_sums.append(modulated.sum().item())

        avg = np.mean(modulated_sums)
        print(f"    {name}: promedio modulado = {avg:.4f}")

    print("\n  >> Si todos son iguales, zeta no crea sesgo espacial")
    return False


def test_geometria_tetraedro():
    """Test: ¿El vertice ANIMA tiene propiedades geometricas especiales?"""
    print("\n" + "=" * 60)
    print("TEST 5: GEOMETRIA DEL TETRAEDRO")
    print("=" * 60)

    space = TetrahedralSpace()

    print("\n  Vertices del tetraedro:")
    archetypes = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
    for i, name in enumerate(archetypes):
        v = space.vertices[i]
        print(f"    {name}: {v.numpy()}")

    print(f"\n  Centro: {space.center.numpy()}")

    # Distancia de cada vertice al centro
    print("\n  Distancia al centro:")
    for i, name in enumerate(archetypes):
        v = space.vertices[i]
        dist = torch.norm(v - space.center).item()
        print(f"    {name}: {dist:.4f}")

    # El tetraedro es regular, todas las distancias deberian ser iguales
    print("\n  >> En tetraedro regular, todas las distancias son iguales")
    return False


def test_dinamica_atraccion():
    """Test: ¿La dinamica de atraccion favorece ANIMA?"""
    print("\n" + "=" * 60)
    print("TEST 6: DINAMICA DE ATRACCION")
    print("=" * 60)

    # La ecuacion de movimiento es:
    # total_delta = delta (red) + attraction (estimulo) + noise + repulsion

    # Probar con una celula en el centro
    psyche = ZetaPsyche(n_cells=1)

    # Poner celula en centro exacto
    psyche.cells[0].position = torch.tensor([0.25, 0.25, 0.25, 0.25])

    # Sin estimulo (estimulo uniforme)
    uniform_stim = torch.tensor([0.25, 0.25, 0.25, 0.25])

    trajectories = {'PERSONA': [], 'SOMBRA': [], 'ANIMA': [], 'ANIMUS': []}

    for step in range(100):
        # Guardar posicion actual
        pos = psyche.cells[0].position.clone()
        for i, name in enumerate(['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']):
            trajectories[name].append(pos[i].item())

        psyche.step(uniform_stim)

    print("\n  Posicion final partiendo del centro con estimulo uniforme:")
    final_pos = psyche.cells[0].position
    for i, name in enumerate(['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']):
        print(f"    {name}: {final_pos[i].item():.3f}")

    dominant = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS'][final_pos.argmax().item()]
    print(f"\n  >> Dominante final: {dominant}")

    if dominant == 'ANIMA':
        print("  >> La dinamica naturalmente deriva hacia ANIMA!")
        return True
    return False


def test_pesos_movement():
    """Test: ¿Los pesos de la red 'movement' favorecen ANIMA?"""
    print("\n" + "=" * 60)
    print("TEST 7: PESOS DE LA RED MOVEMENT")
    print("=" * 60)

    psyche = ZetaPsyche(n_cells=1)

    # Ver los pesos de la ultima capa de movement
    # movement: Linear(hidden_dim, 4) -> Tanh
    movement_weights = list(psyche.movement.parameters())

    if len(movement_weights) >= 2:
        weights = movement_weights[0].data  # Shape: [4, hidden_dim]
        bias = movement_weights[1].data     # Shape: [4]

        print("\n  Bias de la capa movement (hacia cada arquetipo):")
        archetypes = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
        for i, name in enumerate(archetypes):
            print(f"    {name}: {bias[i].item():.4f}")

        print("\n  Suma de pesos absolutos por arquetipo:")
        for i, name in enumerate(archetypes):
            weight_sum = weights[i].abs().sum().item()
            print(f"    {name}: {weight_sum:.4f}")

        # Ver si hay sesgo
        bias_idx = bias.argmax().item()
        print(f"\n  >> Mayor bias hacia: {archetypes[bias_idx]}")

        if archetypes[bias_idx] == 'ANIMA':
            return True

    return False


def test_multiples_runs():
    """Test: ¿Siempre termina en ANIMA o varia?"""
    print("\n" + "=" * 60)
    print("TEST 8: MULTIPLES EJECUCIONES (con diferentes seeds)")
    print("=" * 60)

    final_dominants = []

    for seed in range(10):
        torch.manual_seed(seed)
        np.random.seed(seed)

        psyche = ZetaPsyche(n_cells=50)

        # Correr 200 pasos con estimulo uniforme
        uniform_stim = torch.tensor([0.25, 0.25, 0.25, 0.25])
        for _ in range(200):
            psyche.step(uniform_stim)

        obs = psyche.observe_self()
        final_dominants.append(obs['dominant'].name)

    counts = Counter(final_dominants)
    print(f"\n  Dominante final en 10 seeds diferentes:")
    for arch, count in counts.most_common():
        print(f"    {arch}: {count}")

    if len(counts) == 1:
        print(f"\n  >> SIEMPRE termina en {list(counts.keys())[0]}!")
        return True
    else:
        print("\n  >> Varia segun la seed")
        return False


def main():
    print("\n" + "=" * 70)
    print("   INVESTIGACION: ¿POR QUE ZETA PSYCHE SE ESTABILIZA EN ANIMA?")
    print("=" * 70)

    results = {}

    results['inicializacion'] = test_inicializacion()
    results['softmax'] = test_random_softmax()
    results['red_neuronal'] = test_red_neuronal()
    results['zeta'] = test_zeta_modulation()
    results['geometria'] = test_geometria_tetraedro()
    results['dinamica'] = test_dinamica_atraccion()
    results['pesos'] = test_pesos_movement()
    results['seeds'] = test_multiples_runs()

    # Resumen
    print("\n" + "=" * 70)
    print("   RESUMEN DE HALLAZGOS")
    print("=" * 70)

    causes = [k for k, v in results.items() if v]

    if causes:
        print(f"\n  CAUSAS IDENTIFICADAS:")
        for cause in causes:
            print(f"    - {cause}")
    else:
        print("\n  No se identifico una causa clara.")
        print("  El sesgo puede ser una combinacion de factores sutiles.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
