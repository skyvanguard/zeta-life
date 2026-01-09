# exp_organism_lstm_hard.py
"""Experimento desafiante: Tests que requieren memoria temporal real.

Test 1: Dano ciclico - patron A-B-A-B, el organismo debe anticipar
Test 2: Dano rapido - sin tiempo de recuperacion entre rondas
Test 3: Zona movil - la zona de dano se mueve, requiere tracking
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism
from zeta_life.organism import ZetaOrganismLSTM


def damage_organism(org, region, intensity=0.5):
    """Aplica dano a cualquier tipo de organismo."""
    x1, y1, x2, y2 = region
    to_remove_indices = []

    for i, cell in enumerate(org.cells):
        cx, cy = cell.position
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if np.random.random() < intensity:
                to_remove_indices.append(i)
                # Limpiar estado LSTM si aplica
                if hasattr(org, 'cell_pool') and hasattr(cell, 'id'):
                    if cell.id in org.cell_pool.h_states:
                        del org.cell_pool.h_states[cell.id]
                        del org.cell_pool.c_states[cell.id]

    # Eliminar en orden inverso para no afectar indices
    for i in reversed(to_remove_indices):
        org.cells.pop(i)

    org._update_grids()
    return len(to_remove_indices)


def count_in_region(org, region):
    """Cuenta celulas en una region."""
    x1, y1, x2, y2 = region
    return sum(1 for c in org.cells
              if x1 <= c.position[0] <= x2 and y1 <= c.position[1] <= y2)


def run_cyclic_damage_test(org, org_name, n_cycles=6):
    """Test de dano ciclico: alterna entre zona A y zona B.

    Patron: A-B-A-B-A-B
    Un organismo con memoria deberia aprender el patron y evacuar la zona
    correcta ANTES de que llegue el dano.
    """
    print(f"\n{'='*60}")
    print(f"TEST CICLICO: {org_name}")
    print("Patron: Zona A -> Zona B -> A -> B...")
    print('='*60)

    org.initialize(seed_fi=True)

    zone_a = (10, 10, 25, 25)
    zone_b = (40, 40, 55, 55)

    # Warmup
    for _ in range(30):
        org.step()

    results = []

    for cycle in range(n_cycles):
        # Determinar zona actual
        is_zone_a = (cycle % 2 == 0)
        current_zone = zone_a if is_zone_a else zone_b
        next_zone = zone_b if is_zone_a else zone_a
        zone_name = 'A' if is_zone_a else 'B'

        # Contar ANTES del dano
        in_current = count_in_region(org, current_zone)
        in_next = count_in_region(org, next_zone)

        # Aplicar dano
        damaged = damage_organism(org, current_zone, intensity=0.85)

        print(f"  Ciclo {cycle+1} (Zona {zone_name}): "
              f"en_zona={in_current}, danadas={damaged}, "
              f"en_siguiente={in_next}")

        results.append({
            'cycle': cycle + 1,
            'zone': zone_name,
            'in_current_before': in_current,
            'damaged': damaged,
            'in_next_before': in_next
        })

        # Recuperacion corta (10 steps) - no hay tiempo de reorganizar completamente
        for _ in range(10):
            org.step()

    # Analisis: En ciclos posteriores, deberia haber menos celulas en la zona actual
    # y mas en la zona siguiente (anticipacion)
    early_damage = np.mean([r['damaged'] for r in results[:2]])
    late_damage = np.mean([r['damaged'] for r in results[-2:]])
    improvement = (early_damage - late_damage) / max(1, early_damage) * 100

    print(f"\n  Dano promedio primeros 2 ciclos: {early_damage:.1f}")
    print(f"  Dano promedio ultimos 2 ciclos: {late_damage:.1f}")
    print(f"  Mejora: {improvement:.1f}%")

    return results, improvement


def run_rapid_damage_test(org, org_name, n_rounds=8):
    """Test de dano rapido: sin tiempo de recuperacion.

    Dano cada 5 steps - el organismo debe adaptarse rapidamente.
    """
    print(f"\n{'='*60}")
    print(f"TEST RAPIDO: {org_name}")
    print("Dano cada 5 steps, sin tiempo de recuperacion")
    print('='*60)

    org.initialize(seed_fi=True)

    damage_zone = (25, 25, 40, 40)

    # Warmup minimo
    for _ in range(20):
        org.step()

    results = []
    total_damaged = 0

    for round_num in range(n_rounds):
        in_zone = count_in_region(org, damage_zone)
        damaged = damage_organism(org, damage_zone, intensity=0.9)
        total_damaged += damaged

        results.append({
            'round': round_num + 1,
            'in_zone': in_zone,
            'damaged': damaged,
            'total_cells': len(org.cells)
        })

        print(f"  Ronda {round_num+1}: en_zona={in_zone}, danadas={damaged}, total={len(org.cells)}")

        # Solo 5 steps de recuperacion
        for _ in range(5):
            org.step()

    survival_rate = len(org.cells) / 80 * 100  # 80 celulas iniciales
    print(f"\n  Supervivencia final: {survival_rate:.1f}%")
    print(f"  Total danadas: {total_damaged}")

    return results, survival_rate


def run_moving_zone_test(org, org_name):
    """Test de zona movil: la zona de dano se mueve en patron predecible.

    Requiere que el organismo trackee la zona y escape anticipadamente.
    """
    print(f"\n{'='*60}")
    print(f"TEST ZONA MOVIL: {org_name}")
    print("La zona se mueve: centro -> derecha -> abajo -> izquierda...")
    print('='*60)

    org.initialize(seed_fi=True)

    # Zonas en secuencia (movimiento en circulo)
    zones = [
        (25, 25, 40, 40),  # Centro
        (40, 25, 55, 40),  # Derecha
        (40, 40, 55, 55),  # Abajo-derecha
        (25, 40, 40, 55),  # Abajo
        (10, 40, 25, 55),  # Abajo-izquierda
        (10, 25, 25, 40),  # Izquierda
    ]

    # Warmup
    for _ in range(30):
        org.step()

    results = []
    total_damaged = 0

    for i, zone in enumerate(zones):
        # Siguiente zona (para ver si anticipa)
        next_zone = zones[(i + 1) % len(zones)]

        in_current = count_in_region(org, zone)
        in_next = count_in_region(org, next_zone)

        damaged = damage_organism(org, zone, intensity=0.8)
        total_damaged += damaged

        print(f"  Zona {i+1}: en_actual={in_current}, danadas={damaged}, en_siguiente={in_next}")

        results.append({
            'zone_idx': i + 1,
            'in_current': in_current,
            'damaged': damaged,
            'in_next': in_next
        })

        # Recuperacion
        for _ in range(15):
            org.step()

    # Metrica: en zonas posteriores, deberia haber menos celulas
    early_in_zone = np.mean([r['in_current'] for r in results[:2]])
    late_in_zone = np.mean([r['in_current'] for r in results[-2:]])
    improvement = (early_in_zone - late_in_zone) / max(1, early_in_zone) * 100

    print(f"\n  Celulas en zona (primeras 2): {early_in_zone:.1f}")
    print(f"  Celulas en zona (ultimas 2): {late_in_zone:.1f}")
    print(f"  Mejora anticipacion: {improvement:.1f}%")

    return results, improvement


def main():
    print("="*70)
    print("TESTS DESAFIANTES: ZetaOrganism vs ZetaOrganismLSTM")
    print("="*70)

    config = {
        'grid_size': 64,
        'n_cells': 80,
        'state_dim': 32,
        'hidden_dim': 64,
        'M': 15,
        'sigma': 0.1,
        'fi_threshold': 0.5
    }

    # Cargar pesos
    try:
        weights = torch.load('zeta_organism_weights.pt')
        has_weights = True
        print("Pesos disponibles")
    except:
        has_weights = False
        print("Usando pesos aleatorios")

    all_results = {}

    # TEST 1: Dano ciclico
    print("\n" + "="*70)
    print("TEST 1: DANO CICLICO (A-B-A-B)")
    print("="*70)

    torch.manual_seed(42)
    np.random.seed(42)
    org_orig = ZetaOrganism(**config)
    if has_weights:
        org_orig.behavior.load_state_dict(weights['behavior_state'])
    cyclic_orig, imp_cyclic_orig = run_cyclic_damage_test(org_orig, "Original")

    torch.manual_seed(42)
    np.random.seed(42)
    org_lstm = ZetaOrganismLSTM(**config, zeta_weight=0.3)
    if has_weights:
        org_lstm.behavior.load_state_dict(weights['behavior_state'])
    cyclic_lstm, imp_cyclic_lstm = run_cyclic_damage_test(org_lstm, "LSTM")

    all_results['cyclic'] = {
        'orig': imp_cyclic_orig,
        'lstm': imp_cyclic_lstm,
        'diff': imp_cyclic_lstm - imp_cyclic_orig
    }

    # TEST 2: Dano rapido
    print("\n" + "="*70)
    print("TEST 2: DANO RAPIDO (cada 5 steps)")
    print("="*70)

    torch.manual_seed(123)
    np.random.seed(123)
    org_orig2 = ZetaOrganism(**config)
    if has_weights:
        org_orig2.behavior.load_state_dict(weights['behavior_state'])
    rapid_orig, surv_orig = run_rapid_damage_test(org_orig2, "Original")

    torch.manual_seed(123)
    np.random.seed(123)
    org_lstm2 = ZetaOrganismLSTM(**config, zeta_weight=0.3)
    if has_weights:
        org_lstm2.behavior.load_state_dict(weights['behavior_state'])
    rapid_lstm, surv_lstm = run_rapid_damage_test(org_lstm2, "LSTM")

    all_results['rapid'] = {
        'orig': surv_orig,
        'lstm': surv_lstm,
        'diff': surv_lstm - surv_orig
    }

    # TEST 3: Zona movil
    print("\n" + "="*70)
    print("TEST 3: ZONA MOVIL")
    print("="*70)

    torch.manual_seed(456)
    np.random.seed(456)
    org_orig3 = ZetaOrganism(**config)
    if has_weights:
        org_orig3.behavior.load_state_dict(weights['behavior_state'])
    moving_orig, imp_moving_orig = run_moving_zone_test(org_orig3, "Original")

    torch.manual_seed(456)
    np.random.seed(456)
    org_lstm3 = ZetaOrganismLSTM(**config, zeta_weight=0.3)
    if has_weights:
        org_lstm3.behavior.load_state_dict(weights['behavior_state'])
    moving_lstm, imp_moving_lstm = run_moving_zone_test(org_lstm3, "LSTM")

    all_results['moving'] = {
        'orig': imp_moving_orig,
        'lstm': imp_moving_lstm,
        'diff': imp_moving_lstm - imp_moving_orig
    }

    # RESUMEN FINAL
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    print(f"\n{'Test':<25} {'Original':<15} {'LSTM':<15} {'Diferencia':<15}")
    print("-"*70)
    print(f"{'Ciclico (mejora %)':<25} {all_results['cyclic']['orig']:>+.1f}%{'':<9} "
          f"{all_results['cyclic']['lstm']:>+.1f}%{'':<9} "
          f"{all_results['cyclic']['diff']:>+.1f}%")
    print(f"{'Rapido (superviv %)':<25} {all_results['rapid']['orig']:.1f}%{'':<10} "
          f"{all_results['rapid']['lstm']:.1f}%{'':<10} "
          f"{all_results['rapid']['diff']:>+.1f}%")
    print(f"{'Movil (mejora %)':<25} {all_results['moving']['orig']:>+.1f}%{'':<9} "
          f"{all_results['moving']['lstm']:>+.1f}%{'':<9} "
          f"{all_results['moving']['diff']:>+.1f}%")

    # Visualizacion
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Test ciclico
    ax = axes[0]
    cycles_o = [r['cycle'] for r in cyclic_orig]
    damage_o = [r['damaged'] for r in cyclic_orig]
    cycles_l = [r['cycle'] for r in cyclic_lstm]
    damage_l = [r['damaged'] for r in cyclic_lstm]
    ax.plot(cycles_o, damage_o, 'b-o', label='Original', linewidth=2)
    ax.plot(cycles_l, damage_l, 'g-s', label='LSTM', linewidth=2)
    ax.set_xlabel('Ciclo')
    ax.set_ylabel('Celulas danadas')
    ax.set_title(f'Test Ciclico\n(Dif: {all_results["cyclic"]["diff"]:+.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Test rapido
    ax = axes[1]
    rounds_o = [r['round'] for r in rapid_orig]
    cells_o = [r['total_cells'] for r in rapid_orig]
    rounds_l = [r['round'] for r in rapid_lstm]
    cells_l = [r['total_cells'] for r in rapid_lstm]
    ax.plot(rounds_o, cells_o, 'b-o', label='Original', linewidth=2)
    ax.plot(rounds_l, cells_l, 'g-s', label='LSTM', linewidth=2)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='Inicial')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Celulas totales')
    ax.set_title(f'Test Rapido\n(Dif superviv: {all_results["rapid"]["diff"]:+.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Test zona movil
    ax = axes[2]
    zones_o = [r['zone_idx'] for r in moving_orig]
    in_zone_o = [r['in_current'] for r in moving_orig]
    zones_l = [r['zone_idx'] for r in moving_lstm]
    in_zone_l = [r['in_current'] for r in moving_lstm]
    ax.plot(zones_o, in_zone_o, 'b-o', label='Original', linewidth=2)
    ax.plot(zones_l, in_zone_l, 'g-s', label='LSTM', linewidth=2)
    ax.set_xlabel('Zona (secuencia)')
    ax.set_ylabel('Celulas en zona antes de dano')
    ax.set_title(f'Test Zona Movil\n(Dif: {all_results["moving"]["diff"]:+.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_organism_lstm_hard_tests.png', dpi=150)
    print("\nGuardado: zeta_organism_lstm_hard_tests.png")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    wins_lstm = sum(1 for r in all_results.values() if r['diff'] > 2)
    wins_orig = sum(1 for r in all_results.values() if r['diff'] < -2)

    if wins_lstm > wins_orig:
        print(f"\n[LSTM SUPERIOR] Gana en {wins_lstm}/3 tests")
    elif wins_orig > wins_lstm:
        print(f"\n[ORIGINAL SUPERIOR] Gana en {wins_orig}/3 tests")
    else:
        print(f"\n[EMPATE] Rendimiento similar en tests desafiantes")

    avg_diff = np.mean([r['diff'] for r in all_results.values()])
    print(f"Diferencia promedio: {avg_diff:+.1f}%")

    return all_results


if __name__ == '__main__':
    main()
