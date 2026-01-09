# exp_organism_lstm_comparison.py
"""Experimento: Comparar ZetaOrganism vs ZetaOrganismLSTM.

Hipotesis: El organismo con memoria LSTM deberia mostrar mejor:
1. Anticipacion de dano (evacuacion preventiva)
2. Recuperacion post-dano (memoria de eventos pasados)
3. Coordinacion a largo plazo
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism, CellEntity
from zeta_life.organism import ZetaOrganismLSTM, CellEntityLSTM


def damage_organism(org, region, intensity=0.5):
    """Aplica dano a cualquier tipo de organismo."""
    x1, y1, x2, y2 = region
    damaged = []

    for cell in org.cells:
        cx, cy = cell.position
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if np.random.random() < intensity:
                damaged.append(cell)

    for cell in damaged:
        org.cells.remove(cell)
        # Limpiar estado LSTM si aplica
        if hasattr(org, 'cell_pool') and hasattr(cell, 'id'):
            if cell.id in org.cell_pool.h_states:
                del org.cell_pool.h_states[cell.id]
                del org.cell_pool.c_states[cell.id]

    org._update_grids()
    return len(damaged)


def run_memory_test(org, org_name, damage_region, n_rounds=4):
    """Test de memoria temporal: dano repetido en misma zona.

    Un organismo con buena memoria deberia:
    - Aprender a evacuar la zona peligrosa
    - Mostrar menos dano en rondas posteriores
    """
    print(f"\n{'='*60}")
    print(f"MEMORIA TEMPORAL: {org_name}")
    print('='*60)

    org.initialize(seed_fi=True)

    # Warmup
    for _ in range(50):
        org.step()

    results = {
        'round': [],
        'cells_before': [],
        'cells_damaged': [],
        'cells_in_zone_before': [],
        'fi_damaged': [],
        'recovery_rate': []
    }

    x1, y1, x2, y2 = damage_region

    for round_num in range(n_rounds):
        print(f"\n  Ronda {round_num + 1}:")

        # Contar celulas en zona antes del dano
        cells_in_zone = sum(1 for c in org.cells
                          if x1 <= c.position[0] <= x2 and y1 <= c.position[1] <= y2)
        fi_in_zone = sum(1 for c in org.cells
                        if x1 <= c.position[0] <= x2 and y1 <= c.position[1] <= y2
                        and c.role_idx == 1)

        cells_before = len(org.cells)
        print(f"    Antes: {cells_before} celulas, {cells_in_zone} en zona, {fi_in_zone} Fi en zona")

        # Aplicar dano
        damaged = damage_organism(org, damage_region, intensity=0.9)
        print(f"    Dano: {damaged} celulas eliminadas")

        cells_after = len(org.cells)

        # Recuperacion
        for _ in range(30):
            org.step()

        cells_recovered = len(org.cells)
        recovery = (cells_recovered - cells_after) / max(1, damaged) if damaged > 0 else 1.0

        print(f"    Despues recuperacion: {cells_recovered} celulas")

        results['round'].append(round_num + 1)
        results['cells_before'].append(cells_before)
        results['cells_damaged'].append(damaged)
        results['cells_in_zone_before'].append(cells_in_zone)
        results['fi_damaged'].append(fi_in_zone)
        results['recovery_rate'].append(recovery)

    # Metricas de anticipacion
    # Si el organismo aprende, deberia haber menos celulas en la zona en rondas posteriores
    first_round_in_zone = results['cells_in_zone_before'][0]
    last_round_in_zone = results['cells_in_zone_before'][-1]
    evacuation_improvement = (first_round_in_zone - last_round_in_zone) / max(1, first_round_in_zone) * 100

    print(f"\n  RESUMEN {org_name}:")
    print(f"    Celulas en zona ronda 1: {first_round_in_zone}")
    print(f"    Celulas en zona ronda {n_rounds}: {last_round_in_zone}")
    print(f"    Mejora evacuacion: {evacuation_improvement:.1f}%")

    return results, evacuation_improvement


def run_coordination_test(org, org_name, n_steps=100):
    """Test de coordinacion: medir coherencia del grupo a lo largo del tiempo."""
    print(f"\n{'='*60}")
    print(f"COORDINACION: {org_name}")
    print('='*60)

    org.initialize(seed_fi=True)

    coordination_history = []

    for step in range(n_steps):
        org.step()

        if (step + 1) % 10 == 0:
            # Calcular dispersion espacial (menor = mas coordinado)
            positions = np.array([c.position for c in org.cells])
            centroid = positions.mean(axis=0)
            dispersion = np.sqrt(((positions - centroid)**2).sum(axis=1)).mean()

            # Calcular cohesion de roles
            n_fi = sum(1 for c in org.cells if c.role_idx == 1)
            role_stability = n_fi / max(1, len(org.cells))

            coordination = 1.0 / (1.0 + dispersion / 10)  # Normalizar

            coordination_history.append({
                'step': step + 1,
                'dispersion': dispersion,
                'coordination': coordination,
                'n_fi': n_fi
            })

    avg_coord = np.mean([h['coordination'] for h in coordination_history])
    print(f"  Coordinacion promedio: {avg_coord:.4f}")

    return coordination_history, avg_coord


def run_h_norm_analysis(org_lstm, n_steps=100):
    """Analiza evolucion de h_norm en organismo LSTM."""
    print(f"\n{'='*60}")
    print(f"EVOLUCION MEMORIA LSTM")
    print('='*60)

    org_lstm.initialize(seed_fi=True)

    h_norm_history = []

    for step in range(n_steps):
        org_lstm.step()

        if (step + 1) % 10 == 0:
            m = org_lstm.get_metrics()
            h_norm_history.append({
                'step': step + 1,
                'h_norm': m['avg_h_norm'],
                'n_fi': m['n_fi']
            })
            print(f"  Step {step+1}: h_norm={m['avg_h_norm']:.4f}, Fi={m['n_fi']}")

    return h_norm_history


def main():
    print("="*70)
    print("COMPARACION: ZetaOrganism vs ZetaOrganismLSTM")
    print("="*70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Configuracion comun
    config = {
        'grid_size': 64,
        'n_cells': 80,
        'state_dim': 32,
        'hidden_dim': 64,
        'M': 15,
        'sigma': 0.1,
        'fi_threshold': 0.5
    }

    # Crear organismos
    org_original = ZetaOrganism(**config)
    org_lstm = ZetaOrganismLSTM(**config, zeta_weight=0.2)

    # Cargar pesos si existen
    try:
        weights = torch.load('zeta_organism_weights.pt')
        org_original.behavior.load_state_dict(weights['behavior_state'])
        org_lstm.behavior.load_state_dict(weights['behavior_state'])
        print("Pesos cargados!")
    except:
        print("Usando pesos aleatorios")

    # Test 1: Memoria temporal
    damage_region = (25, 25, 40, 40)

    torch.manual_seed(42)
    np.random.seed(42)
    results_orig, evac_orig = run_memory_test(org_original, "Original", damage_region)

    torch.manual_seed(42)
    np.random.seed(42)
    results_lstm, evac_lstm = run_memory_test(org_lstm, "LSTM", damage_region)

    # Test 2: Coordinacion
    torch.manual_seed(42)
    np.random.seed(42)
    org_original2 = ZetaOrganism(**config)
    coord_orig, avg_coord_orig = run_coordination_test(org_original2, "Original")

    torch.manual_seed(42)
    np.random.seed(42)
    org_lstm2 = ZetaOrganismLSTM(**config, zeta_weight=0.2)
    coord_lstm, avg_coord_lstm = run_coordination_test(org_lstm2, "LSTM")

    # Test 3: Evolucion h_norm
    torch.manual_seed(42)
    np.random.seed(42)
    org_lstm3 = ZetaOrganismLSTM(**config, zeta_weight=0.2)
    h_norm_history = run_h_norm_analysis(org_lstm3)

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO")
    print("="*70)

    print(f"\n{'Metrica':<30} {'Original':<15} {'LSTM':<15} {'Diferencia':<15}")
    print("-"*75)

    print(f"{'Evacuacion preventiva':<30} {evac_orig:>+.1f}%{'':<9} {evac_lstm:>+.1f}%{'':<9} {evac_lstm - evac_orig:>+.1f}%")
    print(f"{'Coordinacion promedio':<30} {avg_coord_orig:.4f}{'':<10} {avg_coord_lstm:.4f}{'':<10} {avg_coord_lstm - avg_coord_orig:>+.4f}")

    # Dano promedio por ronda
    avg_damage_orig = np.mean(results_orig['cells_damaged'])
    avg_damage_lstm = np.mean(results_lstm['cells_damaged'])
    print(f"{'Dano promedio/ronda':<30} {avg_damage_orig:.1f}{'':<14} {avg_damage_lstm:.1f}{'':<14} {avg_damage_lstm - avg_damage_orig:>+.1f}")

    # Visualizacion
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Celulas en zona por ronda
    ax = axes[0, 0]
    rounds = results_orig['round']
    ax.plot(rounds, results_orig['cells_in_zone_before'], 'b-o', label='Original', linewidth=2)
    ax.plot(rounds, results_lstm['cells_in_zone_before'], 'g-s', label='LSTM', linewidth=2)
    ax.set_xlabel('Ronda de dano')
    ax.set_ylabel('Celulas en zona peligrosa')
    ax.set_title('Evacuacion Preventiva')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Dano por ronda
    ax = axes[0, 1]
    ax.bar(np.array(rounds) - 0.15, results_orig['cells_damaged'], 0.3, label='Original', color='blue', alpha=0.7)
    ax.bar(np.array(rounds) + 0.15, results_lstm['cells_damaged'], 0.3, label='LSTM', color='green', alpha=0.7)
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Celulas danadas')
    ax.set_title('Dano Recibido')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Coordinacion a lo largo del tiempo
    ax = axes[0, 2]
    steps_o = [h['step'] for h in coord_orig]
    coord_o = [h['coordination'] for h in coord_orig]
    steps_l = [h['step'] for h in coord_lstm]
    coord_l = [h['coordination'] for h in coord_lstm]
    ax.plot(steps_o, coord_o, 'b-', label='Original', linewidth=2)
    ax.plot(steps_l, coord_l, 'g-', label='LSTM', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Coordinacion')
    ax.set_title('Evolucion de Coordinacion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Evolucion de h_norm
    ax = axes[1, 0]
    steps_h = [h['step'] for h in h_norm_history]
    h_norms = [h['h_norm'] for h in h_norm_history]
    ax.plot(steps_h, h_norms, 'g-', linewidth=2)
    ax.fill_between(steps_h, h_norms, alpha=0.3, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('h_norm promedio')
    ax.set_title('Evolucion Memoria LSTM')
    ax.grid(True, alpha=0.3)

    # 5. Comparacion de Fi
    ax = axes[1, 1]
    fi_o = [h['n_fi'] for h in coord_orig]
    fi_l = [h['n_fi'] for h in coord_lstm]
    ax.plot(steps_o, fi_o, 'b-', label='Original', linewidth=2)
    ax.plot(steps_l, fi_l, 'g-', label='LSTM', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Numero de Fi')
    ax.set_title('Estabilidad de Lideres')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Resumen en barras
    ax = axes[1, 2]
    metrics = ['Evacuacion\n(%)', 'Coordinacion\n(x100)']
    orig_vals = [evac_orig, avg_coord_orig * 100]
    lstm_vals = [evac_lstm, avg_coord_lstm * 100]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, orig_vals, width, label='Original', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, lstm_vals, width, label='LSTM', color='green', alpha=0.7)

    ax.set_ylabel('Valor')
    ax.set_title('Comparacion Final')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Agregar valores sobre las barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('zeta_organism_lstm_comparison.png', dpi=150)
    print("\nGuardado: zeta_organism_lstm_comparison.png")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    improvements = []
    if evac_lstm > evac_orig:
        improvements.append(f"evacuacion (+{evac_lstm - evac_orig:.1f}%)")
    if avg_coord_lstm > avg_coord_orig:
        improvements.append(f"coordinacion (+{(avg_coord_lstm - avg_coord_orig)*100:.1f}%)")

    if improvements:
        print(f"\n[OK] ZetaOrganismLSTM muestra mejora en: {', '.join(improvements)}")
    else:
        print("\n[--] No se observa mejora significativa con LSTM")

    if evac_lstm > evac_orig + 10:
        print("[OK] La memoria LSTM mejora significativamente la anticipacion!")

    return {
        'evac_orig': evac_orig,
        'evac_lstm': evac_lstm,
        'coord_orig': avg_coord_orig,
        'coord_lstm': avg_coord_lstm
    }


if __name__ == '__main__':
    main()
