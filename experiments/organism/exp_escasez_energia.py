# exp_escasez_energia.py
"""Experimento: Escasez de energia en ZetaOrganism.

Hipotesis: Al reducir la energia disponible, los Fi competiran
por recursos limitados, resultando en:
- Reduccion del numero de Fi
- Competencia territorial
- Posible colapso o reorganizacion
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism


def run_scarcity_experiment():
    """Experimento de escasez progresiva de energia."""
    print('='*70)
    print('EXPERIMENTO: ESCASEZ DE ENERGIA')
    print('='*70)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ZetaOrganism(
        grid_size=48,
        n_cells=80,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )

    # Cargar pesos entrenados
    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior.load_state_dict(weights['behavior_state'])
        org.cell_module.load_state_dict(weights['cell_module_state'])
        print('Pesos entrenados cargados!')
    except:
        print('Sin pesos entrenados')

    org.initialize(seed_fi=True)

    # === FASE 1: Estabilizacion con energia normal ===
    print('\n[FASE 1] Estabilizacion con energia normal (100 steps)...')
    for step in range(100):
        org.step()

    baseline = org.get_metrics()
    print(f'Baseline: Fi={baseline["n_fi"]}, Mass={baseline["n_mass"]}, '
          f'Coord={baseline["coordination"]:.3f}, Energy={baseline["avg_energy"]:.3f}')

    # Guardar estado inicial
    full_history = []
    phase_markers = []

    # === FASE 2: Escasez progresiva ===
    print('\n[FASE 2] Escasez progresiva de energia...')

    scarcity_levels = [
        (0.9, 50, "Leve (90%)"),
        (0.7, 50, "Moderada (70%)"),
        (0.5, 50, "Severa (50%)"),
        (0.3, 50, "Critica (30%)"),
        (0.1, 50, "Extrema (10%)")
    ]

    for energy_factor, duration, label in scarcity_levels:
        print(f'\n--- {label}: Factor de energia = {energy_factor} ---')
        phase_markers.append((len(full_history), label))

        for step in range(duration):
            # Aplicar factor de escasez a la energia de todas las celulas
            for cell in org.cells:
                # Drenar energia gradualmente
                cell.energy = cell.energy * (0.98 + 0.02 * energy_factor)
                # Limitar ganancia de energia
                if cell.energy > energy_factor:
                    cell.energy = energy_factor

            org.step()

            # Tambien limitar energia post-step
            for cell in org.cells:
                cell.energy = min(cell.energy, energy_factor)

            m = org.get_metrics()
            m['energy_factor'] = energy_factor
            full_history.append(m)

        m = org.get_metrics()
        print(f'  Fi={m["n_fi"]}, Mass={m["n_mass"]}, '
              f'Coord={m["coordination"]:.3f}, Energy={m["avg_energy"]:.3f}')

    scarcity_end = org.get_metrics()

    # === FASE 3: Recuperacion (restaurar energia) ===
    print('\n[FASE 3] Recuperacion: Restaurando energia normal (100 steps)...')
    phase_markers.append((len(full_history), "Recuperacion"))

    for step in range(100):
        # Permitir recuperacion gradual de energia
        for cell in org.cells:
            cell.energy = min(cell.energy + 0.02, 1.0)

        org.step()
        m = org.get_metrics()
        m['energy_factor'] = 1.0
        full_history.append(m)

        if (step + 1) % 25 == 0:
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Coord={m["coordination"]:.3f}, '
                  f'Energy={m["avg_energy"]:.3f}')

    recovery = org.get_metrics()

    # === ANALISIS ===
    print('\n' + '='*70)
    print('ANALISIS DE ESCASEZ')
    print('='*70)

    print(f'\n{"Fase":<20} {"Fi":<8} {"Mass":<8} {"Coord":<10} {"Energia":<10}')
    print('-'*56)
    print(f'{"Baseline":<20} {baseline["n_fi"]:<8} {baseline["n_mass"]:<8} '
          f'{baseline["coordination"]:<10.3f} {baseline["avg_energy"]:<10.3f}')
    print(f'{"Post-escasez":<20} {scarcity_end["n_fi"]:<8} {scarcity_end["n_mass"]:<8} '
          f'{scarcity_end["coordination"]:<10.3f} {scarcity_end["avg_energy"]:<10.3f}')
    print(f'{"Recuperacion":<20} {recovery["n_fi"]:<8} {recovery["n_mass"]:<8} '
          f'{recovery["coordination"]:<10.3f} {recovery["avg_energy"]:<10.3f}')

    # Calcular metricas de impacto
    fi_loss = baseline['n_fi'] - scarcity_end['n_fi']
    fi_recovery = recovery['n_fi'] - scarcity_end['n_fi']

    print(f'\n*** IMPACTO ***')
    print(f'  Fi perdidos durante escasez: {fi_loss}')
    print(f'  Fi recuperados post-escasez: {fi_recovery}')

    if scarcity_end['n_fi'] == 0:
        print('  *** COLAPSO TOTAL: El sistema perdio todos los Fi ***')
    elif scarcity_end['n_fi'] < baseline['n_fi'] * 0.5:
        print('  *** COLAPSO PARCIAL: El sistema perdio >50% de Fi ***')
    elif recovery['n_fi'] >= baseline['n_fi'] * 0.8:
        print('  *** RESILIENCIA: El sistema se recupero al 80%+ ***')

    # === VISUALIZACION ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Evolucion de Fi
    ax = axes[0, 0]
    fi_vals = [h['n_fi'] for h in full_history]
    ax.plot(fi_vals, 'r-', linewidth=2)
    ax.axhline(y=baseline['n_fi'], color='green', linestyle='--', alpha=0.5, label='Baseline')
    for idx, label in phase_markers:
        ax.axvline(x=idx, color='gray', linestyle=':', alpha=0.7)
        ax.text(idx+2, max(fi_vals)*0.95, label, fontsize=7, rotation=45)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Evolucion de Fi bajo Escasez')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Evolucion de energia
    ax = axes[0, 1]
    energy_vals = [h['avg_energy'] for h in full_history]
    ax.plot(energy_vals, 'orange', linewidth=2)
    for idx, label in phase_markers:
        ax.axvline(x=idx, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Energia promedio')
    ax.set_title('Evolucion de Energia')
    ax.grid(True, alpha=0.3)

    # 3. Coordinacion
    ax = axes[0, 2]
    coord_vals = [h['coordination'] for h in full_history]
    ax.plot(coord_vals, 'g-', linewidth=2)
    ax.axhline(y=baseline['coordination'], color='green', linestyle='--', alpha=0.5)
    for idx, label in phase_markers:
        ax.axvline(x=idx, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Coordinacion')
    ax.set_title('Homeostasis bajo Escasez')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 4. Fi vs Energia (correlacion)
    ax = axes[1, 0]
    ax.scatter(energy_vals, fi_vals, c=range(len(fi_vals)), cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Energia promedio')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Correlacion Fi-Energia')
    ax.grid(True, alpha=0.3)
    # Agregar linea de tendencia
    z = np.polyfit(energy_vals, fi_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(energy_vals), max(energy_vals), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Tendencia')
    ax.legend()

    # 5. Factor de escasez vs Fi
    ax = axes[1, 1]
    # Agrupar por nivel de escasez
    scarcity_data = {}
    for h in full_history:
        ef = h.get('energy_factor', 1.0)
        if ef not in scarcity_data:
            scarcity_data[ef] = []
        scarcity_data[ef].append(h['n_fi'])

    factors = sorted(scarcity_data.keys())
    avg_fi = [np.mean(scarcity_data[f]) for f in factors]
    std_fi = [np.std(scarcity_data[f]) for f in factors]

    ax.bar(range(len(factors)), avg_fi, yerr=std_fi, capsize=5, color='red', alpha=0.7)
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([f'{int(f*100)}%' for f in factors])
    ax.set_xlabel('Factor de Energia')
    ax.set_ylabel('Fi promedio')
    ax.set_title('Impacto de Escasez en Fi')
    ax.grid(True, alpha=0.3)

    # 6. Estado final
    ax = axes[1, 2]
    for cell in org.cells:
        x, y = cell.position
        color = ['blue', 'red', 'black'][cell.role_idx]
        size = 20 + cell.energy * 80
        ax.scatter(x, y, c=color, s=size, alpha=0.7)
    ax.set_xlim(0, org.grid_size)
    ax.set_ylim(0, org.grid_size)
    ax.set_title('Estado Final (tamano=energia)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('zeta_organism_escasez.png', dpi=150)
    print('\nGuardado: zeta_organism_escasez.png')

    return {
        'baseline': baseline,
        'scarcity_end': scarcity_end,
        'recovery': recovery,
        'history': full_history,
        'phase_markers': phase_markers
    }


if __name__ == '__main__':
    run_scarcity_experiment()
