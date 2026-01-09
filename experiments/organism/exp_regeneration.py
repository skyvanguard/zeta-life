# exp_regeneration.py
"""Experimento: Regeneración del organismo.

Prueba si el sistema puede regenerar Fi después de perderlos.
Esto demuestra auto-organización y homeostasis.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism


def run_regeneration_experiment():
    """Experimento de regeneración."""
    print('='*60)
    print('ZetaOrganism: Experimento de Regeneración')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Crear organismo
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
        print('No se encontraron pesos, usando red sin entrenar')

    org.initialize(seed_fi=True)

    # === FASE 1: Estabilización ===
    print('\n[FASE 1] Estabilización inicial (100 steps)...')
    for step in range(100):
        org.step()
        if (step + 1) % 25 == 0:
            m = org.get_metrics()
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Mass={m["n_mass"]}, Coord={m["coordination"]:.3f}')

    pre_damage = org.get_metrics()
    print(f'\nEstado pre-daño:')
    print(f'  Fi: {pre_damage["n_fi"]}')
    print(f'  Mass: {pre_damage["n_mass"]}')
    print(f'  Coordinación: {pre_damage["coordination"]:.3f}')

    # Guardar posiciones de Fi antes del daño
    fi_positions_before = [(c.position, c.energy) for c in org.cells if c.role_idx == 1]

    # === FASE 2: Daño - Eliminar 50% de los Fi ===
    print('\n' + '='*60)
    print('[FASE 2] DAÑO: Eliminando 50% de los Fi...')
    print('='*60)

    fi_cells = [c for c in org.cells if c.role_idx == 1]
    n_to_remove = len(fi_cells) // 2

    print(f'  Fi antes: {len(fi_cells)}')
    print(f'  Eliminando: {n_to_remove} Fi')

    # Convertir Fi a Mass (simula "muerte" del líder)
    removed_positions = []
    for i, cell in enumerate(org.cells):
        if cell.role_idx == 1 and n_to_remove > 0:
            # Convertir a Mass con baja energía
            cell.role = torch.tensor([1.0, 0.0, 0.0])
            cell.energy = 0.1
            removed_positions.append(cell.position)
            n_to_remove -= 1

    org._update_grids()

    post_damage = org.get_metrics()
    print(f'\nEstado post-daño:')
    print(f'  Fi: {post_damage["n_fi"]}')
    print(f'  Mass: {post_damage["n_mass"]}')
    print(f'  Coordinación: {post_damage["coordination"]:.3f}')

    # === FASE 3: Regeneración ===
    print('\n' + '='*60)
    print('[FASE 3] REGENERACIÓN: Observando recuperación (150 steps)...')
    print('='*60)

    # Guardar historia detallada
    regen_history = []
    damage_step = len(org.history)

    for step in range(150):
        org.step()
        m = org.get_metrics()
        regen_history.append(m)

        if (step + 1) % 30 == 0:
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Mass={m["n_mass"]}, '
                  f'Coord={m["coordination"]:.3f}')

    post_regen = org.get_metrics()

    # === ANÁLISIS ===
    print('\n' + '='*60)
    print('ANÁLISIS DE REGENERACIÓN')
    print('='*60)

    print(f'\n{"Métrica":<20} {"Pre-Daño":<12} {"Post-Daño":<12} {"Regenerado":<12} {"Recuperación"}')
    print('-'*70)

    # Fi
    fi_recovery = (post_regen['n_fi'] - post_damage['n_fi']) / max(pre_damage['n_fi'] - post_damage['n_fi'], 1) * 100
    print(f'{"Fi":<20} {pre_damage["n_fi"]:<12} {post_damage["n_fi"]:<12} {post_regen["n_fi"]:<12} {fi_recovery:.1f}%')

    # Coordinación
    coord_recovery = (post_regen['coordination'] - post_damage['coordination']) / max(pre_damage['coordination'] - post_damage['coordination'], 0.01) * 100
    print(f'{"Coordinación":<20} {pre_damage["coordination"]:<12.3f} {post_damage["coordination"]:<12.3f} {post_regen["coordination"]:<12.3f} {coord_recovery:.1f}%')

    # Veredicto
    print('\n' + '='*60)
    if post_regen['n_fi'] >= pre_damage['n_fi'] * 0.8:
        print('*** REGENERACIÓN EXITOSA: El sistema recuperó sus Fi ***')
    elif post_regen['n_fi'] > post_damage['n_fi']:
        print('*** REGENERACIÓN PARCIAL: Emergieron nuevos Fi ***')
    else:
        print('*** SIN REGENERACIÓN: El sistema no se recuperó ***')

    if post_regen['coordination'] >= pre_damage['coordination'] * 0.9:
        print('*** HOMEOSTASIS: Coordinación restaurada ***')
    print('='*60)

    # === VISUALIZACIÓN ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Historia completa
    full_history = org.history

    # 1. Evolución de Fi con marca de daño
    ax = axes[0, 0]
    steps = range(len(full_history))
    fi_vals = [h['n_fi'] for h in full_history]
    ax.plot(steps, fi_vals, 'r-', linewidth=2, label='Fi')
    ax.axvline(x=damage_step, color='black', linestyle='--', linewidth=2, label='DAÑO')
    ax.fill_between(range(damage_step, len(full_history)),
                    [h['n_fi'] for h in full_history[damage_step:]],
                    alpha=0.3, color='green', label='Regeneración')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Evolución de Fi (con daño)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Evolución de Mass
    ax = axes[0, 1]
    mass_vals = [h['n_mass'] for h in full_history]
    ax.plot(steps, mass_vals, 'b-', linewidth=2)
    ax.axvline(x=damage_step, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cantidad Mass')
    ax.set_title('Evolución de Mass')
    ax.grid(True, alpha=0.3)

    # 3. Coordinación
    ax = axes[0, 2]
    coord_vals = [h['coordination'] for h in full_history]
    ax.plot(steps, coord_vals, 'g-', linewidth=2)
    ax.axvline(x=damage_step, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=pre_damage['coordination'], color='green', linestyle=':', alpha=0.5, label='Pre-daño')
    ax.set_xlabel('Step')
    ax.set_ylabel('Coordinación')
    ax.set_title('Coordinación (homeostasis)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 4. Tasa de regeneración
    ax = axes[1, 0]
    fi_regen = [h['n_fi'] for h in regen_history]
    target_fi = pre_damage['n_fi']
    ax.plot(range(len(fi_regen)), fi_regen, 'r-', linewidth=2, label='Fi actual')
    ax.axhline(y=target_fi, color='red', linestyle=':', label=f'Target ({target_fi})')
    ax.axhline(y=post_damage['n_fi'], color='orange', linestyle='--', label=f'Post-daño ({post_damage["n_fi"]})')
    ax.set_xlabel('Steps después del daño')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Curva de Regeneración')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Energía promedio
    ax = axes[1, 1]
    energy_vals = [h['avg_energy'] for h in full_history]
    ax.plot(steps, energy_vals, 'orange', linewidth=2)
    ax.axvline(x=damage_step, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energía promedio')
    ax.set_title('Evolución de Energía')
    ax.grid(True, alpha=0.3)

    # 6. Estado final con posiciones de Fi eliminados
    ax = axes[1, 2]
    for cell in org.cells:
        x, y = cell.position
        color = ['blue', 'red', 'black'][cell.role_idx]
        size = 20 + cell.energy * 80
        ax.scatter(x, y, c=color, s=size, alpha=0.7)

    # Marcar posiciones donde se eliminaron Fi
    for pos in removed_positions:
        ax.scatter(pos[0], pos[1], c='yellow', s=200, marker='x', linewidths=3, label='Fi eliminado')

    ax.set_xlim(0, org.grid_size)
    ax.set_ylim(0, org.grid_size)
    ax.set_title('Estado Final (X=Fi eliminados)')
    ax.set_aspect('equal')

    # Evitar duplicados en leyenda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.savefig('zeta_organism_regeneration.png', dpi=150)
    print('\nGuardado: zeta_organism_regeneration.png')

    return org


if __name__ == '__main__':
    run_regeneration_experiment()
