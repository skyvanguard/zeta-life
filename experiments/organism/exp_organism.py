# exp_organism.py
"""Experimento: Inteligencia colectiva en ZetaOrganism."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism

def run_experiment(n_steps: int = 200, grid_size: int = 64, n_cells: int = 100):
    """Ejecuta simulación y analiza emergencia."""
    print('='*60)
    print('ZetaOrganism: Experimento de Inteligencia Colectiva')
    print('='*60)

    # Crear organismo
    org = ZetaOrganism(
        grid_size=grid_size,
        n_cells=n_cells,
        state_dim=32,
        hidden_dim=64,
        M=15,
        sigma=0.1,
        fi_threshold=0.5  # Más bajo para permitir emergencia
    )

    print(f'\nConfiguracion:')
    print(f'  Grid: {grid_size}x{grid_size}')
    print(f'  Celulas: {n_cells}')
    print(f'  Steps: {n_steps}')

    # Inicializar con semilla Fi
    org.initialize(seed_fi=True)

    print(f'\nIniciando simulacion...')

    # Simular
    for step in range(n_steps):
        org.step()

        if (step + 1) % 50 == 0:
            m = org.get_metrics()
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Mass={m["n_mass"]}, '
                  f'Coord={m["coordination"]:.3f}, Stab={m["stability"]:.3f}')

    # Análisis final
    print('\n' + '='*60)
    print('RESULTADOS FINALES:')

    final = org.get_metrics()
    print(f'  Fi (fuerzas): {final["n_fi"]}')
    print(f'  Mass (seguidores): {final["n_mass"]}')
    print(f'  Corrupt (competidores): {final["n_corrupt"]}')
    print(f'  Coordinacion: {final["coordination"]:.3f}')
    print(f'  Estabilidad: {final["stability"]:.3f}')

    # Análisis de emergencia
    history = org.history

    # ¿Emergieron más Fi?
    fi_history = [h['n_fi'] for h in history]
    if fi_history[-1] > fi_history[0]:
        print('\n*** EMERGENCIA DETECTADA: Nuevos Fi surgieron del sistema ***')

    # ¿Hubo coordinación?
    coord_history = [h['coordination'] for h in history]
    if max(coord_history) > 0.5:
        print('*** COORDINACION: Masas se agruparon alrededor de Fi ***')

    # ¿Sistema estable?
    stab_history = [h['stability'] for h in history]
    if np.mean(stab_history[-20:]) > 0.7:
        print('*** HOMEOSTASIS: Sistema alcanzo equilibrio ***')

    print('='*60)

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Evolución de roles
    ax = axes[0, 0]
    steps = range(len(history))
    ax.plot(steps, fi_history, 'r-', label='Fi (fuerzas)', linewidth=2)
    ax.plot(steps, [h['n_mass'] for h in history], 'b-', label='Mass', linewidth=2)
    ax.plot(steps, [h['n_corrupt'] for h in history], 'k--', label='Corrupt', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cantidad')
    ax.set_title('Evolucion de Roles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Coordinación y estabilidad
    ax = axes[0, 1]
    ax.plot(steps, coord_history, 'g-', label='Coordinacion', linewidth=2)
    ax.plot(steps, stab_history, 'm-', label='Estabilidad', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Valor')
    ax.set_title('Metricas de Inteligencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. Estado final del grid
    ax = axes[1, 0]
    for cell in org.cells:
        x, y = cell.position
        color = ['blue', 'red', 'black'][cell.role_idx]
        size = 20 + cell.energy * 80
        ax.scatter(x, y, c=color, s=size, alpha=0.7)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_title('Estado Final (rojo=Fi, azul=Mass)')
    ax.set_aspect('equal')

    # 4. Energía promedio
    ax = axes[1, 1]
    energy_history = [h['avg_energy'] for h in history]
    ax.plot(steps, energy_history, 'orange', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energia promedio')
    ax.set_title('Evolucion de Energia')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_organism_experiment.png', dpi=150)
    print('\nGuardado: zeta_organism_experiment.png')

    return org, history


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_experiment()
