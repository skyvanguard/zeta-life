# exp_dos_organismos_v2.py
"""Experimento: Dos organismos con interaccion forzada.

Escenarios:
1. Superpuestos: Ambos empiezan en el mismo espacio
2. Colision: Empiezan separados pero migran hacia el centro
3. Invasion: Uno invade el territorio del otro
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from exp_dos_organismos import DualOrganism, DualCellEntity


def initialize_overlapping(dual):
    """Inicializa organismos superpuestos en el centro."""
    dual.cells = []
    center = dual.grid_size // 2

    for org_id in range(2):
        for i in range(dual.n_cells_per_org):
            # Ambos en el centro con leve offset
            offset = 5 if org_id == 0 else -5
            x = np.random.randint(center - 15 + offset, center + 15 + offset)
            y = np.random.randint(center - 15, center + 15)

            state = torch.randn(dual.state_dim) * 0.1

            if i == 0:
                role = torch.tensor([0.0, 1.0, 0.0])
                energy = 0.9
            else:
                role = torch.tensor([1.0, 0.0, 0.0])
                energy = np.random.uniform(0.3, 0.6)

            cell = DualCellEntity(
                position=(x, y),
                state=state,
                role=role,
                energy=energy,
                organism_id=org_id
            )
            dual.cells.append(cell)

    dual._update_grids()


def initialize_collision_course(dual):
    """Inicializa organismos que colisionaran."""
    dual.cells = []

    for org_id in range(2):
        for i in range(dual.n_cells_per_org):
            # Org 0 arriba-izquierda, Org 1 abajo-derecha
            if org_id == 0:
                x = np.random.randint(5, 20)
                y = np.random.randint(dual.grid_size - 20, dual.grid_size - 5)
            else:
                x = np.random.randint(dual.grid_size - 20, dual.grid_size - 5)
                y = np.random.randint(5, 20)

            state = torch.randn(dual.state_dim) * 0.1

            if i == 0:
                role = torch.tensor([0.0, 1.0, 0.0])
                energy = 0.9
            else:
                role = torch.tensor([1.0, 0.0, 0.0])
                energy = np.random.uniform(0.3, 0.6)

            cell = DualCellEntity(
                position=(x, y),
                state=state,
                role=role,
                energy=energy,
                organism_id=org_id
            )
            dual.cells.append(cell)

    dual._update_grids()


def apply_central_gradient(dual, strength=0.3):
    """Aplica gradiente de energia hacia el centro."""
    center = dual.grid_size // 2
    for cell in dual.cells:
        x, y = cell.position
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        max_dist = np.sqrt(2) * dual.grid_size / 2
        bonus = strength * (1 - dist / max_dist)
        cell.energy = min(cell.energy + bonus, 1.0)


def step_with_migration(dual, target='center'):
    """Step con movimiento adicional hacia objetivo."""
    # Step normal
    dual.step()

    # Movimiento adicional hacia centro
    center = dual.grid_size // 2
    for cell in dual.cells:
        x, y = cell.position

        if target == 'center':
            dx = int(np.sign(center - x)) if np.random.random() < 0.2 else 0
            dy = int(np.sign(center - y)) if np.random.random() < 0.2 else 0
        else:
            dx, dy = 0, 0

        new_x = np.clip(x + dx, 0, dual.grid_size - 1)
        new_y = np.clip(y + dy, 0, dual.grid_size - 1)
        cell.position = (new_x, new_y)

    dual._update_grids()


def run_scenario(scenario_name, init_func, n_steps=200, use_migration=False):
    """Ejecuta un escenario."""
    print(f'\n{"="*70}')
    print(f'ESCENARIO: {scenario_name}')
    print('='*70)

    torch.manual_seed(42)
    np.random.seed(42)

    dual = DualOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        dual.behavior_0.load_state_dict(weights['behavior_state'])
        dual.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    init_func(dual)

    initial = dual.get_metrics()
    print(f'Estado inicial:')
    print(f'  Org 0: Fi={initial["org_0"]["n_fi"]}, Total={initial["org_0"]["n_total"]}')
    print(f'  Org 1: Fi={initial["org_1"]["n_fi"]}, Total={initial["org_1"]["n_total"]}')
    print(f'  Contactos frontera: {initial["boundary_contacts"]}')

    # Guardar para visualizacion
    initial_positions = [(c.position[0], c.position[1], c.organism_id, c.role_idx)
                        for c in dual.cells]

    print(f'\nSimulando {n_steps} steps...')
    for step in range(n_steps):
        if use_migration:
            apply_central_gradient(dual, strength=0.1)
            step_with_migration(dual, target='center')
        else:
            dual.step()

        if (step + 1) % 50 == 0:
            m = dual.get_metrics()
            print(f'  Step {step+1}: Org0={m["org_0"]["n_total"]}, Org1={m["org_1"]["n_total"]}, '
                  f'Frontera={m["boundary_contacts"]}')

    final = dual.get_metrics()

    # Determinar resultado
    org0_t = final["org_0"]["n_total"]
    org1_t = final["org_1"]["n_total"]
    conversions = abs(40 - org0_t)

    print(f'\nResultado:')
    print(f'  Org 0: {org0_t} celulas ({org0_t - 40:+d})')
    print(f'  Org 1: {org1_t} celulas ({org1_t - 40:+d})')
    print(f'  Conversiones: {conversions}')
    print(f'  Contactos frontera final: {final["boundary_contacts"]}')

    if conversions > 15:
        if org0_t > org1_t:
            print('  -> DOMINACION: Org 0 absorbio a Org 1')
        else:
            print('  -> DOMINACION: Org 1 absorbio a Org 0')
    elif final["boundary_contacts"] > 20:
        print('  -> FUSION: Los organismos se mezclaron')
    elif conversions > 5:
        print('  -> COMPETENCIA: Intercambio parcial de celulas')
    else:
        print('  -> COEXISTENCIA: Territorios mantenidos')

    return dual, initial_positions, dual.history


def run_all_scenarios():
    """Ejecuta todos los escenarios."""
    print('='*70)
    print('EXPERIMENTO: INTERACCION ENTRE DOS ORGANISMOS')
    print('='*70)

    results = {}

    # Escenario 1: Superpuestos
    dual1, init1, hist1 = run_scenario(
        'SUPERPUESTOS (mismo espacio)',
        initialize_overlapping,
        n_steps=200
    )
    results['superpuestos'] = (dual1, init1, hist1)

    # Escenario 2: Colision con migracion
    dual2, init2, hist2 = run_scenario(
        'COLISION (migracion al centro)',
        initialize_collision_course,
        n_steps=200,
        use_migration=True
    )
    results['colision'] = (dual2, init2, hist2)

    # Visualizacion comparativa
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    scenarios = ['superpuestos', 'colision']
    titles = ['Superpuestos', 'Colision']

    for idx, (scenario, title) in enumerate(zip(scenarios, titles)):
        dual, init_pos, history = results[scenario]

        # Estado inicial
        ax = axes[0, idx*2]
        for x, y, org_id, role_idx in init_pos:
            color = 'royalblue' if org_id == 0 else 'crimson'
            marker = 's' if role_idx == 1 else 'o'
            size = 60 if role_idx == 1 else 20
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.6)
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_title(f'{title}: Inicial')
        ax.set_aspect('equal')

        # Estado final
        ax = axes[0, idx*2 + 1]
        for cell in dual.cells:
            x, y = cell.position
            color = 'royalblue' if cell.organism_id == 0 else 'crimson'
            marker = 's' if cell.role_idx == 1 else 'o'
            size = 60 if cell.role_idx == 1 else 20
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_title(f'{title}: Final')
        ax.set_aspect('equal')

        # Evolucion de poblacion
        ax = axes[1, idx*2]
        steps = range(len(history))
        org0 = [h['org_0']['n_total'] for h in history]
        org1 = [h['org_1']['n_total'] for h in history]
        ax.plot(steps, org0, 'b-', linewidth=2, label='Org 0')
        ax.plot(steps, org1, 'r-', linewidth=2, label='Org 1')
        ax.axhline(y=40, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Poblacion')
        ax.set_title(f'{title}: Poblacion')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Contactos frontera
        ax = axes[1, idx*2 + 1]
        boundary = [h['boundary_contacts'] for h in history]
        ax.plot(steps, boundary, 'g-', linewidth=2)
        ax.fill_between(steps, boundary, alpha=0.3, color='green')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Contactos')
        ax.set_title(f'{title}: Frontera')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_organism_dual_v2.png', dpi=150)
    print('\n' + '='*70)
    print('Guardado: zeta_organism_dual_v2.png')

    return results


if __name__ == '__main__':
    run_all_scenarios()
