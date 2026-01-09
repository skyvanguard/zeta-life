# exp_migracion_v2.py
"""Experimento: Migracion colectiva con gradientes de energia v2.

Modificacion: Las celulas ahora tienen un componente de movimiento
hacia zonas de mayor energia, ademas del movimiento hacia Fi.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism, CellEntity


def create_energy_gradient(grid_size, gradient_type):
    """Crea un mapa de energia basado en el tipo de gradiente."""
    energy_map = np.zeros((grid_size, grid_size))

    for y in range(grid_size):
        for x in range(grid_size):
            norm_x = x / grid_size
            norm_y = y / grid_size

            if gradient_type == 'linear_x':
                energy_map[y, x] = norm_x
            elif gradient_type == 'linear_y':
                energy_map[y, x] = norm_y
            elif gradient_type == 'radial_center':
                cx, cy = grid_size / 2, grid_size / 2
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                max_dist = np.sqrt(2) * grid_size / 2
                energy_map[y, x] = 1 - dist / max_dist
            elif gradient_type == 'radial_corner':
                dist = np.sqrt((x - grid_size)**2 + (y - grid_size)**2)
                max_dist = np.sqrt(2) * grid_size
                energy_map[y, x] = 1 - dist / max_dist

    return energy_map


def compute_energy_gradient(energy_map, x, y):
    """Calcula gradiente de energia en posicion (x, y)."""
    h, w = energy_map.shape

    # Gradiente en X
    x_next = min(x + 1, w - 1)
    x_prev = max(x - 1, 0)
    grad_x = energy_map[y, x_next] - energy_map[y, x_prev]

    # Gradiente en Y
    y_next = min(y + 1, h - 1)
    y_prev = max(y - 1, 0)
    grad_y = energy_map[y_next, x] - energy_map[y_prev, x]

    return grad_x, grad_y


def step_with_energy_gradient(org, energy_map, energy_weight=0.5):
    """Step modificado que incluye movimiento hacia energia."""
    # Primero ejecutar step normal
    org.step()

    # Luego aplicar movimiento adicional basado en gradiente de energia
    new_cells = []
    for cell in org.cells:
        x, y = cell.position

        # Calcular gradiente de energia
        grad_x, grad_y = compute_energy_gradient(energy_map, x, y)

        # Movimiento probabilistico hacia mayor energia
        if np.random.random() < energy_weight:
            if abs(grad_x) > 0.01 or abs(grad_y) > 0.01:
                dx = int(np.sign(grad_x)) if abs(grad_x) > abs(grad_y) else 0
                dy = int(np.sign(grad_y)) if abs(grad_y) >= abs(grad_x) else 0

                new_x = np.clip(x + dx, 0, org.grid_size - 1)
                new_y = np.clip(y + dy, 0, org.grid_size - 1)

                cell.position = (new_x, new_y)

        # Bonus de energia por estar en zona alta
        energy_bonus = energy_map[y, x] * 0.1
        cell.energy = min(cell.energy + energy_bonus, 1.0)

        new_cells.append(cell)

    org.cells = new_cells
    org._update_grids()


def get_centroid(org):
    """Calcula centroide del organismo."""
    if not org.cells:
        return (0, 0)
    x_sum = sum(c.position[0] for c in org.cells)
    y_sum = sum(c.position[1] for c in org.cells)
    n = len(org.cells)
    return (x_sum / n, y_sum / n)


def run_migration_experiment():
    """Ejecuta experimento de migracion."""
    print('='*70)
    print('EXPERIMENTO: MIGRACION CON GRADIENTES DE ENERGIA v2')
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

    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior.load_state_dict(weights['behavior_state'])
        org.cell_module.load_state_dict(weights['cell_module_state'])
        print('Pesos entrenados cargados!')
    except:
        print('Sin pesos entrenados')

    org.initialize(seed_fi=True)

    # Estabilizacion
    print('\n[FASE 0] Estabilizacion sin gradiente (50 steps)...')
    for _ in range(50):
        org.step()

    baseline = org.get_metrics()
    initial_centroid = get_centroid(org)
    print(f'Centroide inicial: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})')
    print(f'Fi={baseline["n_fi"]}, Coord={baseline["coordination"]:.3f}')

    # Guardar posiciones iniciales para visualizacion
    initial_positions = [(c.position[0], c.position[1], c.role_idx) for c in org.cells]

    # Historial completo
    full_history = []
    trajectory = [initial_centroid]

    # === FASE 1: Gradiente hacia derecha ===
    print('\n[FASE 1] Gradiente LINEAR_X (hacia derecha) - 150 steps...')
    energy_map = create_energy_gradient(48, 'linear_x')
    phase1_start = get_centroid(org)

    for step in range(150):
        step_with_energy_gradient(org, energy_map, energy_weight=0.3)
        m = org.get_metrics()
        centroid = get_centroid(org)
        m['centroid_x'] = centroid[0]
        m['centroid_y'] = centroid[1]
        m['phase'] = 'linear_x'
        full_history.append(m)
        trajectory.append(centroid)

        if (step + 1) % 50 == 0:
            print(f'  Step {step+1}: Centroid=({centroid[0]:.1f}, {centroid[1]:.1f}), Fi={m["n_fi"]}')

    phase1_end = get_centroid(org)
    phase1_displacement = np.sqrt((phase1_end[0] - phase1_start[0])**2 +
                                   (phase1_end[1] - phase1_start[1])**2)
    print(f'  Desplazamiento Fase 1: {phase1_displacement:.2f} celdas')
    print(f'  Direccion: dx={phase1_end[0] - phase1_start[0]:.1f}')

    # Guardar posiciones intermedias
    mid_positions = [(c.position[0], c.position[1], c.role_idx) for c in org.cells]

    # === FASE 2: Gradiente hacia centro ===
    print('\n[FASE 2] Gradiente RADIAL_CENTER (hacia centro) - 150 steps...')
    energy_map = create_energy_gradient(48, 'radial_center')
    phase2_start = get_centroid(org)

    for step in range(150):
        step_with_energy_gradient(org, energy_map, energy_weight=0.3)
        m = org.get_metrics()
        centroid = get_centroid(org)
        m['centroid_x'] = centroid[0]
        m['centroid_y'] = centroid[1]
        m['phase'] = 'radial_center'
        full_history.append(m)
        trajectory.append(centroid)

        if (step + 1) % 50 == 0:
            center = (24, 24)
            dist_to_center = np.sqrt((centroid[0] - center[0])**2 + (centroid[1] - center[1])**2)
            print(f'  Step {step+1}: Centroid=({centroid[0]:.1f}, {centroid[1]:.1f}), '
                  f'Dist al centro={dist_to_center:.1f}')

    phase2_end = get_centroid(org)
    center = (24, 24)
    initial_dist_to_center = np.sqrt((phase2_start[0] - center[0])**2 + (phase2_start[1] - center[1])**2)
    final_dist_to_center = np.sqrt((phase2_end[0] - center[0])**2 + (phase2_end[1] - center[1])**2)
    print(f'  Distancia al centro: {initial_dist_to_center:.1f} -> {final_dist_to_center:.1f}')

    # Guardar posiciones finales
    final_positions = [(c.position[0], c.position[1], c.role_idx) for c in org.cells]

    # === ANALISIS ===
    print('\n' + '='*70)
    print('ANALISIS DE MIGRACION')
    print('='*70)

    final_centroid = get_centroid(org)
    total_displacement = np.sqrt((final_centroid[0] - initial_centroid[0])**2 +
                                  (final_centroid[1] - initial_centroid[1])**2)

    print(f'\nCentroide inicial: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})')
    print(f'Centroide final: ({final_centroid[0]:.1f}, {final_centroid[1]:.1f})')
    print(f'Desplazamiento total: {total_displacement:.2f} celdas')

    final = org.get_metrics()
    print(f'\nEstado final:')
    print(f'  Fi: {final["n_fi"]}')
    print(f'  Coordinacion: {final["coordination"]:.3f}')

    # Verificar migracion
    print('\n*** VERIFICACION ***')
    if phase1_end[0] > phase1_start[0]:
        print('  Fase 1 (Linear X): CORRECTO - migro hacia derecha')
    else:
        print('  Fase 1 (Linear X): PARCIAL')

    if final_dist_to_center < initial_dist_to_center:
        print(f'  Fase 2 (Radial): CORRECTO - se acerco al centro')
    else:
        print(f'  Fase 2 (Radial): PARCIAL')

    # === VISUALIZACION ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Trayectoria del centroide
    ax = axes[0, 0]
    traj_x = [t[0] for t in trajectory]
    traj_y = [t[1] for t in trajectory]
    # Colorear por fase
    colors = np.linspace(0, 1, len(trajectory))
    scatter = ax.scatter(traj_x, traj_y, c=colors, cmap='viridis', s=10, alpha=0.7)
    ax.plot(traj_x, traj_y, 'k-', alpha=0.3, linewidth=0.5)
    ax.scatter(traj_x[0], traj_y[0], c='green', s=200, marker='o', label='Inicio', zorder=5)
    ax.scatter(traj_x[-1], traj_y[-1], c='red', s=200, marker='s', label='Final', zorder=5)
    ax.scatter(24, 24, c='yellow', s=100, marker='*', label='Centro', zorder=5)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 48)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trayectoria del Centroide')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 2. Posicion X en el tiempo
    ax = axes[0, 1]
    cx_vals = [h['centroid_x'] for h in full_history]
    ax.plot(cx_vals, 'b-', linewidth=2)
    ax.axvline(x=150, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=24, color='green', linestyle=':', alpha=0.5, label='Centro')
    ax.text(75, max(cx_vals)+1, 'Linear X', ha='center', fontsize=10)
    ax.text(225, max(cx_vals)+1, 'Radial', ha='center', fontsize=10)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Centroide X')
    ax.set_title('Posicion X durante Migracion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Posicion Y en el tiempo
    ax = axes[0, 2]
    cy_vals = [h['centroid_y'] for h in full_history]
    ax.plot(cy_vals, 'g-', linewidth=2)
    ax.axvline(x=150, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=24, color='green', linestyle=':', alpha=0.5, label='Centro')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Centroide Y')
    ax.set_title('Posicion Y durante Migracion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Estado inicial
    ax = axes[1, 0]
    for x, y, role in initial_positions:
        color = ['blue', 'red', 'black'][role]
        ax.scatter(x, y, c=color, s=30, alpha=0.7)
    ax.scatter(initial_centroid[0], initial_centroid[1], c='yellow', s=200, marker='*',
               edgecolors='black', linewidths=2, label='Centroide', zorder=5)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 48)
    ax.set_title('Estado Inicial')
    ax.set_aspect('equal')
    ax.legend()

    # 5. Estado final con gradiente
    ax = axes[1, 1]
    # Mostrar gradiente radial de fondo
    gradient_bg = create_energy_gradient(48, 'radial_center')
    ax.imshow(gradient_bg, extent=[0, 48, 0, 48], origin='lower', cmap='YlOrRd', alpha=0.4)
    for x, y, role in final_positions:
        color = ['blue', 'red', 'black'][role]
        ax.scatter(x, y, c=color, s=30, alpha=0.8, edgecolors='white', linewidths=0.3)
    ax.scatter(final_centroid[0], final_centroid[1], c='yellow', s=200, marker='*',
               edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(24, 24, c='white', s=100, marker='+', linewidths=3, zorder=5)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 48)
    ax.set_title('Estado Final (fondo=gradiente radial)')
    ax.set_aspect('equal')

    # 6. Evolucion de Fi y coordinacion
    ax = axes[1, 2]
    fi_vals = [h['n_fi'] for h in full_history]
    coord_vals = [h['coordination'] for h in full_history]
    ax.plot(fi_vals, 'r-', linewidth=2, label='Fi')
    ax2 = ax.twinx()
    ax2.plot(coord_vals, 'g-', linewidth=2, label='Coord')
    ax.axvline(x=150, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cantidad Fi', color='red')
    ax2.set_ylabel('Coordinacion', color='green')
    ax.set_title('Estabilidad durante Migracion')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_organism_migracion.png', dpi=150)
    print('\nGuardado: zeta_organism_migracion.png')

    return full_history, trajectory


if __name__ == '__main__':
    run_migration_experiment()
