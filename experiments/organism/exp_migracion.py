# exp_migracion.py
"""Experimento: Migracion colectiva con gradientes de energia.

Hipotesis: Al crear gradientes de energia en el espacio,
el organismo deberia migrar colectivamente hacia zonas
de mayor energia, demostrando comportamiento coordinado.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from zeta_life.organism import ZetaOrganism


class MigrationExperiment:
    """Experimento de migracion con gradientes."""

    def __init__(self, org: ZetaOrganism):
        self.org = org
        self.history = []
        self.position_history = []

    def apply_gradient(self, gradient_type: str = 'linear'):
        """Aplica gradiente de energia segun posicion."""
        for cell in self.org.cells:
            x, y = cell.position
            norm_x = x / self.org.grid_size
            norm_y = y / self.org.grid_size

            if gradient_type == 'linear_x':
                # Mas energia a la derecha
                bonus = 0.3 * norm_x
            elif gradient_type == 'linear_y':
                # Mas energia arriba
                bonus = 0.3 * norm_y
            elif gradient_type == 'radial_center':
                # Mas energia en el centro
                cx, cy = self.org.grid_size / 2, self.org.grid_size / 2
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                max_dist = np.sqrt(2) * self.org.grid_size / 2
                bonus = 0.3 * (1 - dist / max_dist)
            elif gradient_type == 'radial_corner':
                # Mas energia en esquina superior derecha
                dist = np.sqrt((x - self.org.grid_size)**2 + (y - self.org.grid_size)**2)
                max_dist = np.sqrt(2) * self.org.grid_size
                bonus = 0.3 * (1 - dist / max_dist)
            elif gradient_type == 'wave':
                # Onda sinusoidal
                bonus = 0.15 * (1 + np.sin(2 * np.pi * norm_x * 2))
            else:
                bonus = 0

            # Aplicar bonus (sin exceder 1.0)
            cell.energy = min(cell.energy + bonus, 1.0)

    def get_centroid(self):
        """Calcula centroide del organismo."""
        if not self.org.cells:
            return (0, 0)
        x_sum = sum(c.position[0] for c in self.org.cells)
        y_sum = sum(c.position[1] for c in self.org.cells)
        n = len(self.org.cells)
        return (x_sum / n, y_sum / n)

    def get_fi_centroid(self):
        """Calcula centroide de los Fi."""
        fi_cells = [c for c in self.org.cells if c.role_idx == 1]
        if not fi_cells:
            return (0, 0)
        x_sum = sum(c.position[0] for c in fi_cells)
        y_sum = sum(c.position[1] for c in fi_cells)
        n = len(fi_cells)
        return (x_sum / n, y_sum / n)

    def run_phase(self, n_steps: int, gradient_type: str, label: str):
        """Ejecuta una fase del experimento."""
        print(f'\n--- {label} ({gradient_type}) ---')

        initial_centroid = self.get_centroid()
        initial_fi_centroid = self.get_fi_centroid()

        phase_positions = []

        for step in range(n_steps):
            # Aplicar gradiente
            self.apply_gradient(gradient_type)

            # Step del organismo
            self.org.step()

            # Guardar metricas
            m = self.org.get_metrics()
            centroid = self.get_centroid()
            fi_centroid = self.get_fi_centroid()

            m['centroid_x'] = centroid[0]
            m['centroid_y'] = centroid[1]
            m['fi_centroid_x'] = fi_centroid[0]
            m['fi_centroid_y'] = fi_centroid[1]
            m['gradient'] = gradient_type

            self.history.append(m)
            phase_positions.append(centroid)

            if (step + 1) % 25 == 0:
                print(f'  Step {step+1}: Centroid=({centroid[0]:.1f}, {centroid[1]:.1f}), '
                      f'Fi={m["n_fi"]}, Coord={m["coordination"]:.3f}')

        final_centroid = self.get_centroid()
        final_fi_centroid = self.get_fi_centroid()

        # Calcular desplazamiento
        dx = final_centroid[0] - initial_centroid[0]
        dy = final_centroid[1] - initial_centroid[1]
        displacement = np.sqrt(dx**2 + dy**2)

        print(f'  Desplazamiento total: {displacement:.2f} celdas')
        print(f'  Direccion: dx={dx:.1f}, dy={dy:.1f}')

        return {
            'initial': initial_centroid,
            'final': final_centroid,
            'displacement': displacement,
            'dx': dx,
            'dy': dy,
            'positions': phase_positions
        }


def run_migration_experiment():
    """Ejecuta experimento completo de migracion."""
    print('='*70)
    print('EXPERIMENTO: MIGRACION CON GRADIENTES DE ENERGIA')
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
    exp = MigrationExperiment(org)
    initial_centroid = exp.get_centroid()
    print(f'Centroide inicial: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})')
    print(f'Fi={baseline["n_fi"]}, Coord={baseline["coordination"]:.3f}')

    # Guardar posiciones iniciales
    initial_positions = [(c.position, c.role_idx) for c in org.cells]

    # Fases con diferentes gradientes
    results = {}

    # Fase 1: Gradiente lineal X (derecha)
    results['linear_x'] = exp.run_phase(100, 'linear_x', 'FASE 1: Gradiente hacia derecha')

    # Fase 2: Gradiente radial al centro
    results['radial_center'] = exp.run_phase(100, 'radial_center', 'FASE 2: Gradiente hacia centro')

    # Fase 3: Gradiente hacia esquina
    results['radial_corner'] = exp.run_phase(100, 'radial_corner', 'FASE 3: Gradiente hacia esquina')

    # Analisis
    print('\n' + '='*70)
    print('ANALISIS DE MIGRACION')
    print('='*70)

    print(f'\n{"Gradiente":<20} {"Desplazamiento":<15} {"Direccion":<20}')
    print('-'*55)
    for name, r in results.items():
        direction = f'dx={r["dx"]:+.1f}, dy={r["dy"]:+.1f}'
        print(f'{name:<20} {r["displacement"]:<15.2f} {direction:<20}')

    # Verificar si la migracion fue en la direccion correcta
    print('\n*** VERIFICACION DE DIRECCION ***')

    # Linear X: deberia ir hacia derecha (dx > 0)
    if results['linear_x']['dx'] > 0:
        print('  Linear X: CORRECTO - migro hacia derecha')
    else:
        print('  Linear X: INCORRECTO - no migro hacia derecha')

    # Radial center: deberia acercarse al centro (24, 24)
    final_c = results['radial_center']['final']
    center = (24, 24)
    dist_to_center = np.sqrt((final_c[0] - center[0])**2 + (final_c[1] - center[1])**2)
    initial_c = results['radial_center']['initial']
    initial_dist = np.sqrt((initial_c[0] - center[0])**2 + (initial_c[1] - center[1])**2)
    if dist_to_center < initial_dist:
        print(f'  Radial centro: CORRECTO - se acerco al centro ({dist_to_center:.1f} < {initial_dist:.1f})')
    else:
        print(f'  Radial centro: PARCIAL - distancia {dist_to_center:.1f} vs {initial_dist:.1f}')

    # Radial corner: deberia ir hacia (48, 48)
    if results['radial_corner']['dx'] > 0 and results['radial_corner']['dy'] > 0:
        print('  Radial esquina: CORRECTO - migro hacia esquina superior derecha')
    else:
        print('  Radial esquina: PARCIAL - migracion no optima')

    # Visualizacion
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Trayectoria del centroide
    ax = axes[0, 0]
    colors = {'linear_x': 'red', 'radial_center': 'green', 'radial_corner': 'blue'}
    for name, r in results.items():
        positions = r['positions']
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.plot(xs, ys, '-', color=colors[name], linewidth=2, label=name, alpha=0.7)
        ax.scatter(xs[0], ys[0], c=colors[name], s=100, marker='o', edgecolors='black')
        ax.scatter(xs[-1], ys[-1], c=colors[name], s=100, marker='s', edgecolors='black')
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 48)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trayectoria del Centroide')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 2. Evolucion de centroide X
    ax = axes[0, 1]
    cx_vals = [h['centroid_x'] for h in exp.history]
    ax.plot(cx_vals, 'b-', linewidth=2)
    # Marcar fases
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
    ax.text(50, max(cx_vals)*0.95, 'Linear X', ha='center', fontsize=9)
    ax.text(150, max(cx_vals)*0.95, 'Radial C', ha='center', fontsize=9)
    ax.text(250, max(cx_vals)*0.95, 'Radial E', ha='center', fontsize=9)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Centroide X')
    ax.set_title('Posicion X del Organismo')
    ax.grid(True, alpha=0.3)

    # 3. Evolucion de centroide Y
    ax = axes[0, 2]
    cy_vals = [h['centroid_y'] for h in exp.history]
    ax.plot(cy_vals, 'g-', linewidth=2)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Centroide Y')
    ax.set_title('Posicion Y del Organismo')
    ax.grid(True, alpha=0.3)

    # 4. Desplazamiento por gradiente
    ax = axes[1, 0]
    names = list(results.keys())
    displacements = [results[n]['displacement'] for n in names]
    bars = ax.bar(names, displacements, color=['red', 'green', 'blue'], alpha=0.7)
    ax.set_ylabel('Desplazamiento (celdas)')
    ax.set_title('Desplazamiento Total por Gradiente')
    ax.grid(True, alpha=0.3)

    # 5. Evolucion de Fi durante migracion
    ax = axes[1, 1]
    fi_vals = [h['n_fi'] for h in exp.history]
    ax.plot(fi_vals, 'r-', linewidth=2)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Estabilidad de Fi durante Migracion')
    ax.grid(True, alpha=0.3)

    # 6. Estado final con gradiente visualizado
    ax = axes[1, 2]
    # Dibujar gradiente de fondo (radial corner)
    gradient_bg = np.zeros((48, 48))
    for i in range(48):
        for j in range(48):
            dist = np.sqrt((i - 48)**2 + (j - 48)**2)
            gradient_bg[j, i] = 1 - dist / (np.sqrt(2) * 48)
    ax.imshow(gradient_bg, extent=[0, 48, 0, 48], origin='lower', cmap='YlOrRd', alpha=0.3)

    # Dibujar celulas
    for cell in org.cells:
        x, y = cell.position
        color = ['blue', 'red', 'black'][cell.role_idx]
        size = 20 + cell.energy * 60
        ax.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='white', linewidths=0.5)

    ax.set_xlim(0, 48)
    ax.set_ylim(0, 48)
    ax.set_title('Estado Final (fondo=gradiente)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('zeta_organism_migracion.png', dpi=150)
    print('\nGuardado: zeta_organism_migracion.png')

    # Resumen final
    final = org.get_metrics()
    total_displacement = np.sqrt(
        (exp.get_centroid()[0] - initial_centroid[0])**2 +
        (exp.get_centroid()[1] - initial_centroid[1])**2
    )

    print(f'\n*** RESUMEN ***')
    print(f'Centroide inicial: ({initial_centroid[0]:.1f}, {initial_centroid[1]:.1f})')
    print(f'Centroide final: ({exp.get_centroid()[0]:.1f}, {exp.get_centroid()[1]:.1f})')
    print(f'Desplazamiento total: {total_displacement:.2f} celdas')
    print(f'Fi final: {final["n_fi"]}')
    print(f'Coordinacion final: {final["coordination"]:.3f}')

    return exp, results


if __name__ == '__main__':
    run_migration_experiment()
