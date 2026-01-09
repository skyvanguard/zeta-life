# exp_tres_organismos_v2.py
"""Experimento: Tres organismos con recursos distribuidos.

Hipotesis: Con recursos distribuidos en 3 zonas (una cerca de cada organismo),
deberia emerger coexistencia territorial en lugar de dominacion total.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from exp_tres_organismos import TripleOrganism, TriCellEntity


def create_distributed_resources(grid_size, n_zones=3):
    """Crea recursos distribuidos en multiples zonas."""
    resource_grid = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    radius = grid_size // 3

    # Posiciones de zonas de recursos (triangulo, como los organismos)
    zones = [
        (center, center - radius),                          # Arriba
        (center - int(radius * 0.866), center + radius // 2),  # Abajo-izq
        (center + int(radius * 0.866), center + radius // 2),  # Abajo-der
    ]

    # Crear cada zona de recursos
    zone_radius = grid_size // 6
    for zx, zy in zones:
        for y in range(grid_size):
            for x in range(grid_size):
                dist = np.sqrt((x - zx)**2 + (y - zy)**2)
                if dist < zone_radius * 2:
                    contribution = max(0, 1 - dist / (zone_radius * 2))
                    resource_grid[y, x] = max(resource_grid[y, x], contribution)

    return resource_grid, zones


def run_distributed_experiment():
    """Ejecuta experimento con recursos distribuidos."""
    print('='*70)
    print('EXPERIMENTO: TRES ORGANISMOS CON RECURSOS DISTRIBUIDOS')
    print('='*70)

    torch.manual_seed(42)
    np.random.seed(42)

    tri = TripleOrganism(
        grid_size=64,
        n_cells_per_org=30,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        total_energy=90.0  # Mas energia total
    )

    # Cargar pesos
    try:
        weights = torch.load('zeta_organism_weights.pt')
        for b in tri.behaviors:
            b.load_state_dict(weights['behavior_state'])
        print('Pesos cargados!')
    except:
        pass

    # Inicializar con recursos distribuidos
    tri.initialize()
    tri.resource_grid, resource_zones = create_distributed_resources(tri.grid_size)

    colors = ['Azul', 'Rojo', 'Verde']
    print(f'\nConfiguracion:')
    print(f'  Recursos: 3 zonas distribuidas')
    print(f'  Energia total: {tri.total_energy}')
    print(f'  Zonas de recursos:')
    for i, (zx, zy) in enumerate(resource_zones):
        print(f'    Zona {i}: ({zx}, {zy})')

    initial = tri.get_metrics()
    print(f'\nEstado inicial:')
    for i in range(3):
        m = initial[f'org_{i}']
        print(f'  Org {i} ({colors[i]}): Total={m["n_total"]}, Energia={m["total_energy"]:.1f}')

    # Guardar posiciones iniciales
    initial_positions = [(c.position[0], c.position[1], c.organism_id, c.role_idx)
                        for c in tri.cells]

    # Simulacion
    print(f'\n[SIMULACION] 400 steps...')
    for step in range(400):
        tri.step()

        if (step + 1) % 100 == 0:
            m = tri.get_metrics()
            print(f'\n  Step {step+1}:')
            for i in range(3):
                org = m[f'org_{i}']
                print(f'    Org {i} ({colors[i]}): Total={org["n_total"]}, Fi={org["n_fi"]}, '
                      f'Energia={org["total_energy"]:.1f}')

    final = tri.get_metrics()

    # Analisis
    print('\n' + '='*70)
    print('ANALISIS DE COEXISTENCIA')
    print('='*70)

    print(f'\n{"Organismo":<15} {"Inicial":<10} {"Final":<10} {"Cambio":<10} {"Energia":<10} {"%Total":<10}')
    print('-'*65)

    total_energy = sum(final[f'org_{i}']['total_energy'] for i in range(3))
    for i in range(3):
        init_total = tri.n_cells_per_org
        final_total = final[f'org_{i}']['n_total']
        change = final_total - init_total
        energy = final[f'org_{i}']['total_energy']
        pct = (energy / total_energy * 100) if total_energy > 0 else 0
        print(f'Org {i} ({colors[i]:<5}) {init_total:<10} {final_total:<10} {change:+d}{"":<6} {energy:<10.1f} {pct:<10.1f}%')

    # Determinar resultado
    print('\n*** RESULTADO ***')

    alive = [i for i in range(3) if final[f'org_{i}']['n_total'] > 0]
    extinct = [i for i in range(3) if final[f'org_{i}']['n_total'] == 0]

    if len(extinct) > 0:
        for e in extinct:
            print(f'  EXTINCION: Organismo {e} ({colors[e]})')

    if len(alive) == 3:
        # Verificar si hay equilibrio
        totals = [final[f'org_{i}']['n_total'] for i in range(3)]
        energies = [final[f'org_{i}']['total_energy'] for i in range(3)]

        if max(totals) < min(totals) * 2:
            print('  COEXISTENCIA EQUILIBRADA: Los tres organismos sobreviven')
        else:
            dominant = totals.index(max(totals))
            print(f'  COEXISTENCIA DESIGUAL: Org {dominant} ({colors[dominant]}) domina pero otros sobreviven')

        # Territorialidad
        print(f'\n  Distribucion de poblacion: {totals[0]} / {totals[1]} / {totals[2]}')
        print(f'  Distribucion de energia: {energies[0]:.1f} / {energies[1]:.1f} / {energies[2]:.1f}')

    elif len(alive) == 2:
        print(f'  DUOPOLIO: Organismos {alive[0]} y {alive[1]} sobreviven')
    else:
        winner = alive[0]
        print(f'  DOMINACION TOTAL: Solo Org {winner} ({colors[winner]}) sobrevive')

    # Calcular indice de diversidad (Shannon)
    total_cells = sum(final[f'org_{i}']['n_total'] for i in range(3))
    if total_cells > 0:
        proportions = [final[f'org_{i}']['n_total'] / total_cells for i in range(3)]
        shannon = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
        max_shannon = np.log(3)  # Diversidad maxima
        evenness = shannon / max_shannon
        print(f'\n  Indice de diversidad Shannon: {shannon:.3f} (max={max_shannon:.3f})')
        print(f'  Equitatividad: {evenness:.3f} (1.0 = perfectamente igual)')

    # Visualizacion
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    org_colors = ['royalblue', 'crimson', 'forestgreen']

    # 1. Estado inicial con zonas de recursos
    ax = axes[0, 0]
    ax.imshow(tri.resource_grid, extent=[0, 64, 0, 64], origin='lower',
              cmap='YlOrRd', alpha=0.4)
    for x, y, org_id, role_idx in initial_positions:
        color = org_colors[org_id]
        marker = 's' if role_idx == 1 else 'o'
        size = 80 if role_idx == 1 else 25
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
    # Marcar centros de zonas
    for i, (zx, zy) in enumerate(resource_zones):
        ax.scatter(zx, zy, c='gold', s=150, marker='*', edgecolors='black', linewidths=1)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Inicial + 3 Zonas de Recursos')
    ax.set_aspect('equal')

    # 2. Estado final
    ax = axes[0, 1]
    ax.imshow(tri.resource_grid, extent=[0, 64, 0, 64], origin='lower',
              cmap='YlOrRd', alpha=0.3)
    for cell in tri.cells:
        x, y = cell.position
        color = org_colors[cell.organism_id]
        marker = 's' if cell.role_idx == 1 else 'o'
        size = 80 if cell.role_idx == 1 else 25
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.8)
    for i, (zx, zy) in enumerate(resource_zones):
        ax.scatter(zx, zy, c='gold', s=150, marker='*', edgecolors='black', linewidths=1)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Final')
    ax.set_aspect('equal')

    # 3. Evolucion de poblacion
    ax = axes[0, 2]
    steps = range(len(tri.history))
    for i in range(3):
        totals = [h[f'org_{i}']['n_total'] for h in tri.history]
        ax.plot(steps, totals, color=org_colors[i], linewidth=2, label=f'Org {i} ({colors[i]})')
    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.5, label='Inicial')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Poblacion')
    ax.set_title('Evolucion de Poblacion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Evolucion de energia
    ax = axes[1, 0]
    for i in range(3):
        energies = [h[f'org_{i}']['total_energy'] for h in tri.history]
        ax.plot(steps, energies, color=org_colors[i], linewidth=2, label=f'Org {i}')
    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.5, label='Equitativo')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Energia Total')
    ax.set_title('Distribucion de Energia')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Evolucion de Fi
    ax = axes[1, 1]
    for i in range(3):
        fis = [h[f'org_{i}']['n_fi'] for h in tri.history]
        ax.plot(steps, fis, color=org_colors[i], linewidth=2, label=f'Org {i}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Evolucion de Lideres')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Diagrama de torta final
    ax = axes[1, 2]
    final_totals = [final[f'org_{i}']['n_total'] for i in range(3)]
    if sum(final_totals) > 0:
        # Solo incluir los que tienen poblacion
        labels = [f'Org {i} ({colors[i]})' for i in range(3) if final_totals[i] > 0]
        sizes = [t for t in final_totals if t > 0]
        plot_colors = [org_colors[i] for i in range(3) if final_totals[i] > 0]
        ax.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%',
               startangle=90, explode=[0.02]*len(sizes))
        ax.set_title('Distribucion Final de Poblacion')
    else:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('zeta_organism_triple_distributed.png', dpi=150)
    print('\nGuardado: zeta_organism_triple_distributed.png')

    return tri


if __name__ == '__main__':
    run_distributed_experiment()
