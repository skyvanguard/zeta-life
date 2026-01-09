# exp_simbiosis.py
"""Experimento: Simbiosis mutualista entre dos organismos.

Hipotesis: Cuando dos organismos intercambian energia mutuamente al estar cerca,
deberian:
1. Acercarse en lugar de segregarse
2. Tener mayor energia promedio que en aislamiento
3. Generar mas Fi por el exceso de energia

Escenarios:
- Control Aislado: Un organismo solo
- Control Competencia: Dos organismos sin mutualismo (comportamiento actual)
- Experimental Mutualismo: Dos organismos con transferencia mutua de energia
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from exp_dos_organismos import DualOrganism, DualCellEntity
from zeta_life.organism import ZetaOrganism


class SymbioticDualOrganism(DualOrganism):
    """Extiende DualOrganism con mecanica de mutualismo."""

    def __init__(self, mutualism_radius: float = 3.0,
                 mutualism_rate: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.mutualism_radius = mutualism_radius
        self.mutualism_rate = mutualism_rate
        self.mutualism_events = 0  # Contador de eventos de mutualismo

    def apply_mutualism(self):
        """Aplica transferencia mutua de energia entre organismos cercanos."""
        org0_cells = [c for c in self.cells if c.organism_id == 0]
        org1_cells = [c for c in self.cells if c.organism_id == 1]

        events = 0
        for cell_a in org0_cells:
            ax, ay = cell_a.position
            for cell_b in org1_cells:
                bx, by = cell_b.position
                dist = np.sqrt((ax - bx)**2 + (ay - by)**2)

                if dist <= self.mutualism_radius:
                    # Transferencia proporcional a proximidad
                    proximity_factor = 1 - (dist / self.mutualism_radius)
                    bonus = self.mutualism_rate * proximity_factor

                    # Ambos ganan energia
                    cell_a.energy = min(1.0, cell_a.energy + bonus)
                    cell_b.energy = min(1.0, cell_b.energy + bonus)
                    events += 1

        self.mutualism_events = events
        return events

    def step(self):
        """Step con mutualismo aplicado antes de la dinamica normal."""
        self.apply_mutualism()
        super().step()

    def get_metrics(self):
        """Metricas extendidas con info de mutualismo."""
        metrics = super().get_metrics()
        metrics['mutualism_events'] = self.mutualism_events

        # Calcular distancia entre centroides
        c0 = metrics['org_0']['centroid']
        c1 = metrics['org_1']['centroid']
        if c0 != (0, 0) and c1 != (0, 0):
            metrics['centroid_distance'] = np.sqrt(
                (c0[0] - c1[0])**2 + (c0[1] - c1[1])**2
            )
        else:
            metrics['centroid_distance'] = 0

        return metrics


def run_isolated_control(n_steps=300):
    """Control: Un organismo aislado."""
    print('\n' + '='*60)
    print('CONTROL: ORGANISMO AISLADO')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ZetaOrganism(
        grid_size=64,
        n_cells=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior.load_state_dict(weights['behavior_state'])
    except:
        pass

    org.initialize(seed_fi=True)

    # Registrar metricas
    history = []
    for step in range(n_steps):
        org.step()
        m = org.get_metrics()
        n_total = len(org.cells)
        avg_energy = np.mean([c.energy for c in org.cells]) if org.cells else 0
        history.append({
            'n_fi': m['n_fi'],
            'n_total': n_total,
            'avg_energy': avg_energy
        })

        if (step + 1) % 100 == 0:
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Energia={avg_energy:.3f}')

    final = history[-1]
    print(f'\n  Final: Fi={final["n_fi"]}, Energia={final["avg_energy"]:.3f}')

    return history


def run_competition_control(n_steps=300, close_start=True):
    """Control: Dos organismos compitiendo (sin mutualismo)."""
    print('\n' + '='*60)
    print('CONTROL: COMPETENCIA (sin mutualismo)')
    print('='*60)

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

    if close_start:
        initialize_close(dual)  # Mismo inicio que mutualismo
    else:
        dual.initialize(separation='horizontal')

    for step in range(n_steps):
        dual.step()

        if (step + 1) % 100 == 0:
            m = dual.get_metrics()
            # Calcular distancia entre centroides
            c0 = m['org_0']['centroid']
            c1 = m['org_1']['centroid']
            dist = np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2) if c0 != (0,0) else 0
            print(f'  Step {step+1}:')
            print(f'    Org 0: Fi={m["org_0"]["n_fi"]}, Energia={m["org_0"]["avg_energy"]:.3f}')
            print(f'    Org 1: Fi={m["org_1"]["n_fi"]}, Energia={m["org_1"]["avg_energy"]:.3f}')
            print(f'    Distancia centroides: {dist:.1f}')

    final = dual.get_metrics()
    c0 = final['org_0']['centroid']
    c1 = final['org_1']['centroid']
    final_dist = np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2) if c0 != (0,0) else 0
    print(f'\n  Final:')
    print(f'    Org 0: Fi={final["org_0"]["n_fi"]}, Total={final["org_0"]["n_total"]}')
    print(f'    Org 1: Fi={final["org_1"]["n_fi"]}, Total={final["org_1"]["n_total"]}')
    print(f'    Distancia centroides: {final_dist:.1f}')

    return dual


def initialize_close(dual):
    """Inicializa organismos cerca del centro (forzando interaccion)."""
    dual.cells = []
    center = dual.grid_size // 2

    for org_id in range(2):
        for i in range(dual.n_cells_per_org):
            # Ambos en el centro, MUY solapados (offset minimo)
            offset_x = -3 if org_id == 0 else 3
            x = np.random.randint(center + offset_x - 8, center + offset_x + 8)
            y = np.random.randint(center - 8, center + 8)
            x = np.clip(x, 0, dual.grid_size - 1)
            y = np.clip(y, 0, dual.grid_size - 1)

            state = torch.randn(dual.state_dim) * 0.1

            if i == 0:
                role = torch.tensor([0.0, 1.0, 0.0])
                energy = 0.6  # Energia inicial mas baja
            else:
                role = torch.tensor([1.0, 0.0, 0.0])
                energy = np.random.uniform(0.2, 0.4)  # Energia inicial mas baja

            cell = DualCellEntity(
                position=(x, y),
                state=state,
                role=role,
                energy=energy,
                organism_id=org_id
            )
            dual.cells.append(cell)

    dual._update_grids()


def run_mutualism_experiment(n_steps=300, mutualism_rate=0.05):
    """Experimental: Dos organismos con mutualismo."""
    print('\n' + '='*60)
    print(f'EXPERIMENTAL: MUTUALISMO (rate={mutualism_rate})')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    symbiotic = SymbioticDualOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        mutualism_radius=8.0,  # Radio mas amplio para capturar interacciones
        mutualism_rate=mutualism_rate
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        symbiotic.behavior_0.load_state_dict(weights['behavior_state'])
        symbiotic.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # Usar inicializacion cercana para forzar interaccion
    initialize_close(symbiotic)

    # Guardar posiciones iniciales
    initial_positions = [(c.position[0], c.position[1], c.organism_id, c.role_idx)
                        for c in symbiotic.cells]

    for step in range(n_steps):
        symbiotic.step()

        if (step + 1) % 100 == 0:
            m = symbiotic.get_metrics()
            print(f'  Step {step+1}:')
            print(f'    Org 0: Fi={m["org_0"]["n_fi"]}, Energia={m["org_0"]["avg_energy"]:.3f}')
            print(f'    Org 1: Fi={m["org_1"]["n_fi"]}, Energia={m["org_1"]["avg_energy"]:.3f}')
            print(f'    Eventos mutualismo: {m["mutualism_events"]}')
            print(f'    Distancia centroides: {m["centroid_distance"]:.1f}')

    final = symbiotic.get_metrics()
    print(f'\n  Final:')
    print(f'    Org 0: Fi={final["org_0"]["n_fi"]}, Total={final["org_0"]["n_total"]}')
    print(f'    Org 1: Fi={final["org_1"]["n_fi"]}, Total={final["org_1"]["n_total"]}')
    print(f'    Eventos mutualismo: {final["mutualism_events"]}')

    return symbiotic, initial_positions


def run_full_comparison():
    """Ejecuta los tres escenarios y compara."""
    print('='*70)
    print('EXPERIMENTO: SIMBIOSIS MUTUALISTA')
    print('='*70)

    n_steps = 300

    # 1. Control aislado
    isolated_history = run_isolated_control(n_steps)

    # 2. Control competencia
    competition = run_competition_control(n_steps)

    # 3. Mutualismo
    mutualism, mut_initial = run_mutualism_experiment(n_steps, mutualism_rate=0.05)

    # === ANALISIS COMPARATIVO ===
    print('\n' + '='*70)
    print('ANALISIS COMPARATIVO')
    print('='*70)

    # Extraer metricas finales
    isolated_final = isolated_history[-1]
    comp_final = competition.get_metrics()
    mut_final = mutualism.get_metrics()

    # Calcular promedios para organismos duales
    comp_avg_energy = (comp_final['org_0']['avg_energy'] + comp_final['org_1']['avg_energy']) / 2
    comp_total_fi = comp_final['org_0']['n_fi'] + comp_final['org_1']['n_fi']

    mut_avg_energy = (mut_final['org_0']['avg_energy'] + mut_final['org_1']['avg_energy']) / 2
    mut_total_fi = mut_final['org_0']['n_fi'] + mut_final['org_1']['n_fi']

    print(f'\n{"Metrica":<25} {"Aislado":<15} {"Competencia":<15} {"Mutualismo":<15}')
    print('-'*70)
    print(f'{"Energia promedio":<25} {isolated_final["avg_energy"]:<15.3f} {comp_avg_energy:<15.3f} {mut_avg_energy:<15.3f}')
    print(f'{"Fi totales":<25} {isolated_final["n_fi"]:<15} {comp_total_fi:<15} {mut_total_fi:<15}')
    print(f'{"Contactos frontera":<25} {"N/A":<15} {comp_final["boundary_contacts"]:<15} {mut_final["boundary_contacts"]:<15}')
    print(f'{"Eventos mutualismo":<25} {"N/A":<15} {"N/A":<15} {mut_final["mutualism_events"]:<15}')

    # Calcular mejoras
    print('\n*** COMPARACION CON COMPETENCIA ***')
    energy_improvement = (mut_avg_energy - comp_avg_energy) / comp_avg_energy * 100
    fi_improvement = (mut_total_fi - comp_total_fi) / max(1, comp_total_fi) * 100
    proximity_improvement = mut_final['boundary_contacts'] - comp_final['boundary_contacts']

    print(f'  Energia: {energy_improvement:+.1f}%')
    print(f'  Fi: {fi_improvement:+.1f}%')
    print(f'  Proximidad: {proximity_improvement:+d} contactos')

    # Determinar resultado
    print('\n*** RESULTADO ***')
    if mut_final['mutualism_events'] > 10:
        if energy_improvement > 5:
            print('  [EXITO] Mutualismo produce beneficio energetico significativo')
        elif energy_improvement > 0:
            print('  [PARCIAL] Mutualismo produce beneficio energetico modesto')
        else:
            print('  [NEUTRAL] Mutualismo no produce beneficio energetico')

        if proximity_improvement > 5:
            print('  [EXITO] Los organismos se acercan (clustering)')
        elif proximity_improvement > 0:
            print('  [PARCIAL] Leve aumento de proximidad')
        else:
            print('  [FALLO] Los organismos no se acercan')
    else:
        print('  [FALLO] Pocos eventos de mutualismo - organismos no interactuan')

    # === VISUALIZACION ===
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Colores
    colors = {'org0': 'royalblue', 'org1': 'crimson'}

    # 1. Estado inicial mutualismo
    ax = axes[0, 0]
    for x, y, org_id, role_idx in mut_initial:
        color = colors['org0'] if org_id == 0 else colors['org1']
        marker = 's' if role_idx == 1 else 'o'
        size = 80 if role_idx == 1 else 25
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
    ax.axvline(x=32, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Mutualismo: Inicial')
    ax.set_aspect('equal')

    # 2. Estado final mutualismo
    ax = axes[0, 1]
    for cell in mutualism.cells:
        x, y = cell.position
        color = colors['org0'] if cell.organism_id == 0 else colors['org1']
        marker = 's' if cell.role_idx == 1 else 'o'
        size = 80 if cell.role_idx == 1 else 25
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
    ax.axvline(x=32, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Mutualismo: Final')
    ax.set_aspect('equal')

    # 3. Estado final competencia
    ax = axes[0, 2]
    for cell in competition.cells:
        x, y = cell.position
        color = colors['org0'] if cell.organism_id == 0 else colors['org1']
        marker = 's' if cell.role_idx == 1 else 'o'
        size = 80 if cell.role_idx == 1 else 25
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
    ax.axvline(x=32, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Competencia: Final')
    ax.set_aspect('equal')

    # 4. Comparacion de energia
    ax = axes[0, 3]
    scenarios = ['Aislado', 'Competencia', 'Mutualismo']
    energies = [isolated_final['avg_energy'], comp_avg_energy, mut_avg_energy]
    bars = ax.bar(scenarios, energies, color=['gray', 'orange', 'green'], alpha=0.7)
    ax.set_ylabel('Energia Promedio')
    ax.set_title('Comparacion de Energia')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # 5. Evolucion de energia (mutualismo)
    ax = axes[1, 0]
    steps = range(len(mutualism.history))
    e0 = [h['org_0']['avg_energy'] for h in mutualism.history]
    e1 = [h['org_1']['avg_energy'] for h in mutualism.history]
    ax.plot(steps, e0, color=colors['org0'], linewidth=2, label='Org 0')
    ax.plot(steps, e1, color=colors['org1'], linewidth=2, label='Org 1')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Energia Promedio')
    ax.set_title('Mutualismo: Evolucion Energia')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Evolucion de eventos mutualismo
    ax = axes[1, 1]
    mut_events = [h['mutualism_events'] for h in mutualism.history]
    ax.plot(steps, mut_events, color='green', linewidth=2)
    ax.fill_between(steps, mut_events, alpha=0.3, color='green')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Eventos de Mutualismo')
    ax.set_title('Interacciones Simbioticas')
    ax.grid(True, alpha=0.3)

    # 7. Evolucion de distancia centroides
    ax = axes[1, 2]
    mut_dist = [h['centroid_distance'] for h in mutualism.history]
    comp_dist = [h.get('centroid_distance', 0) for h in competition.history]
    # Calcular distancia para competencia si no existe
    if comp_dist[0] == 0:
        comp_dist = []
        for h in competition.history:
            c0 = h['org_0']['centroid']
            c1 = h['org_1']['centroid']
            d = np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2) if c0 != (0,0) else 0
            comp_dist.append(d)

    ax.plot(range(len(mut_dist)), mut_dist, color='green', linewidth=2, label='Mutualismo')
    ax.plot(range(len(comp_dist)), comp_dist, color='orange', linewidth=2, label='Competencia')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Distancia entre Centroides')
    ax.set_title('Proximidad Inter-Organismo')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Comparacion de Fi
    ax = axes[1, 3]
    fi_data = [isolated_final['n_fi'], comp_total_fi, mut_total_fi]
    bars = ax.bar(scenarios, fi_data, color=['gray', 'orange', 'green'], alpha=0.7)
    ax.set_ylabel('Fi Totales')
    ax.set_title('Comparacion de Lideres')
    for bar, val in zip(bars, fi_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('zeta_organism_simbiosis.png', dpi=150)
    print('\nGuardado: zeta_organism_simbiosis.png')

    return {
        'isolated': isolated_history,
        'competition': competition,
        'mutualism': mutualism,
        'energy_improvement': energy_improvement,
        'fi_improvement': fi_improvement,
        'proximity_improvement': proximity_improvement
    }


if __name__ == '__main__':
    results = run_full_comparison()
