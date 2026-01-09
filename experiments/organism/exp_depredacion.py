# exp_depredacion.py
"""Experimento: Depredacion asimetrica entre dos organismos.

Hipotesis: Con tasas de conversion asimetricas (depredador 40%, presa 5%):
1. El depredador expandira territorio consumiendo presa
2. La presa desarrollara estrategias de evasion
3. Posible extincion o equilibrio depredador-presa

Escenarios:
- Control: Conversion simetrica (10%/10%)
- Depredacion: Asimetrica (40%/5%)
- Depredacion extrema: Muy asimetrica (60%/2%)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict

from zeta_life.organism import ForceField
from zeta_life.organism import BehaviorEngine
from zeta_life.organism import OrganismCell
from exp_dos_organismos import DualOrganism, DualCellEntity


class PredatorPreyOrganism(DualOrganism):
    """Dos organismos con relacion depredador-presa."""

    def __init__(self, predator_id: int = 0,
                 predator_conversion: float = 0.40,
                 prey_conversion: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.predator_id = predator_id
        self.prey_id = 1 - predator_id
        self.predator_conversion = predator_conversion
        self.prey_conversion = prey_conversion

        # Contadores para estadisticas
        self.conversions_by_predator = 0
        self.conversions_by_prey = 0

    def step(self):
        """Step con logica de depredacion por caza directa.

        Mecanismo de caza:
        - Fi depredadores pueden "cazar" celulas presa cercanas
        - La probabilidad depende de la distancia (mas cerca = mas probable)
        - Celulas presa pueden ser convertidas incluso si son Fi (debiles)
        - Las presas intentan huir de depredadores cercanos
        """
        # Resetear contadores
        self.conversions_by_predator = 0
        self.conversions_by_prey = 0

        # === FASE 1: CAZA DIRECTA ===
        # Los Fi de cada organismo intentan convertir celulas cercanas del enemigo
        hunt_radius = 4.0  # Radio de caza

        # Identificar cazadores (Fi de cada organismo)
        predator_hunters = [c for c in self.cells
                          if c.organism_id == self.predator_id and c.role_idx == 1]
        prey_hunters = [c for c in self.cells
                       if c.organism_id == self.prey_id and c.role_idx == 1]

        # Marcar celulas para conversion
        cells_to_convert = {}  # cell_idx -> new_org_id

        for cell_idx, cell in enumerate(self.cells):
            # Celulas presa cerca de Fi depredador
            if cell.organism_id == self.prey_id:
                for hunter in predator_hunters:
                    dist = np.sqrt((hunter.position[0] - cell.position[0])**2 +
                                  (hunter.position[1] - cell.position[1])**2)
                    if dist <= hunt_radius:
                        # Probabilidad basada en distancia y tipo
                        base_prob = self.predator_conversion * (1 - dist/hunt_radius)
                        # Fi son mas dificiles de convertir que MASS
                        if cell.role_idx == 1:
                            base_prob *= 0.3  # Fi resiste mejor
                        if np.random.random() < base_prob:
                            cells_to_convert[cell_idx] = self.predator_id
                            self.conversions_by_predator += 1
                            break  # Solo una conversion por celula

            # Celulas depredador cerca de Fi presa (raro, pero posible)
            elif cell.organism_id == self.predator_id:
                for hunter in prey_hunters:
                    dist = np.sqrt((hunter.position[0] - cell.position[0])**2 +
                                  (hunter.position[1] - cell.position[1])**2)
                    if dist <= hunt_radius:
                        base_prob = self.prey_conversion * (1 - dist/hunt_radius)
                        if cell.role_idx == 1:
                            base_prob *= 0.3
                        if np.random.random() < base_prob:
                            cells_to_convert[cell_idx] = self.prey_id
                            self.conversions_by_prey += 1
                            break

        # Calcular campo de fuerzas
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        new_cells = []

        for cell_idx, cell in enumerate(self.cells):
            # Aplicar conversion de caza si aplica
            current_org_id = cells_to_convert.get(cell_idx, cell.organism_id)

            # Obtener vecinos
            all_neighbors = self._get_neighbors(cell, radius=5, same_org_only=False)
            # Vecinos del organismo ACTUAL (puede haber cambiado por caza)
            same_org_neighbors = [n for n in all_neighbors if n.organism_id == current_org_id]
            other_org_neighbors = [n for n in all_neighbors if n.organism_id != current_org_id]

            # Contar por tipo
            same_mass = sum(1 for n in same_org_neighbors if n.role_idx == 0)
            same_fi = sum(1 for n in same_org_neighbors if n.role_idx == 1)
            other_fi = sum(1 for n in other_org_neighbors if n.role_idx == 1)

            potential = field[0, 0, cell.position[1], cell.position[0]].item()

            # Seleccionar BehaviorEngine del organismo actual
            behavior = self.behavior_0 if current_org_id == 0 else self.behavior_1

            # Componente neural
            if same_org_neighbors:
                neighbor_states = torch.stack([n.state for n in same_org_neighbors])
                influence_out, influence_in = behavior.bidirectional_influence(
                    cell.state, neighbor_states
                )
                net_influence = (influence_out.mean() - influence_in).item()
                self_pattern = behavior.self_similarity(cell.state)
                cell_enriched = cell.state + 0.1 * self_pattern
                v_input = torch.cat([cell_enriched, torch.tensor([potential])])
                transformed = behavior.transform_net(v_input)
                new_state = transformed + 0.3 * cell.state
            else:
                net_influence = 0.0
                new_state = cell.state.clone()

            # === DINAMICA DE ENERGIA ===
            # Celulas recien convertidas pierden energia
            energy_penalty = 0.3 if cell_idx in cells_to_convert else 0.0

            if cell.role_idx == 1:  # Fi
                energy_gain = 0.02 * same_mass
                energy_loss = 0.03 * other_fi
                new_energy = cell.energy + energy_gain - energy_loss - energy_penalty
            else:  # Mass
                new_energy = cell.energy * 0.995 + 0.05 * max(0, potential)
                if same_fi > 0:
                    new_energy += 0.02
                if other_fi > 0:
                    new_energy -= 0.01
                new_energy -= energy_penalty

            new_energy += 0.02 * max(0, net_influence + 0.5)
            new_energy = np.clip(new_energy, 0, 1)

            # === TRANSICION DE ROL ===
            current_role_idx = cell.role_idx
            influence_score = net_influence + 0.5

            # Celulas convertidas pierden su estado Fi
            if cell_idx in cells_to_convert and cell.role_idx == 1:
                new_role = torch.tensor([1.0, 0.0, 0.0])  # Vuelve a MASS
            elif current_role_idx == 0:  # MASS
                can_become_fi = (
                    new_energy > self.fi_threshold and
                    same_mass >= 2 and
                    (same_fi == 0 or influence_score > 0.3) and
                    other_fi == 0
                )
                if can_become_fi:
                    new_role = torch.tensor([0.0, 1.0, 0.0])
                else:
                    new_role = torch.tensor([1.0, 0.0, 0.0])

            elif current_role_idx == 1:  # FORCE
                loses_fi = (
                    same_mass < 1 or
                    new_energy < 0.2 or
                    (other_fi > 0 and other_fi >= same_fi)
                )
                if loses_fi:
                    new_role = torch.tensor([1.0, 0.0, 0.0])
                else:
                    new_role = torch.tensor([0.0, 1.0, 0.0])
            else:
                new_role = cell.role.clone()

            # === MOVIMIENTO ===
            x, y = cell.position
            new_role_idx = new_role.argmax().item()

            # === MOVIMIENTO DE DEPREDADORES ===
            # Fi depredadores persiguen presas
            if current_org_id == self.predator_id and new_role_idx == 1:
                prey_cells = [c for c in self.cells if c.organism_id == self.prey_id]
                if prey_cells:
                    # Perseguir la presa mas cercana
                    nearest_prey = min(prey_cells, key=lambda p:
                        (p.position[0] - x)**2 + (p.position[1] - y)**2)
                    dx = int(np.sign(nearest_prey.position[0] - x))
                    dy = int(np.sign(nearest_prey.position[1] - y))
                    x = np.clip(x + dx, 0, self.grid_size - 1)
                    y = np.clip(y + dy, 0, self.grid_size - 1)

            elif new_role_idx == 0:  # Mass
                same_fi_cells = [c for c in self.cells
                                if c.organism_id == current_org_id and c.role_idx == 1]
                if same_fi_cells:
                    nearest_fi = min(same_fi_cells, key=lambda f:
                        (f.position[0] - x)**2 + (f.position[1] - y)**2)
                    dx = int(np.sign(nearest_fi.position[0] - x))
                    dy = int(np.sign(nearest_fi.position[1] - y))
                    x = np.clip(x + dx, 0, self.grid_size - 1)
                    y = np.clip(y + dy, 0, self.grid_size - 1)

                # === EVASION DE PRESA ===
                if current_org_id == self.prey_id:
                    nearby_predators = [c for c in self.cells
                                       if c.organism_id == self.predator_id and c.role_idx == 1
                                       and np.sqrt((c.position[0]-x)**2 + (c.position[1]-y)**2) < 10]
                    if nearby_predators:
                        nearest_pred = min(nearby_predators, key=lambda p:
                            (p.position[0] - x)**2 + (p.position[1] - y)**2)
                        # Huir con doble velocidad
                        dx = -2 * int(np.sign(nearest_pred.position[0] - x))
                        dy = -2 * int(np.sign(nearest_pred.position[1] - y))
                        x = np.clip(x + dx, 0, self.grid_size - 1)
                        y = np.clip(y + dy, 0, self.grid_size - 1)

            new_cell = DualCellEntity(
                position=(x, y),
                state=new_state.detach(),
                role=new_role,
                energy=new_energy,
                organism_id=current_org_id,
                controlled_mass=same_mass
            )
            new_cells.append(new_cell)

        self.cells = new_cells
        self._update_grids()
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Metricas extendidas con info de depredacion."""
        metrics = super().get_metrics()
        metrics['conversions_by_predator'] = self.conversions_by_predator
        metrics['conversions_by_prey'] = self.conversions_by_prey

        # Calcular distancia promedio presa-depredador
        predator_cells = [c for c in self.cells if c.organism_id == self.predator_id]
        prey_cells = [c for c in self.cells if c.organism_id == self.prey_id]

        if predator_cells and prey_cells:
            pred_centroid = np.mean([c.position[0] for c in predator_cells]), \
                           np.mean([c.position[1] for c in predator_cells])
            prey_centroid = np.mean([c.position[0] for c in prey_cells]), \
                           np.mean([c.position[1] for c in prey_cells])
            metrics['predator_prey_distance'] = np.sqrt(
                (pred_centroid[0] - prey_centroid[0])**2 +
                (pred_centroid[1] - prey_centroid[1])**2
            )
        else:
            metrics['predator_prey_distance'] = 0

        return metrics


def initialize_overlapping(org):
    """Inicializa organismos solapados para forzar interaccion."""
    org.cells = []
    center = org.grid_size // 2

    for org_id in range(2):
        for i in range(org.n_cells_per_org):
            # Ambos en el centro, muy solapados
            offset_x = -5 if org_id == 0 else 5
            x = np.random.randint(center + offset_x - 10, center + offset_x + 10)
            y = np.random.randint(center - 10, center + 10)
            x = np.clip(x, 0, org.grid_size - 1)
            y = np.clip(y, 0, org.grid_size - 1)

            state = torch.randn(org.state_dim) * 0.1

            if i == 0:
                role = torch.tensor([0.0, 1.0, 0.0])
                energy = 0.9
            else:
                role = torch.tensor([1.0, 0.0, 0.0])
                energy = np.random.uniform(0.2, 0.4)  # Energia baja para mantener MASS

            cell = DualCellEntity(
                position=(x, y),
                state=state,
                role=role,
                energy=energy,
                organism_id=org_id
            )
            org.cells.append(cell)

    org._update_grids()


def run_scenario(name, predator_conv, prey_conv, n_steps=400):
    """Ejecuta un escenario de depredacion."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Depredador->Presa: {predator_conv*100:.0f}%, Presa->Depredador: {prey_conv*100:.0f}%')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = PredatorPreyOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,  # Umbral normal - caza funciona con cualquier rol
        predator_id=0,
        predator_conversion=predator_conv,
        prey_conversion=prey_conv
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior_0.load_state_dict(weights['behavior_state'])
        org.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # Usar inicializacion solapada para forzar interaccion
    initialize_overlapping(org)

    # Guardar posiciones iniciales
    initial_positions = [(c.position[0], c.position[1], c.organism_id, c.role_idx)
                        for c in org.cells]

    initial = org.get_metrics()
    print(f'Estado inicial:')
    print(f'  Depredador (Org 0): {initial["org_0"]["n_total"]} celulas')
    print(f'  Presa (Org 1): {initial["org_1"]["n_total"]} celulas')

    # Tracking de extincion
    extinction_step = None
    total_conversions_pred = 0
    total_conversions_prey = 0

    for step in range(n_steps):
        org.step()

        m = org.get_metrics()
        total_conversions_pred += m['conversions_by_predator']
        total_conversions_prey += m['conversions_by_prey']

        # Detectar extincion
        if extinction_step is None:
            if m['org_0']['n_total'] == 0:
                extinction_step = step + 1
                print(f'\n  [EXTINCION] Depredador extinto en step {extinction_step}!')
            elif m['org_1']['n_total'] == 0:
                extinction_step = step + 1
                print(f'\n  [EXTINCION] Presa extinta en step {extinction_step}!')

        if (step + 1) % 100 == 0:
            print(f'  Step {step+1}:')
            print(f'    Depredador: {m["org_0"]["n_total"]} ({m["org_0"]["n_fi"]} Fi)')
            print(f'    Presa: {m["org_1"]["n_total"]} ({m["org_1"]["n_fi"]} Fi)')
            print(f'    Distancia: {m["predator_prey_distance"]:.1f}')
            print(f'    Conversiones: Pred={total_conversions_pred}, Presa={total_conversions_prey}')

    final = org.get_metrics()

    print(f'\nResultado final:')
    print(f'  Depredador: {final["org_0"]["n_total"]} celulas')
    print(f'  Presa: {final["org_1"]["n_total"]} celulas')
    print(f'  Conversiones totales: Pred={total_conversions_pred}, Presa={total_conversions_prey}')

    if extinction_step:
        print(f'  Extincion en step: {extinction_step}')

    return {
        'org': org,
        'initial_positions': initial_positions,
        'extinction_step': extinction_step,
        'total_conversions_pred': total_conversions_pred,
        'total_conversions_prey': total_conversions_prey
    }


def run_full_experiment():
    """Ejecuta todos los escenarios de depredacion."""
    print('='*70)
    print('EXPERIMENTO: DEPREDACION ASIMETRICA')
    print('='*70)

    n_steps = 400

    scenarios = [
        ('Control (simetrico)', 0.10, 0.10),
        ('Depredador dominante', 0.40, 0.05),
        ('Depredador extremo', 0.60, 0.02),
    ]

    results = {}
    for name, pred_conv, prey_conv in scenarios:
        results[name] = run_scenario(name, pred_conv, prey_conv, n_steps)

    # === ANALISIS COMPARATIVO ===
    print('\n' + '='*70)
    print('ANALISIS COMPARATIVO')
    print('='*70)

    print(f'\n{"Escenario":<25} {"Pred Final":<12} {"Presa Final":<12} {"Extincion":<12} {"Conv Pred":<10}')
    print('-'*75)

    for name, data in results.items():
        org = data['org']
        final = org.get_metrics()
        ext = data['extinction_step'] or 'No'
        print(f'{name:<25} {final["org_0"]["n_total"]:<12} {final["org_1"]["n_total"]:<12} '
              f'{str(ext):<12} {data["total_conversions_pred"]:<10}')

    # === VISUALIZACION ===
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    colors = {'predator': 'crimson', 'prey': 'royalblue'}
    scenario_names = list(results.keys())

    for idx, (name, data) in enumerate(results.items()):
        org = data['org']
        init_pos = data['initial_positions']

        # 1. Estado inicial
        ax = axes[idx, 0]
        for x, y, org_id, role_idx in init_pos:
            color = colors['predator'] if org_id == 0 else colors['prey']
            marker = 's' if role_idx == 1 else 'o'
            size = 80 if role_idx == 1 else 25
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_title(f'{name}\nInicial')
        ax.set_aspect('equal')
        if idx == 0:
            ax.text(5, 58, 'Depredador', color=colors['predator'], fontsize=9, fontweight='bold')
            ax.text(5, 52, 'Presa', color=colors['prey'], fontsize=9, fontweight='bold')

        # 2. Estado final
        ax = axes[idx, 1]
        for cell in org.cells:
            x, y = cell.position
            color = colors['predator'] if cell.organism_id == 0 else colors['prey']
            marker = 's' if cell.role_idx == 1 else 'o'
            size = 80 if cell.role_idx == 1 else 25
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_title('Final')
        ax.set_aspect('equal')

        # 3. Evolucion de poblacion
        ax = axes[idx, 2]
        steps = range(len(org.history))
        pred_pop = [h['org_0']['n_total'] for h in org.history]
        prey_pop = [h['org_1']['n_total'] for h in org.history]
        ax.plot(steps, pred_pop, color=colors['predator'], linewidth=2, label='Depredador')
        ax.plot(steps, prey_pop, color=colors['prey'], linewidth=2, label='Presa')
        ax.axhline(y=40, color='gray', linestyle=':', alpha=0.5)
        if data['extinction_step']:
            ax.axvline(x=data['extinction_step'], color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Poblacion')
        ax.set_title('Evolucion')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Distancia depredador-presa
        ax = axes[idx, 3]
        distances = [h['predator_prey_distance'] for h in org.history]
        ax.plot(steps, distances, color='green', linewidth=2)
        ax.fill_between(steps, distances, alpha=0.3, color='green')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Distancia')
        ax.set_title('Distancia Pred-Presa')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeta_organism_depredacion.png', dpi=150)
    print('\nGuardado: zeta_organism_depredacion.png')

    # === CONCLUSIONES ===
    print('\n' + '='*70)
    print('CONCLUSIONES')
    print('='*70)

    # Analizar resultados
    control = results['Control (simetrico)']
    dominante = results['Depredador dominante']
    extremo = results['Depredador extremo']

    control_final = control['org'].get_metrics()
    dom_final = dominante['org'].get_metrics()
    ext_final = extremo['org'].get_metrics()

    # Determinar patron
    if dom_final['org_1']['n_total'] == 0:
        print('\n[EXTINCION] Depredador dominante causa extincion de presa')
    elif dom_final['org_1']['n_total'] < control_final['org_1']['n_total'] * 0.5:
        print('\n[DOMINACION] Depredador reduce significativamente poblacion de presa')
    else:
        print('\n[COEXISTENCIA] Ambas poblaciones sobreviven')

    if ext_final['org_1']['n_total'] == 0:
        print('[EXTINCION RAPIDA] Depredador extremo elimina presa')

    # Evasion
    dom_dist = [h['predator_prey_distance'] for h in dominante['org'].history]
    if len(dom_dist) > 100:
        early_dist = np.mean(dom_dist[:50])
        late_dist = np.mean(dom_dist[-50:])
        if late_dist > early_dist * 1.2:
            print('[EVASION] La presa desarrolla comportamiento de huida')
        else:
            print('[SIN EVASION] La presa no logra escapar efectivamente')

    return results


if __name__ == '__main__':
    results = run_full_experiment()
