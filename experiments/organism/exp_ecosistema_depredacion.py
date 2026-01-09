# exp_ecosistema_depredacion.py
"""Experimento: Ecosistema dinamico + Depredacion + Refugios.

Hipotesis: Recursos regenerativos que favorecen a la presa pueden
estabilizar la dinamica depredador-presa, creando ciclos Lotka-Volterra
en lugar de extincion.

Combinacion:
- De Ecosistema: parches de recursos regenerativos
- De Depredacion: caza directa asimetrica
- NUEVO: Refugios donde la caza no funciona

Asimetria clave: Presa gana +0.08 de parches, Depredador solo +0.03
Refugios: Parches son zonas seguras donde la caza tiene efectividad reducida
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple

from exp_dos_organismos import DualOrganism, DualCellEntity
from exp_ecosistema import ResourcePatch


class PredatorPreyEcosystem(DualOrganism):
    """Ecosistema con depredacion, recursos asimetricos y refugios."""

    def __init__(self,
                 # Parametros de recursos
                 n_patches: int = 5,
                 patch_radius: float = 8.0,
                 patch_capacity: float = 100.0,
                 regen_delay: int = 50,
                 # Parametros de depredacion
                 predator_id: int = 0,
                 predator_conversion: float = 0.20,
                 prey_conversion: float = 0.03,
                 hunt_radius: float = 4.0,
                 # Asimetria de recursos
                 prey_resource_bonus: float = 0.08,
                 predator_resource_bonus: float = 0.03,
                 # NUEVO: Refugios
                 refuge_effectiveness: float = 0.0,  # 0=sin refugio, 1=inmunidad total
                 predator_hunger_rate: float = 0.0,  # Decay de energia si no caza
                 **kwargs):
        super().__init__(**kwargs)

        # Recursos
        self.patch_radius = patch_radius
        self.patch_capacity = patch_capacity
        self.regen_delay = regen_delay
        self.patches = self._create_patches(n_patches)

        # Depredación
        self.predator_id = predator_id
        self.prey_id = 1 - predator_id
        self.predator_conversion = predator_conversion
        self.prey_conversion = prey_conversion
        self.hunt_radius = hunt_radius

        # Asimetria
        self.prey_resource_bonus = prey_resource_bonus
        self.predator_resource_bonus = predator_resource_bonus

        # Refugios
        self.refuge_effectiveness = refuge_effectiveness
        self.predator_hunger_rate = predator_hunger_rate

        # Estadisticas
        self.conversions_by_predator = 0
        self.conversions_by_prey = 0
        self.prey_in_refuge = 0

    def _create_patches(self, n_patches: int) -> List[ResourcePatch]:
        """Crea parches en patrón X."""
        patches = []
        center = self.grid_size // 2
        offset = self.grid_size // 4

        positions = [
            (center - offset, center - offset),
            (center + offset, center - offset),
            (center, center),
            (center - offset, center + offset),
            (center + offset, center + offset),
        ]

        for pos in positions[:n_patches]:
            patch = ResourcePatch(
                position=pos,
                radius=self.patch_radius,
                capacity=self.patch_capacity,
                current=self.patch_capacity,
                regen_delay=self.regen_delay
            )
            patches.append(patch)

        return patches

    def _is_in_refuge(self, x: int, y: int) -> bool:
        """Verifica si una posicion esta dentro de un refugio (parche)."""
        for patch in self.patches:
            if patch.contains(x, y):
                return True
        return False

    def step(self):
        """Step combinando recursos, depredacion y refugios."""
        self.conversions_by_predator = 0
        self.conversions_by_prey = 0
        self.prey_in_refuge = 0

        # Contar presas en refugio
        for cell in self.cells:
            if cell.organism_id == self.prey_id:
                if self._is_in_refuge(*cell.position):
                    self.prey_in_refuge += 1

        # === FASE 1: CAZA (con refugios) ===
        predator_hunters = [c for c in self.cells
                          if c.organism_id == self.predator_id and c.role_idx == 1]
        prey_hunters = [c for c in self.cells
                       if c.organism_id == self.prey_id and c.role_idx == 1]

        cells_to_convert = {}
        successful_hunts = 0  # Para calcular hambre del depredador

        for cell_idx, cell in enumerate(self.cells):
            if cell.organism_id == self.prey_id:
                # REFUGIO: Si la presa esta en refugio, caza es menos efectiva
                in_refuge = self._is_in_refuge(*cell.position)
                refuge_modifier = (1.0 - self.refuge_effectiveness) if in_refuge else 1.0

                for hunter in predator_hunters:
                    dist = np.sqrt((hunter.position[0] - cell.position[0])**2 +
                                  (hunter.position[1] - cell.position[1])**2)
                    if dist <= self.hunt_radius:
                        base_prob = self.predator_conversion * (1 - dist/self.hunt_radius)
                        base_prob *= refuge_modifier  # Reducir por refugio
                        if cell.role_idx == 1:
                            base_prob *= 0.3
                        if np.random.random() < base_prob:
                            cells_to_convert[cell_idx] = self.predator_id
                            self.conversions_by_predator += 1
                            successful_hunts += 1
                            break

            elif cell.organism_id == self.predator_id:
                for hunter in prey_hunters:
                    dist = np.sqrt((hunter.position[0] - cell.position[0])**2 +
                                  (hunter.position[1] - cell.position[1])**2)
                    if dist <= self.hunt_radius:
                        base_prob = self.prey_conversion * (1 - dist/self.hunt_radius)
                        if cell.role_idx == 1:
                            base_prob *= 0.3
                        if np.random.random() < base_prob:
                            cells_to_convert[cell_idx] = self.prey_id
                            self.conversions_by_prey += 1
                            break

        # Calcular hambre (para penalizar depredadores que no cazan)
        self.last_successful_hunts = successful_hunts

        # === FASE 2: RECURSOS Y MOVIMIENTO ===
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        new_cells = []

        for cell_idx, cell in enumerate(self.cells):
            current_org_id = cells_to_convert.get(cell_idx, cell.organism_id)
            x, y = cell.position

            # === CONSUMO DE RECURSOS (ASIMÉTRICO) ===
            resource_gain = 0.0
            for patch in self.patches:
                if patch.contains(x, y) and not patch.is_depleted:
                    consumed = patch.consume(2.0)
                    if consumed > 0:
                        # Presa gana más de recursos
                        if current_org_id == self.prey_id:
                            resource_gain = self.prey_resource_bonus
                        else:
                            resource_gain = self.predator_resource_bonus

            # Vecinos
            all_neighbors = self._get_neighbors(cell, radius=5, same_org_only=False)
            same_org_neighbors = [n for n in all_neighbors if n.organism_id == current_org_id]
            other_org_neighbors = [n for n in all_neighbors if n.organism_id != current_org_id]

            same_mass = sum(1 for n in same_org_neighbors if n.role_idx == 0)
            same_fi = sum(1 for n in same_org_neighbors if n.role_idx == 1)
            other_fi = sum(1 for n in other_org_neighbors if n.role_idx == 1)

            potential = field[0, 0, y, x].item()
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
            energy_penalty = 0.3 if cell_idx in cells_to_convert else 0.0

            # HAMBRE: Depredadores pierden energia si no cazan
            hunger_penalty = 0.0
            if current_org_id == self.predator_id and self.predator_hunger_rate > 0:
                if successful_hunts == 0:
                    hunger_penalty = self.predator_hunger_rate

            if cell.role_idx == 1:  # Fi
                energy_gain = 0.02 * same_mass + resource_gain
                energy_loss = 0.03 * other_fi + 0.01 + hunger_penalty
                new_energy = cell.energy + energy_gain - energy_loss - energy_penalty
            else:  # Mass
                new_energy = cell.energy * 0.99 + 0.05 * max(0, potential) + resource_gain
                if same_fi > 0:
                    new_energy += 0.01
                if other_fi > 0:
                    new_energy -= 0.02
                if resource_gain == 0:
                    new_energy -= 0.02
                new_energy -= energy_penalty
                new_energy -= hunger_penalty * 0.5  # Mass tambien sufre hambre pero menos

            new_energy += 0.02 * max(0, net_influence + 0.5)
            new_energy = np.clip(new_energy, 0, 1)

            # === TRANSICIÓN DE ROL ===
            current_role_idx = cell.role_idx
            influence_score = net_influence + 0.5

            if cell_idx in cells_to_convert and cell.role_idx == 1:
                new_role = torch.tensor([1.0, 0.0, 0.0])
            elif current_role_idx == 0:
                can_become_fi = (
                    new_energy > self.fi_threshold and
                    same_mass >= 2 and
                    (same_fi == 0 or influence_score > 0.3) and
                    other_fi == 0
                )
                new_role = torch.tensor([0.0, 1.0, 0.0]) if can_become_fi else torch.tensor([1.0, 0.0, 0.0])
            elif current_role_idx == 1:
                loses_fi = (
                    same_mass < 1 or
                    new_energy < 0.2 or
                    (other_fi > 0 and other_fi >= same_fi)
                )
                new_role = torch.tensor([1.0, 0.0, 0.0]) if loses_fi else torch.tensor([0.0, 1.0, 0.0])
            else:
                new_role = cell.role.clone()

            # === MOVIMIENTO ===
            new_x, new_y = x, y
            new_role_idx = new_role.argmax().item()

            # Depredador Fi persigue presa
            if current_org_id == self.predator_id and new_role_idx == 1:
                prey_cells = [c for c in self.cells if c.organism_id == self.prey_id]
                if prey_cells:
                    nearest = min(prey_cells, key=lambda p:
                        (p.position[0] - x)**2 + (p.position[1] - y)**2)
                    dx = int(np.sign(nearest.position[0] - x))
                    dy = int(np.sign(nearest.position[1] - y))
                    new_x = np.clip(x + dx, 0, self.grid_size - 1)
                    new_y = np.clip(y + dy, 0, self.grid_size - 1)

            elif new_role_idx == 0:  # Mass
                # Presa prioriza parches como refugio
                if current_org_id == self.prey_id:
                    best_patch = None
                    best_dist = float('inf')
                    for patch in self.patches:
                        if not patch.is_depleted:
                            dist = patch.distance_to(x, y)
                            if dist < best_dist:
                                best_dist = dist
                                best_patch = patch

                    if best_patch and best_dist > 2:
                        dx = int(np.sign(best_patch.position[0] - x))
                        dy = int(np.sign(best_patch.position[1] - y))
                        new_x = np.clip(x + dx, 0, self.grid_size - 1)
                        new_y = np.clip(y + dy, 0, self.grid_size - 1)

                    # Huir de depredadores cercanos
                    nearby_pred = [c for c in self.cells
                                  if c.organism_id == self.predator_id and c.role_idx == 1
                                  and np.sqrt((c.position[0]-x)**2 + (c.position[1]-y)**2) < 10]
                    if nearby_pred:
                        nearest = min(nearby_pred, key=lambda p:
                            (p.position[0] - x)**2 + (p.position[1] - y)**2)
                        dx = -2 * int(np.sign(nearest.position[0] - x))
                        dy = -2 * int(np.sign(nearest.position[1] - y))
                        new_x = np.clip(x + dx, 0, self.grid_size - 1)
                        new_y = np.clip(y + dy, 0, self.grid_size - 1)
                else:
                    # Depredador mass sigue a su Fi
                    same_fi_cells = [c for c in self.cells
                                    if c.organism_id == current_org_id and c.role_idx == 1]
                    if same_fi_cells:
                        nearest = min(same_fi_cells, key=lambda f:
                            (f.position[0] - x)**2 + (f.position[1] - y)**2)
                        dx = int(np.sign(nearest.position[0] - x))
                        dy = int(np.sign(nearest.position[1] - y))
                        new_x = np.clip(x + dx, 0, self.grid_size - 1)
                        new_y = np.clip(y + dy, 0, self.grid_size - 1)

            new_cell = DualCellEntity(
                position=(new_x, new_y),
                state=new_state.detach(),
                role=new_role,
                energy=new_energy,
                organism_id=current_org_id,
                controlled_mass=same_mass
            )
            new_cells.append(new_cell)

        # Regenerar parches
        for patch in self.patches:
            patch.update()

        self.cells = new_cells
        self._update_grids()
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Metricas combinadas."""
        metrics = super().get_metrics()
        metrics['conversions_pred'] = self.conversions_by_predator
        metrics['conversions_prey'] = self.conversions_by_prey
        metrics['prey_in_refuge'] = self.prey_in_refuge

        # Estado de parches
        metrics['patches_active'] = sum(1 for p in self.patches if not p.is_depleted)
        metrics['total_resources'] = sum(p.current for p in self.patches)

        return metrics


def run_scenario(name: str, predator_conv: float, prey_conv: float,
                 use_resources: bool, refuge_eff: float = 0.0,
                 hunger_rate: float = 0.0, n_steps: int = 600) -> Dict:
    """Ejecuta un escenario."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Pred->Presa: {predator_conv*100:.0f}%, Recursos: {"Si" if use_resources else "No"}, '
          f'Refugio: {refuge_eff*100:.0f}%, Hambre: {hunger_rate}')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = PredatorPreyEcosystem(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        n_patches=5 if use_resources else 0,
        predator_conversion=predator_conv,
        prey_conversion=prey_conv,
        prey_resource_bonus=0.08 if use_resources else 0.0,
        predator_resource_bonus=0.03 if use_resources else 0.0,
        refuge_effectiveness=refuge_eff,
        predator_hunger_rate=hunger_rate
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior_0.load_state_dict(weights['behavior_state'])
        org.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # Inicialización mezclada
    _initialize_mixed(org)

    initial = org.get_metrics()
    print(f'Estado inicial: Pred={initial["org_0"]["n_total"]}, Presa={initial["org_1"]["n_total"]}')

    extinction_step = None
    total_conv_pred = 0
    total_conv_prey = 0
    pop_history = {'pred': [], 'prey': [], 'steps': []}

    for step in range(n_steps):
        org.step()
        m = org.get_metrics()

        total_conv_pred += m['conversions_pred']
        total_conv_prey += m['conversions_prey']

        # Guardar población cada 10 steps para detectar ciclos
        if step % 10 == 0:
            pop_history['pred'].append(m['org_0']['n_total'])
            pop_history['prey'].append(m['org_1']['n_total'])
            pop_history['steps'].append(step)

        if extinction_step is None:
            if m['org_0']['n_total'] == 0:
                extinction_step = step + 1
                print(f'\n  [EXTINCIÓN] Depredador extinto en step {step+1}!')
            elif m['org_1']['n_total'] == 0:
                extinction_step = step + 1
                print(f'\n  [EXTINCIÓN] Presa extinta en step {step+1}!')

        if (step + 1) % 150 == 0:
            print(f'  Step {step+1}: Pred={m["org_0"]["n_total"]} ({m["org_0"]["n_fi"]} Fi), '
                  f'Presa={m["org_1"]["n_total"]} ({m["org_1"]["n_fi"]} Fi), '
                  f'Conv={total_conv_pred}/{total_conv_prey}')

    final = org.get_metrics()
    print(f'\nFinal: Pred={final["org_0"]["n_total"]}, Presa={final["org_1"]["n_total"]}')
    print(f'Coexistencia: {"SÍ" if extinction_step is None else f"NO (step {extinction_step})"}')

    # Detectar ciclos (varianza en poblacion) - solo si no hubo extincion
    has_cycles = False
    cycle_amplitude = 0.0

    if extinction_step is None and len(pop_history['prey']) > 20:
        # Analizar ultimos 400 steps (40 muestras)
        recent_prey = pop_history['prey'][-40:]
        recent_pred = pop_history['pred'][-40:]

        prey_min, prey_max = min(recent_prey), max(recent_prey)
        pred_min, pred_max = min(recent_pred), max(recent_pred)

        prey_amplitude = prey_max - prey_min
        pred_amplitude = pred_max - pred_min

        # Ciclos si hay variacion significativa en ambos
        has_cycles = prey_amplitude > 5 and pred_amplitude > 5
        cycle_amplitude = (prey_amplitude + pred_amplitude) / 2

        if has_cycles:
            print(f'CICLOS DETECTADOS! Amplitud presa: {prey_amplitude:.0f}, pred: {pred_amplitude:.0f}')
        else:
            print(f'Sin ciclos (amplitud presa: {prey_amplitude:.0f}, pred: {pred_amplitude:.0f})')
    elif extinction_step:
        print('Sin ciclos (extincion)')

    return {
        'org': org,
        'extinction_step': extinction_step,
        'pop_history': pop_history,
        'total_conv_pred': total_conv_pred,
        'total_conv_prey': total_conv_prey,
        'has_cycles': has_cycles,
        'cycle_amplitude': cycle_amplitude,
        'name': name
    }


def _initialize_mixed(org):
    """Inicializa organismos mezclados."""
    org.cells = []
    center = org.grid_size // 2

    for org_id in range(2):
        for i in range(org.n_cells_per_org):
            x = center + np.random.randint(-15, 16)
            y = center + np.random.randint(-15, 16)
            x = np.clip(x, 0, org.grid_size - 1)
            y = np.clip(y, 0, org.grid_size - 1)

            state = torch.randn(org.state_dim) * 0.1
            role = torch.tensor([0.0, 1.0, 0.0]) if i == 0 else torch.tensor([1.0, 0.0, 0.0])
            energy = 0.9 if i == 0 else np.random.uniform(0.4, 0.7)

            cell = DualCellEntity(
                position=(x, y), state=state, role=role,
                energy=energy, organism_id=org_id
            )
            org.cells.append(cell)

    org._update_grids()


def run_full_experiment():
    """Ejecuta todos los escenarios con refugios."""
    print('='*70)
    print('EXPERIMENTO: ECOSISTEMA + DEPREDACION + REFUGIOS')
    print('Hipotesis: Refugios + hambre crean ciclos Lotka-Volterra')
    print('='*70)

    # Escenarios: (nombre, pred_conv, prey_conv, recursos, refugio, hambre)
    scenarios = [
        ('Sin refugio (baseline)', 0.30, 0.03, True, 0.0, 0.0),
        ('Refugio 50%', 0.30, 0.03, True, 0.5, 0.0),
        ('Refugio 80%', 0.30, 0.03, True, 0.8, 0.0),
        ('Refugio 80% + Hambre', 0.30, 0.03, True, 0.8, 0.03),
        ('Refugio 95% + Hambre', 0.30, 0.03, True, 0.95, 0.05),
    ]

    results = {}
    for name, pred_conv, prey_conv, use_res, refuge, hunger in scenarios:
        results[name] = run_scenario(name, pred_conv, prey_conv, use_res, refuge, hunger, 800)

    # === ANALISIS ===
    print('\n' + '='*70)
    print('ANALISIS COMPARATIVO')
    print('='*70)

    print(f'\n{"Escenario":<25} {"Pred":<8} {"Presa":<8} {"Extincion":<12} {"Ciclos":<8} {"Amplitud":<10}')
    print('-'*80)

    for name, data in results.items():
        final = data['org'].get_metrics()
        ext = data['extinction_step'] or 'No'
        cycles = 'Si' if data['has_cycles'] else 'No'
        amp = f"{data.get('cycle_amplitude', 0):.1f}" if data['has_cycles'] else '-'
        print(f'{name:<25} {final["org_0"]["n_total"]:<8} {final["org_1"]["n_total"]:<8} '
              f'{str(ext):<12} {cycles:<8} {amp:<10}')

    # === VISUALIZACION ===
    n_scenarios = len(results)
    fig, axes = plt.subplots(2, n_scenarios, figsize=(4*n_scenarios, 10))

    colors = {'pred': 'crimson', 'prey': 'royalblue'}

    for idx, (name, data) in enumerate(results.items()):
        org = data['org']
        pop = data['pop_history']

        # Fila 1: Evolución de población
        ax = axes[0, idx]
        ax.plot(pop['steps'], pop['pred'], color=colors['pred'], linewidth=2, label='Depredador')
        ax.plot(pop['steps'], pop['prey'], color=colors['prey'], linewidth=2, label='Presa')
        ax.axhline(y=40, color='gray', linestyle=':', alpha=0.5)
        if data['extinction_step']:
            ax.axvline(x=data['extinction_step'], color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Población')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 85)

        # Fila 2: Estado final
        ax = axes[1, idx]

        # Dibujar parches si existen
        if hasattr(org, 'patches'):
            for patch in org.patches:
                circle = plt.Circle(patch.position, patch.radius,
                                   color='green', alpha=0.15)
                ax.add_patch(circle)

        # Dibujar células
        for cell in org.cells:
            cx, cy = cell.position
            color = colors['pred'] if cell.organism_id == 0 else colors['prey']
            marker = 's' if cell.role_idx == 1 else 'o'
            size = 50 if cell.role_idx == 1 else 15
            ax.scatter(cx, cy, c=color, s=size, marker=marker, alpha=0.7)

        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_title('Estado Final')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('zeta_organism_ecosistema_depredacion.png', dpi=150)
    print('\nGuardado: zeta_organism_ecosistema_depredacion.png')

    # === CONCLUSIONES ===
    print('\n' + '='*70)
    print('CONCLUSIONES')
    print('='*70)

    baseline = results['Sin refugio (baseline)']

    # Contar exitos
    coexistence_count = sum(1 for r in results.values() if r['extinction_step'] is None)
    cycles_count = sum(1 for r in results.values() if r['has_cycles'])

    print(f'\nCoexistencia: {coexistence_count}/{len(results)} escenarios')
    print(f'Ciclos Lotka-Volterra: {cycles_count}/{len(results)} escenarios')

    # Analizar efecto de refugios
    if baseline['extinction_step']:
        print(f'\n[BASELINE] Sin refugio: extincion en step {baseline["extinction_step"]}')
    else:
        print('\n[BASELINE] Sin refugio: coexistencia')

    for name, data in results.items():
        if name == 'Sin refugio (baseline)':
            continue
        if data['has_cycles']:
            print(f'[CICLOS] {name}: amplitud {data["cycle_amplitude"]:.1f}')
        elif data['extinction_step'] is None:
            print(f'[COEXISTENCIA] {name}: estable sin ciclos')
        else:
            print(f'[EXTINCION] {name}: step {data["extinction_step"]}')

    # Conclusion final
    if cycles_count > 0:
        best_cycle = max((r for r in results.values() if r['has_cycles']),
                        key=lambda x: x['cycle_amplitude'], default=None)
        if best_cycle:
            print(f'\n[EXITO] Ciclos Lotka-Volterra logrados!')
            print(f'  Mejor configuracion: {best_cycle["name"]}')
            print(f'  Amplitud de ciclo: {best_cycle["cycle_amplitude"]:.1f}')
    else:
        print('\n[--] No se lograron ciclos Lotka-Volterra')

    return results


if __name__ == '__main__':
    results = run_full_experiment()
