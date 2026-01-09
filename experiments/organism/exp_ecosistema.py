# exp_ecosistema.py
"""Experimento: Ecosistema dinamico con recursos regenerativos.

Hipotesis: Con parches de recursos distribuidos que se regeneran:
1. Emergerá partición territorial (cada organismo "reclama" parches)
2. Habrá migración entre parches agotados y frescos
3. Posible coexistencia sostenible (vs extinción en depredación)

Diseño:
- 5 parches de recursos en patrón X
- Células ganan energía en parches (+0.05/step)
- Parches se agotan y regeneran después de N steps
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from exp_dos_organismos import DualOrganism, DualCellEntity


@dataclass
class ResourcePatch:
    """Parche de recursos que se regenera."""
    position: Tuple[int, int]
    radius: float = 8.0
    capacity: float = 100.0
    current: float = 100.0
    regen_timer: int = 0
    regen_delay: int = 50

    def distance_to(self, x: int, y: int) -> float:
        """Distancia desde punto al centro del parche."""
        return np.sqrt((x - self.position[0])**2 + (y - self.position[1])**2)

    def contains(self, x: int, y: int) -> bool:
        """Verifica si punto está dentro del parche."""
        return self.distance_to(x, y) <= self.radius

    def consume(self, amount: float = 2.0) -> float:
        """Consume recursos, retorna cantidad efectiva consumida."""
        if self.current <= 0:
            return 0.0
        consumed = min(self.current, amount)
        self.current -= consumed
        return consumed

    def update(self):
        """Actualiza estado de regeneración."""
        if self.current <= 0:
            if self.regen_timer == 0:
                self.regen_timer = self.regen_delay
            self.regen_timer -= 1
            if self.regen_timer <= 0:
                self.current = self.capacity
                self.regen_timer = 0

    @property
    def is_depleted(self) -> bool:
        return self.current <= 0

    @property
    def fill_ratio(self) -> float:
        return self.current / self.capacity


class EcosystemOrganism(DualOrganism):
    """Dos organismos compitiendo por parches de recursos."""

    def __init__(self, n_patches: int = 5, patch_radius: float = 8.0,
                 patch_capacity: float = 100.0, regen_delay: int = 50,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_radius = patch_radius
        self.patch_capacity = patch_capacity
        self.regen_delay = regen_delay

        # Crear parches en patrón X
        self.patches = self._create_patches(n_patches)

        # Estadísticas
        self.patch_occupancy_history = []

    def _create_patches(self, n_patches: int) -> List[ResourcePatch]:
        """Crea parches en patrón X distribuido."""
        patches = []
        center = self.grid_size // 2
        offset = self.grid_size // 4

        # Patrón X: 4 esquinas + centro
        positions = [
            (center - offset, center - offset),  # P1: arriba-izq
            (center + offset, center - offset),  # P2: arriba-der
            (center, center),                     # P3: centro
            (center - offset, center + offset),  # P4: abajo-izq
            (center + offset, center + offset),  # P5: abajo-der
        ]

        for i, pos in enumerate(positions[:n_patches]):
            patch = ResourcePatch(
                position=pos,
                radius=self.patch_radius,
                capacity=self.patch_capacity,
                current=self.patch_capacity,
                regen_delay=self.regen_delay
            )
            patches.append(patch)

        return patches

    def step(self):
        """Step con dinámica de recursos."""
        # Calcular campo de fuerzas
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        new_cells = []

        # Contar células por parche para estadísticas
        patch_cells = {i: {'org_0': 0, 'org_1': 0} for i in range(len(self.patches))}

        for cell in self.cells:
            x, y = cell.position

            # === CONSUMO DE RECURSOS ===
            resource_gain = 0.0
            for i, patch in enumerate(self.patches):
                if patch.contains(x, y) and not patch.is_depleted:
                    consumed = patch.consume(2.0)
                    if consumed > 0:
                        resource_gain = 0.05
                        patch_cells[i][f'org_{cell.organism_id}'] += 1

            # Obtener vecinos
            all_neighbors = self._get_neighbors(cell, radius=5, same_org_only=False)
            same_org_neighbors = self._get_neighbors(cell, radius=5, same_org_only=True)
            other_org_neighbors = [n for n in all_neighbors if n.organism_id != cell.organism_id]

            same_mass = sum(1 for n in same_org_neighbors if n.role_idx == 0)
            same_fi = sum(1 for n in same_org_neighbors if n.role_idx == 1)
            other_fi = sum(1 for n in other_org_neighbors if n.role_idx == 1)

            potential = field[0, 0, y, x].item()

            behavior = self.behavior_0 if cell.organism_id == 0 else self.behavior_1

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

            # === DINÁMICA DE ENERGÍA CON RECURSOS ===
            if cell.role_idx == 1:  # Fi
                energy_gain = 0.02 * same_mass + resource_gain
                energy_loss = 0.03 * other_fi + 0.01  # Decay base para Fi
                new_energy = cell.energy + energy_gain - energy_loss
            else:  # Mass
                new_energy = cell.energy * 0.99 + 0.05 * max(0, potential) + resource_gain
                if same_fi > 0:
                    new_energy += 0.01
                if other_fi > 0:
                    new_energy -= 0.02
                # Sin recursos, decay más rápido
                if resource_gain == 0:
                    new_energy -= 0.02

            new_energy += 0.02 * max(0, net_influence + 0.5)
            new_energy = np.clip(new_energy, 0, 1)

            # === TRANSICIÓN DE ROL ===
            current_role_idx = cell.role_idx
            influence_score = net_influence + 0.5

            if current_role_idx == 0:  # MASS
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
            new_role_idx = new_role.argmax().item()
            new_x, new_y = x, y

            if new_role_idx == 0:  # Mass
                # Prioridad 1: Buscar parche con recursos
                best_patch = None
                best_dist = float('inf')
                for patch in self.patches:
                    if not patch.is_depleted:
                        dist = patch.distance_to(x, y)
                        if dist < best_dist and dist > patch.radius * 0.5:
                            best_dist = dist
                            best_patch = patch

                if best_patch and best_dist > 2:
                    # Moverse hacia parche
                    dx = int(np.sign(best_patch.position[0] - x))
                    dy = int(np.sign(best_patch.position[1] - y))
                    new_x = np.clip(x + dx, 0, self.grid_size - 1)
                    new_y = np.clip(y + dy, 0, self.grid_size - 1)
                else:
                    # Seguir a Fi propio
                    same_fi_cells = [c for c in self.cells
                                    if c.organism_id == cell.organism_id and c.role_idx == 1]
                    if same_fi_cells:
                        nearest_fi = min(same_fi_cells, key=lambda f:
                            (f.position[0] - x)**2 + (f.position[1] - y)**2)
                        dx = int(np.sign(nearest_fi.position[0] - x))
                        dy = int(np.sign(nearest_fi.position[1] - y))
                        new_x = np.clip(x + dx, 0, self.grid_size - 1)
                        new_y = np.clip(y + dy, 0, self.grid_size - 1)

            elif new_role_idx == 1:  # Fi - anclar en parche
                # Fi se queda en parche si tiene recursos
                in_good_patch = any(p.contains(x, y) and not p.is_depleted for p in self.patches)
                if not in_good_patch:
                    # Buscar parche cercano
                    for patch in self.patches:
                        if not patch.is_depleted:
                            dx = int(np.sign(patch.position[0] - x))
                            dy = int(np.sign(patch.position[1] - y))
                            new_x = np.clip(x + dx, 0, self.grid_size - 1)
                            new_y = np.clip(y + dy, 0, self.grid_size - 1)
                            break

            new_cell = DualCellEntity(
                position=(new_x, new_y),
                state=new_state.detach(),
                role=new_role,
                energy=new_energy,
                organism_id=cell.organism_id,
                controlled_mass=same_mass
            )
            new_cells.append(new_cell)

        # Actualizar regeneración de parches
        for patch in self.patches:
            patch.update()

        self.cells = new_cells
        self._update_grids()

        # Guardar estadísticas de parches
        self.patch_occupancy_history.append(patch_cells)
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Métricas extendidas con info de recursos."""
        metrics = super().get_metrics()

        # Estado de parches
        metrics['patches'] = []
        for i, patch in enumerate(self.patches):
            cells_in_patch = {'org_0': 0, 'org_1': 0}
            for cell in self.cells:
                if patch.contains(*cell.position):
                    cells_in_patch[f'org_{cell.organism_id}'] += 1

            metrics['patches'].append({
                'id': i,
                'fill_ratio': patch.fill_ratio,
                'is_depleted': patch.is_depleted,
                'regen_timer': patch.regen_timer,
                'cells_org_0': cells_in_patch['org_0'],
                'cells_org_1': cells_in_patch['org_1']
            })

        # Dominancia territorial
        org_0_patches = sum(1 for p in metrics['patches']
                          if p['cells_org_0'] > p['cells_org_1'])
        org_1_patches = sum(1 for p in metrics['patches']
                          if p['cells_org_1'] > p['cells_org_0'])
        metrics['territorial_dominance'] = {
            'org_0': org_0_patches,
            'org_1': org_1_patches,
            'contested': len(self.patches) - org_0_patches - org_1_patches
        }

        return metrics


def run_scenario(name: str, n_orgs: int = 2, n_patches: int = 5,
                 regen_delay: int = 50, n_steps: int = 500,
                 init_mode: str = 'center') -> Dict:
    """Ejecuta un escenario del ecosistema."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Organismos: {n_orgs}, Parches: {n_patches}, Regen: {regen_delay} steps')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = EcosystemOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        n_patches=n_patches,
        patch_radius=8.0,
        patch_capacity=100.0,
        regen_delay=regen_delay
    )

    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior_0.load_state_dict(weights['behavior_state'])
        org.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # Inicialización según modo
    if init_mode == 'center':
        # Ambos organismos mezclados en el centro
        _initialize_center_mixed(org)
    elif init_mode == 'advantage':
        # Org 0 cerca de 3 parches, Org 1 cerca de 2
        _initialize_with_advantage(org)
    else:
        org.initialize(separation='horizontal')

    # Eliminar org 1 si es control
    if n_orgs == 1:
        org.cells = [c for c in org.cells if c.organism_id == 0]
        org._update_grids()

    initial = org.get_metrics()
    print(f'Estado inicial:')
    print(f'  Org 0: {initial["org_0"]["n_total"]} células')
    if n_orgs > 1:
        print(f'  Org 1: {initial["org_1"]["n_total"]} células')
    print(f'  Parches activos: {sum(1 for p in org.patches if not p.is_depleted)}')

    # Tracking
    coexistence_broken = None
    territorial_history = []

    for step in range(n_steps):
        org.step()

        m = org.get_metrics()
        territorial_history.append(m['territorial_dominance'])

        # Detectar extinción
        if n_orgs > 1 and coexistence_broken is None:
            if m['org_0']['n_total'] == 0 or m['org_1']['n_total'] == 0:
                coexistence_broken = step + 1
                extinct = 0 if m['org_0']['n_total'] == 0 else 1
                print(f'\n  [EXTINCIÓN] Org {extinct} extinto en step {step+1}!')

        if (step + 1) % 100 == 0:
            patches_status = [f"P{i}:{int(p.fill_ratio*100)}%"
                             for i, p in enumerate(org.patches)]
            terr = m['territorial_dominance']
            print(f'  Step {step+1}:')
            print(f'    Org 0: {m["org_0"]["n_total"]} ({m["org_0"]["n_fi"]} Fi)')
            if n_orgs > 1:
                print(f'    Org 1: {m["org_1"]["n_total"]} ({m["org_1"]["n_fi"]} Fi)')
            print(f'    Parches: {" ".join(patches_status)}')
            print(f'    Territorio: O0={terr["org_0"]}, O1={terr["org_1"]}, Cont={terr["contested"]}')

    final = org.get_metrics()

    print(f'\nResultado final:')
    print(f'  Org 0: {final["org_0"]["n_total"]} células')
    if n_orgs > 1:
        print(f'  Org 1: {final["org_1"]["n_total"]} células')
    print(f'  Coexistencia: {"Sí" if coexistence_broken is None else f"Rota en step {coexistence_broken}"}')

    return {
        'org': org,
        'coexistence_broken': coexistence_broken,
        'territorial_history': territorial_history,
        'name': name
    }


def _initialize_center_mixed(org):
    """Inicializa ambos organismos mezclados en el centro."""
    org.cells = []
    center = org.grid_size // 2

    for org_id in range(2):
        for i in range(org.n_cells_per_org):
            # Ambos en el centro, mezclados
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


def _initialize_with_advantage(org):
    """Inicializa con ventaja para Org 0."""
    org.cells = []

    # Org 0 cerca de parches 0, 1, 2 (arriba + centro)
    for i in range(org.n_cells_per_org):
        target_patch = org.patches[i % 3]
        x = target_patch.position[0] + np.random.randint(-5, 6)
        y = target_patch.position[1] + np.random.randint(-5, 6)
        x = np.clip(x, 0, org.grid_size - 1)
        y = np.clip(y, 0, org.grid_size - 1)

        state = torch.randn(org.state_dim) * 0.1
        role = torch.tensor([0.0, 1.0, 0.0]) if i == 0 else torch.tensor([1.0, 0.0, 0.0])
        energy = 0.9 if i == 0 else np.random.uniform(0.4, 0.7)

        cell = DualCellEntity(
            position=(x, y), state=state, role=role,
            energy=energy, organism_id=0
        )
        org.cells.append(cell)

    # Org 1 cerca de parches 3, 4 (abajo)
    for i in range(org.n_cells_per_org):
        target_patch = org.patches[3 + (i % 2)]
        x = target_patch.position[0] + np.random.randint(-5, 6)
        y = target_patch.position[1] + np.random.randint(-5, 6)
        x = np.clip(x, 0, org.grid_size - 1)
        y = np.clip(y, 0, org.grid_size - 1)

        state = torch.randn(org.state_dim) * 0.1
        role = torch.tensor([0.0, 1.0, 0.0]) if i == 0 else torch.tensor([1.0, 0.0, 0.0])
        energy = 0.9 if i == 0 else np.random.uniform(0.4, 0.7)

        cell = DualCellEntity(
            position=(x, y), state=state, role=role,
            energy=energy, organism_id=1
        )
        org.cells.append(cell)

    org._update_grids()


def run_full_experiment():
    """Ejecuta todos los escenarios del ecosistema."""
    print('='*70)
    print('EXPERIMENTO: ECOSISTEMA DINÁMICO CON RECURSOS REGENERATIVOS')
    print('='*70)

    scenarios = [
        ('Control (1 org)', 1, 5, 50, 'center'),
        ('Competencia simétrica', 2, 5, 50, 'center'),
        ('Ventaja inicial', 2, 5, 50, 'advantage'),
        ('Escasez extrema', 2, 2, 100, 'center'),
    ]

    results = {}
    for name, n_orgs, n_patches, regen, init in scenarios:
        results[name] = run_scenario(name, n_orgs, n_patches, regen, 500, init)

    # === ANÁLISIS COMPARATIVO ===
    print('\n' + '='*70)
    print('ANÁLISIS COMPARATIVO')
    print('='*70)

    print(f'\n{"Escenario":<25} {"Org0 Final":<12} {"Org1 Final":<12} {"Coexistencia":<15}')
    print('-'*70)

    for name, data in results.items():
        org = data['org']
        final = org.get_metrics()
        coex = "N/A" if 'Control' in name else (
            "Sí" if data['coexistence_broken'] is None
            else f"Rota step {data['coexistence_broken']}"
        )
        org1_final = final['org_1']['n_total'] if 'org_1' in final else 'N/A'
        print(f'{name:<25} {final["org_0"]["n_total"]:<12} {str(org1_final):<12} {coex:<15}')

    # === VISUALIZACIÓN ===
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    colors = {'org_0': 'crimson', 'org_1': 'royalblue', 'patch': 'forestgreen'}

    for idx, (name, data) in enumerate(results.items()):
        org = data['org']

        # 1. Estado final con parches
        ax = axes[idx, 0]

        # Dibujar parches
        for patch in org.patches:
            circle = plt.Circle(patch.position, patch.radius,
                               color=colors['patch'], alpha=0.2)
            ax.add_patch(circle)
            fill_color = 'green' if not patch.is_depleted else 'gray'
            inner = plt.Circle(patch.position, patch.radius * patch.fill_ratio,
                              color=fill_color, alpha=0.4)
            ax.add_patch(inner)

        # Dibujar células
        for cell in org.cells:
            x, y = cell.position
            color = colors['org_0'] if cell.organism_id == 0 else colors['org_1']
            marker = 's' if cell.role_idx == 1 else 'o'
            size = 60 if cell.role_idx == 1 else 20
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.7)

        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_title(f'{name}\nEstado Final')
        ax.set_aspect('equal')

        # 2. Evolución de población
        ax = axes[idx, 1]
        steps = range(len(org.history))
        pop_0 = [h['org_0']['n_total'] for h in org.history]
        ax.plot(steps, pop_0, color=colors['org_0'], linewidth=2, label='Org 0')
        if 'org_1' in org.history[0]:
            pop_1 = [h['org_1']['n_total'] for h in org.history]
            ax.plot(steps, pop_1, color=colors['org_1'], linewidth=2, label='Org 1')
        ax.axhline(y=40, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Población')
        ax.set_title('Evolución Población')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Estado de parches en el tiempo
        ax = axes[idx, 2]
        for i, patch in enumerate(org.patches):
            fill_history = []
            for h in org.history:
                if 'patches' in h and i < len(h['patches']):
                    fill_history.append(h['patches'][i]['fill_ratio'])
                else:
                    fill_history.append(1.0)
            ax.plot(range(len(fill_history)), fill_history,
                   label=f'P{i}', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Recursos (%)')
        ax.set_title('Estado de Parches')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

        # 4. Dominancia territorial
        ax = axes[idx, 3]
        if data['territorial_history']:
            terr_0 = [t['org_0'] for t in data['territorial_history']]
            terr_1 = [t['org_1'] for t in data['territorial_history']]
            contested = [t['contested'] for t in data['territorial_history']]

            ax.stackplot(range(len(terr_0)), terr_0, contested, terr_1,
                        labels=['Org 0', 'Disputado', 'Org 1'],
                        colors=[colors['org_0'], 'gray', colors['org_1']],
                        alpha=0.7)
            ax.set_xlabel('Steps')
            ax.set_ylabel('# Parches')
            ax.set_title('Dominancia Territorial')
            ax.legend(fontsize=7, loc='upper right')
            ax.set_ylim(0, len(org.patches))

    plt.tight_layout()
    plt.savefig('zeta_organism_ecosistema.png', dpi=150)
    print('\nGuardado: zeta_organism_ecosistema.png')

    # === CONCLUSIONES ===
    print('\n' + '='*70)
    print('CONCLUSIONES')
    print('='*70)

    # Analizar coexistencia
    symmetric = results['Competencia simétrica']
    advantage = results['Ventaja inicial']
    scarcity = results['Escasez extrema']

    coex_count = sum(1 for r in [symmetric, advantage, scarcity]
                     if r['coexistence_broken'] is None)

    print(f'\nCoexistencia lograda en {coex_count}/3 escenarios competitivos')

    if symmetric['coexistence_broken'] is None:
        print('[OK] Competencia simétrica permite coexistencia')
    else:
        print(f'[X] Competencia simétrica: extinción en step {symmetric["coexistence_broken"]}')

    if advantage['coexistence_broken'] is None:
        print('[OK] Ventaja inicial no previene coexistencia')
    else:
        print(f'[X] Ventaja inicial: extinción en step {advantage["coexistence_broken"]}')

    if scarcity['coexistence_broken'] is None:
        print('[OK] Escasez extrema aún permite coexistencia')
    else:
        print(f'[X] Escasez extrema causa extinción en step {scarcity["coexistence_broken"]}')

    # Partición territorial
    sym_final = symmetric['org'].get_metrics()
    if 'territorial_dominance' in sym_final:
        terr = sym_final['territorial_dominance']
        if terr['org_0'] > 0 and terr['org_1'] > 0:
            print(f'\n[PARTICIÓN] Territorio dividido: Org0={terr["org_0"]}, Org1={terr["org_1"]}')
        elif terr['contested'] == len(symmetric['org'].patches):
            print('\n[COMPETENCIA] Todos los parches disputados')

    return results


if __name__ == '__main__':
    results = run_full_experiment()
