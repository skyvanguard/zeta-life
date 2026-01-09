# exp_lotka_volterra.py
"""Experimento: Dinamica Lotka-Volterra con reproduccion y muerte.

Mecanicas nuevas:
- Reproduccion por division celular (energia > umbral)
- Muerte por inanicion (energia <= 0)
- Caza mata en lugar de convertir
- Poblacion variable (no fija en 80)

Objetivo: Lograr ciclos oscilatorios depredador-presa clasicos.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy import signal

from exp_dos_organismos import DualOrganism, DualCellEntity
from exp_ecosistema import ResourcePatch


class LotkaVolterraOrganism(DualOrganism):
    """Ecosistema con reproduccion, muerte y dinamica Lotka-Volterra."""

    def __init__(self,
                 # Parametros de recursos
                 n_patches: int = 5,
                 patch_radius: float = 8.0,
                 patch_capacity: float = 100.0,
                 regen_delay: int = 30,
                 # Parametros de depredacion
                 predator_id: int = 0,
                 hunt_radius: float = 8.0,  # Radio grande para cazar efectivamente
                 hunt_success_rate: float = 0.25,
                 hunt_energy_gain: float = 0.30,
                 # Parametros de reproduccion
                 prey_repro_threshold: float = 0.70,
                 predator_repro_threshold: float = 0.85,
                 max_population: int = 80,
                 # Parametros de energia
                 prey_resource_bonus: float = 0.06,
                 predator_resource_bonus: float = 0.02,
                 base_energy_decay: float = 0.02,
                 predator_hunger_rate: float = 0.03,
                 # Refugios
                 refuge_effectiveness: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)

        # Recursos
        self.patch_radius = patch_radius
        self.patch_capacity = patch_capacity
        self.regen_delay = regen_delay
        self.patches = self._create_patches(n_patches)

        # Depredacion
        self.predator_id = predator_id
        self.prey_id = 1 - predator_id
        self.hunt_radius = hunt_radius
        self.hunt_success_rate = hunt_success_rate
        self.hunt_energy_gain = hunt_energy_gain

        # Reproduccion
        self.prey_repro_threshold = prey_repro_threshold
        self.predator_repro_threshold = predator_repro_threshold
        self.max_population = max_population

        # Energia
        self.prey_resource_bonus = prey_resource_bonus
        self.predator_resource_bonus = predator_resource_bonus
        self.base_energy_decay = base_energy_decay
        self.predator_hunger_rate = predator_hunger_rate

        # Refugios
        self.refuge_effectiveness = refuge_effectiveness

        # Estadisticas por step
        self.births_pred = 0
        self.births_prey = 0
        self.deaths_pred_starve = 0
        self.deaths_prey_starve = 0
        self.deaths_prey_hunted = 0

    def _create_patches(self, n_patches: int) -> List[ResourcePatch]:
        """Crea parches en patron X."""
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
        """Verifica si posicion esta en refugio."""
        for patch in self.patches:
            if patch.contains(x, y):
                return True
        return False

    def _find_empty_adjacent(self, x: int, y: int, radius: int = 3) -> Optional[Tuple[int, int]]:
        """Busca posicion vacia adyacente para reproduccion."""
        occupied = {cell.position for cell in self.cells}

        # Buscar en espiral desde el centro
        for r in range(1, radius + 1):
            candidates = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:  # Solo borde
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if (nx, ny) not in occupied:
                                candidates.append((nx, ny))
            if candidates:
                return candidates[np.random.randint(len(candidates))]
        return None

    def _count_population(self, org_id: int) -> int:
        """Cuenta poblacion de un organismo."""
        return sum(1 for c in self.cells if c.organism_id == org_id)

    def step(self):
        """Step con reproduccion, muerte y caza letal."""
        # Reset estadisticas
        self.births_pred = 0
        self.births_prey = 0
        self.deaths_pred_starve = 0
        self.deaths_prey_starve = 0
        self.deaths_prey_hunted = 0

        # Usar id() para identificar celulas unicamente
        killed_ids = set()
        energy_boosts = {}  # cell_id -> energy gained

        # === FASE 1: CAZA (mata en lugar de convertir) ===
        for pred in self.cells:
            if pred.organism_id != self.predator_id:
                continue
            hunt_bonus = 1.5 if pred.role_idx == 1 else 1.0

            for prey in self.cells:
                if prey.organism_id != self.prey_id:
                    continue
                if id(prey) in killed_ids:
                    continue

                dist = np.sqrt((pred.position[0] - prey.position[0])**2 +
                              (pred.position[1] - prey.position[1])**2)

                if dist <= self.hunt_radius:
                    base_prob = self.hunt_success_rate * (1 - dist / self.hunt_radius) * hunt_bonus

                    if self._is_in_refuge(*prey.position):
                        base_prob *= (1.0 - self.refuge_effectiveness)

                    if prey.role_idx == 1:
                        base_prob *= 0.5

                    if np.random.random() < base_prob:
                        killed_ids.add(id(prey))
                        self.deaths_prey_hunted += 1
                        # Depredador gana energia
                        if id(pred) not in energy_boosts:
                            energy_boosts[id(pred)] = 0
                        energy_boosts[id(pred)] += self.hunt_energy_gain
                        break

        # === FASE 2: ACTUALIZAR ENERGIA Y MOVIMIENTO ===
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        updated_cells = []

        for cell in self.cells:
            if id(cell) in killed_ids:
                continue  # Presa cazada, no procesar

            x, y = cell.position
            org_id = cell.organism_id

            # Consumo de recursos
            resource_gain = 0.0
            for patch in self.patches:
                if patch.contains(x, y) and not patch.is_depleted:
                    consumed = patch.consume(2.0)
                    if consumed > 0:
                        if org_id == self.prey_id:
                            resource_gain = self.prey_resource_bonus
                        else:
                            resource_gain = self.predator_resource_bonus

            # Calcular nueva energia
            new_energy = cell.energy

            # Ganancia por recursos
            new_energy += resource_gain

            # Ganancia por caza (depredadores)
            if id(cell) in energy_boosts:
                new_energy += energy_boosts[id(cell)]

            # Decay base
            new_energy -= self.base_energy_decay

            # Hambre extra para depredadores sin recursos y sin caza
            if org_id == self.predator_id and resource_gain == 0 and id(cell) not in energy_boosts:
                new_energy -= self.predator_hunger_rate

            new_energy = np.clip(new_energy, 0, 1)

            # === MOVIMIENTO ===
            new_x, new_y = x, y

            if org_id == self.predator_id and cell.role_idx == 1:
                # Depredador Fi persigue presa
                prey_list = [c for c in self.cells if c.organism_id == self.prey_id]
                if prey_list:
                    nearest = min(prey_list, key=lambda p:
                        (p.position[0] - x)**2 + (p.position[1] - y)**2)
                    dx = int(np.sign(nearest.position[0] - x))
                    dy = int(np.sign(nearest.position[1] - y))
                    new_x = np.clip(x + dx, 0, self.grid_size - 1)
                    new_y = np.clip(y + dy, 0, self.grid_size - 1)

            elif org_id == self.prey_id:
                # Presa busca recursos y huye de depredadores
                # Primero: huir de depredadores cercanos
                nearby_pred = [c for c in self.cells
                              if c.organism_id == self.predator_id and c.role_idx == 1
                              and np.sqrt((c.position[0]-x)**2 + (c.position[1]-y)**2) < 12]
                if nearby_pred:
                    nearest = min(nearby_pred, key=lambda p:
                        (p.position[0] - x)**2 + (p.position[1] - y)**2)
                    dx = -int(np.sign(nearest.position[0] - x))
                    dy = -int(np.sign(nearest.position[1] - y))
                    new_x = np.clip(x + 2*dx, 0, self.grid_size - 1)
                    new_y = np.clip(y + 2*dy, 0, self.grid_size - 1)
                else:
                    # Buscar recursos
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

            # === TRANSICION DE ROL ===
            neighbors = [c for c in self.cells if c.organism_id == org_id
                        and c.position != cell.position
                        and np.sqrt((c.position[0]-x)**2 + (c.position[1]-y)**2) < 5]
            same_mass = sum(1 for n in neighbors if n.role_idx == 0)
            same_fi = sum(1 for n in neighbors if n.role_idx == 1)

            if cell.role_idx == 0:  # Mass
                can_become_fi = (
                    new_energy > self.fi_threshold and
                    same_mass >= 2 and
                    same_fi == 0
                )
                new_role = torch.tensor([0.0, 1.0, 0.0]) if can_become_fi else torch.tensor([1.0, 0.0, 0.0])
            else:  # Fi
                loses_fi = same_mass < 1 or new_energy < 0.2
                new_role = torch.tensor([1.0, 0.0, 0.0]) if loses_fi else torch.tensor([0.0, 1.0, 0.0])

            updated_cell = DualCellEntity(
                position=(new_x, new_y),
                state=cell.state.clone(),
                role=new_role,
                energy=new_energy,
                organism_id=org_id,
                controlled_mass=same_mass
            )
            updated_cells.append(updated_cell)

        # === FASE 3: MUERTE POR INANICION ===
        surviving_cells = []
        for cell in updated_cells:
            if cell.energy <= 0:
                if cell.organism_id == self.predator_id:
                    self.deaths_pred_starve += 1
                else:
                    self.deaths_prey_starve += 1
            else:
                surviving_cells.append(cell)

        # === FASE 4: REPRODUCCION ===
        new_births = []

        for cell in surviving_cells:
            org_id = cell.organism_id
            pop = sum(1 for c in surviving_cells if c.organism_id == org_id) + \
                  sum(1 for c in new_births if c.organism_id == org_id)

            if pop >= self.max_population:
                continue

            # Determinar umbral segun especie
            threshold = self.prey_repro_threshold if org_id == self.prey_id else self.predator_repro_threshold

            if cell.energy > threshold:
                # Buscar posicion para hija
                pos = self._find_empty_adjacent(*cell.position)
                if pos:
                    # Division: madre pierde energia, hija nace con mitad
                    daughter_energy = cell.energy / 2
                    cell.energy = cell.energy / 2

                    daughter = DualCellEntity(
                        position=pos,
                        state=torch.randn(self.state_dim) * 0.1,
                        role=torch.tensor([1.0, 0.0, 0.0]),  # Nace como Mass
                        energy=daughter_energy,
                        organism_id=org_id,
                        controlled_mass=0
                    )
                    new_births.append(daughter)

                    if org_id == self.predator_id:
                        self.births_pred += 1
                    else:
                        self.births_prey += 1

        # Combinar celulas sobrevivientes + nacimientos
        self.cells = surviving_cells + new_births

        # Regenerar parches
        for patch in self.patches:
            patch.update()

        self._update_grids()
        self.history.append(self.get_metrics())

    def get_metrics(self) -> Dict:
        """Metricas extendidas para Lotka-Volterra."""
        pred_cells = [c for c in self.cells if c.organism_id == self.predator_id]
        prey_cells = [c for c in self.cells if c.organism_id == self.prey_id]

        return {
            'n_pred': len(pred_cells),
            'n_prey': len(prey_cells),
            'n_pred_fi': sum(1 for c in pred_cells if c.role_idx == 1),
            'n_prey_fi': sum(1 for c in prey_cells if c.role_idx == 1),
            'births_pred': self.births_pred,
            'births_prey': self.births_prey,
            'deaths_pred_starve': self.deaths_pred_starve,
            'deaths_prey_starve': self.deaths_prey_starve,
            'deaths_prey_hunted': self.deaths_prey_hunted,
            'avg_energy_pred': np.mean([c.energy for c in pred_cells]) if pred_cells else 0,
            'avg_energy_prey': np.mean([c.energy for c in prey_cells]) if prey_cells else 0,
            'patches_active': sum(1 for p in self.patches if not p.is_depleted),
        }


def initialize_population(org, n_pred: int, n_prey: int):
    """Inicializa poblacion con cantidades especificas."""
    org.cells = []  # Limpiar cualquier celula existente
    center = org.grid_size // 2

    # Depredadores (energia inicial DEBAJO del umbral de reproduccion)
    for i in range(n_pred):
        x = center + np.random.randint(-15, 16)
        y = center + np.random.randint(-15, 16)
        x = np.clip(x, 0, org.grid_size - 1)
        y = np.clip(y, 0, org.grid_size - 1)

        # 30% empiezan como Fi (cazadores)
        is_fi = i < max(2, n_pred // 3)
        cell = DualCellEntity(
            position=(x, y),
            state=torch.randn(org.state_dim) * 0.1,
            role=torch.tensor([0.0, 1.0, 0.0]) if is_fi else torch.tensor([1.0, 0.0, 0.0]),
            energy=0.55 if is_fi else np.random.uniform(0.35, 0.50),  # Debajo de 0.65
            organism_id=org.predator_id
        )
        org.cells.append(cell)

    # Presas (energia inicial DEBAJO del umbral de reproduccion)
    for i in range(n_prey):
        x = center + np.random.randint(-20, 21)
        y = center + np.random.randint(-20, 21)
        x = np.clip(x, 0, org.grid_size - 1)
        y = np.clip(y, 0, org.grid_size - 1)

        cell = DualCellEntity(
            position=(x, y),
            state=torch.randn(org.state_dim) * 0.1,
            role=torch.tensor([0.0, 1.0, 0.0]) if i == 0 else torch.tensor([1.0, 0.0, 0.0]),
            energy=0.50 if i == 0 else np.random.uniform(0.30, 0.45),  # Debajo de 0.55
            organism_id=org.prey_id
        )
        org.cells.append(cell)

    org._update_grids()


def detect_cycles(population_history: List[int], min_periods: int = 2) -> Dict:
    """Detecta ciclos en una serie temporal de poblacion."""
    if len(population_history) < 100:
        return {'has_cycles': False, 'amplitude': 0, 'period': 0}

    data = np.array(population_history)

    # Amplitud
    amplitude = data.max() - data.min()

    if amplitude < 10:
        return {'has_cycles': False, 'amplitude': amplitude, 'period': 0}

    # Detectar periodo usando autocorrelacion
    data_centered = data - data.mean()
    autocorr = np.correlate(data_centered, data_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalizar

    # Buscar primer pico significativo (despues de lag 0)
    peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=10)

    if len(peaks) >= min_periods:
        period = peaks[0]  # Primer pico = periodo
        return {'has_cycles': True, 'amplitude': amplitude, 'period': period}

    return {'has_cycles': False, 'amplitude': amplitude, 'period': 0}


def run_scenario(name: str, n_pred: int, n_prey: int, refuge: float = 0.0,
                 n_steps: int = 1200) -> Dict:
    """Ejecuta un escenario de Lotka-Volterra."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Inicial: Pred={n_pred}, Presa={n_prey}, Refugio={refuge*100:.0f}%')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = LotkaVolterraOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        n_patches=5,
        refuge_effectiveness=refuge,
        prey_repro_threshold=0.55,
        predator_repro_threshold=0.60,  # Depredador reproduce mas facil
        hunt_success_rate=0.45,         # Alta tasa de caza
        hunt_energy_gain=0.35,          # Ganancia moderada por cazar
        prey_resource_bonus=0.06,       # Presa gana de recursos
        predator_resource_bonus=0.04,   # Depredador tambien gana
        base_energy_decay=0.004,        # Decay muy bajo
        predator_hunger_rate=0.006,     # Hambre muy baja
        max_population=100,
    )

    initialize_population(org, n_pred, n_prey)

    # Tracking
    pop_history = {'pred': [], 'prey': [], 'steps': []}
    birth_history = {'pred': [], 'prey': []}
    death_history = {'pred': [], 'prey_starve': [], 'prey_hunted': []}

    extinction_step = None
    extinction_type = None

    for step in range(n_steps):
        org.step()
        m = org.get_metrics()

        # Guardar cada 5 steps
        if step % 5 == 0:
            pop_history['pred'].append(m['n_pred'])
            pop_history['prey'].append(m['n_prey'])
            pop_history['steps'].append(step)
            birth_history['pred'].append(m['births_pred'])
            birth_history['prey'].append(m['births_prey'])
            death_history['pred'].append(m['deaths_pred_starve'])
            death_history['prey_starve'].append(m['deaths_prey_starve'])
            death_history['prey_hunted'].append(m['deaths_prey_hunted'])

        # Detectar extincion
        if extinction_step is None:
            if m['n_pred'] == 0:
                extinction_step = step + 1
                extinction_type = 'predator'
                print(f'\n  [EXTINCION] Depredador extinto en step {step+1}!')
            elif m['n_prey'] == 0:
                extinction_step = step + 1
                extinction_type = 'prey'
                print(f'\n  [EXTINCION] Presa extinta en step {step+1}!')

        # Progreso cada 300 steps
        if (step + 1) % 300 == 0:
            print(f'  Step {step+1}: Pred={m["n_pred"]}, Presa={m["n_prey"]}, '
                  f'Nac={m["births_pred"]}/{m["births_prey"]}, '
                  f'Muertes={m["deaths_pred_starve"]}/{m["deaths_prey_starve"]+m["deaths_prey_hunted"]}')

    # Analisis de ciclos
    cycle_pred = detect_cycles(pop_history['pred'])
    cycle_prey = detect_cycles(pop_history['prey'])

    has_cycles = cycle_pred['has_cycles'] or cycle_prey['has_cycles']

    final = org.get_metrics()
    print(f'\nFinal: Pred={final["n_pred"]}, Presa={final["n_prey"]}')

    if has_cycles:
        print(f'CICLOS DETECTADOS!')
        print(f'  Pred: amplitud={cycle_pred["amplitude"]:.0f}, periodo={cycle_pred["period"]}')
        print(f'  Presa: amplitud={cycle_prey["amplitude"]:.0f}, periodo={cycle_prey["period"]}')
    else:
        print(f'Sin ciclos (amplitud pred={cycle_pred["amplitude"]:.0f}, presa={cycle_prey["amplitude"]:.0f})')

    return {
        'org': org,
        'name': name,
        'pop_history': pop_history,
        'birth_history': birth_history,
        'death_history': death_history,
        'extinction_step': extinction_step,
        'extinction_type': extinction_type,
        'cycle_pred': cycle_pred,
        'cycle_prey': cycle_prey,
        'has_cycles': has_cycles,
    }


def run_full_experiment():
    """Ejecuta todos los escenarios de Lotka-Volterra."""
    print('='*70)
    print('EXPERIMENTO: DINAMICA LOTKA-VOLTERRA')
    print('Mecanicas: Reproduccion por division, muerte por inanicion, caza letal')
    print('='*70)

    scenarios = [
        ('Sin refugio', 20, 40, 0.0),
        ('Refugio 50%', 20, 40, 0.5),
        ('Refugio 80%', 20, 40, 0.8),
        ('Refugio 90%', 20, 40, 0.9),
    ]

    results = {}
    for name, n_pred, n_prey, refuge in scenarios:
        results[name] = run_scenario(name, n_pred, n_prey, refuge, n_steps=1200)

    # === ANALISIS ===
    print('\n' + '='*70)
    print('ANALISIS COMPARATIVO')
    print('='*70)

    print(f'\n{"Escenario":<25} {"Pred":<6} {"Presa":<6} {"Extincion":<12} {"Ciclos":<8} {"Amp Pred":<10} {"Amp Presa":<10}')
    print('-'*85)

    for name, data in results.items():
        final = data['org'].get_metrics()
        ext = data['extinction_step'] or '-'
        cycles = 'Si' if data['has_cycles'] else 'No'
        amp_pred = f"{data['cycle_pred']['amplitude']:.0f}"
        amp_prey = f"{data['cycle_prey']['amplitude']:.0f}"
        print(f'{name:<25} {final["n_pred"]:<6} {final["n_prey"]:<6} '
              f'{str(ext):<12} {cycles:<8} {amp_pred:<10} {amp_prey:<10}')

    # === VISUALIZACION ===
    n_scenarios = len(results)
    fig, axes = plt.subplots(3, n_scenarios, figsize=(5*n_scenarios, 12))

    colors = {'pred': 'crimson', 'prey': 'royalblue'}

    for idx, (name, data) in enumerate(results.items()):
        org = data['org']
        pop = data['pop_history']

        # Fila 1: Evolucion de poblacion
        ax = axes[0, idx]
        ax.plot(pop['steps'], pop['pred'], color=colors['pred'], linewidth=1.5, label='Depredador')
        ax.plot(pop['steps'], pop['prey'], color=colors['prey'], linewidth=1.5, label='Presa')
        if data['extinction_step']:
            ax.axvline(x=data['extinction_step'], color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Poblacion')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 85)

        # Fila 2: Fase (pred vs prey)
        ax = axes[1, idx]
        ax.plot(pop['prey'], pop['pred'], color='purple', linewidth=0.5, alpha=0.7)
        ax.scatter(pop['prey'][0], pop['pred'][0], color='green', s=100, marker='o', label='Inicio', zorder=5)
        ax.scatter(pop['prey'][-1], pop['pred'][-1], color='red', s=100, marker='x', label='Final', zorder=5)
        ax.set_xlabel('Poblacion Presa')
        ax.set_ylabel('Poblacion Depredador')
        ax.set_title('Diagrama de Fase')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Fila 3: Estado final
        ax = axes[2, idx]

        # Dibujar parches
        for patch in org.patches:
            circle = plt.Circle(patch.position, patch.radius, color='green', alpha=0.15)
            ax.add_patch(circle)

        # Dibujar celulas
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
    plt.savefig('zeta_organism_lotka_volterra.png', dpi=150)
    print('\nGuardado: zeta_organism_lotka_volterra.png')

    # === CONCLUSIONES ===
    print('\n' + '='*70)
    print('CONCLUSIONES')
    print('='*70)

    coexistence_count = sum(1 for r in results.values() if r['extinction_step'] is None)
    cycles_count = sum(1 for r in results.values() if r['has_cycles'])

    print(f'\nCoexistencia: {coexistence_count}/{len(results)} escenarios')
    print(f'Ciclos Lotka-Volterra: {cycles_count}/{len(results)} escenarios')

    for name, data in results.items():
        if data['has_cycles']:
            print(f'[CICLOS] {name}:')
            print(f'  Pred: amplitud={data["cycle_pred"]["amplitude"]:.0f}, periodo={data["cycle_pred"]["period"]}')
            print(f'  Presa: amplitud={data["cycle_prey"]["amplitude"]:.0f}, periodo={data["cycle_prey"]["period"]}')
        elif data['extinction_step'] is None:
            print(f'[COEXISTENCIA] {name}: estable sin ciclos')
        else:
            print(f'[EXTINCION] {name}: {data["extinction_type"]} en step {data["extinction_step"]}')

    if cycles_count > 0:
        print('\n[EXITO] Dinamica Lotka-Volterra lograda!')
    else:
        print('\n[PARCIAL] Coexistencia lograda pero sin ciclos claros')

    return results


if __name__ == '__main__':
    results = run_full_experiment()
