# exp_comunicacion_quimica.py
"""Experimento: Comunicacion quimica entre organismos.

Sistema de feromonas:
- ALARMA: Fi emite cuando detecta enemigo -> celulas huyen
- ATRACCION: Fi emite en recursos -> celulas acuden
- TERRITORIAL: Fi emite constantemente -> marca territorio

Mecanica:
- Grids de difusion (64x64) por tipo y organismo
- Difusion gaussiana + evaporacion por step
- Celulas responden a gradientes
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from scipy.ndimage import gaussian_filter

from exp_dos_organismos import DualOrganism, DualCellEntity
from exp_ecosistema import ResourcePatch


@dataclass
class PheromoneSystem:
    """Sistema de feromonas para un organismo."""
    grid_size: int = 64

    # Grids de feromonas
    alarm: np.ndarray = field(default=None)
    attraction: np.ndarray = field(default=None)
    territorial: np.ndarray = field(default=None)

    # Grids de direccion de alarma (NUEVO: alarma dirigida)
    alarm_dir_x: np.ndarray = field(default=None)
    alarm_dir_y: np.ndarray = field(default=None)

    # Parametros
    diffusion_rate: float = 0.8      # Sigma del blur gaussiano
    evaporation_rate: float = 0.05   # Decay por step

    # Intensidades de emision
    alarm_strength: float = 5.0
    attraction_strength: float = 3.0
    territorial_strength: float = 2.0

    # Modo de alarma
    directed_alarm: bool = False  # True = alarma dirigida, False = isotrópica

    def __post_init__(self):
        if self.alarm is None:
            self.alarm = np.zeros((self.grid_size, self.grid_size))
        if self.attraction is None:
            self.attraction = np.zeros((self.grid_size, self.grid_size))
        if self.territorial is None:
            self.territorial = np.zeros((self.grid_size, self.grid_size))
        # Direccion de huida (normalizada)
        if self.alarm_dir_x is None:
            self.alarm_dir_x = np.zeros((self.grid_size, self.grid_size))
        if self.alarm_dir_y is None:
            self.alarm_dir_y = np.zeros((self.grid_size, self.grid_size))

    def emit_alarm(self, x: int, y: int):
        """Fi emite alarma isotrópica (enemigo detectado)."""
        self.alarm[y, x] += self.alarm_strength

    def emit_alarm_directed(self, x: int, y: int, flee_dx: float, flee_dy: float):
        """Fi emite alarma DIRIGIDA con vector de huida."""
        self.alarm[y, x] += self.alarm_strength
        # Acumular direccion de huida (promedio ponderado por intensidad)
        self.alarm_dir_x[y, x] += flee_dx * self.alarm_strength
        self.alarm_dir_y[y, x] += flee_dy * self.alarm_strength

    def emit_attraction(self, x: int, y: int):
        """Fi emite atraccion (recursos encontrados)."""
        self.attraction[y, x] += self.attraction_strength

    def emit_territorial(self, x: int, y: int):
        """Fi emite territorial (marca territorio)."""
        self.territorial[y, x] += self.territorial_strength

    def diffuse_and_evaporate(self):
        """Difusion gaussiana + evaporacion."""
        # Difusion (blur gaussiano)
        self.alarm = gaussian_filter(self.alarm, sigma=self.diffusion_rate)
        self.attraction = gaussian_filter(self.attraction, sigma=self.diffusion_rate)
        self.territorial = gaussian_filter(self.territorial, sigma=self.diffusion_rate)

        # Difundir direccion de alarma tambien
        self.alarm_dir_x = gaussian_filter(self.alarm_dir_x, sigma=self.diffusion_rate)
        self.alarm_dir_y = gaussian_filter(self.alarm_dir_y, sigma=self.diffusion_rate)

        # Evaporacion
        self.alarm *= (1 - self.evaporation_rate)
        self.attraction *= (1 - self.evaporation_rate)
        self.territorial *= (1 - self.evaporation_rate)
        self.alarm_dir_x *= (1 - self.evaporation_rate)
        self.alarm_dir_y *= (1 - self.evaporation_rate)

        # Clip para evitar valores negativos
        self.alarm = np.clip(self.alarm, 0, 10)
        self.attraction = np.clip(self.attraction, 0, 10)
        self.territorial = np.clip(self.territorial, 0, 10)

    def get_flee_direction(self, x: int, y: int) -> Tuple[float, float]:
        """Obtiene direccion de huida en posicion (normalizada)."""
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        dx = self.alarm_dir_x[y, x]
        dy = self.alarm_dir_y[y, x]
        # Normalizar
        mag = np.sqrt(dx*dx + dy*dy) + 0.001
        return (dx / mag, dy / mag)

    def get_gradient(self, grid: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        """Calcula gradiente en posicion (x, y)."""
        # Usar diferencias finitas con bordes
        x = np.clip(x, 1, self.grid_size - 2)
        y = np.clip(y, 1, self.grid_size - 2)

        dx = grid[y, x + 1] - grid[y, x - 1]
        dy = grid[y + 1, x] - grid[y - 1, x]

        return (dx, dy)

    def get_concentration(self, grid: np.ndarray, x: int, y: int) -> float:
        """Obtiene concentracion en posicion."""
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        return grid[y, x]

    def total_concentration(self) -> Dict[str, float]:
        """Concentracion total de cada feromona."""
        return {
            'alarm': float(np.sum(self.alarm)),
            'attraction': float(np.sum(self.attraction)),
            'territorial': float(np.sum(self.territorial))
        }


class ChemicalOrganism(DualOrganism):
    """Organismo con comunicacion quimica."""

    def __init__(self,
                 # Parametros de feromonas
                 diffusion_rate: float = 0.8,
                 evaporation_rate: float = 0.05,
                 alarm_strength: float = 2.0,
                 attraction_strength: float = 1.5,
                 territorial_strength: float = 0.5,
                 # Pesos de respuesta
                 alarm_weight: float = 1.5,
                 attraction_weight: float = 1.0,
                 territorial_weight: float = 1.2,
                 # Radios de deteccion
                 enemy_detection_radius: float = 15.0,
                 # Recursos
                 n_patches: int = 3,
                 patch_radius: float = 8.0,
                 # Activar/desactivar feromonas
                 pheromones_enabled: bool = True,
                 # NUEVO: alarma dirigida
                 directed_alarm: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        # Modo de alarma
        self.directed_alarm = directed_alarm

        # Sistema de feromonas por organismo
        self.pheromones = {
            0: PheromoneSystem(
                grid_size=self.grid_size,
                diffusion_rate=diffusion_rate,
                evaporation_rate=evaporation_rate,
                alarm_strength=alarm_strength,
                attraction_strength=attraction_strength,
                territorial_strength=territorial_strength,
                directed_alarm=directed_alarm
            ),
            1: PheromoneSystem(
                grid_size=self.grid_size,
                diffusion_rate=diffusion_rate,
                evaporation_rate=evaporation_rate,
                alarm_strength=alarm_strength,
                attraction_strength=attraction_strength,
                territorial_strength=territorial_strength,
                directed_alarm=directed_alarm
            )
        }

        # Pesos de respuesta
        self.alarm_weight = alarm_weight
        self.attraction_weight = attraction_weight
        self.territorial_weight = territorial_weight

        # Deteccion
        self.enemy_detection_radius = enemy_detection_radius

        # Recursos
        self.patches = self._create_patches(n_patches, patch_radius)

        # Control
        self.pheromones_enabled = pheromones_enabled

        # Estadisticas
        self.alarm_emissions = {0: 0, 1: 0}
        self.attraction_emissions = {0: 0, 1: 0}
        self.territorial_emissions = {0: 0, 1: 0}

    def _create_patches(self, n_patches: int, radius: float,
                        offset_position: tuple = None) -> List[ResourcePatch]:
        """Crea parches de recursos.

        Args:
            n_patches: Número de parches
            radius: Radio de cada parche
            offset_position: Si se especifica, usa esta posición en lugar de centro
        """
        patches = []
        center = self.grid_size // 2

        if offset_position:
            # Usar posición personalizada
            base_x, base_y = offset_position
        else:
            base_x, base_y = center, center

        if n_patches == 1:
            positions = [(base_x, base_y)]
        elif n_patches == 3:
            positions = [
                (base_x, base_y),
                (base_x - 15, base_y),
                (base_x + 15, base_y)
            ]
        else:
            positions = [(base_x, base_y)]

        for pos in positions[:n_patches]:
            patch = ResourcePatch(
                position=pos,
                radius=radius,
                capacity=100.0,
                current=100.0,
                regen_delay=30
            )
            patches.append(patch)

        return patches

    def _is_in_patch(self, x: int, y: int) -> bool:
        """Verifica si posicion esta en un parche."""
        for patch in self.patches:
            if patch.contains(x, y):
                return True
        return False

    def initialize_conflict(self):
        """Inicializa organismos cerca del centro (zona de conflicto)."""
        self.cells = []
        center = self.grid_size // 2

        for org_id in range(2):
            for i in range(self.n_cells_per_org):
                # Ambos organismos cerca del centro con ligera separacion
                offset = -8 if org_id == 0 else 8
                x = center + offset + np.random.randint(-10, 10)
                y = center + np.random.randint(-15, 15)
                x = np.clip(x, 2, self.grid_size - 3)
                y = np.clip(y, 2, self.grid_size - 3)

                state = torch.randn(self.state_dim) * 0.1

                if i == 0:
                    role = torch.tensor([0.0, 1.0, 0.0])
                    energy = 0.8
                else:
                    role = torch.tensor([1.0, 0.0, 0.0])
                    energy = 0.5

                cell = DualCellEntity(
                    position=(x, y),
                    state=state,
                    role=role,
                    energy=energy,
                    organism_id=org_id
                )
                self.cells.append(cell)

        self._update_grids()

    def initialize_foraging(self):
        """Inicializa organismos en esquinas - Fi con sus Mass.

        Test válido de atracción:
        - TODOS los organismos empiezan en esquinas
        - Recursos en centro (sin células iniciales)
        - La atracción debe guiar a las células hacia recursos
        """
        self.cells = []
        center = self.grid_size // 2

        for org_id in range(2):
            # Posición base de la esquina para este organismo
            if org_id == 0:
                base_x, base_y = 10, 10  # Esquina inferior izquierda
            else:
                base_x, base_y = 54, 54  # Esquina superior derecha

            for i in range(self.n_cells_per_org):
                state = torch.randn(self.state_dim) * 0.1

                # Fi líderes en la esquina
                if i < 2:
                    x = base_x + np.random.randint(-3, 3)
                    y = base_y + np.random.randint(-3, 3)
                    role = torch.tensor([0.0, 1.0, 0.0])  # Fi
                    energy = 0.9
                else:
                    # Mass alrededor de sus Fi
                    x = base_x + np.random.randint(-8, 8)
                    y = base_y + np.random.randint(-8, 8)
                    role = torch.tensor([1.0, 0.0, 0.0])  # Mass
                    energy = 0.5

                x = np.clip(x, 2, self.grid_size - 3)
                y = np.clip(y, 2, self.grid_size - 3)

                cell = DualCellEntity(
                    position=(x, y),
                    state=state,
                    role=role,
                    energy=energy,
                    organism_id=org_id
                )
                self.cells.append(cell)

        self._update_grids()

    def count_cells_in_patches(self) -> Dict[int, int]:
        """Cuenta células de cada organismo en parches de recursos."""
        counts = {0: 0, 1: 0}
        for cell in self.cells:
            x, y = cell.position
            if self._is_in_patch(x, y):
                counts[cell.organism_id] += 1
        return counts

    def set_patch_position(self, position: tuple, radius: float = 10.0):
        """Reposiciona los parches de recursos a nueva ubicación."""
        self.patches = [ResourcePatch(
            position=position,
            radius=radius,
            capacity=100.0,
            current=100.0,
            regen_delay=30
        )]

    def get_distance_to_nearest_patch(self, org_id: int) -> float:
        """Distancia promedio de células al parche más cercano."""
        cells = [c for c in self.cells if c.organism_id == org_id]
        if not cells:
            return 0.0

        total_dist = 0.0
        for cell in cells:
            cx, cy = cell.position
            min_dist = float('inf')
            for patch in self.patches:
                px, py = patch.position
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                min_dist = min(min_dist, dist)
            total_dist += min_dist

        return total_dist / len(cells)

    def _detect_enemy(self, cell: DualCellEntity) -> Tuple[bool, float, float]:
        """Detecta si hay enemigo cerca y retorna direccion de huida.

        Returns:
            (detected, flee_dx, flee_dy): Si hay enemigo y vector de huida normalizado
        """
        x, y = cell.position
        enemy_id = 1 - cell.organism_id

        nearest_dist = float('inf')
        nearest_enemy = None

        for other in self.cells:
            if other.organism_id == enemy_id:
                dist = np.sqrt((other.position[0] - x)**2 + (other.position[1] - y)**2)
                if dist < self.enemy_detection_radius and dist < nearest_dist:
                    nearest_dist = dist
                    nearest_enemy = other

        if nearest_enemy is None:
            return (False, 0.0, 0.0)

        # Calcular direccion de huida (opuesta al enemigo)
        ex, ey = nearest_enemy.position
        flee_dx = x - ex  # Direccion opuesta
        flee_dy = y - ey
        mag = np.sqrt(flee_dx**2 + flee_dy**2) + 0.001
        flee_dx /= mag
        flee_dy /= mag

        return (True, flee_dx, flee_dy)

    def _emit_pheromones(self):
        """Fase de emision de feromonas."""
        self.alarm_emissions = {0: 0, 1: 0}
        self.attraction_emissions = {0: 0, 1: 0}
        self.territorial_emissions = {0: 0, 1: 0}

        if not self.pheromones_enabled:
            return

        for cell in self.cells:
            if cell.role_idx != 1:  # Solo Fi emite
                continue

            x, y = cell.position
            org_id = cell.organism_id
            pheromone = self.pheromones[org_id]

            # ALARMA: si detecta enemigo
            detected, flee_dx, flee_dy = self._detect_enemy(cell)
            if detected:
                if self.directed_alarm:
                    # ALARMA DIRIGIDA: incluye vector de huida
                    pheromone.emit_alarm_directed(x, y, flee_dx, flee_dy)
                else:
                    # ALARMA ISOTRÓPICA: solo intensidad
                    pheromone.emit_alarm(x, y)
                self.alarm_emissions[org_id] += 1

            # ATRACCION: si esta en parche de recursos
            if self._is_in_patch(x, y):
                pheromone.emit_attraction(x, y)
                self.attraction_emissions[org_id] += 1

            # TERRITORIAL: siempre (baja intensidad)
            pheromone.emit_territorial(x, y)
            self.territorial_emissions[org_id] += 1

    def _compute_pheromone_influence(self, cell: DualCellEntity) -> Tuple[float, float]:
        """Calcula influencia de feromonas en movimiento."""
        if not self.pheromones_enabled:
            return (0.0, 0.0)

        x, y = cell.position
        org_id = cell.organism_id
        enemy_id = 1 - org_id

        own_pheromone = self.pheromones[org_id]
        enemy_pheromone = self.pheromones[enemy_id]

        # ALARMA: depende del modo
        if self.directed_alarm:
            # ALARMA DIRIGIDA: usar vector de huida almacenado
            alarm_intensity = own_pheromone.get_concentration(own_pheromone.alarm, x, y)
            if alarm_intensity > 0.1:
                flee_dir = own_pheromone.get_flee_direction(x, y)
                dx = self.alarm_weight * flee_dir[0] * alarm_intensity
                dy = self.alarm_weight * flee_dir[1] * alarm_intensity
            else:
                dx, dy = 0.0, 0.0
        else:
            # ALARMA ISOTRÓPICA: anti-gradiente (huir del centro de alarma)
            alarm_grad = own_pheromone.get_gradient(own_pheromone.alarm, x, y)
            dx = -self.alarm_weight * alarm_grad[0]
            dy = -self.alarm_weight * alarm_grad[1]

        # ATRACCION: acudir (seguir gradiente)
        attract_grad = own_pheromone.get_gradient(own_pheromone.attraction, x, y)
        dx += self.attraction_weight * attract_grad[0]
        dy += self.attraction_weight * attract_grad[1]

        # TERRITORIAL enemigo: evitar (anti-gradiente)
        enemy_terr_grad = enemy_pheromone.get_gradient(enemy_pheromone.territorial, x, y)
        dx -= self.territorial_weight * enemy_terr_grad[0]
        dy -= self.territorial_weight * enemy_terr_grad[1]

        return (dx, dy)

    def step(self):
        """Step con comunicacion quimica."""
        # === FASE 1: EMISION DE FEROMONAS ===
        self._emit_pheromones()

        # === FASE 2: DIFUSION Y EVAPORACION ===
        if self.pheromones_enabled:
            self.pheromones[0].diffuse_and_evaporate()
            self.pheromones[1].diffuse_and_evaporate()

        # === FASE 3: MOVIMIENTO Y DINAMICA ===
        field, gradient = self.force_field.compute_with_gradient(
            self.energy_grid, self.role_grid
        )

        new_cells = []

        for cell in self.cells:
            x, y = cell.position
            org_id = cell.organism_id

            # Vecinos
            all_neighbors = self._get_neighbors(cell, radius=5, same_org_only=False)
            same_org_neighbors = [n for n in all_neighbors if n.organism_id == org_id]
            other_org_neighbors = [n for n in all_neighbors if n.organism_id != org_id]

            same_mass = sum(1 for n in same_org_neighbors if n.role_idx == 0)
            same_fi = sum(1 for n in same_org_neighbors if n.role_idx == 1)
            other_fi = sum(1 for n in other_org_neighbors if n.role_idx == 1)

            # Componente neural
            behavior = self.behavior_0 if org_id == 0 else self.behavior_1
            potential = field[0, 0, y, x].item()

            if same_org_neighbors:
                neighbor_states = torch.stack([n.state for n in same_org_neighbors])
                influence_out, influence_in = behavior.bidirectional_influence(
                    cell.state, neighbor_states
                )
                net_influence = (influence_out.mean() - influence_in).item()
                new_state = cell.state + 0.1 * behavior.self_similarity(cell.state)
            else:
                net_influence = 0.0
                new_state = cell.state.clone()

            # Energia
            new_energy = cell.energy * 0.99 + 0.05 * max(0, potential)
            if same_fi > 0:
                new_energy += 0.01
            if other_fi > 0:
                new_energy -= 0.02

            # Bonus por recursos
            if self._is_in_patch(x, y):
                new_energy += 0.03

            new_energy = np.clip(new_energy, 0, 1)

            # === TRANSICION DE ROL ===
            if cell.role_idx == 0:  # Mass
                can_become_fi = (
                    new_energy > self.fi_threshold and
                    same_mass >= 2 and
                    same_fi == 0 and
                    other_fi == 0
                )
                new_role = torch.tensor([0.0, 1.0, 0.0]) if can_become_fi else torch.tensor([1.0, 0.0, 0.0])
            else:  # Fi
                loses_fi = same_mass < 1 or new_energy < 0.2
                new_role = torch.tensor([1.0, 0.0, 0.0]) if loses_fi else torch.tensor([0.0, 1.0, 0.0])

            # === MOVIMIENTO ===
            new_x, new_y = x, y
            new_role_idx = new_role.argmax().item()

            # Movimiento base
            base_dx, base_dy = 0, 0

            if new_role_idx == 0:  # Mass sigue a Fi
                same_fi_cells = [c for c in self.cells
                                if c.organism_id == org_id and c.role_idx == 1]
                if same_fi_cells:
                    nearest = min(same_fi_cells, key=lambda f:
                        (f.position[0] - x)**2 + (f.position[1] - y)**2)
                    base_dx = int(np.sign(nearest.position[0] - x))
                    base_dy = int(np.sign(nearest.position[1] - y))
            else:  # Fi explora con sesgo hacia centro
                # Sesgo hacia CENTRO DEL GRID (no hacia recursos)
                # Simula comportamiento natural de buscar áreas centrales
                center = self.grid_size // 2
                if np.random.random() < 0.35:  # 35% chance de moverse
                    # 70% del tiempo: hacia centro del grid
                    # 30% del tiempo: aleatorio
                    if np.random.random() < 0.7:
                        base_dx = int(np.sign(center - x)) if x != center else 0
                        base_dy = int(np.sign(center - y)) if y != center else 0
                    else:
                        base_dx = np.random.choice([-1, 0, 1])
                        base_dy = np.random.choice([-1, 0, 1])

            # Influencia de feromonas
            pheromone_dx, pheromone_dy = self._compute_pheromone_influence(cell)

            # Combinar movimiento base + feromonas
            total_dx = base_dx + pheromone_dx
            total_dy = base_dy + pheromone_dy

            # Normalizar y aplicar (umbral bajo para detectar feromonas)
            if abs(total_dx) > 0.05 or abs(total_dy) > 0.05:
                move_x = int(np.sign(total_dx)) if abs(total_dx) > 0.1 else 0
                move_y = int(np.sign(total_dy)) if abs(total_dy) > 0.1 else 0
                new_x = np.clip(x + move_x, 0, self.grid_size - 1)
                new_y = np.clip(y + move_y, 0, self.grid_size - 1)

            new_cell = DualCellEntity(
                position=(new_x, new_y),
                state=new_state.detach(),
                role=new_role,
                energy=new_energy,
                organism_id=org_id,
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
        """Metricas extendidas con feromonas."""
        metrics = super().get_metrics()

        # Concentraciones de feromonas
        for org_id in [0, 1]:
            conc = self.pheromones[org_id].total_concentration()
            metrics[f'org_{org_id}_alarm'] = conc['alarm']
            metrics[f'org_{org_id}_attraction'] = conc['attraction']
            metrics[f'org_{org_id}_territorial'] = conc['territorial']

        # Emisiones este step
        metrics['alarm_emissions'] = self.alarm_emissions
        metrics['attraction_emissions'] = self.attraction_emissions

        # Distancia entre centroides
        org0_cells = [c for c in self.cells if c.organism_id == 0]
        org1_cells = [c for c in self.cells if c.organism_id == 1]

        if org0_cells and org1_cells:
            c0 = np.mean([c.position for c in org0_cells], axis=0)
            c1 = np.mean([c.position for c in org1_cells], axis=0)
            metrics['centroid_distance'] = float(np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2))
        else:
            metrics['centroid_distance'] = 0.0

        # Celulas en territorio enemigo (cruzan frontera)
        center = self.grid_size // 2
        org0_in_enemy = sum(1 for c in org0_cells if c.position[0] > center + 5)
        org1_in_enemy = sum(1 for c in org1_cells if c.position[0] < center - 5)
        metrics['boundary_crossings'] = org0_in_enemy + org1_in_enemy

        return metrics


def run_scenario(name: str, pheromones_enabled: bool, n_steps: int = 600) -> Dict:
    """Ejecuta un escenario."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Feromonas: {"Activadas" if pheromones_enabled else "Desactivadas"}')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ChemicalOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        n_patches=3,
        pheromones_enabled=pheromones_enabled,
        diffusion_rate=0.8,
        evaporation_rate=0.03,          # Menor evaporacion
        alarm_strength=5.0,              # Alta intensidad
        attraction_strength=3.0,
        territorial_strength=2.0,
        alarm_weight=1.5,                # Mayor respuesta
        attraction_weight=1.0,
        territorial_weight=1.2,
        enemy_detection_radius=18.0,     # Mayor radio deteccion
    )

    # Cargar pesos si existen
    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior_0.load_state_dict(weights['behavior_state'])
        org.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # Inicializar con separacion horizontal
    org.initialize(separation='horizontal')

    initial = org.get_metrics()
    print(f'Estado inicial:')
    print(f'  Org 0: {initial["org_0"]["n_total"]} celulas')
    print(f'  Org 1: {initial["org_1"]["n_total"]} celulas')
    print(f'  Distancia centroides: {initial["centroid_distance"]:.1f}')

    # Tracking
    history = {
        'centroid_distance': [],
        'boundary_crossings': [],
        'alarm_0': [], 'alarm_1': [],
        'attraction_0': [], 'attraction_1': [],
        'territorial_0': [], 'territorial_1': [],
        'steps': []
    }

    for step in range(n_steps):
        org.step()
        m = org.get_metrics()

        # Guardar cada 5 steps
        if step % 5 == 0:
            history['centroid_distance'].append(m['centroid_distance'])
            history['boundary_crossings'].append(m['boundary_crossings'])
            history['alarm_0'].append(m['org_0_alarm'])
            history['alarm_1'].append(m['org_1_alarm'])
            history['attraction_0'].append(m['org_0_attraction'])
            history['attraction_1'].append(m['org_1_attraction'])
            history['territorial_0'].append(m['org_0_territorial'])
            history['territorial_1'].append(m['org_1_territorial'])
            history['steps'].append(step)

        # Progreso cada 150 steps
        if (step + 1) % 150 == 0:
            print(f'  Step {step+1}: Dist={m["centroid_distance"]:.1f}, '
                  f'Cruces={m["boundary_crossings"]}, '
                  f'Alarm={m["org_0_alarm"]:.1f}/{m["org_1_alarm"]:.1f}')

    final = org.get_metrics()
    print(f'\nFinal:')
    print(f'  Distancia centroides: {final["centroid_distance"]:.1f}')
    print(f'  Cruces de frontera: {final["boundary_crossings"]}')

    return {
        'org': org,
        'name': name,
        'history': history,
        'pheromones_enabled': pheromones_enabled,
    }


def run_conflict_scenario(name: str, pheromones_enabled: bool, n_steps: int = 600,
                          directed_alarm: bool = False) -> Dict:
    """Ejecuta escenario de conflicto (organismos cerca)."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Feromonas: {"Activadas" if pheromones_enabled else "Desactivadas"}')
    if pheromones_enabled:
        print(f'Alarma: {"DIRIGIDA (vector huida)" if directed_alarm else "ISOTRÓPICA (gradiente)"}')
    print(f'Modo: CONFLICTO (organismos cerca del centro)')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ChemicalOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        n_patches=3,
        pheromones_enabled=pheromones_enabled,
        diffusion_rate=0.8,
        evaporation_rate=0.03,
        alarm_strength=5.0,
        attraction_strength=3.0,
        territorial_strength=2.0,
        alarm_weight=1.5,
        attraction_weight=1.0,
        territorial_weight=1.2,
        enemy_detection_radius=18.0,
        directed_alarm=directed_alarm,
    )

    # Cargar pesos si existen
    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior_0.load_state_dict(weights['behavior_state'])
        org.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # CONFLICTO: organismos cerca del centro
    org.initialize_conflict()

    initial = org.get_metrics()
    print(f'Estado inicial:')
    print(f'  Org 0: {initial["org_0"]["n_total"]} celulas')
    print(f'  Org 1: {initial["org_1"]["n_total"]} celulas')
    print(f'  Distancia centroides: {initial["centroid_distance"]:.1f}')

    history = {
        'centroid_distance': [],
        'boundary_crossings': [],
        'alarm_0': [], 'alarm_1': [],
        'attraction_0': [], 'attraction_1': [],
        'territorial_0': [], 'territorial_1': [],
        'steps': []
    }

    for step in range(n_steps):
        org.step()
        m = org.get_metrics()

        if step % 5 == 0:
            history['centroid_distance'].append(m['centroid_distance'])
            history['boundary_crossings'].append(m['boundary_crossings'])
            history['alarm_0'].append(m['org_0_alarm'])
            history['alarm_1'].append(m['org_1_alarm'])
            history['attraction_0'].append(m['org_0_attraction'])
            history['attraction_1'].append(m['org_1_attraction'])
            history['territorial_0'].append(m['org_0_territorial'])
            history['territorial_1'].append(m['org_1_territorial'])
            history['steps'].append(step)

        if (step + 1) % 150 == 0:
            print(f'  Step {step+1}: Dist={m["centroid_distance"]:.1f}, '
                  f'Cruces={m["boundary_crossings"]}, '
                  f'Alarm={m["org_0_alarm"]:.1f}/{m["org_1_alarm"]:.1f}')

    final = org.get_metrics()
    print(f'\nFinal:')
    print(f'  Distancia centroides: {final["centroid_distance"]:.1f}')
    print(f'  Cruces de frontera: {final["boundary_crossings"]}')

    return {
        'org': org,
        'name': name,
        'history': history,
        'pheromones_enabled': pheromones_enabled,
    }


def run_foraging_scenario(name: str, pheromones_enabled: bool, n_steps: int = 600,
                          attraction_weight: float = 1.0) -> Dict:
    """Ejecuta escenario de forrajeo (organismos lejos de recursos)."""
    print(f'\n{"="*60}')
    print(f'ESCENARIO: {name}')
    print(f'Feromonas: {"Activadas" if pheromones_enabled else "Desactivadas"}')
    print(f'Modo: FORRAJEO (organismos en esquinas, recursos en centro)')
    if pheromones_enabled:
        print(f'Peso atracción: {attraction_weight}')
    print('='*60)

    torch.manual_seed(42)
    np.random.seed(42)

    org = ChemicalOrganism(
        grid_size=64,
        n_cells_per_org=40,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5,
        n_patches=1,  # Solo un parche central grande
        patch_radius=12.0,  # Radio grande
        pheromones_enabled=pheromones_enabled,
        diffusion_rate=2.5,  # Alta difusión para largo alcance
        evaporation_rate=0.005,  # Muy baja evaporación
        alarm_strength=3.0,
        attraction_strength=10.0,  # Muy alta atracción
        territorial_strength=0.5,
        alarm_weight=0.3,  # Menos peso a alarma
        attraction_weight=attraction_weight,  # Variable
        territorial_weight=0.2,
        enemy_detection_radius=12.0,
        directed_alarm=True,
    )

    # Cargar pesos si existen
    try:
        weights = torch.load('zeta_organism_weights.pt')
        org.behavior_0.load_state_dict(weights['behavior_state'])
        org.behavior_1.load_state_dict(weights['behavior_state'])
    except:
        pass

    # FORRAJEO: organismos en esquinas
    org.initialize_foraging()

    initial = org.get_metrics()
    initial_in_patch = org.count_cells_in_patches()
    initial_dist_0 = org.get_distance_to_nearest_patch(0)
    initial_dist_1 = org.get_distance_to_nearest_patch(1)

    print(f'Estado inicial:')
    print(f'  Org 0: {initial["org_0"]["n_total"]} celulas, dist a recurso: {initial_dist_0:.1f}')
    print(f'  Org 1: {initial["org_1"]["n_total"]} celulas, dist a recurso: {initial_dist_1:.1f}')
    print(f'  Células en recursos: Org0={initial_in_patch[0]}, Org1={initial_in_patch[1]}')

    history = {
        'cells_in_patch_0': [],
        'cells_in_patch_1': [],
        'dist_to_patch_0': [],
        'dist_to_patch_1': [],
        'attraction_0': [], 'attraction_1': [],
        'steps': []
    }

    for step in range(n_steps):
        org.step()
        m = org.get_metrics()

        if step % 5 == 0:
            in_patch = org.count_cells_in_patches()
            dist_0 = org.get_distance_to_nearest_patch(0)
            dist_1 = org.get_distance_to_nearest_patch(1)

            history['cells_in_patch_0'].append(in_patch[0])
            history['cells_in_patch_1'].append(in_patch[1])
            history['dist_to_patch_0'].append(dist_0)
            history['dist_to_patch_1'].append(dist_1)
            history['attraction_0'].append(m['org_0_attraction'])
            history['attraction_1'].append(m['org_1_attraction'])
            history['steps'].append(step)

        if (step + 1) % 150 == 0:
            in_patch = org.count_cells_in_patches()
            dist_0 = org.get_distance_to_nearest_patch(0)
            dist_1 = org.get_distance_to_nearest_patch(1)
            print(f'  Step {step+1}: En recurso={in_patch[0]}+{in_patch[1]}, '
                  f'Dist={dist_0:.1f}/{dist_1:.1f}, '
                  f'Attract={m["org_0_attraction"]:.1f}/{m["org_1_attraction"]:.1f}')

    final_in_patch = org.count_cells_in_patches()
    final_dist_0 = org.get_distance_to_nearest_patch(0)
    final_dist_1 = org.get_distance_to_nearest_patch(1)

    print(f'\nFinal:')
    print(f'  Células en recursos: Org0={final_in_patch[0]}, Org1={final_in_patch[1]}')
    print(f'  Distancia a recurso: Org0={final_dist_0:.1f}, Org1={final_dist_1:.1f}')

    return {
        'org': org,
        'name': name,
        'history': history,
        'pheromones_enabled': pheromones_enabled,
        'final_in_patch': final_in_patch,
        'final_dist': (final_dist_0, final_dist_1),
    }


def run_full_experiment():
    """Ejecuta experimento completo: con y sin feromonas."""
    print('='*70)
    print('EXPERIMENTO: COMUNICACION QUIMICA')
    print('Comparacion: Con feromonas vs Sin feromonas')
    print('='*70)

    # Escenarios normales (separados)
    results = {
        'sin_feromonas': run_scenario('Sin Feromonas (Baseline)', False, 600),
        'con_feromonas': run_scenario('Con Feromonas', True, 600),
    }

    # Escenarios de conflicto (cercanos)
    print('\n' + '='*70)
    print('EXPERIMENTO: COMUNICACION QUIMICA - ZONA DE CONFLICTO')
    print('Organismos inicialmente cercanos para maximizar interaccion')
    print('='*70)

    results['conflicto_sin'] = run_conflict_scenario('Conflicto Sin Feromonas', False, 600)
    results['conflicto_iso'] = run_conflict_scenario('Conflicto Alarma Isotrópica', True, 600, directed_alarm=False)
    results['conflicto_dir'] = run_conflict_scenario('Conflicto Alarma DIRIGIDA', True, 600, directed_alarm=True)

    # Escenarios de forrajeo (atracción)
    print('\n' + '='*70)
    print('EXPERIMENTO: COMUNICACION QUIMICA - FORRAJEO')
    print('Organismos en esquinas, recurso en centro - test de atracción')
    print('='*70)

    results['forrajeo_sin'] = run_foraging_scenario('Forrajeo Sin Feromonas', False, 600)
    results['forrajeo_con'] = run_foraging_scenario('Forrajeo Con Atracción', True, 600, attraction_weight=2.0)

    # === ANALISIS COMPARATIVO ===
    print('\n' + '='*70)
    print('ANALISIS COMPARATIVO - ESCENARIOS SEPARADOS')
    print('='*70)

    baseline = results['sin_feromonas']
    chemical = results['con_feromonas']

    # Promedios de ultimos 200 steps
    def avg_last(history, key, n=40):
        return np.mean(history[key][-n:]) if len(history[key]) >= n else np.mean(history[key])

    print(f'\n{"Metrica":<25} {"Sin Feromonas":<15} {"Con Feromonas":<15} {"Diferencia":<15}')
    print('-'*70)

    # Distancia entre centroides
    dist_baseline = avg_last(baseline['history'], 'centroid_distance')
    dist_chemical = avg_last(chemical['history'], 'centroid_distance')
    diff_dist = dist_chemical - dist_baseline
    print(f'{"Distancia centroides":<25} {dist_baseline:<15.1f} {dist_chemical:<15.1f} {diff_dist:+.1f}')

    # Cruces de frontera
    cross_baseline = avg_last(baseline['history'], 'boundary_crossings')
    cross_chemical = avg_last(chemical['history'], 'boundary_crossings')
    diff_cross = cross_chemical - cross_baseline
    print(f'{"Cruces frontera":<25} {cross_baseline:<15.1f} {cross_chemical:<15.1f} {diff_cross:+.1f}')

    # Analisis conflicto
    print('\n' + '='*70)
    print('ANALISIS COMPARATIVO - ZONA DE CONFLICTO')
    print('='*70)

    conf_base = results['conflicto_sin']
    conf_iso = results['conflicto_iso']
    conf_dir = results['conflicto_dir']

    print(f'\n{"Metrica":<20} {"Sin Ferom":<12} {"Isotrópica":<12} {"Dirigida":<12} {"Iso-Dir":<10}')
    print('-'*70)

    # Distancia conflicto
    dist_conf_base = avg_last(conf_base['history'], 'centroid_distance')
    dist_conf_iso = avg_last(conf_iso['history'], 'centroid_distance')
    dist_conf_dir = avg_last(conf_dir['history'], 'centroid_distance')
    diff_iso_dir = dist_conf_dir - dist_conf_iso
    print(f'{"Distancia centroides":<20} {dist_conf_base:<12.1f} {dist_conf_iso:<12.1f} {dist_conf_dir:<12.1f} {diff_iso_dir:+.1f}')

    # Cruces conflicto
    cross_conf_base = avg_last(conf_base['history'], 'boundary_crossings')
    cross_conf_iso = avg_last(conf_iso['history'], 'boundary_crossings')
    cross_conf_dir = avg_last(conf_dir['history'], 'boundary_crossings')
    diff_cross_iso_dir = cross_conf_dir - cross_conf_iso
    print(f'{"Cruces frontera":<20} {cross_conf_base:<12.1f} {cross_conf_iso:<12.1f} {cross_conf_dir:<12.1f} {diff_cross_iso_dir:+.1f}')

    # Alarmas en conflicto
    alarm_conf_base = sum(conf_base['history']['alarm_0']) + sum(conf_base['history']['alarm_1'])
    alarm_conf_iso = sum(conf_iso['history']['alarm_0']) + sum(conf_iso['history']['alarm_1'])
    alarm_conf_dir = sum(conf_dir['history']['alarm_0']) + sum(conf_dir['history']['alarm_1'])
    print(f'{"Alarmas emitidas":<20} {alarm_conf_base:<12.0f} {alarm_conf_iso:<12.0f} {alarm_conf_dir:<12.0f}')

    # Analisis forrajeo
    print('\n' + '='*70)
    print('ANALISIS COMPARATIVO - FORRAJEO (ATRACCION)')
    print('='*70)

    forr_sin = results['forrajeo_sin']
    forr_con = results['forrajeo_con']

    print(f'\n{"Metrica":<25} {"Sin Feromonas":<15} {"Con Atracción":<15} {"Diferencia":<15}')
    print('-'*70)

    # Células en recursos
    cells_sin = forr_sin['final_in_patch'][0] + forr_sin['final_in_patch'][1]
    cells_con = forr_con['final_in_patch'][0] + forr_con['final_in_patch'][1]
    diff_cells = cells_con - cells_sin
    print(f'{"Células en recurso":<25} {cells_sin:<15} {cells_con:<15} {diff_cells:+}')

    # Distancia promedio a recurso
    dist_sin = (forr_sin['final_dist'][0] + forr_sin['final_dist'][1]) / 2
    dist_con = (forr_con['final_dist'][0] + forr_con['final_dist'][1]) / 2
    diff_dist = dist_con - dist_sin
    print(f'{"Dist promedio a recurso":<25} {dist_sin:<15.1f} {dist_con:<15.1f} {diff_dist:+.1f}')

    # Atracción emitida
    attr_sin = sum(forr_sin['history']['attraction_0']) + sum(forr_sin['history']['attraction_1'])
    attr_con = sum(forr_con['history']['attraction_0']) + sum(forr_con['history']['attraction_1'])
    print(f'{"Atracción emitida":<25} {attr_sin:<15.0f} {attr_con:<15.0f}')

    # === VISUALIZACION ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Fila 1: Comparacion temporal
    # Distancia centroides
    ax = axes[0, 0]
    ax.plot(baseline['history']['steps'], baseline['history']['centroid_distance'],
            'b-', linewidth=2, label='Sin feromonas')
    ax.plot(chemical['history']['steps'], chemical['history']['centroid_distance'],
            'r-', linewidth=2, label='Con feromonas')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Distancia')
    ax.set_title('Distancia entre Centroides')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cruces de frontera
    ax = axes[0, 1]
    ax.plot(baseline['history']['steps'], baseline['history']['boundary_crossings'],
            'b-', linewidth=2, label='Sin feromonas')
    ax.plot(chemical['history']['steps'], chemical['history']['boundary_crossings'],
            'r-', linewidth=2, label='Con feromonas')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cruces')
    ax.set_title('Cruces de Frontera')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Concentracion de feromonas
    ax = axes[0, 2]
    ax.plot(chemical['history']['steps'], chemical['history']['alarm_0'],
            'r-', linewidth=1.5, label='Alarma Org 0')
    ax.plot(chemical['history']['steps'], chemical['history']['alarm_1'],
            'b-', linewidth=1.5, label='Alarma Org 1')
    ax.plot(chemical['history']['steps'], chemical['history']['territorial_0'],
            'r--', linewidth=1, alpha=0.7, label='Territorial Org 0')
    ax.plot(chemical['history']['steps'], chemical['history']['territorial_1'],
            'b--', linewidth=1, alpha=0.7, label='Territorial Org 1')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Concentracion')
    ax.set_title('Concentracion Feromonas')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Fila 2: Estados finales y mapas de feromonas
    colors = {0: 'crimson', 1: 'royalblue'}

    # Estado final sin feromonas
    ax = axes[1, 0]
    for patch in baseline['org'].patches:
        circle = plt.Circle(patch.position, patch.radius, color='green', alpha=0.15)
        ax.add_patch(circle)
    for cell in baseline['org'].cells:
        cx, cy = cell.position
        color = colors[cell.organism_id]
        marker = 's' if cell.role_idx == 1 else 'o'
        size = 50 if cell.role_idx == 1 else 15
        ax.scatter(cx, cy, c=color, s=size, marker=marker, alpha=0.7)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Final: Sin Feromonas')
    ax.set_aspect('equal')

    # Estado final con feromonas
    ax = axes[1, 1]
    for patch in chemical['org'].patches:
        circle = plt.Circle(patch.position, patch.radius, color='green', alpha=0.15)
        ax.add_patch(circle)
    for cell in chemical['org'].cells:
        cx, cy = cell.position
        color = colors[cell.organism_id]
        marker = 's' if cell.role_idx == 1 else 'o'
        size = 50 if cell.role_idx == 1 else 15
        ax.scatter(cx, cy, c=color, s=size, marker=marker, alpha=0.7)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_title('Estado Final: Con Feromonas')
    ax.set_aspect('equal')

    # Mapa de feromonas territoriales
    ax = axes[1, 2]
    # Combinar territoriales de ambos organismos
    terr_0 = chemical['org'].pheromones[0].territorial
    terr_1 = chemical['org'].pheromones[1].territorial
    # Mostrar como RGB (rojo = org0, azul = org1)
    combined = np.zeros((64, 64, 3))
    combined[:, :, 0] = terr_0 / (terr_0.max() + 0.01)  # Rojo
    combined[:, :, 2] = terr_1 / (terr_1.max() + 0.01)  # Azul
    ax.imshow(combined, origin='lower', extent=[0, 64, 0, 64])
    ax.set_title('Mapa Territorial (Rojo=Org0, Azul=Org1)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('zeta_organism_comunicacion_quimica.png', dpi=150)
    print('\nGuardado: zeta_organism_comunicacion_quimica.png')

    # === CONCLUSIONES ===
    print('\n' + '='*70)
    print('CONCLUSIONES')
    print('='*70)

    print('\n--- Escenarios Separados ---')
    # Evaluar efectos
    territorial_effect = diff_dist > 2  # Mayor separacion
    boundary_effect = diff_cross < -2   # Menos cruces

    if territorial_effect:
        print('[TERRITORIAL] Feromonas aumentan separacion entre organismos')
    else:
        print('[--] Separados: Feromonas no afectan significativamente la separacion')

    if boundary_effect:
        print('[FRONTERA] Feromonas reducen cruces de territorio')
    else:
        print('[--] Separados: Feromonas no afectan significativamente los cruces')

    # Verificar uso de feromonas
    total_alarm = sum(chemical['history']['alarm_0']) + sum(chemical['history']['alarm_1'])
    total_terr = sum(chemical['history']['territorial_0']) + sum(chemical['history']['territorial_1'])

    if total_alarm > 100:
        print(f'[ALARMA] Sistema de alarma activo (total: {total_alarm:.0f})')

    if total_terr > 500:
        print(f'[TERRITORIAL] Marcaje territorial activo (total: {total_terr:.0f})')

    print('\n--- Zona de Conflicto: Isotrópica vs Dirigida ---')

    # Comparar isotrópica vs dirigida
    if diff_cross_iso_dir < -5:
        print(f'[DIRIGIDA MEJOR] Alarma dirigida reduce cruces ({diff_cross_iso_dir:+.1f} vs isotrópica)')
    elif diff_cross_iso_dir > 5:
        print(f'[ISOTRÓPICA MEJOR] Alarma isotrópica tiene menos cruces ({-diff_cross_iso_dir:+.1f})')
    else:
        print(f'[SIMILAR] Ambos modos tienen cruces similares (dif={diff_cross_iso_dir:+.1f})')

    if diff_iso_dir > 3:
        print(f'[SEPARACIÓN] Alarma dirigida aumenta separación (+{diff_iso_dir:.1f} unidades)')
    elif diff_iso_dir < -3:
        print(f'[AGRUPACIÓN] Alarma dirigida reduce separación ({diff_iso_dir:.1f} unidades)')

    # Efectos generales
    if cross_conf_iso > 10:
        print(f'[PÁNICO ISO] Alarma isotrópica causa pánico ({cross_conf_iso:.0f} cruces)')
    if cross_conf_dir < cross_conf_iso * 0.5:
        print(f'[COORDINACIÓN] Alarma dirigida reduce caos ({cross_conf_dir:.0f} vs {cross_conf_iso:.0f} cruces)')

    print('\n--- Forrajeo: Atracción hacia Recursos ---')

    # Evaluar efectos de atracción
    if diff_cells > 5:
        print(f'[ATRACCIÓN EFECTIVA] Feromonas atraen +{diff_cells} células a recursos')
    elif diff_cells > 0:
        print(f'[ATRACCIÓN LEVE] Feromonas atraen +{diff_cells} células a recursos')
    else:
        print(f'[SIN EFECTO] Feromonas no aumentan células en recursos ({diff_cells:+})')

    if diff_dist < -5:
        print(f'[ACERCAMIENTO] Organismos se acercan {-diff_dist:.1f} unidades a recursos')
    elif diff_dist > 5:
        print(f'[ALEJAMIENTO] Organismos se alejan {diff_dist:.1f} unidades de recursos')

    if attr_con > 1000:
        print(f'[ATRACCIÓN ACTIVA] Sistema de atracción funcionando ({attr_con:.0f} emisiones)')

    return results


if __name__ == '__main__':
    results = run_full_experiment()
