# train_organism.py
"""Entrenamiento de redes neuronales del ZetaOrganism.

Enfoque: Aprendizaje auto-supervisado basado en dinámica emergente.
- BehaviorEngine: aprende a predecir influencia y transformaciones
- OrganismCell: aprende a detectar roles y actualizar estados

Objetivos de entrenamiento:
1. Predicción de rol: ¿quién se convertirá en Fi?
2. Maximizar coordinación: masas cerca de Fi
3. Estabilidad: minimizar cambios bruscos
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from zeta_life.organism import ZetaOrganism, CellEntity
from collections import deque


class OrganismTrainer:
    """Entrenador para las redes del ZetaOrganism."""

    def __init__(self, organism: ZetaOrganism, lr: float = 1e-3):
        self.org = organism

        # Optimizadores separados
        self.opt_behavior = optim.Adam(organism.behavior.parameters(), lr=lr)
        self.opt_cell = optim.Adam(organism.cell_module.parameters(), lr=lr)

        # Historial de entrenamiento
        self.losses = []
        self.metrics_history = []

        # Buffer de experiencias
        self.experience_buffer = deque(maxlen=1000)

    def collect_experience(self, n_steps: int = 20):
        """Recolecta experiencias de simulación."""
        experiences = []

        for _ in range(n_steps):
            # Guardar estado antes
            states_before = [(c.state.clone(), c.role.clone(), c.energy)
                           for c in self.org.cells]

            # Ejecutar paso
            self.org.step()

            # Guardar estado después
            states_after = [(c.state.clone(), c.role.clone(), c.energy)
                          for c in self.org.cells]

            # Métricas
            metrics = self.org.get_metrics()

            experiences.append({
                'before': states_before,
                'after': states_after,
                'metrics': metrics
            })

        self.experience_buffer.extend(experiences)
        return experiences

    def compute_role_prediction_loss(self):
        """Loss para predicción de rol.

        Objetivo: OrganismCell.role_detector debe predecir
        el rol actual basado en el estado.
        """
        loss = torch.tensor(0.0)
        count = 0

        for cell in self.org.cells:
            # Predicción del detector
            state = cell.state.unsqueeze(0)
            predicted_role = self.org.cell_module.detect_role(state)

            # Target: rol actual
            target_role = cell.role.unsqueeze(0)

            # Cross-entropy loss
            loss = loss + F.mse_loss(predicted_role, target_role)
            count += 1

        return loss / max(count, 1)

    def compute_influence_consistency_loss(self):
        """Loss para consistencia de influencia.

        Objetivo: Fi debe tener influencia_out > influencia_in
                  Mass debe tener influencia_in > influencia_out
        """
        loss = torch.tensor(0.0)
        count = 0

        for cell in self.org.cells:
            neighbors = self.org._get_neighbors(cell, radius=5)
            if not neighbors:
                continue

            neighbor_states = torch.stack([n.state for n in neighbors])
            influence_out, influence_in = self.org.behavior.bidirectional_influence(
                cell.state, neighbor_states
            )

            net_influence = influence_out.mean() - influence_in

            # Fi debe tener net_influence positivo
            if cell.role_idx == 1:  # Fi
                # Penalizar si net_influence es negativo
                loss = loss + F.relu(-net_influence)
            else:  # Mass
                # Penalizar si net_influence es muy positivo
                loss = loss + F.relu(net_influence - 0.3)

            count += 1

        return loss / max(count, 1)

    def compute_coordination_loss(self):
        """Loss para maximizar coordinación.

        Objetivo: Masas deben acercarse a Fi.
        """
        fi_cells = [c for c in self.org.cells if c.role_idx == 1]
        mass_cells = [c for c in self.org.cells if c.role_idx == 0]

        if not fi_cells or not mass_cells:
            return torch.tensor(0.0)

        # Calcular distancia promedio de masas a Fi más cercano
        total_dist = 0
        for m in mass_cells:
            min_dist = min(
                np.sqrt((m.position[0] - f.position[0])**2 +
                       (m.position[1] - f.position[1])**2)
                for f in fi_cells
            )
            total_dist += min_dist

        avg_dist = total_dist / len(mass_cells)

        # Normalizar por tamaño del grid
        normalized_dist = avg_dist / self.org.grid_size

        return torch.tensor(normalized_dist)

    def compute_stability_loss(self):
        """Loss para estabilidad.

        Objetivo: Minimizar cambios bruscos de estado.
        """
        if len(self.experience_buffer) < 2:
            return torch.tensor(0.0)

        # Tomar última experiencia
        exp = self.experience_buffer[-1]

        total_change = 0
        for (state_b, role_b, _), (state_a, role_a, _) in zip(
            exp['before'], exp['after']
        ):
            # Cambio de estado
            state_change = (state_a - state_b).norm().item()
            # Cambio de rol
            role_change = (role_a - role_b).norm().item()

            total_change += state_change + role_change

        avg_change = total_change / len(exp['before'])

        return torch.tensor(avg_change)

    def compute_emergence_reward(self):
        """Recompensa por emergencia de Fi.

        Objetivo: Recompensar cuando emergen nuevos Fi.
        """
        if len(self.experience_buffer) < 2:
            return torch.tensor(0.0)

        exp_old = self.experience_buffer[-2]
        exp_new = self.experience_buffer[-1]

        fi_old = exp_old['metrics']['n_fi']
        fi_new = exp_new['metrics']['n_fi']

        # Recompensar emergencia (pero no demasiados Fi)
        target_fi = max(5, self.org.n_cells // 10)  # ~10% Fi ideal

        # Penalizar desviación del target
        deviation = abs(fi_new - target_fi)

        return torch.tensor(deviation / target_fi)

    def train_step(self):
        """Un paso de entrenamiento."""
        # Recolectar experiencias
        self.collect_experience(n_steps=5)

        # Calcular losses
        loss_role = self.compute_role_prediction_loss()
        loss_influence = self.compute_influence_consistency_loss()
        loss_coord = self.compute_coordination_loss()
        loss_stability = self.compute_stability_loss()
        loss_emergence = self.compute_emergence_reward()

        # Loss total ponderado
        total_loss = (
            1.0 * loss_role +
            0.5 * loss_influence +
            0.3 * loss_coord +
            0.1 * loss_stability +
            0.2 * loss_emergence
        )

        # Backprop
        self.opt_behavior.zero_grad()
        self.opt_cell.zero_grad()

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.org.behavior.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.org.cell_module.parameters(), 1.0)

        self.opt_behavior.step()
        self.opt_cell.step()

        # Guardar métricas
        self.losses.append({
            'total': total_loss.item(),
            'role': loss_role.item(),
            'influence': loss_influence.item(),
            'coord': loss_coord.item(),
            'stability': loss_stability.item(),
            'emergence': loss_emergence.item()
        })

        metrics = self.org.get_metrics()
        self.metrics_history.append(metrics)

        return total_loss.item()

    def train(self, n_epochs: int = 50, steps_per_epoch: int = 10):
        """Entrenamiento completo."""
        print('='*60)
        print('Entrenamiento de Redes Neuronales - ZetaOrganism')
        print('='*60)

        for epoch in range(n_epochs):
            epoch_losses = []

            # Reset organismo cada época
            self.org.initialize(seed_fi=True)

            for step in range(steps_per_epoch):
                loss = self.train_step()
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            metrics = self.org.get_metrics()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, '
                      f'Fi={metrics["n_fi"]}, Coord={metrics["coordination"]:.3f}')

        print('='*60)
        print('Entrenamiento completado!')
        return self.losses, self.metrics_history


def run_training():
    """Ejecuta entrenamiento y evaluación."""
    print('Inicializando...')

    torch.manual_seed(42)
    np.random.seed(42)

    # Crear organismo
    org = ZetaOrganism(
        grid_size=48,
        n_cells=80,
        state_dim=32,
        hidden_dim=64,
        fi_threshold=0.5
    )
    org.initialize(seed_fi=True)

    # Crear entrenador
    trainer = OrganismTrainer(org, lr=5e-4)

    # Entrenar
    losses, metrics = trainer.train(n_epochs=30, steps_per_epoch=15)

    # Evaluación post-entrenamiento
    print('\n' + '='*60)
    print('EVALUACION POST-ENTRENAMIENTO')
    print('='*60)

    org.initialize(seed_fi=True)
    org.history = []

    print('\nSimulando 200 steps con redes entrenadas...')
    for step in range(200):
        org.step()
        if (step + 1) % 50 == 0:
            m = org.get_metrics()
            print(f'  Step {step+1}: Fi={m["n_fi"]}, Mass={m["n_mass"]}, '
                  f'Coord={m["coordination"]:.3f}, Stab={m["stability"]:.3f}')

    final = org.get_metrics()
    print(f'\nResultado final:')
    print(f'  Fi: {final["n_fi"]}')
    print(f'  Mass: {final["n_mass"]}')
    print(f'  Coordinación: {final["coordination"]:.3f}')
    print(f'  Estabilidad: {final["stability"]:.3f}')

    # Visualización
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Loss de entrenamiento
    ax = axes[0, 0]
    loss_values = [l['total'] for l in losses]
    ax.plot(loss_values, 'b-', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss Total')
    ax.set_title('Curva de Entrenamiento')
    ax.grid(True, alpha=0.3)

    # 2. Componentes del loss
    ax = axes[0, 1]
    ax.plot([l['role'] for l in losses], label='Role', alpha=0.7)
    ax.plot([l['influence'] for l in losses], label='Influence', alpha=0.7)
    ax.plot([l['coord'] for l in losses], label='Coord', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Componentes del Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Evolución de Fi durante entrenamiento
    ax = axes[0, 2]
    fi_values = [m['n_fi'] for m in metrics]
    ax.plot(fi_values, 'r-', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cantidad Fi')
    ax.set_title('Emergencia de Fi (Entrenamiento)')
    ax.grid(True, alpha=0.3)

    # 4. Evolución de roles (evaluación)
    ax = axes[1, 0]
    history = org.history
    steps = range(len(history))
    ax.plot(steps, [h['n_fi'] for h in history], 'r-', label='Fi', linewidth=2)
    ax.plot(steps, [h['n_mass'] for h in history], 'b-', label='Mass', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cantidad')
    ax.set_title('Evolución de Roles (Evaluación)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Métricas de inteligencia (evaluación)
    ax = axes[1, 1]
    ax.plot(steps, [h['coordination'] for h in history], 'g-', label='Coordinación', linewidth=2)
    ax.plot(steps, [h['stability'] for h in history], 'm-', label='Estabilidad', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Valor')
    ax.set_title('Métricas de Inteligencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 6. Estado final
    ax = axes[1, 2]
    for cell in org.cells:
        x, y = cell.position
        color = ['blue', 'red', 'black'][cell.role_idx]
        size = 20 + cell.energy * 80
        ax.scatter(x, y, c=color, s=size, alpha=0.7)
    ax.set_xlim(0, org.grid_size)
    ax.set_ylim(0, org.grid_size)
    ax.set_title('Estado Final (rojo=Fi, azul=Mass)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('zeta_organism_trained.png', dpi=150)
    print('\nGuardado: zeta_organism_trained.png')

    # Guardar modelo entrenado
    torch.save({
        'behavior_state': org.behavior.state_dict(),
        'cell_module_state': org.cell_module.state_dict(),
    }, 'zeta_organism_weights.pt')
    print('Pesos guardados: zeta_organism_weights.pt')

    return org, trainer


if __name__ == '__main__':
    run_training()
