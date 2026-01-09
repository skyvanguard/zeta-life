"""
Zeta Neural Cellular Automata - Evolucion del Sistema

Combina:
- Kernel Zeta de Riemann (del paper de Francisco Ruiz)
- Neural Cellular Automata diferenciables (Growing NCA de Google)
- Capacidad de regeneracion y aprendizaje

El sistema aprende a:
1. Crecer desde una semilla
2. Regenerarse ante dano
3. Mantener patrones estables con memoria zeta
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional
import warnings

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch no disponible. Instalar con: pip install torch")

# mpmath para ceros exactos
try:
    from mpmath import zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def get_zeta_zeros(M: int) -> List[float]:
    """Obtiene los primeros M ceros de zeta."""
    if HAS_MPMATH:
        return [float(zetazero(k).imag) for k in range(1, M + 1)]
    else:
        known = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                 37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
                 52.970321, 56.446248, 59.347044, 60.831779, 65.112544]
        if M <= len(known):
            return known[:M]
        return known + [2 * np.pi * n / np.log(n + 2) for n in range(len(known) + 1, M + 1)]


if HAS_TORCH:

    class ZetaKernelConv(nn.Module):
        """
        Capa de convolucion con kernel basado en ceros de zeta.

        El kernel NO es aprendible - viene de los ceros de zeta.
        Actua como una "percepcion" estructurada del vecindario.
        """

        def __init__(self, in_channels: int, M: int = 15, R: int = 2, sigma: float = 0.1) -> None:
            super().__init__()
            self.M = M
            self.R = R
            self.sigma = sigma
            self.in_channels = in_channels

            # Construir kernel zeta
            gammas = get_zeta_zeros(M)
            size = 2 * R + 1

            kernel = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    x, y = i - R, j - R
                    r = np.sqrt(x**2 + y**2)
                    if r > 0:
                        for gamma in gammas:
                            kernel[i, j] += np.exp(-sigma * abs(gamma)) * np.cos(gamma * r)

            # Normalizar
            kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)

            # Registrar como buffer (no aprendible)
            kernel_tensor = torch.tensor(kernel, dtype=torch.float32)
            # Expandir para todos los canales de entrada
            kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)
            kernel_tensor = kernel_tensor.repeat(in_channels, 1, 1, 1)
            self.register_buffer('zeta_kernel', kernel_tensor)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Aplica convolucion con kernel zeta."""
            # x: (batch, channels, height, width)
            padding = self.R
            return F.conv2d(x, self.zeta_kernel, padding=padding, groups=self.in_channels)  # type: ignore[arg-type]


    class SobelFilter(nn.Module):
        """Filtros Sobel para detectar gradientes."""

        def __init__(self, in_channels: int) -> None:
            super().__init__()
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

            sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
            sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)
            self.in_channels = in_channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)  # type: ignore[arg-type]
            gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)  # type: ignore[arg-type]
            return torch.cat([gx, gy], dim=1)


    class ZetaNCA(nn.Module):
        """
        Neural Cellular Automata con percepcion basada en kernel zeta.

        Arquitectura:
        1. Percepcion: Kernel Zeta + Sobel
        2. Update Network: MLP que decide el cambio de estado
        3. Stochastic Update: Solo actualiza algunas celulas
        """

        def __init__(
            self,
            channels: int = 16,
            hidden_channels: int = 128,
            M: int = 15,
            R: int = 2,
            sigma: float = 0.1,
            fire_rate: float = 0.5
        ) -> None:
            super().__init__()
            self.channels = channels
            self.fire_rate = fire_rate

            # Canales: [RGBA (4)] + [hidden (12)]
            # Canal 0-3: visualizacion (RGBA)
            # Canal 4+: estado oculto

            # Percepcion con kernel zeta
            self.zeta_conv = ZetaKernelConv(channels, M=M, R=R, sigma=sigma)

            # Filtros Sobel
            self.sobel = SobelFilter(channels)

            # Total de features de percepcion:
            # - Estado original: channels
            # - Zeta perception: channels
            # - Sobel x,y: 2 * channels
            perception_channels = channels * 4

            # Update network (MLP)
            self.update_net = nn.Sequential(
                nn.Conv2d(perception_channels, hidden_channels, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, channels, 1, bias=False)
            )

            # Inicializar ultima capa con ceros (residual learning)
            nn.init.zeros_(self.update_net[-1].weight)

        def perceive(self, x: torch.Tensor) -> torch.Tensor:
            """
            Percepcion del vecindario usando kernel zeta y Sobel.
            """
            # Convolucion zeta
            zeta_perception = self.zeta_conv(x)

            # Gradientes Sobel
            sobel_perception = self.sobel(x)

            # Concatenar todas las percepciones
            return torch.cat([x, zeta_perception, sobel_perception], dim=1)

        def get_alive_mask(self, x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
            """
            Determina que celulas estan "vivas" basandose en el canal alpha.
            Una celula esta viva si tiene al menos un vecino con alpha > threshold.
            """
            # Canal alpha es el canal 3
            alpha = x[:, 3:4, :, :]

            # Max pooling para detectar vecinos vivos
            alive = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > threshold

            return alive.float()

        def forward(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
            """
            Ejecuta N pasos de actualizacion.
            """
            for _ in range(steps):
                x = self.step(x)
            return x

        def step(self, x: torch.Tensor) -> torch.Tensor:
            """
            Un paso de actualizacion del NCA.
            """
            # Mascara de celulas vivas (antes de actualizar)
            pre_alive = self.get_alive_mask(x)

            # Percepcion
            perception = self.perceive(x)

            # Calcular actualizacion
            update = self.update_net(perception)

            # Actualizacion estocastica (solo algunas celulas se actualizan)
            if self.training:
                mask = (torch.rand_like(x[:, :1, :, :]) < self.fire_rate).float()
                update = update * mask

            # Aplicar actualizacion (residual)
            x = x + update

            # Mascara de celulas vivas (despues de actualizar)
            post_alive = self.get_alive_mask(x)

            # Solo mantener celulas que estaban vivas o tienen vecinos vivos
            alive_mask = (pre_alive + post_alive) > 0
            x = x * alive_mask

            return x

        def seed(self, batch_size: int, height: int, width: int, device: str = 'cpu') -> torch.Tensor:
            """
            Crea una semilla inicial (un pixel en el centro).
            """
            x = torch.zeros(batch_size, self.channels, height, width, device=device)
            # Poner semilla en el centro
            cx, cy = height // 2, width // 2
            x[:, 3, cx, cy] = 1.0  # Alpha = 1
            return x


    class ZetaNCATrainer:
        """
        Entrenador para el Zeta NCA.

        Entrena el modelo para:
        1. Crecer hacia una imagen objetivo
        2. Regenerarse ante dano
        """

        def __init__(
            self,
            model: ZetaNCA,
            target: np.ndarray,
            device: str = 'cpu',
            lr: float = 2e-3
        ) -> None:
            self.model = model.to(device)
            self.device = device

            # Convertir target a tensor
            # Target debe ser (H, W, 4) con valores 0-1
            if target.ndim == 2:
                # Si es grayscale, convertir a RGBA
                target = np.stack([target, target, target, (target > 0.5).astype(float)], axis=-1)

            self.target = torch.tensor(target, dtype=torch.float32).to(device)
            self.target = self.target.permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)

            # Padding para el target (agregar canales hidden)
            hidden_pad = torch.zeros(1, model.channels - 4, target.shape[0], target.shape[1], device=device)
            self.target_padded = torch.cat([self.target, hidden_pad], dim=1)

            self.optimizer = Adam(model.parameters(), lr=lr)
            self.losses: List[float] = []

        def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
            """
            Perdida: MSE entre canales RGBA del estado y target.
            """
            # Expandir target para batch
            batch_size = x.shape[0]
            target_expanded = self.target[:, :4].expand(batch_size, -1, -1, -1)
            return F.mse_loss(x[:, :4], target_expanded)

        def train_step(
            self,
            batch_size: int = 8,
            min_steps: int = 64,
            max_steps: int = 96,
            damage_prob: float = 0.5
        ) -> float:
            """
            Un paso de entrenamiento.
            """
            self.model.train()
            self.optimizer.zero_grad()

            h, w = self.target.shape[2], self.target.shape[3]

            # Inicializar desde semilla
            x = self.model.seed(batch_size, h, w, self.device)

            # Numero aleatorio de pasos
            steps = np.random.randint(min_steps, max_steps)

            # Ejecutar pasos
            for step in range(steps):
                x = self.model.step(x)

                # Introducir dano aleatorio durante el entrenamiento
                if step > min_steps // 2 and np.random.random() < damage_prob:
                    # Dano circular aleatorio
                    cx = np.random.randint(0, h)
                    cy = np.random.randint(0, w)
                    r = np.random.randint(5, 15)

                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(h, device=self.device),
                        torch.arange(w, device=self.device),
                        indexing='ij'
                    )
                    damage_mask = ((x_coords - cy)**2 + (y_coords - cx)**2) < r**2
                    x[:, :, damage_mask] = 0

            # Calcular perdida
            loss = self.loss_fn(x)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())
            return loss.item()

        def train(
            self,
            iterations: int = 2000,
            batch_size: int = 8,
            log_every: int = 100
        ) -> None:
            """
            Entrena el modelo.
            """
            print(f"Entrenando Zeta NCA por {iterations} iteraciones...")

            for i in range(iterations):
                loss = self.train_step(batch_size=batch_size)

                if (i + 1) % log_every == 0:
                    print(f"  Iteracion {i+1}/{iterations}, Loss: {loss:.6f}")

            print("Entrenamiento completado.")

        def visualize_growth(self, steps: int = 100, save_path: Optional[str] = None):
            """
            Visualiza el crecimiento desde la semilla.
            """
            self.model.eval()
            h, w = self.target.shape[2], self.target.shape[3]

            with torch.no_grad():
                x = self.model.seed(1, h, w, self.device)

                states = [x.cpu().numpy()[0, :4].transpose(1, 2, 0)]

                for _ in range(steps):
                    x = self.model.step(x)
                    states.append(x.cpu().numpy()[0, :4].transpose(1, 2, 0))

            # Visualizar
            n_show = min(8, len(states))
            indices = np.linspace(0, len(states) - 1, n_show, dtype=int)

            fig, axes = plt.subplots(2, n_show // 2, figsize=(3 * n_show // 2, 6))
            axes = axes.flatten()

            for ax, idx in zip(axes, indices):
                state = states[idx]
                # Mostrar RGB con alpha como mascara
                rgb = state[:, :, :3]
                alpha = state[:, :, 3:4]
                display = rgb * alpha + (1 - alpha)  # Fondo blanco
                ax.imshow(np.clip(display, 0, 1))
                ax.set_title(f'Step {idx}')
                ax.axis('off')

            plt.suptitle('Crecimiento Zeta NCA')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Guardado: {save_path}")

            return fig

        def visualize_regeneration(self, damage_step: int = 50, total_steps: int = 150, save_path: Optional[str] = None):
            """
            Visualiza la regeneracion ante dano.
            """
            self.model.eval()
            h, w = self.target.shape[2], self.target.shape[3]

            with torch.no_grad():
                x = self.model.seed(1, h, w, self.device)

                states = []
                damage_applied = False

                for step in range(total_steps):
                    x = self.model.step(x)

                    # Aplicar dano en el paso indicado
                    if step == damage_step and not damage_applied:
                        states.append(('pre_damage', x.cpu().numpy()[0, :4].transpose(1, 2, 0)))

                        # Dano: eliminar mitad izquierda
                        x[:, :, :, :w//2] = 0
                        damage_applied = True

                        states.append(('post_damage', x.cpu().numpy()[0, :4].transpose(1, 2, 0)))

                    elif step in [damage_step + 10, damage_step + 30, damage_step + 50, damage_step + 80]:
                        states.append((f'regen_{step - damage_step}', x.cpu().numpy()[0, :4].transpose(1, 2, 0)))

            # Visualizar
            fig, axes = plt.subplots(1, len(states), figsize=(3 * len(states), 3))

            for ax, (label, state) in zip(axes, states):
                rgb = state[:, :, :3]
                alpha = state[:, :, 3:4]
                display = rgb * alpha + (1 - alpha)
                ax.imshow(np.clip(display, 0, 1))
                ax.set_title(label)
                ax.axis('off')

            plt.suptitle('Regeneracion Zeta NCA')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Guardado: {save_path}")

            return fig


def create_simple_target(size: int = 64) -> "np.ndarray":
    """
    Crea un target simple (circulo con patron).
    """
    target = np.zeros((size, size, 4))

    cx, cy = size // 2, size // 2
    r = size // 3

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - cx)**2 + (j - cy)**2)
            if dist < r:
                # Color basado en posicion
                target[i, j, 0] = 0.2 + 0.8 * (i / size)  # R
                target[i, j, 1] = 0.8 - 0.6 * (j / size)  # G
                target[i, j, 2] = 0.3  # B
                target[i, j, 3] = 1.0  # Alpha

    return target


def create_zeta_pattern_target(size: int = 64, M: int = 10, sigma: float = 0.1) -> "np.ndarray":
    """
    Crea un target con patron basado en kernel zeta.
    """
    gammas = get_zeta_zeros(M)
    target = np.zeros((size, size, 4))

    cx, cy = size // 2, size // 2

    for i in range(size):
        for j in range(size):
            x, y = i - cx, j - cy
            r = np.sqrt(x**2 + y**2)

            if r < size // 2.5:
                value = 0
                for gamma in gammas:
                    value += np.exp(-sigma * abs(gamma)) * np.cos(gamma * r / 5)

                value = (value - np.min(value)) / (np.max(value) - np.min(value) + 1e-8)

                target[i, j, 0] = 0.1 + 0.6 * value  # R
                target[i, j, 1] = 0.8 * value  # G
                target[i, j, 2] = 0.2 + 0.3 * (1 - value)  # B
                target[i, j, 3] = 1.0 if r < size // 3 else max(0, 1 - (r - size//3) / 10)

    return target


def demo_zeta_nca() -> Optional[Tuple["ZetaNCA", "ZetaNCATrainer"]]:
    """
    Demostracion del Zeta Neural CA.
    """
    print("=" * 70)
    print("ZETA NEURAL CELLULAR AUTOMATA")
    print("Evolucion del sistema: NCA diferenciable con kernel zeta")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nERROR: PyTorch no esta instalado.")
        print("Instalar con: pip install torch")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDispositivo: {device}")

    # 1. Crear target con patron zeta
    print("\n1. Creando target con patron zeta...")
    target = create_zeta_pattern_target(size=64, M=15, sigma=0.1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(target[:, :, :3] * target[:, :, 3:4] + (1 - target[:, :, 3:4]))
    ax.set_title('Target: Patron Zeta')
    ax.axis('off')
    fig.savefig('zeta_nca_target.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_nca_target.png")
    plt.close()

    # 2. Crear modelo
    print("\n2. Creando modelo Zeta NCA...")
    model = ZetaNCA(
        channels=16,
        hidden_channels=96,
        M=15,
        R=2,
        sigma=0.1,
        fire_rate=0.5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametros totales: {total_params:,}")

    # 3. Crear entrenador
    print("\n3. Inicializando entrenador...")
    trainer = ZetaNCATrainer(model, target, device=device, lr=2e-3)

    # 4. Entrenar (pocas iteraciones para demo rapida)
    print("\n4. Entrenando modelo (200 iteraciones para demo rapida)...")
    trainer.train(iterations=200, batch_size=4, log_every=50)

    # 5. Visualizar crecimiento
    print("\n5. Visualizando crecimiento...")
    fig1 = trainer.visualize_growth(steps=100, save_path='zeta_nca_growth.png')
    plt.close()

    # 6. Visualizar regeneracion
    print("\n6. Visualizando regeneracion...")
    fig2 = trainer.visualize_regeneration(
        damage_step=50,
        total_steps=150,
        save_path='zeta_nca_regeneration.png'
    )
    plt.close()

    # 7. Grafica de perdida
    print("\n7. Generando grafica de perdida...")
    fig3, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trainer.losses)
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Curva de Entrenamiento Zeta NCA')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig3.savefig('zeta_nca_loss.png', dpi=150, bbox_inches='tight')
    print("   Guardado: zeta_nca_loss.png")
    plt.close()

    print("\n" + "=" * 70)
    print("Demo Zeta NCA completada.")
    print("=" * 70)

    return model, trainer


if __name__ == "__main__":
    demo_zeta_nca()
