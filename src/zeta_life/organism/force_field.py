# force_field.py
"""Campo de fuerzas con propagacion zeta."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_zeta_zeros(M: int) -> list[float]:
    """Primeros M ceros no triviales de zeta."""
    zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
             37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
             52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
             67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
             79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
             92.491899, 94.651344, 95.870634, 98.831194, 101.317851]
    return zeros[:M]

class ForceField(nn.Module):
    """Campo de fuerzas propagado por convolucion zeta."""

    def __init__(self, grid_size: int = 64, M: int = 15,
                 sigma: float = 0.1, kernel_radius: int = 7) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.M = M
        self.sigma = sigma
        self.kernel_radius = kernel_radius

        # Crear kernel zeta
        self.kernel = self._create_zeta_kernel()

        # Filtros Sobel para gradiente
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _create_zeta_kernel(self) -> nn.Parameter:
        """Crea kernel K_sigma(r) = Sum exp(-sigma|gamma|) * cos(gamma*r)."""
        gammas = get_zeta_zeros(self.M)
        size = 2 * self.kernel_radius + 1

        kernel = np.zeros((size, size))
        center = self.kernel_radius

        for i in range(size):
            for j in range(size):
                r = np.sqrt((i - center)**2 + (j - center)**2)
                for gamma in gammas:
                    weight = np.exp(-self.sigma * abs(gamma))
                    kernel[i, j] += weight * np.cos(gamma * r)

        # Normalizar
        kernel = kernel / (np.abs(kernel).sum() + 1e-8)

        return nn.Parameter(
            torch.tensor(kernel, dtype=torch.float32).view(1, 1, size, size),
            requires_grad=False
        )

    def compute(self, energy: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        """Computa campo de fuerzas.

        Args:
            energy: [B, 1, H, W] energia por celula
            roles: [B, 1, H, W] roles (1=FORCE emite)

        Returns:
            field: [B, 1, H, W] campo de fuerza propagado
        """
        # Solo Fi emiten
        fi_signal = energy * (roles == 1).float()

        # Propagar con kernel zeta
        pad = self.kernel_radius
        padded = F.pad(fi_signal, (pad, pad, pad, pad), mode='circular')
        field = F.conv2d(padded, self.kernel)

        return field

    def compute_with_gradient(self, energy: torch.Tensor,
                              roles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computa campo y gradiente.

        Returns:
            field: [B, 1, H, W]
            gradient: [B, 2, H, W] (dx, dy)
        """
        field = self.compute(energy, roles)

        # Gradiente con Sobel
        padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(padded, self.sobel_x)  # type: ignore[arg-type]
        grad_y = F.conv2d(padded, self.sobel_y)  # type: ignore[arg-type]

        gradient = torch.cat([grad_x, grad_y], dim=1)
        return field, gradient

    def attraction_force(self, position: tuple[int, int], field: torch.Tensor,
                        gradient: torch.Tensor) -> torch.Tensor:
        """Fuerza de atraccion en una posicion."""
        x, y = position
        return gradient[:, :, y, x]  # [B, 2]
