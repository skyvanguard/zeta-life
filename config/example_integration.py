"""
Example integration: How to use the configuration system in main modules.
"""

from config import get_config, get_zeta_zeros

cfg = get_config()
M = cfg.zeta.M
sigma = cfg.zeta.sigma
n_cells = cfg.organism.n_cells
zeros = get_zeta_zeros(M)


class ZetaKernelExample:
    def __init__(self, config=None):
        if config is None:
            config = get_config()
        self.M = config.zeta.M
        self.sigma = config.zeta.sigma
        self.gammas = get_zeta_zeros(self.M, config)

    def evaluate(self, t):
        import numpy as np
        result = 0.0
        for gamma in self.gammas:
            w = np.exp(-self.sigma * abs(gamma))
            result += w * np.cos(gamma * t)
        return 2 * result


def apply_decay(energy, config=None):
    if config is None:
        config = get_config()
    if not config.decay.enabled:
        return energy
    return energy * (1 - config.decay.base_rate)


if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Integration Examples")
    print("=" * 60)

    cfg = get_config()
    print()
    print("Default config loaded:")
    print("  Zeta M:", cfg.zeta.M)
    print("  Zeta sigma:", cfg.zeta.sigma)
    print("  Organism grid_size:", cfg.organism.grid_size)
    print("  Training seed:", cfg.training.seed)

    kernel = ZetaKernelExample()
    print()
    print("ZetaKernelExample:")
    print("  Using M=", kernel.M, "zeros")
    print("  K(0) =", round(kernel.evaluate(0), 4))
    print("  K(1) =", round(kernel.evaluate(1), 4))

    energy = 1.0
    new_energy = apply_decay(energy)
    print()
    print("Decay example:")
    print("  Decay enabled:", cfg.decay.enabled)
    print("  Energy before:", energy, "after:", new_energy)

    print()
    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
