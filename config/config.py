"""Configuration system for the Zeta Life project."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML required. Install with: pip install pyyaml")


@dataclass
class ZetaConfig:
    M: int = 15
    sigma: float = 0.1
    kernel_radius: int = 2
    known_zeros: List[float] = field(default_factory=lambda: [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
        67.079811, 69.546402, 72.067158, 75.704691, 77.144840
    ])


@dataclass
class ConsciousnessConfig:
    n_cells: int = 64
    dream_frequency: int = 100
    dream_duration: int = 20
    warmup_steps: int = 20


@dataclass
class DecayConfig:
    enabled: bool = False
    base_rate: float = 0.005
    stress_rate: float = 0.02
    neglect_rate: float = 0.01
    neglect_threshold: int = 50


@dataclass
class OrganismConfig:
    grid_size: int = 64
    n_cells: int = 100
    state_dim: int = 32
    hidden_dim: int = 64
    fi_threshold: float = 0.7


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    seed: int = 42


@dataclass
class Config:
    zeta: ZetaConfig = field(default_factory=ZetaConfig)
    consciousness: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    organism: OrganismConfig = field(default_factory=OrganismConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data):
        config = cls()
        if data is None:
            return config
        if "zeta" in data:
            for k, v in data["zeta"].items():
                if hasattr(config.zeta, k):
                    setattr(config.zeta, k, v)
        if "consciousness" in data:
            for k, v in data["consciousness"].items():
                if hasattr(config.consciousness, k):
                    setattr(config.consciousness, k, v)
        if "decay" in data:
            for k, v in data["decay"].items():
                if hasattr(config.decay, k):
                    setattr(config.decay, k, v)
        if "organism" in data:
            for k, v in data["organism"].items():
                if hasattr(config.organism, k):
                    setattr(config.organism, k, v)
        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        return config

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


_config_cache = None
_default_config_path = Path(__file__).parent / "default.yaml"


def get_config(config_path=None, use_cache=True):
    global _config_cache
    if use_cache and _config_cache is not None:
        default_config = _config_cache
    else:
        if _default_config_path.exists():
            default_config = Config.from_yaml(str(_default_config_path))
        else:
            default_config = Config()
        if use_cache:
            _config_cache = default_config
    if config_path is None:
        return default_config
    return Config.from_yaml(config_path)


def get_zeta_zeros(M=None, config=None):
    if config is None:
        config = get_config()
    if M is None:
        M = config.zeta.M
    known = config.zeta.known_zeros
    if M <= len(known):
        return known[:M]
    import numpy as np
    extended = list(known)
    for k in range(len(known), M):
        n = k + 1
        extended.append(2 * np.pi * n / np.log(n + 2))
    return extended


if __name__ == "__main__":
    cfg = get_config()
    print("Zeta M:", cfg.zeta.M)
    print("Zeta sigma:", cfg.zeta.sigma)
    print("Decay enabled:", cfg.decay.enabled)
    print("Config loaded successfully!")