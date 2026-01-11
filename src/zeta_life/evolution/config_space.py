"""
Configuration Space for IPUESA Evolution

Defines the 30 evolvable parameters with their valid ranges
and the EvolvableConfig dataclass for type-safe configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple


# Parameter ranges: (min, max) for each evolvable parameter
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    # Group A: Damage and Degradation (10 params)
    'damage_multiplier': (1.5, 5.0),
    'base_degrad_rate': (0.05, 0.30),
    'embedding_protection': (0.05, 0.40),
    'stance_protection': (0.05, 0.25),
    'compound_factor': (0.2, 0.8),
    'module_protection': (0.03, 0.20),
    'resilience_min': (0.1, 0.5),
    'resilience_range': (0.8, 2.0),
    'noise_scale': (0.10, 0.40),
    'residual_cap': (0.20, 0.50),

    # Group B: Recovery (6 params)
    'base_recovery_rate': (0.03, 0.12),
    'embedding_bonus': (0.3, 0.9),
    'cluster_bonus': (0.1, 0.5),
    'degradation_penalty': (0.2, 0.6),
    'degrad_recovery_factor': (0.990, 0.999),
    'corruption_decay': (0.90, 0.98),

    # Group C: Module Effects (8 params)
    'effect_pattern_detector': (0.10, 0.35),
    'effect_threat_filter': (0.10, 0.30),
    'effect_recovery_accelerator': (0.15, 0.40),
    'effect_exploration_dampener': (0.08, 0.25),
    'effect_embedding_protector': (0.15, 0.45),
    'effect_cascade_breaker': (0.12, 0.35),
    'effect_residual_cleaner': (0.10, 0.35),
    'effect_anticipation_enhancer': (0.15, 0.40),

    # Group D: Thresholds and Spreading (6 params)
    'consolidation_threshold': (0.05, 0.25),
    'spread_threshold': (0.08, 0.30),
    'spread_probability': (0.15, 0.50),
    'spread_strength_factor': (0.30, 0.70),
    'module_cap': (4, 10),
    'min_activations': (2, 6),
}


@dataclass
class EvolvableConfig:
    """
    Configuration for IPUESA simulation with all evolvable parameters.

    Default values are from IPUESA-SYNTH-v2 which achieved 8/8 criteria.
    """

    # Group A: Damage and Degradation
    damage_multiplier: float = 3.9
    base_degrad_rate: float = 0.18
    embedding_protection: float = 0.15
    stance_protection: float = 0.12
    compound_factor: float = 0.5
    module_protection: float = 0.08
    resilience_min: float = 0.3
    resilience_range: float = 1.4
    noise_scale: float = 0.25
    residual_cap: float = 0.35

    # Group B: Recovery
    base_recovery_rate: float = 0.06
    embedding_bonus: float = 0.6
    cluster_bonus: float = 0.3
    degradation_penalty: float = 0.4
    degrad_recovery_factor: float = 0.998
    corruption_decay: float = 0.94

    # Group C: Module Effects
    effect_pattern_detector: float = 0.20
    effect_threat_filter: float = 0.18
    effect_recovery_accelerator: float = 0.25
    effect_exploration_dampener: float = 0.15
    effect_embedding_protector: float = 0.30
    effect_cascade_breaker: float = 0.22
    effect_residual_cleaner: float = 0.20
    effect_anticipation_enhancer: float = 0.25

    # Group D: Thresholds and Spreading
    consolidation_threshold: float = 0.10
    spread_threshold: float = 0.15
    spread_probability: float = 0.30
    spread_strength_factor: float = 0.45
    module_cap: int = 6
    min_activations: int = 3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvolvableConfig':
        """Create config from dictionary (e.g., from OpenAlpha output)."""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}

        # Handle int params
        for int_param in ['module_cap', 'min_activations']:
            if int_param in filtered:
                filtered[int_param] = int(round(filtered[int_param]))

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_module_effects(self) -> Dict[str, float]:
        """Get module effects as dictionary for MicroModule."""
        return {
            'pattern_detector': self.effect_pattern_detector,
            'threat_filter': self.effect_threat_filter,
            'recovery_accelerator': self.effect_recovery_accelerator,
            'exploration_dampener': self.effect_exploration_dampener,
            'embedding_protector': self.effect_embedding_protector,
            'cascade_breaker': self.effect_cascade_breaker,
            'residual_cleaner': self.effect_residual_cleaner,
            'anticipation_enhancer': self.effect_anticipation_enhancer,
        }

    def validate(self) -> Tuple[bool, str]:
        """Validate that all parameters are within valid ranges."""
        for key, (min_val, max_val) in PARAM_RANGES.items():
            value = getattr(self, key, None)
            if value is None:
                return False, f"Missing parameter: {key}"
            if not (min_val <= value <= max_val):
                return False, f"{key}={value} outside range [{min_val}, {max_val}]"
        return True, "OK"

    def clamp_to_ranges(self) -> 'EvolvableConfig':
        """Return a new config with all values clamped to valid ranges."""
        d = self.to_dict()
        for key, (min_val, max_val) in PARAM_RANGES.items():
            if key in d:
                d[key] = max(min_val, min(max_val, d[key]))
        return EvolvableConfig.from_dict(d)


def get_baseline_config() -> EvolvableConfig:
    """Get the baseline configuration (SYNTH-v2 defaults)."""
    return EvolvableConfig()


def get_config_as_flat_dict(config: EvolvableConfig) -> Dict[str, float]:
    """Get only the evolvable parameters as a flat dict."""
    d = config.to_dict()
    return {k: float(v) for k, v in d.items() if k in PARAM_RANGES}
