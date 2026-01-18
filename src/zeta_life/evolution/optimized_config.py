"""
Optimized IPUESA Configuration

Evolved over 50 generations with fitness=0.9993, 8/8 criteria passed.
Found at generation 34 on 2026-01-11.

Key improvements over defaults:
- base_recovery_rate: +63.1% (faster recovery)
- spread_probability: +60.6% (easier module spreading)
- min_activations: +52.2% (more selective spreading)
- noise_scale: +48.9% (more variability/anti-fragility)
- damage_multiplier: +13.5% (higher stress, balanced by recovery)
- embedding_protection: -66.7% (less static, more dynamic protection)
- compound_factor: -50.7% (reduced damage cascades)
"""

from .config_space import EvolvableConfig

# Best evolved configuration (fitness=0.9993, 8/8 criteria)
OPTIMIZED_CONFIG = {
    # Damage parameters
    "damage_multiplier": 4.425086756601032,
    "base_degrad_rate": 0.2263893784946433,
    "embedding_protection": 0.05,
    "stance_protection": 0.05297942608001062,
    "compound_factor": 0.24655096638810692,
    "module_protection": 0.03581702077840391,
    "resilience_min": 0.3619189231037474,
    "resilience_range": 0.8685820392727274,
    "noise_scale": 0.3721771471551047,
    "residual_cap": 0.2911412483331081,

    # Recovery parameters
    "base_recovery_rate": 0.09784464708540677,
    "embedding_bonus": 0.7041284088097765,
    "cluster_bonus": 0.20321466581788367,
    "degradation_penalty": 0.5154171267016461,
    "degrad_recovery_factor": 0.999,
    "corruption_decay": 0.9606193298480378,

    # Module effect parameters
    "effect_pattern_detector": 0.16147197022054166,
    "effect_threat_filter": 0.23155353040009838,
    "effect_recovery_accelerator": 0.2955668931610827,
    "effect_exploration_dampener": 0.12438277612564759,
    "effect_embedding_protector": 0.3015158538914583,
    "effect_cascade_breaker": 0.17234564218719955,
    "effect_residual_cleaner": 0.21765559540021426,
    "effect_anticipation_enhancer": 0.26640399272993437,

    # Threshold parameters
    "consolidation_threshold": 0.05,
    "spread_threshold": 0.11129851144016573,
    "spread_probability": 0.4817027977670055,
    "spread_strength_factor": 0.5717624185139087,
    "module_cap": 6.001043522753437,
    "min_activations": 4.56680960109646,
}

def get_optimized_config() -> EvolvableConfig:
    """Get the evolved optimized configuration."""
    return EvolvableConfig.from_dict(OPTIMIZED_CONFIG)

def get_optimized_dict() -> dict:
    """Get the optimized configuration as a dictionary."""
    return OPTIMIZED_CONFIG.copy()
