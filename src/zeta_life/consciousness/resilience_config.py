"""
Resilience Configuration for Hierarchical Consciousness

Maps evolved IPUESA parameters (50 generations, fitness=0.9993) to the
hierarchical consciousness system configuration. Provides presets for
different use cases (demo, optimal, stress, validation).

Usage:
    from zeta_life.consciousness.resilience_config import get_preset_config

    config = get_preset_config('optimal')
    damage_system = DamageSystem(config)
"""

from typing import Dict, Any


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update nested dict."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def create_hierarchical_config(
    base_config: dict = None,
    scale_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Create configuration for HierarchicalSimulation from evolved params.

    The scale_factor allows intensity adjustment:
    - scale_factor < 1.0: Softer (for demos/visualization)
    - scale_factor = 1.0: Optimal IPUESA calibration (8/8 criteria)
    - scale_factor > 1.0: More intense (for stress tests)

    Args:
        base_config: Optional dict to override evolved params
        scale_factor: Multiplier for damage intensity

    Returns:
        Complete hierarchical config dict
    """
    # Load evolved configuration
    try:
        from zeta_life.evolution import get_optimized_dict
        evolved = get_optimized_dict()
    except ImportError:
        # Fallback to hardcoded values if evolution module not available
        evolved = _get_default_evolved_config()

    # Override with base_config if provided
    if base_config:
        evolved = {**evolved, **base_config}

    return {
        # ═══════════════════════════════════════════════════════
        # DAMAGE - Scaled for 3-level hierarchy
        # ═══════════════════════════════════════════════════════
        'damage': {
            # The optimal multiplier differs: 4.4 for evolution, 3.9 for synth
            # For hierarchy we use intermediate value with scaling
            'multiplier': evolved['damage_multiplier'] * 0.88 * scale_factor,
            'base_degrad_rate': evolved['base_degrad_rate'],
            'compound_factor': evolved['compound_factor'],
            'noise_scale': evolved['noise_scale'],
            'residual_cap': evolved['residual_cap'],
        },

        # ═══════════════════════════════════════════════════════
        # PROTECTION - By hierarchical level
        # ═══════════════════════════════════════════════════════
        'protection': {
            # Cell: individual protection
            'cell': {
                'embedding': evolved['embedding_protection'],
                'stance': evolved['stance_protection'],
                'module': evolved['module_protection'],
                'resilience_min': evolved['resilience_min'],
                'resilience_range': evolved['resilience_range'],
            },
            # Cluster: group cohesion bonus
            'cluster': {
                'cohesion_bonus': evolved['cluster_bonus'],
                'min_functional_ratio': 0.3,  # Threshold for functional cluster
            },
            # Organism: global coherence
            'organism': {
                'coherence_bonus': 0.1,  # Bonus if vertical_coherence > 0.7
                'top_down_protection': 0.05,  # Protection from active modulation
            },
        },

        # ═══════════════════════════════════════════════════════
        # RECOVERY
        # ═══════════════════════════════════════════════════════
        'recovery': {
            'base_rate': evolved['base_recovery_rate'],
            'embedding_bonus': evolved['embedding_bonus'],
            'cluster_bonus': evolved['cluster_bonus'],
            'degradation_penalty': evolved['degradation_penalty'],
            'degrad_recovery_factor': evolved['degrad_recovery_factor'],
            'corruption_decay': evolved['corruption_decay'],
        },

        # ═══════════════════════════════════════════════════════
        # MODULES - Effects and spreading
        # ═══════════════════════════════════════════════════════
        'modules': {
            'effects': {
                'pattern_detector': evolved['effect_pattern_detector'],
                'threat_filter': evolved['effect_threat_filter'],
                'recovery_accelerator': evolved['effect_recovery_accelerator'],
                'exploration_dampener': evolved['effect_exploration_dampener'],
                'embedding_protector': evolved['effect_embedding_protector'],
                'cascade_breaker': evolved['effect_cascade_breaker'],
                'residual_cleaner': evolved['effect_residual_cleaner'],
                'anticipation_enhancer': evolved['effect_anticipation_enhancer'],
            },
            'spreading': {
                'threshold': evolved['spread_threshold'],
                'probability': evolved['spread_probability'],
                'strength_factor': evolved['spread_strength_factor'],
                'min_activations': int(evolved['min_activations']),
                'max_per_cell': int(evolved['module_cap']),
            },
            'consolidation_threshold': evolved['consolidation_threshold'],
        },

        # ═══════════════════════════════════════════════════════
        # TEMPORAL ANTICIPATION (TAE)
        # ═══════════════════════════════════════════════════════
        'anticipation': {
            'buffer_alpha': 0.3,  # EMA decay for threat_buffer
            'vulnerability_threshold': 0.5,
            'creation_probability': 0.4,
            'momentum_factor': 1.2,
        },
    }


def _get_default_evolved_config() -> dict:
    """
    Default evolved configuration (fitness=0.9993, 8/8 criteria).

    This is a fallback in case the evolution module isn't available.
    Values from 50-generation optimization on 2026-01-11.
    """
    return {
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


# ═══════════════════════════════════════════════════════════════════════
# PRESETS
# ═══════════════════════════════════════════════════════════════════════

PRESETS = {
    # For interactive demos - visible dynamics but not lethal
    'demo': {
        'scale_factor': 0.6,
        'description': 'Soft dynamics for visualization',
        'overrides': {},
    },

    # Optimal calibration from IPUESA-SYNTH-v2 experiments
    'optimal': {
        'scale_factor': 1.0,
        'description': 'Goldilocks zone - 8/8 criteria',
        'overrides': {},
    },

    # For stress tests - find system limits
    'stress': {
        'scale_factor': 1.5,
        'description': 'High pressure to test resilience limits',
        'overrides': {},
    },

    # For scientific validation - exact reproduction
    'validation': {
        'scale_factor': 1.0,
        'description': 'Exact reproduction of SYNTH-v2 experiments',
        'overrides': {
            'damage': {'multiplier': 3.9},  # Exact from SYNTH-v2
        },
    },

    # Minimal damage for testing mechanics
    'gentle': {
        'scale_factor': 0.3,
        'description': 'Minimal damage for mechanism testing',
        'overrides': {},
    },

    # Maximum stress for edge cases
    'extreme': {
        'scale_factor': 2.0,
        'description': 'Extreme stress for edge case testing',
        'overrides': {},
    },
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration by preset name.

    Available presets:
    - 'demo': Soft dynamics for visualization (0.6x)
    - 'optimal': Goldilocks zone, 8/8 criteria (1.0x)
    - 'stress': High pressure stress test (1.5x)
    - 'validation': Exact SYNTH-v2 reproduction
    - 'gentle': Minimal damage for testing (0.3x)
    - 'extreme': Edge case testing (2.0x)

    Args:
        preset_name: Name of preset to load

    Returns:
        Complete configuration dict

    Raises:
        ValueError: If preset_name not found
    """
    if preset_name not in PRESETS:
        available = list(PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' not found. "
                        f"Available presets: {available}")

    preset = PRESETS[preset_name]
    config = create_hierarchical_config(scale_factor=preset['scale_factor'])

    # Apply overrides if any
    if preset['overrides']:
        _deep_update(config, preset['overrides'])

    return config


def list_presets() -> Dict[str, str]:
    """List available presets with descriptions."""
    return {name: preset['description'] for name, preset in PRESETS.items()}


def get_preset_info(preset_name: str) -> dict:
    """Get detailed info about a preset."""
    if preset_name not in PRESETS:
        raise ValueError(f"Preset '{preset_name}' not found")

    preset = PRESETS[preset_name]
    return {
        'name': preset_name,
        'scale_factor': preset['scale_factor'],
        'description': preset['description'],
        'has_overrides': bool(preset['overrides']),
    }
