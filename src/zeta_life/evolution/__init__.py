"""
Zeta Life Evolution Module

Evolutionary optimization of IPUESA hyperparameters using
genetic algorithms inspired by OpenAlpha_Evolve.
"""

from .config_space import PARAM_RANGES, EvolvableConfig
from .fitness_evaluator import FitnessResult, evaluate_config
from .optimized_config import (
    OPTIMIZED_CONFIG,
    get_optimized_config,
    get_optimized_dict,
)

__all__ = [
    'EvolvableConfig',
    'PARAM_RANGES',
    'evaluate_config',
    'FitnessResult',
    'OPTIMIZED_CONFIG',
    'get_optimized_config',
    'get_optimized_dict',
]
