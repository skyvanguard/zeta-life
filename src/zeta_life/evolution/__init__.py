"""
Zeta Life Evolution Module

Evolutionary optimization of IPUESA hyperparameters using
genetic algorithms inspired by OpenAlpha_Evolve.
"""

from .config_space import EvolvableConfig, PARAM_RANGES
from .fitness_evaluator import evaluate_config, FitnessResult

__all__ = [
    'EvolvableConfig',
    'PARAM_RANGES',
    'evaluate_config',
    'FitnessResult',
]
