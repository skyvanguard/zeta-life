"""
Fitness Evaluator for IPUESA Evolution

Evaluates configurations by running IPUESA simulations and computing
fitness scores based on self-evidence criteria.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config_space import PARAM_RANGES, EvolvableConfig


@dataclass
class FitnessResult:
    """Result of evaluating a configuration."""
    fitness_score: float          # Composite score (0-1)
    criteria_passed: int          # Criteria passed (0-8)
    criteria_total: int           # Total criteria (8)
    criteria_details: dict[str, bool]  # Individual criteria results
    metrics: dict[str, float]     # Raw metrics from simulation
    valid_config: bool            # Whether config was valid
    error: str | None = None   # Error message if failed


def validate_config(config: dict[str, Any]) -> tuple[bool, str]:
    """Validate that config has all required keys within valid ranges."""
    # Check for missing keys
    missing = set(PARAM_RANGES.keys()) - set(config.keys())
    if missing:
        return False, f"Missing keys: {missing}"

    # Check ranges
    for key, (min_val, max_val) in PARAM_RANGES.items():
        val = config.get(key)
        if val is None:
            return False, f"Missing value for {key}"
        if not (min_val <= val <= max_val):
            return False, f"{key}={val} outside range [{min_val}, {max_val}]"

    return True, "OK"


def evaluate_self_evidence(metrics: dict[str, float]) -> dict[str, bool]:
    """
    Evaluate the 8 self-evidence criteria.

    Criteria:
    1. HS_in_range: Holographic Survival in Goldilocks zone [0.30, 0.70]
    2. MSR_pass: Module Spreading Rate > 0.15
    3. TAE_pass: Temporal Anticipation Effectiveness > 0.15
    4. EI_pass: Embedding Integrity > 0.30
    5. ED_pass: Emergent Differentiation > 0.10
    6. diff_pass: HS > baseline HS
    7. gradient_pass: Valid gradient (simplified)
    8. smooth_transition: Degradation variance > 0.02 (not bistable)
    """
    hs = metrics.get('holographic_survival', 0)
    msr = metrics.get('module_spreading_rate', 0)
    tae = metrics.get('temporal_anticipation_effectiveness', 0)
    ei = metrics.get('embedding_integrity', 0)
    ed = metrics.get('emergent_differentiation', 0)
    hs_baseline = metrics.get('baseline_survival', 0)
    deg_var = metrics.get('degradation_variance', 0)

    # For gradient_pass, check if there's a valid ordering
    hs_no_emb = metrics.get('no_embedding_survival', 0)
    gradient_valid = hs >= hs_no_emb >= hs_baseline

    return {
        'HS_in_range': 0.30 <= hs <= 0.70,
        'MSR_pass': msr > 0.15,
        'TAE_pass': tae > 0.15,
        'EI_pass': ei > 0.30,
        'ED_pass': ed > 0.10,
        'diff_pass': hs > hs_baseline,
        'gradient_pass': gradient_valid,
        'smooth_transition': deg_var > 0.02,
    }


def calculate_fitness(criteria: dict[str, bool],
                      metrics: dict[str, float]) -> float:
    """
    Calculate composite fitness score.

    Structure:
    - 70% binary criteria (passed/total)
    - 30% continuous metrics (for gradient improvement)
    """
    # Binary component
    binary_score = sum(criteria.values()) / len(criteria)

    # Continuous component (normalized 0-1)
    hs = metrics.get('holographic_survival', 0)
    msr = metrics.get('module_spreading_rate', 0)
    tae = metrics.get('temporal_anticipation_effectiveness', 0)
    ed = metrics.get('emergent_differentiation', 0)

    # Penalize HS outside Goldilocks (peak at 0.5)
    hs_score = 1.0 - abs(hs - 0.5) * 2
    hs_score = max(0, hs_score)  # Clamp to [0, 1]

    continuous_score = (
        hs_score * 0.25 +
        min(msr / 0.50, 1.0) * 0.25 +
        min(tae / 0.30, 1.0) * 0.25 +
        min(ed / 0.30, 1.0) * 0.25
    )

    # Composite fitness
    fitness = 0.70 * binary_score + 0.30 * continuous_score

    return round(fitness, 4)


def evaluate_config(config: dict[str, Any],
                    n_runs: int = 5,
                    n_steps: int = 100,
                    n_agents: int = 24,
                    n_clusters: int = 4) -> FitnessResult:
    """
    Evaluate a configuration by running IPUESA simulation.

    Args:
        config: Dictionary with 30 parameters
        n_runs: Number of runs to average (reduces variance)
        n_steps: Simulation steps per run
        n_agents: Number of agents
        n_clusters: Number of clusters

    Returns:
        FitnessResult with score and metrics
    """
    # 1. Validate config
    valid, msg = validate_config(config)
    if not valid:
        return FitnessResult(
            fitness_score=0.0,
            criteria_passed=0,
            criteria_total=8,
            criteria_details={},
            metrics={},
            valid_config=False,
            error=msg
        )

    # 2. Run IPUESA simulation
    try:
        from .ipuesa_evolvable import run_ipuesa_with_config

        metrics = run_ipuesa_with_config(
            config=config,
            n_agents=n_agents,
            n_clusters=n_clusters,
            n_steps=n_steps,
            n_runs=n_runs
        )

    except Exception as e:
        return FitnessResult(
            fitness_score=0.0,
            criteria_passed=0,
            criteria_total=8,
            criteria_details={},
            metrics={},
            valid_config=True,
            error=str(e)
        )

    # 3. Evaluate self-evidence criteria
    criteria = evaluate_self_evidence(metrics)

    # 4. Calculate fitness score
    fitness = calculate_fitness(criteria, metrics)

    return FitnessResult(
        fitness_score=fitness,
        criteria_passed=sum(criteria.values()),
        criteria_total=8,
        criteria_details=criteria,
        metrics=metrics,
        valid_config=True
    )


def quick_evaluate(config: dict[str, Any]) -> tuple[float, int]:
    """
    Quick evaluation for debugging (fewer runs, fewer steps).

    Returns:
        (fitness_score, criteria_passed)
    """
    result = evaluate_config(config, n_runs=2, n_steps=50)
    return result.fitness_score, result.criteria_passed
