"""
Statistical utilities for IPUESA validation.

Provides functions for:
- Confidence intervals
- Effect sizes
- Statistical tests
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for a list of values.

    Args:
        data: List of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(data) < 2:
        mean = data[0] if data else 0.0
        return mean, mean, mean

    arr = np.array(data)
    mean = np.mean(arr)
    se = stats.sem(arr)  # Standard error
    n = len(arr)

    # t-distribution critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

    margin = t_crit * se
    return float(mean), float(mean - margin), float(mean + margin)


def compute_effect_size(
    group1: List[float],
    group2: List[float]
) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group1: Control group values
        group2: Treatment group values

    Returns:
        Cohen's d effect size
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    arr1 = np.array(group1)
    arr2 = np.array(group2)

    n1, n2 = len(arr1), len(arr2)
    var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(arr2) - np.mean(arr1)) / pooled_std)


def compare_conditions(
    condition1: List[float],
    condition2: List[float],
    test: str = 'mann_whitney'
) -> Dict:
    """
    Compare two conditions statistically.

    Args:
        condition1: Values from first condition
        condition2: Values from second condition
        test: Statistical test to use ('mann_whitney', 't_test', 'wilcoxon')

    Returns:
        Dictionary with test results
    """
    result = {
        'condition1_mean': np.mean(condition1),
        'condition2_mean': np.mean(condition2),
        'effect_size': compute_effect_size(condition1, condition2),
    }

    if len(condition1) < 3 or len(condition2) < 3:
        result['p_value'] = None
        result['test'] = 'insufficient_data'
        return result

    if test == 'mann_whitney':
        stat, p = stats.mannwhitneyu(condition1, condition2, alternative='two-sided')
        result['statistic'] = float(stat)
        result['p_value'] = float(p)
        result['test'] = 'Mann-Whitney U'
    elif test == 't_test':
        stat, p = stats.ttest_ind(condition1, condition2)
        result['statistic'] = float(stat)
        result['p_value'] = float(p)
        result['test'] = "Student's t"
    elif test == 'wilcoxon':
        # For paired data
        min_len = min(len(condition1), len(condition2))
        stat, p = stats.wilcoxon(condition1[:min_len], condition2[:min_len])
        result['statistic'] = float(stat)
        result['p_value'] = float(p)
        result['test'] = 'Wilcoxon signed-rank'

    return result


def format_ci(
    mean: float,
    lower: float,
    upper: float,
    decimals: int = 3
) -> str:
    """Format confidence interval as string."""
    return f"{mean:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def summarize_distribution(data: List[float]) -> Dict:
    """
    Compute summary statistics for a distribution.

    Returns:
        Dictionary with mean, std, median, min, max, CI
    """
    arr = np.array(data)
    mean, ci_lo, ci_hi = compute_confidence_interval(data)

    return {
        'n': len(data),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'ci_95_lower': ci_lo,
        'ci_95_upper': ci_hi,
        'ci_formatted': format_ci(mean, ci_lo, ci_hi),
    }


def is_significantly_different(
    condition1: List[float],
    condition2: List[float],
    alpha: float = 0.05
) -> bool:
    """Check if two conditions are significantly different at given alpha level."""
    result = compare_conditions(condition1, condition2)
    if result['p_value'] is None:
        return False
    return result['p_value'] < alpha
