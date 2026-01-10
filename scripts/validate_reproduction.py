"""
Validate reproduction of IPUESA results against expected outputs.

Usage:
    python scripts/validate_reproduction.py

Checks:
1. SYNTH-v2 metrics are within expected ranges
2. Ablation results match expected patterns
3. All criteria pass rates meet thresholds
"""

import json
from pathlib import Path
import sys


def load_json(path: Path) -> dict:
    """Load JSON file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def validate_synth_v2(results_path: Path, expected_path: Path) -> tuple:
    """Validate SYNTH-v2 metrics against expected values."""
    results = load_json(results_path)
    expected = load_json(expected_path)

    if results is None:
        return False, "Results file not found"
    if expected is None:
        return False, "Expected file not found"

    issues = []
    passes = []

    # Check metrics from repeatability results if available
    if 'repeatability' in results and 'metrics_with_ci' in results['repeatability']:
        metrics = results['repeatability']['metrics_with_ci']
    elif 'metrics_with_ci' in results:
        metrics = results['metrics_with_ci']
    else:
        return False, "No metrics found in results"

    for metric, exp in expected['expected_values'].items():
        if metric not in metrics:
            issues.append(f"Missing metric: {metric}")
            continue

        actual = metrics[metric]['mean']
        lo, hi = exp['acceptable_range']

        if lo <= actual <= hi:
            passes.append(f"{metric}: {actual:.3f} ✓ (expected {lo}-{hi})")
        else:
            issues.append(f"{metric}: {actual:.3f} ✗ (expected {lo}-{hi})")

    # Check pass rates
    if 'repeatability' in results and 'summary' in results['repeatability']:
        summary = results['repeatability']['summary']
        pass_5 = summary.get('pass_rate_5_of_6', 0)
        pass_6 = summary.get('pass_rate_6_of_6', 0)

        exp_pass_5 = expected['expected_pass_rates']['pass_5_of_6']['min']
        exp_pass_6 = expected['expected_pass_rates']['pass_6_of_6']['min']

        if pass_5 >= exp_pass_5:
            passes.append(f"Pass rate (5/6): {100*pass_5:.0f}% ✓")
        else:
            issues.append(f"Pass rate (5/6): {100*pass_5:.0f}% < {100*exp_pass_5:.0f}%")

        if pass_6 >= exp_pass_6:
            passes.append(f"Pass rate (6/6): {100*pass_6:.0f}% ✓")
        else:
            issues.append(f"Pass rate (6/6): {100*pass_6:.0f}% < {100*exp_pass_6:.0f}%")

    return len(issues) == 0, {'passes': passes, 'issues': issues}


def validate_ablation(results_path: Path, expected_path: Path) -> tuple:
    """Validate ablation results against expected patterns."""
    results = load_json(results_path)
    expected = load_json(expected_path)

    if results is None:
        return False, "Results file not found"
    if expected is None:
        return False, "Expected file not found"

    issues = []
    passes = []

    if 'ablation' not in results:
        return False, "No ablation results found"

    ablation = results['ablation']
    exp_conditions = expected['conditions']

    for condition, exp in exp_conditions.items():
        if condition not in ablation:
            issues.append(f"Missing condition: {condition}")
            continue

        actual = ablation[condition]
        exp_passed = exp['criteria_passed']
        actual_passed = actual.get('criteria_passed', 0)

        # Check criteria passed
        if actual_passed == exp_passed:
            passes.append(f"{condition}: {actual_passed}/6 criteria ✓")
        elif abs(actual_passed - exp_passed) <= 1:
            passes.append(f"{condition}: {actual_passed}/6 criteria ~ (expected {exp_passed})")
        else:
            issues.append(f"{condition}: {actual_passed}/6 criteria (expected {exp_passed})")

    # Check key finding: full should be best
    if 'full' in ablation:
        full_passed = ablation['full'].get('criteria_passed', 0)
        all_others = [
            ablation[c].get('criteria_passed', 0)
            for c in ablation if c != 'full'
        ]
        if full_passed >= max(all_others):
            passes.append("Full config is best or tied ✓")
        else:
            issues.append("Full config is NOT best")

    return len(issues) == 0, {'passes': passes, 'issues': issues}


def main():
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    expected_dir = base_dir / 'expected_outputs'

    print("=" * 60)
    print("IPUESA REPRODUCTION VALIDATION")
    print("=" * 60)

    all_passed = True

    # Validate SYNTH-v2
    print("\n1. SYNTH-v2 Metrics Validation")
    print("-" * 40)

    synth_results = results_dir / 'ipuesa_synth_v2_consolidation.json'
    synth_expected = expected_dir / 'synth_v2_metrics.json'

    passed, details = validate_synth_v2(synth_results, synth_expected)

    if isinstance(details, str):
        print(f"   ERROR: {details}")
        all_passed = False
    else:
        for p in details['passes']:
            print(f"   {p}")
        for i in details['issues']:
            print(f"   {i}")

        if not passed:
            all_passed = False
            print("\n   ⚠ SYNTH-v2 validation FAILED")
        else:
            print("\n   ✓ SYNTH-v2 validation PASSED")

    # Validate Ablation
    print("\n2. Ablation Study Validation")
    print("-" * 40)

    ablation_expected = expected_dir / 'ablation_results.json'

    passed, details = validate_ablation(synth_results, ablation_expected)

    if isinstance(details, str):
        print(f"   ERROR: {details}")
        all_passed = False
    else:
        for p in details['passes']:
            print(f"   {p}")
        for i in details['issues']:
            print(f"   {i}")

        if not passed:
            all_passed = False
            print("\n   ⚠ Ablation validation FAILED")
        else:
            print("\n   ✓ Ablation validation PASSED")

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: ✓ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("OVERALL: ⚠ SOME VALIDATIONS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
