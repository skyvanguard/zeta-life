"""
IPUESA Scale Testing: Generalization Analysis
==============================================

Tests whether SYNTH-v2's Goldilocks zone scales with system size.

Configurations:
- Baseline: 24 agents, 4 clusters (paper configuration)
- Scale 50: 50 agents, 6 clusters
- Scale 100: 100 agents, 8 clusters

Key questions:
1. Does the Goldilocks zone (3.9×) hold at larger scales?
2. Do metrics remain in acceptable ranges?
3. Does the system exhibit bistability at scale?

Author: IPUESA Research
Date: 2026-01-10
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

# Import from main experiment
from exp_ipuesa_synth_v2 import (
    run_episode, run_condition, to_native
)


def compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and 95% CI."""
    if len(data) < 2:
        m = data[0] if data else 0.0
        return m, m, m
    arr = np.array(data)
    mean = np.mean(arr)
    se = stats.sem(arr)
    t_crit = stats.t.ppf(1 - (1-confidence)/2, df=len(arr)-1)
    margin = t_crit * se
    return float(mean), float(mean - margin), float(mean + margin)


def run_scale_test(
    n_agents: int,
    n_clusters: int,
    damage_mult: float,
    n_runs: int = 10,
    n_steps: int = 150
) -> Dict:
    """Run SYNTH-v2 at a specific scale."""

    print(f"\n{'='*60}")
    print(f"Scale Test: {n_agents} agents, {n_clusters} clusters, damage={damage_mult:.2f}×")
    print(f"{'='*60}")

    all_results = []

    for run_idx in range(n_runs):
        np.random.seed(42 + run_idx)

        result = run_episode(
            n_agents=n_agents,
            n_clusters=n_clusters,
            n_steps=n_steps,
            damage_mult=damage_mult,
            use_embeddings=True,
            embedding_dim=8,
            wave_steps=[25, 50, 75, 100, 125]
        )

        # Compute metrics
        survival_rate = result['survival_rate']
        tae = result.get('final_tae', 0)
        msr = result.get('final_msr', 0)
        ei = result.get('final_ei', 0)

        # Emergent differentiation
        if 'final_ic_values' in result:
            ed = float(np.std(result['final_ic_values']))
        else:
            ed = 0.1

        # Degradation variance
        if 'final_degradation_levels' in result:
            deg_var = float(np.var(result['final_degradation_levels']))
        else:
            deg_var = 0.02

        # Count criteria passed
        criteria = {
            'HS': 0.30 <= survival_rate <= 0.70,
            'TAE': tae > 0.15,
            'MSR': msr > 0.15,
            'EI': ei > 0.30,
            'ED': ed > 0.10,
            'deg_var': deg_var > 0.02
        }

        run_result = {
            'run': run_idx,
            'HS': survival_rate,
            'TAE': tae,
            'MSR': msr,
            'EI': ei,
            'ED': ed,
            'deg_var': deg_var,
            'criteria_passed': sum(criteria.values()),
            'criteria_detail': criteria
        }

        all_results.append(run_result)
        print(f"  Run {run_idx+1}/{n_runs}: HS={survival_rate:.3f}, "
              f"criteria={sum(criteria.values())}/6")

    # Aggregate results
    metrics = {}
    for metric in ['HS', 'TAE', 'MSR', 'EI', 'ED', 'deg_var']:
        values = [r[metric] for r in all_results]
        mean, ci_lo, ci_hi = compute_ci(values)
        metrics[metric] = {
            'mean': mean,
            'ci_95': [ci_lo, ci_hi],
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0
        }

    criteria_passed = [r['criteria_passed'] for r in all_results]
    pass_rate_5 = sum(1 for c in criteria_passed if c >= 5) / len(criteria_passed)
    pass_rate_6 = sum(1 for c in criteria_passed if c >= 6) / len(criteria_passed)

    return {
        'config': {
            'n_agents': n_agents,
            'n_clusters': n_clusters,
            'damage_mult': damage_mult,
            'n_runs': n_runs,
            'n_steps': n_steps
        },
        'metrics': metrics,
        'criteria_passed_mean': float(np.mean(criteria_passed)),
        'pass_rate_5_of_6': pass_rate_5,
        'pass_rate_6_of_6': pass_rate_6,
        'all_runs': all_results
    }


def find_goldilocks_at_scale(
    n_agents: int,
    n_clusters: int,
    n_runs: int = 5
) -> Dict:
    """Search for Goldilocks zone at a given scale."""

    print(f"\n{'='*60}")
    print(f"Goldilocks Search: {n_agents} agents, {n_clusters} clusters")
    print(f"{'='*60}")

    # Test damage multipliers
    damage_range = np.linspace(3.0, 5.0, 11)  # 11 points from 3× to 5×
    results = []

    for damage in damage_range:
        hs_values = []
        for run in range(n_runs):
            np.random.seed(42 + run)
            result = run_episode(
                n_agents=n_agents,
                n_clusters=n_clusters,
                n_steps=150,
                damage_mult=damage,
                use_embeddings=True,
                embedding_dim=8,
                wave_steps=[25, 50, 75, 100, 125]
            )
            hs_values.append(result['survival_rate'])

        mean_hs = np.mean(hs_values)
        in_goldilocks = 0.30 <= mean_hs <= 0.70

        results.append({
            'damage_mult': float(damage),
            'mean_hs': float(mean_hs),
            'std_hs': float(np.std(hs_values)),
            'in_goldilocks': in_goldilocks
        })

        symbol = "✓" if in_goldilocks else "✗"
        print(f"  {damage:.2f}×: HS={mean_hs:.3f} {symbol}")

    # Find optimal damage
    goldilocks_candidates = [r for r in results if r['in_goldilocks']]
    if goldilocks_candidates:
        # Choose the one with HS closest to 0.50 (center of zone)
        optimal = min(goldilocks_candidates, key=lambda x: abs(x['mean_hs'] - 0.50))
    else:
        # No Goldilocks zone found
        optimal = None

    return {
        'n_agents': n_agents,
        'n_clusters': n_clusters,
        'search_results': results,
        'goldilocks_zone': goldilocks_candidates,
        'optimal_damage': optimal
    }


def main():
    print("=" * 70)
    print("IPUESA SCALE ANALYSIS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    output = {}

    # Configuration for each scale
    scales = [
        {'n_agents': 24, 'n_clusters': 4, 'name': 'baseline'},
        {'n_agents': 50, 'n_clusters': 6, 'name': 'scale_50'},
        {'n_agents': 100, 'n_clusters': 8, 'name': 'scale_100'},
    ]

    # Test 1: Does 3.9× work at all scales?
    print("\n" + "=" * 70)
    print("TEST 1: Does 3.9× damage work at all scales?")
    print("=" * 70)

    for scale in scales:
        result = run_scale_test(
            n_agents=scale['n_agents'],
            n_clusters=scale['n_clusters'],
            damage_mult=3.9,
            n_runs=10
        )
        output[f"fixed_damage_{scale['name']}"] = to_native(result)

    # Test 2: Search for Goldilocks at each scale
    print("\n" + "=" * 70)
    print("TEST 2: Goldilocks zone search at each scale")
    print("=" * 70)

    for scale in scales:
        result = find_goldilocks_at_scale(
            n_agents=scale['n_agents'],
            n_clusters=scale['n_clusters'],
            n_runs=5
        )
        output[f"goldilocks_search_{scale['name']}"] = to_native(result)

    # Summary
    print("\n" + "=" * 70)
    print("SCALE ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n1. FIXED DAMAGE (3.9×) RESULTS:")
    for scale in scales:
        key = f"fixed_damage_{scale['name']}"
        metrics = output[key]['metrics']
        print(f"\n   {scale['name'].upper()} ({scale['n_agents']} agents):")
        print(f"   - HS: {metrics['HS']['mean']:.3f} "
              f"[{metrics['HS']['ci_95'][0]:.3f}, {metrics['HS']['ci_95'][1]:.3f}]")
        print(f"   - Pass rate (>=5/6): {100*output[key]['pass_rate_5_of_6']:.0f}%")

    print("\n2. GOLDILOCKS ZONE BY SCALE:")
    for scale in scales:
        key = f"goldilocks_search_{scale['name']}"
        zone = output[key]['goldilocks_zone']
        opt = output[key]['optimal_damage']
        print(f"\n   {scale['name'].upper()} ({scale['n_agents']} agents):")
        if zone:
            damages = [z['damage_mult'] for z in zone]
            print(f"   - Zone found: {min(damages):.1f}× to {max(damages):.1f}×")
            if opt:
                print(f"   - Optimal: {opt['damage_mult']:.2f}× (HS={opt['mean_hs']:.3f})")
        else:
            print("   - NO GOLDILOCKS ZONE FOUND")

    # Save results
    results_path = Path(__file__).parent.parent.parent / 'results' / 'ipuesa_scale_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
