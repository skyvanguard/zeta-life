# IPUESA-HG+: Holographic Self Stress Test Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Enhanced stress conditions will create clear differentiation between agents with holographic embeddings and those without, revealing the true value of self-similar identity structures under extreme perturbation.

## Modifications from IPUESA-HG

| Parameter | IPUESA-HG | IPUESA-HG+ |
|-----------|-----------|------------|
| Damage multiplier | 1× | 2× (normal), 3× (high_stress) |
| Wave interval | 15 steps | 10 steps (normal), 8 steps (high_stress) |
| Perturbation types | 5 | 6 (+ structural) |
| Residual damage | None | Cumulative (doesn't fully recover) |
| Embedding dimensions | 8 only | 8 (full), 4 (partial), 0 (none) |

## Enhanced Perturbation System

**6 Wave Types:**
```
Wave 1: history      - Scramble memory       (0.6 damage, 0.15 residual)
Wave 2: prediction   - Add noise             (0.5 damage, 0.10 residual)
Wave 3: social       - Corrupt embeddings    (0.7 damage, 0.20 residual)
Wave 4: structural   - Damage embedding      (0.6 damage, 0.25 residual) [NEW]
Wave 5: identity     - Direct IC damage      (0.4 damage, 0.15 residual)
Wave 6: catastrophic - All systems           (1.0 damage, 0.30 residual)
```

**Cumulative Damage:**
```python
agent.residual_damage += actual_damage * wave.residual_factor
# Residual capped at 0.8, recovers at only 2% per step
```

## New Metrics

| Metric | Formula | Threshold |
|--------|---------|-----------|
| HS | P(survival) | > 0.3 |
| PI | preemptive / total_actions | > 0.2 |
| DS | 1 - max(IC_derivative) | > 0.5 |
| EI | embedding_integrity | > 0.4 |
| RS | successful_recoveries / attempts | > 0.3 |
| CE | corr(actions, survival) | > 0.2 |
| HS_diff | HS(full) - HS(no_emb) | > 0.1 |
| Gradient | no_emb < partial <= full | True |

## Experimental Conditions

| Condition | Embedding | Damage | Interval |
|-----------|-----------|--------|----------|
| `full_hg` | 8-dim | 2× | 10 steps |
| `no_emb` | None | 2× | 10 steps |
| `partial_hg` | 4-dim | 2× | 10 steps |
| `high_stress` | 8-dim | 3× | 8 steps |

## Baseline Results

```
Condition       HS       PI       DS       EI       RS      Resid
----------------------------------------------------------------------
full_hg         0.000    0.000    0.000    0.000    0.211   0.800
no_emb          0.000    0.000    0.000    0.000    0.235   0.800
partial_hg      0.000    0.000    0.000    0.000    0.212   0.800
high_stress     0.000    0.000    0.000    0.000    0.193   0.800
```

**Passed: 0/8 criteria** - No evidence (stress too severe)

## Key Observations

1. **Total extinction**: HS = 0.000 across all conditions
2. **Residual saturation**: All agents hit 0.8 residual cap
3. **No preemptive actions**: Agents don't anticipate (die too fast)
4. **Recovery attempted**: RS ≈ 0.2 shows agents tried to recover
5. **No differentiation**: All conditions equally lethal

## Analysis

The stress test reveals the parameter space:

| Experiment | Survival | Problem |
|------------|----------|---------|
| IPUESA-HG | 100% | Too easy - no differentiation |
| IPUESA-HG+ | 0% | Too hard - total extinction |

**Optimal parameters lie between:**
- Damage multiplier: 1.3-1.5× (not 1× or 2×)
- Wave interval: 12-13 steps (not 15 or 10)
- Residual factor: 0.05-0.10 (not 0.15-0.30)

## Recommended Next Steps

1. **IPUESA-HG-Calibrated**: Find sweet spot parameters
   - Target: 30-70% survival for full_hg, 10-40% for no_emb
   - Binary search on damage multiplier

2. **Survival curve analysis**: Plot survival vs damage multiplier
   - Find the "cliff" where embeddings make difference

3. **Residual tuning**: Reduce residual factors by 50%
   - Allow some long-term recovery

## Philosophical Significance

The HG/HG+ pair demonstrates the importance of **calibrated stress testing**:

- Too little stress: All survive, no selection pressure
- Too much stress: All die, no opportunity for adaptation
- Optimal stress: Creates meaningful differentiation

This mirrors biological and psychological resilience - the "Goldilocks zone" where challenge is survivable but demanding.

## Files

- `experiments/consciousness/exp_ipuesa_hg_plus.py` - Implementation
- `results/ipuesa_hg_plus_results.json` - Output data
