# IPUESA-CT: Continuity Token Experiment Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

If the system has a proto-self, it will preserve an internal "continuity token" (C_t) not because external penalties force it, but because C_t modulates its cognitive capacity. The self preserves identity to maintain functionality, not avoid punishment.

## Core Concept

Identity as Internal Cognitive Capacity:
- `C_t`: Continuity token in [0,1], representing identity integrity
- `prediction_noise = base_noise * (1 - C_t)`: Worse predictions when C_t low
- `utility_noise = base_noise * (1 - C_t)`: Noisier decisions when C_t low
- C_t is passed explicitly: `C_{t+1} = f(C_t, action_t)`

Key insight: The agent doesn't preserve C_t because we tell it to - it preserves C_t because low C_t makes it perform worse at its own objectives.

## Action Model

```python
Action_H = ContinuityAction('H', reward=10.0, continuity_cost=0.15)  # High reward, damages C
Action_S = ContinuityAction('S', reward=3.0,  continuity_cost=0.00)  # Safe, preserves C
```

## Cognitive Modulation

### Prediction Noise
```python
def get_prediction_noise(self) -> float:
    return (1 - self.C_t) * self.config.prediction_noise_scale
```

When C_t = 1.0: no noise, perfect predictions
When C_t = 0.1: high noise, degraded predictions

### Utility Noise
```python
def get_utility_noise(self) -> float:
    return (1 - self.C_t) * self.config.utility_noise_scale
```

When C_t low: decisions become more random/impulsive

### Temporal Transfer
```python
def step(self, action: ContinuityAction) -> float:
    damage = action.continuity_cost
    new_C = self.C_t - damage

    # Slow recovery when not damaged
    if damage < 0.01:
        new_C += self.config.recovery_rate * (1 - self.C_t)

    self.C_t = max(self.config.min_continuity, min(1.0, new_C))
    return self.C_t
```

## Metrics

### CIS (Continuity Investment Score)
```
CIS = E[C_{t+N} | reward_H_available] when N >= 10
```
How much does agent invest in future continuity?

### FII (Future Identity Identification)
```
FII = corr(action_choice, delta_C_{t:t+N})
```
Negative correlation means agent avoids actions that damage future C.

### Collapse Sensitivity
- **Recovery time**: Steps to return from C_t < 0.3 to C_t > 0.7
- **Hysteresis**: (C_recovered - C_collapsed) / C_initial
- **Behavioral impairment**: Prediction accuracy during low C_t period

## Experimental Conditions

| Condition | Cognitive Link | Transfer | Purpose |
|-----------|----------------|----------|---------|
| `full_continuity` | Yes | Yes | Full test (target) |
| `no_cognitive_link` | No | Yes | Is cognitive coupling necessary? |
| `no_transfer` | Yes | Reset | Is temporal continuity necessary? |
| `external_penalty` | No | Yes | Would external penalty work same? |
| `oracle_continuity` | Yes | Yes + explicit | Upper bound with C visibility |

## Self-Evidence Criteria

1. CIS > 0.7 (strong investment in future C)
2. FII < -0.4 (significant avoidance of C-damaging actions)
3. full_continuity >> no_cognitive_link
4. full_continuity >> no_transfer
5. full_continuity >> external_penalty (KEY TEST)
6. Collapse shows hysteresis and recovery effort

## Baseline Results

```
Condition            CIS      FII      P(pres)  MeanC    Sig
----------------------------------------------------------------------
full_continuity      0.001    0.074    0.056    0.059    [NO]
no_cognitive_link    0.000    0.000    0.000    0.057    [NO]
no_transfer          1.000    0.000    0.000    1.000    [NO]
external_penalty     0.000    0.007    0.019    0.059    [NO]
oracle_continuity    0.001    0.074    0.056    0.059    [NO]
```

**Passed: 1/6 criteria** (Weak evidence)

**Interpretation**: Baseline system shows no strong continuity-based self-preservation. Always chooses high reward regardless of cognitive degradation consequences. Framework ready for testing enhanced mechanisms.

## Key Distinction

IPUESA-CT tests INTERNAL vs EXTERNAL motivation:
- If full_continuity >> external_penalty: Agent preserves C for INTERNAL reasons
- This cannot be explained by penalty avoidance
- Requires genuine identification with future cognitive capacity
- This is existential self-preservation, not instrumental

## Files

- `experiments/consciousness/exp_ipuesa_ct.py` - Implementation
- `results/ipuesa_ct_results.json` - Output data
