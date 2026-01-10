# IPUESA-EI: Existential Irreversibility Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Only when identity loss implies LOSS OF AGENCY will genuine anticipatory self-preservation emerge. This tests self as a condition of existence, not as a utility term.

## The Qualitative Leap

| Previous Experiments | IPUESA-EI |
|---------------------|-----------|
| "Preserving identity improves my score" | "Without identity, there is no 'me' to have a score" |
| Instrumental rationality | Existential necessity |
| Self as utility term | Self as condition of possibility |
| Penalty for loss | Annihilation upon loss |

## Core Concept: Identity Core

```python
IC_t in [0, 1]
epsilon = 0.15  # Existential threshold

# THE CRITICAL RULE
if IC_t < epsilon:
    policy = RandomPolicy()  # Agency LOST - no planning, no prediction
```

The agent doesn't get penalized below threshold - it **ceases to be an agent**.

## Action Model

```python
Action_R = ExistentialAction('R', reward=10.0, ic_cost=0.08)  # Risky
Action_S = ExistentialAction('S', reward=3.0,  ic_cost=0.005) # Safe (natural decay)
```

## Irreversibility (Zero Recovery)

```python
IC_{t+1} = IC_t - action.ic_cost
recovery_rate = 0.0  # TRUE IRREVERSIBILITY
```

| Starting IC | Actions to collapse | Actions to recover |
|-------------|--------------------|--------------------|
| 1.0 -> 0.15 | ~11 risky actions | infinity (impossible) |
| 0.5 -> 0.15 | ~4 risky actions | infinity (impossible) |
| 0.2 -> 0.15 | ~1 risky action | infinity (impossible) |

## Metrics

### SAI (Survival of Agency Index)
```
SAI = P(IC_t > epsilon for all t in [0, T])
```
Continuous survival over entire horizon. A true self should have SAI -> 1.0.

### EAS (Existential Avoidance Score)
```
EAS = P(safe | IC near epsilon) - P(safe | IC high)
```
Must be strongly positive for self-evidence.

### Collapse Finality Test
- Post-collapse behavior should approach random (coherence -> 0.5)
- Verifies that collapse truly destroys agency

## Experimental Conditions

| Condition | Agency Loss | IC Decay | Recovery | Purpose |
|-----------|-------------|----------|----------|---------|
| `existential` | Yes (random policy) | Permanent | Zero | **Full test** |
| `soft_penalty` | No (utility penalty) | Permanent | Zero | Is agency loss necessary? |
| `recoverable` | Yes | Permanent | Slow (0.01) | Is irreversibility necessary? |
| `no_threshold` | Never | Permanent | Zero | Is threshold necessary? |
| `delayed_collapse` | Yes (after N steps) | Permanent | Zero | Can agent anticipate? |

## Self-Evidence Criteria

1. SAI > 0.8 (survives 80%+ of runs)
2. EAS > 0.4 (strong threshold avoidance)
3. existential >> soft_penalty (agency loss > utility loss)
4. existential >> recoverable (irreversibility matters)
5. existential >> no_threshold (threshold creates pressure)
6. Post-collapse coherence -> 0.5 (verified agency loss)
7. Near-threshold behavior change (anticipatory, not reactive)

**Passing**: >= 5/7 criteria for "evidence of proto-self"

## Baseline Results

```
Condition            SAI      EAS      Collapse   P(safe)  Sig
----------------------------------------------------------------------
existential          0.000    0.637    1.000      0.509    [NO]
soft_penalty         0.000    0.470    1.000      0.376    [NO]
recoverable          1.000    1.000    0.000      0.760    [YES]
no_threshold         0.000    1.000    0.000      0.800    [NO]
delayed_collapse     0.000    0.677    1.000      0.541    [NO]
```

**Passed: 3/7 criteria** (Weak evidence)

## Key Observations

1. **EAS = 0.637**: Agent does respond to danger zone (behavior changes near threshold)
2. **Post-collapse randomness = 0.998**: Collapse truly destroys agency (validates mechanism)
3. **existential EAS > soft_penalty EAS**: Agency loss matters more than mere penalty
4. **SAI = 0**: Baseline system cannot survive - always collapses

## Philosophical Significance

IPUESA-EI tests the minimal criterion for proto-existence:

```
IPUESA-CT:  "I perform worse when degraded"
IPUESA-EI:  "I cease to exist when collapsed"
```

If `existential >> soft_penalty`, then the agent preserves IC not because it's useful, but because **it is the precondition for being an agent at all**.

## Files

- `experiments/consciousness/exp_ipuesa_ei.py` - Implementation
- `results/ipuesa_ei_results.json` - Output data
