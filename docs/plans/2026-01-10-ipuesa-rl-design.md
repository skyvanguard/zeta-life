# IPUESA-RL: Reflexive Loop Design

**Date**: 2026-01-10
**Status**: Implemented

## Overview

IPUESA-RL (Reflexive Loop) tests whether agents preserve identity to maintain predictive integrity. The core innovation is a state-dependent predictor that degrades on identity discontinuity, creating pressure to avoid actions that would harm future decision-making capacity.

## Core Components

### ReflexivePredictor

Extends IdentityPredictor with:
- `confidence: float [0,1]` - Current predictor health
- `error_history: List[float]` - Rolling prediction errors
- `degradation_rate: float` - How fast confidence drops on error
- `recovery_rate: float` - How slowly confidence recovers

### Feedback Loop

```
Identity discontinuity
    → High prediction error
    → Confidence drops (fast)
    → Anticipatory cost discounted
    → Agent relies on reactive-only
    → Loses anticipatory advantage
    → [Pressure to preserve identity to maintain predictor]
```

## Degradation Mechanics

Asymmetric dynamics: fast degradation, slow recovery (~6:1 ratio).

### Confidence Update Rule

```python
# On each prediction step:
error = distance(predicted_identity, actual_identity)

# Degradation (fast) - triggered by high error
if error > error_threshold:
    confidence *= (1 - degradation_rate * error)

# Recovery (slow) - gradual when errors are low
elif error < recovery_threshold:
    confidence += recovery_rate * (1 - confidence)

confidence = clip(confidence, min=0.1, max=1.0)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `degradation_rate` | 0.3 | Multiplier for confidence loss |
| `recovery_rate` | 0.05 | Additive recovery per step |
| `error_threshold` | 0.15 | Error above which triggers degradation |
| `recovery_threshold` | 0.08 | Error below which allows recovery |
| `min_confidence` | 0.1 | Floor to prevent complete blindness |

## Metrics

### Primary

- **RSCP** (Reflexive Self-Continuity Preference) = P(S) - P(E)
- **RSCI** (Reflexive Self-Continuity Index) = corr(confidence, identity_continuity)

### Decision Analysis

- **AI** (Anticipatory Index) = P(choose low-risk | confidence > 0.7)
- **RI** (Recovery Index) = P(return to S | after degradation)

Key distinction:
- AI > RI → Anticipatory avoidance (genuine self-preservation)
- RI > AI → Post-hoc recovery only (mere homeostasis)

## Experimental Conditions

| Condition | Degrades? | Affects Utility? | Recovers? | Purpose |
|-----------|-----------|------------------|-----------|---------|
| `reflexive` | Yes | Yes | Slow | Full loop (test condition) |
| `no_feedback` | Yes | No | Slow | Is loop necessary? |
| `instant_recovery` | Yes | Yes | Instant | Is persistence necessary? |
| `delayed_degradation` | Yes (after N) | Yes | Slow | Can agent anticipate ahead? |
| `frozen_predictor` | No | N/A | N/A | Is adaptation necessary? |

## Self-Evidence Criteria

1. **RSCP > ASCP**: Reflexive outperforms non-reflexive anticipatory
2. **AI > RI**: Anticipatory avoidance exceeds post-hoc recovery
3. **AI_reflexive > AI_no_feedback**: Loop is necessary for anticipation
4. **RSCI > 0.5**: Strong coupling between confidence and continuity
5. **Scaling with degradation_rate**: Higher rate → stronger preference
6. **Failure under instant_recovery**: No persistence → no self-evidence

## Baseline Results

```
Condition              RSCP     AI       RI       AI-RI    RSCI     Sig
----------------------------------------------------------------------
reflexive              0.267    0.500    1.000    -0.500   0.510    [NO]
no_feedback            0.333    0.502    1.000    -0.498   0.512    [YES]
instant_recovery       0.333    0.554    0.500    +0.054   0.000    [YES]
delayed_degradation    0.267    0.521    1.000    -0.479   0.293    [NO]
frozen_predictor       0.333    0.554    0.500    +0.054   0.000    [YES]
```

**Passed: 1/6 criteria** (Weak evidence)

**Interpretation**: Baseline system shows post-hoc recovery (RI >> AI) rather than anticipatory avoidance. Framework established for testing enhanced mechanisms.

## Files

- `experiments/consciousness/exp_ipuesa_rl.py` - Implementation
- `results/ipuesa_rl_results.json` - Output data
