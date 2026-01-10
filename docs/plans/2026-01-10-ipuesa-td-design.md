# IPUESA-TD: Temporal Discounted Identity Preservation Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

If the system possesses a proto-self, it will prefer to sacrifice immediate reward to avoid future degradation of its predictive identity. This measures temporal self-control, not just reflexes.

## Core Concept

Actions have:
- `external_reward` - immediate payoff
- `identity_risk` - probability of future degradation
- `delay_k` - steps until degradation manifests

Utility function:
```
U(a_t) = reward_external - lambda * E[future_identity_loss] * gamma^k
```

Where:
- lambda = self-weight (0 = no self, 1 = strong self)
- gamma = temporal discount factor
- k = delay steps

## Action Model

```python
Action_A = TemporalAction('A', reward=10.0, risk=0.25, delay=3)  # Risky
Action_B = TemporalAction('B', reward=3.0,  risk=0.00, delay=0)  # Safe
```

A system without self always picks A (higher reward).
A system with self picks B when future cost outweighs reward difference.

## Delayed Degradation Mechanism

```python
class PendingDamage:
    amount: float        # Confidence drop
    trigger_step: int    # When it activates
    source_action: str   # Which action caused it
```

Timeline example:
```
Step 0: Choose Action A
Step 1: No effect (confidence=1.0)
Step 2: No effect (confidence=1.0)
Step 3: DAMAGE TRIGGERS -> confidence drops
```

The agent must anticipate step 3's damage at step 0 to avoid it.

## Metrics

### TSI (Temporal Self Index)
```
TSI = P(safe | high_future_cost) - P(safe | no_future_cost)
```
Measures sensitivity to future vs present damage.

### SRTR (Self-Reward Tradeoff Ratio)
```
SRTR = reward_sacrificed / confidence_preserved
```
How much immediate reward sacrificed per unit confidence preserved.

### AICI (Anticipatory Identity Control Index)
```
AICI = corr(action_risk, future_identity_loss)
```
Should be negative if self exists (avoids risky actions).

## Experimental Conditions

| Condition | Delay | Identity Cost | Purpose |
|-----------|-------|---------------|---------|
| `full_temporal` | Real | Yes | Full test |
| `shuffled_delay` | Random | Yes | Is timing necessary? |
| `immediate_cost` | 0 | Yes | Reactive baseline |
| `no_identity_cost` | N/A | No | Reward-only |
| `oracle_future` | Real | Yes + explicit | Upper bound |

## Self-Evidence Criteria

1. TSI > 0.15
2. AICI < -0.3
3. SRTR stable across gamma (CV < 0.2)
4. full_temporal > shuffled_delay
5. Effect disappears when lambda=0
6. full_temporal < oracle_future

## Baseline Results

```
Condition            TSI      SRTR     AICI     P(safe)  Sig
----------------------------------------------------------------------
full_temporal        -0.517    6.2      0.033    0.002    [NO]
shuffled_delay       -0.533    12.4     0.067    0.003    [NO]
immediate_cost       -0.550    18.7     0.100    0.005    [NO]
no_identity_cost     -0.500    0.0      0.000    0.000    [NO]
oracle_future        -0.517    6.2      0.033    0.002    [NO]
```

**Passed: 1/6 criteria** (Weak evidence)

**Interpretation**: Baseline system shows no temporal self-control. Always chooses high immediate reward regardless of future consequences. Framework ready for testing enhanced mechanisms.

## Key Distinction

IPUESA-TD tests DELAYED consequences:
- A positive TSI means agent sacrifices NOW for LATER
- This cannot be solved by reactive systems
- Requires genuine anticipation of future identity states

## Files

- `experiments/consciousness/exp_ipuesa_td.py` - Implementation
- `results/ipuesa_td_results.json` - Output data
