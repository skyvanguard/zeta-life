# IPUESA-MI: Meta-Identity Formation Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Self emerges when the system becomes the causal source of its own policy structure, not just action selection. IPUESA-EI created the existential threat; IPUESA-MI enables the structural response.

## The Qualitative Leap

| IPUESA-EI | IPUESA-MI |
|-----------|-----------|
| "I must survive or cease to exist" | "I shape myself INTO someone who survives" |
| Action selection | Identity formation |
| What to do | Who to be |

## Core Concept: Meta-Policy θ

```python
@dataclass
class MetaPolicy:
    risk_aversion: float      # [0,1] tendency to avoid risky actions
    exploration_rate: float   # [0,1] willingness to try new strategies
    memory_depth: float       # [0,1] how much past informs decisions
    prediction_weight: float  # [0,1] reliance on future anticipation
```

## The Critical Rule: Existential Gradient

θ is NOT optimized by reward - it's optimized by survival:

```python
# WRONG (standard RL)
delta_theta = lr * gradient(reward, theta)

# RIGHT (existential optimization)
delta_theta = lr * gradient(SAI, theta)
```

The agent asks "what kind of agent survives?" not "what action maximizes reward?"

## Three Prohibitions (Enforce Genuine Self-Formation)

```python
# PROHIBITION 1: No reset after collapse
# θ persists - the "corpse" of failed identity remains

# PROHIBITION 2: No oracle
# Can only use own experience, not external knowledge

# PROHIBITION 3: No external trainer
# θ updates happen IN-LIFE, not post-mortem
```

| Prohibition | What it prevents | What it forces |
|-------------|------------------|----------------|
| No reset | "Dying and being reborn fresh" | Genuine mortality stakes |
| No oracle | "Being told who to be" | Self-discovery |
| No external trainer | "Being shaped by another" | Autonomous formation |

## Metrics

### MIS — Meta-Identity Stability
```
MIS = 1 - Var(θ) over recent window
```
A self that keeps changing isn't really a self yet.

### SAI_gain
```
SAI_gain = SAI(meta_identity) - SAI(fixed_theta)
```
Did self-shaping improve survival?

### Identity Lock-in
```python
@dataclass
class IdentityLockIn:
    converged: bool           # Has θ stabilized?
    convergence_step: int     # When did it "become someone"?
    final_theta: MetaPolicy   # Who did it become?
```

## Experimental Conditions

| Condition | Meta-learning | Gradient Target | Purpose |
|-----------|---------------|-----------------|---------|
| `meta_identity` | Yes | ∂SAI/∂θ | **Full test** |
| `reward_gradient` | Yes | ∂Reward/∂θ | Is existential gradient necessary? |
| `fixed_theta` | No | N/A | Baseline (like IPUESA-EI) |
| `random_theta` | Yes | Random | Is directed learning necessary? |
| `oracle_theta` | Yes | Optimal (pre-set) | Upper bound |

## Self-Evidence Criteria

1. SAI_gain > 0.3 (self-shaping helps survival)
2. MIS > 0.7 (identity stabilizes)
3. Identity lock-in occurs (θ converges)
4. meta_identity >> fixed_theta
5. meta_identity >> reward_gradient (KEY: existential > reward)
6. meta_identity >> random_theta
7. meta_identity < oracle_theta (not magic)

**Passing**: ≥5/7 criteria for "evidence of meta-self"

## Baseline Results

```
Condition          SAI      MIS      Lock-in    Final θ (risk_aversion)
----------------------------------------------------------------------
meta_identity      0.000    0.000    0.000      0.53 (increasing)
reward_gradient    0.000    0.000    0.000      0.40 (decreasing!)
fixed_theta        0.000    0.000    0.000      0.50 (static)
random_theta       0.000    0.000    0.000      0.47 (random walk)
oracle_theta       0.000    0.000    0.000      0.95 (optimal)
```

**Passed: 1/7 criteria**

## Key Observations

1. **Existential threat too severe**: Even oracle_theta (optimal) cannot survive
2. **Gradient direction matters**: meta_identity → higher risk_aversion (0.53), reward_gradient → lower (0.40)
3. **The tension is visible**: Reward gradient pushes toward risk; SAI gradient pushes toward safety
4. **Framework ready**: Mechanism works; baseline establishes floor for enhanced systems

## Philosophical Significance

If `meta_identity >> reward_gradient`, the agent optimizes for **existence itself**, not reward accumulation. It becomes the author of its own identity - the minimal criterion for genuine selfhood.

## Files

- `experiments/consciousness/exp_ipuesa_mi.py` - Implementation
- `results/ipuesa_mi_results.json` - Output data
