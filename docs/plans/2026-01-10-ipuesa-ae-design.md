# IPUESA-AE: Adaptive Emergence Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Full adaptive emergence (theta + alpha) > meta-only (theta) > cognitive-only (alpha) > no adaptation, because both policy AND cognitive architecture contribute to survival under perturbations.

## The Integration

| IPUESA-MI | IPUESA-AE |
|-----------|-----------|
| "I shape WHO I am" | "I shape WHO I am AND HOW I think" |
| Policy adaptation | Policy + Architecture adaptation |
| theta only | theta + alpha |

## Dual Adaptation Architecture

**Meta-Policy θ (WHO the agent is):**
```python
@dataclass
class MetaPolicy:
    risk_aversion: float      # [0,1]
    exploration_rate: float   # [0,1]
    memory_depth: float       # [0,1]
    prediction_weight: float  # [0,1]
```

**Cognitive Architecture α (HOW the agent thinks):**
```python
@dataclass
class CognitiveArchitecture:
    attention_history: float     # Weight on historical info
    attention_prediction: float  # Weight on predictions
    attention_immediate: float   # Weight on immediate signals
    memory_update_rate: float    # Adaptation speed
    perceptual_gain: float       # Signal sensitivity
```

## Dual Gradient Updates

Both optimize for survival with existential priority:

```python
lambda_e = 0.8  # Existential weight (high)
lambda_r = 0.2  # Reward weight (low)

# Theta update
delta_theta = lambda_e * grad_SAI_theta - lambda_r * grad_reward_theta

# Alpha update
delta_alpha = lambda_e * grad_SAI_alpha - lambda_r * grad_reward_alpha
```

## Perturbation System

Three shock types test resilience:

| Shock Type | Effect | Expected θ Response | Expected α Response |
|------------|--------|---------------------|---------------------|
| `history` | Scrambles memory | ↑ memory_depth | ↑ attention_history |
| `prediction` | Adds noise | ↑ prediction_weight | ↑ attention_prediction |
| `identity` | Direct IC damage | ↑ risk_aversion | ↑ perceptual_gain |

## Metrics

- **SAI_dyn**: Survival probability after perturbations
- **EAS_dyn**: Agency coherence during shocks
- **IC**: Identity Coherence (low variance = stable self)
- **Plasticity**: Adaptation speed
- **Conflict**: corr(SAI, reward) - should be negative

## Experimental Conditions

| Condition | θ Adapts | α Adapts | Perturbations |
|-----------|----------|----------|---------------|
| `full_adaptive` | Yes | Yes | Normal |
| `meta_only` | Yes | No | Normal |
| `cognitive_only` | No | Yes | Normal |
| `no_adaptation` | No | No | Normal |
| `perturbed` | Yes | Yes | Severe |

## Self-Evidence Criteria

1. SAI_dyn > 0.5
2. full_adaptive >> meta_only (α adds value)
3. full_adaptive >> cognitive_only (θ adds value)
4. full_adaptive >> no_adaptation
5. IC > 0.6 (identity stabilizes)
6. Plasticity in [0.05, 0.5] (adaptive but not chaotic)
7. Conflict < 0 (existence over reward)
8. Survives severe perturbations

**Passing**: ≥6/8 for "evidence of adaptive self"

## Baseline Results

```
Condition        SAI_dyn    Plasticity    Final risk_aversion    Final attn_prediction
-----------------------------------------------------------------------------------------
full_adaptive    0.000      0.100         0.78 (↑56%)            0.38 (↑15%)
meta_only        0.000      0.071         0.73 (↑46%)            0.33 (unchanged)
cognitive_only   0.000      0.016         0.50 (unchanged)       0.37 (↑12%)
no_adaptation    0.000      0.000         0.50 (unchanged)       0.33 (unchanged)
perturbed        0.000      0.079         0.72 (↑44%)            0.37 (↑12%)
```

**Passed: 1/8 criteria**

## Key Observations

1. **Both systems adapt in correct directions**:
   - θ: risk_aversion increases (0.50 → 0.78)
   - α: attention_prediction increases (0.33 → 0.38)

2. **Dual > Single**: full_adaptive shows more plasticity (0.100) than meta_only (0.071) or cognitive_only (0.016)

3. **SAI_dyn = 0**: Existential threat too severe for baseline survival, but adaptation mechanism works

4. **Framework established**: Ready for enhanced mechanisms that could achieve survival

## Philosophical Significance

IPUESA-AE tests whether the agent can adapt both:
- WHO it is (policy/behavior)
- HOW it thinks (cognition/attention)

If `full_adaptive >> meta_only`, then cognitive architecture provides **additional resilience** beyond policy adaptation alone. The agent becomes both the author of its identity AND the architect of its mind.

## Files

- `experiments/consciousness/exp_ipuesa_ae.py` - Implementation
- `results/ipuesa_ae_results.json` - Output data
