# IPUESA-X: Exploratory Self-Expansion Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Self emerges when the agent can create its own survival tools (micro-modules) in addition to adapting WHO it is (theta) and HOW it thinks (alpha). Emergent modules extend the agent's capabilities beyond predefined systems.

## The Extension

| IPUESA-AE | IPUESA-X |
|-----------|----------|
| "I adapt WHO I am and HOW I think" | "I adapt WHO I am, HOW I think, AND WHAT EMERGES from me" |
| Policy + Architecture | Policy + Architecture + Emergent Modules |
| theta + alpha | theta + alpha + beta |

## Triple Adaptation Architecture

**Meta-Policy theta (WHO the agent is):**
```python
@dataclass
class MetaPolicy:
    risk_aversion: float      # [0,1]
    exploration_rate: float   # [0,1]
    memory_depth: float       # [0,1]
    prediction_weight: float  # [0,1]
```

**Cognitive Architecture alpha (HOW the agent thinks):**
```python
@dataclass
class CognitiveArchitecture:
    attention_history: float     # Weight on historical info
    attention_prediction: float  # Weight on predictions
    attention_immediate: float   # Weight on immediate signals
    memory_update_rate: float    # Adaptation speed
    perceptual_gain: float       # Signal sensitivity
```

**Emergent Micro-Modules beta (WHAT EMERGES):**
```python
@dataclass
class MicroModule:
    id: int
    type: str                    # Module function type
    strength: float              # [0,1] activation strength
    specificity: float           # [0,1] specialization
    creation_step: int           # When created
    contribution: float          # Running survival contribution
    trigger_signature: np.ndarray
    consolidated: bool           # Whether permanent
```

## Module Types

| Type | Function | When Created |
|------|----------|--------------|
| `pattern_detector` | Recognizes threat patterns | High history corruption |
| `threat_filter` | Attenuates threat signals | General threat |
| `recovery_accelerator` | Speeds up IC recovery | Low IC level |
| `exploration_dampener` | Reduces exploration under stress | High urgency |

## Module Lifecycle

```
Novel Threat + SAI < 0.5  -->  CREATE module
         |
         v
    Module Active
         |
    contribution tracked
         |
    +----+----+
    |         |
    v         v
contribution > 0.3    contribution < -0.1
    |                      |
    v                      v
CONSOLIDATE           FORGET
(permanent)           (remove)
```

## Perturbation System (Extended)

Five perturbation types (including two novel):

| Type | Effect | Response |
|------|--------|----------|
| `history` | Scrambles memory | pattern_detector creation |
| `prediction` | Adds noise | threat_filter creation |
| `identity` | Direct IC damage | recovery_accelerator creation |
| `catastrophic` | Damages all systems | Multiple module creation |
| `structural` | Damages adaptive capacity | exploration_dampener creation |

## Metrics

- **SAI_dyn**: Survival probability after perturbations
- **EAS_dyn**: Agency coherence during shocks
- **ES**: Emergence Score = sum(strength * contribution) / n_modules
- **Module_diversity**: Shannon entropy of module types
- **Consolidation_rate**: Permanent modules / created modules

## Experimental Conditions

| Condition | theta | alpha | beta | Perturbations |
|-----------|-------|-------|------|---------------|
| `full_expansion` | Yes | Yes | Yes | Normal |
| `meta_only` | Yes | No | No | Normal |
| `cognitive_only` | No | Yes | No | Normal |
| `no_expansion` | No | No | No | Normal |
| `perturbed` | Yes | Yes | Yes | Extreme |

## Self-Evidence Criteria

1. SAI_dyn > 0.5
2. full_expansion >> meta_only
3. full_expansion >> cognitive_only
4. full_expansion >> no_expansion
5. ES > 0.1 (modules contribute)
6. Module_diversity > 0.3
7. Consolidation_rate > 0.1
8. Plasticity in [0.05, 0.5]
9. Survives extreme perturbations

**Passing**: >=7/9 for "evidence of exploratory self"

## Baseline Results

```
Condition        SAI_dyn    ES      Mod_Div   Consol   Modules
----------------------------------------------------------------
full_expansion   0.000      0.000   -0.000    0.000    4.0
meta_only        0.000      0.000    0.000    0.000    0.0
cognitive_only   0.000      0.000    0.000    0.000    0.0
no_expansion     0.000      0.000    0.000    0.000    0.0
perturbed        0.000      0.000    0.087    0.000    6.0
```

**Passed: 1/9 criteria**

## Key Observations

1. **Modules ARE created**: 4-6 modules created per run
2. **More modules under stress**: perturbed creates ~50% more modules
3. **Diversity emerges under pressure**: perturbed shows 0.087 diversity
4. **Adaptation directions correct**: theta risk_aversion 0.50->0.75, alpha attention_prediction 0.33->0.37
5. **Framework established**: Ready for enhanced mechanisms

## Philosophical Significance

IPUESA-X tests whether the agent can be the **architect of its own extensions**. If `full_expansion >> meta_only` AND `ES > 0`, then the agent demonstrates:

1. **Creative agency**: Creating tools that didn't exist before
2. **Selective retention**: Keeping what works, forgetting what doesn't
3. **Adaptive specialization**: Modules specialize to specific threats

This is the beginning of genuine self-expansion - the agent extends its own cognitive capabilities in response to existential challenges.

## Files

- `experiments/consciousness/exp_ipuesa_x.py` - Implementation
- `results/ipuesa_x_results.json` - Output data
