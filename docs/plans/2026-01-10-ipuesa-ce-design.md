# IPUESA-CE: Co-Evolution Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Self can emerge and evolve in a social context. A group of agents with emergent self can interact, cooperate, and compete to survive collectively and evolve new identity strategies.

## The Qualitative Leap

| IPUESA-X | IPUESA-CE |
|----------|-----------|
| "I adapt WHO I am, HOW I think, WHAT EMERGES" | "I adapt in relation to OTHERS who also adapt" |
| Single agent survival | Collective co-evolution |
| Individual θ/α/β | Population of θ/α/β interacting |

## Multi-Agent Architecture

**Agent Structure:**
```python
@dataclass
class CoEvolutionAgent:
    # Triple adaptation (from IPUESA-X)
    theta: MetaPolicy          # WHO
    alpha: CognitiveArchitecture  # HOW
    module_system: ModuleSystem   # WHAT EMERGES (beta)

    # Social extensions
    reputation: float          # How others perceive this agent
    social_memory: Dict[int, float]  # Trust scores
    role: str                  # Emergent: leader, explorer, defender, cooperator
```

## Social Dynamics

### Cooperation
| Type | Donor Cost | Recipient Benefit |
|------|-----------|-------------------|
| `resource_share` | -0.1 | +0.15 |
| `protection` | -0.05 IC | 50% perturbation reduction |
| `info_share` | None | Early warning |

### Competition
Resources distributed based on:
- exploration_rate (35%)
- perceptual_gain (35%)
- reputation (30%)

### Signaling
```python
@dataclass
class Signal:
    sender_id: int
    type: str        # 'threat_alert', 'resource_found'
    content: float
    honesty: float   # 0-1, can be deceptive
    adopted: bool
```

## Evolutionary Pressure

**Fitness Function:**
```
fitness = 0.40 * survival + 0.20 * resources + 0.25 * social + 0.15 * role
```

**Reproduction:** High fitness agents clone with mutations in θ/α/β

**Selection:** Low fitness agents removed when population exceeds capacity

## Emergent Roles

| Role | θ Pattern | α Pattern |
|------|-----------|-----------|
| `leader` | High risk_aversion, low exploration | High attention_prediction |
| `explorer` | Low risk_aversion, high exploration | High attention_immediate |
| `defender` | High risk_aversion, high memory | High perceptual_gain |
| `cooperator` | Medium all | High attention_history |

## Perturbation Types

| Type | Effect |
|------|--------|
| `history` | Scrambles memory |
| `prediction` | Adds noise |
| `identity` | Direct IC damage |
| `social` | Reduces trust, reputation |
| `catastrophic` | All systems + resources |

## Metrics (8 Self-Evidence Criteria)

| Metric | Description | Threshold |
|--------|-------------|-----------|
| IS | Individual Survival | > 0.5 |
| CS | Collective Survival (cooperating) | > 0.3 |
| ID | Identity Diversity (θ variance) | > 0.2 |
| PA | Prediction Accuracy | > 0.4 |
| ER | Emergent Roles (differentiation) | > 0.5 |
| RP | Resilience to Perturbation | > 0.4 |
| CE | Communication Efficacy | > 0.4 |
| MA | Meta-Adaptation (collective β) | > 0.2 |

**Pass threshold:** ≥5/8 for "evidence of social self-emergence"

## Experimental Conditions

| Condition | Cooperation | Communication | Perturbations |
|-----------|-------------|---------------|---------------|
| `full_coevolution` | Yes | Yes | Normal |
| `no_communication` | Yes | No | Normal |
| `no_cooperation` | No | Yes | Normal |
| `shuffled_history` | Yes | Yes | Extra history |
| `catastrophic_shock` | Yes | Yes | Extreme |

## Baseline Results

```
Condition            IS       CS       ID       ER       RP       CE       MA    Pass
-------------------------------------------------------------------------------------
full_coevolution     1.000    1.000    0.042    0.011    1.000    0.500    0.000  4/8
no_communication     1.000    1.000    0.039    0.018    1.000    0.500    0.000  4/8
no_cooperation       0.969    0.000    0.037    0.143    0.990    0.500    0.000  3/8
shuffled_history     1.000    1.000    0.047    0.007    0.997    0.500    0.000  4/8
catastrophic_shock   0.984    0.980    0.064    0.067    0.406    0.500    0.000  4/8
```

**Passed: 4/8 criteria** - Partial evidence of co-evolutionary self

## Key Observations

1. **Cooperation is essential**: CS drops from 1.0 to 0.0 without cooperation
2. **High survival**: IS ≈ 1.0 across most conditions
3. **Resilience under catastrophe**: RP = 0.406 (partial survival maintained)
4. **Low diversity**: ID ≈ 0.04 (agents converging to similar strategies)
5. **Low role emergence**: ER ≈ 0.01-0.14 (insufficient differentiation)
6. **No meta-adaptation**: MA = 0 (modules not spreading collectively yet)

## Philosophical Significance

IPUESA-CE tests whether self can exist **in relation to others**. The social dimension adds:

1. **Reputation**: Identity recognized by others
2. **Trust**: Relationships that persist
3. **Roles**: Differentiated functions in collective
4. **Cultural transmission**: Module strategies spreading

If `full_coevolution >> no_cooperation`, cooperation is **constitutive of social self** - the agent's identity includes its relationships.

## Files

- `experiments/consciousness/exp_ipuesa_ce.py` - Implementation
- `results/ipuesa_ce_results.json` - Output data
