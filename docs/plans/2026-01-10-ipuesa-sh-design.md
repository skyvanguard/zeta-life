# IPUESA-SH: Self-Hierarchy Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Identity can exist at multiple hierarchical levels simultaneously. An agent's self emerges not just from its own adaptation, but from its position within clusters and the collective - creating a three-level identity structure.

## The Qualitative Leap

| IPUESA-CE | IPUESA-SH |
|-----------|-----------|
| "I adapt in relation to OTHERS" | "I adapt within LEVELS of organization" |
| Flat social structure | Hierarchical identity structure |
| Individual θ/α + social dynamics | Individual + Cluster + Collective identities |

## Three-Level Identity Architecture

**Level Structure:**
```python
@dataclass
class IndividualIdentity:
    agent_id: int
    theta: MetaPolicy           # WHO at individual level
    alpha: CognitiveArchitecture # HOW at individual level
    IC_t: float = 1.0
    history_corruption: float = 0.0
    prediction_noise: float = 0.0

@dataclass
class ClusterIdentity:
    cluster_id: int
    member_ids: Set[int]
    theta_cluster: MetaPolicy   # Aggregated WHO
    alpha_cluster: CognitiveArchitecture  # Aggregated HOW
    cohesion: float = 0.5
    specialization: str = 'balanced'
    IC_cluster: float = 1.0

@dataclass
class CollectiveIdentity:
    theta_collective: MetaPolicy
    alpha_collective: CognitiveArchitecture
    global_coherence: float = 0.5
    collective_purpose: str = 'survival'
    IC_collective: float = 1.0
```

## Bi-Directional Influence

### Bottom-Up Aggregation
| Level | Aggregation Method |
|-------|-------------------|
| Individual → Cluster | Weighted mean by agent IC_t |
| Cluster → Collective | Weighted mean by cluster cohesion × size |

### Top-Down Modulation
| Level | Effect |
|-------|--------|
| Collective → Cluster | Blend cluster theta toward collective (strength × collective_coherence) |
| Cluster → Individual | Blend individual theta toward cluster (strength × cluster_cohesion) |

**Agent Autonomy/Conformity Balance:**
```python
effective_theta = autonomy * individual_theta + conformity * cluster_theta
# where autonomy + conformity = 1.0, default 0.7/0.3
```

## Dissonance Detection & Resolution

### Dissonance Types
| Type | Detection | Impact |
|------|-----------|--------|
| `local` | Individual-Cluster theta distance > 0.3 | Stress on individual |
| `systemic` | Cluster-Collective theta distance > 0.4 | Cluster cohesion loss |
| `crisis` | Both local AND systemic | Major restructuring trigger |

### Resolution Mechanisms
- **Local dissonance**: Agent increases conformity OR migrates to compatible cluster
- **Systemic dissonance**: Cluster adapts purpose OR collective adjusts expectations
- **Crisis**: Emergency restructuring - splits, merges, mass migration

## Dynamic Clustering

### Cluster Manager Operations
| Operation | Trigger | Effect |
|-----------|---------|--------|
| `migration` | High local dissonance + better fit elsewhere | Agent moves cluster |
| `split` | Cluster size > 8 AND cohesion < 0.3 | Divide into two |
| `merge` | Two clusters size < 3 AND similar theta | Combine into one |

## Multi-Level Perturbations

| Level | Types | Cascade |
|-------|-------|---------|
| `individual` | history, prediction, identity | Up to cluster |
| `cluster` | cohesion_attack, split_force | Down to members, up to collective |
| `collective` | purpose_shift, coherence_attack | Down to all |
| `cross_level` | disconnect, cascade_failure | All levels |

**Cascade Mechanics:**
- Individual damage > 0.5 → cluster cohesion -0.1
- Cluster cohesion < 0.3 → member IC_t damage 0.1
- Collective coherence < 0.3 → all cluster cohesion -0.05

## Metrics (8 Self-Evidence Criteria)

### Primary Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| VC | Vertical Coherence (alignment across levels) | > 0.5 |
| HR | Hierarchical Resilience (survival with hierarchy) | > 0.4 |
| ED | Emergent Diversity (cluster specialization variance) | > 0.3 |
| AD | Alignment without forced uniformity | > 0.5 |

### Comparative Criteria
| Criterion | Description |
|-----------|-------------|
| full >> no_cluster | Hierarchy adds value over flat |
| full >> no_collective | Middle level is essential |
| catastrophic HR > 0.2 | Hierarchy survives extreme stress |
| Cluster stability > 0.5 | Clusters persist over time |

**Pass threshold:** ≥5/8 for "evidence of hierarchical self-emergence"

## Experimental Conditions

| Condition | Cluster Level | Collective Level | Perturbations |
|-----------|---------------|------------------|---------------|
| `full_hierarchy` | Yes | Yes | Normal |
| `no_cluster` | No (direct individual-collective) | Yes | Normal |
| `no_collective` | Yes | No | Normal |
| `shuffled_links` | Yes (randomized membership) | Yes | Normal |
| `catastrophic_multi` | Yes | Yes | Multi-level extreme |

## Baseline Results

```
Condition            VC       HR       ED       AD       ClustStab  Pass
--------------------------------------------------------------------------------
full_hierarchy       0.962    0.021    0.275    0.930    1.000      3/8
no_cluster           0.886    0.079    0.000    0.500    0.000      1/8
no_collective        0.950    0.527    0.276    0.714    1.000      5/8
shuffled_links       0.958    0.015    0.274    0.936    1.000      3/8
catastrophic_multi   0.960    0.016    0.273    0.935    1.000      3/8
```

**Passed: 3/8 criteria** - No evidence of hierarchical self (insufficient resilience)

## Key Observations

1. **High vertical coherence**: VC ≈ 0.96 across conditions (hierarchy maintains alignment)
2. **Low hierarchical resilience**: HR ≈ 0.02 (individual survival ~5-6%)
3. **Near-threshold diversity**: ED ≈ 0.275 (close to 0.3 threshold)
4. **Strong alignment**: AD ≈ 0.93 in full hierarchy
5. **Perfect cluster stability**: Clusters persist but don't restructure
6. **Interesting finding**: no_collective passes 5/8 (best condition!)
   - Suggests collective level may add pressure without sufficient benefit
   - Two-level hierarchy (individual-cluster) may be optimal

## Philosophical Significance

IPUESA-SH tests whether identity can exist **across levels of organization**. The hierarchical dimension adds:

1. **Nested identity**: Self exists at individual, cluster, AND collective levels
2. **Vertical coherence**: Identity must be consistent across levels
3. **Emergent specialization**: Clusters develop distinct purposes
4. **Bi-directional causation**: Bottom-up emergence AND top-down constraint

The surprising result that `no_collective >> full_hierarchy` suggests:
- Hierarchy has optimal depth (possibly 2 levels, not 3)
- Adding levels increases coordination cost without proportional benefit
- Middle layer (clusters) provides sufficient structure for emergence

## Files

- `experiments/consciousness/exp_ipuesa_sh.py` - Implementation
- `results/ipuesa_sh_results.json` - Output data
