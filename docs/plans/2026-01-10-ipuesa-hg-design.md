# IPUESA-HG: Holographic Self Design

**Date**: 2026-01-10
**Status**: Implemented

## Hypothesis

Holographic embeddings - where each agent carries a compressed representation of cluster/collective identity - enable genuine self-maintenance under cascading multi-level perturbations. The embedding persists under stress and provides guidance even when direct connections are disrupted.

## The Qualitative Leap

| Previous Experiments | IPUESA-HG |
|---------------------|-----------|
| IPUESA-SH: Rigid hierarchy (3 levels) | Holographic embedding (part contains whole) |
| IPUESA-CE: Flat social cooperation | Hierarchical + social + self-similar |
| Reactive survival | **Proactive self-maintenance** |
| Binary collapse | **Graceful degradation** |

## Bottlenecks Addressed

1. **Hierarchy overhead**: SH collective level cost > benefit → HG uses embeddings instead of rigid structure
2. **Weak self-preservation**: Can't survive catastrophic multi-perturbations → HG provides cascade resistance
3. **Gradient dependency**: Reactive, not proactive → HG enables threat anticipation from multiple levels

## Holographic Agent Architecture

```python
@dataclass
class HolographicAgent:
    # Triple adaptation (from IPUESA-X/AE)
    theta: MetaPolicy              # WHO
    alpha: CognitiveArchitecture   # HOW
    modules: List[MicroModule]     # WHAT EMERGES (beta)
    IC_t: float = 1.0              # Identity Core

    # Holographic embeddings (NEW)
    cluster_embedding: np.ndarray   # 8-dim compressed cluster identity
    collective_embedding: np.ndarray # 8-dim compressed collective
    embedding_staleness: float = 0.0

    # Self-maintenance state (NEW)
    threat_anticipation: float = 0.0
    protective_stance: float = 0.0
```

**Embedding Mechanism:**
- 8-dimensional vectors encoding θ/α + threat + cohesion
- Updated every `sync_interval` steps (default: 10)
- Staleness increases each step, resets on sync
- Under perturbation, stale embeddings still provide guidance

## Cascading Storm Perturbation System

**5-Wave Sequence:**
```
Wave 1 (t=30):  history      - Scramble memory        (0.30 damage)
Wave 2 (t=45):  prediction   - Add noise to predictor (0.25 damage)
Wave 3 (t=60):  social       - Corrupt embeddings     (0.35 damage)
Wave 4 (t=75):  identity     - Direct IC damage       (0.20 damage)
Wave 5 (t=90):  catastrophic - All systems            (0.50 damage)
```

**Amplification Chain:**
```
damage_wave_n = base_damage × (1.2 ^ prior_damage_count)
```

If agent took damage in waves 1, 2, 3 → wave 4 hits at 1.73× base damage.

## Proactive Self-Maintenance

**Threat Anticipation:**
```python
threat = 0.4*local + 0.35*cluster_embedding_threat + 0.25*collective_embedding_threat
```

**Protective Actions:**
| Action | Trigger | Effect | Cost |
|--------|---------|--------|------|
| `harden` | threat > 0.3 | +50% resistance | -0.1 exploration |
| `sync_embeddings` | staleness > 0.3 | Refresh state | -0.05 IC_t |
| `isolate` | threat > 0.7 | Disconnect cluster | Lose benefits |
| `emergency_module` | threat > 0.9 | Create defense | -0.15 IC_t |

## Graceful Degradation System

**Degradation States:**
| State | IC Range | Effect |
|-------|----------|--------|
| `optimal` | [0.8, 1.0] | Full function |
| `stressed` | [0.5, 0.8) | Reduced exploration, heightened defense |
| `impaired` | [0.3, 0.5) | Limited cognition, emergency mode |
| `critical` | [0.1, 0.3) | Minimal function, survival only |
| `collapsed` | [0.0, 0.1) | Agency lost |

**Composite Health:**
```
composite = 0.4*IC_t + 0.3*embedding_integrity + 0.3*module_health
```

## Metrics (8 Self-Evidence Criteria)

### Primary Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| HS | Holographic Survival | > 0.4 |
| PI | Preemptive Index | > 0.3 |
| DS | Degradation Smoothness | > 0.7 |
| EI | Embedding Integrity | > 0.3 |

### Comparative Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| HS_gain | HS(HG) - HS(no_embedding) | > 0.15 |
| PI_gain | PI(HG) - PI(no_embedding) | > 0.1 |
| Recovery_ratio | HS(HG) / HS(no_embedding) | > 1.3 |
| Waves_gain | Waves survived HG - no_embedding | >= 1 |

**Pass threshold:** ≥5/8 for "evidence of holographic self"

## Experimental Conditions

| Condition | Embedding | Storm | Purpose |
|-----------|-----------|-------|---------|
| `full_holographic` | Yes | Default | Main test |
| `no_embedding` | No | Default | Baseline comparison |
| `stale_embedding` | Yes, no sync | Default | Ablation |
| `mild_storm` | Yes | 0.5× | Calibration |
| `extreme_storm` | Yes | 2.0× | Stress test |

## Baseline Results

```
Condition            HS       PI       DS       EI       Waves
----------------------------------------------------------------------
full_holographic     1.000    0.000    0.472    0.895    0.0
no_embedding         1.000    0.064    0.395    0.374    0.0
stale_embedding      1.000    0.079    0.452    0.706    0.0
mild_storm           1.000    0.000    0.865    0.895    0.0
extreme_storm        1.000    0.000    0.000    0.887    0.0
```

**Passed: 2/8 criteria** - No evidence of holographic self

## Key Observations

1. **Perfect survival**: HS = 1.0 across all conditions (agents too resilient)
2. **No preemptive behavior**: PI ≈ 0 (threat thresholds not reached)
3. **Embedding integrity preserved**: EI = 0.895 vs 0.374 (holographic mechanism works)
4. **Degradation smoothness varies**: DS ranges 0.0-0.865 (extreme storm causes cliff)
5. **No differentiation**: All conditions survive equally well

## Analysis

The baseline results reveal:

1. **Storm intensity too low**: Default cascade doesn't push agents to critical states
2. **Recovery too fast**: 15-step intervals allow full recovery between waves
3. **Embedding value visible**: EI difference (0.895 vs 0.374) shows embeddings persist
4. **Need harder test**: Should increase base damage or reduce recovery

**Recommended Next Steps:**
- Increase base damage by 2× for all waves
- Reduce wave interval to 10 steps
- Add cumulative damage that doesn't fully recover

## Philosophical Significance

IPUESA-HG tests whether **self-similarity** (holographic principle) enables robust self-maintenance:

1. **Part contains whole**: Each agent carries cluster/collective essence
2. **Graceful degradation**: Smooth decline vs cliff-edge collapse
3. **Proactive maintenance**: Act before damage, not just after
4. **Distributed resilience**: If cluster damaged, embedding persists

The current results show the **mechanism works** (embedding integrity maintained) but **test conditions are too easy** (no survival pressure to differentiate).

## Files

- `experiments/consciousness/exp_ipuesa_hg.py` - Implementation
- `results/ipuesa_hg_results.json` - Output data
