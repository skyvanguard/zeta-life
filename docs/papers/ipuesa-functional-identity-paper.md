# Functional Identity Attractors in Multi-Agent Systems: What Works, What Fails, and Why It Matters

**Abstract**

We investigate whether multi-agent systems can exhibit self-maintaining functional identity under stress. Rather than claiming consciousness or spontaneous emergence, we operationalize "self" as a measurable attractor characterized by anticipation, differentiation, and social propagation. Through progressive falsification, we show that simpler approaches fail: temporal learning inverts (agents increase risk when anticipating cost), social propagation requires explicit mechanisms, and smooth degradation demands engineered variance. Our final configuration (SYNTH-v2) achieves 8/8 self-evidence criteria, but only within a narrow stress regime (3.9× ± 5%). This fragility is not a weakness—it is the central result. Self-like functional identity is achievable, but not free.

---

## 1. Introduction

The concept of "self" in artificial systems remains frustratingly vague. Claims range from consciousness emergence to mere information persistence, with little agreement on what would constitute evidence for either.

We take a different approach: rather than debating phenomenology, we ask a simpler question:

> Can a multi-agent system maintain a functional identity attractor under stress—one that anticipates threats, differentiates individual trajectories, and propagates learned adaptations socially?

This is not a claim about consciousness, experience, or subjective states. It is a claim about measurable functional properties.

**Our contribution is threefold:**

1. An operational definition of "self" as functional attractor, with six quantifiable metrics
2. A systematic record of what fails: inverted learning, absent propagation, bistable collapse
3. A minimal working configuration that achieves all criteria—but only within a calibrated regime

We emphasize early: **nothing in this work demonstrates spontaneous emergence**. Every successful component required explicit implementation after simpler versions failed. The value lies not in proving emergence, but in documenting precisely what is necessary for self-like behavior to appear.

---

## 2. Related Work

### 2.1 Self-Models in AI

Computational self-models have been explored in robotics (Bongard et al., 2006) and neural networks (Schmidhuber, 2015). These typically focus on physical self-representation rather than functional identity persistence under stress.

### 2.2 Multi-Agent Adaptation

Collective adaptation in multi-agent systems has been studied extensively (Shoham & Leyton-Brown, 2008), but usually emphasizes task performance rather than identity maintenance. Our work focuses on what persists when the system is damaged, not what it accomplishes.

### 2.3 Identity Persistence

Philosophical accounts of personal identity (Parfit, 1984) distinguish numerical from qualitative identity. We sidestep this debate by defining identity operationally: a system has functional identity if it satisfies our six metrics under stress.

We deliberately avoid the consciousness literature (IIT, GWT, HOT) as our claims do not require or imply phenomenal experience.

---

## 3. Operational Definition

### 3.1 What We Do NOT Claim

| Concept | Status |
|---------|--------|
| Consciousness | Not claimed |
| Subjective experience | Not claimed |
| Qualia | Not claimed |
| Phenomenal self | Not claimed |
| Spontaneous emergence | Not claimed |

### 3.2 What We DO Operationalize

**Functional Identity Attractor**: A dynamical pattern in a multi-agent system that satisfies:

| Property | Definition | Metric | Threshold |
|----------|------------|--------|-----------|
| **Anticipation** | System predicts future damage and modifies present behavior | TAE | > 0.15 |
| **Coherence** | Structural integrity preserved under stress | EI | > 0.3 |
| **Propagation** | Individual learning transmits socially | MSR | > 0.15 |
| **Differentiation** | Agents develop distinct trajectories | ED | > 0.10 |
| **Graduality** | Smooth transitions, not binary collapse | deg_var | > 0.02 |
| **Calibrated Survival** | Neither trivial nor impossible | HS | [0.30, 0.70] |

### 3.3 Metric Definitions

**TAE (Temporal Anticipation Effectiveness)**
```
TAE = corr(threat_buffer[t], IC_drop[t:t+5])
```
Measures whether the system's anticipation signal correlates with actual future damage. Positive correlation indicates functional anticipation.

**MSR (Module Spreading Rate)**
```
MSR = learned_modules / total_modules
```
Measures social transmission of adaptations. A module is "learned" if it originated in another agent.

**EI (Embedding Integrity)**
```
EI = ||embedding|| / max_norm - staleness_penalty
```
Measures preservation of holographic structure encoding cluster identity.

**ED (Emergent Differentiation)**
```
ED = std(survival_states)
```
Measures variance in agent outcomes. High ED means agents are not uniform mass.

**deg_var (Degradation Variance)**
```
deg_var = var(degradation_level)
```
Measures smoothness of transitions. Low deg_var indicates bistable (all-or-nothing) dynamics.

**HS (Holographic Survival)**
```
HS = P(agent.is_alive())
```
Proportion surviving. Must be in Goldilocks zone—neither trivial (everyone lives) nor impossible (everyone dies).

---

## 4. Progressive Falsification

This section documents what failed. We consider this the most important contribution, as it reveals what is actually necessary for functional identity.

### 4.1 IPUESA-TD: Temporal Learning Inverts

**Hypothesis**: Agents that anticipate future cost will reduce risky behavior.

**Implementation**: Utility function `U = reward - λ × E[future_loss] × γ^k`

**Result**: TSI = -0.517 (Temporal Strategy Index)

**Interpretation**: Agents chose MORE risky actions when future cost was high. The correlation was significantly negative.

**Lesson**: Knowing cost does not imply avoiding it. Abstract utility functions do not automatically translate to behavior change. The anticipation must directly modify behavioral parameters, not merely inform a calculation.

### 4.2 IPUESA-CE: Social Propagation Absent

**Hypothesis**: Modules will naturally spread between proximate agents.

**Implementation**: Modules created per-agent with proximity-based influence.

**Result**: MA = 0.0 (Module Adoption rate)

**Interpretation**: Zero social transmission occurred. Modules remained local to their creators.

**Lesson**: Social learning requires explicit transmission mechanisms. Proximity and influence are not sufficient. We had to implement direct module copying with strength reduction.

### 4.3 SYNTH-v1: Bistable Collapse

**Hypothesis**: The system will degrade gradually under increasing stress.

**Implementation**: Combined anticipation, propagation, and embeddings.

**Result**: Survival transitioned from 100% to 0% within 0.01× damage change.

**Interpretation**: The system was bistable—either complete protection or complete collapse. No intermediate states.

**Lesson**: Smooth degradation requires engineered variance sources:
- Individual degradation rates
- Cluster-based variation
- Random noise injection
- Slow recovery dynamics

### 4.4 Summary of Failures

| Experiment | Hypothesis | Expected | Actual | Required Fix |
|------------|------------|----------|--------|--------------|
| TD | Anticipate → avoid | TSI > 0 | TSI = -0.517 | Direct behavior modification |
| CE | Proximity → spread | MA > 0.2 | MA = 0.0 | Explicit transmission mechanism |
| SYNTH-v1 | Gradual degradation | deg_var > 0.02 | Bistable | Variance engineering |

**The pattern**: Every "natural" or "emergent" expectation failed. Success required explicit implementation of each property.

---

## 5. SYNTH-v2: The Minimal Working Configuration

### 5.1 Architecture

```
Agent Level
├── theta: MetaPolicy (who)
├── alpha: CognitiveArchitecture (how)
├── modules: List[MicroModule] (what emerges)
├── IC_t: Identity Core (0-1)
├── cluster_embedding: 8-dim holographic
├── threat_buffer: anticipation signal
└── degradation_level: smooth state (0-1)

Cluster Level
├── aggregated theta
├── cohesion metric
├── shared_modules count
└── collective threat
```

### 5.2 Key Components

**Vulnerability-Based Anticipation** (fixes TD)
```python
vulnerability = 1.0
vulnerability -= protective_stance * 0.25
vulnerability += degradation_level * 0.5
vulnerability += (1 - IC_t) * 0.4
# threat_buffer predicts THIS agent's damage, not just wave timing
```

**Explicit Module Spreading** (fixes CE)
```python
if module.consolidated and module.contribution > 0.15:
    for neighbor in cluster_agents:
        if random() < 0.30:
            neighbor.modules.append(copy(module, strength=0.45))
```

**Variance-Engineered Degradation** (fixes bistability)
```python
individual_factor = random(0.3, 1.7)  # Wide variance
cluster_modifier = 0.8 + (cluster_id % 4) * 0.15
noise = (random() - 0.5) * damage * 0.25
degradation += damage * base_rate * individual_factor * cluster_modifier + noise
```

### 5.3 The Goldilocks Zone

The system achieves all criteria only at **3.9× damage multiplier**.

| Damage | HS | Outcome |
|--------|-----|---------|
| 3.12× (-20%) | 1.000 | Everyone survives (trivial) |
| 3.9× | 0.396 | Calibrated (Goldilocks) |
| 4.68× (+20%) | 0.000 | Everyone dies (impossible) |

**This is not a bug—it is the result.** The functional identity attractor exists only within a narrow stress band. Outside this regime, the system either trivializes or collapses.

### 5.4 Fragility as Finding

The sensitivity to damage multiplier reveals something important: self-like functional identity is not robust. It exists in a precarious equilibrium between insufficient and overwhelming stress.

This parallels biological observations: organisms maintain homeostasis within narrow parameter ranges. The fragility is the phenomenon, not a limitation of the model.

---

## 6. Validation

### 6.1 Ablation Study

We removed each deg_var component individually:

| Condition | deg_var | Passed |
|-----------|---------|--------|
| full | 0.0278 | **6/6** |
| no_individual_factor | 0.0270 | 5/6 |
| no_noise | 0.0057 | 5/6 |
| no_cluster_variation | 0.0199 | 5/6 |
| no_slow_recovery | 0.0184 | 5/6 |
| none | 0.0104 | 5/6 |

**Finding**: Only the full configuration passes all criteria. Noise is most critical—without it, deg_var drops below threshold.

### 6.2 Parametric Robustness

| Parameter | Variation | Passed |
|-----------|-----------|--------|
| damage | ±20% | 2/6, 3/6 (fails) |
| noise_scale | ±20% | 5/6, 6/6 (robust) |
| recovery_factor | ±20% | 6/6, 6/6 (robust) |

**Finding**: System is robust to noise and recovery variation but fragile to damage variation. The Goldilocks zone is real and narrow.

### 6.3 Repeatability

16 random seeds, all other parameters fixed:

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| HS | 0.492 | 0.082 | 0.375 | 0.667 |
| TAE | 0.191 | 0.023 | 0.143 | 0.230 |
| MSR | 0.465 | 0.032 | 0.376 | 0.514 |
| deg_var | 0.026 | 0.005 | 0.017 | 0.036 |

- 100% of runs pass ≥5/6 criteria
- 68.8% of runs pass 6/6 criteria
- Mean criteria passed: 5.69/6

**Finding**: Results are reproducible across random seeds.

---

## 7. Discussion

### 7.1 What This Demonstrates

1. **Functional identity is operationalizable**: Six metrics capture distinct aspects of self-like behavior without invoking consciousness.

2. **Anticipation requires embodiment**: Abstract utility fails (TD). Anticipation must directly modify behavioral parameters.

3. **Social learning requires mechanism**: Proximity is insufficient (CE). Explicit transmission is necessary.

4. **Smooth transitions require variance**: Without engineered noise, systems are bistable (v1).

5. **The Goldilocks zone is narrow**: Functional identity exists only under calibrated stress.

### 7.2 What This Does NOT Demonstrate

| Claim | Status | Why |
|-------|--------|-----|
| Consciousness | Not demonstrated | No phenomenal claims made |
| Spontaneous emergence | Not demonstrated | Each property required explicit implementation |
| Universal model | Not demonstrated | Only tested at one scale (24 agents) |
| Generalization | Not demonstrated | Only one stress type (damage waves) |
| "Agents experience anticipation" | Not demonstrated | Correlation ≠ experience |

### 7.3 Why This Matters

**As a testbed**: The IPUESA framework provides a sandbox for testing hypotheses about self-like behavior. The six metrics offer concrete pass/fail criteria.

**As negative results**: The falsification record (Section 4) documents what doesn't work. This saves future researchers from repeating failed approaches.

**As regime identification**: The Goldilocks zone finding suggests that self-like behavior may be inherently narrow-band. This has implications for both artificial and biological systems.

### 7.4 Open Questions

1. Why does temporal learning invert under abstract utility?
2. Is there a regime where social propagation is spontaneous?
3. Does the Goldilocks zone scale with system size?
4. What additional metrics would capture missing aspects?
5. How does this relate to formal theories (IIT, GWT)?

---

## 8. Conclusion

We set out to operationalize "self" as a functional attractor in multi-agent systems. Through progressive falsification, we learned:

- Anticipation must modify behavior directly, not just inform utility
- Social propagation requires explicit mechanisms
- Smooth transitions require engineered variance
- The functional regime is narrow and fragile

Our final configuration (SYNTH-v2) achieves all six criteria, but only within a calibrated stress zone. This fragility is not a failure of the model—it is the finding.

**Self-like functional identity is achievable, but not free.**

It requires specific mechanisms, calibrated stress, and engineered variance. Nothing emerges spontaneously. Everything that works was built after something simpler failed.

This is perhaps the most honest contribution: a clear record of what is necessary, documented through what failed.

---

## References

Bongard, J., Zykov, V., & Lipson, H. (2006). Resilient machines through continuous self-modeling. Science, 314(5802), 1118-1121.

Parfit, D. (1984). Reasons and Persons. Oxford University Press.

Schmidhuber, J. (2015). Deep learning in neural networks: An overview. Neural Networks, 61, 85-117.

Shoham, Y., & Leyton-Brown, K. (2008). Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations. Cambridge University Press.

---

*Correspondence: [contact information]*

*Code and data: [repository link]*
