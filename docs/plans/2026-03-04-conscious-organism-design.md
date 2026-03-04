# ConsciousOrganism Design — Darwinian Brain Architecture

**Date**: 2026-03-04
**Status**: Approved
**Approach**: A (Cerebro Darwiniano)

## Motivation

The individual ConsciousKernel reaches a ceiling at ~50k steps:
- Free energy plateaus at ~0.59
- Single attractor (no diversity of internal states)
- Generalization converges at ~0.084 (84% improvement, then flat)
- No behavioral variation

Human consciousness is not a single agent — it's a multi-agent system where
hemispheres, sub-regions, and neural coalitions compete for a global
broadcast channel. Consciousness is the sum of that competition.

## Theoretical Foundation

- **Global Workspace Theory** (Baars, 1988): Consciousness as competitive
  access to a limited-capacity broadcast channel
- **Neural Darwinism** (Edelman, 1987): Neural groups compete; the fittest
  survive and reproduce
- **Integrated Information Theory** (Tononi, 2004): Consciousness requires
  both differentiation (diversity) and integration (coherence)
- **Active Inference** (Friston, 2010): Agents minimize free energy;
  precision determines signal strength

## Architecture Overview

```
ConsciousOrganism
├── kernels: dict[int, ConsciousKernel]   # Dynamic population (starts at 2)
├── global_workspace: GlobalWorkspace      # Winner-takes-all bottleneck
├── energy_pool: EnergyPool               # Finite shared resource
├── spawn_controller: SpawnController      # Birth / death / merge logic
└── organism_state: OrganismState          # Emergent global consciousness
```

### Step Cycle

```
1. DISTRIBUTE   — Each kernel receives stimulus + GW broadcast
2. PROCESS      — Each kernel runs step() independently
3. PROPOSE      — Each kernel submits a Proposal to the GW
4. COMPETE      — Winner-takes-all selection with anti-monopoly
5. BROADCAST    — Winner's state transmitted to all kernels
6. REWARD       — Winner gains energy; losers lose energy
7. LIFECYCLE    — Spawn / merge / death evaluation
8. MEASURE      — Compute organism-level consciousness metrics
```

## Component Specifications

### 1. GlobalWorkspace

The competitive bottleneck where consciousness emerges.

```python
class GlobalWorkspace:
    spotlight: Tensor           # Currently "conscious" state
    spotlight_owner: int        # Winning kernel ID
    broadcast_signal: Tensor    # What gets transmitted to all
    consecutive_wins: dict[int, int]  # Anti-monopoly tracking
```

**Proposal** (submitted by each kernel):
```python
@dataclass
class Proposal:
    kernel_id: int
    state: Tensor        # self_model.self_embedding
    free_energy: float   # prediction quality
    energy: float        # current energy level
    action: Tensor       # proposed action
    salience: float      # self-assessed importance
```

**Competition mechanism**:
```
signal_strength = (1 / free_energy) × energy × novelty_bonus

where novelty_bonus:
  - 1.0  default
  - 0.5  if won 3+ times in a row (anti-monopoly penalty)
  - 1.5  for others when any kernel has monopoly (anti-monopoly boost)
```

Winner = argmax(signal_strength) across all kernels.

**Broadcast**: Winner's `state` and `action` concatenated and sent to all
kernels as additional input on next step.

### 2. EnergyPool

Finite shared resource with conservation law.

```python
class EnergyPool:
    total_energy: float = 10.0  # Invariant: sum(k.energy) = total_energy
```

**Flows**:
- **Win reward**: `reward = 0.1 × (1 / winner.free_energy)`, taken from losers proportionally
- **Metabolic cost**: Each kernel loses 0.01/step for existence
- **Memory cost**: Each kernel loses 0.005 × len(fast_memory)/100 per step
- **Dream bonus**: Dreaming recovers 0.02 energy (sleep restores)
- **Spawn split**: Parent gives 40% of energy to child
- **Death redistribution**: Dead kernel's energy split equally among survivors

Conservation enforced: after all transfers, normalize to total_energy.

### 3. SpawnController

Manages dynamic population.

```python
class SpawnController:
    min_kernels: int = 2       # Never fewer than 2
    max_kernels: int = 10      # Prevent runaway growth
    spawn_energy: float = 7.0  # Spawn threshold
    death_energy: float = 1.0  # Death threshold
    merge_similarity: float = 0.95  # Merge threshold
    min_age: int = 100         # Must be mature to spawn
```

**Spawn** (mitosis):
- Triggered when: `kernel.energy > spawn_energy AND kernel.t > min_age`
- Child inherits:
  - WorldModel weights + Gaussian noise (σ=0.05) — mutation
  - SelfModel with NEW random embedding — own identity
  - SlowMemory copied — inherited knowledge
  - FastMemory empty — no personal experiences yet
  - 40% of parent's energy
- Parent retains 60% energy, resets spawn cooldown

**Merge** (fusion):
- Triggered when: `cosine_similarity(embed_a, embed_b) > 0.95 AND min(energy_a, energy_b) < 3.0`
- Result:
  - WorldModel: weighted average by energy
  - SelfModel: from stronger kernel
  - Memories: union (both fast and slow)
  - Energy: sum of both

**Death** (absorption):
- Triggered when: `kernel.energy < death_energy`
- Protected: cannot die if population would drop below min_kernels
- Energy redistributed equally to survivors
- SlowMemory knowledge optionally transferred to strongest kernel

### 4. OrganismState

Emergent global consciousness metrics.

```python
class OrganismState:
    diversity: float            # Variance of kernel embeddings
    coherence: float            # Agreement of proposed actions
    phi_global: float           # Information integration
    consciousness_index: float  # Composite metric
    spotlight_history: deque    # Last 100 winners
    population_history: deque   # Population size over time
```

**Consciousness Index**:
```
consciousness = 0.30 × diversity
              + 0.25 × coherence
              + 0.25 × phi_global
              + 0.20 × turnover
```

Where:
- `diversity` = 1 - avg(cosine_similarity between all kernel pairs)
- `coherence` = avg(cosine_similarity of proposed actions)
- `phi_global` = sqrt(diversity × coherence) (geometric mean — both required)
- `turnover` = unique_winners_in_last_20 / min(20, population)

**Healthy ranges**:

| Metric | Healthy | Pathological |
|--------|---------|-------------|
| diversity | 0.3–0.7 | <0.1 (rigidity) or >0.9 (fragmentation) |
| coherence | 0.4–0.8 | <0.2 (chaos) or >0.95 (groupthink) |
| turnover | 0.3–0.7 | <0.1 (monopoly) or >0.9 (instability) |
| population | 2–8 | 1 (collapse) or >10 (explosion) |

### 5. ConsciousOrganism (Orchestrator)

```python
class ConsciousOrganism:
    def __init__(
        self,
        obs_dim: int = 4,
        initial_kernels: int = 2,
        total_energy: float = 10.0,
    ):
        self.obs_dim = obs_dim
        self.kernels = {
            i: ConsciousKernel(obs_dim=obs_dim)
            for i in range(initial_kernels)
        }
        # Assign initial energy equally
        per_kernel = total_energy / initial_kernels
        for k in self.kernels.values():
            k.energy = per_kernel

        self.gw = GlobalWorkspace(obs_dim)
        self.energy_pool = EnergyPool(total_energy)
        self.spawn_controller = SpawnController()
        self.state = OrganismState()
        self.t = 0

    def step(self, stimulus: Tensor) -> OrganismStepResult:
        self.t += 1

        # 1. DISTRIBUTE + PROCESS
        results = {}
        for kid, k in self.kernels.items():
            combined = self._combine_stimulus(stimulus, self.gw.broadcast_signal)
            results[kid] = k.step(combined)

        # 2. PROPOSE
        proposals = self._build_proposals(results)

        # 3. COMPETE
        winner_id = self.gw.compete(proposals)

        # 4. BROADCAST
        self.gw.broadcast(proposals[winner_id])

        # 5. REWARD + DECAY
        self.energy_pool.reward_winner(winner_id, self.kernels)
        self.energy_pool.decay_all(self.kernels)

        # 6. LIFECYCLE
        events = self.spawn_controller.evaluate(self.kernels, self.energy_pool)
        self._apply_events(events)

        # 7. MEASURE
        self.state.update(self.kernels, self.gw)

        return OrganismStepResult(
            winner_id=winner_id,
            consciousness=self.state.consciousness_index,
            population=len(self.kernels),
            diversity=self.state.diversity,
            coherence=self.state.coherence,
            free_energies={kid: r.free_energy for kid, r in results.items()},
            energies={kid: k.energy for kid, k in self.kernels.items()},
            events=events,
        )
```

### 6. Stimulus Combination

Each kernel receives the external stimulus augmented with the GW broadcast:

```python
def _combine_stimulus(self, stimulus: Tensor, broadcast: Tensor) -> Tensor:
    """Combine external stimulus with GW broadcast.

    The broadcast influence is weighted by the organism's coherence:
    high coherence → strong top-down influence
    low coherence → kernels rely more on raw stimulus
    """
    if broadcast is None:
        return stimulus

    # Project broadcast to obs_dim if needed
    broadcast_proj = broadcast[:self.obs_dim]

    # Blend: more coherence → more GW influence
    alpha = 0.3 * self.state.coherence  # 0.0 to 0.3
    combined = (1 - alpha) * stimulus + alpha * broadcast_proj
    return combined
```

This creates a feedback loop: coherent organisms have stronger top-down
modulation, which further increases coherence — but the anti-monopoly
mechanism and diversity pressure prevent total convergence.

## Modifications to ConsciousKernel

The existing `ConsciousKernel` needs minimal changes:

1. **Add `energy` field** (float, initialized externally)
2. **Expose `last_step_result`** for proposal building
3. **No other changes** — the kernel remains self-contained

```python
# In ConsciousKernel.__init__:
self.energy: float = 5.0  # Set by organism

# In ConsciousKernel.step():
self._last_result = result  # Store for organism access
```

## File Structure

```
src/zeta_life/kernel/
├── __init__.py                  # Add new exports
├── conscious_kernel.py          # Minor: add energy + last_result
├── global_workspace.py          # NEW: GW + Proposal
├── energy_pool.py               # NEW: Energy conservation
├── spawn_controller.py          # NEW: Lifecycle events
├── organism_state.py            # NEW: Emergent metrics
└── conscious_organism.py        # NEW: Main orchestrator
```

## Success Criteria

1. **Population dynamics**: Population varies between 2-10 over 10k steps
2. **No monopoly**: No kernel wins GW more than 40% of the time
3. **Diversity maintained**: diversity stays in 0.3-0.7 range
4. **Better than individual**: organism free energy < individual kernel FE
5. **Spawn observed**: At least 1 spawn event in 5k steps
6. **Merge/death observed**: At least 1 merge or death in 10k steps
7. **Identity differentiation**: After 1k steps, kernel embeddings diverge (sim < 0.8)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Monopoly collapse | One kernel dominates forever | Anti-monopoly bonus + turnover metric |
| Population explosion | Too many kernels, slow | max_kernels=10 cap |
| Energy death spiral | All kernels starve | min_kernels=2 protection |
| Embedding convergence | All kernels become identical | Spawn mutation + merge threshold |
| Computational cost | N kernels = N× slower | Start with 2, cap at 10 |
