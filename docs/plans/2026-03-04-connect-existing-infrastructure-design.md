# Design: Connect Existing Infrastructure for Non-Trivial Proto-Language

**Date**: 2026-03-04
**Status**: Approved
**Goal**: Break the softmax pass-through (linearity=0.9999) by connecting already-built but disconnected components in the ConsciousKernel action selection pipeline.

## Problem Statement

The proto-language broadcast is essentially `softmax(stimulus)` -- a deterministic pass-through. The ConsciousKernel has rich infrastructure (PrecisionController, SlowMemory, SelfModel) that is built, tested, and maintained, but **never used in action selection**. Additionally:

- Grounding experiment shows no difference between causal/random feedback
- Organism doesn't outperform individual kernel on any compositionality metric
- Context effect is 0.000001 (effectively zero)

## Three Fronts

### Front 1: Rich Action Selection (break softmax linearity)

**File**: `src/zeta_life/kernel/conscious_kernel.py`

Replace the current ACT phase (lines 181-186) with precision-weighted, semantically-guided action:

```python
# New ACT phase (when use_rich_action=True)
recent_errors = self.error_engine.recent_errors()
precisions = self.precision_controller(stimulus, recent_errors)
weighted_stimulus = stimulus * precisions

semantic = self.slow_memory.generalize(stimulus)
latent_bias = self._latent_to_action(self.world_model.latent_state.detach())

combined = weighted_stimulus + semantic_weight * semantic + latent_weight * latent_bias
actual_self = F.softmax(combined, dim=-1)
```

**New parameter**: `use_rich_action: bool = False` (backward compatible)
**New parameter**: `semantic_weight: float = 0.2`

**Why it works**: PrecisionController learns to upweight reliable channels and downweight noisy ones. SlowMemory provides a semantic prior based on accumulated experience. Together they make action context-dependent.

### Front 2: Non-Linear Grounding (close causal loop)

**File**: `experiments/kernel/exp_grounding.py`

Enhance ReactiveEnvironment with non-linear dynamics:

```python
def step(self, action):
    delta = action.detach() - self.state
    self.state = self.state + self.reactivity * torch.tanh(delta * 3.0)
    obs = (self.state + torch.randn(self.obs_dim) * self.noise).abs()
    return obs / (obs.sum() + 1e-8)
```

Add fourth condition `grounded_rich` using `use_rich_action=True`.

**Why it works**: tanh(3x) creates non-linear dynamics where prediction matters. With rich action selection, the organism's broadcast depends on history, making causal feedback distinguishable from noise.

### Front 3: Weighted Broadcast (social emergence)

**File**: `src/zeta_life/kernel/global_workspace.py`

Add weighted broadcast mode alongside winner-takes-all:

```python
def broadcast_weighted(self, proposals: dict[int, Proposal]) -> None:
    total_salience = sum(p.salience for p in proposals.values())
    blended = sum(
        (p.salience / total_salience) * p.action
        for p in proposals.values()
    )
    self.broadcast_signal = blended.detach()
```

**File**: `src/zeta_life/kernel/conscious_organism.py`

Add `broadcast_mode: str = 'winner'` parameter. When `'weighted'`, use `broadcast_weighted()` instead of single winner.

**Why it works**: Weighted broadcast integrates perspectives from all kernels, creating a richer signal than any individual kernel could produce.

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| Linearity | 0.9999 | < 0.97 |
| Context Effect | 0.000001 | > 0.01 |
| FE grounded < ungrounded | FAIL | PASS |
| Org > Ind (bigram pred) | 0.93 = 0.93 | Org > Ind |

## Validation Experiments

1. `exp_compositionality.py --use-rich-action --latent-weight 0.2 --top-down-strength 0.5`
2. `exp_grounding.py --use-rich-action` (with non-linear env)
3. `exp_grammar.py` (with `broadcast_mode='weighted'`)

## Backward Compatibility

- `use_rich_action=False` (default): behavior identical to before
- `broadcast_mode='winner'` (default): behavior identical to before
- All existing 699 tests must pass unchanged
