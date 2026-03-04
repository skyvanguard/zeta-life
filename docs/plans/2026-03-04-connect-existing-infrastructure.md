# Connect Existing Infrastructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Break the softmax pass-through (linearity=0.9999) by connecting PrecisionController, SlowMemory, and weighted broadcast to the action selection pipeline.

**Architecture:** Three parallel fronts: (1) Rich action selection in ConsciousKernel using precision-weighted stimulus + semantic guidance, (2) Non-linear reactive environment for grounding, (3) Weighted broadcast in GlobalWorkspace for social emergence. All behind opt-in flags for backward compat.

**Tech Stack:** PyTorch, numpy, matplotlib, scipy

---

### Task 1: Rich Action Selection — Tests

**Files:**
- Modify: `tests/test_conscious_kernel.py`

**Step 1: Write failing tests for `use_rich_action`**

Add `TestRichAction` class after `TestLatentBias` in `tests/test_conscious_kernel.py`:

```python
class TestRichAction:
    """Tests for precision-weighted semantic action selection."""

    def test_default_rich_action_off(self):
        ck = _make_kernel()
        assert ck.use_rich_action is False

    def test_rich_action_param_accepted(self):
        ck = _make_kernel(use_rich_action=True, semantic_weight=0.3)
        assert ck.use_rich_action is True
        assert ck.semantic_weight == 0.3

    def test_rich_action_off_is_pure_softmax(self):
        """use_rich_action=False should produce softmax(stimulus) exactly."""
        ck = _make_kernel(use_rich_action=False, latent_weight=0.0)
        stimulus = torch.tensor([1.0, 0.0, 0.5, 0.2])
        result = ck.step(stimulus)
        expected = F.softmax(stimulus, dim=-1)
        assert torch.allclose(result.action, expected, atol=1e-5)

    def test_rich_action_differs_from_softmax(self):
        """use_rich_action=True should produce action != softmax(stimulus)."""
        ck = _make_kernel(use_rich_action=True, semantic_weight=0.3)
        stimulus = torch.tensor([1.0, 0.0, 0.5, 0.2])
        # Warm up so slow_memory has data and precision_controller is non-trivial
        for _ in range(50):
            ck.step(stimulus)
        result = ck.step(stimulus)
        pure = F.softmax(stimulus, dim=-1)
        assert not torch.allclose(result.action, pure, atol=1e-3), (
            "Rich action should differ from pure softmax"
        )

    def test_rich_action_valid_distribution(self):
        """Rich action must produce valid probability distribution."""
        ck = _make_kernel(use_rich_action=True, semantic_weight=0.3)
        for _ in range(20):
            result = ck.step(torch.randn(4))
        assert result.action.sum().item() == pytest.approx(1.0, abs=1e-4)
        assert (result.action >= 0).all()

    def test_precision_controller_called_in_rich(self):
        """PrecisionController should be invoked when use_rich_action=True."""
        ck = _make_kernel(use_rich_action=True)
        stimulus = torch.tensor([0.7, 0.1, 0.1, 0.1])
        # After step, precision_controller has been called — verify it works
        ck.step(stimulus)
        recent = ck.error_engine.recent_errors()
        precisions = ck.precision_controller(stimulus, recent)
        assert precisions.shape == (4,)
        assert (precisions > 0).all()

    def test_semantic_weight_zero_still_uses_precision(self):
        """With semantic_weight=0, only precision weighting should modulate."""
        ck = _make_kernel(use_rich_action=True, semantic_weight=0.0)
        stimulus = torch.tensor([1.0, 0.0, 0.5, 0.2])
        for _ in range(30):
            ck.step(stimulus)
        result = ck.step(stimulus)
        pure = F.softmax(stimulus, dim=-1)
        # Precision weighting alone can shift the action
        assert result.action.shape == (4,)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_conscious_kernel.py::TestRichAction -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'use_rich_action'`

**Step 3: Commit failing tests**

```bash
git add tests/test_conscious_kernel.py
git commit -m "test(kernel): add failing tests for rich action selection"
```

---

### Task 2: Rich Action Selection — Implementation

**Files:**
- Modify: `src/zeta_life/kernel/conscious_kernel.py:98-142` (init) and `:181-186` (step ACT phase)

**Step 1: Add parameters to `__init__`**

In `conscious_kernel.py`, add `use_rich_action` and `semantic_weight` parameters to `__init__`:

```python
def __init__(
    self,
    obs_dim: int = 4,
    latent_dim: int = 32,
    embed_dim: int = 16,
    reflect_interval: int = 5,
    dream_interval: int = 50,
    save_interval: int = 100,
    latent_weight: float = 0.0,
    use_rich_action: bool = False,
    semantic_weight: float = 0.2,
) -> None:
    # ... existing assignments ...
    self.use_rich_action = use_rich_action
    self.semantic_weight = semantic_weight
```

**Step 2: Replace the ACT computation in `step()`**

Replace lines 181-186 (the `raw_self`/`actual_self` block) with:

```python
        predicted_self = self.self_model.predict_self(self.last_action)

        if self.use_rich_action:
            # Precision-weighted stimulus
            recent_errors = self.error_engine.recent_errors()
            precisions = self.precision_controller(stimulus, recent_errors)
            weighted_stim = stimulus * precisions

            # Semantic guidance from slow memory
            semantic = self.slow_memory.generalize(stimulus)

            # Latent world context
            latent_bias = self._latent_to_action(
                self.world_model.latent_state.detach()
            )

            combined = (
                weighted_stim
                + self.semantic_weight * semantic
                + self.latent_weight * latent_bias
            )
            actual_self = F.softmax(combined, dim=-1)
        else:
            raw_self = F.softmax(stimulus, dim=-1)
            if self.latent_weight > 0.0:
                latent_bias = self._latent_to_action(
                    self.world_model.latent_state.detach()
                )
                actual_self = F.softmax(
                    raw_self + self.latent_weight * latent_bias, dim=-1
                )
            else:
                actual_self = raw_self
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_conscious_kernel.py -v`
Expected: ALL PASS (existing + new TestRichAction)

**Step 4: Commit**

```bash
git add src/zeta_life/kernel/conscious_kernel.py
git commit -m "feat(kernel): add precision-weighted semantic action selection"
```

---

### Task 3: Wire `use_rich_action` Through ConsciousOrganism

**Files:**
- Modify: `src/zeta_life/kernel/conscious_organism.py:58-98` (init) and `:174-180` (spawn)
- Modify: `tests/test_conscious_organism.py`

**Step 1: Write failing test**

Add to `tests/test_conscious_organism.py`:

```python
class TestRichActionOrganism:
    """Tests for use_rich_action passed through to kernels."""

    def test_default_off(self):
        org = ConsciousOrganism()
        assert org.use_rich_action is False

    def test_passes_to_kernels(self):
        org = ConsciousOrganism(use_rich_action=True, semantic_weight=0.3)
        for k in org.kernels.values():
            assert k.use_rich_action is True
            assert k.semantic_weight == 0.3

    def test_runs_without_crash(self):
        org = ConsciousOrganism(use_rich_action=True, semantic_weight=0.2)
        for _ in range(50):
            org.step(torch.randn(4))
        assert org.t == 50
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_conscious_organism.py::TestRichActionOrganism -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'use_rich_action'`

**Step 3: Add params to `ConsciousOrganism.__init__`**

Add `use_rich_action: bool = False` and `semantic_weight: float = 0.2` to `__init__` signature. Store as `self.use_rich_action` and `self.semantic_weight`. Pass both to `ConsciousKernel(...)` in the initial kernel creation loop and in `_spawn()`.

**Step 4: Run all organism tests**

Run: `python -m pytest tests/test_conscious_organism.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/zeta_life/kernel/conscious_organism.py tests/test_conscious_organism.py
git commit -m "feat(kernel): wire use_rich_action through ConsciousOrganism"
```

---

### Task 4: Weighted Broadcast — Tests

**Files:**
- Create: `tests/test_global_workspace.py`

**Step 1: Write tests for weighted broadcast**

```python
"""Tests for GlobalWorkspace weighted broadcast."""
import torch
import pytest

from zeta_life.kernel.global_workspace import GlobalWorkspace, Proposal


def _make_proposal(kid: int, action: torch.Tensor, fe: float = 1.0,
                   energy: float = 5.0) -> Proposal:
    return Proposal(
        kernel_id=kid,
        state=torch.randn(16),
        free_energy=fe,
        energy=energy,
        action=action,
        salience=1.0 / (1.0 + fe),
    )


class TestWeightedBroadcast:
    def test_default_mode_is_winner(self):
        gw = GlobalWorkspace()
        assert gw.broadcast_mode == 'winner'

    def test_weighted_mode_accepted(self):
        gw = GlobalWorkspace(broadcast_mode='weighted')
        assert gw.broadcast_mode == 'weighted'

    def test_weighted_blends_proposals(self):
        gw = GlobalWorkspace(broadcast_mode='weighted')
        p1 = _make_proposal(0, torch.tensor([1.0, 0.0, 0.0, 0.0]), fe=1.0)
        p2 = _make_proposal(1, torch.tensor([0.0, 1.0, 0.0, 0.0]), fe=1.0)
        proposals = {0: p1, 1: p2}
        gw.compete(proposals)
        gw.broadcast_weighted(proposals)
        # Equal salience → blended should be ~[0.5, 0.5, 0, 0]
        assert gw.broadcast_signal[0].item() == pytest.approx(0.5, abs=0.01)
        assert gw.broadcast_signal[1].item() == pytest.approx(0.5, abs=0.01)

    def test_weighted_favors_high_salience(self):
        gw = GlobalWorkspace(broadcast_mode='weighted')
        p1 = _make_proposal(0, torch.tensor([1.0, 0.0, 0.0, 0.0]), fe=0.1)  # high salience
        p2 = _make_proposal(1, torch.tensor([0.0, 1.0, 0.0, 0.0]), fe=10.0)  # low salience
        proposals = {0: p1, 1: p2}
        gw.compete(proposals)
        gw.broadcast_weighted(proposals)
        # p1 has much higher salience → broadcast skewed toward [1,0,0,0]
        assert gw.broadcast_signal[0].item() > 0.7

    def test_winner_mode_unchanged(self):
        gw = GlobalWorkspace(broadcast_mode='winner')
        p1 = _make_proposal(0, torch.tensor([1.0, 0.0, 0.0, 0.0]), fe=0.1, energy=5.0)
        p2 = _make_proposal(1, torch.tensor([0.0, 1.0, 0.0, 0.0]), fe=10.0, energy=5.0)
        proposals = {0: p1, 1: p2}
        winner = gw.compete(proposals)
        gw.broadcast(proposals[winner])
        # Winner takes all — broadcast should be exactly the winner's action
        assert torch.allclose(gw.broadcast_signal, proposals[winner].action)
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_global_workspace.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'broadcast_mode'`

**Step 3: Commit failing tests**

```bash
git add tests/test_global_workspace.py
git commit -m "test(kernel): add failing tests for weighted broadcast"
```

---

### Task 5: Weighted Broadcast — Implementation

**Files:**
- Modify: `src/zeta_life/kernel/global_workspace.py`

**Step 1: Add `broadcast_mode` param and `broadcast_weighted` method**

In `GlobalWorkspace.__init__`, add `broadcast_mode: str = 'winner'` and store it.

Add method:

```python
def broadcast_weighted(self, proposals: dict[int, Proposal]) -> None:
    """Broadcast weighted blend of all proposals (alternative to winner-takes-all)."""
    total_salience = sum(p.salience for p in proposals.values())
    if total_salience < 1e-8:
        return
    blended = torch.zeros_like(next(iter(proposals.values())).action)
    for p in proposals.values():
        blended = blended + (p.salience / total_salience) * p.action
    self.broadcast_signal = blended.clone().detach()
    # Keep spotlight and history from winner for consistency
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_global_workspace.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/zeta_life/kernel/global_workspace.py
git commit -m "feat(kernel): add weighted broadcast mode to GlobalWorkspace"
```

---

### Task 6: Wire Weighted Broadcast in ConsciousOrganism

**Files:**
- Modify: `src/zeta_life/kernel/conscious_organism.py:58-104` (init) and `:106-150` (step)

**Step 1: Add `broadcast_mode` parameter**

In `ConsciousOrganism.__init__`, add `broadcast_mode: str = 'winner'`. Store as `self.broadcast_mode`. Pass to `GlobalWorkspace(obs_dim=obs_dim, broadcast_mode=broadcast_mode)`.

Wait — `GlobalWorkspace.__init__` doesn't take `broadcast_mode` in the constructor yet (Task 5 stores it as attribute). Actually Task 5 does add it to `__init__`. So just pass it through.

**Step 2: Modify `step()` to use weighted broadcast when configured**

In `ConsciousOrganism.step()`, replace:

```python
# 4. BROADCAST
self.gw.broadcast(proposals[winner_id])
```

with:

```python
# 4. BROADCAST
if self.broadcast_mode == 'weighted':
    self.gw.broadcast_weighted(proposals)
else:
    self.gw.broadcast(proposals[winner_id])
```

**Step 3: Write test**

Add to `tests/test_conscious_organism.py`:

```python
class TestBroadcastMode:
    def test_default_mode_winner(self):
        org = ConsciousOrganism()
        assert org.broadcast_mode == 'winner'

    def test_weighted_mode_runs(self):
        org = ConsciousOrganism(broadcast_mode='weighted')
        for _ in range(30):
            org.step(torch.randn(4))
        assert org.t == 30

    def test_weighted_broadcast_non_zero(self):
        org = ConsciousOrganism(broadcast_mode='weighted')
        org.step(torch.randn(4))
        assert org.gw.broadcast_signal.abs().sum().item() > 0
```

**Step 4: Run all tests**

Run: `python -m pytest tests/test_conscious_organism.py tests/test_global_workspace.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/zeta_life/kernel/conscious_organism.py tests/test_conscious_organism.py
git commit -m "feat(kernel): wire broadcast_mode through ConsciousOrganism"
```

---

### Task 7: Update Experiment CLI Args

**Files:**
- Modify: `experiments/kernel/exp_compositionality.py`
- Modify: `experiments/kernel/exp_grammar.py`
- Modify: `experiments/kernel/exp_grounding.py`

**Step 1: Add `--use-rich-action` and `--broadcast-mode` to all 3 experiments**

In each experiment's `argparse` block, add:

```python
parser.add_argument("--use-rich-action", action="store_true",
                    help="Enable precision-weighted semantic action")
parser.add_argument("--semantic-weight", type=float, default=0.2,
                    help="Semantic guidance weight (requires --use-rich-action)")
parser.add_argument("--broadcast-mode", choices=["winner", "weighted"],
                    default="winner", help="GW broadcast mode")
```

Pass these to `ConsciousOrganism(...)` and `ConsciousKernel(...)` constructors in each experiment.

**Step 2: Update `exp_grounding.py` ReactiveEnvironment with non-linear dynamics**

Replace the `step` method in `ReactiveEnvironment`:

```python
def step(self, action: torch.Tensor) -> torch.Tensor:
    """Update state with non-linear dynamics."""
    delta = action.detach() - self.state
    self.state = self.state + self.reactivity * torch.tanh(delta * 3.0)
    self.state = self.state.clamp(min=0.0)
    obs = (self.state + torch.randn(self.obs_dim) * self.noise).abs()
    obs = obs / (obs.sum() + 1e-8)
    return obs
```

**Step 3: Commit**

```bash
git add experiments/kernel/exp_compositionality.py experiments/kernel/exp_grammar.py experiments/kernel/exp_grounding.py
git commit -m "feat(kernel): add --use-rich-action and --broadcast-mode to all experiments"
```

---

### Task 8: Run Full Test Suite

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: 699+ tests ALL PASS (existing + ~16 new)

**Step 2: Commit if any test file was touched but not yet committed**

---

### Task 9: Run Experiments with New Features

**Step 1: Run compositionality with rich action**

Run: `python experiments/kernel/exp_compositionality.py --steps 16000 --use-rich-action --latent-weight 0.2 --top-down-strength 0.5`

Expected: Linearity < 0.99, Context Effect > 0.001

**Step 2: Run grammar with weighted broadcast**

Run: `python experiments/kernel/exp_grammar.py --steps 20000 --broadcast-mode weighted --use-rich-action`

Expected: Org bigram_pred > Ind bigram_pred

**Step 3: Run grounding with rich action**

Run: `python experiments/kernel/exp_grounding.py --steps 10000 --use-rich-action`

Expected: FE grounded < FE ungrounded (PASS)

**Step 4: Analyze and report results**

Compare before/after for all metrics in the design doc success criteria table.

**Step 5: Commit results**

```bash
git add results/
git commit -m "feat(kernel): experiment results with connected infrastructure"
```
