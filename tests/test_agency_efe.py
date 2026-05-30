"""Tests for active-inference action selection (agency) in ConsciousKernel.

Covers:
- Backward compatibility: action_mode="reactive" (default) is byte-identical to
  the pre-agency kernel (action = softmax(stimulus) at latent_weight=0).
- EFE mode falls back to reactive when no preference is given.
- EFE actions remain valid probability distributions.
- explore_eps produces exploratory (varying) actions.
- Integration: in a reactive environment with a goal, the EFE planner drives the
  environment toward the preference and beats the reactive baseline.

The integration test encodes the core claim of the agency work: a planner that
imagines action outcomes and minimises expected free energy toward a preference
acts with purpose, where a reactive agent (action = softmax(obs)) cannot.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from zeta_life.kernel import ConsciousKernel


def test_default_action_mode_is_reactive():
    assert ConsciousKernel().action_mode == "reactive"


def test_reactive_mode_is_pure_softmax():
    # Regression: default mode must keep action = softmax(stimulus).
    ck = ConsciousKernel(action_mode="reactive", latent_weight=0.0)
    stimulus = torch.tensor([1.0, 0.0, 0.5, 0.2])
    result = ck.step(stimulus)
    assert torch.allclose(result.action, F.softmax(stimulus, dim=-1), atol=1e-5)


def test_efe_without_preference_falls_back_to_reactive():
    # No preference -> the efe branch is inert and behaves reactively.
    ck = ConsciousKernel(action_mode="efe", preference=None, latent_weight=0.0)
    stimulus = torch.tensor([1.0, 0.0, 0.5, 0.2])
    result = ck.step(stimulus)
    assert torch.allclose(result.action, F.softmax(stimulus, dim=-1), atol=1e-5)


def test_preference_is_normalised():
    # A raw (unnormalised) preference is stored as a distribution.
    ck = ConsciousKernel(action_mode="efe", preference=torch.tensor([7.0, 1.0, 1.0, 1.0]))
    assert abs(float(ck.preference.sum()) - 1.0) < 1e-6


def test_efe_action_is_valid_distribution():
    pref = torch.tensor([0.7, 0.1, 0.1, 0.1])
    ck = ConsciousKernel(action_mode="efe", preference=pref)
    for _ in range(30):
        result = ck.step(torch.rand(4))
    assert abs(float(result.action.sum()) - 1.0) < 1e-4
    assert bool((result.action >= 0).all())


def test_explore_eps_produces_varying_actions():
    # With full exploration, consecutive actions should differ (random, normalised).
    torch.manual_seed(0)
    ck = ConsciousKernel(action_mode="efe", preference=torch.tensor([0.7, 0.1, 0.1, 0.1]),
                         explore_eps=1.0)
    actions = [ck.step(torch.rand(4)).action for _ in range(5)]
    assert not all(torch.allclose(actions[0], a, atol=1e-3) for a in actions[1:])
    for a in actions:
        assert abs(float(a.sum()) - 1.0) < 1e-4


class _ReactiveEnv:
    """state_{t+1} = (1-r)*state + r*action; obs = normalize(state)."""
    def __init__(self, r: float = 0.3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.state = torch.rand(4, generator=g)
        self.state = self.state / self.state.sum()
        self.r = r

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.state = ((1 - self.r) * self.state + self.r * action).clamp(min=1e-6)
        return self.state / self.state.sum()


def _goal_run(planner: bool, n: int = 600, warmup: int = 300, seed: int = 0) -> float:
    """Run a kernel in a reactive env toward a goal; return mean tail cosine."""
    torch.manual_seed(seed)
    target = torch.tensor([0.7, 0.1, 0.1, 0.1])
    env = _ReactiveEnv(seed=seed)
    ck = ConsciousKernel(
        action_mode="efe" if planner else "reactive",
        preference=target if planner else None,
    )
    obs = env.state / env.state.sum()
    sims = []
    for t in range(n):
        result = ck.step(obs)
        if t < warmup:
            a = torch.rand(4)
            a = a / a.sum()
            ck.last_action = a.detach()  # align WM training with executed action
        else:
            a = result.action  # efe action (planner) or softmax(obs) (reactive)
        obs = env.step(a)
        sims.append(float(F.cosine_similarity(env.state.unsqueeze(0), target.unsqueeze(0))))
    return sum(sims[-100:]) / 100


def test_efe_planner_reaches_goal_and_beats_reactive():
    # Core agency claim: the planner steers the environment toward the preference
    # and clearly outperforms the reactive baseline.
    planner = sum(_goal_run(planner=True, seed=s) for s in (0, 1, 2)) / 3
    reactive = sum(_goal_run(planner=False, seed=s) for s in (0, 1, 2)) / 3
    assert planner > 0.85, f"planner should reach the goal (cosine={planner:.3f})"
    assert planner - reactive > 0.1, (
        f"agency should beat reactive (planner={planner:.3f}, reactive={reactive:.3f})"
    )
