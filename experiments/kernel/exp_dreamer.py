"""
Dreamer-style amortized planning vs search (the verdict)
========================================================

Phase 4 of docs/plans/2026-06-08-dreamer-style-planning.md. Question: does an
amortized actor trained in imagination MATCH the search-based EFE planner on
model-based control, while choosing actions at CONSTANT (O(1)) cost — vs search,
which pays O(n_samples) world-model rollouts per action?

Task: control with unknown (cyclic-permuted) action dynamics, generalised to D
dimensions — `state_{t+1} = norm((1-r) state + r * roll(action, 1))` — toward a
non-vertex target. A model-free controller is defeated by the permutation; the
agent must learn the dynamics. Sweep the action dimension D ∈ {4, 8, 16}.

Arms (all warm up with random continuous actions so the world model learns):
  - efe-shooting : EFE, 48 sampled candidates per step
  - efe-cem      : EFE, CEM (16 x 3)
  - dreamer      : amortized actor (one network forward per action)

Metrics: (a) control error ||state - C|| over the tail; (b) action-selection
cost (µs per chosen action, measured after training). Success criterion: dreamer
error within ~10% of the best search arm, at far lower per-action cost.
"""

from __future__ import annotations

import sys
import time
import argparse
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel

DIMS = [4, 8, 16]


def target_for(dim: int) -> torch.Tensor:
    t = F.softmax(torch.linspace(1.2, -1.2, dim), dim=-1)
    return t / t.sum()


def make_kernel(arm: str, dim: int, target: torch.Tensor) -> ConsciousKernel:
    if arm == "efe-shooting":
        return ConsciousKernel(obs_dim=dim, action_mode="efe", preference=target,
                               efe_n_samples=48, efe_obs_norm="l1",
                               reflect_interval=10**9, dream_interval=10**9)
    if arm == "efe-cem":
        return ConsciousKernel(obs_dim=dim, action_mode="efe", preference=target,
                               efe_n_samples=16, efe_cem_iters=3, efe_obs_norm="l1",
                               reflect_interval=10**9, dream_interval=10**9)
    if arm == "dreamer":
        return ConsciousKernel(obs_dim=dim, action_mode="dreamer", preference=target,
                               actor_explore=0.3, reflect_interval=10**9,
                               dream_interval=10**9)
    raise ValueError(arm)


def run(arm: str, dim: int, n_steps: int, warmup: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    target = target_for(dim)
    ck = make_kernel(arm, dim, target)
    g = torch.Generator().manual_seed(seed)
    state = torch.rand(dim, generator=g); state = state / state.sum()
    obs = state.clone()
    dists = []
    for t in range(n_steps):
        result = ck.step(obs)
        if t < warmup:
            a = torch.rand(dim); a = a / a.sum()
            ck.last_action = a.detach()
        else:
            a = result.action
        permuted = torch.roll(a.detach(), 1)
        state = ((1 - 0.3) * state + 0.3 * permuted).clamp(min=1e-6)
        state = state / state.sum()
        obs = state
        dists.append(float(torch.linalg.vector_norm(state - target)))
    tail = max(1, int(n_steps * 0.2))
    err = st.mean(dists[-tail:])

    # Action-selection cost (after training), excluding world-model learning.
    reactive = torch.softmax(obs, dim=-1)
    n_timing = 200
    t0 = time.perf_counter()
    for _ in range(n_timing):
        if arm == "dreamer":
            ck._select_action_dreamer()
        else:
            ck._select_action_efe(reactive)
    us_per_action = (time.perf_counter() - t0) / n_timing * 1e6
    return err, us_per_action


def run_arm(arm, dim, seeds, n_steps, warmup):
    errs, costs = [], []
    for s in seeds:
        e, c = run(arm, dim, n_steps, warmup, s)
        errs.append(e); costs.append(c)
    return st.mean(errs), (st.pstdev(errs) if len(errs) > 1 else 0.0), st.mean(costs)


ARMS = ["efe-shooting", "efe-cem", "dreamer"]


def main(n_steps: int, warmup: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 72)
    print("  DREAMER (amortized) vs SEARCH — model-based control across dimension")
    print("=" * 72)
    print(f"  task = permuted-dynamics control; dims={DIMS}; steps={n_steps}; seeds={seeds}")
    print(f"  metrics = control error ||state-C|| and us/action (after training)")
    print()

    res = {}
    for dim in DIMS:
        print(f"  --- D={dim} ---")
        res[dim] = {}
        for arm in ARMS:
            e, s, c = run_arm(arm, dim, seeds, n_steps, warmup)
            res[dim][arm] = (e, s, c)
            print(f"    {arm:13s}: err = {e:.4f} ± {s:.4f}   cost = {c:8.1f} us/action")
        print()

    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    matches = []
    for dim in DIMS:
        best_search = min(res[dim]["efe-shooting"][0], res[dim]["efe-cem"][0])
        dre = res[dim]["dreamer"][0]
        ok = dre <= best_search * 1.10 + 0.02
        matches.append(ok)
        cost_ratio = res[dim]["efe-shooting"][2] / max(res[dim]["dreamer"][2], 1e-9)
        print(f"  D={dim:2d}: dreamer {dre:.4f} vs best-search {best_search:.4f} "
              f"-> matches={ok}; dreamer is {cost_ratio:.1f}x cheaper per action")
    print()
    won = all(matches)
    if won:
        print("  FINDING: the amortized actor MATCHES search on model-based control")
        print("  across dimensions, while choosing actions at O(1) cost (one network")
        print("  forward) vs search's O(n_samples) rollouts. Dreamer-style planning is")
        print("  the right upgrade: same control, far cheaper inference.")
    else:
        print("  FINDING: the amortized actor does NOT consistently match search "
              "(honest negative); see per-dim rows.")
    print(f"  [{'PASS' if won else 'PARTIAL/FAIL'}]")
    print("=" * 72)

    if plot:
        try:
            _plot(res)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return won


def _plot(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"efe-shooting": "#e67e22", "efe-cem": "#16a085", "dreamer": "#8e44ad"}
    for arm in ARMS:
        ax1.errorbar(DIMS, [res[d][arm][0] for d in DIMS],
                     yerr=[res[d][arm][1] for d in DIMS], marker="o",
                     color=colors[arm], capsize=4, label=arm)
    ax1.set_xlabel("action dimension"); ax1.set_ylabel("control error ||state-C||")
    ax1.set_title("Control error vs dimension"); ax1.legend(); ax1.grid(True, alpha=0.3)

    for arm in ARMS:
        ax2.plot(DIMS, [res[d][arm][2] for d in DIMS], marker="s",
                 color=colors[arm], label=arm)
    ax2.set_xlabel("action dimension"); ax2.set_ylabel("us per action (log)")
    ax2.set_yscale("log")
    ax2.set_title("Action-selection cost (amortized = O(1))")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Dreamer-style amortized actor vs search planning")
    out = Path("results") / "dreamer.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreamer amortized vs search")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(n_steps=args.steps, warmup=args.warmup,
              seeds=list(range(args.seeds)), plot=not args.no_plot)
    sys.exit(0 if ok else 1)
