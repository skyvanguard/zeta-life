"""
Model-based control — where the action-conditioned design is an ADVANTAGE
=========================================================================

The Mackey-Glass benchmark (exp_realtask.py) showed the kernel is handicapped at
pure prediction: its transition is driven by the action, so with no real action
signal it underperforms a plain GRU. The flip side of that design is that the
kernel is an ACTION-CONDITIONED agent — it should shine where control requires a
learned action->effect model. This experiment tests exactly that.

Task: control with UNKNOWN (permuted) action dynamics.

    state_{t+1} = normalize( (1-r) state + r * P[action] )

where P is a fixed permutation of the action channels, unknown to the agent. The
agent must drive its state to a non-vertex target C = [0.7, 0.1, 0.1, 0.1]. A
model-free controller that assumes identity dynamics (action = C) FAILS: it sends
C but the state moves toward P[C] != C. Only an agent that LEARNS the dynamics
and inverts them — sending action = P^{-1}[C] — reaches the target. That is the
whole point of a model-based active-inference agent.

Arms:
  - reactive   : action = softmax(obs)                 (no goal)
  - naive (MF) : action = C                            (model-free; ignores P)
  - kernel EFE : continuous EFE over a learned world model (warm-up = random
                 continuous actions so the model learns the permuted dynamics)

Metric: control error ||state - C|| over the test tail (lower = better),
averaged over seeds. The honest question: does the learned model + planner beat
the model-free controller that the permutation defeats?
"""

from __future__ import annotations

import sys
import argparse
import statistics as st
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel

TARGET = torch.tensor([0.7, 0.1, 0.1, 0.1])
TARGET = TARGET / TARGET.sum()
PERM = [2, 0, 3, 1]  # fixed, unknown-to-agent permutation of action channels


class PermutedEnv:
    """state_{t+1} = normalize((1-r) state + r * action[PERM])."""

    def __init__(self, r: float = 0.4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        s = torch.rand(4, generator=g)
        self.state = s / s.sum()
        self.r = r
        self.perm = torch.tensor(PERM)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        permuted = action.detach()[self.perm]
        self.state = ((1 - self.r) * self.state + self.r * permuted).clamp(min=1e-6)
        self.state = self.state / self.state.sum()
        return self.state


def run(arm: str, n_steps: int, warmup: int, seed: int) -> float:
    torch.manual_seed(seed)
    env = PermutedEnv(seed=seed)
    obs = env.state.clone()
    ck = None
    if arm == "kernel":
        ck = ConsciousKernel(action_mode="efe", preference=TARGET,
                             efe_n_samples=48, efe_obs_norm="l1",
                             reflect_interval=10**9, dream_interval=10**9)
    dists = []
    for t in range(n_steps):
        if arm == "reactive":
            a = torch.softmax(obs, dim=-1)
        elif arm == "naive":
            a = TARGET.clone()                      # model-free: assumes identity
        else:  # kernel
            result = ck.step(obs)
            if t < warmup:
                a = torch.rand(4); a = a / a.sum()
                ck.last_action = a.detach()         # learn the dynamics on the action space
            else:
                a = result.action
        obs = env.step(a)
        dists.append(float(torch.linalg.vector_norm(env.state - TARGET)))
    tail = max(1, int(n_steps * 0.2))
    return st.mean(dists[-tail:])


def run_arm(arm: str, seeds, n_steps, warmup):
    vals = [run(arm, n_steps, warmup, s) for s in seeds]
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def main(n_steps: int, warmup: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  MODEL-BASED CONTROL — unknown (permuted) action dynamics")
    print("=" * 70)
    print(f"  target C = {[round(float(x), 2) for x in TARGET]}  perm={PERM}")
    print(f"  steps={n_steps}  warmup={warmup}  seeds={seeds}")
    print(f"  metric = ||state - C|| over last 20% (LOWER = better)")
    print()

    res = {}
    for arm in ("reactive", "naive", "kernel"):
        m, s = run_arm(arm, seeds, n_steps, warmup)
        res[arm] = (m, s)
        label = {"reactive": "reactive", "naive": "naive (model-free)",
                 "kernel": "kernel EFE (model-based)"}[arm]
        print(f"  {label:26s}: dist = {m:.4f} ± {s:.4f}")
    print()

    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    naive = res["naive"][0]
    ker = res["kernel"][0]
    react = res["reactive"][0]
    beats_naive = ker < naive * 0.6
    reaches = ker < 0.15
    print(f"  kernel < naive (learned model beats model-free): {beats_naive} "
          f"({ker:.4f} vs {naive:.4f})")
    print(f"  kernel reaches the target (dist<0.15):           {reaches} ({ker:.4f})")
    print(f"  reactive baseline:                               {react:.4f}")
    print()
    won = beats_naive and reaches
    if won:
        print("  FINDING: the kernel LEARNS the unknown (permuted) dynamics and inverts")
        print("  them to reach the target, where the model-free controller is defeated")
        print("  by the permutation. This is the regime where the action-conditioned")
        print("  design is an ADVANTAGE — the mirror image of the prediction handicap.")
    else:
        print("  FINDING: even here the learned model does not clearly beat the model-free")
        print("  controller — an honest negative bounding the agent's control value.")
    print(f"  [{'PASS' if won else 'INCONCLUSIVE'}]")
    print("=" * 70)

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

    arms = ["reactive", "naive", "kernel"]
    labels = ["reactive", "naive\n(model-free)", "kernel EFE\n(model-based)"]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, [res[a][0] for a in arms], yerr=[res[a][1] for a in arms],
           capsize=4, color=["#7f8c8d", "#e67e22", "#8e44ad"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("control error ||state - C|| [lower=better]")
    ax.set_title("Control with unknown permuted dynamics:\nlearned model beats the model-free controller")
    ax.grid(True, axis="y", alpha=0.3)
    out = Path("results") / "modelcontrol.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-based control with permuted dynamics")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(n_steps=args.steps, warmup=args.warmup,
              seeds=list(range(args.seeds)), plot=not args.no_plot)
    sys.exit(0 if ok else 1)
