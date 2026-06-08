"""
Control — closing the "controla" verb (continuous EFE action selection)
=======================================================================

The original aprende/controla question left "controla" unanswered: the EFE
planner could only choose among DISCRETE one-hot candidate actions, which the
agency investigation (docs/AGENCY_2026.md) flagged as OUT OF DISTRIBUTION for a
world model trained on continuous actions, and which cannot represent a non-vertex
target at all. This experiment closes it.

Task: an inertial control environment

    state_{t+1} = normalize( (1 - r) * state_t + r * action_t )

whose fixed point under a constant action is the action itself. The agent must
drive its state to a NON-VERTEX target C = [0.7, 0.1, 0.1, 0.1]. A one-hot action
can only push toward a pure vertex (overshooting C); a continuous action can BE
C, so the state converges exactly onto it.

Arms:
  - reactive       : action = softmax(obs)            (no goal)
  - efe-discrete   : EFE over one-hots only           (the old coarse planner)
  - efe-continuous : EFE + 48 sampled continuous actions (training-consistent)
  - efe-cont-h4    : continuous + planning horizon 4  (sustained-action rollout)

All EFE arms share an exploratory warm-up with random CONTINUOUS actions, so the
world model learns the dynamics on the SAME action space the planner evaluates
(the train/eval consistency the agency investigation identified as the real fix).

Metric: distance ||state - C|| over the last 100 steps (lower = better control),
plus cosine(state, C). Averaged over seeds.
"""

from __future__ import annotations

import sys
import argparse
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel

TARGET = torch.tensor([0.7, 0.1, 0.1, 0.1])
TARGET = TARGET / TARGET.sum()

ARMS = {
    "reactive":       dict(action_mode="reactive"),
    "efe-discrete":   dict(action_mode="efe", efe_n_samples=0, efe_horizon=1),
    "efe-continuous": dict(action_mode="efe", efe_n_samples=48, efe_horizon=1,
                           efe_obs_norm="l1"),
    "efe-cont-h4":    dict(action_mode="efe", efe_n_samples=48, efe_horizon=4,
                           efe_obs_norm="l1"),
}


class InertiaEnv:
    """state_{t+1} = normalize((1-r) state + r action). Fixed point = action."""

    def __init__(self, r: float = 0.3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        s = torch.rand(4, generator=g)
        self.state = s / s.sum()
        self.r = r

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.state = ((1 - self.r) * self.state + self.r * action.detach()).clamp(min=1e-6)
        self.state = self.state / self.state.sum()
        return self.state


def run(arm: str, n_steps: int, warmup: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    kwargs = ARMS[arm]
    is_efe = kwargs["action_mode"] == "efe"
    env = InertiaEnv(seed=seed)
    ck = ConsciousKernel(preference=TARGET if is_efe else None, **kwargs)

    obs = env.state.clone()
    dists, coss = [], []
    for t in range(n_steps):
        result = ck.step(obs)
        if is_efe and t < warmup:
            a = torch.rand(4)
            a = a / a.sum()
            ck.last_action = a.detach()  # train WM on the continuous action space
        else:
            a = result.action
        obs = env.step(a)
        dists.append(float(torch.linalg.vector_norm(env.state - TARGET)))
        coss.append(float(F.cosine_similarity(
            env.state.unsqueeze(0), TARGET.unsqueeze(0))))
    tail = max(1, int(n_steps * 0.2))
    return st.mean(dists[-tail:]), st.mean(coss[-tail:])


def run_arm(arm: str, seeds: list[int], n_steps: int, warmup: int) -> tuple[float, float, float]:
    ds, cs = [], []
    for s in seeds:
        d, c = run(arm, n_steps, warmup, s)
        ds.append(d)
        cs.append(c)
    std = st.pstdev(ds) if len(ds) > 1 else 0.0
    return st.mean(ds), std, st.mean(cs)


def main(n_steps: int, warmup: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  CONTROL — continuous EFE action selection (closing 'controla')")
    print("=" * 70)
    print(f"  target C = {[round(float(x), 2) for x in TARGET]}  (non-vertex)")
    print(f"  steps={n_steps}  warmup={warmup}  seeds={seeds}")
    print(f"  metric = ||state - C|| over last 20% (LOWER = better control)")
    print()

    res = {}
    for arm in ARMS:
        d, std, c = run_arm(arm, seeds, n_steps, warmup)
        res[arm] = (d, std, c)
        print(f"  {arm:15s}: dist = {d:.4f} ± {std:.4f}   cosine = {c:.4f}")
    print()

    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    disc = res["efe-discrete"][0]
    cont = res["efe-continuous"][0]
    react = res["reactive"][0]
    h4 = res["efe-cont-h4"][0]
    cont_beats_disc = cont < disc * 0.6   # clearly closer to target
    cont_reaches = cont < 0.1
    print(f"  continuous < discrete (fine actions help):  {cont_beats_disc} "
          f"({cont:.4f} vs {disc:.4f})")
    print(f"  continuous reaches the target (dist<0.1):   {cont_reaches} ({cont:.4f})")
    print(f"  vs reactive baseline:                       {react:.4f}")
    print(f"  horizon-4 vs horizon-1:                     {h4:.4f} vs {cont:.4f} "
          f"({'helps' if h4 < cont * 0.95 else 'no clear gain (YAGNI, per agency)'})")
    print()
    passed = cont_beats_disc and cont_reaches
    print(f"  [{'PASS' if passed else 'FAIL'}] continuous EFE controls the state to a "
          f"non-vertex target; the one-hot planner cannot")
    print("=" * 70)

    if plot:
        try:
            _plot(res)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return passed


def _plot(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = list(ARMS)
    x = np.arange(len(arms))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#7f8c8d", "#e67e22", "#2980b9", "#27ae60"]

    ax1.bar(x, [res[a][0] for a in arms], yerr=[res[a][1] for a in arms],
            capsize=4, color=colors)
    ax1.set_xticks(x); ax1.set_xticklabels(arms, rotation=15)
    ax1.set_ylabel("||state - C||  [lower = better control]")
    ax1.set_title("Control error to a non-vertex target")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, [res[a][2] for a in arms], color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(arms, rotation=15)
    ax2.set_ylabel("cosine(state, C)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Cosine to target")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Closing 'controla': continuous EFE reaches arbitrary targets")
    out = Path("results") / "control.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous EFE control benchmark")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(n_steps=args.steps, warmup=args.warmup,
              seeds=list(range(args.seeds)), plot=not args.no_plot)
    sys.exit(0 if ok else 1)
