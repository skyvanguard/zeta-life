"""
Curiosity — does a REAL dynamics-ensemble disagreement drive exploration?
=========================================================================

Earlier finding (the retraction): the EFE's disagreement-curiosity, sourced from
the shared-latent predictor HEADS (`wm_disagreement_heads`), gave no controlled
gain — the heads share the transition and differ only in a linear readout over an
identical next-latent, so their disagreement is near-flat. RELATED_WORK item #3:
retry with a Plan2Explore-faithful INDEPENDENT dynamics ensemble
(`dynamics_ensemble`), where each member is its own (latent, action) -> next-obs
model, so they disagree where the dynamics are unlearned.

Task: a two-regime environment that hides dynamics behind exploration.

    regime A (argmax(state) == 0):  state_{t+1} = (1-r) state + r action
    regime B (argmax(state) != 0):  state_{t+1} = (1-r) state + r roll(action, 1)

The agent's preferred C is peaked at vertex 0 (regime A = home). A purely
pragmatic agent never leaves A and never learns B. A curious agent is pulled
toward the high-disagreement frontier (B is unlearned).

Two CONTROLLED paired comparisons (each curious arm differs from its pragmatic
control ONLY in efe_epistemic_weight; both carry the same machinery so the global
RNG stream is matched):
  - heads   : curious-heads   vs pragmatic-heads     (the old, flat signal)
  - ensemble: curious-ens     vs pragmatic-ens       (the real dynamics signal)

Metric: fraction of steps in regime B (coverage of the novel regime). The honest
question: does the real ensemble drive exploration where the heads did not?
"""

from __future__ import annotations

import sys
import math
import argparse
import statistics as st
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel

C = torch.tensor([0.7, 0.1, 0.1, 0.1])
C = C / C.sum()


def arms_for(weight: float) -> dict:
    base = dict(action_mode="efe", efe_n_samples=24, efe_obs_norm="l1",
                efe_epistemic_mode="disagreement")
    return {
        "reactive":        dict(action_mode="reactive"),
        "pragmatic-heads": dict(**base, wm_disagreement_heads=5, efe_epistemic_weight=0.0),
        "curious-heads":   dict(**base, wm_disagreement_heads=5, efe_epistemic_weight=weight),
        "pragmatic-ens":   dict(**base, dynamics_ensemble=5, efe_epistemic_weight=0.0),
        "curious-ens":     dict(**base, dynamics_ensemble=5, efe_epistemic_weight=weight),
    }


def true_step(state: torch.Tensor, action: torch.Tensor, r: float = 0.3) -> torch.Tensor:
    if int(state.argmax()) == 0:                       # regime A
        nxt = (1 - r) * state + r * action.detach()
    else:                                              # regime B: permuted action
        nxt = (1 - r) * state + r * torch.roll(action.detach(), 1)
    nxt = nxt.clamp(min=1e-6)
    return nxt / nxt.sum()


def run(kwargs: dict, n_steps: int, seed: int) -> float:
    torch.manual_seed(seed)
    is_efe = kwargs["action_mode"] == "efe"
    ck = ConsciousKernel(preference=C if is_efe else None, reflect_interval=10**9,
                         dream_interval=10**9, **kwargs)
    state = C.clone()
    obs = state.clone()
    in_b = []
    for _ in range(n_steps):
        result = ck.step(obs)
        state = true_step(state, result.action)
        obs = state
        in_b.append(int(state.argmax()) != 0)
    tail = max(1, n_steps // 2)
    return sum(in_b[-tail:]) / tail


def paired(curious: list[float], pragmatic: list[float]) -> tuple[float, float, int, int]:
    diffs = [c - p for c, p in zip(curious, pragmatic)]
    n = len(diffs)
    md = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("inf")
    t = md / se if se not in (0.0, float("inf")) else 0.0
    wins = sum(d > 0 for d in diffs)
    return md, t, wins, n


def main(n_steps: int, seeds: list[int], weight: float, plot: bool) -> bool:
    arms = arms_for(weight)
    print("=" * 72)
    print("  CURIOSITY — real dynamics ensemble vs shared-latent heads (controlled)")
    print("=" * 72)
    print(f"  epistemic_weight={weight}  steps={n_steps}  seeds={len(seeds)}")
    print(f"  metric = fraction of steps in regime B (the novel regime)")
    print()

    per_arm = {name: [run(kw, n_steps, s) for s in seeds] for name, kw in arms.items()}
    res = {name: (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0)
           for name, v in per_arm.items()}
    for name in arms:
        print(f"  {name:16s}: time in regime B = {res[name][0]:.3f} ± {res[name][1]:.3f}")
    print()

    md_h, t_h, w_h, n = paired(per_arm["curious-heads"], per_arm["pragmatic-heads"])
    md_e, t_e, w_e, _ = paired(per_arm["curious-ens"], per_arm["pragmatic-ens"])

    print("=" * 72)
    print("  VERDICT (paired, controlled)")
    print("=" * 72)
    print(f"  HEADS    : curious {res['curious-heads'][0]:.3f} vs pragmatic "
          f"{res['pragmatic-heads'][0]:.3f}  | diff {md_h:+.3f} (t={t_h:.2f}, wins {w_h}/{n})")
    print(f"  ENSEMBLE : curious {res['curious-ens'][0]:.3f} vs pragmatic "
          f"{res['pragmatic-ens'][0]:.3f}  | diff {md_e:+.3f} (t={t_e:.2f}, wins {w_e}/{n})")
    print()
    ens_works = md_e > 0 and abs(t_e) > 2.0
    heads_works = md_h > 0 and abs(t_h) > 2.0
    if ens_works and not heads_works:
        print("  FINDING: the REAL dynamics ensemble RELIABLY drives exploration where")
        print("  the shared-latent heads do not — the disagreement signal needed to be")
        print("  over independent dynamics, not a linear readout. Validates Plan2Explore")
        print("  in our regime and explains the earlier null.")
    elif ens_works and heads_works:
        print("  FINDING: both signals drive exploration here; the ensemble is the more")
        print("  principled source (independent dynamics).")
    elif md_e > md_h and md_e > 0:
        print("  FINDING: the ensemble shows a stronger positive DIRECTION than the heads")
        print(f"  but not significant at n={n} (t={t_e:.2f}) — a tendency, reported honestly.")
    else:
        print("  FINDING: even the real ensemble does not reliably drive exploration in")
        print("  this regime — an honest negative; the pragmatic anchor dominates.")
    print(f"  [{'PASS' if ens_works else 'INCONCLUSIVE'}]")
    print("=" * 72)

    if plot:
        try:
            _plot(res)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return ens_works


def _plot(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = list(res)
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#7f8c8d", "#e67e22", "#d35400", "#2980b9", "#8e44ad"]
    ax.bar(x, [res[a][0] for a in arms], yerr=[res[a][1] for a in arms],
           capsize=4, color=colors[:len(arms)])
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=12, fontsize=8)
    ax.set_ylabel("fraction of time in the novel regime B")
    ax.set_title("Curiosity: real dynamics ensemble vs shared-latent heads")
    ax.grid(True, axis="y", alpha=0.3)
    out = Path("results") / "curiosity.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disagreement curiosity: ensemble vs heads")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--weight", type=float, default=500.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(n_steps=args.steps, seeds=list(range(args.seeds)),
              weight=args.weight, plot=not args.no_plot)
    sys.exit(0 if ok else 1)
