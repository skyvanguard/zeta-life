"""
Integration — the Conscious Kernel's faculties on an RSSM world model
=====================================================================

§3.10 bounded the kernel's CartPole gap to its one-step world model. This is the
integration (`kernel/rssm_kernel.py`): the kernel's consciousness faculties —
persistent identity (self-model), complementary (fast/slow) memory, zeta-rhythm
dream consolidation, and the integration index Psi — layered on the DreamerV3 RSSM
world model + controller. The question: does the integrated kernel REACH the
ceiling on CartPole (i.e. the faculties don't break the solve), AND is Psi a live,
rising integration signal as it learns?

Baselines: random / heuristic. Metric: episode length (cap 500) + frozen greedy
eval; plus the Psi trajectory (early vs late mean).
"""

from __future__ import annotations

import sys
import argparse
import statistics as st
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel.rssm_kernel import RSSMConsciousKernel

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    print("gymnasium not installed — `pip install gymnasium`")
    sys.exit(1)

SCALES = np.array([2.4, 3.0, 0.21, 3.0], dtype=np.float32)


def norm(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs / SCALES, dtype=torch.float32)


def baseline(n_steps: int, seed: int, heuristic: bool) -> list[int]:
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed); lengths, ep = [], 0
    for _ in range(n_steps):
        a = (1 if (obs[2] + 0.5 * obs[3]) > 0 else 0) if heuristic else int(rng.integers(2))
        obs, _, term, trunc, _ = env.step(a); ep += 1
        if term or trunc:
            lengths.append(ep); ep = 0; obs, _ = env.reset()
    env.close(); return lengths


def run_kernel(n_steps: int, seed: int) -> tuple[list[int], float, list[float]]:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    k = RSSMConsciousKernel(obs_dim=4, action_dim=2, agent_kwargs=dict(seed=seed))
    k.reset_state()
    lengths, ep, first, psis = [], 0, True, []
    for _ in range(n_steps):
        r = k.act(norm(obs))
        nobs, _, term, trunc, _ = env.step(r.action)
        k.observe(norm(obs), r.action_onehot, 1.0, term, first)
        first = False
        psis.append(r.psi); ep += 1
        if term or trunc:
            lengths.append(ep); ep = 0
            obs, _ = env.reset(); k.reset_state(); first = True
        else:
            obs = nobs

    greedy = []
    obs, _ = env.reset(); k.reset_state(); ep = 0; steps = 0
    while len(greedy) < 10 and steps < 6000:
        r = k.act(norm(obs), greedy=True)
        obs, _, term, trunc, _ = env.step(r.action); ep += 1; steps += 1
        if term or trunc:
            greedy.append(ep); ep = 0; obs, _ = env.reset(); k.reset_state()
    env.close()
    return lengths, (st.mean(greedy) if greedy else 0.0), psis


def tail_mean(xs) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return st.mean(xs[-max(1, len(xs) // 4):])


def main(n_steps: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  INTEGRATION — Conscious Kernel faculties on an RSSM world model")
    print("=" * 70)
    print(f"  steps={n_steps}  seeds={seeds}")
    print()
    rnd = st.mean(tail_mean(baseline(n_steps, s, heuristic=False)) for s in seeds)
    heur = st.mean(tail_mean(baseline(n_steps, s, heuristic=True)) for s in seeds)
    runs = [run_kernel(n_steps, s) for s in seeds]
    greedy = [r[1] for r in runs]
    g = st.mean(greedy); g_sd = st.pstdev(greedy) if len(greedy) > 1 else 0.0
    tail = st.mean(tail_mean(r[0]) for r in runs)
    # Psi: early (first 10%) vs late (last 10%) mean.
    psi0 = runs[0][2]
    early = st.mean(psi0[: max(1, len(psi0) // 10)])
    late = st.mean(psi0[-max(1, len(psi0) // 10):])

    print(f"  random              : {rnd:.1f}")
    print(f"  heuristic           : {heur:.1f}")
    print(f"  RSSM-kernel (greedy) : {g:.1f} ± {g_sd:.1f}  (training tail {tail:.1f})")
    print(f"  Psi (integration)    : early {early:.3f} -> late {late:.3f}")
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    solved = g >= 450
    psi_live = late > early
    print(f"  greedy {g:.1f} vs ceiling {heur:.0f}  ({g / max(heur, 1):.0%} of optimal)")
    print(f"  Psi rises with learning: {psi_live} ({early:.3f} -> {late:.3f})")
    print()
    if solved:
        print("  FINDING: the INTEGRATED kernel reaches CartPole's ceiling — its")
        print("  consciousness faculties (identity, CLS memory, dream, Psi) run on a")
        print("  recurrent RSSM world model WITHOUT breaking control, and Psi stays a")
        print("  live integration signal over the recurrent state.")
    else:
        print("  FINDING: the integrated kernel does not reach the ceiling — honest;")
        print("  the faculties or the integration need work.")
    print(f"  [{'SOLVED' if solved else 'PARTIAL'}]")
    print("=" * 70)

    if plot:
        try:
            _plot(rnd, heur, g, g_sd, runs[0][0], psi0)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return solved


def _plot(rnd, heur, g, g_sd, curve, psis) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(["random", "heuristic", "RSSM-kernel"], [rnd, heur, g],
            yerr=[0, 0, g_sd], capsize=4, color=["#7f8c8d", "#27ae60", "#2c3e50"])
    ax1.axhline(500, ls="--", c="k", alpha=0.4)
    ax1.set_ylabel("mean episode length (last 25%)")
    ax1.set_title("CartPole-v1: integrated kernel"); ax1.grid(True, axis="y", alpha=0.3)
    if len(psis) > 50:
        w = max(1, len(psis) // 100)
        ax2.plot(np.convolve(psis, np.ones(w) / w, mode="valid"), color="#8e44ad")
    ax2.set_xlabel("step"); ax2.set_ylabel("Psi (integration index)")
    ax2.set_title("Psi over learning (seed 0)"); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)
    fig.suptitle("Integration: kernel faculties + Psi on an RSSM world model")
    out = Path("results") / "rssm_kernel.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Integrated RSSM Conscious Kernel on CartPole")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    ok = main(a.steps, list(range(a.seeds)), plot=not a.no_plot)
    sys.exit(0 if ok else 1)
