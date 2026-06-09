"""
DreamerV3 parity — does a real RSSM crack CartPole-v1?
======================================================

The kernel's one-step world model plateaus at ~33% of CartPole's ceiling and
collapses late (exp_cartpole.py). This trains a self-contained DreamerV2/V3-style
agent (kernel/dreamerv3_agent.py) with an RSSM world model (recurrent state +
sequence training) and a LEARNED reward model, on the same CartPole-v1, to test
whether proper Dreamer parity reaches the ceiling — which would pin the kernel's
gap to its world-model architecture, not its active-inference design.

Baselines: random (~22) and a near-optimal heuristic (~500). Metric: mean episode
length over the last 25% of episodes, plus a frozen greedy eval (cap 500).
"""

from __future__ import annotations

import sys
import argparse
import statistics as st
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel.dreamerv3_agent import DreamerV3Agent

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    print("gymnasium not installed — `pip install gymnasium`")
    sys.exit(1)

SCALES = np.array([2.4, 3.0, 0.21, 3.0], dtype=np.float32)


def norm(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs / SCALES, dtype=torch.float32)


def baseline_random(n_steps: int, seed: int) -> list[int]:
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(seed); env.reset(seed=seed)
    lengths, ep = [], 0
    for _ in range(n_steps):
        _, _, term, trunc, _ = env.step(int(rng.integers(2))); ep += 1
        if term or trunc:
            lengths.append(ep); ep = 0; env.reset()
    env.close(); return lengths


def baseline_heuristic(n_steps: int, seed: int) -> list[int]:
    env = gym.make("CartPole-v1"); obs, _ = env.reset(seed=seed)
    lengths, ep = [], 0
    for _ in range(n_steps):
        a = 1 if (obs[2] + 0.5 * obs[3]) > 0 else 0
        obs, _, term, trunc, _ = env.step(a); ep += 1
        if term or trunc:
            lengths.append(ep); ep = 0; obs, _ = env.reset()
    env.close(); return lengths


def train_dreamer(n_steps: int, train_every: int, seed: int) -> tuple[list[int], float]:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    agent = DreamerV3Agent(obs_dim=4, action_dim=2, seed=seed)
    agent.reset_state()
    lengths, ep, first = [], 0, True
    for t in range(n_steps):
        a, a_oh = agent.act(norm(obs))
        nobs, _, term, trunc, _ = env.step(a)
        done = term or trunc
        agent.replay.add(norm(obs), a_oh, 1.0, 0.0 if term else 1.0, first)
        first = False
        ep += 1
        if t % train_every == 0:
            agent.train()
        if done:
            lengths.append(ep); ep = 0
            obs, _ = env.reset(); agent.reset_state(); first = True
        else:
            obs = nobs

    # Frozen greedy evaluation.
    greedy = []
    obs, _ = env.reset(); agent.reset_state(); ep = 0; steps = 0
    while len(greedy) < 10 and steps < 6000:
        a, _ = agent.act(norm(obs), greedy=True)
        obs, _, term, trunc, _ = env.step(a); ep += 1; steps += 1
        if term or trunc:
            greedy.append(ep); ep = 0; obs, _ = env.reset(); agent.reset_state()
    env.close()
    return lengths, (st.mean(greedy) if greedy else 0.0)


def tail_mean(lengths: list[int]) -> float:
    if not lengths:
        return 0.0
    k = max(1, len(lengths) // 4)
    return st.mean(lengths[-k:])


def main(n_steps: int, train_every: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  DREAMERV3 PARITY (RSSM) on CartPole-v1")
    print("=" * 70)
    print(f"  steps={n_steps}  train_every={train_every}  seeds={seeds}")
    print()
    rnd = st.mean(tail_mean(baseline_random(n_steps, s)) for s in seeds)
    heur = st.mean(tail_mean(baseline_heuristic(n_steps, s)) for s in seeds)
    runs = [train_dreamer(n_steps, train_every, s) for s in seeds]
    curves = [r[0] for r in runs]
    greedy = [r[1] for r in runs]
    g = st.mean(greedy); g_sd = st.pstdev(greedy) if len(greedy) > 1 else 0.0
    train_tail = st.mean(tail_mean(c) for c in curves)

    print(f"  random        : {rnd:.1f}")
    print(f"  heuristic     : {heur:.1f}")
    print(f"  DreamerV3 RSSM : greedy {g:.1f} ± {g_sd:.1f}  (training tail {train_tail:.1f})")
    print(f"    (episodes/seed: {[len(c) for c in curves]})")
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    solved = g >= 450
    strong = g >= 0.8 * heur
    print(f"  greedy {g:.1f} vs ceiling {heur:.0f}  ({g / max(heur,1):.0%} of optimal)")
    if solved:
        print("  FINDING: the RSSM agent SOLVES CartPole (greedy >= 450). The kernel's")
        print("  ~33% plateau was its one-step world model — a recurrent state-space")
        print("  model trained on sequences + a learned reward closes the gap.")
    elif strong:
        print("  FINDING: the RSSM agent reaches near-ceiling — strong evidence the")
        print("  bottleneck was the world-model architecture, not active inference.")
    else:
        print("  FINDING: even a proper RSSM does not (yet) reach the ceiling here —")
        print("  honest; the limit is deeper than the one-step model or needs more tuning.")
    print(f"  [{'SOLVED' if solved else 'PARTIAL'}]")
    print("=" * 70)

    if plot:
        try:
            _plot(rnd, heur, g, g_sd, curves[0])
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return solved


def _plot(rnd, heur, g, g_sd, curve) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(["random", "heuristic", "DreamerV3\nRSSM"], [rnd, heur, g],
            yerr=[0, 0, g_sd], capsize=4, color=["#7f8c8d", "#27ae60", "#c0392b"])
    ax1.axhline(500, ls="--", c="k", alpha=0.4, label="cap")
    ax1.set_ylabel("mean episode length (last 25%)")
    ax1.set_title("CartPole-v1: RSSM agent"); ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)
    if len(curve) > 1:
        w = max(1, len(curve) // 20)
        ax2.plot(np.convolve(curve, np.ones(w) / w, mode="valid"), color="#c0392b")
    ax2.set_xlabel("episode"); ax2.set_ylabel("episode length (smoothed)")
    ax2.set_title("RSSM learning curve (seed 0)"); ax2.grid(True, alpha=0.3)
    fig.suptitle("DreamerV3 parity (RSSM) vs the kernel's one-step world model")
    out = Path("results") / "dreamerv3.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DreamerV3 RSSM on CartPole")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    ok = main(a.steps, a.train_every, list(range(a.seeds)), plot=not a.no_plot)
    sys.exit(0 if ok else 1)
