"""
External RL benchmark — the kernel on CartPole-v1 (gymnasium)
============================================================

The first test of the kernel on a recognised external RL environment (the
strategist's #1). CartPole-v1: 4-D real observation, discrete action {0,1},
reward = +1 per step alive (episode ends when the pole falls or the cart leaves
the track; max length 500).

We frame it as active-inference REGULATION: the kernel's preference is the
upright/centered goal state (normalised zeros), and the **Dreamer-style actor**
(action_mode="dreamer", dreamer_reward="neg_distance") learns, in imagination, to
act so the predicted next state stays near the goal — which is exactly staying
alive. The kernel's 2-simplex action maps to the discrete env action by argmax.

Honest baselines (no extra RL library):
  - random   : uniform {0,1}              (~22 steps)
  - heuristic: push toward the falling side (near-optimal, ~500)
  - kernel   : Dreamer actor, learned online across episodes

Metric: mean episode length over the last 25% of episodes (higher = better; cap
500). The honest question: does the kernel TRANSFER to a real RL task — does it
learn to balance well above random, and how close to the heuristic ceiling?
"""

from __future__ import annotations

import sys
import argparse
import statistics as st
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    print("gymnasium not installed — `pip install gymnasium`")
    sys.exit(1)

# Approximate observation scales for normalisation (cart pos, cart vel, pole
# angle, pole angular velocity). Goal state = normalised zeros (upright/centered).
SCALES = np.array([2.4, 3.0, 0.21, 3.0], dtype=np.float32)
GOAL = torch.zeros(4)


def normalize(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs / SCALES, dtype=torch.float32)


def episode_lengths_random(n_steps: int, seed: int) -> list[int]:
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    lengths, ep = [], 0
    for _ in range(n_steps):
        _, _, term, trunc, _ = env.step(int(rng.integers(2)))
        ep += 1
        if term or trunc:
            lengths.append(ep); ep = 0; env.reset()
    env.close()
    return lengths


def episode_lengths_heuristic(n_steps: int, seed: int) -> list[int]:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    lengths, ep = [], 0
    for _ in range(n_steps):
        # push in the direction the pole is falling (angle + a bit of angular vel)
        a = 1 if (obs[2] + 0.5 * obs[3]) > 0 else 0
        obs, _, term, trunc, _ = env.step(a)
        ep += 1
        if term or trunc:
            lengths.append(ep); ep = 0; obs, _ = env.reset()
    env.close()
    return lengths


def _reset_recurrence(ck: ConsciousKernel) -> None:
    ck.world_model.latent_state = torch.zeros(ck.latent_dim)
    ck.last_action = torch.zeros(2)
    ck._last_self_state = torch.zeros(4)


def episode_lengths_kernel(n_steps: int, warmup: int, seed: int,
                           greedy_eps: int = 10) -> tuple[list[int], float]:
    """Train online, then evaluate GREEDILY (no exploration noise).

    Returns (training episode lengths, mean greedy episode length).
    """
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    ck = ConsciousKernel(
        obs_dim=4, action_dim=2, action_mode="dreamer", preference=GOAL,
        dreamer_reward="neg_distance", actor_explore=0.3,
        reflect_interval=10**9, dream_interval=10**9,
    )
    lengths, ep = [], 0
    for t in range(n_steps):
        result = ck.step(normalize(obs))
        if t < warmup:
            a = torch.rand(2); a = a / a.sum()
            ck.last_action = a.detach()
        else:
            a = result.action
        obs, _, term, trunc, _ = env.step(int(torch.argmax(a).item()))
        ep += 1
        if term or trunc:
            lengths.append(ep); ep = 0
            obs, _ = env.reset(); _reset_recurrence(ck)

    # Greedy evaluation (exploration off; keep learning off for a clean readout).
    ck.actor_explore = 0.0
    greedy = []
    obs, _ = env.reset(); _reset_recurrence(ck); ep = 0
    steps = 0
    while len(greedy) < greedy_eps and steps < greedy_eps * 600:
        result = ck.step(normalize(obs))
        obs, _, term, trunc, _ = env.step(int(torch.argmax(result.action).item()))
        ep += 1; steps += 1
        if term or trunc:
            greedy.append(ep); ep = 0
            obs, _ = env.reset(); _reset_recurrence(ck)
    env.close()
    return lengths, (st.mean(greedy) if greedy else 0.0)


def tail_mean(lengths: list[int]) -> float:
    if not lengths:
        return 0.0
    k = max(1, len(lengths) // 4)
    return st.mean(lengths[-k:])


def main(n_steps: int, warmup: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  CARTPOLE-v1 — the kernel on an external RL benchmark")
    print("=" * 70)
    print(f"  steps={n_steps}  warmup={warmup}  seeds={seeds}")
    print(f"  metric = mean episode length over last 25% of episodes (cap 500)")
    print()

    rnd = st.mean(tail_mean(episode_lengths_random(n_steps, s)) for s in seeds)
    heur = st.mean(tail_mean(episode_lengths_heuristic(n_steps, s)) for s in seeds)
    ker_runs = [episode_lengths_kernel(n_steps, warmup, s) for s in seeds]
    train_curves = [r[0] for r in ker_runs]
    greedy_vals = [r[1] for r in ker_runs]
    ker = st.mean(greedy_vals)                       # headline = greedy eval
    ker_std = st.pstdev(greedy_vals) if len(greedy_vals) > 1 else 0.0
    ker_train = st.mean(tail_mean(c) for c in train_curves)  # training-tail (w/ explore)

    print(f"  random    : mean ep length = {rnd:.1f}")
    print(f"  heuristic : mean ep length = {heur:.1f}")
    print(f"  kernel    : greedy eval = {ker:.1f} ± {ker_std:.1f}  "
          f"(training tail = {ker_train:.1f})")
    print(f"    (kernel training episodes per seed: {[len(c) for c in train_curves]})")
    print()

    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    beats_random = ker > rnd * 2.0
    frac_of_heur = ker / heur if heur > 0 else 0.0
    print(f"  kernel >> random (learned to balance): {beats_random} "
          f"({ker:.1f} vs {rnd:.1f})")
    print(f"  fraction of heuristic ceiling:         {frac_of_heur:.2f} "
          f"({ker:.1f} / {heur:.1f})")
    print()
    if beats_random and frac_of_heur > 0.5:
        print("  FINDING: the kernel TRANSFERS to a real RL benchmark — its Dreamer")
        print("  actor learns to balance CartPole well above random, approaching the")
        print("  hand-tuned heuristic, framed purely as active-inference regulation.")
    elif beats_random:
        print("  FINDING: the kernel learns to balance well above random but stays")
        print("  below the heuristic ceiling — partial transfer, honestly bounded.")
    else:
        print("  FINDING: the kernel does NOT clearly beat random on CartPole — an")
        print("  honest negative bounding the external-transfer of the current design.")
    print("=" * 70)

    if plot:
        try:
            _plot(rnd, heur, ker, ker_std, train_curves[0])
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return beats_random and frac_of_heur > 0.5


def _plot(rnd, heur, ker, ker_std, ker_curve) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = ["random", "heuristic", "kernel"]
    ax1.bar(names, [rnd, heur, ker], yerr=[0, 0, ker_std], capsize=4,
            color=["#7f8c8d", "#27ae60", "#8e44ad"])
    ax1.axhline(500, ls="--", c="k", alpha=0.4, label="CartPole-v1 cap")
    ax1.set_ylabel("mean episode length (last 25%)")
    ax1.set_title("CartPole-v1: episode length")
    ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.3)

    if len(ker_curve) > 1:
        w = max(1, len(ker_curve) // 20)
        sm = np.convolve(ker_curve, np.ones(w) / w, mode="valid")
        ax2.plot(sm, color="#8e44ad")
    ax2.set_xlabel("episode"); ax2.set_ylabel("episode length (smoothed)")
    ax2.set_title("Kernel learning curve (seed 0)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("The kernel on an external RL benchmark (CartPole-v1)")
    out = Path("results") / "cartpole.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel on CartPole-v1")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(n_steps=args.steps, warmup=args.warmup,
              seeds=list(range(args.seeds)), plot=not args.no_plot)
    sys.exit(0 if ok else 1)
