"""
Continuous control — the fused kernel on Pendulum-v1
====================================================

Extends the in-situ RSSM fusion (§3.12) from discrete (CartPole) to CONTINUOUS
control. Pendulum-v1 (gymnasium classic control, no MuJoCo): obs 3-D
(cos θ, sin θ, θ̇), action a 1-D continuous torque in [-2, 2], dense reward
≈ −(θ² + 0.1 θ̇² + 0.001 u²) (0 = upright & still; ~−16 worst). The agent must
SWING UP and balance — the canonical simple continuous benchmark.

The canonical ``ConsciousKernel(world_model_type="rssm")`` is given a CONTINUOUS
actor (tanh-Gaussian, trained by value gradients through the differentiable
rollout). Metric: episode return (sum of rewards, 200 steps; higher/less-negative
is better). Random ≈ −1200..−1600; a good swing-up controller ≈ > −250.

NOTE: DeepMind Control (dm_control) proper needs MuJoCo (heavier); Pendulum is the
no-MuJoCo continuous benchmark used here.
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

SCALES = np.array([1.0, 1.0, 8.0], dtype=np.float32)  # cos, sin, theta_dot


def norm(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs / SCALES, dtype=torch.float32)


def baseline_random(n_steps: int, seed: int) -> list[float]:
    env = gym.make("Pendulum-v1"); env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    returns, ret = [], 0.0
    for _ in range(n_steps):
        _, r, term, trunc, _ = env.step(rng.uniform(-2, 2, size=1).astype(np.float32))
        ret += r
        if term or trunc:
            returns.append(ret); ret = 0.0; env.reset()
    env.close(); return returns


def run_kernel(n_steps: int, seed: int) -> tuple[list[float], float, list[float]]:
    torch.manual_seed(seed)
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    k = ConsciousKernel(
        obs_dim=3, action_dim=1, world_model_type="rssm",
        rssm_kwargs=dict(action_type="continuous", action_high=2.0, seed=seed,
                         deter_dim=96, stoch_dim=24, hidden=96, seq_len=20,
                         batch_size=16, horizon=12, entropy=1e-3))
    k.reset_rssm_state()
    returns, ret, psis = [], 0.0, []
    for _ in range(n_steps):
        r = k.step(norm(obs))
        env_action = np.array([float(r.action) * 2.0], dtype=np.float32)  # [-1,1]->[-2,2]
        nobs, reward, term, trunc, _ = env.step(env_action)
        k.learn_rssm(reward=float(reward), done=False)   # Pendulum never terminates
        ret += reward; psis.append(r.psi)
        if term or trunc:
            returns.append(ret); ret = 0.0
            obs, _ = env.reset(); k.reset_rssm_state()
        else:
            obs = nobs

    greedy = []
    obs, _ = env.reset(); k.reset_rssm_state(); ret = 0.0; eps = 0
    while eps < 5:
        r = k.step(norm(obs), greedy=True)
        obs, reward, term, trunc, _ = env.step(np.array([float(r.action) * 2.0], dtype=np.float32))
        ret += reward
        if term or trunc:
            greedy.append(ret); ret = 0.0; eps += 1; obs, _ = env.reset(); k.reset_rssm_state()
    env.close()
    return returns, (st.mean(greedy) if greedy else 0.0), psis


def main(n_steps: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  CONTINUOUS CONTROL — fused kernel(rssm) on Pendulum-v1")
    print("=" * 70)
    print(f"  steps={n_steps}  seeds={seeds}")
    print()
    rnd = st.mean(st.mean(baseline_random(n_steps, s)) for s in seeds)
    runs = [run_kernel(n_steps, s) for s in seeds]
    greedy = [r[1] for r in runs]
    g = st.mean(greedy); g_sd = st.pstdev(greedy) if len(greedy) > 1 else 0.0
    curve0 = runs[0][0]
    early = st.mean(curve0[: max(1, len(curve0) // 5)]) if curve0 else 0.0
    late = st.mean(curve0[-max(1, len(curve0) // 5):]) if curve0 else 0.0
    psi0 = runs[0][2]
    psi_e = st.mean(psi0[: max(1, len(psi0) // 10)])
    psi_l = st.mean(psi0[-max(1, len(psi0) // 10):])

    print(f"  random (return)        : {rnd:.0f}")
    print(f"  fused kernel (greedy)  : {g:.0f} ± {g_sd:.0f}")
    print(f"  training return        : early {early:.0f} -> late {late:.0f}")
    print(f"  Psi (integration)      : early {psi_e:.3f} -> late {psi_l:.3f}")
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    solved = g > -250
    learns = (late > early + 200) or (g > rnd + 300)
    print(f"  greedy return {g:.0f} vs random {rnd:.0f}  (solved>~-250: {solved})")
    print(f"  learns (beats random / improves): {learns}")
    print()
    if solved:
        print("  FINDING: the fused kernel SOLVES continuous control — its tanh-Gaussian")
        print("  actor (value gradients) swings up and balances the pendulum, on the")
        print("  same RSSM cycle, extending the in-situ fusion from discrete to continuous.")
    elif learns:
        print("  FINDING: the fused kernel LEARNS continuous control (well above random)")
        print("  but does not fully solve in this budget — honest partial result.")
    else:
        print("  FINDING: the fused kernel does not clearly learn Pendulum here — honest;")
        print("  continuous control may need more steps/tuning.")
    print(f"  [{'SOLVED' if solved else ('LEARNS' if learns else 'PARTIAL')}]")
    print("=" * 70)

    if plot:
        try:
            _plot(rnd, g, g_sd, curve0, psi0)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return solved or learns


def _plot(rnd, g, g_sd, curve, psis) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if len(curve) > 1:
        w = max(1, len(curve) // 20)
        ax1.plot(np.convolve(curve, np.ones(w) / w, mode="valid"), color="#16537e", label="kernel")
    ax1.axhline(rnd, ls="--", c="#7f8c8d", label=f"random ({rnd:.0f})")
    ax1.axhline(-250, ls=":", c="#27ae60", label="~solved")
    ax1.set_xlabel("episode"); ax1.set_ylabel("episode return")
    ax1.set_title("Pendulum-v1: fused kernel(rssm) continuous"); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    if len(psis) > 50:
        w = max(1, len(psis) // 100)
        ax2.plot(np.convolve(psis, np.ones(w) / w, mode="valid"), color="#8e44ad")
    ax2.set_xlabel("step"); ax2.set_ylabel("Psi (integration index)")
    ax2.set_title("Psi over learning (seed 0)"); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)
    fig.suptitle("Continuous control: the fused kernel on Pendulum-v1")
    out = Path("results") / "pendulum.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Continuous fused kernel on Pendulum-v1")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    ok = main(a.steps, list(range(a.seeds)), plot=not a.no_plot)
    sys.exit(0 if ok else 1)
