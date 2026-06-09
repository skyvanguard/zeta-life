"""
Richer continuous control — the fused kernel on MuJoCo (Reacher-v5)
==================================================================

Extends §3.13 (Pendulum, 1-D action) to a **higher-dimensional MuJoCo** task:
Reacher-v5 (gymnasium MuJoCo) — a 2-joint arm with a **2-D continuous action**
and 10-D observation, dense reward (−distance to a random target − control cost),
50-step episodes. The same fused ``ConsciousKernel(world_model_type="rssm",
action_type="continuous")`` controls it.

NOTE on dm_control: dm_control 1.0.41 conflicts with the installed mujoco 3.9
(`MjModel.flex_bandwidth` removed), so we use gymnasium's MuJoCo suite — the SAME
MuJoCo engine — for a richer, higher-dim continuous task. Honest constraint: our
training is CPU-bound, so MuJoCo locomotion (Hopper/HalfCheetah, ~1e6 steps) is out
of reach; Reacher (short episodes, dense reward) is the tractable higher-dim target.

Obs are standardised by mean/std from a random rollout (robust across MuJoCo envs).
Metric: episode return vs random + a frozen greedy eval.
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
    print("gymnasium not installed — `pip install gymnasium mujoco imageio`")
    sys.exit(1)


def obs_normalizer(env_name: str, n: int = 2000):
    env = gym.make(env_name)
    obs, _ = env.reset(seed=12345)
    buf = [obs]
    for _ in range(n):
        o, _, term, trunc, _ = env.step(env.action_space.sample())
        buf.append(o)
        if term or trunc:
            o, _ = env.reset()
            buf.append(o)
    env.close()
    arr = np.array(buf, dtype=np.float32)
    mean = arr.mean(0)
    std = arr.std(0).clip(min=1e-3)
    return mean, std


def _act_repeat(env, action, repeat):
    total, done, nobs = 0.0, False, None
    for _ in range(repeat):
        nobs, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            done = True
            break
    return nobs, total, done


def baseline_random(env_name: str, n_steps: int, seed: int) -> list[float]:
    env = gym.make(env_name); env.reset(seed=seed)
    returns, ret = [], 0.0
    for _ in range(n_steps):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        ret += r
        if term or trunc:
            returns.append(ret); ret = 0.0; env.reset()
    env.close(); return returns


def run_kernel(env_name, n_steps, seed, action_repeat, mean, std):
    torch.manual_seed(seed)
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    adim = env.action_space.shape[0]
    high = torch.tensor(env.action_space.high, dtype=torch.float32)
    m = torch.tensor(mean); s = torch.tensor(std)
    norm = lambda o: ((torch.tensor(o, dtype=torch.float32) - m) / s)
    k = ConsciousKernel(
        obs_dim=len(mean), action_dim=adim, world_model_type="rssm",
        rssm_kwargs=dict(action_type="continuous", action_high=1.0, seed=seed,
                         deter_dim=128, stoch_dim=32, hidden=128, seq_len=24,
                         batch_size=16, horizon=15, entropy=3e-3))
    k.reset_rssm_state()
    returns, ret, psis = [], 0.0, []
    for _ in range(n_steps):
        r = k.step(norm(obs))
        env_action = (r.action * high).numpy().astype(np.float32)   # model [-1,1] -> env bounds
        nobs, reward, done = _act_repeat(env, env_action, action_repeat)
        k.learn_rssm(reward=float(reward), done=False)
        ret += reward; psis.append(r.psi)
        if done:
            returns.append(ret); ret = 0.0
            obs, _ = env.reset(); k.reset_rssm_state()
        else:
            obs = nobs

    greedy, ret, eps = [], 0.0, 0
    obs, _ = env.reset(); k.reset_rssm_state()
    while eps < 8:
        r = k.step(norm(obs), greedy=True)
        env_action = (r.action * high).numpy().astype(np.float32)
        obs, reward, done = _act_repeat(env, env_action, action_repeat)
        ret += reward
        if done:
            greedy.append(ret); ret = 0.0; eps += 1; obs, _ = env.reset(); k.reset_rssm_state()
    env.close()
    return returns, (st.mean(greedy) if greedy else 0.0), psis


def main(env_name, n_steps, seeds, action_repeat, plot) -> bool:
    print("=" * 70)
    print(f"  RICHER CONTINUOUS CONTROL — fused kernel(rssm) on {env_name} (MuJoCo)")
    print("=" * 70)
    print(f"  agent_steps={n_steps}  action_repeat={action_repeat}  seeds={seeds}")
    mean, std = obs_normalizer(env_name)
    print(f"  obs dim={len(mean)}  (standardised)")
    print()
    rnd = st.mean(st.mean(baseline_random(env_name, n_steps * action_repeat, s)) for s in seeds)
    runs = [run_kernel(env_name, n_steps, s, action_repeat, mean, std) for s in seeds]
    greedy = [r[1] for r in runs]
    g = st.mean(greedy); g_sd = st.pstdev(greedy) if len(greedy) > 1 else 0.0
    c0 = runs[0][0]
    early = st.mean(c0[: max(1, len(c0) // 5)]) if c0 else 0.0
    late = st.mean(c0[-max(1, len(c0) // 5):]) if c0 else 0.0
    psi0 = runs[0][2]
    psi_e = st.mean(psi0[: max(1, len(psi0) // 10)]); psi_l = st.mean(psi0[-max(1, len(psi0) // 10):])

    print(f"  random (return)        : {rnd:.1f}")
    print(f"  fused kernel (greedy)  : {g:.1f} ± {g_sd:.1f}")
    print(f"  training return        : early {early:.1f} -> late {late:.1f}")
    print(f"  Psi (integration)      : early {psi_e:.3f} -> late {psi_l:.3f}")
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    learns = g > rnd + abs(rnd) * 0.25 + 1.0
    print(f"  greedy {g:.1f} vs random {rnd:.1f}  (learns: {learns})")
    if learns:
        print(f"  FINDING: the fused kernel LEARNS a higher-dim MuJoCo continuous task")
        print(f"  ({env_name}, {len(mean)}-D obs / {runs and ''}2-D action) — well above random,")
        print(f"  extending continuous control beyond Pendulum. CPU budget bounds the depth.")
    else:
        print(f"  FINDING: no clear learning on {env_name} in this budget — honest;")
        print(f"  richer MuJoCo tasks need far more steps (GPU) than CPU allows here.")
    print(f"  [{'LEARNS' if learns else 'PARTIAL'}]")
    print("=" * 70)

    if plot:
        try:
            _plot(env_name, rnd, g, g_sd, c0, psi0)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return learns


def _plot(env_name, rnd, g, g_sd, curve, psis) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if len(curve) > 1:
        w = max(1, len(curve) // 20)
        ax1.plot(np.convolve(curve, np.ones(w) / w, mode="valid"), color="#b8431f", label="kernel")
    ax1.axhline(rnd, ls="--", c="#7f8c8d", label=f"random ({rnd:.0f})")
    ax1.set_xlabel("episode"); ax1.set_ylabel("episode return")
    ax1.set_title(f"{env_name}: fused kernel(rssm)"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    if len(psis) > 50:
        w = max(1, len(psis) // 100)
        ax2.plot(np.convolve(psis, np.ones(w) / w, mode="valid"), color="#8e44ad")
    ax2.set_xlabel("step"); ax2.set_ylabel("Psi (integration index)")
    ax2.set_title("Psi over learning (seed 0)"); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)
    fig.suptitle(f"Higher-dim MuJoCo continuous control: {env_name}")
    out = Path("results") / "mujoco.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fused kernel on a MuJoCo continuous task")
    p.add_argument("--env", type=str, default="Reacher-v5")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    ok = main(a.env, a.steps, list(range(a.seeds)), a.action_repeat, plot=not a.no_plot)
    sys.exit(0 if ok else 1)
