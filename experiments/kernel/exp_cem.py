"""
CEM vs random shooting — an honest negative for this control regime
===================================================================

Deepens the "controla" planner (exp_control.py): the continuous EFE planner used
one-shot random shooting; the Cross-Entropy Method (CEM) refines the sampling
distribution over iterations. CEM is implemented in the kernel (efe_cem_iters);
this experiment asks whether it actually helps here.

HONEST FINDING: it does not, reliably. Across action dimensions (4..32) and
sample budgets, the CEM-vs-random difference flips sign between configurations
and stays within seed noise — no robust advantage. The reason is structural: the
inertial control task is UNIMODAL (the optimal action is essentially the target
itself, and the fixed point of (1-r)s + r a is a), so one-shot random shooting
already finds it; there is no rugged/multimodal action landscape for CEM's
refinement to exploit. This is the agency "scissor" again: the extension only
matters where the simple method fails, and here it does not.

CEM is retained as a capability for harder regimes (multimodal action
landscapes, sequential/long-horizon plans, much higher dimension) where random
shooting's coverage genuinely breaks down -- not claimed as a win here.

Task: the inertial control environment generalised to D dims; target = a smooth
decreasing D-dim distribution. Metric: control error ||state - C|| over the tail.
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

CEM_ITERS = 3
DIMS = [4, 8, 16, 32]
BUDGET = 48  # total samples per step (matched between methods)


def target_for(dim: int) -> torch.Tensor:
    # Smooth decreasing distribution over all D components (a real D-dim profile).
    t = F.softmax(torch.linspace(1.5, -1.5, dim), dim=-1)
    return t / t.sum()


class InertiaEnv:
    def __init__(self, dim: int, r: float = 0.3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        s = torch.rand(dim, generator=g)
        self.state = s / s.sum()
        self.r = r

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.state = ((1 - self.r) * self.state + self.r * action.detach()).clamp(min=1e-6)
        self.state = self.state / self.state.sum()
        return self.state


def run(method: str, dim: int, n_steps: int, warmup: int, seed: int) -> float:
    torch.manual_seed(seed)
    target = target_for(dim)
    if method == "cem":
        kwargs = dict(efe_n_samples=max(2, BUDGET // CEM_ITERS), efe_cem_iters=CEM_ITERS)
    else:
        kwargs = dict(efe_n_samples=BUDGET, efe_cem_iters=0)
    env = InertiaEnv(dim, seed=seed)
    ck = ConsciousKernel(obs_dim=dim, action_mode="efe", preference=target,
                         efe_obs_norm="l1", **kwargs)
    obs = env.state.clone()
    dists = []
    for t in range(n_steps):
        result = ck.step(obs)
        if t < warmup:
            a = torch.rand(dim)
            a = a / a.sum()
            ck.last_action = a.detach()
        else:
            a = result.action
        obs = env.step(a)
        dists.append(float(torch.linalg.vector_norm(env.state - target)))
    tail = max(1, int(n_steps * 0.2))
    return st.mean(dists[-tail:])


def run_cell(method: str, dim: int, seeds, n_steps, warmup):
    ds = [run(method, dim, n_steps, warmup, s) for s in seeds]
    return st.mean(ds), (st.pstdev(ds) if len(ds) > 1 else 0.0)


def main(n_steps: int, warmup: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 70)
    print("  CEM vs RANDOM SHOOTING — does refinement help this control task?")
    print("=" * 70)
    print(f"  matched budget={BUDGET}/step  CEM iters={CEM_ITERS}  dims={DIMS}")
    print(f"  steps={n_steps}  seeds={seeds}  metric=||state-C|| over last 20%")
    print()

    res = {}
    print(f"  {'dim':>4s} {'random':>16s} {'cem':>16s} {'rel.improv':>11s}")
    for d in DIMS:
        rm, rs = run_cell("random", d, seeds, n_steps, warmup)
        cm, cs = run_cell("cem", d, seeds, n_steps, warmup)
        rel = (rm - cm) / rm if rm > 0 else 0.0
        res[d] = {"random": (rm, rs), "cem": (cm, cs), "rel": rel}
        print(f"  {d:4d} {rm:9.4f} ± {rs:.4f} {cm:9.4f} ± {cs:.4f} {100*rel:9.1f}%")
    print()

    print("=" * 70)
    print("  VERDICT (characterisation, not a win/lose gate)")
    print("=" * 70)
    rels = [res[d]["rel"] for d in DIMS]
    # "within noise" if |rel improvement| is small vs the seed spread.
    within_noise = []
    for d in DIMS:
        rm, rs = res[d]["random"]
        cm, cs = res[d]["cem"]
        gap = abs(rm - cm)
        within_noise.append(gap <= (rs + cs))
    reliable = any((res[d]["rel"] > 0.10) and within_noise[i] is False
                   for i, d in enumerate(DIMS))
    print(f"  rel. improvement by dim: "
          f"{ {d: round(100*res[d]['rel'], 1) for d in DIMS} } (%)")
    print(f"  within seed noise by dim: { dict(zip(DIMS, within_noise)) }")
    print()
    if not reliable:
        print("  FINDING: no reliable CEM advantage in this regime — the sign flips")
        print("  across dims and the gaps sit within seed noise. The inertial control")
        print("  task is unimodal, so random shooting already suffices (agency scissor).")
        print("  CEM is kept as a capability for harder action landscapes.")
    else:
        print("  FINDING: CEM shows a reliable advantage at some dimension here.")
    print("=" * 70)

    if plot:
        try:
            _plot(res)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot skipped: {e})")
    return True


def _plot(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.errorbar(DIMS, [res[d]["random"][0] for d in DIMS],
                 yerr=[res[d]["random"][1] for d in DIMS], marker="s",
                 color="#e67e22", capsize=4, label="random shooting")
    ax1.errorbar(DIMS, [res[d]["cem"][0] for d in DIMS],
                 yerr=[res[d]["cem"][1] for d in DIMS], marker="o",
                 color="#2980b9", capsize=4, label=f"CEM ({CEM_ITERS} iters)")
    ax1.set_xlabel("action dimension")
    ax1.set_ylabel("control error ||state - C||")
    ax1.set_title(f"Control error vs dimension (budget={BUDGET})")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.bar([str(d) for d in DIMS], [100 * res[d]["rel"] for d in DIMS], color="#2980b9")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("action dimension")
    ax2.set_ylabel("CEM relative improvement (%)")
    ax2.set_title("CEM vs random (sign flips, within noise)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("CEM vs random shooting: no reliable advantage in this control regime")
    out = Path("results") / "cem.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEM vs random shooting vs dimension")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    BUDGET = args.budget
    ok = main(n_steps=args.steps, warmup=args.warmup,
              seeds=list(range(args.seeds)), plot=not args.no_plot)
    sys.exit(0 if ok else 1)
