"""
Agency — Active inference action selection
==========================================

Does the kernel act with PURPOSE when given a preference, or only react?

We place the kernel in a reactive environment whose state it can influence
through its actions, give it a preferred observation C, and compare two action
policies:

  - reactive : action = softmax(observation)          (no goal — just reflects)
  - efe      : action = argmin_a KL(C || imagine(a))   (expected free energy)

Both share an exploratory warm-up so the world model learns the
action -> consequence dynamics (without exploration the model never sees
diverse actions and planning is blind). After warm-up, the EFE planner exploits
the learned model to steer the environment toward C.

Success criterion: the EFE planner drives cosine(state, C) clearly higher than
the reactive baseline — agency adds value on top of perception.

Ref: Friston et al. 2015, "Active inference and epistemic value".
"""
import sys
import argparse
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel

TARGET = torch.tensor([0.7, 0.1, 0.1, 0.1])


class ReactiveEnv:
    def __init__(self, r: float = 0.3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.state = torch.rand(4, generator=g)
        self.state = self.state / self.state.sum()
        self.r = r

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.state = ((1 - self.r) * self.state + self.r * action).clamp(min=1e-6)
        return self.state / self.state.sum()


def run(planner: bool, n_steps: int, warmup: int, seed: int) -> float:
    torch.manual_seed(seed)
    env = ReactiveEnv(seed=seed)
    ck = ConsciousKernel(
        action_mode="efe" if planner else "reactive",
        preference=TARGET if planner else None,
    )
    obs = env.state / env.state.sum()
    sims = []
    for t in range(n_steps):
        result = ck.step(obs)
        if t < warmup:
            a = torch.rand(4)
            a = a / a.sum()
            ck.last_action = a.detach()  # align WM training with executed action
        else:
            a = result.action
        obs = env.step(a)
        sims.append(float(F.cosine_similarity(env.state.unsqueeze(0), TARGET.unsqueeze(0))))
    return st.mean(sims[-100:])


def main(n_steps: int = 900, warmup: int = 400):
    print("=" * 60)
    print("  AGENCY — Active Inference Action Selection")
    print("=" * 60)
    print(f"  preference C = {TARGET.tolist()}")
    print(f"  steps={n_steps}  warmup={warmup} (exploratory)")
    print()
    seeds = [0, 1, 2, 3, 4]
    reactive = st.mean([run(False, n_steps, warmup, s) for s in seeds])
    planner = st.mean([run(True, n_steps, warmup, s) for s in seeds])

    print(f"  reactive policy (softmax obs):     cosine(state, C) = {reactive:.4f}")
    print(f"  efe policy      (active inference): cosine(state, C) = {planner:.4f}")
    print(f"  improvement: {planner - reactive:+.4f}")
    print()
    passed = (planner > 0.85) and (planner - reactive > 0.1)
    print(f"  [{'PASS' if passed else 'FAIL'}] agency steers the world toward the "
          f"preference and beats the reactive baseline")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agency / active inference experiment")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--warmup", type=int, default=400)
    args = parser.parse_args()
    ok = main(n_steps=args.steps, warmup=args.warmup)
    sys.exit(0 if ok else 1)
