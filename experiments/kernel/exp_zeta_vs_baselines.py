"""
Zeta vs Baselines — does putting zeta frequencies IN the model help it LEARN?
============================================================================

Research question (the one this experiment exists to answer):

    Does the ConsciousKernel learn/control a task better than
      (a) a parameter-matched RNN, and
      (b) the same kernel with RANDOM frequencies instead of zeta?

To make (b) a fair test, the zeta frequencies must actually influence the model.
We wire them into the world model's transition as a temporal feature bank
(:mod:`zeta_life.kernel.temporal_features`), so the model can ANTICIPATE
time-structured dynamics instead of only reacting to them.

Why this measures *learning*, not coarse control
-------------------------------------------------
An earlier control variant (drive the state to a fixed target via EFE) could not
differentiate the arms: with a fixed target and the discrete one-hot action set,
the optimal action is stationary, so the world model's temporal prediction never
changes which action is chosen — a null by action granularity, not by hypothesis.
The half of the question with real leverage is "aprende" (learn/predict): the
temporal feature flows directly into ``predict() -> perceptual error``. So the
task here is **online next-step prediction** of a time-structured observation
stream — the clean test of arms (a) and (b).

Task
----
The environment emits an exogenous signal ``o(t)`` on the 4-simplex whose
structure is driven by a chosen set of frequencies. The kernel observes ``o(t)``
and its world model predicts ``o(t+1)`` from the time code. We measure the
one-step perceptual prediction error over the last 30% of steps (lower = the
model learned the signal better).

  - zeta-world    : o(t) driven by the non-trivial zeta zeros
  - neutral-world : o(t) driven by non-zeta frequencies (specificity control)

Four arms (same world-model architecture except the temporal bank):

  - zeta    : temporal features = zeta oscillators            (the hypothesis)
  - random  : temporal features = random oscillators          (param-matched to zeta)
  - learned : temporal features = learnable oscillators       (param-matched "RNN")
  - rnn     : no temporal features (plain GRU world model)     (classic RNN baseline)

Honest expectation:
  - zeta-world    : zeta should predict better than random and rnn (basis matches).
  - neutral-world : zeta should NOT specially beat random — the control that
    distinguishes "zeta is special" from "oscillators help in general".

Ref: Friston et al. 2015 (active inference); project ZetaLSTM finding — temporal
zeta structure helps most when the data carries zeta-frequency correlations.
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
from zeta_life.kernel.temporal_features import OscillatorBank
from zeta_life.core.zeta_constants import get_zeta_zeros

# Non-zeta frequencies for the neutral world: low-order values that do NOT
# coincide with the zeta zeros' spectrum.
NEUTRAL_FREQS = [0.7, 1.3, 2.1, 3.4, 5.5, 7.9, 9.2, 11.6]


# ---------------------------------------------------------------------------
# Environment: an exogenous time-structured signal to be predicted
# ---------------------------------------------------------------------------

class SignalStream:
    """Emits o(t) = softmax(s(t)) with s_c(t) = sum_k a_k cos(freq_k t + phase_ck).

    The signal is purely a function of time (no action feedback), so prediction
    quality reflects how well the model's temporal code matches the signal's
    spectrum — nothing else.
    """

    def __init__(self, freqs: list[float], sigma: float = 0.1, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.freqs = freqs
        self.phase = torch.rand(4, len(freqs), generator=g) * 2 * math.pi
        self.amps = torch.tensor([math.exp(-sigma * abs(f)) for f in freqs])

    def observe(self, t: float) -> torch.Tensor:
        angles = torch.tensor([[f * t for f in self.freqs] for _ in range(4)]) + self.phase
        s = (self.amps * torch.cos(angles)).sum(dim=1)
        return torch.softmax(s, dim=-1)


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------

def make_bank(arm: str, M: int, sigma: float, seed: int) -> OscillatorBank | None:
    # All fixed banks share the zeta-zero band so the comparison is fair.
    zeros = get_zeta_zeros(M)
    band = (min(zeros), max(zeros))
    if arm == "zeta":
        return OscillatorBank.zeta(M=M, sigma=sigma)
    if arm == "fourier":
        return OscillatorBank.fourier(M=M, sigma=sigma, freq_range=band)
    if arm == "log_spaced":
        return OscillatorBank.log_spaced(M=M, sigma=sigma, freq_range=band)
    if arm == "random":
        return OscillatorBank.random(M=M, sigma=sigma, seed=seed)
    if arm == "learned":
        return OscillatorBank.learned(M=M, sigma=sigma, seed=seed)
    if arm == "rnn":
        return None
    raise ValueError(f"unknown arm: {arm}")


def param_count(ck: ConsciousKernel) -> int:
    """Trainable parameters in the control path (world model + temporal bank)."""
    n = sum(p.numel() for p in ck.world_model.parameters() if p.requires_grad)
    if ck.temporal_features is not None:
        n += sum(p.numel() for p in ck.temporal_features.parameters() if p.requires_grad)
    return n


# ---------------------------------------------------------------------------
# Single run: online next-step prediction error
# ---------------------------------------------------------------------------

def run(arm: str, world_freqs: list[float], n_steps: int, M: int,
        sigma: float, seed: int) -> tuple[float, int]:
    torch.manual_seed(seed)
    stream = SignalStream(world_freqs, sigma=sigma, seed=seed)
    bank = make_bank(arm, M, sigma, seed)
    ck = ConsciousKernel(
        action_mode="reactive",
        temporal_features=bank,
        # Isolate the world-model learning effect: no dream/reflect overhead.
        reflect_interval=10_000,
        dream_interval=10_000,
    )
    errors: list[float] = []
    for t in range(n_steps):
        obs = stream.observe(float(t))
        result = ck.step(obs)
        # Perceptual error = ||predicted o(t) - actual o(t)|| for this step's
        # prediction (the world model predicted from the previous latent + time
        # code). Lower over time = it learned the signal.
        errors.append(result.errors["perceptual"])
    tail = max(1, int(n_steps * 0.3))
    return st.mean(errors[-tail:]), param_count(ck)


def run_arm(arm: str, world_freqs: list[float], seeds: list[int],
            n_steps: int, M: int, sigma: float) -> tuple[float, float, int]:
    scores, params = [], 0
    for s in seeds:
        score, params = run(arm, world_freqs, n_steps, M, sigma, s)
        scores.append(score)
    mean = st.mean(scores)
    std = st.pstdev(scores) if len(scores) > 1 else 0.0
    return mean, std, params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ARMS = ["zeta", "fourier", "log_spaced", "random", "learned", "rnn"]


def main(n_steps: int, seeds: list[int], M: int, sigma: float, plot: bool) -> bool:
    print("=" * 70)
    print("  ZETA vs BASELINES — zeta frequencies IN the model (prediction)")
    print("=" * 70)
    print(f"  task = online next-step prediction of a time-structured signal")
    print(f"  steps={n_steps}  seeds={seeds}  M={M}  sigma={sigma}")
    print(f"  metric = mean perceptual error over last 30% (LOWER is better)")
    print()

    worlds = {
        "zeta-world": get_zeta_zeros(min(M, 8)),
        "neutral-world": NEUTRAL_FREQS,
    }

    results: dict[str, dict[str, tuple[float, float, int]]] = {}
    for world_name, wfreqs in worlds.items():
        print(f"  --- {world_name} (signal freqs: "
              f"{[round(f, 2) for f in wfreqs[:4]]}...) ---")
        results[world_name] = {}
        for arm in ARMS:
            mean, std, params = run_arm(arm, wfreqs, seeds, n_steps, M, sigma)
            results[world_name][arm] = (mean, std, params)
            print(f"    {arm:8s}: pred error = {mean:.5f} ± {std:.5f}  "
                  f"(params={params})")
        print()

    # ----- verdict -----
    print("=" * 70)
    print("  VERDICT  (lower error is better)")
    print("=" * 70)
    zw = results["zeta-world"]
    nw = results["neutral-world"]

    def rel_gain(a: float, b: float) -> float:
        # how much lower is a than b, as a fraction of b
        return (b - a) / b if b > 0 else 0.0

    margin = 0.02  # require >2% relative improvement to call it a win
    zeta_beats_random_zw = rel_gain(zw["zeta"][0], zw["random"][0]) > margin
    zeta_beats_rnn_zw = rel_gain(zw["zeta"][0], zw["rnn"][0]) > margin
    zeta_beats_learned_zw = rel_gain(zw["zeta"][0], zw["learned"][0]) > margin
    # Specificity control: on the neutral world zeta must NOT clearly beat random.
    specificity = rel_gain(nw["zeta"][0], nw["random"][0]) <= margin

    print(f"  [zeta-world]  zeta < random  : {zeta_beats_random_zw} "
          f"({zw['zeta'][0]:.5f} vs {zw['random'][0]:.5f}, "
          f"{100*rel_gain(zw['zeta'][0], zw['random'][0]):+.1f}%)")
    print(f"  [zeta-world]  zeta < rnn     : {zeta_beats_rnn_zw} "
          f"({zw['zeta'][0]:.5f} vs {zw['rnn'][0]:.5f}, "
          f"{100*rel_gain(zw['zeta'][0], zw['rnn'][0]):+.1f}%)")
    print(f"  [zeta-world]  zeta < learned : {zeta_beats_learned_zw} "
          f"({zw['zeta'][0]:.5f} vs {zw['learned'][0]:.5f}, "
          f"{100*rel_gain(zw['zeta'][0], zw['learned'][0]):+.1f}%)")
    print(f"  [neutral]     zeta NOT special vs random : {specificity} "
          f"({nw['zeta'][0]:.5f} vs {nw['random'][0]:.5f})")
    # The realistic case: on a world NOT built from zeta, is the recommended
    # fixed lattice (fourier) as good as zeta? If so, fourier is the better
    # default (principled, reproducible, no number-theoretic baggage).
    fourier_ok_neutral = rel_gain(nw["fourier"][0], nw["zeta"][0]) >= -margin
    print(f"  [neutral]     fourier >= zeta (good default) : {fourier_ok_neutral} "
          f"({nw['fourier'][0]:.5f} vs {nw['zeta'][0]:.5f})")
    print()

    hypothesis_supported = zeta_beats_random_zw and zeta_beats_rnn_zw and specificity
    print(f"  HYPOTHESIS SUPPORTED: {hypothesis_supported}")
    print("    (zeta predicts better than random+rnn when the world has zeta")
    print("     structure, AND loses its edge when it does not — a specific,")
    print("     not generic, effect)")
    print("=" * 70)

    if plot:
        try:
            _plot(results)
        except Exception as e:  # noqa: BLE001 - plotting is best-effort
            print(f"  (plot skipped: {e})")

    return hypothesis_supported


def _plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ARMS))
    width = 0.38
    colors = {"zeta-world": "#8e44ad", "neutral-world": "#95a5a6"}
    for i, (world, res) in enumerate(results.items()):
        means = [res[a][0] for a in ARMS]
        errs = [res[a][1] for a in ARMS]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=4,
               label=world, color=colors.get(world, None), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("mean perceptual prediction error  [lower = better]")
    ax.set_title("Zeta frequencies in the model: prediction error, zeta vs baselines")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out = Path("results") / "zeta_vs_baselines.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zeta vs baselines prediction benchmark")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--M", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(
        n_steps=args.steps,
        seeds=list(range(args.seeds)),
        M=args.M,
        sigma=args.sigma,
        plot=not args.no_plot,
    )
    sys.exit(0 if ok else 1)
