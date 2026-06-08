"""
Psi robustness — does the self-calibrating metric remove the clamp dependence?
=============================================================================

Research question (validates the 2026-06 Psi fix):

    The integration index Psi used to depend on a hand-tuned clamp constant
    `psi_prec_half`: F_i = w_prec * prec_mean/(prec_mean + psi_prec_half). When
    the precisions were trained (they grow toward inverse error variance), that
    fixed constant had to be retuned or Psi degraded — too small a half throttled
    F_i high and suppressed the coherent signal; too large a half let noise leak
    through. The fix replaces the fixed half with a SELF-CALIBRATING one (an EMA
    of the mean precision), so prec_term sits near 0.5 regardless of scale.

    This experiment tests the claim head-on: sweep `psi_prec_half` across two
    decades, in BOTH modes (adaptive vs fixed), and measure the discrimination
    gap  Psi(coherent) - Psi(noise).  If the fix works:
      - adaptive: the gap is ~flat and large across the whole sweep (the former
        constant is no longer load-bearing — it only bootstraps the EMA);
      - fixed:    the gap degrades at the extremes (the old fragility).

Metric: steady-state Psi (mean of the last 30 steps), averaged over seeds.
"""

from __future__ import annotations

import sys
import argparse
import statistics as st
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel import ConsciousKernel

PATTERN = torch.tensor([0.5, 0.2, 0.2, 0.1])
HALVES = [0.5, 1.0, 2.0, 5.0, 20.0, 50.0, 100.0]


def coherent(_t: int) -> torch.Tensor:
    return PATTERN + 0.01 * torch.randn(4)


def noise(_t: int) -> torch.Tensor:
    return torch.softmax(torch.randn(4), dim=-1)


def run(stimulus_fn, half: float, adaptive: bool, n_steps: int, seed: int) -> float:
    torch.manual_seed(seed)
    ck = ConsciousKernel(
        psi_mode="hill",
        psi_prec_half=half,
        psi_prec_adaptive=adaptive,
        reflect_interval=10_000,
        dream_interval=10_000,
    )
    psis = [ck.step(stimulus_fn(t)).psi for t in range(n_steps)]
    return sum(psis[-30:]) / 30


def cell(stimulus_fn, half: float, adaptive: bool, n_steps: int, seeds: list[int]) -> float:
    return st.mean(run(stimulus_fn, half, adaptive, n_steps, s) for s in seeds)


def main(n_steps: int, seeds: list[int], plot: bool) -> bool:
    print("=" * 72)
    print("  PSI ROBUSTNESS — self-calibrating vs fixed clamp")
    print("=" * 72)
    print(f"  steps={n_steps}  seeds={seeds}  psi_prec_half sweep={HALVES}")
    print(f"  metric = steady-state Psi (last 30 steps); gap = coherent - noise")
    print()

    res: dict[tuple[str, float], tuple[float, float, float]] = {}
    for adaptive in (True, False):
        mode = "adaptive" if adaptive else "fixed"
        print(f"  --- {mode} ---")
        print(f"  {'half':>7s} {'coherent':>9s} {'noise':>8s} {'gap':>8s}")
        for h in HALVES:
            coh = cell(coherent, h, adaptive, n_steps, seeds)
            noi = cell(noise, h, adaptive, n_steps, seeds)
            res[(mode, h)] = (coh, noi, coh - noi)
            print(f"  {h:7.1f} {coh:9.4f} {noi:8.4f} {coh - noi:8.4f}")
        print()

    # ----- verdict -----
    adaptive_gaps = [res[("adaptive", h)][2] for h in HALVES]
    fixed_gaps = [res[("fixed", h)][2] for h in HALVES]
    adaptive_spread = max(adaptive_gaps) - min(adaptive_gaps)
    fixed_spread = max(fixed_gaps) - min(fixed_gaps)

    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print(f"  discrimination gap across the sweep:")
    print(f"    adaptive: min={min(adaptive_gaps):.4f} max={max(adaptive_gaps):.4f} "
          f"spread={adaptive_spread:.4f}")
    print(f"    fixed:    min={min(fixed_gaps):.4f} max={max(fixed_gaps):.4f} "
          f"spread={fixed_spread:.4f}")
    print()
    robust = adaptive_spread < fixed_spread
    discriminates = min(adaptive_gaps) > 0.5
    print(f"  adaptive gap is flatter than fixed (clamp not load-bearing): {robust}")
    print(f"  adaptive discriminates across the whole sweep (gap>0.5):     {discriminates}")
    print()
    passed = robust and discriminates
    print(f"  [{'PASS' if passed else 'FAIL'}] the self-calibrating Psi removes the "
          f"clamp dependence while keeping discrimination")
    print("=" * 72)

    if plot:
        try:
            _plot(res)
        except Exception as e:  # noqa: BLE001 - plotting is best-effort
            print(f"  (plot skipped: {e})")
    return passed


def _plot(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: discrimination gap vs half
    ax1.plot(HALVES, [res[("adaptive", h)][2] for h in HALVES], "o-",
             color="#2980b9", label="adaptive (self-calibrating)")
    ax1.plot(HALVES, [res[("fixed", h)][2] for h in HALVES], "s--",
             color="#e67e22", label="fixed clamp (old)")
    ax1.axhline(0.5, ls=":", c="k", alpha=0.4, label="discrimination floor")
    ax1.set_xscale("log")
    ax1.set_xlabel("psi_prec_half (former clamp constant)")
    ax1.set_ylabel("discrimination gap  Psi(coherent) - Psi(noise)")
    ax1.set_title("Gap vs clamp constant\n(adaptive should be flat; fixed degrades)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: raw coherent / noise lines
    for mode, ls, m in (("adaptive", "-", "o"), ("fixed", "--", "s")):
        ax2.plot(HALVES, [res[(mode, h)][0] for h in HALVES], ls, marker=m,
                 color="#27ae60", label=f"coherent ({mode})")
        ax2.plot(HALVES, [res[(mode, h)][1] for h in HALVES], ls, marker=m,
                 color="#c0392b", label=f"noise ({mode})")
    ax2.set_xscale("log")
    ax2.set_xlabel("psi_prec_half (former clamp constant)")
    ax2.set_ylabel("steady-state Psi")
    ax2.set_title("Raw Psi: coherent vs noise\n(fixed: coherent dips low-half, noise rises high-half)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Self-calibrating Psi removes the hand-tuned clamp dependence")
    out = Path("results") / "psi_robustness.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psi robustness vs the clamp constant")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok = main(n_steps=args.steps, seeds=list(range(args.seeds)), plot=not args.no_plot)
    sys.exit(0 if ok else 1)
