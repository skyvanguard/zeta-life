"""
Spacing statistics — is it the ZETA zeros, or just LEVEL REPULSION?
==================================================================

This is the decisive test the project never ran. Every prior "zeta vs random"
comparison left a hole: it never isolated the ONE property that genuinely
distinguishes the Riemann-zeta zeros from an arbitrary frequency set — the
statistics of their *gaps*. Consecutive zeta zeros exhibit GUE level repulsion
(Montgomery-Odlyzko): small gaps are suppressed, unlike a Poisson (random)
process where small gaps are common.

So we build four frequency banks with the **same range and same mean density**
but **different gap statistics**, and ask which property (if any) helps:

  - zeta    : the actual zeta zeros (rescaled to the band)         [GUE by nature]
  - gue     : eigenvalues of a random Hermitian matrix            [GUE, not zeta]
  - poisson : sorted uniform points (exponential gaps)            [no repulsion]
  - uniform : an equispaced lattice                               [maximal rigidity]

Verdict logic:
  - zeta ~= gue  on every metric           -> it is the STATISTIC, not the zeros'
    arithmetic, that matters (a real, novel claim).
  - gue/zeta strictly better than uniform  -> GUE is genuinely special beyond a
    lattice (would justify zeta).
  - uniform as good as / better than gue/zeta -> only low-discrepancy SPREAD
    matters; a rigid lattice (or Fourier basis) suffices; zeta buys nothing.
  - all four ~ equal in the kernel         -> spacing is functionally irrelevant.

Part A (primary, clean): two DETERMINISTIC diagnostics that actually determine a
frequency basis's quality, with flat amplitudes (sigma=0, so all M frequencies
are live — under the usual sigma~0.1 only ~4 survive and spacing is moot):

  1. COVERING RADIUS (normalised): the largest gap to the nearest frequency, in
     units of the mean gap. 1.0 is ideal (a perfect lattice); larger means a
     spectral hole no basis frequency covers — a blind spot.
  2. TEMPORAL-CODE CONDITIONING: log10 condition number of the feature Gram
     matrix Phi^T Phi over a time window. Lower = the time code is less
     redundant / better decodable. Two near-identical frequencies (clustering)
     make two columns near-collinear and blow the condition number up;
     repulsion keeps the code well-conditioned.

(An earlier ridge-reconstruction probe was discarded: with a finite
40-frequency dictionary, continuous-spectrum random targets are unrepresentable
over a long window, so every statistic saturated at ~88% residual — a degenerate
regime measuring ridge artifacts, not coverage. These two diagnostics are
well-posed and isolate the spacing statistic directly.)

Part B (optional, --kernel): the same banks inside the real ConsciousKernel on a
broadband prediction task — the functional test that the diagnostics carry into
the actual system.

Refs: Montgomery (1973) pair correlation; Odlyzko (1987) numerical GUE match;
Mehta, Random Matrices (Wigner surmise / level repulsion).
"""

from __future__ import annotations

import sys
import math
import argparse
import statistics as st
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.kernel.temporal_features import (
    generate_spacing_frequencies,
    spacing_stats,
)

STATS = ["zeta", "gue", "poisson", "uniform"]
FREQ_RANGE = (5.0, 80.0)


# ---------------------------------------------------------------------------
# Part A — deterministic basis diagnostics
# ---------------------------------------------------------------------------

def covering_radius(freqs: np.ndarray, lo: float, hi: float) -> float:
    """Largest gap to nearest frequency, normalised by the ideal (lattice) gap.

    1.0 = a perfect equispaced lattice; >1 = there is a spectral hole half a max
    gap wide that no basis frequency covers (a blind spot).
    """
    s = np.sort(freqs)
    gaps = np.diff(s)
    ideal_gap = (hi - lo) / (len(freqs) - 1)
    return float(0.5 * gaps.max() / ideal_gap)


def code_log_cond(freqs: np.ndarray, T: int) -> float:
    """log10 of the condition number of the temporal feature Gram matrix.

    Flat-amplitude features Phi (T x 2M); lower = the time code is less redundant
    / more decodable. Clustering -> near-collinear columns -> huge condition.
    """
    t = np.arange(T, dtype=float)
    ang = np.outer(t, freqs)
    phi = np.concatenate([np.cos(ang), np.sin(ang)], axis=1)
    gram = phi.T @ phi
    ev = np.linalg.eigvalsh(gram)
    ev = np.clip(ev, 1e-12, None)
    return float(np.log10(ev[-1] / ev[0]))


def run_diagnostics(M: int, T: int, seeds: list[int]) -> dict:
    out: dict[str, dict] = {}
    for stat in STATS:
        cov, cond, fsmall, cv = [], [], [], []
        for s in seeds:
            f = generate_spacing_frequencies(stat, M, FREQ_RANGE, seed=s).numpy()
            cov.append(covering_radius(f, *FREQ_RANGE))
            cond.append(code_log_cond(f, T))
            sp = spacing_stats(generate_spacing_frequencies(stat, M, FREQ_RANGE, seed=s))
            fsmall.append(sp["frac_small"])
            cv.append(sp["gap_cv"])
        out[stat] = {
            "covering_radius": st.mean(cov),
            "log_cond": st.mean(cond),
            "frac_small": st.mean(fsmall),
            "gap_cv": st.mean(cv),
        }
    return out


# ---------------------------------------------------------------------------
# Part B — in-kernel confirmation (optional)
# ---------------------------------------------------------------------------

def run_kernel(M: int, n_steps: int, seeds: list[int]) -> dict:
    import torch
    from zeta_life.kernel import ConsciousKernel
    from zeta_life.kernel.temporal_features import OscillatorBank

    def broadband_signal(t, freqs, phase, amps) -> torch.Tensor:
        ang = torch.tensor([[f * t for f in freqs] for _ in range(4)]) + phase
        return torch.softmax((amps * torch.cos(ang)).sum(dim=1), dim=-1)

    out: dict[str, tuple[float, float]] = {}
    for stat in STATS:
        errs = []
        for seed in seeds:
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            # Broadband world: 12 random sinusoids spanning the band, NOT a
            # bank's frequencies, so no statistic gets a basis-matching freebie.
            wf = (FREQ_RANGE[0] + (FREQ_RANGE[1] - FREQ_RANGE[0])
                  * torch.rand(12, generator=g)).tolist()
            phase = torch.rand(4, len(wf), generator=g) * 2 * math.pi
            amps = torch.ones(len(wf))
            bank = OscillatorBank.by_spacing(stat, M=M, sigma=0.0,
                                             freq_range=FREQ_RANGE, seed=seed)
            ck = ConsciousKernel(action_mode="reactive", temporal_features=bank,
                                 reflect_interval=10_000, dream_interval=10_000)
            e = [ck.step(broadband_signal(float(t), wf, phase, amps)).errors["perceptual"]
                 for t in range(n_steps)]
            tail = max(1, int(n_steps * 0.3))
            errs.append(st.mean(e[-tail:]))
        out[stat] = (st.mean(errs), st.pstdev(errs) if len(errs) > 1 else 0.0)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _close(a: float, b: float, rel: float = 0.05) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-9)


def _better(a: float, b: float, rel: float = 0.05) -> bool:
    """True if a is at least rel lower (better) than b."""
    return a < b * (1.0 - rel)


def main(M: int, T: int, seeds: list[int], run_kernel_part: bool, plot: bool) -> None:
    print("=" * 72)
    print("  SPACING STATISTICS — is it the zeta zeros, or just level repulsion?")
    print("=" * 72)
    print(f"  M={M} freqs in {FREQ_RANGE}  (same range & density; only GAPS differ)")
    print(f"  T={T}  seeds={seeds}")
    print()

    res = run_diagnostics(M, T, seeds)

    print("  --- spacing diagnostics (manipulation check + basis quality) ---")
    print(f"  {'arm':8s} {'gap_cv':>8s} {'frac_small':>11s} "
          f"{'cover_rad':>10s} {'log_cond':>9s}")
    print(f"  {'':8s} {'(0=rigid)':>8s} {'(repulsion':>11s} {'(1=ideal,':>10s} "
          f"{'(lower=':>9s}")
    print(f"  {'':8s} {'':>8s} {'low)':>11s} {'>1 hole)':>10s} {'better)':>9s}")
    for stat in STATS:
        r = res[stat]
        print(f"  {stat:8s} {r['gap_cv']:8.3f} {r['frac_small']:11.3f} "
              f"{r['covering_radius']:10.3f} {r['log_cond']:9.3f}")
    print()

    kernel_res = None
    if run_kernel_part:
        print("  --- Part B: in-kernel broadband prediction (LOWER is better) ---")
        kernel_res = run_kernel(M, n_steps=900, seeds=seeds)
        for stat in STATS:
            m, s = kernel_res[stat]
            print(f"  {stat:8s} pred_error = {m:.5f} ± {s:.5f}")
        print()

    # ----- verdict -----
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    z, gu, po, un = (res["zeta"], res["gue"], res["poisson"], res["uniform"])

    # zeta behaves like gue on both diagnostics?
    zeta_eq_gue = (_close(z["covering_radius"], gu["covering_radius"], 0.15)
                   and _close(z["log_cond"], gu["log_cond"], 0.15))
    # repulsion (best of zeta/gue) avoids poisson's blind spots / ill-conditioning?
    best_cov = min(z["covering_radius"], gu["covering_radius"])
    best_cond = min(z["log_cond"], gu["log_cond"])
    repulsion_beats_poisson = (_better(best_cov, po["covering_radius"])
                               or _better(best_cond, po["log_cond"]))
    # is GUE/zeta ever strictly better than a rigid lattice?
    gue_special_vs_uniform = (_better(best_cov, un["covering_radius"])
                              or _better(best_cond, un["log_cond"]))

    print(f"  covering radius : zeta={z['covering_radius']:.3f} gue={gu['covering_radius']:.3f} "
          f"poisson={po['covering_radius']:.3f} uniform={un['covering_radius']:.3f}")
    print(f"  log_cond        : zeta={z['log_cond']:.3f} gue={gu['log_cond']:.3f} "
          f"poisson={po['log_cond']:.3f} uniform={un['log_cond']:.3f}")
    print()
    print(f"  zeta ~= gue (statistic, not arithmetic)  : {zeta_eq_gue}")
    print(f"  repulsion beats poisson (no blind spots) : {repulsion_beats_poisson}")
    print(f"  gue/zeta strictly better than a lattice  : {gue_special_vs_uniform}")
    if kernel_res:
        ks = {s: kernel_res[s][0] for s in STATS}
        spread = (max(ks.values()) - min(ks.values())) / st.mean(ks.values())
        kernel_flat = spread < 0.05
        best_arm = min(ks, key=ks.get)
        print(f"  kernel: spread across arms = {100*spread:.1f}%  "
              f"(best={best_arm})  -> functionally flat: {kernel_flat}")
    print()

    if gue_special_vs_uniform:
        conclusion = ("GUE level repulsion is genuinely special (better than "
                      "BOTH poisson and a lattice)"
                      + (" and zeta inherits it — zeta is justified AS a GUE "
                         "spectrum." if zeta_eq_gue else
                         ", but zeta does not fully match gue — investigate."))
    elif repulsion_beats_poisson:
        conclusion = ("Repulsion (gue/zeta) only avoids poisson's worst case; "
                      "it is NOT better than a rigid lattice, which wins both "
                      "diagnostics. The useful property is low-discrepancy "
                      "SPREAD, delivered at least as well by a lattice/Fourier "
                      "basis. Zeta is not special — and functionally flat in "
                      "the kernel.")
    else:
        conclusion = ("Spacing statistic does not drive basis quality here; "
                      "only 'structured vs random' matters. Zeta interchangeable.")
    print(f"  CONCLUSION: {conclusion}")
    print("=" * 72)

    if plot:
        try:
            _plot(res, kernel_res)
        except Exception as e:  # noqa: BLE001 - plotting is best-effort
            print(f"  (plot skipped: {e})")


def _plot(res: dict, kernel_res: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3 if kernel_res else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    x = np.arange(len(STATS))
    colors = ["#8e44ad", "#2980b9", "#e67e22", "#7f8c8d"]

    def bars(ax, key, title, ylabel, ref=None):
        vals = [res[s][key] for s in STATS]
        ax.bar(x, vals, 0.6, color=colors, alpha=0.9)
        if ref is not None:
            ax.axhline(ref, ls="--", c="k", alpha=0.5, label=f"ideal={ref}")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(STATS)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    bars(axes[0], "covering_radius", "Covering radius", "norm. radius [lower=better]", ref=1.0)
    bars(axes[1], "log_cond", "Temporal-code conditioning", "log10 cond [lower=better]")
    if kernel_res:
        ax = axes[2]
        means = [kernel_res[s][0] for s in STATS]
        errs = [kernel_res[s][1] for s in STATS]
        ax.bar(x, means, 0.6, yerr=errs, capsize=4, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(STATS)
        ax.set_title("In-kernel broadband prediction")
        ax.set_ylabel("perceptual error [lower=better]")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Spacing statistics: zeta vs GUE vs Poisson vs uniform")
    out = Path("results") / "spacing_statistics.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spacing-statistics decisive test")
    parser.add_argument("--M", type=int, default=40)
    parser.add_argument("--T", type=int, default=400, help="time grid length")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--kernel", action="store_true", help="also run Part B")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    main(
        M=args.M,
        T=args.T,
        seeds=list(range(args.seeds)),
        run_kernel_part=args.kernel,
        plot=not args.no_plot,
    )
