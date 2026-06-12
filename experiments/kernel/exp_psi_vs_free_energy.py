"""
Psi vs free energy -- is the integration index a real instrument or a moving average?
=====================================================================================

Phase 1 of the science plan (docs/SCIENCE_PLAN.md). Before trusting Psi as an
instrument we validate it in the bench, adopting the method of Olesen, Waade,
Albantakis & Mathys (2023, "Phi fluctuates with surprisal", PLOS Comput Biol):

  1. MANIPULATION CHECK. Run the kernel on a COHERENT stimulus (smoothly
     rotating point on the simplex) vs a FRAGMENTED one (i.i.d. jumps). If Psi
     measures integration, Psi(coherent) > Psi(fragmented) and free energy is
     lower on coherent input.

  2. CO-FLUCTUATION (Albantakis method). On a long run, correlate the
     *fluctuations* (first differences) of Psi and free energy -- NOT their
     levels (which share trends and inflate correlation). Report effect size
     (slope beta), the lag profile, and the distribution of per-window
     correlation signs (their key finding: the mean hides that ~half the
     windows correlate the other way).

  3. TRAINING-PROGRESS CONTROL (the fitness confound). Albantakis show the
     long-timescale Phi<->surprisal link is largely because both track fitness.
     We split the run into early/late halves: if the Psi<->free_energy coupling
     is just a shared training trend, it vanishes within a stationary late
     window.

  4. AR(1) BASELINE (beat the trivial alternative). The lesson of zeta-life's
     own RNG confound: a new signal must beat the dumb baseline. We ask whether
     d(Psi) adds explanatory power for d(free_energy) beyond an AR(1) of free
     energy itself. If it does not, Psi is a moving average with prestige.

Outputs: results/psi_vs_free_energy_run.txt and results/psi_vs_free_energy.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.instrumentation import TickLogger  # noqa: E402
from zeta_life.kernel import ConsciousKernel  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
PHASES = torch.tensor([0.0, 1.6, 3.1, 4.7])


# ---------------------------------------------------------------------------
# Stimuli on the 4-simplex (the kernel expects non-negative, sum-1 vertices)
# ---------------------------------------------------------------------------

def coherent_stimulus(t: int, rng: torch.Generator) -> torch.Tensor:
    """Smoothly rotating point on the simplex -- learnable temporal structure."""
    base = 1.0 + torch.sin(0.05 * t + PHASES)
    noisy = base + 0.1 * torch.rand(4, generator=rng)
    return noisy / noisy.sum()


def fragmented_stimulus(rng: torch.Generator) -> torch.Tensor:
    """I.i.d. random point on the simplex -- no temporal structure to integrate."""
    x = torch.rand(4, generator=rng).abs() + 1e-3
    return x / x.sum()


# ---------------------------------------------------------------------------
# Run the kernel, logging (psi, free_energy) per tick
# ---------------------------------------------------------------------------

def run(kind: str, n_steps: int, seed: int, logpath: Path) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    kernel = ConsciousKernel(obs_dim=4, latent_dim=32)
    psis, fes = [], []
    with TickLogger(logpath) as log:
        for t in range(1, n_steps + 1):
            stim = coherent_stimulus(t, rng) if kind == "coherent" else fragmented_stimulus(rng)
            r = kernel.step(stim)
            psis.append(r.psi)
            fes.append(r.free_energy)
            log.log({
                "kind": kind,
                "psi": float(r.psi),
                "free_energy": float(r.free_energy),
                "second_order_error": None,   # reserved for Phase 2
                "gw_winner": None,
                "mode": "bench",
            })
    return np.array(psis), np.array(fes)


# ---------------------------------------------------------------------------
# Analysis helpers (numpy only; honest small-sample stats)
# ---------------------------------------------------------------------------

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ols_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, R^2) of y ~ a + b*x."""
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(coef[1]), r2


def lag_profile(dpsi: np.ndarray, dfe: np.ndarray, max_lag: int = 5) -> dict[int, float]:
    """Correlation of d(Psi)[t] with d(fe)[t+lag] for lag in [-max_lag, max_lag]."""
    out: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = dpsi[-lag:], dfe[:lag]
        elif lag > 0:
            a, b = dpsi[:-lag], dfe[lag:]
        else:
            a, b = dpsi, dfe
        out[lag] = pearson(a, b) if len(a) > 2 else 0.0
    return out


def sign_distribution(dpsi: np.ndarray, dfe: np.ndarray, window: int = 25) -> dict[str, int]:
    """Per-window correlation sign distribution (Albantakis' key honesty check)."""
    pos = neg = neu = 0
    for i in range(0, len(dpsi) - window, window):
        c = pearson(dpsi[i:i + window], dfe[i:i + window])
        if c > 0.1:
            pos += 1
        elif c < -0.1:
            neg += 1
        else:
            neu += 1
    return {"positive": pos, "negative": neg, "neutral": neu}


def ar1_residual_test(psi: np.ndarray, fe: np.ndarray) -> dict[str, float]:
    """Does d(Psi) explain d(fe) beyond an AR(1) of fe?

    Baseline: d(fe)[t] ~ d(fe)[t-1]  (AR(1) on the fluctuations).
    Augmented: add d(Psi)[t] as a regressor. Report the R^2 gain.
    """
    dfe = np.diff(fe)
    dpsi = np.diff(psi)
    # align: predict dfe[t] from dfe[t-1] and dpsi[t]
    y = dfe[1:]
    x_ar = dfe[:-1]
    x_psi = dpsi[1:]
    _, r2_ar = ols_slope_r2(x_ar, y)
    # augmented: two regressors
    A = np.vstack([np.ones_like(y), x_ar, x_psi]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_aug = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"r2_ar1": r2_ar, "r2_augmented": r2_aug, "r2_gain_from_psi": r2_aug - r2_ar}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 74)
    out("PHASE 1 -- Psi vs free energy: instrument validation (Albantakis method)")
    out("=" * 74)
    out(f"steps={args.steps}  seeds={args.seeds}")
    out("")

    # --- 1. Manipulation check: coherent vs fragmented ---
    coh_psi_all, frag_psi_all, coh_fe_all, frag_fe_all = [], [], [], []
    # keep one representative run for the co-fluctuation analysis
    coh_psi_rep = coh_fe_rep = None
    for s in range(args.seeds):
        cp, cf = run("coherent", args.steps, s, RESULTS / f"_phase1_coh_{s}.jsonl")
        fp, ff = run("fragmented", args.steps, 100 + s, RESULTS / f"_phase1_frag_{s}.jsonl")
        tail = slice(-args.steps // 3, None)  # steady-state tail
        coh_psi_all.append(cp[tail].mean()); frag_psi_all.append(fp[tail].mean())
        coh_fe_all.append(cf[tail].mean()); frag_fe_all.append(ff[tail].mean())
        if s == 0:
            coh_psi_rep, coh_fe_rep = cp, cf

    out("[1] MANIPULATION CHECK (steady-state tail, mean +/- sd over seeds)")
    out(f"    Psi  coherent  = {np.mean(coh_psi_all):.4f} +/- {np.std(coh_psi_all):.4f}")
    out(f"    Psi  fragmented= {np.mean(frag_psi_all):.4f} +/- {np.std(frag_psi_all):.4f}")
    out(f"    FE   coherent  = {np.mean(coh_fe_all):.4f} +/- {np.std(coh_fe_all):.4f}")
    out(f"    FE   fragmented= {np.mean(frag_fe_all):.4f} +/- {np.std(frag_fe_all):.4f}")
    psi_gap = np.mean(coh_psi_all) - np.mean(frag_psi_all)
    fe_gap = np.mean(frag_fe_all) - np.mean(coh_fe_all)
    out(f"    => Psi gap (coh-frag) = {psi_gap:+.4f}   (expect > 0)")
    out(f"    => FE  gap (frag-coh) = {fe_gap:+.4f}   (expect > 0: more surprise on noise)")
    verdict1 = "PASS" if psi_gap > 0.05 and fe_gap > 0 else "FAIL"
    out(f"    VERDICT: {verdict1}")
    out("")

    # --- 2. Co-fluctuation on the representative coherent run ---
    dpsi = np.diff(coh_psi_rep)
    dfe = np.diff(coh_fe_rep)
    beta, r2 = ols_slope_r2(dfe, dpsi)  # d(Psi) ~ d(fe)
    lag0 = pearson(dpsi, dfe)
    lags = lag_profile(dpsi, dfe)
    signs = sign_distribution(dpsi, dfe)
    out("[2] CO-FLUCTUATION  d(Psi) vs d(free_energy)  (fluctuations, not levels)")
    out(f"    lag-0 correlation       = {lag0:+.4f}")
    out(f"    slope beta (dPsi~dFE)    = {beta:+.5f}   R^2 = {r2:.4f}")
    out("    lag profile             = " + ", ".join(f"{k}:{v:+.2f}" for k, v in lags.items()))
    peak_lag = max(lags, key=lambda k: abs(lags[k]))
    out(f"    peak |corr| at lag      = {peak_lag} (expect ~0: simultaneous, per Albantakis)")
    out(f"    per-window sign dist     = {signs}  (mean hides the spread)")
    out("")

    # --- 3. Training-progress control: early vs late half ---
    half = len(dpsi) // 2
    c_early = pearson(dpsi[:half], dfe[:half])
    c_late = pearson(dpsi[half:], dfe[half:])
    out("[3] TRAINING-PROGRESS CONTROL (fitness confound)")
    out(f"    coupling early half     = {c_early:+.4f}")
    out(f"    coupling late half      = {c_late:+.4f}")
    out(f"    => persists in stationary late window? {'yes' if abs(c_late) > 0.1 else 'NO (was a trend)'}")
    out("")

    # --- 4. AR(1) baseline ---
    ar = ar1_residual_test(coh_psi_rep, coh_fe_rep)
    out("[4] AR(1) BASELINE -- does d(Psi) beat a moving average?")
    out(f"    R^2  AR(1) of d(FE)            = {ar['r2_ar1']:.4f}")
    out(f"    R^2  AR(1) + d(Psi)           = {ar['r2_augmented']:.4f}")
    out(f"    R^2  gain attributable to Psi = {ar['r2_gain_from_psi']:+.4f}")
    beats = ar['r2_gain_from_psi'] > 0.01
    out(f"    => Psi adds signal beyond AR(1)? {'yes' if beats else 'NO (redundant)'}")
    out("")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(coh_psi_rep, label="Psi", color="C0")
        ax0b = ax[0].twinx()
        ax0b.plot(coh_fe_rep, label="free energy", color="C1", alpha=0.7)
        ax[0].set_title("Coherent run: Psi (left) vs free energy (right)")
        ax[0].set_xlabel("tick"); ax[0].set_ylabel("Psi", color="C0")
        ax0b.set_ylabel("free energy", color="C1")
        ax[1].scatter(dfe, dpsi, s=8, alpha=0.5)
        ax[1].set_title(f"Fluctuations: d(Psi) vs d(FE)  (lag-0 r={lag0:+.2f})")
        ax[1].set_xlabel("d(free energy)"); ax[1].set_ylabel("d(Psi)")
        fig.tight_layout()
        fig.savefig(RESULTS / "psi_vs_free_energy.png", dpi=110)
        out("[plot] results/psi_vs_free_energy.png")
    except Exception as e:
        out(f"[plot skipped] {e}")

    # --- cleanup temp logs ---
    for f in RESULTS.glob("_phase1_*.jsonl"):
        f.unlink()

    out("")
    out("SUMMARY")
    out(f"  manipulation check : {verdict1}")
    out(f"  coupling lag-0     : {lag0:+.4f} (peak lag {peak_lag})")
    out(f"  late-half coupling : {c_late:+.4f}")
    out(f"  Psi beats AR(1)    : {'yes' if beats else 'no'} (R2 gain {ar['r2_gain_from_psi']:+.4f})")

    (RESULTS / "psi_vs_free_energy_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
