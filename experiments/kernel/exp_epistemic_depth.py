"""
Epistemic depth -- the second-order error over precision as a regime signal
===========================================================================

Phase 2 of the science plan (docs/SCIENCE_PLAN.md). Tests the precision
hyper-model (the "beautiful loop" third condition) inside the kernel.

Two claims, tested head-on:

  A. SIGNATURE. The second-order error over precision should be LOW when the
     world is stationary (the system has learned to predict its own confidence)
     and SPIKE at regime changes (it is surprised by its own precision). We feed
     coherent -> fragmented -> coherent and look for the spikes at the seams.

  B. INDEPENDENCE FROM FREE ENERGY. Phase 1 found Psi is partly tautological
     with free energy (Psi is computed from it). For the hyper-model's signal to
     add anything, the second-order error must NOT be a direct function of free
     energy. We compare |corr(second_order, free_energy)| against
     |corr(Psi, free_energy)|: the hyper-model signal should be markedly less
     coupled to free energy than Psi is.

Output: results/epistemic_depth_run.txt and results/epistemic_depth.png
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


def coherent(t: int, rng: torch.Generator) -> torch.Tensor:
    base = 1.0 + torch.sin(0.05 * t + PHASES)
    x = base + 0.1 * torch.rand(4, generator=rng)
    return x / x.sum()


def fragmented(rng: torch.Generator) -> torch.Tensor:
    x = torch.rand(4, generator=rng).abs() + 1e-3
    return x / x.sum()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=200, help="ticks per regime block")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    B = args.block

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    kernel = ConsciousKernel(obs_dim=4, latent_dim=32, precision_hypermodel=True)

    psi, fe, so = [], [], []
    logpath = RESULTS / "_epistemic_depth.jsonl"
    # regimes: coherent [0,B) -> fragmented [B,2B) -> coherent [2B,3B)
    with TickLogger(logpath) as log:
        for t in range(1, 3 * B + 1):
            if t <= B or t > 2 * B:
                stim = coherent(t, rng)
            else:
                stim = fragmented(rng)
            r = kernel.step(stim)
            psi.append(r.psi); fe.append(r.free_energy); so.append(r.second_order_error)
            log.log({"psi": float(r.psi), "free_energy": float(r.free_energy),
                     "second_order_error": float(r.second_order_error),
                     "gw_winner": None, "mode": "bench"})

    psi = np.array(psi); fe = np.array(fe); so = np.array(so)
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 72)
    out("PHASE 2 -- Epistemic depth: second-order error over precision")
    out("=" * 72)
    out(f"block={B} ticks/regime  seed={args.seed}")
    out(f"regimes: coherent [0,{B}) -> fragmented [{B},{2 * B}) -> coherent [{2 * B},{3 * B})")
    out("")

    # --- A. Signature: per-regime means + transition spikes ---
    settle = B // 2  # ignore the first half of each block (settling)
    coh1 = so[settle:B]
    frag = so[B + settle:2 * B]
    coh2 = so[2 * B + settle:3 * B]
    # spike windows around the seams
    seam1 = so[B:B + 10].max()
    seam2 = so[2 * B:2 * B + 10].max()
    out("[A] SIGNATURE (second-order error)")
    out(f"    coherent-1 settled mean = {coh1.mean():.4f}")
    out(f"    fragmented settled mean = {frag.mean():.4f}")
    out(f"    coherent-2 settled mean = {coh2.mean():.4f}")
    out(f"    spike at seam->frag (max first 10) = {seam1:.4f}")
    out(f"    spike at seam->coh  (max first 10) = {seam2:.4f}")
    base_settled = max(coh1.mean(), coh2.mean())
    spike_ok = seam1 > base_settled * 1.5 or frag.mean() > coh1.mean() * 1.3
    out(f"    => regime change visible in the signal? {'yes' if spike_ok else 'NO'}")
    out("")

    # --- B. Independence from free energy ---
    # use fluctuations to be consistent with Phase 1
    dfe = np.diff(fe)
    dso = np.diff(so)
    dpsi = np.diff(psi)
    c_so_fe = abs(pearson(dso, dfe))
    c_psi_fe = abs(pearson(dpsi, dfe))
    out("[B] INDEPENDENCE FROM FREE ENERGY (|corr| of fluctuations)")
    out(f"    |corr(d second_order, d FE)| = {c_so_fe:.4f}")
    out(f"    |corr(d Psi,           d FE)| = {c_psi_fe:.4f}   (Phase 1: ~0.74, tautological)")
    independent = c_so_fe < c_psi_fe
    out(f"    => hyper-model signal less coupled to FE than Psi? {'yes' if independent else 'NO'}")
    out("")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for a in ax:
            a.axvline(B, color="k", ls="--", alpha=0.4)
            a.axvline(2 * B, color="k", ls="--", alpha=0.4)
        ax[0].plot(psi, color="C0"); ax[0].set_ylabel("Psi")
        ax[0].set_title("coherent | fragmented | coherent")
        ax[1].plot(fe, color="C1"); ax[1].set_ylabel("free energy")
        ax[2].plot(so, color="C3"); ax[2].set_ylabel("2nd-order error\n(epistemic depth)")
        ax[2].set_xlabel("tick")
        fig.tight_layout()
        fig.savefig(RESULTS / "epistemic_depth.png", dpi=110)
        out("[plot] results/epistemic_depth.png")
    except Exception as e:
        out(f"[plot skipped] {e}")

    logpath.unlink(missing_ok=True)
    out("")
    out("SUMMARY")
    out(f"  signature visible      : {'yes' if spike_ok else 'no'}")
    out(f"  |corr(2nd-order, FE)|  : {c_so_fe:.4f}  vs  |corr(Psi, FE)| {c_psi_fe:.4f}")
    out(f"  more independent of FE : {'yes' if independent else 'no'}")

    (RESULTS / "epistemic_depth_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
