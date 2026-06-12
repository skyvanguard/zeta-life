"""
Psi recalibration for Yvyra's regime
====================================

Phase 1 found Psi sits at 0 on Yvyra's real ticks: her free energy (median ~2.4)
is far higher than the bench's coherent input (~0.02), so with the default
``psi_fe_scale=5`` the phi-base collapses to ~0.08 (subcritical) and Psi never
crosses threshold. This experiment re-simulates the bridge kernel over Yvyra's
*real* 4-axis scores (from the deployed paired log) across a sweep of
``psi_fe_scale`` and picks the value that makes Psi discriminate again.

Discrimination criterion: Psi should VARY (std > 0) and track inverse free energy
(low FE / coherent tick -> high Psi), i.e. a strongly negative corr(Psi, FE).

IMPORTANT -- the simulation is FAITHFUL to the deployment: it drives the
YvyraBridge with a load/save per tick (a fresh process each tick, like
yvyra_kernel.py). This matters: an in-memory run (one process) keeps the
recurrent state that the checkpoint does NOT persist (_prec_ref, world-model
latent), so it reports a very different Psi than the real per-tick deployment.

Usage:
    PYTHONPATH=src python experiments/kernel/exp_psi_recalibration.py \
        --log ~/.hermes/zeta/state/zeta_ticks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.bridge import YvyraBridge  # noqa: E402

AXES = ("novedad", "introspeccion", "conexion", "resolucion")
C = [0.30, 0.40, 0.10, 0.20]
SCALES = [5.0, 2.0, 1.5, 1.0, 0.5, 0.3]


def load_real_scores(logpath: str) -> list[list[float]]:
    p = os.path.expanduser(logpath)
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        if "scores" in t:
            out.append([float(t["scores"][a]) for a in AXES])
    return out


def simulate(scale: float, scores: list[list[float]], warmup: int) -> tuple[np.ndarray, np.ndarray]:
    """Faithful deployment sim: YvyraBridge with load/save per tick. (psi, fe)."""
    d = tempfile.mkdtemp()
    torch.manual_seed(0)
    # warmup like the deployment: smooth random walk around the character C
    rng = random.Random(0)
    mood = list(C)
    br = YvyraBridge(mode="silent", save_dir=d, psi_fe_scale=scale)
    for _ in range(warmup):
        mood = [min(1.0, max(0.0, m + 0.1 * (rng.random() - 0.5))) for m in mood]
        br.step(mood)
    br.save("y")
    # feed the REAL Yvyra scores, reloading per tick (fresh process each tick)
    psis, fes = [], []
    for sc in scores:
        b = YvyraBridge(mode="feedback", save_dir=d, psi_fe_scale=scale)
        b.load("y")
        out = b.step(sc)
        b.save("y")
        psis.append(out["psi"])
        fes.append(out["free_energy"])
    return np.array(psis), np.array(fes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="~/.hermes/zeta/state/zeta_ticks.jsonl")
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    scores = load_real_scores(args.log)
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 72)
    out("PSI RECALIBRATION over Yvyra's real scores")
    out("=" * 72)
    out(f"real ticks: {len(scores)}  warmup: {args.warmup}")
    out("")
    out(f"{'scale':>6} {'psi_mean':>9} {'psi_std':>8} {'psi_range':>16} {'corr(psi,-fe)':>14}  verdict")

    best = None
    for scale in SCALES:
        psis, fes = simulate(scale, scores, args.warmup)
        std = float(psis.std())
        corr = float(np.corrcoef(psis, -fes)[0, 1]) if std > 1e-9 and fes.std() > 1e-9 else 0.0
        rng = f"[{psis.min():.3f},{psis.max():.3f}]"
        # a good scale: Psi varies enough AND tracks inverse free energy
        discriminates = std > 0.05 and corr > 0.3
        verdict = "discriminates" if discriminates else ("flat" if std <= 0.05 else "weak")
        out(f"{scale:>6.1f} {psis.mean():>9.3f} {std:>8.3f} {rng:>16} {corr:>+14.3f}  {verdict}")
        # score by std * max(corr,0): want both spread and correct direction
        sc_score = std * max(corr, 0.0)
        if best is None or sc_score > best[1]:
            best = (scale, sc_score, std, corr)

    out("")
    if best and best[2] > 0.05 and best[3] > 0.3:
        out(f"RECOMMENDED psi_fe_scale = {best[0]}  (std={best[2]:.3f}, corr={best[3]:+.3f})")
    else:
        out("No scale gives clean discrimination on these scores -- Psi heuristic may")
        out("need more than a scale change for Yvyra's regime (the epistemic-depth")
        out("signal remains the robust integration measure).")

    RESULTS = Path(__file__).resolve().parents[2] / "results"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "psi_recalibration_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
