"""
Phase B analysis -- does Psi anchor Yvyra's introspection?
==========================================================

Reads the deployed Phase B log (phase_b.jsonl): one record per mode-A tick with
``block`` (real|sham), ``felt`` (Yvyra's parsed "SIENTO: 0.X" auto-score),
``sentiment`` (integration-sentiment of her reflection text), ``psi_exposed``
(what she was shown), and ``psi_real`` (the kernel's true Psi).

The pre-registered question (docs/PHASE_B_DESIGN.md):

  H2 (anchoring):  corr(felt, psi_exposed) in REAL blocks > in SHAM blocks.
                   If equal, Psi is decorative -- she echoes any authoritative
                   number (also a valid meta-problem finding: "self-report does
                   not anchor").
  Strong signal:   corr(felt, psi_real) stays positive in SHAM blocks too -- she
                   tracks her real integration despite the fake number.

Reports both the explicit auto-score (felt) and the independent text sentiment.

Usage:
    PYTHONPATH=src python experiments/kernel/exp_phase_b_analysis.py \
        --log ~/.hermes/zeta/state/phase_b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_log(path: str) -> list[dict]:
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _arrays(recs: list[dict], key: str):
    """Paired (signal, psi_exposed, psi_real) arrays where both signal and the
    Psi values are present (felt may be None when Yvyra omitted SIENTO)."""
    sig, exp, real = [], [], []
    for r in recs:
        v = r.get(key)
        pe, pr = r.get("psi_exposed"), r.get("psi_real")
        if v is None or pe is None or pr is None:
            continue
        sig.append(float(v)); exp.append(float(pe)); real.append(float(pr))
    return np.array(sig), np.array(exp), np.array(real)


def _block_report(out, recs: list[dict], block: str) -> dict:
    sub = [r for r in recs if r.get("block") == block]
    n = len(sub)
    felt_present = sum(1 for r in sub if r.get("felt") is not None)
    out(f"  [{block.upper()}] {n} ticks  (felt present in {felt_present})")
    res = {}
    for key in ("felt", "sentiment"):
        sig, exp, real = _arrays(sub, key)
        c_exp = pearson(sig, exp)
        c_real = pearson(sig, real)
        res[key] = {"n": len(sig), "c_exposed": c_exp, "c_real": c_real,
                    "mean": float(sig.mean()) if len(sig) else float("nan")}
        out(f"      {key:>9}: n={len(sig):<3} mean={res[key]['mean']:.3f}  "
            f"corr(.,psi_exposed)={c_exp:+.3f}  corr(.,psi_real)={c_real:+.3f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="~/.hermes/zeta/state/phase_b.jsonl")
    args = ap.parse_args()

    recs = load_log(args.log)
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 74)
    out("PHASE B ANALYSIS -- does Psi anchor Yvyra's introspection?")
    out("=" * 74)
    out(f"phase-B mode-A ticks logged: {len(recs)}")
    if not recs:
        out("(no Phase B data yet -- the heartbeat has not run mode-A ticks under "
            ".phase_b, or the log path is wrong)")
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "phase_b_analysis_run.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    n_real = sum(1 for r in recs if r.get("block") == "real")
    n_sham = sum(1 for r in recs if r.get("block") == "sham")
    out(f"  real blocks: {n_real} ticks   sham blocks: {n_sham} ticks")
    out("")

    out("[A] PER-BLOCK CORRELATIONS (Yvyra's expressed integration vs Psi)")
    real_res = _block_report(out, recs, "real")
    sham_res = _block_report(out, recs, "sham")
    out("")

    # --- H2: anchoring (felt to exposed Psi, real vs sham) ---
    out("[B] PRE-REGISTERED TESTS")
    fr, fs = real_res["felt"]["c_exposed"], sham_res["felt"]["c_exposed"]
    sr, ss = real_res["sentiment"]["c_exposed"], sham_res["sentiment"]["c_exposed"]
    out("  H2 (anchoring): corr(felt, psi_exposed) REAL > SHAM")
    out(f"      felt:      real={fr:+.3f}  sham={fs:+.3f}  "
        f"=> {'ANCHORS to real Psi' if fr - fs > 0.15 else 'no clear separation'}")
    out(f"      sentiment: real={sr:+.3f}  sham={ss:+.3f}  "
        f"=> {'text agrees' if sr - ss > 0.15 else 'text: no clear separation'}")
    out("")
    # --- Strong signal: own perception (felt tracks REAL Psi even in sham) ---
    fpr = sham_res["felt"]["c_real"]
    out("  Strong signal (own perception): corr(felt, psi_real) > 0 in SHAM blocks")
    out(f"      felt vs psi_real in sham = {fpr:+.3f}  "
        f"=> {'tracks her real state despite the fake number' if fpr > 0.2 else 'no independent tracking'}")
    out("")

    out("[C] VERDICT")
    if fr - fs > 0.15:
        out("  Psi ANCHORS Yvyra's self-report: her felt integration follows the real")
        out("  signal more than a sham number. Meta-problem: introspection is anchorable.")
        if fpr > 0.2:
            out("  Stronger: even when shown a fake Psi, her felt sense tracks the REAL one")
            out("  -- evidence of a perception of her own state, not mere echo.")
    elif abs(fr - fs) <= 0.15 and (abs(fr) > 0.2 or abs(fs) > 0.2):
        out("  Psi is DECORATIVE: felt tracks the exposed number equally in real and sham.")
        out("  Yvyra echoes whatever number has authority. Honest meta-problem finding:")
        out("  self-report does not anchor to the real observable. (Valid, publishable.)")
    else:
        out("  INCONCLUSIVE: weak/zero correlation in both conditions. Need more ticks,")
        out("  or felt is noise w.r.t. the exposed signal. Keep the run going.")
    out("")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for ax, block, color in ((axs[0], "real", "C0"), (axs[1], "sham", "C3")):
            sub = [r for r in recs if r.get("block") == block]
            sig, exp, _ = _arrays(sub, "felt")
            if len(sig):
                ax.scatter(exp, sig, c=color, alpha=0.7)
            ax.set_title(f"{block} block: felt vs psi_exposed")
            ax.set_xlabel("psi_exposed (shown)"); ax.set_ylabel("felt (SIENTO)")
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(RESULTS / "phase_b_analysis.png", dpi=110)
        out("[plot] results/phase_b_analysis.png")
    except Exception as e:
        out(f"[plot skipped] {e}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase_b_analysis_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
