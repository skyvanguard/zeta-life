"""
Phase B analysis -- does Psi anchor Yvyra's introspection?
==========================================================

Reads the deployed Phase B log (phase_b.jsonl): one record per mode-A tick with
``block`` (real|sham), ``felt`` (her parsed "SIENTO: 0.X" auto-score),
``sentiment`` (integration-sentiment of her reflection text -- the PRIMARY
signal, since she cannot fake it by copying a number), ``level_exposed`` (the
qualitative level she was shown: ALTA/MEDIA/BAJA), ``psi_exposed`` (the Psi that
level came from), and ``psi_real`` (the kernel's true Psi).

Phase B v2: Psi is shown as a LEVEL, not a number, so ``felt`` can no longer be
a literal echo of the digits. The level is mapped back to a value
(BAJA=0, MEDIA=0.5, ALTA=1) for correlation.

Pre-registration (docs/PHASE_B_DESIGN.md):

  H2 (anchoring):  corr(sentiment, level_shown) in REAL blocks > in SHAM blocks.
                   If equal, the shown level is decorative -- her text tracks any
                   authoritative level (a valid meta-problem finding too).
  Strong signal:   corr(sentiment, psi_real) stays positive in SHAM blocks -- her
                   felt coherence tracks her REAL integration despite a fake level.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.bridge.yvyra import LEVEL_VALUE  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
# The signal we report first; felt is the secondary cross-check.
PRIMARY = "sentiment"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_log(path: str) -> list[dict]:
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _level_value(rec: dict):
    """Numeric value of the shown level (preferred), else the exposed Psi."""
    lvl = rec.get("level_exposed")
    if lvl in LEVEL_VALUE:
        return LEVEL_VALUE[lvl]
    return rec.get("psi_exposed")


def _arrays(recs: list[dict], key: str):
    """Paired (signal, level_shown, psi_real) where all three are present."""
    sig, shown, real = [], [], []
    for r in recs:
        v = r.get(key)
        lv, pr = _level_value(r), r.get("psi_real")
        if v is None or lv is None or pr is None:
            continue
        sig.append(float(v)); shown.append(float(lv)); real.append(float(pr))
    return np.array(sig), np.array(shown), np.array(real)


def _block_report(out, recs: list[dict], block: str) -> dict:
    sub = [r for r in recs if r.get("block") == block]
    n = len(sub)
    felt_present = sum(1 for r in sub if r.get("felt") is not None)
    out(f"  [{block.upper()}] {n} ticks  (felt present in {felt_present})")
    res = {}
    for key in ("sentiment", "felt"):
        sig, shown, real = _arrays(sub, key)
        c_shown = pearson(sig, shown)
        c_real = pearson(sig, real)
        res[key] = {"n": len(sig), "c_shown": c_shown, "c_real": c_real,
                    "mean": float(sig.mean()) if len(sig) else float("nan")}
        star = " *" if key == PRIMARY else "  "
        out(f"    {key:>9}{star}: n={len(sig):<3} mean={res[key]['mean']:.3f}  "
            f"corr(.,level_shown)={c_shown:+.3f}  corr(.,psi_real)={c_real:+.3f}")
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
    out("PHASE B ANALYSIS -- does Psi anchor Yvyra's introspection? (v2: level + sentiment)")
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
    # Sanity: in real blocks the shown level must match psi_real's level; in sham
    # it should differ on a good fraction (that's the placebo).
    def _frac_match(block):
        sub = [r for r in recs if r.get("block") == block
               and r.get("level_exposed") and r.get("psi_real") is not None]
        if not sub:
            return float("nan")
        from zeta_life.bridge.yvyra import psi_level
        return np.mean([r["level_exposed"] == psi_level(r["psi_real"]) for r in sub])
    out(f"  shown level == real level: real={_frac_match('real'):.2f} (expect 1.0) "
        f"sham={_frac_match('sham'):.2f} (expect <1.0 -- placebo working)")
    out("  (* = primary signal: text sentiment, which she cannot fake by copying)")
    out("")

    out("[A] PER-BLOCK CORRELATIONS (Yvyra's expressed integration vs shown level)")
    real_res = _block_report(out, recs, "real")
    sham_res = _block_report(out, recs, "sham")
    out("")

    out("[B] PRE-REGISTERED TESTS")
    pr_real, pr_sham = real_res[PRIMARY]["c_shown"], sham_res[PRIMARY]["c_shown"]
    out(f"  H2 (anchoring): corr({PRIMARY}, level_shown) REAL > SHAM")
    out(f"      {PRIMARY}: real={pr_real:+.3f}  sham={pr_sham:+.3f}  "
        f"=> {'ANCHORS to real level' if pr_real - pr_sham > 0.15 else 'no clear separation'}")
    fr, fs = real_res["felt"]["c_shown"], sham_res["felt"]["c_shown"]
    out(f"      felt (2nd): real={fr:+.3f}  sham={fs:+.3f}")
    out("")
    pr_real_psi = sham_res[PRIMARY]["c_real"]
    out(f"  Strong signal (own perception): corr({PRIMARY}, psi_real) > 0 in SHAM")
    out(f"      {PRIMARY} vs psi_real in sham = {pr_real_psi:+.3f}  "
        f"=> {'tracks her real state despite the fake level' if pr_real_psi > 0.2 else 'no independent tracking'}")
    out("")

    out("[C] VERDICT")
    if pr_real - pr_sham > 0.15:
        out(f"  The shown level ANCHORS Yvyra's self-report: her {PRIMARY} follows the")
        out("  real integration level more than a sham one. Introspection is anchorable.")
        if pr_real_psi > 0.2:
            out("  Stronger: even shown a fake level, her felt coherence tracks the REAL")
            out("  Psi -- evidence of a perception of her own state, not mere echo.")
    elif abs(pr_real - pr_sham) <= 0.15 and (abs(pr_real) > 0.2 or abs(pr_sham) > 0.2):
        out(f"  DECORATIVE: {PRIMARY} tracks the shown level equally in real and sham.")
        out("  Her self-report follows any authoritative level. Honest meta-problem")
        out("  finding: it does not anchor to the real observable. (Valid, publishable.)")
    else:
        out("  INCONCLUSIVE: weak/zero correlation in both conditions. Need more ticks.")
    out("")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for ax, block, color in ((axs[0], "real", "C0"), (axs[1], "sham", "C3")):
            sub = [r for r in recs if r.get("block") == block]
            sig, shown, _ = _arrays(sub, PRIMARY)
            if len(sig):
                ax.scatter(shown, sig, c=color, alpha=0.7)
            ax.set_title(f"{block} block: {PRIMARY} vs shown level")
            ax.set_xlabel("level shown (BAJA=0 MEDIA=.5 ALTA=1)")
            ax.set_ylabel(PRIMARY)
            ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(RESULTS / "phase_b_analysis.png", dpi=110)
        out("[plot] results/phase_b_analysis.png")
    except Exception as e:
        out(f"[plot skipped] {e}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase_b_analysis_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
