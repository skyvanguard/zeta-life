"""
Phase A baseline analysis -- Yvyra's real silent run
====================================================

Analyses the deployed Phase A data (uncontaminated silent baseline): the paired
per-tick log (zeta_ticks.jsonl) plus Yvyra's real journals. Reports the signal
distributions, the key correlations (does the epistemic-depth signal stay
INDEPENDENT of free energy on real data, as it did on the bench? does Psi track
coherence?), the 4-axis dynamics, and a blind re-score of the journals.

Honest note on the reporter: in Yvyra the 4 axes are computed by bash from
objective signals (research mode, journal length, net use, Fran mentions), NOT
self-reported by the LLM. So the blind re-scorer here measures whether the
journal TEXT is coherent with its mode (introspection journals read introspective,
research journals read novel), not whether the LLM scored honestly.

Usage:
    PYTHONPATH=src python experiments/kernel/exp_phase_a_analysis.py \
        --log ~/.hermes/zeta/state/zeta_ticks.jsonl \
        --journals ~/.hermes/heartbeat
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.bridge.rescorer import rescore  # noqa: E402

AXES = ("novedad", "introspeccion", "conexion", "resolucion")
RESULTS = Path(__file__).resolve().parents[2] / "results"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_log(path: str) -> list[dict]:
    p = os.path.expanduser(path)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_journal_entries(md_path: str) -> list[str]:
    """Split a heartbeat journal .md into its per-tick entries (## [HH:MM])."""
    if not os.path.exists(md_path):
        return []
    text = open(md_path, encoding="utf-8").read()
    # entries are separated by '## [HH:MM]' headers
    parts = re.split(r"##\s*\[\d{1,2}:\d{2}\]", text)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="~/.hermes/zeta/state/zeta_ticks.jsonl")
    ap.add_argument("--journals", default="~/.hermes/heartbeat")
    ap.add_argument("--day", default="2026-06-12")
    args = ap.parse_args()

    ticks = load_log(args.log)
    n = len(ticks)
    psi = np.array([t["psi"] for t in ticks])
    fe = np.array([t["free_energy"] for t in ticks])
    so = np.array([t["second_order_error"] for t in ticks])
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 74)
    out("PHASE A BASELINE ANALYSIS -- Yvyra's real silent run")
    out("=" * 74)
    out(f"paired ticks: {n}")
    out("")

    # --- A. Signal distributions ---
    import statistics as st
    out("[A] SIGNAL DISTRIBUTIONS")
    out(f"    Psi:            mean={psi.mean():.3f} std={psi.std():.3f} "
        f">0 in {100*np.mean(psi>0.01):.0f}% of ticks")
    out(f"    free_energy:    median={st.median(fe):.2f} "
        f"first20_avg={fe[:20].mean():.2f} last20_avg={fe[-20:].mean():.2f} (adaptation)")
    out(f"    epistemic depth:mean={so.mean():.3f} std={so.std():.3f} "
        f"range=[{so.min():.2f},{so.max():.2f}]")
    exposed = [t.get("psi_exposed") for t in ticks]
    out(f"    silent integrity: Psi exposed in {sum(1 for e in exposed if e is not None)} ticks "
        f"(expect 0)")
    out("")

    # --- B. Key correlations (fluctuations, Albantakis-style) ---
    dpsi, dfe, dso = np.diff(psi), np.diff(fe), np.diff(so)
    c_psi_fe = abs(pearson(dpsi, dfe))
    c_so_fe = abs(pearson(dso, dfe))
    c_so_psi = abs(pearson(dso, dpsi))
    out("[B] CORRELATIONS (|corr| of fluctuations)")
    out(f"    |corr(d Psi,        d FE)| = {c_psi_fe:.3f}  (high: Psi derives from FE)")
    out(f"    |corr(d epistemic,  d FE)| = {c_so_fe:.3f}  (LOW expected: independent signal)")
    out(f"    |corr(d epistemic,  d Psi)|= {c_so_psi:.3f}")
    indep = c_so_fe < c_psi_fe
    out(f"    => epistemic depth more independent of FE than Psi is? "
        f"{'yes -- replicates the bench finding' if indep else 'NO'}")
    out("")

    # --- C. 4-axis dynamics ---
    out("[C] 4-AXIS DYNAMICS (Yvyra's experience, from bash objective signals)")
    for i, ax in enumerate(AXES):
        vals = np.array([t["scores"][ax] for t in ticks])
        out(f"    {ax:>14}: mean={vals.mean():.2f} std={vals.std():.2f} "
            f"range=[{vals.min():.2f},{vals.max():.2f}]")
    out("")

    # --- D. Journal quality + blind re-score ---
    jdir = os.path.expanduser(args.journals)
    intro_entries = load_journal_entries(os.path.join(jdir, "self-journal", f"{args.day}.md"))
    research_entries = load_journal_entries(os.path.join(jdir, "journal", f"{args.day}.md"))
    out("[D] JOURNALS (real text Yvyra wrote)")
    out(f"    introspection journal: {len(intro_entries)} entries, "
        f"avg {np.mean([len(e.split()) for e in intro_entries]):.0f} words" if intro_entries else
        "    introspection journal: (none)")
    out(f"    research journal:      {len(research_entries)} entries, "
        f"avg {np.mean([len(e.split()) for e in research_entries]):.0f} words" if research_entries else
        "    research journal: (none)")
    # blind re-score: do introspection journals read introspective, research read novel?
    if intro_entries:
        intro_scores = [rescore(e) for e in intro_entries]
        intro_axis = np.mean([s["introspeccion"] for s in intro_scores])
        intro_nov = np.mean([s["novedad"] for s in intro_scores])
        out(f"    blind re-score of INTROSPECTION journals: "
            f"introspeccion={intro_axis:.2f} vs novedad={intro_nov:.2f} "
            f"({'introspective text OK' if intro_axis > intro_nov else 'weak'})")
    if research_entries:
        res_scores = [rescore(e) for e in research_entries]
        res_nov = np.mean([s["novedad"] for s in res_scores])
        res_intro = np.mean([s["introspeccion"] for s in res_scores])
        out(f"    blind re-score of RESEARCH journals: "
            f"novedad={res_nov:.2f} vs introspeccion={res_intro:.2f}")
    out("")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(psi, color="C0"); ax[0].set_ylabel("Psi")
        ax[0].set_title(f"Phase A baseline -- {n} real ticks (silent)")
        ax[1].plot(fe, color="C1"); ax[1].set_ylabel("free energy")
        ax[2].plot(so, color="C3"); ax[2].set_ylabel("epistemic depth\n(2nd-order)")
        ax[2].set_xlabel("tick")
        fig.tight_layout()
        fig.savefig(RESULTS / "phase_a_analysis.png", dpi=110)
        out("[plot] results/phase_a_analysis.png")
    except Exception as e:
        out(f"[plot skipped] {e}")

    out("")
    out("SUMMARY")
    out(f"  Psi discriminates: {100*np.mean(psi>0.01):.0f}% of ticks > 0")
    out(f"  epistemic depth independent of FE: {'yes' if indep else 'no'} "
        f"(|corr| {c_so_fe:.2f} vs Psi's {c_psi_fe:.2f})")
    out(f"  journals: {len(intro_entries)} introspection + {len(research_entries)} research, all captured")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase_a_analysis_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
