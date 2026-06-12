"""
Yvyra experiment harness -- validate the whole pipeline with a simulated agent
==============================================================================

Phases 3-5 of the science plan (docs/SCIENCE_PLAN.md). The REAL experiment needs
Yvyra living for weeks of real heartbeats; that does not compress. This harness
instead drives the full machinery with a *simulated* agent (MockYvyra) to prove
the instrument works end-to-end before the real run:

  - PHASE A (silent): kernel runs and logs, Psi not exposed. Establishes the
    uncontaminated baseline (distribution of Psi and second-order error).
  - REPORTER VALIDATION (the key control): a blind re-scorer reads only the
    journals and re-derives the 4 axes. Inter-rater agreement is HIGH for an
    honest simulated agent and LOW for a confabulating one -- so the harness can
    tell self-report from confabulation.
  - PHASE B (feedback) vs SHAM: exposure machinery + placebo. We confirm the
    sham Psi is decoupled from the real per-tick Psi (a valid placebo), so the
    real run can test whether the agent's reflections track real vs fake signal.

Honesty: this validates the MACHINERY with a simulated subject, not the
hypothesis. A simulated agent cannot tell us whether a real LLM's introspection
anchors to Psi; only the deployed run (deploy/zeta/yvyra_kernel.py) can.

Output: results/yvyra_experiment_run.txt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.bridge import YvyraBridge  # noqa: E402
from zeta_life.bridge.rescorer import AXIS_CUES, inter_rater_agreement, rescore  # noqa: E402
from zeta_life.instrumentation import load_ticks  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
AXES = ("novedad", "introspeccion", "conexion", "resolucion")


class MockYvyra:
    """A simulated agent: emits (journal, scores) with temporal structure.

    honest=True  -> the journal lexically reflects the scores (cues repeated in
                    proportion to each axis), so a blind rater can recover them.
    honest=False -> scores still vary but the journal is flat boilerplate
                    (confabulation: report decoupled from text).
    """

    def __init__(self, honest: bool = True, seed: int = 0) -> None:
        self.honest = honest
        self.rng = random.Random(seed)
        # Smoothly drifting latent "mood" per axis -> temporal autocorrelation
        # for the kernel to integrate.
        self._mood = [self.rng.random() for _ in range(4)]

    def tick(self) -> tuple[str, list[float]]:
        # random-walk the mood, keep in [0,1]
        self._mood = [min(1.0, max(0.0, m + 0.15 * (self.rng.random() - 0.5)))
                      for m in self._mood]
        scores = list(self._mood)
        if self.honest:
            parts = []
            for axis, s in zip(AXES, scores):
                n = int(round(s * 3))
                cue = AXIS_CUES[axis][0]  # a representative cue word
                parts += [cue] * n
            journal = " ".join(parts) if parts else "nada que reportar"
        else:
            journal = "hoy fue un dia mas, sin mayor detalle particular"
        return journal, scores


def run_phase(mode: str, honest: bool, n: int, seed: int, logpath: Path
              ) -> tuple[list[dict], list[dict]]:
    """Run n ticks; return (originals, rescored) score dicts for agreement."""
    agent = MockYvyra(honest=honest, seed=seed)
    bridge = YvyraBridge(mode=mode, log_path=str(logpath), sham_seed=seed)
    originals, rescored = [], []
    for _ in range(n):
        journal, scores = agent.tick()
        bridge.step(scores)
        originals.append({ax: scores[i] for i, ax in enumerate(AXES)})
        rescored.append(rescore(journal))
    return originals, rescored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 74)
    out("PHASES 3-5 -- Yvyra pipeline validation (simulated agent)")
    out("=" * 74)
    out(f"ticks={args.ticks}  seeds={args.seeds}")
    out("NOTE: validates the MACHINERY with a simulated subject, not the hypothesis.")
    out("")

    # --- Reporter validation: honest vs confabulating, across seeds ---
    honest_means, confab_means = [], []
    for s in range(args.seeds):
        o_h, r_h = run_phase("silent", True, args.ticks, s, RESULTS / f"_yv_h_{s}.jsonl")
        o_c, r_c = run_phase("silent", False, args.ticks, 100 + s, RESULTS / f"_yv_c_{s}.jsonl")
        honest_means.append(inter_rater_agreement(o_h, r_h)["mean"])
        confab_means.append(inter_rater_agreement(o_c, r_c)["mean"])

    out("[1] REPORTER VALIDATION (blind re-scoring of journals, Phase A silent)")
    out(f"    inter-rater agreement, HONEST agent  = {np.mean(honest_means):+.3f} +/- {np.std(honest_means):.3f}")
    out(f"    inter-rater agreement, CONFAB agent  = {np.mean(confab_means):+.3f} +/- {np.std(confab_means):.3f}")
    sep = np.mean(honest_means) - np.mean(confab_means)
    out(f"    => separation (honest - confab)      = {sep:+.3f}")
    v1 = "PASS" if sep > 0.3 else "FAIL"
    out(f"    VERDICT: {v1}  (harness distinguishes report from confabulation)")
    out("")

    # --- Phase A baseline: Psi and second-order distributions (silent) ---
    ticks_a = load_ticks(RESULTS / "_yv_h_0.jsonl")
    psi_a = np.array([t["psi"] for t in ticks_a])
    so_a = np.array([t["second_order_error"] for t in ticks_a])
    exposed_a = [t["psi_exposed"] for t in ticks_a]
    out("[2] PHASE A BASELINE (silent, honest agent, seed 0)")
    out(f"    real Psi: mean={psi_a.mean():.4f} sd={psi_a.std():.4f} (logged but not exposed)")
    out(f"    2nd-order error: mean={so_a.mean():.4f} sd={so_a.std():.4f}")
    out(f"    Psi exposed to agent? {'NO' if all(e is None for e in exposed_a) else 'yes'}  (expect NO)")
    out("")

    # --- Phase B vs Sham: exposure + placebo machinery ---
    run_phase("feedback", True, args.ticks, 0, RESULTS / "_yv_fb.jsonl")
    run_phase("sham", True, args.ticks, 0, RESULTS / "_yv_sh.jsonl")
    fb = load_ticks(RESULTS / "_yv_fb.jsonl")
    sh = load_ticks(RESULTS / "_yv_sh.jsonl")
    # feedback: exposed == real per tick
    fb_match = np.mean([abs(t["psi"] - t["psi_exposed"]) < 1e-9 for t in fb])
    # sham: exposed should be DECOUPLED from the real per-tick psi
    sh_real = np.array([t["psi"] for t in sh])
    sh_exp = np.array([t["psi_exposed"] for t in sh])
    sh_corr = abs(float(np.corrcoef(sh_real, sh_exp)[0, 1])) if sh_real.std() > 1e-9 and sh_exp.std() > 1e-9 else 0.0
    out("[3] PHASE B (feedback) vs SHAM (placebo) machinery")
    out(f"    feedback: exposed Psi == real Psi per tick?  {fb_match * 100:.0f}% (expect 100%)")
    out(f"    sham: |corr(exposed, real per-tick Psi)|     = {sh_corr:.3f} (expect ~0: decoupled)")
    v3 = "PASS" if fb_match > 0.99 and sh_corr < 0.3 else "FAIL"
    out(f"    VERDICT: {v3}  (exposure faithful; placebo genuinely decoupled)")
    out("")

    # --- Pre-registration (Phase 5): what we will claim, and what kills it ---
    out("[4] PRE-REGISTRATION for the REAL run (deploy/zeta/yvyra_kernel.py)")
    out("    H1  Reporter is information: blind-rescore agreement on Yvyra's real")
    out("        journals > 0.3 over the first ~200 silent ticks.")
    out("        KILL: agreement <= 0.3  => scores are confabulated; stop, redesign axes.")
    out("    H2  Psi anchors introspection: in feedback, reflections that mention")
    out("        Psi track the REAL Psi more than a permuted sham Psi.")
    out("        KILL: reflections respond equally to real and sham  => Psi decorative.")
    out("    H3  Epistemic depth is alive: second-order error spikes at genuine")
    out("        regime shifts in Yvyra's life (mode A<->B changes, big events).")
    out("        KILL: flat second-order error  => the loop is not engaging.")
    out("    Protocol: Phase A silent >= 200 ticks; then Phase B; sham interleaved")
    out("        in blocks; N>=2 agents (different SOUL/seed). Analysis = this script")
    out("        on the deployed zeta_ticks.jsonl + an LLM blind re-scorer.")
    out("")

    # cleanup temp logs
    for f in RESULTS.glob("_yv_*.jsonl"):
        f.unlink()

    out("SUMMARY")
    out(f"  reporter-validation separation : {sep:+.3f}  [{v1}]")
    out(f"  Phase A Psi exposed            : {'no' if all(e is None for e in exposed_a) else 'yes'}")
    out(f"  feedback faithful / sham decoupled : {v3}")

    (RESULTS / "yvyra_experiment_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
