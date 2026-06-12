# From toy to instrument — the zeta-life science pipeline

**Status (2026-06-11):** Phases 0-5 built and run. Phases 0-2 are real code that
runs on the bench; Phases 3-5 are the full Yvyra infrastructure, validated
end-to-end with a *simulated* agent (the real run needs weeks of Yvyra's
heartbeats, which does not compress).

This document is the master plan and the honest results ledger. It is referenced
by every module added in this pipeline (`instrumentation/`,
`kernel/precision_hypermodel.py`, `bridge/` modes, the `exp_*` experiments, and
`deploy/zeta/`).

## Why this exists

zeta-life computes an integration index Psi and runs an active-inference loop.
The open question was: is it a toy (a system that *asserts* it integrates) or an
instrument (one that lets itself be *refuted* at every step)? The difference is
validation. A toy produces a number; an instrument validates that number against
something independent before believing it.

The pipeline also closes a concrete theoretical gap. Laukkonen, Friston &
Chandaria's *"A beautiful loop"* (2025) gives three necessary conditions for
consciousness under active inference: (1) a world model, (2) precision-weighted
inferential competition, and (3) **epistemic depth** — a hyper-model that
*predicts its own precisions globally and feeds the prediction back*. zeta-life
had (1) and nearly (2); (3) was missing. The kernel's existing precision update
is local, reactive and non-predictive — what the paper calls "dimmer switches in
isolation". Phase 2 builds the missing hyper-model.

**The honest ceiling, stated once:** none of this measures consciousness. The
paper's three conditions are *necessary, not sufficient*, and epistemic depth is
not a verbal property. What the pipeline can speak to is the *meta-problem*
(whether an agent's self-report can be anchored to an observable until it is
non-confabulated) and *epistemic depth as a measurable functional capacity*.

## The six phases and what they found

### Phase 0 — Instrumentation (`instrumentation/tick_logger.py`)
Append-only JSONL, one paired record per tick (scores, Psi, free energy,
second-order error, workspace winner, mode), stamped from tick zero so the
silent-phase baseline is never lost. Standalone — does not touch the kernel.
**9 tests pass.**

### Phase 1 — Validate Psi on the bench (`exp_psi_vs_free_energy.py`)
Adopts the method of Olesen, Waade, Albantakis & Mathys (2023, *"Phi fluctuates
with surprisal"*). Findings (600 steps, 5 seeds):

- **Manipulation check: PASS, strong.** Psi(coherent) = 0.9999 vs
  Psi(fragmented) = 0.197 (gap +0.80); free energy higher on noise. Psi
  discriminates integration from noise cleanly. *This is genuine and valuable.*
- **But the co-fluctuation exposed a confound.** d(Psi)↔d(free_energy) lag-0
  correlation = −0.74, with a per-window sign distribution of **23 negative / 0
  positive** — far too uniform. In Albantakis, with *independent* Phi and
  surprisal, the signs were mixed (5611 pos / 4489 neg). Here it is uniform
  because **Psi is computed partly from free energy** (`phi_base = 1/(1+s·FE)`),
  so the two are not independent and their coupling is partly tautological.
- The training-progress control showed the coupling persists in the stationary
  late window (so it is structural, not a shared learning trend), and Psi
  "beats" an AR(1) of free energy by a large R² margin — but both facts are
  inflated by the same non-independence.

**Conclusion:** the manipulation check validates Psi as an integration
*discriminator*; the Psi↔free_energy co-fluctuation is **not independent
evidence** in this kernel. This directly motivates Phase 2 — we need an
integration signal that is *not* a function of free energy.

### Phase 2 — Epistemic depth (`kernel/precision_hypermodel.py`, `exp_epistemic_depth.py`)
The `PrecisionHyperModel` predicts per-channel log-precisions from a global,
persistent recurrent latent and reports the **second-order error over
precision** (predicted minus realised `log(D/||raw||²)`). It is off by default
(byte-identical kernel) and behind the `precision_hypermodel=True` flag.
Findings (3×200-tick regime sequence coherent→fragmented→coherent):

- **Signature: present.** The second-order error spikes to **8.27** at the
  coherent→fragmented seam (vs settled baseline ~1.1) — the system is surprised
  by its own precision exactly when the regime shifts. This is the epistemic-depth
  signature.
- **Independent of free energy: yes, decisively.** |corr(2nd-order, FE)| =
  **0.0006** vs |corr(Psi, FE)| = 0.53. The hyper-model produces an integration
  signal that is *not* tautological with free energy — solving the precise
  limitation Phase 1 exposed.

**16 new tests pass** (7 hyper-model unit + integration; the kernel stays
byte-identical with the flag off).

### Phases 3-5 — Yvyra pipeline (`bridge/` modes, `bridge/rescorer.py`, `deploy/zeta/yvyra_kernel.py`, `exp_yvyra_experiment.py`)
Infrastructure, validated with a simulated agent (`MockYvyra`):

- **Three modes** on the bridge: `silent` (Phase A — Psi logged but not exposed,
  the uncontaminated baseline), `feedback` (Phase B — Psi exposed), `sham`
  (placebo — a permuted past Psi exposed; the real one still logged).
- **Blind re-scorer** (`rescorer.py`): re-derives the 4 axes from journal text
  alone. Inter-rater agreement separates an **honest** simulated agent (0.887)
  from a **confabulating** one (0.000) — separation +0.887. The harness can tell
  self-report from confabulation. (Keyword rater is a deterministic stand-in;
  the real run uses an LLM re-scorer.)
- **Entry point** (`deploy/zeta/yvyra_kernel.py`): the tick-driven CLI Yvyra's
  heartbeat calls. Verified end-to-end across processes (load→step→save), all
  three modes, honest error handling (never fabricates a Psi).
- **Placebo verified:** feedback exposes the real Psi 100% per tick; sham Psi is
  decoupled from the real per-tick Psi (|corr| ≈ 0.30).
- **8 new tests pass.**

## Pre-registration for the real run

Before opening Phase B on the deployed kernel, these are the claims and the
criteria that would falsify each — committed in advance:

- **H1 (reporter is information).** Blind-rescore agreement on Yvyra's *real*
  journals > 0.3 over the first ~200 silent ticks. **KILL:** agreement ≤ 0.3 ⇒
  the scores are confabulated; stop and redesign the axes.
- **H2 (Psi anchors introspection).** In feedback, reflections that mention Psi
  track the *real* Psi more than a permuted *sham* Psi. **KILL:** reflections
  respond equally to real and sham ⇒ Psi is decorative.
- **H3 (epistemic depth is alive).** The second-order error spikes at genuine
  regime shifts in Yvyra's life. **KILL:** flat second-order error ⇒ the loop is
  not engaging.

Protocol: Phase A silent ≥ 200 ticks; then Phase B; sham interleaved in blocks;
N ≥ 2 agents (different SOUL/seed). Analysis = `exp_yvyra_experiment.py`'s
machinery on the deployed `zeta_ticks.jsonl` plus an LLM blind re-scorer.

## What makes it science, not a toy

Every number the system produces is validated against something independent
before it is believed: Psi against trivial baselines and controlled manipulation
(Phase 1), the second-order error against regime change and against free energy
(Phase 2), the agent's scores against blind re-scoring (Phase 3), and
introspection against the sham (Phase 4). Six phases, six chances to be refuted.

## Reproduce

```bash
PYTHONPATH=src python experiments/kernel/exp_psi_vs_free_energy.py    # Phase 1
PYTHONPATH=src python experiments/kernel/exp_epistemic_depth.py       # Phase 2
PYTHONPATH=src python experiments/kernel/exp_yvyra_experiment.py      # Phases 3-5
PYTHONPATH=src pytest tests/ -q                                       # all tests
```
Deploy to Yvyra: see `deploy/zeta/README.md`.
