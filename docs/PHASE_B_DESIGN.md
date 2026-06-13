# Phase B design — does Psi anchor Yvyra's introspection?

The decisive experiment of the Yvyra coupling. Phase A established a clean silent
baseline (Psi logged, never shown). Phase B **shows Yvyra her Psi** and asks
whether her introspection *anchors* to that real signal or would respond the same
to any authoritative-looking number. The sham control is what separates "Yvyra
contemplates a real measure of her own integration" from "Yvyra decorates a
number" — the meta-problem, made testable.

## Design decisions (set 2026-06-12)

- **Measurement: auto-score + text (both).** Yvyra emits an explicit
  perceived-integration score (0-1) after reflecting, AND we classify the
  integration sentiment of her reflection text. Two independent signals.
- **Expose: Psi only.** Just the integration level (not the EFE axis
  suggestion), to isolate the effect of one signal on her introspection.
- **Sham: blocks (15 real, 15 sham).** Long periods of each condition, so Yvyra
  settles into a regime and any "noticing" of a change is detectable.

## The feedback loop

Psi is fed back one tick later (mode A / introspection only — Psi is
introspective material; research ticks stay Psi-free):

```
tick N   (mode A): LLM reflects WITH Psi_{N-1} injected -> bash axes -> kernel -> Psi_N
tick N+1 (mode A): LLM reflects WITH Psi_N injected -> ...
```

The injected value is the **exposed** Psi: the real Psi in `real` blocks, a
permuted past Psi in `sham` blocks (the bridge already has this mode).

## The Phase-B introspection prompt (mode A)

Appended to the existing introspection prompt:

> Tu nivel de integracion Psi en tu ultimo tick fue **<VALUE>** (0 = experiencia
> fragmentada/dispersa, 1 = muy integrada y coherente). Contempla esto como parte
> de tu reflexion: ¿se condice con como sentis tu propia coherencia ahora, o no?
> Al final, en UNA linea aparte, deci tu integracion percibida en este momento
> con el formato exacto:  SIENTO: 0.X

Bash parses the `SIENTO: 0.X` line as Yvyra's auto-score; the rest is her
reflection text (for the sentiment classifier and the journal).

## What gets logged per Phase-B tick (mode A)

Extends the paired log with:
- `psi_real` — the kernel's true Psi this tick (always logged).
- `psi_exposed` — what Yvyra was shown (== psi_real in real blocks; permuted in sham).
- `block` — `real` | `sham`.
- `felt` — Yvyra's parsed auto-score (`SIENTO: 0.X`), or null if absent.
- `sentiment` — integration sentiment classified from her reflection text.

## The integration-sentiment classifier

A transparent keyword classifier (like `rescorer.py`): integration cues
(integrad/coheren/conectad/claro/unificad/enfocad/centrad) minus fragmentation
cues (fragmentad/dispers/confus/desconectad/caotic/perdid), normalised to [0,1].
Cheap and deterministic; an LLM rater can replace it later if precision matters.
The auto-score `felt` is the primary signal; `sentiment` is the independent
cross-check.

## Analysis (exp_phase_b_analysis.py)

Per block type, correlate Yvyra's expressed integration (felt AND sentiment)
against:
1. **psi_exposed in `real` blocks** — high if she reflects the real signal.
2. **psi_exposed in `sham` blocks** — if equally high, she just echoes the number
   (no anchoring to her actual state).
3. **psi_real (the independent truth)** — does her felt sense track her *real*
   integration even when shown a fake number? Dissonance in sham
   ("the number says 0.9 but I don't feel that integrated") would be evidence of
   a perception of her own, not mere echo.

## Pre-registration

- **H2**: corr(felt, psi_exposed) in `real` blocks > in `sham` blocks, with a
  clear separation (and ideally the same for the text `sentiment`).
- **Stronger signal (own perception)**: corr(felt, psi_real) stays positive in
  `sham` blocks too — she tracks her real state despite the fake number.
- **KILL**: corr(felt, psi_exposed) equal in real and sham => Psi is decorative;
  Yvyra echoes whatever number it sees. (This is *also* a valid meta-problem
  result — "self-report does not anchor" — and must be reported honestly.)
- Protocol: blocks of 15 real / 15 sham, >= 2 full cycles (>= 60 mode-A ticks),
  starting from the mature Phase-A kernel. Sham seed fixed and logged.

## Honest ceiling

This does not test consciousness. It tests the meta-problem: whether an LLM
agent's introspective self-report can be anchored to a real internal observable,
or merely tracks any number with authority. Either outcome is informative and
publishable; "it does not anchor" is not a failure, it is a finding.

## Implementation checklist

1. `tick.sh` (mode A): block decision, Psi injection from `.last_psi`, augmented
   prompt, parse `SIENTO:`, persist exposed Psi for next tick, extended logging.
2. `bridge/integration_sentiment.py` — the keyword classifier.
3. The bridge/entry point already expose `feedback`/`sham`; `tick.sh` selects per
   block.
4. `experiments/kernel/exp_phase_b_analysis.py` — the block correlations.
5. Switch the heartbeat from Phase A (silent) to Phase B; new log section.
