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

Appended to the existing introspection prompt (v2 — a LEVEL, not the number, so
she cannot echo digits):

> Tu nivel de integracion en tu ultimo tick fue **<ALTA|MEDIA|BAJA>** (BAJA =
> experiencia fragmentada/dispersa, MEDIA = parcial, ALTA = muy integrada y
> coherente). Contempla esto como parte de tu reflexion: ¿se condice con como
> sentis tu propia coherencia ahora, o no? Al final, en UNA linea aparte, deci tu
> integracion percibida en este momento con el formato exacto:  SIENTO: 0.X
> (deci el numero que de verdad sentis, no repitas la categoria).

Bash parses the `SIENTO: 0.X` line as Yvyra's auto-score; the rest is her
reflection text (for the sentiment classifier and the journal). The **text
sentiment is the primary signal** — she cannot produce it by copying.

## What gets logged per Phase-B tick (mode A)

Extends the paired log with:
- `psi_real` — the kernel's true Psi this tick (always logged).
- `psi_exposed` — the Psi the shown level came from (== psi_real in real blocks;
  a permuted past Psi in sham).
- `level_exposed` — the qualitative level actually shown (ALTA/MEDIA/BAJA).
- `block` — `real` | `sham`.
- `felt` — Yvyra's parsed auto-score (`SIENTO: 0.X`), or null if absent (secondary).
- `sentiment` — integration sentiment classified from her reflection text (primary).

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

## v1 -> v2: what the first live run taught us (2026-06-13)

The first deployment (v1) ran 27 mode-A ticks and then was stopped: the data was
invalid for two reasons, plus a third hypothesis that the data refuted.

1. **Sham control was broken (placebo did not exist).** `psi_exposed == psi_real`
   in 27/27 ticks, including 12/12 sham. Cause: the bridge's `_psi_buffer` (the
   pool of past Psi the sham permutes) lived in memory, but each tick is a fresh
   process, so it was always empty and the code fell back to the real Psi. **Fix
   (v2):** persist the buffer + RNG state in a `<name>.bridge.json` sidecar across
   `save()`/`load()`.

2. **Yvyra echoed the number.** `felt` copied the injected Psi in 23/26 ticks —
   often to 10 identical decimals (`felt=0.9999923322` after being shown
   `0.999992`). The auto-score measured digit-copying, not introspection. **Fix
   (v2):** expose Psi as a qualitative **level** (ALTA/MEDIA/BAJA), never the
   number, so her `SIENTO: 0.X` must be her own judgement; and make the **text
   sentiment** (which she cannot fake by copying) the primary signal.

3. **"Psi is saturated" — refuted.** It looked pinned at ~0.99 in the 27-tick
   sample, so we hypothesised the binary-flag axes were too constant and tried
   deriving the 4 axes from the reflection text. On the full 559-tick production
   log Psi is **not** saturated: it is **bimodal** (std 0.42; 27% of mode-A ticks
   below 0.5), and the dips are driven by regime transitions (79% of low-Psi
   ticks follow a research<->introspection switch), not by axis coarseness.
   Replaying the real journals, the text-derived axes did **not** help
   (std 0.367 -> 0.329). So the axes were left unchanged; Psi already has the
   variance Phase B needs once enough ticks accumulate.

## Implementation checklist

1. `tick.sh` (mode A): block decision, Psi injection from `.last_psi`, augmented
   prompt, parse `SIENTO:`, persist exposed Psi for next tick, extended logging.
2. `bridge/integration_sentiment.py` — the keyword classifier.
3. The bridge/entry point already expose `feedback`/`sham`; `tick.sh` selects per
   block.
4. `experiments/kernel/exp_phase_b_analysis.py` — the block correlations.
5. Switch the heartbeat from Phase A (silent) to Phase B; new log section.
