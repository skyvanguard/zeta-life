# Anthropic as the north — research dossier

Two verified investigations into Anthropic's published work (2026-06-13), to guide
the "north": make Ψ a property of an LLM's own activations and test privileged
access. Companion to `RESEARCH_PHASE_B.md`. References verified against primary
sources (transformer-circuits.pub, anthropic.com, arXiv) where stated; items
marked "verify" were surfaced but not read verbatim — do not cite blind.

## The two load-bearing takeaways

1. **The raw hidden state is the wrong basis.** Anthropic's core finding is
   *superposition*: activations pack more concepts than dimensions, so neurons are
   polysemantic and a raw hidden-state dimension ≠ a concept. The right unit is a
   **feature** (a learned direction), extracted with **sparse autoencoders (SAE) /
   dictionary learning**. Our four Ψ_act metrics run on raw hidden states — fine
   as v1, but Anthropic would compute them over **SAE features**.

2. **Concept injection is the gold-standard test, and it's replicable on Qwen3-8B
   without an SAE.** Inject a known concept vector (obtained by *difference of
   means* over contrastive prompts) into the residual stream and ask whether the
   model reports it. This turns our Binder test from *correlational* to *causal*,
   covering Lindsey's grounding + internality criteria. Cheap activation steering;
   no SAE needed for this step.

## Mechanistic interpretability (investigation A)

- **Toy Models of Superposition** (Elhage et al., 2022) — superposition, feature
  importance/sparsity, *feature dimensionality*, dense↔sparse phase transition.
  https://transformer-circuits.pub/2022/toy_model/index.html
- **Towards Monosemanticity** (Bricken, Templeton et al., 2023) — SAE on a 1-layer
  MLP; features more monosemantic/causal than neurons; feature splitting.
  https://transformer-circuits.pub/2023/monosemantic-features/index.html
- **Scaling Monosemanticity: Claude 3 Sonnet** (Templeton et al., 2024) — SAE on a
  mid-layer residual stream; up to 34M features, ~100 active at once; abstract,
  multilingual, multimodal, safety-relevant features.
  https://transformer-circuits.pub/2024/scaling-monosemanticity/
- **Golden Gate Claude** (2024) + **Evaluating feature steering** (Durmus et al.,
  2024) — causal control by clamping a feature; a "sweet spot" ≈ −5..+5, off-target
  effects beyond. https://www.anthropic.com/news/golden-gate-claude ·
  https://www.anthropic.com/research/evaluating-feature-steering
- **Circuit Tracing** + **On the Biology of a Large Language Model** (Ameisen,
  Lindsey et al., 27 Mar 2025) — attribution graphs over cross-layer transcoders;
  multi-hop reasoning, planning, shared multilingual features. Explain only ~¼ of
  prompts, attention frozen, much manual work.
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html ·
  https://transformer-circuits.pub/2025/attribution-graphs/biology.html

*Anthropic does NOT publish a scalar "integration" metric like Ψ.* Their
contribution is (a) a better representation (features), (b) ways to measure the
state's geometry (feature dimensionality, manifolds), and (c) a causal validation
protocol for the state↔report link.

## Introspection + model welfare / consciousness (investigation B)

- **Emergent Introspective Awareness in LLMs** (Jack Lindsey, 29 Oct 2025) —
  *concept injection*: inject a concept vector (contrastive pairs / mean of 50
  random words), measure if the model detects+reports it. **Four criteria for
  genuine introspection: accuracy, grounding (causal), internality, metacognitive
  representation.** Detection *before* it affects output (not confabulation).
  Numbers (verified): "Opus 4.1 succeeds on about **20%** of trials"; "**failures
  of introspection remain the norm**"; **0 false positives in 100 control trials**;
  detection peaks ~2/3 model depth; helpful-only variants > production. Explicit
  disclaimer: results "**don't tell us whether Claude... might be conscious**";
  at most a rudimentary "**access consciousness**", not phenomenal.
  https://transformer-circuits.pub/2025/introspection/index.html
- **Exploring model welfare** (Anthropic, 24 Apr 2025) — model-welfare program (Kyle
  Fish). "No scientific consensus on whether... AI systems could be conscious";
  "humility... as few assumptions as possible". https://www.anthropic.com/research/exploring-model-welfare
- **Self-reports are not reliable evidence** (Eleos AI, Claude-4 eval notes) — no
  independent evidence of welfare-relevant states; no obvious introspective
  mechanism; self-reports shaped by pretraining imitation, system prompt,
  post-training; extreme suggestibility. https://eleosai.org/post/claude-4-interview-notes/
- **Taking AI Welfare Seriously** (Long, Sebo, Fish et al., 2024, arXiv:2411.00986)
  and **Consciousness in AI** (Butlin, Long et al., 2023, arXiv:2308.08708) — the
  indicator-properties framework the project already uses (`INDICATOR_PROPERTIES.md`).
- ~15% (Kyle Fish) / 15–20% (Claude self-assessed) consciousness probability —
  *personal hedges / sycophancy-prone self-reports, not paper results*. Cite as such.

## Implications for the north (refined plan)

**The Binder test (#17) should follow Lindsey's 4 criteria, not just accuracy:**
1. **Accuracy** — does the self-report predict Ψ_act better than an external
   predictor that sees only the text? (what we had)
2. **Grounding (causal)** — *concept-injection*: perturb the state Ψ_act measures
   and check the self-report changes coherently. **Replicable on Qwen3-8B via
   difference-of-means steering, no SAE.** This is the upgrade.
3. **Internality** — detection happens before/independent of output effects.
4. **Metacognitive representation** — reproduce the control: no-injection trials
   where the model must NOT report the state (target the 0/100 false-positive bar).

**Sequence (from the research):**
1. Concept injection on Qwen3-8B (mean-diff vectors) — replicate Lindsey, get causal
   grounding now, cheaply.
2. (Optional, later) train a mid-layer SAE on Qwen3-8B (SAELens ecosystem; verify
   if public SAEs for Qwen3-8B exist before assuming) and recompute Ψ_act over
   features vs raw hidden states.
3. Full Binder test under the 4 criteria.

**Framing discipline (inherit from Anthropic):** say "introspective access /
self-report fidelity", never "consciousness"; Ψ is a monotone integration signal,
not a consciousness measure; indicators ≠ consciousness (Butlin). Expect a small,
fragile effect (~20% ceiling); confabulation is the default; injection is an
unnatural setting (limited ecological validity); a passing mechanism "could still
be rather shallow". A small effect with clean controls (the 0/100 bar) is
publishable exactly as Anthropic presents it.

**Resource gap (honest):** Anthropic runs production models with 34M-feature SAEs
and dedicated teams. Realistic for us: a smaller SAE on Qwen3-8B, mean-diff concept
injection, and the 4-criteria rubric. The full attribution-graph / circuit-tracing
pipeline is the least realistic to replicate at our scale.

## Verify before citing
SAE feature counts/intermediate sizes (1M/4M) and dead-feature/reconstruction
numbers from Scaling Monosemanticity (page exceeded fetch); the Opus 4.6 system
card page/version (212 pages, Feb 2026 — secondary sources); *When Models
Manipulate Manifolds* (arXiv:2601.04480) transformer-circuits entry; third-party
introspection follow-ups (arXiv 2603.21396, 2602.20031, 2512.12411 — not read).
The Lindsey introspection paper and the model-welfare post were verified directly.
