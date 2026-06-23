# Research dossier — Phase B and the introspection question

Consolidated findings from three parallel literature investigations (2026-06-13),
run to decide how to proceed after Phase B v2 came back **inconclusive by Psi
saturation** (see `PHASE_B_DESIGN.md`, `fase-b-inconclusa-saturacion-psi.md`).
References were verified against primary sources where marked; items flagged
"verificar" were surfaced by search but not opened directly — do not cite them
without checking.

## The question

Phase B asks whether Yvyra's introspective self-report **anchors to her real Psi**
(she perceives her integration) or merely **describes her experience** / echoes an
authoritative number. The bridge exposes Psi as a level (ALTA/MEDIA/BAJA); a sham
control shows a fake level.

---

## A. Metacognition / introspection in LLMs

**State of the art.** LLMs have *some* genuine introspective access to internal
states, but it is **narrow, unreliable (~20% in the best causal tests),
content-blind (detects magnitude, not semantics), prompt-fragile, and only in the
most capable models**. The default explanation for fluent self-reports remains
**confabulation** (plausible post-hoc narrative, not a readout). Critically,
self-reports of confidence/state **anchor to prompt cues** (personas, "are you
sure?", injected hints) rather than tracking a stable internal state — exactly
Phase B's risk.

**Key findings.**
- *Genuine but minimal introspection exists* — proven by cross-prediction
  (privileged access): a model predicts its own behaviour better than another
  model trained on its outputs, and the edge persists after deliberately changing
  its behaviour. **Binder et al., "Looking Inward", ICLR 2025 (arXiv:2410.13787).**
  Authors' own limit: succeeds on simple tasks, fails on complex/OOD ones.
- *Causal introspection ~20%* — concept injection (activation steering): inject a
  concept vector, ask if the model notices an injected thought. Claude Opus 4.1
  detects ~20%, **0 false positives in 100 control trials**. **Lindsey/Anthropic,
  "Emergent Introspective Awareness", transformer-circuits.pub, 2025-10-29.**
  Explicitly: "failures of introspection remain the norm"; not a consciousness claim.
- *Confabulation is real and triggerable* — CoT often doesn't name the real cause
  of an answer (injected bias changes the answer but not the explanation).
  **Turpin et al., NeurIPS 2023 (arXiv:2305.04388); Lanham et al. 2023
  (arXiv:2307.13702); Anthropic 2025 (arXiv:2505.05410).**
- *Self-report anchors to cues, not state* — confidence follows personas /
  "are you sure?" while real accuracy is unchanged. **Findings ACL 2025
  (arXiv:2506.00582); sycophancy: Sharma et al. 2023 (arXiv:2310.13548).** This is
  the most direct mirror of Phase B's null hypothesis.
- *Skeptic criterion* — a report is introspective only if it describes an internal
  state via a causal state→report process (privileged access). **Song et al. 2025
  (arXiv:2508.14802); Comșa & Shanahan 2025 (arXiv:2506.05068).**

**Application to Yvyra.** An observed anchoring of her self-report to the injected
level is **the default prediction of confabulation/sycophancy**, not evidence of
Psi perception. Controls the literature mandates: (1) sham + cue-anchoring
analysis; (2) **external-predictor test (Binder)** — the most decisive and cheap;
(3) activation probe vs verbal report; (4) factorial (injected level × real Psi)
with incongruent trials; (5) report AUROC, expect weak discrimination;
(6) format-robustness; (7) pre-reg + blind re-scorer (already in pipeline).

---

## B. Separating "perceiving a state" from "describing experience" (common cause)

**The core.** When two signals covary (Psi and text), a **fork** (common cause:
experience → Psi and experience → text) and a **chain** (mediation:
experience → Psi → text, "report anchors to Psi") produce **the same observational
pattern** — indistinguishable from passive data, and **no amount of data fixes it**
(Reichenbach; Pearl). The sham controls "obey an authoritative number" but **does
not break the fork**. Only **intervention** — `do(Psi)` with content held fixed —
separates them.

**Methods to separate perception from description.**
- *Three dissociable interoception dimensions* (accuracy/sensibility/awareness): a
  system can confidently report on a state it cannot detect. **Garfinkel et al.,
  Biological Psychology 104 (2015), n=80.**
- *meta-d′ / M-ratio* — separates first-order (content discrimination) from
  second-order metacognitive sensitivity. **Maniscalco & Lau 2012; Fleming 2017.**
- *No-report paradigms* — isolate the confound of the act of reporting; but
  removing report doesn't remove the cognition that shares its cause (Block 2019).
  **Tsuchiya et al., TiCS 2015.**
- *Confabulation prototype* — verbal reports of one's own causes are generated from
  a priori theories, decoupled from the real process. **Nisbett & Wilson,
  Psych. Review 84 (1977).**
- *LLM-specific*: biasing-feature test (Turpin 2023), CoT perturbation (Lanham
  2023), **concept injection as do(state) (Anthropic 2025)**.

**Causal/statistical tools.** Fork ≡ chain in conditional independence; Reichenbach
screening-off is necessary but not sufficient; only interventions partition the
Markov-equivalence class (Hauser & Bühlmann, JMLR 2012). `do(Psi)` is the clean
separator. Observational substitutes (causal mediation w/ sequential ignorability;
instrumental variables) trade the problem for untestable assumptions; beware
collider bias when filtering trials (Munafò et al. 2016).

**Application to Yvyra.** Correlation + sham cannot distinguish anchoring from
common cause. The decisive move is to **turn observation into intervention**.
Operationalize content (first-order: richness/novelty/length) vs state (Psi);
measure Psi-sensitivity of the text **above content** (partial coefficient, with a
confounding sensitivity analysis). Diagnostic cases = **dissociations** where Psi
decouples from content (the regime transitions). Honest expectation: genuine
grounding, if any, is partial and noisy (~20% ceiling).

---

## C. Varied / non-saturated experience for an autonomous agent

**The core.** The valuable signal is not static novelty but the **change in
predictability** (prediction error, ensemble disagreement, or **learning
progress** = the *derivative* of the error). When an agent masters its environment
those signals collapse — exactly Yvyra's case (world model predicts her well, free
energy low, Psi pinned high). The fix is not injected noise but moving the agent to
regimes where its own prediction still fails, and regulating difficulty to keep it
in its zone of progress.

**Approaches.**
- *ICM — curiosity as prediction error in feature space.* **Pathak et al., ICML
  2017 (arXiv:1705.05363).** Robust to the noisy-TV problem.
- *Random Network Distillation* — cheap, stable novelty bonus; easiest to port to
  LLMs. **Burda et al. 2018 (arXiv:1810.12894).**
- *Plan2Explore* — explore toward *expected* surprise via ensemble disagreement;
  maps onto the kernel's `dynamics_ensemble`/`wm_disagreement_heads`. **Sekar et
  al., ICML 2020.** (Caveat: `exp_curiosity.py` found disagreement functionally
  flat in our 4-D regime — verify before relying on it.)
- *Never Give Up / Agent57* — episodic + long-term novelty (two timescales),
  maps to FastMemory/SlowMemory. **Badia et al. 2020 (arXiv:2003.13350).**
- *Novelty search* — reward behavioural novelty, ignore the objective; covers the
  experience space by construction. **Lehman & Stanley, ECJ 2011.**
- *POET / open-endedness* — co-evolve tasks with solutions; emergent curriculum.
  **Wang, Lehman et al. 2019.**
- *Learning progress (Oudeyer & Kaplan, IAC/SAGG-RIAC, IEEE TEC 2007)* — reward the
  *reduction* of error over time; both mastered (error≈0) and irreducibly random
  (error high, progress≈0) get low reward → agent drawn to the learnable frontier.
  **This explains the saturation directly and is the canonical fix.** See also
  Automatic Curriculum Learning (Portelas et al., IJCAI 2020, arXiv:2003.04664).

**LLM-specific, applicable now.**
- *i-MENTOR* — RND at the sequence level for LLM reasoning, O(1) cost, small nets,
  4–15% overhead, larger gains on harder tasks. **arXiv:2505.17621.** The closest
  proven port of RND to an LLM-by-ticks setting.
- *Self-Questioning LMs / PAPRIKA* — agent generates its own tasks of increasing
  difficulty. **arXiv:2508.03682; PAPRIKA (OpenReview UeB3Hdrhda).**

**Application to Yvyra (mechanisms by ticks, ordered by likelihood of producing
Psi↔content dissociations):** (1) RND over journal embeddings (4-axis + topic, not
raw tokens); (2) learning-progress reward (error derivative per topic);
(3) novelty search over the 4-axis behaviour footprint; (4) self-generated tasks
of growing difficulty; (5) ensemble-disagreement curiosity (reuse existing infra —
but verify it isn't flat). Keep a slow long-term term (SlowMemory + dream replay)
to preserve identity while short-term novelty varies.

---

## Synthesis — the reframe

All three converge on one point we had not stated clearly:

> **Psi is not an internal state of Yvyra.** It lives in the *kernel*, which runs
> *outside* her and is computed *downstream* of her text. Yvyra (the LLM) does not
> have Psi in any of her activations.

The introspection literature (Anthropic, Binder) assumes the state being
introspected is **inside** the model — which is why activation steering can inject
it. Here there is **no `do(Psi)` possible on Yvyra**: Psi is not in her. The level
we inject in the prompt is an *external cue*, not an intervention on a state of
hers.

Therefore **"does Yvyra perceive her Psi?" is ill-posed** — she has no channel for
it. The most that can exist is **common cause** (her experience generates both text
and, downstream, Psi) or **textual metacognition** (she infers her state by
re-reading her own text). And Psi has a component — the world model's surprise /
regime transitions — that Yvyra does **not see** when she writes; that is where
Psi decouples from content (the dissociations).

### Three paths

1. **Privileged-access test (Binder)** — cheap, decisive, doable now on the 144
   existing ticks. Train an external predictor that sees only Yvyra's journal and
   predicts Psi; compare to how well her own self-report predicts Psi. If the
   external predictor matches her → all Psi info is in the text = pure common
   cause, no privileged introspection. **First obvious step.**
2. **De-saturate to create dissociations (not just range)** — RND / novelty-search
   / learning-progress so Psi decouples from content. Secondary to (1).
3. **Real `do(Psi)`** — ruled out for Yvyra: Psi is not in the LLM. Only possible
   if redesigned so Psi were an internal variable of the agent (a different
   project).

## Verification note

Verified against primary sources (fetch/search): Binder 2410.13787; Lindsey/
Anthropic introspection 2025; Garfinkel 2015; Reichenbach (SEP); Turpin
2305.04388; Pathak 1705.05363; Burda 1810.12894; Sekar (PMLR v119); Oudeyer &
Kaplan (IAC 2007); Lehman & Stanley 2011; i-MENTOR 2505.17621. **Flagged
"verificar" (surfaced but not opened — do not cite blind):** Hahami "Feeling the
Strength" (2512.12411), Saadat & Nemzer "Certainty Robustness" (2603.03330),
NeuroFaith (2506.09277), exact primary cite for Gazzaniga's interpreter, do-calculus
completeness attribution. Several future-dated arXiv IDs surfaced by search were
discarded as likely index hallucinations and not used.
