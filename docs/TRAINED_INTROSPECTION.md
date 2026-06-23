# Trained introspection — research dossier (the north, cycle 2)

Two verified investigations (2026-06-13) before committing to "trained
introspection" (Fran's original zeta-life vision: the model *adapts*, no giant
model needed). Companion to `ANTHROPIC_NORTH.md`. Cycle 1 (spontaneous
introspection) was negative; this scopes cycle 2.

## Headline (verified)

**An official pretrained SAE for Qwen3-8B exists** — "Qwen-Scope" (Qwen/Alibaba,
arXiv 2605.11887). Repo `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` confirmed live
(HTTP 200) with a per-layer SAE (`layer0.sae.pt` … one per layer), 64K features,
Top-k (k=50). So **Phase 1 (a Psi_act that varies) needs no SAE training** — just
download the mid-layer SAE, run hidden states through `encode()`, compute Psi_act
over sparse features instead of raw activations. Caveat: it is for Qwen3-8B-**Base**,
not Instruct — verify alignment, or use the base model. Also verify the SAE behaves
under bitsandbytes-8bit (weight-only int8 leaves activations ~fp16; run an
explained-variance check vs bf16).

## A. Can introspection be TRAINED? (yes, with a big confabulation caveat)

- **Binder et al., "Looking Inward" (ICLR 2025, arXiv:2410.13787)** — fine-tune M1
  to predict properties of its OWN behaviour; M1 predicts itself better than an
  external M2 trained on M1's ground truth (privileged access), and keeps the edge
  **after its behaviour is intentionally changed** (rules out memorised recitation).
  *Fails on complex / OOD tasks.* This is the training protocol + the validation
  (M1 vs M2) for our test.
- **Betley et al., "Tell me about yourself" (ICLR 2025, arXiv:2501.11120)** — models
  fine-tuned on data exhibiting a behaviour (no explicit description) later
  *verbalize* it ("the code I write is insecure"). Closest bridge to our case.
  Caveat: implanted behaviours, not reading a continuous activation metric.
- **Bozoukov et al. (arXiv:2511.04875, 2025)** — behavioural self-awareness induced
  with a single **LoRA rank-1**, mostly captured by **one steering vector**;
  domain-specific. Caveat: if it's a linear steering vector, it's hard to tell
  "introspection" from a disguised linear readout — a direct precaution for Psi_act.
- **Lin, Hilton & Evans (arXiv:2205.14334)** — GPT-3 trained to emit calibrated
  verbalized confidence without logits. Proof you can train a faithful self-report
  of an internal state (its own uncertainty). Degrades under distribution shift.
- **Kadavath et al. (arXiv:2207.05221)** — trained P(IK) ("I know"); good
  in-distribution, **miscalibrated OOD**. Same pattern as Binder.
- **Premakumar et al., "Unexpected Benefits of Self-Modeling" (arXiv:2407.10188)** —
  auxiliary objective of predicting one's OWN activations makes the network simpler
  / more predictable. A possible complement (hypothesis, not guarantee).
- **Readout != introspection** — "Geometry of Truth" (arXiv:2310.06824): a linear
  probe reads a state even when the model outputs the opposite. A probe reading
  Psi_act is NOT the model introspecting it. Must validate separately.

**Verdict:** trained self-report with privileged access works for SIMPLE,
domain-bounded tasks, cheaply (LoRA rank-1). Psi_act (an activation metric) sits in
the complex/OOD regime where confabulation and OOD failure dominate. So **the real
experiment is not training the report — it's proving it's perception, not
confabulation** (M2-control + dissociations + ablations). Realistic expectation in
an 8B: a functional self-report is reachable; demonstrating genuine privileged
access is doubtful (consistent with our spontaneous negative).

## B. SAEs / features that vary (the practical path)

- **Qwen-Scope** (verified): pretrained SAEs for Qwen3 (1.7B/8B/30B) + Qwen3.5;
  residual stream, Top-k, 64K width. Use `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`.
- **Gemma Scope** (DeepMind, arXiv:2408.05147) — high-quality open SAEs for Gemma 2
  (2B/9B/27B). Fallback only if the Qwen SAE disappoints; would mean switching model
  (Gemma-2-9B ~10-11.5GB in 8-bit, tight in 12GB; Gemma-2-2B trivial).
- **Llama Scope** (arXiv:2410.20526), EleutherAI, Goodfire SAEs also exist.
- **Training a SAE ourselves** (SAELens): viable but EXPENSIVE — bottleneck is disk
  (~8KB/token; 100M tokens ≈ 800GB) and days of time. Redundant given Qwen-Scope.
- **Cheaper alternatives for "features that vary"**: PCA over activations (free
  baseline, do this first), linear probes, concept directions (diff-of-means).

**Recommendation:** (1) free PCA baseline over mid-layer hidden states to see if
variance is already recoverable; (2) else use the pretrained Qwen-Scope SAE and
compute Psi_act over sparse features; (3) train our own SAE only as a last resort.

## Refined 3-phase plan (cycle 2)

1. **Psi_act that varies** — PCA baseline (free) → Qwen-Scope SAE features. *No SAE
   training.* Much easier than first thought.
2. **Train introspection** — LoRA fine-tuning with **Binder's protocol** (self-
   prediction; always build the M2 external control), training on **dissociations**
   (cases where Psi_act is NOT inferable from text), optional DPO anti-confabulation.
3. **Validate** — Binder (M1 > M2) + ablations of the source activations + measured
   false-positive rate. Pre-register success criterion (M1>M2 AND survives
   dissociation) before running. Honest: result may be confabulation that fools us —
   the controls ARE the experiment. Expectation in 8B: moderate.

## Verify before citing
arXiv IDs 2605.11887 (Qwen-Scope), 2511.04875, 2501.11120 surfaced via search;
Qwen-Scope repo confirmed live by direct HF API call. The L0_100 repo returned a
transient connection error (not a 404). Some figures ("~20%", "16/52", Gemma Scope
"400+") are from fetch/summaries — confirm against PDFs before a paper. The
"Introspection Adapters" item is a LessWrong post, not peer-reviewed.
