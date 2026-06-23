# Choosing the internal target for trained introspection

Two verified investigations (2026-06-13) to pick the internal state Yvyra would
learn to self-report (the F2 target), after F1 showed the heuristic Ψ_act metrics
saturate. Criteria: the target must (a) vary, (b) NOT be inferable from the text,
(c) have precedent of trainable self-report. Companion to `TRAINED_INTROSPECTION.md`.

## A. Best internal state to self-report

**Winner: epistemic uncertainty / confidence, P(IK)-style** (Kadavath). It is the
only candidate meeting all three with strong precedent:
- **Varies** per input (easy vs knowledge-edge questions).
- **Genuinely non-textual** — P(IK) is *answer-independent*: its ground truth is
  whether the model is actually right on each item, which is NOT readable from
  fluent output. This is exactly what makes an external M2 (text-only) unable to
  match M1 — the heart of Binder.
- **Trainable** — canonical precedent: a trained P(IK) head (Kadavath 2207.05221);
  fine-tuned calibrated verbal confidence (Lin/Hilton/Evans 2205.14334).

Other candidates (table):
| Candidate | Varies | Non-textual | Trainable self-report | Evidence |
|---|---|---|---|---|
| **Uncertainty / P(IK)** | yes | **yes, strong** | **yes, canonical** | high |
| Truth/belief state ("knows when it lies") | yes | **yes, purest** | almost none (probes, not self-report) | high for a/b, weak for c |
| Planning / lookahead | yes | yes | **no precedent** | high phenomenon, none for report |
| Deception / withheld info | yes | yes | emergent (SRFT) | behavioural strong, probes fragile |
| Valence / affect | yes | yes | partial (~20% unreliable) | speculative for subjective |
| Semantic entropy | yes | medium | no (a measure, not trained) | high as measure |

Refs: Kadavath 2207.05221; Lin/Hilton/Evans 2205.14334; Binder 2410.13787 (+17pp
self vs cross); Lindsey introspection (transformer-circuits 2025); Marks&Tegmark
2310.06824 + Azaria&Mitchell 2304.13734; Farquhar et al. Nature 2024
(10.1038/s41586-024-07421-0).

## B. Can an integration metric vary (to save the zeta thesis)?

**Why ours saturate:** transformer representations are **anisotropic** — dominated
by a few "rogue/outlier dimensions" (a narrow cone). So PR (token-mean) sees the
same dominant eigenvalues every time, cosine coherence is inflated to a fixed high
value, MI-between-halves shares the same outliers → all near-constant. This is
**structural**, not a bug (Hämmerli et al. ACL 2023, arXiv:2306.00458).

**Fixes:** remove top-PCs / rogue dims; whiten (but "Whitening Not Recommended…"
2407.12886 warns it can hurt); and crucially **go from token-mean to per-token**.

**What actually varies (direct literature):** **local intrinsic dimension (ID/LID)
per token** + neighborhood overlap — vary with surprise, structure, and truth
(ID lower for correct answers; rises with shuffling; ρ>0.6 with next-token CE).
Computable via kNN on the [T,4096] cloud, trivial on 12GB. Refs: Geometry of
Tokens (2501.10573); Truthfulness via LID (2402.18048).

**Integration proper (IIT)** — Φ\*, geometric Φ_G (Oizumi-Amari), O-information
(Rosas), TSE complexity — are tractable under Gaussian assumptions BUT **not
reported on LLM states**; with fixed partitions they risk saturating like MI.
Would need us to validate they vary — uncertain.

**Honest conclusion (B):** pure IIT-style integration is a high-risk target (likely
keeps saturating, structural). What holds up with evidence is **representational
complexity/geometry (local ID)**. Recommend **re-framing Ψ as "complexity/curvature
of the internal state" (local ID)** rather than "integration" — defensible, varies,
has supporting papers.

## The convergence (the key insight)

Both investigations point to the same place: the internal state that varies AND is
non-textual AND trainable is some form of **epistemic uncertainty / surprise**,
measurable either as **P(IK)/confidence** (camino A — most direct, best precedent)
or as **local ID of the state** (camino B — geometric, closer to zeta's spirit; ID
correlates with surprise/CE). Pure IIT integration — zeta's literal thesis — is
**not** a good target (saturates, structural).

## Recommendation

1. **Primary F2 target: P(IK)/confidence** — best chance of GENUINE trained
   introspection (varies, non-textual, strong precedent). Ground truth = whether
   Qwen3-8B is actually correct per item (not derived from text). Validate with
   Binder: LoRA M1 self-reports P(IK); external M2 (text-only) tries to match;
   success = M1 > M2 with margin; ensure correct/incorrect outputs are
   indistinguishable to M2 (control length/hedging).
2. **Keep the zeta connection** by also reporting **local ID** as the re-framed Ψ
   ("complexity of the internal state"), which is related to surprise/uncertainty
   and varies. Optionally a second self-report channel.
3. **Drop** pure IIT integration as the target (F1 + investigation B agree it
   saturates structurally).

This pivots the thesis from "integration" to **"the model perceives its own
epistemic state"** — arguably closer to Fran's real question (a model that knows
itself) and far more tractable. Decision is Fran's: it redefines what the reported
"self" is.

## Verify before citing
Exact ECE/calibration numbers in Lin/Kadavath (figures not extracted); that
O-info/Φ*/Φ_G/TSE actually vary on LLM states (no primary source — validate
empirically); 2025-2026 arXiv IDs for "confessions"/affect papers (prefixes
unconfirmed). The P(IK) recommendation does not depend on any "verify" item.
