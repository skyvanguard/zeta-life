# F2.3/F2.4 plan — training P(IK) self-report + Binder validation

Two verified investigations (2026-06-13) before the LoRA. Companion to
`TARGET_SELECTION.md`. F2.2 confirmed the "do I know" signal is in the state
(AUROC 0.717; softmax conf 0.806; MMLU acc 0.723).

## A. How to train P(IK) self-report (verified methods)

- **Kadavath (P(IK), arXiv:2207.05221):** target is **answer-independent** — predict
  "will I get this right?" from the *question alone*. Ground truth = did the model
  answer correctly (sample N at temp 1 → fraction correct as soft label, or hard
  acierta/no). A separate **value head**, **cross-entropy** loss. Calibrates well
  in-distribution, **poorly OOD** → keep the test within MMLU.
- **Lin/Hilton/Evans (verbal confidence, arXiv:2205.14334):** fine-tune the model
  to *emit* calibrated confidence in words ("Confidence: 19%" or 5 words). Beats
  logit baselines under distribution shift. Train label = empirical accuracy per
  sub-task. Metric = MSE/Brier.
- **Binder (privileged access, arXiv:2410.13787):** the validation. M1 self-predicts
  its correctness; **M2 = a different base model trained on the SAME labels** but
  only sees the text. **M1 > M2 (and the gap doesn't close with more M2 data) ⇒
  privileged access; M2 ≈ M1 ⇒ confabulation.** Llama-70B self 48.5% vs cross 31.8%
  (plateau ~35%). Anti-memorisation: change the model's behaviour, self-report
  must follow without retraining on the new behaviour.

**Design for Qwen3-8B + LoRA + MMLU:**
- Dataset: MMLU question + options + an **answer-independent** ask ("¿sabrás la
  respuesta? SÍ/NO y confianza 0-1") — NOT "what's the answer". Target = did Qwen
  get it right (from F2.1; ideally N-sample soft label).
- Output: verbal `SÍ/NO` + a `0.00-1.00` confidence (simplest for LoRA on a causal
  LM); or a value head + BCE (purer Kadavath, more code).
- Loss: cross-entropy on the confidence/decision tokens only (mask the rest).
- **M2 control (F2.4):** a different model (or text-only predictor) fine-tuned on
  the SAME (text→correct) labels. NOTE: the F2.2 activation probe is NOT M2 (it has
  internal access) — it's the ceiling. Metric: AUROC/Brier/ECE of M1 vs M2 on the
  same held-out split; significance + M2 scaling plateau.
- Anti-confabulation: answer-independent target + M2 same-labels + (optional)
  behaviour-modification test.

## B. QLoRA on 12GB Blackwell (verified, practical)

**Viable** — the canonical QLoRA use case (7-8B in ~10-12GB). Critical good news:
**bitsandbytes install docs confirm Windows x86-64 CUDA 12.8-12.9 wheels target
sm120 (Blackwell)**; NF4 needs CC≥6.0 (sm_120 qualifies). Issue #1851: NF4 4-bit
produces correct output on RTX 5090 Blackwell (bnb 0.49.1, CUDA 12.8). **Risk: that
was 5090/Linux; ours is 5070 Ti Laptop/Windows → smoke-test 5-10 real steps first.**

**Config (recommended):**
- Quant: `load_in_4bit, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16`.
- LoRA: `r=16, alpha=32, dropout=0.05, target_modules=all linears
  (q/k/v/o/gate/up/down_proj), task_type=CAUSAL_LM, bias=none`.
- Train: `lr=2e-4, cosine, warmup 0.05-0.1, epochs 2, batch=2, grad_accum=8
  (eff 16), gradient_checkpointing=True, optim=paged_adamw_8bit, bf16=True,
  max_seq=256-512`. Call `prepare_model_for_kbit_training` before LoRA.
- Library: `trl.SFTTrainer` (simplest; pass `peft_config`) or peft+Trainer.
- VRAM ~8-11GB peak. Throughput est. ~150-350 tok/s → ~15-45 min/epoch for short
  MC examples; total ~30 min-2h.

**Pitfalls:** OOM → batch 1 / lower max_seq / r; NaN → bf16 (not fp16), lower lr;
must free the 8-bit inference process first; `prepare_model_for_kbit_training` +
gradient checkpointing to avoid "requires grad" errors.

**Fallback if 4-bit fails on Blackwell:** (1) LoRA in 8-bit (int8 already confirmed
working) — safest; (2) Unsloth NVFP4 (Blackwell-native); (3) smaller model (Qwen3-4B).

## Plan (de-risked)
1. **Smoke test** (~5 min): load Qwen3-8B 4-bit, prepare_for_kbit + LoRA, run 5-10
   real training steps. Check: no kernel error, finite decreasing loss, coherent
   `generate()`. Confirms 4-bit on this exact GPU before committing hours.
2. If OK → build the answer-independent P(IK) dataset (reuse F2.1 ground truth).
3. LoRA train (M1).
4. M2 control + Binder (F2.4): AUROC/Brier M1 vs M2; M1 > M2 = trained introspection.

## Verify
bnb 0.49.2 4-bit on Windows + 5070 Ti Laptop specifically (smoke test); that `trl`
doesn't downgrade transformers 5.12; exact throughput (extrapolated). Kadavath/Lin/
Binder figures from HTML/ar5iv — confirm in PDF before citing in a paper.
Refs: 2207.05221, 2205.14334, 2410.13787, 2305.14314 (QLoRA), bitsandbytes install
docs, bnb issue #1851, Unsloth LoRA guide.
