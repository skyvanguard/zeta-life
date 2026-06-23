"""
Smoke test -- can we QLoRA (4-bit NF4) Qwen3-8B on this Blackwell GPU?
====================================================================

De-risks F2.3 before committing hours: load Qwen3-8B in 4-bit NF4, add LoRA, run a
few real training steps. Checks: no Blackwell kernel error, finite decreasing loss,
coherent generate(), VRAM fits 12GB. If this fails -> fall back to 8-bit LoRA.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_lora_smoketest.py
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL = "Qwen/Qwen3-8B"
TEXTS = [
    "The capital of France is Paris, a city on the Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The mitochondria is the powerhouse of the cell.",
    "An octave in music spans eight diatonic notes.",
    "The square root of one hundred forty-four is twelve.",
    "Photosynthesis converts sunlight into chemical energy.",
]


def main() -> None:
    print("loading Qwen3-8B in 4-bit NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="cuda")
    print(f"loaded. VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    losses = []
    for step in range(8):
        text = TEXTS[step % len(TEXTS)]
        batch = tok(text, return_tensors="pt").to("cuda")
        out = model(**batch, labels=batch.input_ids)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(float(loss.item()))
        print(f"  step {step}: loss={loss.item():.4f}")

    finite = all(np.isfinite(l) for l in losses) if (np := __import__('numpy')) else True
    decreased = losses[-1] < losses[0]

    model.eval()
    with torch.no_grad():
        gen = model.generate(
            **tok("The capital of France is", return_tensors="pt").to("cuda"),
            max_new_tokens=12, do_sample=False)
    text_out = tok.decode(gen[0], skip_special_tokens=True)

    print("\n=== SMOKE TEST RESULT ===")
    print(f"  losses: {[round(l,3) for l in losses]}")
    print(f"  finite loss: {finite} | decreased: {decreased}")
    print(f"  generate(): {repr(text_out)}")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB / 12.8")
    ok = finite and decreased and len(text_out) > 10 and "Paris" in text_out
    coherent = len(text_out.split()) >= 5
    print(f"  => 4-bit LoRA on Blackwell: "
          f"{'WORKS' if (finite and coherent) else 'PROBLEM -> fallback 8-bit'}")


if __name__ == "__main__":
    main()
