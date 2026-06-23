"""
F2.3b -- LoRA fine-tune Qwen3-8B to self-report P(IK) (M1)
=========================================================

Trains the model to emit YES/NO ("will I answer correctly?") on the
answer-independent prompts from exp_pik_dataset. Loss only on the YES/NO token
(prompt masked). 4-bit QLoRA (the smoke-test config that works on Blackwell).
Saves the LoRA adapter.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_pik_train.py --epochs 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL = "Qwen/Qwen3-8B"


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/pik_train.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--adapter-dir", default="data/pik_lora")
    args = ap.parse_args()

    data = load_jsonl(args.train)
    print(f"train examples: {len(data)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()
    model.config.use_cache = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)

    def make(ex):
        full = ex["prompt"] + " " + ex["target"]
        ids = tok(full, return_tensors="pt").input_ids
        plen = tok(ex["prompt"], return_tensors="pt").input_ids.shape[1]
        labels = ids.clone()
        labels[0, :plen] = -100
        return ids.to("cuda"), labels.to("cuda")

    import random
    rng = random.Random(0)
    n = len(data)
    step = 0
    for epoch in range(args.epochs):
        order = list(range(n)); rng.shuffle(order)
        run = 0.0
        for k, idx in enumerate(order):
            ids, labels = make(data[idx])
            out = model(input_ids=ids, labels=labels)
            (out.loss / args.accum).backward()
            run += float(out.loss.item())
            step += 1
            if step % args.accum == 0:
                opt.step(); opt.zero_grad()
            if (k + 1) % 500 == 0:
                print(f"  epoch {epoch} [{k+1}/{n}] avg_loss={run/500:.4f} "
                      f"VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                run = 0.0
        opt.step(); opt.zero_grad()

    Path(args.adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    print(f"\nsaved LoRA adapter to {args.adapter_dir}")


if __name__ == "__main__":
    main()
