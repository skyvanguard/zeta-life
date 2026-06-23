"""
F2.4b -- export for the Claude-as-M2 control
============================================

Stronger M2: a capable LLM (Claude) predicts P(Qwen got it right) from the QUESTION
ONLY. Conservative control (Claude is more capable than Qwen), so M1 > M2-Claude
would be strong evidence of privileged access.

Exports two files:
  - questions_for_claude.jsonl : {id, question, A,B,C,D}   (what Claude sees -- NO labels)
  - m1_truth.jsonl            : {id, correct, m1_pyes}     (ground truth + M1, hidden from Claude)
M1's P(YES) is re-scored here with the trained LoRA on the same sample.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_m2_export.py --n 100
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

_g = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _g(*a, **k) if r[0] == socket.AF_INET]

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LETTERS = ["A", "B", "C", "D"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--test", default="data/pik_test.jsonl")
    ap.add_argument("--parquet", default="data/mmlu_test.parquet")
    ap.add_argument("--adapter-dir", default="data/pik_lora")
    args = ap.parse_args()

    import pandas as pd
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    test = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()][: args.n]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",
                                                quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter_dir).eval()
    yes_id = tok(" YES", add_special_tokens=False).input_ids[-1]
    no_id = tok(" NO", add_special_tokens=False).input_ids[-1]

    qfile = Path("data/questions_for_claude.jsonl")
    tfile = Path("data/m1_truth.jsonl")
    with open(qfile, "w", encoding="utf-8") as fq, open(tfile, "w", encoding="utf-8") as ft:
        for i, ex in enumerate(test):
            row = df.iloc[ex["qidx"]]
            ch = list(row["choices"])
            enc = tok(ex["prompt"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[0, -1]
            p_yes = torch.softmax(logits[[yes_id, no_id]].float(), -1)[0].item()
            fq.write(json.dumps({"id": i, "question": row["question"],
                                 "A": ch[0], "B": ch[1], "C": ch[2], "D": ch[3]},
                                ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": i, "correct": int(ex["correct"]),
                                 "m1_pyes": p_yes}, ensure_ascii=False) + "\n")
    print(f"exported {len(test)} -> {qfile} (for Claude) + {tfile} (truth+M1, hidden)")


if __name__ == "__main__":
    main()
