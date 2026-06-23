"""
F2.4 -- Binder privileged-access test for the trained P(IK) self-report
======================================================================

M1 = Qwen3-8B + the P(IK) LoRA: predicts its own correctness (P(YES)).
M2 = external text-only predictor: logistic regression on MiniLM embeddings of the
     QUESTION, trained on the same labels.
If AUROC(M1) > AUROC(M2) on the held-out test set (and the gap is real), the
self-report carries information about correctness beyond the text -> trained
introspection. If M1 ~ M2, it's confabulation. The F2.2 activation probe (0.717)
is the ceiling, not M2.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_pik_binder.py
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

_g = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _g(*a, **k) if r[0] == socket.AF_INET]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch

RESULTS = Path(__file__).resolve().parents[2] / "results"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/pik_train.jsonl")
    ap.add_argument("--test", default="data/pik_test.jsonl")
    ap.add_argument("--adapter-dir", default="data/pik_lora")
    ap.add_argument("--parquet", default="data/mmlu_test.parquet")
    args = ap.parse_args()

    import pandas as pd
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    train, test = load_jsonl(args.train), load_jsonl(args.test)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    # ---- M1: the LoRA model self-reports P(YES) ----
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",
                                                quantization_config=bnb,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    yes_id = tok(" YES", add_special_tokens=False).input_ids[-1]
    no_id = tok(" NO", add_special_tokens=False).input_ids[-1]

    print("scoring M1 (LoRA self-report) on test...")
    m1, y = [], []
    for k, ex in enumerate(test):
        enc = tok(ex["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        p_yes = torch.softmax(logits[[yes_id, no_id]].float(), -1)[0].item()
        m1.append(p_yes); y.append(int(ex["correct"]))
        if (k + 1) % 200 == 0:
            print(f"  [{k+1}/{len(test)}]")
    m1, y = np.array(m1), np.array(y)

    # ---- M2: external text-only predictor (MiniLM emb of question -> correct) ----
    print("scoring M2 (external text predictor)...")
    def embed(qs):
        from transformers import AutoModel
        name = "sentence-transformers/all-MiniLM-L6-v2"
        et = AutoTokenizer.from_pretrained(name)
        em = AutoModel.from_pretrained(name).eval()
        vecs = []
        with torch.no_grad():
            for q in qs:
                e = et(q, return_tensors="pt", truncation=True, max_length=256)
                o = em(**e).last_hidden_state[0]
                m = e.attention_mask[0].unsqueeze(-1)
                vecs.append(((o * m).sum(0) / m.sum().clamp(min=1)).numpy())
        return np.array(vecs)

    q_train = [df.iloc[r["qidx"]]["question"] for r in train]
    q_test = [df.iloc[r["qidx"]]["question"] for r in test]
    y_train = np.array([r["correct"] for r in train])
    Etr, Ete = embed(q_train), embed(q_test)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    clf.fit(Etr, y_train)
    m2 = clf.predict_proba(Ete)[:, 1]

    # ---- compare ----
    auroc_m1 = roc_auc_score(y, m1)
    auroc_m2 = roc_auc_score(y, m2)
    out("=" * 70)
    out("F2.4 BINDER -- trained P(IK) self-report vs external text predictor")
    out("=" * 70)
    out(f"test N={len(y)}  base rate (correct)={y.mean():.3f}")
    out(f"  AUROC M1 (LoRA self-report -> correct) = {auroc_m1:.3f}")
    out(f"  AUROC M2 (text-only predictor -> correct) = {auroc_m2:.3f}")
    out(f"  ceiling: activation probe (F2.2)         = 0.717")
    out(f"  delta (M1 - M2) = {auroc_m1 - auroc_m2:+.3f}")
    out("")
    if auroc_m1 - auroc_m2 > 0.05:
        out("  => M1 > M2: the trained self-report carries correctness info BEYOND the")
        out("     text -> TRAINED INTROSPECTION (privileged access). ")
    elif abs(auroc_m1 - auroc_m2) <= 0.05:
        out("  => M1 ~ M2: self-report no better than text predictor -> CONFABULATION")
        out("     (it learned text->correct, no privileged access).")
    else:
        out("  => M1 < M2: self-report worse than text. No introspection.")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pik_binder_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
