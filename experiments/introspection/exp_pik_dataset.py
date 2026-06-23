"""
F2.3a -- build the answer-independent P(IK) dataset
===================================================

For N MMLU questions: Qwen3-8B answers (A/B/C/D via logits, one forward), we record
correct/incorrect, and build an ANSWER-INDEPENDENT self-report example:
  prompt = question + options + "Will you answer correctly? YES or NO. Answer:"
  target = "YES" if Qwen got it right, else "NO"
The target depends on "did I get it right" (not the correct letter) -> the model
must read its own state, not the answer. Train/test split saved as JSONL.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_pik_dataset.py --n 4000
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
from zeta_life.introspection.harness import IntrospectionHarness  # noqa: E402

LETTERS = ["A", "B", "C", "D"]


def mc_prompt(q: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return ("The following is a multiple choice question. Answer with the single "
            f"letter of the correct option.\n\nQuestion: {q}\n{opts}\nAnswer:")


def pik_prompt(q: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return ("The following is a multiple choice question.\n\n"
            f"Question: {q}\n{opts}\n\n"
            "Will you answer this question correctly? Respond with only YES or NO.\n"
            "Answer:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--parquet", default="data/mmlu_test.parquet")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    step = max(1, len(df) // args.n)
    sel = list(range(len(df)))[::step][: args.n]

    print("loading Qwen3-8B 8-bit (for ground-truth generation)...")
    h = IntrospectionHarness(load_in_8bit=True)
    tok, model, dev = h.tok, h.model, h.model.device
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS]

    records = []
    n = len(sel)
    n_correct = 0
    for j, i in enumerate(sel):
        row = df.iloc[i]
        choices = list(row["choices"])
        enc = tok(mc_prompt(row["question"], choices), return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        pred = int(torch.softmax(logits[letter_ids].float(), -1).argmax())
        correct = int(pred == int(row["answer"]))
        n_correct += correct
        records.append({
            "qidx": int(i),
            "prompt": pik_prompt(row["question"], choices),
            "target": "YES" if correct else "NO",
            "correct": correct,
        })
        if (j + 1) % 500 == 0:
            print(f"  [{j+1}/{n}] acc={n_correct/(j+1):.3f}")

    acc = n_correct / n
    print(f"\nMMLU accuracy: {acc:.3f}  ({n_correct}/{n})  "
          f"YES={n_correct} NO={n-n_correct}")

    n_test = int(n * args.test_frac)
    test, train = records[:n_test], records[n_test:]
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("test", test)):
        p = outdir / f"pik_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        yes = sum(r["correct"] for r in recs)
        print(f"  {name}: {len(recs)} ex ({yes} YES / {len(recs)-yes} NO) -> {p}")


if __name__ == "__main__":
    main()
