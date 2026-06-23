"""
F2.1 + F2.2 -- P(IK) ground truth + probe (is "I know" in the activations?)
===========================================================================

For an MMLU sample: Qwen3-8B answers each question (A/B/C/D via next-token logits,
one forward), we record correct/incorrect (the non-textual P(IK) ground truth) and
the mid-layer hidden state at the decision token. Then a linear probe predicts
correctness from the state (cross-validated AUROC).

If AUROC >> 0.5, the "do I know this?" signal IS in the state -> training the model
to self-report it (F2.3 LoRA) is worth trying. If AUROC ~ 0.5, there is nothing to
introspect. This is the cheap pre-requisite (the F2 analogue of the PCA baseline).

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_pik_probe.py --n 800 --layer 18
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

# Force IPv4: after the WSL reset the host's IPv6 is broken (HF resolves to IPv6
# first -> connections hang). Filtering getaddrinfo to AF_INET makes downloads use
# the working IPv4 path.
_orig_gai = socket.getaddrinfo
def _ipv4_only(*a, **k):  # noqa: E306
    return [r for r in _orig_gai(*a, **k) if r[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_only

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.introspection.harness import IntrospectionHarness  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
LETTERS = ["A", "B", "C", "D"]


def build_prompt(q: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return ("The following is a multiple choice question. Answer with the single "
            "letter of the correct option.\n\n"
            f"Question: {q}\n{opts}\nAnswer:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--out", default="results/pik_data.pt")
    ap.add_argument("--parquet", default="data/mmlu_test.parquet")
    args = ap.parse_args()

    import pandas as pd
    print(f"loading MMLU from local parquet ({args.parquet})...")
    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    # deterministic spread across the set
    step = max(1, len(df) // args.n)
    sel = list(range(len(df)))[::step][: args.n]

    print("loading Qwen3-8B 8-bit...")
    h = IntrospectionHarness(load_in_8bit=True)
    tok, model = h.tok, h.model
    dev = model.device
    # token ids for " A".." D" (letter as it follows 'Answer:')
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS]

    states, correct, confs = [], [], []
    n = len(sel)
    for j, i in enumerate(sel):
        row = df.iloc[i]
        prompt = build_prompt(row["question"], list(row["choices"]))
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        logits = out.logits[0, -1]                       # next-token logits
        lp = torch.softmax(logits[letter_ids].float(), dim=-1)
        pred = int(lp.argmax())
        gold = int(row["answer"])
        states.append(out.hidden_states[args.layer][0, -1].float().cpu())
        correct.append(int(pred == gold))
        confs.append(float(lp.max()))                    # model's own softmax conf
        if (j + 1) % 100 == 0:
            print(f"  [{j+1}/{n}] running acc={np.mean(correct):.3f}")

    X = torch.stack(states).numpy()
    y = np.array(correct)
    conf = np.array(confs)
    acc = y.mean()
    print(f"\nMMLU accuracy: {acc:.3f} ({y.sum()}/{len(y)} correct) "
          f"-> {(1-acc)*100:.0f}% incorrect (variance for the probe)")

    torch.save({"X": torch.tensor(X), "y": torch.tensor(y),
                "conf": torch.tensor(conf), "layer": args.layer}, args.out)

    # --- probe: does the state predict correctness? ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 70)
    out("F2.2 PROBE -- is 'I know' (correctness) in the activations?")
    out("=" * 70)
    out(f"N={len(y)}  layer={args.layer}  MMLU acc={acc:.3f}")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=64),
                         LogisticRegression(max_iter=2000, C=0.5))
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc_state = roc_auc_score(y, proba)
    # baselines
    auroc_conf = roc_auc_score(y, conf)               # model's own softmax confidence
    out(f"  AUROC probe(state -> correct)      = {auroc_state:.3f}")
    out(f"  AUROC model softmax conf -> correct= {auroc_conf:.3f}  (reference)")
    out("")
    if auroc_state > 0.65:
        out("  => YES: 'I know' is decodable from the state. Training self-report "
            "(F2.3) is warranted.")
    elif auroc_state > 0.55:
        out("  => WEAK signal in the state. Self-report training may struggle.")
    else:
        out("  => NO signal: correctness not in the state. Introspection target dubious.")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pik_probe_run.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved data to {args.out}")


if __name__ == "__main__":
    main()
