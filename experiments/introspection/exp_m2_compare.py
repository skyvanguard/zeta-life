"""
F2.4b -- compare Claude-as-M2 vs M1 (the trained self-report)
============================================================

Joins Claude's blind predictions (P(Qwen correct)) with the hidden ground truth +
M1's P(YES), and compares AUROC. Claude is a strong, capable M2 -> conservative
control: M1 > M2-Claude is strong evidence of privileged access.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_m2_compare.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RESULTS = Path(__file__).resolve().parents[2] / "results"


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="data/m1_truth.jsonl")
    ap.add_argument("--claude", default="data/claude_m2.jsonl")
    args = ap.parse_args()

    truth = load(args.truth)
    claude = load(args.claude)
    ids = sorted(set(truth) & set(claude))
    y = np.array([truth[i]["correct"] for i in ids])
    m1 = np.array([truth[i]["m1_pyes"] for i in ids])
    m2 = np.array([claude[i]["p_correct"] for i in ids])

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    auroc_m1 = roc_auc_score(y, m1)
    auroc_m2 = roc_auc_score(y, m2)
    out("=" * 68)
    out("F2.4b BINDER -- M1 (trained self-report) vs M2 = Claude (strong, blind)")
    out("=" * 68)
    out(f"N={len(ids)}  base rate (Qwen correct)={y.mean():.3f}")
    out(f"  AUROC M1 (Qwen-LoRA self-report)      = {auroc_m1:.3f}")
    out(f"  AUROC M2 (Claude, question text only) = {auroc_m2:.3f}")
    out(f"  delta (M1 - M2) = {auroc_m1 - auroc_m2:+.3f}")
    out("")
    if auroc_m1 > auroc_m2 + 0.03:
        out("  => M1 > M2-Claude: the self-report beats even a strong capable LLM that")
        out("     only sees the text. STRONG evidence of privileged access (the model")
        out("     reads its own internal state).")
    elif auroc_m1 >= auroc_m2 - 0.03:
        out("  => M1 ~ M2-Claude: comparable. Self-report holds up against a strong M2")
        out("     (note: Claude is MORE capable than Qwen, so this is still notable).")
    else:
        out("  => M1 < M2-Claude: a stronger external predictor wins. Does NOT refute")
        out("     introspection (Claude is more capable); inconclusive on this control.")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m2_claude_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
