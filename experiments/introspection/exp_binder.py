"""
Binder privileged-access test over Psi_act
==========================================

Question (accuracy criterion): does Yvyra's self-report carry information about
her own Psi_act BEYOND what an external reader of her text can infer? If adding
the self-report (felt) to a text-only predictor does NOT improve prediction of
Psi_act, there is no privileged access -- the link is common-cause (the text
already contains it).

For each Psi_act metric:
  R2_text   -- cross-validated R^2 predicting the metric from text embeddings (PCA)
  R2_text+felt -- same, with the self-report added as a feature
  delta     -- improvement from adding felt (>0 and CI-positive => privileged info)
  corr(felt, metric), corr(sentiment, metric) for reference

Honest caveat: with ~33 ticks this is low-powered; treat as a first look. Run more
ticks (repeat prompts with sampling) to strengthen it. Given the concept-injection
negative, a common-cause result (delta ~ 0) is expected.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_binder.py --log results/introspection_ticks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PSI_KEYS = ("participation_ratio", "phi_proxy", "interlayer_coherence",
            "trajectory_predictability")
RESULTS = Path(__file__).resolve().parents[2] / "results"


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def embed(texts: list[str]) -> np.ndarray:
    """MiniLM sentence embeddings (mean-pooled) via transformers."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name)
    mdl.eval()
    embs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=256)
            out = mdl(**enc).last_hidden_state[0]          # [T, D]
            mask = enc.attention_mask[0].unsqueeze(-1)
            v = (out * mask).sum(0) / mask.sum().clamp(min=1)
            embs.append(v.numpy())
    return np.array(embs)


def cv_r2(X: np.ndarray, y: np.ndarray, n_comp: int, folds: int = 5) -> float:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if y.std() < 1e-9:
        return float("nan")
    k = min(n_comp, X.shape[1], max(2, len(y) - len(y) // folds - 1))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k), Ridge(alpha=1.0))
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    scores = []
    for tr, te in kf.split(X):
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        ss_res = ((y[te] - pred) ** 2).sum()
        ss_tot = ((y[te] - y[tr].mean()) ** 2).sum()
        scores.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
    return float(np.mean(scores))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/introspection_ticks.jsonl")
    ap.add_argument("--n-comp", type=int, default=8)
    args = ap.parse_args()

    recs = [r for r in load(args.log) if r.get("felt") is not None]
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 74)
    out("BINDER PRIVILEGED-ACCESS TEST over Psi_act")
    out("=" * 74)
    out(f"ticks with a self-report (felt): {len(recs)}")
    if len(recs) < 12:
        out("too few ticks for a meaningful test (need >= ~12). Collect more.")
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "binder_run.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    felt = np.array([float(r["felt"]) for r in recs])
    sent = np.array([float(r["sentiment"]) for r in recs])
    out("embedding reflections (MiniLM)...")
    E = embed([r["reflection"] for r in recs])
    E_felt = np.concatenate([E, felt[:, None]], axis=1)

    out(f"{'metric':>26} {'R2_text':>9} {'R2_+felt':>9} {'delta':>7} "
        f"{'r(felt)':>8} {'r(sent)':>8}")
    privileged = []
    for k in PSI_KEYS:
        y = np.array([float(r["psi_act"][k]) for r in recs])
        r2_t = cv_r2(E, y, args.n_comp)
        r2_tf = cv_r2(E_felt, y, args.n_comp)
        delta = r2_tf - r2_t
        rf, rs = pearson(felt, y), pearson(sent, y)
        if delta > 0.05:
            privileged.append(k)
        out(f"{k:>26} {r2_t:>9.3f} {r2_tf:>9.3f} {delta:>+7.3f} {rf:>+8.3f} {rs:>+8.3f}")
    out("")
    out("VERDICT")
    if privileged:
        out(f"  felt adds predictive power for: {privileged}")
        out("  => some PRIVILEGED ACCESS: the self-report carries Psi_act info beyond the text.")
    else:
        out("  felt adds no predictive power over the text for any metric.")
        out("  => COMMON CAUSE: no privileged access. The self-report is inferable from the")
        out("     text; consistent with the concept-injection negative.")
    out("  (low-powered at this N; treat as first look)")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "binder_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
