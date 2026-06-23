"""
PCA baseline -- is there between-prompt variance in the raw activations?
=======================================================================

Reads captured token-mean hidden states and asks, per layer: do different prompts
produce different states, or is everything collapsed (saturated, like Psi_act)?
Also strips the dominant common direction (anisotropy) to see residual variance,
and whether states cluster by prompt kind.

If raw activations DO vary between prompts but Psi_act doesn't, the problem is the
METRIC (raw basis) -> motivates the SAE. If even the activations are collapsed,
that's a deeper issue.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_pca_baseline.py --acts results/acts.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="results/acts.pt")
    args = ap.parse_args()

    data = torch.load(args.acts, weights_only=False)
    TM = data["token_means"].numpy()           # [N, L+1, D]
    meta = data["meta"]
    N, Lp1, D = TM.shape
    kinds = [m["kind"] for m in meta]
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 74)
    out("PCA BASELINE -- between-prompt variance in raw token-mean activations")
    out("=" * 74)
    out(f"ticks N={N}, layers={Lp1} (incl. embeddings), dim={D}")
    out("")
    out(f"{'layer':>6} {'|mean|':>9} {'between-std':>11} {'ratio':>7} "
        f"{'PC1%':>6} {'PC1-3%':>7} {'kind-sep':>8}")

    probe_layers = sorted(set([0, Lp1 // 4, Lp1 // 2, (3 * Lp1) // 4, Lp1 - 1]))
    for l in probe_layers:
        X = TM[:, l, :]                          # [N, D]
        mean_norm = np.linalg.norm(X, axis=1).mean()
        Xc = X - X.mean(0, keepdims=True)
        between_std = np.sqrt((Xc ** 2).sum(1).mean())   # rms distance to centroid
        ratio = between_std / (mean_norm + 1e-9)
        # PCA via SVD on centred X
        s = np.linalg.svd(Xc, compute_uv=False)
        var = s ** 2
        pc1 = 100 * var[0] / var.sum() if var.sum() > 0 else 0
        pc13 = 100 * var[:3].sum() / var.sum() if var.sum() > 0 else 0
        # kind separability: between-kind var / total var on PC1-3 scores
        U = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T   # [N,3] scores
        sep = _kind_separability(U, kinds)
        out(f"{l:>6} {mean_norm:>9.1f} {between_std:>11.2f} {ratio:>7.3f} "
            f"{pc1:>6.1f} {pc13:>7.1f} {sep:>8.3f}")

    out("")
    out("READING")
    out("  ratio   = between-prompt spread / state norm. Near 0 => collapsed/saturated.")
    out("  PC1%    = variance in the top direction (high => dominated by 1 axis / anisotropy).")
    out("  kind-sep= between-kind / total variance on PC1-3 (high => states encode the task).")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pca_baseline_run.txt").write_text("\n".join(lines), encoding="utf-8")


def _kind_separability(U: np.ndarray, kinds: list[str]) -> float:
    uniq = sorted(set(kinds))
    grand = U.mean(0)
    ss_between = 0.0
    ss_total = ((U - grand) ** 2).sum()
    for k in uniq:
        idx = [i for i, kk in enumerate(kinds) if kk == k]
        if not idx:
            continue
        gm = U[idx].mean(0)
        ss_between += len(idx) * ((gm - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 1e-9 else 0.0


if __name__ == "__main__":
    main()
