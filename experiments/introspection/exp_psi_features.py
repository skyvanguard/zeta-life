"""
Psi_act over SAE features (north cycle 2, Phase 1)
==================================================

Loads the Qwen-Scope TopK SAE (layer 18) and the captured activations, then:
  1) Picks the right (layer index, b_dec convention) by reconstruction
     explained-variance -- ALSO the Base-vs-Instruct alignment check: if EV is low,
     the SAE (trained on Qwen3-8B-Base) does not align with our activations and we
     must recapture with the Base model.
  2) Encodes activations to sparse features and recomputes Psi_act over FEATURES.
  3) Compares between-tick variance of Psi_act over features vs over raw (the raw
     ones were saturated: pr~0.004, coh~0.969 constant). Does the SAE de-saturate?

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_psi_features.py --acts results/acts.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.introspection.psi_act import (  # noqa: E402
    participation_ratio, phi_proxy, trajectory_predictability)

RESULTS = Path(__file__).resolve().parents[2] / "results"


def find_sae() -> Path:
    hits = list(Path.home().glob(
        ".cache/huggingface/hub/models--Qwen--SAE-Res-Qwen3-8B-Base-*/snapshots/*/layer18.sae.pt"))
    if not hits:
        raise FileNotFoundError("layer18.sae.pt not found in HF cache")
    return hits[0]


def topk_encode(x, W_enc, b_enc, b_dec, k, sub_bdec):
    """x [T,D] -> sparse features [T, S] keeping top-k per row (ReLU)."""
    pre = ((x - b_dec) if sub_bdec else x) @ W_enc.T + b_enc   # [T,S]
    pre = torch.relu(pre)
    vals, idx = pre.topk(k, dim=-1)
    z = torch.zeros_like(pre)
    z.scatter_(-1, idx, vals)
    return z


def ev(x, xhat) -> float:
    num = ((x - xhat) ** 2).sum()
    den = ((x - x.mean(0, keepdim=True)) ** 2).sum()
    return float(1 - num / den) if den > 1e-9 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="results/acts.pt")
    args = ap.parse_args()

    data = torch.load(args.acts, weights_only=False)
    per_token = data["per_token"]                 # {layer_idx: [ [T,D], ... ]}
    sae = torch.load(find_sae(), weights_only=True)
    W_enc, W_dec = sae["W_enc"].float(), sae["W_dec"].float()
    b_enc, b_dec = sae["b_enc"].float(), sae["b_dec"].float()
    k = 50
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    out("=" * 74)
    out("PSI_ACT OVER SAE FEATURES (Qwen-Scope layer 18, TopK k=50)")
    out("=" * 74)

    # 1) Pick layer index + convention by reconstruction EV (Base/Instruct check)
    out("[A] reconstruction explained-variance (alignment check)")
    best = None
    for layer_idx in sorted(per_token.keys()):
        sample = per_token[layer_idx][0].float()           # [T,D] first tick
        for sub in (True, False):
            z = topk_encode(sample, W_enc, b_enc, b_dec, k, sub)
            xhat = z @ W_dec.T + b_dec
            e = ev(sample, xhat)
            out(f"    hidden_states[{layer_idx}] sub_bdec={sub!s:5}: EV={e:+.3f}")
            if best is None or e > best[2]:
                best = (layer_idx, sub, e)
    layer_idx, sub, bev = best
    out(f"  -> best: hidden_states[{layer_idx}] sub_bdec={sub} (EV={bev:.3f})")
    if bev < 0.5:
        out("  WARNING: low EV -> SAE (Base) does NOT align with these (Instruct) "
            "activations. Recapture with Qwen3-8B-Base before trusting features.")
    out("")

    # 2) Psi_act over features (pr/phi/traj apply to single-layer features)
    out("[B] Psi_act over FEATURES vs raw (between-tick std)")
    feats_metrics = {"participation_ratio": [], "phi_proxy": [],
                     "trajectory_predictability": []}
    raw_metrics = {k2: [] for k2 in feats_metrics}
    for acts in per_token[layer_idx]:
        x = acts.float()
        z = topk_encode(x, W_enc, b_enc, b_dec, k, sub)      # [T,S]
        Hf = z.unsqueeze(0)                                   # [1,T,S]
        Hr = x.unsqueeze(0)                                  # [1,T,D]
        feats_metrics["participation_ratio"].append(participation_ratio(Hf))
        feats_metrics["phi_proxy"].append(phi_proxy(Hf))
        feats_metrics["trajectory_predictability"].append(trajectory_predictability(Hf))
        raw_metrics["participation_ratio"].append(participation_ratio(Hr))
        raw_metrics["phi_proxy"].append(phi_proxy(Hr))
        raw_metrics["trajectory_predictability"].append(trajectory_predictability(Hr))

    out(f"  {'metric':>26} {'raw mean':>9} {'raw std':>8} {'feat mean':>10} {'feat std':>9} {'x more var':>11}")
    for m in feats_metrics:
        rm, rs = np.mean(raw_metrics[m]), np.std(raw_metrics[m])
        fm, fs = np.mean(feats_metrics[m]), np.std(feats_metrics[m])
        ratio = fs / rs if rs > 1e-9 else float("inf")
        out(f"  {m:>26} {rm:>9.4f} {rs:>8.4f} {fm:>10.4f} {fs:>9.4f} {ratio:>11.1f}")
    out("")
    out("  (feat std >> raw std => the SAE de-saturates Psi_act: features carry the")
    out("   between-prompt variance the raw basis hid.)")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "psi_features_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
