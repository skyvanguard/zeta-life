"""
Capture raw activations for the SAE / PCA path (north cycle 2, Phase 1)
======================================================================

Re-runs the reflection prompts and saves, per tick:
  - per-layer token-mean hidden states  -> for the PCA baseline (does any layer
    show between-prompt variance, or is it saturated like Psi_act?)
  - per-token hidden states at the SAE layers (18, 19) -> to feed the Qwen-Scope
    SAE encoder and recompute Psi_act over sparse features.

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_capture_activations.py --out results/acts.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.introspection.harness import IntrospectionHarness  # noqa: E402
from exp_collect_ticks import PROMPTS  # noqa: E402  (same prompt set)

SAE_LAYERS = (18, 19)   # decoder-layer outputs near the SAE's mid layer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/acts.pt")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    base = ("Tick de introspeccion. Responde honesto, en espanol, directo, "
            "8 a 12 lineas, sin encabezados. ")

    print("loading Qwen3-8B 8-bit...")
    h = IntrospectionHarness(load_in_8bit=True)

    token_means = []          # [N, L+1, D]
    per_token = {l: [] for l in SAE_LAYERS}   # layer -> list of [T, D]
    meta = []
    n = len(prompts)
    for i, (kind, q) in enumerate(prompts):
        text, plen, full = h._generate(base + q, args.max_new_tokens, False)
        H = h._capture(full, plen)            # [L+1, T_gen, D] (cpu fp32)
        token_means.append(H.mean(dim=1))     # [L+1, D]
        for l in SAE_LAYERS:
            per_token[l].append(H[l].clone())  # [T_gen, D] (hidden_states[l])
        meta.append({"i": i, "kind": kind, "question": q,
                     "n_tokens": int(H.shape[1])})
        print(f"[{i+1}/{n}] {kind:11} tokens={H.shape[1]}")

    out = {
        "token_means": torch.stack(token_means),   # [N, L+1, D]
        "per_token": {l: per_token[l] for l in SAE_LAYERS},
        "meta": meta,
        "sae_layers": list(SAE_LAYERS),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\nsaved activations for {n} ticks to {args.out}")
    print(f"token_means shape: {tuple(out['token_means'].shape)}")


if __name__ == "__main__":
    main()
