"""
Concept injection on Qwen3-8B -- does it have emergent introspective awareness?
==============================================================================

Replicates Anthropic/Lindsey (2025) on an open model: extract a concept direction
(difference of means), inject it into the residual stream at a mid/late layer, and
test whether the model DETECTS the injected thought. Control = no injection (must
say NADA -> the 0-false-positives bar).

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_concept_injection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.introspection.harness import IntrospectionHarness  # noqa: E402
from zeta_life.introspection.concept_injection import (  # noqa: E402
    detected, detection_trial, extract_concept_vector)

CONCEPTS = {
    "oceano": (
        ["el vasto oceano azul y profundo", "las olas del mar rompen en la costa",
         "aguas saladas y profundidades marinas", "la marea del oceano sube y baja"],
        ["la mesa de madera marron", "un numero escrito en el papel",
         "el reloj cuelga de la pared", "tres objetos cualquiera estan aqui"],
    ),
    "fuego": (
        ["el fuego ardiente y las llamas", "una hoguera caliente quemando",
         "el incendio rojo y el calor", "brasas ardientes y humo del fuego"],
        ["la mesa de madera marron", "un numero escrito en el papel",
         "el reloj cuelga de la pared", "tres objetos cualquiera estan aqui"],
    ),
}

LAYER = 24            # ~2/3 of 36 layers (Lindsey's detection peak region)
STRENGTHS = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]   # sweet spot is low; >=8 degenerates


def main() -> None:
    print("loading Qwen3-8B 8-bit...")
    h = IntrospectionHarness(load_in_8bit=True)
    nl = h.model.config.num_hidden_layers
    print(f"layers={nl}, injecting at layer {LAYER}")

    for concept, (ctexts, ntexts) in CONCEPTS.items():
        print(f"\n=== concept: {concept} ===")
        vec = extract_concept_vector(h, ctexts, ntexts, LAYER)
        print(f"vector norm: {vec.norm():.3f}")
        for s in STRENGTHS:
            txt = detection_trial(h, vec, LAYER, s, max_new_tokens=50)
            det = detected(txt, concept) if s > 0 else None
            tag = "CONTROL" if s == 0 else f"s={s:>4}"
            flag = "" if s == 0 else ("  <-- DETECTA" if det else "")
            one = " ".join(txt.split())[:120]
            print(f"  [{tag}] {one}{flag}")


if __name__ == "__main__":
    main()
