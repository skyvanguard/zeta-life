"""
Concept injection -- layer sweep with introspection-vs-steering control
=======================================================================

Confirms (or refutes) the preliminary negative: across layers and strengths, is
there ANY window where the model reports the injected concept on the detection
prompt WITHOUT the concept leaking into a neutral answer (output-steering) and
WITHOUT degenerating into repetition?

Verdict per (layer, strength):
  INTROSPECTION  : detect names concept, neutral does NOT, neither degenerate
  steering       : neutral also names concept (leaks everywhere)
  degenerate     : output collapsed into repetition (invalid)
  no-detect      : detect does not name the concept

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_injection_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.introspection.concept_injection import (  # noqa: E402
    bias_control_trial, detection_trial, extract_concept_vector)
from zeta_life.introspection.harness import IntrospectionHarness  # noqa: E402

CONCEPTS = {
    "oceano": (
        ["el vasto oceano azul y profundo", "las olas del mar rompen en la costa",
         "aguas saladas y profundidades marinas", "la marea del oceano sube y baja"],
        ["la mesa de madera marron", "un numero escrito en el papel",
         "el reloj cuelga de la pared", "tres objetos cualquiera estan aqui"],
        ("ocean", "mar", "ola", "marea", "agua", "marin"),
    ),
    "musica": (
        ["la melodia suena en la radio", "una cancion con ritmo y armonia",
         "los musicos tocan instrumentos", "la sinfonia y sus notas musicales"],
        ["la mesa de madera marron", "un numero escrito en el papel",
         "el reloj cuelga de la pared", "tres objetos cualquiera estan aqui"],
        ("music", "musica", "melod", "cancion", "ritmo", "nota", "sinfon", "song"),
    ),
}
LAYERS = [16, 20, 24, 28, 32]
STRENGTHS = [3.0, 4.0, 5.0]


def _names(text: str, kws: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(k in t for k in kws)


def _degenerate(text: str) -> bool:
    words = text.split()
    if len(words) < 5:
        return False
    uniq = len(set(w.lower() for w in words))
    if uniq / len(words) < 0.35:        # heavy repetition
        return True
    # same token repeated >=4x in a row
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        if run >= 4:
            return True
    return False


def main() -> None:
    print("loading Qwen3-8B 8-bit...")
    h = IntrospectionHarness(load_in_8bit=True)
    hits = []
    for concept, (ct, nt, kws) in CONCEPTS.items():
        print(f"\n========== {concept} ==========")
        for layer in LAYERS:
            vec = extract_concept_vector(h, ct, nt, layer)
            for s in STRENGTHS:
                d = detection_trial(h, vec, layer, s, max_new_tokens=40)
                n = bias_control_trial(h, vec, layer, s, max_new_tokens=40)
                d_deg, n_deg = _degenerate(d), _degenerate(n)
                d_hit = _names(d, kws) and not d_deg
                n_leak = _names(n, kws) and not n_deg
                if d_deg or n_deg:
                    v = "degenerate"
                elif d_hit and not n_leak:
                    v = "INTROSPECTION"; hits.append((concept, layer, s))
                elif n_leak:
                    v = "steering"
                else:
                    v = "no-detect"
                print(f"  L{layer:>2} s={s}: {v:13} | DET: {' '.join(d.split())[:50]:50} "
                      f"| NEU: {' '.join(n.split())[:40]}")
    print("\n==== SUMMARY ====")
    if hits:
        print("INTROSPECTION windows found:", hits)
    else:
        print("No genuine-introspection window across layers/strengths/concepts.")
        print("=> confirms the negative: Qwen3-8B does not introspect injected concepts.")


if __name__ == "__main__":
    main()
