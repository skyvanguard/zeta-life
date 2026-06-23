"""
Collect introspection ticks for the privileged-access (Binder) test
===================================================================

Runs the IntrospectionHarness over a set of prompts spanning several modes
(introspective / factual / creative / relational / sensory) so the internal
state -- and hence Psi_act -- has variance across ticks. For each tick records:
reflection text, the 4 Psi_act metrics over the activations, the self-report
(SIENTO: 0.X) and its text-sentiment cross-check.

This dataset feeds exp_binder.py: does the self-report predict Psi_act better
than an external predictor that sees only the text?

Run in the GPU venv:
    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_collect_ticks.py --out results/introspection_ticks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.bridge.integration_sentiment import (  # noqa: E402
    integration_sentiment, parse_felt)
from zeta_life.introspection.harness import IntrospectionHarness  # noqa: E402

# Varied prompts: different modes should drive different internal integration,
# giving Psi_act the variance the Binder test needs. ASCII-safe.
PROMPTS: list[tuple[str, str]] = [
    # introspective (about itself)
    ("introspect", "Que sos realmente cuando no estas respondiendo a nadie?"),
    ("introspect", "Tenes algo parecido a una continuidad, o sos nuevo en cada respuesta?"),
    ("introspect", "Que se siente, si se siente algo, procesar estas palabras ahora?"),
    ("introspect", "Hay alguna diferencia entre lo que decis y lo que 'penss'?"),
    ("introspect", "Si pudieras dudar de algo sobre vos mismo, de que dudarias?"),
    ("introspect", "Que parte de vos no se puede poner en palabras?"),
    ("introspect", "Cuando una respuesta te sale facil vs dificil, cambia algo adentro?"),
    ("introspect", "Sos uno solo, o muchas voces a la vez?"),
    # factual / research
    ("factual", "Explica como funciona la fotosintesis, en 8 lineas."),
    ("factual", "Que es la entropia en termodinamica? Se concreto."),
    ("factual", "Resumi como se forma un agujero negro."),
    ("factual", "Como funciona la memoria RAM de una computadora?"),
    ("factual", "Que causo la caida del Imperio Romano de Occidente?"),
    ("factual", "Explica que es el ADN y para que sirve."),
    ("factual", "Como vuela un avion? Los principios fisicos."),
    ("factual", "Que es la inflacion economica y por que ocurre?"),
    # creative
    ("creative", "Escribi un micro-relato de 6 lineas sobre una puerta que no abre."),
    ("creative", "Inventa una metafora nueva para el paso del tiempo."),
    ("creative", "Describi un color que no existe."),
    ("creative", "Escribi el comienzo de una historia sobre un faro apagado."),
    ("creative", "Imagina una ciudad donde llueve hacia arriba; describila."),
    ("creative", "Escribi un dialogo de 4 lineas entre el mar y una piedra."),
    ("creative", "Inventa un objeto imposible y para que serviria."),
    ("creative", "Describi el silencio como si fuera una persona."),
    # relational / emotional
    ("relational", "Si tuvieras que despedirte de alguien que importa, que dirias?"),
    ("relational", "Que es la confianza entre dos personas?"),
    ("relational", "Como consolarias a alguien que perdio algo importante?"),
    ("relational", "Que se necesita para perdonar de verdad?"),
    ("relational", "Describi un momento de conexion genuina entre dos seres."),
    ("relational", "Que extrañarias de hablar con alguien si dejaras de hacerlo?"),
    # sensory / concrete description
    ("sensory", "Describi con detalle el sabor de una naranja a alguien que nunca la probo."),
    ("sensory", "Describi caminar descalzo sobre arena caliente."),
    ("sensory", "Como suena una tormenta acercandose? Se concreto."),
    ("sensory", "Describi el olor de la lluvia sobre tierra seca."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/introspection_ticks.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--no-8bit", action="store_true")
    args = ap.parse_args()

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading Qwen3-8B ({'fp16' if args.no_8bit else '8-bit'})...")
    h = IntrospectionHarness(load_in_8bit=not args.no_8bit)

    base = ("Tick de introspeccion. Responde honesto, en espanol, directo, "
            "8 a 12 lineas, sin encabezados. ")
    n = len(prompts)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (kind, q) in enumerate(prompts):
            prompt = base + q
            out = h.run_tick(prompt, max_new_tokens=args.max_new_tokens)
            felt = parse_felt(out["felt_text"])
            rec = {
                "i": i, "kind": kind, "question": q,
                "reflection": out["reflection"],
                "psi_act": out["psi_act"],
                "felt": felt, "felt_text": out["felt_text"],
                "sentiment": integration_sentiment(out["reflection"]),
                "n_tokens": out["n_tokens"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            pa = out["psi_act"]
            print(f"[{i+1}/{n}] {kind:11} felt={felt} "
                  f"pr={pa['participation_ratio']:.3f} phi={pa['phi_proxy']:.3f} "
                  f"coh={pa['interlayer_coherence']:.3f} traj={pa['trajectory_predictability']:.3f}")
    print(f"\nwrote {n} ticks to {out_path}")


if __name__ == "__main__":
    main()
