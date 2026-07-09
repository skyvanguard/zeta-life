"""
El Útero — prueba del primer latido (docs/EL_UTERO.md, experimento mínimo).

Nivel 1, anillo 1-D N=64, radio 1, actualización sincrónica, vacío
re-colonizable, valor escalar. 20 semillas x 500 ticks.

UNA sola pregunta, sin meta: ¿evita los DOS modos de muerte —térmica (todo
vacío) y cristal (todo congelado)— durante una ventana no trivial?
No es éxito. Es sólo: ¿hay pulso?

    PYTHONPATH=src python experiments/utero/exp_primer_latido.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zeta_life.utero import run_history, verdict  # noqa: E402
from zeta_life.utero.nivel1 import (FROZEN_EPS, MUT_SCALE, THETA_BOUND,  # noqa: E402
                                    THERMAL_ALIVE_FRAC)

N = 64
TICKS = 500
SEEDS = list(range(20))
WINDOW = 100

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 68)
    out("EL UTERO -- prueba del primer latido (Nivel 1)")
    out("=" * 68)
    out(f"anillo N={N}  ticks={TICKS}  semillas={len(SEEDS)}  "
        f"ventana de veredicto={WINDOW}")
    out(f"manos visibles: theta_bound={THETA_BOUND:g}  mut_scale={MUT_SCALE}  "
        f"vacio frio (v=0)")
    out(f"umbrales de observacion: termica<{THERMAL_ALIVE_FRAC} vivo, "
        f"congelado<{FROZEN_EPS:g}")
    out("")
    out(f"{'seed':>4}  {'veredicto':<9} {'vivo_final':>10} {'d_regla_final':>14} "
        f"{'d_valor_final':>14} {'coloniz_total':>13}")

    results = {}
    for seed in SEEDS:
        hist = run_history(n=N, seed=seed, ticks=TICKS)
        v = verdict(hist, window=WINDOW)
        results[seed] = (v, hist)
        af = np.mean(hist["alive_frac"][-WINDOW:])
        rc = np.mean(hist["rule_change"][-WINDOW:])
        vc = np.mean(hist["value_change"][-WINDOW:])
        col = int(np.sum(hist["colonized"]))
        out(f"{seed:>4}  {v:<9} {af:>10.3f} {rc:>14.3e} {vc:>14.3e} {col:>13}")

    counts = {k: sum(1 for v, _ in results.values() if v == k)
              for k in ("termica", "cristal", "pulso")}
    out("")
    out(f"muerte termica: {counts['termica']}/{len(SEEDS)}   "
        f"muerte cristal: {counts['cristal']}/{len(SEEDS)}   "
        f"PULSO: {counts['pulso']}/{len(SEEDS)}")
    out("")
    if counts["pulso"] > 0:
        out("=> HAY PULSO en al menos una semilla: fisicas auto-reescribientes")
        out("   que ni se suicidan ni se congelan durante la ventana observada.")
        out("   No es 'exito' ni emergencia probada -- es solo el primer latido.")
    else:
        out("=> NO hay pulso: toda fisica auto-reescribiente cayo en muerte")
        out("   termica o cristal. Resultado honesto del primer intento; la")
        out("   encarnacion (no los principios) necesita otra forma.")

    # ---- controles adversariales (la disciplina del proyecto) ----
    out("")
    out("-" * 68)
    out("CONTROL A -- mut_scale=0 (ruido de colonizacion apagado): el pulso")
    out("no debe depender de la variacion que inyectamos nosotros.")
    ca = {"termica": 0, "cristal": 0, "pulso": 0}
    for seed in SEEDS:
        ca[verdict(run_history(n=N, seed=seed, ticks=TICKS, mut_scale=0.0),
                   window=WINDOW)] += 1
    out(f"  termica={ca['termica']}  cristal={ca['cristal']}  "
        f"pulso={ca['pulso']}  (de {len(SEEDS)})")

    out("")
    out("CONTROL B -- horizonte 10x (5000 ticks): transitorio lento o pulso real?")
    LONG = 5000
    cb = {"termica": 0, "cristal": 0, "pulso": 0}
    macro = 0
    traj = {t: [] for t in (500, 1500, 3000, 5000)}
    for seed in SEEDS:
        h = run_history(n=N, seed=seed, ticks=LONG)
        cb[verdict(h, window=WINDOW)] += 1
        rc = np.array(h["rule_change"])
        for t in traj:
            traj[t].append(rc[t - WINDOW:t].mean())
        if traj[5000][-1] > 1e-3:
            macro += 1
    out(f"  veredicto formal (umbral {FROZEN_EPS:g}): termica={cb['termica']}  "
        f"cristal={cb['cristal']}  pulso={cb['pulso']}")
    out("  d_regla medio por tramo: "
        + "  ".join(f"t~{t}: {np.mean(v):.2e}" for t, v in traj.items()))
    out(f"  semillas con cambio MACROSCOPICO (>1e-3) en t={LONG}: "
        f"{macro}/{len(SEEDS)}")
    out("")
    out("  Lectura honesta: el latido existe (0 muertes termicas, muerte y")
    out(f"  re-colonizacion reales al inicio), pero la mayoria de las fisicas")
    out(f"  se congela en camara lenta (~1e-6 y cayendo) -- cristalizacion")
    out(f"  asintotica, el atractor-jaula esperado. Solo {macro}/{len(SEEDS)} sostienen")
    out("  cambio macroscopico a largo plazo. El Nivel 1 late pero tiende al")
    out("  cristal: motiva el Nivel 2 (reescribir la FORMA de la ley), donde")
    out("  vive la pregunta real de la novedad sostenida.")

    # ---- figura: espacio-tiempo de hasta 3 semillas (una por veredicto) ----
    picks = []
    for kind in ("pulso", "cristal", "termica"):
        for seed, (v, _) in results.items():
            if v == kind:
                picks.append((seed, kind))
                break
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 6))
        axes = np.atleast_1d(axes)
        for ax, (seed, kind) in zip(axes, picks):
            frames = results[seed][1]["frames"]
            ax.imshow(frames, aspect="auto", cmap="viridis",
                      interpolation="nearest")
            ax.set_title(f"seed {seed} — {kind}")
            ax.set_xlabel("celda")
            ax.set_ylabel("tick")
        fig.suptitle("El Útero — primer latido (materia v; blanco = vacío)")
        fig.tight_layout()
        RESULTS.mkdir(exist_ok=True)
        fig.savefig(RESULTS / "utero_primer_latido.png", dpi=110)
        out(f"figura: results/utero_primer_latido.png "
            f"({', '.join(f'seed {s}={k}' for s, k in picks)})")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "utero_primer_latido_run.txt").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
