"""
El Útero — Nivel 2: ¿la física que reescribe su FORMA escapa del cristal?

El Nivel 1 latió pero tendió al atractor-cristal (17/20 congeladas en cámara
lenta a 5000 ticks). El Nivel 2 es la apuesta del boceto: reglas-PROGRAMA que
reescriben su propia forma (MUTO/COPY) y colonizan el vacío escribiendo su
código (SPAWN, literal). Sin ruido inyectado: la variación es sólo la sopa
inicial + la dinámica del código.

Preguntas (describir, no premiar):
  1. ¿Sobrevive algo a la extinción inicial (los ciegos a la materia mueren)?
  2. ¿El cambio de código se SOSTIENE a largo plazo (anti-cristal), o el
     Nivel 2 también se congela?
  3. ¿SPAWN se enriquece en el código vivo (selección sin recompensa)?
  4. ¿La diversidad de genomas persiste o colapsa a un monocultivo?

    PYTHONPATH=src python experiments/utero/exp_nivel2_latido.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zeta_life.utero.nivel2 import (SPAWN, run_history, verdict,  # noqa: E402
                                    FROZEN_VALUE_EPS, THERMAL_ALIVE_FRAC)

N = 64
TICKS = 500
LONG = 5000
SEEDS = list(range(20))
WINDOW = 100

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 72)
    out("EL UTERO -- Nivel 2: reglas-programa que reescriben su propia forma")
    out("=" * 72)
    out(f"anillo N={N}  ticks={TICKS}  semillas={len(SEEDS)}  ventana={WINDOW}")
    out("manos visibles: sonda ciega-a-la-materia, orden sembrado de SPAWN,")
    out("vacio frio. SIN ruido inyectado (menos manos que el Nivel 1).")
    out("")
    out(f"{'seed':>4}  {'veredicto':<9} {'vivo_fin':>8} {'d_codigo_fin':>13} "
        f"{'coloniz':>8} {'divers':>7} {'spawn_ini':>9} {'spawn_fin':>9}")

    results = {}
    for seed in SEEDS:
        hist = run_history(n=N, seed=seed, ticks=TICKS)
        v = verdict(hist, window=WINDOW)
        results[seed] = (v, hist)
        af = np.mean(hist["alive_frac"][-WINDOW:])
        cc = np.mean(hist["code_change"][-WINDOW:])
        col = int(np.sum(hist["colonized"]))
        div = hist["diversity"][-1]
        s0 = hist["op_hist_init"][SPAWN]
        s1 = hist["op_hist_final"][SPAWN]
        out(f"{seed:>4}  {v:<9} {af:>8.3f} {cc:>13.3e} {col:>8} {div:>7} "
            f"{s0:>9.3f} {s1:>9.3f}")

    counts = {k: sum(1 for v, _ in results.values() if v == k)
              for k in ("termica", "cristal", "pulso")}
    ext = np.mean([r[1]["alive_frac"][0] for r in results.values()])
    rec = np.mean([np.mean(r[1]["alive_frac"][-WINDOW:]) for r in results.values()])
    d_spawn = [float(r[1]["op_hist_final"][SPAWN] - r[1]["op_hist_init"][SPAWN])
               for r in results.values()]
    out("")
    out(f"muerte termica: {counts['termica']}/{len(SEEDS)}   "
        f"muerte cristal: {counts['cristal']}/{len(SEEDS)}   "
        f"PULSO: {counts['pulso']}/{len(SEEDS)}")
    out(f"extincion inicial media (tick 1): vivo={ext:.2f} -> "
        f"recuperacion final: vivo={rec:.2f} (via SPAWN literal)")
    out(f"enriquecimiento de SPAWN en codigo vivo: {np.mean(d_spawn):+.4f} medio "
        f"({sum(1 for d in d_spawn if d > 0)}/{len(SEEDS)} semillas al alza)")

    # ---- control: horizonte 10x -- la pregunta anti-cristal ----
    out("")
    out("-" * 72)
    out(f"CONTROL -- horizonte {LONG} ticks: el cambio de codigo se sostiene?")
    out("(la razon de ser del Nivel 2: el Nivel 1 se congelaba aqui)")
    cb = {"termica": 0, "cristal": 0, "pulso": 0}
    macro = 0
    traj = {t: [] for t in (500, 1500, 3000, 5000)}
    div_fin, col_late = [], []
    for seed in SEEDS:
        h = run_history(n=N, seed=seed, ticks=LONG)
        cb[verdict(h, window=WINDOW)] += 1
        cc = np.array(h["code_change"])
        for t in traj:
            traj[t].append(cc[t - WINDOW:t].mean())
        if traj[LONG][-1] > 1e-4:
            macro += 1
        div_fin.append(h["diversity"][-1])
        col_late.append(int(np.sum(h["colonized"][-1000:])))
    out(f"  veredicto formal: termica={cb['termica']}  cristal={cb['cristal']}  "
        f"pulso={cb['pulso']}")
    out("  d_codigo medio por tramo: "
        + "  ".join(f"t~{t}: {np.mean(v):.2e}" for t, v in traj.items()))
    out(f"  semillas con cambio de codigo MACROSCOPICO (>1e-4) en t={LONG}: "
        f"{macro}/{len(SEEDS)}")
    out(f"  diversidad final media: {np.mean(div_fin):.1f} genomas distintos; "
        f"colonizaciones en los ultimos 1000 ticks (media): {np.mean(col_late):.1f}")

    # ---- control anti-ciclo: pulso real o cristal disfrazado? ----
    out("")
    out("CONTROL ANTI-CICLO -- estados distintos en los ultimos 1000 ticks de")
    out("5000 (pocos estados = ciclo limite = una jaula dinamica, no novedad):")
    from zeta_life.utero.nivel2 import UteroNivel2
    uniq = []
    for seed in SEEDS:
        u = UteroNivel2(n=N, seed=seed)
        for _ in range(LONG - 1000):
            u.step()
        seen = set()
        for _ in range(1000):
            u.step()
            seen.add((u.v.tobytes(), u.code.tobytes(), u.alive.tobytes()))
        uniq.append(len(seen))
    uniq = np.array(uniq)
    out(f"  estados distintos por semilla: {uniq.tolist()}")
    out(f"  en ciclo corto (<50 estados): {(uniq < 50).sum()}/{len(SEEDS)}   "
        f"aperiodicas (>900): {(uniq > 900).sum()}/{len(SEEDS)}")
    out("")
    if (uniq < 50).all():
        out("  => VEREDICTO HONESTO: el 'cambio sostenido' es un CICLO LIMITE --")
        out("     la fisica reescribe y des-reescribe lo mismo para siempre (hasta")
        out("     la muerte/re-colonizacion entra en el bucle). El Nivel 2 v0")
        out("     encuentra atractores mas ricos que el Nivel 1 (ciclos con")
        out("     reescritura de codigo y ecologia de muerte-renacimiento) pero")
        out("     sigue siendo un atractor-jaula. La novedad perpetua no emergio;")
        out("     el problema abierto muestra los dientes tal como el boceto")
        out("     advirtio ('aqui casi todo colapsa').")
    else:
        out("  => hay semillas aperiodicas: cambio genuinamente no-ciclico.")

    # ---- figura ----
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
            ax.imshow(results[seed][1]["frames"], aspect="auto", cmap="viridis",
                      interpolation="nearest")
            ax.set_title(f"seed {seed} — {kind}")
            ax.set_xlabel("celda")
            ax.set_ylabel("tick")
        fig.suptitle("El Útero Nivel 2 — materia v (blanco = vacío)")
        fig.tight_layout()
        RESULTS.mkdir(exist_ok=True)
        fig.savefig(RESULTS / "utero_nivel2_latido.png", dpi=110)
        out("")
        out(f"figura: results/utero_nivel2_latido.png "
            f"({', '.join(f'seed {s}={k}' for s, k in picks)})")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "utero_nivel2_run.txt").write_text("\n".join(lines),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
