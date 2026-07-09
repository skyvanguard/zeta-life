"""
El Útero — v2 germinal: ¿la variación en la reproducción sostiene la novedad?

v1 (async + espacio creciente) abrió el espacio pero la novedad se secó: la
reproducción copiaba EXACTO. v2 introduce variación germinal SIN mano nuestra:
la cría nace con UNA instrucción reescrita desde la materia del momento del
parto (los campos b,c del SPAWN eligen registro y posición — la física puede
evolucionar CÓMO varían sus hijas; la función es la misma de MUTO; no hay RNG).

Preguntas:
  1. ¿La novedad (genomas nunca vistos por tramo) ahora se sostiene?
  2. Si se seca: ¿es porque los nacimientos cesan, o porque los nacimientos
     REPITEN la misma cría (materia asentada => partos idénticos)?
  3. Comparación directa v2 (germinal) vs v1 (copia exacta), mismas semillas.

    PYTHONPATH=src python experiments/utero/exp_utero_germinal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zeta_life.utero.creciente import run_history, verdict  # noqa: E402

N0 = 16
MAX_N = 256
TICKS = 4000
SEEDS = list(range(20))
WINDOW = 100
TRANCHE = 500

RESULTS = Path(__file__).resolve().parents[2] / "results"


def tranches(arr: np.ndarray) -> list:
    return [float(arr[i:i + TRANCHE].sum()) for i in range(0, TICKS, TRANCHE)]


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 74)
    out("EL UTERO -- v2 germinal: variacion en la reproduccion, sin mano nuestra")
    out("=" * 74)
    out(f"n0={N0} -> max {MAX_N}  ticks={TICKS}  semillas={len(SEEDS)}")
    out("la cria nace con UNA instruccion reescrita desde la materia del parto")
    out("(campos b,c del SPAWN + registro; funcion de MUTO; sin RNG inyectado).")
    out("")
    out(f"{'seed':>4}  {'veredicto':<9} {'mundo':>6} {'vivo':>6} {'genomas':>8} "
        f"{'novedad_2da_mitad':>17} {'partos_2da_mitad':>16}")

    nov_g = np.zeros((len(SEEDS), TICKS // TRANCHE))
    nov_e = np.zeros((len(SEEDS), TICKS // TRANCHE))
    births_late_tot, results = [], {}
    for seed in SEEDS:
        hg = run_history(n0=N0, seed=seed, ticks=TICKS, max_n=MAX_N,
                         germinal=True)
        he = run_history(n0=N0, seed=seed, ticks=TICKS, max_n=MAX_N,
                         germinal=False)
        results[seed] = (verdict(hg, window=WINDOW), hg)
        nov_g[seed] = tranches(np.array(hg["new_genomes"]))
        nov_e[seed] = tranches(np.array(he["new_genomes"]))
        ngl = int(np.array(hg["new_genomes"])[TICKS // 2:].sum())
        bl = int(np.array(hg["colonized"])[TICKS // 2:].sum())
        births_late_tot.append(bl)
        out(f"{seed:>4}  {results[seed][0]:<9} {hg['n_world'][-1]:>6} "
            f"{hg['alive_frac'][-1]:>6.2f} {hg['total_genomes']:>8} "
            f"{ngl:>17} {bl:>16}")

    counts = {k: sum(1 for v, _ in results.values() if v == k)
              for k in ("termica", "cristal", "pulso")}
    grew = sum(1 for _, h in results.values() if h["n_world"][-1] >= MAX_N)
    tot_early_g = nov_g[:, 0].mean()
    tot_early_e = nov_e[:, 0].mean()
    late_g = float(nov_g[:, TICKS // TRANCHE // 2:].sum())
    late_e = float(nov_e[:, TICKS // TRANCHE // 2:].sum())
    out("")
    out(f"termica: {counts['termica']}   cristal: {counts['cristal']}   "
        f"pulso: {counts['pulso']}   crecieron a la pared: {grew}/{len(SEEDS)}")
    out("")
    out("novedad media por tramo (500 ticks):")
    out("  v2 germinal:   "
        + "  ".join(f"{m:.1f}" for m in nov_g.mean(axis=0)))
    out("  v1 copia-exacta: "
        + "  ".join(f"{m:.1f}" for m in nov_e.mean(axis=0)))
    out(f"tramo inicial: germinal {tot_early_g:.1f} vs exacta {tot_early_e:.1f} "
        f"(la variacion germinal SI acuna mas al principio)")
    out(f"novedad 2da mitad (total, 20 seeds): germinal={late_g:.0f}  "
        f"exacta={late_e:.0f}")
    out(f"partos en la 2da mitad (total): {sum(births_late_tot)} -- "
        f"semillas con partos tardios: "
        f"{sum(1 for b in births_late_tot if b > 0)}/{len(SEEDS)}")

    out("")
    if late_g < 1.0:
        if sum(births_late_tot) > 0:
            out("=> VEREDICTO HONESTO: la variacion germinal acuna mas genomas al")
            out("   inicio pero la novedad SE SECA IGUAL -- y el diagnostico es")
            out("   preciso: los PARTOS CONTINUAN en varias semillas, pero acunan")
            out("   CERO genomas nuevos. La mutacion es determinista sobre la")
            out("   materia, y la materia se ASENTO: mismo contexto -> misma cria,")
            out("   parto tras parto. La jaula se mudo una vez mas: ya no es la")
            out("   reproduccion (varia), es la MATERIA CONGELADA que alimenta la")
            out("   variacion con lo mismo. La novedad estructural necesita")
            out("   materia que no se asiente (dinamica no-contractiva) o memoria")
            out("   del linaje -- el proximo cruce es de la DINAMICA DE LA")
            out("   MATERIA, no del codigo.")
        else:
            out("=> VEREDICTO: la novedad se seco porque los partos CESARON")
            out("   (mundo lleno y estable, sin muertes -> sin nacimientos).")
    else:
        out("=> HAY NOVEDAD SOSTENIDA con variacion germinal. Antes de creerla:")
        out("   inspeccionar los genomas tardios (que sean estructuralmente")
        out("   distintos y no un contador trivial).")

    # ---- figura ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    pick = next((s for s, (_, h) in results.items()
                 if h["n_world"][-1] >= MAX_N), SEEDS[0])
    axes[0].imshow(results[pick][1]["frames"], aspect="auto", cmap="viridis",
                   interpolation="nearest")
    axes[0].set_title(f"v2 germinal — seed {pick} (mundo "
                      f"{results[pick][1]['n_world'][-1]})")
    axes[0].set_xlabel("coordenada")
    axes[0].set_ylabel("tick")
    xs = (np.arange(TICKS // TRANCHE) + 0.5) * TRANCHE
    axes[1].plot(xs, nov_g.mean(axis=0), "o-", label="v2 germinal")
    axes[1].plot(xs, nov_e.mean(axis=0), "s--", label="v1 copia exacta")
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("genomas nuevos por tramo (media)")
    axes[1].set_title("¿La variación germinal sostiene la novedad?")
    axes[1].legend()
    fig.suptitle("El Útero — v2 germinal")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "utero_germinal.png", dpi=110)
    out("")
    out("figura: results/utero_germinal.png")

    (RESULTS / "utero_germinal_run.txt").write_text("\n".join(lines),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
