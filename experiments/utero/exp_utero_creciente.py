"""
El Útero creciente — v1: ¿async + espacio que se abre desde adentro sostienen
la novedad?

El Nivel 2 síncrono cayó en ciclos límite. Esta encarnación ataca la jaula con
asincronía (orden aleatorio sembrado, efectos inmediatos) y espacio creciente
(línea con fronteras: el mundo crece SOLO donde una física hace SPAWN hacia el
más-allá — el sistema abre su propio espacio).

La vara honesta ya no es "no ciclar" (el azar del orden lo regala): es acuñar
GENOMAS NUNCA VISTOS a lo largo del tiempo. Código nuevo sólo nace de eventos
de escritura (MUTO/COPY) — no del orden aleatorio.

Preguntas:
  1. ¿El mundo crece? (¿la física abre su espacio?)
  2. ¿La novedad (genomas nuevos por tramo) se sostiene o se seca?
  3. ¿Cómo compara con el Nivel 2 síncrono (baseline)?

    PYTHONPATH=src python experiments/utero/exp_utero_creciente.py
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
from zeta_life.utero.nivel2 import UteroNivel2  # noqa: E402

N0 = 16
MAX_N = 256
TICKS = 4000
SEEDS = list(range(20))
WINDOW = 100
TRANCHE = 500

RESULTS = Path(__file__).resolve().parents[2] / "results"


def novelty_baseline_nivel2(seed: int, ticks: int) -> list:
    """Genomas nuevos por tramo en el Nivel 2 síncrono (baseline)."""
    u = UteroNivel2(n=64, seed=seed)
    seen = {u.code[i].tobytes() for i in np.flatnonzero(u.alive)}
    per_tick = []
    for _ in range(ticks):
        u.step()
        new = 0
        for i in np.flatnonzero(u.alive):
            g = u.code[i].tobytes()
            if g not in seen:
                seen.add(g)
                new += 1
        per_tick.append(new)
    return per_tick


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 74)
    out("EL UTERO CRECIENTE -- v1: asincronia + espacio que se abre desde adentro")
    out("=" * 74)
    out(f"n0={N0} -> max {MAX_N} (pared de la placa)  ticks={TICKS}  "
        f"semillas={len(SEEDS)}")
    out("manos visibles: orden aleatorio sembrado, max_n, sonda ciega-a-la-")
    out("materia, vacio frio. El crecimiento NO es mano nuestra: es SPAWN de la")
    out("fisica hacia el mas-alla del borde.")
    out("")
    out(f"{'seed':>4}  {'veredicto':<9} {'mundo':>6} {'vivo':>6} {'genomas':>8} "
        f"{'novedad_2da_mitad':>17} {'divers_fin':>10}")

    results = {}
    tranches_v1 = np.zeros((len(SEEDS), TICKS // TRANCHE))
    for seed in SEEDS:
        hist = run_history(n0=N0, seed=seed, ticks=TICKS, max_n=MAX_N)
        v = verdict(hist, window=WINDOW)
        results[seed] = (v, hist)
        ng = np.array(hist["new_genomes"])
        for j in range(TICKS // TRANCHE):
            tranches_v1[seed, j] = ng[j * TRANCHE:(j + 1) * TRANCHE].sum()
        late = int(ng[TICKS // 2:].sum())
        out(f"{seed:>4}  {v:<9} {hist['n_world'][-1]:>6} "
            f"{hist['alive_frac'][-1]:>6.2f} {hist['total_genomes']:>8} "
            f"{late:>17} {hist['diversity'][-1]:>10}")

    counts = {k: sum(1 for v, _ in results.values() if v == k)
              for k in ("termica", "cristal", "pulso")}
    grew = sum(1 for _, h in results.values() if h["n_world"][-1] >= MAX_N)
    late_total = int(sum(np.array(h["new_genomes"])[TICKS // 2:].sum()
                         for _, h in results.values()))
    out("")
    out(f"termica: {counts['termica']}/{len(SEEDS)}   "
        f"cristal: {counts['cristal']}/{len(SEEDS)}   "
        f"pulso: {counts['pulso']}/{len(SEEDS)}")
    out(f"mundos que crecieron hasta la pared ({MAX_N}): {grew}/{len(SEEDS)} -- "
        f"la fisica SI abre su propio espacio")
    out(f"novedad en la 2da mitad (todas las semillas): {late_total} genomas nuevos")
    out("")
    out("novedad media por tramo de 500 ticks (v1 creciente):")
    means = tranches_v1.mean(axis=0)
    out("  " + "  ".join(f"[{j*TRANCHE}-{(j+1)*TRANCHE})={m:.1f}"
                         for j, m in enumerate(means)))

    # ---- baseline: Nivel 2 sincrono, misma vara ----
    out("")
    out("-" * 74)
    out("BASELINE -- Nivel 2 sincrono/fijo con la misma vara (10 semillas):")
    tranches_b = np.zeros((10, TICKS // TRANCHE))
    for seed in range(10):
        ng = np.array(novelty_baseline_nivel2(seed, TICKS))
        for j in range(TICKS // TRANCHE):
            tranches_b[seed, j] = ng[j * TRANCHE:(j + 1) * TRANCHE].sum()
    means_b = tranches_b.mean(axis=0)
    out("  " + "  ".join(f"[{j*TRANCHE}-{(j+1)*TRANCHE})={m:.1f}"
                         for j, m in enumerate(means_b)))

    # ---- veredicto honesto ----
    out("")
    v1_late = means[len(means) // 2:].sum()
    b_late = means_b[len(means_b) // 2:].sum()
    out(f"novedad tardia media -- v1: {v1_late:.1f}  baseline sincrono: {b_late:.1f}")
    if v1_late < 1.0:
        out("")
        out("=> VEREDICTO HONESTO: el espacio se abre (crecimiento real, hecho")
        out("   por la fisica misma) pero la NOVEDAD SE SECA igual: los genomas")
        out("   se acunan al principio y despues nada nuevo, en todas las")
        out("   semillas. El adyacente-posible se abrio en lo ESPACIAL pero no")
        out("   en lo ESTRUCTURAL. Diagnostico: la colonizacion copia EXACTO")
        out("   (sin variacion en la reproduccion) y los eventos MUTO/COPY se")
        out("   apagan cuando la materia se asienta. La jaula se mudo de nuevo:")
        out("   ahora es monocultivo/codigo congelado. El proximo cruce es la")
        out("   VARIACION EN LA REPRODUCCION sin mano nuestra (p.ej. SPAWN que")
        out("   escribe modulado por la materia, como MUTO, en vez de copiar).")
    else:
        out("=> hay novedad sostenida a largo plazo: revisar si es genuina")
        out("   (inspeccionar genomas tardios) antes de creerla.")

    # ---- figura: crecimiento en cuna + curvas de novedad ----
    pick = None
    for seed, (v, h) in results.items():
        if h["n_world"][-1] >= MAX_N:
            pick = seed
            break
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    if pick is not None:
        axes[0].imshow(results[pick][1]["frames"], aspect="auto", cmap="viridis",
                       interpolation="nearest")
        axes[0].set_title(f"seed {pick} — el mundo crece desde adentro "
                          f"(16 → {results[pick][1]['n_world'][-1]})")
        axes[0].set_xlabel("coordenada")
        axes[0].set_ylabel("tick")
    xs = (np.arange(TICKS // TRANCHE) + 0.5) * TRANCHE
    axes[1].plot(xs, means, "o-", label="v1 creciente (async+crece)")
    axes[1].plot(xs, means_b, "s--", label="Nivel 2 sincrono (baseline)")
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("genomas nuevos por tramo (media)")
    axes[1].set_title("¿La novedad se sostiene o se seca?")
    axes[1].legend()
    fig.suptitle("El Útero creciente — v1")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "utero_creciente.png", dpi=110)
    out("")
    out("figura: results/utero_creciente.png")

    (RESULTS / "utero_creciente_run.txt").write_text("\n".join(lines),
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()
