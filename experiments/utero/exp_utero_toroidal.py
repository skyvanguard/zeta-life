"""
El Útero — v3 toroidal: ¿la materia en un círculo desbloquea la novedad?

v2 afiló el diagnóstico: la variación germinal funciona pero la MATERIA se
congela (sigmoid + registros acotados => dinámica contractiva) y alimenta cada
parto con lo mismo. v3 ataca la causa raíz: v' = R3 mod 1 — la materia vive en
un TORO; el wrap permite mapas expansivos (ADD v,v->R3 es el doubling map).
Cambio de física, no de metas; más simple que la sigmoide. La sonda de muerte
usa separación irracional (0 y 0.618...) porque 0 y 1 son el mismo punto del
toro.

Preguntas:
  1. ¿La novedad tardía deja de ser cero? (en v1/v2 fue 0 en 40/40 corridas)
  2. En las semillas sostenidas: ¿la curva decae a cero o se estabiliza?
     (control de horizonte largo)
  3. ¿La materia sigue moviéndose a largo plazo? (d_materia tardío)

    PYTHONPATH=src python experiments/utero/exp_utero_toroidal.py
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
LONG = 12000
SEEDS = list(range(20))
WINDOW = 100
TRANCHE = 500

RESULTS = Path(__file__).resolve().parents[2] / "results"


def tranches(arr: np.ndarray, ticks: int) -> list:
    return [float(arr[i:i + TRANCHE].sum()) for i in range(0, ticks, TRANCHE)]


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 74)
    out("EL UTERO -- v3 toroidal: materia en un circulo (v' = R3 mod 1)")
    out("=" * 74)
    out(f"n0={N0} -> max {MAX_N}  ticks={TICKS}  semillas={len(SEEDS)}  "
        f"(germinal=True en ambos brazos)")
    out("manos declaradas: sonda con separacion irracional (0, 0.618) en el toro.")
    out("")
    out(f"{'seed':>4}  {'veredicto':<9} {'mundo':>6} {'vivo':>6} {'genomas':>8} "
        f"{'novedad_2da_mitad':>17} {'d_materia_fin':>13}")

    nov_t = np.zeros((len(SEEDS), TICKS // TRANCHE))
    nov_s = np.zeros((len(SEEDS), TICKS // TRANCHE))
    late_by_seed, results = {}, {}
    for seed in SEEDS:
        ht = run_history(n0=N0, seed=seed, ticks=TICKS, max_n=MAX_N,
                         germinal=True, toroidal=True)
        hs = run_history(n0=N0, seed=seed, ticks=TICKS, max_n=MAX_N,
                         germinal=True, toroidal=False)
        results[seed] = (verdict(ht, window=WINDOW), ht)
        nov_t[seed] = tranches(np.array(ht["new_genomes"]), TICKS)
        nov_s[seed] = tranches(np.array(hs["new_genomes"]), TICKS)
        late = int(np.array(ht["new_genomes"])[TICKS // 2:].sum())
        late_by_seed[seed] = late
        vc = float(np.mean(ht["value_change"][-WINDOW:]))
        out(f"{seed:>4}  {results[seed][0]:<9} {ht['n_world'][-1]:>6} "
            f"{ht['alive_frac'][-1]:>6.2f} {ht['total_genomes']:>8} "
            f"{late:>17} {vc:>13.4f}")

    sustained = [s for s, l in late_by_seed.items() if l > 100]
    total_late_t = sum(late_by_seed.values())
    total_late_s = float(nov_s[:, TICKS // TRANCHE // 2:].sum())
    out("")
    out("novedad media por tramo (500 ticks):")
    out("  v3 toroidal:      "
        + "  ".join(f"{m:.0f}" for m in nov_t.mean(axis=0)))
    out("  v2 sigmoid (base):"
        + "  ".join(f"{m:.1f}" for m in nov_s.mean(axis=0)))
    out(f"novedad 2da mitad (total 20 seeds): toroidal={total_late_t}  "
        f"sigmoid={total_late_s:.0f}")
    out(f"semillas con novedad tardia SOSTENIDA (>100): {len(sustained)}/20 "
        f"-> {sustained}")

    # ---- control de horizonte largo en las semillas sostenidas ----
    long_curves = {}
    if sustained:
        out("")
        out("-" * 74)
        out(f"CONTROL horizonte {LONG} ticks (semillas sostenidas): la curva")
        out("decae a cero o se estabiliza?")
        for seed in sustained[:3]:
            h = run_history(n0=N0, seed=seed, ticks=LONG, max_n=MAX_N,
                            germinal=True, toroidal=True)
            tr = tranches(np.array(h["new_genomes"]), LONG)
            long_curves[seed] = tr
            last4 = tr[-4:]
            out(f"  seed {seed}: genomas totales {h['total_genomes']}, "
                f"ultimos 4 tramos {[int(x) for x in last4]}, "
                f"d_materia fin {np.mean(h['value_change'][-WINDOW:]):.4f}")
        still = sum(1 for tr in long_curves.values() if tr[-1] > 10)
        out("")
        if still:
            out(f"  => {still}/{len(long_curves)} siguen acunando genomas nuevos")
            out(f"     a t={LONG}: PRIMERA encarnacion con novedad estructural")
            out("     sostenida a largo plazo. Cautela honesta: la curva decae;")
            out("     si es transitorio largo o regimen estable requiere mas")
            out("     horizonte. Y 'genomas nuevos' = combinaciones nunca vistas")
            out("     (vara estructural), pero su RIQUEZA funcional no esta")
            out("     evaluada -- ese es el proximo control.")
        else:
            out("  => todas decaen a ~0: el toro alarga el transitorio pero no")
            out("     sostiene novedad. Honesto: aun no.")
    else:
        out("")
        out("=> ninguna semilla sostiene novedad tardia: el toro no alcanzo.")

    # ---- figura ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    pick = sustained[0] if sustained else SEEDS[0]
    axes[0].imshow(results[pick][1]["frames"], aspect="auto", cmap="twilight",
                   interpolation="nearest")
    axes[0].set_title(f"v3 toroidal — seed {pick} (materia circular)")
    axes[0].set_xlabel("coordenada")
    axes[0].set_ylabel("tick")
    xs = (np.arange(TICKS // TRANCHE) + 0.5) * TRANCHE
    axes[1].plot(xs, nov_t.mean(axis=0), "o-", label="v3 toroidal")
    axes[1].plot(xs, nov_s.mean(axis=0), "s--", label="v2 sigmoid")
    for seed, tr in long_curves.items():
        xl = (np.arange(len(tr)) + 0.5) * TRANCHE
        axes[1].plot(xl, tr, ":", alpha=0.7, label=f"seed {seed} @ {LONG}")
    axes[1].set_yscale("symlog")
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("genomas nuevos por tramo")
    axes[1].set_title("¿El toro sostiene la novedad?")
    axes[1].legend(fontsize=8)
    fig.suptitle("El Útero — v3 materia toroidal")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "utero_toroidal.png", dpi=110)
    out("")
    out("figura: results/utero_toroidal.png")

    (RESULTS / "utero_toroidal_run.txt").write_text("\n".join(lines),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
