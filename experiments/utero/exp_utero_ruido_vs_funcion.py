"""
El Útero — control ruido-vs-función sobre la seed 13 (v3 toroidal).

v3 dio la primera novedad sostenida (26k genomas a 12k ticks, curva subiendo).
Pero "genoma nuevo" es una vara ESTRUCTURAL: podría ser espuma caótica —
código que nace y muere en su celda sin dejar nada. Este control decide.

Vara definida ANTES de mirar:
  ESPUMA (ruido): cada genoma vive sólo en su celda natal, un lapso comparable
  al intervalo medio de reescritura de una celda (línea base medida del propio
  sistema), población máxima 1, y la ablación de la zona-bomba mata la novedad
  sin regeneración.
  FUNCIÓN: algún genoma tardío se PROPAGA (población >= 2) o PERSISTE >> línea
  base; y/o la bomba de novedad se REGENERA tras la ablación (el régimen se
  auto-mantiene — no depende de células particulares).

    PYTHONPATH=src python experiments/utero/exp_utero_ruido_vs_funcion.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zeta_life.utero.creciente import UteroCreciente  # noqa: E402

SEED = 13
TICKS = 12000
LATE_FROM = 6000          # "genoma tardío" = acuñado después de este tick
ABLATE_AT = 8000
ACTIVE_WIN = 200          # "zona-bomba" = código cambiado en esta ventana
TRANCHE = 500

RESULTS = Path(__file__).resolve().parents[2] / "results"


def coords_and_genomes(u: UteroCreciente) -> dict:
    """Mapa coordenada -> genoma (bytes) de las celdas vivas."""
    return {i - u.left_grown: u.code[i].tobytes()
            for i in np.flatnonzero(u.alive)}


def instrumented_run(ablate: bool):
    """Correr seed 13 registrando el censo de genomas (y ablación opcional)."""
    u = UteroCreciente(n0=16, seed=SEED, germinal=True, toroidal=True)
    reg: dict = {}            # genoma -> [first, last, max_pop, natal, spread]
    last_code: dict = {}      # coord -> genoma del tick anterior
    last_change: dict = {}    # coord -> último tick con cambio de código
    rewrite_events = 0
    rewrite_cells = 0
    new_per_tick = []
    ablated_cells = 0

    for t in range(TICKS):
        u.step()
        cg = coords_and_genomes(u)
        pop = Counter(cg.values())
        new = 0
        for coord, g in cg.items():
            if g not in reg:
                reg[g] = [t, t, pop[g], coord, {coord}]
                new += 1
            else:
                r = reg[g]
                r[1] = t
                r[2] = max(r[2], pop[g])
                r[4].add(coord)
            if coord in last_code and last_code[coord] != g:
                rewrite_events += 1
                last_change[coord] = t
            rewrite_cells += 1 if coord in last_code else 0
        last_code = cg
        new_per_tick.append(new)

        if ablate and t == ABLATE_AT:
            pump = [c for c, tc in last_change.items()
                    if t - tc <= ACTIVE_WIN]
            for coord in pump:
                i = coord + u.left_grown
                if 0 <= i < u.n and u.alive[i]:
                    u.alive[i] = False
                    u.v[i] = 0.0
                    u.code[i] = 0
                    ablated_cells += 1
            last_code = coords_and_genomes(u)

    rewrite_rate = rewrite_events / max(rewrite_cells, 1)
    return u, reg, new_per_tick, rewrite_rate, ablated_cells


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 74)
    out("EL UTERO -- control ruido-vs-funcion (seed 13, v3 toroidal)")
    out("=" * 74)

    # ---- corrida instrumentada (sin ablacion) ----
    u, reg, new_per_tick, rewrite_rate, _ = instrumented_run(ablate=False)
    interval = 1.0 / max(rewrite_rate, 1e-9)
    out(f"ticks={TICKS}  genomas totales={len(reg)}  "
        f"mundo final={u.n}  vivo final={u.alive.mean():.2f}")
    out(f"LINEA BASE espuma: tasa de reescritura por celda = {rewrite_rate:.3f}"
        f" -> intervalo medio ~{interval:.1f} ticks")
    out("")

    late = {g: r for g, r in reg.items() if r[0] >= LATE_FROM}
    lifespans = np.array([r[1] - r[0] for r in late.values()])
    maxpops = np.array([r[2] for r in late.values()])
    spreads = np.array([len(r[4]) for r in late.values()])
    out(f"GENOMAS TARDIOS (acunados en t>={LATE_FROM}): {len(late)}")
    out(f"  vida (ticks): mediana={np.median(lifespans):.0f}  "
        f"p90={np.percentile(lifespans, 90):.0f}  max={lifespans.max()}")
    out(f"  poblacion max: mediana={np.median(maxpops):.0f}  "
        f"max={maxpops.max()}  con pop>=2: "
        f"{(maxpops >= 2).sum()} ({100*(maxpops >= 2).mean():.1f}%)")
    out(f"  celdas visitadas: con >=2 celdas: {(spreads >= 2).sum()} "
        f"({100*(spreads >= 2).mean():.1f}%)  max={spreads.max()}")
    n_outlive = int((lifespans > 10 * interval).sum())
    out(f"  que viven >10x el intervalo de reescritura ({10*interval:.0f} "
        f"ticks): {n_outlive} ({100*n_outlive/max(len(late),1):.1f}%)")
    top = sorted(late.values(), key=lambda r: r[1] - r[0], reverse=True)[:5]
    out("  top-5 por vida: "
        + "; ".join(f"vida={r[1]-r[0]} pop={r[2]} celdas={len(r[4])}"
                    for r in top))

    # ---- ablacion de la zona-bomba ----
    out("")
    out("-" * 74)
    out(f"ABLACION en t={ABLATE_AT}: matar las celdas con codigo activo en los")
    out(f"ultimos {ACTIVE_WIN} ticks; seguir hasta t={TICKS}.")
    _, reg_a, new_a, _, ablated = instrumented_run(ablate=True)
    out(f"celdas ablacionadas: {ablated}")
    na = np.array(new_a)
    tr_pre = [int(na[i:i + TRANCHE].sum())
              for i in range(ABLATE_AT - 2000, ABLATE_AT, TRANCHE)]
    tr_post = [int(na[i:i + TRANCHE].sum())
               for i in range(ABLATE_AT, TICKS, TRANCHE)]
    out(f"novedad por tramo PRE-ablacion  (t={ABLATE_AT-2000}..{ABLATE_AT}): "
        f"{tr_pre}")
    out(f"novedad por tramo POST-ablacion (t={ABLATE_AT}..{TICKS}): {tr_post}")
    pump_back = sum(tr_post[2:]) > 0.25 * sum(tr_pre) if tr_pre else False

    # ---- veredicto (vara pre-definida) ----
    out("")
    spread_ok = (maxpops >= 2).mean() > 0.05 or (spreads >= 2).mean() > 0.10
    persist_ok = n_outlive / max(len(late), 1) > 0.05
    out("VEREDICTO (vara definida antes de mirar):")
    out(f"  propagacion (>=2 pop o celdas, umbral 5-10%): "
        f"{'SI' if spread_ok else 'NO'}")
    out(f"  persistencia (>10x linea base, umbral 5%): "
        f"{'SI' if persist_ok else 'NO'}")
    out(f"  regeneracion de la bomba tras ablacion: "
        f"{'SI' if pump_back else 'NO'}")
    hits = sum([spread_ok, persist_ok, pump_back])
    if hits >= 2:
        out("  => FUNCION: la novedad deja huella (se propaga/persiste) y/o el")
        out("     regimen se auto-mantiene. Mas que espuma caotica.")
    elif hits == 1:
        out("  => MIXTO: una sola sonda positiva -- senal debil, no concluyente.")
        out("     Honesto: no declarar funcion todavia.")
    else:
        out("  => RUIDO: los genomas tardios son espuma caotica -- nacen y")
        out("     mueren en su celda, nada se propaga, y la bomba no se")
        out("     regenera. La novedad sostenida de v3 es calor, no vida.")
        out("     Resultado honesto; la vara estructural no alcanza.")

    # ---- figura ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(np.clip(lifespans, 0, 500), bins=50)
    axes[0].axvline(interval, color="r", ls="--",
                    label=f"línea base espuma (~{interval:.0f})")
    axes[0].axvline(10 * interval, color="darkred", ls=":",
                    label="10x línea base")
    axes[0].set_xlabel("vida del genoma tardío (ticks, recortado a 500)")
    axes[0].set_ylabel("genomas")
    axes[0].set_title("¿Los genomas tardíos viven más que la espuma?")
    axes[0].legend()
    xs = (np.arange(len(na) // TRANCHE) + 0.5) * TRANCHE
    tr_all = [na[i:i + TRANCHE].sum() for i in range(0, TICKS, TRANCHE)]
    axes[1].plot(xs, tr_all, "o-")
    axes[1].axvline(ABLATE_AT, color="r", ls="--", label="ablación de la bomba")
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("genomas nuevos por tramo")
    axes[1].set_title("¿La bomba se regenera?")
    axes[1].legend()
    fig.suptitle("El Útero — ruido vs función (seed 13)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "utero_ruido_vs_funcion.png", dpi=110)
    out("")
    out("figura: results/utero_ruido_vs_funcion.png")

    (RESULTS / "utero_ruido_vs_funcion_run.txt").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
