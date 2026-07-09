"""
El Útero — v4: muerte por equilibrio. ¿El motor de novedad se vuelve
auto-reparable?

El control ruido-vs-función dejó UNA falla: la bomba de novedad no se regenera
tras la ablación (los colonizadores vienen de llanuras asentadas y re-congelan
la zona). v4 completa el principio 3 con la observación del propio boceto —
«cristal = muerto de pie»: una celda cuya materia queda quieta eq_window ticks
muere. Lo que deja de devenir, deja de ser. Asentarse = morir = reintentar:
la regeneración tiene mecanismo.

Preguntas:
  1. ¿El régimen sostenido se vuelve más típico que 1/20? ¿O la muerte por
     equilibrio extingue mundos enteros (cascada térmica)?
  2. LA DECISIVA: ¿la bomba se regenera ahora tras la ablación? (en v3: NO —
     383→70→…→3)
  3. ¿Aparece la jaula nueva "materia oscilante + código congelado"?

    PYTHONPATH=src python experiments/utero/exp_utero_motor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zeta_life.utero.creciente import UteroCreciente, run_history, verdict  # noqa: E402

N0 = 16
MAX_N = 256
TICKS = 4000
SEEDS = list(range(20))
WINDOW = 100
TRANCHE = 500
ABLATE_AT = 8000
ABL_TICKS = 12000
ACTIVE_WIN = 200

RESULTS = Path(__file__).resolve().parents[2] / "results"

# referencia commiteada: v3 seed 13 post-ablacion (utero_ruido_vs_funcion_run.txt)
V3_POST_ABLATION = [2367, 383, 70, 58, 68, 20, 15, 3]


def ablation_run(seed: int, muerte_eq: bool) -> list:
    """Correr con ablación de la zona activa en t=ABLATE_AT; devolver novedad
    por tick."""
    u = UteroCreciente(n0=N0, seed=seed, max_n=MAX_N, germinal=True,
                       toroidal=True, muerte_equilibrio=muerte_eq)
    last_code: dict = {}
    last_change: dict = {}
    new_per_tick = []
    for t in range(ABL_TICKS):
        u.step()
        cg = {i - u.left_grown: u.code[i].tobytes()
              for i in np.flatnonzero(u.alive)}
        for coord, g in cg.items():
            if coord in last_code and last_code[coord] != g:
                last_change[coord] = t
        last_code = cg
        new_per_tick.append(u._register_genomes())
        if t == ABLATE_AT:
            pump = [c for c, tc in last_change.items()
                    if t - tc <= ACTIVE_WIN]
            for coord in pump:
                i = coord + u.left_grown
                if 0 <= i < u.n and u.alive[i]:
                    u.alive[i] = False
                    u.v[i] = 0.0
                    u.code[i] = 0
                    u.eq_count[i] = 0
            last_code = {i - u.left_grown: u.code[i].tobytes()
                         for i in np.flatnonzero(u.alive)}
    return new_per_tick


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 74)
    out("EL UTERO -- v4: muerte por equilibrio (cristal = muerto de pie)")
    out("=" * 74)
    out(f"n0={N0} -> max {MAX_N}  ticks={TICKS}  semillas={len(SEEDS)}  "
        f"(toroidal+germinal en ambos brazos)")
    out("manos nuevas declaradas: eq_eps=1e-9, eq_window=100.")
    out("")
    out(f"{'seed':>4}  {'veredicto':<9} {'mundo':>6} {'vivo':>6} {'genomas':>8} "
        f"{'novedad_2da_mitad':>17}")

    late_by_seed = {}
    counts = {"termica": 0, "cristal": 0, "pulso": 0}
    frozen_osc = 0
    for seed in SEEDS:
        h = run_history(n0=N0, seed=seed, ticks=TICKS, max_n=MAX_N,
                        germinal=True, toroidal=True, muerte_equilibrio=True)
        v = verdict(h, window=WINDOW)
        counts[v] += 1
        late = int(np.array(h["new_genomes"])[TICKS // 2:].sum())
        late_by_seed[seed] = late
        # jaula nueva: materia moviendose (vivo) pero codigo congelado
        cc = np.array(h["code_change"][-WINDOW:])
        if v == "pulso" and cc.max() == 0.0:
            frozen_osc += 1
        out(f"{seed:>4}  {v:<9} {h['n_world'][-1]:>6} "
            f"{h['alive_frac'][-1]:>6.2f} {h['total_genomes']:>8} {late:>17}")

    sustained = [s for s, l in late_by_seed.items() if l > 100]
    out("")
    out(f"termica: {counts['termica']}   cristal: {counts['cristal']}   "
        f"pulso: {counts['pulso']}")
    out(f"regimen sostenido (novedad tardia >100): {len(sustained)}/20 "
        f"-> {sustained}   (v3 era 1/20)")
    out(f"jaula nueva 'materia oscilante + codigo congelado': {frozen_osc}/20")

    # ---- LA PREGUNTA DECISIVA: regeneracion tras ablacion ----
    out("")
    out("-" * 74)
    out(f"ABLACION (t={ABLATE_AT}, zona activa de {ACTIVE_WIN} ticks) en las")
    out("semillas sostenidas -- ¿la bomba se regenera con muerte por equilibrio?")
    out(f"referencia v3 (seed 13, SIN muerte-eq): {V3_POST_ABLATION} -> murio")
    regen = {}
    for seed in (sustained[:3] if sustained else [13]):
        npt = np.array(ablation_run(seed, muerte_eq=True))
        post = [int(npt[i:i + TRANCHE].sum())
                for i in range(ABLATE_AT, ABL_TICKS, TRANCHE)]
        pre = [int(npt[i:i + TRANCHE].sum())
               for i in range(ABLATE_AT - 2000, ABLATE_AT, TRANCHE)]
        regen[seed] = (pre, post)
        out(f"  seed {seed}: pre {pre} | post {post}")

    ok = sum(1 for pre, post in regen.values()
             if sum(post[2:]) > 0.25 * sum(pre) and sum(post[-2:]) > 0)
    out("")
    if ok:
        out(f"  => {ok}/{len(regen)} REGENERAN: la novedad tardia post-ablacion")
        out("     se sostiene (>25% del nivel pre y viva al final). Con muerte")
        out("     por equilibrio, asentarse = morir = reintentar: el motor deja")
        out("     de ser un lugar fragil y pasa a ser la condicion del mundo.")
        out("     AUTO-REPARACION del regimen de novedad -- el 3er criterio que")
        out("     v3 fallo. Cautela: sigue siendo el regimen de pocas semillas.")
    else:
        out("  => NO regeneran: la muerte por equilibrio no basta para que el")
        out("     motor se auto-repare (o extingue el mundo antes). Honesto.")

    # ---- figura ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = (np.arange(len(V3_POST_ABLATION)) + 0.5) * TRANCHE + ABLATE_AT
    ax.plot(xs, V3_POST_ABLATION, "s--", color="gray",
            label="v3 (sin muerte-eq): murió")
    for seed, (pre, post) in regen.items():
        xp = (np.arange(len(post)) + 0.5) * TRANCHE + ABLATE_AT
        ax.plot(xp, post, "o-", label=f"v4 seed {seed}")
    ax.axvline(ABLATE_AT, color="r", ls=":", label="ablación")
    ax.set_yscale("symlog")
    ax.set_xlabel("tick")
    ax.set_ylabel("genomas nuevos por tramo")
    ax.set_title("El Útero v4 — ¿la bomba se regenera tras la ablación?")
    ax.legend()
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "utero_motor.png", dpi=110)
    out("")
    out("figura: results/utero_motor.png")

    (RESULTS / "utero_motor_run.txt").write_text("\n".join(lines),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
