"""
El Útero — Nivel 1: la física reescribe sus propios parámetros.

Encarna docs/EL_UTERO.md (los tres principios: reglas-como-estado, lazo cerrado
física↔física, persistencia como único filtro) en su versión "semilla segura":
la FORMA de la ley es fija; su CONTENIDO (theta) se auto-modifica.

Sustrato: anillo 1-D de N celdas. Cada celda viva carga materia y física:

    v_i      — valor escalar en (0,1)                        (materia)
    theta_i  — 8 parámetros [a1,a2,a3,b, m1,m2,m3,eta]       (física, mutable)

Paso sincrónico (el lazo cerrado):

    v'     = sigmoid( a1*v_izq + a2*v + a3*v_der + b )
    theta' = theta + eta * ( m1 * (theta_vecinas_media - theta)   # difusión de física
                           + m2 * lap_v * theta                   # la materia modula la física
                           + m3 * theta )                         # auto-amplificación/decaimiento

theta COMPLETO se reescribe con su propia fórmula — los meta-parámetros
(m1,m2,m3,eta) también son reescritos por la regla que ellos mismos definen.

Persistencia (único filtro, sin meta): una celda se vuelve VACÍO si su próxima
física es degenerada — no-finita, o |theta'| > theta_bound (la física se voló).
El vacío no computa, pero es re-colonizable: una vecina viva copia su física
con una perturbación mínima (aproximación Nivel-1 de "la regla vecina escribe
en él"; en Nivel 2 la escritura será literal).

Manos visibles (los ÚNICOS knobs nuestros — mantener mínimos y declarados):
  - theta_bound : qué cuenta como "degenerado" (el filtro de persistencia)
  - mut_scale   : ruido al colonizar (la fuente de variación)
  - el vacío aporta v=0 al vecindario ("vacío frío")
No hay recompensa, no hay objetivo, no hay selección dirigida.
"""

from __future__ import annotations

import numpy as np

D = 8  # [a1, a2, a3, b, m1, m2, m3, eta]

# Manos visibles (ver docstring)
THETA_BOUND = 1e4
MUT_SCALE = 0.01
INIT_THETA_SCALE = 0.5

# Umbrales de OBSERVACIÓN (no afectan la dinámica; sólo describen el veredicto)
THERMAL_ALIVE_FRAC = 0.05
FROZEN_EPS = 1e-7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class Utero1D:
    """Anillo 1-D Nivel 1. Estado mutable de simulación (materia+física+vida)."""

    def __init__(self, n: int = 64, seed: int = 0,
                 theta_bound: float = THETA_BOUND,
                 mut_scale: float = MUT_SCALE):
        self.n = n
        self.theta_bound = theta_bound
        self.mut_scale = mut_scale
        self.rng = np.random.default_rng(seed)
        self.v = self.rng.uniform(0.0, 1.0, size=n)
        self.theta = self.rng.normal(0.0, INIT_THETA_SCALE, size=(n, D))
        self.alive = np.ones(n, dtype=bool)

    # ---- el paso (sincrónico) ----

    def step(self) -> dict:
        """Un tick: aplicar cada física local, filtrar por persistencia,
        re-colonizar el vacío. Devuelve métricas descriptivas del tick."""
        v, th, alive = self.v, self.theta, self.alive
        prev_v, prev_th, prev_alive = v.copy(), th.copy(), alive.copy()

        al = np.roll(alive, 1)          # vecina izquierda viva?
        ar = np.roll(alive, -1)         # vecina derecha viva?
        vl = np.where(al, np.roll(v, 1), 0.0)    # vacío frío: aporta 0
        vr = np.where(ar, np.roll(v, -1), 0.0)

        # media de física de las vecinas VIVAS (si ninguna, la propia)
        thl, thr = np.roll(th, 1, axis=0), np.roll(th, -1, axis=0)
        n_vec = (al.astype(float) + ar.astype(float))[:, None]
        th_vec = (thl * al[:, None] + thr * ar[:, None]) / np.maximum(n_vec, 1.0)
        th_vec = np.where(n_vec > 0, th_vec, th)

        a1, a2, a3, b = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
        m1, m2, m3, eta = th[:, 4], th[:, 5], th[:, 6], th[:, 7]

        new_v = _sigmoid(a1 * vl + a2 * v + a3 * vr + b)
        lap_v = vl + vr - 2.0 * v
        g = (m1[:, None] * (th_vec - th)
             + m2[:, None] * lap_v[:, None] * th
             + m3[:, None] * th)
        new_th = th + eta[:, None] * g

        # ---- persistencia: única ley de muerte (sin meta) ----
        degenerate = (~np.isfinite(new_v)
                      | ~np.isfinite(new_th).all(axis=1)
                      | (np.abs(new_th).max(axis=1) > self.theta_bound))
        new_alive = alive & ~degenerate

        new_v = np.where(new_alive, new_v, 0.0)
        new_th = np.where(new_alive[:, None], new_th, 0.0)

        # ---- re-colonización del vacío por física viva vecina ----
        cl, cr = np.roll(new_alive, 1), np.roll(new_alive, -1)
        colonizable = ~new_alive & (cl | cr)
        idx = np.flatnonzero(colonizable)
        if idx.size:
            take_left = np.where(cl[idx] & cr[idx],
                                 self.rng.random(idx.size) < 0.5, cl[idx])
            parent = np.where(take_left, (idx - 1) % self.n, (idx + 1) % self.n)
            new_th[idx] = new_th[parent] + self.rng.normal(
                0.0, self.mut_scale, size=(idx.size, D))
            new_v[idx] = new_v[parent]
            new_alive[idx] = True

        self.v, self.theta, self.alive = new_v, new_th, new_alive

        # ---- métricas descriptivas (observar, no premiar) ----
        both = prev_alive & new_alive
        rule_change = (float(np.abs(new_th[both] - prev_th[both]).mean())
                       if both.any() else 0.0)
        value_change = (float(np.abs(new_v[both] - prev_v[both]).mean())
                        if both.any() else 0.0)
        return {
            "alive_frac": float(new_alive.mean()),
            "rule_change": rule_change,
            "value_change": value_change,
            "colonized": int(idx.size),
        }


def run_history(n: int = 64, seed: int = 0, ticks: int = 500,
                **kwargs) -> dict:
    """Correr un útero y registrar la historia completa (para mirar y juzgar)."""
    u = Utero1D(n=n, seed=seed, **kwargs)
    frames = np.full((ticks + 1, n), np.nan)
    frames[0] = np.where(u.alive, u.v, np.nan)
    hist: dict = {"alive_frac": [], "rule_change": [], "value_change": [],
                  "colonized": []}
    for t in range(ticks):
        m = u.step()
        for k in hist:
            hist[k].append(m[k])
        frames[t + 1] = np.where(u.alive, u.v, np.nan)
    hist["frames"] = frames
    return hist


def verdict(hist: dict, window: int = 100) -> str:
    """Clasificar la ventana final: 'termica' | 'cristal' | 'pulso'.

    Sólo describe — no retro-alimenta la dinámica. térmica = (casi) todo vacío;
    cristal = vivo pero congelado (ni la física ni la materia cambian);
    pulso = vivo y todavía deviniendo.
    """
    af = np.array(hist["alive_frac"][-window:])
    rc = np.array(hist["rule_change"][-window:])
    vc = np.array(hist["value_change"][-window:])
    if af.mean() < THERMAL_ALIVE_FRAC:
        return "termica"
    if rc.max() < FROZEN_EPS and vc.max() < FROZEN_EPS:
        return "cristal"
    return "pulso"
