"""
El Útero creciente — v1: asincronía + espacio que se abre desde adentro.

El Nivel 2 síncrono en anillo fijo cayó en ciclos límite (20/20): determinismo
+ espacio fijo + sincronía ⇒ recurrencia. Esta encarnación ataca la jaula con
las dos direcciones que el boceto dejó abiertas (docs/EL_UTERO.md):

1. **Asincronía**: una celda actúa por vez, en orden aleatorio sembrado, y sus
   efectos (materia, código, colonización) se aplican DE INMEDIATO. El orden
   es azar sin dirección: perturba, no elige.

2. **Espacio creciente**: el mundo es una LÍNEA con dos fronteras, no un
   anillo. Cuando una física en el borde hace SPAWN hacia el más-allá, el
   mundo CRECE una celda y su código la habita. No hay regla nuestra de
   crecimiento: el espacio se abre sólo donde la física lo abre — el sistema
   genera su propia variable adyacente (la elección de Fran), al menos en lo
   espacial. El vacío interior sigue siendo colonizable por SPAWN.

El lenguaje de reglas es el mismo del Nivel 2 (10 ops totales, MUTO/COPY
reescriben la propia forma). Sin ruido inyectado sobre el código.

La vara de la novedad (anti-ilusión): con orden aleatorio, "no ciclar" ya no
prueba nada. El criterio honesto es acuñar GENOMAS NUNCA VISTOS: código nuevo
sólo puede nacer de eventos de escritura (MUTO/COPY), no del azar del orden.
Medimos genomas nuevos por tramo — ¿la novedad se sostiene o se seca?

Manos visibles:
  - el orden aleatorio sembrado (perturbación sin dirección)
  - max_n (la pared de la placa de Petri — límite físico, no meta)
  - la sonda de muerte ciega-a-la-materia (heredada del Nivel 2)
  - el vacío / el más-allá aportan v=0 y lectura nula (frío)
"""

from __future__ import annotations

import math

import numpy as np

from zeta_life.utero.nivel2 import (F, K, PROBE_EPS, execute, _output_only)

N0 = 16
MAX_N = 256

# Umbrales de OBSERVACIÓN (sólo describen el veredicto)
THERMAL_ALIVE_FRAC = 0.05
FROZEN_VALUE_EPS = 1e-7


class UteroCreciente:
    """Línea 1-D asincrónica que crece donde su física la abre."""

    def __init__(self, n0: int = N0, seed: int = 0, max_n: int = MAX_N,
                 germinal: bool = False, toroidal: bool = False,
                 muerte_equilibrio: bool = False, eq_eps: float = 1e-9,
                 eq_window: int = 100, memoria: bool = False):
        """germinal=True (v2): SPAWN no copia exacto — la cría nace con UNA
        instrucción reescrita desde la materia del momento del parto (campos
        b,c del SPAWN + registro; la misma función de MUTO). La variación sale
        del estado del mundo, no de un RNG nuestro. False = v1 (copia exacta),
        byte-idéntica a los resultados commiteados.

        toroidal=True (v3): la materia vive en un círculo — v' = R3 mod 1 en
        vez de sigmoid. El wrap permite mapas expansivos (caos expresable de
        verdad), atacando la causa raíz de v2: la materia congelada. La sonda
        de muerte usa separación irracional (0 y 0.618…) porque 0 y 1 son el
        MISMO punto del toro (con la sonda vieja, todo mapa lineal colapsaría
        y mataría física sana).

        muerte_equilibrio=True (v4): extensión del principio 3 — «cristal =
        muerto de pie». Una celda cuya materia queda quieta (|Δv| < eq_eps)
        durante eq_window ticks seguidos se vuelve VACÍO. Lo que deja de
        devenir, deja de ser (Schrödinger: lejos del equilibrio o muerto).
        No es meta ni recompensa: completa el filtro de persistencia, que ya
        mataba a la física ciega a la materia, matando también a la materia
        quieta. Manos declaradas: eq_eps, eq_window.

        memoria=True (v5): la DIMENSIÓN que Fran no había tocado. Cada celda
        retiene su R3 crudo (el potencial interno, antes de envolver — NO la
        materia observable v, que es su proyección con pérdida) y lo re-inyecta
        como R3 inicial el tick siguiente. Recurrencia / integración temporal:
        estado oculto tipo potencial de membrana. Da dinámica de 2º orden, que
        ensancha el borde-del-caos. La cría nace SIN recuerdos (mem=0). Sin
        manos nuevas."""
        self.germinal = germinal
        self.toroidal = toroidal
        self.muerte_eq = muerte_equilibrio
        self.eq_eps = eq_eps
        self.eq_window = eq_window
        self.eq_count = np.zeros(n0, dtype=np.int64)
        self.memoria = memoria
        self.mem = np.zeros(n0, dtype=np.float64)     # R3 crudo persistente
        # materias de prueba de la sonda ciega-a-la-materia (mano declarada)
        self._probe_hi = 0.6180339887498949 if toroidal else 1.0
        self.max_n = max_n
        self.rng = np.random.default_rng(seed)
        self.v = self.rng.uniform(0.0, 1.0, size=n0)
        self.code = np.zeros((n0, K, F), dtype=np.int64)
        self.code[:, :, 0] = self.rng.integers(0, 10, size=(n0, K))
        self.code[:, :, 1:] = self.rng.integers(0, 16, size=(n0, K, F - 1))
        self.alive = np.ones(n0, dtype=bool)
        self.left_grown = 0                # celdas añadidas por la izquierda
        self.seen: set = set()             # genomas vistos (para la novedad)
        self._register_genomes()

    @property
    def n(self) -> int:
        return len(self.v)

    def _register_genomes(self) -> int:
        new = 0
        for i in np.flatnonzero(self.alive):
            g = self.code[i].tobytes()
            if g not in self.seen:
                self.seen.add(g)
                new += 1
        return new

    def _ctx(self, i: int) -> tuple:
        n = self.n
        cl = self.code[i - 1] if i > 0 and self.alive[i - 1] else None
        cr = self.code[i + 1] if i < n - 1 and self.alive[i + 1] else None
        vl = float(self.v[i - 1]) if i > 0 and self.alive[i - 1] else 0.0
        vr = float(self.v[i + 1]) if i < n - 1 and self.alive[i + 1] else 0.0
        return (cl, self.code[i], cr), vl, vr

    def step(self) -> dict:
        """Un tick asincrónico: cada celda viva (orden aleatorio) actúa sobre
        el mundo ACTUAL. El borde crece si una física escribe en el más-allá."""
        n0_tick = self.n
        prev_v = self.v.copy()
        prev_code = self.code.copy()
        prev_alive = self.alive.copy()

        edge_left = None   # (code_copy, v) — primer reclamo gana
        edge_right = None
        colonized = 0

        for i in self.rng.permutation(np.flatnonzero(self.alive)):
            i = int(i)
            if not self.alive[i]:          # murió antes de su turno
                continue
            ctx, vl, vr = self._ctx(i)
            mi = float(self.mem[i]) if self.memoria else 0.0
            v_new, own_next, spawn, raw = execute(
                self.code[i], vl, float(self.v[i]), vr, ctx,
                wrap=self.toroidal, r3_init=mi)
            # persistencia: la física ciega a la materia muere (sonda, misma
            # memoria fija -> prueba de sensibilidad a la MATERIA sola)
            h = self._probe_hi
            p1 = _output_only(self.code[i], 0.0, 0.0, 0.0, ctx,
                              wrap=self.toroidal, r3_init=mi)
            p2 = _output_only(self.code[i], h, h, h, ctx,
                              wrap=self.toroidal, r3_init=mi)
            if (not math.isfinite(v_new)
                    or (abs(v_new - p1) < PROBE_EPS
                        and abs(v_new - p2) < PROBE_EPS
                        and abs(p1 - p2) < PROBE_EPS)):
                self.alive[i] = False
                self.v[i] = 0.0
                self.code[i] = 0
                self.eq_count[i] = 0
                self.mem[i] = 0.0
                continue
            # v4: muerte por equilibrio — lo que deja de devenir, deja de ser
            if self.muerte_eq:
                if abs(v_new - float(self.v[i])) < self.eq_eps:
                    self.eq_count[i] += 1
                else:
                    self.eq_count[i] = 0
                if self.eq_count[i] > self.eq_window:
                    self.alive[i] = False
                    self.v[i] = 0.0
                    self.code[i] = 0
                    self.eq_count[i] = 0
                    self.mem[i] = 0.0
                    continue
            # async: efectos inmediatos
            self.v[i] = v_new
            self.code[i] = own_next
            self.mem[i] = raw            # memoria: R3 crudo persistente
            if spawn is None:
                continue
            side, mpos, mop = spawn
            child = own_next.copy()
            if self.germinal:               # v2: nace con UNA instrucción
                child[mpos, 0] = mop        # reescrita desde la materia
            t = i - 1 if side == 0 else i + 1
            if t < 0:                       # escribe en el más-allá izquierdo
                if edge_left is None:
                    edge_left = (child, v_new)
            elif t >= self.n:               # más-allá derecho
                if edge_right is None:
                    edge_right = (child, v_new)
            elif not self.alive[t]:         # vacío interior: colonización
                self.code[t] = child
                self.v[t] = v_new
                self.alive[t] = True
                self.eq_count[t] = 0
                self.mem[t] = 0.0           # la cría nace sin recuerdos
                colonized += 1

        # crecimiento del mundo (fin de tick; tope = la placa de Petri)
        grown = 0
        grew_left = False
        if edge_right is not None and self.n < self.max_n:
            c, val = edge_right
            self.v = np.concatenate([self.v, [val]])
            self.code = np.concatenate([self.code, c[None]])
            self.alive = np.concatenate([self.alive, [True]])
            self.eq_count = np.concatenate([self.eq_count, [0]])
            self.mem = np.concatenate([self.mem, [0.0]])
            grown += 1
        if edge_left is not None and self.n < self.max_n:
            c, val = edge_left
            self.v = np.concatenate([[val], self.v])
            self.code = np.concatenate([c[None], self.code])
            self.alive = np.concatenate([[True], self.alive])
            self.eq_count = np.concatenate([[0], self.eq_count])
            self.mem = np.concatenate([[0.0], self.mem])
            self.left_grown += 1
            grown += 1
            grew_left = True

        # métricas descriptivas (observar, no premiar) — comparar las celdas
        # que existían al inicio del tick (índices corridos si creció a la izq)
        shift = 1 if grew_left else 0
        cur_v = self.v[shift:shift + n0_tick]
        cur_code = self.code[shift:shift + n0_tick]
        cur_alive = self.alive[shift:shift + n0_tick]
        both = prev_alive & cur_alive
        code_change = (float((cur_code[both] != prev_code[both]).mean())
                       if both.any() else 0.0)
        value_change = (float(np.abs(cur_v[both] - prev_v[both]).mean())
                        if both.any() else 0.0)
        new_genomes = self._register_genomes()
        n_alive = int(self.alive.sum())
        return {
            "alive_frac": float(self.alive.mean()),
            "n_world": self.n,
            "code_change": code_change,
            "value_change": value_change,
            "colonized": colonized,
            "grown": grown,
            "new_genomes": new_genomes,
            "diversity": (len({self.code[i].tobytes()
                               for i in np.flatnonzero(self.alive)})
                          if n_alive else 0),
        }


def run_history(n0: int = N0, seed: int = 0, ticks: int = 2000,
                max_n: int = MAX_N, germinal: bool = False,
                toroidal: bool = False,
                muerte_equilibrio: bool = False, memoria: bool = False) -> dict:
    """Correr un útero creciente registrando historia + frames alineados."""
    u = UteroCreciente(n0=n0, seed=seed, max_n=max_n, germinal=germinal,
                       toroidal=toroidal, muerte_equilibrio=muerte_equilibrio,
                       memoria=memoria)
    keys = ("alive_frac", "n_world", "code_change", "value_change",
            "colonized", "grown", "new_genomes", "diversity")
    hist: dict = {k: [] for k in keys}
    raw = [(np.where(u.alive, u.v, np.nan).copy(), u.left_grown)]
    for _ in range(ticks):
        m = u.step()
        for k in keys:
            hist[k].append(m[k])
        raw.append((np.where(u.alive, u.v, np.nan).copy(), u.left_grown))
    # alinear frames por coordenada (el mundo crece hacia ambos lados)
    lf = u.left_grown
    frames = np.full((len(raw), u.n), np.nan)
    for t, (vals, l_t) in enumerate(raw):
        start = lf - l_t
        frames[t, start:start + len(vals)] = vals
    hist["frames"] = frames
    hist["total_genomes"] = len(u.seen)
    return hist


def verdict(hist: dict, window: int = 100) -> str:
    """'termica' | 'cristal' | 'pulso' (sólo describe la ventana final)."""
    af = np.array(hist["alive_frac"][-window:])
    cc = np.array(hist["code_change"][-window:])
    vc = np.array(hist["value_change"][-window:])
    if af.mean() < THERMAL_ALIVE_FRAC:
        return "termica"
    if cc.max() == 0.0 and vc.max() < FROZEN_VALUE_EPS:
        return "cristal"
    return "pulso"
