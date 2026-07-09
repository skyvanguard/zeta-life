"""
El Útero — Nivel 2: la física reescribe su propia FORMA.

En el Nivel 1 la ley tenía forma fija y sólo movía sus parámetros — y tendió al
cristal. Aquí la regla es un PROGRAMA corto en un lenguaje mínimo y total (sin
crashes): aritmética sobre la materia + operaciones que leen y ESCRIBEN el
código de las reglas mismas (espíritu von Neumann / AlChemy de Fontana,
docs/EL_UTERO.md Nivel 2). La celda puede inventar física nueva.

Cada celda viva: materia `v` (escalar en (0,1)) + regla `code` (K=16
instrucciones de 4 campos [op, a, b, c]). Por tick, cada celda ejecuta SU
programa sobre 4 registros (R0=v_izq, R1=v, R2=v_der, R3=0; todo acotado):

    op   semántica (campos a,b,c; registros = campo % 4)
    ---  ---------------------------------------------------------------
    NOP    nada
    ADD    Rc = clip(Ra + Rb)          SUB   Rc = clip(Ra - Rb)
    MUL    Rc = clip(Ra * Rb)          THR   Rc = 1 si Ra > Rb sino 0
    CONST  Rc = (a - 8) / 4
    READ   Rc = opcode de la regla de {izq,self,der}[a%3] en posición b%K
    MUTO   reescribe el opcode PROPIO en posición a%K desde |Rb|   (auto-mod)
    COPY   injerta la instrucción b%K de {izq,self,der}[a%3] en la propia c%K
    SPAWN  si la vecina {izq,der}[a%2] es VACÍO, escribe en ella su código
           nuevo completo y su materia (colonización LITERAL)

Salida: v' = sigmoid(R3) — la materia debe fluir por el código hacia R3 para
que la celda viva. MUTO/COPY escriben sobre el código PRÓXIMO (genoma staged);
READ/COPY leen el código ACTUAL. Actualización sincrónica.

Persistencia (único filtro, sin meta): muere la física CIEGA A LA MATERIA —
si su salida es idéntica ante materias de prueba distintas (mapa constante,
"física muerta" del boceto). No hay blowup posible: todo está acotado.

Manos visibles (menos que en Nivel 1 — no hay ruido inyectado):
  - la sonda de muerte (ciega-a-la-materia = muerta), con sus 2 materias de prueba
  - el orden aleatorio (sembrado) al resolver conflictos de SPAWN
  - el vacío frío (aporta v=0 y lectura 0)
La variación viene SOLO de la sopa inicial y de la dinámica del código.
"""

from __future__ import annotations

import math

import numpy as np

N_OPS = 10
NOP, ADD, SUB, MUL, THR, CONST, READ, MUTO, COPY, SPAWN = range(N_OPS)
K = 16            # instrucciones por regla
F = 4             # campos por instrucción [op, a, b, c]
ARG_RANGE = 16    # rango de los campos a,b,c
REG_CLIP = 4.0
PROBE_EPS = 1e-12

# Umbrales de OBSERVACIÓN (sólo describen el veredicto)
THERMAL_ALIVE_FRAC = 0.05
FROZEN_VALUE_EPS = 1e-7


def _clip(x: float) -> float:
    return -REG_CLIP if x < -REG_CLIP else (REG_CLIP if x > REG_CLIP else x)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def execute(code: np.ndarray, vl: float, v: float, vr: float,
            ctx: tuple) -> tuple:
    """Ejecutar una regla. ctx = (code_izq|None, code_self, code_der|None).

    Devuelve (v', own_next, spawn) con spawn en {None, 0=izq, 1=der}.
    Total por construcción: nunca lanza, siempre termina en K pasos.
    """
    r = [vl, v, vr, 0.0]
    own_next = code.copy()
    spawn = None
    for k in range(K):
        op, a, b, c = int(code[k, 0]), int(code[k, 1]), int(code[k, 2]), int(code[k, 3])
        if op == ADD:
            r[c % 4] = _clip(r[a % 4] + r[b % 4])
        elif op == SUB:
            r[c % 4] = _clip(r[a % 4] - r[b % 4])
        elif op == MUL:
            r[c % 4] = _clip(r[a % 4] * r[b % 4])
        elif op == THR:
            r[c % 4] = 1.0 if r[a % 4] > r[b % 4] else 0.0
        elif op == CONST:
            r[c % 4] = (a - 8) / 4.0
        elif op == READ:
            src = ctx[a % 3]
            r[c % 4] = float(src[b % K, 0]) / N_OPS if src is not None else 0.0
        elif op == MUTO:
            own_next[a % K, 0] = int(abs(r[b % 4]) * N_OPS) % N_OPS
        elif op == COPY:
            src = ctx[a % 3]
            if src is not None:
                own_next[c % K] = src[b % K]
        elif op == SPAWN:
            # side=a%2; los campos b,c codifican la VARIACIÓN GERMINAL de la
            # cría (usada por la encarnación 'germinal'; nivel2/v1 la ignoran
            # y copian exacto): posición c%K, opcode nuevo desde |R[b]| en el
            # momento del parto — la misma función de MUTO, acoplada a materia.
            spawn = (a % 2, c % K, int(abs(r[b % 4]) * N_OPS) % N_OPS)
        # NOP: nada
    return _sigmoid(r[3]), own_next, spawn


def _output_only(code: np.ndarray, vl: float, v: float, vr: float,
                 ctx: tuple) -> float:
    """Ejecución fantasma (sin efectos): sólo la salida de materia."""
    return execute(code, vl, v, vr, ctx)[0]


class UteroNivel2:
    """Anillo 1-D Nivel 2: reglas-programa que reescriben su propia forma."""

    def __init__(self, n: int = 64, seed: int = 0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.v = self.rng.uniform(0.0, 1.0, size=n)
        self.code = np.zeros((n, K, F), dtype=np.int64)
        self.code[:, :, 0] = self.rng.integers(0, N_OPS, size=(n, K))
        self.code[:, :, 1:] = self.rng.integers(0, ARG_RANGE, size=(n, K, F - 1))
        self.alive = np.ones(n, dtype=bool)

    def step(self) -> dict:
        """Un tick sincrónico: ejecutar, filtrar por persistencia, colonizar."""
        n = self.n
        v, code, alive = self.v, self.code, self.alive
        prev_code, prev_alive = code.copy(), alive.copy()
        prev_v = v.copy()

        new_v, new_code = v.copy(), code.copy()
        new_alive = alive.copy()
        spawns: list[tuple[int, int]] = []  # (celda_destino, engendradora)

        for i in np.flatnonzero(alive):
            l, r_ = (i - 1) % n, (i + 1) % n
            ctx = (code[l] if alive[l] else None, code[i],
                   code[r_] if alive[r_] else None)
            vl = v[l] if alive[l] else 0.0
            vr = v[r_] if alive[r_] else 0.0

            v_new, own_next, spawn = execute(code[i], vl, float(v[i]), vr, ctx)

            # persistencia: muere la física ciega a la materia (mapa constante)
            p1 = _output_only(code[i], 0.0, 0.0, 0.0, ctx)
            p2 = _output_only(code[i], 1.0, 1.0, 1.0, ctx)
            dead = (not math.isfinite(v_new)
                    or (abs(v_new - p1) < PROBE_EPS and abs(v_new - p2) < PROBE_EPS
                        and abs(p1 - p2) < PROBE_EPS))
            if dead:
                new_alive[i] = False
                new_v[i] = 0.0
                new_code[i] = 0
            else:
                new_v[i] = v_new
                new_code[i] = own_next
                if spawn is not None:
                    t = l if spawn[0] == 0 else r_
                    if not alive[t]:      # sólo hacia vacío de inicio de tick
                        spawns.append((t, i))

        # colonización LITERAL: la regla vecina escribe su código en el vacío.
        # (orden sembrado para resolver conflictos; engendradora muerta no escribe)
        colonized = 0
        if spawns:
            claimed: set[int] = set()
            for j in self.rng.permutation(len(spawns)):
                t, i = spawns[j]
                if t in claimed or new_alive[t] or not new_alive[i]:
                    continue
                new_code[t] = new_code[i].copy()
                new_v[t] = new_v[i]
                new_alive[t] = True
                claimed.add(t)
                colonized += 1

        self.v, self.code, self.alive = new_v, new_code, new_alive

        # métricas descriptivas (observar, no premiar)
        both = prev_alive & new_alive
        code_change = (float((new_code[both] != prev_code[both]).mean())
                       if both.any() else 0.0)
        value_change = (float(np.abs(new_v[both] - prev_v[both]).mean())
                        if both.any() else 0.0)
        n_alive = int(new_alive.sum())
        diversity = (len({new_code[i].tobytes()
                          for i in np.flatnonzero(new_alive)}) if n_alive else 0)
        return {
            "alive_frac": float(new_alive.mean()),
            "code_change": code_change,
            "value_change": value_change,
            "colonized": colonized,
            "diversity": diversity,
        }

    def op_histogram(self) -> np.ndarray:
        """Frecuencia de cada op en el código VIVO (para ver selección sin meta)."""
        if not self.alive.any():
            return np.zeros(N_OPS)
        ops = self.code[self.alive][:, :, 0].ravel()
        h = np.bincount(ops % N_OPS, minlength=N_OPS).astype(float)
        return h / h.sum()


def run_history(n: int = 64, seed: int = 0, ticks: int = 500) -> dict:
    """Correr un útero Nivel 2 registrando la historia completa."""
    u = UteroNivel2(n=n, seed=seed)
    frames = np.full((ticks + 1, n), np.nan)
    frames[0] = np.where(u.alive, u.v, np.nan)
    hist: dict = {"alive_frac": [], "code_change": [], "value_change": [],
                  "colonized": [], "diversity": [],
                  "op_hist_init": u.op_histogram()}
    for t in range(ticks):
        m = u.step()
        for k in ("alive_frac", "code_change", "value_change",
                  "colonized", "diversity"):
            hist[k].append(m[k])
        frames[t + 1] = np.where(u.alive, u.v, np.nan)
    hist["frames"] = frames
    hist["op_hist_final"] = u.op_histogram()
    return hist


def verdict(hist: dict, window: int = 100) -> str:
    """'termica' | 'cristal' | 'pulso' sobre la ventana final (sólo describe)."""
    af = np.array(hist["alive_frac"][-window:])
    cc = np.array(hist["code_change"][-window:])
    vc = np.array(hist["value_change"][-window:])
    if af.mean() < THERMAL_ALIVE_FRAC:
        return "termica"
    if cc.max() == 0.0 and vc.max() < FROZEN_VALUE_EPS:
        return "cristal"
    return "pulso"
