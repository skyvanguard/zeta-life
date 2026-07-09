"""Tests del Útero Nivel 2 (reglas-programa auto-reescribientes)."""

import numpy as np
import pytest

from zeta_life.utero.nivel2 import (ADD, CONST, COPY, K, MUTO, N_OPS, NOP,
                                    SPAWN, UteroNivel2, execute, run_history,
                                    verdict)


def prog(*instrs):
    """Construir una regla: instrucciones dadas + relleno NOP hasta K."""
    code = np.zeros((K, 4), dtype=np.int64)
    for i, ins in enumerate(instrs):
        code[i, :len(ins)] = ins
    return code


def test_totality_random_programs_never_crash():
    rng = np.random.default_rng(0)
    for _ in range(200):
        code = np.zeros((K, 4), dtype=np.int64)
        code[:, 0] = rng.integers(0, N_OPS, K)
        code[:, 1:] = rng.integers(0, 16, (K, 3))
        ctx = (code, code, code)
        v, nxt, spawn, _ = execute(code, rng.random(), rng.random(), rng.random(), ctx)
        assert np.isfinite(v) and 0.0 < v < 1.0
        assert nxt.shape == (K, 4)


def test_determinism_same_seed():
    a, b = UteroNivel2(n=32, seed=7), UteroNivel2(n=32, seed=7)
    for _ in range(30):
        a.step()
        b.step()
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.code, b.code)
    np.testing.assert_array_equal(a.alive, b.alive)


def test_matter_blind_physics_dies():
    u = UteroNivel2(n=8, seed=0)
    u.code[3] = prog()                       # todo NOP: salida constante 0.5
    u.code[4] = prog((CONST, 12, 0, 3))      # constante pura a R3
    u.step()
    assert not u.alive[3]
    assert not u.alive[4]


def test_matter_sensitive_physics_survives():
    u = UteroNivel2(n=8, seed=0)
    for i in range(8):
        u.code[i] = prog((ADD, 1, 1, 3))     # R3 = 2*v: depende de la materia
    u.step()
    assert u.alive.all()


def test_muto_rewrites_own_next_opcode():
    # CONST 10 -> R0 = (10-8)/4 = 0.5 ; MUTO pos 5 desde R0 -> op = int(0.5*10) = 5
    code = prog((CONST, 10, 0, 0), (MUTO, 5, 0, 0), (ADD, 1, 1, 3))
    _, nxt, _, _ = execute(code, 0.2, 0.5, 0.8, (None, code, None))
    assert nxt[5, 0] == 5
    assert code[5, 0] == NOP  # la regla ORIGINAL no cambió (escritura staged)


def test_copy_grafts_neighbor_instruction():
    left = prog((ADD, 2, 2, 3))
    code = prog((COPY, 0, 0, 9), (ADD, 1, 1, 3))   # copia instr 0 de izq -> propia 9
    _, nxt, _, _ = execute(code, 0.2, 0.5, 0.8, (left, code, None))
    np.testing.assert_array_equal(nxt[9], left[0])


def test_spawn_colonizes_void_with_new_code():
    u = UteroNivel2(n=8, seed=1)
    for i in range(8):
        u.code[i] = prog((ADD, 1, 1, 3))
    u.code[4] = prog((ADD, 1, 1, 3), (SPAWN, 0, 0, 0))   # engendra a la izquierda
    u.alive[3] = False
    u.v[3] = 0.0
    u.step()
    assert u.alive[3]
    np.testing.assert_array_equal(u.code[3], u.code[4])  # escribió SU código


def test_dead_spawner_does_not_colonize():
    u = UteroNivel2(n=8, seed=1)
    for i in range(8):
        u.code[i] = prog((ADD, 1, 1, 3))
    u.code[4] = prog((CONST, 12, 0, 3), (SPAWN, 0, 0, 0))  # ciega a la materia + SPAWN
    u.alive[3] = False
    u.step()
    assert not u.alive[4]      # murió por la sonda
    assert not u.alive[3]      # su escritura no ocurrió


def test_void_without_spawner_stays_void():
    # diferencia clave con Nivel 1: la colonización YA NO es automática
    u = UteroNivel2(n=8, seed=1)
    for i in range(8):
        u.code[i] = prog((ADD, 1, 1, 3))     # nadie tiene SPAWN
    u.alive[3] = False
    u.v[3] = 0.0
    for _ in range(5):
        u.step()
    assert not u.alive[3]


def test_run_history_shapes_and_ops():
    hist = run_history(n=32, seed=0, ticks=60)
    assert hist["frames"].shape == (61, 32)
    assert len(hist["alive_frac"]) == 60
    assert hist["op_hist_init"].shape == (N_OPS,)
    assert hist["op_hist_final"].shape == (N_OPS,)


def test_verdict_cases():
    base = {"alive_frac": [0.0] * 100, "code_change": [0.0] * 100,
            "value_change": [0.0] * 100}
    assert verdict(base) == "termica"
    frozen = {"alive_frac": [0.9] * 100, "code_change": [0.0] * 100,
              "value_change": [0.0] * 100}
    assert verdict(frozen) == "cristal"
    living = {"alive_frac": [0.9] * 100, "code_change": [0.01] * 100,
              "value_change": [0.02] * 100}
    assert verdict(living) == "pulso"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
