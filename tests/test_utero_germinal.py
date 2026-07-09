"""Tests de la variación germinal (v2): SPAWN escribe modulado por la materia."""

import numpy as np
import pytest

from zeta_life.utero.creciente import UteroCreciente, run_history
from zeta_life.utero.nivel2 import ADD, CONST, K, N_OPS, SPAWN, execute


def prog(*instrs):
    code = np.zeros((K, 4), dtype=np.int64)
    for i, ins in enumerate(instrs):
        code[i, :len(ins)] = ins
    return code


def test_spawn_carries_birth_variation_fields():
    # CONST 10 -> R0 = 0.5 ; SPAWN side=izq, b=0 (usa R0), c=9 (posición)
    code = prog((CONST, 10, 0, 0), (ADD, 1, 1, 3), (SPAWN, 0, 0, 9))
    _, _, spawn, _ = execute(code, 0.2, 0.5, 0.8, (None, code, None))
    side, mpos, mop = spawn
    assert side == 0
    assert mpos == 9
    assert mop == int(0.5 * N_OPS) % N_OPS == 5


def test_germinal_child_differs_in_exactly_one_instruction():
    u = UteroCreciente(n0=5, seed=2, max_n=8, germinal=True)
    parent = prog((CONST, 10, 0, 0), (ADD, 1, 1, 3), (SPAWN, 0, 0, 9))
    for i in range(5):
        u.code[i] = prog((ADD, 1, 1, 3))
    u.code[3] = parent
    u.alive[2] = False
    u.v[2] = 0.0
    u.step()
    assert u.alive[2]
    diff = (u.code[2] != u.code[3]).any(axis=1)
    assert diff.sum() == 1                      # exactamente UNA instrucción
    assert u.code[2][9, 0] == 5                 # el opcode nacido de la materia


def test_non_germinal_child_is_exact_copy():
    u = UteroCreciente(n0=5, seed=2, max_n=8, germinal=False)
    parent = prog((CONST, 10, 0, 0), (ADD, 1, 1, 3), (SPAWN, 0, 0, 9))
    for i in range(5):
        u.code[i] = prog((ADD, 1, 1, 3))
    u.code[3] = parent
    u.alive[2] = False
    u.step()
    assert u.alive[2]
    np.testing.assert_array_equal(u.code[2], u.code[3])


def test_germinal_determinism():
    a = UteroCreciente(n0=16, seed=9, germinal=True)
    b = UteroCreciente(n0=16, seed=9, germinal=True)
    for _ in range(30):
        a.step()
        b.step()
    assert a.n == b.n
    np.testing.assert_array_equal(a.code, b.code)


def test_germinal_soup_runs_and_stays_bounded():
    u = UteroCreciente(n0=16, seed=0, max_n=64, germinal=True)
    for _ in range(100):
        u.step()
    assert u.n <= 64
    assert np.isfinite(u.v[u.alive]).all()


def test_run_history_germinal_flag():
    h = run_history(n0=16, seed=1, ticks=60, max_n=64, germinal=True)
    assert len(h["new_genomes"]) == 60


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
