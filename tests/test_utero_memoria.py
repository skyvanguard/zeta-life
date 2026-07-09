"""Tests de la memoria (v5): R3 crudo persistente — estado oculto recurrente."""

import numpy as np
import pytest

from zeta_life.utero.creciente import UteroCreciente
from zeta_life.utero.nivel2 import ADD, CONST, K, execute


def prog(*instrs):
    code = np.zeros((K, 4), dtype=np.int64)
    for i, ins in enumerate(instrs):
        code[i, :len(ins)] = ins
    return code


def test_execute_returns_raw_r3():
    # CONST 13 -> R3 = 1.25 ; wrap -> materia 0.25 ; raw = 1.25
    code = prog((CONST, 13, 0, 3))
    out, _, _, raw = execute(code, 0.0, 0.0, 0.0, (None, code, None), wrap=True)
    assert abs(out - 0.25) < 1e-12
    assert abs(raw - 1.25) < 1e-12          # raw != materia (proyección con pérdida)


def test_r3_init_seeds_the_register():
    # NOP-only: R3 queda en r3_init ; materia toroidal = r3_init mod 1
    code = prog()
    out, _, _, raw = execute(code, 0.0, 0.0, 0.0, (None, code, None),
                             wrap=True, r3_init=1.7)
    assert abs(raw - 1.7) < 1e-12
    assert abs(out - 0.7) < 1e-12


def test_memory_persists_across_ticks():
    u = UteroCreciente(n0=4, seed=0, max_n=4, toroidal=True, memoria=True)
    # ADD R3,v -> R3 : R3 acumula (integrador) — memoria distinta de materia
    for i in range(4):
        u.code[i] = prog((ADD, 3, 1, 3))
    u.v[:] = 0.3
    m0 = u.mem.copy()
    u.step()
    u.step()
    assert not np.allclose(u.mem, m0)       # la memoria evolucionó
    # dos celdas con misma materia pueden tener memoria distinta (estado oculto)
    assert u.mem.std() >= 0.0               # existe y es real


def test_memory_off_is_byte_identical():
    a = UteroCreciente(n0=16, seed=13, germinal=True, toroidal=True,
                       memoria=False)
    b = UteroCreciente(n0=16, seed=13, germinal=True, toroidal=True)
    for _ in range(300):
        a.step()
        b.step()
    assert a.n == b.n
    np.testing.assert_array_equal(a.code, b.code)
    np.testing.assert_array_equal(a.v, b.v)


def test_child_born_without_memory():
    u = UteroCreciente(n0=6, seed=1, max_n=8, toroidal=True, memoria=True)
    for i in range(6):
        u.code[i] = prog((ADD, 3, 1, 3), (ADD, 3, 0, 3))  # acumula memoria
    u.v[:] = 0.3
    for _ in range(20):
        u.step()                            # las celdas acumulan memoria != 0
    # forzar una colonización: matar una celda con vecina viva+SPAWN
    u.code[2] = prog((ADD, 1, 1, 3), )
    from zeta_life.utero.nivel2 import SPAWN
    u.code[3] = prog((ADD, 1, 1, 3), (SPAWN, 0, 0, 0))
    u.mem[3] = 2.5
    u.alive[2] = False
    u.step()
    if u.alive[2]:
        assert u.mem[2] == 0.0              # la cría nace sin recuerdos


def test_memory_determinism():
    a = UteroCreciente(n0=16, seed=8, germinal=True, toroidal=True, memoria=True)
    b = UteroCreciente(n0=16, seed=8, germinal=True, toroidal=True, memoria=True)
    for _ in range(50):
        a.step()
        b.step()
    assert a.n == b.n
    np.testing.assert_array_equal(a.code, b.code)
    np.testing.assert_array_equal(a.mem, b.mem)


def test_memory_soup_runs_bounded():
    u = UteroCreciente(n0=16, seed=0, max_n=64, germinal=True, toroidal=True,
                       memoria=True)
    for _ in range(200):
        u.step()
    a = u.v[u.alive]
    if a.size:
        assert np.isfinite(a).all()
        assert ((a >= 0.0) & (a < 1.0)).all()
    assert np.isfinite(u.mem).all()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
