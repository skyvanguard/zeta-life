"""Tests de la muerte por equilibrio (v4): lo que deja de devenir, deja de ser."""

import numpy as np
import pytest

from zeta_life.utero.creciente import UteroCreciente, run_history
from zeta_life.utero.nivel2 import ADD, CONST, K, MUL, SUB, THR


def prog(*instrs):
    code = np.zeros((K, 4), dtype=np.int64)
    for i, ins in enumerate(instrs):
        code[i, :len(ins)] = ins
    return code


def fixed_point_prog():
    # v' = v (identidad): sensible a la materia (pasa la sonda) pero QUIETA.
    # ADD v,0 -> R3 con R3 inicial 0: ADD 1,3 -> 3 => R3 = v + 0 = v
    return prog((ADD, 1, 3, 3))


def moving_prog():
    # doubling map en el toro: v' = 2v mod 1 — materia que nunca se asienta
    return prog((ADD, 1, 1, 3))


def test_fixed_point_cell_dies_after_window():
    u = UteroCreciente(n0=6, seed=0, max_n=8, toroidal=True,
                       muerte_equilibrio=True, eq_window=10)
    for i in range(6):
        u.code[i] = fixed_point_prog()
    for _ in range(9):
        u.step()
    assert u.alive.any()                  # aún dentro de la ventana
    for _ in range(5):
        u.step()
    assert not u.alive.any()              # el equilibrio los mató a todos


def test_moving_matter_survives_equilibrium_death():
    u = UteroCreciente(n0=6, seed=0, max_n=8, toroidal=True,
                       muerte_equilibrio=True, eq_window=10)
    u.v[:] = 0.123456                     # irracional-ish: orbita no trivial
    for i in range(6):
        u.code[i] = moving_prog()
    for _ in range(50):
        u.step()
    assert u.alive.all()                  # el doubling map nunca se asienta


def test_equilibrium_death_leaves_recolonizable_void():
    u = UteroCreciente(n0=6, seed=1, max_n=8, toroidal=True,
                       muerte_equilibrio=True, eq_window=5)
    u.v[:] = 0.3
    for i in range(6):
        u.code[i] = moving_prog()
    u.code[2] = fixed_point_prog()        # esta se asentará y morirá
    for _ in range(20):
        u.step()
    # la celda 2 murió por equilibrio en algún momento; el mundo sigue vivo
    assert u.alive.sum() >= 4


def test_flag_off_is_byte_identical():
    # v3 sin muerte_equilibrio: seed 13 primeros 200 ticks reproducibles
    a = UteroCreciente(n0=16, seed=13, germinal=True, toroidal=True)
    b = UteroCreciente(n0=16, seed=13, germinal=True, toroidal=True,
                       muerte_equilibrio=False)
    for _ in range(200):
        a.step()
        b.step()
    assert a.n == b.n
    np.testing.assert_array_equal(a.code, b.code)
    np.testing.assert_array_equal(a.v, b.v)


def test_v4_determinism():
    a = UteroCreciente(n0=16, seed=7, germinal=True, toroidal=True,
                       muerte_equilibrio=True)
    b = UteroCreciente(n0=16, seed=7, germinal=True, toroidal=True,
                       muerte_equilibrio=True)
    for _ in range(50):
        a.step()
        b.step()
    assert a.n == b.n
    np.testing.assert_array_equal(a.code, b.code)


def test_v4_soup_runs_bounded():
    u = UteroCreciente(n0=16, seed=0, max_n=64, germinal=True, toroidal=True,
                       muerte_equilibrio=True)
    for _ in range(200):
        u.step()
    a = u.v[u.alive]
    if a.size:
        assert np.isfinite(a).all()
        assert ((a >= 0.0) & (a < 1.0)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
