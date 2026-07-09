"""Tests de la materia toroidal (v3): v' = R3 mod 1 — el wrap permite caos."""

import numpy as np
import pytest

from zeta_life.utero.creciente import UteroCreciente, run_history
from zeta_life.utero.nivel2 import ADD, CONST, K, execute


def prog(*instrs):
    code = np.zeros((K, 4), dtype=np.int64)
    for i, ins in enumerate(instrs):
        code[i, :len(ins)] = ins
    return code


def test_wrap_output_is_mod_1():
    # CONST 13 -> R3 = (13-8)/4 = 1.25 ; wrap -> 0.25
    code = prog((CONST, 13, 0, 3))
    v, _, _ = execute(code, 0.0, 0.0, 0.0, (None, code, None), wrap=True)
    assert abs(v - 0.25) < 1e-12


def test_wrap_negative_wraps_into_unit():
    # CONST 4 -> R3 = (4-8)/4 = -1.0... -1.0 % 1 = 0.0 ; usar -0.75: CONST 5
    code = prog((CONST, 5, 0, 3))            # R3 = -0.75 -> 0.25
    v, _, _ = execute(code, 0.0, 0.0, 0.0, (None, code, None), wrap=True)
    assert abs(v - 0.25) < 1e-12


def test_doubling_map_is_expressible_and_survives_probe():
    # ADD v,v -> R3: en el toro es el doubling map (caos canonico).
    # Con la sonda VIEJA (0 y 1) moriria: 2*0=0, 2*1 mod 1=0 -> "ciega".
    # Con la sonda irracional sobrevive: 2*0.618 mod 1 = 0.236 != 0.
    u = UteroCreciente(n0=6, seed=0, max_n=8, toroidal=True)
    for i in range(6):
        u.code[i] = prog((ADD, 1, 1, 3))
    u.step()
    assert u.alive.all()


def test_matter_blind_still_dies_on_torus():
    u = UteroCreciente(n0=6, seed=0, max_n=8, toroidal=True)
    for i in range(6):
        u.code[i] = prog((ADD, 1, 1, 3))
    u.code[2] = prog((CONST, 13, 0, 3))      # constante pura
    u.step()
    assert not u.alive[2]


def test_toroidal_values_stay_in_unit_interval():
    u = UteroCreciente(n0=16, seed=3, max_n=64, germinal=True, toroidal=True)
    for _ in range(100):
        u.step()
        a = u.v[u.alive]
        assert np.isfinite(a).all()
        assert ((a >= 0.0) & (a < 1.0)).all()


def test_toroidal_determinism():
    a = UteroCreciente(n0=16, seed=11, germinal=True, toroidal=True)
    b = UteroCreciente(n0=16, seed=11, germinal=True, toroidal=True)
    for _ in range(30):
        a.step()
        b.step()
    assert a.n == b.n
    np.testing.assert_array_equal(a.code, b.code)


def test_sigmoid_path_unchanged():
    # v1/v2 byte-idénticos: seed 0, 1000 ticks -> mundo 16, 22 genomas
    h = run_history(n0=16, seed=0, ticks=1000, max_n=256)
    assert h["n_world"][-1] == 16
    assert h["total_genomes"] == 22


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
