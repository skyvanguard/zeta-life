"""Tests del Útero creciente (async + espacio que se abre desde adentro)."""

import numpy as np
import pytest

from zeta_life.utero.creciente import UteroCreciente, run_history, verdict
from zeta_life.utero.nivel2 import ADD, CONST, K, SPAWN


def prog(*instrs):
    code = np.zeros((K, 4), dtype=np.int64)
    for i, ins in enumerate(instrs):
        code[i, :len(ins)] = ins
    return code


def sensitive():
    return prog((ADD, 1, 1, 3))          # R3 = 2*v: vive


def test_init_and_determinism():
    a, b = UteroCreciente(n0=16, seed=5), UteroCreciente(n0=16, seed=5)
    for _ in range(30):
        a.step()
        b.step()
    assert a.n == b.n
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.code, b.code)
    np.testing.assert_array_equal(a.alive, b.alive)


def test_random_soup_runs_without_crash():
    u = UteroCreciente(n0=16, seed=0, max_n=64)
    for _ in range(50):
        m = u.step()
    assert u.n <= 64
    assert np.isfinite(u.v[u.alive]).all()


def test_spawn_at_right_edge_grows_world():
    u = UteroCreciente(n0=4, seed=1, max_n=16)
    for i in range(4):
        u.code[i] = sensitive()
    u.code[3] = prog((ADD, 1, 1, 3), (SPAWN, 1, 0, 0))   # engendra a la derecha
    n_before = u.n
    u.step()
    assert u.n == n_before + 1
    assert u.alive[-1]
    np.testing.assert_array_equal(u.code[-1], u.code[-2])  # hija del borde


def test_spawn_at_left_edge_grows_world_and_tracks_origin():
    u = UteroCreciente(n0=4, seed=1, max_n=16)
    for i in range(4):
        u.code[i] = sensitive()
    u.code[0] = prog((ADD, 1, 1, 3), (SPAWN, 0, 0, 0))   # engendra a la izquierda
    u.step()
    assert u.n == 5
    assert u.left_grown == 1
    assert u.alive[0]


def test_petri_wall_caps_growth():
    u = UteroCreciente(n0=4, seed=1, max_n=4)
    for i in range(4):
        u.code[i] = prog((ADD, 1, 1, 3), (SPAWN, 1, 0, 0))
    for _ in range(5):
        u.step()
    assert u.n == 4                       # la pared de la placa


def test_interior_void_colonization_still_works():
    u = UteroCreciente(n0=5, seed=2, max_n=8)
    for i in range(5):
        u.code[i] = sensitive()
    u.code[3] = prog((ADD, 1, 1, 3), (SPAWN, 0, 0, 0))   # escribe a su izquierda
    u.alive[2] = False
    u.v[2] = 0.0
    u.step()
    assert u.alive[2]


def test_matter_blind_dies_in_async_too():
    u = UteroCreciente(n0=6, seed=0, max_n=8)
    for i in range(6):
        u.code[i] = sensitive()
    u.code[2] = prog((CONST, 12, 0, 3))   # constante pura: ciega a la materia
    u.step()
    assert not u.alive[2]


def test_novelty_counts_new_genomes():
    hist = run_history(n0=16, seed=0, ticks=50, max_n=32)
    assert hist["total_genomes"] >= 16 - 5   # la sopa inicial ya cuenta
    assert len(hist["new_genomes"]) == 50
    assert all(g >= 0 for g in hist["new_genomes"])


def test_run_history_frames_aligned():
    hist = run_history(n0=8, seed=3, ticks=40, max_n=32)
    frames = hist["frames"]
    assert frames.shape[0] == 41
    assert frames.shape[1] >= 8           # el mundo pudo crecer


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
