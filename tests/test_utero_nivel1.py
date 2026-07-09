"""Tests del Útero Nivel 1 (sustrato auto-reescribiente, docs/EL_UTERO.md)."""

import numpy as np
import pytest

from zeta_life.utero import Utero1D, run_history, verdict
from zeta_life.utero.nivel1 import D


def test_init_shapes_and_all_alive():
    u = Utero1D(n=64, seed=0)
    assert u.v.shape == (64,)
    assert u.theta.shape == (64, D)
    assert u.alive.all()


def test_determinism_same_seed():
    a, b = Utero1D(n=32, seed=7), Utero1D(n=32, seed=7)
    for _ in range(50):
        a.step()
        b.step()
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.theta, b.theta)
    np.testing.assert_array_equal(a.alive, b.alive)


def test_values_stay_bounded():
    u = Utero1D(n=32, seed=1)
    for _ in range(100):
        u.step()
        assert np.isfinite(u.v[u.alive]).all()
        assert ((u.v[u.alive] >= 0.0) & (u.v[u.alive] <= 1.0)).all()


def test_blown_up_physics_dies():
    u = Utero1D(n=16, seed=0)
    u.theta[5] = 0.0
    u.theta[5, 0] = u.theta_bound * 2  # física ya degenerada, eta=0 (no cambia)
    u.alive[:] = True
    # aislar: matar vecinas para que no la re-colonicen en el mismo tick
    u.alive[4] = u.alive[6] = False
    u.step()
    assert not u.alive[5]


def test_void_without_neighbors_stays_void():
    u = Utero1D(n=16, seed=0)
    u.alive[:] = False
    u.step()
    assert not u.alive.any()


def test_void_next_to_life_gets_colonized():
    u = Utero1D(n=16, seed=3)
    u.alive[8] = False
    u.step()
    assert u.alive[8]  # una vecina viva la colonizó


def test_colonized_cell_inherits_perturbed_physics():
    u = Utero1D(n=16, seed=3, mut_scale=0.01)
    u.alive[8] = False
    u.step()
    # heredó una física cercana (pero no idéntica) a una vecina viva
    d_left = np.abs(u.theta[8] - u.theta[7]).max()
    d_right = np.abs(u.theta[8] - u.theta[9]).max()
    assert min(d_left, d_right) < 1.0
    assert min(d_left, d_right) > 0.0


def test_run_history_shapes():
    hist = run_history(n=32, seed=0, ticks=100)
    assert hist["frames"].shape == (101, 32)
    assert len(hist["alive_frac"]) == 100


def test_verdict_termica():
    hist = {"alive_frac": [0.0] * 100, "rule_change": [0.0] * 100,
            "value_change": [0.0] * 100}
    assert verdict(hist) == "termica"


def test_verdict_cristal():
    hist = {"alive_frac": [1.0] * 100, "rule_change": [0.0] * 100,
            "value_change": [0.0] * 100}
    assert verdict(hist) == "cristal"


def test_verdict_pulso():
    hist = {"alive_frac": [0.8] * 100, "rule_change": [0.01] * 100,
            "value_change": [0.02] * 100}
    assert verdict(hist) == "pulso"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
