"""Tests for the Yvyra <-> zeta-life bridge."""

from __future__ import annotations

import json
import tempfile

import pytest
import torch

from zeta_life.bridge import AXES, DEFAULT_C, YvyraBridge


def _coherent_scores():
    return {"novedad": 0.3, "introspeccion": 0.8, "conexion": 0.1, "resolucion": 0.2}


class TestStep:
    def test_step_with_dict(self):
        b = YvyraBridge()
        out = b.step(_coherent_scores())
        assert out["tick"] == 1
        assert set(out["action"].keys()) == set(AXES)

    def test_step_with_list(self):
        b = YvyraBridge()
        out = b.step([0.3, 0.8, 0.1, 0.2])
        assert out["tick"] == 1

    def test_psi_in_unit_interval(self):
        b = YvyraBridge()
        for _ in range(10):
            out = b.step([0.3, 0.8, 0.1, 0.2])
            assert 0.0 <= out["psi"] <= 1.0

    def test_suggested_axis_is_valid(self):
        b = YvyraBridge()
        out = b.step(_coherent_scores())
        assert out["suggested_axis"] in AXES
        assert isinstance(out["suggestion"], str)

    def test_action_is_distribution(self):
        b = YvyraBridge()
        out = b.step(_coherent_scores())
        assert sum(out["action"].values()) == pytest.approx(1.0, abs=1e-4)

    def test_missing_axis_raises(self):
        b = YvyraBridge()
        with pytest.raises(ValueError):
            b.step({"novedad": 0.3, "introspeccion": 0.8})

    def test_wrong_length_list_raises(self):
        b = YvyraBridge()
        with pytest.raises(ValueError):
            b.step([0.3, 0.8, 0.1])


class TestPreference:
    def test_default_preference_is_C(self):
        b = YvyraBridge()
        st = b.state()
        pref = [st["preference"][a] for a in AXES]
        expected = [c / sum(DEFAULT_C) for c in DEFAULT_C]
        assert pref == pytest.approx(expected, abs=1e-5)

    def test_custom_preference_normalised(self):
        b = YvyraBridge(preference=[1.0, 1.0, 1.0, 1.0])
        st = b.state()
        assert st["preference"]["novedad"] == pytest.approx(0.25, abs=1e-5)

    def test_bad_preference_raises(self):
        with pytest.raises(ValueError):
            YvyraBridge(preference=[1.0, 2.0, 3.0])


class TestJSON:
    def test_step_json_roundtrip(self):
        b = YvyraBridge()
        out = json.loads(b.step_json(json.dumps(_coherent_scores())))
        for key in ("tick", "psi", "free_energy", "suggested_axis", "action"):
            assert key in out

    def test_state_json(self):
        b = YvyraBridge()
        b.step(_coherent_scores())
        st = json.loads(b.state_json())
        assert st["tick"] == 1

    def test_dream_json(self):
        b = YvyraBridge()
        for _ in range(5):
            b.step(_coherent_scores())
        rep = json.loads(b.dream_json())
        assert "transfers" in rep


class TestStateAndDream:
    def test_state_recent_experience(self):
        b = YvyraBridge()
        b.step([0.3, 0.8, 0.1, 0.2])
        st = b.state()
        assert set(st["recent_experience"].keys()) == set(AXES)

    def test_dream_runs(self):
        b = YvyraBridge()
        for _ in range(5):
            b.step(_coherent_scores())
        rep = b.dream()
        assert rep["duration"] == 30

    def test_auto_dream_at_interval(self):
        b = YvyraBridge(dream_every=5)
        out = None
        for _ in range(5):
            out = b.step(_coherent_scores())
        assert "dream" in out and out["dreamed"] is True


class TestPersistence:
    def test_save_load_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = YvyraBridge(save_dir=tmp)
            for _ in range(8):
                b.step(_coherent_scores())
            assert b.kernel.t == 8
            b.save("yv")

            b2 = YvyraBridge(save_dir=tmp)
            assert b2.kernel.t == 0
            b2.load("yv")
            assert b2.kernel.t == 8

    def test_save_without_dir_raises(self):
        b = YvyraBridge()
        with pytest.raises(ValueError):
            b.save()


class TestBehaviour:
    def test_psi_higher_for_coherent_than_noise(self):
        torch.manual_seed(0)
        pattern = [0.3, 0.8, 0.1, 0.2]
        bc = YvyraBridge()
        coherent = [bc.step([p + 0.01 * float(torch.randn(1)) for p in pattern])["psi"]
                    for _ in range(120)]
        bn = YvyraBridge()
        noise = [bn.step(torch.softmax(torch.randn(4), dim=-1).tolist())["psi"]
                 for _ in range(120)]
        assert sum(coherent[-30:]) / 30 > sum(noise[-30:]) / 30
