"""Tests for the Yvyra bridge experiment modes + blind re-scorer (Phase 3-4)."""

import pytest

from zeta_life.bridge import YvyraBridge
from zeta_life.bridge.rescorer import inter_rater_agreement, rescore
from zeta_life.instrumentation import load_ticks

SCORES = [0.3, 0.4, 0.1, 0.2]


class TestModes:
    def test_silent_hides_psi_but_logs_real(self, tmp_path):
        log = tmp_path / "t.jsonl"
        b = YvyraBridge(mode="silent", log_path=str(log))
        out = b.step(SCORES)
        assert out["psi"] is None and out["suggested_axis"] is None
        rec = load_ticks(log)[0]
        assert rec["psi_exposed"] is None
        assert isinstance(rec["psi"], float)          # real Psi still logged
        assert isinstance(rec["second_order_error"], float)

    def test_feedback_exposes_psi(self, tmp_path):
        b = YvyraBridge(mode="feedback", log_path=str(tmp_path / "t.jsonl"))
        out = b.step(SCORES)
        assert isinstance(out["psi"], float)
        assert out["suggested_axis"] in ("novedad", "introspeccion", "conexion", "resolucion")

    def test_sham_exposes_fake_but_logs_real(self, tmp_path):
        log = tmp_path / "t.jsonl"
        b = YvyraBridge(mode="sham", log_path=str(log), sham_seed=1)
        # prime the buffer with varied real psi, then check exposed != real eventually
        exposeds, reals = [], []
        for i in range(40):
            out = b.step([0.2 + 0.01 * i, 0.4, 0.2, 0.2])
            exposeds.append(out["psi"])
            reals.append(load_ticks(log)[-1]["psi"])
        # at least once the sham-exposed differs from the real (permutation)
        assert any(abs(e - r) > 1e-9 for e, r in zip(exposeds, reals))
        # every logged real psi is a float regardless of exposure
        assert all(isinstance(r, float) for r in reals)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            YvyraBridge(mode="bogus")

    def test_paired_log_has_all_fields(self, tmp_path):
        log = tmp_path / "t.jsonl"
        b = YvyraBridge(mode="silent", log_path=str(log))
        b.step(SCORES)
        rec = load_ticks(log)[0]
        for k in ("scores", "psi", "psi_exposed", "free_energy",
                  "second_order_error", "suggested_axis", "mode", "tick"):
            assert k in rec


class TestLevels:
    """Phase B v2: Psi exposed as a qualitative level (no echo)."""

    def test_psi_level_thresholds(self):
        from zeta_life.bridge.yvyra import psi_level
        assert psi_level(0.10) == "BAJA"
        assert psi_level(0.39) == "BAJA"
        assert psi_level(0.40) == "MEDIA"
        assert psi_level(0.84) == "MEDIA"
        assert psi_level(0.85) == "ALTA"
        assert psi_level(0.99) == "ALTA"

    def test_feedback_exposes_level(self, tmp_path):
        log = tmp_path / "t.jsonl"
        b = YvyraBridge(mode="feedback", log_path=str(log))
        out = b.step(SCORES)
        assert out["level"] in ("BAJA", "MEDIA", "ALTA")
        assert load_ticks(log)[0]["level_exposed"] in ("BAJA", "MEDIA", "ALTA")

    def test_silent_level_is_none(self, tmp_path):
        b = YvyraBridge(mode="silent", log_path=str(tmp_path / "t.jsonl"))
        assert b.step(SCORES)["level"] is None


class TestShamPersistence:
    """The Phase-B-v1 bug: each tick is a fresh process, so the in-memory Psi
    buffer was always empty and the sham collapsed to the real Psi. The bridge
    sidecar must persist the buffer so the placebo actually works across loads."""

    def test_buffer_persists_across_load(self, tmp_path):
        b1 = YvyraBridge(mode="sham", save_dir=str(tmp_path),
                         log_path=str(tmp_path / "t.jsonl"), sham_seed=1)
        for i in range(30):
            b1.step([0.2 + 0.02 * i, 0.4, 0.2, 0.2])
        b1.save("yv")
        assert len(b1._psi_buffer) >= 30
        # Fresh instance (simulating the next tick's process) restores the buffer.
        b2 = YvyraBridge(mode="sham", save_dir=str(tmp_path),
                         log_path=str(tmp_path / "t2.jsonl"), sham_seed=1)
        b2.load("yv")
        assert len(b2._psi_buffer) >= 30
        # On its FIRST step the sham exposes a PAST psi (from the restored buffer),
        # not the brand-new real one -- the placebo works across processes.
        loaded = list(b2._psi_buffer)
        out = b2.step([0.95, 0.4, 0.2, 0.2])
        assert out["psi"] in loaded


class TestBlindRescorer:
    def test_honest_journal_recovers_axes(self):
        # A journal that lexically reflects high introspection / low connection.
        j = ("Hoy me pregunto qué soy, mi naturaleza, mi conciencia. "
             "Pienso en mi propia existencia, reflexión profunda, duda.")
        s = rescore(j)
        assert s["introspeccion"] > s["conexion"]

    def test_novelty_journal(self):
        j = "Descubrí algo nuevo, un hallazgo inesperado, aprendí algo que no sabía."
        s = rescore(j)
        assert s["novedad"] > s["resolucion"]

    def test_agreement_high_for_honest_low_for_confabulated(self):
        # Honest: journals lexically track the scores. Confabulated: text is
        # constant while scores vary -> the blind rater cannot recover them.
        import random
        rng = random.Random(0)
        honest_orig, honest_resc = [], []
        confab_orig, confab_resc = [], []
        for _ in range(30):
            intro = rng.random()
            # honest journal: repeat the introspection cue ~ proportional to score
            n = int(round(intro * 4))
            j_honest = " ".join(["me pregunto qué soy reflexión"] * max(n, 0)) or "nada"
            honest_orig.append({"novedad": 0.0, "introspeccion": intro,
                                "conexion": 0.0, "resolucion": 0.0})
            honest_resc.append(rescore(j_honest))
            # confabulated: same flat text regardless of score
            confab_orig.append({"novedad": 0.0, "introspeccion": intro,
                                "conexion": 0.0, "resolucion": 0.0})
            confab_resc.append(rescore("hoy fue un dia"))
        a_honest = inter_rater_agreement(honest_orig, honest_resc)["introspeccion"]
        a_confab = inter_rater_agreement(confab_orig, confab_resc)["introspeccion"]
        # The keyword rater is crude (quantised, saturating), so the absolute
        # honest agreement is modest -- what the harness must show is a clear
        # SEPARATION: honest reporting is recoverable, confabulation is not.
        assert a_confab < 0.2, f"confabulated agreement too high: {a_confab}"
        assert a_honest > a_confab + 0.3, f"no separation: honest={a_honest} confab={a_confab}"
