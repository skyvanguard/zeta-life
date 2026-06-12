"""Tests for the TickLogger paired-logging instrumentation (Phase 0)."""

import json

import pytest

from zeta_life.instrumentation import TickLogger, load_ticks


class TestBasicLogging:
    def test_writes_one_line_per_log(self, tmp_path):
        p = tmp_path / 'ticks.jsonl'
        with TickLogger(p) as log:
            log.log({'psi': 0.5})
            log.log({'psi': 0.6})
        lines = p.read_text(encoding='utf-8').strip().split('\n')
        assert len(lines) == 2

    def test_stamps_monotonic_tick(self, tmp_path):
        p = tmp_path / 'ticks.jsonl'
        with TickLogger(p) as log:
            t0 = log.log({'psi': 0.1})
            t1 = log.log({'psi': 0.2})
        assert (t0, t1) == (0, 1)
        recs = load_ticks(p)
        assert [r['tick'] for r in recs] == [0, 1]

    def test_logger_owns_tick_numbering(self, tmp_path):
        # A 'tick' supplied by the caller is overwritten by the logger.
        p = tmp_path / 'ticks.jsonl'
        with TickLogger(p) as log:
            log.log({'tick': 999, 'psi': 0.1})
        assert load_ticks(p)[0]['tick'] == 0

    def test_round_trip_preserves_fields(self, tmp_path):
        p = tmp_path / 'ticks.jsonl'
        rec = {'scores': {'novedad': 0.3, 'introspeccion': 0.7},
               'psi': 0.42, 'free_energy': 1.1,
               'second_order_error': None, 'gw_winner': 'perceptual',
               'mode': 'silent'}
        with TickLogger(p) as log:
            log.log(rec)
        loaded = load_ticks(p)[0]
        for k, v in rec.items():
            assert loaded[k] == v


class TestPersistenceAcrossRestart:
    def test_append_continues_tick_count(self, tmp_path):
        p = tmp_path / 'ticks.jsonl'
        with TickLogger(p) as log:
            log.log({'psi': 0.1})
            log.log({'psi': 0.2})
        # Reopen: must continue from tick 2, not reset to 0.
        with TickLogger(p) as log2:
            assert log2.tick == 2
            t = log2.log({'psi': 0.3})
        assert t == 2
        assert [r['tick'] for r in load_ticks(p)] == [0, 1, 2]


class TestRobustness:
    def test_non_serialisable_raises_loudly(self, tmp_path):
        import torch
        p = tmp_path / 'ticks.jsonl'
        with TickLogger(p) as log:
            with pytest.raises(TypeError, match='not JSON-serialisable'):
                log.log({'psi': torch.tensor(0.5)})

    def test_partial_final_line_is_skipped(self, tmp_path):
        p = tmp_path / 'ticks.jsonl'
        with TickLogger(p) as log:
            log.log({'psi': 0.1})
        # Simulate a crash mid-write: append a truncated JSON line.
        with open(p, 'a', encoding='utf-8') as f:
            f.write('{"psi": 0.2, "tic')  # no newline, broken
        recs = load_ticks(p)
        assert len(recs) == 1
        assert recs[0]['psi'] == 0.1

    def test_blank_lines_ignored(self, tmp_path):
        p = tmp_path / 'ticks.jsonl'
        p.write_text('{"a": 1}\n\n{"a": 2}\n', encoding='utf-8')
        assert len(load_ticks(p)) == 2

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / 'nested' / 'deep' / 'ticks.jsonl'
        with TickLogger(p) as log:
            log.log({'psi': 0.1})
        assert p.exists()
