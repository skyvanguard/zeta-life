#!/usr/bin/env python3
"""yvyra_kernel.py -- the tick-driven entry point Yvyra's heartbeat calls.

Deployed to ``/opt/data/zeta/yvyra_kernel.py`` in Yvyra's container. Each
heartbeat tick is a fresh process: it loads the persisted kernel identity,
advances one step from the 4-axis self-report, logs the paired record, persists,
and prints a JSON line. Never invents a Psi: on any error it prints
``{"ok": false, ...}`` and exits 0 so the heartbeat logs it and moves on.

Usage (from the heartbeat, see ~/.hermes/HEARTBEAT.md):

    ZETA_LIFE_SRC=/opt/data/zeta/pysrc \\
    ZETA_MODE=silent \\
    python /opt/data/zeta/yvyra_kernel.py step "0.3,0.4,0.1,0.2"

    python /opt/data/zeta/yvyra_kernel.py state
    python /opt/data/zeta/yvyra_kernel.py dream

Environment:
    ZETA_LIFE_SRC  path to the zeta_life `src/` (prepended to sys.path)
    ZETA_DATA      data dir for checkpoint + log (default: dir of this file)
    ZETA_MODE      silent | feedback | sham  (default: silent -- Phase A)
    ZETA_IDENTITY  identity name for save/load (default: yvyra)

Modes (docs/SCIENCE_PLAN.md):
    silent   -- Phase A: real Psi is logged but NOT returned (psi=null in the
                JSON). Establishes the uncontaminated baseline.
    feedback -- Phase B: real Psi and suggestion returned.
    sham     -- placebo: a permuted past Psi is returned; the real one is logged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    src = os.environ.get("ZETA_LIFE_SRC")
    if src:
        sys.path.insert(0, src)


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def main(argv: list[str]) -> int:
    _bootstrap_path()
    try:
        from zeta_life.bridge import YvyraBridge
    except Exception as e:  # import failure -> never fabricate a result
        _emit({"ok": False, "error": f"import zeta_life failed: {e}"})
        return 0

    data_dir = Path(os.environ.get("ZETA_DATA", Path(__file__).resolve().parent))
    data_dir.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("ZETA_MODE", "silent")
    identity = os.environ.get("ZETA_IDENTITY", "yvyra")
    log_path = data_dir / "zeta_ticks.jsonl"

    if not argv:
        _emit({"ok": False, "error": "no command (expected: step <scores> | state | dream)"})
        return 0
    cmd = argv[0]

    try:
        bridge = YvyraBridge(mode=mode, save_dir=str(data_dir), log_path=str(log_path))
        # Restore identity if a prior tick saved one (tick-driven continuity).
        if bridge.kernel is not None:
            try:
                bridge.load(identity)
            except FileNotFoundError:
                pass  # first ever tick

        if cmd == "step":
            if len(argv) < 2:
                _emit({"ok": False, "error": "step requires scores '<nov>,<intro>,<con>,<res>'"})
                return 0
            try:
                vals = [float(x) for x in argv[1].split(",")]
            except ValueError as e:
                _emit({"ok": False, "error": f"bad scores: {e}"})
                return 0
            if len(vals) != 4:
                _emit({"ok": False, "error": f"expected 4 scores, got {len(vals)}"})
                return 0
            out = bridge.step(vals)
            bridge.save(identity)
            _emit({"ok": True, "tick": out["tick"], "mode": out["mode"],
                   "psi": out["psi"], "suggest": out["suggested_axis"],
                   "suggestion": out["suggestion"]})
            return 0

        if cmd == "state":
            _emit({"ok": True, **bridge.state()})
            return 0

        if cmd == "dream":
            _emit({"ok": True, "dream": bridge.dream()})
            return 0

        _emit({"ok": False, "error": f"unknown command: {cmd}"})
        return 0

    except Exception as e:  # any runtime failure -> honest error, no fake Psi
        _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
