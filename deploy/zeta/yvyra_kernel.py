#!/usr/bin/env python3
"""yvyra_kernel.py -- the tick-driven entry point Yvyra's heartbeat calls.

Deployed to ``/opt/data/zeta/yvyra_kernel.py`` in Yvyra's container (which is the
host ``~/.hermes/zeta/`` via the ``~/.hermes:/opt/data`` bind mount). Each
heartbeat tick is a fresh process: it loads the persisted kernel identity,
advances one step from the 4-axis self-report, logs the paired record, persists,
and prints a JSON line. Never invents a Psi: on any error it prints
``{"ok": false, ...}`` and exits 0 so the heartbeat logs it and moves on.

Subcommands:
  step "<nov>,<intro>,<con>,<res>"   advance one tick; print JSON {psi, suggest, ...}
  state                              print current state without advancing
  dream                              run a consolidation dream
  warmup [N]                         mature the kernel with N ticks (default 600)

Usage (from ~/.hermes/HEARTBEAT.md, run inside the container):

    ZETA_LIFE_SRC=/opt/data/zeta/pysrc /opt/data/zeta/venv/bin/python \\
      /opt/data/zeta/yvyra_kernel.py step "0.3,0.4,0.1,0.2"

Environment:
    ZETA_LIFE_SRC  path to the zeta_life src tree (prepended to sys.path)
    ZETA_DATA      data dir for checkpoint + paired log (default: <script dir>/state)
    ZETA_MODE      silent | feedback | sham  (default: silent -- Phase A)
    ZETA_IDENTITY  identity name for save/load (default: yvyra)

Modes (docs/SCIENCE_PLAN.md):
    silent   -- Phase A: real Psi is logged but NOT returned (psi=null). The
                uncontaminated baseline. THIS IS THE DEFAULT.
    feedback -- Phase B: real Psi and suggestion returned.
    sham     -- placebo: a permuted past Psi returned; the real one logged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PREFERENCE = [0.30, 0.40, 0.10, 0.20]  # Yvyra's character C (see YVYRA_BRIDGE.md)


def _bootstrap_path() -> None:
    src = os.environ.get("ZETA_LIFE_SRC")
    if src and (Path(src) / "zeta_life").is_dir():
        sys.path.insert(0, src)


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _data_dir() -> Path:
    d = Path(os.environ.get("ZETA_DATA", Path(__file__).resolve().parent / "state"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv: list[str]) -> int:
    _bootstrap_path()
    try:
        from zeta_life.bridge import YvyraBridge
    except Exception as e:  # import failure -> never fabricate a result
        _emit({"ok": False, "error": f"import zeta_life failed: {e}"})
        return 0

    data_dir = _data_dir()
    mode = os.environ.get("ZETA_MODE", "silent")
    identity = os.environ.get("ZETA_IDENTITY", "yvyra")
    log_path = data_dir / "zeta_ticks.jsonl"

    if not argv:
        _emit({"ok": False, "error": "no command (expected: step <scores> | state | dream | warmup [N])"})
        return 0
    cmd = argv[0]

    try:
        # Warmup uses SYNTHETIC scores to mature the checkpoint -- it must NOT
        # write to the science log, which holds only Yvyra's real experience.
        warmup_log = None if cmd == "warmup" else str(log_path)
        bridge = YvyraBridge(preference=PREFERENCE, mode=mode,
                             save_dir=str(data_dir), log_path=warmup_log)
        try:
            bridge.load(identity)
        except FileNotFoundError:
            pass  # first ever tick / fresh warmup

        if cmd == "warmup":
            import random
            n = int(argv[1]) if len(argv) > 1 else 600
            rng = random.Random(0)
            mood = list(PREFERENCE)
            for _ in range(n):
                mood = [min(1.0, max(0.0, m + 0.1 * (rng.random() - 0.5))) for m in mood]
                bridge.step(mood)
            bridge.save(identity)
            # Report the last real Psi (the bridge buffers them even in silent).
            real_psi = bridge._psi_buffer[-1] if bridge._psi_buffer else None
            _emit({"ok": True, "warmup": n, "tick": bridge.kernel.t, "psi": real_psi})
            return 0

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
                   "psi": out["psi"], "psi_real": out.get("psi_real"),
                   "suggest": out["suggested_axis"], "suggestion": out["suggestion"]})
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
