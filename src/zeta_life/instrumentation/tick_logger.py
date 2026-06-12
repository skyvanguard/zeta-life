"""TickLogger -- append-only paired logging, one JSON record per tick.

The science pipeline (see ``docs/SCIENCE_PLAN.md``) hinges on logging every
tick's signals *paired* and from the first tick, so the silent-phase baseline
is never lost. This logger is intentionally standalone: it does not touch the
kernel. The caller (an experiment, or the Yvyra bridge) builds a record dict and
hands it over; the logger stamps a monotonic ``tick`` index and appends one JSON
line.

Design choices
--------------
- **JSONL append-only**: crash-safe, streamable, greppable. Each line is a
  complete record; a partial final line (power loss mid-write) is simply
  dropped by :func:`load_ticks`.
- **Schema-flexible**: known fields (scores, psi, free_energy,
  second_order_error, gw_winner) are documented but the logger accepts any
  JSON-serialisable keys, so a field reserved now (e.g. second_order_error,
  null until the hyper-model lands in Phase 2) costs nothing.
- **No tensors**: the caller passes plain floats/lists. The logger refuses
  non-serialisable values loudly rather than silently dropping them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Documented (not enforced) keys, so callers and analysis stay in sync.
KNOWN_FIELDS = (
    'tick',                  # monotonic index, stamped by the logger
    'scores',                # dict[str, float] -- the 4 experiential axes (Yvyra)
    'psi',                   # float -- integration index
    'free_energy',           # float -- variational free energy / surprise proxy
    'second_order_error',    # float | None -- precision prediction error (Phase 2)
    'gw_winner',             # str | int | None -- global-workspace winner
    'mode',                  # str -- 'silent' | 'feedback' | 'sham' (Yvyra)
)


class TickLogger:
    """Append one JSON record per tick to a ``.jsonl`` file.

    Parameters
    ----------
    path : str | Path
        Destination ``.jsonl`` file. Parent directories are created.
    flush : bool
        Flush after every write (default True) so a crash keeps all prior
        ticks. Disable only for tight loops where the file is closed cleanly.
    """

    def __init__(self, path: str | Path, flush: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush = flush
        self._tick = 0
        # Append mode: a reused logger continues the tick count from the file
        # so restarts don't reset the baseline.
        if self.path.exists():
            self._tick = _count_lines(self.path)
        self._fh = open(self.path, 'a', encoding='utf-8')

    def log(self, record: dict[str, Any]) -> int:
        """Stamp a monotonic ``tick`` and append the record as one JSON line.

        The caller's ``record`` is copied; a ``tick`` key in it is overwritten
        by the logger's own counter (the logger owns tick numbering). Returns
        the tick index assigned to this record.

        Raises
        ------
        TypeError
            If the record is not JSON-serialisable (caught here, loudly, rather
            than corrupting the log).
        """
        stamped = {**record, 'tick': self._tick}
        try:
            line = json.dumps(stamped, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"TickLogger record is not JSON-serialisable (tick {self._tick}): {e}. "
                f"Pass plain floats/lists, not tensors."
            ) from e
        self._fh.write(line + '\n')
        if self._flush:
            self._fh.flush()
        self._tick += 1
        return stamped['tick']

    @property
    def tick(self) -> int:
        """Next tick index that :meth:`log` will assign."""
        return self._tick

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TickLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _count_lines(path: Path) -> int:
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in f)


def load_ticks(path: str | Path) -> list[dict[str, Any]]:
    """Load a ``.jsonl`` tick log into a list of records.

    A trailing partial line (e.g. from a crash mid-write) is skipped rather
    than raising, so a half-written final record never breaks analysis.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Trailing partial line from an interrupted write; drop it.
                continue
    return records
