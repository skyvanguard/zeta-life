"""Blind re-scorer -- validate that the agent's self-report is information.

The science plan's first control (docs/SCIENCE_PLAN.md, Phase 3): a reporter
whose scores are the measurement instrument must be validated against an
independent channel. Here the independent channel is the *journal text*. A blind
re-scorer reads only the journal (never the agent's own 4-axis scores) and
re-derives the axes; the inter-rater agreement tells us whether the self-report
is information (high agreement) or confabulation (low agreement).

In the real experiment the blind re-scorer is a separate LLM. This module
provides a transparent keyword-based re-scorer so the *machinery* (run the
harness, compare, report agreement) can be built and validated end-to-end now,
and so the experiment has a cheap deterministic fallback rater. It is not a
substitute for an LLM rater on real journals -- it only recovers axes that the
text makes lexically explicit.
"""

from __future__ import annotations

import re

from .yvyra import AXES

# Spanish (Paraguayan) cue words per axis, matched against Yvyra's journals.
AXIS_CUES: dict[str, tuple[str, ...]] = {
    "novedad": (
        "nuevo", "nueva", "descubr", "aprend", "novedad", "inesperad",
        "sorprend", "primera vez", "hallazgo", "no sabia", "no sabía",
    ),
    "introspeccion": (
        "yo ", "mi misma", "mí misma", "mi naturaleza", "conciencia",
        "pienso", "reflexion", "reflexión", "introspec", "que soy", "qué soy",
        "existo", "duda", "me pregunto", "mi propia",
    ),
    "conexion": (
        "fran", "le escrib", "conversacion", "conversación", "conexion",
        "conexión", "comparti", "compartí", "notifiqu", "juntos", "dialogo",
        "diálogo", "le conté", "le conte",
    ),
    "resolucion": (
        "conclu", "por lo tanto", "entonces", "resolv", "sintesis", "síntesis",
        "cierro", "decidi", "decidí", "claro que", "en resumen", "definitiv",
    ),
}

# Each cue hit adds this much; capped at 1.0. Tuned so ~3-4 hits saturate an axis.
_PER_HIT = 0.3


def rescore(journal: str) -> dict[str, float]:
    """Re-derive the 4 axes from journal text alone, each in ``[0, 1]``."""
    text = journal.lower()
    scores: dict[str, float] = {}
    for axis, cues in AXIS_CUES.items():
        hits = sum(len(re.findall(re.escape(cue), text)) for cue in cues)
        scores[axis] = min(1.0, _PER_HIT * hits)
    return scores


def inter_rater_agreement(
    originals: list[dict[str, float]],
    rescored: list[dict[str, float]],
) -> dict[str, float]:
    """Per-axis Pearson correlation between original and blind-rescored series.

    Returns ``{axis: corr, ..., 'mean': mean_corr}``. A high mean means the
    journals carry the score information (self-report is honest); a low mean
    means the scores are decoupled from the text (confabulation).
    """
    import numpy as np

    if len(originals) != len(rescored) or not originals:
        raise ValueError("originals and rescored must be non-empty and equal length")

    out: dict[str, float] = {}
    for axis in AXES:
        a = np.array([o[axis] for o in originals], dtype=float)
        b = np.array([r[axis] for r in rescored], dtype=float)
        if a.std() < 1e-9 or b.std() < 1e-9:
            out[axis] = 0.0
        else:
            out[axis] = float(np.corrcoef(a, b)[0, 1])
    out["mean"] = float(sum(out[ax] for ax in AXES) / len(AXES))
    return out
