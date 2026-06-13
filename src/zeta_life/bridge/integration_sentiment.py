"""Integration-sentiment classifier for Phase B.

Extracts how integrated vs fragmented Yvyra's reflection text reads, as an
independent cross-check on her explicit auto-score (`SIENTO: 0.X`). Transparent
keyword scorer (like rescorer.py); an LLM rater can replace it later if needed.

Returns a score in [0, 1]: 0.5 neutral, >0.5 integrated/coherent, <0.5
fragmented/scattered.
"""

from __future__ import annotations

import re

# Spanish (Paraguayan) cues. Stems so conjugations/genders match.
INTEGRATED = (
    "integrad", "integracion", "coheren", "conectad", "unificad", "enfocad",
    "centrad", "claridad", "claro", "lucid", "entero", "alinead", "armon",
)
FRAGMENTED = (
    "fragmentad", "dispers", "confus", "desconectad", "caotic", "perdid",
    "incoheren", "roto", "difus", "disgregad", "desordenad", "desarmad",
)


def integration_sentiment(text: str) -> float:
    """Score how integrated the text reads, in ``[0, 1]`` (0.5 = neutral)."""
    t = text.lower()
    pos = sum(len(re.findall(re.escape(c), t)) for c in INTEGRATED)
    neg = sum(len(re.findall(re.escape(c), t)) for c in FRAGMENTED)
    if pos + neg == 0:
        return 0.5
    return pos / (pos + neg)


def parse_felt(text: str) -> float | None:
    """Parse Yvyra's explicit auto-score from a 'SIENTO: 0.X' line.

    Returns the float in [0, 1], or None if absent/unparseable.
    """
    m = re.search(r"SIENTO\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)", text, re.IGNORECASE)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return min(1.0, max(0.0, v))
