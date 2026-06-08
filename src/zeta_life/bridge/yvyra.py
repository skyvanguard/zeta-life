"""YvyraBridge -- the zeta-life side of the Yvyra coupling.

Implements the semantic contract in ``docs/YVYRA_BRIDGE.md``:

- ENCODE: a live agent (Yvyra) scores each tick on 4 experiential axes in [0,1]
  -> a 4-D stimulus.
- STEP: ``kernel.step(stimulus)`` integrates the experience over time (Psi) and,
  separately, the bridge computes an EFE *suggestion* (toward the preferred
  character C) of which axis to lean into next.
- DREAM/SAVE: periodic consolidation, and (when ``save_dir`` is set) automatic
  persistence on the dream cadence, give continuity across restarts.

Design note -- why the kernel runs *reactive*, not in EFE mode:
The kernel's world model must learn ``experience(t) -> experience(t+1)`` so that
Psi reflects the *actual* temporal coherence of Yvyra's life. So the executed
"action" is the real experience (reactive: action = softmax(stimulus)), and the
EFE suggestion is computed read-only from the world model (``imagine``) without
hijacking what the world model trains on. The suggestion is advisory, exactly as
the contract specifies ("sugerencia, no orden").
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from ..kernel import ConsciousKernel

# The 4 experiential axes (ASCII keys for safe JSON; see the contract table).
AXES: tuple[str, str, str, str] = ("novedad", "introspeccion", "conexion", "resolucion")

# Default preferred character C (derived from Yvyra's SOUL): a curious,
# introspective thinker who does not settle (resolucion kept low on purpose).
DEFAULT_C: list[float] = [0.30, 0.40, 0.10, 0.20]

# Human-readable suggestion verbs per axis.
_SUGGESTION = {
    "novedad": "busca novedad (material/ideas nuevas)",
    "introspeccion": "profundiza la introspeccion",
    "conexion": "busca conexion con Fran",
    "resolucion": "busca cerrar / sintetizar",
}


class YvyraBridge:
    """Couples a live agent's self-report to a ConsciousKernel.

    Parameters
    ----------
    preference : list[float] | None
        The preferred character C over the 4 axes (defaults to ``DEFAULT_C``).
        Normalised internally to a distribution.
    save_dir : str | None
        Directory for ``save()``/``load()`` persistence.
    dream_every : int
        Run a consolidation dream every this many ticks (0 disables).
    score_ema : float
        EMA decay for the "recent experience" summary reported by ``state()``.
    """

    def __init__(
        self,
        preference: list[float] | None = None,
        save_dir: str | None = None,
        dream_every: int = 20,
        score_ema: float = 0.8,
    ) -> None:
        pref = torch.tensor(preference if preference is not None else DEFAULT_C,
                            dtype=torch.float32)
        if pref.numel() != 4:
            raise ValueError("preference must have 4 entries (the 4 axes)")
        self._C = (pref / pref.sum()).detach()

        # Reactive kernel: the world model learns on the ACTUAL experience.
        # Auto-dreaming is disabled here; the bridge controls dreaming per the
        # contract ("cada N ticks: kernel.dream()").
        self.kernel = ConsciousKernel(
            obs_dim=4,
            action_mode="reactive",
            preference=self._C,
            efe_obs_norm="l1",   # faithful projection for the EFE suggestion
            dream_interval=10**9,
        )
        self.save_dir = save_dir
        self.dream_every = dream_every
        self.score_ema = score_ema

        self._last: dict | None = None
        self._recent_scores: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Core: one tick
    # ------------------------------------------------------------------

    def step(self, scores) -> dict:
        """Advance one tick from the agent's 4-axis self-report.

        Parameters
        ----------
        scores : dict[str, float] | sequence[float]
            Either ``{axis: score}`` for the 4 axes or a length-4 sequence, each
            score in ``[0, 1]``.

        Returns
        -------
        dict
            JSON-serialisable: ``tick``, ``psi``, ``free_energy``, per-axis
            ``errors``, ``suggested_axis`` + ``suggestion``, the ``action``
            distribution over axes, ``dreamed``, and ``dream`` (when one ran).
        """
        stim = self._encode(scores)
        result = self.kernel.step(stim)

        # Track an EMA of the raw scores for the state() summary.
        if self._recent_scores is None:
            self._recent_scores = stim.clone()
        else:
            self._recent_scores = (
                self.score_ema * self._recent_scores + (1 - self.score_ema) * stim
            )

        idx = self._suggest_axis()
        action = result.action.tolist()
        out = {
            "tick": self.kernel.t,
            "psi": result.psi,
            "free_energy": result.free_energy,
            "errors": result.errors,
            "suggested_axis": AXES[idx],
            "suggestion": _SUGGESTION[AXES[idx]],
            "action": {AXES[i]: action[i] for i in range(4)},
            "dreamed": result.dreamed,
        }
        if self.dream_every and self.kernel.t % self.dream_every == 0:
            out["dream"] = self.dream()
            out["dreamed"] = True
            # Persist on the dream cadence (the contract's "cada N ticks:
            # dream() + save()") when a save_dir is configured.
            if self.save_dir is not None:
                self.save()
                out["saved"] = True
        self._last = out
        return out

    # ------------------------------------------------------------------
    # EFE suggestion (read-only; does not affect world-model training)
    # ------------------------------------------------------------------

    def _suggest_axis(self) -> int:
        """Axis whose imagined outcome best moves experience toward C.

        Scored by the kernel's OWN EFE cost (``_efe_cost``), so it uses the same
        observation normalisation as the planner -- no second, divergent scoring.
        One-hot candidates are appropriate here: we recommend an AXIS to lean
        into, not a continuous action. Read-only (imagine does not mutate state).
        """
        costs = [self.kernel._efe_cost(F.one_hot(torch.tensor(i), 4).float())
                 for i in range(4)]
        return int(min(range(4), key=lambda i: costs[i]))

    # ------------------------------------------------------------------
    # State / dream / persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """Summary of the bridge/kernel state (JSON-serialisable)."""
        recent = (
            {AXES[i]: float(self._recent_scores[i]) for i in range(4)}
            if self._recent_scores is not None else None
        )
        return {
            "tick": self.kernel.t,
            "psi": self._last["psi"] if self._last else 0.0,
            "preference": {AXES[i]: float(self._C[i]) for i in range(4)},
            "recent_experience": recent,
            "suggested_axis": self._last["suggested_axis"] if self._last else None,
        }

    def dream(self, duration: int = 30) -> dict:
        """Run one consolidation dream cycle and return its summary."""
        report = self.kernel.dream_engine.dream_cycle(duration)
        return {
            "duration": report.duration,
            "transfers": report.transfers,
            "replays": report.replays,
            "identity_updated": report.identity_updated,
        }

    def save(self, name: str = "yvyra") -> None:
        if self.save_dir is None:
            raise ValueError("save_dir was not set")
        self.kernel.save(self.save_dir, name)

    def load(self, name: str = "yvyra") -> None:
        if self.save_dir is None:
            raise ValueError("save_dir was not set")
        self.kernel.load(self.save_dir, name)

    # ------------------------------------------------------------------
    # JSON string API (for the tool/skill wiring in Yvyra's container)
    # ------------------------------------------------------------------

    def step_json(self, payload: str) -> str:
        """``kernel_step``: JSON in (scores), JSON out (the step result)."""
        return json.dumps(self.step(json.loads(payload)))

    def state_json(self) -> str:
        """``kernel_state``: JSON out."""
        return json.dumps(self.state())

    def dream_json(self) -> str:
        """``kernel_dream``: JSON out."""
        return json.dumps(self.dream())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(self, scores) -> torch.Tensor:
        if isinstance(scores, dict):
            try:
                vals = [float(scores[a]) for a in AXES]
            except KeyError as e:
                raise ValueError(f"missing axis in scores: {e}") from e
        else:
            vals = [float(x) for x in scores]
            if len(vals) != 4:
                raise ValueError(f"expected 4 scores, got {len(vals)}")
        return torch.tensor(vals, dtype=torch.float32)
