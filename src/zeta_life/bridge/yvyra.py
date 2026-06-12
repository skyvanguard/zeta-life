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
import random
from collections import deque

import torch
import torch.nn.functional as F

from ..instrumentation import TickLogger
from ..kernel import ConsciousKernel

# Experiment modes (see docs/SCIENCE_PLAN.md):
#   silent   -- Phase A: kernel runs and logs, but Psi/suggestion are NOT
#               exposed to the agent. Establishes the uncontaminated baseline.
#   feedback -- Phase B: Psi and the suggestion are returned to the agent.
#   sham     -- placebo control: a permuted (fake) Psi is returned instead of
#               the real one; the real Psi is still logged for analysis.
MODES = ("silent", "feedback", "sham")

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
        mode: str = "feedback",
        log_path: str | None = None,
        sham_seed: int = 0,
        psi_fe_scale: float = 1.0,
    ) -> None:
        pref = torch.tensor(preference if preference is not None else DEFAULT_C,
                            dtype=torch.float32)
        if pref.numel() != 4:
            raise ValueError("preference must have 4 entries (the 4 axes)")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._C = (pref / pref.sum()).detach()
        self.mode = mode

        # Reactive kernel WITH the precision hyper-model, so each tick also
        # reports the second-order error over precision (epistemic depth) for
        # the science log. The world model learns on the ACTUAL experience;
        # auto-dreaming is off (the bridge controls dreaming per the contract).
        # psi_fe_scale default 1.0 (not the kernel's 5.0): Yvyra's free energy
        # runs higher than the bench, so scale 5 keeps phi-base subcritical and
        # Psi pins at 0. Under a FAITHFUL deployment simulation (load/save per
        # tick, exp_psi_recalibration.py), scale 1.0 maximises Psi variance
        # (std 0.45) while tracking coherence (corr +0.54).
        self.kernel = ConsciousKernel(
            obs_dim=4,
            action_mode="reactive",
            preference=self._C,
            efe_obs_norm="l1",   # faithful projection for the EFE suggestion
            dream_interval=10**9,
            precision_hypermodel=True,
            psi_fe_scale=psi_fe_scale,
        )
        self.save_dir = save_dir
        self.dream_every = dream_every
        self.score_ema = score_ema

        # Paired logging (Phase 0): scores + psi(real) + free_energy +
        # second_order + suggestion + mode, one record per tick, append-only.
        self._logger = TickLogger(log_path) if log_path is not None else None
        # Buffer of real Psi values for the sham control (temporal permutation).
        self._psi_buffer: deque[float] = deque(maxlen=200)
        self._sham_rng = random.Random(sham_seed)

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
        psi_real = result.psi

        # Decide what the AGENT sees, per experiment mode.
        if self.mode == "silent":
            psi_exposed: float | None = None
            axis_exposed: str | None = None
        elif self.mode == "sham":
            # Placebo: a Psi value sampled from past real values (temporal
            # permutation) -- same marginal distribution, no real-time link.
            psi_exposed = (self._sham_rng.choice(self._psi_buffer)
                           if self._psi_buffer else psi_real)
            axis_exposed = AXES[self._sham_rng.randrange(4)]
        else:  # feedback
            psi_exposed = psi_real
            axis_exposed = AXES[idx]
        self._psi_buffer.append(psi_real)

        # Paired log: the REAL signals always recorded, plus what was exposed.
        if self._logger is not None:
            self._logger.log({
                "scores": {AXES[i]: float(stim[i]) for i in range(4)},
                "psi": psi_real,
                "psi_exposed": psi_exposed,
                "free_energy": result.free_energy,
                "second_order_error": result.second_order_error,
                "suggested_axis": AXES[idx],       # real suggestion (for analysis)
                "gw_winner": None,
                "mode": self.mode,
            })

        out = {
            "tick": self.kernel.t,
            "psi": psi_exposed,
            "free_energy": result.free_energy,
            "errors": result.errors,
            "suggested_axis": axis_exposed,
            "suggestion": _SUGGESTION[axis_exposed] if axis_exposed else None,
            "action": {AXES[i]: action[i] for i in range(4)},
            "dreamed": result.dreamed,
            "mode": self.mode,
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
