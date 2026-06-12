"""PrecisionHyperModel -- epistemic depth for the Conscious Kernel.

Implements the missing third condition of Laukkonen, Friston & Chandaria's
"A beautiful loop" (2025): a hyper-model that **predicts its own precisions
globally and feeds the prediction back**, recursively. The signal that matters
is the *second-order prediction error over precision* -- the system being
surprised by its own confidence.

Why this is not what the kernel already has
-------------------------------------------
``PredictionErrorEngine.update_precisions`` adapts each channel's precision
toward its realised inverse-error-variance ``D / ||raw||^2``. That is **local,
reactive, and non-predictive** -- the paper's "dimmer switches in isolation".
This module turns those three absences into presences:

- **Predictive**: it predicts *next* tick's log-precisions before the error is
  seen, rather than estimating the error that just happened.
- **Global / non-local**: a single recurrent latent (a persistent ``GRUCell``
  hidden state) conditions all channels jointly, so each channel's predicted
  precision depends on the system-wide state.
- **Recurrent (ad infinitum)**: the hidden state persists across ticks; the
  predicted precision biases binding, the realised error yields the empirical
  precision, the hyper-model updates from the mismatch, and predicts again.

The realised target ``log(D / ||raw||^2)`` is exactly the optimum that
``update_precisions`` already chases -- so this module learns to predict, one
tick ahead, the quantity the local rule estimates after the fact.

Honesty: this is active-inference engineering, not consciousness. The paper's
three conditions are *necessary, not sufficient*, and epistemic depth is not a
verbal property. The second-order error is reported *alongside* Psi, never as a
replacement, until the bench decides which discriminates better.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Channels whose error is below this norm carry no signal; their empirical
# precision (D/||raw||^2) would be unbounded, so they are masked out of the
# second-order error rather than dominating it.
_NO_SIGNAL = 1e-6
# Clamp realised log-precision so a near-zero error doesn't produce +inf targets.
_LOGPREC_CLAMP = 20.0


class PrecisionHyperModel(nn.Module):
    """Predict per-channel log-precisions from a global recurrent latent.

    Parameters
    ----------
    n_channels : int
        Number of error channels (matches the PredictionErrorEngine).
    hidden : int
        Size of the recurrent latent (the "epistemic field" of precision).
    hyper_lr : float
        Learning rate for the hyper-model's own optimiser.
    """

    def __init__(self, n_channels: int = 4, hidden: int = 32, hyper_lr: float = 0.01) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        # Input is the per-channel context (recent error magnitudes); the GRU's
        # persistent hidden state carries the global, recurrent memory.
        self.gru = nn.GRUCell(n_channels, hidden)
        self.to_logprec = nn.Linear(hidden, n_channels)
        self.opt = torch.optim.Adam(
            list(self.gru.parameters()) + list(self.to_logprec.parameters()),
            lr=hyper_lr,
        )
        # Recurrent state and last prediction live outside the parameter set.
        self._h = torch.zeros(1, hidden)
        self._last_pred: Tensor | None = None

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Clear the recurrent latent (episode boundary)."""
        self._h = torch.zeros(1, self.hidden)
        self._last_pred = None

    # ------------------------------------------------------------------
    def predict(self, context: Tensor) -> Tensor:
        """Predict next tick's log-precisions; advances the recurrent latent.

        Parameters
        ----------
        context : Tensor
            Global context of shape ``(n_channels,)`` (e.g. recent per-channel
            error magnitudes). Detached internally -- the hyper-model trains on
            its prediction error, not on the kernel's graph.
        """
        ctx = context.detach().reshape(1, self.n_channels)
        self._h = self.gru(ctx, self._h)
        pred = self.to_logprec(self._h).squeeze(0)
        self._last_pred = pred
        return pred

    # ------------------------------------------------------------------
    @staticmethod
    def realised_logprec(errors: dict[str, dict[str, Tensor]],
                         channels: list[str]) -> tuple[Tensor, Tensor]:
        """Empirical log-precision ``log(D / ||raw||^2)`` per channel + a mask.

        Returns ``(logprec, mask)`` where ``mask[i]`` is False for channels with
        no signal (their empirical precision is unbounded).
        """
        vals, mask = [], []
        for ch in channels:
            raw = errors[ch]['raw'].detach()
            ss = float(torch.sum(raw ** 2))
            D = raw.numel()
            if ss < _NO_SIGNAL or D == 0:
                vals.append(0.0)
                mask.append(False)
            else:
                lp = torch.log(torch.tensor(D / ss))
                vals.append(float(torch.clamp(lp, -_LOGPREC_CLAMP, _LOGPREC_CLAMP)))
                mask.append(True)
        return torch.tensor(vals), torch.tensor(mask, dtype=torch.bool)

    # ------------------------------------------------------------------
    def update(self, realised: Tensor, mask: Tensor) -> float:
        """Train on the second-order error and return its (detached) magnitude.

        The second-order error is ``predicted_logprec - realised_logprec`` over
        the channels that carry signal. Returns ``||error||`` for logging, or
        ``0.0`` when there is nothing to learn from.
        """
        if self._last_pred is None or not bool(mask.any()):
            return 0.0
        diff = (self._last_pred - realised)[mask]
        loss = torch.sum(diff ** 2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # Truncate BPTT: next tick starts from a detached latent.
        self._h = self._h.detach()
        with torch.no_grad():
            mag = float(torch.norm((self._last_pred.detach() - realised)[mask]))
        return mag

    # ------------------------------------------------------------------
    # Persistence (mirrors DreamerV3Agent.state_dict / load_state_dict)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:  # type: ignore[override]
        return {
            'gru': self.gru.state_dict(),
            'to_logprec': self.to_logprec.state_dict(),
            'opt': self.opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:  # type: ignore[override]
        self.gru.load_state_dict(state['gru'])
        self.to_logprec.load_state_dict(state['to_logprec'])
        self.opt.load_state_dict(state['opt'])
        self.reset_state()
