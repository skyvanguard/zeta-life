"""RSSMConsciousKernel — the kernel's faculties on a DreamerV3 RSSM world model.

§3.10 bounded the kernel's CartPole gap to its one-step world model: a reference
RSSM solves CartPole where the kernel plateaus. This is the integration: layer the
Conscious Kernel's *consciousness faculties* — persistent identity (self-model),
complementary (fast/slow) memory, zeta-rhythm dream consolidation, and the
integration index Psi — on top of the RSSM world model + controller
(``DreamerV3Agent``). The agent provides perception (recurrent posterior state),
world model (sequence-trained) and control (actor-critic in imagination); the
kernel's faculties run on the RSSM feature ``s = [h, z]`` without altering the
control path, so the agent still SOLVES CartPole while Psi stays live.

The faculty classes are reused unchanged (``SelfModel``, ``PredictionErrorEngine``,
``FastMemory``/``SlowMemory``, ``DreamEngine``, the Psi equations) — instantiated
over the RSSM feature space. The interoceptive/identity channel tracks the
softmax-normalised feature (matching the self-model's simplex design).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .dreamerv3_agent import DreamerV3Agent
from .self_model import SelfModel
from .prediction_error import PredictionErrorEngine
from .complementary_memory import Episode, FastMemory, SlowMemory
from .dream_engine import DreamEngine
from ..integration.formal_equations import compute_phi_c, compute_psi_hill


@dataclass
class RSSMStepResult:
    action: int
    action_onehot: Tensor
    psi: float
    free_energy: float


class RSSMConsciousKernel:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        embed_dim: int = 16,
        reflect_interval: int = 5,
        dream_interval: int = 200,
        alpha: float = 1.5,
        psi_fe_scale: float = 5.0,
        psi_prec_decay: float = 0.99,
        psi_hill_n: float = 4.0,
        psi_hill_K: float = 0.1,
        psi_w_prec: float = 1.0,
        psi_w_ref: float = 0.5,
        agent_kwargs: dict | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent = DreamerV3Agent(obs_dim, action_dim, **(agent_kwargs or {}))
        feat = self.agent.rssm.feat_dim

        # Reused faculty classes, instantiated over the RSSM feature space.
        self.self_model = SelfModel(state_dim=feat, embed_dim=embed_dim)
        self.error_engine = PredictionErrorEngine(2)        # perceptual, interoceptive
        self.fast_memory = FastMemory(capacity=500, surprise_threshold=0.3)
        self.slow_memory = SlowMemory(context_dim=feat, outcome_dim=feat)
        self.dream_engine = DreamEngine(self.fast_memory, self.slow_memory, self.self_model)
        self.replay = self.agent.replay                     # passthrough for the loop

        self.alpha = alpha
        self.psi_fe_scale = psi_fe_scale
        self.psi_prec_decay = psi_prec_decay
        self.psi_hill_n = psi_hill_n
        self.psi_hill_K = psi_hill_K
        self.psi_w_prec = psi_w_prec
        self.psi_w_ref = psi_w_ref
        self._prec_ref: float | None = None

        self.reflect_interval = reflect_interval
        self.dream_interval = dream_interval
        self.t = 0
        self.last_psi = 0.0
        self.last_free_energy = 0.0
        self._last_self_state = torch.zeros(feat)

    @property
    def feat_dim(self) -> int:
        return self.agent.rssm.feat_dim

    def reset_state(self) -> None:
        self.agent.reset_state()

    def _feature(self) -> Tensor:
        return self.agent.rssm.feat(self.agent._h, self.agent._z).squeeze(0).detach()

    # ------------------------------------------------------------------
    def act(self, obs: Tensor, greedy: bool = False) -> RSSMStepResult:
        """Pick an action (via the RSSM agent) and run the kernel's faculties."""
        self.t += 1
        a, a_oh = self.agent.act(obs, greedy=greedy)         # advances h,z; picks action
        s = self._feature()                                  # (feat,)
        self_state = F.softmax(s, dim=-1)                    # simplex over features
        recon = self.agent.rssm.decoder(s.unsqueeze(0)).squeeze(0).detach()
        self_pred = self.self_model.predict_self(self._last_self_state)  # (feat,) simplex

        predictions = {"perceptual": recon, "interoceptive": self_pred}
        observations = {"perceptual": obs, "interoceptive": self_state}
        errors = self.error_engine.compute_errors(predictions, observations)
        free_energy = self.error_engine.free_energy(errors)
        # Train the identity faculty from the interoceptive error; learn precisions.
        self.self_model.update_from_error(errors["interoceptive"]["raw"])
        self.error_engine.update_precisions(errors)

        # Complementary memory + consolidation.
        surprise = max(errors[ch]["magnitude"].item() for ch in self.error_engine.channels)
        self.fast_memory.store(Episode(
            stimulus=obs.detach(), observation=obs.detach(),
            archetype_state=self_state.detach(), surprise=surprise,
            dominant=f"f{int(self_state.argmax())}", timestamp=self.t,
            prediction_errors={ch: errors[ch]["magnitude"].item()
                               for ch in self.error_engine.channels}))
        self.slow_memory.integrate(self_state.detach(), self_state.detach())
        if self.t % self.reflect_interval == 0:
            self.self_model.reflect(self_state.detach(), depth=3)
        if self.t % self.dream_interval == 0 and len(self.fast_memory) > 0:
            self.dream_engine.dream_cycle(30)

        self.last_free_energy = free_energy.item()
        self.last_psi = self._compute_psi(self.last_free_energy)
        self._last_self_state = self_state.detach()
        return RSSMStepResult(a, a_oh, self.last_psi, self.last_free_energy)

    def observe(self, obs: Tensor, action_onehot: Tensor, reward: float,
                term: bool, first: bool) -> None:
        """Store the transition and train the RSSM + actor-critic."""
        self.agent.replay.add(obs, action_onehot, reward, 0.0 if term else 1.0, first)
        self.agent.train()

    # ------------------------------------------------------------------
    def _compute_psi(self, free_energy: float) -> float:
        """Integration index Psi over the RSSM state (same heuristic as the kernel)."""
        phi = 1.0 / (1.0 + self.psi_fe_scale * free_energy) + 0.2 * (len(self.fast_memory) / 500.0)
        prec_mean = float(self.error_engine.precisions.mean().item())
        if self._prec_ref is None:
            self._prec_ref = max(prec_mean, 1e-6)
        else:
            d = self.psi_prec_decay
            self._prec_ref = d * self._prec_ref + (1.0 - d) * prec_mean
        denom = prec_mean + self._prec_ref
        F_i = self.psi_w_prec * (prec_mean / denom if denom > 0 else 0.0)
        if self.self_model.reflection_history:
            last = self.self_model.reflection_history[-1]
            F_i += self.psi_w_ref * (1.0 / (1.0 + last[-1]["prediction_error"]))
        C = float(self.error_engine.recent_errors().mean().item()) / 5.0
        phi_c = compute_phi_c(F_i, self.alpha, C)
        return compute_psi_hill(phi, phi_c, self.psi_hill_n, self.psi_hill_K)
