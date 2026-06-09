"""RSSM — recurrent state-space world model (DreamerV2/V3-style).

A reference implementation built to BOUND the Conscious Kernel's limitation: the
kernel's one-step GRU world model plateaus at ~33% of CartPole's ceiling and
collapses late. A proper recurrent state-space model trained on SEQUENCES, with a
learned reward and continue head, is the principled fix Dreamer uses. If this
solves CartPole where the kernel does not, the bottleneck was the world-model
architecture, not the active-inference design.

Honest scope: the stochastic latent is **Gaussian** with KL balancing + free nats
(DreamerV2-flavored), not the categorical / two-hot / symlog-everything stack of
DreamerV3 verbatim — but it is the recurrent-state + sequence-training + learned
reward core that the kernel lacked.

State: ``h`` (deterministic GRU recurrent state) + ``z`` (stochastic). The model
feature fed to every head is ``s = [h, z]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deter_dim: int = 128,
        stoch_dim: int = 32,
        hidden: int = 128,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
        kl_scale: float = 1.0,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.feat_dim = deter_dim + stoch_dim
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.min_std = min_std

        self.encoder = _mlp([obs_dim, hidden, hidden])              # obs -> embed
        self.img_in = _mlp([stoch_dim + action_dim, hidden])        # [z,a] -> gru input
        self.cell = nn.GRUCell(hidden, deter_dim)
        self.prior_net = _mlp([deter_dim, hidden, 2 * stoch_dim])    # h -> (mean,std)
        self.post_net = _mlp([deter_dim + hidden, hidden, 2 * stoch_dim])  # [h,embed]
        self.decoder = _mlp([self.feat_dim, hidden, obs_dim])
        self.reward_head = _mlp([self.feat_dim, hidden, 1])
        self.cont_head = _mlp([self.feat_dim, hidden, 1])           # logit of "continue"

    # ------------------------------------------------------------------
    def initial_state(self, batch: int) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        return (torch.zeros(batch, self.deter_dim, device=device),
                torch.zeros(batch, self.stoch_dim, device=device))

    def feat(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat([h, z], dim=-1)

    def _stats(self, net: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
        out = net(x)
        mean, std = out.chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        return mean, std

    def _recur(self, z: Tensor, a: Tensor, h: Tensor) -> Tensor:
        return self.cell(self.img_in(torch.cat([z, a], dim=-1)), h)

    def img_step(self, z: Tensor, a: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """One imagined step using the PRIOR (no observation)."""
        h = self._recur(z, a, h)
        mean, std = self._stats(self.prior_net, h)
        z = mean + std * torch.randn_like(std)
        return h, z

    def obs_step(
        self, z: Tensor, a: Tensor, h: Tensor, embed: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """One step incorporating an observation (posterior)."""
        h = self._recur(z, a, h)
        prior_mean, prior_std = self._stats(self.prior_net, h)
        post_mean, post_std = self._stats(self.post_net, torch.cat([h, embed], dim=-1))
        z = post_mean + post_std * torch.randn_like(post_std)
        return h, z, (prior_mean, prior_std), (post_mean, post_std)

    @staticmethod
    def _kl(qm: Tensor, qs: Tensor, pm: Tensor, ps: Tensor) -> Tensor:
        """KL(q || p) for diagonal Gaussians, summed over the last dim."""
        kl = (torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5)
        return kl.sum(-1)

    def observe(
        self, obs_seq: Tensor, act_seq: Tensor, reward_seq: Tensor,
        cont_seq: Tensor, is_first: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Train-time pass over a batch of sequences (B, T, ...).

        ``is_first`` (B, T) marks episode starts within a sampled sequence; the
        recurrent state is reset there (so flat cross-episode sequences are valid).
        Returns ``(hs, zs, loss, metrics)`` where ``hs, zs`` are the posterior
        states (B, T, ·) used as imagination starts, and ``loss`` is the total
        world-model loss (recon + reward + continue + balanced KL).
        """
        B, T, _ = obs_seq.shape
        embed = self.encoder(obs_seq)                                # (B,T,hidden)
        h, z = self.initial_state(B)
        a_prev = torch.zeros(B, self.action_dim, device=obs_seq.device)

        hs, zs = [], []
        kl_total = obs_seq.new_zeros(())
        for t in range(T):
            if is_first is not None:
                keep = (1.0 - is_first[:, t]).unsqueeze(-1)          # 0 at episode start
                h, z, a_prev = h * keep, z * keep, a_prev * keep
            h, z, (pm, ps), (qm, qs) = self.obs_step(z, a_prev, h, embed[:, t])
            hs.append(h)
            zs.append(z)
            kl_lhs = self._kl(qm.detach(), qs.detach(), pm, ps)      # train prior
            kl_rhs = self._kl(qm, qs, pm.detach(), ps.detach())      # train posterior
            kl = self.kl_balance * kl_lhs + (1 - self.kl_balance) * kl_rhs
            kl_total = kl_total + torch.clamp(kl, min=self.free_nats).mean()
            a_prev = act_seq[:, t]

        hs = torch.stack(hs, dim=1)                                  # (B,T,deter)
        zs = torch.stack(zs, dim=1)                                  # (B,T,stoch)
        feats = self.feat(hs, zs)                                    # (B,T,feat)

        recon = ((self.decoder(feats) - obs_seq) ** 2).sum(-1).mean()
        reward_pred = self.reward_head(feats).squeeze(-1)
        reward_loss = ((reward_pred - symlog(reward_seq)) ** 2).mean()
        cont_logit = self.cont_head(feats).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logit, cont_seq)
        kl_loss = kl_total / T

        loss = recon + reward_loss + cont_loss + self.kl_scale * kl_loss
        metrics = {
            "recon": recon.item(), "reward": reward_loss.item(),
            "cont": cont_loss.item(), "kl": kl_loss.item(),
        }
        return hs.detach(), zs.detach(), loss, metrics

    # heads for imagination ------------------------------------------------
    def predict_reward(self, feat: Tensor) -> Tensor:
        return symexp(self.reward_head(feat).squeeze(-1))

    def predict_continue(self, feat: Tensor) -> Tensor:
        return torch.sigmoid(self.cont_head(feat).squeeze(-1))
