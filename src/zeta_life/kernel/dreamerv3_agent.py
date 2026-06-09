"""DreamerV3-style agent — RSSM world model + actor-critic in imagination.

A self-contained reference agent (does NOT touch ConsciousKernel) used to bound
the kernel's CartPole limitation. Components:

- a flat sequence replay (DreamerV3-style: stores a stream with ``is_first`` flags
  so cross-episode windows are valid);
- the RSSM world model (``rssm.py``), trained on sampled length-L sequences;
- a categorical actor + critic trained purely in imagination from the posterior
  states, with lambda-returns, REINFORCE + entropy for the discrete action, an EMA
  target critic, and advantage normalisation.
"""

from __future__ import annotations

from collections import deque

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical

from .rssm import RSSM, _mlp


class SequenceReplay:
    """Flat transition stream; samples (B, L) windows with episode-start flags."""

    def __init__(self, capacity: int = 100000, seed: int = 20260608) -> None:
        self.obs: deque[Tensor] = deque(maxlen=capacity)
        self.act: deque[Tensor] = deque(maxlen=capacity)
        self.rew: deque[Tensor] = deque(maxlen=capacity)
        self.cont: deque[Tensor] = deque(maxlen=capacity)
        self.first: deque[float] = deque(maxlen=capacity)
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def add(self, obs: Tensor, act: Tensor, rew: float, cont: float, first: bool) -> None:
        self.obs.append(obs.detach().clone())
        self.act.append(act.detach().clone())
        self.rew.append(torch.tensor(float(rew)))
        self.cont.append(torch.tensor(float(cont)))
        self.first.append(1.0 if first else 0.0)

    def __len__(self) -> int:
        return len(self.obs)

    def sample(self, batch: int, length: int):
        n = len(self.obs)
        starts = torch.randint(0, n - length, (batch,), generator=self._rng)
        obs = torch.stack([torch.stack([self.obs[int(s) + t] for t in range(length)]) for s in starts])
        act = torch.stack([torch.stack([self.act[int(s) + t] for t in range(length)]) for s in starts])
        rew = torch.stack([torch.tensor([self.rew[int(s) + t] for t in range(length)]) for s in starts])
        cont = torch.stack([torch.tensor([self.cont[int(s) + t] for t in range(length)]) for s in starts])
        first = torch.stack([torch.tensor([self.first[int(s) + t] for t in range(length)]) for s in starts])
        return obs, act, rew, cont, first


class DreamerV3Agent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deter_dim: int = 128,
        stoch_dim: int = 32,
        hidden: int = 128,
        seq_len: int = 24,
        batch_size: int = 16,
        horizon: int = 15,
        gamma: float = 0.99,
        lam: float = 0.95,
        wm_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy: float = 3e-3,
        grad_clip: float = 100.0,
        critic_tau: float = 0.98,
        warmup: int = 500,
        action_type: str = "discrete",
        action_high: float = 1.0,
        min_std: float = 0.1,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.entropy = entropy
        self.grad_clip = grad_clip
        self.critic_tau = critic_tau
        self.warmup = warmup
        # Continuous control: a tanh-Gaussian actor trained by value gradients
        # (reparameterised actions backprop'd through the differentiable rollout),
        # vs the discrete categorical actor (REINFORCE). action_high scales the
        # tanh output to the env range (assumed symmetric [-high, high]).
        self.action_type = action_type
        self.action_high = action_high
        self.min_std = min_std

        self.rssm = RSSM(obs_dim, action_dim, deter_dim, stoch_dim, hidden)
        feat = self.rssm.feat_dim
        actor_out = 2 * action_dim if action_type == "continuous" else action_dim
        self.actor = _mlp([feat, hidden, actor_out])
        self.critic = _mlp([feat, hidden, 1])
        self.critic_target = _mlp([feat, hidden, 1])
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.wm_opt = torch.optim.Adam(self.rssm.parameters(), lr=wm_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = SequenceReplay()
        self._adv_std = 1.0
        self.reset_state()

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        self._h, self._z = self.rssm.initial_state(1)
        self._prev_a = torch.zeros(1, self.action_dim)

    def _actor_gaussian(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        """Continuous actor: (mean, std) of the pre-tanh Gaussian."""
        mean, log_std = self.actor(feat).chunk(2, dim=-1)
        return mean, F.softplus(log_std) + self.min_std

    @staticmethod
    def _gauss_entropy(std: Tensor) -> Tensor:
        """Differential entropy of a diagonal Gaussian, summed over action dims."""
        return (0.5 * (1.0 + math.log(2 * math.pi)) + std.log()).sum(-1)

    @torch.no_grad()
    def act(self, obs: Tensor, greedy: bool = False):
        """Step the posterior with the real obs, then pick an action.

        Returns ``(env_action, model_action)``: for discrete, (int, one-hot); for
        continuous, (scaled action tensor for the env, the [-1,1] tanh action used
        by the world model / replay).
        """
        embed = self.rssm.encoder(obs.unsqueeze(0))
        self._h, self._z, _, _ = self.rssm.obs_step(self._z, self._prev_a, self._h, embed)
        feat = self.rssm.feat(self._h, self._z)
        if self.action_type == "continuous":
            mean, std = self._actor_gaussian(feat)
            a_pre = mean if greedy else mean + std * torch.randn_like(std)
            a = torch.tanh(a_pre).squeeze(0)                 # (A,) in [-1, 1]
            self._prev_a = a.unsqueeze(0)
            return a * self.action_high, a                   # (env action, model action)
        logits = self.actor(feat)
        a = int(logits.argmax(-1).item()) if greedy else int(Categorical(logits=logits).sample().item())
        a_oh = F.one_hot(torch.tensor(a), self.action_dim).float()
        self._prev_a = a_oh.unsqueeze(0)
        return a, a_oh

    # ------------------------------------------------------------------
    def train(self) -> dict | None:
        if len(self.replay) < max(self.warmup, self.seq_len + 1):
            return None
        obs, act, rew, cont, first = self.replay.sample(self.batch_size, self.seq_len)
        hs, zs, wm_loss, metrics = self.rssm.observe(obs, act, rew, cont, first)
        self.wm_opt.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), self.grad_clip)
        self.wm_opt.step()
        beh = self._train_behavior(hs, zs)
        metrics.update(beh)
        return metrics

    def _lambda_returns(self, rewards, conts, values):
        """rewards/conts: list len H (transition i->i+1); values: list len H+1."""
        H = len(rewards)
        returns = [None] * H
        nxt = values[H]
        for i in reversed(range(H)):
            disc = self.gamma * conts[i]
            nxt = rewards[i] + disc * ((1 - self.lam) * values[i + 1] + self.lam * nxt)
            returns[i] = nxt
        return returns

    def _train_behavior(self, hs: Tensor, zs: Tensor) -> dict:
        # Flatten posterior states to imagination starts (N, ·).
        h = hs.reshape(-1, self.rssm.deter_dim)
        z = zs.reshape(-1, self.rssm.stoch_dim)
        H = self.horizon

        continuous = self.action_type == "continuous"
        feats = [self.rssm.feat(h, z)]
        logps, ents, rewards, conts = [], [], [], []
        for _ in range(H):
            feat = feats[-1]
            if continuous:
                mean, std = self._actor_gaussian(feat)
                a = torch.tanh(mean + std * torch.randn_like(std))   # reparam, grad
                ents.append(self._gauss_entropy(std))
                logps.append(None)
            else:
                dist = Categorical(logits=self.actor(feat))
                ai = dist.sample()
                logps.append(dist.log_prob(ai))
                ents.append(dist.entropy())
                a = F.one_hot(ai, self.action_dim).float()
            h, z = self.rssm.img_step(z, a, h)
            nf = self.rssm.feat(h, z)
            feats.append(nf)
            rewards.append(self.rssm.predict_reward(nf))
            conts.append(self.rssm.predict_continue(nf))

        with torch.no_grad():
            values_t = [self.critic_target(f).squeeze(-1) for f in feats]   # H+1
            det_returns = self._lambda_returns(
                [r.detach() for r in rewards], [c.detach() for c in conts], values_t)
            weights = [torch.ones_like(det_returns[0])]
            for i in range(H - 1):
                weights.append(weights[-1] * self.gamma * conts[i].detach())
            adv = [(det_returns[i] - values_t[i]) for i in range(H)]
            self._adv_std = 0.99 * self._adv_std + 0.01 * float(torch.stack(adv).std() + 1e-8)

        # Critic: regress live V(feat_i) to (detached) lambda-returns.
        values = [self.critic(feats[i].detach()).squeeze(-1) for i in range(H)]
        critic_loss = torch.stack(
            [((values[i] - det_returns[i]) ** 2) * weights[i] for i in range(H)]).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        # Actor.
        if continuous:
            # Value gradients: lambda-returns differentiable through the rewards
            # (which depend on the reparameterised actions) + detached bootstrap.
            grad_returns = self._lambda_returns(rewards, conts, values_t)
            obj = torch.stack([weights[i] * grad_returns[i] for i in range(H)]).mean()
            ent = torch.stack([weights[i] * ents[i] for i in range(H)]).mean()
            actor_loss = -(obj + self.entropy * ent)
        else:
            norm_adv = [a / max(self._adv_std, 1e-3) for a in adv]
            actor_loss = -torch.stack(
                [weights[i] * (logps[i] * norm_adv[i] + self.entropy * ents[i])
                 for i in range(H)]).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # EMA target critic.
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.mul_(self.critic_tau).add_((1.0 - self.critic_tau) * p)

        return {"critic": critic_loss.detach().item(), "actor": actor_loss.detach().item(),
                "ret": torch.stack(det_returns).mean().detach().item()}
