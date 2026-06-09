"""Tests for the DreamerV3-style RSSM agent (rssm + sequence replay + actor-critic)."""

from __future__ import annotations

import math

import torch

from zeta_life.kernel.rssm import RSSM, symlog, symexp
from zeta_life.kernel.dreamerv3_agent import SequenceReplay, DreamerV3Agent


class TestSymlog:
    def test_inverse(self):
        x = torch.tensor([-5.0, -0.3, 0.0, 0.3, 5.0, 100.0])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)


class TestRSSM:
    def test_observe_shapes_and_finite_loss(self):
        torch.manual_seed(0)
        rssm = RSSM(obs_dim=4, action_dim=2, deter_dim=16, stoch_dim=8, hidden=16)
        B, T = 3, 6
        hs, zs, loss, metrics = rssm.observe(
            torch.randn(B, T, 4), torch.rand(B, T, 2),
            torch.rand(B, T), torch.ones(B, T))
        assert hs.shape == (B, T, 16) and zs.shape == (B, T, 8)
        assert math.isfinite(float(loss)) and loss.requires_grad
        assert all(math.isfinite(v) for v in metrics.values())

    def test_is_first_resets_state(self):
        torch.manual_seed(0)
        rssm = RSSM(obs_dim=4, action_dim=2, deter_dim=16, stoch_dim=8, hidden=16)
        B, T = 2, 5
        first = torch.zeros(B, T); first[:, 0] = 1.0; first[:, 3] = 1.0
        hs, zs, loss, _ = rssm.observe(
            torch.randn(B, T, 4), torch.rand(B, T, 2),
            torch.rand(B, T), torch.ones(B, T), is_first=first)
        assert hs.shape == (B, T, 16) and math.isfinite(float(loss))

    def test_img_step_shapes(self):
        rssm = RSSM(obs_dim=4, action_dim=2, deter_dim=16, stoch_dim=8, hidden=16)
        h, z = rssm.initial_state(5)
        h2, z2 = rssm.img_step(z, torch.rand(5, 2), h)
        assert h2.shape == (5, 16) and z2.shape == (5, 8)
        feat = rssm.feat(h2, z2)
        assert feat.shape == (5, 24)
        assert rssm.predict_reward(feat).shape == (5,)
        assert rssm.predict_continue(feat).shape == (5,)


class TestSequenceReplay:
    def test_add_sample_shapes(self):
        rb = SequenceReplay(capacity=1000)
        for t in range(100):
            rb.add(torch.randn(4), torch.rand(2), 1.0, 1.0, first=(t % 20 == 0))
        obs, act, rew, cont, first = rb.sample(batch=8, length=16)
        assert obs.shape == (8, 16, 4) and act.shape == (8, 16, 2)
        assert rew.shape == (8, 16) and cont.shape == (8, 16) and first.shape == (8, 16)


class TestDreamerV3Agent:
    def _fill(self, agent, n=600):
        agent.reset_state()
        obs = torch.randn(4)
        for t in range(n):
            a, a_oh = agent.act(obs)
            done = (t % 30 == 29)
            agent.replay.add(obs, a_oh, 1.0, 0.0 if done else 1.0, first=(t % 30 == 0))
            obs = torch.randn(4) if done else obs + 0.1 * torch.randn(4)
            if done:
                agent.reset_state()

    def test_act_returns_valid(self):
        agent = DreamerV3Agent(obs_dim=4, action_dim=2)
        agent.reset_state()
        a, a_oh = agent.act(torch.randn(4))
        assert a in (0, 1) and a_oh.shape == (2,)

    def test_train_finite_and_moves_params(self):
        torch.manual_seed(0)
        agent = DreamerV3Agent(obs_dim=4, action_dim=2, warmup=200,
                               seq_len=12, batch_size=8, horizon=8)
        self._fill(agent)
        before = [p.detach().clone() for p in agent.actor.parameters()]
        metrics = None
        for _ in range(3):
            metrics = agent.train()
        assert metrics is not None
        assert all(math.isfinite(v) for v in metrics.values())
        after = list(agent.actor.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_train_returns_none_before_warmup(self):
        agent = DreamerV3Agent(obs_dim=4, action_dim=2, warmup=500)
        assert agent.train() is None
