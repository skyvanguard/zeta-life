"""Tests for the DreamerV3-style RSSM agent (rssm + sequence replay + actor-critic)."""

from __future__ import annotations

import math

import torch

from zeta_life.kernel.rssm import RSSM, symlog, symexp
from zeta_life.kernel.dreamerv3_agent import SequenceReplay, DreamerV3Agent
from zeta_life.kernel.rssm_kernel import RSSMConsciousKernel, RSSMStepResult


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
        assert math.isfinite(loss.detach().item()) and loss.requires_grad
        assert all(math.isfinite(v) for v in metrics.values())

    def test_is_first_resets_state(self):
        torch.manual_seed(0)
        rssm = RSSM(obs_dim=4, action_dim=2, deter_dim=16, stoch_dim=8, hidden=16)
        B, T = 2, 5
        first = torch.zeros(B, T); first[:, 0] = 1.0; first[:, 3] = 1.0
        hs, zs, loss, _ = rssm.observe(
            torch.randn(B, T, 4), torch.rand(B, T, 2),
            torch.rand(B, T), torch.ones(B, T), is_first=first)
        assert hs.shape == (B, T, 16) and math.isfinite(loss.detach().item())

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


class TestRSSMConsciousKernel:
    """Kernel faculties (identity, CLS memory, dream, Psi) on the RSSM agent."""

    def _kernel(self):
        return RSSMConsciousKernel(
            obs_dim=4, action_dim=2,
            agent_kwargs=dict(warmup=100, seq_len=12, batch_size=8, horizon=8))

    def test_act_returns_result_with_finite_psi(self):
        k = self._kernel(); k.reset_state()
        r = k.act(torch.randn(4))
        assert isinstance(r, RSSMStepResult)
        assert r.action in (0, 1) and r.action_onehot.shape == (2,)
        assert 0.0 <= r.psi <= 1.0 and math.isfinite(r.free_energy)

    def test_faculties_live_after_steps(self):
        torch.manual_seed(0)
        k = self._kernel(); k.reset_state()
        obs, first = torch.randn(4), True
        for t in range(150):
            r = k.act(obs)
            term = (t % 30 == 29)
            k.observe(obs, r.action_onehot, 1.0, term, first)
            first = False
            obs = torch.randn(4) if term else obs + 0.1 * torch.randn(4)
            if term:
                k.reset_state(); first = True
        assert len(k.fast_memory) > 0          # CLS memory active
        assert k.feat_dim == k.agent.rssm.feat_dim
        assert 0.0 <= k.last_psi <= 1.0         # integration index live

    def test_faculties_do_not_break_control_action(self):
        k = self._kernel(); k.reset_state()
        r = k.act(torch.randn(4), greedy=True)
        assert r.action in (0, 1)


class TestInSituRSSMKernel:
    """Canonical ConsciousKernel(world_model_type='rssm') — the in-situ fusion."""

    def test_default_kernel_is_gru(self):
        from zeta_life.kernel import ConsciousKernel
        k = ConsciousKernel()
        assert k.world_model_type == "gru" and k._rssm_agent is None

    def test_rssm_kernel_resizes_faculties_to_feature_space(self):
        from zeta_life.kernel import ConsciousKernel
        k = ConsciousKernel(obs_dim=4, action_dim=2, world_model_type="rssm",
                            rssm_kwargs=dict(warmup=100, seq_len=10, batch_size=6, horizon=6))
        feat = k._rssm_agent.rssm.feat_dim
        assert k.self_model.state_dim == feat
        assert k._last_self_state.shape == (feat,)

    def test_rssm_kernel_step_and_learn(self):
        from zeta_life.kernel import ConsciousKernel
        torch.manual_seed(0)
        k = ConsciousKernel(obs_dim=4, action_dim=2, world_model_type="rssm",
                            rssm_kwargs=dict(warmup=80, seq_len=10, batch_size=6, horizon=6))
        k.reset_rssm_state()
        obs, result = torch.randn(4), None
        for t in range(150):
            result = k.step(obs)
            k.learn_rssm(reward=1.0, done=(t % 30 == 29))
            obs = torch.randn(4) if (t % 30 == 29) else obs + 0.1 * torch.randn(4)
        assert result.action.shape == (2,)
        assert 0.0 <= result.psi <= 1.0 and math.isfinite(result.free_energy)
        assert len(k.fast_memory) > 0       # faculties live in the fused class

    def test_greedy_eval_runs(self):
        from zeta_life.kernel import ConsciousKernel
        k = ConsciousKernel(obs_dim=4, action_dim=2, world_model_type="rssm",
                            rssm_kwargs=dict(warmup=100))
        k.reset_rssm_state()
        r = k.step(torch.randn(4), greedy=True)
        assert r.action.shape == (2,)


class TestContinuousControl:
    """Continuous (tanh-Gaussian, value-gradient) actor for continuous control."""

    def test_agent_continuous_action_bounds(self):
        ag = DreamerV3Agent(obs_dim=3, action_dim=1, action_type="continuous", action_high=2.0)
        ag.reset_state()
        env_a, model_a = ag.act(torch.randn(3))
        assert env_a.shape == (1,) and model_a.shape == (1,)
        assert -2.001 <= float(env_a) <= 2.001       # scaled to env range
        assert -1.001 <= float(model_a) <= 1.001      # tanh model action

    def test_agent_continuous_trains(self):
        torch.manual_seed(0)
        ag = DreamerV3Agent(obs_dim=3, action_dim=1, action_type="continuous",
                            action_high=2.0, warmup=100, seq_len=10, batch_size=6, horizon=6)
        ag.reset_state()
        obs = torch.randn(3)
        for t in range(200):
            _, model_a = ag.act(obs)
            ag.replay.add(obs, model_a, -float((obs[0] - 1) ** 2), 1.0, first=(t % 40 == 0))
            obs = torch.randn(3) if (t % 40 == 39) else obs + 0.05 * torch.randn(3)
            if t % 40 == 39:
                ag.reset_state()
        m = ag.train()
        assert m is not None and all(math.isfinite(v) for v in m.values())

    def test_fused_kernel_continuous(self):
        from zeta_life.kernel import ConsciousKernel
        k = ConsciousKernel(obs_dim=3, action_dim=1, world_model_type="rssm",
                            rssm_kwargs=dict(action_type="continuous", action_high=2.0,
                                             warmup=100, seq_len=10, batch_size=6, horizon=6))
        k.reset_rssm_state()
        r = k.step(torch.randn(3))
        k.learn_rssm(reward=-0.5, done=False)
        assert r.action.shape == (1,) and -1.001 <= float(r.action) <= 1.001
        assert 0.0 <= r.psi <= 1.0

    def test_multidim_continuous_action(self):
        """Higher-dim continuous action (e.g. MuJoCo Reacher: 2-D action)."""
        ag = DreamerV3Agent(obs_dim=10, action_dim=2, action_type="continuous", action_high=1.0)
        ag.reset_state()
        env_a, model_a = ag.act(torch.randn(10))
        assert env_a.shape == (2,) and model_a.shape == (2,)
        assert bool((model_a.abs() <= 1.001).all())

    def test_fused_kernel_multidim_continuous(self):
        from zeta_life.kernel import ConsciousKernel
        k = ConsciousKernel(obs_dim=10, action_dim=2, world_model_type="rssm",
                            rssm_kwargs=dict(action_type="continuous", action_high=1.0,
                                             warmup=80, seq_len=8, batch_size=6, horizon=6))
        k.reset_rssm_state()
        r = k.step(torch.randn(10))
        k.learn_rssm(reward=-1.0, done=False)
        assert r.action.shape == (2,) and 0.0 <= r.psi <= 1.0
