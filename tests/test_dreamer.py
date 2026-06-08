"""Tests for Dreamer-style amortized planning (imagine_grad, Actor/Critic, kernel)."""

from __future__ import annotations

import statistics as st

import pytest
import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.world_model import WorldModel
from zeta_life.kernel.policy import Actor, Critic
from zeta_life.kernel.replay import ReplayBuffer

TARGET = torch.tensor([0.7, 0.1, 0.1, 0.1])
TARGET = TARGET / TARGET.sum()


class TestImagineGrad:
    def test_returns_grad_and_shapes(self):
        wm = WorldModel()
        z = torch.zeros(3, 32)
        a = torch.softmax(torch.randn(3, 4), dim=-1)
        nz, pred = wm.imagine_grad(z, a)
        assert nz.shape == (3, 32) and pred.shape == (3, 4)
        assert nz.requires_grad and pred.requires_grad

    def test_does_not_mutate_latent_state(self):
        wm = WorldModel()
        before = wm.latent_state.clone()
        wm.imagine_grad(torch.zeros(1, 32), torch.softmax(torch.randn(1, 4), dim=-1))
        assert torch.allclose(wm.latent_state, before)

    def test_imagine_remains_nograd(self):
        wm = WorldModel()
        preds = wm.imagine([torch.softmax(torch.randn(4), dim=-1)])
        assert preds[0].requires_grad is False


class TestPolicy:
    def test_actor_returns_simplex(self):
        a = Actor(32, 4)(torch.randn(5, 32))
        assert a.shape == (5, 4)
        assert torch.allclose(a.sum(-1), torch.ones(5), atol=1e-5)
        assert (a >= 0).all()

    def test_critic_returns_scalar_per_item(self):
        v = Critic(32)(torch.randn(5, 32))
        assert v.shape == (5,)


class TestDreamerKernel:
    def test_default_has_no_actor(self):
        ck = ConsciousKernel()
        assert ck.actor is None and ck.critic is None and ck._replay is None

    def test_dreamer_steps_valid_action(self):
        ck = ConsciousKernel(action_mode="dreamer", preference=TARGET)
        result = None
        for _ in range(10):
            result = ck.step(torch.rand(4))
        assert result.action.shape == (4,)
        assert float(result.action.sum()) == pytest.approx(1.0, abs=1e-4)

    def test_lambda_returns_shapes(self):
        ck = ConsciousKernel(action_mode="dreamer", preference=TARGET)
        rewards = [torch.zeros(3) for _ in range(4)]
        values = [torch.ones(3) for _ in range(5)]
        rets = ck._lambda_returns(rewards, values, 0.97, 0.95)
        assert len(rets) == 4 and rets[0].shape == (3,)

    def test_actor_params_move_under_training(self):
        torch.manual_seed(0)
        ck = ConsciousKernel(action_mode="dreamer", preference=TARGET, actor_explore=0.3)
        before = [p.detach().clone() for p in ck.actor.parameters()]
        for t in range(60):
            ck.step(torch.rand(4))
            if t < 30:
                a = torch.rand(4)
                ck.last_action = (a / a.sum()).detach()
        after = list(ck.actor.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))


class TestDreamerControl:
    def test_amortized_actor_reaches_target(self):
        """The imagination-trained actor drives the state to a non-vertex target."""
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        state = torch.rand(4, generator=g)
        state = state / state.sum()
        ck = ConsciousKernel(action_mode="dreamer", preference=TARGET,
                             actor_explore=0.3, reflect_interval=10**9,
                             dream_interval=10**9)
        obs = state.clone()
        dists = []
        for t in range(500):
            result = ck.step(obs)
            if t < 250:
                a = torch.rand(4); a = a / a.sum()
                ck.last_action = a.detach()
            else:
                a = result.action
            state = ((1 - 0.3) * state + 0.3 * a.detach()).clamp(min=1e-6)
            state = state / state.sum()
            obs = state
            dists.append(float(torch.linalg.vector_norm(state - TARGET)))
        assert st.mean(dists[-60:]) < 0.2


class TestReplayBuffer:
    def test_add_len_and_capacity(self):
        rb = ReplayBuffer(capacity=3)
        for i in range(5):
            rb.add(torch.tensor([float(i)]), torch.tensor([0.0]), torch.tensor([float(i + 1)]))
        assert len(rb) == 3  # capacity bound

    def test_sample_shapes(self):
        rb = ReplayBuffer(capacity=100)
        for _ in range(10):
            rb.add(torch.rand(4), torch.rand(2), torch.rand(4))
        obs, act, nxt = rb.sample(8)
        assert obs.shape == (8, 4) and act.shape == (8, 2) and nxt.shape == (8, 4)

    def test_kernel_dreamer_populates_replay(self):
        ck = ConsciousKernel(action_mode="dreamer", preference=TARGET)
        for _ in range(12):
            ck.step(torch.rand(4))
        assert ck._replay is not None and len(ck._replay) > 0


class TestDecoupledAction:
    """action_dim decoupled from obs_dim + neg_distance regulation reward."""

    def test_default_action_dim_equals_obs(self):
        ck = ConsciousKernel()
        assert ck.action_dim == ck.obs_dim == 4

    def test_decoupled_dreamer_action_shape(self):
        ck = ConsciousKernel(obs_dim=4, action_dim=2, action_mode="dreamer",
                             preference=torch.zeros(4), dreamer_reward="neg_distance")
        result = None
        for _ in range(8):
            result = ck.step(torch.randn(4) * 0.1)
        assert result.action.shape == (2,)

    def test_neg_distance_preference_not_normalized(self):
        goal = torch.tensor([0.0, 0.0, 0.0, 0.0])
        ck = ConsciousKernel(obs_dim=4, action_dim=2, action_mode="dreamer",
                             preference=goal, dreamer_reward="neg_distance")
        assert torch.allclose(ck.preference, goal)  # raw target, not a distribution

    def test_neg_distance_reward_value(self):
        ck = ConsciousKernel(obs_dim=4, action_dim=2, action_mode="dreamer",
                             preference=torch.zeros(4), dreamer_reward="neg_distance")
        r = ck._reward_from_pred(torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
        assert r.shape == (2,)
        assert float(r[0]) == pytest.approx(-5.0, abs=1e-4)   # -||[3,4,0,0]||
        assert float(r[1]) == pytest.approx(0.0, abs=1e-4)


class TestCartpoleSmoke:
    def test_kernel_runs_on_cartpole(self):
        gym = pytest.importorskip("gymnasium")
        import numpy as np
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=0)
        scales = np.array([2.4, 3.0, 0.21, 3.0], dtype=np.float32)
        ck = ConsciousKernel(obs_dim=4, action_dim=2, action_mode="dreamer",
                             preference=torch.zeros(4), dreamer_reward="neg_distance",
                             actor_explore=0.3, reflect_interval=10**9,
                             dream_interval=10**9)
        for _ in range(30):
            result = ck.step(torch.tensor(obs / scales, dtype=torch.float32))
            obs, _, term, trunc, _ = env.step(int(torch.argmax(result.action).item()))
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        assert result.action.shape == (2,)
