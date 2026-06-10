"""Tests for RSSM persistence — save/load of the DreamerV3Agent inside the kernel.

The kernel's identity persistence (PersistenceLayer) must include the RSSM
world model when ``world_model_type="rssm"``: otherwise the trained dynamics,
actor and critic are silently lost across sessions.
"""

import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.dreamerv3_agent import DreamerV3Agent

RSSM_KW = dict(warmup=80, seq_len=8, batch_size=4, horizon=4,
               deter_dim=32, stoch_dim=8, hidden=32)


def _make_rssm_kernel(seed: int = 0) -> ConsciousKernel:
    torch.manual_seed(seed)
    k = ConsciousKernel(obs_dim=4, action_dim=2, world_model_type="rssm",
                        rssm_kwargs=dict(RSSM_KW))
    k.reset_rssm_state()
    return k


class TestDreamerAgentStateDict:
    """DreamerV3Agent exposes a full state_dict/load_state_dict pair."""

    def test_state_dict_has_all_modules(self):
        ag = DreamerV3Agent(obs_dim=4, action_dim=2, **RSSM_KW)
        sd = ag.state_dict()
        for key in ('rssm', 'actor', 'critic', 'critic_target',
                    'wm_opt', 'actor_opt', 'critic_opt', 'adv_std'):
            assert key in sd, f"missing '{key}' in DreamerV3Agent.state_dict()"

    def test_round_trip_restores_weights(self):
        torch.manual_seed(0)
        src = DreamerV3Agent(obs_dim=4, action_dim=2, **RSSM_KW)
        torch.manual_seed(99)
        dst = DreamerV3Agent(obs_dim=4, action_dim=2, **RSSM_KW)
        src._adv_std = 3.14

        dst.load_state_dict(src.state_dict())

        for p_src, p_dst in zip(src.rssm.parameters(), dst.rssm.parameters()):
            assert torch.equal(p_src, p_dst)
        for p_src, p_dst in zip(src.actor.parameters(), dst.actor.parameters()):
            assert torch.equal(p_src, p_dst)
        for p_src, p_dst in zip(src.critic.parameters(), dst.critic.parameters()):
            assert torch.equal(p_src, p_dst)
        assert dst._adv_std == 3.14


class TestKernelRSSMPersistence:
    """ConsciousKernel(world_model_type='rssm') save/load round trip."""

    def _train_briefly(self, k: ConsciousKernel, steps: int = 120) -> None:
        obs = torch.randn(4)
        for t in range(steps):
            k.step(obs)
            k.learn_rssm(reward=1.0, done=(t % 30 == 29))
            obs = torch.randn(4) if (t % 30 == 29) else obs + 0.1 * torch.randn(4)

    def test_save_load_restores_rssm_weights(self, tmp_path):
        src = _make_rssm_kernel(seed=0)
        self._train_briefly(src)
        src.save(str(tmp_path), 'rssm_id')

        dst = _make_rssm_kernel(seed=42)
        # Sanity: fresh kernel differs before loading
        same = all(torch.equal(a, b) for a, b in
                   zip(src._rssm_agent.rssm.parameters(),
                       dst._rssm_agent.rssm.parameters()))
        assert not same, "fresh kernel should differ before load"

        dst.load(str(tmp_path), 'rssm_id')

        for p_src, p_dst in zip(src._rssm_agent.rssm.parameters(),
                                dst._rssm_agent.rssm.parameters()):
            assert torch.equal(p_src, p_dst)
        for p_src, p_dst in zip(src._rssm_agent.actor.parameters(),
                                dst._rssm_agent.actor.parameters()):
            assert torch.equal(p_src, p_dst)

    def test_save_load_restores_step_counter(self, tmp_path):
        src = _make_rssm_kernel(seed=0)
        self._train_briefly(src, steps=50)
        src.save(str(tmp_path), 'rssm_id')

        dst = _make_rssm_kernel(seed=1)
        dst.load(str(tmp_path), 'rssm_id')
        assert dst.t == src.t

    def test_loaded_kernel_keeps_stepping(self, tmp_path):
        src = _make_rssm_kernel(seed=0)
        self._train_briefly(src, steps=50)
        src.save(str(tmp_path), 'rssm_id')

        dst = _make_rssm_kernel(seed=1)
        dst.load(str(tmp_path), 'rssm_id')
        dst.reset_rssm_state()
        r = dst.step(torch.randn(4))
        dst.learn_rssm(reward=0.5, done=False)
        assert r.action.shape == (2,)
        assert 0.0 <= r.psi <= 1.0


class TestGRUBackwardCompat:
    """The default GRU path must be unaffected by RSSM persistence support."""

    def test_gru_save_load_round_trip_unchanged(self, tmp_path):
        torch.manual_seed(0)
        src = ConsciousKernel(obs_dim=4, latent_dim=16)
        for _ in range(10):
            src.step(torch.randn(4))
        src.save(str(tmp_path), 'gru_id')

        torch.manual_seed(7)
        dst = ConsciousKernel(obs_dim=4, latent_dim=16)
        dst.load(str(tmp_path), 'gru_id')

        for p_src, p_dst in zip(src.world_model.parameters(),
                                dst.world_model.parameters()):
            assert torch.equal(p_src, p_dst)
        assert dst.t == src.t

    def test_gru_checkpoint_has_no_rssm_entry(self, tmp_path):
        """GRU checkpoints stay byte-compatible: no rssm_agent payload."""
        torch.manual_seed(0)
        gru = ConsciousKernel(obs_dim=4, latent_dim=16)
        gru.step(torch.randn(4))
        gru.save(str(tmp_path), 'gru_id')

        ckpt = torch.load(tmp_path / 'gru_id.ckpt', weights_only=False)
        assert 'rssm_agent' not in ckpt

    def test_rssm_checkpoint_contains_rssm_entry(self, tmp_path):
        k = _make_rssm_kernel(seed=0)
        k.step(torch.randn(4))
        k.learn_rssm(reward=0.0, done=False)
        k.save(str(tmp_path), 'rssm_id')

        ckpt = torch.load(tmp_path / 'rssm_id.ckpt', weights_only=False)
        assert 'rssm_agent' in ckpt
        assert 'rssm' in ckpt['rssm_agent']
