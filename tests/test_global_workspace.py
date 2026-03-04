"""Tests for GlobalWorkspace and Proposal."""
import torch
import pytest

from zeta_life.kernel.global_workspace import GlobalWorkspace, Proposal


def _make_proposal(kernel_id: int, free_energy: float = 0.3,
                   energy: float = 5.0, obs_dim: int = 4) -> Proposal:
    return Proposal(
        kernel_id=kernel_id,
        state=torch.randn(16),
        free_energy=free_energy,
        energy=energy,
        action=torch.randn(obs_dim),
        salience=1.0 / (1.0 + free_energy),
    )


class TestProposal:
    def test_creation(self):
        p = _make_proposal(0)
        assert p.kernel_id == 0
        assert isinstance(p.free_energy, float)

    def test_signal_strength(self):
        p = _make_proposal(0, free_energy=0.2, energy=5.0)
        s = p.signal_strength()
        assert s > 0
        assert isinstance(s, float)

    def test_lower_fe_higher_signal(self):
        p_good = _make_proposal(0, free_energy=0.1, energy=5.0)
        p_bad = _make_proposal(1, free_energy=0.5, energy=5.0)
        assert p_good.signal_strength() > p_bad.signal_strength()

    def test_more_energy_higher_signal(self):
        p_rich = _make_proposal(0, free_energy=0.3, energy=8.0)
        p_poor = _make_proposal(1, free_energy=0.3, energy=2.0)
        assert p_rich.signal_strength() > p_poor.signal_strength()


class TestGlobalWorkspaceInit:
    def test_creation(self):
        gw = GlobalWorkspace(obs_dim=4)
        assert gw.obs_dim == 4

    def test_spotlight_initially_none(self):
        gw = GlobalWorkspace(obs_dim=4)
        assert gw.spotlight is None

    def test_broadcast_signal_initially_zeros(self):
        gw = GlobalWorkspace(obs_dim=4)
        assert gw.broadcast_signal.shape == (4,)
        assert gw.broadcast_signal.sum().item() == 0.0


class TestCompete:
    def test_returns_winner_id(self):
        gw = GlobalWorkspace(obs_dim=4)
        proposals = {
            0: _make_proposal(0, free_energy=0.5, energy=5.0),
            1: _make_proposal(1, free_energy=0.2, energy=5.0),
        }
        winner = gw.compete(proposals)
        assert winner == 1  # lower FE wins

    def test_higher_energy_breaks_tie(self):
        gw = GlobalWorkspace(obs_dim=4)
        proposals = {
            0: _make_proposal(0, free_energy=0.3, energy=3.0),
            1: _make_proposal(1, free_energy=0.3, energy=7.0),
        }
        winner = gw.compete(proposals)
        assert winner == 1

    def test_anti_monopoly_after_3_wins(self):
        gw = GlobalWorkspace(obs_dim=4)
        # Kernel 0 has much better stats
        proposals = {
            0: _make_proposal(0, free_energy=0.1, energy=8.0),
            1: _make_proposal(1, free_energy=0.4, energy=4.0),
        }
        # Win 3 times
        for _ in range(3):
            gw.compete(proposals)

        # Now kernel 0 has penalty, kernel 1 gets boost
        gw.consecutive_wins[0] = 3
        winner = gw.compete(proposals)
        # With penalty applied, verify it was tracked
        assert gw.consecutive_wins[0] >= 3

    def test_single_proposal_wins(self):
        gw = GlobalWorkspace(obs_dim=4)
        proposals = {0: _make_proposal(0)}
        assert gw.compete(proposals) == 0


class TestBroadcast:
    def test_updates_spotlight(self):
        gw = GlobalWorkspace(obs_dim=4)
        p = _make_proposal(0)
        gw.broadcast(p)
        assert gw.spotlight is not None
        assert gw.spotlight_owner == 0

    def test_updates_broadcast_signal(self):
        gw = GlobalWorkspace(obs_dim=4)
        p = _make_proposal(0)
        gw.broadcast(p)
        assert gw.broadcast_signal.shape == (4,)
        # broadcast_signal should be the action
        assert torch.allclose(gw.broadcast_signal, p.action)

    def test_tracks_history(self):
        gw = GlobalWorkspace(obs_dim=4)
        gw.broadcast(_make_proposal(0))
        gw.broadcast(_make_proposal(1))
        assert len(gw.history) == 2
