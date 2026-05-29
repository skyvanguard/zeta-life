"""Tests for world model learning (Phase 3): encoder + recurrence.

Two bugs fixed by the prior+posterior flow:
- #4 the encoder was frozen (latent_state was always detached, so the
  perceptual loss never reached the encoder).
- #5 the recurrence was destroyed (latent_state was overwritten each step with
  a detached encode(stimulus), discarding the transition GRU's output).

observe() now folds the observation into the latent via a posterior blend and
trains the encoder on the reconstruction loss; predict() keeps the recurrent
prior.
"""
from __future__ import annotations

import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.world_model import WorldModel


def _encoder_magnitude(wm: WorldModel) -> float:
    return sum(float(p.detach().abs().sum()) for p in wm.encoder.parameters())


# --- #4 encoder learns -----------------------------------------------------
def test_observe_trains_encoder():
    torch.manual_seed(0)
    wm = WorldModel()
    before = _encoder_magnitude(wm)
    for _ in range(30):
        wm.predict(torch.zeros(4))
        wm.observe(torch.tensor([0.5, 0.2, 0.2, 0.1]))
    assert abs(_encoder_magnitude(wm) - before) > 1e-3  # encoder moved


def test_encoder_learns_inside_kernel():
    torch.manual_seed(0)
    ck = ConsciousKernel()
    before = _encoder_magnitude(ck.world_model)
    pattern = torch.tensor([0.5, 0.2, 0.2, 0.1])
    for _ in range(40):
        ck.step(pattern + 0.01 * torch.randn(4))
    assert abs(_encoder_magnitude(ck.world_model) - before) > 1e-3


# --- #5 recurrence / temporal memory --------------------------------------
def test_latent_is_not_pure_encode():
    # After observe(), the latent must be a blend of prior and encoding, NOT a
    # detached pure encode(stimulus) (the old broken behaviour).
    torch.manual_seed(0)
    wm = WorldModel(posterior_blend=0.5)
    stim = torch.tensor([0.5, 0.2, 0.2, 0.1])
    wm.predict(torch.ones(4))     # build a non-trivial prior
    wm.observe(stim)
    pure_encode = wm.encode(stim).detach()
    assert not torch.allclose(wm.latent_state, pure_encode)


def test_temporal_memory_predicts_cyclic_sequence():
    # A cyclic sequence requires remembering the phase; a model with real
    # recurrence should predict it better than chance.
    torch.manual_seed(0)
    cycle = [torch.tensor([1.0, 0, 0, 0]), torch.tensor([0, 1.0, 0, 0]),
             torch.tensor([0, 0, 1.0, 0]), torch.tensor([0, 0, 0, 1.0])]
    wm = WorldModel()
    errs = []
    last_action = torch.zeros(4)
    for t in range(160):
        stim = cycle[t % 4]
        pred, _ = wm.predict(last_action)
        errs.append(float(torch.sum((pred.detach() - stim) ** 2)))
        wm.update_from_error(pred - stim)
        wm.observe(stim)
        last_action = torch.softmax(stim, dim=-1)
    first = sum(errs[:20]) / 20
    last = sum(errs[-20:]) / 20
    assert last < first  # learns the cyclic dynamics over time


# --- regression: predict() still works -------------------------------------
def test_predict_still_returns_obs_and_latent():
    wm = WorldModel(obs_dim=4, latent_dim=32, action_dim=4)
    pred, latent = wm.predict(torch.zeros(4))
    assert pred.shape == (4,)
    assert latent.shape == (32,)
