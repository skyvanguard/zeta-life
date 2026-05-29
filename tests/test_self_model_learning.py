"""Tests for self-model learning (Phase 4, point #6).

The self-model's prediction pathway was never trained: the kernel only
backpropagated the perceptual error, so the interoceptive channel contributed a
fixed, non-reducible offset to the free energy. Now the self-model has its own
optimizer and is trained from the interoceptive error, while the identity
embedding still drifts only via the slow EMA in reflect().
"""
from __future__ import annotations

import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.self_model import SelfModel


def _pred_magnitude(sm: SelfModel) -> float:
    return sum(
        float(p.detach().abs().sum())
        for p in sm.embed_to_prediction.parameters()
    )


def test_update_trains_prediction_pathway():
    torch.manual_seed(0)
    sm = SelfModel(state_dim=4, embed_dim=16)
    before = _pred_magnitude(sm)
    target = torch.tensor([0.4, 0.3, 0.2, 0.1])
    for _ in range(50):
        pred = sm.predict_self(torch.zeros(4))
        sm.update_from_error(pred - target)
    assert abs(_pred_magnitude(sm) - before) > 1e-3


def test_interoceptive_error_decreases_in_kernel():
    torch.manual_seed(0)
    ck = ConsciousKernel()
    pattern = torch.tensor([0.5, 0.2, 0.2, 0.1])
    errs = []
    for _ in range(80):
        r = ck.step(pattern + 0.01 * torch.randn(4))
        errs.append(r.errors["interoceptive"])
    first = sum(errs[:20]) / 20
    last = sum(errs[-20:]) / 20
    assert last < first  # self-model learns to predict its own state


def test_self_embedding_not_moved_by_gradient():
    # The identity embedding must NOT be in the prediction optimizer; it only
    # changes via the EMA in reflect(). update_from_error alone must leave it put.
    torch.manual_seed(0)
    sm = SelfModel(state_dim=4, embed_dim=16)
    before = sm.self_embedding.detach().clone()
    target = torch.tensor([0.4, 0.3, 0.2, 0.1])
    for _ in range(30):
        pred = sm.predict_self(torch.zeros(4))
        sm.update_from_error(pred - target)  # no reflect() called here
    assert torch.allclose(sm.self_embedding.detach(), before)


def test_update_from_error_no_grad_is_safe():
    # A detached error (no grad_fn) must not raise, just report the loss.
    sm = SelfModel(state_dim=4, embed_dim=16)
    loss = sm.update_from_error(torch.tensor([0.1, 0.2, 0.0, 0.0]))
    assert isinstance(loss, float)


def test_predict_self_still_distribution():
    sm = SelfModel(state_dim=4, embed_dim=16)
    out = sm.predict_self(torch.zeros(4)).detach()
    assert abs(float(out.sum()) - 1.0) < 1e-5
