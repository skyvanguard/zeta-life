"""Tests for precision learning (Phase 2).

The per-channel precisions must actually be trained toward inverse error
variance — previously they were frozen at softplus(0)=0.693 forever because no
optimiser ever touched them, and the free energy lacked the -log precision term
(so minimising it would have driven precision to 0 anyway).
"""
from __future__ import annotations

import torch

from zeta_life.kernel import ConsciousKernel
from zeta_life.kernel.prediction_error import PredictionErrorEngine


def test_precisions_change_with_training():
    torch.manual_seed(0)
    ck = ConsciousKernel()
    before = ck.error_engine.precisions.detach().clone()
    pattern = torch.tensor([0.5, 0.2, 0.2, 0.1])
    for _ in range(40):
        ck.step(pattern + 0.01 * torch.randn(4))
    after = ck.error_engine.precisions.detach()
    # Active channels must have moved away from the frozen init (~0.693).
    assert not torch.allclose(before, after)
    assert after[0] > 1.0  # perceptual precision rose (reliable channel)


def test_dead_channels_keep_init_precision():
    # temporal/epistemic carry no signal -> their precision is not trained.
    torch.manual_seed(0)
    ck = ConsciousKernel()
    for _ in range(30):
        ck.step(torch.tensor([0.5, 0.2, 0.2, 0.1]))
    precs = ck.error_engine.precisions.detach()
    init = torch.nn.functional.softplus(torch.zeros(1)).item()  # 0.6931
    assert abs(float(precs[2]) - init) < 1e-4  # temporal unchanged
    assert abs(float(precs[3]) - init) < 1e-4  # epistemic unchanged


def test_precision_does_not_collapse_to_zero():
    # With the -log precision term, the training objective has an interior
    # optimum (precision* = D/||err||^2), so precisions stay positive and finite.
    torch.manual_seed(0)
    eng = PredictionErrorEngine(4)
    raw = torch.tensor([0.3, -0.2, 0.1, 0.4])
    for _ in range(100):
        errors = {
            "perceptual": {"raw": raw, "precision": eng.precisions[0]},
            "interoceptive": {"raw": raw * 0.5, "precision": eng.precisions[1]},
            "temporal": {"raw": torch.zeros(4), "precision": eng.precisions[2]},
            "epistemic": {"raw": torch.zeros(4), "precision": eng.precisions[3]},
        }
        eng.update_precisions(errors)
    precs = eng.precisions.detach()
    assert bool((precs > 0).all())        # never collapses to 0
    assert bool(torch.isfinite(precs).all())  # never blows up to inf


def test_lower_error_yields_higher_precision():
    # Inverse variance: the channel with smaller error should end with higher
    # precision than the channel with larger error.
    torch.manual_seed(0)
    eng = PredictionErrorEngine(4)
    small = torch.tensor([0.05, 0.0, 0.0, 0.0])   # reliable channel
    large = torch.tensor([0.8, 0.5, -0.6, 0.3])   # noisy channel
    for _ in range(200):
        errors = {
            "perceptual": {"raw": small, "precision": eng.precisions[0]},
            "interoceptive": {"raw": large, "precision": eng.precisions[1]},
            "temporal": {"raw": torch.zeros(4), "precision": eng.precisions[2]},
            "epistemic": {"raw": torch.zeros(4), "precision": eng.precisions[3]},
        }
        eng.update_precisions(errors)
    precs = eng.precisions.detach()
    assert precs[0] > precs[1]  # reliable channel has higher precision
