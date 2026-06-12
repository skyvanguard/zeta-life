"""Tests for PrecisionHyperModel -- epistemic depth (Phase 2)."""

import torch

from zeta_life.kernel.precision_hypermodel import PrecisionHyperModel


def _errors(raws: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Build the errors dict shape that the engine produces."""
    return {ch: {'raw': raw} for ch, raw in raws.items()}


CHANNELS = ['perceptual', 'interoceptive', 'temporal', 'epistemic']


class TestShapesAndState:
    def test_predict_returns_per_channel(self):
        hm = PrecisionHyperModel(n_channels=4)
        out = hm.predict(torch.rand(4))
        assert out.shape == (4,)

    def test_reset_clears_state(self):
        hm = PrecisionHyperModel(n_channels=4)
        hm.predict(torch.rand(4))
        assert hm._last_pred is not None
        hm.reset_state()
        assert hm._last_pred is None
        assert torch.equal(hm._h, torch.zeros(1, hm.hidden))


class TestRealisedLogprec:
    def test_masks_channels_without_signal(self):
        hm = PrecisionHyperModel(n_channels=4)
        raws = {
            'perceptual': torch.tensor([0.5, 0.5]),    # has signal
            'interoceptive': torch.tensor([0.1, 0.1]),  # has signal
            'temporal': torch.zeros(2),                 # no signal
            'epistemic': torch.zeros(2),                # no signal
        }
        lp, mask = hm.realised_logprec(_errors(raws), CHANNELS)
        assert mask.tolist() == [True, True, False, False]
        # smaller error => higher precision => higher log-precision
        assert lp[1] > lp[0]

    def test_logprec_value_matches_formula(self):
        hm = PrecisionHyperModel(n_channels=2)
        raw = torch.tensor([0.5, 0.5])               # ss = 0.5, D = 2
        lp, mask = hm.realised_logprec(_errors({'a': raw, 'b': torch.zeros(2)}), ['a', 'b'])
        import math
        assert abs(float(lp[0]) - math.log(2 / 0.5)) < 1e-5


class TestLearning:
    def test_second_order_error_drops_on_stationary_signal(self):
        """On a stationary regime the hyper-model learns to predict its precisions."""
        torch.manual_seed(0)
        hm = PrecisionHyperModel(n_channels=4, hyper_lr=0.02)
        # Fixed-statistics errors => fixed realised precision target.
        raws = {
            'perceptual': torch.tensor([0.3, 0.3]),
            'interoceptive': torch.tensor([0.2, 0.2]),
            'temporal': torch.zeros(2),
            'epistemic': torch.zeros(2),
        }
        errs = _errors(raws)
        context = torch.tensor([0.42, 0.28, 0.0, 0.0])  # stable context
        mags = []
        for _ in range(200):
            hm.predict(context)
            realised, mask = hm.realised_logprec(errs, CHANNELS)
            mags.append(hm.update(realised, mask))
        early = sum(mags[:20]) / 20
        late = sum(mags[-20:]) / 20
        assert late < early * 0.5, f"did not learn: early={early:.3f} late={late:.3f}"

    def test_signature_spikes_on_regime_change(self):
        """Epistemic-depth signature: error spikes when the precision regime shifts."""
        torch.manual_seed(0)
        hm = PrecisionHyperModel(n_channels=4, hyper_lr=0.02)
        low_err = _errors({'perceptual': torch.tensor([0.1, 0.1]),
                           'interoceptive': torch.tensor([0.1, 0.1]),
                           'temporal': torch.zeros(2), 'epistemic': torch.zeros(2)})
        ctx = torch.tensor([0.14, 0.14, 0.0, 0.0])
        # settle on the low-error regime
        last = 0.0
        for _ in range(150):
            hm.predict(ctx)
            r, m = hm.realised_logprec(low_err, CHANNELS)
            last = hm.update(r, m)
        # abrupt regime change: error magnitude jumps (precision collapses)
        high_err = _errors({'perceptual': torch.tensor([2.0, 2.0]),
                            'interoceptive': torch.tensor([2.0, 2.0]),
                            'temporal': torch.zeros(2), 'epistemic': torch.zeros(2)})
        hm.predict(torch.tensor([2.8, 2.8, 0.0, 0.0]))
        r, m = hm.realised_logprec(high_err, CHANNELS)
        spike = hm.update(r, m)
        assert spike > last * 2.0, f"no spike: settled={last:.3f} spike={spike:.3f}"


class TestPersistence:
    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        src = PrecisionHyperModel(n_channels=4)
        for _ in range(10):
            src.predict(torch.rand(4))
            raws = {'perceptual': torch.rand(2), 'interoceptive': torch.rand(2),
                    'temporal': torch.zeros(2), 'epistemic': torch.zeros(2)}
            r, m = src.realised_logprec(_errors(raws), CHANNELS)
            src.update(r, m)
        torch.manual_seed(99)
        dst = PrecisionHyperModel(n_channels=4)
        dst.load_state_dict(src.state_dict())
        for p_s, p_d in zip(src.gru.parameters(), dst.gru.parameters()):
            assert torch.equal(p_s, p_d)
