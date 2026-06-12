"""Adaptive psi_fe_scale: Psi discriminates regardless of the free-energy regime.

A fixed psi_fe_scale is a moving target: when Yvyra's dynamics change (e.g.
richer journals raise free energy), the scale calibrated for the old regime
leaves Psi pinned at 0 in the new one. The adaptive scale auto-calibrates to an
EMA of free energy (like _prec_ref does for F_i), so phi-base stays in a
discriminating band whatever the absolute free-energy level.
"""

import numpy as np
import torch

from zeta_life.kernel import ConsciousKernel


def _frag(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(4, generator=g).abs() + 1e-3
    return x / x.sum()


class TestAdaptiveScale:
    def test_adaptive_discriminates_where_fixed_collapses(self):
        """In a high-free-energy regime, fixed-scale Psi flattens; adaptive doesn't."""
        torch.manual_seed(0)
        k_fixed = ConsciousKernel(obs_dim=4, latent_dim=16, psi_fe_scale=1.0)
        torch.manual_seed(0)
        k_adapt = ConsciousKernel(obs_dim=4, latent_dim=16, psi_fe_adaptive=True)

        pf, pa = [], []
        for t in range(80):
            s = _frag(t)  # i.i.d. -> world model predicts poorly -> high free energy
            pf.append(k_fixed.step(s).psi)
            pa.append(k_adapt.step(s).psi)

        std_fixed = float(np.std(pf[-40:]))
        std_adapt = float(np.std(pa[-40:]))
        assert std_adapt > std_fixed, (
            f"adaptive should discriminate more: fixed std={std_fixed:.3f} "
            f"adaptive std={std_adapt:.3f}")
        assert std_adapt > 0.05, f"adaptive Psi should vary, got std={std_adapt:.3f}"

    def test_off_by_default_byte_identical(self):
        """Adaptive is opt-in; default kernel is unchanged."""
        torch.manual_seed(0)
        k_def = ConsciousKernel(obs_dim=4, latent_dim=16)
        torch.manual_seed(0)
        k_fixed = ConsciousKernel(obs_dim=4, latent_dim=16, psi_fe_adaptive=False)
        for t in range(20):
            s = _frag(t)
            r1, r2 = k_def.step(s), k_fixed.step(s)
            assert r1.psi == r2.psi

    def test_fe_ref_persists_across_save_load(self, tmp_path):
        k = ConsciousKernel(obs_dim=4, latent_dim=16, psi_fe_adaptive=True)
        for t in range(30):
            k.step(_frag(t))
        assert k._fe_ref is not None
        k.save(str(tmp_path), "id")
        torch.manual_seed(9)
        k2 = ConsciousKernel(obs_dim=4, latent_dim=16, psi_fe_adaptive=True)
        k2.load(str(tmp_path), "id")
        assert abs(k2._fe_ref - k._fe_ref) < 1e-6
