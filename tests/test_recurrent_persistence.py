"""Recurrent-state continuity across save/load (tick-driven deployment).

In the Yvyra deployment each tick is a fresh process: load -> step -> save. For
the kernel to be continuous (not restart semi-amnesic each tick, which keeps
free energy high and Psi pinned low), ALL behaviour-affecting state must survive
a checkpoint -- not just the nn.Module weights/buffers, but the runtime tensors
and scalars used by step()/_compute_psi (e.g. _prec_ref, recent errors, the last
self-state).

The test: take a kernel with history, snapshot it (save), and compare the SAME
next step run (a) in memory vs (b) after load into a fresh kernel. If every
behaviour-affecting state is persisted, the two steps produce the same free
energy and Psi.
"""

import torch

from zeta_life.kernel import ConsciousKernel

TOL = 1e-4


def _simplex(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(4, generator=g).abs() + 1e-3
    return x / x.sum()


def _kernel_with_history(steps: int = 40, **kw) -> ConsciousKernel:
    torch.manual_seed(0)
    k = ConsciousKernel(obs_dim=4, latent_dim=16, **kw)
    for t in range(steps):
        k.step(_simplex(t))
    return k


class TestRecurrentContinuity:
    def test_free_energy_and_psi_continuous_across_save_load(self, tmp_path):
        k = _kernel_with_history()
        k.save(str(tmp_path), "id")
        nxt = _simplex(999)

        # (a) continue in memory
        r_mem = k.step(nxt)

        # (b) same step after load into a fresh kernel
        torch.manual_seed(7)  # different seed: only loaded state should matter
        k2 = ConsciousKernel(obs_dim=4, latent_dim=16)
        k2.load(str(tmp_path), "id")
        r_load = k2.step(nxt)

        assert abs(r_mem.free_energy - r_load.free_energy) < TOL, (
            f"free energy discontinuous: in-mem={r_mem.free_energy:.5f} "
            f"loaded={r_load.free_energy:.5f}")
        assert abs(r_mem.psi - r_load.psi) < TOL, (
            f"Psi discontinuous: in-mem={r_mem.psi:.5f} loaded={r_load.psi:.5f}")

    def test_continuity_with_hypermodel(self, tmp_path):
        k = _kernel_with_history(precision_hypermodel=True)
        k.save(str(tmp_path), "id")
        nxt = _simplex(123)
        r_mem = k.step(nxt)

        torch.manual_seed(7)
        k2 = ConsciousKernel(obs_dim=4, latent_dim=16, precision_hypermodel=True)
        k2.load(str(tmp_path), "id")
        r_load = k2.step(nxt)

        assert abs(r_mem.free_energy - r_load.free_energy) < TOL
        assert abs(r_mem.psi - r_load.psi) < TOL
        # the epistemic-depth signal must be continuous too
        assert abs((r_mem.second_order_error or 0) - (r_load.second_order_error or 0)) < 1e-2
