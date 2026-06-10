"""Quickstart demo — the Conscious Kernel in ~60 lines.

Runs the full active-inference cycle (perceive -> predict -> compare ->
update -> memorize -> act -> reflect -> dream) on a synthetic structured
signal, prints the consciousness index Psi and free energy as they evolve,
and demonstrates identity persistence (save -> load round trip).

Usage::

    PYTHONPATH=src python demos/quickstart.py
    # or, after `pip install -e .`:
    python demos/quickstart.py
"""

import tempfile

import torch

from zeta_life.kernel import ConsciousKernel

N_STEPS = 300
REPORT_EVERY = 50


def synthetic_stimulus(t: int) -> torch.Tensor:
    """A predictable 4-D stimulus on the simplex — learnable structure.

    The kernel expects vertex-space stimuli (non-negative, summing to 1).
    A slowly rotating pattern with mild noise gives the world model
    something real to learn.
    """
    phases = torch.tensor([0.0, 1.6, 3.1, 4.7])
    base = 1.0 + torch.sin(0.05 * t + phases)
    noisy = base + 0.1 * torch.randn(4).abs()
    return noisy / noisy.sum()


def main() -> None:
    torch.manual_seed(0)
    print("=== Zeta-Life quickstart: the Conscious Kernel ===\n")

    kernel = ConsciousKernel(obs_dim=4, latent_dim=32)
    print(f"Kernel created (obs_dim=4, latent_dim=32, "
          f"world_model_type={kernel.world_model_type!r})\n")

    print(f"{'step':>6} {'Psi':>8} {'free_energy':>12} {'error':>10}")
    result = None
    for t in range(1, N_STEPS + 1):
        result = kernel.step(synthetic_stimulus(t))
        if t % REPORT_EVERY == 0:
            print(f"{t:>6} {result.psi:>8.4f} {result.free_energy:>12.4f} "
                  f"{sum(result.errors.values()):>10.4f}")

    print(f"\nAfter {N_STEPS} steps: {len(kernel.fast_memory)} episodes "
          f"in fast memory, Psi={result.psi:.4f}")

    # --- Identity persistence: save, load into a fresh kernel, continue ---
    with tempfile.TemporaryDirectory() as ckpt_dir:
        kernel.save(ckpt_dir, identity_name='quickstart')
        print(f"\nIdentity saved to a checkpoint (step={kernel.t}).")

        reborn = ConsciousKernel(obs_dim=4, latent_dim=32)
        reborn.load(ckpt_dir, identity_name='quickstart')
        print(f"Fresh kernel loaded the identity (step={reborn.t}) "
              f"and keeps stepping:")
        r = reborn.step(synthetic_stimulus(N_STEPS + 1))
        print(f"{reborn.t:>6} {r.psi:>8.4f} {r.free_energy:>12.4f} "
              f"{sum(r.errors.values()):>10.4f}")

    print("\nNext steps: see experiments/kernel/ (e.g. "
          "exp_conscious_kernel_validation.py) and README.md.")


if __name__ == '__main__':
    main()
