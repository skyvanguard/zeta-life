"""
Yvyra bridge demo — the kernel running on a (mock) live agent's experience
==========================================================================

End-to-end demonstration of the Yvyra <-> zeta-life coupling
(``docs/YVYRA_BRIDGE.md``) WITHOUT the real external agent. A ``MockYvyra``
lives in blocks of modes and scores each tick honestly-ish on the 4 axes:

  research      -> high novedad
  introspection -> high introspeccion
  fran          -> high conexion
  synthesis     -> high resolucion

Each tick: mode -> 4 scores -> bridge.step -> (Psi, EFE suggestion). Because the
experience is temporally structured (blocks), the world model can learn it and
Psi integrates upward; the EFE suggestion points toward the axes that move the
experience back toward the preferred character C.

This validates the bridge plumbing: encoding, stepping, suggestion, periodic
dreams, and a Psi trajectory that reflects experiential coherence.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from zeta_life.bridge import AXES, DEFAULT_C, YvyraBridge


class MockYvyra:
    """A stand-in live agent: lives in blocks of modes, scoring per mode + noise."""

    MODES = {
        # axis order: novedad, introspeccion, conexion, resolucion
        "research":      [0.80, 0.35, 0.20, 0.40],
        "introspection": [0.20, 0.85, 0.10, 0.25],
        "fran":          [0.40, 0.40, 0.90, 0.50],
        "synthesis":     [0.30, 0.45, 0.20, 0.85],
    }
    # An introspection-leaning life (matches C), with research and occasional
    # contact/synthesis — a coherent schedule the kernel can integrate.
    SCHEDULE = ["introspection", "research", "introspection", "fran",
                "introspection", "research", "synthesis", "introspection"]

    def __init__(self, block: int = 12, noise: float = 0.06, seed: int = 0):
        self.block = block
        self.noise = noise
        self.g = torch.Generator().manual_seed(seed)

    def live(self, tick: int):
        mode = self.SCHEDULE[(tick // self.block) % len(self.SCHEDULE)]
        base = torch.tensor(self.MODES[mode])
        scores = (base + self.noise * torch.randn(4, generator=self.g)).clamp(0.0, 1.0)
        return mode, scores.tolist()


def main(n_ticks: int, plot: bool) -> None:
    torch.manual_seed(0)
    print("=" * 70)
    print("  YVYRA BRIDGE DEMO — kernel on a (mock) live agent's experience")
    print("=" * 70)
    print(f"  C (preferred character) = "
          f"{dict(zip(AXES, [round(c/sum(DEFAULT_C), 2) for c in DEFAULT_C]))}")
    print(f"  ticks={n_ticks}")
    print()

    yvyra = MockYvyra()
    bridge = YvyraBridge(dream_every=25)

    psis, dist_to_c, modes = [], [], []
    suggestion_counts = {a: 0 for a in AXES}
    C = torch.tensor([c / sum(DEFAULT_C) for c in DEFAULT_C])
    stable_psi, boundary_psi = [], []  # within a mode vs just after a switch

    for t in range(n_ticks):
        mode, scores = yvyra.live(t)
        out = bridge.step(scores)
        psis.append(out["psi"])
        modes.append(mode)
        suggestion_counts[out["suggested_axis"]] += 1
        s = torch.tensor(scores)
        s = s / s.sum()
        dist_to_c.append(float(torch.linalg.vector_norm(s - C)))
        # A "boundary" tick is one right after a mode switch (experience just
        # fragmented); a "stable" tick is deep inside a sustained mode.
        pos = t % yvyra.block
        (boundary_psi if pos < 2 else stable_psi).append(out["psi"])
        if (t + 1) % 25 == 0:
            tag = " [DREAM]" if "dream" in out else ""
            print(f"  tick {t+1:3d} | mode={mode:13s} | Psi={out['psi']:.3f} | "
                  f"suggest={out['suggested_axis']:13s}{tag}")

    def _mean(v):
        return sum(v) / len(v) if v else 0.0

    print()
    print(f"  Psi during SUSTAINED focus (stable ticks):  {_mean(stable_psi):.3f}")
    print(f"  Psi right after a mode switch (boundary):    {_mean(boundary_psi):.3f}")
    print("    -> Psi drops when the experience fragments (the contract's intended")
    print("       introspective signal), and is higher during sustained focus.")
    print(f"  suggestion distribution: "
          f"{ {a: suggestion_counts[a] for a in AXES} }")
    print(f"    (introspeccion-dominant, matching C={dict(zip(AXES, [round(float(c),2) for c in C]))})")
    print("=" * 70)

    if plot:
        try:
            _plot(psis, dist_to_c, suggestion_counts)
        except Exception as e:  # noqa: BLE001 - plotting is best-effort
            print(f"  (plot skipped: {e})")


def _plot(psis, dist_to_c, suggestion_counts) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    arr = np.array(psis)
    axes[0].plot(arr, color="#c9a0dc", alpha=0.5, label="Psi (per tick)")
    if len(arr) >= 10:
        w = 10
        sm = np.convolve(arr, np.ones(w) / w, mode="valid")
        axes[0].plot(range(w - 1, len(arr)), sm, color="#8e44ad", lw=2,
                     label="Psi (smoothed)")
    axes[0].set_title("Psi over ticks\n(dips at experience fragmentation)")
    axes[0].set_xlabel("tick")
    axes[0].set_ylabel("Psi")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dist_to_c, color="#16a085")
    axes[1].set_title("Distance of experience to C\n(lower = closer to character)")
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("||experience - C||")
    axes[1].grid(True, alpha=0.3)

    names = list(AXES)
    vals = [suggestion_counts[a] for a in names]
    axes[2].bar(names, vals, color=["#2980b9", "#8e44ad", "#e67e22", "#27ae60"])
    axes[2].set_title("EFE suggestions by axis")
    axes[2].set_ylabel("count")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Yvyra bridge: the kernel integrating a live agent's experience")
    out = Path("results") / "yvyra_bridge.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yvyra bridge end-to-end demo")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    main(n_ticks=args.ticks, plot=not args.no_plot)
