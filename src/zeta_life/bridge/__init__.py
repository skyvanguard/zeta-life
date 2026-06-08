"""Yvyra <-> zeta-life bridge.

The zeta-life side of the contract in ``docs/YVYRA_BRIDGE.md``: turn a live
agent's per-tick self-report (4 experiential axes) into the ConsciousKernel's
world, and return its integration index Psi plus an EFE *suggestion* of which
axis to lean into next.

Usage::

    from zeta_life.bridge import YvyraBridge
    bridge = YvyraBridge(save_dir="/opt/data/zeta")
    out = bridge.step({"novedad": 0.6, "introspeccion": 0.8,
                       "conexion": 0.1, "resolucion": 0.3})
    print(out["psi"], out["suggested_axis"])
"""

from .yvyra import AXES, DEFAULT_C, YvyraBridge

__all__ = ["YvyraBridge", "AXES", "DEFAULT_C"]
