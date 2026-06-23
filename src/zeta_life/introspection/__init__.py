"""Introspection / Psi-internal experiments (the "north": make Psi a property of
the agent's own activations, then test privileged access -- see
docs/RESEARCH_PHASE_B.md).

- ``psi_act``: candidate integration metrics computed over an LLM's hidden states.
"""

from .psi_act import (  # noqa: F401
    ALL_METRICS,
    interlayer_coherence,
    participation_ratio,
    phi_proxy,
    psi_act_all,
    trajectory_predictability,
)
