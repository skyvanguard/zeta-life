"""
GPUBrainAdapter: Interface between ZetaOrganism and Brain10K.

Translates organism state into thalamic stimulus, runs the GPU brain,
and returns consciousness metrics for integration with the hierarchical
simulation.

Graceful degradation: returns None if no GPU available or torch not found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .organism_consciousness import OrganismConsciousness


class GPUBrainAdapter:
    """
    Adapter between ZetaOrganism consciousness and the GPU Brain10K.

    Maps organism state to thalamic input, runs the brain simulation,
    and provides output for top-down modulation and metrics.
    """

    def __init__(
        self,
        npl: int = 350,
        F_i: float = 2.5,
        alpha: float = 1.0,
        C_param: float = 0.3,
    ):
        self.npl = npl
        self.brain = None
        self._available = False

        try:
            import torch
            from .brain10k import Brain10K

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.brain = Brain10K(
                npl=npl, F_i=F_i, alpha=alpha, C_param=C_param, device=device
            )
            self._available = True
        except Exception:
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self.brain is not None

    def step(self, organism_state: OrganismConsciousness) -> dict | None:
        """
        Run one brain step using organism state as input.

        Args:
            organism_state: Current organism consciousness

        Returns:
            Dict with psi, phi, self_ref, is_conscious, or None if unavailable
        """
        if not self.is_available:
            return None

        import torch

        stimulus = self._organism_to_stimulus(organism_state)

        # Gradually increase coupling based on organism consciousness level
        ci = organism_state.consciousness_index.compute_total()
        if ci > 0.3:
            self.brain.increase_coupling(0.002 * ci)

        result = self.brain.step(stimulus)
        return result

    def _organism_to_stimulus(self, organism: OrganismConsciousness):
        """Map organism state to thalamic input tensor."""
        import torch

        thal_size = self.brain.thal_size
        stimulus = torch.zeros(thal_size, device=self.brain.device)

        # Map archetype state to stimulus (first 4 values → spread across thalamus)
        archetype = organism.global_archetype
        if archetype is not None and len(archetype) >= 4:
            chunk = thal_size // 4
            for i in range(4):
                val = float(archetype[i])
                stimulus[i * chunk: (i + 1) * chunk] = val * 0.3

        # Phi global modulates overall stimulus strength
        stimulus *= (0.5 + organism.phi_global)

        return stimulus

    def get_top_down_modulation(self, brain_output: dict | None) -> dict:
        """
        Convert brain output to top-down modulation parameters.

        Args:
            brain_output: Result from step(), or None

        Returns:
            Dict with modulation parameters for TopDownModulator
        """
        if brain_output is None:
            return {}

        return {
            'brain_consciousness': brain_output.get('psi', 0.0),
            'brain_self_reference': brain_output.get('self_ref', 0.0),
            'brain_is_conscious': brain_output.get('is_conscious', False),
        }
