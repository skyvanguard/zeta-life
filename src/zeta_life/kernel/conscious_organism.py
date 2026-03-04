"""ConsciousOrganism -- multi-agent Darwinian Brain orchestrator.

Multiple ConsciousKernels compete for a GlobalWorkspace via
winner-takes-all. Energy conservation and dynamic spawn/merge/death
create selection pressure. Consciousness emerges from the competition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .conscious_kernel import ConsciousKernel, StepResult
from .global_workspace import GlobalWorkspace, Proposal
from .energy_pool import EnergyPool
from .spawn_controller import (
    SpawnController, SpawnEvent, MergeEvent, DeathEvent, LifecycleEvent,
)
from .organism_state import OrganismState


@dataclass
class OrganismStepResult:
    """Result of a single ConsciousOrganism step."""
    winner_id: int
    consciousness: float
    population: int
    diversity: float
    coherence: float
    free_energies: dict[int, float] = field(default_factory=dict)
    energies: dict[int, float] = field(default_factory=dict)
    events: list[LifecycleEvent] = field(default_factory=list)


class ConsciousOrganism:
    """Multi-agent orchestrator with Darwinian selection pressure.

    Parameters
    ----------
    obs_dim : int
        Observation/action dimensionality for each kernel.
    initial_kernels : int
        Starting population size.
    total_energy : float
        Total conserved energy in the system.
    latent_dim : int
        World model latent space dim for each kernel.
    embed_dim : int
        Self model embedding dim for each kernel.
    reflect_interval : int
        Reflection interval for each kernel.
    dream_interval : int
        Dream interval for each kernel.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        initial_kernels: int = 2,
        total_energy: float = 10.0,
        latent_dim: int = 32,
        embed_dim: int = 16,
        reflect_interval: int = 5,
        dream_interval: int = 50,
    ) -> None:
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.reflect_interval = reflect_interval
        self.dream_interval = dream_interval
        self.t: int = 0
        self._next_id: int = initial_kernels

        # Create initial kernels
        self.kernels: dict[int, ConsciousKernel] = {}
        per_kernel_energy = total_energy / initial_kernels
        for i in range(initial_kernels):
            k = ConsciousKernel(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                reflect_interval=reflect_interval,
                dream_interval=dream_interval,
            )
            k.energy = per_kernel_energy
            self.kernels[i] = k

        # Components
        self.gw = GlobalWorkspace(obs_dim=obs_dim)
        self.energy_pool = EnergyPool(total_energy=total_energy)
        self.spawn_controller = SpawnController()
        self.state = OrganismState()

    def step(self, stimulus: Tensor) -> OrganismStepResult:
        """Advance the organism by one step."""
        self.t += 1

        # 1. DISTRIBUTE + PROCESS
        results: dict[int, StepResult] = {}
        for kid, k in self.kernels.items():
            combined = self._combine_stimulus(stimulus)
            results[kid] = k.step(combined)

        # 2. PROPOSE
        proposals: dict[int, Proposal] = {}
        for kid, k in self.kernels.items():
            r = results[kid]
            proposals[kid] = Proposal(
                kernel_id=kid,
                state=k.self_model.self_embedding.data.clone().detach(),
                free_energy=r.free_energy,
                energy=k.energy,
                action=r.action.clone().detach(),
                salience=1.0 / (1.0 + r.free_energy),
            )

        # 3. COMPETE
        winner_id = self.gw.compete(proposals)

        # 4. BROADCAST
        self.gw.broadcast(proposals[winner_id])

        # 5. REWARD + DECAY + NORMALIZE
        self.energy_pool.reward_winner(winner_id, self.kernels)
        self.energy_pool.decay_all(self.kernels)
        self.energy_pool.normalize(self.kernels)

        # 6. LIFECYCLE
        events = self.spawn_controller.evaluate(self.kernels)
        self._apply_events(events)
        if events:
            self.energy_pool.normalize(self.kernels)

        # 7. MEASURE
        self.state.update(self.kernels, self.gw)

        return OrganismStepResult(
            winner_id=winner_id,
            consciousness=self.state.consciousness_index,
            population=len(self.kernels),
            diversity=self.state.diversity,
            coherence=self.state.coherence,
            free_energies={kid: results[kid].free_energy for kid in results},
            energies={kid: k.energy for kid, k in self.kernels.items()},
            events=events,
        )

    def _combine_stimulus(self, stimulus: Tensor) -> Tensor:
        """Combine external stimulus with GW broadcast."""
        broadcast = self.gw.broadcast_signal
        if broadcast is None or broadcast.sum().item() == 0.0:
            return stimulus
        alpha = 0.3 * self.state.coherence
        combined = (1 - alpha) * stimulus + alpha * broadcast[:self.obs_dim]
        return combined

    def _apply_events(self, events: list[LifecycleEvent]) -> None:
        """Apply lifecycle events to the kernel population."""
        for event in events:
            if isinstance(event, SpawnEvent):
                self._spawn(event.parent_id)
            elif isinstance(event, MergeEvent):
                self._merge(event.kernel_a, event.kernel_b)
            elif isinstance(event, DeathEvent):
                self._death(event.kernel_id)

    def _spawn(self, parent_id: int) -> None:
        """Clone parent kernel with mutation."""
        parent = self.kernels[parent_id]
        child = ConsciousKernel(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            reflect_interval=self.reflect_interval,
            dream_interval=self.dream_interval,
        )

        # Inherit world model with mutation
        child_sd = child.world_model.state_dict()
        parent_sd = parent.world_model.state_dict()
        for key in child_sd:
            if key in parent_sd:
                child_sd[key] = parent_sd[key] + torch.randn_like(parent_sd[key]) * 0.05
        child.world_model.load_state_dict(child_sd)

        # Copy slow memory (inherited knowledge)
        child.slow_memory.load_state_dict(parent.slow_memory.state_dict())

        # Energy split: parent 60%, child 40%
        child_energy = parent.energy * 0.4
        parent.energy *= 0.6
        child.energy = child_energy

        new_id = self._next_id
        self._next_id += 1
        self.kernels[new_id] = child

    def _merge(self, id_a: int, id_b: int) -> None:
        """Merge two kernels into one."""
        if id_a not in self.kernels or id_b not in self.kernels:
            return
        ka, kb = self.kernels[id_a], self.kernels[id_b]

        # Keep the stronger kernel
        if ka.energy >= kb.energy:
            survivor, absorbed = ka, kb
            survivor_id, absorbed_id = id_a, id_b
        else:
            survivor, absorbed = kb, ka
            survivor_id, absorbed_id = id_b, id_a

        # Combine energy
        survivor.energy += absorbed.energy

        # Merge world model (weighted average)
        total_e = survivor.energy
        w_s = (survivor.energy - absorbed.energy) / max(total_e, 1e-6)
        w_a = 1.0 - w_s
        surv_sd = survivor.world_model.state_dict()
        abs_sd = absorbed.world_model.state_dict()
        for key in surv_sd:
            if key in abs_sd:
                surv_sd[key] = w_s * surv_sd[key] + w_a * abs_sd[key]
        survivor.world_model.load_state_dict(surv_sd)

        del self.kernels[absorbed_id]

    def _death(self, kernel_id: int) -> None:
        """Remove a kernel and redistribute its energy."""
        if kernel_id not in self.kernels:
            return
        dead = self.kernels[kernel_id]
        released = dead.energy
        del self.kernels[kernel_id]

        # Distribute energy to survivors
        if self.kernels:
            per_survivor = released / len(self.kernels)
            for k in self.kernels.values():
                k.energy += per_survivor
