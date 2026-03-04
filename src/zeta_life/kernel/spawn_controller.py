"""SpawnController -- dynamic population lifecycle management.

Manages spawn (mitosis), merge (fusion), and death (absorption) of
ConsciousKernels based on energy levels and embedding similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LifecycleEvent:
    """Base class for lifecycle events."""
    pass


@dataclass
class SpawnEvent(LifecycleEvent):
    parent_id: int


@dataclass
class MergeEvent(LifecycleEvent):
    kernel_a: int
    kernel_b: int


@dataclass
class DeathEvent(LifecycleEvent):
    kernel_id: int


class SpawnController:
    """Evaluates population dynamics and generates lifecycle events.

    Parameters
    ----------
    min_kernels : int
        Minimum population (death protected).
    max_kernels : int
        Maximum population (spawn blocked).
    spawn_energy : float
        Energy threshold to trigger spawn.
    death_energy : float
        Energy threshold to trigger death.
    merge_similarity : float
        Cosine similarity threshold to trigger merge.
    min_age : int
        Minimum steps before a kernel can spawn.
    """

    def __init__(
        self,
        min_kernels: int = 2,
        max_kernels: int = 10,
        spawn_energy: float = 8.0,
        death_energy: float = 1.0,
        merge_similarity: float = 0.90,
        min_age: int = 200,
    ) -> None:
        self.min_kernels = min_kernels
        self.max_kernels = max_kernels
        self.spawn_energy = spawn_energy
        self.death_energy = death_energy
        self.merge_similarity = merge_similarity
        self.min_age = min_age

    def evaluate(self, kernels: dict) -> list[LifecycleEvent]:
        """Evaluate all kernels and return lifecycle events."""
        events: list[LifecycleEvent] = []
        n = len(kernels)

        # DEATH: low energy kernels die (if above min population)
        deaths = 0
        for kid, k in kernels.items():
            if k.energy < self.death_energy and (n - deaths) > self.min_kernels:
                events.append(DeathEvent(kernel_id=kid))
                deaths += 1

        # MERGE: similar kernels fuse (respecting min population)
        merged_ids: set[int] = set()
        effective_n = n - deaths
        kids = list(kernels.keys())
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                if effective_n - len(merged_ids) // 2 <= self.min_kernels:
                    break
                a_id, b_id = kids[i], kids[j]
                if a_id in merged_ids or b_id in merged_ids:
                    continue
                ka, kb = kernels[a_id], kernels[b_id]
                embed_a = ka.self_model.self_embedding.data
                embed_b = kb.self_model.self_embedding.data
                sim = F.cosine_similarity(
                    embed_a.unsqueeze(0), embed_b.unsqueeze(0)
                ).item()
                if sim > self.merge_similarity and min(ka.energy, kb.energy) < 3.0:
                    events.append(MergeEvent(kernel_a=a_id, kernel_b=b_id))
                    merged_ids.update([a_id, b_id])

        # SPAWN: high-energy mature kernels reproduce
        future_n = n - deaths - len([e for e in events if isinstance(e, MergeEvent)])
        for kid, k in kernels.items():
            if future_n >= self.max_kernels:
                break
            if (
                k.energy > self.spawn_energy
                and k.t > self.min_age
                and kid not in merged_ids
            ):
                events.append(SpawnEvent(parent_id=kid))
                future_n += 1

        return events
