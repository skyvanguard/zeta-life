"""OrganismState -- emergent global consciousness metrics.

Tracks diversity, coherence, information integration, and turnover
across the kernel population to compute a composite consciousness index.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor


class OrganismState:
    """Emergent consciousness metrics for the organism.

    Consciousness Index:
        0.30 * diversity + 0.25 * coherence + 0.25 * phi + 0.20 * turnover
    """

    def __init__(self) -> None:
        self.diversity: float = 0.0
        self.coherence: float = 0.0
        self.phi_global: float = 0.0
        self.turnover: float = 0.0
        self.consciousness_index: float = 0.0
        self.population_history: deque[int] = deque(maxlen=1000)
        self.consciousness_history: deque[float] = deque(maxlen=1000)

    def update(self, kernels: dict, gw) -> None:
        """Recompute all metrics from current kernel states."""
        if len(kernels) < 2:
            self.diversity = 0.0
            self.coherence = 1.0
            self.phi_global = 0.0
            self.turnover = 0.0
            self.consciousness_index = 0.0
            return

        # Diversity: 1 - avg cosine similarity of embeddings
        embeds = [k.self_model.self_embedding.data for k in kernels.values()]
        sims = []
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                sim = F.cosine_similarity(
                    embeds[i].unsqueeze(0), embeds[j].unsqueeze(0)
                ).item()
                sims.append(sim)
        self.diversity = max(0.0, min(1.0, 1.0 - (sum(sims) / max(len(sims), 1))))

        # Coherence: avg cosine similarity of actions
        actions = [k.last_action for k in kernels.values()]
        action_sims = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                sim = F.cosine_similarity(
                    actions[i].unsqueeze(0), actions[j].unsqueeze(0)
                ).item()
                action_sims.append(sim)
        self.coherence = sum(action_sims) / max(len(action_sims), 1)

        # Phi: geometric mean of diversity and coherence
        self.phi_global = (max(self.diversity, 0) * max(self.coherence, 0)) ** 0.5

        # Turnover: transition rate in last 20 winners
        history = list(gw.history)
        last_20 = history[-20:] if len(history) >= 20 else history
        if len(last_20) >= 2:
            transitions = sum(
                1 for i in range(1, len(last_20)) if last_20[i] != last_20[i - 1]
            )
            self.turnover = transitions / (len(last_20) - 1)
        else:
            self.turnover = 0.0

        # Consciousness index
        self.consciousness_index = (
            0.30 * self.diversity
            + 0.25 * max(self.coherence, 0)
            + 0.25 * self.phi_global
            + 0.20 * min(self.turnover, 1.0)
        )
        self.consciousness_index = max(0.0, min(1.0, self.consciousness_index))

        # History
        self.population_history.append(len(kernels))
        self.consciousness_history.append(self.consciousness_index)
