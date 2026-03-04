"""EnergyPool -- finite shared resource with conservation law.

Energy is conserved: sum of all kernel energies equals total_energy.
Winning the Global Workspace earns energy; existing costs energy.
"""

from __future__ import annotations


class EnergyPool:
    """Manages finite energy distributed among kernels.

    Parameters
    ----------
    total_energy : float
        Total conserved energy in the system.
    metabolic_cost : float
        Energy cost per step per kernel for existing.
    memory_cost : float
        Energy cost per step per 100 memories stored.
    dream_bonus : float
        Energy recovered when a kernel dreams.
    win_reward_scale : float
        Scale factor for winner reward (multiplied by 1/FE).
    """

    def __init__(
        self,
        total_energy: float = 10.0,
        metabolic_cost: float = 0.01,
        memory_cost: float = 0.005,
        dream_bonus: float = 0.02,
        win_reward_scale: float = 0.1,
    ) -> None:
        self.total_energy = total_energy
        self.metabolic_cost = metabolic_cost
        self.memory_cost = memory_cost
        self.dream_bonus = dream_bonus
        self.win_reward_scale = win_reward_scale

    def reward_winner(self, winner_id: int, kernels: dict) -> None:
        """Transfer energy from losers to the winner."""
        winner = kernels[winner_id]
        fe = getattr(winner, '_last_result', None)
        fe_val = fe.free_energy if fe else 0.3
        reward = self.win_reward_scale * (1.0 / max(fe_val, 1e-6))
        reward = min(reward, 1.0)  # cap reward

        n_losers = len(kernels) - 1
        if n_losers <= 0:
            return

        per_loser = reward / n_losers
        for kid, k in kernels.items():
            if kid == winner_id:
                k.energy += reward
            else:
                k.energy -= per_loser

    def decay_all(self, kernels: dict) -> None:
        """Apply metabolic costs and dream bonuses."""
        for k in kernels.values():
            k.energy -= self.metabolic_cost
            mem_len = len(k.fast_memory) if hasattr(k, 'fast_memory') else 0
            k.energy -= self.memory_cost * (mem_len / 100.0)

            result = getattr(k, '_last_result', None)
            if result and getattr(result, 'dreamed', False):
                k.energy += self.dream_bonus

    def normalize(self, kernels: dict) -> None:
        """Enforce energy conservation by scaling all energies."""
        if not kernels:
            return
        total = sum(k.energy for k in kernels.values())
        if total <= 0:
            per_kernel = self.total_energy / len(kernels)
            for k in kernels.values():
                k.energy = per_kernel
            return
        scale = self.total_energy / total
        for k in kernels.values():
            k.energy *= scale
