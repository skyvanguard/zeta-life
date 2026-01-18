"""
Zeta Cellular Automata - Game of Life with zeta-derived kernels.

Phases:
1. Zeta-structured initialization
2. Zeta-weighted kernel replacing Moore neighborhood
3. Temporal memory via Laplace transform
4. Neural CA with zeta perception (differentiable)
"""

from .zeta_game_of_life import ZetaGameOfLife, ZetaKernel
from .zeta_gol_fase2 import ZetaNeighborhoodKernel, ZetaWeightedGoL
from .zeta_gol_fase3 import ZetaFullSystem, ZetaLaplaceOperator
# Note: ZetaNCA classes require PyTorch - import directly if needed:
# from .zeta_neural_ca import ZetaNCA, ZetaNCATrainer
