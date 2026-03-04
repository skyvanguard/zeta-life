"""ConsciousKernel -- main orchestrator for the Conscious Kernel.

Ties together all kernel components into a single step-by-step loop that
implements the full Active Inference cycle:

    PERCEIVE -> PREDICT -> COMPARE -> UPDATE -> MEMORIZE -> ACT -> REFLECT -> DREAM

Each call to :meth:`step` advances the system by one time step, producing
a :class:`StepResult` that summarises the free energy, per-channel errors,
selected action, and flags for reflection and dreaming.

Components:
    - WorldModel:          Predictive model of the environment
    - SelfModel:           Recursive self-modeling with Strange Loop
    - PredictionErrorEngine: Multi-channel precision-weighted errors
    - PrecisionController: Learned precision hyper-model
    - FastMemory:          Episodic hippocampal buffer
    - SlowMemory:          Semantic neocortical network
    - DreamEngine:         Zeta-driven sleep consolidation
    - PersistenceLayer:    Identity save/load across sessions
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .world_model import WorldModel
from .prediction_error import PredictionErrorEngine
from .self_model import SelfModel
from .complementary_memory import Episode, FastMemory, SlowMemory
from .precision_controller import PrecisionController
from .dream_engine import DreamEngine
from .persistence import PersistenceLayer


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single :meth:`ConsciousKernel.step` call.

    Parameters
    ----------
    free_energy : float
        Total variational free energy (>= 0).
    errors : dict
        Per-channel error magnitudes ``{channel_name: float}``.
    action : Tensor
        Action vector produced by the kernel this step.
    consciousness : float
        Scalar consciousness index (0-1 range, heuristic).
    reflected : bool
        Whether self-reflection ran this step.
    dreamed : bool
        Whether a dream cycle ran this step.
    """

    free_energy: float
    errors: dict  # {channel: magnitude}
    action: Tensor
    consciousness: float = 0.0
    reflected: bool = False
    dreamed: bool = False


# ---------------------------------------------------------------------------
# ConsciousKernel
# ---------------------------------------------------------------------------

class ConsciousKernel:
    """Main orchestrator that implements the full Active Inference loop.

    Creates and wires all kernel components on construction, then exposes
    a single :meth:`step` method that advances by one time step.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation/action space.
    latent_dim : int
        World model latent space dimensionality.
    embed_dim : int
        Self model identity embedding dimensionality.
    reflect_interval : int
        Run self-reflection every *reflect_interval* steps.
    dream_interval : int
        Run a dream consolidation cycle every *dream_interval* steps.
    save_interval : int
        Auto-save interval (reserved for future use).
    """

    def __init__(
        self,
        obs_dim: int = 4,
        latent_dim: int = 32,
        embed_dim: int = 16,
        reflect_interval: int = 5,
        dream_interval: int = 50,
        save_interval: int = 100,
    ) -> None:
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.reflect_interval = reflect_interval
        self.dream_interval = dream_interval
        self.save_interval = save_interval

        # --- Core components ---
        self.world_model = WorldModel(obs_dim, latent_dim, obs_dim)
        self.self_model = SelfModel(obs_dim, embed_dim)
        self.error_engine = PredictionErrorEngine(4)
        self.precision_controller = PrecisionController(obs_dim, 4)
        self.fast_memory = FastMemory(500, 0.3)
        self.slow_memory = SlowMemory(obs_dim, outcome_dim=obs_dim)
        self.dream_engine = DreamEngine(
            self.fast_memory,
            self.slow_memory,
            self.self_model,
        )

        # --- State ---
        self.last_action = torch.zeros(obs_dim)
        self.t: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self, stimulus: Tensor) -> StepResult:
        """Advance the kernel by one time step.

        Implements the full Active Inference cycle:

        1. PERCEIVE  -- encode stimulus into latent space
        2. PREDICT   -- predict observation and self-state from last action
        3. COMPARE   -- compute multi-channel prediction errors + free energy
        4. UPDATE    -- train world model from perceptual error
        5. MEMORIZE  -- store episode in fast memory, integrate into slow memory
        6. ACT       -- select action (current self-state, detached)
        7. REFLECT   -- periodic Strange Loop self-reflection
        8. DREAM     -- periodic dream consolidation cycle

        Parameters
        ----------
        stimulus : Tensor
            Observation vector of shape ``(obs_dim,)``.

        Returns
        -------
        StepResult
            Summary of this step's computations.
        """
        self.t += 1

        # ---- 1. PERCEIVE ----
        observation = self.world_model.encode(stimulus)

        # ---- 2. PREDICT ----
        # predict() returns tensors with grad_fn for learning
        predicted_obs, _ = self.world_model.predict(self.last_action)
        predicted_self = self.self_model.predict_self(self.last_action)
        actual_self = F.softmax(stimulus, dim=-1)

        # ---- 3. COMPARE ----
        # Build predictions/observations dicts for 4 channels
        predictions = {
            'perceptual': predicted_obs,
            'interoceptive': predicted_self,
            'temporal': torch.zeros(self.obs_dim),
            'epistemic': torch.zeros(self.obs_dim),
        }
        observations = {
            'perceptual': stimulus,
            'interoceptive': actual_self,
            'temporal': torch.zeros(self.obs_dim),
            'epistemic': torch.zeros(self.obs_dim),
        }

        errors = self.error_engine.compute_errors(predictions, observations)
        free_energy = self.error_engine.free_energy(errors)

        # ---- 4. UPDATE ----
        # Train world model from perceptual prediction error.
        # The raw error from predicted_obs - stimulus has grad_fn from predict(),
        # so update_from_error can backpropagate.
        self.world_model.update_from_error(errors['perceptual']['raw'])
        self.world_model.latent_state = self.world_model.encode(stimulus).detach()

        # ---- 5. MEMORIZE ----
        surprise = max(
            errors[ch]['magnitude'].item() for ch in self.error_engine.channels
        )
        dominant = self._dominant_name(actual_self)

        episode = Episode(
            stimulus=stimulus.detach(),
            observation=stimulus.detach(),
            archetype_state=actual_self.detach(),
            surprise=surprise,
            dominant=dominant,
            timestamp=self.t,
            prediction_errors={
                ch: errors[ch]['magnitude'].item()
                for ch in self.error_engine.channels
            },
        )
        self.fast_memory.store(episode)
        self.slow_memory.integrate(
            actual_self.detach(),
            actual_self.detach(),
        )

        # ---- 6. ACT ----
        action = actual_self.detach()
        self.last_action = action

        # ---- 7. REFLECT ----
        reflected = False
        if self.t % self.reflect_interval == 0:
            self.self_model.reflect(actual_self.detach(), depth=3)
            reflected = True

        # ---- 8. DREAM ----
        dreamed = False
        if self.t % self.dream_interval == 0 and len(self.fast_memory) > 0:
            self.dream_engine.dream_cycle(30)
            dreamed = True

        # Build simplified errors dict for the result
        errors_summary = {
            ch: errors[ch]['magnitude'].item()
            for ch in self.error_engine.channels
        }

        return StepResult(
            free_energy=free_energy.item(),
            errors=errors_summary,
            action=action,
            consciousness=0.0,
            reflected=reflected,
            dreamed=dreamed,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, base_path: str, identity_name: str = 'default') -> None:
        """Save all kernel state to disk.

        Parameters
        ----------
        base_path : str
            Root directory for checkpoints.
        identity_name : str
            Name for the identity checkpoint.
        """
        pl = PersistenceLayer(base_path)
        pl.save_state(self._get_components(), identity_name)

    def load(self, base_path: str, identity_name: str = 'default') -> None:
        """Restore kernel state from disk.

        Parameters
        ----------
        base_path : str
            Root directory for checkpoints.
        identity_name : str
            Name of the identity to load.
        """
        pl = PersistenceLayer(base_path)
        self.t = pl.load_state(self._get_components(), identity_name)
        # Wake-up reflection after loading
        self.self_model.reflect(torch.zeros(self.obs_dim), depth=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_components(self) -> dict:
        """Build the components dict expected by PersistenceLayer."""
        return {
            'world_model': self.world_model,
            'self_model': self.self_model,
            'error_engine': self.error_engine,
            'precision_controller': self.precision_controller,
            'fast_memory': self.fast_memory,
            'slow_memory': self.slow_memory,
            'step': self.t,
        }

    @staticmethod
    def _dominant_name(state: Tensor) -> str:
        """Return the name of the dominant vertex in a state vector.

        Parameters
        ----------
        state : Tensor
            State vector (typically a softmax distribution).

        Returns
        -------
        str
            One of 'PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS', or 'UNKNOWN'.
        """
        names = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS']
        idx = state.argmax().item()
        return names[idx] if idx < len(names) else 'UNKNOWN'
