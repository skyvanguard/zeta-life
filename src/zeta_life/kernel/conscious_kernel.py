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
from .dream_engine import DreamEngine
from .persistence import PersistenceLayer
from .temporal_features import OscillatorBank
from ..integration.formal_equations import compute_phi_c, compute_psi, compute_psi_hill


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
    psi : float
        Scalar consciousness index Psi (0-1 range, formal equation).
    reflected : bool
        Whether self-reflection ran this step.
    dreamed : bool
        Whether a dream cycle ran this step.
    """

    free_energy: float
    errors: dict  # {channel: magnitude}
    action: Tensor
    psi: float = 0.0
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
        latent_weight: float = 0.0,
        alpha: float = 1.0,
        psi_mode: str = "hill",
        psi_fe_scale: float = 5.0,
        psi_hill_n: float = 4.0,
        psi_hill_K: float = 0.1,
        psi_prec_half: float = 5.0,
        psi_w_prec: float = 0.45,
        psi_w_ref: float = 0.15,
        action_mode: str = "reactive",
        preference: Tensor | None = None,
        action_candidates: list[Tensor] | None = None,
        explore_eps: float = 0.0,
        efe_epistemic_weight: float = 0.0,
        temporal_features: OscillatorBank | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.reflect_interval = reflect_interval
        self.dream_interval = dream_interval
        self.save_interval = save_interval
        self.latent_weight = latent_weight
        self.alpha = alpha
        # Psi metric configuration.
        #   psi_mode="hill"   -> DEFAULT. Bounded Hill metric that discriminates
        #                        degrees of integration (see compute_psi_hill).
        #                        The cubic form saturates to 1.0 for any
        #                        supercritical input, so it cannot separate
        #                        coherent input from noise — hill is the usable
        #                        metric and is therefore the default.
        #   psi_mode="cubic"  -> original Psi = B^3 + Phi (clamped). Kept to
        #                        reproduce earlier (pre-fix) paper results.
        # psi_fe_scale controls how strongly free energy maps to Phi:
        #   Phi = 1/(1 + psi_fe_scale * free_energy). Larger values are needed when
        #   the system's free energy is small (e.g. after the interoceptive fix).
        self.psi_mode = psi_mode
        self.psi_fe_scale = psi_fe_scale
        self.psi_hill_n = psi_hill_n
        self.psi_hill_K = psi_hill_K
        # Binding-force (F_i) calibration for the critical threshold Phi_c.
        #   F_i = psi_w_prec * prec_term + psi_w_ref * reflection_convergence
        #   prec_term = prec_mean / (prec_mean + psi_prec_half)   in [0, 1)
        # The bounded prec_term replaces the old unbounded `prec_mean / 10`, which
        # was calibrated for precisions ~O(1). Once update_precisions() began
        # training precisions toward inverse-error-variance they grew to O(10-50),
        # inflating F_i above alpha so Phi_c = F_i/(alpha-C) exceeded Phi for ALL
        # inputs -> B<0 -> Psi collapsed to 0 and stopped discriminating coherence
        # from noise. Capping F_i at (psi_w_prec + psi_w_ref) keeps Phi_c safely
        # below Phi for coherent input even as precisions saturate, so Psi stays
        # supercritical for structure and subcritical for noise indefinitely.
        self.psi_prec_half = psi_prec_half
        self.psi_w_prec = psi_w_prec
        self.psi_w_ref = psi_w_ref

        # --- Action selection (agency) ---
        #   action_mode="reactive" -> DEFAULT. action = softmax(stimulus) (with the
        #                             optional latent_weight bias). Byte-identical to
        #                             the pre-agency kernel.
        #   action_mode="efe"      -> Active-inference action selection: pick the
        #                             candidate action whose imagined outcome
        #                             minimises expected free energy G(a) toward the
        #                             preference C. Requires `preference`.
        # G(a) = KL(C || softmax(imagine([a])))            [pragmatic value]
        #        - efe_epistemic_weight * H(softmax(imagine([a])))   [epistemic, coarse]
        # The chosen action becomes `actual_self`, so it flows into the
        # interoceptive channel, memory and last_action — which keeps the world
        # model trained on the action ACTUALLY taken (the alignment that makes
        # planning work; see the agency investigation).
        self.action_mode = action_mode
        self.preference = (
            (preference / preference.sum()).detach()
            if preference is not None else None
        )
        self.explore_eps = explore_eps
        self.efe_epistemic_weight = efe_epistemic_weight
        if action_candidates is not None:
            self._action_candidates = [a.detach() for a in action_candidates]
        else:
            # Default candidate set: the pure (one-hot) actions plus the uniform
            # action — a minimal basis that spans "push one channel" vs "stay flat".
            cands = [F.one_hot(torch.tensor(i), obs_dim).float() for i in range(obs_dim)]
            cands.append(torch.full((obs_dim,), 1.0 / obs_dim))
            self._action_candidates = cands

        # --- Temporal features (zeta in the control path) ---
        # When supplied, an OscillatorBank turns the step index t into a feature
        # vector fed to the world model's transition, so the model can ANTICIPATE
        # time-structured dynamics. This is the path through which the zeta
        # frequencies actually influence learning/control (vs the dream rhythm,
        # which only schedules consolidation). Default None -> temporal_dim=0 ->
        # byte-identical to the pre-temporal kernel.
        self.temporal_features = temporal_features
        temporal_dim = temporal_features.dim if temporal_features is not None else 0

        # --- Core components ---
        self.world_model = WorldModel(obs_dim, latent_dim, obs_dim, temporal_dim=temporal_dim)
        # If the feature bank has trainable frequencies (the "learned" arm), fold
        # its parameters into the world model's optimizer so they are trained by
        # the prior prediction loss alongside the transition.
        if temporal_features is not None:
            trainable = [p for p in temporal_features.parameters() if p.requires_grad]
            if trainable:
                self.world_model.optimizer.add_param_group({"params": trainable})
        self.self_model = SelfModel(obs_dim, embed_dim)
        self.error_engine = PredictionErrorEngine(4)
        self.fast_memory = FastMemory(500, 0.3)
        self.slow_memory = SlowMemory(obs_dim, outcome_dim=obs_dim)
        self.dream_engine = DreamEngine(
            self.fast_memory,
            self.slow_memory,
            self.self_model,
        )

        # --- Latent bias projection (fixed random -- natural diversity) ---
        self._latent_to_action = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim // 2, obs_dim),
        )
        for p in self._latent_to_action.parameters():
            p.requires_grad_(False)

        # --- State ---
        self.last_action = torch.zeros(obs_dim)
        self.t: int = 0
        self.energy: float = 5.0
        self._last_result: StepResult | None = None

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

        # Temporal feature for this step (None when no bank is configured). For a
        # trainable bank this carries grad so the prior loss can train it.
        tf_vec = (
            self.temporal_features(self.t)
            if self.temporal_features is not None else None
        )

        # ---- 1. PREDICT (prior) ----
        # predict() returns tensors with grad_fn for learning
        predicted_obs, _ = self.world_model.predict(self.last_action, tf_vec)
        predicted_self = self.self_model.predict_self(self.last_action)
        raw_self = F.softmax(stimulus, dim=-1)
        if self.action_mode == "efe" and self.preference is not None:
            # Active-inference action selection. predict() above already advanced
            # the latent, so imagine() inside the planner uses the current latent.
            actual_self = self._select_action_efe(raw_self, tf_vec)
        elif self.latent_weight > 0.0:
            latent_bias = self._latent_to_action(self.world_model.latent_state.detach())
            actual_self = F.softmax(raw_self + self.latent_weight * latent_bias, dim=-1)
        else:
            actual_self = raw_self

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
        # Posterior step: fold the observation into the latent and train the
        # encoder (replaces the old line that overwrote the recurrent latent
        # with a detached encode(stimulus), which froze the encoder and erased
        # the transition's temporal memory).
        self.world_model.observe(stimulus)

        # Train the self-model from the interoceptive error so its channel
        # reflects learned self-prediction instead of a fixed offset.
        self.self_model.update_from_error(errors['interoceptive']['raw'])

        # Train the per-channel precisions toward inverse error variance, so the
        # precision term in Psi actually reflects channel reliability instead of
        # staying frozen at its initial value.
        self.error_engine.update_precisions(errors)

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

        result = StepResult(
            free_energy=free_energy.item(),
            errors=errors_summary,
            action=action,
            psi=self._compute_psi(free_energy.item()),
            reflected=reflected,
            dreamed=dreamed,
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Consciousness computation (Psi = B^3 + Phi)
    # ------------------------------------------------------------------

    def _compute_psi(self, free_energy: float) -> float:
        """Derive formal consciousness index Psi from kernel signals.

        Maps internal kernel signals to the formal equation parameters:
        - Phi (integrated information): inverse of free energy + memory bonus
        - F_i (binding force): mean precision + reflection convergence bonus
        - C (coherence cost): mean recent prediction errors
        - alpha: coupling parameter (constructor arg)

        Returns
        -------
        float
            Consciousness index clamped to [0, 1].
        """
        # Phi: inverse of free energy + episodic memory bonus.
        # The cubic mode keeps the historical /10 scaling; the hill mode uses a
        # configurable scale so Phi actually responds to free energy (the /10
        # compresses Phi to ~0.96 regardless of input, which the cubic form then
        # saturates anyway).
        if self.psi_mode == "hill":
            phi_base = 1.0 / (1.0 + self.psi_fe_scale * free_energy)
        else:
            phi_base = 1.0 / (1.0 + free_energy / 10.0)
        mem_ratio = len(self.fast_memory) / 500.0
        phi = phi_base + 0.2 * mem_ratio  # range ~[0.0, 1.2]

        # F_i: bounded precision term + reflection convergence bonus.
        # prec_term saturates in [0, 1) so trained precisions (which grow without
        # bound as inverse error variance) cannot inflate F_i past its cap of
        # (psi_w_prec + psi_w_ref). That cap keeps Phi_c below Phi for coherent
        # input, which is what lets Psi discriminate structure from noise even at
        # long horizons. See the constructor note for the full rationale.
        precisions = self.error_engine.precisions  # tensor (4,)
        prec_mean = float(precisions.mean().item())
        prec_term = prec_mean / (prec_mean + self.psi_prec_half)
        F_i = self.psi_w_prec * prec_term
        if self.self_model.reflection_history:
            last_ref = self.self_model.reflection_history[-1]
            ref_convergence = 1.0 / (1.0 + last_ref[-1]['prediction_error'])
            F_i += self.psi_w_ref * ref_convergence

        # C: coherence cost from recent errors
        recent = self.error_engine.recent_errors()  # tensor (4,)
        C = float(recent.mean().item()) / 5.0

        # Psi via formal equations
        phi_c = compute_phi_c(F_i, self.alpha, C)
        if self.psi_mode == "hill":
            return compute_psi_hill(phi, phi_c, self.psi_hill_n, self.psi_hill_K)
        psi = compute_psi(phi, phi_c)
        return min(1.0, max(0.0, psi))

    # ------------------------------------------------------------------
    # Action selection (active inference)
    # ------------------------------------------------------------------

    def _select_action_efe(
        self, reactive_action: Tensor, temporal_feat: Tensor | None = None
    ) -> Tensor:
        """Select an action by minimising expected free energy toward `preference`.

        For each candidate action ``a`` (the configured candidate set plus the
        reactive action), the world model imagines the resulting observation
        without mutating state, and we score:

            G(a) = KL(preference || softmax(imagine([a])))     [pragmatic value]
                   - efe_epistemic_weight * H(softmax(imagine([a])))  [epistemic]

        and return ``argmin_a G(a)``. With probability ``explore_eps`` we instead
        return a random (normalised) action — exploration, which is what lets the
        world model learn the action->outcome dynamics in the first place.

        The pragmatic term drives the agent toward preferred outcomes; the
        epistemic term (a coarse outcome-entropy proxy, off by default) rewards
        informative actions. See Friston et al. 2015, "Active inference and
        epistemic value".
        """
        if self.explore_eps > 0.0 and float(torch.rand(1).item()) < self.explore_eps:
            a = torch.rand(self.obs_dim)
            return (a / a.sum()).detach()

        pref = self.preference
        log_pref = pref.clamp(min=1e-6).log()
        feat = temporal_feat.detach() if temporal_feat is not None else None
        best, best_g = reactive_action.detach(), float("inf")
        for a in self._action_candidates + [reactive_action]:
            pred_obs = self.world_model.imagine([a], feat)[0]
            proj = F.softmax(pred_obs, dim=-1)
            log_proj = proj.clamp(min=1e-6).log()
            pragmatic = float((pref * (log_pref - log_proj)).sum())  # KL(pref || proj)
            entropy = float(-(proj * log_proj).sum())
            g = pragmatic - self.efe_epistemic_weight * entropy
            if g < best_g:
                best_g, best = g, a.detach()
        return best

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
