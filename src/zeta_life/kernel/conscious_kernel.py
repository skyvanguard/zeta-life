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

from collections import deque
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
from .policy import Actor, Critic
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
        psi_prec_adaptive: bool = True,
        psi_prec_decay: float = 0.99,
        psi_w_prec: float = 0.45,
        psi_w_ref: float = 0.15,
        action_mode: str = "reactive",
        preference: Tensor | None = None,
        action_candidates: list[Tensor] | None = None,
        explore_eps: float = 0.0,
        efe_epistemic_weight: float = 0.0,
        efe_n_samples: int = 0,
        efe_sample_scale: float = 1.5,
        efe_horizon: int = 1,
        efe_discount: float = 1.0,
        efe_obs_norm: str = "softmax",
        efe_cem_iters: int = 0,
        efe_cem_elite_frac: float = 0.3,
        efe_epistemic_mode: str = "entropy",
        wm_disagreement_heads: int = 0,
        imag_horizon: int = 5,
        imag_rollouts: int = 8,
        imag_lambda: float = 0.95,
        imag_gamma: float = 0.97,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_explore: float = 0.0,
        dreamer_epistemic_weight: float = 0.0,
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
        # --- Psi: integration index (ENGINEERING HEURISTIC) ---
        # IMPORTANT: Psi is a bounded, monotone integration *heuristic*, not a
        # derived or proven measure of consciousness (it is neither IIT's phi nor
        # a free energy). It maps kernel signals (free energy, precision, coherence
        # cost) onto a [0,1) order parameter that discriminates coherent input from
        # noise. The constants below are CALIBRATION, not theory.
        #
        #   psi_mode="hill"  -> DEFAULT and recommended. Bounded Hill metric that
        #                       discriminates degrees of integration.
        #   psi_mode="cubic" -> DEPRECATED. Psi = B^3 + Phi saturates to 1.0 for
        #                       any supercritical input (cannot discriminate);
        #                       kept only to reproduce earlier paper results.
        #   psi_fe_scale     -> calibration of the free-energy -> Phi reference
        #                       scale: Phi = 1/(1 + psi_fe_scale * free_energy).
        #                       Phi is NOT scale-invariant; this sets the
        #                       free-energy operating point and is task-dependent.
        self.psi_mode = psi_mode
        self.psi_fe_scale = psi_fe_scale
        self.psi_hill_n = psi_hill_n
        self.psi_hill_K = psi_hill_K
        # Binding force F_i = psi_w_prec * prec_term + psi_w_ref * reflection_conv,
        # with prec_term = prec_mean / (prec_mean + half)  in [0, 1).
        # SELF-CALIBRATING half-point (psi_prec_adaptive=True, default): `half`
        # tracks an EMA of the mean precision, so prec_term stays near 0.5 however
        # large the trained precisions grow. This REPLACES the old fixed
        # psi_prec_half clamp: that constant had to be retuned every time the
        # substrate improved (training precisions toward inverse-error-variance
        # made them grow to O(10-50), inflating F_i past alpha so Phi_c > Phi for
        # ALL inputs -> Psi collapsed to 0). With the adaptive half, F_i is bounded
        # and ASYMPTOTICALLY scale-invariant -- there is a brief transient while
        # the EMA tracks a new precision scale -- with no reactive clamp to retune.
        # psi_prec_half is now only the fixed fallback when psi_prec_adaptive=False
        # (adaptive mode bootstraps the EMA from the initial precision scale, so it
        # is genuinely independent of this constant).
        self.psi_prec_half = psi_prec_half
        self.psi_prec_adaptive = psi_prec_adaptive
        self.psi_prec_decay = psi_prec_decay
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
        # Continuous action candidates + planning horizon (the "controla" work).
        #   efe_n_samples > 0 -> in addition to the discrete candidates, sample
        #     this many CONTINUOUS simplex actions (softmax of Gaussian logits,
        #     scale efe_sample_scale). The agency investigation found the original
        #     one-hot-only candidate set is OUT OF DISTRIBUTION for a world model
        #     trained on continuous actions, which caps control on non-vertex
        #     targets; sampled continuous candidates are training-consistent and
        #     let the planner actually reach arbitrary targets.
        #   efe_horizon > 1 -> evaluate each candidate as a SUSTAINED action over
        #     this many imagined steps, summing discounted (efe_discount) expected
        #     free energy. Helps when the environment has inertia.
        # Defaults (0, 1) are byte-identical to the prior discrete 1-step planner.
        self.efe_n_samples = efe_n_samples
        self.efe_sample_scale = efe_sample_scale
        self.efe_horizon = efe_horizon
        self.efe_discount = efe_discount
        # How the imagined observation is projected to the simplex before the KL
        # to the preference (see _to_simplex). "l1" floors at 0 and L1-normalises:
        # exact for the near-simplex outputs the predictor learns (it is trained
        # on simplex observations), with negative outliers floored to ~0 -- NOT a
        # general faithful map of arbitrary R^n. It lets the planner score
        # continuous actions near a non-vertex target correctly. "softmax"
        # (default, legacy) re-flattens the prediction, rewarding only EXTREME
        # (one-hot) actions and capping control on non-vertex targets.
        self.efe_obs_norm = efe_obs_norm
        # Cross-Entropy Method (CEM) for continuous action selection. When
        # efe_cem_iters > 0 the planner refines a Gaussian sampling distribution
        # (in logit space) over iterations, keeping the top efe_cem_elite_frac
        # each round -- finding better continuous actions per sample, especially
        # under a tight budget. efe_n_samples is the per-iteration population.
        # efe_cem_iters=0 (default) keeps the flat candidate set (random shooting).
        self.efe_cem_iters = efe_cem_iters
        self.efe_cem_elite_frac = efe_cem_elite_frac
        # Epistemic term of the EFE: "entropy" (default, coarse outcome-entropy
        # proxy) or "disagreement" (the world model's ensemble disagreement, a
        # real info-gain signal -- requires wm_disagreement_heads > 0). NOTE:
        # disagreement is ~O(1e-2), so efe_epistemic_weight must be large
        # (O(10-100)) for it to compete with the pragmatic KL.
        self.efe_epistemic_mode = efe_epistemic_mode
        # Dreamer-style behaviour learning (action_mode="dreamer"): an amortized
        # actor trained in imagination with a critic and value gradients through
        # the differentiable latent dynamics; reward = -EFE toward the preference.
        # The actor acts at constant cost per step (no search). Modules are created
        # below once latent_dim/obs_dim are known.
        self.imag_horizon = imag_horizon
        self.imag_rollouts = imag_rollouts
        self.imag_lambda = imag_lambda
        self.imag_gamma = imag_gamma
        self.actor_explore = actor_explore
        self.dreamer_epistemic_weight = dreamer_epistemic_weight
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
        self.world_model = WorldModel(
            obs_dim, latent_dim, obs_dim, temporal_dim=temporal_dim,
            disagreement_heads=wm_disagreement_heads,
        )
        # If the feature bank has trainable frequencies (the "learned" arm), fold
        # its parameters into the world model's optimizer so they are trained by
        # the prior prediction loss alongside the transition.
        if temporal_features is not None:
            trainable = [p for p in temporal_features.parameters() if p.requires_grad]
            if trainable:
                self.world_model.optimizer.add_param_group({"params": trainable})

        # Dreamer actor/critic + a small replay buffer of recent latents.
        self.actor = None
        self.critic = None
        self._latent_buffer = None
        if action_mode == "dreamer":
            self.actor = Actor(latent_dim, obs_dim)
            self.critic = Critic(latent_dim)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
            self._latent_buffer = deque(maxlen=200)

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
        # EMA of the mean precision; the self-calibrating half-point for F_i.
        self._prec_ref: float | None = None
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
            # the latent, so imagine() inside the planner uses the current latent;
            # the planner rebuilds the future temporal codes phi(t+1..t+H) itself.
            actual_self = self._select_action_efe(raw_self)
        elif self.action_mode == "dreamer" and self.preference is not None:
            # Amortized actor: act from the current latent at constant cost.
            actual_self = self._select_action_dreamer()
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

        # Dreamer: buffer the posterior latent and improve the actor/critic in
        # imagination (no effect in other action modes).
        if self.action_mode == "dreamer" and self._latent_buffer is not None:
            self._latent_buffer.append(self.world_model.latent_state.detach().clone())
            self._train_behavior()

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
    # Integration index Psi (engineering heuristic; bounded Hill metric)
    # ------------------------------------------------------------------

    def _compute_psi(self, free_energy: float) -> float:
        """Compute the integration index Psi from kernel signals.

        Psi is an ENGINEERING HEURISTIC (bounded, monotone), NOT a proven
        consciousness measure. It maps kernel signals onto the integration
        equations:
        - Phi (integration): inverse free energy (calibrated) + memory bonus
        - F_i (binding force): self-calibrated precision term + reflection conv.
        - C (coherence cost): mean recent prediction errors
        - alpha: coupling parameter (constructor arg)

        Returns
        -------
        float
            Integration index in [0, 1] (Hill mode) or clamped to [0, 1] (cubic).
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

        # F_i: self-calibrated precision term + reflection convergence bonus.
        # The saturating half-point tracks an EMA of the mean precision (adaptive
        # mode), so prec_term settles near 0.5 however large the trained precisions
        # grow -> F_i is bounded and ASYMPTOTICALLY scale-invariant (a brief
        # transient while the EMA tracks), with no fixed clamp to retune.
        precisions = self.error_engine.precisions  # tensor (4,)
        prec_mean = float(precisions.mean().item())
        if self._prec_ref is None:
            # Bootstrap from the system's OWN initial precision scale, not from
            # psi_prec_half — this keeps adaptive mode genuinely independent of the
            # former clamp constant (psi_prec_half only matters when adaptive=False).
            self._prec_ref = max(prec_mean, 1e-6)
        else:
            d = self.psi_prec_decay
            self._prec_ref = d * self._prec_ref + (1.0 - d) * prec_mean
        half = self._prec_ref if self.psi_prec_adaptive else self.psi_prec_half
        denom = prec_mean + half
        prec_term = prec_mean / denom if denom > 0 else 0.0
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

    def _select_action_efe(self, reactive_action: Tensor) -> Tensor:
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

        # CEM refinement (continuous) takes over when configured.
        if self.efe_cem_iters > 0:
            return self._select_action_cem(reactive_action)

        # Otherwise: flat candidate set = discrete basis + sampled CONTINUOUS
        # simplex actions (training-consistent) + the reactive action.
        candidates = list(self._action_candidates)
        if self.efe_n_samples > 0:
            logits = torch.randn(self.efe_n_samples, self.obs_dim) * self.efe_sample_scale
            samples = F.softmax(logits, dim=-1)
            candidates += [samples[i] for i in range(self.efe_n_samples)]
        candidates.append(reactive_action)

        best, best_g = reactive_action.detach(), float("inf")
        for a in candidates:
            g = self._efe_cost(a)
            if g < best_g:
                best_g, best = g, a.detach()
        return best

    def _to_simplex(self, x: Tensor) -> Tensor:
        """Project a (possibly unbounded) predictor output onto the simplex.

        "l1": floor at 0 and L1-normalise. Exact for the near-simplex outputs the
        predictor learns (it is trained on simplex-valued observations); negative
        outliers are floored to ~0 ("channel absent"), with a uniform fallback if
        nothing is positive. NOT a general faithful map of arbitrary R^n -- only
        of the predictor's trained range. "softmax": the legacy flattening map
        (over-rewards extreme one-hot actions; caps control on non-vertex targets).
        """
        if self.efe_obs_norm == "l1":
            p = x.clamp(min=0.0)
            s = p.sum()
            if float(s) <= 1e-8:
                return torch.full_like(x, 1.0 / x.numel())
            return p / s
        return F.softmax(x, dim=-1)

    def _horizon_features(self, horizon: int) -> list[Tensor] | None:
        """Future temporal codes phi(t+1..t+horizon) for an anticipatory rollout.

        Returns None when no temporal bank is configured. Using the FUTURE codes
        (not a frozen phi(t) broadcast to every step) is what makes a horizon>1
        plan genuinely anticipatory.
        """
        if self.temporal_features is None:
            return None
        return [self.temporal_features(self.t + k).detach()
                for k in range(1, horizon + 1)]

    def _efe_cost(self, a: Tensor) -> float:
        """Expected free energy of committing to action ``a``.

        Pragmatic value is summed and discounted over the horizon:
            sum_t discount^t * KL(preference || simplex(imagine_t)).
        Epistemic value depends on the mode:
          - "entropy":      per-step, discounted outcome entropy (a coarse proxy);
          - "disagreement": a single-step info-gain BONUS from the world model's
            ensemble disagreement on the committed action (the immediate
            information gain, NOT horizon-summed). Its magnitude is ~O(1e-2), so
            efe_epistemic_weight must be large (O(10-100)) to compete with the
            pragmatic KL.
        With a temporal bank the rollout uses the FUTURE codes phi(t+1..t+H), so a
        horizon>1 plan is anticipatory. imagine() does not mutate state (read-only).
        """
        pref = self.preference
        log_pref = pref.clamp(min=1e-6).log()
        horizon = max(1, self.efe_horizon)
        feats = self._horizon_features(horizon)
        preds = self.world_model.imagine([a] * horizon, feats)
        g, disc = 0.0, 1.0
        for pred_obs in preds:
            proj = self._to_simplex(pred_obs)
            log_proj = proj.clamp(min=1e-6).log()
            pragmatic = float((pref * (log_pref - log_proj)).sum())  # KL(pref||proj)
            g += disc * pragmatic
            if self.efe_epistemic_mode == "entropy":
                entropy = float(-(proj * log_proj).sum())
                g -= disc * self.efe_epistemic_weight * entropy
            disc *= self.efe_discount
        if self.efe_epistemic_mode == "disagreement":
            first_feat = feats[0] if feats is not None else None
            g -= self.efe_epistemic_weight * self.world_model.disagreement(a, first_feat)
        return g

    def _select_action_cem(self, reactive_action: Tensor) -> Tensor:
        """Cross-Entropy Method action search in logit space.

        Iteratively samples a population of continuous simplex actions from a
        Gaussian over logits, keeps the elite (lowest expected free energy), and
        refits the Gaussian to them. The discrete basis and the reactive action
        are ALSO scored, so CEM never returns worse than the flat candidate set.
        """
        obs_dim = self.obs_dim
        pop = self.efe_n_samples if self.efe_n_samples > 0 else 16
        elite_n = max(2, int(pop * self.efe_cem_elite_frac))
        # Seed the incumbent with the discrete basis + reactive action.
        best, best_g = reactive_action.detach(), float("inf")
        for a0 in self._action_candidates + [reactive_action]:
            c0 = self._efe_cost(a0)
            if c0 < best_g:
                best_g, best = c0, a0.detach()
        mu = torch.zeros(obs_dim)
        sigma = torch.full((obs_dim,), self.efe_sample_scale)
        for _ in range(self.efe_cem_iters):
            logits = mu + sigma * torch.randn(pop, obs_dim)
            actions = F.softmax(logits, dim=-1)
            costs = [self._efe_cost(actions[i]) for i in range(pop)]
            elite = sorted(range(pop), key=lambda i: costs[i])[:elite_n]
            if costs[elite[0]] < best_g:
                best_g, best = costs[elite[0]], actions[elite[0]].detach()
            elite_logits = logits[elite]
            mu = elite_logits.mean(dim=0)
            sigma = elite_logits.std(dim=0) + 1e-3
        return best

    # ------------------------------------------------------------------
    # Dreamer-style behaviour learning (amortized actor + critic)
    # ------------------------------------------------------------------

    def _select_action_dreamer(self) -> Tensor:
        """Act with the amortized actor on the current latent (constant cost)."""
        z = self.world_model.latent_state.detach().unsqueeze(0)
        with torch.no_grad():
            a = self.actor(z).squeeze(0)
        if self.actor_explore > 0.0:
            logits = a.clamp(min=1e-6).log() + self.actor_explore * torch.randn(self.obs_dim)
            a = F.softmax(logits, dim=-1)
        return a.detach()

    def _reward_from_pred(self, pred: Tensor) -> Tensor:
        """Reward = -KL(preference || simplex(pred)), batched. Shape (B,)."""
        p = pred.clamp(min=0.0)
        s = p.sum(dim=-1, keepdim=True)
        proj = torch.where(
            s > 1e-8, p / s.clamp(min=1e-8),
            torch.full_like(pred, 1.0 / pred.shape[-1]),
        )
        log_proj = proj.clamp(min=1e-6).log()
        log_C = self.preference.clamp(min=1e-6).log()
        kl = (self.preference * (log_C - log_proj)).sum(dim=-1)
        return -kl

    @staticmethod
    def _lambda_returns(rewards, values, gamma: float, lam: float):
        """TD(lambda) returns. rewards: list[(B,)] len H; values: list[(B,)] len H+1.

        R_t = r_t + gamma[(1-lam) V_{t+1} + lam R_{t+1}], R_H bootstraps with V_H.
        """
        H = len(rewards)
        returns = [None] * H
        nxt = values[H]
        for t in reversed(range(H)):
            nxt = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * nxt)
            returns[t] = nxt
        return returns

    def _train_behavior(self) -> None:
        """Improve actor & critic in imagination (Dreamer-style value gradients)."""
        if (self.preference is None or self._latent_buffer is None
                or len(self._latent_buffer) < self.imag_rollouts):
            return
        B, H = self.imag_rollouts, self.imag_horizon
        idxs = torch.randint(0, len(self._latent_buffer), (B,))
        z = torch.stack([self._latent_buffer[int(i)] for i in idxs]).detach()
        zs, rewards = [z], []
        for _ in range(H):
            a = self.actor(z)
            z, pred = self.world_model.imagine_grad(z, a)
            zs.append(z)
            rewards.append(self._reward_from_pred(pred))
        values = [self.critic(zt) for zt in zs]  # len H+1

        # Critic: regress V(z_t) onto detached lambda-returns.
        crit_targets = self._lambda_returns(
            [r.detach() for r in rewards], [v.detach() for v in values],
            self.imag_gamma, self.imag_lambda)
        critic_loss = torch.stack(
            [(values[t] - crit_targets[t]) ** 2 for t in range(H)]).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Actor: maximize lambda-returns (value gradients through the dynamics;
        # bootstrap value detached so the actor does not backprop into the critic).
        actor_returns = self._lambda_returns(
            rewards, [v.detach() for v in values], self.imag_gamma, self.imag_lambda)
        actor_loss = -torch.stack(actor_returns).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Hygiene: clear imagination grads that leaked into the world model
        # (the world model is trained only by real data, never by imagination).
        self.world_model.optimizer.zero_grad()

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
