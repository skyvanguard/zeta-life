# The Conscious Kernel — Design Document

**Date**: 2026-03-03
**Author**: skyvanguard + Claude
**Status**: Approved
**Supersedes**: N/A (new subsystem)

## 1. Motivation

Current AI systems (LLMs, AlphaZero) lack three fundamental properties that biological intelligence possesses:

1. **Continuous learning**: They don't learn from experience after deployment
2. **Persistent identity**: Close the session and everything is lost
3. **Self-awareness**: They don't model themselves as agents in the world

Zeta Life already has pieces that address these gaps (Strange Loop, AttractorMemory, IPUESA, DreamConsolidator), but they operate as independent subsystems. The Conscious Kernel integrates them into a unified Active Inference architecture.

### Theoretical Foundation

Based on "A Beautiful Loop" (Laukkonen, Friston & Chandaria, 2025), consciousness requires:

1. **Epistemic Field** — A generative model of the world
2. **Bayesian Binding** — Inferential competition where uncertainty-reducing hypotheses win
3. **Epistemic Depth** — Recursive self-modeling (the system models its own modeling)

The unique contribution of Zeta Life: using the Riemann zeta zeros as the temporal binding mechanism — analogous to brain oscillations (SO-Spindle-SWR) that synchronize information across levels during consolidation.

### References

- Laukkonen, Friston & Chandaria (2025). "A Beautiful Loop: Active Inference Theory of Consciousness". Neuroscience & Biobehavioral Reviews.
- COGITATE (Nature, 2025). Adversarial testing of IIT vs GNWT — neither survived intact, suggesting integration.
- Butlin & Schwitzgebel (2025). "Indicators of Consciousness in AI". Trends in Cognitive Sciences.
- Neural ODE + Memory-Augmented Transformers (Nature Scientific Reports, 2025) — -24% forgetting.
- Universal Neural Cellular Automata (GECCO 2025) — computational universality in CA.
- Biological Computationalism (ScienceDirect, 2025) — substrate-dependent consciousness.
- DreamerV3 (Nature, 2025) — learning within self-generated dreams.
- IWMT (Frontiers, 2020) — Integrated World Modeling Theory unifying IIT + GWT + FEP.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    CONSCIOUS KERNEL                      │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ WORLD MODEL  │◄──►│  SELF MODEL  │◄──►│ PRECISION │  │
│  │ (predicts    │    │ (Strange     │    │ CONTROLLER│  │
│  │  environment)│    │  Loop +      │    │ (hyper-   │  │
│  │              │    │  self-as-    │    │  model)   │  │
│  │              │    │  variable)   │    │           │  │
│  └──────┬───────┘    └──────┬───────┘    └─────┬─────┘  │
│         │                   │                   │        │
│         ▼                   ▼                   ▼        │
│  ┌─────────────────────────────────────────────────────┐│
│  │           PREDICTION ERROR ENGINE                    ││
│  │  error = (prediction - reality) * precision          ││
│  │  Channels: perceptual, interoceptive, temporal,      ││
│  │            epistemic                                  ││
│  └──────────────────────┬──────────────────────────────┘│
│                         │                                │
│         ┌───────────────┼───────────────┐                │
│         ▼               ▼               ▼                │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐        │
│  │FAST MEMORY │  │SLOW MEMORY │  │ ATTRACTOR   │        │
│  │(hippocampus│  │(neocortex) │  │ MEMORY      │        │
│  │ one-shot   │  │ gradual    │  │ (identity)  │        │
│  │ sparse)    │  │ dense)     │  │ (existing)  │        │
│  └─────┬──────┘  └─────┬──────┘  └──────┬──────┘        │
│        │               │                │                │
│        └───────┬───────┘                │                │
│                ▼                        │                │
│  ┌──────────────────┐                   │                │
│  │  DREAM ENGINE    │◄──────────────────┘                │
│  │  (consolidation) │                                    │
│  │  3-phase zeta    │                                    │
│  │  binding cycle   │                                    │
│  └──────────────────┘                                    │
│                                                          │
│  ┌──────────────────────────────────────────────────────┐│
│  │  PERSISTENCE LAYER                                    ││
│  │  Complete state → disk → restore on resume            ││
│  └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## 3. Component Specifications

### 3.1 World Model

**Purpose**: Generative model of the environment that predicts before perceiving.

**Evolution from**: `ZetaAttentivePredictive` (L1/L2/L3 hierarchical prediction)

**New capabilities**:
- Persistent `latent_state: Tensor[latent_dim]` that carries forward between steps
- `predict(action) -> Prediction` — top-down expectation BEFORE input arrives
- `encode(observation) -> Tensor` — bottom-up perception to latent space
- `imagine(action_sequence) -> list[Prediction]` — counterfactual simulation ("what if...")

**Architecture**:
```python
class WorldModel(nn.Module):
    def __init__(self, obs_dim=4, latent_dim=32, action_dim=4):
        self.encoder = nn.Sequential(          # observation -> latent
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.transition = nn.GRUCell(          # latent + action -> next latent
            input_size=action_dim,
            hidden_size=latent_dim
        )
        self.predictor = nn.Linear(latent_dim, obs_dim)  # latent -> predicted obs

    def predict(self, action):
        next_latent = self.transition(action, self.latent_state)
        return self.predictor(next_latent), next_latent

    def encode(self, observation):
        return self.encoder(observation)

    def imagine(self, action_sequence):
        """Simulate without real input — internal 'thinking'."""
        imagined = []
        state = self.latent_state.clone()
        for action in action_sequence:
            state = self.transition(action, state)
            imagined.append(self.predictor(state))
        return imagined
```

**Design decisions**:
- GRUCell for transition (lighter than LSTM, sufficient for state tracking)
- `latent_dim=32` (4 archetypes expanded to richer representation)
- Predictor outputs in observation space for direct error computation
- `imagine()` detaches from real input — enables planning and counterfactual reasoning

### 3.2 Self Model

**Purpose**: Recursive self-modeling with variable depth (Strange Loop evolution).

**Evolution from**: `_self_reflection_cycle()` in `zeta_conscious_self.py`

**New capabilities**:
- `self_embedding: Tensor[embed_dim]` — persistent numerical identity
- 4 levels of epistemic depth (vs 1 today)
- `predict_self(action) -> Tensor` — "how will I feel if I do X?"
- `identity_distance(state) -> float` — distance from core identity

**Epistemic depth levels**:

| Level | Name | Description | Existing? |
|-------|------|-------------|-----------|
| 1 | Perception | "I receive stimulus X" | Yes |
| 2 | Agency | "I caused Y" | Partial |
| 3 | Meta-cognition | "I am thinking about X" | Yes (Strange Loop) |
| 4 | Recursive depth | "I notice that I am thinking about thinking..." | New |

**Architecture**:
```python
class SelfModel(nn.Module):
    def __init__(self, state_dim=4, embed_dim=16):
        self.self_embedding = nn.Parameter(torch.randn(embed_dim) * 0.1)
        self.state_to_embed = nn.Linear(state_dim, embed_dim)
        self.reflection_net = nn.GRUCell(embed_dim, embed_dim)
        self.embed_to_prediction = nn.Linear(embed_dim, state_dim)

    def reflect(self, current_state, depth=3):
        """Strange Loop with controlled depth."""
        embed = self.state_to_embed(current_state)
        reflections = []

        for d in range(depth):
            # Each level: observe current embed, process through self-reference
            combined = embed + self.self_embedding  # self-as-variable
            embed = self.reflection_net(combined, embed)
            reflections.append({
                'depth': d,
                'state': embed.clone(),
                'prediction_error': torch.norm(embed - self.self_embedding).item()
            })

        # Update self_embedding (slow moving average)
        with torch.no_grad():
            self.self_embedding.data = (
                0.95 * self.self_embedding.data + 0.05 * embed.detach()
            )

        return reflections

    def predict_self(self, future_action):
        """How will I feel if I take this action?"""
        projected = self.reflection_net(future_action, self.self_embedding)
        return self.embed_to_prediction(projected)

    def identity_distance(self, state):
        """Distance from core identity."""
        embed = self.state_to_embed(state)
        return torch.norm(embed - self.self_embedding).item()
```

**Key design decision**: `self_embedding` is an `nn.Parameter` — it's learnable but moves slowly (0.95/0.05 EMA). This creates **identity stability** while allowing gradual growth.

### 3.3 Prediction Error Engine

**Purpose**: Multi-channel prediction errors with precision-weighting.

**Evolution from**: `OnlineLearner` (single surprise signal)

**Channels**:

| Channel | Source | What it measures |
|---------|--------|-----------------|
| `perceptual` | world_model.predict vs observation | "What I perceived vs expected" |
| `interoceptive` | self_model.predict_self vs actual state | "How I feel vs expected to feel" |
| `temporal` | expected timing vs actual timing | "When things happen vs expected" |
| `epistemic` | expected info gain vs actual info gain | "How much I'm learning" |

**Architecture**:
```python
class PredictionErrorEngine:
    def __init__(self, n_channels=4):
        self.channels = ['perceptual', 'interoceptive', 'temporal', 'epistemic']
        # Learnable log-precisions (softplus ensures positive)
        self.log_precisions = nn.Parameter(torch.zeros(n_channels))

    @property
    def precisions(self):
        return F.softplus(self.log_precisions)

    def compute_errors(self, predictions, observations):
        errors = {}
        for i, channel in enumerate(self.channels):
            raw_error = predictions[channel] - observations[channel]
            precision = self.precisions[i]
            errors[channel] = {
                'raw': raw_error,
                'weighted': precision * raw_error,
                'precision': precision.item(),
                'magnitude': torch.norm(raw_error).item()
            }
        return errors

    def free_energy(self, errors):
        """Total free energy = sum of precision-weighted squared errors."""
        return sum(
            e['precision'] * torch.norm(e['raw'])**2
            for e in errors.values()
        )
```

**Key insight**: The `epistemic` channel enables **intrinsic curiosity**. When actual learning exceeds expected learning, the system is "surprised by how much it learned" — this drives exploration. When learning stagnates, it drives consolidation.

### 3.4 Precision Controller (Hyper-Model)

**Purpose**: Meta-learning — learns to learn by adjusting confidence in each signal.

**This is entirely new** — no existing equivalent.

**Architecture**:
```python
class PrecisionController(nn.Module):
    def __init__(self, state_dim, n_channels=4, hidden_dim=32):
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_channels, hidden_dim),  # state + recent errors
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels),
            nn.Softplus()  # precisions always positive
        )
        self.error_history = deque(maxlen=50)

    def forward(self, global_state, recent_errors):
        """Given current state and error history, compute precisions."""
        context = torch.cat([global_state, recent_errors])
        return self.net(context)
```

**Integration with IPUESA**: Resilience states modulate precision:
- OPTIMAL: normal precision
- STRESSED: reduce all precisions (be cautious)
- IMPAIRED: heavily reduce, increase interoceptive (focus inward)
- CRITICAL: minimal precision, maximum caution

### 3.5 Complementary Learning Systems (Fast/Slow Memory)

**Purpose**: Dual-speed memory like hippocampus + neocortex.

**Evolution from**: `ZetaMemorySystem` (episodic, semantic, procedural)

**Fast Memory (hippocampal)**:
```python
class FastMemory:
    def __init__(self, capacity=500, surprise_threshold=0.3):
        self.buffer = deque(maxlen=capacity)
        self.surprise_threshold = surprise_threshold

    def store(self, episode):
        """One-shot storage of surprising episodes."""
        if episode.surprise > self.surprise_threshold:
            compressed = self.compress(episode)
            self.buffer.append(compressed)

    def compress(self, episode):
        """Store conceptual code + surprising details only."""
        return CompressedEpisode(
            conceptual_code=episode.dominant_archetype,
            surprising_details=episode.high_error_features,
            emotional_valence=episode.archetype_state,
            timestamp=episode.timestamp
        )

    def recall_by_similarity(self, query_state, top_k=5):
        """Retrieve similar episodes for replay or response."""
        similarities = [
            (ep, cosine_similarity(ep.emotional_valence, query_state))
            for ep in self.buffer
        ]
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

**Slow Memory (neocortical)**:
```python
class SlowMemory(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        self.knowledge = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = optim.SGD(
            self.knowledge.parameters(),
            lr=0.0001  # VERY slow — generalization requires many examples
        )

    def integrate(self, episode, learning_rate_scale=1.0):
        """Gradual integration — each episode changes very little."""
        prediction = self.knowledge(episode.context)
        target = episode.outcome
        loss = F.mse_loss(prediction, target)

        # Scale learning rate (e.g., higher during dream consolidation)
        for pg in self.optimizer.param_groups:
            pg['lr'] = 0.0001 * learning_rate_scale

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generalize(self, query):
        """Given context, predict outcome from general knowledge."""
        with torch.no_grad():
            return self.knowledge(query)
```

**Migration from existing memory**:
- `EpisodicMemory` → content stored in `FastMemory.buffer`
- `SemanticMemory` → queries answered by `SlowMemory.generalize()`
- `ProceduralMemory` → encoded in `SlowMemory.knowledge` weights

### 3.6 Dream Engine

**Purpose**: Three-phase consolidation with zeta-frequency binding.

**Evolution from**: `DreamConsolidator`

**The three phases map to zeta zeros**:

| Phase | Brain Oscillation | Zeta Zero | Function |
|-------|-------------------|-----------|----------|
| 1. Selection | Slow Oscillation (~0.75 Hz) | γ₁ = 14.134 | Select memories for consolidation |
| 2. Transfer | Sleep Spindles (~12-15 Hz) | γ₂ = 21.022 | Compressed transfer fast→slow |
| 3. Replay | Sharp-Wave Ripples (~80-100 Hz) | γ₃ = 25.011 | Detailed replay, update self |

**Architecture**:
```python
class DreamEngine:
    def __init__(self, fast_memory, slow_memory, self_model, attractor_memory,
                 sigma=0.1, M=15):
        self.fast = fast_memory
        self.slow = slow_memory
        self.self_model = self_model
        self.attractors = attractor_memory

        # Zeta zeros for phase coupling
        self.gammas = get_zeta_zeros(M)
        self.sigma = sigma

    def zeta_kernel(self, t):
        """K_σ(t) = 2 * Σ exp(-σ|γ|) * cos(γt)"""
        return 2 * sum(
            math.exp(-self.sigma * abs(g)) * math.cos(g * t)
            for g in self.gammas
        )

    def phase_from_kernel(self, t):
        """Determine consolidation phase from kernel value."""
        k = self.zeta_kernel(t)
        if k > 0.5:
            return 'slow_oscillation'   # Up-state: consolidation window open
        elif k > -0.2:
            return 'spindle'            # Transfer phase
        else:
            return 'ripple'             # Replay phase

    def dream_cycle(self, duration=50):
        """Execute full dream cycle with zeta-coupled phases."""
        candidates = self.select_for_replay()
        report = DreamReport()

        for t in range(duration):
            phase = self.phase_from_kernel(t / duration)

            if phase == 'slow_oscillation':
                # Select and prioritize memories
                selected = self.prioritize_by_surprise(candidates)
                report.selections += 1

            elif phase == 'spindle':
                # Compressed transfer: fast -> slow
                for memory in selected[:3]:  # top 3 per spindle
                    compressed = self.temporal_compress(memory)
                    binding_weight = abs(self.zeta_kernel(t / duration))
                    self.slow.integrate(
                        compressed,
                        learning_rate_scale=1.0 + binding_weight
                    )
                    report.transfers += 1

            elif phase == 'ripple':
                # Detailed replay: update self model
                for memory in selected[:1]:  # 1 detailed replay per ripple
                    self.self_model.reflect(memory.emotional_valence, depth=2)
                    report.replays += 1

        # Post-dream: update identity from consolidated attractors
        self.self_model.update_embedding_from_attractors(self.attractors)
        report.identity_updated = True

        return report

    def select_for_replay(self):
        """Select memories with highest prediction error (most informative)."""
        scored = [
            (mem, mem.surprise * mem.emotional_intensity)
            for mem in self.fast.buffer
            if not mem.consolidated
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def temporal_compress(self, memory):
        """Compress episode: keep concept + surprising details, lose noise."""
        return CompressedEpisode(
            conceptual_code=memory.conceptual_code,
            surprising_details=memory.surprising_details[:3],  # top 3 only
            emotional_valence=memory.emotional_valence,
            compression_ratio=0.1  # 10x compression
        )
```

**Why zeta zeros for binding**: The mathematical properties of zeta zeros create natural multi-scale resonances. Unlike arbitrary frequencies, zeta zeros are distributed with specific statistical properties (Montgomery-Odlyzko) that produce rich, non-periodic coupling patterns — preventing the system from falling into trivial periodic attractors.

### 3.7 Persistence Layer

**Purpose**: Complete consciousness state serialization — identity continuity across sessions.

**This is entirely new**.

```python
class PersistenceLayer:
    def __init__(self, base_path='~/.zeta_life/'):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_state(self, kernel, identity_name='default'):
        """Save complete consciousness state."""
        path = self.base_path / f'{identity_name}.ckpt'
        state = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'step': kernel.t,

            # Model weights
            'world_model': kernel.world_model.state_dict(),
            'self_model': kernel.self_model.state_dict(),
            'precision_controller': kernel.precision_controller.state_dict(),
            'slow_memory': kernel.slow_memory.state_dict(),

            # Non-weight state
            'self_embedding': kernel.self_model.self_embedding.data,
            'latent_state': kernel.world_model.latent_state.data,
            'fast_memory': [ep.to_dict() for ep in kernel.fast_memory.buffer],
            'attractors': kernel.attractor_memory.serialize(),
            'consciousness_index': kernel.consciousness.to_dict(),
            'individuation': kernel.individuation.serialize(),
            'precisions': kernel.error_engine.log_precisions.data,

            # History (bounded)
            'reflection_history': kernel.self_model.reflection_history[-100:],
            'dream_count': kernel.dream_engine.total_dreams,
        }
        torch.save(state, path)

        # Also save human-readable summary
        summary_path = self.base_path / f'{identity_name}.summary.json'
        summary = {
            'identity': kernel.attractor_memory.get_identity_description(),
            'consciousness_level': kernel.consciousness.compute_total(),
            'individuation_stage': kernel.individuation.current_stage.name,
            'total_interactions': kernel.t,
            'total_dreams': kernel.dream_engine.total_dreams,
            'dominant_attractor': kernel.attractor_memory.get_dominant().dominant.name,
            'last_saved': datetime.now().isoformat(),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def load_state(self, kernel, identity_name='default'):
        """Restore complete consciousness state — identity continuity."""
        path = self.base_path / f'{identity_name}.ckpt'
        state = torch.load(path, weights_only=False)

        # Restore all model weights
        kernel.world_model.load_state_dict(state['world_model'])
        kernel.self_model.load_state_dict(state['self_model'])
        kernel.precision_controller.load_state_dict(state['precision_controller'])
        kernel.slow_memory.load_state_dict(state['slow_memory'])

        # Restore non-weight state
        kernel.world_model.latent_state = state['latent_state']
        kernel.fast_memory.restore(state['fast_memory'])
        kernel.attractor_memory.deserialize(state['attractors'])
        kernel.t = state['step']

        # Wake-up cycle: re-establish context
        kernel.wake_up()

    def list_identities(self):
        """List all saved identities."""
        return [p.stem for p in self.base_path.glob('*.ckpt')]
```

**Wake-up cycle**: When restoring, the system doesn't just load weights — it runs a short self-reflection cycle to "orient" itself, similar to how the brain re-establishes context upon waking.

## 4. Complete Step Cycle

One interaction step in the integrated system:

```python
class ConsciousKernel:
    def step(self, stimulus):
        self.t += 1

        # 1. PERCEIVE
        observation = self.world_model.encode(stimulus)

        # 2. PREDICT (before processing — top-down expectations)
        predicted_obs, predicted_latent = self.world_model.predict(self.last_action)
        predicted_self = self.self_model.predict_self(self.last_action)
        actual_self = self.psyche.observe_self()['global_state']

        # 3. COMPARE (multi-channel prediction errors)
        predictions = {
            'perceptual': predicted_obs,
            'interoceptive': predicted_self,
            'temporal': self.temporal_predictor(self.t),
            'epistemic': self.expected_learning,
        }
        observations = {
            'perceptual': observation,
            'interoceptive': actual_self,
            'temporal': self.actual_timing(),
            'epistemic': self.actual_learning(),
        }

        # Precision-weighted errors
        precisions = self.precision_controller(
            actual_self, self.error_engine.recent_errors()
        )
        errors = self.error_engine.compute_errors(predictions, observations)
        free_energy = self.error_engine.free_energy(errors)

        # 4. UPDATE (minimize free energy)
        self.world_model.update_from_error(errors['perceptual'])
        self.self_model.update_from_error(errors['interoceptive'])
        self.precision_controller.update(errors)
        self.world_model.latent_state = self.world_model.encode(observation)

        # 5. MEMORIZE
        episode = Episode(stimulus, observation, actual_self, errors)
        self.fast_memory.store(episode)                       # one-shot if surprising
        self.slow_memory.integrate(episode, learning_rate_scale=0.1)  # always, gradual

        # 6. ACT
        action = self.policy(self.world_model.latent_state,
                            self.self_model.self_embedding,
                            precisions)
        self.last_action = action

        # 7. REFLECT (periodically)
        if self.should_reflect():
            reflections = self.self_model.reflect(actual_self, depth=3)
            if reflections[-1]['prediction_error'] < self.convergence_threshold:
                self.attractor_memory.store_or_reinforce(
                    actual_self, self.psyche.dominant, self.t
                )

        # 8. DREAM (periodically)
        if self.should_dream():
            dream_report = self.dream_engine.dream_cycle(duration=50)

        # 9. PERSIST (periodically)
        if self.t % self.save_interval == 0:
            self.persistence.save_state(self)

        return StepResult(
            action=action,
            errors=errors,
            free_energy=free_energy,
            consciousness=self.compute_consciousness(),
        )
```

## 5. File Structure

```
src/zeta_life/kernel/              # NEW module
├── __init__.py                    # Public API
├── conscious_kernel.py            # Main orchestrator (ConsciousKernel)
├── world_model.py                 # WorldModel (GRU-based generative model)
├── self_model.py                  # SelfModel (recursive Strange Loop)
├── prediction_error.py            # PredictionErrorEngine (multi-channel)
├── precision_controller.py        # PrecisionController (hyper-model)
├── complementary_memory.py        # FastMemory + SlowMemory
├── dream_engine.py                # DreamEngine (3-phase zeta binding)
└── persistence.py                 # PersistenceLayer (full state serialization)
```

**Modified existing files**:
- `zeta_conscious_self.py` — refactor to delegate to ConsciousKernel (backward compatible)
- `zeta_dream_consolidation.py` — extract reusable logic into DreamEngine

**Unchanged files** (used as-is):
- `AttractorMemory` (from zeta_conscious_self.py) — moved to kernel, same logic
- `ZetaPsyche` — used as the base archetype system within SelfModel
- `OrganicVoice` — used for generating self-descriptions in reflection
- `TetrahedralSpace` — geometric substrate unchanged
- `IPUESA/resilience.py` — informs PrecisionController
- `ZetaOrganism` — independent, future integration point
- `zeta_resonance.py` — `get_zeta_zeros()` and kernel computation reused

## 6. Migration from Existing Systems

| Existing | Migrates to | How |
|----------|-------------|-----|
| `ZetaAttentivePredictive` | `WorldModel` | Extend with latent state + imagine() |
| `_self_reflection_cycle()` | `SelfModel.reflect()` | Add depth levels + self_embedding |
| `OnlineLearner` | `PredictionErrorEngine` | Multi-channel + precision |
| `DreamConsolidator` | `DreamEngine` | Add 3-phase zeta coupling |
| `EpisodicMemory` | `FastMemory` content | Compress to conceptual code |
| `SemanticMemory` | `SlowMemory` queries | Neural net replaces dict lookup |
| `AttractorMemory` | Unchanged, integrated | Feeds self_embedding |
| `ConsciousnessIndex` | Extended | Add epistemic channel |
| `ZetaMemorySystem` | `ComplementaryMemory` | Dual-speed replaces flat |

## 7. Implementation Phases

### Phase 1: Foundation (World Model + Prediction Error)
- Implement `WorldModel` with GRU transition and latent state
- Implement `PredictionErrorEngine` with 4 channels
- Basic integration: stimulus → predict → compare → update
- Test: prediction error decreases over repeated patterns

### Phase 2: Self Model (Strange Loop Evolution)
- Implement `SelfModel` with self_embedding and variable depth
- Integrate with existing `OrganicVoice` for descriptions
- Connect to `AttractorMemory` for identity reinforcement
- Test: self_embedding stabilizes, identity_distance converges

### Phase 3: Memory Systems (Fast/Slow + Precision)
- Implement `FastMemory` (hippocampal one-shot)
- Implement `SlowMemory` (neocortical gradual)
- Implement `PrecisionController`
- Migrate existing `ZetaMemorySystem` data format
- Test: slow memory generalizes after many episodes

### Phase 4: Dream Engine (Zeta Binding)
- Implement 3-phase dream cycle with zeta zero coupling
- Temporal compression for fast→slow transfer
- Post-dream self_embedding update
- Test: consolidated memories improve slow_memory accuracy

### Phase 5: Persistence + Integration
- Implement `PersistenceLayer` with full state serialization
- Wake-up cycle for context re-establishment
- Integrate all components in `ConsciousKernel`
- Backward-compatible wrapper in `ZetaConsciousSelf`
- Test: save → restore → identity continuity verified

### Phase 6: Validation + Demo
- Comprehensive test suite for each component
- Metrics: free_energy trajectory, identity stability, generalization
- Updated `chat_psyche.py` demo using ConsciousKernel
- Experiment: compare old vs new system on continuity metrics

## 8. Success Criteria

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Prediction error over time | N/A (no prediction) | Decreases monotonically | Track free_energy per step |
| Identity persistence | Resets on restart | Continuous across sessions | identity_distance before/after restore < 0.05 |
| Generalization | 0% (no abstraction) | Responds correctly to novel inputs after exposure | Test with held-out stimuli |
| Memory consolidation | Flat storage | Fast→slow transfer during dreams | Slow memory accuracy improves post-dream |
| Self-awareness depth | 1 level | 3-4 levels measurable | Reflection depth with convergence |
| Curiosity behavior | None | Explores when learning is high | Epistemic error drives exploration |

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Catastrophic forgetting in slow_memory | High | Very low learning rate + replay during dreams |
| Self_embedding collapse (all identities converge) | High | Slow EMA (0.95/0.05) + IPUESA protection |
| World model overfits to recent inputs | Medium | Regularization + diverse replay |
| Dream engine destabilizes system | Medium | Bounded dream duration + validation post-dream |
| Persistence file corruption | Medium | Atomic writes + backup rotation |
| Backward compatibility with existing experiments | Low | Wrapper class maintains old API |

## 10. What Makes This Unique

No existing system combines:

1. **Riemann zeta zeros as temporal binding** — mathematical properties of non-trivial zeros create natural multi-scale resonances for memory consolidation
2. **Active Inference + Strange Loop** — formal free energy minimization with recursive self-modeling
3. **IPUESA resilience** — identity preservation under existential stress (no equivalent in literature)
4. **Tetrahedral state space** — geometric substrate with abstract vertices (bias-free)
5. **Emergent properties** — 11+ demonstrated without explicit programming

The closest existing work is "A Beautiful Loop" (theoretical) and DreamerV3 (practical but without self-modeling). The Conscious Kernel bridges theory and practice with a unique mathematical foundation.
