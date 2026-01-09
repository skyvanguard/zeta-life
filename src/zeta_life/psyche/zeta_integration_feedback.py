# ZetaIntegrationFeedback: Sistema de Retroalimentacion Metrica-Psique
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
from .zeta_psyche import ZetaPsyche, Archetype
from .zeta_individuation import IndividuationStage, IntegrationMetrics

@dataclass
class FeedbackEffect:
    archetype: Archetype
    influence_modifier: float
    repulsion_reduction: float
    cooperation_bonus: float

class IntegrationFeedback:
    ARCHETYPE_PARAMS = {
        Archetype.PERSONA: {"base_influence": 0.8, "integration_boost": 0.3, "repulsion_factor": 0.2},
        Archetype.SOMBRA: {"base_influence": 0.6, "integration_boost": 0.5, "repulsion_factor": 0.5},
        Archetype.ANIMA: {"base_influence": 0.5, "integration_boost": 0.4, "repulsion_factor": 0.3},
        Archetype.ANIMUS: {"base_influence": 0.5, "integration_boost": 0.4, "repulsion_factor": 0.3},
    }
    STAGE_COOPERATION = {
        IndividuationStage.INCONSCIENTE: 0.0, IndividuationStage.CRISIS_PERSONA: 0.1,
        IndividuationStage.ENCUENTRO_SOMBRA: 0.25, IndividuationStage.INTEGRACION_SOMBRA: 0.4,
        IndividuationStage.ENCUENTRO_ANIMA: 0.55, IndividuationStage.INTEGRACION_ANIMA: 0.7,
        IndividuationStage.EMERGENCIA_SELF: 0.85, IndividuationStage.SELF_REALIZADO: 0.95,
    }
    def __init__(self, smoothing_factor=0.1):
        self.smoothing_factor = smoothing_factor
        self._previous_effects = {}
    def compute_archetype_influence(self, archetype, metrics, stage):
        params = self.ARCHETYPE_PARAMS[archetype]
        cooperation = self.STAGE_COOPERATION[stage]
        metric_map = {Archetype.PERSONA: metrics.persona_flexibility, Archetype.SOMBRA: metrics.shadow_acceptance,
                      Archetype.ANIMA: metrics.anima_connection, Archetype.ANIMUS: metrics.animus_balance}
        metric_value = metric_map.get(archetype, 0.5)
        influence_modifier = params["base_influence"] + params["integration_boost"] * metric_value * (0.5 + 0.5 * cooperation)
        repulsion_reduction = metric_value * (0.5 + 0.5 * cooperation)
        cooperation_bonus = cooperation * metrics.self_coherence * 0.3
        return FeedbackEffect(archetype, influence_modifier, repulsion_reduction, cooperation_bonus)
    def apply_feedback(self, psyche, metrics, stage):
        effects = {}
        for archetype in Archetype:
            effect = self.compute_archetype_influence(archetype, metrics, stage)
            if archetype in self._previous_effects:
                prev = self._previous_effects[archetype]
                effect = FeedbackEffect(archetype,
                    self._smooth(prev.influence_modifier, effect.influence_modifier),
                    self._smooth(prev.repulsion_reduction, effect.repulsion_reduction),
                    self._smooth(prev.cooperation_bonus, effect.cooperation_bonus))
            effects[archetype] = effect
            self._previous_effects[archetype] = effect
        self._apply_to_psyche_cells(psyche, effects, stage)
        self._modify_psyche_parameters(psyche, metrics, stage)
        return effects
    def _smooth(self, old_value, new_value):
        return old_value + self.smoothing_factor * (new_value - old_value)
    def _apply_to_psyche_cells(self, psyche, effects, stage):
        cooperation = self.STAGE_COOPERATION[stage]
        for cell in psyche.cells:
            dominant_idx = torch.argmax(cell.position.detach()).item()
            effect = effects[Archetype(dominant_idx)]
            cell.energy = max(0.1, min(1.0, cell.energy + (effect.influence_modifier - 0.5) * 0.1))
            if cooperation > 0.3:
                center = torch.tensor([0.25, 0.25, 0.25, 0.25])
                with torch.no_grad():
                    new_pos = cell.position + cooperation * effect.cooperation_bonus * (center - cell.position)
                    new_pos = torch.clamp(new_pos, min=0.01)
                    cell.position.copy_(new_pos / new_pos.sum())
    def _modify_psyche_parameters(self, psyche, metrics, stage):
        cooperation = self.STAGE_COOPERATION[stage]
        if hasattr(psyche, "global_state") and psyche.global_state is not None:
            with torch.no_grad():
                noise_reduction = 1.0 - metrics.self_coherence * 0.1 * cooperation
                mean_val = psyche.global_state.mean()
                psyche.global_state = mean_val + noise_reduction * (psyche.global_state - mean_val)
    def get_archetype_dynamics(self, metrics, stage):
        cooperation = self.STAGE_COOPERATION[stage]
        return {"cooperation_level": cooperation, "overall_integration": metrics.self_coherence * cooperation}

class IntegrationWorkFeedback:
    WORK_EFFECTS = {
        "shadow_dialogue": {"target": Archetype.SOMBRA, "energy": 0.15, "shift": 0.05, "metric": "shadow_acceptance"},
        "persona_examination": {"target": Archetype.PERSONA, "energy": 0.1, "shift": 0.03, "metric": "persona_flexibility"},
        "anima_encounter": {"target": Archetype.ANIMA, "energy": 0.12, "shift": 0.04, "metric": "anima_connection"},
        "animus_balance": {"target": Archetype.ANIMUS, "energy": 0.12, "shift": 0.04, "metric": "animus_balance"},
        "mandala_meditation": {"target": None, "energy": 0.08, "shift": 0.02, "metric": "self_coherence"},
        "dream_analysis": {"target": None, "energy": 0.1, "shift": 0.03, "metric": "self_coherence"},
    }
    def __init__(self, intensity=1.0):
        self.intensity = intensity
    def apply_work_effect(self, work_type, psyche, metrics, stage):
        if work_type not in self.WORK_EFFECTS:
            return f"Unknown: {work_type}", 0.0
        effect = self.WORK_EFFECTS[work_type]
        cooperation = IntegrationFeedback.STAGE_COOPERATION.get(stage, 0.5)
        effectiveness = 0.5 + 0.5 * cooperation
        target = effect["target"]
        energy_boost = effect["energy"] * self.intensity * effectiveness
        for cell in psyche.cells:
            if target is None:
                cell.energy = min(1.0, cell.energy + energy_boost * 0.5)
            elif Archetype(torch.argmax(cell.position).item()) == target:
                cell.energy = min(1.0, cell.energy + energy_boost)
        center = torch.tensor([0.25, 0.25, 0.25, 0.25])
        shift = effect["shift"] * self.intensity * effectiveness
        for cell in psyche.cells:
            with torch.no_grad():
                new_pos = cell.position + shift * (center - cell.position)
                cell.position.copy_(torch.clamp(new_pos, min=0.01) / torch.clamp(new_pos, min=0.01).sum())
        current = getattr(metrics, effect["metric"])
        new_value = min(1.0, current + 0.05 * self.intensity * effectiveness)
        setattr(metrics, effect["metric"], new_value)
        return f"{work_type} applied with effectiveness {effectiveness:.2f}", effectiveness

def create_feedback_system(smoothing_factor=0.1, work_intensity=1.0):
    return IntegrationFeedback(smoothing_factor), IntegrationWorkFeedback(work_intensity)
