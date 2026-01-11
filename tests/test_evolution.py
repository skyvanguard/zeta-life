"""
Tests for the Evolution module.

Tests config_space, fitness_evaluator, and ipuesa_evolvable.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zeta_life.evolution.config_space import (
    EvolvableConfig, PARAM_RANGES, get_baseline_config, get_config_as_flat_dict
)
from zeta_life.evolution.fitness_evaluator import (
    validate_config, evaluate_self_evidence, calculate_fitness, FitnessResult
)
from zeta_life.evolution.ipuesa_evolvable import (
    EvolvableAgent, EvolvableMicroModule, ClusterState,
    gradual_damage, gradual_recovery, create_agents, create_clusters
)


class TestConfigSpace:
    """Tests for config_space.py"""

    def test_param_ranges_complete(self):
        """All 30 parameters should be defined."""
        assert len(PARAM_RANGES) == 30

    def test_evolvable_config_defaults(self):
        """Default config should have all parameters."""
        config = EvolvableConfig()
        assert config.damage_multiplier == 3.9
        assert config.base_degrad_rate == 0.18
        assert config.module_cap == 6

    def test_evolvable_config_from_dict(self):
        """Config should be creatable from dict."""
        d = {'damage_multiplier': 4.0, 'base_degrad_rate': 0.2}
        config = EvolvableConfig.from_dict(d)
        assert config.damage_multiplier == 4.0
        assert config.base_degrad_rate == 0.2
        # Others should have defaults
        assert config.module_cap == 6

    def test_evolvable_config_to_dict(self):
        """Config should be convertible to dict."""
        config = EvolvableConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert 'damage_multiplier' in d
        assert d['damage_multiplier'] == 3.9

    def test_config_validation_valid(self):
        """Valid config should pass validation."""
        config = EvolvableConfig()
        valid, msg = config.validate()
        assert valid is True
        assert msg == "OK"

    def test_config_validation_out_of_range(self):
        """Out-of-range config should fail validation."""
        config = EvolvableConfig(damage_multiplier=10.0)  # Max is 5.0
        valid, msg = config.validate()
        assert valid is False
        assert "damage_multiplier" in msg

    def test_config_clamp_to_ranges(self):
        """Clamp should bring values into valid ranges."""
        config = EvolvableConfig(damage_multiplier=10.0, base_degrad_rate=0.01)
        clamped = config.clamp_to_ranges()
        assert clamped.damage_multiplier == 5.0  # Max
        assert clamped.base_degrad_rate == 0.05  # Min

    def test_get_module_effects(self):
        """Should return dict of module effects."""
        config = EvolvableConfig()
        effects = config.get_module_effects()
        assert isinstance(effects, dict)
        assert len(effects) == 8
        assert 'pattern_detector' in effects
        assert effects['pattern_detector'] == 0.20

    def test_get_baseline_config(self):
        """Baseline should be default config."""
        baseline = get_baseline_config()
        default = EvolvableConfig()
        assert baseline.damage_multiplier == default.damage_multiplier

    def test_get_config_as_flat_dict(self):
        """Should return only evolvable params as float dict."""
        config = EvolvableConfig()
        flat = get_config_as_flat_dict(config)
        assert len(flat) == 30
        assert all(isinstance(v, float) for v in flat.values())


class TestFitnessEvaluator:
    """Tests for fitness_evaluator.py"""

    def test_validate_config_valid(self):
        """Valid config dict should pass."""
        config = get_config_as_flat_dict(EvolvableConfig())
        valid, msg = validate_config(config)
        assert valid is True

    def test_validate_config_missing_key(self):
        """Missing key should fail."""
        config = {'damage_multiplier': 3.9}  # Missing other keys
        valid, msg = validate_config(config)
        assert valid is False
        assert "Missing" in msg

    def test_validate_config_out_of_range(self):
        """Out of range value should fail."""
        config = get_config_as_flat_dict(EvolvableConfig())
        config['damage_multiplier'] = 10.0  # Max is 5.0
        valid, msg = validate_config(config)
        assert valid is False
        assert "outside range" in msg

    def test_evaluate_self_evidence_all_pass(self):
        """Good metrics should pass all criteria."""
        metrics = {
            'holographic_survival': 0.5,  # In [0.30, 0.70]
            'module_spreading_rate': 0.3,  # > 0.15
            'temporal_anticipation_effectiveness': 0.25,  # > 0.15
            'embedding_integrity': 0.8,  # > 0.30
            'emergent_differentiation': 0.2,  # > 0.10
            'baseline_survival': 0.1,  # HS > baseline
            'no_embedding_survival': 0.3,  # For gradient
            'degradation_variance': 0.05,  # > 0.02
        }
        criteria = evaluate_self_evidence(metrics)
        assert all(criteria.values())
        assert sum(criteria.values()) == 8

    def test_evaluate_self_evidence_partial(self):
        """Bad metrics should fail some criteria."""
        metrics = {
            'holographic_survival': 0.1,  # NOT in [0.30, 0.70]
            'module_spreading_rate': 0.05,  # NOT > 0.15
            'temporal_anticipation_effectiveness': 0.25,
            'embedding_integrity': 0.8,
            'emergent_differentiation': 0.2,
            'baseline_survival': 0.1,
            'no_embedding_survival': 0.05,
            'degradation_variance': 0.05,
        }
        criteria = evaluate_self_evidence(metrics)
        assert criteria['HS_in_range'] is False
        assert criteria['MSR_pass'] is False
        assert sum(criteria.values()) < 8

    def test_calculate_fitness(self):
        """Fitness should be between 0 and 1."""
        criteria = {
            'HS_in_range': True,
            'MSR_pass': True,
            'TAE_pass': True,
            'EI_pass': True,
            'ED_pass': True,
            'diff_pass': True,
            'gradient_pass': True,
            'smooth_transition': True,
        }
        metrics = {
            'holographic_survival': 0.5,
            'module_spreading_rate': 0.5,
            'temporal_anticipation_effectiveness': 0.3,
            'emergent_differentiation': 0.3,
        }
        fitness = calculate_fitness(criteria, metrics)
        assert 0 <= fitness <= 1
        # All criteria pass + good continuous = high fitness
        assert fitness > 0.8


class TestIPUESAEvolvable:
    """Tests for ipuesa_evolvable.py"""

    def test_evolvable_agent_creation(self):
        """Agent should be creatable with defaults."""
        agent = EvolvableAgent(agent_id=0, cluster_id=0)
        assert agent.IC_t == 1.0
        assert agent.is_alive()
        assert len(agent.modules) == 0

    def test_agent_is_alive(self):
        """is_alive should reflect IC_t threshold."""
        agent = EvolvableAgent(agent_id=0, cluster_id=0)
        assert agent.is_alive()
        agent.IC_t = 0.01
        assert not agent.is_alive()

    def test_agent_embedding_integrity(self):
        """Embedding integrity should be calculated."""
        agent = EvolvableAgent(
            agent_id=0, cluster_id=0,
            cluster_embedding=np.ones(8)
        )
        ei = agent.get_embedding_integrity()
        assert 0 <= ei <= 1

    def test_micro_module_apply(self):
        """Module should apply effect based on config."""
        config = EvolvableConfig()
        module = EvolvableMicroModule(
            module_type='pattern_detector',
            strength=0.5
        )
        effect = module.apply(config)
        expected = 0.5 * config.effect_pattern_detector
        assert effect == pytest.approx(expected)
        assert module.activation_count == 1

    def test_gradual_damage(self):
        """Damage should reduce IC_t."""
        agent = EvolvableAgent(agent_id=0, cluster_id=0)
        config = EvolvableConfig()
        initial_ic = agent.IC_t

        damage = gradual_damage(agent, 0.1, config)

        assert agent.IC_t < initial_ic
        assert damage > 0

    def test_gradual_recovery(self):
        """Recovery should increase IC_t."""
        agent = EvolvableAgent(agent_id=0, cluster_id=0, IC_t=0.5)
        cluster = ClusterState(cluster_id=0, cohesion=0.5)
        config = EvolvableConfig()
        initial_ic = agent.IC_t

        recovery, success = gradual_recovery(agent, cluster, config)

        assert agent.IC_t >= initial_ic
        assert recovery >= 0

    def test_create_agents(self):
        """Should create correct number of agents."""
        config = EvolvableConfig()
        agents = create_agents(24, 4, config)
        assert len(agents) == 24
        # Should be distributed across clusters
        cluster_counts = {}
        for a in agents:
            cluster_counts[a.cluster_id] = cluster_counts.get(a.cluster_id, 0) + 1
        assert len(cluster_counts) == 4

    def test_create_clusters(self):
        """Should create correct number of clusters."""
        clusters = create_clusters(4)
        assert len(clusters) == 4
        assert clusters[0].cluster_id == 0
        assert clusters[3].cluster_id == 3


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_run_ipuesa_with_config(self):
        """Full simulation should run and return metrics."""
        from zeta_life.evolution.ipuesa_evolvable import run_ipuesa_with_config

        config = get_config_as_flat_dict(EvolvableConfig())
        results = run_ipuesa_with_config(
            config,
            n_agents=12,
            n_clusters=3,
            n_steps=30,
            n_runs=2
        )

        assert isinstance(results, dict)
        assert 'holographic_survival' in results
        assert 'module_spreading_rate' in results
        assert 'baseline_survival' in results
        assert 0 <= results['holographic_survival'] <= 1

    def test_evaluate_config_full(self):
        """Full evaluation should return FitnessResult."""
        from zeta_life.evolution.fitness_evaluator import evaluate_config

        config = get_config_as_flat_dict(EvolvableConfig())
        result = evaluate_config(
            config,
            n_runs=2,
            n_steps=30,
            n_agents=12,
            n_clusters=3
        )

        assert isinstance(result, FitnessResult)
        assert result.valid_config is True
        assert 0 <= result.fitness_score <= 1
        assert 0 <= result.criteria_passed <= 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
