"""Tests for PersistenceLayer -- identity persistence for the Conscious Kernel.

Covers:
- save_state creates .ckpt file
- save_state creates .summary.json file with step and timestamp
- load_state restores world_model weights correctly
- load_state restores self_embedding correctly
- load_state restores step counter
- list_identities returns saved names
- identity_exists works
"""

import json

import torch
import pytest

from zeta_life.kernel.persistence import PersistenceLayer
from zeta_life.kernel.world_model import WorldModel
from zeta_life.kernel.self_model import SelfModel
from zeta_life.kernel.prediction_error import PredictionErrorEngine
from zeta_life.kernel.precision_controller import PrecisionController
from zeta_life.kernel.complementary_memory import FastMemory, SlowMemory


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_components():
    """Create a fresh set of kernel components for testing."""
    return {
        'world_model': WorldModel(obs_dim=4, latent_dim=32, action_dim=4),
        'self_model': SelfModel(state_dim=4, embed_dim=16),
        'error_engine': PredictionErrorEngine(n_channels=4),
        'precision_controller': PrecisionController(state_dim=4),
        'fast_memory': FastMemory(capacity=50),
        'slow_memory': SlowMemory(context_dim=4, outcome_dim=4),
        'step': 0,
    }


# ---------------------------------------------------------------------------
# save_state creates .ckpt file
# ---------------------------------------------------------------------------

class TestSaveStateCreatesCheckpoint:
    """save_state should create a .ckpt file on disk."""

    def test_ckpt_file_created(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        path = pl.save_state(components, identity_name='test_agent')
        assert path.exists()
        assert path.suffix == '.ckpt'

    def test_ckpt_file_name_matches_identity(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        path = pl.save_state(components, identity_name='alpha')
        assert path.stem == 'alpha'

    def test_default_identity_name(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        path = pl.save_state(components)
        assert path.stem == 'default'


# ---------------------------------------------------------------------------
# save_state creates .summary.json
# ---------------------------------------------------------------------------

class TestSaveStateCreatesSummary:
    """save_state should create a .summary.json alongside the .ckpt."""

    def test_summary_file_created(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        pl.save_state(components, identity_name='test_agent')
        summary_path = tmp_path / 'test_agent.summary.json'
        assert summary_path.exists()

    def test_summary_contains_step(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        components['step'] = 42
        pl.save_state(components, identity_name='test_agent')
        summary_path = tmp_path / 'test_agent.summary.json'
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['step'] == 42

    def test_summary_contains_timestamp(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        pl.save_state(components, identity_name='test_agent')
        summary_path = tmp_path / 'test_agent.summary.json'
        with open(summary_path) as f:
            summary = json.load(f)
        assert 'timestamp' in summary
        # Timestamp should be an ISO 8601 string
        assert isinstance(summary['timestamp'], str)
        assert len(summary['timestamp']) > 10


# ---------------------------------------------------------------------------
# load_state restores world_model weights
# ---------------------------------------------------------------------------

class TestLoadStateRestoresWorldModel:
    """load_state should restore world_model weights so predictions match."""

    def test_world_model_weights_restored(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        # Create and train original components
        original = _make_components()
        wm = original['world_model']
        obs = torch.tensor([1.0, 0.0, -1.0, 0.5])
        action = torch.tensor([0.0, 1.0, 0.0, 0.0])

        # Train the world model a bit to change its weights from init
        wm.latent_state = wm.encode(obs).detach()
        for _ in range(20):
            pred_obs, _ = wm.predict(action)
            error = obs - pred_obs
            wm.update_from_error(error)
            wm.latent_state = wm.encode(obs).detach()

        # Save
        original['step'] = 100
        pl.save_state(original, identity_name='trained')

        # Get prediction from trained model
        wm.latent_state = wm.encode(obs).detach()
        with torch.no_grad():
            trained_pred, _ = wm.predict(action)

        # Create fresh components and load
        fresh = _make_components()
        pl.load_state(fresh, identity_name='trained')

        # Get prediction from loaded model
        fresh_wm = fresh['world_model']
        fresh_wm.latent_state = fresh_wm.encode(obs).detach()
        with torch.no_grad():
            loaded_pred, _ = fresh_wm.predict(action)

        assert torch.allclose(trained_pred, loaded_pred, atol=1e-5), (
            f"Predictions differ: trained={trained_pred}, loaded={loaded_pred}"
        )

    def test_world_model_latent_state_restored(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        original = _make_components()
        wm = original['world_model']
        # Set a non-zero latent state
        obs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        wm.latent_state = wm.encode(obs).detach()
        latent_before = wm.latent_state.clone()

        pl.save_state(original, identity_name='latent_test')

        fresh = _make_components()
        pl.load_state(fresh, identity_name='latent_test')

        assert torch.allclose(fresh['world_model'].latent_state, latent_before, atol=1e-6)


# ---------------------------------------------------------------------------
# load_state restores self_embedding
# ---------------------------------------------------------------------------

class TestLoadStateRestoresSelfEmbedding:
    """load_state should restore self_model's self_embedding after reflections."""

    def test_self_embedding_restored(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        original = _make_components()
        sm = original['self_model']

        # Reflect several times to drift the embedding from its initial value
        for i in range(10):
            state = torch.randn(4)
            sm.reflect(state, depth=3)

        embedding_before = sm.self_embedding.data.clone()

        pl.save_state(original, identity_name='reflected')

        # Load into fresh components
        fresh = _make_components()
        pl.load_state(fresh, identity_name='reflected')

        embedding_after = fresh['self_model'].self_embedding.data

        assert torch.allclose(embedding_before, embedding_after, atol=1e-6), (
            "self_embedding was not restored correctly"
        )

    def test_self_model_network_weights_restored(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        original = _make_components()
        sm = original['self_model']

        # Capture a specific weight tensor
        weight_before = sm.state_to_embed.weight.data.clone()

        pl.save_state(original, identity_name='sm_weights')

        fresh = _make_components()
        pl.load_state(fresh, identity_name='sm_weights')

        weight_after = fresh['self_model'].state_to_embed.weight.data

        assert torch.allclose(weight_before, weight_after, atol=1e-6)


# ---------------------------------------------------------------------------
# load_state restores step counter
# ---------------------------------------------------------------------------

class TestLoadStateRestoresStep:
    """load_state should return the step counter from the checkpoint."""

    def test_step_returned(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        components = _make_components()
        components['step'] = 999
        pl.save_state(components, identity_name='step_test')

        fresh = _make_components()
        step = pl.load_state(fresh, identity_name='step_test')
        assert step == 999

    def test_step_zero(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        components = _make_components()
        components['step'] = 0
        pl.save_state(components, identity_name='zero_step')

        fresh = _make_components()
        step = pl.load_state(fresh, identity_name='zero_step')
        assert step == 0

    def test_step_large_value(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        components = _make_components()
        components['step'] = 1_000_000
        pl.save_state(components, identity_name='big_step')

        fresh = _make_components()
        step = pl.load_state(fresh, identity_name='big_step')
        assert step == 1_000_000


# ---------------------------------------------------------------------------
# list_identities
# ---------------------------------------------------------------------------

class TestListIdentities:
    """list_identities should return names of saved identities."""

    def test_empty_initially(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        assert pl.list_identities() == []

    def test_returns_saved_names(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()

        pl.save_state(components, identity_name='alice')
        pl.save_state(components, identity_name='bob')

        identities = pl.list_identities()
        assert sorted(identities) == ['alice', 'bob']

    def test_no_duplicates_on_overwrite(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()

        pl.save_state(components, identity_name='agent')
        pl.save_state(components, identity_name='agent')

        identities = pl.list_identities()
        assert identities == ['agent']


# ---------------------------------------------------------------------------
# identity_exists
# ---------------------------------------------------------------------------

class TestIdentityExists:
    """identity_exists should return True only for saved identities."""

    def test_false_when_not_saved(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        assert not pl.identity_exists('nonexistent')

    def test_true_after_save(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        pl.save_state(components, identity_name='exists_test')
        assert pl.identity_exists('exists_test')

    def test_false_for_different_name(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        pl.save_state(components, identity_name='saved_one')
        assert not pl.identity_exists('saved_two')


# ---------------------------------------------------------------------------
# Integration: full save/load round-trip
# ---------------------------------------------------------------------------

class TestPersistenceIntegration:
    """End-to-end round-trip: save all components, load, verify consistency."""

    def test_full_round_trip(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))

        # Build and exercise all components
        original = _make_components()
        original['step'] = 250

        # Train world model
        wm = original['world_model']
        obs = torch.tensor([1.0, -1.0, 0.5, 0.0])
        action = torch.tensor([0.0, 0.0, 1.0, 0.0])
        wm.latent_state = wm.encode(obs).detach()
        for _ in range(10):
            pred, _ = wm.predict(action)
            wm.update_from_error(obs - pred)
            wm.latent_state = wm.encode(obs).detach()

        # Reflect with self model
        sm = original['self_model']
        for _ in range(5):
            sm.reflect(torch.randn(4), depth=3)

        # Save
        pl.save_state(original, identity_name='full_test')

        # Verify both files exist
        assert (tmp_path / 'full_test.ckpt').exists()
        assert (tmp_path / 'full_test.summary.json').exists()

        # Load into fresh components
        fresh = _make_components()
        step = pl.load_state(fresh, identity_name='full_test')

        assert step == 250
        assert torch.allclose(
            original['self_model'].self_embedding.data,
            fresh['self_model'].self_embedding.data,
            atol=1e-6,
        )
        assert torch.allclose(
            original['world_model'].latent_state,
            fresh['world_model'].latent_state,
            atol=1e-6,
        )

    def test_summary_json_has_expected_fields(self, tmp_path):
        pl = PersistenceLayer(base_path=str(tmp_path))
        components = _make_components()
        components['step'] = 77

        # Add some episodes to fast_memory to check fast_memory_size
        from zeta_life.kernel.complementary_memory import Episode
        ep = Episode(
            stimulus=torch.randn(4),
            observation=torch.randn(4),
            archetype_state=torch.randn(4),
            surprise=0.9,
            dominant='V0',
            timestamp=1,
        )
        components['fast_memory'].store(ep)

        pl.save_state(components, identity_name='fields_test')

        with open(tmp_path / 'fields_test.summary.json') as f:
            summary = json.load(f)

        assert summary['step'] == 77
        assert 'timestamp' in summary
        assert 'fast_memory_size' in summary
        assert summary['fast_memory_size'] == 1
        assert 'self_embedding_norm' in summary
        assert isinstance(summary['self_embedding_norm'], float)

    def test_base_path_created_if_not_exists(self, tmp_path):
        nested = tmp_path / 'deep' / 'nested' / 'path'
        pl = PersistenceLayer(base_path=str(nested))
        assert nested.exists()
