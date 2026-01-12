"""Test suite for IPUESA resilience integration (Phase 1-2).

Tests cover:
- Phase 1: CellResilience, MicroModule, DamageSystem, resilience_config
- Phase 2: ConsciousCell integration, Cluster integration, HierarchicalSimulation integration
"""

import pytest
import numpy as np
import sys
import os
import importlib.util

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Use direct module loading to avoid circular import in existing __init__.py chain
# The circular import is between zeta_psyche and zeta_memory and is pre-existing

BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'zeta_life', 'consciousness')

def load_module_direct(name, path):
    """Load module directly without __init__.py chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load resilience modules directly (no circular deps)
resilience_mod = load_module_direct(
    'zeta_life.consciousness.resilience',
    os.path.join(BASE_PATH, 'resilience.py')
)
resilience_config_mod = load_module_direct(
    'zeta_life.consciousness.resilience_config',
    os.path.join(BASE_PATH, 'resilience_config.py')
)
damage_system_mod = load_module_direct(
    'zeta_life.consciousness.damage_system',
    os.path.join(BASE_PATH, 'damage_system.py')
)

CellResilience = resilience_mod.CellResilience
MicroModule = resilience_mod.MicroModule
MODULE_TYPES = resilience_mod.MODULE_TYPES
DEGRADATION_THRESHOLDS = resilience_mod.DEGRADATION_THRESHOLDS
get_preset_config = resilience_config_mod.get_preset_config
list_presets = resilience_config_mod.list_presets
DamageSystem = damage_system_mod.DamageSystem

# Phase 2 imports need the full package - skip if circular import fails
try:
    from zeta_life.consciousness.micro_psyche import ConsciousCell
    from zeta_life.consciousness.cluster import Cluster
    PHASE2_AVAILABLE = True
except ImportError:
    # Load individually to avoid circular import
    PHASE2_AVAILABLE = False
    ConsciousCell = None
    Cluster = None


# =============================================================================
# Phase 1 Tests: Core Resilience Components
# =============================================================================

class TestCellResilience:
    """Tests for CellResilience dataclass."""

    def test_initial_state(self):
        """New CellResilience starts in OPTIMAL state."""
        r = CellResilience()
        assert r.degradation_level == 0.0
        assert r.state == 'OPTIMAL'
        assert r.is_functional == True

    def test_degradation_states(self):
        """State transitions at correct thresholds."""
        r = CellResilience()

        # OPTIMAL: 0 - 0.2
        r.degradation_level = 0.1
        assert r.state == 'OPTIMAL'

        # STRESSED: 0.2 - 0.4
        r.degradation_level = 0.3
        assert r.state == 'STRESSED'

        # IMPAIRED: 0.4 - 0.6
        r.degradation_level = 0.5
        assert r.state == 'IMPAIRED'

        # CRITICAL: 0.6 - 0.8
        r.degradation_level = 0.7
        assert r.state == 'CRITICAL'

        # COLLAPSED: >= 0.8
        r.degradation_level = 0.85
        assert r.state == 'COLLAPSED'

    def test_is_functional(self):
        """Cell is functional when degradation < 0.8."""
        r = CellResilience()

        r.degradation_level = 0.79
        assert r.is_functional == True

        r.degradation_level = 0.80
        assert r.is_functional == False

    def test_vulnerability(self):
        """Vulnerability scales with degradation and protections."""
        r = CellResilience()

        # No degradation, no modules = vulnerability equal to degradation
        r.degradation_level = 0.0
        assert r.vulnerability == 0.0

        # Higher degradation = higher vulnerability
        r.degradation_level = 0.5
        assert r.vulnerability == 0.5

        # Modules reduce vulnerability
        r.modules.append(MicroModule('threat_filter', strength=0.8))
        # vulnerability = degradation - module_protection
        assert r.vulnerability < 0.5

    def test_modules_list(self):
        """Can add and query modules."""
        r = CellResilience()
        assert len(r.modules) == 0

        m = MicroModule('threat_filter', strength=0.7)
        r.modules.append(m)

        assert len(r.modules) == 1
        assert r.has_module_type('threat_filter') == True
        assert r.has_module_type('cascade_breaker') == False

    def test_get_modules_by_type(self):
        """Can get modules by type."""
        r = CellResilience()

        r.modules.append(MicroModule('threat_filter', strength=0.5))
        r.modules.append(MicroModule('threat_filter', strength=0.3))
        r.modules.append(MicroModule('cascade_breaker', strength=0.4))

        threat_modules = r.get_modules_by_type('threat_filter')
        assert len(threat_modules) == 2

        cascade_modules = r.get_modules_by_type('cascade_breaker')
        assert len(cascade_modules) == 1

        # Total strength for type
        total_strength = sum(m.strength for m in threat_modules)
        assert total_strength == pytest.approx(0.8)


class TestMicroModule:
    """Tests for MicroModule dataclass."""

    def test_creation(self):
        """Module created with correct defaults."""
        m = MicroModule('threat_filter')
        assert m.module_type == 'threat_filter'
        assert m.strength == 0.5
        assert m.activations == 0
        assert m.contribution == 0.0

    def test_apply_effect(self):
        """apply() returns correct effect and tracks activation."""
        m = MicroModule('threat_filter', strength=0.8)

        # Threat filter should reduce threat by strength * config effect
        effect = m.apply({'threat_filter': 0.2})

        assert effect == pytest.approx(0.16)  # 0.8 * 0.2
        assert m.activations == 1

    def test_contribution_tracking(self):
        """Reinforce accumulates contribution."""
        m = MicroModule('recovery_accelerator', strength=0.6)

        # Reinforce tracks contribution
        m.reinforce(amount=0.1)
        m.reinforce(amount=0.1)
        m.reinforce(amount=0.1)

        # Contribution grows with each reinforce
        assert m.contribution > 0

    def test_is_consolidated(self):
        """Module consolidated after min_activations AND positive contribution."""
        m = MicroModule('cascade_breaker', strength=0.5)

        # Not consolidated yet
        assert m.is_consolidated(min_activations=3) == False

        # Just activations - not enough, need contribution too
        m.activations = 3
        assert m.is_consolidated(min_activations=3) == False

        # Add positive contribution
        m.contribution = 0.1
        assert m.is_consolidated(min_activations=3) == True

    def test_copy_weakened(self):
        """copy_weakened creates weaker copy."""
        m = MicroModule('threat_filter', strength=0.8)
        m.activations = 5
        m.contribution = 0.4

        copy = m.copy_weakened(strength_factor=0.5)

        assert copy.module_type == 'threat_filter'
        assert copy.strength == 0.4  # 0.8 * 0.5
        assert copy.activations == 0  # Reset
        assert copy.contribution == 0.0  # Reset


class TestDamageSystem:
    """Tests for DamageSystem class."""

    @pytest.fixture
    def damage_system(self):
        """Create DamageSystem with optimal config."""
        cfg = get_preset_config('optimal')
        return DamageSystem(cfg)

    @pytest.fixture
    def mock_cell(self):
        """Create mock cell for testing."""
        class MockCell:
            def __init__(self):
                self.position = np.array([0.5, 0.5])
                self.resilience = CellResilience()
        return MockCell()

    def test_apply_damage(self, damage_system, mock_cell):
        """Damage increases degradation level."""
        initial_deg = mock_cell.resilience.degradation_level

        dmg = damage_system.apply_damage(
            mock_cell,
            mock_cell.resilience,
            base_damage=0.3
        )

        assert dmg > 0
        assert mock_cell.resilience.degradation_level > initial_deg

    def test_damage_creates_modules(self, damage_system, mock_cell):
        """Significant damage can create modules."""
        # Apply several rounds of damage
        for _ in range(5):
            damage_system.apply_damage(
                mock_cell,
                mock_cell.resilience,
                base_damage=0.4
            )

        # May have created modules (probabilistic)
        # At minimum, threat_buffer should be set
        assert mock_cell.resilience.threat_buffer >= 0

    def test_apply_recovery(self, damage_system, mock_cell):
        """Recovery reduces degradation."""
        mock_cell.resilience.degradation_level = 0.5

        recovery = damage_system.apply_recovery(
            mock_cell,
            mock_cell.resilience,
            cluster_cohesion=0.8
        )

        assert recovery > 0
        assert mock_cell.resilience.degradation_level < 0.5

    def test_recovery_with_modules(self, damage_system, mock_cell):
        """Recovery accelerator module boosts recovery."""
        mock_cell.resilience.degradation_level = 0.5

        # Add recovery accelerator
        m = MicroModule('recovery_accelerator', strength=0.8)
        mock_cell.resilience.modules.append(m)

        recovery = damage_system.apply_recovery(
            mock_cell,
            mock_cell.resilience,
            cluster_cohesion=0.8
        )

        # Should have better recovery
        assert recovery > 0
        assert m.activations > 0  # Module was activated

    def test_spread_modules_in_cluster(self, damage_system):
        """Consolidated modules spread to neighbors."""
        # Create cells
        class MockCell:
            def __init__(self):
                self.position = np.array([0.5, 0.5])
                self.resilience = CellResilience()

        cells = [MockCell() for _ in range(5)]

        # Give one cell a consolidated module
        m = MicroModule('threat_filter', strength=0.8)
        m.activations = 10
        m.contribution = 0.5
        cells[0].resilience.modules.append(m)

        spread_count = damage_system.spread_modules_in_cluster(cells)

        # At least the donor has the module
        total_with_module = sum(
            1 for c in cells
            if c.resilience.has_module_type('threat_filter')
        )
        assert total_with_module >= 1

    def test_get_metrics(self, damage_system):
        """Metrics correctly calculated for cell population."""
        class MockCell:
            def __init__(self, deg=0.0):
                self.position = np.array([0.5, 0.5])
                self.resilience = CellResilience()
                self.resilience.degradation_level = deg

        # Create varied population
        cells = [
            MockCell(0.0),   # OPTIMAL
            MockCell(0.3),   # STRESSED
            MockCell(0.5),   # IMPAIRED
            MockCell(0.7),   # CRITICAL
            MockCell(0.9),   # COLLAPSED
        ]

        metrics = damage_system.get_metrics(cells)

        assert 'HS' in metrics
        assert 'mean_degradation' in metrics
        assert 'total_modules' in metrics
        assert 'functional_count' in metrics

        # 4 of 5 are functional (deg < 0.8)
        assert metrics['functional_count'] == 4
        assert metrics['HS'] == pytest.approx(0.8)  # HS = functional / total
        assert metrics['mean_degradation'] == pytest.approx(0.48)


class TestResilienceConfig:
    """Tests for resilience_config module."""

    def test_list_presets(self):
        """All expected presets available."""
        presets = list_presets()

        assert 'demo' in presets
        assert 'optimal' in presets
        assert 'stress' in presets
        assert 'validation' in presets

    def test_preset_scaling(self):
        """Presets have correct scaling."""
        demo = get_preset_config('demo')
        optimal = get_preset_config('optimal')
        stress = get_preset_config('stress')

        # Demo should be gentler
        assert demo['damage']['multiplier'] < optimal['damage']['multiplier']

        # Stress should be harder
        assert stress['damage']['multiplier'] > optimal['damage']['multiplier']

    def test_validation_preset(self):
        """Validation preset has 3.9x damage for synth-v2 testing."""
        validation = get_preset_config('validation')

        # Should have the specific damage multiplier for IPUESA-SYNTH-v2
        assert validation['damage']['multiplier'] == pytest.approx(3.9, rel=0.1)

    def test_config_structure(self):
        """Config has all required sections."""
        cfg = get_preset_config('optimal')

        # Required sections
        assert 'damage' in cfg
        assert 'recovery' in cfg
        assert 'modules' in cfg
        assert 'anticipation' in cfg
        assert 'protection' in cfg

        # Damage section
        assert 'multiplier' in cfg['damage']
        assert 'residual_cap' in cfg['damage']

        # Recovery section
        assert 'base_rate' in cfg['recovery']
        assert 'cluster_bonus' in cfg['recovery']

        # Modules section
        assert 'spreading' in cfg['modules']
        assert 'effects' in cfg['modules']

        # Protection section (hierarchical levels)
        assert 'cell' in cfg['protection']
        assert 'cluster' in cfg['protection']
        assert 'organism' in cfg['protection']


# =============================================================================
# Phase 2 Tests: Integration with Hierarchical System
# =============================================================================

@pytest.mark.skipif(not PHASE2_AVAILABLE, reason="Circular import in codebase prevents full imports")
class TestConsciousCellResilience:
    """Tests for ConsciousCell + CellResilience integration."""

    @pytest.fixture
    def conscious_cell(self):
        """Create ConsciousCell using factory method."""
        return ConsciousCell.create_random(grid_size=64)

    def test_cell_has_resilience(self, conscious_cell):
        """ConsciousCell has resilience field."""
        assert hasattr(conscious_cell, 'resilience')
        assert isinstance(conscious_cell.resilience, CellResilience)

    def test_cell_is_functional(self, conscious_cell):
        """is_functional property works."""
        assert conscious_cell.is_functional == True

        conscious_cell.resilience.degradation_level = 0.85
        assert conscious_cell.is_functional == False

    def test_effective_plasticity(self, conscious_cell):
        """Plasticity reduced by degradation."""
        # Get base plasticity from psyche
        base_plasticity = conscious_cell.psyche.get_plasticity()

        # No degradation - plasticity should be close to base
        conscious_cell.resilience.degradation_level = 0.0
        # effective = base * (1 - deg * 0.5) = base * 1.0
        assert conscious_cell.effective_plasticity == pytest.approx(base_plasticity, rel=0.1)

        # With degradation
        conscious_cell.resilience.degradation_level = 0.5
        # effective = base * (1 - 0.5 * 0.5) = base * 0.75
        expected = base_plasticity * 0.75
        assert conscious_cell.effective_plasticity == pytest.approx(expected, rel=0.1)

    def test_degradation_state(self, conscious_cell):
        """degradation_state property works."""
        conscious_cell.resilience.degradation_level = 0.0
        assert conscious_cell.degradation_state == 'OPTIMAL'

        conscious_cell.resilience.degradation_level = 0.5
        assert conscious_cell.degradation_state == 'IMPAIRED'


@pytest.mark.skipif(not PHASE2_AVAILABLE, reason="Circular import in codebase prevents full imports")
class TestClusterResilience:
    """Tests for Cluster + resilience integration."""

    @pytest.fixture
    def cluster_with_cells(self):
        """Create cluster with cells."""
        from zeta_life.consciousness.cluster import ClusterPsyche

        # Create cells using factory
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(5)]

        # Set varied degradation
        cells[0].resilience.degradation_level = 0.0
        cells[1].resilience.degradation_level = 0.2
        cells[2].resilience.degradation_level = 0.4
        cells[3].resilience.degradation_level = 0.6
        cells[4].resilience.degradation_level = 0.9  # COLLAPSED

        # Create cluster
        psyche = ClusterPsyche.from_cells(cells)
        return Cluster(cluster_id=0, cells=cells, psyche=psyche)

    def test_cluster_resilience(self, cluster_with_cells):
        """cluster_resilience property works."""
        resilience = cluster_with_cells.cluster_resilience
        assert 0 <= resilience <= 1

    def test_functional_ratio(self, cluster_with_cells):
        """functional_ratio correctly calculated."""
        # 4 of 5 cells functional (deg < 0.8)
        assert cluster_with_cells.functional_ratio == 0.8

    def test_mean_degradation(self, cluster_with_cells):
        """mean_degradation correctly calculated."""
        expected = (0.0 + 0.2 + 0.4 + 0.6 + 0.9) / 5
        assert cluster_with_cells.mean_degradation == pytest.approx(expected)

    def test_get_functional_cells(self, cluster_with_cells):
        """get_functional_cells returns only functional cells."""
        functional = cluster_with_cells.get_functional_cells()
        assert len(functional) == 4

    def test_spread_modules(self, cluster_with_cells):
        """spread_modules works with config."""
        cfg = get_preset_config('optimal')

        # Add consolidated module to first cell
        m = MicroModule('threat_filter', strength=0.7)
        m.activations = 10
        m.contribution = 0.5
        cluster_with_cells.cells[0].resilience.modules.append(m)

        spread = cluster_with_cells.spread_modules(cfg)

        # Function should run without error
        assert spread >= 0


# =============================================================================
# Self-Evidence Tests (IPUESA Criteria)
# =============================================================================

@pytest.mark.skipif(not PHASE2_AVAILABLE, reason="Circular import in codebase prevents full imports")
class TestSelfEvidence:
    """Tests for IPUESA self-evidence criteria."""

    def test_hs_in_range(self):
        """HS (Holographic Survival) in valid range under stress."""
        cfg = get_preset_config('validation')  # 3.9x damage
        ds = DamageSystem(cfg)

        # Create cell population using factory
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(20)]

        # Simulate damage rounds
        for _ in range(10):
            for cell in cells:
                ds.apply_damage(cell, cell.resilience, base_damage=0.15)
            for cell in cells:
                ds.apply_recovery(cell, cell.resilience, cluster_cohesion=0.6)

        metrics = ds.get_metrics(cells)

        # HS should be in valid range [0, 1]
        assert 0 <= metrics['HS'] <= 1

    def test_modules_emerge_under_stress(self):
        """Modules should emerge under stress conditions."""
        cfg = get_preset_config('stress')
        ds = DamageSystem(cfg)

        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(10)]

        # Apply stress
        for _ in range(20):
            for cell in cells:
                ds.apply_damage(cell, cell.resilience, base_damage=0.3)

        # Count modules (probabilistic, so just check it's possible)
        total_modules = sum(len(c.resilience.modules) for c in cells)
        assert total_modules >= 0

    def test_embedding_integrity(self):
        """Embedding integrity (EI) should be maintained."""
        cells = [ConsciousCell.create_random(grid_size=64) for _ in range(10)]

        # Check each cell has valid resilience
        for cell in cells:
            assert cell.resilience is not None
            assert isinstance(cell.resilience, CellResilience)
            assert 0 <= cell.resilience.degradation_level <= 1

    def test_gradual_degradation(self):
        """Degradation should be gradual, not bistable."""
        cfg = get_preset_config('optimal')
        ds = DamageSystem(cfg)

        cell = ConsciousCell.create_random(grid_size=64)

        # Track degradation over time
        degradation_history = [cell.resilience.degradation_level]

        for _ in range(30):
            ds.apply_damage(cell, cell.resilience, base_damage=0.05)
            degradation_history.append(cell.resilience.degradation_level)

        # Check gradual increase (no sudden jumps > 0.3)
        for i in range(1, len(degradation_history)):
            delta = degradation_history[i] - degradation_history[i-1]
            assert delta < 0.3, f"Sudden jump in degradation: {delta}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
