"""
Damage System for Hierarchical Consciousness

Manages damage application, recovery, and module creation/spreading
for the consciousness system. Uses evolved parameters from IPUESA
optimization (fitness=0.9993, 8/8 criteria).

The DamageSystem implements:
- Gradual damage with module mitigation
- Recovery with cluster cohesion bonus
- Temporal anticipation for proactive protection
- Module creation under stress
- Module spreading within clusters
"""

from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np

from .resilience import CellResilience, MicroModule, MODULE_TYPES

if TYPE_CHECKING:
    from .micro_psyche import ConsciousCell


class DamageSystem:
    """
    Manages damage and recovery for cells in the hierarchical system.

    Uses configuration from evolved IPUESA parameters to control
    damage intensity, protection factors, recovery rates, and
    module dynamics.
    """

    def __init__(self, config: dict):
        """
        Initialize damage system with configuration.

        Args:
            config: Hierarchical config dict with 'damage', 'protection',
                   'recovery', 'modules', and 'anticipation' sections
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate configuration has required sections."""
        required_sections = ['damage', 'protection', 'recovery', 'modules', 'anticipation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Config missing required section: {section}")

    def apply_damage(self, cell: 'ConsciousCell', resilience: CellResilience,
                     base_damage: float) -> float:
        """
        Apply damage to cell, return actual damage dealt.

        Damage is mitigated by:
        - Active modules (threat_filter, cascade_breaker)
        - Embedding protection
        - Protective stance (from anticipation)
        - Random noise (for anti-fragility)

        Args:
            cell: The ConsciousCell being damaged
            resilience: Cell's CellResilience state
            base_damage: Raw damage amount before mitigation

        Returns:
            Actual damage dealt after mitigation
        """
        cfg_damage = self.config['damage']
        cfg_protection = self.config['protection']['cell']
        cfg_modules = self.config['modules']['effects']

        # 1. Base calculation with multiplier
        damage = base_damage * cfg_damage['multiplier']

        # 2. Module mitigation
        for module in resilience.modules:
            if module.module_type in ['threat_filter', 'cascade_breaker']:
                effect = module.apply(cfg_modules)
                damage *= (1.0 - effect)
                module.contribution += 0.1 * module.strength

        # 3. Embedding protection
        if resilience.embedding is not None:
            embedding_factor = cfg_protection['embedding'] * resilience.embedding_strength
            damage *= (1.0 - embedding_factor)

        # 4. Protective stance (from anticipation)
        stance_factor = cfg_protection['stance'] * resilience.protective_stance
        damage *= (1.0 - stance_factor)

        # 5. Module protection bonus
        module_factor = cfg_protection['module'] * len(resilience.modules) * 0.1
        damage *= (1.0 - min(module_factor, 0.3))

        # 6. Noise for anti-fragility
        noise = np.random.normal(0, cfg_damage['noise_scale'])
        damage *= (1.0 + noise)
        damage = max(0, damage)

        # 7. Apply to degradation
        degrad_rate = cfg_damage['base_degrad_rate']
        resilience.degradation_level += damage * degrad_rate
        resilience.degradation_level = min(1.0, resilience.degradation_level)

        # 8. Accumulate residual damage
        compound = cfg_damage['compound_factor']
        resilience.residual_damage += damage * compound
        resilience.residual_damage = min(cfg_damage['residual_cap'],
                                         resilience.residual_damage)

        # 9. Update threat buffer for TAE
        cfg_antic = self.config['anticipation']
        resilience.update_anticipation(
            damage,
            alpha=cfg_antic['buffer_alpha'],
            momentum=cfg_antic['momentum_factor']
        )

        # 10. Maybe create module if vulnerable
        self._maybe_create_module(resilience, damage)

        return damage

    def apply_recovery(self, cell: 'ConsciousCell', resilience: CellResilience,
                       cluster_cohesion: float = 0.5) -> float:
        """
        Apply recovery to cell, return recovery amount.

        Recovery is enhanced by:
        - Active modules (recovery_accelerator, residual_cleaner)
        - Cluster cohesion bonus
        - Embedding bonus
        - Lower degradation (easier to recover when less damaged)

        Args:
            cell: The ConsciousCell recovering
            resilience: Cell's CellResilience state
            cluster_cohesion: Cohesion of cell's cluster [0, 1]

        Returns:
            Recovery amount
        """
        cfg_recovery = self.config['recovery']
        cfg_modules = self.config['modules']['effects']

        # 1. Base recovery rate
        rate = cfg_recovery['base_rate']

        # 2. Module bonuses
        for module in resilience.modules:
            if module.module_type == 'recovery_accelerator':
                effect = module.apply(cfg_modules)
                rate *= (1 + effect)
                module.contribution += 0.1 * module.strength

            elif module.module_type == 'residual_cleaner':
                effect = module.apply(cfg_modules)
                cleaned = resilience.residual_damage * effect
                resilience.residual_damage -= cleaned
                module.contribution += 0.05 * module.strength

            elif module.module_type == 'embedding_protector':
                effect = module.apply(cfg_modules)
                resilience.embedding_strength = min(
                    1.0,
                    resilience.embedding_strength + effect * 0.1
                )
                module.contribution += 0.05 * module.strength

        # 3. Cluster cohesion bonus
        rate += cfg_recovery['cluster_bonus'] * cluster_cohesion

        # 4. Embedding bonus
        if resilience.embedding is not None:
            rate += cfg_recovery['embedding_bonus'] * resilience.embedding_strength * 0.1

        # 5. Degradation penalty (harder to recover when damaged)
        penalty = resilience.degradation_level * cfg_recovery['degradation_penalty']
        rate *= (1 - penalty)

        # 6. Residual damage penalty
        rate *= cfg_recovery['degrad_recovery_factor'] ** resilience.residual_damage

        # 7. Apply recovery
        rate = max(0, rate)
        recovery = resilience.degradation_level * rate
        resilience.degradation_level -= recovery
        resilience.degradation_level = max(0, resilience.degradation_level)

        # 8. Residual decay
        resilience.residual_damage *= cfg_recovery['corruption_decay']

        # 9. Decay unused modules
        self._decay_unused_modules(resilience)

        return recovery

    def _maybe_create_module(self, resilience: CellResilience, recent_damage: float):
        """
        Maybe create a new module if cell is vulnerable.

        Modules are created proactively when:
        - Vulnerability is high (> threshold)
        - Random check passes
        - Cell is under module cap
        """
        cfg_antic = self.config['anticipation']
        cfg_modules = self.config['modules']['spreading']

        # Check vulnerability threshold
        if resilience.vulnerability < cfg_antic['vulnerability_threshold']:
            return

        # Random creation check
        if np.random.random() > cfg_antic['creation_probability']:
            return

        # Check module cap
        if len(resilience.modules) >= cfg_modules['max_per_cell']:
            return

        # Choose module type based on current deficits
        module_type = self._choose_module_type(resilience)

        # Create module
        resilience.add_module(
            module_type=module_type,
            strength=0.5,
            max_modules=cfg_modules['max_per_cell']
        )

    def _choose_module_type(self, resilience: CellResilience) -> str:
        """
        Choose module type based on current deficits.

        Prioritizes module types the cell doesn't have yet,
        weighted by current needs.
        """
        # Count existing module types
        existing_types = set(m.module_type for m in resilience.modules)

        # Weights based on state
        weights = {}
        for mtype in MODULE_TYPES:
            if mtype in existing_types:
                weights[mtype] = 0.1  # Low weight if already have
            else:
                weights[mtype] = 1.0  # Higher weight if missing

        # Adjust weights based on state
        if resilience.degradation_level > 0.5:
            weights['recovery_accelerator'] *= 2.0
            weights['cascade_breaker'] *= 1.5

        if resilience.residual_damage > 0.3:
            weights['residual_cleaner'] *= 2.0

        if resilience.threat_buffer > 0.4:
            weights['threat_filter'] *= 2.0
            weights['anticipation_enhancer'] *= 1.5

        if resilience.embedding is not None and resilience.embedding_strength < 0.7:
            weights['embedding_protector'] *= 2.0

        # Normalize and sample
        total = sum(weights.values())
        probs = [weights[mt] / total for mt in MODULE_TYPES]

        return np.random.choice(MODULE_TYPES, p=probs)

    def _decay_unused_modules(self, resilience: CellResilience):
        """Decay modules that weren't used this step and remove weak ones."""
        cfg_modules = self.config['modules']

        # Decay all modules slightly
        for module in resilience.modules:
            module.decay(rate=0.98)

        # Remove modules below consolidation threshold
        threshold = cfg_modules['consolidation_threshold']
        resilience.remove_weak_modules(threshold)

    def spread_modules_in_cluster(self, cells: List['ConsciousCell']) -> int:
        """
        Spread consolidated modules between cells in a cluster.

        Consolidated modules (high activations, positive contribution)
        can be copied to neighboring cells that lack that module type.

        Args:
            cells: List of cells in the cluster

        Returns:
            Number of modules spread
        """
        cfg_spreading = self.config['modules']['spreading']
        spread_count = 0

        # Find all consolidated modules
        consolidated: List[Tuple['ConsciousCell', MicroModule]] = []
        for cell in cells:
            for module in cell.resilience.get_consolidated_modules(
                min_activations=cfg_spreading['min_activations']
            ):
                consolidated.append((cell, module))

        # Try to spread each consolidated module
        for source_cell, module in consolidated:
            if np.random.random() > cfg_spreading['probability']:
                continue

            # Find target cell without this module type
            for target_cell in cells:
                if target_cell is source_cell:
                    continue

                # Skip if target already has this type
                if target_cell.resilience.has_module_type(module.module_type):
                    continue

                # Skip if target at module cap
                if len(target_cell.resilience.modules) >= cfg_spreading['max_per_cell']:
                    continue

                # Create weakened copy
                new_module = module.copy_weakened(cfg_spreading['strength_factor'])
                target_cell.resilience.modules.append(new_module)
                spread_count += 1
                break  # Only spread to one target per source module

        return spread_count

    def calculate_cluster_resilience(self, cells: List['ConsciousCell']) -> float:
        """
        Calculate aggregate resilience for a cluster.

        Returns average resilience (1 - degradation) of functional cells.
        """
        functional = [c for c in cells if c.resilience.is_functional]
        if not functional:
            return 0.0

        return 1.0 - np.mean([c.resilience.degradation_level for c in functional])

    def calculate_functional_ratio(self, cells: List['ConsciousCell']) -> float:
        """Calculate proportion of functional cells."""
        if not cells:
            return 0.0
        return sum(1 for c in cells if c.resilience.is_functional) / len(cells)

    def get_metrics(self, cells: List['ConsciousCell']) -> dict:
        """
        Calculate IPUESA-compatible metrics for a set of cells.

        Returns:
            Dict with HS, MSR, EI, ED, mean_degradation, total_modules
        """
        functional = [c for c in cells if c.resilience.is_functional]

        # HS: Holographic Survival
        hs = len(functional) / len(cells) if cells else 0.0

        # MSR: Module Spreading Rate (approximated)
        total_consolidated = 0
        total_spread = 0
        for cell in cells:
            for m in cell.resilience.modules:
                if m.is_consolidated(min_activations=3):
                    total_consolidated += 1
                    if m.strength < 0.8:  # Was copied (reduced strength)
                        total_spread += 1
        msr = total_spread / max(total_consolidated, 1)

        # EI: Embedding Integrity
        ei = np.mean([c.resilience.embedding_strength for c in cells]) if cells else 0.0

        # ED: Entropy of Degradation (std as proxy)
        ed = np.std([c.resilience.degradation_level for c in cells]) if cells else 0.0

        return {
            'HS': hs,
            'MSR': msr,
            'EI': ei,
            'ED': ed,
            'mean_degradation': np.mean([c.resilience.degradation_level for c in cells]) if cells else 0.0,
            'total_modules': sum(len(c.resilience.modules) for c in cells),
            'functional_count': len(functional),
        }
