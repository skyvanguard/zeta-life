#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEMO: IPUESA Resilience System
==============================

Interactive demonstration of the resilience mechanics integrated
into the hierarchical consciousness system.

Features demonstrated:
1. Gradual degradation (5 states)
2. Module creation under stress
3. Module spreading within clusters
4. Temporal anticipation (TAE)
5. Recovery with cluster cohesion

Usage:
    python demos/demo_resilience.py

Date: 2026-01-12
"""

import sys
import os
import time

# Fix Windows console
if sys.platform == 'win32':
    os.system('')

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zeta_life.consciousness.resilience import CellResilience, MicroModule, MODULE_TYPES
from zeta_life.consciousness.resilience_config import get_preset_config, list_presets
from zeta_life.consciousness.damage_system import DamageSystem


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

COLORS = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'gray': '\033[90m',
}

STATE_COLORS = {
    'OPTIMAL': 'green',
    'STRESSED': 'yellow',
    'IMPAIRED': 'magenta',
    'CRITICAL': 'red',
    'COLLAPSED': 'gray',
}


def color(text: str, color_name: str) -> str:
    return f"{COLORS.get(color_name, '')}{text}{COLORS['reset']}"


def print_header(title: str, char: str = "=", width: int = 60):
    print(f"\n{color(char * width, 'cyan')}")
    print(f"  {color(title, 'bold')}")
    print(f"{color(char * width, 'cyan')}")


def print_subheader(title: str):
    print(f"\n  {color('>>> ' + title, 'yellow')}")


def progress_bar(value: float, width: int = 30, filled_char: str = "█", empty_char: str = "░") -> str:
    value = max(0, min(1, value))
    filled = int(value * width)
    empty = width - filled

    if value < 0.2:
        col = 'green'
    elif value < 0.4:
        col = 'yellow'
    elif value < 0.6:
        col = 'magenta'
    else:
        col = 'red'

    bar = color(filled_char * filled, col) + color(empty_char * empty, 'gray')
    return f"[{bar}] {value:.1%}"


def state_indicator(state: str) -> str:
    col = STATE_COLORS.get(state, 'reset')
    symbols = {
        'OPTIMAL': '●',
        'STRESSED': '◐',
        'IMPAIRED': '◑',
        'CRITICAL': '◒',
        'COLLAPSED': '○',
    }
    return color(f"{symbols.get(state, '?')} {state}", col)


# =============================================================================
# MOCK CELL FOR DEMO
# =============================================================================

class DemoCell:
    """Simple cell for demonstration."""
    def __init__(self, cell_id: int):
        self.cell_id = cell_id
        self.position = np.random.uniform(0, 64, size=2)
        self.resilience = CellResilience()
        # Initialize with embedding for holographic protection
        self.resilience.embedding = np.random.randn(8)
        self.resilience.embedding /= np.linalg.norm(self.resilience.embedding)


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_degradation_states():
    """Demonstrate the 5 degradation states."""
    print_header("1. DEGRADATION STATES")

    print("\n  The resilience system has 5 degradation states:")
    print()

    cell = DemoCell(0)

    levels = [0.0, 0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

    print(f"  {'Level':<8} {'State':<20} {'Functional':<12} {'Bar'}")
    print(f"  {'-'*60}")

    for level in levels:
        cell.resilience.degradation_level = level
        state = cell.resilience.state
        functional = "Yes" if cell.resilience.is_functional else "No"
        bar = progress_bar(level, width=20)

        func_color = 'green' if cell.resilience.is_functional else 'red'
        print(f"  {level:<8.2f} {state_indicator(state):<28} {color(functional, func_color):<20} {bar}")

    print(f"\n  {color('Threshold:', 'bold')} Cell becomes non-functional at degradation >= 0.8")


def demo_damage_and_recovery(preset: str = 'demo'):
    """Demonstrate damage and recovery mechanics."""
    print_header(f"2. DAMAGE & RECOVERY (preset: {preset})")

    config = get_preset_config(preset)
    ds = DamageSystem(config)

    print(f"\n  Config: damage_multiplier = {config['damage']['multiplier']:.2f}")
    print(f"          recovery_rate = {config['recovery']['base_rate']:.4f}")

    cell = DemoCell(0)

    print_subheader("Applying damage over 10 steps")
    print()
    print(f"  {'Step':<6} {'Degradation':<14} {'State':<15} {'Modules':<10} {'Threat Buffer'}")
    print(f"  {'-'*65}")

    for step in range(10):
        # Apply damage
        dmg = ds.apply_damage(cell, cell.resilience, base_damage=0.25)

        state = cell.resilience.state
        deg = cell.resilience.degradation_level
        modules = len(cell.resilience.modules)
        threat = cell.resilience.threat_buffer

        print(f"  {step+1:<6} {progress_bar(deg, 15):<22} {state_indicator(state):<23} {modules:<10} {threat:.3f}")

        time.sleep(0.1)

    print_subheader("Applying recovery over 10 steps (cohesion=0.8)")
    print()

    for step in range(10):
        # Apply recovery with good cluster cohesion
        rec = ds.apply_recovery(cell, cell.resilience, cluster_cohesion=0.8)

        state = cell.resilience.state
        deg = cell.resilience.degradation_level
        modules = len(cell.resilience.modules)

        print(f"  {step+1:<6} {progress_bar(deg, 15):<22} {state_indicator(state):<23} {modules:<10} +{rec:.4f}")

        time.sleep(0.1)


def demo_module_types():
    """Demonstrate the 8 module types."""
    print_header("3. MICRO-MODULES (8 Types)")

    print("\n  Modules are created under stress and provide protection:")
    print()

    config = get_preset_config('optimal')
    effects = config['modules']['effects']

    print(f"  {'Type':<25} {'Effect':<10} {'Description'}")
    print(f"  {'-'*70}")

    descriptions = {
        'pattern_detector': 'Recognizes threat patterns',
        'threat_filter': 'Reduces incoming damage',
        'recovery_accelerator': 'Speeds up recovery',
        'exploration_dampener': 'Reduces exploration under stress',
        'embedding_protector': 'Preserves embedding integrity',
        'cascade_breaker': 'Prevents damage cascades',
        'residual_cleaner': 'Clears accumulated damage',
        'anticipation_enhancer': 'Improves threat prediction',
    }

    for mtype in MODULE_TYPES:
        effect = effects.get(mtype, 0.1)
        desc = descriptions.get(mtype, '')
        print(f"  {color(mtype, 'cyan'):<33} {effect:<10.2f} {desc}")


def demo_module_spreading():
    """Demonstrate module spreading within a cluster."""
    print_header("4. MODULE SPREADING")

    config = get_preset_config('optimal')
    ds = DamageSystem(config)

    # Create cluster of cells
    cells = [DemoCell(i) for i in range(6)]

    print("\n  Initial state (no modules):")
    print()
    for cell in cells:
        modules = len(cell.resilience.modules)
        print(f"    Cell {cell.cell_id}: {modules} modules")

    # Give first cell a consolidated module
    print_subheader("Adding consolidated module to Cell 0")

    m = MicroModule('threat_filter', strength=0.8)
    m.activations = 10
    m.contribution = 0.5
    cells[0].resilience.modules.append(m)

    print(f"\n    Cell 0 now has: {color('threat_filter', 'cyan')} (strength=0.8, activations=10)")

    print_subheader("Spreading modules in cluster")

    # Spread multiple times
    total_spread = 0
    for round_num in range(3):
        spread = ds.spread_modules_in_cluster(cells)
        total_spread += spread
        print(f"\n    Round {round_num + 1}: {spread} modules spread")

        for cell in cells:
            modules = cell.resilience.modules
            if modules:
                module_str = ", ".join([f"{m.module_type}({m.strength:.2f})" for m in modules])
                print(f"      Cell {cell.cell_id}: {color(module_str, 'cyan')}")
            else:
                print(f"      Cell {cell.cell_id}: {color('(none)', 'gray')}")

    print(f"\n  {color(f'Total spread: {total_spread} modules', 'green')}")


def demo_temporal_anticipation():
    """Demonstrate temporal anticipation (TAE)."""
    print_header("5. TEMPORAL ANTICIPATION (TAE)")

    config = get_preset_config('optimal')
    ds = DamageSystem(config)

    cell = DemoCell(0)

    print("\n  TAE components:")
    print("    - threat_buffer: EMA of recent damage")
    print("    - anticipated_damage: Predicted future damage")
    print("    - protective_stance: Proactive protection level")
    print()

    print(f"  {'Step':<6} {'Damage':<10} {'Threat Buf':<12} {'Anticipated':<12} {'Stance':<10}")
    print(f"  {'-'*55}")

    # Simulate increasing threat
    damages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    for step, base_dmg in enumerate(damages):
        ds.apply_damage(cell, cell.resilience, base_damage=base_dmg)

        tb = cell.resilience.threat_buffer
        ad = cell.resilience.anticipated_damage
        ps = cell.resilience.protective_stance

        print(f"  {step+1:<6} {base_dmg:<10.2f} {tb:<12.3f} {ad:<12.3f} {ps:<10.3f}")

        time.sleep(0.1)

    print(f"\n  {color('Note:', 'bold')} Protective stance increases when anticipated_damage > 0.3")


def demo_cluster_simulation():
    """Demonstrate full cluster simulation with storm."""
    print_header("6. CLUSTER STORM SIMULATION")

    config = get_preset_config('demo')  # Use gentle preset
    ds = DamageSystem(config)

    # Create cluster
    n_cells = 12
    cells = [DemoCell(i) for i in range(n_cells)]

    print(f"\n  Cluster: {n_cells} cells")
    print(f"  Preset: demo (0.6x damage)")
    print()

    # Run simulation
    n_steps = 30

    print(f"  {'Step':<6} {'Functional':<12} {'Mean Deg':<12} {'Modules':<10} {'Status'}")
    print(f"  {'-'*55}")

    for step in range(n_steps):
        # Storm phase (first 15 steps)
        if step < 15:
            base_damage = 0.2 + (step / 15) * 0.2  # Escalating
            status = color("STORM", 'red')
        else:
            base_damage = 0.0
            status = color("RECOVERY", 'green')

        # Apply to all cells
        for cell in cells:
            if base_damage > 0:
                ds.apply_damage(cell, cell.resilience, base_damage=base_damage)
            ds.apply_recovery(cell, cell.resilience, cluster_cohesion=0.7)

        # Spread modules every 5 steps
        if step % 5 == 0:
            ds.spread_modules_in_cluster(cells)

        # Calculate metrics
        functional = sum(1 for c in cells if c.resilience.is_functional)
        mean_deg = np.mean([c.resilience.degradation_level for c in cells])
        total_modules = sum(len(c.resilience.modules) for c in cells)

        func_color = 'green' if functional >= n_cells * 0.7 else ('yellow' if functional >= n_cells * 0.4 else 'red')

        print(f"  {step+1:<6} {color(f'{functional}/{n_cells}', func_color):<20} {mean_deg:<12.3f} {total_modules:<10} {status}")

        time.sleep(0.05)

    # Final state
    print_subheader("Final State")

    states = {}
    for cell in cells:
        state = cell.resilience.state
        states[state] = states.get(state, 0) + 1

    print()
    for state, count in sorted(states.items()):
        print(f"    {state_indicator(state)}: {count} cells")


def demo_presets():
    """Show available presets."""
    print_header("7. AVAILABLE PRESETS")

    presets = list_presets()

    print()
    print(f"  {'Preset':<15} {'Description'}")
    print(f"  {'-'*50}")

    for name, desc in presets.items():
        cfg = get_preset_config(name)
        mult = cfg['damage']['multiplier']
        print(f"  {color(name, 'cyan'):<23} {desc} ({mult:.1f}x)")


# =============================================================================
# MAIN
# =============================================================================

def main(interactive: bool = True):
    print(color("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   IPUESA RESILIENCE SYSTEM DEMO                          ║
    ║   Identity-Preserving Unified Emergent Self-Architecture  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """, 'cyan'))

    def pause():
        if interactive:
            try:
                input(color("\n  Press Enter to continue...", 'gray'))
            except EOFError:
                pass

    try:
        demo_degradation_states()
        pause()

        demo_damage_and_recovery('demo')
        pause()

        demo_module_types()
        pause()

        demo_module_spreading()
        pause()

        demo_temporal_anticipation()
        pause()

        demo_cluster_simulation()
        pause()

        demo_presets()

        print_header("DEMO COMPLETE")
        print(f"\n  {color('All resilience mechanics demonstrated!', 'green')}")
        print(f"\n  Run validation experiment:")
        print(f"    {color('python experiments/consciousness/exp_hierarchical_resilience_validation.py', 'cyan')}")
        print()

    except KeyboardInterrupt:
        print(f"\n\n  {color('Demo interrupted.', 'yellow')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='IPUESA Resilience Demo')
    parser.add_argument('--no-interactive', '-n', action='store_true',
                       help='Run without pausing for input')
    args = parser.parse_args()
    main(interactive=not args.no_interactive)
