#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zeta-Life Quickstart Demo
=========================

A 5-minute introduction to the main concepts:
1. Zeta Kernel visualization
2. ZetaPsyche archetype dynamics
3. IPUESA self-evidence (simplified)

Run: python demos/quickstart.py
"""

import numpy as np
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, Exception):
        pass


def demo_zeta_kernel():
    """Demo 1: Visualize the zeta kernel."""
    print("\n" + "=" * 60)
    print("DEMO 1: Zeta Kernel")
    print("=" * 60)

    # First few imaginary parts of zeta zeros
    gammas = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]

    print("\nRiemann zeta zeros (imaginary parts):")
    for i, g in enumerate(gammas):
        print(f"  γ_{i+1} = {g:.6f}")

    # Kernel function
    sigma = 0.1
    t = np.linspace(0, 5, 100)

    def K(t, sigma, gammas):
        return 2 * sum(np.exp(-sigma * abs(g)) * np.cos(g * t) for g in gammas)

    kernel_values = K(t, sigma, gammas)

    print(f"\nKernel K_σ(t) with σ={sigma}:")
    print(f"  K(0) = {K(0, sigma, gammas):.4f}")
    print(f"  K(1) = {K(1, sigma, gammas):.4f}")
    print(f"  K(2) = {K(2, sigma, gammas):.4f}")

    # ASCII visualization
    print("\nKernel shape (ASCII):")
    min_k, max_k = min(kernel_values), max(kernel_values)
    for i in range(0, 50, 5):
        val = kernel_values[i * 2]
        normalized = int(20 * (val - min_k) / (max_k - min_k + 1e-6))
        bar = "#" * normalized
        print(f"  t={t[i*2]:.1f}: {bar}")


def demo_psyche():
    """Demo 2: ZetaPsyche archetype dynamics."""
    print("\n" + "=" * 60)
    print("DEMO 2: ZetaPsyche Archetypes")
    print("=" * 60)

    # Simplified archetype space
    archetypes = {
        'PERSONA': np.array([1, 0, 0, 0]),   # Social mask
        'SOMBRA': np.array([0, 1, 0, 0]),    # Shadow
        'ANIMA': np.array([0, 0, 1, 0]),     # Receptive
        'ANIMUS': np.array([0, 0, 0, 1]),    # Active
    }

    print("\nJungian Archetypes (tetrahedral vertices):")
    for name, vec in archetypes.items():
        print(f"  {name}: {vec}")

    # Simulate simple dynamics
    print("\nSimulating archetype evolution...")
    state = np.array([0.25, 0.25, 0.25, 0.25])  # Start balanced

    for step in range(5):
        # Random perturbation
        noise = np.random.randn(4) * 0.1
        state = state + noise
        state = np.abs(state)
        state = state / state.sum()  # Normalize

        dominant = ['PERSONA', 'SOMBRA', 'ANIMA', 'ANIMUS'][np.argmax(state)]
        print(f"  Step {step+1}: {state.round(3)} -> Dominant: {dominant}")

    # Integration metric
    dist_to_center = np.linalg.norm(state - 0.25)
    integration = 1 - dist_to_center / 0.5
    print(f"\nSelf Integration: {integration:.2%}")


def demo_ipuesa():
    """Demo 3: IPUESA self-evidence (simplified)."""
    print("\n" + "=" * 60)
    print("DEMO 3: IPUESA Self-Evidence")
    print("=" * 60)

    print("\nIPUESA = Investigación de Persistencia Universal")
    print("          de Entidades con Self Anticipatorio")

    # Simulate simplified SYNTH-v2 metrics
    print("\nSimulating SYNTH-v2 metrics...")
    np.random.seed(42)

    # Metrics with noise
    metrics = {
        'HS': 0.40 + np.random.randn() * 0.05,   # Holographic Survival
        'TAE': 0.22 + np.random.randn() * 0.02,  # Temporal Anticipation
        'MSR': 0.50 + np.random.randn() * 0.03,  # Module Spreading
        'EI': 0.95 + np.random.randn() * 0.02,   # Embedding Integrity
        'ED': 0.36 + np.random.randn() * 0.04,   # Emergent Differentiation
        'deg_var': 0.028 + np.random.randn() * 0.003,  # Degradation Variance
    }

    thresholds = {
        'HS': (0.30, 0.70),
        'TAE': (0.15, 1.0),
        'MSR': (0.15, 1.0),
        'EI': (0.30, 1.0),
        'ED': (0.10, 1.0),
        'deg_var': (0.02, 1.0),
    }

    print("\nMetrics vs Thresholds:")
    passed = 0
    for name, value in metrics.items():
        lo, hi = thresholds[name]
        ok = lo <= value <= hi
        passed += ok
        symbol = "✓" if ok else "✗"
        print(f"  {name}: {value:.3f} [{lo:.2f}-{hi:.2f}] {symbol}")

    print(f"\nCriteria Passed: {passed}/6")

    if passed >= 5:
        print("\n→ System exhibits FUNCTIONAL IDENTITY ATTRACTOR")
    else:
        print("\n→ System does NOT exhibit self-evidence")


def main():
    print("=" * 60)
    print("ZETA-LIFE QUICKSTART DEMO")
    print("=" * 60)
    print("\nThis demo introduces 3 key concepts:")
    print("1. Zeta Kernel - Mathematical foundation")
    print("2. ZetaPsyche - Jungian archetype dynamics")
    print("3. IPUESA - Functional identity metrics")

    demo_zeta_kernel()
    demo_psyche()
    demo_ipuesa()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Run notebooks: jupyter notebook notebooks/")
    print("  • Run experiments: python experiments/consciousness/exp_ipuesa_synth_v2.py")
    print("  • Read paper: docs/papers/ipuesa-identidad-funcional-paper.md")
    print("  • Interactive chat: python demos/chat_psyche.py --reflection")


if __name__ == '__main__':
    main()
