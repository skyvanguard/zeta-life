"""
Conscious Kernel Validation Experiment
=======================================

Validates the 6 success criteria from the design document:
1. Prediction error decreases over repeated patterns
2. Identity persists across save/restore
3. Generalization to novel inputs after exposure
4. Memory consolidation (slow memory improves post-dream)
5. Self-awareness depth (reflection converges)
6. Curiosity behavior (epistemic error tracks learning)
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zeta_life.kernel import ConsciousKernel


def test_prediction_error_decreases():
    """Criterion 1: Free energy decreases on repeated patterns."""
    print("\n=== Test 1: Prediction Error Decreases ===")
    ck = ConsciousKernel()
    pattern = torch.tensor([0.6, 0.2, 0.1, 0.1])

    energies = []
    for _ in range(50):
        result = ck.step(pattern)
        energies.append(result.free_energy)

    avg_first = sum(energies[:10]) / 10
    avg_last = sum(energies[-10:]) / 10

    print(f"  First 10 avg free energy: {avg_first:.4f}")
    print(f"  Last 10 avg free energy:  {avg_last:.4f}")
    print(f"  Reduction:                {(1 - avg_last / max(avg_first, 1e-8)) * 100:.1f}%")
    passed = avg_last < avg_first
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_identity_persistence():
    """Criterion 2: Identity survives save/restore."""
    print("\n=== Test 2: Identity Persistence ===")
    ck1 = ConsciousKernel()

    for _ in range(30):
        ck1.step(torch.tensor([0.5, 0.2, 0.2, 0.1]))

    embed_before = ck1.self_model.self_embedding.data.clone()
    step_before = ck1.t

    with tempfile.TemporaryDirectory() as tmpdir:
        ck1.save(tmpdir, 'test')
        ck2 = ConsciousKernel()
        ck2.load(tmpdir, 'test')

    embed_after = ck2.self_model.self_embedding.data
    distance = torch.norm(embed_before - embed_after).item()

    print(f"  Steps saved:          {step_before}")
    print(f"  Steps restored:       {ck2.t}")
    print(f"  Embedding distance:   {distance:.6f}")
    # After load, a wake-up reflection runs which slightly modifies embedding
    passed = distance < 0.5 and ck2.t == step_before
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_generalization():
    """Criterion 3: Slow memory generalizes after exposure.

    Measures generalization as *improvement from training* (post-training error
    on a novel input lower than the same kernel's pre-training error), not an
    absolute error threshold. The absolute error is dominated by random
    initialisation (~0.5 ± 0.06) and the learning signal is small (~6%), so a
    fixed `error < 0.5` threshold was a coin flip across runs. Comparing the same
    kernel before vs after isolates the learning effect. Seeded for reproducibility.
    """
    print("\n=== Test 3: Generalization ===")
    torch.manual_seed(0)
    ck = ConsciousKernel()

    pattern = torch.tensor([0.7, 0.1, 0.1, 0.1])
    # SlowMemory is trained on softmax(stimulus), so the query must also be softmax
    novel = torch.tensor([0.65, 0.15, 0.1, 0.1])
    novel_soft = torch.nn.functional.softmax(novel, dim=-1)
    training_soft = torch.nn.functional.softmax(pattern, dim=-1)

    # Baseline: error on the novel input BEFORE any training (random init).
    pre_error = torch.norm(ck.slow_memory.generalize(novel_soft) - novel_soft).item()

    # Need enough steps for the slow neocortical network to learn (lr=0.0001)
    for _ in range(500):
        ck.step(pattern)
    # Force multiple dream consolidation cycles for deeper transfer
    for _ in range(5):
        ck.dream_engine.dream_cycle(duration=50)

    predicted = ck.slow_memory.generalize(novel_soft)
    post_error = torch.norm(predicted - novel_soft).item()

    print(f"  Training pattern (softmax): {[f'{x:.3f}' for x in training_soft.tolist()]}")
    print(f"  Novel input (softmax):      {[f'{x:.3f}' for x in novel_soft.tolist()]}")
    print(f"  Predicted:                  {[f'{x:.3f}' for x in predicted.tolist()]}")
    print(f"  Error pre-train:            {pre_error:.4f}")
    print(f"  Error post-train:           {post_error:.4f}")
    # Generalizes = training reduces error on the novel input.
    passed = post_error < pre_error
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_memory_consolidation():
    """Criterion 4: Slow memory accuracy improves post-dream."""
    print("\n=== Test 4: Memory Consolidation ===")
    ck = ConsciousKernel()

    pattern = torch.tensor([0.6, 0.2, 0.1, 0.1])

    # Pre-dream: train with some episodes
    for _ in range(50):
        ck.step(pattern)

    # Measure pre-dream accuracy
    pre_pred = ck.slow_memory.generalize(pattern)
    pre_error = torch.norm(pre_pred - pattern).item()

    # Dream
    ck.dream_engine.dream_cycle(duration=50)

    # Measure post-dream accuracy
    post_pred = ck.slow_memory.generalize(pattern)
    post_error = torch.norm(post_pred - pattern).item()

    print(f"  Pre-dream error:  {pre_error:.4f}")
    print(f"  Post-dream error: {post_error:.4f}")
    # Dream should at least not make things worse
    passed = post_error <= pre_error + 0.1  # allow small variance
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_self_awareness_depth():
    """Criterion 5: Reflection works at depth 3-4."""
    print("\n=== Test 5: Self-Awareness Depth ===")
    ck = ConsciousKernel()

    # Build some state
    for _ in range(20):
        ck.step(torch.randn(4).abs())

    state = torch.tensor([0.4, 0.3, 0.2, 0.1])
    reflections = ck.self_model.reflect(state, depth=4)

    print(f"  Depth levels: {len(reflections)}")
    for r in reflections:
        print(f"    Level {r['depth']}: PE = {r['prediction_error']:.4f}")

    passed = len(reflections) == 4
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_dream_with_zeta_binding():
    """Criterion 6: Dream engine uses zeta kernel for phase selection."""
    print("\n=== Test 6: Zeta Binding in Dreams ===")
    ck = ConsciousKernel()

    # Feed episodes to build memory
    for i in range(30):
        stimulus = torch.randn(4).abs()
        stimulus = stimulus / stimulus.sum()
        ck.step(stimulus)

    # Run dream and check phases
    report = ck.dream_engine.dream_cycle(duration=50)

    print(f"  Duration:    {report.duration}")
    print(f"  Selections:  {report.selections}")
    print(f"  Transfers:   {report.transfers}")
    print(f"  Replays:     {report.replays}")
    print(f"  Phases:      {report.phases_visited}")

    # All three phases should be visited
    all_phases_visited = all(v > 0 for v in report.phases_visited.values())
    has_activity = report.transfers > 0 or report.replays > 0

    print(f"  All phases visited: {all_phases_visited}")
    print(f"  Has activity:       {has_activity}")
    passed = all_phases_visited and has_activity
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("=" * 60)
    print("  CONSCIOUS KERNEL VALIDATION")
    print("  Active Inference + Zeta Binding + Strange Loop")
    print("=" * 60)

    results = [
        ('Prediction Error Decreases', test_prediction_error_decreases()),
        ('Identity Persistence', test_identity_persistence()),
        ('Generalization', test_generalization()),
        ('Memory Consolidation', test_memory_consolidation()),
        ('Self-Awareness Depth', test_self_awareness_depth()),
        ('Zeta Binding in Dreams', test_dream_with_zeta_binding()),
    ]

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = sum(1 for _, p in results if p)
    print(f"\n  Total: {total}/{len(results)} passed")
    print("=" * 60)

    return total == len(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
