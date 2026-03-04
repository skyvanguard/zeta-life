"""
Proto-Language Analysis — Does the GW broadcast encode stimulus meaning?
=========================================================================

Without changing any code, we analyze whether the ConsciousOrganism's
Global Workspace broadcast signal develops consistent mappings to
stimulus types. If the same broadcast patterns emerge for the same
stimuli, this is proto-language: a shared internal code.

Measures:
1. Mutual Information between stimulus type and broadcast signal
2. Cluster purity: do broadcast vectors cluster by stimulus type?
3. Signal consistency: cosine similarity of broadcasts for same stimulus
4. Signal discriminability: can we decode stimulus type from broadcast?
5. Temporal stability: does the "vocabulary" stabilize over time?

A positive result means the organism spontaneously develops
a primitive referential system — proto-language.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zeta_life.kernel import ConsciousOrganism


# ---------------------------------------------------------------------------
# Stimulus environment with labeled types
# ---------------------------------------------------------------------------

STIMULUS_TYPES = {
    'A': torch.tensor([0.7, 0.1, 0.1, 0.1]),
    'B': torch.tensor([0.1, 0.7, 0.1, 0.1]),
    'C': torch.tensor([0.1, 0.1, 0.7, 0.1]),
    'D': torch.tensor([0.1, 0.1, 0.1, 0.7]),
}

def get_stimulus(t: int, noise: float = 0.03) -> tuple[torch.Tensor, str]:
    """Cycle through 4 stimulus types with some noise."""
    types = list(STIMULUS_TYPES.keys())
    # Each type for 50 steps, then switch
    idx = (t // 50) % len(types)
    label = types[idx]
    base = STIMULUS_TYPES[label]
    return (base + torch.randn(4) * noise).abs(), label


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def within_class_similarity(signals: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    """Average cosine similarity of signals within each class."""
    result = {}
    for label, vecs in signals.items():
        if len(vecs) < 2:
            result[label] = 0.0
            continue
        sims = []
        # Sample pairs to avoid O(n^2) for large sets
        n = min(len(vecs), 200)
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
                sims.append(cosine_sim(vecs[i], vecs[j]))
        result[label] = sum(sims) / max(len(sims), 1)
    return result


def between_class_similarity(signals: dict[str, list[torch.Tensor]]) -> float:
    """Average cosine similarity of signals between different classes."""
    labels = list(signals.keys())
    sims = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            vecs_a = signals[labels[i]][-100:]
            vecs_b = signals[labels[j]][-100:]
            for a in vecs_a[:20]:
                for b in vecs_b[:20]:
                    sims.append(cosine_sim(a, b))
    return sum(sims) / max(len(sims), 1)


def decode_accuracy(signals: dict[str, list[torch.Tensor]], n_test: int = 100) -> float:
    """Nearest-centroid classifier: can we decode stimulus type from broadcast?"""
    # Compute centroids from first half
    centroids = {}
    for label, vecs in signals.items():
        half = len(vecs) // 2
        if half < 5:
            return 0.0
        stacked = torch.stack(vecs[:half])
        centroids[label] = stacked.mean(dim=0)

    # Test on second half
    correct = 0
    total = 0
    for label, vecs in signals.items():
        half = len(vecs) // 2
        test_vecs = vecs[half:half + n_test]
        for v in test_vecs:
            best_label = max(
                centroids.keys(),
                key=lambda l: cosine_sim(v, centroids[l])
            )
            if best_label == label:
                correct += 1
            total += 1

    return correct / max(total, 1)


def temporal_stability(signals_by_window: list[dict[str, torch.Tensor]]) -> float:
    """How stable are the centroids across time windows?"""
    if len(signals_by_window) < 2:
        return 0.0
    stabilities = []
    for i in range(1, len(signals_by_window)):
        for label in signals_by_window[i]:
            if label in signals_by_window[i - 1]:
                sim = cosine_sim(signals_by_window[i][label],
                                 signals_by_window[i - 1][label])
                stabilities.append(sim)
    return sum(stabilities) / max(len(stabilities), 1)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(n_steps: int = 20000, window_size: int = 2000):
    print("=" * 70)
    print("  Proto-Language Analysis — GW Broadcast Signal Consistency")
    print("=" * 70)

    org = ConsciousOrganism(obs_dim=4, initial_kernels=2, total_energy=10.0)

    # Collect broadcast signals by stimulus type
    signals: dict[str, list[torch.Tensor]] = defaultdict(list)
    # Track centroids per window for temporal stability
    window_centroids: list[dict[str, torch.Tensor]] = []
    window_signals: dict[str, list[torch.Tensor]] = defaultdict(list)

    start = time.time()

    for t in range(1, n_steps + 1):
        stimulus, label = get_stimulus(t)
        org.step(stimulus)

        broadcast = org.gw.broadcast_signal.clone().detach()
        signals[label].append(broadcast)
        window_signals[label].append(broadcast)

        # Window checkpoint
        if t % window_size == 0:
            centroids = {}
            for lbl, vecs in window_signals.items():
                if vecs:
                    centroids[lbl] = torch.stack(vecs).mean(dim=0)
            window_centroids.append(centroids)
            window_signals = defaultdict(list)

            elapsed = time.time() - start
            print(f"  [{t:6d}] {t/elapsed:.0f} steps/s | "
                  f"signals: {', '.join(f'{l}={len(signals[l])}' for l in sorted(signals))}")

    elapsed = time.time() - start
    print(f"\n  Total: {n_steps} steps in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # 1. Within-class similarity (consistency)
    wc = within_class_similarity(signals)
    avg_wc = sum(wc.values()) / len(wc)
    print(f"\n  1. WITHIN-CLASS SIMILARITY (consistency of signal per stimulus)")
    print(f"     Goal: high = same stimulus -> similar broadcast")
    for label in sorted(wc):
        print(f"       Type {label}: {wc[label]:.3f}")
    print(f"       Average: {avg_wc:.3f}")

    # 2. Between-class similarity (discriminability)
    bc = between_class_similarity(signals)
    print(f"\n  2. BETWEEN-CLASS SIMILARITY (confusion between stimulus types)")
    print(f"     Goal: low = different stimuli -> different broadcasts")
    print(f"       Average: {bc:.3f}")

    # 3. Discrimination index
    disc = avg_wc - bc
    print(f"\n  3. DISCRIMINATION INDEX (within - between)")
    print(f"     > 0 means broadcasts carry stimulus information")
    print(f"       Index: {disc:.3f}")

    # 4. Decode accuracy
    acc = decode_accuracy(signals)
    chance = 1.0 / len(STIMULUS_TYPES)
    print(f"\n  4. DECODE ACCURACY (nearest centroid classifier)")
    print(f"     Chance level: {chance:.1%}")
    print(f"       Accuracy: {acc:.1%}")

    # 5. Temporal stability
    stab = temporal_stability(window_centroids)
    print(f"\n  5. TEMPORAL STABILITY (centroid consistency across windows)")
    print(f"     1.0 = perfectly stable vocabulary")
    print(f"       Stability: {stab:.3f}")

    # --- Verdict ---
    print("\n" + "-" * 70)
    print("  PROTO-LANGUAGE VERDICT")
    print("-" * 70)

    criteria = [
        ("Discrimination > 0", disc > 0),
        ("Decode > chance", acc > chance),
        ("Within-class > 0.5", avg_wc > 0.5),
        ("Stability > 0.5", stab > 0.5),
    ]

    for name, passed in criteria:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    passed_count = sum(1 for _, p in criteria if p)
    print(f"\n  {passed_count}/{len(criteria)} criteria met")

    if passed_count == 4:
        print("\n  >>> PROTO-LANGUAGE DETECTED <<<")
        print("  The organism's GW broadcast spontaneously encodes")
        print("  stimulus type with consistent, stable, discriminable signals.")
    elif passed_count >= 2:
        print("\n  >>> WEAK PROTO-LANGUAGE <<<")
        print("  Some signal structure present but not fully consistent.")
    else:
        print("\n  >>> NO PROTO-LANGUAGE <<<")
        print("  Broadcast signals do not consistently encode stimulus type.")

    # Try plot
    try:
        _save_plot(signals, window_centroids, n_steps)
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")


def _save_plot(signals, window_centroids, n_steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Proto-Language Analysis — GW Broadcast Signals", fontsize=14)

    # 1. PCA of all signals colored by type
    all_vecs = []
    all_labels = []
    colors_map = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange'}
    for label in sorted(signals):
        vecs = signals[label][-500:]  # last 500 per type
        all_vecs.extend([v.numpy() for v in vecs])
        all_labels.extend([label] * len(vecs))

    if len(all_vecs) > 10:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(np.array(all_vecs))
        for label in sorted(signals):
            mask = [l == label for l in all_labels]
            pts = coords[mask]
            axes[0].scatter(pts[:, 0], pts[:, 1], c=colors_map[label],
                          alpha=0.3, s=10, label=f"Type {label}")
        axes[0].legend()
        axes[0].set_title("Broadcast Signals (PCA)")

    # 2. Within vs between class similarity over time
    windows = range(len(window_centroids))
    if len(window_centroids) >= 2:
        wc_per_window = []
        bc_per_window = []
        for i, centroids in enumerate(window_centroids):
            labels = list(centroids.keys())
            if len(labels) < 2:
                continue
            # Within: similarity of centroid to prior window
            if i > 0:
                wc_sims = []
                bc_sims = []
                for l1 in labels:
                    if l1 in window_centroids[i-1]:
                        wc_sims.append(cosine_sim(centroids[l1], window_centroids[i-1][l1]))
                    for l2 in labels:
                        if l1 != l2:
                            bc_sims.append(cosine_sim(centroids[l1], centroids[l2]))
                if wc_sims:
                    wc_per_window.append(sum(wc_sims)/len(wc_sims))
                if bc_sims:
                    bc_per_window.append(sum(bc_sims)/len(bc_sims))

        if wc_per_window:
            axes[1].plot(wc_per_window, label='Within-class (stability)', color='green')
        if bc_per_window:
            axes[1].plot(bc_per_window, label='Between-class (confusion)', color='red')
        axes[1].legend()
        axes[1].set_title("Signal Structure Over Time")
        axes[1].set_xlabel("Window")

    # 3. Centroid evolution
    if len(window_centroids) >= 2:
        for label in sorted(STIMULUS_TYPES):
            norms = []
            for c in window_centroids:
                if label in c:
                    norms.append(c[label].norm().item())
            if norms:
                axes[2].plot(norms, label=f"Type {label}", color=colors_map[label])
        axes[2].legend()
        axes[2].set_title("Centroid Norm Over Time")
        axes[2].set_xlabel("Window")

    plt.tight_layout()
    out = Path("results") / "proto_language.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--window", type=int, default=2000)
    args = parser.parse_args()
    main(n_steps=args.steps, window_size=args.window)
