"""
Emergent Grammar Analysis — Temporal structure in proto-language broadcasts
==========================================================================

Does the ConsciousOrganism develop grammar-like temporal structure?
We measure whether broadcast sequences show:
  1. Predictable bigram transitions (not random)
  2. Low transition entropy (vs high for random input)
  3. Position encoding within stimulus cycles
  4. Mutual information at multiple time lags (anticipation)
  5. Structural difference between organized vs random input

Two conditions:
  - Structured: A→B→C→D repeating cycle (50 steps each, cycle=200)
  - Random: uniform random selection from {A,B,C,D}

Two systems:
  - Organism (multi-agent with GW)
  - Individual kernel (single agent, no GW)
"""

import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zeta_life.kernel import ConsciousKernel, ConsciousOrganism


# ---------------------------------------------------------------------------
# Stimulus definitions
# ---------------------------------------------------------------------------

STIMULUS_TYPES = {
    'A': torch.tensor([0.7, 0.1, 0.1, 0.1]),
    'B': torch.tensor([0.1, 0.7, 0.1, 0.1]),
    'C': torch.tensor([0.1, 0.1, 0.7, 0.1]),
    'D': torch.tensor([0.1, 0.1, 0.1, 0.7]),
}
LABELS = ['A', 'B', 'C', 'D']
CYCLE_LEN = 200  # 50 steps per stimulus type


def structured_stimulus(t: int, noise: float = 0.03) -> tuple[torch.Tensor, str]:
    """A→B→C→D repeating cycle, 50 steps each."""
    idx = (t // 50) % 4
    label = LABELS[idx]
    base = STIMULUS_TYPES[label]
    return (base + torch.randn(4) * noise).abs(), label


def random_stimulus(noise: float = 0.03) -> tuple[torch.Tensor, str]:
    """Uniformly random stimulus from {A,B,C,D}."""
    idx = torch.randint(0, 4, (1,)).item()
    label = LABELS[idx]
    base = STIMULUS_TYPES[label]
    return (base + torch.randn(4) * noise).abs(), label


def discretize_broadcast(broadcast: torch.Tensor) -> int:
    """Map broadcast to nearest stimulus centroid (argmax)."""
    sims = []
    bc = broadcast[:4]  # obs_dim dimensions
    for label in LABELS:
        centroid = F.softmax(STIMULUS_TYPES[label], dim=-1)
        sim = F.cosine_similarity(bc.unsqueeze(0), centroid.unsqueeze(0)).item()
        sims.append(sim)
    return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Run condition
# ---------------------------------------------------------------------------

def run_condition(
    system, n_steps: int, is_organism: bool, structured: bool,
) -> list[int]:
    """Run system for n_steps, return sequence of discretized broadcast symbols."""
    symbols = []
    for t in range(1, n_steps + 1):
        if structured:
            stimulus, _ = structured_stimulus(t)
        else:
            stimulus, _ = random_stimulus()

        system.step(stimulus)

        if is_organism:
            bc = system.gw.broadcast_signal.clone().detach()
        else:
            bc = system.last_action.clone().detach()

        symbols.append(discretize_broadcast(bc))

        if t % 5000 == 0:
            print(f"      [{t:6d}/{n_steps}]")

    return symbols


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bigram_transition_matrix(symbols: list[int], n_symbols: int = 4) -> np.ndarray:
    """Build row-normalized bigram transition matrix."""
    counts = np.zeros((n_symbols, n_symbols))
    for i in range(len(symbols) - 1):
        counts[symbols[i]][symbols[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def transition_entropy(trans_matrix: np.ndarray) -> float:
    """Average Shannon entropy per row of transition matrix."""
    entropies = []
    for row in trans_matrix:
        row = row[row > 0]
        if len(row) > 0:
            h = -np.sum(row * np.log2(row + 1e-12))
            entropies.append(h)
    return float(np.mean(entropies)) if entropies else 0.0


def bigram_predictability(trans_matrix: np.ndarray) -> float:
    """1 - H_norm(P): higher = more predictable transitions."""
    h = transition_entropy(trans_matrix)
    h_max = np.log2(trans_matrix.shape[1])
    if h_max == 0:
        return 0.0
    return 1.0 - h / h_max


def position_decode_accuracy(symbols: list[int], cycle_len: int = CYCLE_LEN) -> float:
    """Can we decode position-in-cycle from broadcast symbol?

    For structured input, broadcast at position p should map to specific symbol.
    >25% = encodes position (chance = 25% for 4 classes).
    """
    correct = 0
    total = 0
    for t, sym in enumerate(symbols):
        pos_in_cycle = (t % cycle_len) // 50  # Expected symbol index (0-3)
        if sym == pos_in_cycle:
            correct += 1
        total += 1
    return correct / max(total, 1)


def mutual_information(symbols: list[int], stimulus_seq: list[int], lag: int) -> float:
    """MI(broadcast_t, stimulus_{t+lag}) for anticipation measurement."""
    n = len(symbols) - lag
    if n <= 0:
        return 0.0

    n_sym = 4
    joint = np.zeros((n_sym, n_sym))
    for t in range(n):
        joint[symbols[t]][stimulus_seq[t + lag]] += 1

    joint /= joint.sum() + 1e-12
    p_bc = joint.sum(axis=1)
    p_stim = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_sym):
        for j in range(n_sym):
            if joint[i, j] > 1e-12 and p_bc[i] > 1e-12 and p_stim[j] > 1e-12:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_bc[i] * p_stim[j]))
    return float(mi)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(results: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Emergent Grammar Analysis — Proto-Language Temporal Structure",
                 fontsize=14, y=0.98)

    # --- Panel 1: Transition heatmaps ---
    ax = axes[0, 0]
    # Show organism structured transition matrix
    tm = results['org_struct_tm']
    im = ax.imshow(tm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{tm[i, j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title(f"Organism Structured Transitions\n"
                 f"H={results['org_struct_entropy']:.3f}, "
                 f"Pred={results['org_struct_bigram']:.3f}")
    ax.set_xlabel("Next symbol")
    ax.set_ylabel("Current symbol")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- Panel 2: MI vs lag ---
    ax = axes[0, 1]
    lags = results['lags']
    for key, label, color, marker in [
        ('org_struct_mi', 'Org Structured', '#3498db', 'o'),
        ('org_rand_mi', 'Org Random', '#3498db', 's'),
        ('ind_struct_mi', 'Ind Structured', '#e67e22', 'o'),
        ('ind_rand_mi', 'Ind Random', '#e67e22', 's'),
    ]:
        vals = results[key]
        ls = '-' if 'Structured' in label else '--'
        ax.plot(lags, vals, color=color, marker=marker, linestyle=ls,
                label=label, markersize=5)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("Mutual Information (bits)")
    ax.set_title("Broadcast-Stimulus MI vs Lag\n(Anticipation)")
    ax.legend(fontsize=8)
    ax.set_yscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Position decode accuracy ---
    ax = axes[1, 0]
    systems = ['Org\nStructured', 'Org\nRandom', 'Ind\nStructured', 'Ind\nRandom']
    accs = [
        results['org_struct_pos'],
        results['org_rand_pos'],
        results['ind_struct_pos'],
        results['ind_rand_pos'],
    ]
    colors = ['#3498db', '#85c1e9', '#e67e22', '#f0b27a']
    bars = ax.bar(systems, accs, color=colors)
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Chance (25%)')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Position Decode Accuracy\n(>25% = encodes cycle position)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # --- Panel 4: Summary bigram predictability bars ---
    ax = axes[1, 1]
    metrics = ['Bigram\nPred', 'Trans\nEntropy', 'Entropy\nRatio']
    org_vals = [
        results['org_struct_bigram'],
        results['org_struct_entropy'],
        results['org_entropy_ratio'],
    ]
    ind_vals = [
        results['ind_struct_bigram'],
        results['ind_struct_entropy'],
        results['ind_entropy_ratio'],
    ]
    x = np.arange(len(metrics))
    width = 0.3
    bars1 = ax.bar(x - width/2, org_vals, width, label='Organism', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, ind_vals, width, label='Individual', color='#e67e22', alpha=0.8)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Grammar Metrics (Structured Input)\nEntropy ratio < 1.0 = input structure reflected")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path("results") / "grammar.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {out}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(n_steps: int = 20000):
    lags = [1, 2, 5, 10, 20, 50]
    results: dict = {'lags': lags}

    print("=" * 70)
    print("  Emergent Grammar Analysis — Proto-Language Temporal Structure")
    print("=" * 70)
    print(f"  Steps per condition: {n_steps}")
    print(f"  Cycle length: {CYCLE_LEN} (50 steps per symbol)")
    print()

    configs = [
        ('org', 'Organism', True),
        ('ind', 'Individual', False),
    ]

    for sys_key, sys_name, is_organism in configs:
        for cond_key, cond_name, structured in [('struct', 'Structured', True),
                                                  ('rand', 'Random', False)]:
            prefix = f"{sys_key}_{cond_key}"
            print(f"\n  {'-' * 60}")
            print(f"  {sys_name} — {cond_name}")
            print(f"  {'-' * 60}")

            if is_organism:
                system = ConsciousOrganism(obs_dim=4, initial_kernels=2, total_energy=10.0)
            else:
                system = ConsciousKernel(obs_dim=4)

            start = time.time()
            symbols = run_condition(system, n_steps, is_organism, structured)
            elapsed = time.time() - start
            print(f"    Done in {elapsed:.1f}s")

            # Build stimulus sequence for MI computation
            stim_seq = []
            for t in range(1, n_steps + 1):
                if structured:
                    idx = (t // 50) % 4
                else:
                    idx = torch.randint(0, 4, (1,)).item()
                stim_seq.append(idx)

            # Metrics
            tm = bigram_transition_matrix(symbols)
            results[f'{prefix}_tm'] = tm
            results[f'{prefix}_entropy'] = transition_entropy(tm)
            results[f'{prefix}_bigram'] = bigram_predictability(tm)
            results[f'{prefix}_pos'] = position_decode_accuracy(symbols)

            # MI at multiple lags
            mi_vals = []
            for lag in lags:
                mi = mutual_information(symbols, stim_seq, lag)
                mi_vals.append(mi)
            results[f'{prefix}_mi'] = mi_vals

            print(f"    Transition entropy: {results[f'{prefix}_entropy']:.4f}")
            print(f"    Bigram predictability: {results[f'{prefix}_bigram']:.4f}")
            print(f"    Position decode: {results[f'{prefix}_pos']:.1%}")
            print(f"    MI lags: {', '.join(f'{v:.4f}' for v in mi_vals)}")

    # Entropy ratio: structured/random (< 1.0 = structure reflected)
    for sys_key in ['org', 'ind']:
        s_ent = results[f'{sys_key}_struct_entropy']
        r_ent = results[f'{sys_key}_rand_entropy']
        results[f'{sys_key}_entropy_ratio'] = s_ent / max(r_ent, 1e-6)

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  Metric                  Org-Struct  Org-Rand  Ind-Struct  Ind-Rand")
    print(f"  {'-' * 66}")
    print(f"  Bigram predictability   "
          f"{results['org_struct_bigram']:10.4f}"
          f"{results['org_rand_bigram']:10.4f}"
          f"{results['ind_struct_bigram']:12.4f}"
          f"{results['ind_rand_bigram']:10.4f}")
    print(f"  Transition entropy      "
          f"{results['org_struct_entropy']:10.4f}"
          f"{results['org_rand_entropy']:10.4f}"
          f"{results['ind_struct_entropy']:12.4f}"
          f"{results['ind_rand_entropy']:10.4f}")
    print(f"  Position decode         "
          f"{results['org_struct_pos']:10.1%}"
          f"{results['org_rand_pos']:10.1%}"
          f"{results['ind_struct_pos']:12.1%}"
          f"{results['ind_rand_pos']:10.1%}")

    print(f"\n  Entropy ratio (struct/rand):")
    print(f"    Organism:   {results['org_entropy_ratio']:.4f}")
    print(f"    Individual: {results['ind_entropy_ratio']:.4f}")

    # Criteria
    print(f"\n  {'-' * 60}")
    print(f"  GRAMMAR CRITERIA")
    print(f"  {'-' * 60}")

    criteria = [
        ("struct_entropy < rand_entropy (Org)",
         results['org_struct_entropy'] < results['org_rand_entropy'],
         f"{results['org_struct_entropy']:.4f} < {results['org_rand_entropy']:.4f}"),
        ("position_decode > 25% (Org-Struct)",
         results['org_struct_pos'] > 0.25,
         f"{results['org_struct_pos']:.1%}"),
        ("bigram_pred > 0.2 (Org-Struct)",
         results['org_struct_bigram'] > 0.2,
         f"{results['org_struct_bigram']:.4f}"),
        ("MI_lag1 > 0.01 (Org-Struct)",
         results['org_struct_mi'][0] > 0.01,
         f"{results['org_struct_mi'][0]:.4f}"),
        ("Org > Ind (bigram pred, structured)",
         results['org_struct_bigram'] > results['ind_struct_bigram'],
         f"{results['org_struct_bigram']:.4f} vs {results['ind_struct_bigram']:.4f}"),
    ]

    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} ({value})")

    passed_count = sum(1 for _, p, _ in criteria if p)
    print(f"\n  {passed_count}/{len(criteria)} criteria met")

    # Plot
    try:
        save_plot(results)
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergent grammar analysis")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Steps per condition")
    args = parser.parse_args()
    main(n_steps=args.steps)
