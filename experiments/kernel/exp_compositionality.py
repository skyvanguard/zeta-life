"""
Compositionality Analysis — Does the proto-language support composition?
========================================================================

The ConsciousOrganism's GW broadcast encodes stimulus type with 100% decode
accuracy and perfect stability (proto-language). But is this mapping merely a
deterministic pass-through (softmax), or does the multi-agent architecture
introduce emergent compositional structure?

If we know the broadcast signals for stimuli A and B individually, can we
PREDICT the signal for the mixture A+B? Factors that could create non-trivial
compositionality:
  1. Top-down modulation: combined = (1-alpha)*stimulus + alpha*broadcast
  2. Winner selection: different kernels win for different stimuli
  3. Anti-monopoly: penalizes consecutive winners
  4. World model context: latent state depends on history

Measures:
  1. Linearity Index: f(aA+bB) ~ a*f(A)+b*f(B)?
  2. Systematicity Index: distance preservation (Spearman correlation)
  3. Context Effect: does previous stimulus affect current broadcast?
  4. Organism vs Individual: does multi-agent architecture add compositionality?

A positive result means the organism spontaneously develops compositional
structure — a hallmark of true language.
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

PAIRS = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]
RATIOS = [0.3, 0.5, 0.7]


def make_mixture(label1: str, label2: str, ratio: float, noise: float = 0.03) -> torch.Tensor:
    """Create mixture stimulus: ratio*A + (1-ratio)*B with noise."""
    s = ratio * STIMULUS_TYPES[label1] + (1 - ratio) * STIMULUS_TYPES[label2]
    return (s + torch.randn(4) * noise).abs()


def get_stimulus(t: int, noise: float = 0.03) -> tuple[torch.Tensor, str]:
    """Cycle through individual stimulus types (50 steps each)."""
    types = list(STIMULUS_TYPES.keys())
    idx = (t // 50) % len(types)
    label = types[idx]
    base = STIMULUS_TYPES[label]
    return (base + torch.randn(4) * noise).abs(), label


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Phase 1: Learn vocabulary
# ---------------------------------------------------------------------------

def phase1_vocabulary(system, n_steps: int, is_organism: bool) -> dict[str, torch.Tensor]:
    """Present individual stimuli and collect broadcast centroids."""
    signals: dict[str, list[torch.Tensor]] = defaultdict(list)

    for t in range(1, n_steps + 1):
        stimulus, label = get_stimulus(t)
        system.step(stimulus)

        if is_organism:
            broadcast = system.gw.broadcast_signal.clone().detach()
        else:
            broadcast = system.last_action.clone().detach()

        signals[label].append(broadcast)

        if t % 2000 == 0:
            print(f"    [{t:6d}/{n_steps}] "
                  f"{', '.join(f'{l}={len(signals[l])}' for l in sorted(signals))}")

    # Compute centroids from second half (after learning)
    centroids = {}
    for label, vecs in signals.items():
        half = len(vecs) // 2
        centroids[label] = torch.stack(vecs[half:]).mean(dim=0)

    return centroids


# ---------------------------------------------------------------------------
# Phase 2: Test combinations
# ---------------------------------------------------------------------------

def phase2_combinations(
    system, centroids: dict[str, torch.Tensor], n_steps: int, is_organism: bool
) -> dict[str, list[torch.Tensor]]:
    """Present mixtures and collect broadcast signals."""
    combo_signals: dict[str, list[torch.Tensor]] = defaultdict(list)

    # Build schedule: 6 pairs x 3 ratios = 18 combos
    combos = []
    for l1, l2 in PAIRS:
        for ratio in RATIOS:
            key = f"{l1}{l2}_{ratio:.1f}"
            combos.append((l1, l2, ratio, key))

    steps_per_combo = n_steps // len(combos)

    for combo_idx, (l1, l2, ratio, key) in enumerate(combos):
        for s in range(steps_per_combo):
            stimulus = make_mixture(l1, l2, ratio)
            system.step(stimulus)

            if is_organism:
                broadcast = system.gw.broadcast_signal.clone().detach()
            else:
                broadcast = system.last_action.clone().detach()

            combo_signals[key].append(broadcast)

        if (combo_idx + 1) % 6 == 0:
            done = (combo_idx + 1) * steps_per_combo
            print(f"    [{done:6d}/{n_steps}] {combo_idx + 1}/{len(combos)} combos done")

    return combo_signals


# ---------------------------------------------------------------------------
# Phase 2b: Context effect measurement
# ---------------------------------------------------------------------------

def measure_context_effect(
    system, n_trials: int, is_organism: bool
) -> float:
    """Measure how previous stimulus context affects broadcast for same stimulus.

    For stimulus X, compare broadcast when X follows A vs when X follows B.
    Higher score = more context sensitivity.
    """
    types = list(STIMULUS_TYPES.keys())
    context_broadcasts: dict[str, dict[str, list[torch.Tensor]]] = {
        target: defaultdict(list) for target in types
    }

    for trial in range(n_trials):
        for target in types:
            for context in types:
                if context == target:
                    continue
                # Present context stimulus for 10 steps
                for _ in range(10):
                    system.step(STIMULUS_TYPES[context] + torch.randn(4) * 0.03)
                # Present target stimulus and capture broadcast
                system.step(STIMULUS_TYPES[target])
                if is_organism:
                    bc = system.gw.broadcast_signal.clone().detach()
                else:
                    bc = system.last_action.clone().detach()
                context_broadcasts[target][context].append(bc)

    # For each target, compute variance of broadcast across different contexts
    context_effects = []
    for target in types:
        context_centroids = []
        for context, vecs in context_broadcasts[target].items():
            if vecs:
                context_centroids.append(torch.stack(vecs).mean(dim=0))
        if len(context_centroids) >= 2:
            stacked = torch.stack(context_centroids)
            # Variance across contexts (higher = more context-dependent)
            variance = stacked.var(dim=0).mean().item()
            context_effects.append(variance)

    return sum(context_effects) / max(len(context_effects), 1)


# ---------------------------------------------------------------------------
# Phase 3: Analysis metrics
# ---------------------------------------------------------------------------

def compute_linearity(
    centroids: dict[str, torch.Tensor],
    combo_signals: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    """Linearity Index: f(aA+bB) ~ a*f(A) + b*f(B)?

    Returns per-combo and average linearity scores.
    """
    scores = {}
    for key, vecs in combo_signals.items():
        # Parse key: "AB_0.5"
        parts = key.split('_')
        l1, l2 = parts[0][0], parts[0][1]
        ratio = float(parts[1])

        # Linear prediction from individual centroids
        predicted = ratio * centroids[l1] + (1 - ratio) * centroids[l2]

        # Actual broadcast centroid (second half for stability)
        half = len(vecs) // 2
        actual = torch.stack(vecs[half:]).mean(dim=0)

        scores[key] = cosine_sim(predicted, actual)

    return scores


def compute_systematicity(
    centroids: dict[str, torch.Tensor],
    combo_signals: dict[str, list[torch.Tensor]],
) -> float:
    """Systematicity Index: do distances in stimulus space map to distances in broadcast space?

    Spearman correlation between stimulus-space distances and broadcast-space distances.
    """
    keys = list(combo_signals.keys())
    if len(keys) < 3:
        return 0.0

    # Compute centroid for each combo
    combo_centroids = {}
    for key, vecs in combo_signals.items():
        half = len(vecs) // 2
        combo_centroids[key] = torch.stack(vecs[half:]).mean(dim=0)

    # Also include individual stimuli
    all_keys = list(STIMULUS_TYPES.keys()) + keys
    all_stim = {}
    all_bc = {}

    for label, stim in STIMULUS_TYPES.items():
        all_stim[label] = stim
        all_bc[label] = centroids[label]

    for key in keys:
        parts = key.split('_')
        l1, l2 = parts[0][0], parts[0][1]
        ratio = float(parts[1])
        all_stim[key] = ratio * STIMULUS_TYPES[l1] + (1 - ratio) * STIMULUS_TYPES[l2]
        all_bc[key] = combo_centroids[key]

    # Pairwise distances
    ordered = list(all_stim.keys())
    stim_dists = []
    bc_dists = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            sd = (all_stim[ordered[i]] - all_stim[ordered[j]]).norm().item()
            bd = (all_bc[ordered[i]] - all_bc[ordered[j]]).norm().item()
            stim_dists.append(sd)
            bc_dists.append(bd)

    # Spearman rank correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(stim_dists, bc_dists)
    return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(
    org_centroids: dict[str, torch.Tensor],
    org_combo_signals: dict[str, list[torch.Tensor]],
    org_linearity: dict[str, float],
    org_systematicity: float,
    ind_centroids: dict[str, torch.Tensor],
    ind_combo_signals: dict[str, list[torch.Tensor]],
    ind_linearity: dict[str, float],
    ind_systematicity: float,
    org_context: float,
    ind_context: float,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Compositionality Analysis — Proto-Language Structure", fontsize=14, y=0.98)

    colors_base = {'A': '#e74c3c', 'B': '#3498db', 'C': '#2ecc71', 'D': '#f39c12'}

    # --- Panel 1: PCA of all broadcasts (individual + mixtures) ---
    ax = axes[0, 0]
    all_vecs = []
    all_labels = []
    all_types = []  # 'individual' or 'mixture'

    # Individual centroids
    for label in sorted(org_centroids):
        all_vecs.append(org_centroids[label].numpy())
        all_labels.append(label)
        all_types.append('individual')

    # Mixture signals (sample)
    for key, vecs in org_combo_signals.items():
        sample = vecs[-min(50, len(vecs)):]
        for v in sample:
            all_vecs.append(v.numpy())
            all_labels.append(key)
            all_types.append('mixture')

    if len(all_vecs) > 5:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(np.array(all_vecs))

        # Plot mixtures first (lighter)
        for i, (label, tp) in enumerate(zip(all_labels, all_types)):
            if tp == 'mixture':
                l1 = label.split('_')[0][0]
                ax.scatter(coords[i, 0], coords[i, 1],
                          c=colors_base.get(l1, 'gray'), alpha=0.15, s=8)

        # Plot individual centroids (bold)
        for i, (label, tp) in enumerate(zip(all_labels, all_types)):
            if tp == 'individual':
                ax.scatter(coords[i, 0], coords[i, 1],
                          c=colors_base[label], s=200, marker='*',
                          edgecolors='black', linewidths=1, zorder=5,
                          label=f'Type {label}')

        ax.legend(fontsize=8)
        ax.set_title("Broadcast Space (PCA)\nStars=individual, dots=mixtures")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")

    # --- Panel 2: Linearity scores per combination and ratio ---
    ax = axes[0, 1]
    pair_labels = []
    org_scores = []
    ind_scores = []

    for key in sorted(org_linearity.keys()):
        pair_labels.append(key)
        org_scores.append(org_linearity[key])
        ind_scores.append(ind_linearity.get(key, 0))

    x = np.arange(len(pair_labels))
    width = 0.35
    ax.bar(x - width/2, org_scores, width, label='Organism', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, ind_scores, width, label='Individual', color='#e67e22', alpha=0.8)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Threshold (0.8)')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Linearity Index per Combination")
    ax.legend(fontsize=8)

    # --- Panel 3: Systematicity scatter ---
    ax = axes[1, 0]
    for system_name, centroids, combo_signals, color, marker in [
        ('Organism', org_centroids, org_combo_signals, '#3498db', 'o'),
        ('Individual', ind_centroids, ind_combo_signals, '#e67e22', 's'),
    ]:
        all_keys = list(STIMULUS_TYPES.keys()) + list(combo_signals.keys())
        all_stim = {}
        all_bc = {}
        for label, stim in STIMULUS_TYPES.items():
            all_stim[label] = stim
            all_bc[label] = centroids[label]
        for key, vecs in combo_signals.items():
            parts = key.split('_')
            l1, l2 = parts[0][0], parts[0][1]
            ratio = float(parts[1])
            all_stim[key] = ratio * STIMULUS_TYPES[l1] + (1 - ratio) * STIMULUS_TYPES[l2]
            half = len(vecs) // 2
            all_bc[key] = torch.stack(vecs[half:]).mean(dim=0)

        ordered = list(all_stim.keys())
        stim_dists = []
        bc_dists = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                sd = (all_stim[ordered[i]] - all_stim[ordered[j]]).norm().item()
                bd = (all_bc[ordered[i]] - all_bc[ordered[j]]).norm().item()
                stim_dists.append(sd)
                bc_dists.append(bd)

        ax.scatter(stim_dists, bc_dists, c=color, marker=marker, alpha=0.4, s=20,
                  label=system_name)

    ax.set_xlabel("Stimulus Distance")
    ax.set_ylabel("Broadcast Distance")
    syst_org = org_systematicity
    syst_ind = ind_systematicity
    ax.set_title(f"Systematicity (Spearman)\nOrg={syst_org:.3f}, Ind={syst_ind:.3f}")
    ax.legend(fontsize=8)

    # --- Panel 4: Summary comparison bars ---
    ax = axes[1, 1]
    metrics = ['Linearity\n(avg)', 'Systematicity', 'Context\nEffect']
    org_vals = [
        np.mean(list(org_linearity.values())),
        org_systematicity,
        org_context,
    ]
    ind_vals = [
        np.mean(list(ind_linearity.values())),
        ind_systematicity,
        ind_context,
    ]

    x = np.arange(len(metrics))
    width = 0.3
    bars1 = ax.bar(x - width/2, org_vals, width, label='Organism', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, ind_vals, width, label='Individual', color='#e67e22', alpha=0.8)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                   f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Organism vs Individual Kernel")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path("results") / "compositionality.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {out}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(n_steps: int = 16000):
    phase1_steps = n_steps // 2
    phase2_steps = n_steps // 2
    context_trials = 10

    print("=" * 70)
    print("  Compositionality Analysis — Proto-Language Structure")
    print("=" * 70)
    print(f"  Phase 1: {phase1_steps} steps (vocabulary learning)")
    print(f"  Phase 2: {phase2_steps} steps (combination testing)")
    print(f"  Context trials: {context_trials}")
    print()

    # ===================================================================
    # ORGANISM
    # ===================================================================
    print("-" * 70)
    print("  ORGANISM (multi-agent with Global Workspace)")
    print("-" * 70)

    org = ConsciousOrganism(obs_dim=4, initial_kernels=2, total_energy=10.0)
    start = time.time()

    print("\n  Phase 1: Learning vocabulary...")
    org_centroids = phase1_vocabulary(org, phase1_steps, is_organism=True)

    print("\n  Phase 2: Testing combinations...")
    org_combo = phase2_combinations(org, org_centroids, phase2_steps, is_organism=True)

    print("\n  Phase 2b: Measuring context effect...")
    org_context = measure_context_effect(org, context_trials, is_organism=True)

    org_time = time.time() - start
    print(f"\n  Organism done in {org_time:.1f}s")

    # ===================================================================
    # INDIVIDUAL KERNEL (control)
    # ===================================================================
    print("\n" + "-" * 70)
    print("  INDIVIDUAL KERNEL (single agent, no GW)")
    print("-" * 70)

    kernel = ConsciousKernel(obs_dim=4)
    start = time.time()

    print("\n  Phase 1: Learning vocabulary...")
    ind_centroids = phase1_vocabulary(kernel, phase1_steps, is_organism=False)

    print("\n  Phase 2: Testing combinations...")
    ind_combo = phase2_combinations(kernel, ind_centroids, phase2_steps, is_organism=False)

    print("\n  Phase 2b: Measuring context effect...")
    ind_context = measure_context_effect(kernel, context_trials, is_organism=False)

    ind_time = time.time() - start
    print(f"\n  Individual done in {ind_time:.1f}s")

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # 1. Linearity
    org_linearity = compute_linearity(org_centroids, org_combo)
    ind_linearity = compute_linearity(ind_centroids, ind_combo)

    avg_org_lin = np.mean(list(org_linearity.values()))
    avg_ind_lin = np.mean(list(ind_linearity.values()))

    print(f"\n  1. LINEARITY INDEX: f(aA+bB) ~ a*f(A) + b*f(B)?")
    print(f"     1.0 = perfectly linear (predictable from components)")
    print(f"     Organism avg:   {avg_org_lin:.4f}")
    print(f"     Individual avg: {avg_ind_lin:.4f}")
    print(f"     Per-combo (organism):")
    for key in sorted(org_linearity):
        org_v = org_linearity[key]
        ind_v = ind_linearity.get(key, 0)
        diff = org_v - ind_v
        print(f"       {key:10s}: org={org_v:.4f}  ind={ind_v:.4f}  diff={diff:+.4f}")

    # 2. Systematicity
    org_syst = compute_systematicity(org_centroids, org_combo)
    ind_syst = compute_systematicity(ind_centroids, ind_combo)

    print(f"\n  2. SYSTEMATICITY INDEX: distance preservation (Spearman rho)")
    print(f"     1.0 = perfect isomorphism between stimulus and broadcast spaces")
    print(f"     Organism:   {org_syst:.4f}")
    print(f"     Individual: {ind_syst:.4f}")

    # 3. Context effect
    print(f"\n  3. CONTEXT EFFECT: influence of previous stimulus on broadcast")
    print(f"     Higher = more context-dependent (compositional context)")
    print(f"     Organism:   {org_context:.6f}")
    print(f"     Individual: {ind_context:.6f}")
    if ind_context > 0:
        ratio = org_context / ind_context
        print(f"     Ratio (org/ind): {ratio:.2f}x")

    # --- Verdict ---
    print("\n" + "-" * 70)
    print("  COMPOSITIONALITY VERDICT")
    print("-" * 70)

    criteria = [
        ("Linearity > 0.8", avg_org_lin > 0.8, f"{avg_org_lin:.4f}"),
        ("Systematicity > 0.7", org_syst > 0.7, f"{org_syst:.4f}"),
        ("Org linearity > Ind linearity", avg_org_lin > avg_ind_lin,
         f"{avg_org_lin:.4f} vs {avg_ind_lin:.4f}"),
        ("Org systematicity > Ind systematicity", org_syst > ind_syst,
         f"{org_syst:.4f} vs {ind_syst:.4f}"),
    ]

    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} ({value})")

    passed_count = sum(1 for _, p, _ in criteria if p)
    print(f"\n  {passed_count}/{len(criteria)} criteria met")

    if passed_count >= 3:
        print("\n  >>> COMPOSITIONAL PROTO-LANGUAGE <<<")
        print("  The organism's broadcast signals exhibit compositional structure:")
        print("  mixtures are predictable from individual components, and the")
        print("  multi-agent architecture adds emergent compositionality.")
    elif passed_count >= 2:
        print("\n  >>> PARTIAL COMPOSITIONALITY <<<")
        print("  Some compositional structure present, but not fully emergent.")
    else:
        print("\n  >>> LIMITED COMPOSITIONALITY <<<")
        print("  The broadcast mapping is largely non-compositional or the")
        print("  multi-agent architecture does not add compositional structure.")

    # --- Plot ---
    try:
        save_plot(
            org_centroids, org_combo, org_linearity, org_syst,
            ind_centroids, ind_combo, ind_linearity, ind_syst,
            org_context, ind_context,
        )
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compositionality analysis of proto-language")
    parser.add_argument("--steps", type=int, default=16000,
                       help="Total steps (split 50/50 between phases)")
    args = parser.parse_args()
    main(n_steps=args.steps)
