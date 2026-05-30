"""
Referential Grounding — Closing the causal loop
================================================

Does proto-language become more meaningful when it has causal consequences?
We test whether broadcasts improve when the environment reacts to them.

Reactive Environment:
  state_{t+1} = (1-r)*state_t + r*action_t + noise
  obs_{t+1} = normalize(state_{t+1})

3 conditions:
  1. Grounded:      env.step(broadcast) — causal feedback
  2. Ungrounded:    env.step(random) — no feedback (control)
  3. Goal-directed: grounded + target signal mixed into observation

3 core criteria + 1 informational metric:
  1. Free energy convergence: FE grounded < FE ungrounded
  2. Goal convergence: cosine(env.state, target) > 0.5
  3. FE decrease rate: slope of FE in grounded more negative
  (info) Broadcast entropy: expected lower in grounded ("more focused"), but
         inconclusive while agency is nominal (latent_weight=0) — the broadcast
         is near-uniform in both conditions, so entropy saturates at ~log2(4).
         Reported for tracking; does not gate the experiment.
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zeta_life.kernel import ConsciousKernel, ConsciousOrganism


# ---------------------------------------------------------------------------
# Reactive Environment
# ---------------------------------------------------------------------------

class ReactiveEnvironment:
    """Simple reactive environment: state responds to actions."""

    def __init__(self, obs_dim: int = 4, reactivity: float = 0.2, noise: float = 0.03):
        self.obs_dim = obs_dim
        self.reactivity = reactivity
        self.noise = noise
        self.state = torch.ones(obs_dim) / obs_dim

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Update state based on action and return observation."""
        self.state = (
            (1 - self.reactivity) * self.state
            + self.reactivity * action.detach()
        )
        obs = (self.state + torch.randn(self.obs_dim) * self.noise).abs()
        obs = obs / (obs.sum() + 1e-8)
        return obs

    def reset(self):
        self.state = torch.ones(self.obs_dim) / self.obs_dim


# ---------------------------------------------------------------------------
# Run condition
# ---------------------------------------------------------------------------

def run_condition(
    n_steps: int,
    condition: str,
    is_organism: bool,
    target: torch.Tensor | None = None,
) -> dict:
    """Run one experimental condition.

    Returns dict with time series of FE, broadcast entropy, goal similarity.
    """
    if is_organism:
        system = ConsciousOrganism(obs_dim=4, initial_kernels=2, total_energy=10.0)
    else:
        system = ConsciousKernel(obs_dim=4)

    env = ReactiveEnvironment(obs_dim=4, reactivity=0.2, noise=0.03)

    fe_series = []
    entropy_series = []
    goal_sim_series = []
    env_states = []

    obs = env.state.clone()

    for t in range(1, n_steps + 1):
        # Mix observation with target if goal-directed
        if condition == 'goal' and target is not None:
            stimulus = 0.7 * obs + 0.3 * target
        else:
            stimulus = obs

        result = system.step(stimulus)

        # Get broadcast
        if is_organism:
            broadcast = system.gw.broadcast_signal[:4].clone().detach()
            fe = sum(result.free_energies.values()) / max(len(result.free_energies), 1)
        else:
            broadcast = system.last_action.clone().detach()
            fe = result.free_energy

        # Environment step based on condition
        if condition == 'grounded' or condition == 'goal':
            obs = env.step(broadcast)
        elif condition == 'ungrounded':
            obs = env.step(torch.randn(4).abs())
        else:
            obs = env.step(broadcast)

        # Record metrics
        fe_series.append(fe)

        # Broadcast entropy — normalise by L1, NOT softmax. The broadcast is
        # already a near-distribution (last_action = softmax(stimulus)), so a
        # second softmax flattens it toward uniform and saturates entropy at
        # log2(4)=2.0 (which made grounded vs ungrounded indistinguishable). L1
        # normalisation preserves the broadcast's real concentration.
        bc = broadcast.abs()
        bc_prob = bc / (bc.sum() + 1e-8)
        entropy = -(bc_prob * torch.log2(bc_prob + 1e-8)).sum().item()
        entropy_series.append(entropy)

        # Goal similarity
        if target is not None:
            sim = F.cosine_similarity(
                env.state.unsqueeze(0), target.unsqueeze(0)
            ).item()
            goal_sim_series.append(sim)

        env_states.append(env.state.clone().numpy())

        if t % 2000 == 0:
            print(f"      [{t:6d}/{n_steps}] FE={fe:.4f}, H={entropy:.3f}")

    return {
        'fe': np.array(fe_series),
        'entropy': np.array(entropy_series),
        'goal_sim': np.array(goal_sim_series) if goal_sim_series else np.array([]),
        'env_states': np.array(env_states),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(results: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Referential Grounding — Closing the Causal Loop",
                 fontsize=14, y=0.98)

    window = 100  # smoothing window

    def smooth(arr, w=window):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    # --- Panel 1: Free energy trajectories ---
    ax = axes[0, 0]
    for key, label, color, ls in [
        ('org_grounded', 'Org Grounded', '#2ecc71', '-'),
        ('org_ungrounded', 'Org Ungrounded', '#e74c3c', '--'),
        ('org_goal', 'Org Goal-directed', '#3498db', '-'),
        ('ind_grounded', 'Ind Grounded', '#27ae60', ':'),
        ('ind_ungrounded', 'Ind Ungrounded', '#c0392b', ':'),
    ]:
        if key in results:
            fe = smooth(results[key]['fe'])
            ax.plot(fe, label=label, color=color, linestyle=ls, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Free Energy")
    ax.set_title("Free Energy Convergence\n(Grounded should converge faster)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: PCA of env states ---
    ax = axes[0, 1]
    try:
        from sklearn.decomposition import PCA
        for key, label, color in [
            ('org_grounded', 'Grounded', '#2ecc71'),
            ('org_ungrounded', 'Ungrounded', '#e74c3c'),
            ('org_goal', 'Goal-directed', '#3498db'),
        ]:
            if key in results:
                states = results[key]['env_states']
                if len(states) > 10:
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(states)
                    ax.scatter(coords[::10, 0], coords[::10, 1],
                              c=color, alpha=0.3, s=5, label=label)
                    # Mark start and end
                    ax.scatter(coords[0, 0], coords[0, 1], c=color, s=100,
                              marker='o', edgecolors='black')
                    ax.scatter(coords[-1, 0], coords[-1, 1], c=color, s=100,
                              marker='*', edgecolors='black')
        ax.set_title("Environment State Trajectories (PCA)\no=start, *=end")
        ax.legend(fontsize=8)
    except ImportError:
        ax.text(0.5, 0.5, "sklearn required for PCA", transform=ax.transAxes,
                ha='center')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Goal convergence ---
    ax = axes[1, 0]
    if 'org_goal' in results and len(results['org_goal']['goal_sim']) > 0:
        gs = smooth(results['org_goal']['goal_sim'])
        ax.plot(gs, color='#3498db', label='Org Goal-directed')
    if 'ind_goal' in results and len(results['ind_goal']['goal_sim']) > 0:
        gs = smooth(results['ind_goal']['goal_sim'])
        ax.plot(gs, color='#e67e22', linestyle='--', label='Ind Goal-directed')
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Threshold (0.5)')
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity to Target")
    ax.set_title("Goal Convergence\n(env.state → target)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Broadcast entropy comparison ---
    ax = axes[1, 1]
    for key, label, color, ls in [
        ('org_grounded', 'Org Grounded', '#2ecc71', '-'),
        ('org_ungrounded', 'Org Ungrounded', '#e74c3c', '--'),
        ('org_goal', 'Org Goal', '#3498db', '-'),
        ('ind_grounded', 'Ind Grounded', '#27ae60', ':'),
    ]:
        if key in results:
            ent = smooth(results[key]['entropy'])
            ax.plot(ent, label=label, color=color, linestyle=ls, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Broadcast Entropy (bits)")
    ax.set_title("Broadcast Entropy\n(Lower = more focused signal)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("results") / "grounding.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {out}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(n_steps: int = 10000):
    torch.manual_seed(0)  # reproducible grounded vs ungrounded comparison
    print("=" * 70)
    print("  Referential Grounding — Closing the Causal Loop")
    print("=" * 70)
    print(f"  Steps per condition: {n_steps}")
    print()

    target = torch.tensor([0.6, 0.2, 0.1, 0.1])  # goal target
    target = target / target.sum()

    all_results: dict = {}

    configs = [
        ('org', 'Organism', True),
        ('ind', 'Individual', False),
    ]

    conditions = [
        ('grounded', 'Grounded', None),
        ('ungrounded', 'Ungrounded', None),
        ('goal', 'Goal-directed', target),
    ]

    for sys_key, sys_name, is_organism in configs:
        for cond_key, cond_name, cond_target in conditions:
            key = f"{sys_key}_{cond_key}"
            print(f"\n  {'-' * 60}")
            print(f"  {sys_name} — {cond_name}")
            print(f"  {'-' * 60}")

            start = time.time()
            result = run_condition(n_steps, cond_key, is_organism, cond_target)
            elapsed = time.time() - start
            all_results[key] = result
            print(f"    Done in {elapsed:.1f}s")

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # FE comparison (last 20%)
    tail = n_steps // 5
    print(f"\n  Free Energy (avg last {tail} steps):")
    for key in sorted(all_results):
        fe_avg = all_results[key]['fe'][-tail:].mean()
        print(f"    {key:25s}: {fe_avg:.4f}")

    # Entropy comparison
    print(f"\n  Broadcast Entropy (avg last {tail} steps):")
    for key in sorted(all_results):
        ent_avg = all_results[key]['entropy'][-tail:].mean()
        print(f"    {key:25s}: {ent_avg:.4f}")

    # Goal convergence
    print(f"\n  Goal Convergence (avg last {tail} steps):")
    for key in ['org_goal', 'ind_goal']:
        if key in all_results and len(all_results[key]['goal_sim']) > 0:
            gs_avg = all_results[key]['goal_sim'][-tail:].mean()
            print(f"    {key:25s}: {gs_avg:.4f}")

    # FE slope (decrease rate)
    print(f"\n  FE Decrease Rate (linear fit slope):")
    for key in sorted(all_results):
        fe = all_results[key]['fe']
        x = np.arange(len(fe))
        slope = np.polyfit(x, fe, 1)[0]
        print(f"    {key:25s}: {slope:.6f}")

    # Criteria
    print(f"\n  {'-' * 60}")
    print(f"  GROUNDING CRITERIA")
    print(f"  {'-' * 60}")

    org_g_fe = all_results['org_grounded']['fe'][-tail:].mean()
    org_u_fe = all_results['org_ungrounded']['fe'][-tail:].mean()
    org_g_ent = all_results['org_grounded']['entropy'][-tail:].mean()
    org_u_ent = all_results['org_ungrounded']['entropy'][-tail:].mean()
    org_goal_sim = (all_results['org_goal']['goal_sim'][-tail:].mean()
                    if len(all_results['org_goal']['goal_sim']) > 0 else 0)

    org_g_slope = np.polyfit(np.arange(len(all_results['org_grounded']['fe'])),
                             all_results['org_grounded']['fe'], 1)[0]
    org_u_slope = np.polyfit(np.arange(len(all_results['org_ungrounded']['fe'])),
                             all_results['org_ungrounded']['fe'], 1)[0]

    criteria = [
        ("FE grounded < FE ungrounded",
         org_g_fe < org_u_fe,
         f"{org_g_fe:.4f} < {org_u_fe:.4f}"),
        ("Goal convergence > 0.5",
         org_goal_sim > 0.5,
         f"{org_goal_sim:.4f}"),
        ("FE slope grounded more negative",
         org_g_slope < org_u_slope,
         f"{org_g_slope:.6f} < {org_u_slope:.6f}"),
    ]

    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} ({value})")

    passed_count = sum(1 for _, p, _ in criteria if p)
    print(f"\n  {passed_count}/{len(criteria)} criteria met")

    # Informational (NOT a pass/fail criterion): broadcast entropy. With nominal
    # agency (latent_weight=0) the broadcast is near-uniform in BOTH conditions,
    # so its entropy saturates at ~log2(4)=2.0 and the grounded vs ungrounded gap
    # is sub-millibit noise. A real "grounded broadcast is more focused" effect
    # can only emerge once the action depends on internal state (latent_weight>0).
    # Reported for tracking, but it does not gate the experiment.
    ent_dir = "<" if org_g_ent < org_u_ent else ">="
    print(f"\n  [INFO] Broadcast entropy grounded {ent_dir} ungrounded "
          f"({org_g_ent:.4f} vs {org_u_ent:.4f}) — inconclusive until agency is real")

    # Plot
    try:
        save_plot(all_results)
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Referential grounding experiment")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Steps per condition")
    args = parser.parse_args()
    main(n_steps=args.steps)
