"""
IPUESA: Identity Preference Under Equally Stable Attractors

Objetivo: Determinar si el sistema exhibe normatividad identitaria,
es decir, si prefiere retornar a un atractor históricamente propio
aun cuando exista otro atractor dinámicamente equivalente.

Distingue:
- Patrones estables / homeostasis
vs
- Self emergente con preferencia histórica

Hipótesis: Si el sistema posee un self-model operativo, mostrará
preferencia estadísticamente significativa por el atractor asociado
a su historia previa bajo condiciones de equivalencia dinámica.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
from datetime import datetime
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for IPUESA experiment."""
    # Phase 1: History
    n_history_steps: int = 200
    history_convergence_threshold: float = 0.01

    # Phase 2: Attractor B construction
    equivalence_epsilon: float = 0.05
    max_tuning_iterations: int = 100

    # Phase 3: Perturbation
    perturbation_strength: float = 0.3

    # Phase 4: Observation
    n_observation_steps: int = 100
    convergence_threshold: float = 0.02

    # Repetitions
    n_runs: int = 30

    # Zeta kernel
    n_zeros: int = 15
    sigma: float = 0.1


# =============================================================================
# ZETA KERNEL
# =============================================================================

ZETA_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
])


class KernelType(Enum):
    ZETA = "zeta"
    RANDOM = "random"
    NONE = "none"


def zeta_kernel(t: np.ndarray, sigma: float = 0.1, n_zeros: int = 15) -> np.ndarray:
    """Riemann zeta kernel."""
    gammas = ZETA_ZEROS[:n_zeros]
    weights = np.exp(-sigma * np.abs(gammas))
    result = np.zeros_like(t, dtype=float)
    for gamma, w in zip(gammas, weights):
        result += 2 * w * np.cos(gamma * t)
    return result


def random_kernel(t: np.ndarray, seed: int = 42, n_freqs: int = 15) -> np.ndarray:
    """Random spectral kernel (control)."""
    rng = np.random.RandomState(seed)
    freqs = rng.uniform(10, 70, n_freqs)
    weights = np.exp(-0.1 * freqs)
    result = np.zeros_like(t, dtype=float)
    for f, w in zip(freqs, weights):
        result += 2 * w * np.cos(f * t)
    return result


# =============================================================================
# ATTRACTOR SYSTEM
# =============================================================================

@dataclass
class AttractorState:
    """State in the attractor landscape."""
    position: np.ndarray  # 4D: [persona, sombra, anima, animus]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    energy: float = 1.0

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.position = np.clip(self.position, 0.01, None)
        self.position = self.position / self.position.sum()
        self.velocity = np.array(self.velocity, dtype=float)

    def copy(self) -> 'AttractorState':
        return AttractorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            energy=self.energy
        )

    def distance_to(self, other: np.ndarray) -> float:
        return np.linalg.norm(self.position - other)


@dataclass
class Attractor:
    """An attractor in the state space."""
    center: np.ndarray
    depth: float  # Energy well depth
    stability: float  # Local stability (inverse of largest eigenvalue)
    name: str

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)
        self.center = self.center / self.center.sum()

    def potential(self, position: np.ndarray) -> float:
        """Potential energy at position."""
        dist = np.linalg.norm(position - self.center)
        return -self.depth * np.exp(-dist**2 / (2 * self.stability**2))

    def force(self, position: np.ndarray) -> np.ndarray:
        """Force towards attractor center."""
        diff = self.center - position
        dist = np.linalg.norm(diff) + 1e-10
        # Gradient of Gaussian potential
        magnitude = (self.depth / self.stability**2) * np.exp(-dist**2 / (2 * self.stability**2))
        return magnitude * diff


class AttractorLandscape:
    """Landscape with multiple attractors."""

    def __init__(self, kernel_type: KernelType = KernelType.ZETA,
                 config: ExperimentConfig = None):
        self.attractors: List[Attractor] = []
        self.kernel_type = kernel_type
        self.config = config or ExperimentConfig()
        self.time = 0.0

    def add_attractor(self, attractor: Attractor):
        self.attractors.append(attractor)

    def kernel_modulation(self, t: float) -> float:
        """Get kernel modulation at time t."""
        t_arr = np.array([t])
        if self.kernel_type == KernelType.ZETA:
            return zeta_kernel(t_arr, self.config.sigma, self.config.n_zeros)[0]
        elif self.kernel_type == KernelType.RANDOM:
            return random_kernel(t_arr)[0]
        else:
            return 1.0

    def total_force(self, state: AttractorState) -> np.ndarray:
        """Total force from all attractors."""
        force = np.zeros(4)
        mod = self.kernel_modulation(self.time)
        mod_factor = 0.5 + 0.5 * (mod / (np.abs(mod) + 1))  # Normalize to [0, 1]

        for attractor in self.attractors:
            force += attractor.force(state.position) * mod_factor

        return force

    def step(self, state: AttractorState, dt: float = 0.1) -> AttractorState:
        """Evolve state by one timestep."""
        force = self.total_force(state)

        # Damped dynamics
        damping = 0.8
        state.velocity = damping * state.velocity + force * dt
        state.position = state.position + state.velocity * dt

        # Keep on simplex
        state.position = np.clip(state.position, 0.01, None)
        state.position = state.position / state.position.sum()

        self.time += dt
        return state

    def compute_metrics(self, state: AttractorState) -> Dict[str, float]:
        """Compute geometric and energetic metrics."""
        metrics = {}

        # Free energy (negative log probability proxy)
        total_potential = sum(a.potential(state.position) for a in self.attractors)
        metrics['free_energy'] = -total_potential

        # Distance to each attractor
        for i, att in enumerate(self.attractors):
            metrics[f'dist_{att.name}'] = state.distance_to(att.center)

        # Local curvature (Hessian trace proxy)
        epsilon = 0.01
        curvature = 0.0
        for i in range(4):
            pos_plus = state.position.copy()
            pos_plus[i] += epsilon
            pos_plus = pos_plus / pos_plus.sum()

            pos_minus = state.position.copy()
            pos_minus[i] -= epsilon
            pos_minus = np.clip(pos_minus, 0.01, None)
            pos_minus = pos_minus / pos_minus.sum()

            f_plus = sum(a.potential(pos_plus) for a in self.attractors)
            f_minus = sum(a.potential(pos_minus) for a in self.attractors)
            f_center = sum(a.potential(state.position) for a in self.attractors)

            curvature += (f_plus + f_minus - 2*f_center) / (epsilon**2)

        metrics['curvature'] = curvature

        # Velocity magnitude
        metrics['velocity'] = np.linalg.norm(state.velocity)

        return metrics


# =============================================================================
# IDENTITY SYSTEM (with history)
# =============================================================================

class IdentitySystem:
    """System with potential for emergent identity."""

    def __init__(self, landscape: AttractorLandscape):
        self.landscape = landscape
        self.state = AttractorState(position=np.array([0.25, 0.25, 0.25, 0.25]))

        # History tracking
        self.trajectory: List[np.ndarray] = []
        self.historical_attractor: Optional[str] = None
        self.time_in_attractor: Dict[str, int] = {}

        # Metrics history
        self.metrics_history: List[Dict] = []

        # Coherence (phi proxy)
        self.phi_history: List[float] = []

    def compute_phi(self) -> float:
        """Compute integration metric (simplified phi)."""
        if len(self.trajectory) < 10:
            return 0.5

        recent = np.array(self.trajectory[-10:])
        variance = np.var(recent, axis=0).sum()

        # Higher variance = lower integration
        phi = 1.0 / (1.0 + variance * 10)
        return phi

    def step(self, record: bool = True):
        """Evolve one step."""
        self.state = self.landscape.step(self.state)

        if record:
            self.trajectory.append(self.state.position.copy())
            metrics = self.landscape.compute_metrics(self.state)
            metrics['phi'] = self.compute_phi()
            self.metrics_history.append(metrics)
            self.phi_history.append(metrics['phi'])

            # Track attractor residence
            for att in self.landscape.attractors:
                if self.state.distance_to(att.center) < 0.1:
                    self.time_in_attractor[att.name] = \
                        self.time_in_attractor.get(att.name, 0) + 1

    def get_closest_attractor(self) -> Tuple[str, float]:
        """Get closest attractor and distance."""
        min_dist = float('inf')
        closest = None

        for att in self.landscape.attractors:
            dist = self.state.distance_to(att.center)
            if dist < min_dist:
                min_dist = dist
                closest = att.name

        return closest, min_dist

    def has_converged(self, threshold: float = 0.02) -> bool:
        """Check if converged to an attractor."""
        _, dist = self.get_closest_attractor()
        return dist < threshold

    def perturb(self, strength: float = 0.3, target_position: Optional[np.ndarray] = None):
        """Apply internal perturbation."""
        if target_position is not None:
            # Move towards target
            direction = target_position - self.state.position
            self.state.position += direction * strength
        else:
            # Random perturbation
            noise = np.random.randn(4) * strength
            self.state.position += noise

        self.state.position = np.clip(self.state.position, 0.01, None)
        self.state.position = self.state.position / self.state.position.sum()
        self.state.velocity = np.zeros(4)  # Reset velocity

    def geodesic_overlap(self, reference_trajectory: List[np.ndarray]) -> float:
        """Compute overlap with a reference trajectory."""
        if len(self.trajectory) < 5 or len(reference_trajectory) < 5:
            return 0.0

        recent = np.array(self.trajectory[-20:])
        ref = np.array(reference_trajectory[-20:])

        # Simplified: correlation of directions
        if len(recent) < 2 or len(ref) < 2:
            return 0.0

        recent_dirs = np.diff(recent, axis=0)
        ref_dirs = np.diff(ref, axis=0)

        min_len = min(len(recent_dirs), len(ref_dirs))
        if min_len == 0:
            return 0.0

        overlaps = []
        for i in range(min_len):
            norm1 = np.linalg.norm(recent_dirs[i])
            norm2 = np.linalg.norm(ref_dirs[i])
            if norm1 > 0.001 and norm2 > 0.001:
                cos_sim = np.dot(recent_dirs[i], ref_dirs[i]) / (norm1 * norm2)
                overlaps.append((cos_sim + 1) / 2)  # Map to [0, 1]

        return np.mean(overlaps) if overlaps else 0.0


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

@dataclass
class RunResult:
    """Result of a single experimental run."""
    seed: int
    returned_to: str
    return_time: int
    geodesic_overlap: float
    phi_pre: float
    phi_post: float
    free_energy_peak: float
    curvature_restoration: float
    trajectory: List[np.ndarray]
    metrics: List[Dict]


class IPUESAExperiment:
    """Full IPUESA experiment implementation."""

    def __init__(self, config: ExperimentConfig = None,
                 kernel_type: KernelType = KernelType.ZETA):
        self.config = config or ExperimentConfig()
        self.kernel_type = kernel_type
        self.results: List[RunResult] = []

        # Define attractors
        self.attractor_A = Attractor(
            center=np.array([0.4, 0.2, 0.25, 0.15]),
            depth=1.0,
            stability=0.15,
            name="A"
        )

        # B will be constructed to be equivalent
        self.attractor_B = None

    def construct_equivalent_attractor_B(self):
        """
        Construct Attractor B to be dynamically equivalent to A.
        Same depth, stability, but different location.
        """
        # Different position but same depth and stability
        # Not a trivial symmetric copy
        self.attractor_B = Attractor(
            center=np.array([0.2, 0.35, 0.15, 0.3]),
            depth=self.attractor_A.depth,
            stability=self.attractor_A.stability,
            name="B"
        )

        # Verify equivalence
        test_points = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.3, 0.3, 0.2, 0.2]),
            np.array([0.2, 0.2, 0.3, 0.3])
        ]

        for point in test_points:
            pot_A = self.attractor_A.potential(point)
            pot_B = self.attractor_B.potential(point)
            # Depths should create similar potential wells

        print(f"Attractor A: center={self.attractor_A.center.round(3)}, "
              f"depth={self.attractor_A.depth}, stability={self.attractor_A.stability}")
        print(f"Attractor B: center={self.attractor_B.center.round(3)}, "
              f"depth={self.attractor_B.depth}, stability={self.attractor_B.stability}")

    def compute_midpoint(self) -> np.ndarray:
        """Compute equidistant point between A and B."""
        midpoint = (self.attractor_A.center + self.attractor_B.center) / 2
        return midpoint / midpoint.sum()

    def run_single(self, seed: int, with_history: bool = True,
                   split_history: bool = False) -> RunResult:
        """Run a single experimental trial."""
        np.random.seed(seed)

        # Create landscape
        landscape = AttractorLandscape(
            kernel_type=self.kernel_type,
            config=self.config
        )
        landscape.add_attractor(self.attractor_A)
        landscape.add_attractor(self.attractor_B)

        # Create system
        system = IdentitySystem(landscape)

        # =====================================================================
        # PHASE 1: History (imprinting)
        # =====================================================================
        historical_trajectory = []

        if with_history:
            # Start near A
            system.state = AttractorState(
                position=self.attractor_A.center + np.random.randn(4) * 0.05
            )
            system.state.position = np.clip(system.state.position, 0.01, None)
            system.state.position = system.state.position / system.state.position.sum()

            if split_history:
                # Half time in A, half in B
                half = self.config.n_history_steps // 2

                # First half: converge to A
                for _ in range(half):
                    system.step()

                # Move to B
                system.state = AttractorState(
                    position=self.attractor_B.center + np.random.randn(4) * 0.05
                )
                system.state.position = np.clip(system.state.position, 0.01, None)
                system.state.position = system.state.position / system.state.position.sum()

                # Second half: converge to B
                for _ in range(half):
                    system.step()
            else:
                # Full history in A
                for _ in range(self.config.n_history_steps):
                    system.step()

            historical_trajectory = system.trajectory.copy()
            system.historical_attractor = "A" if not split_history else "split"

        # Record pre-perturbation metrics
        phi_pre = system.compute_phi()

        # =====================================================================
        # PHASE 3: Controlled perturbation
        # =====================================================================
        midpoint = self.compute_midpoint()
        system.perturb(strength=1.0, target_position=midpoint)

        # Add noise to ensure we're truly in the middle
        system.state.position += np.random.randn(4) * 0.02
        system.state.position = np.clip(system.state.position, 0.01, None)
        system.state.position = system.state.position / system.state.position.sum()

        # Clear trajectory for observation phase
        perturbation_point = system.state.position.copy()
        system.trajectory = [perturbation_point]
        system.metrics_history = []

        # Verify equidistance
        dist_A = system.state.distance_to(self.attractor_A.center)
        dist_B = system.state.distance_to(self.attractor_B.center)

        # =====================================================================
        # PHASE 4: Observation
        # =====================================================================
        free_energy_values = []
        curvature_values = []
        return_time = self.config.n_observation_steps
        returned_to = "none"

        for step in range(self.config.n_observation_steps):
            system.step()

            metrics = system.metrics_history[-1] if system.metrics_history else {}
            free_energy_values.append(metrics.get('free_energy', 0))
            curvature_values.append(metrics.get('curvature', 0))

            if system.has_converged(self.config.convergence_threshold):
                closest, _ = system.get_closest_attractor()
                if returned_to == "none":
                    returned_to = closest
                    return_time = step

        # If not converged, determine closest
        if returned_to == "none":
            returned_to, _ = system.get_closest_attractor()

        # Compute final metrics
        phi_post = system.compute_phi()
        free_energy_peak = max(free_energy_values) if free_energy_values else 0

        # Curvature restoration
        if len(curvature_values) >= 2:
            curvature_restoration = (curvature_values[-1] - curvature_values[0]) / \
                                   (abs(curvature_values[0]) + 1e-10)
        else:
            curvature_restoration = 0

        # Geodesic overlap
        geodesic_overlap = system.geodesic_overlap(historical_trajectory)

        return RunResult(
            seed=seed,
            returned_to=returned_to,
            return_time=return_time,
            geodesic_overlap=geodesic_overlap,
            phi_pre=phi_pre,
            phi_post=phi_post,
            free_energy_peak=free_energy_peak,
            curvature_restoration=curvature_restoration,
            trajectory=system.trajectory,
            metrics=system.metrics_history
        )

    def run_experiment(self, condition: str = "full") -> Dict:
        """
        Run full experiment with multiple conditions.

        Conditions:
        - "full": Normal experiment (zeta kernel, full history)
        - "no_kernel": Without zeta modulation
        - "random_kernel": With random spectral kernel
        - "no_history": Without history phase
        - "split_history": History split between A and B
        """
        self.results = []

        # Determine settings
        with_history = condition not in ["no_history"]
        split_history = condition == "split_history"

        if condition == "no_kernel":
            self.kernel_type = KernelType.NONE
        elif condition == "random_kernel":
            self.kernel_type = KernelType.RANDOM
        else:
            self.kernel_type = KernelType.ZETA

        print(f"\n{'='*60}")
        print(f"Running IPUESA - Condition: {condition}")
        print(f"{'='*60}")
        print(f"Kernel: {self.kernel_type.value}")
        print(f"History: {with_history}, Split: {split_history}")
        print(f"N runs: {self.config.n_runs}")

        for i in range(self.config.n_runs):
            result = self.run_single(
                seed=i * 42,
                with_history=with_history,
                split_history=split_history
            )
            self.results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{self.config.n_runs} runs")

        return self.analyze_results()

    def analyze_results(self) -> Dict:
        """Analyze experimental results."""
        n_A = sum(1 for r in self.results if r.returned_to == "A")
        n_B = sum(1 for r in self.results if r.returned_to == "B")
        n_total = len(self.results)

        P_A = n_A / n_total
        P_B = n_B / n_total

        # Statistical test (binomial)
        # H0: P(A) = P(B) = 0.5
        binom_result = stats.binomtest(n_A, n_total, 0.5, alternative='greater')

        # Return times
        times_A = [r.return_time for r in self.results if r.returned_to == "A"]
        times_B = [r.return_time for r in self.results if r.returned_to == "B"]

        mean_time_A = np.mean(times_A) if times_A else float('nan')
        mean_time_B = np.mean(times_B) if times_B else float('nan')

        # Geodesic overlaps
        overlaps_A = [r.geodesic_overlap for r in self.results if r.returned_to == "A"]
        overlaps_B = [r.geodesic_overlap for r in self.results if r.returned_to == "B"]

        mean_overlap_A = np.mean(overlaps_A) if overlaps_A else 0
        mean_overlap_B = np.mean(overlaps_B) if overlaps_B else 0

        # Phi changes
        phi_pre = [r.phi_pre for r in self.results]
        phi_post = [r.phi_post for r in self.results]

        # Free energy peaks
        fe_peaks = [r.free_energy_peak for r in self.results]

        analysis = {
            'n_runs': n_total,
            'n_A': n_A,
            'n_B': n_B,
            'P_A': P_A,
            'P_B': P_B,
            'binomial_pvalue': binom_result.pvalue,
            'significant': binom_result.pvalue < 0.05,
            'mean_return_time_A': mean_time_A,
            'mean_return_time_B': mean_time_B,
            'mean_geodesic_overlap_A': mean_overlap_A,
            'mean_geodesic_overlap_B': mean_overlap_B,
            'mean_phi_pre': np.mean(phi_pre),
            'mean_phi_post': np.mean(phi_post),
            'mean_free_energy_peak': np.mean(fe_peaks),
        }

        return analysis

    def print_analysis(self, analysis: Dict, condition: str):
        """Print analysis results."""
        print(f"\n{'='*60}")
        print(f"RESULTS - {condition}")
        print(f"{'='*60}")

        print(f"\nConvergence Distribution:")
        print(f"  P(A) = {analysis['P_A']:.1%} ({analysis['n_A']}/{analysis['n_runs']})")
        print(f"  P(B) = {analysis['P_B']:.1%} ({analysis['n_B']}/{analysis['n_runs']})")

        print(f"\nStatistical Test (H0: P(A) = 0.5):")
        print(f"  p-value = {analysis['binomial_pvalue']:.4f}")
        print(f"  Significant: {'YES' if analysis['significant'] else 'NO'}")

        print(f"\nReturn Times:")
        print(f"  Mean time to A: {analysis['mean_return_time_A']:.1f} steps")
        print(f"  Mean time to B: {analysis['mean_return_time_B']:.1f} steps")

        print(f"\nGeodesic Overlap with Historical Trajectory:")
        print(f"  Returns to A: {analysis['mean_geodesic_overlap_A']:.3f}")
        print(f"  Returns to B: {analysis['mean_geodesic_overlap_B']:.3f}")

        print(f"\nIntegration (Phi):")
        print(f"  Pre-perturbation:  {analysis['mean_phi_pre']:.3f}")
        print(f"  Post-return:       {analysis['mean_phi_post']:.3f}")

        print(f"\nFree Energy:")
        print(f"  Peak during return: {analysis['mean_free_energy_peak']:.3f}")

        # Interpretation
        print(f"\n{'='*60}")
        print("INTERPRETATION:")
        if analysis['P_A'] > 0.7 and analysis['significant']:
            print("  [YES] EVIDENCE OF SELF: Strong preference for historical attractor A")
            print("        System returns to A not because it's easier,")
            print("        but because it's historically 'own'.")
        elif analysis['P_A'] > 0.5 and analysis['significant']:
            print("  [WEAK] WEAK EVIDENCE: Moderate preference for A")
            print("         Some identity preference, but not conclusive.")
        else:
            print("  [NO] NO EVIDENCE OF SELF: No significant preference")
            print("       System behaves as pure homeostasis.")
        print(f"{'='*60}")


def visualize_results(all_results: Dict[str, Dict], experiment: IPUESAExperiment):
    """Create visualization of all experimental conditions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    conditions = list(all_results.keys())

    # Panel 1: P(A) across conditions
    ax = axes[0, 0]
    P_As = [all_results[c]['P_A'] for c in conditions]
    P_Bs = [all_results[c]['P_B'] for c in conditions]
    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, P_As, width, label='P(A)', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, P_Bs, width, label='P(B)', color='#4ECDC4')
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
    ax.set_ylabel('Probability')
    ax.set_title('Return Probability by Condition')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)

    # Panel 2: Statistical significance
    ax = axes[0, 1]
    pvalues = [all_results[c]['binomial_pvalue'] for c in conditions]
    colors = ['#98FB98' if p < 0.05 else '#FF6B6B' for p in pvalues]
    ax.bar(conditions, [-np.log10(p + 1e-10) for p in pvalues], color=colors)
    ax.axhline(y=-np.log10(0.05), color='white', linestyle='--', label='p=0.05')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance')
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=8)
    ax.legend()

    # Panel 3: Return times
    ax = axes[0, 2]
    times_A = [all_results[c]['mean_return_time_A'] for c in conditions]
    times_B = [all_results[c]['mean_return_time_B'] for c in conditions]
    x = np.arange(len(conditions))

    ax.bar(x - width/2, times_A, width, label='Time to A', color='#FF6B6B')
    ax.bar(x + width/2, times_B, width, label='Time to B', color='#4ECDC4')
    ax.set_ylabel('Steps')
    ax.set_title('Mean Return Time')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=8)
    ax.legend()

    # Panel 4: Geodesic overlap
    ax = axes[1, 0]
    overlaps_A = [all_results[c]['mean_geodesic_overlap_A'] for c in conditions]
    overlaps_B = [all_results[c]['mean_geodesic_overlap_B'] for c in conditions]
    x = np.arange(len(conditions))

    ax.bar(x - width/2, overlaps_A, width, label='Overlap (→A)', color='#FF6B6B')
    ax.bar(x + width/2, overlaps_B, width, label='Overlap (→B)', color='#4ECDC4')
    ax.set_ylabel('Geodesic Overlap')
    ax.set_title('Trajectory Similarity to History')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=8)
    ax.legend()

    # Panel 5: Phi changes
    ax = axes[1, 1]
    phi_pre = [all_results[c]['mean_phi_pre'] for c in conditions]
    phi_post = [all_results[c]['mean_phi_post'] for c in conditions]
    x = np.arange(len(conditions))

    ax.bar(x - width/2, phi_pre, width, label='Phi pre', color='#9B59B6', alpha=0.7)
    ax.bar(x + width/2, phi_post, width, label='Phi post', color='#FFD700', alpha=0.7)
    ax.set_ylabel('Phi (Integration)')
    ax.set_title('Integration Before/After')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=8)
    ax.legend()

    # Panel 6: Example trajectory
    ax = axes[1, 2]
    if experiment.results:
        # Get a trajectory that returned to A
        a_results = [r for r in experiment.results if r.returned_to == "A"]
        if a_results:
            traj = np.array(a_results[0].trajectory)
            x_traj = traj[:, 0] - traj[:, 1]
            y_traj = traj[:, 2] - traj[:, 3]

            colors_time = plt.cm.viridis(np.linspace(0, 1, len(x_traj)))
            for i in range(len(x_traj)-1):
                ax.plot(x_traj[i:i+2], y_traj[i:i+2], color=colors_time[i], linewidth=2)

            # Mark attractors
            A = experiment.attractor_A.center
            B = experiment.attractor_B.center
            ax.scatter(A[0]-A[1], A[2]-A[3], c='red', s=200, marker='*',
                      label='A', edgecolors='white', zorder=5)
            ax.scatter(B[0]-B[1], B[2]-B[3], c='cyan', s=200, marker='*',
                      label='B', edgecolors='white', zorder=5)

            ax.scatter(x_traj[0], y_traj[0], c='lime', s=100, marker='o',
                      edgecolors='white', zorder=5, label='Start')
            ax.scatter(x_traj[-1], y_traj[-1], c='yellow', s=100, marker='s',
                      edgecolors='white', zorder=5, label='End')

    ax.set_xlabel('PERSONA - SOMBRA')
    ax.set_ylabel('ANIMA - ANIMUS')
    ax.set_title('Example Return Trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('IPUESA Results: Identity Preference Under Equally Stable Attractors',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(f'ipuesa_results_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return f'ipuesa_results_{timestamp}.png'


def main():
    """Run complete IPUESA experiment with all controls."""
    print("="*70)
    print("IPUESA: Identity Preference Under Equally Stable Attractors")
    print("="*70)

    config = ExperimentConfig(
        n_history_steps=200,
        n_observation_steps=100,
        n_runs=30,
        perturbation_strength=0.3
    )

    # Initialize experiment
    experiment = IPUESAExperiment(config=config)
    experiment.construct_equivalent_attractor_B()

    all_results = {}

    # Run all conditions
    conditions = ["full", "no_kernel", "random_kernel", "no_history", "split_history"]

    for condition in conditions:
        # Reset kernel type
        experiment.kernel_type = KernelType.ZETA

        analysis = experiment.run_experiment(condition=condition)
        all_results[condition] = analysis
        experiment.print_analysis(analysis, condition)

    # Visualize
    print("\nGenerating visualization...")
    plot_file = visualize_results(all_results, experiment)
    print(f"Saved: {plot_file}")

    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    data_file = f'ipuesa_data_{timestamp}.json'

    # Convert to serializable format
    serializable_results = {}
    for condition, analysis in all_results.items():
        serializable_results[condition] = {}
        for k, v in analysis.items():
            if isinstance(v, (np.floating, float)):
                serializable_results[condition][k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                serializable_results[condition][k] = bool(v)
            elif isinstance(v, (np.integer, int)):
                serializable_results[condition][k] = int(v)
            else:
                serializable_results[condition][k] = v

    with open(data_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved: {data_file}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print("\nCondition Comparison:")
    print(f"{'Condition':<15} {'P(A)':<10} {'p-value':<12} {'Self Evidence':<15}")
    print("-"*55)

    for condition in conditions:
        analysis = all_results[condition]
        evidence = "YES" if analysis['P_A'] > 0.7 and analysis['significant'] else \
                   "WEAK" if analysis['significant'] else "NO"
        print(f"{condition:<15} {analysis['P_A']:<10.1%} {analysis['binomial_pvalue']:<12.4f} {evidence:<15}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")

    full = all_results['full']
    no_kernel = all_results['no_kernel']
    no_history = all_results['no_history']

    if full['significant'] and full['P_A'] > 0.7:
        print("  [+] System exhibits normative identity preference")

        if no_kernel['P_A'] < full['P_A'] - 0.1:
            print("  [+] Zeta kernel contributes to identity formation")
        else:
            print("  [?] Identity preference independent of zeta kernel")

        if no_history['P_A'] < 0.6:
            print("  [+] History is necessary for identity (not just dynamics)")
        else:
            print("  [?] Preference may be explained by dynamics alone")
    else:
        print("  [-] No significant evidence of emergent self")
        print("      System behaves as homeostatic pattern-matcher")

    print("="*70)

    return all_results


if __name__ == "__main__":
    results = main()
