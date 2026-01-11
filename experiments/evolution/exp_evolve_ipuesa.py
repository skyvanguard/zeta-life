"""
IPUESA Hyperparameter Evolution Orchestrator

Runs evolutionary optimization to discover optimal IPUESA configurations
that maximize self-evidence criteria.

Usage:
    python exp_evolve_ipuesa.py --generations 50 --population 20
    python exp_evolve_ipuesa.py --resume evolved_configs/checkpoint_gen25.json
    python exp_evolve_ipuesa.py --quick  # Fast validation run

Author: Zeta Life Research
Date: 2026-01-11
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EVOLVED_CONFIGS_DIR = PROJECT_ROOT / "evolved_configs"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zeta_life.evolution.config_space import (
    EvolvableConfig, PARAM_RANGES, get_baseline_config, get_config_as_flat_dict
)
from zeta_life.evolution.fitness_evaluator import evaluate_config, FitnessResult


@dataclass
class Individual:
    """An individual in the evolutionary population."""
    config: Dict[str, float]
    fitness: float = 0.0
    criteria_passed: int = 0
    generation: int = 0
    parent_ids: Optional[List[int]] = None
    mutations: Optional[List[str]] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutations is None:
            self.mutations = []


@dataclass
class EvolutionState:
    """Complete state of evolution (for checkpoint/resume)."""
    generation: int
    population: List[Individual]
    best_ever: Individual
    history: List[Dict]
    evo_config: Dict

    def save(self, path: Path):
        """Save checkpoint to file."""
        data = {
            'generation': self.generation,
            'population': [asdict(ind) for ind in self.population],
            'best_ever': asdict(self.best_ever),
            'history': self.history,
            'evo_config': self.evo_config
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        print(f"[Checkpoint] Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'EvolutionState':
        """Load checkpoint from file."""
        data = json.loads(path.read_text())
        return cls(
            generation=data['generation'],
            population=[Individual(**d) for d in data['population']],
            best_ever=Individual(**data['best_ever']),
            history=data['history'],
            evo_config=data['evo_config']
        )


class IPUESAEvolver:
    """
    Evolutionary optimizer for IPUESA configurations.

    Implements:
    - Initialization with variation around baseline
    - Tournament selection
    - Gaussian mutation with adaptive strength
    - Arithmetic crossover
    - Elitism
    """

    def __init__(self,
                 population_size: int = 20,
                 elite_size: int = 2,
                 mutation_rate: float = 0.3,
                 mutation_strength: float = 0.15,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 3,
                 n_eval_runs: int = 5,
                 n_eval_steps: int = 100,
                 n_agents: int = 24,
                 n_clusters: int = 4):

        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.n_eval_runs = n_eval_runs
        self.n_eval_steps = n_eval_steps
        self.n_agents = n_agents
        self.n_clusters = n_clusters

        # Baseline config
        self.baseline = get_baseline_config()
        self.param_ranges = PARAM_RANGES

    def initialize_population(self) -> List[Individual]:
        """
        Create initial population with variation around baseline.
        First individual is the exact baseline (initial elitism).
        """
        population = []

        # Individual 0: exact baseline
        baseline_config = get_config_as_flat_dict(self.baseline)
        population.append(Individual(config=baseline_config, generation=0))

        # Rest: variations of baseline
        for i in range(1, self.population_size):
            config = self._mutate_config(baseline_config, strength=0.25)
            population.append(Individual(config=config, generation=0))

        return population

    def _mutate_config(self, config: Dict[str, float],
                       strength: Optional[float] = None) -> Dict[str, float]:
        """Apply Gaussian mutation to a config."""
        strength = strength if strength is not None else self.mutation_strength
        new_config = {}

        for key, value in config.items():
            if np.random.random() < self.mutation_rate:
                min_val, max_val = self.param_ranges[key]
                range_size = max_val - min_val

                # Gaussian mutation
                delta = np.random.normal(0, strength * range_size)
                new_value = value + delta

                # Clamp to valid range
                new_value = max(min_val, min(max_val, new_value))
                new_config[key] = float(new_value)
            else:
                new_config[key] = value

        return new_config

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Arithmetic crossover between two configs."""
        child = {}
        alpha = np.random.random()  # Mixing weight

        for key in parent1.keys():
            if np.random.random() < self.crossover_rate:
                # Arithmetic mix
                child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
            else:
                # Direct inheritance
                child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]

        return child

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Tournament selection."""
        candidates = np.random.choice(
            len(population),
            size=min(self.tournament_size, len(population)),
            replace=False
        )
        winner_idx = max(candidates, key=lambda i: population[i].fitness)
        return population[winner_idx]

    def evaluate_population(self, population: List[Individual],
                            generation: int) -> List[Individual]:
        """Evaluate fitness of all individuals."""
        print(f"\n[Gen {generation}] Evaluating {len(population)} individuals...")

        for i, ind in enumerate(population):
            # Skip already evaluated elites
            if ind.fitness > 0 and ind.generation < generation:
                print(f"  [{i+1}/{len(population)}] (elite, skipped)")
                continue

            result = evaluate_config(
                ind.config,
                n_runs=self.n_eval_runs,
                n_steps=self.n_eval_steps,
                n_agents=self.n_agents,
                n_clusters=self.n_clusters
            )

            ind.fitness = result.fitness_score
            ind.criteria_passed = result.criteria_passed
            ind.generation = generation

            status = "+" if result.criteria_passed >= 6 else "o"
            print(f"  [{i+1}/{len(population)}] {status} "
                  f"fitness={ind.fitness:.4f} "
                  f"criteria={ind.criteria_passed}/8")

        return population

    def evolve_generation(self, population: List[Individual],
                          generation: int) -> List[Individual]:
        """Create new generation via selection + crossover + mutation."""
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        new_population = []

        # Elitism: keep best individuals
        for i in range(self.elite_size):
            elite = Individual(
                config=population[i].config.copy(),
                fitness=population[i].fitness,
                criteria_passed=population[i].criteria_passed,
                generation=generation,
                parent_ids=[i],
                mutations=["elite"]
            )
            new_population.append(elite)

        # Generate rest via crossover + mutation
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            # Crossover
            child_config = self._crossover(parent1.config, parent2.config)

            # Mutation
            child_config = self._mutate_config(child_config)

            child = Individual(
                config=child_config,
                generation=generation,
                parent_ids=[
                    population.index(parent1),
                    population.index(parent2)
                ],
                mutations=["crossover", "mutation"]
            )
            new_population.append(child)

        return new_population

    def run(self,
            generations: int = 50,
            target_criteria: int = 8,
            checkpoint_interval: int = 5,
            resume_from: Optional[Path] = None) -> EvolutionState:
        """
        Run the complete evolutionary cycle.

        Args:
            generations: Maximum number of generations
            target_criteria: Target criteria for early stop
            checkpoint_interval: Checkpoint save frequency
            resume_from: Path to checkpoint for resume

        Returns:
            EvolutionState with final results
        """
        # Initialize or resume
        if resume_from and resume_from.exists():
            state = EvolutionState.load(resume_from)
            population = state.population
            start_gen = state.generation + 1
            best_ever = state.best_ever
            history = state.history
            print(f"[Resume] Continuing from generation {start_gen}")
        else:
            population = self.initialize_population()
            start_gen = 0
            best_ever = None
            history = []

        # Config for checkpoint
        evo_config = {
            'population_size': self.population_size,
            'elite_size': self.elite_size,
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
            'crossover_rate': self.crossover_rate,
            'tournament_size': self.tournament_size,
            'n_eval_runs': self.n_eval_runs,
            'n_eval_steps': self.n_eval_steps,
        }

        print("=" * 60)
        print("IPUESA Hyperparameter Evolution")
        print("=" * 60)
        print(f"Population: {self.population_size}")
        print(f"Generations: {generations}")
        print(f"Target: {target_criteria}/8 criteria")
        print(f"Eval runs: {self.n_eval_runs}, steps: {self.n_eval_steps}")
        print("=" * 60)

        final_gen = start_gen

        for gen in range(start_gen, generations):
            final_gen = gen

            # Evaluate
            population = self.evaluate_population(population, gen)

            # Best of this generation
            best_gen = max(population, key=lambda x: x.fitness)

            # Update global best
            if best_ever is None or best_gen.fitness > best_ever.fitness:
                best_ever = Individual(
                    config=best_gen.config.copy(),
                    fitness=best_gen.fitness,
                    criteria_passed=best_gen.criteria_passed,
                    generation=gen
                )
                print(f"\n  [NEW BEST] Gen {gen}: "
                      f"fitness={best_ever.fitness:.4f} "
                      f"criteria={best_ever.criteria_passed}/8")

            # Generation statistics
            fitnesses = [ind.fitness for ind in population]
            gen_stats = {
                'generation': gen,
                'best_fitness': max(fitnesses),
                'avg_fitness': float(np.mean(fitnesses)),
                'std_fitness': float(np.std(fitnesses)),
                'best_criteria': best_gen.criteria_passed,
                'best_ever_fitness': best_ever.fitness,
                'best_ever_criteria': best_ever.criteria_passed,
            }
            history.append(gen_stats)

            print(f"[Gen {gen}] best={gen_stats['best_fitness']:.4f} "
                  f"avg={gen_stats['avg_fitness']:.4f} "
                  f"std={gen_stats['std_fitness']:.4f}")

            # Early stop if target reached
            if best_ever.criteria_passed >= target_criteria:
                print(f"\n[SUCCESS] Reached {target_criteria}/8 criteria at gen {gen}!")
                break

            # Checkpoint
            if gen > 0 and gen % checkpoint_interval == 0:
                state = EvolutionState(gen, population, best_ever, history, evo_config)
                checkpoint_path = EVOLVED_CONFIGS_DIR / f"checkpoint_gen{gen}.json"
                state.save(checkpoint_path)

            # Evolve (except last generation)
            if gen < generations - 1:
                population = self.evolve_generation(population, gen + 1)

        # Save final results
        final_state = EvolutionState(final_gen, population, best_ever, history, evo_config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = EVOLVED_CONFIGS_DIR / f"evolution_result_{timestamp}.json"
        final_state.save(final_path)

        # Save clean config
        clean_config_path = EVOLVED_CONFIGS_DIR / f"evolved_config_{timestamp}.json"
        clean_config_path.write_text(json.dumps(best_ever.config, indent=2))
        print(f"\n[Final] Best config saved to {clean_config_path}")

        return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Evolve IPUESA hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations")
    parser.add_argument("--population", type=int, default=20,
                        help="Population size")
    parser.add_argument("--elite", type=int, default=2,
                        help="Number of elites to preserve")
    parser.add_argument("--mutation-rate", type=float, default=0.3,
                        help="Probability of mutating each parameter")
    parser.add_argument("--mutation-strength", type=float, default=0.15,
                        help="Mutation strength (fraction of range)")
    parser.add_argument("--crossover-rate", type=float, default=0.7,
                        help="Crossover probability per parameter")
    parser.add_argument("--eval-runs", type=int, default=5,
                        help="IPUESA runs per evaluation")
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Steps per IPUESA run")
    parser.add_argument("--target", type=int, default=8,
                        help="Target criteria for early stop")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint file")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Generations between checkpoints")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation run (3 gen, 8 pop)")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.generations = 3
        args.population = 8
        args.eval_runs = 2
        args.eval_steps = 50
        print("[Quick mode] Using reduced parameters for validation")

    # Create output directory
    EVOLVED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create evolver
    evolver = IPUESAEvolver(
        population_size=args.population,
        elite_size=args.elite,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        n_eval_runs=args.eval_runs,
        n_eval_steps=args.eval_steps
    )

    # Run evolution
    state = evolver.run(
        generations=args.generations,
        target_criteria=args.target,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume
    )

    # Final summary
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best fitness: {state.best_ever.fitness:.4f}")
    print(f"Criteria passed: {state.best_ever.criteria_passed}/8")
    print(f"Found at generation: {state.best_ever.generation}")
    print(f"\nBest config ({len(state.best_ever.config)} parameters):")

    # Group parameters for display
    groups = {
        'Damage': ['damage_multiplier', 'base_degrad_rate', 'embedding_protection',
                   'stance_protection', 'compound_factor', 'module_protection',
                   'resilience_min', 'resilience_range', 'noise_scale', 'residual_cap'],
        'Recovery': ['base_recovery_rate', 'embedding_bonus', 'cluster_bonus',
                     'degradation_penalty', 'degrad_recovery_factor', 'corruption_decay'],
        'Effects': ['effect_pattern_detector', 'effect_threat_filter',
                    'effect_recovery_accelerator', 'effect_exploration_dampener',
                    'effect_embedding_protector', 'effect_cascade_breaker',
                    'effect_residual_cleaner', 'effect_anticipation_enhancer'],
        'Thresholds': ['consolidation_threshold', 'spread_threshold',
                       'spread_probability', 'spread_strength_factor',
                       'module_cap', 'min_activations']
    }

    for group_name, params in groups.items():
        print(f"\n  [{group_name}]")
        for param in params:
            if param in state.best_ever.config:
                value = state.best_ever.config[param]
                baseline = getattr(get_baseline_config(), param)
                diff = ((value - baseline) / baseline * 100) if baseline != 0 else 0
                sign = "+" if diff > 0 else ""
                print(f"    {param}: {value:.4f} ({sign}{diff:.1f}%)")


if __name__ == "__main__":
    main()
