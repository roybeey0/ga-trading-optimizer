"""
Genetic Algorithm Engine
Optimizes trading strategy parameters using evolutionary computation.
"""

import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import time
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# GENE DEFINITIONS — each parameter with bounds
# ─────────────────────────────────────────────
GENE_DEFINITIONS = {
    # Technical Indicators
    'ema_fast':          {'min': 5,    'max': 50,   'type': 'int',   'step': 1},
    'ema_slow':          {'min': 20,   'max': 200,  'type': 'int',   'step': 5},
    'rsi_period':        {'min': 5,    'max': 30,   'type': 'int',   'step': 1},
    'rsi_oversold':      {'min': 15,   'max': 40,   'type': 'float', 'step': 1},
    'rsi_overbought':    {'min': 60,   'max': 85,   'type': 'float', 'step': 1},
    'atr_period':        {'min': 7,    'max': 30,   'type': 'int',   'step': 1},
    'atr_multiplier':    {'min': 0.5,  'max': 4.0,  'type': 'float', 'step': 0.1},
    'bb_period':         {'min': 10,   'max': 50,   'type': 'int',   'step': 1},
    'bb_std':            {'min': 1.0,  'max': 3.5,  'type': 'float', 'step': 0.1},
    
    # Risk Management
    'stop_loss_pct':     {'min': 0.005,'max': 0.08, 'type': 'float', 'step': 0.005},
    'take_profit_pct':   {'min': 0.01, 'max': 0.25, 'type': 'float', 'step': 0.005},
    'position_size_pct': {'min': 0.05, 'max': 0.5,  'type': 'float', 'step': 0.05},
    
    # Filters
    'volume_filter':     {'min': 0.8,  'max': 3.0,  'type': 'float', 'step': 0.1},
}

GENE_KEYS = list(GENE_DEFINITIONS.keys())


@dataclass
class Individual:
    """Represents one set of strategy parameters (chromosome)."""
    genes: Dict = field(default_factory=dict)
    fitness: float = -999.0
    generation: int = 0
    
    def __post_init__(self):
        if not self.genes:
            self.genes = self._random_genes()
    
    def _random_genes(self) -> Dict:
        genes = {}
        for key, g in GENE_DEFINITIONS.items():
            if g['type'] == 'int':
                genes[key] = random.randint(int(g['min']), int(g['max']))
            else:
                steps = round((g['max'] - g['min']) / g['step'])
                genes[key] = round(g['min'] + random.randint(0, steps) * g['step'], 4)
        return genes
    
    def clone(self):
        return Individual(genes=deepcopy(self.genes), fitness=self.fitness)


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_individual: Individual
    population_diversity: float
    elapsed_time: float


class GeneticAlgorithm:
    """
    Genetic Algorithm for Trading Strategy Optimization.
    
    Features:
    - Tournament selection
    - Uniform + arithmetic crossover
    - Gaussian + uniform mutation
    - Elitism preservation
    - Adaptive mutation rate
    - Population diversity tracking
    """
    
    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        elite_size: int = 5,
        tournament_size: int = 3,
        fitness_fn: Callable = None,
        verbose: bool = True,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.fitness_fn = fitness_fn
        self.verbose = verbose
        
        self.population: List[Individual] = []
        self.generation_stats: List[GenerationStats] = []
        self.best_ever: Individual = None
        self.start_time = None
    
    # ─────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────
    
    def _initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        population = []
        
        # Add some heuristic seed individuals
        seeds = self._get_seed_individuals()
        population.extend(seeds)
        
        # Fill rest with random individuals
        while len(population) < self.population_size:
            population.append(Individual())
        
        return population[:self.population_size]
    
    def _get_seed_individuals(self) -> List[Individual]:
        """Pre-defined good starting points based on domain knowledge."""
        seeds_params = [
            # Conservative trend follower
            {'ema_fast': 10, 'ema_slow': 30, 'rsi_period': 14, 'rsi_oversold': 30, 
             'rsi_overbought': 70, 'atr_period': 14, 'atr_multiplier': 2.0,
             'bb_period': 20, 'bb_std': 2.0, 'stop_loss_pct': 0.02, 
             'take_profit_pct': 0.06, 'position_size_pct': 0.2, 'volume_filter': 1.2},
            # Aggressive momentum
            {'ema_fast': 5, 'ema_slow': 20, 'rsi_period': 10, 'rsi_oversold': 25, 
             'rsi_overbought': 75, 'atr_period': 10, 'atr_multiplier': 1.5,
             'bb_period': 15, 'bb_std': 1.8, 'stop_loss_pct': 0.015, 
             'take_profit_pct': 0.05, 'position_size_pct': 0.3, 'volume_filter': 1.0},
            # Slow trend
            {'ema_fast': 20, 'ema_slow': 50, 'rsi_period': 21, 'rsi_oversold': 35, 
             'rsi_overbought': 65, 'atr_period': 21, 'atr_multiplier': 2.5,
             'bb_period': 25, 'bb_std': 2.2, 'stop_loss_pct': 0.03, 
             'take_profit_pct': 0.09, 'position_size_pct': 0.15, 'volume_filter': 1.5},
        ]
        return [Individual(genes=p) for p in seeds_params]
    
    # ─────────────────────────────────────────
    # SELECTION
    # ─────────────────────────────────────────
    
    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Select individual via tournament."""
        contestants = random.sample(population, min(self.tournament_size, len(population)))
        return max(contestants, key=lambda x: x.fitness)
    
    def _get_elite(self, population: List[Individual]) -> List[Individual]:
        """Return top N individuals (elitism)."""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return [ind.clone() for ind in sorted_pop[:self.elite_size]]
    
    # ─────────────────────────────────────────
    # CROSSOVER
    # ─────────────────────────────────────────
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover with arithmetic blend for continuous genes.
        """
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        child1_genes = {}
        child2_genes = {}
        
        for key in GENE_KEYS:
            g = GENE_DEFINITIONS[key]
            p1_val = parent1.genes[key]
            p2_val = parent2.genes[key]
            
            if random.random() < 0.5:
                # Swap genes
                child1_genes[key] = p1_val
                child2_genes[key] = p2_val
            else:
                # Arithmetic blend (BLX-alpha)
                alpha = random.uniform(0, 1)
                if g['type'] == 'int':
                    child1_genes[key] = int(alpha * p1_val + (1 - alpha) * p2_val)
                    child2_genes[key] = int((1 - alpha) * p1_val + alpha * p2_val)
                else:
                    child1_genes[key] = round(alpha * p1_val + (1 - alpha) * p2_val, 4)
                    child2_genes[key] = round((1 - alpha) * p1_val + alpha * p2_val, 4)
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    # ─────────────────────────────────────────
    # MUTATION
    # ─────────────────────────────────────────
    
    def _mutate(self, individual: Individual, adaptive_rate: float = None) -> Individual:
        """
        Gaussian mutation with adaptive rate based on generation progress.
        """
        rate = adaptive_rate or self.mutation_rate
        mutant = individual.clone()
        
        for key in GENE_KEYS:
            if random.random() < rate:
                g = GENE_DEFINITIONS[key]
                current = mutant.genes[key]
                gene_range = g['max'] - g['min']
                
                if g['type'] == 'int':
                    # Gaussian integer mutation
                    noise = int(np.random.normal(0, gene_range * 0.1))
                    new_val = int(np.clip(current + noise, g['min'], g['max']))
                    mutant.genes[key] = new_val
                else:
                    # Gaussian float mutation
                    noise = np.random.normal(0, gene_range * 0.1)
                    new_val = round(np.clip(current + noise, g['min'], g['max']), 4)
                    mutant.genes[key] = new_val
        
        # Ensure ema_fast < ema_slow
        if mutant.genes['ema_fast'] >= mutant.genes['ema_slow']:
            mutant.genes['ema_fast'] = max(5, mutant.genes['ema_slow'] - 5)
        
        return mutant
    
    # ─────────────────────────────────────────
    # DIVERSITY TRACKING
    # ─────────────────────────────────────────
    
    def _population_diversity(self, population: List[Individual]) -> float:
        """Measure genetic diversity as normalized gene variance."""
        if len(population) < 2:
            return 0.0
        
        diversities = []
        for key in GENE_KEYS:
            g = GENE_DEFINITIONS[key]
            vals = [ind.genes[key] for ind in population]
            gene_range = g['max'] - g['min']
            if gene_range > 0:
                normalized_std = np.std(vals) / gene_range
                diversities.append(normalized_std)
        
        return np.mean(diversities) if diversities else 0.0
    
    # ─────────────────────────────────────────
    # MAIN EVOLUTION LOOP
    # ─────────────────────────────────────────
    
    def evolve(self, df: pd.DataFrame) -> Individual:
        """
        Main GA loop. Returns the best individual found.
        
        Args:
            df: DataFrame with OHLCV + indicators (pre-computed)
        
        Returns:
            Best Individual after evolution
        """
        assert self.fitness_fn is not None, "fitness_fn must be set before calling evolve()"
        
        self.start_time = time.time()
        self.population = self._initialize_population()
        self.generation_stats = []
        
        if self.verbose:
            print("\n" + "═" * 65)
            print("   🧬 GENETIC ALGORITHM — TRADING STRATEGY OPTIMIZER")
            print("═" * 65)
            print(f"   Population: {self.population_size} | Generations: {self.n_generations}")
            print(f"   Crossover: {self.crossover_rate:.0%} | Mutation: {self.mutation_rate:.0%}")
            print(f"   Elite: {self.elite_size} | Tournament: {self.tournament_size}")
            print("═" * 65 + "\n")
        
        # Evaluate initial population
        self._evaluate_population(self.population, df)
        
        for gen in range(self.n_generations):
            gen_start = time.time()
            
            # Adaptive mutation: increase when diversity is low
            diversity = self._population_diversity(self.population)
            adaptive_mutation = self.mutation_rate * (1 + max(0, 0.3 - diversity) * 5)
            
            # Elitism: preserve top individuals
            elite = self._get_elite(self.population)
            
            # Create next generation
            next_gen = elite.copy()
            
            while len(next_gen) < self.population_size:
                parent1 = self._tournament_select(self.population)
                parent2 = self._tournament_select(self.population)
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1, adaptive_mutation)
                child2 = self._mutate(child2, adaptive_mutation)
                
                next_gen.extend([child1, child2])
            
            next_gen = next_gen[:self.population_size]
            
            # Evaluate new individuals only (not elite, already evaluated)
            new_individuals = next_gen[self.elite_size:]
            self._evaluate_population(new_individuals, df)
            
            self.population = next_gen
            
            # Statistics
            fitnesses = [ind.fitness for ind in self.population if ind.fitness > -999]
            
            if fitnesses:
                best_ind = max(self.population, key=lambda x: x.fitness)
                best_ind.generation = gen
                
                if self.best_ever is None or best_ind.fitness > self.best_ever.fitness:
                    self.best_ever = best_ind.clone()
                
                stats = GenerationStats(
                    generation=gen + 1,
                    best_fitness=max(fitnesses),
                    avg_fitness=np.mean(fitnesses),
                    worst_fitness=min(fitnesses),
                    best_individual=best_ind.clone(),
                    population_diversity=diversity,
                    elapsed_time=time.time() - self.start_time
                )
                self.generation_stats.append(stats)
                
                if self.verbose:
                    elapsed = time.time() - gen_start
                    print(
                        f"  Gen {gen+1:>3}/{self.n_generations} │ "
                        f"Best: {stats.best_fitness:+.4f} │ "
                        f"Avg: {stats.avg_fitness:+.4f} │ "
                        f"Diversity: {diversity:.3f} │ "
                        f"Mut: {adaptive_mutation:.3f} │ "
                        f"{elapsed:.1f}s"
                    )
        
        if self.verbose:
            print("\n" + "═" * 65)
            print(f"   ✅ Evolution complete! Best fitness: {self.best_ever.fitness:.4f}")
            print(f"   ⏱️  Total time: {time.time() - self.start_time:.1f}s")
            print("═" * 65 + "\n")
        
        return self.best_ever
    
    def _evaluate_population(self, population: List[Individual], df: pd.DataFrame):
        """Evaluate fitness for each individual in the population."""
        for ind in population:
            if ind.fitness == -999.0:  # Only evaluate unevaluated
                result = self.fitness_fn(ind.genes, df)
                ind.fitness = result.fitness
    
    def get_evolution_history(self) -> pd.DataFrame:
        """Return generation statistics as DataFrame."""
        records = []
        for s in self.generation_stats:
            records.append({
                'generation': s.generation,
                'best_fitness': s.best_fitness,
                'avg_fitness': s.avg_fitness,
                'worst_fitness': s.worst_fitness,
                'diversity': s.population_diversity,
                'elapsed_time': s.elapsed_time
            })
        return pd.DataFrame(records)
