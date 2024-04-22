from genetic_algorithms import population, selection, crossover, mutation
import numpy as np


class KnapsackProblem:
    def __init__(self, num_items=900, max_weight=500):
        self.num_items = num_items
        self.max_weight = max_weight
        self.initialize_items()

    def initialize_items(self):
        self.weights = np.random.randint(
            2 * self.max_weight / self.num_items, size=(1, self.num_items)
        )
        self.values = np.random.randint(20, size=(1, self.num_items))

    def solve_problem(self, max_generation=256, population_size=128):
        # Initialize the population
        kwargs = {"chromosome_size": self.num_items, "population_size": population_size}
        # For The main population
        bin_pop = population.BinaryPopulation(**kwargs)
        bin_pop.initialize_population()
        # initialize necessery classes
        selector = selection.TournamentSelector()
        cross_over = crossover.UniformCrossover()
        mutator = mutation.ReverseSequenceMutation()
        # truncate selector for elitism
        truncate_selector = selection.TruncationSelector()
        num_keep_elite = 4 
        # Save history
        history = []
        for i in range(max_generation):
            # Select parents
            bin_pop = self.calculate_fitness(bin_pop)
            parents_a = selector(bin_pop, population_size // 2 - num_keep_elite // 2, 4)
            parents_b = selector(bin_pop, population_size // 2 - num_keep_elite // 2, 4)
            # Crossover parents
            offsprings = cross_over(parents_a, parents_b)
            # Mutation
            offsprings = mutator(offsprings)
            # Keep the best (elitism)
            # get best individuals in parents
            best_individuals = truncate_selector(bin_pop, num_keep_elite)
            bin_pop.population = np.concatenate(
                [offsprings.chromosomes, best_individuals.chromosomes], axis=0
            )
            history.append(int(np.flip(np.sort(bin_pop.fitness))[0]))

        return history

    def calculate_fitness(self, population: population.Population):
        total_weights = np.sum(population.population * self.weights, axis=1)
        over_weight_mask = total_weights > self.max_weight
        total_value = np.sum(population.population * self.values, axis=1)
        fitness = np.where(over_weight_mask, 0, total_value)
        population.fitness = fitness
        return population

    def optimal(self, wt, val, W, n): 
        # Copied from GeeksForGeeks
        # base conditions 
        if n == 0 or W == 0: 
            return 0
        if self.t[n][W] != -1: 
            return self.t[n][W] 
    
        # choice diagram code 
        if wt[n-1] <= W: 
            self.t[n][W] = max( 
                val[n-1] + self.optimal( 
                    wt, val, W-wt[n-1], n-1), 
                self.optimal(wt, val, W, n-1)) 
            return self.t[n][W] 
        elif wt[n-1] > W: 
            self.t[n][W] = self.optimal(wt, val, W, n-1) 
            return self.t[n][W] 


if __name__ == "__main__":
    knapsack_problem = KnapsackProblem()
    history = knapsack_problem.solve_problem()
    print(history)
    profit = list(knapsack_problem.values.squeeze())
    weight = list(knapsack_problem.weights.squeeze())
    W = knapsack_problem.max_weight
    n = len(profit)
    knapsack_problem.t = [[-1 for i in range(W + 1)] for j in range(n + 1)] 
    optimal = knapsack_problem.optimal(weight, profit, W, n)
    print("optimal = ", optimal)
    if optimal == history[-1]:
        print("Optimal found")
