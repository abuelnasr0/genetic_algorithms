from genetic_algorithms import population, selection, crossover, mutation
import numpy as np
import matplotlib.pyplot as plt

from sys import maxsize 
from itertools import permutations

class SalesmanProblem:
    def __init__(self, num_cities=900, boundry=10.0):
        self.num_cities = num_cities
        self.boundry = boundry
        self.initialize_cities()

    def initialize_cities(self):
        self.cities = np.random.rand(self.num_cities, 2) * self.boundry
        self.cities_distances = np.sqrt(
            np.sum(
                np.square(
                    np.expand_dims(self.cities, 0) - np.expand_dims(self.cities, 1)
                ),
                axis=-1,
            )
        )

    def solve_problem(self, max_generation=256, population_size=128):
        # Initialize the population
        kwargs = {
            "chromosome_size": self.num_cities,
            "population_size": population_size,
        }
        # For The main population
        pop = population.SequencePopulation(**kwargs)
        pop.initialize_population()
        # initialize necessery classes
        selector = selection.TournamentSelector()
        mutator = mutation.ReverseSequenceMutation()
        # truncate selector for elitism
        truncate_selector = selection.TruncationSelector()
        num_keep_elite = 4
        # Save history
        history = []
        for i in range(max_generation):
            # Select parents
            pop = self.calculate_fitness(pop)
            # Mutation
            offsprings = selector(pop, population_size - num_keep_elite , 4)
            offsprings = mutator(offsprings)
            # Keep the best (elitism)
            # get best individuals in parents
            best_individuals = truncate_selector(pop, num_keep_elite)
            pop.population = np.concatenate(
                [offsprings.chromosomes, best_individuals.chromosomes], axis=0
            )
            history.append(
                self.convert_fitness_to_distance((np.flip(np.sort(pop.fitness))[0]))
            )

        sorted_indexes = np.flip(np.argsort(pop.fitness))

        return history, pop.population[sorted_indexes[0]]

    def convert_fitness_to_distance(self, fitness):
        return self.num_cities**2 / fitness

    def convert_distance_to_fitness(self, distance):
        return self.num_cities**2 / distance

    def calculate_fitness(self, population: population.Population):
        # Will generate
        # [[0, 1],
        # [1, 2],
        # [2, 3],
        # [3, 0]]
        population_indexes = np.c_[
            [
                np.arange(population.chromosome_size),
                np.roll(np.arange(population.chromosome_size), -1),
            ]
        ].T

        city_i_j = population.population[:, population_indexes]
        distance = np.sum(
            np.squeeze(self.cities_distances[city_i_j[..., :1], city_i_j[..., 1:]]),
            axis=-1,
        )
        population.fitness = self.convert_distance_to_fitness(distance)

        return population

    def optimal(self, s=0): 
 
        # store all vertex apart from source vertex 
        vertex = [] 
        for i in range(self.num_cities): 
            if i != s: 
                vertex.append(i) 
    
        # store minimum weight Hamiltonian Cycle 
        min_path = maxsize 
        next_permutation=permutations(vertex)
        for i in next_permutation:
    
            # store current Path weight(cost) 
            current_pathweight = 0
    
            # compute current path weight 
            k = s 
            for j in i: 
                current_pathweight += self.cities_distances[k][j] 
                k = j 
            current_pathweight += self.cities_distances[k][s] 
    
            # update minimum 
            min_path = min(min_path, current_pathweight) 
            
        return min_path 
    



if __name__ == "__main__":
    # Solve small problem and compare to optimal solution
    num_cities = 10
    sales = SalesmanProblem(num_cities=num_cities)
    history, best_solution = sales.solve_problem(max_generation=128, population_size=32)
    optimal_solution = sales.optimal()
    print("************Small Problem************")
    print("Genetic algorithm solution = ", best_solution)
    print("Genetic algorithm distance = ", history[-1])
    print("optimal distance = ", optimal_solution)

    # Create figure for two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs.flatten()
    
    # Draw the genetic algorithm path
    axs[0].scatter(sales.cities[:, 0], sales.cities[:, 1], c='r')
    plt_xs = sales.cities[best_solution, 0]
    plt_xs = np.hstack([plt_xs, plt_xs[:1]])
    plt_ys = sales.cities[best_solution, 1]
    plt_ys = np.hstack([plt_ys, plt_ys[:1]])

    axs[0].plot(plt_xs, plt_ys)

    axs[0].set_title(f"Number of cities = {num_cities}")
    # axs[0].imshow()

    # Solve large problem and draw it
    num_cities = 64
    sales = SalesmanProblem(num_cities=num_cities)
    history, best_solution = sales.solve_problem(max_generation=512, population_size=512)
    print("************Large Problem************")
    print("Genetic algorithm solution = ", best_solution)
    print("Genetic algorithm distance = ", history[-1])
    
    # Draw the genetic algorithm path
    axs[1].scatter(sales.cities[:, 0], sales.cities[:, 1], c='r')
    plt_xs = sales.cities[best_solution, 0]
    plt_xs = np.hstack([plt_xs, plt_xs[:1]])
    plt_ys = sales.cities[best_solution, 1]
    plt_ys = np.hstack([plt_ys, plt_ys[:1]])
    axs[1].plot(plt_xs, plt_ys)
    axs[1].set_title(f"Number of cities = {num_cities}")
    # axs[1].imshow()

    plt.show()






