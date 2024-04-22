from abc import ABC, abstractmethod
import numpy as np

from genetic_algorithms.population import *


class Selector(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def select(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)


class RandomSelector(Selector):
    def __call__(self, generation: Population, selection_size: int):
        indexes = np.random.randint(generation.population_size, size=selection_size)
        return ChromosomesCollection.from_nparray(generation.population[indexes, :])


class TruncationSelector(Selector):
    def __call__(self, generation, selection_size: int):
        if selection_size > generation.population_size:
            raise ValueError(
                "`selection_size` must be less than or equal to"
                " `generation.population_size`"
            )
        sort_indexes = np.argsort(generation.fitness)[::-1]
        return ChromosomesCollection.from_nparray(
            generation.population[sort_indexes, :][:selection_size, :]
        )


class TournamentSelector(Selector):
    def __call__(self, generation, selection_size: int, tournament_size: int = None):
        indexes = np.arange(generation.population_size)
        indexes = np.repeat(np.expand_dims(indexes, 0), selection_size, axis=0)
        random_indexes = (
            np.random.default_rng()
            .permuted(indexes, axis=1)[:, :tournament_size]
            .reshape(selection_size, tournament_size)
        )
        fitness_values = generation.fitness[random_indexes]
        max_fitnesses_indexes = np.argmax(fitness_values, axis=1)
        chromosomes_indexes = random_indexes[
            np.arange(selection_size), max_fitnesses_indexes
        ]
        return ChromosomesCollection.from_nparray(
            generation.population[chromosomes_indexes]
        )


class ProportionalSelection(Selector):
    def __call__(self, generation, selection_size: int):
        probabilities = generation.fitness / np.sum(generation.fitness)
        if selection_size > generation.population_size:
            raise ValueError(
                "`selection_size` must be less than or equal to"
                "`generation.population_size`"
            )
        selection_indexes = np.random.choice(
            np.arange(generation.population_size),
            size=selection_size,
            replace=False,
            p=probabilities,
        )
        return ChromosomesCollection.from_nparray(
            generation.population[selection_indexes]
        )
