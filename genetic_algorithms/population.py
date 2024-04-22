from abc import ABC, abstractmethod
import numpy as np


class ChromosomesCollection:
    def __init__(self, chromosome_size=16, collection_size=64):
        self.chromosome_size = chromosome_size
        self.collection_size = collection_size
        self._chromosomes = None

    @property
    def chromosomes(self):
        return self._chromosomes

    @chromosomes.setter
    def chromosomes(self, value):
        self._chromosomes = value

    @classmethod
    def from_nparray(cls, population):
        if len(population.shape) != 2:
            raise ValueError(
                "`population.shape` must equal to `(collection_size, chromosome_size)`"
            )
        collection_size, chromosome_size = population.shape
        object_ = cls(chromosome_size=chromosome_size, collection_size=collection_size)
        object_.chromosomes = population
        return object_


class Population(ABC):
    """
    A class that descripes a Population of chromosomes
    """

    def __init__(self, chromosome_size=16, population_size=64):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.chromosomes_collection = ChromosomesCollection(
            chromosome_size=chromosome_size, collection_size=population_size
        )
        self._fitness = None
        self._current_gen = 0

    @abstractmethod
    def initialize_population(self):
        pass

    @property
    def population(self):
        return self.chromosomes_collection.chromosomes

    @population.setter
    def population(self, value):
        self.chromosomes_collection.chromosomes = value

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @property
    def current_gen(self):
        return self._current_gen

    @current_gen.setter
    def current_gen(self, value):
        self._current_gen = value

    def set_next_generation(self, new_generation):
        self.population = new_generation
        self.current_gen = self.current_gen + 1


class FloatPopulation(Population):
    def initialize_population(self):
        self.population = np.random.rand(self.population_size, self.chromosome_size)


class BinaryPopulation(Population):
    def initialize_population(self, zero_probability=0.5):
        one_probability = 1.0 - zero_probability
        self.population = np.random.choice(
            [0, 1],
            (self.population_size, self.chromosome_size),
            p=[one_probability, zero_probability],
        ).astype(bool)


class SequencePopulation(Population):
    """
    Args:
        genes_values: an array like. The values that the genes can take. `genes_values` length
            must be equal to `chromosome_size`
    """

    def __init__(self, genes_values=None, **kwargs):
        super().__init__(**kwargs)
        self.genes_values = genes_values

    def initialize_population(self):
        values = np.repeat(
            np.expand_dims(self.genes_values, 0), self.population_size, axis=0
        )
        self.population = np.random.default_rng().permuted(values, axis=1)

    @property
    def genes_values(self):
        return self._genes_values

    @genes_values.setter
    def genes_values(self, value):

        if value is None:
            values = np.arange(self.chromosome_size)
        else:
            if len(value) != self.chromosome_size:
                raise ValueError("`gene_values` length must equal `chromosome_size`")
            values = np.array(value)

        self._genes_values = values


class DiscretePopulation(Population):
    """
    Args:
        discrete_values: an array of discrete values of genes.
        discrete_probabilities: an array of probabilities of discrete values. the
            array length must equal `discrete_values` length
    """

    def __init__(self, discrete_values, discrete_probabilities=None, **kwargs):
        super().__init__(**kwargs)
        self.discrete_values = discrete_values
        self.discrete_probabilities = discrete_probabilities

    def initialize_population(self):
        self.population = np.random.choice(
            self.discrete_values,
            (self.population_size, self.chromosome_size),
            p=self.discrete_probabilities,
        )

    @property
    def discrete_values(self):
        return self._discrete_values

    @discrete_values.setter
    def discrete_values(self, value):
        self._discrete_values = np.array(value)

    @property
    def discrete_probabilities(self):
        return self._discrete_probabilities

    @discrete_probabilities.setter
    def discrete_probabilities(self, value):
        if value is not None:
            value = np.array(value)
            if value.shape[0] != self.discrete_values.shape[0]:
                raise ValueError(
                    "`genes_probabilities` length must equal `genes_values` length"
                )
        self._discrete_probabilities = value
