from abc import ABC, abstractmethod
import numpy as np
from genetic_algorithms import population


class Crossover(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class OnePointCrossover(Crossover):
    def __call__(
        self,
        parents_a: population.ChromosomesCollection,
        parents_b: population.ChromosomesCollection,
        accepted_probability: float = 0.5,
        x_point: int = None,
    ) -> population.ChromosomesCollection:
        if parents_a.chromosome_size != parents_b.chromosome_size:
            raise ValueError("sizes must match")

        if x_point is None:
            x_point = np.random.randint(0, parents_a.chromosome_size)
        else:
            if x_point < 0 or x_point >= parents_a.chromosome_size:
                raise ValueError("point must be right")

        selection_probability = np.random.rand(parents_a.collection_size)
        accepted_mask = selection_probability < accepted_probability
        accepted_parents_a = parents_a.chromosomes[accepted_mask]
        accepted_parents_b = parents_b.chromosomes[accepted_mask]
        declined_parents_a = parents_a.chromosomes[~accepted_mask]
        declined_parents_b = parents_b.chromosomes[~accepted_mask]

        # Accepted parents mutation
        childs_a = np.concatenate(
            [
                accepted_parents_a[:, :x_point],
                accepted_parents_b[:, x_point:],
            ],
            axis=1,
        )
        childs_b = np.concatenate(
            [
                accepted_parents_a[:, :x_point],
                accepted_parents_b[:, x_point:],
            ],
            axis=1,
        )

        return population.ChromosomesCollection.from_nparray(
            np.concatenate(
                [childs_a, childs_b, declined_parents_a, declined_parents_b], axis=0
            )
        )


class UniformCrossover(Crossover):
    def __call__(
        self,
        parents_a: population.ChromosomesCollection,
        parents_b: population.ChromosomesCollection,
        accepted_probability: float = 0.5,
        ones_probability: float = 0.5,
    ) -> population.ChromosomesCollection:
        if parents_a.chromosome_size != parents_b.chromosome_size:
            raise ValueError("sizes must match")

        selection_probability = np.random.rand(parents_a.collection_size)
        accepted_mask = selection_probability <= accepted_probability
        accepted_parents_a = parents_a.chromosomes[accepted_mask]
        accepted_parents_b = parents_b.chromosomes[accepted_mask]
        declined_parents_a = parents_a.chromosomes[~accepted_mask]
        declined_parents_b = parents_b.chromosomes[~accepted_mask]

        crossover_mask = np.random.rand(*accepted_parents_a.shape) <= ones_probability
        childs_a = np.where(crossover_mask, accepted_parents_a, accepted_parents_b)
        childs_b = np.where(crossover_mask, accepted_parents_b, accepted_parents_a)

        return population.ChromosomesCollection.from_nparray(
            np.concatenate(
                [childs_a, childs_b, declined_parents_a, declined_parents_b], axis=0
            )
        )


class HeuristicCrossover(Crossover):
    """
    for Float
    """

    def __call__(
        self,
        parents_a: population.ChromosomesCollection,
        parents_b: population.ChromosomesCollection,
    ) -> population.ChromosomesCollection:
        if parents_a.chromosome_size != parents_b.chromosome_size:
            raise ValueError("sizes must match")

        random_parameter = np.random.rand(parents_a.collection_size)
        childs = parents_a.chromosomes + np.expand_dims(random_parameter, 1) * np.abs(
            parents_a.chromosomes - parents_b.chromosomes
        )

        return population.ChromosomesCollection.from_nparray(childs)


class FlatCrossover(Crossover):
    """
    for Float
    """

    def __call__(
        self,
        parents_a: population.ChromosomesCollection,
        parents_b: population.ChromosomesCollection,
    ) -> population.ChromosomesCollection:
        if parents_a.chromosome_size != parents_b.chromosome_size:
            raise ValueError("sizes must match")

        random_parameter = np.random.rand(
            parents_a.collection_size, parents_a.chromosome_size
        )
        childs = random_parameter * np.abs(
            parents_a.chromosomes - parents_b.chromosomes
        ) + np.min(
            np.concatenate(
                [
                    np.expand_dims(parents_a.chromosomes, -1),
                    np.expand_dims(parents_b.chromosomes, -1),
                ],
                axis=-1,
            ),
            axis=-1,
        )

        return population.ChromosomesCollection.from_nparray(childs)
