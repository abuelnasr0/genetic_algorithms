from abc import ABC, abstractmethod
import numpy as np
from genetic_algorithms import population


class Mutation(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BitFlipMutation(Mutation):
    def __call__(
        self,
        offsprings: population.ChromosomesCollection,
        mutation_probability: float = 0.2,
    ):
        mutation_mask = (
            np.random.rand(offsprings.collection_size, offsprings.chromosome_size)
            <= mutation_probability
        )
        new_offsprings = np.where(
            mutation_mask, ~offsprings.chromosomes, offsprings.chromosomes
        )
        return population.ChromosomesCollection.from_nparray(new_offsprings)


class CreepMutation(Mutation):
    """
    For float
    """

    def __call__(
        self,
        offsprings: population.ChromosomesCollection,
        step_size: float = 0.05,
        mutation_probability: float = 0.6,
        add_random_to_offspring: bool = False,
    ):
        mutation_mask = (
            np.random.rand(offsprings.collection_size, offsprings.chromosome_size)
            <= mutation_probability
        )
        random_value = (
            np.random.rand(offsprings.collection_size, offsprings.chromosome_size)
            * step_size
        )
        random_value = (
            random_value + offsprings.chromosomes
            if add_random_to_offspring
            else random_value
        )
        new_offsprings = np.where(mutation_mask, random_value, offsprings.chromosomes)
        return population.ChromosomesCollection.from_nparray(new_offsprings)


class ReverseSequenceMutation(Mutation):
    """
    For sequence and may be discrete and binary
    """

    def __call__(
        self,
        offsprings: population.ChromosomesCollection,
        mutation_probability: float = 0.75,
    ):
        mutation_mask = (
            np.random.rand(offsprings.collection_size) <= mutation_probability
        )
        ranges = np.random.randint(
            offsprings.chromosome_size,
            size=(2, offsprings.collection_size),
        )
        ranges = np.sort(ranges, axis=0) * np.expand_dims(mutation_mask, 0)
        new_offsprings = self._reverse_ranges(offsprings, ranges)

        return population.ChromosomesCollection.from_nparray(new_offsprings)

    @staticmethod
    def _reverse_ranges(
        offsprings: population.ChromosomesCollection,
        ranges: np.ndarray,
    ):
        # Reverse the array
        reversed = np.flip(offsprings.chromosomes, axis=-1)
        # Shift each row by the proper amount
        starts, ends = ranges[0], ranges[1]
        shift_steps = np.expand_dims(starts + ends + 1, 1)
        row_shifted = reversed[
            np.c_[: reversed.shape[0]],
            (np.r_[: reversed.shape[1]] - shift_steps) % reversed.shape[1],
        ]
        indexes_range = np.repeat(
            np.expand_dims(np.arange(offsprings.chromosome_size), 0),
            offsprings.collection_size,
            axis=0,
        )
        mask = (indexes_range >= np.expand_dims(starts, 1)) & (
            indexes_range <= np.expand_dims(ends, 1)
        )
        new_offsprings = np.where(mask, row_shifted, offsprings.chromosomes)
        return new_offsprings
