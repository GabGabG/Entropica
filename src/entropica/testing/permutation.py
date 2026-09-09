from dataclasses import dataclass

from cupy.typing import ArrayLike
import cupy as cp

from ..estimators.base import BaseEstimator
from .strategies import PermutationStrategy, YPermutation


@dataclass
class PermutationTestResult:
    statistic: cp.ndarray
    pvalue: cp.ndarray
    null_distribution: cp.ndarray


class PermutationTest:

    def __init__(self, estimator: BaseEstimator, strategy: PermutationStrategy | None = None,
                 n_permutations: int = 1_000, batch_size: int = 128,
                 random_state: int | cp.random.Generator | None = None, *estimator_compute_args,
                 **estimator_compute_kwargs):
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("estimator must be an instance of BaseEstimator.")
        self._estimator = estimator

        if strategy is None:
            strategy = YPermutation()
        elif not isinstance(strategy, YPermutation):
            raise TypeError("strategy must be an instance of PermutationStrategy.")
        self._strategy = strategy

        if n_permutations < 1:
            raise ValueError("n_permutations must be at least one.")
        self._n_permutations = n_permutations

        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        self._batch_size = batch_size

        self.random_state = random_state
        self._args = estimator_compute_args
        self._kwargs = estimator_compute_kwargs

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

    @property
    def strategy(self) -> PermutationStrategy:
        return self._strategy

    @property
    def n_permutations(self) -> int:
        return self._n_permutations

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def random_state(self) -> cp.random.Generator:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: int | cp.random.Generator | None):
        if random_state is None:
            self._random_state = cp.random.default_rng()
        elif isinstance(random_state, int):
            self._random_state = cp.random.default_rng(random_state)
        elif isinstance(random_state, cp.random.Generator):
            self._random_state = random_state
        else:
            raise TypeError(f"Unknown random_state type: {type(random_state)}")

    def compute(self, x: ArrayLike, y: ArrayLike) -> PermutationTestResult:

        statistic = self._estimator.compute(x, y, *self._args, **self._kwargs)
        null_distribution = cp.empty((self._n_permutations, *statistic.shape), dtype=statistic.dtype)

        for start in range(0, self._n_permutations, self._batch_size):
            stop = min(start + self._batch_size, self._n_permutations)
            current_batch_size = stop - start

            x_permuted, y_permuted = self._strategy.permute(x, y, current_batch_size, self.random_state)

            statistics_permuted = self._estimator.compute(x_permuted, y_permuted, *self._args, **self._kwargs)
            null_distribution[start:stop] = statistics_permuted
