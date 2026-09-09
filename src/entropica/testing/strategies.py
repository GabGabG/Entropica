from abc import ABC, abstractmethod

import cupy as cp
from cupy.typing import ArrayLike


class PermutationStrategy(ABC):
    @staticmethod
    def _set_random_state(random_state: int | cp.Generator | None) -> cp.random.Generator:
        if random_state is None:
            random_state = cp.random.default_rng()
        elif isinstance(random_state, int):
            random_state = cp.random.default_rng(random_state)
        elif isinstance(random_state, cp.random.Generator):
            random_state = random_state
        else:
            raise TypeError(f"Unknown random_state type: {type(random_state)}")
        return random_state

    @abstractmethod
    def permute(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_permutations: int,
        random_state: int | cp.generator | None,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        raise NotImplementedError("To implement in derived classes.")


class YPermutation(PermutationStrategy):
    def permute(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_permutations: int,
        random_state: int | cp.generator | None,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        if n_permutations < 1:
            raise ValueError("The number of permutations must be at least 1.")
        x = cp.asarray(x)
        y = cp.asarray(y)

        if y.ndim not in (1, 2):
            raise ValueError("y must be one- or two-dimensional.")

        n_samples = y.shape[0]

        keys = self._set_random_state(random_state).random(size=(n_samples, n_permutations))
        indices = cp.argsort(keys, axis=0)

        if y.ndim == 1:
            y = y[:, None]

        y_permuted = y[indices]

        return x, y_permuted
