from abc import ABC, abstractmethod

import cupy as cp


class BaseEstimator(ABC):
    def __init__(self, dtype: cp.dtype, random_state: int | cp.random.Generator | None):
        self._dtype = dtype
        self.random_state = random_state

    @property
    def dtype(self) -> cp.dtype:
        return self._dtype

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

    @abstractmethod
    def compute(self, *args, **kwargs) -> cp.ndarray:
        raise NotImplementedError("To implement in derived classes.")
