import cupy as cp
from cupy.typing import ArrayLike
from cupyx.scipy.special import digamma

from ..backends.cupy import knn_statistics


class KNNMutualInformation:
    def __init__(
            self,
            k: int = 3,
            add_noise: bool = True,
            noise_intensity: float = 1e-8,
            dtype: cp.dtype = cp.float32,
            random_state: int | cp.random.Generator | None = None,
    ):
        if k < 1:
            raise ValueError("k must be at least one.")
        self._k = k
        self._add_noise = add_noise
        self._noise_intensity = noise_intensity
        dtype = cp.dtype(dtype)

        if dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
            msg = f"dtype must be float32 or float64, got {dtype}"
            raise TypeError(msg)
        self._dtype = dtype
        self.random_state = random_state

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

    def _noisy_data(self, data: cp.ndarray) -> cp.ndarray:
        noisy_data = data + self._noise_intensity * self._random_state.standard_normal(
            size=data.shape, dtype=self._dtype
        )
        return noisy_data

    def _compute_from_pairs(
            self, x_pairs: cp.ndarray, y_pairs: cp.ndarray, n_samples: int
    ) -> cp.ndarray:
        nx, ny = knn_statistics(x_pairs, y_pairs, self._k)

        c = digamma(self._k).astype(self._dtype)
        d = digamma(n_samples).astype(self._dtype)
        avg_digamma = cp.mean(
            digamma(nx).astype(self._dtype) + digamma(ny).astype(self._dtype), axis=1
        )
        return c + d - avg_digamma

    def compute(self, x: ArrayLike, y: ArrayLike) -> cp.ndarray:
        x = cp.asarray(x, dtype=self._dtype)
        y = cp.asarray(y, dtype=self._dtype)

        if x.ndim not in (1, 2) or y.ndim not in (1, 2):
            raise ValueError("x and y must be one- or two-dimensional.")

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples.")

        n_samples = x.shape[0]

        # (N,) -> (N, 1)
        if x.ndim == 1:
            x = x[:, None]

        # (N,) -> (N, 1)
        if y.ndim == 1:
            y = y[:, None]

        if x.shape[1] == 1:
            x = cp.broadcast_to(x, shape=(n_samples, y.shape[1]))
        elif y.shape[1] == 1:
            y = cp.broadcast_to(y, shape=(n_samples, x.shape[1]))
        elif x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same number of features or one must have one feature.")

        if self._add_noise:
            x = self._noisy_data(x)
            y = self._noisy_data(y)
        x_pairs = cp.ascontiguousarray(x.T)
        y_pairs = cp.ascontiguousarray(y.T)
        print(x_pairs.shape, y_pairs.shape)

        mi = self._compute_from_pairs(x_pairs, y_pairs, n_samples)
        return mi

    def compute_pairwise(self, data: ArrayLike) -> cp.ndarray:
        data = cp.asarray(data, dtype=self._dtype)

        dim = data.ndim
        if dim not in (2, 3):
            raise ValueError("data must be two-dimensional or three-dimensional.")
        if dim == 3:
            m, n1, n2 = data.shape
            if n1 != n2:
                raise ValueError(
                    "data must be square in its last two dimensions if 3D."
                )
            data = data.reshape(m, n1 * n2)

        n_samples, n_variables = data.shape
        if self._add_noise:
            data = self._noisy_data(data)

        idx_i, idx_j = cp.triu_indices(n_variables, k=1)

        n_pairs = len(idx_i)

        if n_pairs == 0:
            return cp.zeros((n_variables, n_variables), dtype=self._dtype)

        x_pairs = cp.ascontiguousarray(data[:, idx_i].T)
        y_pairs = cp.ascontiguousarray(data[:, idx_j].T)

        mi_scores = self._compute_from_pairs(x_pairs, y_pairs, n_samples)

        mi_matrix = cp.full((n_variables, n_variables), cp.nan, dtype=self._dtype)

        mi_matrix[idx_i, idx_j] = mi_scores
        mi_matrix[idx_j, idx_i] = mi_scores
        return mi_matrix
