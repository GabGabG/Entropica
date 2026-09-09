import cupy as cp
from cupy.typing import ArrayLike
from cupyx.scipy.special import digamma

from ..backends.cupy import knn_statistics
from .base import BaseEstimator


class KNNMutualInformation(BaseEstimator):
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
        super().__init__(dtype, random_state)

    def _noisy_data(self, data: cp.ndarray) -> cp.ndarray:
        noisy_data = data + self._noise_intensity * self._random_state.standard_normal(
            size=data.shape, dtype=self._dtype
        )
        return noisy_data

    @staticmethod
    def _as_batched(data: cp.ndarray) -> cp.ndarray:
        if data.ndim == 1:
            # shape (N,), implicit batchsize = 1 and n vars = 1
            return data[:, None, None]

        if data.ndim == 2:
            # shape (N, V), implicit batchsize = 1 (n vars = V)
            return data[:, None, :]

        if data.ndim == 3:
            return data

        raise ValueError("data must have one, two or three dimensions")

    @staticmethod
    def _restore_shape(mi: cp.ndarray, x_was_1D: bool, y_was_1D: bool) -> cp.ndarray:
        if x_was_1D and y_was_1D:
            return mi[0, 0, 0]
        if x_was_1D:
            return mi[:, 0, :]
        if y_was_1D:
            return mi[:, :, 0]
        return mi

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

        x_was_1D = x.ndim == 1
        y_was_1D = y.ndim == 1

        x = self._as_batched(x)
        y = self._as_batched(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples.")

        n_samples = x.shape[0]

        bx = x.shape[1]
        by = y.shape[1]

        dx = x.shape[2]
        dy = y.shape[2]

        if bx != by:
            if bx == 1:
                x = cp.broadcast_to(x, (n_samples, by, dx))
            elif by == 1:
                y = cp.broadcast_to(y, (n_samples, bx, dy))
            else:
                msg = "Batch dimensions of x and y must match, or one of them must be one."
                msg += f"\nGot {bx} for x and {by} for y."
                raise ValueError(msg)
        batch_size = max(bx, by)

        if self._add_noise:
            x = self._noisy_data(x)
            y = self._noisy_data(y)

        # Build all (batch size, variable_x, variable_y) pairs
        # x_pairs, y_pairs: (B * dx * dy, N)

        idx_x, idx_y = cp.meshgrid(cp.arange(dx), cp.arange(dy), indexing="ij")
        idx_x = idx_x.ravel()
        idx_y = idx_y.ravel()

        x_pairs = x[:, :, idx_x]
        y_pairs = y[:, :, idx_y]

        x_pairs = cp.transpose(x_pairs, (1, 2, 0))
        y_pairs = cp.transpose(y_pairs, (1, 2, 0))

        x_pairs = x_pairs.reshape(batch_size * dx * dy, n_samples)
        y_pairs = y_pairs.reshape(batch_size * dx * dy, n_samples)

        x_pairs = cp.ascontiguousarray(x_pairs)
        y_pairs = cp.ascontiguousarray(y_pairs)

        mi = self._compute_from_pairs(x_pairs, y_pairs, n_samples)
        mi = mi.reshape(batch_size, dx, dy)
        mi = self._restore_shape(mi, x_was_1D, y_was_1D)

        return mi

    def compute_pairwise(self, data: ArrayLike) -> cp.ndarray:
        data = cp.asarray(data, dtype=self._dtype)

        dim = data.ndim
        if dim not in (2, 3):
            raise ValueError("data must be two-dimensional or three-dimensional.")
        if dim == 3:
            m, n1, n2 = data.shape
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
