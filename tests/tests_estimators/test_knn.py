from collections import namedtuple

import cupy as cp
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.special import digamma

from entropica.estimators.knn import KNNMutualInformation


def mutual_information_kdtree(x: np.ndarray, y: np.ndarray, k: int) -> float:
    n_samples = x.shape[0]
    points = np.column_stack((x, y))

    tree = cKDTree(points)
    distances = tree.query(points, k=k + 1, p=np.inf)[0]

    radius = distances[:, k]

    x_tree = cKDTree(x[:, None])
    y_tree = cKDTree(y[:, None])

    nx = x_tree.query_ball_point(x[:, None], radius, p=np.inf, return_length=True)
    ny = y_tree.query_ball_point(y[:, None], radius, p=np.inf, return_length=True)

    return digamma(k) + digamma(n_samples) - np.mean(digamma(nx) + digamma(ny))




class TestKNNMutualInformation:

    def test_obj_possible(self):
        try:
            KNNMutualInformation()
        except Exception as e:
            pytest.fail(f"Exception raised:\m{e}")

    def test_default_attributes(self):
        obj = KNNMutualInformation()
        assert obj._k == 3
        assert obj._add_noise
        assert obj._noise_intensity == 1e-8
        assert obj._dtype == cp.float32
        assert isinstance(obj._random_state, cp.random.Generator)

    @pytest.mark.parametrize("k", range(-10, 0))
    def test_k_is_less_than_one(self, k: int):
        msg = "k must be at least one."
        with pytest.raises(ValueError, match=msg):
            KNNMutualInformation(k=k)

    @pytest.mark.parametrize("dtype", [np.single, np.double])
    def test_numpy_floats_ok(self, dtype: np.dtype):
        try:
            KNNMutualInformation(dtype=dtype)
        except Exception as e:
            pytest.fail(f"Exception raised:\n{e}")

    @pytest.mark.parametrize("dtype", [np.complex64, cp.long, np.bool, cp.float16])
    def test_wrong_dtypes(self, dtype: np.dtype):
        msg = f"dtype must be float32 or float64, got {cp.dtype(dtype)}"
        with pytest.raises(TypeError, match=msg):
            KNNMutualInformation(dtype=dtype)

    def test_random_state_getter(self):
        obj = KNNMutualInformation()
        assert isinstance(obj.random_state, cp.random.Generator)
        assert obj._random_state == obj.random_state

    def test_random_state_setter_none(self):
        obj = KNNMutualInformation()
        initial_gen = obj.random_state
        obj.random_state = None
        final_gen = obj.random_state
        assert isinstance(final_gen, cp.random.Generator)
        assert final_gen != initial_gen

    @pytest.mark.parametrize("seed", range(0, 10))
    def test_random_state_setter_seed(self, seed:int):
        obj = KNNMutualInformation()
        obj.random_state = seed
        assert isinstance(obj.random_state, cp.random.Generator)

    def test_random_state_setter_random_state(self):
        obj = KNNMutualInformation()
        initial_gen = obj.random_state
        obj.random_state = initial_gen
        assert isinstance(obj.random_state, cp.random.Generator)
        assert obj.random_state == initial_gen

    def test_random_state_setter_unknown_type(self):
        obj = KNNMutualInformation()
        random_state = "I am a random state"
        msg = f"Unknown random_state type: {type(random_state)}"
        with pytest.raises(TypeError, match=msg):
            obj.random_state = random_state

    @pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
    @pytest.mark.parametrize("intensity", [-2, -1, 1, 2])
    def test_noisy_data(self, dtype:cp.dtype, intensity:int):
        data = cp.random.standard_normal((10, 1_000), dtype=dtype)
        obj = KNNMutualInformation(noise_intensity=intensity)
        noisy = obj._noisy_data(data)
        mean = noisy.mean()
        std = noisy.std()
        cp.testing.assert_allclose(mean, data.mean(), atol=1e-3, rtol=cp.inf)
        cp.testing.assert_allclose(std, data.std() * intensity, atol=1e-3, rtol=cp.inf)
        assert noisy.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_compute_from_pairs(self, dtype:np.dtype, k:int):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, 100), dtype=dtype)
        y = rng.standard_normal((1, 100), dtype=dtype)
        obj = KNNMutualInformation(k=k, add_noise=False, dtype=dtype)
        mi_gpu = obj._compute_from_pairs(cp.asarray(x), cp.asarray(y), 100)[0]
        mi_cpu = mutual_information_kdtree(x[0], y[0], k=k)
        cp.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-5, atol=1e-6)

    def test_compute_wrong_x_dim(self):
        obj = KNNMutualInformation()
        msg = "x and y must be one-dimensional."
        with pytest.raises(ValueError, match=msg):
            obj.compute(cp.arange(100).reshape(1, 100), cp.arange(100))

    def test_compute_wrong_y_dim(self):
        obj = KNNMutualInformation()
        msg = "x and y must be one-dimensional."
        with pytest.raises(ValueError, match=msg):
            obj.compute(cp.arange(100), cp.arange(100).reshape(1, 100))

    def test_compute_shape_mismatch(self):
        obj = KNNMutualInformation()
        msg = "x and y must have the same shape."
        with pytest.raises(ValueError, match=msg):
            obj.compute(cp.arange(100), cp.arange(99))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_compute_no_noise(self, dtype: np.dtype, k:int):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100, dtype=dtype)
        y = rng.standard_normal(100, dtype=dtype)
        obj = KNNMutualInformation(k=k, add_noise=False, dtype=dtype)
        mi_gpu = obj.compute(cp.asarray(x), cp.asarray(y))
        mi_cpu = mutual_information_kdtree(x, y, k=k)
        cp.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_compute_noise(self, dtype: np.dtype, k:int):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100, dtype=dtype)
        y = rng.standard_normal(100, dtype=dtype)
        obj = KNNMutualInformation(k=k, add_noise=True, dtype=dtype)
        mi_gpu = obj.compute(cp.asarray(x), cp.asarray(y))
        mi_cpu = mutual_information_kdtree(x, y, k=k)
        cp.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-5, atol=1e-6)

    def test_compute_cross_wrong_x_dim(self):
        obj = KNNMutualInformation()
        msg = "x and y must be two-dimensional."
        with pytest.raises(ValueError, match=msg):
            obj.compute_cross(cp.arange(100), cp.arange(100).reshape(10, 10))

    def test_compute_cross_wrong_y_dim(self):
        obj = KNNMutualInformation()
        msg = "x and y must be two-dimensional."
        with pytest.raises(ValueError, match=msg):
            obj.compute_cross(cp.arange(100).reshape(10, 10), cp.arange(100))

    def test_compute_cross_samples_mismatch(self):
        obj = KNNMutualInformation()
        msg = "x and y must have the same number of samples."
        with pytest.raises(ValueError, match=msg):
            obj.compute_cross(cp.arange(100).reshape(10, 10), cp.arange(90).reshape(9, 10))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_compute_cross_no_noise(self, dtype: np.dtype, k:int):
        rng = np.random.default_rng(0)
        nx = 3
        ny = 2
        x = rng.standard_normal((100, nx), dtype=dtype)
        y = rng.standard_normal((100, ny), dtype=dtype)
        obj = KNNMutualInformation(k=k, add_noise=False, dtype=dtype)
        mi_gpu = obj.compute_cross(cp.asarray(x), cp.asarray(y))
        mi_cpu = np.empty((nx, ny), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                mi_cpu[i, j] = mutual_information_kdtree(x[:, i], y[:, j], k=k)
        cp.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_compute_cross_noise(self, dtype: np.dtype, k:int):
        rng = np.random.default_rng(0)
        nx = 3
        ny = 2
        x = rng.standard_normal((100, nx), dtype=dtype)
        y = rng.standard_normal((100, ny), dtype=dtype)
        obj = KNNMutualInformation(k=k, add_noise=True, dtype=dtype)
        mi_gpu = obj.compute_cross(cp.asarray(x), cp.asarray(y))
        mi_cpu = np.empty((nx, ny), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                mi_cpu[i, j] = mutual_information_kdtree(x[:, i], y[:, j], k=k)
        cp.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-5, atol=1e-6)


    def test_compute_pairwise_wrong_dim_1D(self):
        obj = KNNMutualInformation()
        msg = "data must be two-dimensional or three-dimensional."
        with pytest.raises(ValueError, match=msg):
            obj.compute_pairwise(cp.arange(100))

    def test_compute_pariwise_wrong_dim_4D(self):
        obj = KNNMutualInformation()
        msg = "data must be two-dimensional or three-dimensional."
        with pytest.raises(ValueError, match=msg):
            obj.compute_pairwise(cp.arange(100).reshape(1, 1, 1, 100))

    def test_compute_pairwise_n_pairs_is_0(self):
        obj = KNNMutualInformation()
        data = cp.arange(100).reshape(100, 1)
        mi = obj.compute_pairwise(data)
        cp.testing.assert_array_equal(mi, 0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_compute_pairwise_no_noise(self, dtype: np.dtype, k:int):
        rng = np.random.default_rng(0)
        n = 2
        data = rng.standard_normal((100, n), dtype=dtype)
        obj = KNNMutualInformation(k=k, add_noise=False, dtype=dtype)
        mi_gpu = obj.compute_pairwise(cp.asarray(data))
        mi_cpu = np.full((n, n), np.nan, dtype=dtype)
        for i in range(n):
            for j in range(i + 1, n):
                mi_cpu[i, j] = mutual_information_kdtree(data[:, i], data[:, j], k=k)
                mi_cpu[j, i] = mi_cpu[i, j]
        cp.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_compute_pairwise_noise(self, dtype: np.dtype, k: int):
        rng = np.random.default_rng(0)
        n = 5
        data = rng.standard_normal((100, n), dtype=dtype)
        obj = KNNMutualInformation(k=k, add_noise=True, dtype=dtype)
        mi_gpu = obj.compute_pairwise(cp.asarray(data))
        mi_cpu = np.full((n, n), np.nan, dtype=dtype)
        for i in range(n):
            for j in range(i + 1, n):
                mi_cpu[i, j] = mutual_information_kdtree(data[:, i], data[:, j], k=k)
                mi_cpu[j, i] = mi_cpu[i, j]
        cp.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-5, atol=1e-6)
