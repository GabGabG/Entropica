from collections import namedtuple

import cupy as cp
import numpy as np
import pytest
from scipy.spatial import cKDTree

from entropica.backends.cupy import (
    _LINEAR_KNN_CODE,
    _get_knn_kernel,
    knn_statistics,
)


def knn_statistics_kdtree(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_pairs, n_samples = x.shape

    nx = np.empty((n_pairs, n_samples), dtype=np.int32)
    ny = np.empty((n_pairs, n_samples), dtype=np.int32)

    for pair in range(n_pairs):
        points = np.column_stack((x[pair], y[pair]))
        tree = cKDTree(points)
        distances = tree.query(points, k=k + 1, p=np.inf)[0]
        r = distances[:, k]
        x_tree = cKDTree(x[pair, :, None])
        y_tree = cKDTree(y[pair, :, None])

        nx[pair] = x_tree.query_ball_point(
            x[pair, :, None], r, p=np.inf, return_length=True
        )
        ny[pair] = y_tree.query_ball_point(
            y[pair, :, None], r, p=np.inf, return_length=True
        )

    return nx, ny


class TestGetKNNKernel:
    @pytest.fixture(autouse=True)
    def setup(self):
        _get_knn_kernel.cache_clear()

    def test_get_knn_kernel_type(self):
        assert isinstance(_get_knn_kernel(k=3), cp.ElementwiseKernel)

    @pytest.mark.parametrize("k", range(1, 11))
    def test_get_knn_kernel_attributes(self, k: int):
        params = namedtuple("params", ["name", "dtype", "ctype", "raw", "is_const"])
        kernel = _get_knn_kernel(k=k)
        in_params = [
            params("x_pairs", None, "T", True, True),
            params("y_pairs", None, "T", True, True),
            params("sample_size", cp.dtype("int32"), "int", False, True),
        ]

        out_params = [
            params("nx", cp.dtype("int32"), "int", False, False),
            params("ny", cp.dtype("int32"), "int", False, False),
        ]

        assert kernel.name == "linear_knn_kernel"
        assert kernel.preamble == f"#define MAX_DISTS {k + 1}"

        assert len(kernel.in_params) == len(in_params)
        for i, p in enumerate(kernel.in_params):
            current_params = in_params[i]
            assert p.name == current_params.name
            assert p.dtype is current_params.dtype
            assert p.ctype == current_params.ctype
            assert p.raw == current_params.raw
            assert p.is_const == current_params.is_const

        assert len(kernel.out_params) == len(out_params)
        for i, p in enumerate(kernel.out_params):
            current_params = out_params[i]
            assert p.name == current_params.name
            assert p.dtype is current_params.dtype
            assert p.ctype == current_params.ctype
            assert p.raw == current_params.raw
            assert p.is_const == current_params.is_const

        assert kernel.operation == _LINEAR_KNN_CODE

    def test_knn_kernel_is_cached(self):
        kernel_1 = _get_knn_kernel(3)
        kernel_2 = _get_knn_kernel(3)
        assert kernel_1 is kernel_2

        info = _get_knn_kernel.cache_info()

        assert info.misses == 1
        assert info.hits == 1

    def test_knn_kernel_not_cached_different_k(self):
        kernel_1 = _get_knn_kernel(3)
        kernel_2 = _get_knn_kernel(4)
        assert kernel_1 is not kernel_2


class TestKNNStatistics:
    def test_wrong_x_3D(self):
        x = cp.arange(100).reshape(2, 5, 10)
        y = cp.arange(100).reshape(10, 10)
        msg = "x_pairs and y_pairs must be 2D."
        with pytest.raises(ValueError, match=msg):
            knn_statistics(x, y, 3)

    def test_wrong_y_3D(self):
        y = cp.arange(100).reshape(2, 5, 10)
        x = cp.arange(100).reshape(10, 10)
        msg = "x_pairs and y_pairs must be 2D."
        with pytest.raises(ValueError, match=msg):
            knn_statistics(x, y, 3)

    def test_wrong_x_y_shape_mismatch(self):
        x = cp.arange(24).reshape(4, 6)
        y = cp.arange(24).reshape(6, 4)
        msg = "x_pairs and y_pairs must have the same shape."
        with pytest.raises(ValueError, match=msg):
            knn_statistics(x, y, 3)

    @pytest.mark.parametrize("k", range(-10, 0))
    def test_k_is_too_small(self, k: int):
        msg = f"k must satisfy 1 <= k < n_samples, got k={k}, n_samples=10."
        x = cp.arange(100).reshape(10, 10)
        with pytest.raises(ValueError, match=msg):
            knn_statistics(x, x, k)

    @pytest.mark.parametrize("k", range(11, 20))
    def test_k_is_too_big(self, k: int):
        msg = f"k must satisfy 1 <= k < n_samples, got k={k}, n_samples=10."
        x = cp.arange(100).reshape(10, 10)
        with pytest.raises(ValueError, match=msg):
            knn_statistics(x, x, k)

    @pytest.mark.parametrize("k", range(1, 11))
    def test_no_errors(self, k: int):
        x = cp.random.randn(2 * k, 2 * k)
        y = cp.random.randn(2 * k, 2 * k)
        try:
            knn_statistics(x, y, k)
        except Exception as e:
            pytest.fail(f"Exception raised:\n{e}")

    @pytest.mark.parametrize("k", range(1, 11))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_knn_statistics_against_kdtree_variable_k(self, k: int, dtype: np.dtype):
        seed = 42
        generator = np.random.default_rng(seed=seed)
        x = generator.standard_normal(size=(10, 100), dtype=dtype)
        y = generator.standard_normal(size=(10, 100), dtype=dtype)
        nx_tree, ny_tree = knn_statistics_kdtree(x, y, k=k)

        nx_gpu, ny_gpu = knn_statistics(cp.asarray(x), cp.asarray(y), k=k)
        np.testing.assert_array_equal(nx_tree, nx_gpu.get())
        np.testing.assert_array_equal(ny_tree, ny_gpu.get())

    @pytest.mark.parametrize("n_pairs", [1, 2, 5, 10, 100, 1_000])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_knn_statistics_against_kdtree_variable_number_pairs(
            self, n_pairs: int, dtype: np.dtype
    ):
        seed = 42
        generator = np.random.default_rng(seed=seed)
        x = generator.standard_normal(size=(n_pairs, 100), dtype=dtype)
        y = generator.standard_normal(size=(n_pairs, 100), dtype=dtype)

        nx_tree, ny_tree = knn_statistics_kdtree(x, y, k=3)

        nx_gpu, ny_gpu = knn_statistics(cp.asarray(x), cp.asarray(y), k=3)
        np.testing.assert_array_equal(nx_tree, nx_gpu.get())
        np.testing.assert_array_equal(ny_tree, ny_gpu.get())

    @pytest.mark.parametrize("n_samples", [5, 10, 100, 1_000])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_knn_statistics_against_kdtree_variable_number_samples(
            self, n_samples: int, dtype: np.dtype
    ):
        seed = 42
        generator = np.random.default_rng(seed=seed)
        x = generator.standard_normal(size=(10, n_samples), dtype=dtype)
        y = generator.standard_normal(size=(10, n_samples), dtype=dtype)
        k = 3
        nx_tree, ny_tree = knn_statistics_kdtree(x, y, k)

        nx_gpu, ny_gpu = knn_statistics(cp.asarray(x), cp.asarray(y), k=k)
        np.testing.assert_array_equal(nx_tree, nx_gpu.get())
        np.testing.assert_array_equal(ny_tree, ny_gpu.get())

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_single_pair(self, dtype: np.dtype):
        seed = 42
        generator = np.random.default_rng(seed=seed)

        x = generator.standard_normal((1, 100), dtype=dtype)
        y = generator.standard_normal((1, 100), dtype=dtype)

        nx_tree, ny_tree = knn_statistics_kdtree(x, y, k=3)
        nx_gpu, ny_gpu = knn_statistics(cp.asarray(x), cp.asarray(y), k=3)

        np.testing.assert_array_equal(nx_tree, nx_gpu.get())
        np.testing.assert_array_equal(ny_tree, ny_gpu.get())
