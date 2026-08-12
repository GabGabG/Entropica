import cupy as cp
from cupy.typing import ArrayLike

from .estimators.knn import KNNMutualInformation


def mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    *,
    k: int = 3,
    add_noise: bool = True,
    noise_intensity: float = 1e-8,
    dtype: cp.dtype = cp.float32,
    random_state: int | cp.random.Generator | None = None,
) -> cp.ndarray:
    estimator = KNNMutualInformation(
        k=k,
        add_noise=add_noise,
        noise_intensity=noise_intensity,
        dtype=dtype,
        random_state=random_state,
    )
    return estimator.compute(x, y)


def pairwise_mutual_information(
    data: ArrayLike,
    *,
    k: int = 3,
    add_noise: bool = True,
    noise_intensity: float = 1e-8,
    dtype: cp.dtype = cp.float32,
    random_state: int | cp.random.Generator | None = None,
) -> cp.ndarray:
    estimator = KNNMutualInformation(
        k=k,
        add_noise=add_noise,
        noise_intensity=noise_intensity,
        dtype=dtype,
        random_state=random_state,
    )
    return estimator.compute_pairwise(data)
