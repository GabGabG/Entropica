import cupy as cp
from functools import lru_cache

_LINEAR_KNN_CODE = """
int pair_idx = i / sample_size;
int row = i % sample_size;
int offset = pair_idx * sample_size;
T best_dists[MAX_DISTS];

for (int a = 0; a < MAX_DISTS; a++){
    best_dists[a] = 1e30;
}

// --------------------------------------------------
// 1. Find distance to the k-th neighbour
// --------------------------------------------------

T current_x = x_pairs[offset + row];
T current_y = y_pairs[offset + row];

for (int col = 0; col < sample_size; col++){
    T diff_x = abs(current_x - x_pairs[offset + col]);
    T diff_y = abs(current_y - y_pairs[offset + col]);

    T max_diff = (diff_x > diff_y) ? diff_x : diff_y;

    if (max_diff < best_dists[MAX_DISTS - 1]){
        int pos = MAX_DISTS - 1;
        while (pos > 0 && max_diff < best_dists[pos - 1]){
            best_dists[pos] = best_dists[pos - 1];
            pos--;
        }
        best_dists[pos] = max_diff;
    }
}

T r = best_dists[MAX_DISTS - 1];

// --------------------------------------------------
// 2. Count marginal neighbours
// --------------------------------------------------

int count_x = 0;
int count_y = 0;

for (int col = 0; col < sample_size; col++){
    T diff_x = abs(current_x - x_pairs[offset + col]);
    T diff_y = abs(current_y - y_pairs[offset + col]);

    if (diff_x <= r){
        count_x++;
    }
    if (diff_y <= r){
        count_y++;
    }
}
nx = count_x;
ny = count_y;
"""


@lru_cache(maxsize=None)
def _get_knn_kernel(k: int) -> cp.ElementwiseKernel:
    return cp.ElementwiseKernel(
        in_params="raw T x_pairs, raw T y_pairs, int32 sample_size",
        out_params="int32 nx, int32 ny",
        operation=_LINEAR_KNN_CODE,
        name="linear_knn_kernel",
        preamble=f"#define MAX_DISTS {k + 1}"
    )


def knn_statistics(x_pairs: cp.ndarray, y_pairs: cp.ndarray, k: int) -> tuple[cp.ndarray, cp.ndarray]:
    if x_pairs.ndim != 2 or y_pairs.ndim != 2:
        raise ValueError("x_pairs and y_pairs must be 2D.")

    if x_pairs.shape != y_pairs.shape:
        raise ValueError("x_pairs and y_pairs must have the same shape.")

    n_samples = x_pairs.shape[1]

    if not 1 <= k < n_samples:
        msg = f"k must satisfy 1 <= k < n_samples, got k={k}, n_samples={n_samples}."
        raise ValueError(msg)
    kernel = _get_knn_kernel(k)
    nx_tensor = cp.zeros(x_pairs.shape, dtype=cp.int32)
    ny_tensor = cp.zeros(x_pairs.shape, dtype=cp.int32)
    kernel(x_pairs, y_pairs, n_samples, nx_tensor, ny_tensor)
    return nx_tensor, ny_tensor
