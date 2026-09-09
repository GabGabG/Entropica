import cupy as cp
import pytest
from numpy._typing import ArrayLike

from entropica.testing.strategies import PermutationStrategy, YPermutation


class TestPermutationStrategy:
    def test_is_abstract(self):
        msg = (
            "Can't instantiate abstract class PermutationStrategy without an "
            "implementation for abstract method 'permute'"
        )
        with pytest.raises(TypeError, match=msg):
            PermutationStrategy()

    def test_set_random_state_None(self):
        generator = PermutationStrategy._set_random_state(None)
        assert isinstance(generator, cp.random.Generator)

    @pytest.mark.parametrize("seed", range(0, 10))
    def test_set_random_state_seed(self, seed: int):
        generator = PermutationStrategy._set_random_state(seed)
        assert isinstance(generator, cp.random.Generator)

    def test_set_random_state_generator(self):
        generator = PermutationStrategy._set_random_state(cp.random.default_rng())
        assert isinstance(generator, cp.random.Generator)

    def test_set_random_state_unknown(self):
        random_state = "I am a random state"
        msg = f"Unknown random_state type: {type(random_state)}"
        with pytest.raises(TypeError, match=msg):
            PermutationStrategy._set_random_state(random_state)


class TestDummyPermutationStrategy:
    @pytest.fixture(autouse=True)
    def setup(self):
        class DummyPermutationStrategy(PermutationStrategy):
            def permute(
                self,
                x: ArrayLike,
                y: ArrayLike,
                n_permutations: int,
                random_state: int | cp.generator | None,
            ) -> tuple[cp.ndarray, cp.ndarray]:
                return super().permute(x, y, n_permutations, random_state)

        self.dummy_class = DummyPermutationStrategy

    def test_permute_raises(self):
        obj = self.dummy_class()
        msg = "To implement in derived classes."
        with pytest.raises(NotImplementedError, match=msg):
            obj.permute(cp.ones(10), cp.ones(10), 10, 42)


class TestYPermutation:
    def test_obj_possible(self):
        try:
            YPermutation()
        except Exception as e:
            pytest.fail(f"Exception raised:\n{e}")

    def test_is_subclass(self):
        assert issubclass(YPermutation, PermutationStrategy)

    @pytest.mark.parametrize("n_permutations", range(-10, 1))
    def test_permute_n_permutations_too_small(self, n_permutations: int):
        x = cp.ones(10)
        y = cp.array(2)
        msg = "The number of permutations must be at least 1."
        permute = YPermutation()
        with pytest.raises(ValueError, match=msg):
            permute.permute(x, y, n_permutations, None)

    def test_permute_wrong_dimension(self):
        x = cp.ones(10)
        y = cp.array(2)
        msg = "y must be one- or two-dimensional."
        permute = YPermutation()
        with pytest.raises(ValueError, match=msg):
            permute.permute(x, y, 100, None)

    def test_permute_1D(self):
        generator = cp.random.default_rng(42)
        n_samples = 5
        n_permutations = 4
        x = cp.arange(n_samples)
        y = cp.arange(n_samples)
        supposed_y_p = cp.array(
            [[3, 4, 3, 1], [2, 1, 1, 3], [1, 2, 2, 0], [0, 3, 4, 4], [4, 0, 0, 2]]
        )
        supposed_y_p = supposed_y_p[..., None]  # Var dimension (1)
        permutation_obj = YPermutation()
        x_p, y_p = permutation_obj.permute(x, y, n_permutations, generator)
        cp.testing.assert_array_equal(x_p, x)  # x doesn't change
        cp.testing.assert_array_equal(y_p, supposed_y_p)

    def test_permute_2D(self):
        generator = cp.random.default_rng(42)
        n_samples = 5
        n_permutations = 4
        x = cp.arange(n_samples)
        y = cp.arange(n_samples)
        y = cp.vstack([y, y + n_samples]).T
        supposed_y_p = cp.array(
            [[3, 4, 3, 1], [2, 1, 1, 3], [1, 2, 2, 0], [0, 3, 4, 4], [4, 0, 0, 2]]
        )
        supposed_y_p = cp.dstack([supposed_y_p, supposed_y_p + n_samples])
        permutation_obj = YPermutation()
        x_p, y_p = permutation_obj.permute(x, y, n_permutations, generator)
        cp.testing.assert_array_equal(x_p, x)  # x doesn't change
        cp.testing.assert_array_equal(y_p, supposed_y_p)
