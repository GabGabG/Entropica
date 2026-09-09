import cupy as cp
import numpy as np
import pytest

from entropica.estimators.base import BaseEstimator


class TestBaseEstimator:
    def test_is_abstract(self):
        msg = (
            "Can't instantiate abstract class BaseEstimator without an "
            "implementation for abstract method 'compute'"
        )
        with pytest.raises(TypeError, match=msg):
            BaseEstimator(cp.float32, None)


class TestDummyBaseEstimator:
    @pytest.fixture(autouse=True)
    def setup(self):
        class DummyEstimator(BaseEstimator):
            def compute(self):
                super().compute()

        self.dummy_class = DummyEstimator

    def test_obj_possible(self):
        try:
            self.dummy_class(cp.float32, None)
        except Exception as e:
            pytest.fail(f"Exception raised:\n{e}")

    def test_default_attributes(self):
        obj = self.dummy_class(cp.float32, None)
        assert obj._dtype == cp.float32
        assert isinstance(obj._random_state, cp.random.Generator)

    @pytest.mark.parametrize("dtype", [cp.float32, cp.float64, cp.int32, cp.int64])
    def test_dtype_property(self, dtype: cp.dtype):
        obj = self.dummy_class(dtype, None)
        dtype_property = obj.dtype
        assert obj._dtype == dtype_property
        assert dtype_property == dtype

    @pytest.mark.parametrize("dtype", [np.single, np.double])
    def test_numpy_floats_ok(self, dtype: np.dtype):
        try:
            self.dummy_class(dtype, None)
        except Exception as e:
            pytest.fail(f"Exception raised:\n{e}")

    def test_random_state_getter(self):
        obj = self.dummy_class(cp.float32, None)
        assert isinstance(obj.random_state, cp.random.Generator)
        assert obj._random_state == obj.random_state

    def test_random_state_setter_none(self):
        obj = self.dummy_class(cp.float32, None)
        initial_gen = obj.random_state
        obj.random_state = None
        final_gen = obj.random_state
        assert isinstance(final_gen, cp.random.Generator)
        assert final_gen != initial_gen

    @pytest.mark.parametrize("seed", range(0, 10))
    def test_random_state_setter_seed(self, seed: int):
        obj = self.dummy_class(cp.float32, seed)
        obj.random_state = seed
        assert isinstance(obj.random_state, cp.random.Generator)

    def test_random_state_setter_random_state(self):
        obj = self.dummy_class(cp.float32, None)
        initial_gen = obj.random_state
        obj.random_state = initial_gen
        assert isinstance(obj.random_state, cp.random.Generator)
        assert obj.random_state == initial_gen

    def test_random_state_setter_unknown_type(self):
        obj = self.dummy_class(cp.float32, None)
        random_state = "I am a random state"
        msg = f"Unknown random_state type: {type(random_state)}"
        with pytest.raises(TypeError, match=msg):
            obj.random_state = random_state

    def test_compute_raises(self):
        obj = self.dummy_class(cp.float32, None)
        msg = "To implement in derived classes."
        with pytest.raises(NotImplementedError, match=msg):
            obj.compute()
