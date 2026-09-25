import numpy as np
import pytest

from upd_apf.utils.saturation import saturate_vector


def test_vector_inside_limit_unchanged():
    assert np.allclose(saturate_vector([3, 4], 6), [3, 4])


def test_vector_saturation_preserves_direction():
    result = saturate_vector([3, 4, 0], 2)
    assert np.isclose(np.linalg.norm(result), 2)
    assert np.allclose(result, [1.2, 1.6, 0])


def test_zero_limit():
    assert np.allclose(saturate_vector([1, 0], 0), [0, 0])


def test_invalid_limit():
    with pytest.raises(ValueError):
        saturate_vector([1], -1)
