import numpy as np
import pytest

from upd_apf.utils.time_grid import prediction_time_grid


def test_divisible_grid_contains_both_endpoints():
    assert np.allclose(prediction_time_grid(1, 0.25), [0, .25, .5, .75, 1])


def test_nondivisible_grid_appends_horizon():
    assert np.allclose(prediction_time_grid(1, 0.3), [0, .3, .6, .9, 1])


def test_zero_horizon():
    assert np.array_equal(prediction_time_grid(0, 0.1), [0])


@pytest.mark.parametrize("horizon, dt", [(-1, .1), (1, 0), (1, -1)])
def test_invalid_grid_inputs(horizon, dt):
    with pytest.raises(ValueError):
        prediction_time_grid(horizon, dt)
