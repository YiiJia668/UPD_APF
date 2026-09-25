import numpy as np
import pytest

from upd_apf.safety.relative_geometry import relative_covariance, relative_geometry, relative_mean, unit_direction


def test_relative_mean_sign_points_uav_to_obstacle():
    assert np.array_equal(relative_mean([4, 2, 3], [1, 2, 3]), [3, 0, 0])


def test_relative_geometry_distance_and_direction():
    mean, distance, direction = relative_geometry([3, 4, 0], [0, 0, 0])
    assert np.array_equal(mean, [3, 4, 0])
    assert distance == 5
    assert np.allclose(direction, [.6, .8, 0])


def test_zero_distance_is_explicitly_degenerate():
    with pytest.raises(ValueError, match="degenerate"):
        unit_direction([0, 0, 0])


def test_relative_covariance_adds_independent_covariances():
    assert np.allclose(relative_covariance(np.eye(3), 2 * np.eye(3)), 3 * np.eye(3))


def test_translation_invariance():
    shift = np.array([9, -2, 4])
    assert np.array_equal(relative_mean(np.array([1, 2, 3]) + shift, shift), [1, 2, 3])
