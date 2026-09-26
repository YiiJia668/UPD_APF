import numpy as np
import pytest

from upd_apf.potential.damping import force, reference_velocity


def test_stationary_uav_accelerates_toward_goal():
    reference = reference_velocity([0, 0, 0], [3, 4, 0], 2)
    np.testing.assert_allclose(reference, [1.2, 1.6, 0])
    np.testing.assert_allclose(force([0, 0, 0], reference, .5), [.6, .8, 0])
    assert reference.shape == (3,) and reference.dtype == np.float64


def test_velocity_matches_reference():
    reference = reference_velocity([0, 0, 0], [1, 2, 3], 2)
    np.testing.assert_array_equal(force(reference, reference, .4), np.zeros(3))


@pytest.mark.parametrize("velocity", [[1., -2., 3.], [0., 0., 0.]])
def test_goal_has_zero_reference_and_brakes(velocity):
    reference = reference_velocity([1, 2, 3], [1, 2, 3], 2)
    np.testing.assert_array_equal(reference, np.zeros(3))
    np.testing.assert_allclose(force(velocity, reference, .4), -.4 * np.array(velocity))


def test_exact_formula():
    velocity, reference = np.array([1., 2., -3.]), np.array([2., -1., .5])
    np.testing.assert_allclose(force(velocity, reference, .7), -.7 * (velocity - reference))


def test_translation_rotation_and_input_purity():
    p, goal, velocity = np.array([1., 2., 3.]), np.array([3., -1., 5.]), np.array([.5, -1., 2.])
    copies = p.copy(), goal.copy(), velocity.copy()
    shift = np.array([10., -3., 4.])
    rotation = np.array([[.6, -.8, 0], [.8, .6, 0], [0, 0, 1.]])
    reference = reference_velocity(p, goal, 2)
    original = force(velocity, reference, .7)
    np.testing.assert_allclose(reference_velocity(p + shift, goal + shift, 2), reference)
    rotated_reference = reference_velocity(rotation @ p, rotation @ goal, 2)
    np.testing.assert_allclose(rotated_reference, rotation @ reference)
    np.testing.assert_allclose(
        force(rotation @ velocity, rotated_reference, .7), rotation @ original,
    )
    for value, before in zip((p, goal, velocity), copies):
        np.testing.assert_array_equal(value, before)
    np.testing.assert_array_equal(force(velocity, reference, .7), original)


def test_zero_speed_gain_and_numerical_goal_threshold():
    np.testing.assert_array_equal(reference_velocity([0, 0, 0], [1, 0, 0], 0), np.zeros(3))
    np.testing.assert_array_equal(force([1, 2, 3], [4, 5, 6], 0), np.zeros(3))
    np.testing.assert_array_equal(
        reference_velocity([0, 0, 0], [1e-9, 0, 0], 2), np.zeros(3),
    )
    np.testing.assert_allclose(
        reference_velocity([0, 0, 0], [1e-9, 0, 0], 2, eps_distance=0), [2, 0, 0],
    )


@pytest.mark.parametrize("field,value", [
    ("desired_speed", -1), ("desired_speed", np.inf), ("desired_speed", np.nan),
    ("eps_distance", -1), ("eps_distance", np.nan), ("eps_distance", np.inf),
    ("position", [1, 2]), ("position", [[0, 0, 0]]),
    ("position", [np.nan, 0, 0]), ("goal", [0, np.inf, 0]), ("goal", [1, 2]),
])
def test_invalid_reference_inputs(field, value):
    args = dict(position=[0, 0, 0], goal=[1, 2, 3], desired_speed=2)
    args[field] = value
    with pytest.raises(ValueError):
        reference_velocity(**args)


@pytest.mark.parametrize("field,value", [
    ("gain", -1), ("gain", np.inf), ("gain", np.nan),
    ("velocity", [1, 2]), ("velocity", [[0, 0, 0]]),
    ("velocity", [0, np.nan, 0]), ("reference_velocity", [0, 0, np.inf]),
    ("reference_velocity", [1, 2]),
])
def test_invalid_force_inputs(field, value):
    args = dict(velocity=[0, 0, 0], reference_velocity=[1, 2, 3], gain=.4)
    args[field] = value
    with pytest.raises(ValueError):
        force(**args)
