import numpy as np
import pytest

from upd_apf.potential.attractive import force, potential


def test_at_goal():
    p = np.array([1., -2., 3.])
    assert potential(p, p, 2, 3) == 0.0
    np.testing.assert_array_equal(force(p, p, 2, 3), np.zeros(3))


@pytest.mark.parametrize("distance", [2., 8.])
def test_exact_formulas_and_goal_direction(distance):
    p = np.array([1., 2., -1.])
    direction = np.array([.6, .8, 0.])
    goal = p + distance * direction
    gain, switch = 2., 3.
    expected_u = (gain * distance**2 / 2 if distance <= switch
                  else gain * switch * distance - gain * switch**2 / 2)
    expected_norm = gain * min(distance, switch)
    assert potential(p, goal, gain, switch) == pytest.approx(expected_u)
    result = force(p, goal, gain, switch)
    np.testing.assert_allclose(result, expected_norm * direction)
    assert np.linalg.norm(result) == pytest.approx(expected_norm)
    assert result @ (goal - p) > 0
    assert isinstance(potential(p, goal, gain, switch), float)
    assert result.shape == (3,) and result.dtype == np.float64


def test_boundary_continuity():
    gain, switch, delta = 2., 3., 1e-8
    values = [potential([0, 0, 0], [d, 0, 0], gain, switch)
              for d in (switch - delta, switch, switch + delta)]
    forces = [force([0, 0, 0], [d, 0, 0], gain, switch)
              for d in (switch - delta, switch, switch + delta)]
    np.testing.assert_allclose(values, [9., 9., 9.], rtol=0, atol=7e-8)
    np.testing.assert_allclose(forces, [[6., 0, 0]] * 3, rtol=0, atol=3e-8)


@pytest.mark.parametrize("goal", [[1., 2., .5], [7., -4., 3.]])
def test_force_is_negative_central_difference_gradient(goal):
    p = np.array([.2, -.1, .3])
    gain, switch, h = 1.7, 3., 1e-6
    gradient = np.array([
        (potential(p + h * axis, goal, gain, switch)
         - potential(p - h * axis, goal, gain, switch)) / (2 * h)
        for axis in np.eye(3)
    ])
    np.testing.assert_allclose(
        force(p, goal, gain, switch), -gradient, rtol=2e-8, atol=2e-9,
    )


@pytest.mark.parametrize("switch", [1., 20.])
def test_translation_rotation_and_input_purity(switch):
    p, goal = np.array([.2, -1., 2.]), np.array([3., 4., -2.])
    before = p.copy(), goal.copy()
    shift = np.array([10., -5., 2.])
    rotation = np.array([[.6, -.8, 0], [.8, .6, 0], [0, 0, 1.]])
    original_u, original_f = potential(p, goal, 2, switch), force(p, goal, 2, switch)
    assert potential(p + shift, goal + shift, 2, switch) == pytest.approx(original_u)
    assert potential(rotation @ p, rotation @ goal, 2, switch) == pytest.approx(original_u)
    np.testing.assert_allclose(force(p + shift, goal + shift, 2, switch), original_f)
    np.testing.assert_allclose(
        force(rotation @ p, rotation @ goal, 2, switch), rotation @ original_f,
    )
    np.testing.assert_array_equal(p, before[0])
    np.testing.assert_array_equal(goal, before[1])


@pytest.mark.parametrize("goal", [[1, 0, 0], [20, 0, 0]])
def test_zero_gain(goal):
    assert potential([0, 0, 0], goal, 0, 3) == 0
    np.testing.assert_array_equal(force([0, 0, 0], goal, 0, 3), np.zeros(3))


@pytest.mark.parametrize("function", [potential, force])
@pytest.mark.parametrize("field,value", [
    ("gain", -1), ("gain", np.nan), ("gain", np.inf),
    ("switch_distance", 0), ("switch_distance", -1),
    ("switch_distance", np.nan), ("switch_distance", np.inf),
    ("position", [1, 2]), ("position", [[1, 2, 3]]),
    ("position", [np.nan, 0, 0]), ("goal", [0, np.inf, 0]),
    ("goal", [1, 2, 3, 4]),
])
def test_invalid_inputs(function, field, value):
    args = dict(position=[0, 0, 0], goal=[1, 2, 3], gain=1, switch_distance=3)
    args[field] = value
    with pytest.raises(ValueError):
        function(**args)
