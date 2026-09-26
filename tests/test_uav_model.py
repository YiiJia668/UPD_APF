"""Analytical tests for force-driven point-mass dynamics."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from upd_apf.config import UPDAPFConfig, config_from_mapping
from upd_apf.data_types import UAVState
from upd_apf.simulation import PointMassUAVModel


def state(p=(0, 0, 0), v=(0, 0, 0)):
    return UAVState(p, v, np.diag([1., 2., 3.]))


def test_exact_force_gain_and_ownership():
    original = state(v=(1, 0, 0))
    output = PointMassUAVModel(force_to_acceleration_gain=.5).step(original, [2, 0, 0], 1)
    assert_array_equal(output.position, [1.5, 0, 0])
    assert_array_equal(output.velocity, [2, 0, 0])
    assert_array_equal(original.position, [0, 0, 0])
    assert_array_equal(original.velocity, [1, 0, 0])
    assert_array_equal(output.position_covariance, original.position_covariance)
    assert not np.shares_memory(output.position_covariance, original.position_covariance)


@pytest.mark.parametrize("velocity", [(0, 0, 0), (2, -3, 1)])
def test_zero_force(velocity):
    original = state((1, 2, 3), velocity)
    output = PointMassUAVModel().step(original, [0, 0, 0], .3)
    assert_allclose(output.position, original.position + .3 * original.velocity)
    assert_array_equal(output.velocity, velocity)


@pytest.mark.parametrize("gain", [1., .5, 2.])
def test_gain_scaling_without_resaturation(gain):
    original = state((3, -2, 4), (1, 2, 3))
    force = np.array([200., -400., 100.])
    dt = .2
    baseline = PointMassUAVModel().step(original, force, dt)
    result = PointMassUAVModel(force_to_acceleration_gain=gain).step(original, force, dt)
    assert_allclose(baseline.velocity - original.velocity, force * dt)
    assert_allclose(result.velocity - original.velocity, gain * (baseline.velocity - original.velocity))
    ballistic_position = original.position + original.velocity * dt
    assert_allclose(result.position - ballistic_position, gain * .5 * force * dt**2)


def test_translation_and_rotation():
    original = state((1, 2, 3), (-2, 3, 1))
    force = np.array([4., -5., 2.])
    shift = np.array([3., -2., 8.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    model = PointMassUAVModel(3)
    result = model.step(original, force, .4)
    moved = model.step(state(original.position + shift, original.velocity), force, .4)
    rotated = model.step(state(rotation @ original.position, rotation @ original.velocity), rotation @ force, .4)
    assert_allclose(moved.position, result.position + shift)
    assert_allclose(moved.velocity, result.velocity)
    assert_allclose(rotated.position, rotation @ result.position)
    assert_allclose(rotated.velocity, rotation @ result.velocity)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_gain_and_dt(bad):
    with pytest.raises(ValueError):
        PointMassUAVModel(bad)
    with pytest.raises(ValueError):
        PointMassUAVModel().step(state(), [0, 0, 0], bad)


@pytest.mark.parametrize("force", [[1, 2], [[1, 2, 3]], [0, np.nan, 0], [np.inf, 0, 0]])
def test_invalid_force(force):
    with pytest.raises(ValueError):
        PointMassUAVModel().step(state(), force, .1)


@pytest.mark.parametrize("field", ["position", "velocity", "position_covariance"])
def test_revalidates_nonfinite_state(field):
    original = state()
    object.__setattr__(original, field, np.full(getattr(original, field).shape, np.nan))
    with pytest.raises(ValueError):
        PointMassUAVModel().step(original, [0, 0, 0], .1)


def test_overflow_is_error():
    with pytest.raises(ValueError):
        PointMassUAVModel().step(state(), [1e308, 0, 0], 10)


@pytest.mark.parametrize("group,key,values", [
    ("uav", "force_to_acceleration_gain", [0, -1, np.nan, np.inf, -np.inf]),
    ("uav", "radius", [-1, np.nan, np.inf]),
    ("simulation", "dt", [0, -1, np.nan, np.inf]),
    ("simulation", "max_time", [0, -1, np.nan, np.inf]),
    ("simulation", "goal_tolerance", [0, -1, np.nan, np.inf]),
    ("simulation", "goal_speed_tolerance", [-1, np.nan, np.inf]),
])
def test_config_validation(group, key, values):
    for value in values:
        with pytest.raises(ValueError):
            config_from_mapping({group: {key: value}})
    assert UPDAPFConfig().uav.force_to_acceleration_gain == 1
    assert config_from_mapping({"simulation": {"goal_speed_tolerance": 0}}).simulation.goal_speed_tolerance == 0


def test_mass_is_not_a_second_public_scale():
    assert "mass" not in UPDAPFConfig().to_dict()["uav"]
    with pytest.raises(ValueError, match="invalid uav"):
        config_from_mapping({"uav": {"mass": 2.}})
    with pytest.raises(TypeError):
        PointMassUAVModel(mass=2.)
