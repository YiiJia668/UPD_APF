"""Truth ownership and measurement generation regressions."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from upd_apf.data_types import ObstacleMetadata
from upd_apf.environment import ConstantVelocityObstacle, PositionSensor, World
from upd_apf.prediction.kalman_filter import ConstantVelocityKalmanFilter


def obstacle(name="a", p=(1, 2, 3), v=(2, -1, 0), radius=.4):
    return ConstantVelocityObstacle(ObstacleMetadata(name, radius), p, v)


def test_world_cv_snapshots_and_estimator_independence():
    a, b = obstacle(), obstacle("b", (3, 4, 5), (-1, 2, 3))
    world = World([a, b])
    saved = world.snapshots()
    estimator = ConstantVelocityKalmanFilter(np.r_[a.position, a.velocity], np.eye(6), .2, np.eye(3))
    before = estimator.state
    world.advance(.5)
    for old, new in zip(saved, world.snapshots()):
        assert_allclose(new.position, old.position + .5 * old.velocity)
        assert_array_equal(new.velocity, old.velocity)
        assert not np.shares_memory(old.position, new.position)
        with pytest.raises(ValueError):
            old.position[0] = 99
    assert_array_equal(saved[0].position, [1, 2, 3])
    assert_array_equal(estimator.state, before)
    assert estimator.time == 0
    assert not hasattr(a, "predict_distribution")


def test_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        World([obstacle(), obstacle()])


@pytest.mark.parametrize("kwargs", [{"name": ""}, {"name": 3}, {"radius": -1}, {"radius": np.nan},
    {"radius": np.inf}, {"p": [1, 2]}, {"v": [1, 2]}, {"p": [np.inf, 0, 0]}, {"v": [0, np.nan, 0]}])
def test_truth_validation(kwargs):
    with pytest.raises(ValueError):
        obstacle(**kwargs)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_truth_dt_validation(dt):
    for target in [obstacle(), World(), World([obstacle()])]:
        with pytest.raises(ValueError):
            target.advance(dt)


def test_truth_overflow():
    with pytest.raises(ValueError):
        obstacle(v=(1e308, 0, 0)).advance(10)


def test_sensor_exact_ids_time_and_independent_arrays():
    world = World([obstacle(), obstacle("b")])
    truths = world.snapshots()
    measurements = PositionSensor(np.zeros((3, 3))).measure(truths, timestamp=1.25)
    for truth, measured in zip(truths, measurements):
        assert measured.obstacle_id == truth.metadata.obstacle_id
        assert measured.timestamp == 1.25
        assert_array_equal(measured.position, truth.position)
        assert not np.shares_memory(measured.position, truth.position)
    world.advance(1)
    assert_array_equal(measurements[0].position, [1, 2, 3])
    assert_array_equal(truths[0].position, [1, 2, 3])


def test_sensor_seed_replay_and_truth_untouched():
    truths = World([obstacle()]).snapshots()
    covariance = np.diag([.1, .2, .3])
    a = PositionSensor(covariance, seed=19)
    b = PositionSensor(covariance, rng=np.random.default_rng(19))
    for time in [0, .2, .4]:
        ma, mb = a.measure(truths, time)[0], b.measure(truths, time)[0]
        assert_array_equal(ma.position, mb.position)
        assert not np.array_equal(ma.position, truths[0].position)
    assert_array_equal(truths[0].position, [1, 2, 3])
    covariance[:] = 100
    assert_array_equal(a.measurement_covariance, np.diag([.1, .2, .3]))


@pytest.mark.parametrize("covariance", [np.eye(2), np.diag([1, 1, -1]), np.full((3, 3), np.nan),
    [[1, 1, 0], [0, 1, 0], [0, 0, 1]]])
def test_sensor_bad_covariance(covariance):
    with pytest.raises(ValueError):
        PositionSensor(covariance, seed=1)


def test_sensor_rng_contract_and_timestamp():
    with pytest.raises(ValueError):
        PositionSensor(np.eye(3))
    with pytest.raises(ValueError):
        PositionSensor(np.eye(3), seed=1, rng=np.random.default_rng(1))
    with pytest.raises(TypeError):
        PositionSensor(np.eye(3), rng=1)
    with pytest.raises(ValueError):
        PositionSensor(np.zeros((3, 3))).measure([], np.nan)


@pytest.mark.parametrize("advance_external", [True, False])
def test_world_owns_truth_independently_of_caller(advance_external):
    original = obstacle()
    initial = original.snapshot()
    world = World([original])
    if advance_external:
        original.advance(.5)
        assert_array_equal(world.snapshots()[0].position, initial.position)
        assert_array_equal(world.snapshots()[0].velocity, initial.velocity)
        assert_allclose(original.position, initial.position + .5 * initial.velocity)
    else:
        world.advance(.5)
        assert_array_equal(original.position, initial.position)
        assert_array_equal(original.velocity, initial.velocity)
        assert_allclose(world.snapshots()[0].position, initial.position + .5 * initial.velocity)
