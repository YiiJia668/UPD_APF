"""Closed-loop timing, truth/belief separation, termination and replay."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from upd_apf.config import UPDAPFConfig
from upd_apf.data_types import UAVState, UPDAPFResult, ObstacleMetadata, ObstacleMeasurement
from upd_apf.environment import World, ConstantVelocityObstacle, PositionSensor
from upd_apf.planner import UPDAPFPlanner
from upd_apf.prediction.kalman_filter import ConstantVelocityKalmanFilter
from upd_apf.simulation import Simulator, build_no_obstacle_scenario, build_single_crossing_scenario
from upd_apf.safety.chance_constraint import confidence_beta


def state(p=(0, 0, 0), v=(0, 0, 0)):
    return UAVState(p, v, np.zeros((3, 3)))


def config(**groups):
    cfg = UPDAPFConfig()
    defaults = {"dt": .2, "max_time": 1., "goal_tolerance": .1, "goal_speed_tolerance": .1}
    defaults.update(groups.pop("simulation", {}))
    cfg = replace(cfg, simulation=replace(cfg.simulation, **defaults))
    return replace(cfg, **{name: replace(getattr(cfg, name), **values) for name, values in groups.items()})


def truth(p=(10, 2, 0), v=(0, 1, 0), name="a", radius=.4):
    return ConstantVelocityObstacle(ObstacleMetadata(name, radius), p, v)


def estimator(p=(8, 3, 0), v=(0, 1, 0)):
    return ConstantVelocityKalmanFilter(np.r_[p, v], np.eye(6), .1, np.eye(3))


class FixedPlanner:
    def __init__(self, force=(0, 0, 0)):
        self.calls = []
        self.result = UPDAPFResult(np.zeros(3), np.zeros(3), np.zeros(3), force, 0., 0., np.inf, np.inf, ())

    def compute_control(self, uav, goal, obstacles):
        self.calls.append(tuple(obstacles))
        return self.result


def make_sim(cfg=None, planner=None, obstacles=(), estimators=None, sensor=None):
    cfg = config() if cfg is None else cfg
    return Simulator(cfg, FixedPlanner() if planner is None else planner, World(obstacles),
                     PositionSensor(np.zeros((3, 3))) if sensor is None else sensor,
                     {} if estimators is None else estimators)


def assert_tree_equal(a, b):
    if is_dataclass(a):
        for field in fields(a):
            assert_tree_equal(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    elif isinstance(a, np.ndarray):
        assert_array_equal(a, b)
    else:
        assert a == b


def test_timing_partial_step_and_terminal_has_no_extra_work():
    events = []
    class RecordingEstimator(ConstantVelocityKalmanFilter):
        def predict(self, dt):
            events.append(("predict", dt))
            super().predict(dt)
        def update(self, measurement):
            events.append(("update", measurement.timestamp))
            assert_allclose(self.time, measurement.timestamp)
            assert_allclose(measurement.position, [10, 2 + self.time, 0])
            super().update(measurement)
    class RecordingSensor(PositionSensor):
        def measure(self, truths, timestamp):
            events.append(("sensor", timestamp))
            return super().measure(truths, timestamp)
    class RecordingPlanner(FixedPlanner):
        def compute_control(self, uav, goal, obstacles):
            events.append(("planner", obstacles[0].predictor.time))
            return super().compute_control(uav, goal, obstacles)
    cfg = config(simulation={"dt": .3})
    kf = RecordingEstimator([10, 2, 0, 0, 1, 0], np.eye(6), .1, np.eye(3))
    sim = make_sim(cfg, RecordingPlanner(), [truth()], {"a": kf}, RecordingSensor(np.zeros((3, 3))))
    result = sim.run(state(), [100, 0, 0])
    assert result.termination_reason == "max_time"
    assert result.final_time == 1.
    assert_allclose([s.dt for s in result.steps], [.3, .3, .3, .1])
    expected = []
    for i, step in enumerate(result.steps):
        if i:
            expected.append(("predict", result.steps[i-1].dt))
        expected.extend([("sensor", step.time), ("update", step.time), ("planner", step.time)])
    assert [e[0] for e in events] == [e[0] for e in expected]
    assert_allclose([e[1] for e in events], [e[1] for e in expected])
    assert_allclose(kf.time, .9)  # no unnecessary prediction by final .1
    assert_allclose(result.final_obstacle_truths[0].position, [10, 3, 0])


def test_no_accumulated_tiny_final_step():
    result = make_sim(config(simulation={"dt": .1})).run(state(), [100, 0, 0])
    assert len(result.steps) == 10
    assert result.final_time == 1


def test_belief_not_truth_or_measurement_and_planner_purity():
    cfg = config(simulation={"max_time": .6}, prediction={"horizon": .8, "dt": .3})
    kf = estimator()
    original_planner = UPDAPFPlanner(cfg)
    posterior_positions = []
    class CheckingPlanner:
        def compute_control(self, uav, goal, obstacles):
            assert obstacles[0].predictor is kf
            assert not hasattr(obstacles[0], "position")
            x, p, t = kf.state, kf.covariance, kf.time
            posterior_positions.append(x[:3])
            result = original_planner.compute_control(uav, goal, obstacles)
            assert_array_equal(kf.state, x)
            assert_array_equal(kf.covariance, p)
            assert kf.time == t
            sample = result.per_obstacle_results[0].safety_samples[0]
            assert_allclose(sample.distance, np.linalg.norm(x[:3] - uav.position))
            n = (x[:3] - uav.position) / sample.distance
            expected_margin = sample.distance - confidence_beta(cfg.chance_constraint.collision_probability) * np.sqrt(n @ p[:3, :3] @ n) - cfg.uav.radius - .4 - cfg.chance_constraint.extra_safety_margin
            assert_allclose(sample.safety_margin, expected_margin)
            return result
    sim = make_sim(cfg, CheckingPlanner(), [truth()], {"a": kf}, PositionSensor(np.eye(3), seed=27))
    result = sim.run(state(), [50, 0, 0])
    first = result.steps[0]
    assert not np.allclose(first.measurements[0].position, first.obstacle_truths[0].position)
    assert not np.allclose(posterior_positions[0], first.measurements[0].position)
    assert not np.allclose(posterior_positions[0], first.obstacle_truths[0].position)
    assert not np.allclose(posterior_positions[0], posterior_positions[-1])
    assert_allclose([s.dt for s in result.steps], [.2, .2, .2])
    assert_allclose([s.time for s in first.planner_result.per_obstacle_results[0].safety_samples], [0, .3, .6, .8])


@pytest.mark.parametrize("gain,position,velocity", [
    (.5, [.4, -.1, 0], [3, -1, 0]),
    (1., [.6, -.2, 0], [5, -2, 0]),
    (2., [1., -.4, 0], [9, -4, 0]),
])
def test_config_gain_changes_dynamics_without_second_saturation(gain, position, velocity):
    cfg = config(uav={"force_to_acceleration_gain": gain, "max_speed": .01, "max_acceleration": .01}, control={"max_total_force": .01})
    planner = FixedPlanner([20, -10, 0])
    result = make_sim(cfg, planner).run(state(v=(1, 0, 0)), [100, 0, 0])
    first = result.steps[0]
    assert_array_equal(first.planner_result.total_force, [20, -10, 0])
    assert_allclose(first.uav_state_after.position, position)
    assert_allclose(first.uav_state_after.velocity, velocity)
    assert all(call == () for call in planner.calls)


@pytest.mark.parametrize("position,velocity,goal,expected", [
    ((0, 0, 0), (0, 0, 0), (0, 0, 0), "goal_reached"),
    ((.1, 0, 0), (.1, 0, 0), (0, 0, 0), "goal_reached"),
    ((0, 0, 0), (1, 0, 0), (0, 0, 0), "max_time"),
    ((.2, 0, 0), (0, 0, 0), (0, 0, 0), "max_time"),
])
def test_goal_requires_position_and_speed(position, velocity, goal, expected):
    planner = FixedPlanner()
    result = make_sim(planner=planner).run(state(position, velocity), goal)
    assert result.termination_reason == expected
    if expected == "goal_reached":
        assert result.steps == ()
        assert planner.calls == []


def test_initial_collision_over_goal_uses_truth_not_far_belief():
    class ForbiddenSensor:
        def measure(self, *args, **kwargs):
            raise AssertionError("initial termination cannot measure")
    planner = FixedPlanner()
    kf = estimator((100, 100, 0))
    result = make_sim(planner=planner, obstacles=[truth((.7, 0, 0))], estimators={"a": kf}, sensor=ForbiddenSensor()).run(state(), [0, 0, 0])
    assert result.termination_reason == "collision"
    assert result.final_time == 0
    assert not result.steps and not planner.calls
    assert kf.time == 0
    assert_array_equal(kf.state[:3], [100, 100, 0])


@pytest.mark.parametrize("obstacle_position,goal,reason", [
    ((1, 0, 0), (10, 0, 0), "collision"),
    (None, (1, 0, 0), "goal_reached"),
    (None, (10, 0, 0), "max_time"),
])
def test_terminal_at_max_time_priority(obstacle_position, goal, reason):
    cfg = config(simulation={"dt": 1., "max_time": 1., "goal_speed_tolerance": 1.})
    obstacles = [] if obstacle_position is None else [truth(obstacle_position, (0, 0, 0))]
    estimators = {} if not obstacles else {"a": estimator()}
    result = make_sim(cfg, obstacles=obstacles, estimators=estimators).run(state(v=(1, 0, 0)), goal)
    assert result.final_time == 1
    assert len(result.steps) == 1
    assert result.termination_reason == reason


def test_safety_violation_does_not_terminate_or_enlarge_physical_radius():
    cfg = config(simulation={"max_time": .2}, control={"max_total_force": 0})
    # .7 physical radius < .9 truth distance < 1.2 buffered safety radius.
    kf = estimator((.2, 0, 0), (0, 0, 0))
    result = make_sim(cfg, UPDAPFPlanner(cfg), [truth((.9, 0, 0), (0, 0, 0))], {"a": kf}).run(state(), [50, 0, 0])
    assert result.termination_reason == "max_time"
    assert len(result.steps) == 1
    assert result.steps[0].planner_result.min_safety_margin < 0
    assert result.steps[0].planner_result.min_ttc == 0


@pytest.mark.parametrize("mapping", [{}, {"b": estimator()}, {"a": estimator(), "b": estimator()}])
def test_estimator_id_binding_rejected(mapping):
    with pytest.raises(ValueError, match="IDs"):
        make_sim(obstacles=[truth()], estimators=mapping)


def test_duplicate_estimator_instance_rejected():
    kf = estimator()
    with pytest.raises(ValueError, match="own estimator"):
        make_sim(obstacles=[truth(), truth(name="b")], estimators={"a": kf, "b": kf})


@pytest.mark.parametrize("ids,offset", [(["b"], 0), (["a", "a"], 0), ([], 0), (["a"], .1)])
def test_sensor_id_and_timestamp_rejected(ids, offset):
    class BadSensor:
        def measure(self, truths, timestamp):
            return [ObstacleMeasurement(i, timestamp + offset, [1, 2, 3]) for i in ids]
    with pytest.raises(ValueError):
        make_sim(obstacles=[truth()], estimators={"a": estimator()}, sensor=BadSensor()).run(state(), [100, 0, 0])


def test_history_independent_of_world_planner_and_final_state():
    planner = FixedPlanner([1, 2, 0])
    ob = truth()
    sim = make_sim(planner=planner, obstacles=[ob], estimators={"a": estimator()})
    result = sim.run(state(), [100, 0, 0])
    before = deepcopy(result)
    ob.advance(3)
    planner.result.total_force.setflags(write=True)
    planner.result.total_force[:] = 99
    result.final_uav_state.position.setflags(write=True)
    result.final_uav_state.position[:] = -9
    assert_tree_equal(result.steps, before.steps)
    assert_tree_equal(result.final_obstacle_truths, before.final_obstacle_truths)
    for a, b in zip(result.steps, result.steps[1:]):
        assert not np.shares_memory(a.uav_state_after.position, b.uav_state_before.position)
        assert not np.shares_memory(a.planner_result.total_force, b.planner_result.total_force)
        assert not np.shares_memory(a.measurements[0].position, b.measurements[0].position)
    with pytest.raises(RuntimeError, match="single-use"):
        sim.run(state(), [100, 0, 0])


def test_invalid_planner_force_fails_loudly():
    planner = FixedPlanner()
    object.__setattr__(planner.result, "total_force", np.array([np.nan, 0, 0]))
    with pytest.raises(ValueError):
        make_sim(planner=planner).run(state(), [100, 0, 0])


def test_no_obstacle_scenario_reaches_stable_goal_and_first_step_exact():
    setup = build_no_obstacle_scenario()
    result = setup.simulator.run(setup.initial_uav_state, setup.goal)
    cfg = setup.simulator.config
    assert result.termination_reason == "goal_reached"
    assert np.linalg.norm(result.final_uav_state.position - setup.goal) <= cfg.simulation.goal_tolerance
    assert np.linalg.norm(result.final_uav_state.velocity) <= cfg.simulation.goal_speed_tolerance
    first = result.steps[0]
    acceleration = cfg.uav.force_to_acceleration_gain * first.planner_result.total_force
    assert_allclose(first.uav_state_after.position, first.uav_state_before.position + first.uav_state_before.velocity * first.dt + .5 * acceleration * first.dt**2)
    assert_allclose(first.uav_state_after.velocity, first.uav_state_before.velocity + acceleration * first.dt)
    assert first.planner_result.per_obstacle_results == ()


@pytest.mark.parametrize("noisy", [False, True])
def test_crossing_closed_loop_and_seed_replay(noisy):
    cfg = config(simulation={"max_time": 2., "dt": .2}, prediction={"horizon": 1., "dt": .3})
    a = build_single_crossing_scenario(cfg, seed=42, noisy=noisy)
    b = build_single_crossing_scenario(cfg, seed=42, noisy=noisy)
    ra = a.simulator.run(a.initial_uav_state, a.goal)
    rb = b.simulator.run(b.initial_uav_state, b.goal)
    assert_tree_equal(ra, rb)
    assert len(ra.steps) > 2
    initial = ra.steps[0].obstacle_truths[0]
    for step in ra.steps:
        assert_allclose(step.obstacle_truths[0].position, initial.position + step.time * initial.velocity)
        assert_array_equal(step.obstacle_truths[0].velocity, initial.velocity)
        assert np.isfinite(step.uav_state_after.position).all()
        assert np.isfinite(step.uav_state_after.velocity).all()
        assert np.isfinite(step.planner_result.total_force).all()
        diagnostics = step.planner_result.per_obstacle_results[0]
        assert np.isfinite(diagnostics.collision_risk.total_risk)
        assert np.isfinite([s.safety_margin for s in diagnostics.safety_samples]).all()
    assert not np.array_equal(ra.steps[0].planner_result.total_force, ra.steps[-1].planner_result.total_force)
    assert ra.steps[0].planner_result.global_risk != ra.steps[-1].planner_result.global_risk
    assert not np.array_equal(a.simulator.estimators["crossing"].state[:3], initial.position)


def test_measurements_bind_by_id_not_list_order():
    class ReversedSensor(PositionSensor):
        def measure(self, truths, timestamp):
            return tuple(reversed(super().measure(truths, timestamp)))
    cfg = config(simulation={"max_time": .2})
    a, b = estimator((0, 10, 0)), estimator((0, 20, 0))
    result = make_sim(cfg, obstacles=[truth((2, 10, 0), name="a"), truth((4, 20, 0), name="b")],
                      estimators={"b": b, "a": a}, sensor=ReversedSensor(np.zeros((3, 3)))).run(state(), [100, 0, 0])
    assert_allclose(a.state[:3], [1, 10, 0])
    assert_allclose(b.state[:3], [2, 20, 0])
    assert [m.obstacle_id for m in result.steps[0].measurements] == ["b", "a"]


def test_default_crossing_setup_runs_multiple_cycles():
    setup = build_single_crossing_scenario()
    result = setup.simulator.run(setup.initial_uav_state, setup.goal)
    assert len(result.steps) > 2
    # This is infrastructure validation, not a universal avoidance claim.
    assert result.termination_reason in {"goal_reached", "collision", "max_time"}
    assert all(np.isfinite(s.planner_result.total_force).all() for s in result.steps)
