"""Deterministic integration regressions for the Phase-6 control cycle."""
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from upd_apf.config import UPDAPFConfig, ControlConfig
from upd_apf.data_types import (
    GaussianPrediction, ObstacleMetadata, UAVState, UPDAPFResult, ObstacleRiskResult,
)
from upd_apf.interfaces import Planner
from upd_apf.planner import PlannerObstacle, UPDAPFPlanner
from upd_apf.prediction.kalman_filter import ConstantVelocityKalmanFilter
from upd_apf.prediction.uav_predictor import predict_uav_distribution
from upd_apf.potential import attractive, damping, repulsive
from upd_apf.safety.chance_constraint import confidence_beta, collision_radius, evaluate_safety_sample
from upd_apf.safety.collision_metrics import (
    compute_cpa, compute_cpa_margin, compute_chance_constraint_ttc, build_collision_risk,
)
from upd_apf.safety.relative_geometry import relative_covariance
from upd_apf.utils.saturation import saturate_vector
from upd_apf.utils.time_grid import prediction_time_grid


class LinearPredictor:
    def __init__(self, position, velocity=(0, 0, 0), covariance=None):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.covariance = np.zeros((3, 3)) if covariance is None else np.asarray(covariance)
        self.queries = []

    def predict_distribution(self, tau):
        self.queries.append(tau)
        return GaussianPrediction(17 + tau, self.position + tau * self.velocity, self.velocity, self.covariance)

    def predict(self, dt):
        raise AssertionError("planner must not propagate estimator")

    def update(self, measurement):
        raise AssertionError("planner must not update estimator")


def state(position=(0, 0, 0), velocity=(0, 0, 0), covariance=None):
    return UAVState(position, velocity, np.zeros((3, 3)) if covariance is None else covariance)


def obstacle(position=(3, 0, 0), velocity=(0, 0, 0), covariance=None, radius=0.4, name="a"):
    return PlannerObstacle(ObstacleMetadata(name, radius), LinearPredictor(position, velocity, covariance))


def config(**groups):
    cfg = UPDAPFConfig()
    return replace(cfg, **{name: replace(getattr(cfg, name), **values) for name, values in groups.items()})


def assert_equal_tree(a, b):
    if is_dataclass(a):
        for field in fields(a):
            assert_equal_tree(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal_tree(left, right)
    elif isinstance(a, (str, type(None))):
        assert a == b
    else:
        assert_allclose(a, b, atol=1e-11)


def test_public_api_binding_and_compatibility():
    planner = UPDAPFPlanner(UPDAPFConfig())
    assert isinstance(planner, Planner)
    result = planner.compute_control(state(), [4, 0, 0], [obstacle()])
    assert isinstance(result, UPDAPFResult)
    binding = obstacle()
    with pytest.raises(FrozenInstanceError):
        binding.metadata = ObstacleMetadata("other", 1)
    item = result.per_obstacle_results[0]
    old_item = ObstacleRiskResult(item.obstacle_id, item.safety_samples, item.collision_risk, item.repulsive_force)
    assert old_item.repulsive_force_raw is None and old_item.repulsive_potential is None
    old = UPDAPFResult(*[getattr(result, field.name) for field in fields(UPDAPFResult)][:9])
    assert old.total_force_raw is None and old.reference_velocity is None


@pytest.mark.parametrize("position,velocity,goal", [
    ((0, 0, 0), (0, 0, 0), (4, 1, 0)),
    ((1, 2, 3), (0, 0, 0), (1, 2, 3)),
    ((1, 2, 3), (1, -2, 3), (1, 2, 3)),
])
def test_empty_and_goal(position, velocity, goal):
    cfg = UPDAPFConfig()
    result = UPDAPFPlanner(cfg).compute_control(state(position, velocity), goal, [])
    assert_array_equal(result.repulsive_force, np.zeros(3))
    assert result.global_risk == result.mean_risk == 0
    assert result.min_safety_margin == result.min_ttc == float("inf")
    assert result.per_obstacle_results == ()
    att = attractive.force(position, goal, cfg.attractive.gain, cfg.attractive.switch_distance)
    ref = damping.reference_velocity(position, goal, cfg.uav.desired_speed)
    damp = damping.force(velocity, ref, cfg.damping.gain)
    assert_allclose(result.attractive_force, att)
    assert_allclose(result.reference_velocity, ref)
    assert_allclose(result.damping_force, damp)
    assert_allclose(result.total_force_raw, att + damp)
    assert_allclose(result.total_force, saturate_vector(att + damp, cfg.control.max_total_force))
    if position == goal:
        assert_array_equal(att, np.zeros(3))
        assert_array_equal(ref, np.zeros(3))
        assert_allclose(damp, -cfg.damping.gain * np.asarray(velocity))


def test_goal_preserves_repulsion_and_sign():
    result = UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), [0, 0, 0], [obstacle()])
    assert result.repulsive_force[0] < 0
    assert_array_equal(result.repulsive_force[1:], [0, 0])
    assert_allclose(result.total_force_raw, result.repulsive_force)


@pytest.mark.parametrize("real_kf", [False, True])
@pytest.mark.parametrize("uav_uncertainty", [False, True])
def test_full_chain_matches_direct_lower_layers(real_kf, uav_uncertainty):
    cfg = config(prediction=dict(horizon=1.0, dt=0.3), uav=dict(use_uncertainty=uav_uncertainty))
    uav = state(velocity=(0.3, 0, 0), covariance=np.diag([0.03, 0.02, 0.01]))
    position, velocity = np.array([2., 1., 0.4]), np.array([-2., 0.1, 0.])
    predictor = (ConstantVelocityKalmanFilter(
        np.r_[position, velocity], np.diag([.04, .03, .02, .01, .02, .01]), .05, np.eye(3)*.01, time=17,
    ) if real_kf else LinearPredictor(position, velocity, np.diag([.04, .03, .02])))
    binding = PlannerObstacle(ObstacleMetadata("moving", .7), predictor)
    result = UPDAPFPlanner(cfg).compute_control(uav, [5, 2, 1], [binding])
    item = result.per_obstacle_results[0]
    times = prediction_time_grid(1., .3)
    assert_array_equal([sample.time for sample in item.safety_samples], times)
    assert times[0] == 0 and times[-1] == 1
    assert len(item.safety_samples) == len(times)
    beta = confidence_beta(cfg.chance_constraint.collision_probability)
    radius = collision_radius(cfg.uav.radius, .7, cfg.chance_constraint.extra_safety_margin)
    expected = []
    for t in times:
        obs = predictor.predict_distribution(t)
        up = predict_uav_distribution(uav, t, use_uncertainty=uav_uncertainty)
        expected.append(evaluate_safety_sample(t, obs.position_mean, up.position_mean,
                        relative_covariance(obs.position_covariance, up.position_covariance), beta, radius))
    assert_equal_tree(item.safety_samples, expected)
    t_cpa, d_cpa = compute_cpa(position-uav.position, velocity-uav.velocity, 1.)
    assert not np.any(np.isclose(times, t_cpa))  # exact CPA query is off-grid
    cpa_margin = compute_cpa_margin(t_cpa, predictor.predict_distribution(t_cpa),
        predict_uav_distribution(uav, t_cpa, use_uncertainty=uav_uncertainty), beta, radius)
    ttc = compute_chance_constraint_ttc(expected)
    risk = build_collision_risk(t_cpa, d_cpa, cpa_margin, ttc,
        warning_margin=cfg.risk.cpa_warning_margin, sigmoid_scale=cfg.risk.cpa_sigmoid_scale,
        time_scale=cfg.risk.ttc_time_scale, cpa_weight=cfg.risk.cpa_weight, ttc_weight=cfg.risk.ttc_weight)
    assert_equal_tree(item.collision_risk, risk)
    margins = [s.safety_margin for s in expected]
    gradients = [s.margin_gradient for s in expected]
    raw = repulsive.obstacle_force(margins, gradients, times, risk.total_risk, cfg.repulsive, saturate=False)
    saturated = repulsive.obstacle_force(margins, gradients, times, risk.total_risk, cfg.repulsive, saturate=True)
    assert_allclose(item.repulsive_force_raw, raw)
    assert_allclose(item.repulsive_force, saturated)
    assert_allclose(item.repulsive_potential, repulsive.potential(margins, times, risk.total_risk, cfg.repulsive))
    assert_allclose(result.repulsive_force, saturated)  # no second risk/gain multiplication
    assert 0 < risk.total_risk < 1
    assert not np.allclose(result.repulsive_force, risk.total_risk * saturated)


def test_grid_generated_once_and_shared(monkeypatch):
    import upd_apf.planner.upd_apf_planner as module
    calls = []
    def grid(*args):
        calls.append(args)
        return prediction_time_grid(*args)
    monkeypatch.setattr(module, "prediction_time_grid", grid)
    bindings = [obstacle(name="a"), obstacle((0, 4, 0), name="b")]
    result = UPDAPFPlanner(config(prediction=dict(horizon=1., dt=.3))).compute_control(state(), [4, 0, 0], bindings)
    assert calls == [(1., .3)]
    for binding, item in zip(bindings, result.per_obstacle_results):
        assert_array_equal([s.time for s in item.safety_samples], prediction_time_grid(1., .3))
        assert item.safety_samples[0].time == 0.
        assert item.safety_samples[-1].time == 1.
        assert_allclose(binding.predictor.queries[:5], [0., .3, .6, .9, 1.])


@pytest.mark.parametrize("positions", [[(3, 0, 0), (4, 0, 0)], [(3, 0, 0), (-3, 0, 0)]])
def test_multiple_obstacles_add_or_cancel(positions):
    bindings = [obstacle(p, name=str(i)) for i, p in enumerate(positions)]
    result = UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), [0, 0, 0], bindings)
    forces = [r.repulsive_force for r in result.per_obstacle_results]
    assert_allclose(result.repulsive_force, np.sum(forces, axis=0))
    if positions[1][0] > 0:
        assert result.repulsive_force[0] < min(f[0] for f in forces)
    else:
        assert_allclose(result.repulsive_force, 0, atol=1e-12)


@pytest.mark.parametrize("obstacle_limit,total_limit", [(1000., 1000.), (.4, 1000.), (1000., .2), (.4, .2)])
def test_two_stage_saturation_order(obstacle_limit, total_limit):
    cfg = config(repulsive=dict(max_obstacle_force=obstacle_limit, max_total_repulsive_force=.001),
                 control=dict(max_total_force=total_limit))
    bindings = [obstacle((1.4, 0, 0), name="a"), obstacle((0, 1.8, 0), name="b")]
    result = UPDAPFPlanner(cfg).compute_control(state(), [0, 0, 0], bindings)
    raw = [r.repulsive_force_raw for r in result.per_obstacle_results]
    expected = np.sum([saturate_vector(f, obstacle_limit) for f in raw], axis=0)
    assert_allclose(result.repulsive_force, expected)
    assert_allclose(result.total_force_raw, expected)
    assert_allclose(result.total_force, saturate_vector(expected, total_limit))
    if obstacle_limit == .4:
        for item in result.per_obstacle_results:
            assert np.linalg.norm(item.repulsive_force_raw) > obstacle_limit
            assert_allclose(np.linalg.norm(item.repulsive_force), obstacle_limit)
        assert not np.allclose(expected, saturate_vector(np.sum(raw, axis=0), obstacle_limit))
    else:
        for item in result.per_obstacle_results:
            assert_allclose(item.repulsive_force, item.repulsive_force_raw)
    if total_limit == .2:
        assert_allclose(np.linalg.norm(result.total_force), total_limit)
        assert_allclose(result.total_force/np.linalg.norm(result.total_force), expected/np.linalg.norm(expected))
    else:
        assert_allclose(result.total_force, result.total_force_raw)


def test_summaries_accounting_and_permutation():
    bindings = [obstacle((2, 1, 0), (-.2, 0, 0), name="a"),
                obstacle((-4, 1, 0), radius=.8, name="b"), obstacle((0, 3, 1), radius=.1, name="c")]
    planner = UPDAPFPlanner(UPDAPFConfig())
    result = planner.compute_control(state(velocity=(.1, .2, 0)), [4, 3, 2], bindings)
    other = planner.compute_control(state(velocity=(.1, .2, 0)), [4, 3, 2], [bindings[2], bindings[0], bindings[1]])
    assert [r.obstacle_id for r in other.per_obstacle_results] == ["c", "a", "b"]
    for name in ("repulsive_force", "total_force_raw", "total_force", "global_risk", "mean_risk", "min_safety_margin", "min_ttc"):
        assert_allclose(getattr(result, name), getattr(other, name))
    risks = [r.collision_risk.total_risk for r in result.per_obstacle_results]
    assert result.global_risk == max(risks)
    assert_allclose(result.mean_risk, np.mean(risks))
    assert result.min_safety_margin == min(s.safety_margin for r in result.per_obstacle_results for s in r.safety_samples)
    assert result.min_ttc == min(r.collision_risk.probabilistic_ttc for r in result.per_obstacle_results)
    assert_allclose(result.total_force_raw, result.attractive_force + result.repulsive_force + result.damping_force)


def test_predictor_purity_determinism_and_result_ownership():
    kf = ConstantVelocityKalmanFilter([3, 1, .2, -.3, .1, 0], np.eye(6)*.03, .02, np.eye(3)*.01, time=7)
    before_state, before_cov = kf.state, kf.covariance
    binding = PlannerObstacle(ObstacleMetadata("kf", .4), kf)
    uav = state(velocity=(.2, 0, 0))
    planner = UPDAPFPlanner(UPDAPFConfig())
    first = planner.compute_control(uav, [6, 0, 0], [binding])
    import copy
    snapshot = copy.deepcopy(first)
    second = planner.compute_control(uav, [6, 0, 0], [binding])
    assert_equal_tree(first, second)
    planner.compute_control(state((1, 0, 0)), [0, 4, 0], [])
    assert_equal_tree(first, snapshot)
    assert_array_equal(kf.state, before_state)
    assert_array_equal(kf.covariance, before_cov)
    assert kf.time == 7
    assert not np.shares_memory(first.total_force, second.total_force)
    for array in (first.total_force_raw, first.reference_velocity, first.per_obstacle_results[0].repulsive_force_raw):
        assert not array.flags.writeable


@pytest.mark.parametrize("transform", ["translation", "rotation"])
def test_coordinate_properties(transform):
    cfg = config(uav=dict(use_uncertainty=True), prediction=dict(horizon=1.3, dt=.2), control=dict(max_total_force=.3), repulsive=dict(max_obstacle_force=.5))
    q = np.array([[0.36, -0.8, 0.48], [0.48, 0.6, 0.64], [-0.8, 0., 0.6]]) if transform == "rotation" else np.eye(3)
    offset = np.array([8., -4., 2.]) if transform == "translation" else np.zeros(3)
    assert_allclose(q.T @ q, np.eye(3), atol=1e-14)
    assert_allclose(np.linalg.det(q), 1.)
    uav = state((.1, -.2, .3), (.4, .1, -.1), np.diag([.01, .02, .03]))
    goal = np.array([5., 2., 1.])
    obs = obstacle((2., .4, .8), (-.5, .1, 0), np.diag([.04, .08, .02]))
    planner = UPDAPFPlanner(cfg)
    base = planner.compute_control(uav, goal, [obs])
    moved = planner.compute_control(state(q@uav.position+offset, q@uav.velocity, q@uav.position_covariance@q.T),
        q@goal+offset, [obstacle(q@obs.predictor.position+offset, q@obs.predictor.velocity, q@obs.predictor.covariance@q.T)])
    for name in ("attractive_force", "repulsive_force", "damping_force", "total_force_raw", "total_force", "reference_velocity"):
        assert_allclose(getattr(moved, name), q@getattr(base, name), atol=1e-11)
    for name in ("global_risk", "mean_risk", "min_safety_margin", "min_ttc"):
        assert_allclose(getattr(base, name), getattr(moved, name))
    a, b = base.per_obstacle_results[0], moved.per_obstacle_results[0]
    assert_equal_tree(a.collision_risk, b.collision_risk)
    for left, right in zip(a.safety_samples, b.safety_samples):
        assert_allclose(left.safety_margin, right.safety_margin)
        assert_allclose(right.margin_gradient, q@left.margin_gradient, atol=1e-11)


def test_uncertainty_changes_margins_and_risk():
    planner = UPDAPFPlanner(config(repulsive=dict(max_obstacle_force=1000), control=dict(max_total_force=1000)))
    low = planner.compute_control(state(), [0, 0, 0], [obstacle(covariance=np.eye(3)*.01)])
    high = planner.compute_control(state(), [0, 0, 0], [obstacle(covariance=np.eye(3)*.16)])
    assert high.min_safety_margin < low.min_safety_margin
    assert high.global_risk > low.global_risk
    for result in (low, high):
        item = result.per_obstacle_results[0]
        assert_allclose(item.repulsive_force, item.repulsive_force_raw)
        assert_allclose(result.total_force, result.total_force_raw)


@pytest.mark.parametrize("margin_position,expected", [((.9, 0, 0), 0.), ((20., 0, 0), float("inf"))])
def test_ttc_current_violation_and_safe_horizon(margin_position, expected):
    result = UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), [0, 0, 0], [obstacle(margin_position)])
    assert result.min_ttc == expected


@pytest.mark.parametrize("field", ["position", "velocity", "position_covariance"])
@pytest.mark.parametrize("bad", [[1, 2], [np.nan, 0, 0], [np.inf, 0, 0]])
def test_malformed_uav_rejected(field, bad):
    uav = state()
    object.__setattr__(uav, field, bad)
    with pytest.raises(ValueError):
        UPDAPFPlanner(UPDAPFConfig()).compute_control(uav, [0, 0, 0], [])


@pytest.mark.parametrize("goal", [[1, 2], [[1, 2, 3]], [np.inf, 0, 0], [np.nan, 0, 0]])
def test_invalid_goal(goal):
    with pytest.raises(ValueError, match="goal"):
        UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), goal, [])


@pytest.mark.parametrize("metadata,predictor", [(None, LinearPredictor([3, 0, 0])),
    (ObstacleMetadata("", .3), LinearPredictor([3, 0, 0])),
    (ObstacleMetadata("a", -1), LinearPredictor([3, 0, 0])),
    (ObstacleMetadata("a", np.nan), LinearPredictor([3, 0, 0])),
    (ObstacleMetadata("a", .3), object())])
def test_invalid_binding(metadata, predictor):
    with pytest.raises((TypeError, ValueError)):
        PlannerObstacle(metadata, predictor)


@pytest.mark.parametrize("binding", [object(), LinearPredictor([3, 0, 0]), (ObstacleMetadata("a", .4), LinearPredictor([3, 0, 0]))])
def test_unbound_obstacle_rejected(binding):
    with pytest.raises(TypeError, match="PlannerObstacle"):
        UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), [0, 0, 0], [binding])


@pytest.mark.parametrize("field,bad", [("time", np.nan), ("position_mean", [1, 2]),
    ("velocity_mean", [np.inf, 0, 0]), ("position_covariance", np.eye(2)),
    ("position_covariance", -np.eye(3)), ("position_mean", [np.nan, 0, 0]), (None, None)])
def test_invalid_predictor_output(field, bad):
    class BadPredictor:
        def predict_distribution(self, tau):
            if field is None:
                return object()
            result = GaussianPrediction(tau, [3, 0, 0], [0, 0, 0], np.eye(3))
            object.__setattr__(result, field, bad)
            return result
    binding = PlannerObstacle(ObstacleMetadata("bad", .3), BadPredictor())
    with pytest.raises((TypeError, ValueError)):
        UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), [0, 0, 0], [binding])


@pytest.mark.parametrize("limit", [-1., np.nan, np.inf, -np.inf])
def test_invalid_total_limit(limit):
    with pytest.raises(ValueError, match="max_total_force"):
        replace(UPDAPFConfig(), control=ControlConfig(limit))


def test_zero_total_limit():
    result = UPDAPFPlanner(config(control=dict(max_total_force=0))).compute_control(state(), [4, 0, 0], [])
    assert np.linalg.norm(result.total_force_raw) > 0
    assert_array_equal(result.total_force, np.zeros(3))


@pytest.mark.parametrize("position,velocity", [((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (-1, 0, 0))])
def test_degenerate_geometry_propagates(position, velocity):
    with pytest.raises(ValueError, match="degenerate"):
        UPDAPFPlanner(config(prediction=dict(horizon=1., dt=.3))).compute_control(state(), [5, 0, 0], [obstacle(position, velocity)])


def test_symmetric_head_on_has_no_lateral_escape():
    result = UPDAPFPlanner(config(prediction=dict(horizon=1.))).compute_control(
        state(velocity=(1, 0, 0)), [10, 0, 0], [obstacle((5, 0, 0), (-1, 0, 0), np.eye(3)*.04)])
    assert_array_equal(result.total_force[1:], [0, 0])
    assert_array_equal(result.repulsive_force[1:], [0, 0])


def test_ttc_crossing_is_interpolated_by_lower_layer():
    cfg = config(prediction=dict(horizon=1., dt=.3))
    result = UPDAPFPlanner(cfg).compute_control(state(), [0, 0, 0], [obstacle((2, 0, 0), (-1, 0, 0))])
    item = result.per_obstacle_results[0]
    assert_allclose(result.min_ttc, .8)
    assert result.min_ttc == compute_chance_constraint_ttc(item.safety_samples)


def test_each_obstacle_uses_its_own_radius():
    result = UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), [0, 0, 0],
        [obstacle(radius=.2, name="small"), obstacle(radius=.8, name="large")])
    small, large = result.per_obstacle_results
    assert_allclose([s.safety_margin for s in small.safety_samples],
                    np.array([s.safety_margin for s in large.safety_samples]) + .6)
    assert large.collision_risk.total_risk > small.collision_risk.total_risk


def test_off_grid_degenerate_cpa_propagates():
    # Grid 0,.3,.6,.9,1 is nondegenerate; exact CPA at .5 is coincident.
    with pytest.raises(ValueError, match="degenerate"):
        UPDAPFPlanner(config(prediction=dict(horizon=1., dt=.3))).compute_control(
            state(), [3, 0, 0], [obstacle((1, 0, 0), (-2, 0, 0))])


def test_frozen_risk_is_built_once_and_passed_unchanged(monkeypatch):
    import upd_apf.planner.upd_apf_planner as module
    real_build = module.build_collision_risk
    real_force = repulsive.obstacle_force
    real_potential = repulsive.potential
    risks, force_calls, potential_calls = [], [], []
    def build(*args, **kwargs):
        value = real_build(*args, **kwargs)
        risks.append(value.total_risk)
        return value
    def force(margins, gradients, times, risk, cfg, *, saturate):
        force_calls.append((risk, saturate, times))
        return real_force(margins, gradients, times, risk, cfg, saturate=saturate)
    def potential(margins, times, risk, cfg):
        potential_calls.append((risk, times))
        return real_potential(margins, times, risk, cfg)
    monkeypatch.setattr(module, "build_collision_risk", build)
    monkeypatch.setattr(repulsive, "obstacle_force", force)
    monkeypatch.setattr(repulsive, "potential", potential)
    UPDAPFPlanner(UPDAPFConfig()).compute_control(state(), [4, 0, 0],
        [obstacle(name="a"), obstacle((4, 1, 0), name="b")])
    assert len(risks) == len(potential_calls) == 2
    assert [(r, sat) for r, sat, _ in force_calls] == [(risks[0], False), (risks[0], True), (risks[1], False), (risks[1], True)]
    assert [r for r, _ in potential_calls] == risks
    assert all(t is force_calls[0][2] for _, _, t in force_calls)


@pytest.mark.parametrize("potential", [0, 1.5, np.float32(2.5), np.array(3.)])
def test_repulsive_potential_is_stored_as_python_float(potential):
    item = UPDAPFPlanner(UPDAPFConfig()).compute_control(
        state(), [0, 0, 0], [obstacle()]).per_obstacle_results[0]
    result = replace(item, repulsive_potential=potential)
    assert type(result.repulsive_potential) is float
    assert result.repulsive_potential == float(potential)


@pytest.mark.parametrize("potential", [-1., np.nan, np.inf, -np.inf, [1.], np.array([1., 2.]), np.ones((1, 1))])
def test_invalid_repulsive_potential_rejected(potential):
    item = UPDAPFPlanner(UPDAPFConfig()).compute_control(
        state(), [0, 0, 0], [obstacle()]).per_obstacle_results[0]
    with pytest.raises(ValueError, match="repulsive_potential"):
        replace(item, repulsive_potential=potential)
