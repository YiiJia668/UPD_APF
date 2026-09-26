from dataclasses import replace

import numpy as np
import pytest

from upd_apf.config import RepulsiveConfig
from upd_apf.potential.repulsive import (
    obstacle_force,
    potential,
    risk_modulated_gain,
    temporal_weights,
)
from upd_apf.safety.chance_constraint import evaluate_safety_sample
from upd_apf.utils.time_grid import prediction_time_grid


def test_temporal_grid_weights_and_repeatability():
    times = prediction_time_grid(1., .3)
    assert times[0] == 0 and times[-1] == 1
    weights = temporal_weights(times, .7)
    expected = np.exp(-.7 * times)
    np.testing.assert_allclose(weights, expected / expected.sum(), rtol=1e-14)
    assert weights.sum() == pytest.approx(1)
    assert np.all(weights >= 0) and np.all(np.diff(weights) <= 0)
    assert weights[0] > 0
    np.testing.assert_array_equal(weights, temporal_weights(times, .7))
    np.testing.assert_allclose(temporal_weights(times, 0), np.full(len(times), 1 / len(times)))


def test_temporal_numerical_stability():
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(temporal_weights([1e6, 1e6 + 1], 1e6), [1, 0])
        np.testing.assert_array_equal(temporal_weights([0, 1e308], 1e308), [1, 0])
        np.testing.assert_array_equal(temporal_weights([4], 1e308), [1])


@pytest.mark.parametrize("times,decay", [
    ([], .7), ([[0, 1]], .7), (0, .7), ([-1, 0], .7),
    ([0, np.nan], .7), ([0, np.inf], .7), ([0, 1], -1),
    ([0, 1], np.nan), ([0, 1], np.inf),
])
def test_invalid_temporal_inputs(times, decay):
    with pytest.raises(ValueError):
        temporal_weights(times, decay)


def test_frozen_gain_endpoints_monotonicity_and_zero_parameters():
    assert risk_modulated_gain(0, 2, 3) == 2
    assert risk_modulated_gain(1, 2, 3) == 8
    gains = [risk_modulated_gain(risk, 2, 3) for risk in (0, .3, .7, 1)]
    assert np.all(np.diff(gains) > 0)
    assert risk_modulated_gain(0, 2, 0) == risk_modulated_gain(1, 2, 0) == 2
    assert risk_modulated_gain(.5, 0, 3) == 0
    assert isinstance(risk_modulated_gain(.5, 2, 3), float)


@pytest.mark.parametrize("risk,base,gain", [
    (-.1, 2, 3), (1.1, 2, 3), (np.nan, 2, 3), (np.inf, 2, 3),
    (.5, -1, 3), (.5, np.nan, 3), (.5, np.inf, 3),
    (.5, 2, -1), (.5, 2, np.nan), (.5, 2, np.inf),
])
def test_invalid_gain_inputs(risk, base, gain):
    with pytest.raises(ValueError):
        risk_modulated_gain(risk, base, gain)


@pytest.mark.parametrize("margins,times,gradients", [
    ([.6], [0], [[-1, 0, 0]]),
    ([2., .5, -.2], [0, .4, 1.], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
])
def test_exact_unclipped_unsaturated_formula(margins, times, gradients):
    config = RepulsiveConfig(max_obstacle_force=1e6)
    risk = .4
    weights = np.exp(-config.future_decay * np.array(times))
    weights /= weights.sum()
    response = weights * np.exp(-np.array(margins) / config.length_scale)
    eta = config.eta_0 * (1 + config.risk_gain * risk)
    expected_u = eta * response.sum()
    expected_f = eta / config.length_scale * (response @ np.array(gradients))
    actual_u = potential(margins, times, risk, config)
    actual_f = obstacle_force(margins, gradients, times, risk, config, saturate=False)
    assert isinstance(actual_u, float) and np.isfinite(actual_u) and actual_u >= 0
    assert actual_f.shape == (3,) and actual_f.dtype == np.float64
    assert actual_u == pytest.approx(expected_u, rel=1e-14)
    np.testing.assert_allclose(actual_f, expected_f, rtol=1e-14)
    np.testing.assert_array_equal(
        obstacle_force(margins, gradients, times, risk, config), actual_f,
    )


def test_margin_and_risk_monotonicity():
    margins = [-2., 0., 2., 4.]
    potentials = [potential([c], [0], .2) for c in margins]
    norms = [np.linalg.norm(obstacle_force([c], [[-1, 0, 0]], [0], .2, saturate=False))
             for c in margins]
    assert np.all(np.diff(potentials) < 0) and np.all(np.diff(norms) < 0)
    low_u, high_u = potential([1], [0], 0), potential([1], [0], 1)
    low_f = obstacle_force([1], [[-1, 0, 0]], [0], 0, saturate=False)
    high_f = obstacle_force([1], [[-1, 0, 0]], [0], 1, saturate=False)
    assert high_u == pytest.approx(3 * low_u)
    np.testing.assert_allclose(high_f, 3 * low_f)


def test_discount_reduces_identical_future_threat_contribution():
    uniform = RepulsiveConfig(future_decay=0)
    discounted = replace(uniform, future_decay=2)
    # The stronger threat is in the future.
    margins, gradients, times = [4., -1.], [[-1, 0, 0]] * 2, [0, 2]
    assert potential(margins, times, 0, discounted) < potential(margins, times, 0, uniform)
    assert np.linalg.norm(obstacle_force(margins, gradients, times, 0, discounted)) < np.linalg.norm(
        obstacle_force(margins, gradients, times, 0, uniform),
    )


@pytest.mark.parametrize("variance", [0., .25, 1.])
def test_positive_leading_sign_pushes_away_without_lateral_bias(variance):
    sample = evaluate_safety_sample(
        0, [4, 0, 0], [0, 0, 0], variance * np.eye(3), 2, 1,
    )
    np.testing.assert_allclose(sample.margin_gradient, [-1, 0, 0])
    result = obstacle_force(
        [sample.safety_margin], [sample.margin_gradient], [sample.time], .3,
        saturate=False,
    )
    assert result[0] < 0  # An accidental extra minus MUST fail this assertion.
    np.testing.assert_array_equal(result[1:], [0, 0])


def test_isotropic_uncertainty_strengthens_repulsion():
    results = []
    for variance in (0., .25, 1.):
        sample = evaluate_safety_sample(
            0, [4, 0, 0], [0, 0, 0], variance * np.eye(3), 2, 1,
        )
        results.append(potential([sample.safety_margin], [0], .2))
    assert np.all(np.diff(results) > 0)


@pytest.mark.parametrize("covariance", [
    np.zeros((3, 3)), .2 * np.eye(3),
    np.array([[.4, .08, 0], [.08, .2, .03], [0, .03, .3]]),
])
def test_raw_force_matches_phase3_central_difference_with_frozen_risk(covariance):
    p = np.array([.2, -.3, .1])
    times = prediction_time_grid(1., .4)
    positions = np.array([[4., 2., 1.], [3.8, 2.1, 1.], [3.6, 2.2, 1.], [3.5, 2.3, 1.]])
    # These are frozen across every position perturbation.
    risk, config = .4, RepulsiveConfig(max_obstacle_force=100.)
    weights = temporal_weights(times, config.future_decay)
    eta = risk_modulated_gain(risk, config.eta_0, config.risk_gain)

    def samples(position):
        evaluated = [
            evaluate_safety_sample(t, obstacle, position, covariance, 2., 1.)
            for t, obstacle in zip(times, positions)
        ]
        return (np.array([s.safety_margin for s in evaluated]),
                np.array([s.margin_gradient for s in evaluated]))

    def energy(position):
        margins, _ = samples(position)
        assert np.all(-margins / config.length_scale > config.exponent_clip_min)
        assert np.all(-margins / config.length_scale < config.exponent_clip_max)
        assert potential(margins, times, risk, config) == pytest.approx(
            eta * np.sum(weights * np.exp(-margins / config.length_scale)),
        )
        return potential(margins, times, risk, config)

    margins, gradients = samples(p)
    raw = obstacle_force(margins, gradients, times, risk, config, saturate=False)
    assert np.linalg.norm(raw) < config.max_obstacle_force
    np.testing.assert_array_equal(obstacle_force(margins, gradients, times, risk, config), raw)
    h = 1e-6
    numeric = np.array([
        (energy(p + h * axis) - energy(p - h * axis)) / (2 * h)
        for axis in np.eye(3)
    ])
    np.testing.assert_allclose(raw, -numeric, rtol=2e-8, atol=2e-9)


def test_active_single_obstacle_saturation_preserves_direction_and_potential():
    config = RepulsiveConfig(max_obstacle_force=.3)
    args = ([-2., -.5], [[-1., 2., 0], [0, 0, -1.]], [0, 1], .5)
    raw = obstacle_force(*args, config, saturate=False)
    bounded = obstacle_force(*args, config)
    assert np.linalg.norm(raw) > config.max_obstacle_force
    assert np.linalg.norm(bounded) == pytest.approx(config.max_obstacle_force)
    np.testing.assert_allclose(bounded / np.linalg.norm(bounded), raw / np.linalg.norm(raw))
    assert potential(args[0], args[2], args[3], config) == potential(
        args[0], args[2], args[3], replace(config, max_obstacle_force=1000),
    )
    np.testing.assert_array_equal(
        obstacle_force(*args, replace(config, max_obstacle_force=0)), np.zeros(3),
    )


def test_extreme_margins_are_clipped_without_nan_inf_or_warnings():
    config = RepulsiveConfig(length_scale=.5)
    with np.errstate(all="raise"):
        u = potential([-1e308], [0], .4, config)
        raw = obstacle_force([-1e308], [[-1, 0, 0]], [0], .4, config, saturate=False)
        bounded = obstacle_force([-1e308], [[-1, 0, 0]], [0], .4, config)
        far = potential([1e308], [0], .4, config)
    eta = risk_modulated_gain(.4, config.eta_0, config.risk_gain)
    assert u == pytest.approx(eta * np.exp(config.exponent_clip_max))
    assert far == pytest.approx(eta * np.exp(config.exponent_clip_min), abs=0)
    assert np.isfinite(u) and np.all(np.isfinite(raw))
    np.testing.assert_allclose(bounded, [-config.max_obstacle_force, 0, 0])


def test_configured_clipping_and_zero_gain():
    config = RepulsiveConfig(exponent_clip_min=-2, exponent_clip_max=2)
    assert potential([-100], [0], 0, config) == pytest.approx(config.eta_0 * np.exp(2))
    assert potential([100], [0], 0, config) == pytest.approx(config.eta_0 * np.exp(-2))
    disabled = replace(config, eta_0=0)
    assert potential([-100], [0], 1, disabled) == 0
    np.testing.assert_array_equal(
        obstacle_force([-100], [[-1, 0, 0]], [0], 1, disabled), np.zeros(3),
    )


def test_no_mutation_and_determinism():
    margins = np.array([1., 2.])
    gradients = np.array([[-1., 0, 0], [0, -1., 0]])
    times = np.array([0., 1.])
    copies = [a.copy() for a in (margins, gradients, times)]
    for value in (margins, gradients, times):
        value.setflags(write=False)
    first = obstacle_force(margins, gradients, times, .5)
    assert potential(margins, times, .5) == potential(margins, times, .5)
    np.testing.assert_array_equal(first, obstacle_force(margins, gradients, times, .5))
    for value, before in zip((margins, gradients, times), copies):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("field,value", [
    ("length_scale", 0), ("length_scale", -1), ("length_scale", np.nan),
    ("length_scale", np.inf), ("eta_0", -1), ("eta_0", np.inf),
    ("risk_gain", -1), ("risk_gain", np.nan), ("future_decay", -1),
    ("max_obstacle_force", -1), ("max_obstacle_force", np.nan),
    ("max_obstacle_force", np.inf), ("exponent_clip_min", np.nan),
    ("exponent_clip_max", np.inf), ("exponent_clip_max", 1000),
    ("exponent_clip_min", 51),
])
def test_invalid_repulsive_configuration(field, value):
    config = replace(RepulsiveConfig(), **{field: value})
    with pytest.raises(ValueError):
        potential([1], [0], .5, config)
    with pytest.raises(ValueError):
        obstacle_force([1], [[-1, 0, 0]], [0], .5, config)


@pytest.mark.parametrize("margins,times", [
    ([], []), ([1, 2], [0]), ([[1]], [0]), ([np.nan], [0]),
    ([np.inf], [0]), ([1], [np.nan]), ([1], [-1]),
])
def test_invalid_margin_time_arrays(margins, times):
    with pytest.raises(ValueError):
        potential(margins, times, .5)
    with pytest.raises(ValueError):
        obstacle_force(margins, [[-1, 0, 0]], times, .5)


@pytest.mark.parametrize("gradients", [
    [-1, 0, 0], [[-1, 0]], [[-1, 0, 0], [-1, 0, 0]],
    [[np.nan, 0, 0]], [[0, np.inf, 0]],
])
def test_invalid_gradient_arrays(gradients):
    with pytest.raises(ValueError):
        obstacle_force([1], gradients, [0], .5)


@pytest.mark.parametrize("risk", [-.1, 1.1, np.nan, np.inf])
def test_repulsive_rejects_invalid_risk(risk):
    with pytest.raises(ValueError):
        potential([1], [0], risk)
    with pytest.raises(ValueError):
        obstacle_force([1], [[-1, 0, 0]], [0], risk)


def test_large_configured_exponent_saturates_without_norm_overflow():
    config = RepulsiveConfig(
        eta_0=1, risk_gain=0, length_scale=1, exponent_clip_max=400,
    )
    args = ([-500], [[-1., .5, .25]], [0], 0)
    with np.errstate(all="raise"):
        raw = obstacle_force(*args, config, saturate=False)
        bounded = obstacle_force(*args, config)
        unbounded_config = replace(config, max_obstacle_force=1e200)
        unchanged = obstacle_force(*args, unbounded_config)
    direction = np.array([-1., .5, .25])
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(bounded, config.max_obstacle_force * direction)
    np.testing.assert_array_equal(unchanged, raw)


def test_unrepresentable_gain_or_output_is_explicitly_rejected():
    with pytest.raises(ValueError, match="floating-point range"):
        risk_modulated_gain(1, 1e308, 10)
    config = RepulsiveConfig(eta_0=1e308)
    with pytest.raises(ValueError, match="floating-point range"):
        potential([-100], [0], 0, config)
    with pytest.raises(ValueError, match="floating-point range"):
        obstacle_force([-100], [[-1, 0, 0]], [0], 0, config)


def test_repulsive_rotation_covariance():
    rotation = np.array([[.6, -.8, 0], [.8, .6, 0], [0, 0, 1.]])
    gradients = np.array([[-1., .2, 0], [0, -1., .3]])
    margins, times = [1., .5], [0, 1]
    for limit in (.1, 100):
        config = RepulsiveConfig(max_obstacle_force=limit)
        original = obstacle_force(margins, gradients, times, .3, config)
        rotated = obstacle_force(margins, gradients @ rotation.T, times, .3, config)
        np.testing.assert_allclose(rotated, rotation @ original, atol=1e-14)


@pytest.mark.parametrize("lower,upper", [(10., 50.), (-50., -10.)])
def test_exponent_clipping_interval_must_contain_zero(lower, upper):
    config = RepulsiveConfig(
        exponent_clip_min=lower, exponent_clip_max=upper,
    )
    with pytest.raises(ValueError, match="contain zero"):
        potential([0], [0], 0, config)
    with pytest.raises(ValueError, match="contain zero"):
        obstacle_force([0], [[-1, 0, 0]], [0], 0, config)


@pytest.mark.parametrize("lower,upper", [
    (-50., 50.), (0., 50.), (-50., 0.), (0., 0.),
])
def test_zero_margin_has_exact_unit_exponential_response(lower, upper):
    # A single sample and unit eta/length remove all multiplicative factors.
    config = RepulsiveConfig(
        eta_0=1, risk_gain=0, length_scale=1,
        exponent_clip_min=lower, exponent_clip_max=upper,
    )
    assert potential([0], [0], 0, config) == 1.0
    np.testing.assert_array_equal(
        obstacle_force(
            [0], [[-1, 0, 0]], [0], 0, config, saturate=False,
        ),
        [-1., 0., 0.],
    )
