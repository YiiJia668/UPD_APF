import numpy as np
import pytest

from upd_apf.safety.chance_constraint import (
    collision_radius,
    confidence_beta,
    directional_sigma,
    evaluate_safety_sample,
    margin_gradient,
    safety_margin,
)


def test_lower_collision_probability_gives_larger_beta():
    assert confidence_beta(.001) > confidence_beta(.01)


def test_probability_validation():
    with pytest.raises(ValueError):
        confidence_beta(.5)


def test_collision_radius_sums_components():
    assert collision_radius(.3, .7, .5) == 1.5


def test_known_directional_sigma():
    assert directional_sigma([1, 0, 0], np.diag([4, 1, 1])) == 2


def test_zero_uncertainty_is_physically_zero():
    sigma = directional_sigma([1, 0, 0], np.zeros((3, 3)))
    assert sigma == 0
    assert safety_margin(4, 2, sigma, 1) == 3


def test_materially_non_psd_covariance_rejected():
    with pytest.raises(ValueError, match="PSD"):
        directional_sigma([0, 1, 0], np.diag([1, -1, 1]))


def test_isotropic_gradient_is_negative_direction():
    n = np.array([.6, .8, 0])
    gradient = margin_gradient(n, 5, 2 * np.eye(3), 2)
    assert np.allclose(gradient, -n)


def test_zero_uncertainty_gradient_branch():
    assert np.array_equal(margin_gradient([1, 0, 0], 4, np.zeros((3, 3)), 2), [-1, 0, 0])


def test_radial_gradient_identity():
    n = np.array([.6, .8, 0])
    gradient = margin_gradient(n, 5, np.diag([4, 1, 2]), 2)
    assert np.isclose(n @ gradient, -1)


def test_anisotropic_gradient_matches_finite_difference():
    obstacle = np.array([4., 2., 1.])
    uav = np.array([.2, -.3, .1])
    covariance = np.array([[3., .2, 0], [.2, 1., .1], [0, .1, 2.]])
    beta, radius = 2., 1.

    def margin(position):
        mean = obstacle - position
        distance = np.linalg.norm(mean)
        n = mean / distance
        return safety_margin(distance, beta, directional_sigma(n, covariance), radius)

    mean = obstacle - uav
    distance = np.linalg.norm(mean)
    n = mean / distance
    analytic = margin_gradient(n, distance, covariance, beta)
    h = 1e-6
    numeric = np.array([(margin(uav + h * np.eye(3)[i]) - margin(uav - h * np.eye(3)[i])) / (2*h) for i in range(3)])
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_evaluate_sample_uses_shared_dataclass():
    sample = evaluate_safety_sample(1, [5, 0, 0], [0, 0, 0], np.eye(3), 2, 1)
    assert sample.time == 1 and sample.safety_margin == 2
    assert np.array_equal(sample.margin_gradient, [-1, 0, 0])


def test_rotation_consistency():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    n = np.array([1., 0, 0])
    covariance = np.diag([4., 1, 2])
    original = margin_gradient(n, 3, covariance, 2)
    rotated = margin_gradient(rotation @ n, 3, rotation @ covariance @ rotation.T, 2)
    assert np.allclose(rotated, rotation @ original)
