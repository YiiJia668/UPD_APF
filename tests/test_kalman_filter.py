import numpy as np

from upd_apf.prediction.kalman_filter import ConstantVelocityKalmanFilter
from upd_apf.prediction.process_noise import cv_transition, white_acceleration_process_noise


def make_filter():
    return ConstantVelocityKalmanFilter(
        [0, 0, 0, 1, 2, 3], np.eye(6), .2, .1 * np.eye(3), time=5
    )


def test_predict_matches_analytic_propagation():
    kf = make_filter()
    expected = cv_transition(.5) @ kf.state
    kf.predict(.5)
    assert np.allclose(kf.state, expected)
    assert kf.time == 5.5


def test_predict_covariance_includes_cross_covariance_and_q():
    kf = make_filter()
    a = cv_transition(.5)
    expected = a @ np.eye(6) @ a.T + white_acceleration_process_noise(.5, .2)
    kf.predict(.5)
    assert np.allclose(kf.covariance, expected)


def test_zero_horizon_distribution_equals_posterior():
    kf = make_filter()
    prediction = kf.predict_distribution(0)
    assert np.array_equal(prediction.position_mean, kf.state[:3])
    assert np.array_equal(prediction.position_covariance, kf.covariance[:3, :3])


def test_prediction_query_is_pure():
    kf = make_filter()
    state, covariance, time = kf.state, kf.covariance, kf.time
    kf.predict_distribution(3)
    assert np.array_equal(kf.state, state)
    assert np.array_equal(kf.covariance, covariance)
    assert kf.time == time


def test_update_reduces_position_variance():
    kf = make_filter()
    before = np.trace(kf.covariance[:3, :3])
    kf.update([1, 1, 1])
    assert np.trace(kf.covariance[:3, :3]) < before


def test_joseph_update_stays_symmetric_psd():
    kf = make_filter()
    for _ in range(20):
        kf.predict(.1)
        kf.update([1, 2, 3])
    assert np.array_equal(kf.covariance, kf.covariance.T)
    assert np.linalg.eigvalsh(kf.covariance).min() >= -1e-12


def test_properties_return_copies():
    kf = make_filter()
    state = kf.state
    state[0] = 99
    assert kf.state[0] == 0
