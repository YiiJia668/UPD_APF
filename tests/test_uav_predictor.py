import numpy as np

from upd_apf.data_types import UAVState
from upd_apf.prediction.uav_predictor import UAVPredictor, predict_position, predict_uav_distribution


def state():
    return UAVState([1, 2, 3], [2, 0, -1], 2 * np.eye(3))


def test_predict_position():
    assert np.allclose(predict_position([1, 2, 3], [2, 0, -1], 2), [5, 2, 1])


def test_uncertainty_disabled_returns_zero_covariance():
    assert np.array_equal(predict_uav_distribution(state(), 1).position_covariance, np.zeros((3, 3)))


def test_uncertainty_enabled_preserves_covariance():
    result = UAVPredictor(state(), use_uncertainty=True).predict_distribution(2)
    assert np.array_equal(result.position_covariance, 2 * np.eye(3))
    assert np.allclose(result.position_mean, [5, 2, 1])
