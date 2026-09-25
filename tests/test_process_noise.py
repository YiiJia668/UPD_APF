import numpy as np
import pytest

from upd_apf.prediction.process_noise import cv_transition, white_acceleration_process_noise


def test_cv_transition_propagates_position():
    x = np.array([1, 2, 3, 4, 5, 6.])
    assert np.allclose(cv_transition(2) @ x, [9, 12, 15, 4, 5, 6])


def test_zero_time_matrices():
    assert np.array_equal(cv_transition(0), np.eye(6))
    assert np.array_equal(white_acceleration_process_noise(0, .3), np.zeros((6, 6)))


def test_white_acceleration_blocks():
    q = white_acceleration_process_noise(2, 3)
    assert np.allclose(q[:3, :3], 8 * np.eye(3))
    assert np.allclose(q[:3, 3:], 6 * np.eye(3))
    assert np.allclose(q[3:, 3:], 6 * np.eye(3))


def test_matrix_spectral_density():
    sa = np.diag([1, 2, 3])
    assert np.allclose(white_acceleration_process_noise(1, sa)[3:, 3:], sa)


def test_process_noise_is_symmetric_psd():
    q = white_acceleration_process_noise(.7, .2)
    assert np.allclose(q, q.T)
    assert np.linalg.eigvalsh(q).min() >= -1e-12


@pytest.mark.parametrize("tau", [-1, np.inf])
def test_invalid_time(tau):
    with pytest.raises(ValueError):
        cv_transition(tau)


def test_negative_density_rejected():
    with pytest.raises(ValueError):
        white_acceleration_process_noise(1, -1)
