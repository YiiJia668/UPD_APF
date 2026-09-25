import numpy as np
import pytest

from upd_apf.data_types import GaussianPrediction, UAVState


def test_uav_state_copies_inputs_and_makes_arrays_read_only():
    position = np.zeros(3)
    state = UAVState(position, np.ones(3), np.eye(3))
    position[0] = 4
    assert state.position[0] == 0
    with pytest.raises(ValueError):
        state.position[0] = 2


def test_vector_shape_is_validated():
    with pytest.raises(ValueError, match="shape"):
        UAVState(np.zeros(2), np.zeros(3), np.eye(3))


def test_prediction_fields_are_normalized_arrays():
    prediction = GaussianPrediction(0, [1, 2, 3], [0, 0, 0], np.eye(3))
    assert prediction.position_mean.dtype == float
