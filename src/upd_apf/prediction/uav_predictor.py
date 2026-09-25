"""Pure constant-velocity prediction for the UAV."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from upd_apf.data_types import GaussianPrediction, UAVState


def predict_position(position: ArrayLike, velocity: ArrayLike, tau: float) -> NDArray[np.float64]:
    if tau < 0 or not np.isfinite(tau):
        raise ValueError("prediction time must be finite and non-negative")
    position_array = np.asarray(position, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    if position_array.shape != (3,) or velocity_array.shape != (3,):
        raise ValueError("position and velocity must have shape (3,)")
    return position_array + float(tau) * velocity_array


def predict_uav_distribution(
    state: UAVState,
    tau: float,
    *,
    use_uncertainty: bool = False,
) -> GaussianPrediction:
    covariance = state.position_covariance if use_uncertainty else np.zeros((3, 3))
    return GaussianPrediction(
        time=float(tau),
        position_mean=predict_position(state.position, state.velocity, tau),
        velocity_mean=state.velocity,
        position_covariance=covariance,
    )


class UAVPredictor:
    def __init__(self, state: UAVState, *, use_uncertainty: bool = False) -> None:
        self.state = state
        self.use_uncertainty = use_uncertainty

    def predict_distribution(self, tau: float) -> GaussianPrediction:
        return predict_uav_distribution(self.state, tau, use_uncertainty=self.use_uncertainty)
