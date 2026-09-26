"""Exact zero-order-held force integration, without secondary saturation."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from upd_apf.data_types import UAVState, _array


@dataclass(frozen=True)
class PointMassUAVModel:
    force_to_acceleration_gain: float = 1.0

    def __post_init__(self):
        gain = self.force_to_acceleration_gain
        if not np.isscalar(gain) or not np.isfinite(gain) or gain <= 0:
            raise ValueError("force_to_acceleration_gain must be finite and positive")

    def step(self, state: UAVState, control_force: ArrayLike, dt: float) -> UAVState:
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        state = UAVState(state.position, state.velocity, state.position_covariance)
        force = _array(control_force, (3,), "control_force")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            acceleration = self.force_to_acceleration_gain * force
            position = state.position + state.velocity * dt + 0.5 * acceleration * dt * dt
            velocity = state.velocity + acceleration * dt
        # UAVState validates finite outputs and independently copies covariance.
        return UAVState(position, velocity, state.position_covariance)
