"""Goal-directed reference velocity and linear velocity damping."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from upd_apf.config import NumericalConfig

from ._validation import scalar, vector


def reference_velocity(
    position: ArrayLike,
    goal: ArrayLike,
    desired_speed: float,
    *,
    eps_distance: float = NumericalConfig().eps_distance,
) -> NDArray[np.float64]:
    """Return goal-directed velocity (3,) in m/s, or zero within eps_distance.

    Position and goal have shape (3,) in metres; desired_speed is non-negative.
    No fallback direction is introduced at the goal.
    """
    error = vector(goal, "goal") - vector(position, "position")
    speed = scalar(desired_speed, "desired_speed")
    threshold = scalar(eps_distance, "eps_distance")
    distance = float(np.linalg.norm(error))
    if distance <= threshold:
        return np.zeros(3)
    return speed * (error / distance)


def force(
    velocity: ArrayLike, reference_velocity: ArrayLike, gain: float,
) -> NDArray[np.float64]:
    """Return -gain*(velocity-reference_velocity), all vectors shape (3,).

    Velocities are in m/s; gain is non-negative. At goal, use a zero reference.
    """
    velocity = vector(velocity, "velocity")
    reference = vector(reference_velocity, "reference_velocity")
    return scalar(gain, "gain") * (reference - velocity)
