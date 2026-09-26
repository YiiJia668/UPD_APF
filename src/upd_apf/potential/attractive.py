"""Quadratic-near / conic-far goal attraction, without extra saturation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import scalar, vector


def _inputs(
    position: ArrayLike, goal: ArrayLike, gain: float, switch_distance: float,
) -> tuple[NDArray[np.float64], float, float, float]:
    error = vector(goal, "goal") - vector(position, "position")
    gain = scalar(gain, "gain")
    switch_distance = scalar(switch_distance, "switch_distance", positive=True)
    return error, float(np.linalg.norm(error)), gain, switch_distance


def potential(
    position: ArrayLike, goal: ArrayLike, gain: float, switch_distance: float,
) -> float:
    """Return scalar attraction for positions (3,) and switch distance in metres.

    U = gain*d**2/2 inside the switch, otherwise gain*switch*(d-switch/2).
    """
    _, distance, gain, switch = _inputs(position, goal, gain, switch_distance)
    if distance <= switch:
        return float(0.5 * gain * distance**2)
    return float(gain * switch * (distance - 0.5 * switch))


def force(
    position: ArrayLike, goal: ArrayLike, gain: float, switch_distance: float,
) -> NDArray[np.float64]:
    """Return -grad(U), shape (3,), toward goal; far norm is gain*switch_distance."""
    error, distance, gain, switch = _inputs(position, goal, gain, switch_distance)
    if distance <= switch:
        return gain * error
    return gain * switch * (error / distance)
