"""Relative position and covariance geometry."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from upd_apf.utils.covariance import validate_psd


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of shape (3,)")
    return result


def relative_mean(obstacle_position: ArrayLike, uav_position: ArrayLike) -> NDArray[np.float64]:
    """Return ``p_obstacle - p_uav`` (a vector pointing toward the obstacle)."""
    return _vector(obstacle_position, "obstacle_position") - _vector(uav_position, "uav_position")


def relative_covariance(obstacle_covariance: ArrayLike, uav_covariance: ArrayLike) -> NDArray[np.float64]:
    obstacle = validate_psd(obstacle_covariance)
    uav = validate_psd(uav_covariance)
    if obstacle.shape != (3, 3) or uav.shape != (3, 3):
        raise ValueError("position covariances must have shape (3, 3)")
    return validate_psd(obstacle + uav)


def distance(relative_position: ArrayLike) -> float:
    return float(np.linalg.norm(_vector(relative_position, "relative_position")))


def unit_direction(relative_position: ArrayLike, eps_distance: float = 1e-8) -> NDArray[np.float64]:
    relative = _vector(relative_position, "relative_position")
    separation = float(np.linalg.norm(relative))
    if separation <= eps_distance:
        raise ValueError("degenerate relative geometry: distance is at or below eps_distance")
    return relative / separation


def relative_geometry(
    obstacle_position: ArrayLike,
    uav_position: ArrayLike,
    eps_distance: float = 1e-8,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    mean = relative_mean(obstacle_position, uav_position)
    separation = distance(mean)
    return mean, separation, unit_direction(mean, eps_distance)
