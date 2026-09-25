"""Chance-constrained safety margin and its UAV-position gradient."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from upd_apf.data_types import SafetySample
from upd_apf.utils.covariance import validate_psd

from .relative_geometry import relative_geometry


def confidence_beta(collision_probability: float) -> float:
    if not 0 < collision_probability < 0.5:
        raise ValueError("collision probability must be in (0, 0.5)")
    return float(norm.ppf(1.0 - collision_probability))


def collision_radius(uav_radius: float, obstacle_radius: float, extra_safety_margin: float = 0.0) -> float:
    values = (uav_radius, obstacle_radius, extra_safety_margin)
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("radii and extra safety margin must be finite and non-negative")
    return float(sum(values))


def directional_variance(
    direction: ArrayLike,
    relative_covariance: ArrayLike,
    psd_tolerance: float = 1e-10,
) -> float:
    n = np.asarray(direction, dtype=float)
    if n.shape != (3,) or not np.all(np.isfinite(n)):
        raise ValueError("direction must be a finite vector of shape (3,)")
    covariance = validate_psd(relative_covariance, psd_tolerance)
    if covariance.shape != (3, 3):
        raise ValueError("relative covariance must have shape (3, 3)")
    variance = float(n @ covariance @ n)
    if variance < -psd_tolerance:
        raise ValueError("directional variance is materially negative")
    return max(variance, 0.0)


def directional_sigma(direction: ArrayLike, relative_covariance: ArrayLike, psd_tolerance: float = 1e-10) -> float:
    return float(np.sqrt(directional_variance(direction, relative_covariance, psd_tolerance)))


def safety_margin(distance: float, beta: float, sigma: float, collision_radius: float) -> float:
    if distance < 0 or sigma < 0 or collision_radius < 0:
        raise ValueError("distance, sigma, and collision radius must be non-negative")
    return float(distance - beta * sigma - collision_radius)


def margin_gradient(
    direction: ArrayLike,
    distance: float,
    relative_covariance: ArrayLike,
    beta: float,
    sigma: float | None = None,
    *,
    eps_sigma: float = 1e-10,
    psd_tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    n = np.asarray(direction, dtype=float)
    covariance = validate_psd(relative_covariance, psd_tolerance)
    if n.shape != (3,) or covariance.shape != (3, 3):
        raise ValueError("direction and covariance must have shapes (3,) and (3, 3)")
    if distance <= 0:
        raise ValueError("margin gradient is undefined for degenerate geometry")
    physical_sigma = directional_sigma(n, covariance, psd_tolerance) if sigma is None else float(sigma)
    if physical_sigma < 0:
        raise ValueError("sigma must be non-negative")
    if physical_sigma <= eps_sigma:
        return -n.copy()
    projection = (np.eye(3) - np.outer(n, n)) @ covariance @ n
    return -n + beta * projection / (distance * physical_sigma)


def evaluate_safety_sample(
    time: float,
    obstacle_position: ArrayLike,
    uav_position: ArrayLike,
    relative_covariance: ArrayLike,
    beta: float,
    collision_radius: float,
    *,
    eps_distance: float = 1e-8,
    eps_sigma: float = 1e-10,
    psd_tolerance: float = 1e-10,
) -> SafetySample:
    _, separation, direction = relative_geometry(obstacle_position, uav_position, eps_distance)
    sigma = directional_sigma(direction, relative_covariance, psd_tolerance)
    margin = safety_margin(separation, beta, sigma, collision_radius)
    gradient = margin_gradient(
        direction, separation, relative_covariance, beta, sigma,
        eps_sigma=eps_sigma, psd_tolerance=psd_tolerance,
    )
    return SafetySample(time, separation, direction, sigma, margin, gradient)
