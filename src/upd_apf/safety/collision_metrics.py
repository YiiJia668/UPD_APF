"""Closest-approach, chance-boundary TTC, and dynamic risk metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from upd_apf.data_types import CollisionRisk, GaussianPrediction, SafetySample

from .chance_constraint import evaluate_safety_sample
from .relative_geometry import relative_covariance, relative_mean


def compute_cpa(
    relative_position: ArrayLike,
    relative_velocity: ArrayLike,
    horizon: float,
    eps_velocity_squared: float = 1e-12,
) -> tuple[float, float]:
    """Return clipped CPA time and analytic mean separation.

    Inputs follow the fixed convention ``r=p_obstacle-p_uav`` and
    ``v_r=v_obstacle-v_uav``.
    """
    r = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    if r.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite([*r, *velocity])):
        raise ValueError("relative position and velocity must be finite 3-vectors")
    if horizon < 0 or not np.isfinite(horizon):
        raise ValueError("horizon must be finite and non-negative")
    speed_squared = float(velocity @ velocity)
    if speed_squared <= eps_velocity_squared:
        time = 0.0
    else:
        time = float(np.clip(-(r @ velocity) / speed_squared, 0.0, horizon))
    return time, float(np.linalg.norm(r + time * velocity))


def compute_cpa_margin(
    t_cpa: float,
    obstacle_prediction: GaussianPrediction,
    uav_prediction: GaussianPrediction,
    beta: float,
    collision_radius: float,
    **tolerances: float,
) -> float:
    """Evaluate CPA margin through the accepted Phase-3 implementation."""
    covariance = relative_covariance(
        obstacle_prediction.position_covariance,
        uav_prediction.position_covariance,
    )
    sample = evaluate_safety_sample(
        t_cpa,
        obstacle_prediction.position_mean,
        uav_prediction.position_mean,
        covariance,
        beta,
        collision_radius,
        **tolerances,
    )
    return sample.safety_margin


def compute_chance_constraint_ttc(
    samples_or_times: Sequence[SafetySample] | ArrayLike,
    margins: ArrayLike | None = None,
) -> float:
    """Find the first sampled/interpolated time at which ``c <= 0``."""
    if margins is None:
        samples = tuple(samples_or_times)  # type: ignore[arg-type]
        times = np.asarray([sample.time for sample in samples], dtype=float)
        margin_values = np.asarray([sample.safety_margin for sample in samples], dtype=float)
    else:
        times = np.asarray(samples_or_times, dtype=float)
        margin_values = np.asarray(margins, dtype=float)
    if times.ndim != 1 or margin_values.ndim != 1 or len(times) != len(margin_values) or len(times) == 0:
        raise ValueError("times and margins must be non-empty one-dimensional arrays of equal length")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(margin_values)):
        raise ValueError("times and margins must be finite")
    if np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("times must be non-negative and strictly increasing")
    if margin_values[0] <= 0:
        return 0.0
    crossing_indices = np.flatnonzero(margin_values[1:] <= 0)
    if not len(crossing_indices):
        return float("inf")
    left = int(crossing_indices[0])
    right = left + 1
    alpha = margin_values[left] / (margin_values[left] - margin_values[right])
    return float(times[left] + alpha * (times[right] - times[left]))


def compute_cpa_risk(cpa_margin: float, warning_margin: float, sigmoid_scale: float) -> float:
    if sigmoid_scale <= 0:
        raise ValueError("CPA sigmoid scale must be positive")
    return float(expit((warning_margin - cpa_margin) / sigmoid_scale))


def compute_ttc_risk(probabilistic_ttc: float, time_scale: float) -> float:
    if time_scale <= 0:
        raise ValueError("TTC time scale must be positive")
    if probabilistic_ttc < 0 or np.isnan(probabilistic_ttc):
        raise ValueError("TTC must be non-negative or infinity")
    return 0.0 if np.isinf(probabilistic_ttc) else float(np.exp(-probabilistic_ttc / time_scale))


def combine_risk(cpa_risk: float, ttc_risk: float, cpa_weight: float, ttc_weight: float) -> float:
    if min(cpa_risk, ttc_risk, cpa_weight, ttc_weight) < 0:
        raise ValueError("risks and weights must be non-negative")
    if not np.isclose(cpa_weight + ttc_weight, 1.0, atol=1e-12, rtol=0):
        raise ValueError("risk weights must sum to one")
    return float(cpa_weight * cpa_risk + ttc_weight * ttc_risk)


def build_collision_risk(
    t_cpa: float,
    d_cpa: float,
    cpa_margin: float,
    probabilistic_ttc: float,
    *,
    warning_margin: float,
    sigmoid_scale: float,
    time_scale: float,
    cpa_weight: float,
    ttc_weight: float,
) -> CollisionRisk:
    cpa_risk = compute_cpa_risk(cpa_margin, warning_margin, sigmoid_scale)
    ttc_risk = compute_ttc_risk(probabilistic_ttc, time_scale)
    return CollisionRisk(
        t_cpa, d_cpa, cpa_margin, probabilistic_ttc, cpa_risk, ttc_risk,
        combine_risk(cpa_risk, ttc_risk, cpa_weight, ttc_weight),
    )


chance_constraint_ttc = compute_chance_constraint_ttc
