"""Single-obstacle predictive repulsion consuming computed safety/risk values.

Risk is an index in [0, 1], not a collision probability. Hold risk (and hence
eta), times, and covariance fixed throughout each control cycle's local field
evaluation. No risk or gain gradient is computed. Negative margins mean the
sufficient chance constraint is violated, not a proven probability violation.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from upd_apf.config import RepulsiveConfig
from upd_apf.utils.saturation import saturate_vector

from ._validation import scalar


def temporal_weights(times: ArrayLike, future_decay: float) -> NDArray[np.float64]:
    """Normalize exp(-future_decay*times) for nonempty times (M,) in seconds.

    Retains every sample, including tau=0. Use the common prediction_time_grid
    upstream. Subtracting the earliest time prevents all-weight underflow even
    for grids starting after zero; the normalized mathematical weights agree.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size == 0
            or not np.all(np.isfinite(times)) or np.any(times < 0)):
        raise ValueError("times must be a nonempty finite non-negative (M,) array")
    decay = scalar(future_decay, "future_decay")
    # Overflow of this non-negative product means a negligible future weight.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-decay * (times - np.min(times)))
    return weights / np.sum(weights)


def risk_modulated_gain(risk: float, eta_0: float, risk_gain: float) -> float:
    """Return frozen eta_0*(1+risk_gain*risk); risk is a scalar index in [0, 1]."""
    risk = scalar(risk, "risk")
    if risk > 1:
        raise ValueError("risk must be in [0, 1]")
    base = scalar(eta_0, "eta_0")
    modulation = scalar(risk_gain, "risk_gain")
    result = base * (1.0 + modulation * risk)
    if not np.isfinite(result):
        raise ValueError("risk-modulated gain exceeds floating-point range")
    return float(result)


def _response(
    margins: ArrayLike, times: ArrayLike, risk: float, config: RepulsiveConfig,
) -> tuple[NDArray[np.float64], float, float]:
    weights = temporal_weights(times, config.future_decay)
    margins = np.asarray(margins, dtype=float)
    if margins.shape != weights.shape or not np.all(np.isfinite(margins)):
        raise ValueError("margins must be finite with the same (M,) shape as times")
    length = scalar(config.length_scale, "length_scale", positive=True)
    eta = risk_modulated_gain(risk, config.eta_0, config.risk_gain)
    scalar(config.max_obstacle_force, "max_obstacle_force")
    bounds = np.asarray(
        [config.exponent_clip_min, config.exponent_clip_max], dtype=float,
    )
    if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] > bounds[1]
            or not bounds[0] <= 0 <= bounds[1]
            or bounds[1] > np.log(np.finfo(float).max)):
        raise ValueError(
            "exponent bounds must be finite, ordered, exp-safe, and contain zero"
        )
    # Finite margins / tiny positive length may overflow before clipping.
    with np.errstate(over="ignore", under="ignore"):
        response = weights * np.exp(np.clip(-margins / length, *bounds))
    return response, eta, length


def potential(
    margins: ArrayLike,
    times: ArrayLike,
    risk: float,
    config: RepulsiveConfig = RepulsiveConfig(),
) -> float:
    """Return eta*sum(omega*exp(-c/ell)) for margins/times of shape (M,).

    Margins and ell are in metres. Exponents use the configured numerical
    clipping bounds; force saturation never changes this scalar potential.
    """
    response, eta, _ = _response(margins, times, risk, config)
    with np.errstate(over="ignore"):
        result = float(eta * np.sum(response))
    if not np.isfinite(result):
        raise ValueError("repulsive potential exceeds floating-point range")
    return result


def obstacle_force(
    margins: ArrayLike,
    margin_gradients: ArrayLike,
    times: ArrayLike,
    risk: float,
    config: RepulsiveConfig = RepulsiveConfig(),
    *,
    saturate: bool = True,
) -> NDArray[np.float64]:
    """Return one obstacle's force (3,); set saturate=False for the raw force.

    Inputs c/times have shape (M,); UAV-position margin gradients have (M, 3).
    F_raw = +(eta/ell)*sum(omega*exp(-c/ell)*grad(c)): the leading sign is
    POSITIVE because grad(c)=-n for isotropic covariance, pointing away.
    With frozen eta/weights this is -grad(U) only where clipping is inactive.
    Active norm saturation also breaks the original potential-gradient identity.
    """
    response, eta, length = _response(margins, times, risk, config)
    gradients = np.asarray(margin_gradients, dtype=float)
    if (gradients.shape != (response.size, 3)
            or not np.all(np.isfinite(gradients))):
        raise ValueError("margin_gradients must be finite with shape (M, 3)")
    if eta == 0:
        return np.zeros(3)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = (eta / length) * np.sum(response[:, None] * gradients, axis=0)
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw repulsive force exceeds floating-point range")
    if saturate:
        # Preserve the shared saturation API while avoiding overflow in its
        # squared norm for unusually large, but representable, raw forces.
        scale = float(np.max(np.abs(raw)))
        if scale > np.sqrt(np.finfo(float).max / 3):
            scaled = raw / scale
            scaled_limit = config.max_obstacle_force / scale
            if np.linalg.norm(scaled) <= scaled_limit:
                return raw.copy()
            return scale * saturate_vector(scaled, scaled_limit)
        return saturate_vector(raw, config.max_obstacle_force)
    return raw
