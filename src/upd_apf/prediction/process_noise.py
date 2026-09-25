"""Constant-velocity dynamics with continuous white-acceleration noise."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from upd_apf.utils.covariance import validate_psd


def _nonnegative_time(tau: float) -> float:
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("prediction time must be finite and non-negative")
    return tau


def cv_transition(tau: float) -> NDArray[np.float64]:
    tau = _nonnegative_time(tau)
    result = np.eye(6)
    result[:3, 3:] = tau * np.eye(3)
    return result


def white_acceleration_process_noise(
    tau: float,
    acceleration_noise_spectral_density: float | ArrayLike,
) -> NDArray[np.float64]:
    """Return exact CV discretization for acceleration spectral density ``Sa``."""
    tau = _nonnegative_time(tau)
    density = np.asarray(acceleration_noise_spectral_density, dtype=float)
    if density.ndim == 0:
        if not np.isfinite(density) or density < 0:
            raise ValueError("acceleration-noise spectral density must be non-negative")
        sa = float(density) * np.eye(3)
    else:
        if density.shape != (3, 3):
            raise ValueError("acceleration-noise spectral density matrix must be 3x3")
        sa = validate_psd(density)
    return np.block([
        [tau**3 / 3.0 * sa, tau**2 / 2.0 * sa],
        [tau**2 / 2.0 * sa, tau * sa],
    ])


state_transition = cv_transition
process_noise = white_acceleration_process_noise
