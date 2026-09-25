"""Covariance matrix validation helpers."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def symmetrize(matrix: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError("covariance must be finite")
    return 0.5 * (value + value.T)


def validate_psd(matrix: ArrayLike, tolerance: float = 1e-10, *, psd_tolerance: float | None = None) -> NDArray[np.float64]:
    """Return a symmetric covariance, rejecting material negative eigenvalues."""
    if psd_tolerance is not None:
        tolerance = psd_tolerance
    if tolerance < 0:
        raise ValueError("PSD tolerance must be non-negative")
    value = symmetrize(matrix)
    minimum = float(np.linalg.eigvalsh(value)[0])
    if minimum < -tolerance:
        raise ValueError(f"covariance is not PSD (minimum eigenvalue {minimum})")
    return value


def is_psd(matrix: ArrayLike, tolerance: float = 1e-10) -> bool:
    try:
        validate_psd(matrix, tolerance)
    except ValueError:
        return False
    return True
