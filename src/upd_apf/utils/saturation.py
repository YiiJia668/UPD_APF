"""Norm-preserving vector saturation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def saturate_vector(vector: ArrayLike, max_norm: float) -> NDArray[np.float64]:
    if max_norm < 0 or not np.isfinite(max_norm):
        raise ValueError("max_norm must be finite and non-negative")
    value = np.asarray(vector, dtype=float)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("vector must be a finite one-dimensional array")
    norm = float(np.linalg.norm(value))
    if norm == 0.0 or norm <= max_norm:
        return value.copy()
    return value * (max_norm / norm)


norm_saturate = saturate_vector
