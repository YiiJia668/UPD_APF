"""Small input validators shared by the potential-field functions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def scalar(value: float, name: str, *, positive: bool = False) -> float:
    result = np.asarray(value, dtype=float)
    if result.ndim != 0 or not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    number = float(result)
    if number < 0 or (positive and number == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of shape (3,)")
    return result
