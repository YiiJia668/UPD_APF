"""Common prediction time-grid construction."""

import numpy as np
from numpy.typing import NDArray


def prediction_time_grid(horizon: float, prediction_dt: float, *, atol: float = 1e-12) -> NDArray[np.float64]:
    if horizon < 0 or not np.isfinite(horizon):
        raise ValueError("horizon must be finite and non-negative")
    if prediction_dt <= 0 or not np.isfinite(prediction_dt):
        raise ValueError("prediction_dt must be finite and positive")
    if horizon == 0:
        return np.array([0.0])
    count = int(np.floor(horizon / prediction_dt))
    grid = prediction_dt * np.arange(count + 1, dtype=float)
    if np.isclose(grid[-1], horizon, atol=atol, rtol=0.0):
        grid[-1] = horizon
    else:
        grid = np.append(grid, horizon)
    return grid


build_prediction_time_grid = prediction_time_grid
