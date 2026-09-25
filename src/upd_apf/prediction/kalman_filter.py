"""Six-state constant-velocity Kalman filter."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from upd_apf.data_types import GaussianPrediction, ObstacleMeasurement
from upd_apf.utils.covariance import symmetrize, validate_psd

from .process_noise import cv_transition, white_acceleration_process_noise


class ConstantVelocityKalmanFilter:
    """Position-measurement CV filter with pure future-distribution queries."""

    def __init__(
        self,
        state: ArrayLike,
        covariance: ArrayLike,
        acceleration_noise_spectral_density: float | ArrayLike,
        measurement_covariance: ArrayLike,
        *,
        time: float = 0.0,
    ) -> None:
        self._state = np.array(state, dtype=float, copy=True)
        if self._state.shape != (6,) or not np.all(np.isfinite(self._state)):
            raise ValueError("state must be a finite vector of shape (6,)")
        covariance_array = np.asarray(covariance, dtype=float)
        if covariance_array.shape != (6, 6):
            raise ValueError("covariance must have shape (6, 6)")
        self._covariance = validate_psd(covariance_array)
        measurement_array = np.asarray(measurement_covariance, dtype=float)
        if measurement_array.shape != (3, 3):
            raise ValueError("measurement covariance must have shape (3, 3)")
        self.measurement_covariance = validate_psd(measurement_array)
        # Validate once; process-noise construction remains the source of truth.
        white_acceleration_process_noise(0.0, acceleration_noise_spectral_density)
        density = np.asarray(acceleration_noise_spectral_density, dtype=float)
        self.acceleration_noise_spectral_density = float(density) if density.ndim == 0 else density.copy()
        self.time = float(time)
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")

    @property
    def state(self) -> NDArray[np.float64]:
        return self._state.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        return self._covariance.copy()

    def predict(self, dt: float) -> None:
        transition = cv_transition(dt)
        noise = white_acceleration_process_noise(dt, self.acceleration_noise_spectral_density)
        self._state = transition @ self._state
        self._covariance = symmetrize(transition @ self._covariance @ transition.T + noise)
        self.time += float(dt)

    def update(self, measurement: ObstacleMeasurement | ArrayLike) -> None:
        measured = measurement.position if isinstance(measurement, ObstacleMeasurement) else measurement
        z = np.asarray(measured, dtype=float)
        if z.shape != (3,) or not np.all(np.isfinite(z)):
            raise ValueError("position measurement must be a finite vector of shape (3,)")
        h = np.zeros((3, 6))
        h[:, :3] = np.eye(3)
        innovation_covariance = h @ self._covariance @ h.T + self.measurement_covariance
        # Solve S K^T = (P H^T)^T instead of explicitly forming S^-1.
        kalman_gain = np.linalg.solve(innovation_covariance, (self._covariance @ h.T).T).T
        self._state = self._state + kalman_gain @ (z - h @ self._state)
        identity = np.eye(6)
        residual_map = identity - kalman_gain @ h
        self._covariance = symmetrize(
            residual_map @ self._covariance @ residual_map.T
            + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        )

    def predict_distribution(self, tau: float) -> GaussianPrediction:
        transition = cv_transition(tau)
        noise = white_acceleration_process_noise(tau, self.acceleration_noise_spectral_density)
        state = transition @ self._state
        covariance = symmetrize(transition @ self._covariance @ transition.T + noise)
        return GaussianPrediction(
            time=self.time + float(tau),
            position_mean=state[:3],
            velocity_mean=state[3:],
            position_covariance=covariance[:3, :3],
        )


KalmanFilter = ConstantVelocityKalmanFilter
