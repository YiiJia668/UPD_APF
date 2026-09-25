"""Structural interfaces at subsystem boundaries."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .data_types import GaussianPrediction, ObstacleMeasurement, UPDAPFResult, UAVState


@runtime_checkable
class Predictor(Protocol):
    def predict_distribution(self, tau: float) -> GaussianPrediction: ...


@runtime_checkable
class ObstacleEstimator(Predictor, Protocol):
    def predict(self, dt: float) -> None: ...
    def update(self, measurement: ObstacleMeasurement | NDArray[np.float64]) -> None: ...


@runtime_checkable
class Planner(Protocol):
    def compute_control(
        self, uav_state: UAVState, goal: NDArray[np.float64], obstacles: Sequence[Predictor]
    ) -> UPDAPFResult | NDArray[np.float64]: ...
