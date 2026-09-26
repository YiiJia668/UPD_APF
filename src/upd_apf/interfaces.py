"""Structural interfaces at subsystem boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .data_types import GaussianPrediction, ObstacleMetadata, ObstacleMeasurement, UPDAPFResult, UAVState


@runtime_checkable
class Predictor(Protocol):
    def predict_distribution(self, tau: float) -> GaussianPrediction: ...


@runtime_checkable
class ObstacleEstimator(Predictor, Protocol):
    def predict(self, dt: float) -> None: ...
    def update(self, measurement: ObstacleMeasurement | NDArray[np.float64]) -> None: ...


@dataclass(frozen=True)
class PlannerObstacle:
    """Explicit immutable association of public metadata and a pure predictor."""

    metadata: ObstacleMetadata
    predictor: Predictor

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ObstacleMetadata):
            raise TypeError("metadata must be ObstacleMetadata")
        if not isinstance(self.metadata.obstacle_id, str) or not self.metadata.obstacle_id:
            raise ValueError("obstacle_id must be a nonempty string")
        if not np.isscalar(self.metadata.radius) or not np.isfinite(self.metadata.radius) or self.metadata.radius < 0:
            raise ValueError("obstacle radius must be finite and non-negative")
        if not callable(getattr(self.predictor, "predict_distribution", None)):
            raise TypeError("predictor must provide predict_distribution(tau)")


@runtime_checkable
class Planner(Protocol):
    def compute_control(
        self, uav_state: UAVState, goal: NDArray[np.float64], obstacles: Sequence[PlannerObstacle]
    ) -> UPDAPFResult | NDArray[np.float64]: ...
