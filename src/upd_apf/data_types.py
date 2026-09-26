"""Shared data models used across prediction, safety, and planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class UAVState:
    position: FloatArray
    velocity: FloatArray
    position_covariance: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _array(self.position, (3,), "position"))
        object.__setattr__(self, "velocity", _array(self.velocity, (3,), "velocity"))
        object.__setattr__(self, "position_covariance", _array(self.position_covariance, (3, 3), "position_covariance"))


@dataclass(frozen=True)
class ObstacleMeasurement:
    obstacle_id: str
    timestamp: float
    position: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _array(self.position, (3,), "position"))


@dataclass(frozen=True)
class ObstacleMetadata:
    obstacle_id: str
    radius: float


@dataclass(frozen=True)
class GaussianPrediction:
    time: float
    position_mean: FloatArray
    velocity_mean: FloatArray
    position_covariance: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_mean", _array(self.position_mean, (3,), "position_mean"))
        object.__setattr__(self, "velocity_mean", _array(self.velocity_mean, (3,), "velocity_mean"))
        object.__setattr__(self, "position_covariance", _array(self.position_covariance, (3, 3), "position_covariance"))


@dataclass(frozen=True)
class SafetySample:
    time: float
    distance: float
    direction: FloatArray
    directional_sigma: float
    safety_margin: float
    margin_gradient: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _array(self.direction, (3,), "direction"))
        object.__setattr__(self, "margin_gradient", _array(self.margin_gradient, (3,), "margin_gradient"))


@dataclass(frozen=True)
class CollisionRisk:
    t_cpa: float
    d_cpa: float
    cpa_margin: float
    probabilistic_ttc: float
    cpa_risk: float
    ttc_risk: float
    total_risk: float


@dataclass(frozen=True)
class ObstacleRiskResult:
    obstacle_id: str
    safety_samples: Sequence[SafetySample]
    collision_risk: CollisionRisk
    repulsive_force: FloatArray
    repulsive_force_raw: FloatArray | None = None
    repulsive_potential: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "safety_samples", tuple(self.safety_samples))
        object.__setattr__(self, "repulsive_force", _array(self.repulsive_force, (3,), "repulsive_force"))
        if self.repulsive_force_raw is not None:
            object.__setattr__(self, "repulsive_force_raw", _array(self.repulsive_force_raw, (3,), "repulsive_force_raw"))
        if self.repulsive_potential is not None:
            potential = np.asarray(self.repulsive_potential, dtype=float)
            if potential.ndim != 0 or not np.isfinite(potential) or potential < 0:
                raise ValueError("repulsive_potential must be a finite non-negative scalar")
            object.__setattr__(self, "repulsive_potential", float(potential))


@dataclass(frozen=True)
class UPDAPFResult:
    attractive_force: FloatArray
    repulsive_force: FloatArray
    damping_force: FloatArray
    total_force: FloatArray
    global_risk: float
    mean_risk: float
    min_safety_margin: float
    min_ttc: float
    per_obstacle_results: Sequence[ObstacleRiskResult]
    total_force_raw: FloatArray | None = None
    reference_velocity: FloatArray | None = None

    def __post_init__(self) -> None:
        for name in ("attractive_force", "repulsive_force", "damping_force", "total_force"):
            object.__setattr__(self, name, _array(getattr(self, name), (3,), name))
        object.__setattr__(self, "per_obstacle_results", tuple(self.per_obstacle_results))

        for name in ("total_force_raw", "reference_velocity"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _array(getattr(self, name), (3,), name))
