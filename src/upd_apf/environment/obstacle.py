"""Constant-velocity ground truth; never a planner predictor."""
from dataclasses import dataclass

import numpy as np

from upd_apf.data_types import FloatArray, ObstacleMetadata, _array


@dataclass(frozen=True)
class ObstacleTruthSnapshot:
    metadata: ObstacleMetadata
    position: FloatArray
    velocity: FloatArray

    def __post_init__(self):
        if not isinstance(self.metadata, ObstacleMetadata):
            raise TypeError("metadata must be ObstacleMetadata")
        if not isinstance(self.metadata.obstacle_id, str) or not self.metadata.obstacle_id:
            raise ValueError("obstacle_id must be a nonempty string")
        radius = self.metadata.radius
        if not np.isscalar(radius) or not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and non-negative")
        object.__setattr__(self, "position", _array(self.position, (3,), "position"))
        object.__setattr__(self, "velocity", _array(self.velocity, (3,), "velocity"))


class ConstantVelocityObstacle:
    def __init__(self, metadata: ObstacleMetadata, position, velocity):
        self._truth = ObstacleTruthSnapshot(metadata, position, velocity)

    @property
    def metadata(self):
        return self._truth.metadata

    @property
    def position(self):
        return self.snapshot().position

    @property
    def velocity(self):
        return self.snapshot().velocity

    def snapshot(self) -> ObstacleTruthSnapshot:
        return ObstacleTruthSnapshot(self.metadata, self._truth.position, self._truth.velocity)

    def advance(self, dt: float) -> None:
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        with np.errstate(over="ignore", invalid="ignore"):
            position = self._truth.position + dt * self._truth.velocity
        self._truth = ObstacleTruthSnapshot(self.metadata, position, self._truth.velocity)
