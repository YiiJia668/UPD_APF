"""Fixed collection of obstacle truths, without estimator or planner ownership."""
from collections.abc import Sequence
import numpy as np

from .obstacle import ConstantVelocityObstacle, ObstacleTruthSnapshot


class World:
    def __init__(self, obstacles: Sequence[ConstantVelocityObstacle] = ()):
        obstacles = tuple(obstacles)
        if any(not isinstance(o, ConstantVelocityObstacle) for o in obstacles):
            raise TypeError("world requires ConstantVelocityObstacle objects")
        ids = [o.metadata.obstacle_id for o in obstacles]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate obstacle IDs")
        # Own independent truth entities; external advance() cannot alter this world.
        snapshots = (o.snapshot() for o in obstacles)
        self._obstacles = tuple(ConstantVelocityObstacle(s.metadata, s.position, s.velocity)
                                for s in snapshots)

    def snapshots(self) -> tuple[ObstacleTruthSnapshot, ...]:
        return tuple(o.snapshot() for o in self._obstacles)

    def advance(self, dt: float) -> None:
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        for obstacle in self._obstacles:
            obstacle.advance(dt)
