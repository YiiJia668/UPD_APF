"""Minimal sampled-state closed loop. Truth is never passed to the planner."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from upd_apf.config import UPDAPFConfig, validate_config
from upd_apf.data_types import UAVState, UPDAPFResult, ObstacleMeasurement, _array
from upd_apf.environment.obstacle import ObstacleTruthSnapshot
from upd_apf.environment.world import World
from upd_apf.environment.sensor import PositionSensor
from upd_apf.interfaces import ObstacleEstimator, Planner, PlannerObstacle
from .uav_model import PointMassUAVModel


@dataclass(frozen=True)
class SimulationStep:
    time: float
    dt: float
    uav_state_before: UAVState
    uav_state_after: UAVState
    obstacle_truths: tuple[ObstacleTruthSnapshot, ...]
    measurements: tuple[ObstacleMeasurement, ...]
    planner_result: UPDAPFResult


@dataclass(frozen=True)
class SimulationResult:
    termination_reason: Literal["collision", "goal_reached", "max_time"]
    final_time: float
    final_uav_state: UAVState
    steps: tuple[SimulationStep, ...]
    final_obstacle_truths: tuple[ObstacleTruthSnapshot, ...]


def _copy_state(state):
    return UAVState(state.position, state.velocity, state.position_covariance)


class Simulator:
    """One run consumes world, estimator and sensor state; rebuild for replay.

    Initial estimator belief/prior is defined at t=0 before the first t=0
    measurement update. At k=0: measure, update, plan, with no time prediction.
    At k>0: predict(previous actual dt), measure, update, plan. UAV self-state
    is directly available. No final estimator update is needed after termination.
    """
    def __init__(self, config: UPDAPFConfig, planner: Planner, world: World,
                 sensor: PositionSensor, estimators: Mapping[str, ObstacleEstimator]):
        validate_config(config)
        self.config = config
        self.planner = planner
        self.world = world
        self.sensor = sensor
        self.estimators = dict(estimators)
        ids = {s.metadata.obstacle_id for s in world.snapshots()}
        if ids != set(self.estimators):
            raise ValueError("world and estimator IDs must correspond exactly")
        for estimator in self.estimators.values():
            if not isinstance(estimator, ObstacleEstimator):
                raise TypeError("estimators must implement predict, update and predict_distribution")
        if len({id(e) for e in self.estimators.values()}) != len(self.estimators):
            raise ValueError("each obstacle requires its own estimator")
        self.uav_model = PointMassUAVModel(config.uav.force_to_acceleration_gain)
        self._started = False

    def _termination(self, state, goal, truths, time):
        if any(np.linalg.norm(state.position - o.position) <=
               self.config.uav.radius + o.metadata.radius for o in truths):
            return "collision"
        cfg = self.config.simulation
        if (np.linalg.norm(state.position - goal) <= cfg.goal_tolerance and
                np.linalg.norm(state.velocity) <= cfg.goal_speed_tolerance):
            return "goal_reached"
        if time >= cfg.max_time:
            return "max_time"
        return None

    def run(self, initial_uav_state: UAVState, goal: ArrayLike) -> SimulationResult:
        if self._started:
            raise RuntimeError("Simulator is single-use; rebuild components for replay")
        state = _copy_state(initial_uav_state)
        goal = _array(goal, (3,), "goal")
        self._started = True
        time = 0.0
        previous_dt = None
        steps = []
        cfg = self.config.simulation
        while True:
            truths = self.world.snapshots()
            reason = self._termination(state, goal, truths, time)
            if reason is not None:
                return SimulationResult(reason, time, _copy_state(state), tuple(steps), truths)
            if previous_dt is not None:
                for estimator in self.estimators.values():
                    estimator.predict(previous_dt)
            measurements = tuple(self.sensor.measure(truths, timestamp=time))
            ids = [m.obstacle_id for m in measurements]
            if len(ids) != len(set(ids)) or set(ids) != set(self.estimators):
                raise ValueError("measurement IDs must match each estimator exactly once")
            if any(not np.isfinite(m.timestamp) or m.timestamp != time for m in measurements):
                raise ValueError("measurement timestamps must equal current simulation time")
            # Snapshot before passing measurements to arbitrary estimator implementations.
            saved_measurements = tuple(ObstacleMeasurement(m.obstacle_id, m.timestamp, m.position)
                                       for m in measurements)
            for measurement in measurements:
                self.estimators[measurement.obstacle_id].update(measurement)
            bindings = tuple(PlannerObstacle(o.metadata, self.estimators[o.metadata.obstacle_id])
                             for o in truths)
            result = self.planner.compute_control(state, goal, bindings)
            if not isinstance(result, UPDAPFResult):
                raise TypeError("Phase 7 planner must return UPDAPFResult")
            _array(result.total_force, (3,), "total_force")
            # Indexed boundaries avoid accumulated drift and spurious tiny final steps.
            next_time = min(cfg.max_time, (len(steps) + 1) * cfg.dt)
            dt = next_time - time
            next_state = self.uav_model.step(state, result.total_force, dt)
            steps.append(SimulationStep(time, dt, _copy_state(state), _copy_state(next_state),
                                        truths, saved_measurements, deepcopy(result)))
            self.world.advance(dt)
            time = next_time
            previous_dt = dt
            state = next_state
