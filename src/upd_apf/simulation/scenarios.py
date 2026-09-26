"""Two minimal setups, not a benchmark or a trajectory generator."""
from dataclasses import dataclass, replace
import numpy as np

from upd_apf.config import UPDAPFConfig
from upd_apf.data_types import UAVState, ObstacleMetadata, FloatArray, _array
from upd_apf.environment.obstacle import ConstantVelocityObstacle
from upd_apf.environment.world import World
from upd_apf.environment.sensor import PositionSensor
from upd_apf.planner import UPDAPFPlanner
from upd_apf.prediction.kalman_filter import ConstantVelocityKalmanFilter
from .simulator import Simulator


@dataclass(frozen=True)
class ScenarioSetup:
    simulator: Simulator
    initial_uav_state: UAVState
    goal: FloatArray

    def __post_init__(self):
        object.__setattr__(self, "goal", _array(self.goal, (3,), "goal"))


def _minimal_config():
    # Explicit scenario engineering values; scientific/global defaults unchanged.
    cfg = UPDAPFConfig()
    return replace(cfg, uav=replace(cfg.uav, desired_speed=0.0),
                   damping=replace(cfg.damping, gain=2.0))


def _initial_state(config):
    return UAVState(np.zeros(3), np.zeros(3), np.eye(3) * config.uav.initial_position_covariance)


def build_no_obstacle_scenario(config: UPDAPFConfig | None = None) -> ScenarioSetup:
    config = _minimal_config() if config is None else config
    simulator = Simulator(config, UPDAPFPlanner(config), World(), PositionSensor(np.zeros((3, 3))), {})
    return ScenarioSetup(simulator, _initial_state(config), np.array([10., 0., 0.]))


def build_single_crossing_scenario(config: UPDAPFConfig | None = None, *, seed: int | None = None,
                                   noisy: bool = False) -> ScenarioSetup:
    config = _minimal_config() if config is None else config
    obstacle = ConstantVelocityObstacle(ObstacleMetadata("crossing", 0.4), [5., -3., 0.], [0., 1., 0.])
    kf = config.kalman
    covariance = np.diag([kf.initial_position_std**2] * 3 + [kf.initial_velocity_std**2] * 3)
    measurement_covariance = np.eye(3) * kf.measurement_position_std**2
    estimator = ConstantVelocityKalmanFilter(
        np.concatenate([obstacle.position, obstacle.velocity]), covariance,
        kf.acceleration_noise_spectral_density, measurement_covariance)
    sensor = PositionSensor(measurement_covariance if noisy else np.zeros((3, 3)),
                            seed=config.simulation.seed if seed is None else seed)
    simulator = Simulator(config, UPDAPFPlanner(config), World([obstacle]), sensor, {"crossing": estimator})
    return ScenarioSetup(simulator, _initial_state(config), np.array([12., 0., 0.]))
