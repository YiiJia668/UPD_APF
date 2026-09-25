"""Typed configuration loading and validation for UPD-APF."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.05
    max_time: float = 60.0
    goal_tolerance: float = 0.5
    goal_speed_tolerance: float = 0.5
    seed: int = 42


@dataclass(frozen=True)
class PredictionConfig:
    horizon: float = 3.0
    dt: float = 0.1


@dataclass(frozen=True)
class UAVConfig:
    radius: float = 0.3
    max_speed: float = 5.0
    max_acceleration: float = 4.0
    desired_speed: float = 2.0
    force_to_acceleration_gain: float = 1.0
    use_uncertainty: bool = False
    initial_position_covariance: float = 0.0


@dataclass(frozen=True)
class KalmanConfig:
    acceleration_noise_spectral_density: float = 0.25
    measurement_position_std: float = 0.1
    initial_position_std: float = 0.2
    initial_velocity_std: float = 0.5


@dataclass(frozen=True)
class ChanceConstraintConfig:
    collision_probability: float = 0.01
    extra_safety_margin: float = 0.5


@dataclass(frozen=True)
class AttractiveConfig:
    gain: float = 1.0
    switch_distance: float = 5.0


@dataclass(frozen=True)
class RepulsiveConfig:
    eta_0: float = 2.0
    length_scale: float = 1.5
    risk_gain: float = 2.0
    future_decay: float = 0.7
    max_obstacle_force: float = 15.0
    max_total_repulsive_force: float = 20.0
    exponent_clip_min: float = -50.0
    exponent_clip_max: float = 50.0


@dataclass(frozen=True)
class RiskConfig:
    cpa_weight: float = 0.5
    ttc_weight: float = 0.5
    cpa_warning_margin: float = 1.5
    cpa_sigmoid_scale: float = 0.5
    ttc_time_scale: float = 2.0


@dataclass(frozen=True)
class DampingConfig:
    gain: float = 0.4


@dataclass(frozen=True)
class ControlConfig:
    max_total_force: float = 30.0


@dataclass(frozen=True)
class NumericalConfig:
    eps_distance: float = 1e-8
    eps_velocity_squared: float = 1e-12
    eps_sigma: float = 1e-10
    psd_tolerance: float = 1e-10


@dataclass(frozen=True)
class TraditionalAPFConfig:
    repulsive_gain: float = 2.0
    influence_clearance: float = 4.0
    min_clearance: float = 1e-4


@dataclass(frozen=True)
class AblationConfig:
    use_uncertainty: bool = True
    use_cpa_risk: bool = True
    use_ttc_risk: bool = True
    use_predictive_horizon: bool = True
    use_damping: bool = True
    use_uav_uncertainty: bool = False


@dataclass(frozen=True)
class UPDAPFConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    chance_constraint: ChanceConstraintConfig = field(default_factory=ChanceConstraintConfig)
    attractive: AttractiveConfig = field(default_factory=AttractiveConfig)
    repulsive: RepulsiveConfig = field(default_factory=RepulsiveConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    damping: DampingConfig = field(default_factory=DampingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    traditional_apf: TraditionalAPFConfig = field(default_factory=TraditionalAPFConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def __post_init__(self) -> None:
        validate_config(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Config = UPDAPFConfig

_GROUPS = {f.name: f.type for f in __import__("dataclasses").fields(UPDAPFConfig)}
# ``from __future__ import annotations`` leaves field types as strings; the concrete
# mapping keeps construction explicit and rejects misspelled groups and keys.
_GROUP_CLASSES = {
    "simulation": SimulationConfig, "prediction": PredictionConfig, "uav": UAVConfig,
    "kalman": KalmanConfig, "chance_constraint": ChanceConstraintConfig,
    "attractive": AttractiveConfig, "repulsive": RepulsiveConfig, "risk": RiskConfig,
    "damping": DampingConfig, "control": ControlConfig, "numerical": NumericalConfig,
    "traditional_apf": TraditionalAPFConfig, "ablation": AblationConfig,
}


def config_from_mapping(values: Mapping[str, Any]) -> UPDAPFConfig:
    unknown = set(values) - set(_GROUP_CLASSES)
    if unknown:
        raise ValueError(f"unknown configuration group(s): {sorted(unknown)}")
    groups: dict[str, Any] = {}
    for name, cls in _GROUP_CLASSES.items():
        raw = values.get(name, {})
        if not isinstance(raw, Mapping):
            raise TypeError(f"configuration group {name!r} must be a mapping")
        try:
            groups[name] = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid {name} configuration: {exc}") from exc
    return UPDAPFConfig(**groups)


def load_config(path: str | Path) -> UPDAPFConfig:
    """Load a complete YAML configuration from *path*."""
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, Mapping):
        raise TypeError("configuration root must be a mapping")
    return config_from_mapping(values)


def validate_config(config: UPDAPFConfig) -> None:
    positive = {
        "simulation.dt": config.simulation.dt,
        "simulation.max_time": config.simulation.max_time,
        "prediction.dt": config.prediction.dt,
        "uav.max_speed": config.uav.max_speed,
        "uav.max_acceleration": config.uav.max_acceleration,
        "repulsive.length_scale": config.repulsive.length_scale,
        "risk.cpa_sigmoid_scale": config.risk.cpa_sigmoid_scale,
        "risk.ttc_time_scale": config.risk.ttc_time_scale,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.prediction.horizon < 0:
        raise ValueError("prediction.horizon must be non-negative")
    epsilon = config.chance_constraint.collision_probability
    if not 0 < epsilon < 0.5:
        raise ValueError("chance_constraint.collision_probability must be in (0, 0.5)")
    weights = (config.risk.cpa_weight, config.risk.ttc_weight)
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("risk weights must be non-negative and sum to one")
    if config.kalman.acceleration_noise_spectral_density < 0:
        raise ValueError("acceleration-noise spectral density must be non-negative")
    tolerances = vars(config.numerical).values()
    if any(value < 0 for value in tolerances):
        raise ValueError("numerical tolerances must be non-negative")
    if config.repulsive.exponent_clip_min > config.repulsive.exponent_clip_max:
        raise ValueError("repulsive exponent clipping bounds are reversed")
