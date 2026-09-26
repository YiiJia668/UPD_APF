"""Pure control-cycle orchestration of the Phase 2--5 public mathematics.

Risk is an index, not a probability. Probabilistic TTC denotes the first
chance-boundary violation time. No state propagation or escape heuristic lives
here, and the saturated command is not a global scalar-potential gradient.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from upd_apf.config import UPDAPFConfig, validate_config
from upd_apf.data_types import GaussianPrediction, ObstacleRiskResult, UAVState, UPDAPFResult
from upd_apf.interfaces import PlannerObstacle
from upd_apf.potential import attractive, damping, repulsive
from upd_apf.prediction.uav_predictor import predict_uav_distribution
from upd_apf.safety.chance_constraint import confidence_beta, collision_radius, evaluate_safety_sample
from upd_apf.safety.collision_metrics import (
    compute_cpa, compute_cpa_margin, compute_chance_constraint_ttc, build_collision_risk,
)
from upd_apf.safety.relative_geometry import relative_mean, relative_covariance
from upd_apf.safety.risk_field import aggregate_risk
from upd_apf.utils.covariance import validate_psd
from upd_apf.utils.saturation import saturate_vector
from upd_apf.utils.time_grid import prediction_time_grid


class UPDAPFPlanner:
    """Consume metadata/predictor bindings and return an independent force result.

    UAV positional uncertainty follows config.uav.use_uncertainty. Ablation
    switches are reserved for later benchmark work. There is deliberately no
    intermediate total-repulsion saturation: sum individually saturated forces.
    """

    def __init__(self, config: UPDAPFConfig):
        if not isinstance(config, UPDAPFConfig):
            raise TypeError("config must be UPDAPFConfig")
        validate_config(config)
        self.config = config

    def _prediction(self, obstacle: PlannerObstacle, tau: float) -> GaussianPrediction:
        prediction = obstacle.predictor.predict_distribution(float(tau))
        if not isinstance(prediction, GaussianPrediction):
            raise TypeError("predictor must return GaussianPrediction")
        if not np.isscalar(prediction.time) or not np.isfinite(prediction.time):
            raise ValueError("prediction time must be finite")
        # Revalidate and copy public arrays, including deliberately malformed
        # instances; do not repair or mutate the predictor's returned object.
        prediction = GaussianPrediction(
            prediction.time, prediction.position_mean, prediction.velocity_mean,
            prediction.position_covariance,
        )
        validate_psd(prediction.position_covariance, self.config.numerical.psd_tolerance)
        return prediction

    def compute_control(
        self,
        uav_state: UAVState,
        goal: ArrayLike,
        obstacles: Sequence[PlannerObstacle],
    ) -> UPDAPFResult:
        """Query one cycle without modifying UAV state or obstacle estimators.

        Each obstacle binds ObstacleMetadata(id, radius) to a Predictor. Output
        sample times are relative offsets even if predictor timestamps are
        absolute. Degenerate geometry exceptions propagate from Phase 3.
        """
        if not isinstance(uav_state, UAVState):
            raise TypeError("uav_state must be UAVState")
        state = UAVState(uav_state.position, uav_state.velocity, uav_state.position_covariance)
        config = self.config
        validate_config(config)
        validate_psd(state.position_covariance, config.numerical.psd_tolerance)
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal must be a finite vector of shape (3,)")
        obstacles = tuple(obstacles)
        for obstacle in obstacles:
            if not isinstance(obstacle, PlannerObstacle):
                raise TypeError("obstacles must contain PlannerObstacle bindings")

        times = prediction_time_grid(config.prediction.horizon, config.prediction.dt)
        beta = confidence_beta(config.chance_constraint.collision_probability)
        use_uncertainty = config.uav.use_uncertainty
        uav_predictions = [predict_uav_distribution(state, t, use_uncertainty=use_uncertainty) for t in times]
        attraction = attractive.force(state.position, goal, config.attractive.gain, config.attractive.switch_distance)
        reference = damping.reference_velocity(
            state.position, goal, config.uav.desired_speed,
            eps_distance=config.numerical.eps_distance,
        )
        damp = damping.force(state.velocity, reference, config.damping.gain)
        tolerances = dict(
            eps_distance=config.numerical.eps_distance,
            eps_sigma=config.numerical.eps_sigma,
            psd_tolerance=config.numerical.psd_tolerance,
        )
        results = []
        for obstacle in obstacles:
            radius = collision_radius(config.uav.radius, obstacle.metadata.radius, config.chance_constraint.extra_safety_margin)
            predictions = [self._prediction(obstacle, t) for t in times]
            samples = tuple(
                evaluate_safety_sample(
                    float(t), obs.position_mean, uav.position_mean,
                    relative_covariance(obs.position_covariance, uav.position_covariance),
                    beta, radius, **tolerances,
                )
                for t, obs, uav in zip(times, predictions, uav_predictions)
            )
            current = predictions[0]
            t_cpa, d_cpa = compute_cpa(
                relative_mean(current.position_mean, state.position),
                current.velocity_mean - state.velocity,
                config.prediction.horizon, config.numerical.eps_velocity_squared,
            )
            cpa_margin = compute_cpa_margin(
                t_cpa, self._prediction(obstacle, t_cpa),
                predict_uav_distribution(state, t_cpa, use_uncertainty=use_uncertainty),
                beta, radius, **tolerances,
            )
            ttc = compute_chance_constraint_ttc(samples)
            risk = build_collision_risk(
                t_cpa, d_cpa, cpa_margin, ttc,
                warning_margin=config.risk.cpa_warning_margin,
                sigmoid_scale=config.risk.cpa_sigmoid_scale,
                time_scale=config.risk.ttc_time_scale,
                cpa_weight=config.risk.cpa_weight, ttc_weight=config.risk.ttc_weight,
            )
            margins = np.array([sample.safety_margin for sample in samples])
            gradients = np.array([sample.margin_gradient for sample in samples])
            potential = repulsive.potential(margins, times, risk.total_risk, config.repulsive)
            raw = repulsive.obstacle_force(margins, gradients, times, risk.total_risk, config.repulsive, saturate=False)
            force = repulsive.obstacle_force(margins, gradients, times, risk.total_risk, config.repulsive, saturate=True)
            results.append(ObstacleRiskResult(
                obstacle.metadata.obstacle_id, samples, risk, force, raw, potential,
            ))

        repulsion = np.sum([item.repulsive_force for item in results], axis=0) if results else np.zeros(3)
        global_risk, mean_risk = aggregate_risk(results)
        min_margin = min((sample.safety_margin for item in results for sample in item.safety_samples), default=float("inf"))
        min_ttc = min((item.collision_risk.probabilistic_ttc for item in results), default=float("inf"))
        total_raw = attraction + repulsion + damp
        total = saturate_vector(total_raw, config.control.max_total_force)
        return UPDAPFResult(
            attraction, repulsion, damp, total, global_risk, mean_risk,
            min_margin, min_ttc, results, total_raw, reference,
        )
