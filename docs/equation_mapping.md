# Equation-to-Function Mapping

| Mathematical quantity | Formula / concept | Intended Python location |
|---|---|---|
| Relative mean | `mu = p_obs - p_uav` | `safety/relative_geometry.py::relative_mean` |
| Relative covariance | `Sigma_r = Sigma_o + Sigma_u` | `safety/relative_geometry.py::relative_covariance` |
| Distance | `d = ||mu||` | `safety/relative_geometry.py::distance` |
| Unit direction | `n = mu / d` | `safety/relative_geometry.py::unit_direction` |
| CV transition | `A(tau)` | `prediction/process_noise.py::cv_transition` |
| White-acceleration Q | `Q(tau)` | `prediction/process_noise.py::white_acceleration_process_noise` |
| KF prediction | `x-=Ax`, `P-=APA^T+Q` | `prediction/kalman_filter.py::predict` |
| KF update | Joseph covariance update | `prediction/kalman_filter.py::update` |
| Future Gaussian | `A(tau)x`, `A P A^T + Q` | `prediction/kalman_filter.py::predict_distribution` |
| UAV future mean | `p+v*tau` | `prediction/uav_predictor.py::predict_position` |
| Directional sigma | `sqrt(n^T Sigma n)` | `safety/chance_constraint.py::directional_sigma` |
| Confidence beta | `Phi^-1(1-epsilon)` | `safety/chance_constraint.py::confidence_beta` |
| Collision radius | `r_u+r_o+d_margin` | `safety/chance_constraint.py::collision_radius` |
| Safety margin | `c=d-beta*sigma-R` | `safety/chance_constraint.py::safety_margin` |
| Margin gradient | `grad c` | `safety/chance_constraint.py::margin_gradient` |
| CPA | `t_cpa`, `d_cpa` | `safety/collision_metrics.py::compute_cpa` |
| CPA chance margin | predicted distribution at `t_cpa` | `safety/collision_metrics.py::compute_cpa_margin` |
| Chance TTC | first `c(tau)<=0` | `safety/collision_metrics.py::compute_chance_constraint_ttc` |
| CPA risk | sigmoid of CPA margin | `safety/collision_metrics.py::compute_cpa_risk` |
| TTC risk | `exp(-TTC/Tc)` | `safety/collision_metrics.py::compute_ttc_risk` |
| Combined risk | weighted sum | `safety/collision_metrics.py::combine_risk` |
| Global/mean risk | max / mean | `safety/risk_field.py::aggregate_risk` |
| Temporal weights | normalized `exp(-lambda*tau)` | `potential/repulsive.py::temporal_weights` |
| Frozen risk gain | `eta0(1+lambda_R*R)` | `potential/repulsive.py::risk_modulated_gain` |
| Attractive potential | piecewise | `potential/attractive.py::potential` |
| Attractive force | piecewise | `potential/attractive.py::force` |
| Predictive repulsive potential | multi-step exponential | `potential/repulsive.py::potential` |
| Predictive repulsive force | weighted gradient sum | `potential/repulsive.py::obstacle_force` |
| Traditional APF potential | clearance inverse potential | `potential/traditional_repulsive.py::potential` |
| Traditional APF force | clearance inverse force | `potential/traditional_repulsive.py::force` |
| Reference velocity | direction to goal | `potential/damping.py::reference_velocity` |
| Damping | `-kv(v-v_ref)` | `potential/damping.py::force` |
| Vector saturation | norm-preserving saturation | `utils/saturation.py::saturate_vector` |
| Covariance symmetrization | `(P+P.T)/2` | `utils/covariance.py::symmetrize` |
| PSD validation | eigenvalue tolerance | `utils/covariance.py::validate_psd` |
| Time grid | includes 0 and horizon | `utils/time_grid.py::prediction_time_grid` |
| Total UPD-APF command | attraction + repulsion + damping | `planner/upd_apf_planner.py::UPDAPFPlanner.compute_control` |
| Point-mass acceleration | `sat(kF*Fcmd, amax)` | `simulation/uav_model.py::command_acceleration` |
| UAV integration | point-mass step | `simulation/uav_model.py::step` |
| Path length | sum segment lengths | `evaluation/metrics.py::path_length` |
| Integrated risk | sum `R*dt` | `evaluation/metrics.py::integrated_risk` |
| Smoothness cost | sum acceleration differences squared | `evaluation/metrics.py::acceleration_difference_cost` |

## Important distinction

`margin_gradient` is the gradient of the chance-constrained margin.

It is **not** the gradient of the composite CPA/TTC risk. Do not rename or expose it as `risk_gradient`.

## Phase 5 public API

- `attractive.potential(position, goal, gain, switch_distance)` returns a Python
  float; `attractive.force(...)` returns a (3,) ndarray.
- `repulsive.temporal_weights(times, future_decay)` accepts a nonempty (M,)
  time array, normally supplied by `prediction_time_grid`.
- `repulsive.risk_modulated_gain(risk, eta_0, risk_gain)` returns the frozen gain.
- `repulsive.potential(margins, times, risk, config=RepulsiveConfig())` returns
  the scalar predictive potential with configured numerical exponent clipping.
- `repulsive.obstacle_force(margins, margin_gradients, times, risk, config,
  saturate=True)` consumes (M,), (M, 3), and (M,) arrays. The optional config
  defaults to `RepulsiveConfig()`; `saturate` is keyword-only. Use
  `saturate=False` for raw-force finite-difference checks. The leading
  coefficient is positive because the supplied gradient is with respect to UAV
  position. Only the per-obstacle limit is used.
- `damping.reference_velocity(position, goal, desired_speed, eps_distance=...)`
  accepts (3,) positions and returns (3,) velocity; the numerical threshold is
  keyword-only and defaults to `NumericalConfig().eps_distance`.
- `damping.force(velocity, reference_velocity, gain)` returns (3,) damping.

Risk and weights are held fixed during a cycle's local field evaluation.
Clipping and active force saturation invalidate the original exact gradient
identity. At the Phase-5 boundary, traditional repulsion and total-command equations
were future mappings. Phase 6 now implements total-command orchestration;
traditional repulsion remains a future-phase mapping.


## Phase 6 integration mapping

Phase 6 does NOT add new avoidance mathematics. UPDAPFPlanner.compute_control
composes the Phase 2--5 public APIs with these accounting operations:

| Quantity | Definition / owner |
|---|---|
| Common times | One prediction_time_grid(horizon, dt), including 0 and exact horizon |
| UAV distributions | predict_uav_distribution, with config.uav.use_uncertainty |
| Obstacle distributions | Pure PlannerObstacle.predictor.predict_distribution(tau) |
| Current obstacle velocity | Zero-time distribution velocity_mean |
| Collision radius | collision_radius using each binding metadata radius |
| Safety samples | relative_covariance then evaluate_safety_sample on common grid |
| CPA and CPA margin | compute_cpa; exact-time queries; compute_cpa_margin |
| Chance-boundary TTC | compute_chance_constraint_ttc(safety_samples) |
| Frozen risk | One build_collision_risk call; unchanged total_risk passed to Phase 5 |
| Obstacle diagnostics | Phase-5 potential, raw force and individually saturated force |
| Repulsive sum | Sum individually saturated forces; no extra sign/risk multiplier |
| Raw total force | Attraction + repulsive sum + damping |
| Command | saturate_vector(raw_total, config.control.max_total_force) |
| Risk summaries | aggregate_risk: maximum and arithmetic mean |
| Minimum margin/TTC | Minimum across samples / obstacle TTCs; infinity when empty |

PlannerObstacle(metadata, predictor) is frozen and defined in interfaces.py;
import it and UPDAPFPlanner from upd_apf.planner. No intermediate total repulsive
saturation is used. Ablation switches are not activated in this phase.

ObstacleRiskResult adds repulsive_force_raw and repulsive_potential;
UPDAPFResult adds total_force_raw and reference_velocity. New fields default to
None only for legacy constructors; planner calls always populate them.
Arrays are independently owned and read-only. No global potential is claimed.

Risk is an index, not probability. Negative margin violates a conservative
sufficient chance condition. CPA is nominal mean-distance CPA and probabilistic_ttc
is sampled/interpolated chance-boundary TTC. Degenerate errors propagate.
Perfectly symmetric head-on geometry has no manufactured lateral escape.
Phase 6 performs no state or motion integration.
