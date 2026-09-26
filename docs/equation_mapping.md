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
| Point-mass acceleration | `a = force_to_acceleration_gain * F_total` (Phase 7; no secondary saturation) | `simulation/uav_model.py::PointMassUAVModel.step` |
| UAV integration | `p_next=p+v*dt+0.5*a*dt^2`, `v_next=v+a*dt` | `simulation/uav_model.py::PointMassUAVModel.step` |
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


## Phase 7 closed-loop mapping (authoritative for minimal simulation)

The explicit Phase-7 request supersedes the old scaffold's semi-implicit
integration and speed/acceleration clamps. Phase 1--6 mathematics is unchanged.

| Quantity | Formula / owner |
|---|---|
| Physical acceleration | `a = force_to_acceleration_gain * F_total`, gain finite and positive; `PointMassUAVModel.step` |
| UAV truth integration | Constant held acceleration: `p+v*dt+0.5*a*dt^2`, `v+a*dt` |
| UAV covariance | Copy the input position covariance; no new covariance dynamics |
| Obstacle truth | `p_next=p+v*dt`, unchanged velocity; `ConstantVelocityObstacle.advance` |
| Measurement | `z=p_true+nu`, `nu~N(0,R_sensor)`; `PositionSensor.measure` |
| Measurement reproducibility | Explicit Generator or seed for nonzero covariance; exact copy for zero covariance |
| Estimator initialization | Initial belief/prior at t=0, before the first t=0 measurement update |
| Posterior scheduling | At k=0 measure/update/plan without prediction; at k>0 predict(previous actual dt), then measure/update/plan |
| Planner binding | Public metadata plus estimator, never truth position/velocity |
| Physical collision | `norm(p_u-p_o_true) <= r_u+r_o`; `Simulator._termination` |
| Goal arrival | Position distance <= goal_tolerance AND speed <= goal_speed_tolerance |
| Termination priority | Collision, then goal_reached, then max_time; before measurements |
| Final step | End boundary `min(max_time,(k+1)*simulation.dt)`, dt=boundary-current time |
| Prediction grid | Existing prediction.dt/horizon remains independent of simulation.dt |

The boundary-index clock is mathematically equivalent to a final partial step
and avoids accumulation of floating-point clock error. Physical collision is
sampled at cycle boundaries, including the final state. No swept detection is
implemented. Safety-buffer overlap or chance-boundary violation alone never
terminates a run. The final estimator can lag terminal truth by the final dt;
there is no terminal measurement/planner cycle.

Legacy UAVConfig.max_speed and max_acceleration remain compatible configuration
fields but are not applied by Phase 7. The existing force_to_acceleration_gain
(default 1.0) is the single force-to-acceleration source, interpreted as inverse
effective mass (gain=1/m_eff); no independent mass parameter is exposed. Existing simulation
goal tolerances are retained at 0.5 for both distance and speed. Scenario-only
settings desired_speed=0 and damping.gain=2 demonstrate stable goal approach
without changing the global planner defaults or adding goal slowdown logic.
