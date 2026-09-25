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
| Total UPD-APF command | attraction + repulsion + damping | `planner/upd_apf_planner.py::compute_control` |
| Point-mass acceleration | `sat(kF*Fcmd, amax)` | `simulation/uav_model.py::command_acceleration` |
| UAV integration | point-mass step | `simulation/uav_model.py::step` |
| Path length | sum segment lengths | `evaluation/metrics.py::path_length` |
| Integrated risk | sum `R*dt` | `evaluation/metrics.py::integrated_risk` |
| Smoothness cost | sum acceleration differences squared | `evaluation/metrics.py::acceleration_difference_cost` |

## Important distinction

`margin_gradient` is the gradient of the chance-constrained margin.

It is **not** the gradient of the composite CPA/TTC risk. Do not rename or expose it as `risk_gradient`.
