# UPD-APF Project Specification

## Objective

Implement a modular, testable, reproducible Python research codebase for 3D UAV local path planning with dynamic uncertain obstacles.

The first complete research version shall contain:

1. Traditional APF baseline.
2. UPD-APF planner.
3. Dynamic obstacle state estimation.
4. Constant-velocity Kalman Filter.
5. Multi-step uncertainty propagation.
6. Chance-constrained safety margin.
7. CPA and chance-constraint TTC.
8. CPA/TTC risk modulation.
9. Predictive multi-step repulsive field.
10. Damping and point-mass UAV motion constraints.
11. Scenario library.
12. CSV logging.
13. Matplotlib visualization.
14. Pytest tests.
15. Benchmark/ablation infrastructure.

## Technology

- Python >= 3.11
- NumPy
- SciPy
- pandas
- Matplotlib
- PyYAML
- pytest

Avoid large frameworks in the first implementation.

## Out of scope for the initial version

- PSO/GWO/DE
- Risk-Guided PSO
- Warm-Start
- event-triggered replanning
- neural trajectory prediction
- multi-UAV coordination
- ROS
- PX4
- Gazebo
- GPU frameworks

## UAV model

Use a 3D point-mass approximation.

\[
a_{\rm raw}=k_FF_{\rm cmd}
\]

Apply:
\[
\|a_{\rm cmd}\|\le a_{\max}.
\]

Then:
\[
v_{k+1}
=
\operatorname{sat}(v_k+a_{\rm cmd}\Delta t,v_{\max})
\]

\[
p_{k+1}
=
p_k+v_{k+1}\Delta t.
\]

## Obstacle simulation

The planner may use only sensor measurements and estimates.

Ground truth exists only for:
- environment evolution;
- collision checking;
- evaluation.

Default matched-model dynamic obstacles use the same CV/white-acceleration stochastic family used by the KF.

A separate sudden-maneuver scenario intentionally violates the model.

## Required dataclasses

### UAVState
- position `(3,)`
- velocity `(3,)`
- position_covariance `(3,3)`

### ObstacleMeasurement
- obstacle_id
- timestamp
- position `(3,)`

### ObstacleMetadata
- obstacle_id
- radius

### GaussianPrediction
- time
- position_mean `(3,)`
- velocity_mean `(3,)`
- position_covariance `(3,3)`

### SafetySample
- time
- distance
- direction `(3,)`
- directional_sigma
- safety_margin
- margin_gradient `(3,)`

### CollisionRisk
- t_cpa
- d_cpa
- cpa_margin
- probabilistic_ttc
- cpa_risk
- ttc_risk
- total_risk

### ObstacleRiskResult
- obstacle_id
- safety_samples
- collision_risk
- repulsive_force

### UPDAPFResult
- attractive_force
- repulsive_force
- damping_force
- total_force
- global_risk
- mean_risk
- min_safety_margin
- min_ttc
- per_obstacle_results

## Configuration

All constants must be configurable.

Required groups include:
- simulation;
- prediction;
- UAV;
- Kalman/filter noise;
- chance constraint;
- attractive field;
- repulsive field;
- CPA/TTC risk;
- damping;
- saturation;
- numerical tolerances;
- baseline APF;
- scenario termination;
- random seed.

## Core runtime order per control step

1. Read UAV state.
2. Generate obstacle measurements.
3. Advance each obstacle KF exactly once to current time.
4. Fuse current measurement.
5. Query future distributions without mutating filter state.
6. Build common future time grid including 0 and horizon endpoint.
7. Evaluate relative mean/covariance.
8. Evaluate chance-constrained safety samples.
9. Evaluate CPA.
10. Evaluate chance-constraint TTC.
11. Compute CPA/TTC risk.
12. Freeze risk gain for this cycle.
13. Construct per-obstacle predictive repulsive force.
14. Sum/saturate repulsive force.
15. Compute attractive and damping forces.
16. Compute/saturate total command.
17. Advance UAV model.
18. Advance obstacle ground truth.
19. Check continuous/step-aware collision under the chosen simulator motion assumption.
20. Log all state/control/risk/evaluation outputs.

## Global risk

First implementation:
\[
R_{\mathrm{global}}=\max_i R_i.
\]

Also log:
\[
R_{\mathrm{mean}}=\frac1N\sum_iR_i.
\]

No obstacles:
- global risk = 0
- mean risk = 0
- repulsive force = zero vector
- min safety margin = +inf
- min TTC = +inf

## Scenario library

Required scenarios:

1. `no_obstacle`
2. `static_obstacle`
3. `near_head_on`
4. `symmetric_head_on`
5. `crossing`
6. `moving_away`
7. `same_distance_different_velocity`
8. `uncertainty_comparison`
9. `multi_obstacle`
10. `sudden_maneuver`

The perfectly symmetric head-on case is diagnostic and is not required to produce successful lateral bypass in the initial method.

## Required metrics

Per run:
- success
- collision
- path length
- travel time
- minimum true clearance
- minimum probabilistic safety margin
- mean control computation time
- p95 control computation time
- mean global risk
- maximum global risk
- integrated risk
- acceleration-difference smoothness cost
- maximum acceleration
- mean acceleration

## Required logs

Trajectory log:
- time
- UAV position/velocity
- goal distance
- requested force components/norms
- applied acceleration
- global/mean risk
- min safety margin
- min TTC

Long-format obstacle-risk log:
- time
- obstacle_id
- total risk
- CPA time
- CPA mean distance
- CPA safety margin
- chance-constraint TTC
- current directional sigma

## Visualization

Use Matplotlib only.

Produce separate figures for:
- 3D trajectory
- global risk vs time
- minimum safety margin vs time
- TTC vs time
- attractive-force norm
- repulsive-force norm
- command-force norm
- obstacle prediction error
- Kalman position-covariance trace

Optional 3D animation must not be required for headless tests.

## CLI target

Example:

```bash
python scripts/run_simulation.py \
  --planner upd_apf \
  --scenario near_head_on \
  --config configs/default.yaml \
  --seed 42 \
  --save-results \
  --no-animation
```

Baseline:

```bash
python scripts/run_simulation.py \
  --planner traditional_apf \
  --scenario near_head_on \
  --seed 42
```

## Ablation switches

The UPD-APF design must allow:
- `use_uncertainty`
- `use_cpa_risk`
- `use_ttc_risk`
- `use_predictive_horizon`
- `use_damping`
- `use_uav_uncertainty`

Do not build five independent planners.

## Reproducibility

Use explicit NumPy generators.

Prefer separate reproducible RNG streams for:
- scenario/ground truth;
- process noise;
- sensor noise.

Planner comparisons under the same seed should use the same physical/noise realization where practical.

## Acceptance principles

Correctness and reproducibility have priority over visual appearance or premature optimization.

Never make a benchmark pass by:
- reading truth state in the planner;
- adding hidden lateral offsets;
- changing the mathematical contract;
- weakening tests without explanation;
- silently clipping invalid covariance to arbitrary values.
