# Implementation Report

## Status

Phases 1 through 6 are implemented. Phase 6 is validated on
`codex/upd-apf-phase6-planner`, based on `fb3e209` (`upd-apf-phase5`).
Phase-6 changes remain uncommitted for human review.

## Current phase

Phase 6: complete, awaiting human review. Phase 7 has not started.

## Phase 1

- Added typed, validated configuration models and YAML loading.
- Added the shared UAV, measurement, prediction, safety, collision-risk, and planner-result dataclasses.
- Added subsystem protocols and dedicated covariance, vector-saturation, and prediction-grid utilities.
- The prediction grid includes both zero and the exact horizon endpoint.
- Added behavioral tests for configuration, shared types, covariance handling, saturation, and time grids.

## Phase 2

- Added the six-state constant-velocity transition and continuous white-acceleration process-noise model.
- Added constant-velocity Kalman prediction, position measurement updates in Joseph form, covariance symmetrization, and pure future-distribution queries.
- Added deterministic constant-velocity UAV prediction with explicit optional UAV covariance.
- Added tests for analytic propagation, covariance blocks and cross-covariance, zero-horizon identity, query purity, and posterior covariance properties.

## Phase 3

- Added relative mean, distance, direction, and independent relative-covariance operations using the UAV-to-obstacle sign convention.
- Added confidence coefficients, directional uncertainty, collision radius, chance-constrained margin, and its non-degenerate UAV-position gradient.
- Kept zero distance explicit as degenerate geometry and zero uncertainty as physical zero with the required `-n` gradient branch.
- Added tests for monotonic confidence, known and zero uncertainty, isotropic and anisotropic gradients, covariance validity, translation invariance, and rotation consistency.

## Phase 4

- Added clipped analytic CPA using estimated relative position and velocity.
- Added CPA chance-margin evaluation through the Phase-3 safety implementation.
- Added sampled/interpolated chance-constraint TTC, stable CPA risk, TTC risk, weighted risk, and construction of the shared `CollisionRisk` model.
- Added global maximum and mean risk aggregation without exposing a false risk gradient.
- Added tests for head-on and receding CPA, horizon clipping, TTC boundary cases and interpolation, stable risks, shared models, and aggregation.


## Phase 5 — Potential Fields

- Implemented `potential/attractive.py::potential/force`: quadratic near the
  goal, conic beyond the switch distance, with zero force at the goal.
  The far-field force norm is `gain * switch_distance`.
- Implemented `potential/repulsive.py::temporal_weights` using normalized
  `exp(-future_decay * tau)`. Subtracting the earliest time prevents all-weight
  underflow without changing the normalized weights. All supplied time samples
  are retained, including zero and the exact horizon from the existing grid.
- Implemented `risk_modulated_gain(risk, eta_0, risk_gain)` as
  `eta = eta_0 * (1 + risk_gain * risk)`, rejecting risk outside [0, 1].
  Risk is an index, not a probability. The caller must freeze risk and times
  throughout each control cycle's local field evaluation; no gradient of risk
  or eta is calculated.
- Implemented scalar `potential(margins, times, risk, config)`:
  `U = eta * sum(omega * exp(-c / length_scale))`.
- Implemented `obstacle_force(margins, margin_gradients, times, risk, config)`:
  `F_raw = +(eta / length_scale) * sum(omega * exp(-c / length_scale) * grad(c))`.
  The positive leading coefficient is essential: Phase 3 defines the
  UAV-position margin gradient, which is `-n` for isotropic covariance and
  therefore already points away from the obstacle.
- `obstacle_force(..., saturate=False)` exposes raw force. The default applies
  only single-obstacle norm saturation via the existing `saturate_vector` and
  `max_obstacle_force`. Extreme finite vectors are rescaled before calling that
  utility to avoid overflow of its squared norm; inactive saturation returns
  the raw values unchanged. The scalar potential does not depend on this limit.
- Reused existing `RepulsiveConfig` fields, including exponent clipping defaults
  [-50, 50]. Bounds must be finite, ordered, contain zero, and be safe for float64
  exponentiation. Pre-merge regression tests reject all-positive/all-negative
  intervals and confirm an exact unit exponential response at zero margin,
  including when zero is a clipping endpoint.
  Overflow of the finite margin/length quotient is handled before clipping.
  Unrepresentable gain, potential or raw-force results raise ValueError rather
  than returning NaN/Inf. No dependency or configuration field was added.
- Implemented `damping.py::reference_velocity` as the goal-directed desired
  speed, or zero within the existing numerical distance threshold.
  `damping.py::force` returns `-gain * (velocity - reference_velocity)`.
- Added a private shared `potential/_validation.py` for finite non-negative
  scalar parameters and finite vectors of exact shape (3,). Predictive inputs
  require matching nonempty (M,) arrays and (M, 3) gradients. Inputs are not
  mutated, reshaped, randomly perturbed, or silently repaired.
- Replaced only the three Phase-5 test placeholders. Added 138 cases covering
  exact equations, attractive continuity, central-difference gradients in both
  attractive regions, repulsive central differences using the existing Phase-3
  evaluation with zero/isotropic/anisotropic covariance and frozen risk/weights,
  repulsive sign and absence of hidden lateral force, temporal discount,
  risk/uncertainty monotonicity, clipping, active/inactive saturation (including
  very large finite forces), damping at goal, rigid transformations, purity,
  determinism and invalid input handling. Existing 85 tests are unchanged.

### Scope and API compatibility

The potential modules previously contained docstrings only, so no existing
public signatures needed alteration. Names follow docs/equation_mapping.md.
The user-authorized Phase-5 scope takes precedence over the older broader
docs/development_plan.md entry: total-force saturation and Traditional APF are
not implemented. There is no planner, multi-obstacle force aggregation,
simulation, risk gradient, or escape heuristic in this change. Phase 1–4 source
and mathematical definitions are unchanged.

### Verification

- Baseline: 85 passed with the specified Python environment.
- Full cumulative pytest: 223 passed (85 existing + 138 new), no skips/xfails.
- `F:\Anaconda\envs\upd-apf\python.exe -m compileall -q src`: passed.
- `git diff --check`: passed.
- Starting branch: `codex/upd-apf-phase5-potential-fields`, clean at
  `12d41e3`; local main and queried remote main both matched that commit.
  The `upd-apf-phase4` tag is present.

### Potential-field limitations

- The raw repulsive force equals the negative potential gradient only when
  eta/weights/covariance are frozen and exponent clipping is inactive.
  Active exponent clipping or norm saturation breaks that original identity;
  no globally conservative saturated controller is claimed.
- Perfectly symmetric isotropic head-on geometry has no manufactured lateral
  component and may stop, reverse or reach equilibrium.
- Negative margin violates the conservative sufficient chance condition;
  it does not prove the exact collision probability exceeds epsilon.
- Pathological magnitudes beyond float64 output range are rejected explicitly
  for repulsion; exponent clipping does not make arbitrary physical parameters
  representable. No Phase-5 TODO remains within the requested scope.

## Regression

Full cumulative suite: 286 passed (223 existing + 63 Phase-6 tests).
No failures, skips, xfails, or pytest warnings.

## Known limitations

- Perfectly symmetric collinear head-on geometry with isotropic covariance can stop, reverse, or reach equilibrium because no artificial lateral perturbation is introduced.
- The chance constraint is a conservative sufficient condition, not the exact three-dimensional collision probability.
- Fixed process and measurement noise do not automatically adapt to unexpected obstacle maneuvers.
- Phase 4 computes scalar CPA/TTC risk only; a full composite risk gradient is intentionally not defined.
- Accepted historical commit objects referenced during reconciliation were not present in any local ref, reflog, unreachable object, or remote ref, so the cumulative phase commits were reconstructed transparently from the repository's mathematical contract.

## Next step

Human review of Phase 6. Phase 7 requires a separate explicit request.


## Phase 6: planner integration

Phase 6 does NOT add new avoidance mathematics. It integrates Phase 2
prediction, Phase 3 chance safety, Phase 4 CPA/TTC/risk and Phase 5 potential
forces, adding multi-obstacle aggregation, total-force saturation and structured
diagnostics. No prediction, safety or potential source module was modified.
No environment, simulation, visualization or Traditional APF work was added.
No commit or push was performed.

### Public API and compatibility

```python
UPDAPFPlanner(config: UPDAPFConfig)
compute_control(self, uav_state: UAVState, goal: ArrayLike,
                obstacles: Sequence[PlannerObstacle]) -> UPDAPFResult
```

`PlannerObstacle(metadata: ObstacleMetadata, predictor: Predictor)` is frozen,
defined in interfaces.py and exported from upd_apf.planner alongside the planner.
This avoids circular imports. The insufficient Sequence[Predictor] annotation
in Planner now names Sequence[PlannerObstacle]. Bare predictors are rejected.
Metadata requires a nonempty string id and finite nonnegative radius.
No private estimator fields, fabricated ids or shared obstacle radius are used.

ObstacleRiskResult appends repulsive_force_raw and repulsive_potential;
its existing repulsive_force is individually saturated force.
UPDAPFResult appends total_force_raw and reference_velocity.
New fields default to None for old constructor compatibility; planner results
always populate them. New vector fields are copied and read-only. There are no
redundant sample arrays or new global potential fields.

### Control cycle

1. Validate state, goal, bindings, distributions and total-force limit. Existing
   control.max_total_force must be finite and nonnegative; zero is allowed,
   consistently with saturate_vector.
2. Call prediction_time_grid(horizon, dt) once. All obstacles share it,
   including zero and exact horizon (also for horizon=1 and dt=0.3).
3. Call predict_uav_distribution at each time. UAV covariance follows
   config.uav.use_uncertainty, with no new covariance dynamics. Obstacle queries
   use predict_distribution only. Current obstacle velocity comes from the
   zero-time GaussianPrediction.velocity_mean. Predictor absolute timestamps
   are validated; SafetySample.time uses relative prediction offsets.
4. Call collision_radius separately for each metadata radius and confidence_beta
   once per cycle. Call relative_covariance and evaluate_safety_sample per pair.
5. Call compute_cpa on current relative position/velocity. Query both predictors
   at exact CPA time, including off-grid times, then call compute_cpa_margin.
   CPA minimizes nominal mean distance, not chance margin.
6. Pass samples to compute_chance_constraint_ttc. Phase 4 owns crossing and
   interpolation. Call build_collision_risk once per obstacle with RiskConfig.
7. Freeze total_risk and pass it unchanged to repulsive.potential and
   repulsive.obstacle_force with saturate=False and saturate=True.
8. Sum individually saturated forces, without an extra sign/risk multiplier
   or intermediate total-repulsion saturation. max_total_repulsive_force is
   intentionally unused under the authoritative two-stage Phase-6 contract.
9. Obtain attraction, reference velocity and damping from Phase 5. Compose
   total_force_raw = attractive_force + repulsive_force + damping_force, then
   call saturate_vector(total_force_raw, config.control.max_total_force).
10. Call aggregate_risk for maximum and arithmetic mean. Minimum margin covers
    all samples; minimum TTC covers all obstacles. Diagnostics preserve input
    order; aggregate outputs are permutation invariant within floating tolerance.

Empty obstacles return empty diagnostics, zero repulsion/risks and infinite
minimum margin/TTC; attraction, damping and final saturation still run.
At goal, attraction/reference velocity vanish and damping brakes remaining
velocity. Obstacle repulsion remains active.

### Validation

The prior planner test file was only a docstring placeholder. Added 63 tests
cover binding/API and legacy constructors; empty/goal behavior; one common grid;
direct Phase 2--5 numerical comparisons; exact off-grid CPA; chance-TTC;
no double sign/risk weighting; addition/cancellation; two-stage saturation;
summaries/accounting; permutation and translation invariance; proper-rotation
covariance with anisotropic covariance and active saturation; uncertainty;
real KF state/covariance/time purity; deterministic calls and result ownership;
malformed inputs/predictions/limits; per-obstacle radii; frozen risk call counts;
degenerate grid/CPA geometry and no lateral kick in symmetric head-on geometry.

An initial new grid assertion compared 0.3*3 and literal 0.9 bitwise. It was
corrected to compare exactly against the common grid API, keeping exact endpoint
checks. No existing test or algorithm was changed to conceal a failure.

Full pytest: 286 passed (223 existing + 63 new), no skips or xfails.
python -m compileall -q src: passed. git diff --check: passed.

### Interpretation, limitations and TODO

- Risk is a dimensionless index, not calibrated collision probability.
- c<0 violates a conservative sufficient condition; it does not establish that
  true collision probability exceeds epsilon.
- probabilistic_ttc is sampled/interpolated chance-boundary TTC, not exact
  geometric TTC. Finite sampling can miss between-sample events.
- Degenerate geometry errors propagate, including exact CPA coincidences.
  No fallback direction or fabricated gradient is introduced.
- Perfectly symmetric isotropic head-on geometry has no manufactured lateral
  escape; stopping, reversing or equilibrium remain possible.
- Frozen risk, damping, clipping and saturation do not make the command the
  exact negative gradient of a global conservative potential.
- Third-party predictors must honor the pure-query protocol. Real KF purity
  is regression tested; arbitrary predictor implementations remain responsible
  for their own query purity.
- Ablation switches remain reserved for later benchmark work. This phase uses
  the full-chain pseudocode and the existing config.uav.use_uncertainty option.
- No Phase-6 implementation TODO remains. Motion integration and simulation
  require later explicit authorization.
