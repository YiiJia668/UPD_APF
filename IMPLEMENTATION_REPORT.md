# Implementation Report

## Status

Phases 1 through 5 are complete. Phase 5 is implemented on
`codex/upd-apf-phase5-potential-fields`; it has not been committed or merged.

## Current phase

Phase 5 — COMPLETE
Phase 6 has not started.

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

Full cumulative suite: 223 passed (85 existing + 138 Phase-5 tests).
No failures, skips, xfails, or pytest warnings.

## Known limitations

- Perfectly symmetric collinear head-on geometry with isotropic covariance can stop, reverse, or reach equilibrium because no artificial lateral perturbation is introduced.
- The chance constraint is a conservative sufficient condition, not the exact three-dimensional collision probability.
- Fixed process and measurement noise do not automatically adapt to unexpected obstacle maneuvers.
- Phase 4 computes scalar CPA/TTC risk only; a full composite risk gradient is intentionally not defined.
- Accepted historical commit objects referenced during reconciliation were not present in any local ref, reflog, unreachable object, or remote ref, so the cumulative phase commits were reconstructed transparently from the repository's mathematical contract.

## Next step

Phase 6 — Planner, pending explicit approval.
