# Implementation Report

## Status

Phases 1 through 4 are complete on the same cumulative branch.
Phase 5 has not started.

## Current phase

Phase 4 — COMPLETE
Awaiting approval for Phase 5.

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

## Regression

Full cumulative suite: 85 passed, 0 failed, 0 skipped, 0 warnings.

## Known limitations

- Perfectly symmetric collinear head-on geometry with isotropic covariance can stop, reverse, or reach equilibrium because no artificial lateral perturbation is introduced.
- The chance constraint is a conservative sufficient condition, not the exact three-dimensional collision probability.
- Fixed process and measurement noise do not automatically adapt to unexpected obstacle maneuvers.
- Phase 4 computes scalar CPA/TTC risk only; a full composite risk gradient is intentionally not defined.
- Accepted historical commit objects referenced during reconciliation were not present in any local ref, reflog, unreachable object, or remote ref, so the cumulative phase commits were reconstructed transparently from the repository's mathematical contract.

## Next step

Phase 5 — Potential Fields, pending explicit approval.
