# Development Plan

## Phase 1 — Repository and mathematical infrastructure

Implement:
- config loading/validation;
- dataclasses;
- protocols/interfaces;
- saturation utility;
- covariance utilities;
- prediction time grid;
- initial tests;
- documentation consistency.

Exit conditions:
- package imports successfully;
- pytest runs;
- Phase-1 tests pass;
- no algorithm Phase-2+ code is added.

## Phase 2 — Prediction and Kalman Filter

Implement:
- CV transition;
- continuous white-acceleration process noise;
- KF predict/update;
- Joseph covariance update;
- pure `predict_distribution(tau)`;
- UAV predictor.

Tests:
- analytic state propagation;
- covariance propagation including cross-covariance;
- zero horizon;
- PSD/symmetry;
- query purity;
- one long prediction vs composition consistency where applicable.

## Phase 3 — Chance constraint

Implement:
- relative geometry;
- directional sigma;
- beta;
- collision radius;
- safety margin;
- exact non-degenerate margin gradient;
- zero-uncertainty branch.

Tests:
- beta monotonicity;
- known directional sigma;
- zero uncertainty;
- isotropic gradient;
- anisotropic finite-difference gradient;
- `n.T @ grad == -1` where applicable.

## Phase 4 — CPA / TTC / risk

Implement:
- CPA;
- CPA predicted chance margin;
- chance-constraint TTC;
- CPA risk;
- TTC risk;
- weighted total risk;
- global risk aggregation.

Tests:
- head-on analytic CPA;
- receding case;
- current violation TTC=0;
- no violation TTC=inf;
- interpolation;
- same-distance velocity comparison.

## Phase 5 — Potential fields

Implement:
- attractive potential/force;
- predictive repulsive potential/force;
- temporal weights;
- frozen risk gain;
- damping;
- single-obstacle and total saturation;
- traditional repulsive field math.

Tests:
- force directions;
- distance monotonicity in controlled case;
- uncertainty monotonicity in controlled isotropic case;
- potential/force finite-difference consistency before clipping/saturation;
- saturation norms.

## Phase 6 — UPD-APF planner

Implement orchestration only.

Requirements:
- no ground-truth access;
- one KF time advance per cycle;
- pure future distribution queries;
- ablation switches;
- structured results.

## Phase 7 — Minimal simulation

Implement:
- point-mass UAV;
- dynamic obstacle truth model;
- position sensor;
- world;
- no-obstacle;
- static-obstacle;
- near-head-on;
- collision checking;
- termination.

## Phase 8 — Full scenarios

Add:
- symmetric head-on diagnostic;
- crossing;
- moving away;
- same-distance/different-velocity;
- uncertainty comparison;
- multiple obstacles;
- sudden maneuver mismatch.

Do not hide failures.

## Phase 9 — Logging and visualization

Add:
- CLI;
- run metadata;
- resolved config;
- trajectory CSV;
- obstacle risk CSV;
- run metrics CSV;
- required Matplotlib figures;
- optional animation.

## Phase 10 — Traditional APF baseline integration

Add:
- unified planner interface;
- baseline CLI selection;
- baseline tests;
- no KF/risk feedback in baseline controller.

## Phase 11 — Benchmark and ablations

Add:
- multi-seed batch runner;
- success/collision aggregation;
- timing statistics;
- risk statistics;
- ablation configurations;
- reproducible result directories.

## Stop rule

After each requested phase:
1. run relevant tests;
2. update `IMPLEMENTATION_REPORT.md`;
3. report unresolved issues;
4. stop unless explicitly asked to continue.
