# Codex Initial Task

Read, in this order:

1. `AGENTS.md`
2. `docs/mathematical_contract.md`
3. `docs/PROJECT_SPEC.md`
4. `docs/equation_mapping.md`
5. `docs/development_plan.md`

Do not modify the mathematical contract silently.

## Current task

Implement **Phase 1 only**:

- package skeleton cleanup;
- configuration model;
- YAML configuration loading;
- dataclasses;
- protocol/interface definitions;
- numerical saturation utilities;
- covariance validation/symmetrization utilities;
- prediction time-grid utility;
- test scaffolding and Phase-1 tests.

Do NOT implement:
- Kalman Filter;
- chance constraint;
- CPA/TTC;
- potential fields;
- UPD-APF planner;
- simulation;
- benchmark.

After implementation:

1. run `pytest`;
2. fix Phase-1 failures;
3. update `IMPLEMENTATION_REPORT.md`;
4. report:
   - files created/changed,
   - tests executed,
   - test results,
   - unresolved issues;
5. STOP and wait for approval before Phase 2.
