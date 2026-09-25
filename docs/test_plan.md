# Test Plan

Hard requirements include:

1. Lower collision probability tolerance -> larger beta.
2. `Sigma=diag(4,1,1)`, `n=[1,0,0]` -> sigma=2.
3. `Sigma=0` -> chance margin exactly `d-R` (within floating tolerance).
4. Isotropic covariance -> margin gradient approximately `-n`.
5. Non-degenerate full gradient -> `n.T @ grad ≈ -1`.
6. Anisotropic margin gradient matches finite differences.
7. Increasing distance reduces unsaturated single-obstacle repulsion in controlled geometry.
8. Increasing isotropic covariance reduces safety margin and increases unsaturated single-obstacle repulsion in controlled geometry.
9. Head-on CPA analytic case matches expected time.
10. Receding obstacle CPA clips to zero.
11. `c(0)<=0` -> chance TTC=0.
12. Safe horizon -> TTC=inf and TTC risk=0.
13. TTC interpolation is correct at the first positive-to-nonpositive crossing.
14. Approaching obstacle risk > receding obstacle risk under same distance/covariance.
15. Far obstacle repulsion is negligible.
16. No-obstacle command points primarily toward goal.
17. Repulsive and total command norm saturation holds.
18. No NaN/Inf in states, forces or covariance during multi-obstacle simulation; TTC=inf is an allowed sentinel.
19. Prediction queries do not mutate current KF state.
20. Planner never receives ground-truth obstacle state.
21. Same seed reproduces physical/noise trajectories.
22. Traditional APF does not invoke KF/covariance/CPA/TTC.
23. Translation invariance for scalar geometry/risk quantities.
24. Rotation consistency for vectors/covariances in controlled tests.
25. Symmetric-head-on diagnostic is not forced to pass by hidden perturbation.
