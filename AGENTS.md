# AGENTS.md — UPD-APF Repository Instructions

## Project
UPD-APF = Uncertainty-aware Predictive Dynamic Artificial Potential Field.

This repository is a research codebase for dynamic 3D UAV local path planning under obstacle-state uncertainty.

## Non-negotiable workflow
1. Work phase-by-phase.
2. Do not implement later phases unless explicitly requested.
3. Before changing a mathematical formula, explain the issue and wait for approval.
4. Run the full relevant pytest suite after each phase.
5. Never change tests merely to hide an algorithmic failure.
6. Preserve reproducibility: all stochastic components must accept an explicit seed / RNG.
7. Planner code must not access ground-truth obstacle state.
8. Keep mathematical operations in dedicated modules; planners orchestrate only.

## Mathematical sign convention
Relative mean:
    mu = p_obstacle - p_uav

For non-degenerate geometry:
    d = ||mu||
    n = mu / d

Therefore n points UAV -> obstacle.

Chance-constrained safety margin:
    c = d - beta * sigma - R_collision

For isotropic covariance:
    grad_{p_uav} c = -n

The repulsive force is:
    F_rep = +(eta / ell_r) * sum_m[
        omega_m * exp(-c_m / ell_r) * grad(c_m)
    ]

DO NOT add another leading minus sign to this implemented force formula.
Under isotropic covariance, the force must point approximately obstacle -> UAV.

## Zero-distance rule
If d <= eps_distance:
- classify as degenerate geometry / collision condition;
- do not invent a random avoidance direction;
- do not silently normalize by d + eps and pretend it is the exact gradient.

## Zero-uncertainty rule
Physical directional variance:
    q = n.T @ Sigma @ n

If q is a tiny negative number within PSD tolerance, clamp it to zero.
If q is materially negative, raise an error.

Physical sigma:
    sigma = sqrt(q)

Do NOT define sigma = sqrt(max(q, eps_sigma)).

If sigma <= eps_sigma in margin-gradient evaluation:
    grad(c) = -n

eps_sigma is a branch threshold, not artificial uncertainty.

## Process-noise model
Use a continuous white-acceleration driven constant-velocity model.

A(tau) = [[I, tau I],
          [0,    I  ]]

Q(tau) = [[tau^3/3 Sa, tau^2/2 Sa],
          [tau^2/2 Sa, tau Sa]]

with Sa = q_a * I_3 unless a matrix is explicitly configured.

The configuration parameter represents acceleration-noise spectral density,
not a per-step acceleration standard deviation.

## Kalman covariance update
Use Joseph form:
    P = (I-KH) P_minus (I-KH)^T + K R K^T

Then symmetrize:
    P = 0.5 * (P + P.T)

Do not explicitly invert the innovation covariance if a stable solve is available.

## Prediction time grid
The common safety/risk grid includes tau = 0 and T_p.

If T_p is not exactly divisible by prediction_dt, explicitly append T_p.

At tau = 0:
    A(0) = I
    Q(0) = 0

predict_distribution(0) must equal the current posterior estimate.

## Chance constraint interpretation
The implemented condition
    c = d - beta*sigma - R_collision >= 0
is a conservative sufficient safety condition under the stated Gaussian projection model.

If c < 0, say:
    "the sufficient chance constraint is violated"

Do NOT claim that the true collision probability must therefore exceed epsilon.

## Probabilistic TTC interpretation
The quantity called probabilistic TTC is:
    TTC_cc = inf { tau in [0, T_p] : c(tau) <= 0 }

It is the first predicted violation time of the chance-constrained safety boundary.

If c(0) <= 0:
    TTC_cc = 0

If the horizon is safe:
    TTC_cc = +inf

## Risk-gradient rule
margin_gradient is NOT risk_gradient.

Do not expose margin_gradient as "risk_gradient".
A full risk gradient is not part of the initial implementation.

## Frozen-risk local field
CPA/TTC risk modulation eta is evaluated once per control cycle and frozen while
constructing the local predictive repulsive field.

Do not claim the final saturated/clipped controller is globally the exact negative
gradient of one scalar potential.

## Symmetric head-on limitation
A perfectly collinear head-on case with isotropic covariance may contain no lateral
component and may stop, reverse, or reach equilibrium.

Do not inject hidden random lateral forces to make this case pass.
Keep a separate near-head-on benchmark and a symmetric-head-on diagnostic scenario.

## Dependency constraints
- prediction must not depend on planner/simulation
- safety must not read true obstacle state
- potential consumes computed safety/risk objects
- planner orchestrates but does not own base mathematics
- visualization never feeds back into control
- baseline Traditional APF must not use KF/covariance/CPA/TTC

## Current development status
Repository scaffold only.

Start with Phase 1 unless the user explicitly requests another phase.
