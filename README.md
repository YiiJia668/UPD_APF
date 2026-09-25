# UPD-APF

**Uncertainty-aware Predictive Dynamic Artificial Potential Field** for dynamic 3D UAV obstacle avoidance.

This repository is intentionally initialized as a **research scaffold** for Codex-assisted implementation.  
The mathematical contract and development rules are fixed in `AGENTS.md` and `docs/mathematical_contract.md`.

## Core idea

UPD-APF combines:

- goal attraction;
- Kalman-filter obstacle state estimation;
- multi-step obstacle prediction;
- covariance propagation;
- chance-constrained safety margins;
- CPA (Closest Point of Approach);
- chance-constraint TTC;
- multi-step predictive repulsion;
- CPA/TTC risk modulation;
- damping and UAV kinematic saturation.

The central safety margin is

\[
c_i(\tau)
=
d_i(\tau)
-
\beta \sigma_i(\tau)
-
R_{\mathrm{collision},i}.
\]

The intended predictive repulsive force is

\[
F_{\mathrm{rep},i}
=
\frac{\eta_i}{\ell_r}
\sum_m
\omega_m
\exp\left(-\frac{c_{im}}{\ell_r}\right)
\nabla_{p_u} c_{im}.
\]

## Important

This repository does **not** yet contain the final algorithm implementation.

Codex should implement it phase-by-phase according to:

- `AGENTS.md`
- `docs/PROJECT_SPEC.md`
- `docs/mathematical_contract.md`
- `docs/equation_mapping.md`
- `docs/development_plan.md`

## Recommended first Codex request

> Read AGENTS.md and all files in docs/. Do not implement Phase 2 or later. Implement Phase 1 only, run pytest, summarize files changed and any unresolved issues, then stop.

## Planned installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Planned commands

```bash
pytest

python scripts/run_simulation.py \
  --planner upd_apf \
  --scenario near_head_on \
  --seed 42 \
  --save-results \
  --no-animation
```

## Planned repository structure

```text
configs/
docs/
src/upd_apf/
tests/
scripts/
results/
```

## Research scope

Initial implementation:
- Traditional APF baseline
- UPD-APF
- dynamic obstacle scenarios
- reproducible logging and metrics
- ablations

Not in the initial implementation:
- PSO/GWO/DE
- Risk-Guided PSO
- Warm-Start replanning
- event-triggered replanning
- deep-learning prediction
- ROS/PX4/Gazebo
