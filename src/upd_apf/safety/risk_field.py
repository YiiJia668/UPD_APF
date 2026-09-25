"""Aggregation of already-computed obstacle risks.

This module deliberately exposes no ``risk_gradient``: a Phase-3 margin gradient
is not the gradient of the composite CPA/TTC risk.
"""

from collections.abc import Iterable

import numpy as np

from upd_apf.data_types import CollisionRisk, ObstacleRiskResult


def aggregate_risk(risks: Iterable[float | CollisionRisk | ObstacleRiskResult]) -> tuple[float, float]:
    values: list[float] = []
    for risk in risks:
        if isinstance(risk, ObstacleRiskResult):
            value = risk.collision_risk.total_risk
        elif isinstance(risk, CollisionRisk):
            value = risk.total_risk
        else:
            value = float(risk)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("individual risks must be finite and in [0, 1]")
        values.append(value)
    if not values:
        return 0.0, 0.0
    return max(values), float(np.mean(values))


def global_risk(risks: Iterable[float | CollisionRisk | ObstacleRiskResult]) -> float:
    return aggregate_risk(risks)[0]
