import numpy as np
import pytest

from upd_apf.data_types import CollisionRisk, GaussianPrediction
from upd_apf.safety.collision_metrics import (
    build_collision_risk,
    combine_risk,
    compute_chance_constraint_ttc,
    compute_cpa,
    compute_cpa_margin,
    compute_cpa_risk,
    compute_ttc_risk,
)


def prediction(time, position, velocity=(0, 0, 0), covariance=None):
    return GaussianPrediction(time, position, velocity, np.zeros((3, 3)) if covariance is None else covariance)


def test_head_on_cpa_analytic_case():
    time, distance = compute_cpa([10, 0, 0], [-2, 0, 0], 10)
    assert time == 5 and distance == 0


def test_receding_case_clips_to_zero():
    time, distance = compute_cpa([10, 0, 0], [2, 0, 0], 10)
    assert time == 0 and distance == 10


def test_cpa_clips_to_horizon():
    time, distance = compute_cpa([10, 0, 0], [-1, 0, 0], 3)
    assert time == 3 and distance == 7


def test_stationary_relative_motion_uses_current_time():
    assert compute_cpa([3, 4, 0], [0, 0, 0], 3) == (0, 5)


def test_cpa_sign_convention_crossing():
    # Obstacle is to the right and moving left relative to the UAV.
    assert compute_cpa([4, 0, 0], [-2, 0, 0], 4)[0] == 2


def test_cpa_margin_reuses_phase3_equation():
    obstacle = prediction(2, [5, 0, 0], covariance=np.eye(3))
    uav = prediction(2, [0, 0, 0])
    assert compute_cpa_margin(2, obstacle, uav, beta=2, collision_radius=1) == 2


def test_current_violation_ttc_is_zero():
    assert compute_chance_constraint_ttc([0, 1, 2], [-.1, 1, 2]) == 0


def test_safe_horizon_ttc_is_infinite():
    assert np.isinf(compute_chance_constraint_ttc([0, 1, 2], [3, 2, 1]))


def test_ttc_interpolates_first_crossing():
    assert compute_chance_constraint_ttc([0, 1, 2], [2, 1, -1]) == 1.5


def test_boundary_sample_counts_as_crossing():
    assert compute_chance_constraint_ttc([0, 1], [1, 0]) == 1


def test_cpa_risk_matches_stable_sigmoid():
    assert compute_cpa_risk(1.5, 1.5, .5) == .5
    assert compute_cpa_risk(-1e6, 0, 1) == 1


def test_ttc_risk_definition_and_infinity_sentinel():
    assert np.isclose(compute_ttc_risk(2, 2), np.exp(-1))
    assert compute_ttc_risk(float("inf"), 2) == 0


def test_weighted_total_risk():
    assert combine_risk(.8, .2, .25, .75) == pytest.approx(.35)


def test_weights_are_validated():
    with pytest.raises(ValueError, match="sum"):
        combine_risk(.5, .5, .5, .6)


def test_builder_returns_shared_collision_risk_dataclass():
    result = build_collision_risk(
        1, 2, 0, float("inf"), warning_margin=0, sigmoid_scale=1,
        time_scale=2, cpa_weight=.5, ttc_weight=.5,
    )
    assert isinstance(result, CollisionRisk)
    assert result.cpa_risk == .5 and result.ttc_risk == 0 and result.total_risk == .25


def test_approaching_has_more_risk_than_receding_at_same_distance():
    approaching_t, _ = compute_cpa([5, 0, 0], [-1, 0, 0], 5)
    receding_t, _ = compute_cpa([5, 0, 0], [1, 0, 0], 5)
    # Their first-boundary times reflect the same velocity distinction.
    approaching = compute_ttc_risk(approaching_t, 2)
    receding = compute_ttc_risk(float("inf") if receding_t == 0 else receding_t, 2)
    assert approaching > receding


def test_cpa_is_translation_invariant():
    r = np.array([3., 4, 0])
    assert compute_cpa(r, [-1, 0, 0], 3) == compute_cpa(r, [-1, 0, 0], 3)
