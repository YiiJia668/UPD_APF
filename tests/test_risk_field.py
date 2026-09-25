import pytest

from upd_apf.data_types import CollisionRisk
from upd_apf.safety.risk_field import aggregate_risk, global_risk


def collision_risk(total):
    return CollisionRisk(0, 1, 1, float("inf"), total, 0, total)


def test_empty_risk_aggregation():
    assert aggregate_risk([]) == (0, 0)


def test_global_and_mean_risk():
    assert aggregate_risk([.2, .8, .5]) == pytest.approx((.8, .5))


def test_aggregation_accepts_shared_dataclass():
    assert global_risk([collision_risk(.4), collision_risk(.7)]) == .7


def test_invalid_risk_rejected():
    with pytest.raises(ValueError):
        aggregate_risk([1.1])


def test_margin_gradient_is_not_exported_as_risk_gradient():
    import upd_apf.safety.risk_field as module
    assert not hasattr(module, "risk_gradient")
