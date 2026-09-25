import pytest

from upd_apf.config import UPDAPFConfig, config_from_mapping, load_config


def test_default_yaml_loads():
    config = load_config("configs/default.yaml")
    assert config.prediction.horizon == 3.0
    assert config.simulation.seed == 42


def test_defaults_construct():
    assert UPDAPFConfig().risk.cpa_weight == 0.5


@pytest.mark.parametrize("epsilon", [0.0, 0.5, -0.1])
def test_probability_validation(epsilon):
    with pytest.raises(ValueError):
        config_from_mapping({"chance_constraint": {"collision_probability": epsilon}})


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        config_from_mapping({"risk": {"cpa_weight": 0.8, "ttc_weight": 0.8}})


def test_unknown_group_rejected():
    with pytest.raises(ValueError, match="unknown"):
        config_from_mapping({"prediciton": {}})


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="invalid prediction"):
        config_from_mapping({"prediction": {"horizn": 2}})


def test_to_dict_is_nested_mapping():
    assert UPDAPFConfig().to_dict()["kalman"]["initial_velocity_std"] == 0.5
