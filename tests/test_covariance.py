import numpy as np
import pytest

from upd_apf.utils.covariance import is_psd, symmetrize, validate_psd


def test_symmetrize():
    assert np.allclose(symmetrize([[1, 2], [0, 1]]), [[1, 1], [1, 1]])


def test_validate_psd_accepts_semidefinite():
    assert np.allclose(validate_psd([[1, 0], [0, 0]]), np.diag([1, 0]))


def test_validate_psd_allows_tiny_negative_roundoff():
    assert validate_psd(np.diag([1, -1e-12]), tolerance=1e-10)[1, 1] == -1e-12


def test_validate_psd_rejects_negative_eigenvalue():
    with pytest.raises(ValueError, match="not PSD"):
        validate_psd(np.diag([1, -1e-3]))


def test_is_psd():
    assert is_psd(np.eye(3))
    assert not is_psd(np.diag([1, 1, -1]))


def test_covariance_must_be_square():
    with pytest.raises(ValueError, match="square"):
        symmetrize(np.ones((2, 3)))
