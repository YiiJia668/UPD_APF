"""Position-only Gaussian measurements with an explicit independent RNG."""
from collections.abc import Sequence
import numpy as np

from upd_apf.data_types import ObstacleMeasurement, _array
from upd_apf.utils.covariance import validate_psd
from .obstacle import ObstacleTruthSnapshot


class PositionSensor:
    def __init__(self, measurement_covariance, *, rng=None, seed=None):
        covariance = _array(measurement_covariance, (3, 3), "measurement_covariance")
        if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10):
            raise ValueError("measurement covariance must be symmetric")
        covariance = validate_psd(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        self._factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        self._covariance = _array(covariance, (3, 3), "measurement_covariance")
        if rng is not None and seed is not None:
            raise ValueError("provide rng or seed, not both")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if np.any(covariance) and rng is None and seed is None:
            raise ValueError("noisy sensor requires explicit rng or seed")
        self._rng = rng if rng is not None else (np.random.default_rng(seed) if seed is not None else None)

    @property
    def measurement_covariance(self):
        return self._covariance.copy()

    def measure(self, truths: Sequence[ObstacleTruthSnapshot], timestamp: float):
        if not np.isscalar(timestamp) or not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        measurements = []
        for truth in truths:
            position = truth.position
            if np.any(self._covariance):
                position = position + self._factor @ self._rng.standard_normal(3)
            measurements.append(ObstacleMeasurement(truth.metadata.obstacle_id, float(timestamp), position))
        return tuple(measurements)
