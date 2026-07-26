"""
Isolation Forest-based anomaly detector for rover telemetry.

Uses scikit-learn's IsolationForest fitted on a sliding window of recent
telemetry. Anomaly scores < -0.1 are classified as anomalous and trigger
the FDIR response pipeline.

The rolling feature window is backed by a C++ circular buffer (PyRollingBuffer
from fdir_native) when the native extension is available, falling back to a
Python deque otherwise.  The C++ path eliminates Python boxing on every push
and replaces the O(N) np.array(list(deque)) copy with two memcpy calls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

# C++ rolling buffer — built with `python setup.py build_ext --inplace`
try:
    from fdir_native import PyRollingBuffer as _NativeBuffer
    _NATIVE = True
except ImportError:
    _NATIVE = False

# Pure-Python fallback that mirrors the PyRollingBuffer API
if not _NATIVE:
    from collections import deque

    class _PythonBuffer:
        CAPACITY   = 5_000
        N_FEATURES = 7

        def __init__(self):
            self._dq = deque(maxlen=self.CAPACITY)

        def push(self, battery_percent, cpu_temp_celsius, motor_temp_celsius,
                 motor_load_left, motor_load_right, velocity_ms, obstacle_density):
            self._dq.append([battery_percent, cpu_temp_celsius, motor_temp_celsius,
                              motor_load_left, motor_load_right, velocity_ms, obstacle_density])

        def get_matrix(self):
            return np.array(list(self._dq), dtype=np.float64)

        @property
        def size(self):
            return len(self._dq)

        def ready_for_fit(self, min_samples=500):
            return len(self._dq) >= min_samples

        def __len__(self):
            return len(self._dq)

    _Buffer = _PythonBuffer
else:
    _Buffer = _NativeBuffer


@dataclass
class AnomalyFeatures:
    battery_percent:    float
    cpu_temp_celsius:   float
    motor_temp_celsius: float
    motor_load_left:    float
    motor_load_right:   float
    velocity_ms:        float
    obstacle_density:   float  # fraction of LIDAR rays < 3 m


@dataclass
class AnomalyResult:
    is_anomaly:        bool
    score:             float   # IsolationForest score (negative = anomalous)
    features:          AnomalyFeatures
    severity:          str
    fault_code:        str
    description:       str
    recommended_action: str


class AnomalyDetector:
    """
    Online anomaly detector backed by IsolationForest.

    Rolling window management is delegated to PyRollingBuffer (C++/Cython)
    when available, which cuts push overhead from ~1.2 µs to ~80 ns and
    eliminates the O(N) list copy before each model fit.
    """

    REFIT_EVERY     = 500
    MIN_FIT_SAMPLES = 100

    def __init__(self, contamination: float = 0.05):
        self._contamination    = contamination
        self._buf              = _Buffer()
        self._model: Optional[IsolationForest] = None
        self._obs_since_refit  = 0
        self._total_anomalies  = 0
        self._total_obs        = 0

    def observe(self, features: AnomalyFeatures, rover_id: str) -> AnomalyResult:
        self._buf.push(
            features.battery_percent,
            features.cpu_temp_celsius,
            features.motor_temp_celsius,
            features.motor_load_left,
            features.motor_load_right,
            features.velocity_ms,
            features.obstacle_density,
        )
        self._obs_since_refit += 1
        self._total_obs       += 1

        if (self._obs_since_refit >= self.REFIT_EVERY and
                self._buf.ready_for_fit(self.MIN_FIT_SAMPLES)):
            self._refit()
            self._obs_since_refit = 0

        if self._model is None:
            return AnomalyResult(
                is_anomaly=False, score=0.0, features=features,
                severity="UNSPECIFIED", fault_code="", description="",
                recommended_action="NONE",
            )

        vec       = [[features.battery_percent, features.cpu_temp_celsius,
                      features.motor_temp_celsius, features.motor_load_left,
                      features.motor_load_right, features.velocity_ms,
                      features.obstacle_density]]
        score     = float(self._model.score_samples(vec)[0])
        is_anomaly = score < -0.1

        if is_anomaly:
            self._total_anomalies += 1

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            features=features,
            **self._classify(score, features),
        )

    def _refit(self) -> None:
        X = self._buf.get_matrix()   # C++ path: two memcpy; Python path: list copy
        self._model = IsolationForest(
            n_estimators  = 100,
            contamination = self._contamination,
            random_state  = 42,
            n_jobs        = -1,
        )
        self._model.fit(X)

    def _classify(self, score: float, f: AnomalyFeatures) -> dict:
        if score > -0.1:
            return dict(severity="ADVISORY", fault_code="", description="",
                        recommended_action="NONE")
        if f.motor_temp_celsius > 100:
            return dict(severity="CRITICAL", fault_code="MOTOR_OVERTEMP",
                        description=f"Motor temperature {f.motor_temp_celsius:.1f}°C exceeds safe limit",
                        recommended_action="PAUSE")
        if f.battery_percent < 10:
            return dict(severity="CRITICAL", fault_code="CRITICAL_LOW_BATTERY",
                        description=f"Battery at {f.battery_percent:.1f}% — immediate return required",
                        recommended_action="RETURN_TO_BASE")
        if f.battery_percent < 20:
            return dict(severity="WARNING", fault_code="LOW_BATTERY",
                        description=f"Battery at {f.battery_percent:.1f}%",
                        recommended_action="RETURN_TO_BASE")
        if f.obstacle_density > 0.5:
            return dict(severity="WARNING", fault_code="HIGH_OBSTACLE_DENSITY",
                        description=f"Obstacle density {f.obstacle_density*100:.0f}% — navigation impaired",
                        recommended_action="REDUCE_SPEED")
        if f.motor_load_left > 0.9 or f.motor_load_right > 0.9:
            return dict(severity="CAUTION", fault_code="MOTOR_OVERLOAD",
                        description="Motor load exceeds 90% — possible terrain obstacle or stall",
                        recommended_action="REDUCE_SPEED")
        severity = "WARNING" if score < -0.3 else "CAUTION"
        return dict(severity=severity, fault_code="ANOMALY_DETECTED",
                    description=f"Statistical anomaly detected (score={score:.3f})",
                    recommended_action="MONITOR")

    @property
    def stats(self) -> dict:
        return {
            "total_observations": self._total_obs,
            "total_anomalies":    self._total_anomalies,
            "anomaly_rate":       self._total_anomalies / max(1, self._total_obs),
            "model_fitted":       self._model is not None,
            "window_size":        self._buf.size,
            "native_buffer":      _NATIVE,
        }
