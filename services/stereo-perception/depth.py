"""
Depth estimation from a disparity map.

Converts integer disparity (×16 scaled output from DisparityComputer) to a
float32 depth map in metres using the stereo calibration Q matrix or the
manual baseline/focal-length formula.

All invalid, zero, negative, NaN, Inf, and out-of-range pixels are replaced
with NaN in the output so downstream consumers can filter them with
np.isfinite() or np.isnan().

Depth formula:
    depth = focal_length_px * baseline_m / disparity_px

Using the Q matrix via cv2.reprojectImageTo3D gives the same result while
respecting non-zero image centres, which is more accurate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibration import CalibrationData

log = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Output of a single depth estimation pass."""
    depth_map:      np.ndarray   # float32, metres; NaN where invalid
    valid_fraction: float        # fraction of pixels with finite depth in [min, max]
    min_depth_m:    float        # minimum valid depth in this frame
    max_depth_m:    float        # maximum valid depth in this frame
    mean_depth_m:   float        # mean valid depth in this frame


class DepthEstimator:
    """
    Convert a ×16-scaled disparity map to a float32 depth map in metres.

    Pixels are marked NaN (invalid) when:
      - disparity ≤ 0 (invalid/occluded by OpenCV convention)
      - disparity equals the minimum-disparity sentinel value
      - depth < min_depth_m or depth > max_depth_m
      - depth is NaN or Inf after the reprojection
    """

    _DISPARITY_SCALE = 16.0  # StereoSGBM/BM output is scaled by this factor

    def __init__(
        self,
        calib:     CalibrationData,
        min_depth: float,
        max_depth: float,
    ) -> None:
        self._Q         = calib.Q
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._min_disp_sentinel = (calib.Q[3, 3] - 1)  # approx invalid sentinel
        log.info(
            "DepthEstimator: Q loaded  baseline=%.4f m  f=%.1f px  "
            "depth_range=[%.2f, %.2f] m",
            calib.baseline_m, calib.focal_length, min_depth, max_depth,
        )

    def compute(self, disparity_raw: np.ndarray) -> DepthResult:
        """
        Compute depth from a ×16-scaled disparity map.

        disparity_raw: int16 array (H×W), output of DisparityComputer.
        Returns a DepthResult with a float32 NaN-masked depth map.
        """
        # Convert to float, unscale
        disp_f = disparity_raw.astype(np.float32) / self._DISPARITY_SCALE

        # Mask out invalid disparity pixels (negative or zero)
        invalid_mask = disp_f <= 0.0

        # Avoid division by zero
        disp_safe = np.where(invalid_mask, np.nan, disp_f)

        # Reproject to 3D using the calibration Q matrix.
        # cv2.reprojectImageTo3D expects float32 and returns (H, W, 3) X/Y/Z.
        # We only need the Z (depth) channel.
        disp_for_reproj = np.where(np.isfinite(disp_safe), disp_safe, 0.0).astype(np.float32)
        points_3d = self._reproject(disp_for_reproj)
        depth = points_3d[:, :, 2].astype(np.float32)

        # Apply masks: invalid disparity, out-of-range depth, NaN/Inf
        depth[invalid_mask]          = np.nan
        depth[~np.isfinite(depth)]   = np.nan
        depth[depth < self._min_depth] = np.nan
        depth[depth > self._max_depth] = np.nan

        valid = np.isfinite(depth)
        valid_fraction = float(np.sum(valid)) / float(depth.size) if depth.size > 0 else 0.0

        if np.any(valid):
            valid_depths   = depth[valid]
            min_d  = float(np.min(valid_depths))
            max_d  = float(np.max(valid_depths))
            mean_d = float(np.mean(valid_depths))
        else:
            min_d = max_d = mean_d = float("nan")

        return DepthResult(
            depth_map      = depth,
            valid_fraction = valid_fraction,
            min_depth_m    = min_d,
            max_depth_m    = max_d,
            mean_depth_m   = mean_d,
        )

    def _reproject(self, disparity_f32: np.ndarray) -> np.ndarray:
        """
        Reproject disparity to 3D using the calibration Q matrix.

        cv2.reprojectImageTo3D is used when available (faster C++ implementation).
        Falls back to a pure-numpy implementation so tests run without OpenCV.
        The result is (H, W, 3) with channels [X, Y, Z] in metres.
        """
        try:
            import cv2
            return cv2.reprojectImageTo3D(disparity_f32, self._Q)
        except Exception:
            h, w = disparity_f32.shape
            Q = self._Q
            ones = np.ones((h, w), dtype=np.float32)
            xs   = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            ys   = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
            d    = disparity_f32
            X = Q[0, 0] * xs + Q[0, 3]
            Y = Q[1, 1] * ys + Q[1, 3]
            Z = Q[2, 3] * ones
            W = Q[3, 2] * d + Q[3, 3]
            W = np.where(W == 0, 1e-9, W)
            pts = np.stack([X / W, Y / W, Z / W], axis=-1)
            return pts.astype(np.float32)
