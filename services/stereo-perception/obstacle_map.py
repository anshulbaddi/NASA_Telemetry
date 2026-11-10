"""
Obstacle region analysis from a stereo depth map.

The depth image is divided into three horizontal regions (LEFT, CENTER, RIGHT)
and a vertical ROI band that excludes the sky and near-ground rows.

For each region we compute a robust "nearest reliable obstacle" distance using
a low percentile of the valid depth values.  A simple minimum would be
dominated by noise pixels; the Nth percentile suppresses isolated bad pixels
while still detecting real obstacles.

Output:
    ObstacleMap — left / center / right nearest depths + metadata

This is intentionally compact: only statistics, not the raw depth arrays,
are passed to the Kafka telemetry publisher and the path planner.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from depth import DepthResult

log = logging.getLogger(__name__)


@dataclass
class RegionStats:
    """Depth statistics for one horizontal image region."""
    nearest_m:      float   # robust nearest obstacle (low percentile of valid depths)
    median_m:       float   # median of valid depths
    valid_fraction: float   # fraction of pixels with valid depth in this region
    n_pixels:       int     # total pixels in region

    @property
    def has_obstacle(self) -> bool:
        """True if there are enough valid pixels to make a reliable estimate."""
        return self.valid_fraction > 0.05 and np.isfinite(self.nearest_m)


@dataclass
class ObstacleMap:
    """
    Regional obstacle representation derived from the stereo depth map.

    Three regions cover the full image width:
      left   — approximately ±30° left of forward
      center — approximately ±15° around forward heading
      right  — approximately ±30° right of forward

    These correspond roughly to:
      LEFT clearance   → rover can turn left if clear
      CENTER clearance → path ahead; triggers STOP if blocked
      RIGHT clearance  → rover can turn right if clear
    """
    left:       RegionStats
    center:     RegionStats
    right:      RegionStats
    timestamp:  float = 0.0

    @property
    def nearest_m(self) -> float:
        """Overall nearest valid obstacle across all regions."""
        candidates = [r.nearest_m for r in (self.left, self.center, self.right)
                      if np.isfinite(r.nearest_m)]
        return min(candidates) if candidates else float("nan")

    def recommended_direction(
        self,
        stop_dist_m:    float = 1.5,
        warning_dist_m: float = 3.0,
    ) -> str:
        """
        Simple reactive navigation decision based on regional clearances.

        Priority:
          1. Emergency stop if anything is critically close
          2. Stop if center is blocked
          3. Turn to the clearer side if center is warning-distance
          4. Forward if all clear
        """
        c_near = self.center.nearest_m  if np.isfinite(self.center.nearest_m)  else float("inf")
        l_near = self.left.nearest_m    if np.isfinite(self.left.nearest_m)    else float("inf")
        r_near = self.right.nearest_m   if np.isfinite(self.right.nearest_m)   else float("inf")

        if min(c_near, l_near, r_near) < stop_dist_m * 0.33:
            return "EMERGENCY_STOP"
        if c_near < stop_dist_m:
            # Center blocked — try to steer to a clear side
            left_clear  = l_near >= stop_dist_m
            right_clear = r_near >= stop_dist_m
            if left_clear and (not right_clear or l_near >= r_near):
                return "TURN_LEFT"
            if right_clear:
                return "TURN_RIGHT"
            return "STOP"
        if c_near < warning_dist_m:
            # Warning zone — steer to whichever side has more clearance
            if r_near >= l_near:
                return "TURN_RIGHT"
            return "TURN_LEFT"
        return "FORWARD"

    def to_telemetry_dict(self) -> dict:
        return {
            "timestamp":         self.timestamp,
            "nearest_obstacle_m": round(self.nearest_m, 3) if np.isfinite(self.nearest_m) else None,
            "left_clearance_m":  round(self.left.nearest_m, 3) if np.isfinite(self.left.nearest_m) else None,
            "center_clearance_m": round(self.center.nearest_m, 3) if np.isfinite(self.center.nearest_m) else None,
            "right_clearance_m": round(self.right.nearest_m, 3) if np.isfinite(self.right.nearest_m) else None,
            "obstacle_detected": np.isfinite(self.nearest_m) and self.nearest_m < 5.0,
        }


def _region_stats(depth_slice: np.ndarray, percentile: float) -> RegionStats:
    """
    Compute robust depth statistics for a depth array slice.

    depth_slice: 2D float32 array (H×W) with NaN for invalid pixels.
    percentile:  Low percentile to use as "nearest reliable obstacle" (e.g. 10.0).
    """
    valid = depth_slice[np.isfinite(depth_slice)]
    n_total = depth_slice.size
    n_valid = valid.size

    if n_valid < 5:
        return RegionStats(
            nearest_m      = float("nan"),
            median_m       = float("nan"),
            valid_fraction = n_valid / max(n_total, 1),
            n_pixels       = n_total,
        )

    nearest_m = float(np.percentile(valid, percentile))
    median_m  = float(np.median(valid))
    return RegionStats(
        nearest_m      = nearest_m,
        median_m       = median_m,
        valid_fraction = n_valid / n_total,
        n_pixels       = n_total,
    )


class ObstacleAnalyzer:
    """
    Converts a DepthResult into an ObstacleMap by analysing regional statistics.

    Region boundaries and ROI are configured through the config module.
    """

    def __init__(self, config) -> None:
        self._left_frac   = config.LEFT_REGION_FRAC
        self._center_frac = config.CENTER_REGION_FRAC
        self._percentile  = config.OBSTACLE_PERCENTILE
        self._roi_top     = config.DEPTH_ROI_TOP_FRAC
        self._roi_bot     = config.DEPTH_ROI_BOTTOM_FRAC

    def analyze(self, depth_result: DepthResult) -> ObstacleMap:
        depth = depth_result.depth_map
        h, w  = depth.shape

        # Vertical ROI — exclude sky above and near-ground below
        row_start = int(h * self._roi_top)
        row_end   = int(h * self._roi_bot)
        roi       = depth[row_start:row_end, :]

        # Horizontal region splits
        col_left_end   = int(w * self._left_frac)
        col_center_end = int(w * (self._left_frac + self._center_frac))

        left_slice   = roi[:, :col_left_end]
        center_slice = roi[:, col_left_end:col_center_end]
        right_slice  = roi[:, col_center_end:]

        left_stats   = _region_stats(left_slice,   self._percentile)
        center_stats = _region_stats(center_slice, self._percentile)
        right_stats  = _region_stats(right_slice,  self._percentile)

        return ObstacleMap(
            left      = left_stats,
            center    = center_stats,
            right     = right_stats,
            timestamp = time.time(),
        )
