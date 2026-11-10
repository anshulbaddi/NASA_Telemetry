"""
Unit tests for the obstacle map module.

Tests do NOT require a camera or physical hardware.
"""
from __future__ import annotations

import math
import os
import sys
import types
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../../services/stereo-perception"))

from depth import DepthResult
from obstacle_map import ObstacleAnalyzer, ObstacleMap, _region_stats


def _make_config(**overrides) -> types.SimpleNamespace:
    defaults = dict(
        LEFT_REGION_FRAC        = 0.33,
        CENTER_REGION_FRAC      = 0.34,
        OBSTACLE_PERCENTILE     = 10.0,
        DEPTH_ROI_TOP_FRAC      = 0.25,
        DEPTH_ROI_BOTTOM_FRAC   = 0.75,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_depth_result(depth_map: np.ndarray) -> DepthResult:
    valid  = np.isfinite(depth_map)
    vfrac  = float(np.sum(valid)) / depth_map.size
    depths = depth_map[valid]
    return DepthResult(
        depth_map      = depth_map.astype(np.float32),
        valid_fraction = vfrac,
        min_depth_m    = float(np.min(depths)) if depths.size > 0 else math.nan,
        max_depth_m    = float(np.max(depths)) if depths.size > 0 else math.nan,
        mean_depth_m   = float(np.mean(depths)) if depths.size > 0 else math.nan,
    )


class TestRegionStats(unittest.TestCase):

    def test_all_nan_returns_nan_stats(self):
        arr   = np.full((50, 50), np.nan, dtype=np.float32)
        stats = _region_stats(arr, percentile=10.0)
        self.assertFalse(math.isfinite(stats.nearest_m))
        self.assertEqual(stats.valid_fraction, 0.0)

    def test_uniform_depth_returns_same_value(self):
        arr   = np.full((50, 50), 3.0, dtype=np.float32)
        stats = _region_stats(arr, percentile=10.0)
        self.assertAlmostEqual(stats.nearest_m, 3.0, places=2)
        self.assertAlmostEqual(stats.median_m,  3.0, places=2)
        self.assertAlmostEqual(stats.valid_fraction, 1.0, places=3)

    def test_percentile_picks_low_value(self):
        # 20% of pixels at 1.0 m, 80% at 5.0 m → 10th percentile is unambiguously 1.0 m
        arr = np.full((100, 10), 5.0, dtype=np.float32)
        arr[:20, :] = 1.0  # first 20 rows = 20% of pixels
        stats = _region_stats(arr, percentile=10.0)
        self.assertLessEqual(stats.nearest_m, 2.0)

    def test_few_valid_pixels_returns_nan(self):
        arr          = np.full((50, 50), np.nan, dtype=np.float32)
        arr[0, 0]    = 1.0
        arr[0, 1]    = 2.0
        arr[0, 2]    = 3.0
        # 3 valid pixels < threshold of 5
        stats = _region_stats(arr, percentile=10.0)
        self.assertFalse(math.isfinite(stats.nearest_m))


class TestObstacleAnalyzer(unittest.TestCase):

    def setUp(self):
        self.cfg      = _make_config()
        self.analyzer = ObstacleAnalyzer(self.cfg)

    def _run(self, depth_map: np.ndarray) -> ObstacleMap:
        return self.analyzer.analyze(_make_depth_result(depth_map))

    def test_open_scene_has_large_clearances(self):
        depth = np.full((480, 640), 8.0, dtype=np.float32)
        omap  = self._run(depth)
        for region in (omap.left, omap.center, omap.right):
            self.assertGreater(region.nearest_m, 5.0)

    def test_obstacle_in_center_detected(self):
        depth = np.full((480, 640), 8.0, dtype=np.float32)
        # Put a close obstacle in the center columns
        w = 640
        center_start = int(w * 0.33)
        center_end   = int(w * 0.67)
        depth[:, center_start:center_end] = 1.0
        omap = self._run(depth)
        self.assertLess(omap.center.nearest_m, 2.0)
        self.assertGreater(omap.left.nearest_m,  5.0)
        self.assertGreater(omap.right.nearest_m, 5.0)

    def test_all_invalid_depth_returns_nan_clearances(self):
        depth = np.full((480, 640), np.nan, dtype=np.float32)
        omap  = self._run(depth)
        self.assertFalse(math.isfinite(omap.center.nearest_m))
        self.assertFalse(math.isfinite(omap.nearest_m))

    def test_nearest_m_is_minimum_of_regions(self):
        depth    = np.full((480, 640), 8.0, dtype=np.float32)
        depth[:, :100] = 2.0  # left region obstacle
        omap = self._run(depth)
        self.assertAlmostEqual(omap.nearest_m, omap.left.nearest_m, places=1)

    def test_output_shape_independent_of_input(self):
        for h, w in [(240, 320), (480, 640), (600, 800)]:
            depth = np.full((h, w), 5.0, dtype=np.float32)
            omap  = self._run(depth)
            self.assertIsInstance(omap.center.nearest_m, float)


class TestObstacleMapDirection(unittest.TestCase):

    def _make_map(self, left_m, center_m, right_m) -> ObstacleMap:
        from obstacle_map import RegionStats
        def rs(d): return RegionStats(nearest_m=d, median_m=d, valid_fraction=1.0, n_pixels=100)
        return ObstacleMap(left=rs(left_m), center=rs(center_m), right=rs(right_m))

    def test_clear_ahead_is_forward(self):
        omap = self._make_map(8.0, 8.0, 8.0)
        self.assertEqual(omap.recommended_direction(stop_dist_m=1.5, warning_dist_m=3.0), "FORWARD")

    def test_center_blocked_no_sides_is_stop(self):
        omap = self._make_map(1.0, 1.0, 1.0)
        self.assertEqual(omap.recommended_direction(stop_dist_m=1.5, warning_dist_m=3.0), "STOP")

    def test_center_blocked_right_clear_turns_right(self):
        omap = self._make_map(1.0, 1.0, 5.0)
        direction = omap.recommended_direction(stop_dist_m=1.5, warning_dist_m=3.0)
        self.assertEqual(direction, "TURN_RIGHT")

    def test_center_blocked_left_clear_turns_left(self):
        omap = self._make_map(5.0, 1.0, 1.0)
        direction = omap.recommended_direction(stop_dist_m=1.5, warning_dist_m=3.0)
        self.assertEqual(direction, "TURN_LEFT")

    def test_emergency_stop_when_very_close(self):
        omap = self._make_map(0.1, 0.1, 0.1)
        direction = omap.recommended_direction(stop_dist_m=1.5, warning_dist_m=3.0)
        self.assertEqual(direction, "EMERGENCY_STOP")


if __name__ == "__main__":
    unittest.main()
