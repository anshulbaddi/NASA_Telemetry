"""
Unit tests for the depth estimation module.

Tests do NOT require a camera or physical hardware.
All inputs are synthetic numpy arrays.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Add stereo-perception to path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../../services/stereo-perception"))

from calibration import make_dummy_calibration
from depth import DepthEstimator


class TestDepthEstimator(unittest.TestCase):

    def setUp(self):
        self.calib  = make_dummy_calibration(width=640, height=480, baseline_m=0.12)
        self.est    = DepthEstimator(self.calib, min_depth=0.25, max_depth=10.0)

    # ── Basic correctness ─────────────────────────────────────────────────────

    def test_uniform_disparity_produces_finite_depth(self):
        """A uniform valid disparity field should produce finite depth."""
        H, W    = 100, 100
        # 30 px disparity × 16 = 480 raw units
        disp_px = 30.0
        disp_raw = np.full((H, W), int(disp_px * 16), dtype=np.int16)
        result   = self.est.compute(disp_raw)
        valid    = np.isfinite(result.depth_map)
        self.assertGreater(np.sum(valid), H * W * 0.5,
                           "Most pixels should be valid for uniform disparity")

    def test_zero_disparity_is_invalid(self):
        """Zero disparity should produce NaN depth (division by zero guard)."""
        disp_raw = np.zeros((50, 50), dtype=np.int16)
        result   = self.est.compute(disp_raw)
        self.assertEqual(result.valid_fraction, 0.0)
        self.assertTrue(np.all(~np.isfinite(result.depth_map)))

    def test_negative_disparity_is_invalid(self):
        """Negative disparity (OpenCV invalid marker) should be NaN."""
        disp_raw = np.full((50, 50), -16, dtype=np.int16)
        result   = self.est.compute(disp_raw)
        self.assertEqual(result.valid_fraction, 0.0)

    def test_depth_within_range(self):
        """All returned valid depths must be within [min_depth, max_depth]."""
        disp_raw = np.random.randint(16, 640, size=(100, 100), dtype=np.int16)
        result   = self.est.compute(disp_raw)
        valid    = result.depth_map[np.isfinite(result.depth_map)]
        if valid.size > 0:
            self.assertGreaterEqual(float(np.min(valid)), 0.25)
            self.assertLessEqual(float(np.max(valid)),    10.0)

    def test_depth_below_min_is_invalid(self):
        """Depths below MIN_DEPTH_M must be masked to NaN."""
        # Very large disparity → very small depth → should be rejected
        disp_raw = np.full((50, 50), 10_000, dtype=np.int16)
        result   = self.est.compute(disp_raw)
        # Either valid (if depth lands in range) or all NaN
        valid = result.depth_map[np.isfinite(result.depth_map)]
        if valid.size > 0:
            self.assertGreaterEqual(float(np.min(valid)), 0.25)

    def test_valid_fraction_range(self):
        """valid_fraction must always be between 0 and 1."""
        for _ in range(5):
            disp = np.random.randint(-16, 1024, size=(80, 80), dtype=np.int16)
            r    = self.est.compute(disp)
            self.assertGreaterEqual(r.valid_fraction, 0.0)
            self.assertLessEqual(r.valid_fraction, 1.0)

    def test_all_nan_if_no_valid_disparity(self):
        """A disparity image with all -1 should yield no valid depth."""
        disp_raw = np.full((60, 60), -1 * 16, dtype=np.int16)
        result   = self.est.compute(disp_raw)
        self.assertEqual(result.valid_fraction, 0.0)
        self.assertTrue(np.isnan(result.min_depth_m))
        self.assertTrue(np.isnan(result.max_depth_m))
        self.assertTrue(np.isnan(result.mean_depth_m))

    def test_output_dtype_is_float32(self):
        """Depth map must be float32."""
        disp_raw = np.ones((40, 40), dtype=np.int16) * 32
        result   = self.est.compute(disp_raw)
        self.assertEqual(result.depth_map.dtype, np.float32)

    def test_output_shape_matches_input(self):
        """Output shape must match input disparity shape."""
        H, W     = 123, 456
        disp_raw = np.ones((H, W), dtype=np.int16) * 64
        result   = self.est.compute(disp_raw)
        self.assertEqual(result.depth_map.shape, (H, W))


class TestCalibrationDummy(unittest.TestCase):
    """Sanity-check the dummy calibration values."""

    def test_baseline_positive(self):
        calib = make_dummy_calibration(baseline_m=0.12)
        self.assertGreater(calib.baseline_m, 0)

    def test_focal_length_positive(self):
        calib = make_dummy_calibration()
        self.assertGreater(calib.focal_length, 0)

    def test_q_matrix_shape(self):
        calib = make_dummy_calibration()
        self.assertEqual(calib.Q.shape, (4, 4))

    def test_k_matrices_shape(self):
        calib = make_dummy_calibration()
        self.assertEqual(calib.K_left.shape,  (3, 3))
        self.assertEqual(calib.K_right.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
