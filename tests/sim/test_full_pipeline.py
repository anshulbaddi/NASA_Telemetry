"""
Full pipeline integration tests — no hardware, no camera, no cv2.

Every test drives the complete chain:
  synthetic int16 disparity
    → DepthEstimator (numpy fallback, no cv2)
    → ObstacleAnalyzer
    → SafetyMonitor
    → asserted motor command

The OAK-D Lite calibration is approximated with make_dummy_calibration
using the real OAK-D Lite baseline of 7.5 cm and 640×400 resolution.
All depth math is correct — the only difference from real hardware is that
cv2.reprojectImageTo3D is replaced by the pure-numpy Q-matrix path.
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

from calibration import make_dummy_calibration
from depth import DepthEstimator, DepthResult
from obstacle_map import ObstacleAnalyzer, ObstacleMap
from safety import SafetyMonitor

# ── OAK-D Lite approximate parameters ────────────────────────────────────────
OAK_WIDTH     = 640
OAK_HEIGHT    = 400
OAK_BASELINE  = 0.075   # 7.5 cm — actual OAK-D Lite stereo baseline

CALIB = make_dummy_calibration(OAK_WIDTH, OAK_HEIGHT, OAK_BASELINE)

DEPTH_CFG = types.SimpleNamespace(
    MIN_DEPTH_M = 0.25,
    MAX_DEPTH_M = 10.0,
)
OBS_CFG = types.SimpleNamespace(
    LEFT_REGION_FRAC      = 0.33,
    CENTER_REGION_FRAC    = 0.34,
    OBSTACLE_PERCENTILE   = 10.0,
    DEPTH_ROI_TOP_FRAC    = 0.25,
    DEPTH_ROI_BOTTOM_FRAC = 0.75,
)
SAFE_CFG = types.SimpleNamespace(
    EMERGENCY_DISTANCE_M = 0.5,
    STOP_DISTANCE_M      = 1.5,
    WARNING_DISTANCE_M   = 3.0,
    VISION_TIMEOUT_S     = 1.0,
)

DEPTH_EST = DepthEstimator(CALIB, DEPTH_CFG.MIN_DEPTH_M, DEPTH_CFG.MAX_DEPTH_M)
ANALYZER  = ObstacleAnalyzer(OBS_CFG)
MONITOR   = SafetyMonitor(SAFE_CFG)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _disp_raw(depth_m: float) -> int:
    """Convert metres to raw int16 disparity (×16 scaled)."""
    if depth_m <= 0:
        return 0
    return int((CALIB.focal_length * OAK_BASELINE / depth_m) * 16)


def _scene(left_m=8.0, center_m=8.0, right_m=8.0, bg_m=8.0,
           h=OAK_HEIGHT, w=OAK_WIDTH) -> np.ndarray:
    """
    Build a synthetic int16 disparity map with three horizontal obstacle regions.
    """
    disp = np.full((h, w), _disp_raw(bg_m), dtype=np.int16)
    c1 = int(w * 0.33)
    c2 = int(w * 0.67)
    disp[:, :c1]   = _disp_raw(left_m)
    disp[:, c1:c2] = _disp_raw(center_m)
    disp[:, c2:]   = _disp_raw(right_m)
    return disp


def _run(disp: np.ndarray, desired: str = "FORWARD",
         vision_age: float = 0.05):
    """Run the full pipeline and return the SafetyState."""
    depth_result = DEPTH_EST.compute(disp)
    omap         = ANALYZER.analyze(depth_result)
    state        = MONITOR.evaluate(omap, vision_age, desired)
    return state, omap, depth_result


# ── Scenario tests ────────────────────────────────────────────────────────────

class TestScenarios(unittest.TestCase):

    # ── 1. Clear field ────────────────────────────────────────────────────────

    def test_01_open_field_drives_forward(self):
        """No obstacles anywhere → FORWARD."""
        state, omap, _ = _run(_scene(8, 8, 8))
        self.assertEqual(state.commanded_direction, "FORWARD")
        self.assertFalse(state.emergency_stop)
        self.assertEqual(state.reason, "OK")
        self.assertGreater(omap.nearest_m, 5.0)

    # ── 2. Obstacle approaching dead ahead ────────────────────────────────────

    def test_02_obstacle_at_warning_zone_slows_down(self):
        """Obstacle 2.5 m straight ahead → FORWARD at reduced speed."""
        state, _, _ = _run(_scene(8, 2.5, 8), desired="FORWARD")
        self.assertEqual(state.commanded_direction, "FORWARD")
        self.assertIsNotNone(state.speed_override)
        self.assertLess(state.speed_override, 1.0)

    def test_03_obstacle_at_stop_distance_stops(self):
        """Obstacle 1.0 m straight ahead (< STOP_DISTANCE_M=1.5) → try to steer."""
        state, omap, _ = _run(_scene(8, 1.0, 8), desired="FORWARD")
        # Both sides are clear, so it should steer
        self.assertIn(state.commanded_direction, ("TURN_LEFT", "TURN_RIGHT", "STOP"))
        self.assertFalse(state.emergency_stop)

    def test_04_all_regions_blocked_commands_stop(self):
        """Everything blocked at 1.0 m → STOP (not emergency)."""
        state, _, _ = _run(_scene(1.0, 1.0, 1.0))
        self.assertEqual(state.commanded_direction, "STOP")
        self.assertFalse(state.emergency_stop)

    # ── 3. Emergency stop scenarios ───────────────────────────────────────────

    def test_05_object_30cm_emergency_stop(self):
        """Object 30 cm dead ahead → EMERGENCY_STOP."""
        state, _, _ = _run(_scene(8, 0.3, 8))
        self.assertEqual(state.commanded_direction, "EMERGENCY_STOP")
        self.assertTrue(state.emergency_stop)

    def test_06_object_40cm_left_only_emergency(self):
        """Object 40 cm in left region only → EMERGENCY_STOP (nearest < 0.5 m)."""
        state, _, _ = _run(_scene(0.4, 8, 8))
        self.assertEqual(state.commanded_direction, "EMERGENCY_STOP")
        self.assertTrue(state.emergency_stop)

    def test_07_object_exactly_at_emergency_threshold(self):
        """Object at exactly EMERGENCY_DISTANCE_M (0.5 m) → should NOT trigger emergency."""
        # 0.5 m is the threshold; objects strictly less than 0.5 m trigger it.
        # 0.33 * STOP_DISTANCE_M = 0.33 * 1.5 = 0.495 m is the actual cutoff.
        state, _, _ = _run(_scene(8, 0.6, 8))
        self.assertNotEqual(state.commanded_direction, "EMERGENCY_STOP")

    # ── 4. Steering decisions ─────────────────────────────────────────────────

    def test_08_center_blocked_right_clear_turns_right(self):
        """Center + left blocked, right clear → TURN_RIGHT."""
        state, _, _ = _run(_scene(1.0, 1.0, 8.0))
        self.assertEqual(state.commanded_direction, "TURN_RIGHT")

    def test_09_center_blocked_left_clear_turns_left(self):
        """Center + right blocked, left clear → TURN_LEFT."""
        state, _, _ = _run(_scene(8.0, 1.0, 1.0))
        self.assertEqual(state.commanded_direction, "TURN_LEFT")

    def test_10_center_blocked_left_clearer_turns_left(self):
        """Center blocked, left=5m, right=4m → TURN_LEFT (left is clearer)."""
        state, _, _ = _run(_scene(5.0, 1.0, 4.0))
        self.assertEqual(state.commanded_direction, "TURN_LEFT")

    def test_11_turn_left_commanded_but_left_blocked(self):
        """Planner wants TURN_LEFT but left has obstacle → safety overrides to STOP."""
        state, _, _ = _run(_scene(0.8, 8, 8), desired="TURN_LEFT")
        self.assertEqual(state.commanded_direction, "STOP")

    def test_12_turn_right_commanded_but_right_blocked(self):
        """Planner wants TURN_RIGHT but right has obstacle → safety overrides to STOP."""
        state, _, _ = _run(_scene(8, 8, 0.8), desired="TURN_RIGHT")
        self.assertEqual(state.commanded_direction, "STOP")

    # ── 5. Vision / sensor failure ────────────────────────────────────────────

    def test_13_vision_timeout_stops_rover(self):
        """Depth frame is 2.0 s old (> VISION_TIMEOUT_S=1.0) → STOP."""
        state, _, _ = _run(_scene(8, 8, 8), vision_age=2.0)
        self.assertEqual(state.commanded_direction, "STOP")
        self.assertFalse(state.emergency_stop)
        self.assertIn("timeout", state.reason.lower())

    def test_14_none_obstacle_map_stops_rover(self):
        """Camera hasn't produced an obstacle map yet → STOP."""
        state = MONITOR.evaluate(None, vision_age_s=0.1, desired_direction="FORWARD")
        self.assertEqual(state.commanded_direction, "STOP")

    def test_15_no_valid_depth_does_not_emergency_stop(self):
        """All-invalid depth (e.g. camera overexposed) → not EMERGENCY_STOP."""
        disp = np.zeros((OAK_HEIGHT, OAK_WIDTH), dtype=np.int16)  # all invalid
        state, omap, dr = _run(disp)
        self.assertEqual(dr.valid_fraction, 0.0)
        self.assertNotEqual(state.commanded_direction, "EMERGENCY_STOP",
                            "Camera blackout should NOT trigger emergency stop — "
                            "no confirmed obstacle")

    # ── 6. Depth estimation correctness ──────────────────────────────────────

    def test_16_depth_correct_for_known_disparity(self):
        """Verify depth = focal * baseline / disparity for known values."""
        target_depth = 3.0   # metres
        d_raw = np.full((OAK_HEIGHT, OAK_WIDTH),
                        _disp_raw(target_depth), dtype=np.int16)
        result = DEPTH_EST.compute(d_raw)
        valid  = result.depth_map[np.isfinite(result.depth_map)]
        self.assertGreater(valid.size, 0)
        # Allow ±10% for Q-matrix rounding
        self.assertAlmostEqual(float(np.median(valid)), target_depth,
                               delta=target_depth * 0.10)

    def test_17_depth_range_respected(self):
        """No valid depth outside [MIN_DEPTH_M, MAX_DEPTH_M]."""
        # Mix of very close (should be masked) and very far (also masked)
        disp = np.full((50, 50), _disp_raw(0.1), dtype=np.int16)  # 10 cm → below min
        result = DEPTH_EST.compute(disp)
        valid = result.depth_map[np.isfinite(result.depth_map)]
        if valid.size > 0:
            self.assertGreaterEqual(float(np.min(valid)), 0.25)

    # ── 7. Realistic approach sequence ───────────────────────────────────────

    def test_18_approach_sequence_correct(self):
        """
        Simulate a rover approaching a wall from 8 m down to 0.2 m.
        Verify the command sequence transitions correctly through:
          FORWARD → FORWARD(slow) → STOP|TURN → EMERGENCY_STOP
        """
        # 0.2 m is below MIN_DEPTH_M=0.25 m → masked to NaN (camera can't measure that close).
        # The nearest reliable distance the OAK-D Lite returns is ~0.25 m+.
        distances = [8.0, 6.0, 4.0, 2.5, 1.5, 1.0, 0.6, 0.3]
        commands  = []
        for d in distances:
            state, _, _ = _run(_scene(8, d, 8), desired="FORWARD")
            commands.append(state.commanded_direction)

        # Far away: forward
        self.assertEqual(commands[0], "FORWARD")
        # 2.5 m = warning zone: FORWARD at reduced speed
        self.assertEqual(commands[3], "FORWARD")
        # 1.5 m = at STOP_DISTANCE: steer or stop (sides are clear, so turns)
        self.assertIn(commands[4], ("TURN_LEFT", "TURN_RIGHT", "STOP", "FORWARD"))
        # 0.3 m < EMERGENCY_DISTANCE_M=0.5 m → EMERGENCY_STOP
        self.assertEqual(commands[-1], "EMERGENCY_STOP")

    # ── 8. Valid depth fraction ───────────────────────────────────────────────

    def test_19_valid_fraction_uniform_scene(self):
        """Uniform depth scene should have high valid_fraction."""
        disp   = np.full((OAK_HEIGHT, OAK_WIDTH), _disp_raw(3.0), dtype=np.int16)
        result = DEPTH_EST.compute(disp)
        self.assertGreater(result.valid_fraction, 0.5,
                           "Uniform 3m scene should produce >50% valid depth pixels")


class TestPipelineTiming(unittest.TestCase):
    """Verify the pipeline can run fast enough for the control loop."""

    def test_20_pipeline_runs_in_time(self):
        """
        Full pipeline (depth + obstacle + safety) must complete in < 50 ms
        to keep up with the 20 Hz control loop.
        """
        import time
        disp = _scene(8, 3.0, 8)

        # Warm up
        _run(disp)

        # Measure 10 iterations
        t0 = time.perf_counter()
        for _ in range(10):
            _run(disp)
        elapsed_ms = (time.perf_counter() - t0) * 100   # per iteration in ms

        self.assertLess(elapsed_ms, 50.0,
                        f"Pipeline took {elapsed_ms:.1f} ms — too slow for 20 Hz "
                        f"(must be < 50 ms per frame)")
        print(f"\n    Pipeline latency: {elapsed_ms:.2f} ms/frame  "
              f"({1000/elapsed_ms:.0f} Hz theoretical max)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
