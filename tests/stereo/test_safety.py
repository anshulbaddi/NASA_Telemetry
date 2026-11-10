"""
Unit tests for the safety monitor.

Tests do NOT require hardware.
"""
from __future__ import annotations

import math
import os
import sys
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../../services/stereo-perception"))

from obstacle_map import ObstacleMap, RegionStats
from safety import SafetyMonitor, SafetyState


def _make_config(**overrides) -> types.SimpleNamespace:
    defaults = dict(
        EMERGENCY_DISTANCE_M = 0.5,
        STOP_DISTANCE_M      = 1.5,
        WARNING_DISTANCE_M   = 3.0,
        VISION_TIMEOUT_S     = 1.0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_omap(left_m: float, center_m: float, right_m: float) -> ObstacleMap:
    def rs(d): return RegionStats(nearest_m=d, median_m=d, valid_fraction=1.0, n_pixels=100)
    return ObstacleMap(left=rs(left_m), center=rs(center_m), right=rs(right_m))


class TestSafetyMonitor(unittest.TestCase):

    def setUp(self):
        self.cfg     = _make_config()
        self.monitor = SafetyMonitor(self.cfg)

    # ── Vision timeout ────────────────────────────────────────────────────────

    def test_vision_timeout_commands_stop(self):
        omap  = _make_omap(5.0, 5.0, 5.0)
        state = self.monitor.evaluate(omap, vision_age_s=2.0)
        self.assertEqual(state.commanded_direction, "STOP")
        self.assertFalse(state.emergency_stop)

    def test_fresh_vision_does_not_timeout(self):
        omap  = _make_omap(5.0, 5.0, 5.0)
        state = self.monitor.evaluate(omap, vision_age_s=0.1)
        self.assertNotEqual(state.commanded_direction, "STOP")

    def test_none_obstacle_map_commands_stop(self):
        state = self.monitor.evaluate(None, vision_age_s=0.1)
        self.assertEqual(state.commanded_direction, "STOP")

    # ── Emergency stop ────────────────────────────────────────────────────────

    def test_emergency_stop_when_very_close(self):
        omap  = _make_omap(0.3, 0.3, 0.3)
        state = self.monitor.evaluate(omap, vision_age_s=0.05)
        self.assertEqual(state.commanded_direction, "EMERGENCY_STOP")
        self.assertTrue(state.emergency_stop)

    def test_emergency_stop_only_one_region(self):
        omap  = _make_omap(5.0, 0.2, 5.0)
        state = self.monitor.evaluate(omap, vision_age_s=0.05)
        self.assertEqual(state.commanded_direction, "EMERGENCY_STOP")

    # ── Center blocked ────────────────────────────────────────────────────────

    def test_center_blocked_right_clear(self):
        omap  = _make_omap(1.0, 1.0, 5.0)
        state = self.monitor.evaluate(omap, vision_age_s=0.05)
        self.assertEqual(state.commanded_direction, "TURN_RIGHT")
        self.assertFalse(state.emergency_stop)

    def test_center_blocked_left_clear(self):
        omap  = _make_omap(5.0, 1.0, 1.0)
        state = self.monitor.evaluate(omap, vision_age_s=0.05)
        self.assertEqual(state.commanded_direction, "TURN_LEFT")

    def test_all_blocked_commands_stop(self):
        omap  = _make_omap(1.0, 1.0, 1.0)
        state = self.monitor.evaluate(omap, vision_age_s=0.05)
        self.assertEqual(state.commanded_direction, "STOP")
        self.assertFalse(state.emergency_stop)

    # ── Warning zone ──────────────────────────────────────────────────────────

    def test_warning_zone_reduces_speed(self):
        omap  = _make_omap(5.0, 2.5, 5.0)  # center in warning range
        state = self.monitor.evaluate(omap, vision_age_s=0.05, desired_direction="FORWARD")
        self.assertEqual(state.commanded_direction, "FORWARD")
        self.assertIsNotNone(state.speed_override)
        self.assertLess(state.speed_override, 1.0)  # reduced speed

    # ── Pass-through ──────────────────────────────────────────────────────────

    def test_clear_scene_passes_forward(self):
        omap  = _make_omap(8.0, 8.0, 8.0)
        state = self.monitor.evaluate(omap, vision_age_s=0.05, desired_direction="FORWARD")
        self.assertEqual(state.commanded_direction, "FORWARD")
        self.assertFalse(state.emergency_stop)
        self.assertEqual(state.reason, "OK")

    def test_turn_left_blocked_commands_stop(self):
        omap  = _make_omap(1.0, 8.0, 8.0)  # left blocked
        state = self.monitor.evaluate(omap, vision_age_s=0.05, desired_direction="TURN_LEFT")
        self.assertEqual(state.commanded_direction, "STOP")

    def test_turn_right_blocked_commands_stop(self):
        omap  = _make_omap(8.0, 8.0, 1.0)  # right blocked
        state = self.monitor.evaluate(omap, vision_age_s=0.05, desired_direction="TURN_RIGHT")
        self.assertEqual(state.commanded_direction, "STOP")

    # ── Nan clearances (camera returning no valid depth) ─────────────────────

    def test_nan_clearances_treat_as_safe(self):
        omap  = _make_omap(math.nan, math.nan, math.nan)
        state = self.monitor.evaluate(omap, vision_age_s=0.05)
        # With all-NaN clearances, should not emergency stop (not a confirmed obstacle)
        self.assertNotEqual(state.commanded_direction, "EMERGENCY_STOP")


class TestSafetyStateFields(unittest.TestCase):

    def test_reason_is_string(self):
        monitor = SafetyMonitor(_make_config())
        omap    = _make_omap(5.0, 5.0, 5.0)
        state   = monitor.evaluate(omap, vision_age_s=0.05)
        self.assertIsInstance(state.reason, str)

    def test_emergency_stop_is_bool(self):
        monitor = SafetyMonitor(_make_config())
        omap    = _make_omap(5.0, 5.0, 5.0)
        state   = monitor.evaluate(omap, vision_age_s=0.05)
        self.assertIsInstance(state.emergency_stop, bool)


if __name__ == "__main__":
    unittest.main()
