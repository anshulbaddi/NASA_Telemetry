"""
Safety monitor — independent last-resort check between perception and motors.

This layer runs AFTER the obstacle map is produced and BEFORE any motor command
is issued.  Its job is to enforce hard distance limits regardless of what the
path planner decided.

Rules (evaluated in order, first match wins):
  1. Vision timeout — if stereo depth is stale, command STOP
  2. Emergency stop — if anything is closer than EMERGENCY_DISTANCE_M
  3. Center blocked  — if center clearance < STOP_DISTANCE_M
  4. Warning zone    — if center clearance < WARNING_DISTANCE_M → reduce speed / steer
  5. Normal           — pass through the planner's direction

The safety monitor is intentionally simple.  It must be correct and fast.
It must never be bypassed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from obstacle_map import ObstacleMap

log = logging.getLogger(__name__)


@dataclass
class SafetyState:
    """Result of the safety evaluation for one control cycle."""
    commanded_direction: str    # FORWARD | TURN_LEFT | TURN_RIGHT | STOP | EMERGENCY_STOP
    emergency_stop:      bool
    reason:              str
    speed_override:      Optional[float] = None  # None = use default for this direction


class SafetyMonitor:
    """
    Evaluate the current perception state and return a safe motor command.

    Args:
        config: module with EMERGENCY_DISTANCE_M, STOP_DISTANCE_M,
                WARNING_DISTANCE_M, VISION_TIMEOUT_S
    """

    def __init__(self, config) -> None:
        self._emergency_dist = config.EMERGENCY_DISTANCE_M
        self._stop_dist      = config.STOP_DISTANCE_M
        self._warning_dist   = config.WARNING_DISTANCE_M
        self._vision_timeout = config.VISION_TIMEOUT_S

    def evaluate(
        self,
        obstacle_map: Optional[ObstacleMap],
        vision_age_s: float,
        desired_direction: str = "FORWARD",
    ) -> SafetyState:
        """
        Evaluate safety constraints and return the allowed direction.

        obstacle_map:      Latest obstacle map (may be None if camera hasn't produced one yet).
        vision_age_s:      Seconds since the last valid depth frame.
        desired_direction: What the path planner wants to do.
        """
        # ── 1. Vision timeout — stereo data is too old ────────────────────────
        if vision_age_s > self._vision_timeout:
            log.warning(
                "Safety: vision timeout (age=%.2f s > %.2f s) → STOP",
                vision_age_s, self._vision_timeout,
            )
            return SafetyState(
                commanded_direction = "STOP",
                emergency_stop      = False,
                reason              = f"Vision timeout: last depth {vision_age_s:.1f}s ago",
            )

        if obstacle_map is None:
            return SafetyState(
                commanded_direction = "STOP",
                emergency_stop      = False,
                reason              = "No obstacle map available yet",
            )

        nearest_m  = obstacle_map.nearest_m
        center_m   = obstacle_map.center.nearest_m
        left_m     = obstacle_map.left.nearest_m
        right_m    = obstacle_map.right.nearest_m

        def finite(v: float, fallback: float = 999.0) -> float:
            return v if math.isfinite(v) else fallback

        nearest_m  = finite(nearest_m)
        center_m   = finite(center_m)
        left_m     = finite(left_m)
        right_m    = finite(right_m)

        # ── 2. Emergency stop — anything critically close ─────────────────────
        if nearest_m < self._emergency_dist:
            log.warning(
                "Safety: EMERGENCY STOP nearest=%.2f m < emergency_dist=%.2f m",
                nearest_m, self._emergency_dist,
            )
            return SafetyState(
                commanded_direction = "EMERGENCY_STOP",
                emergency_stop      = True,
                reason = f"Obstacle at {nearest_m:.2f} m < emergency threshold {self._emergency_dist:.2f} m",
            )

        # ── 3. Center blocked ─────────────────────────────────────────────────
        if center_m < self._stop_dist:
            # Try to steer; if both sides also blocked, stop
            if left_m >= self._stop_dist and left_m >= right_m:
                return SafetyState(
                    commanded_direction = "TURN_LEFT",
                    emergency_stop      = False,
                    reason = f"Center blocked ({center_m:.2f} m); left clear ({left_m:.2f} m)",
                )
            if right_m >= self._stop_dist:
                return SafetyState(
                    commanded_direction = "TURN_RIGHT",
                    emergency_stop      = False,
                    reason = f"Center blocked ({center_m:.2f} m); right clear ({right_m:.2f} m)",
                )
            log.warning(
                "Safety: STOP all directions blocked  center=%.2f  left=%.2f  right=%.2f",
                center_m, left_m, right_m,
            )
            return SafetyState(
                commanded_direction = "STOP",
                emergency_stop      = False,
                reason = f"All directions blocked: C={center_m:.2f} L={left_m:.2f} R={right_m:.2f} m",
            )

        # ── 4. Warning zone — reduce speed or steer ───────────────────────────
        if center_m < self._warning_dist and desired_direction == "FORWARD":
            # Pass through at reduced speed — speed_override handled by control loop
            log.debug(
                "Safety: warning zone center=%.2f m < %.2f m → reduced speed",
                center_m, self._warning_dist,
            )
            return SafetyState(
                commanded_direction = "FORWARD",
                emergency_stop      = False,
                reason = f"Warning zone: center={center_m:.2f} m",
                speed_override      = 0.3,  # 30% of max speed
            )

        # ── 5. Normal operation — pass through desired direction ──────────────
        # Validate that the desired direction doesn't conflict with obstacles
        if desired_direction == "TURN_LEFT"  and left_m  < self._stop_dist:
            return SafetyState(
                commanded_direction = "STOP",
                emergency_stop      = False,
                reason = f"Cannot turn left: obstacle at {left_m:.2f} m",
            )
        if desired_direction == "TURN_RIGHT" and right_m < self._stop_dist:
            return SafetyState(
                commanded_direction = "STOP",
                emergency_stop      = False,
                reason = f"Cannot turn right: obstacle at {right_m:.2f} m",
            )

        return SafetyState(
            commanded_direction = desired_direction,
            emergency_stop      = False,
            reason              = "OK",
        )
