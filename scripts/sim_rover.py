#!/usr/bin/env python3
"""
Rover simulation — no hardware, no camera, no cv2.

Generates synthetic OAK-D Lite depth data and drives the full pipeline:
  synthetic disparity → DepthEstimator → ObstacleAnalyzer → SafetyMonitor
  → prints what the motor controller would receive

Plays a sequence of real-world scenarios automatically.
Each scenario runs for several frames so you can watch the rover react.

Usage:
    python3 scripts/sim_rover.py              # full auto mode
    python3 scripts/sim_rover.py --fps 5      # slower
    python3 scripts/sim_rover.py --scenario wall_approach
    python3 scripts/sim_rover.py --list       # show available scenarios
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import math
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../services/stereo-perception"))

from calibration  import make_dummy_calibration
from depth        import DepthEstimator
from obstacle_map import ObstacleAnalyzer
from safety       import SafetyMonitor

# ── OAK-D Lite calibration (7.5 cm baseline, 640×400) ────────────────────────
CALIB   = make_dummy_calibration(640, 400, baseline_m=0.075)
FOCAL   = CALIB.focal_length    # ~1000 px
BASE    = 0.075                  # m

DEPTH_EST = DepthEstimator(CALIB, min_depth=0.25, max_depth=10.0)

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
ANALYZER = ObstacleAnalyzer(OBS_CFG)
MONITOR  = SafetyMonitor(SAFE_CFG)


# ── Synthetic depth generation ────────────────────────────────────────────────

def _disp(depth_m: float) -> int:
    if depth_m <= 0:
        return 0
    return int((FOCAL * BASE / depth_m) * 16)


def make_scene(left_m=8.0, center_m=8.0, right_m=8.0,
               h=400, w=640) -> np.ndarray:
    """Build a synthetic int16 disparity map with three depth zones."""
    disp = np.full((h, w), _disp(8.0), dtype=np.int16)
    c1 = int(w * 0.33)
    c2 = int(w * 0.67)
    disp[:, :c1]   = _disp(left_m)
    disp[:, c1:c2] = _disp(center_m)
    disp[:, c2:]   = _disp(right_m)
    return disp


def make_corridor(width_m=1.5, center_dist_m=8.0, h=400, w=640) -> np.ndarray:
    """
    Corridor: walls on both sides at width_m, open ahead.
    Simulates a narrow hallway or doorway.
    """
    disp = np.full((h, w), _disp(center_dist_m), dtype=np.int16)
    wall_cols = int(w * 0.25)
    disp[:, :wall_cols]    = _disp(width_m)
    disp[:, w-wall_cols:]  = _disp(width_m)
    return disp


def make_diagonal_wall(approach_m: float, bias: float = 0.5,
                       h=400, w=640) -> np.ndarray:
    """
    Wall at an angle — closer on the left, farther on the right.
    bias=0.5 → symmetric; bias>0.5 → closer on the left.
    """
    disp = np.full((h, w), _disp(8.0), dtype=np.int16)
    for col in range(w):
        frac = col / w
        dist = approach_m + (1 - bias) * frac * 3.0
        disp[:, col] = _disp(max(dist, 0.3))
    return disp


# ── Scenarios ─────────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return a + (b - a) * t


SCENARIOS = {
    # name: list of (label, frames, disparity_fn_args, vision_age)
    "open_field": {
        "title": "Open field — rover drives forward with no obstacles",
        "frames": [
            ("Clear ahead", 6, dict(left_m=8, center_m=8, right_m=8), 0.05),
        ],
    },
    "wall_approach": {
        "title": "Wall approaching straight ahead — full braking + emergency stop sequence",
        "frames": [
            ("8.0 m — cruise",        4, dict(left_m=8, center_m=8.0, right_m=8), 0.05),
            ("4.0 m — cruise",        3, dict(left_m=8, center_m=4.0, right_m=8), 0.05),
            ("2.5 m — warning zone",  4, dict(left_m=8, center_m=2.5, right_m=8), 0.05),
            ("1.4 m — stop zone",     4, dict(left_m=8, center_m=1.4, right_m=8), 0.05),
            ("0.8 m — still stopped", 3, dict(left_m=8, center_m=0.8, right_m=8), 0.05),
            ("0.3 m — EMERGENCY",     4, dict(left_m=8, center_m=0.3, right_m=8), 0.05),
        ],
    },
    "obstacle_right": {
        "title": "Obstacle drifts in from the right — rover steers left to avoid",
        "frames": [
            ("Clear",                  3, dict(left_m=8, center_m=8, right_m=8),   0.05),
            ("Obstacle 4m right",      3, dict(left_m=8, center_m=8, right_m=4),   0.05),
            ("Obstacle 2m right/ctr",  4, dict(left_m=8, center_m=1.2, right_m=1.2), 0.05),
            ("Clear path right",       4, dict(left_m=8, center_m=1.2, right_m=8), 0.05),
            ("All clear — resume",     3, dict(left_m=8, center_m=8, right_m=8),   0.05),
        ],
    },
    "obstacle_left": {
        "title": "Obstacle from the left — rover steers right",
        "frames": [
            ("Clear",                  3, dict(left_m=8,   center_m=8,   right_m=8), 0.05),
            ("Obstacle 3m left",       3, dict(left_m=3,   center_m=8,   right_m=8), 0.05),
            ("Obstacle blocks center", 4, dict(left_m=1.0, center_m=1.0, right_m=8), 0.05),
            ("Resume",                 3, dict(left_m=8,   center_m=8,   right_m=8), 0.05),
        ],
    },
    "narrow_corridor": {
        "title": "Narrow corridor (walls 1.2m on each side) — rover navigates through",
        "frames": [
            ("Corridor entry — walls 2.0m", 3, None, 0.05),
            ("Corridor mid — walls 1.2m",   4, None, 0.05),
            ("Corridor tight — walls 0.9m", 4, None, 0.05),
            ("Exit — clearing",             3, None, 0.05),
        ],
        "_corridor": [2.0, 1.2, 0.9, 8.0],
    },
    "camera_timeout": {
        "title": "Camera timeout — depth frames go stale, rover stops",
        "frames": [
            ("Normal operation",         4, dict(left_m=8, center_m=8, right_m=8), 0.05),
            ("Camera lag 0.5s",          3, dict(left_m=8, center_m=8, right_m=8), 0.5),
            ("Camera timeout 1.5s",      4, dict(left_m=8, center_m=8, right_m=8), 1.5),
            ("Camera recovers",          4, dict(left_m=8, center_m=8, right_m=8), 0.05),
        ],
    },
    "emergency_side": {
        "title": "Object appears unexpectedly close on the left — emergency stop",
        "frames": [
            ("Clear",               3, dict(left_m=8,   center_m=8, right_m=8), 0.05),
            ("Object 0.4m left",    4, dict(left_m=0.4, center_m=8, right_m=8), 0.05),
            ("Object gone",         3, dict(left_m=8,   center_m=8, right_m=8), 0.05),
        ],
    },
    "all_blocked": {
        "title": "Completely boxed in — all directions blocked",
        "frames": [
            ("Clear",           3, dict(left_m=8,   center_m=8,   right_m=8),   0.05),
            ("Closing in",      3, dict(left_m=2.5, center_m=2.5, right_m=2.5), 0.05),
            ("All blocked",     5, dict(left_m=1.0, center_m=1.0, right_m=1.0), 0.05),
            ("One side opens",  4, dict(left_m=1.0, center_m=1.0, right_m=6.0), 0.05),
        ],
    },
}


# ── ASCII rendering ───────────────────────────────────────────────────────────

DEPTH_CHARS = " ·:;+=xX#@█"   # dark (far) to bright (close)

def _depth_char(d_m: float) -> str:
    if not math.isfinite(d_m) or d_m > 10:
        return " "
    idx = int((1.0 - min(d_m, 10) / 10.0) * (len(DEPTH_CHARS) - 1))
    return DEPTH_CHARS[idx]


def _clearance_bar(d_m: float, width: int = 18) -> str:
    """Horizontal bar showing clearance with colour coding."""
    if not math.isfinite(d_m):
        return "  ??? m  [no depth data]"
    filled = max(0, min(width, int((min(d_m, 10) / 10.0) * width)))
    bar    = "█" * filled + "░" * (width - filled)
    if d_m < 0.5:
        tag = "🚨 EMERGENCY"
    elif d_m < 1.5:
        tag = "🛑 STOP ZONE"
    elif d_m < 3.0:
        tag = "⚠  WARNING"
    else:
        tag = "✓  CLEAR"
    return f"  {d_m:5.2f}m  [{bar}]  {tag}"


_CMD_ICONS = {
    "FORWARD":       "→  FORWARD",
    "TURN_LEFT":     "↰  TURN LEFT",
    "TURN_RIGHT":    "↱  TURN RIGHT",
    "STOP":          "■  STOP",
    "EMERGENCY_STOP":"🚨 EMERGENCY STOP",
}


def render_frame(scenario_title: str, phase_label: str,
                 omap, state, depth_result,
                 frame_no: int, fps: float,
                 vision_age: float) -> str:
    W = 70
    lines = []
    lines.append("═" * W)
    lines.append(f"  ROVER SIM  │  {scenario_title[:46]}")
    lines.append(f"  Frame {frame_no:04d}  │  {fps:.0f} Hz control loop  │  vision age {vision_age*1000:.0f} ms")
    lines.append("─" * W)
    lines.append(f"  Phase: {phase_label}")
    lines.append("")

    # Depth heat-map (condensed 3-zone view)
    l = omap.left.nearest_m
    c = omap.center.nearest_m
    r = omap.right.nearest_m

    lines.append("  CLEARANCES")
    lines.append(f"  Left  :" + _clearance_bar(l))
    lines.append(f"  Center:" + _clearance_bar(c))
    lines.append(f"  Right :" + _clearance_bar(r))
    lines.append(f"  Nearest obstacle: "
                 + (f"{omap.nearest_m:.2f} m" if math.isfinite(omap.nearest_m) else "none"))
    lines.append("")

    # Depth validity
    vf = depth_result.valid_fraction * 100
    lines.append(f"  Valid depth pixels: {vf:.0f}%  "
                 + ("▓" * int(vf / 5) + "░" * (20 - int(vf / 5))))
    lines.append("")

    # Motor command
    cmd  = state.commanded_direction
    icon = _CMD_ICONS.get(cmd, cmd)
    spd  = f"  speed: {state.speed_override:.0%}" if state.speed_override is not None else ""
    lines.append(f"  {'─'*50}")
    lines.append(f"  MOTOR COMMAND  ►  {icon}{spd}")
    lines.append(f"  Reason: {state.reason}")
    lines.append(f"  {'─'*50}")
    lines.append("")
    return "\n".join(lines)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_scenario(name: str, scenario: dict, frame_delay: float) -> None:
    title   = scenario["title"]
    corridor = scenario.get("_corridor")
    frame_no = 0
    t_start  = time.perf_counter()

    for i, (label, n_frames, scene_args, vision_age) in enumerate(scenario["frames"]):
        for _ in range(n_frames):
            # Build disparity map
            if corridor:
                w = corridor[i]
                disp = make_corridor(width_m=w)
            elif scene_args is None:
                disp = make_scene()
            else:
                disp = make_scene(**scene_args)

            # Run full pipeline
            depth_result = DEPTH_EST.compute(disp)
            omap         = ANALYZER.analyze(depth_result)
            desired      = "FORWARD"
            state        = MONITOR.evaluate(omap, vision_age_s=vision_age,
                                            desired_direction=desired)

            elapsed = time.perf_counter() - t_start
            fps_est = frame_no / elapsed if elapsed > 0 else 0

            # Clear screen (works in most terminals)
            print("\033[H\033[J", end="")
            print(render_frame(title, label, omap, state, depth_result,
                               frame_no, fps_est, vision_age))
            frame_no += 1
            time.sleep(frame_delay)

    # Pause at end of scenario
    time.sleep(0.8)


def run_all(frame_delay: float) -> None:
    for name, scenario in SCENARIOS.items():
        run_scenario(name, scenario, frame_delay)

    # Final summary
    print("\033[H\033[J", end="")
    print("═" * 70)
    print("  SIMULATION COMPLETE")
    print("═" * 70)
    print()
    print("  All rover scenarios passed through the full pipeline:")
    print()
    for name, sc in SCENARIOS.items():
        print(f"    ✓  {name:<22}  {sc['title'][:42]}")
    print()
    print("  The same code runs on the Jetson Nano with CAMERA_TYPE=oak.")
    print("  The only difference: OAK-D Lite depth maps replace synthetic ones.")
    print()
    print("  To run on the Jetson:")
    print("    1. python3 scripts/calibrate_stereo.py oak-calib \\")
    print("               --output config/stereo_calibration.yaml")
    print("    2. CAMERA_TYPE=oak python3 services/stereo-perception/main.py")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Rover simulation (no hardware)")
    p.add_argument("--fps",      type=float, default=8,
                   help="Simulation speed in frames/sec (default: 8)")
    p.add_argument("--scenario", default=None,
                   help="Run a single named scenario")
    p.add_argument("--list",     action="store_true",
                   help="List available scenarios")
    args = p.parse_args()

    if args.list:
        print("\nAvailable scenarios:\n")
        for name, sc in SCENARIOS.items():
            print(f"  {name:<22}  {sc['title']}")
        print()
        return

    delay = 1.0 / args.fps

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario '{args.scenario}'. Use --list to see options.")
            sys.exit(1)
        run_scenario(args.scenario, SCENARIOS[args.scenario], delay)
    else:
        run_all(delay)


if __name__ == "__main__":
    main()
