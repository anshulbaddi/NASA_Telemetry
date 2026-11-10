#!/usr/bin/env python3
"""
Depth pipeline test — exercises calibration → rectification → disparity → depth → obstacle_map.

Works without a physical camera by reading pre-recorded stereo images from:
  data/test_stereo/left/
  data/test_stereo/right/

Or by generating a synthetic stereo pair for pure-algorithm testing.

Usage:
  # Test with file images
  python3 scripts/test_depth.py --camera-type file

  # Benchmark: measure fps on N frames
  python3 scripts/test_depth.py --camera-type file --benchmark --frames 100

  # Save visualisation (depth colormap + obstacle overlay)
  python3 scripts/test_depth.py --camera-type file --save-dir debug_output

  # Generate a synthetic scene and test
  python3 scripts/test_depth.py --synthetic
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../services/stereo-perception"))

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Depth pipeline test")
    p.add_argument("--camera-type",  default="file", choices=["file", "usb", "csi"])
    p.add_argument("--left-dir",     default="data/test_stereo/left")
    p.add_argument("--right-dir",    default="data/test_stereo/right")
    p.add_argument("--calibration",  default="config/stereo_calibration.yaml")
    p.add_argument("--save-dir",     default=None)
    p.add_argument("--frames",       type=int, default=10)
    p.add_argument("--benchmark",    action="store_true")
    p.add_argument("--no-display",   action="store_true")
    p.add_argument("--synthetic",    action="store_true", help="Use synthetic stereo test scene")
    return p.parse_args()


def make_synthetic_pair(width: int = 640, height: int = 480) -> tuple:
    """Create a synthetic stereo pair with a known foreground object."""
    left  = np.zeros((height, width, 3), dtype=np.uint8)
    right = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient
    for x in range(width):
        col = int(x / width * 128) + 64
        left[:, x]  = [col, col, col]
        right[:, x] = [col, col, col]

    # Foreground "wall" at disparity ~20 pixels (simulated obstacle)
    cv2.rectangle(left,  (220, 100), (420, 380), (180, 100, 60), -1)
    cv2.rectangle(right, (200, 100), (400, 380), (180, 100, 60), -1)  # shifted left by 20px

    # Close object at disparity ~40 pixels
    cv2.circle(left,  (320, 240), 40, (60, 180, 100), -1)
    cv2.circle(right, (280, 240), 40, (60, 180, 100), -1)  # shifted left by 40px

    return left, right


def colorize_depth(depth: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """Convert float depth map to BGR colour image for visualisation."""
    valid = np.isfinite(depth)
    norm  = np.zeros_like(depth)
    norm[valid] = depth[valid] / max_depth
    norm  = np.clip(norm, 0, 1)
    grey  = (norm * 255).astype(np.uint8)
    # Use a perceptual colormap: closer = red, further = blue
    colour = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
    colour[~valid] = 0  # black for invalid pixels
    return colour


def main() -> None:
    args = parse_args()

    from calibration import load_calibration, make_dummy_calibration
    from rectification import StereoRectifier
    from disparity import DisparityComputer
    from depth import DepthEstimator
    from obstacle_map import ObstacleAnalyzer

    calib = load_calibration(args.calibration)
    if calib is None:
        print(f"No calibration file at {args.calibration!r}; using dummy calibration.")
        calib = make_dummy_calibration(640, 480)

    # Build pipeline config namespace
    cfg = types.SimpleNamespace(
        STEREO_ALGORITHM  = "sgbm",
        MIN_DISPARITY     = 0,
        NUM_DISPARITIES   = 64,
        BLOCK_SIZE        = 7,
        UNIQUENESS_RATIO  = 10,
        SPECKLE_WIN_SIZE  = 100,
        SPECKLE_RANGE     = 32,
        DISP12_MAX_DIFF   = 1,
        PRE_FILTER_CAP    = 63,
        P1_MULT           = 8,
        P2_MULT           = 32,
        USE_GPU           = False,
        USE_WLS_FILTER    = False,
        MIN_DEPTH_M       = 0.25,
        MAX_DEPTH_M       = 10.0,
        LEFT_REGION_FRAC   = 0.33,
        CENTER_REGION_FRAC = 0.34,
        OBSTACLE_PERCENTILE = 10.0,
        DEPTH_ROI_TOP_FRAC  = 0.25,
        DEPTH_ROI_BOTTOM_FRAC = 0.75,
    )

    rectifier  = StereoRectifier(calib)
    disparity  = DisparityComputer(cfg)
    depth_est  = DepthEstimator(calib, cfg.MIN_DEPTH_M, cfg.MAX_DEPTH_M)
    obstacle   = ObstacleAnalyzer(cfg)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.synthetic:
        left_imgs  = [make_synthetic_pair()[0]] * max(args.frames, 1)
        right_imgs = [make_synthetic_pair()[1]] * max(args.frames, 1)
    elif args.camera_type == "file":
        left_files  = sorted(f for f in os.listdir(args.left_dir)  if f.endswith((".png",".jpg")))
        right_files = sorted(f for f in os.listdir(args.right_dir) if f.endswith((".png",".jpg")))
        n = min(len(left_files), len(right_files), args.frames or 999)
        left_imgs  = [cv2.imread(os.path.join(args.left_dir,  left_files[i]))  for i in range(n)]
        right_imgs = [cv2.imread(os.path.join(args.right_dir, right_files[i])) for i in range(n)]
    else:
        print("For USB/CSI, use test_camera.py directly.")
        sys.exit(1)

    latencies = []
    for i, (left, right) in enumerate(zip(left_imgs, right_imgs)):
        if left is None or right is None:
            print(f"Skipping pair {i}: load failure")
            continue

        t0 = time.perf_counter()
        lr, rr    = rectifier.rectify(left, right)
        disp      = disparity.compute(lr, rr)
        dr        = depth_est.compute(disp)
        omap      = obstacle.analyze(dr)
        elapsed   = (time.perf_counter() - t0) * 1000

        latencies.append(elapsed)

        print(
            f"Frame {i:4d}  latency={elapsed:6.1f}ms  valid={dr.valid_fraction*100:5.1f}%"
            f"  L={omap.left.nearest_m:5.2f}m  C={omap.center.nearest_m:5.2f}m"
            f"  R={omap.right.nearest_m:5.2f}m  → {omap.recommended_direction()}"
        )

        if args.save_dir:
            depth_vis = colorize_depth(dr.depth_map, cfg.MAX_DEPTH_M)
            disp_vis  = cv2.normalize(disp.astype(np.float32), None, 0, 255,
                                      cv2.NORM_MINMAX, cv2.CV_8U)
            disp_col  = cv2.applyColorMap(disp_vis, cv2.COLORMAP_BONE)
            rect_vis  = rectifier.debug_annotate(lr, rr)

            # Overlay obstacle labels on depth image
            h, w = depth_vis.shape[:2]
            for region, label in [
                (omap.left,   f"L:{omap.left.nearest_m:.1f}m"),
                (omap.center, f"C:{omap.center.nearest_m:.1f}m"),
                (omap.right,  f"R:{omap.right.nearest_m:.1f}m"),
            ]:
                pass  # labels added to canvas below

            cmd = omap.recommended_direction()
            cv2.putText(depth_vis, cmd, (w//2 - 60, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if cmd == "FORWARD" else (0, 0, 255), 2)

            top = np.hstack([rect_vis[:, :w], rect_vis[:, w:]])
            bot = np.hstack([disp_col, depth_vis])
            canvas = np.vstack([top, bot])
            cv2.imwrite(os.path.join(args.save_dir, f"depth_{i:06d}.jpg"), canvas)

        if not args.no_display and not args.benchmark:
            if args.save_dir:
                cv2.imshow("Depth test", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()

    if latencies:
        print(f"\n── Summary ──────────────────────────────────────")
        print(f"  Frames processed: {len(latencies)}")
        print(f"  Mean latency:     {sum(latencies)/len(latencies):.1f} ms")
        print(f"  Min latency:      {min(latencies):.1f} ms")
        print(f"  Max latency:      {max(latencies):.1f} ms")
        print(f"  Mean FPS:         {1000.0 / (sum(latencies)/len(latencies)):.1f}")


if __name__ == "__main__":
    main()
