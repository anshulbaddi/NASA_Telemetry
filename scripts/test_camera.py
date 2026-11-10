#!/usr/bin/env python3
"""
Quick camera verification and debug tool.

Usage examples:

  # Verify both cameras open and produce frames
  python3 scripts/test_camera.py

  # Show rectified side-by-side with horizontal guide lines
  python3 scripts/test_camera.py --debug-rectify

  # Test with pre-recorded images
  python3 scripts/test_camera.py --camera-type file \
      --left-dir data/test_stereo/left --right-dir data/test_stereo/right

  # Save debug frames to disk without display
  python3 scripts/test_camera.py --no-display --save-dir debug_output --frames 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../services/stereo-perception"))

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo camera test")
    p.add_argument("--camera-type",    default="usb", choices=["csi", "usb", "file"])
    p.add_argument("--left-id",        type=int, default=0)
    p.add_argument("--right-id",       type=int, default=1)
    p.add_argument("--width",          type=int, default=640)
    p.add_argument("--height",         type=int, default=480)
    p.add_argument("--fps",            type=int, default=30)
    p.add_argument("--left-dir",       default="data/test_stereo/left")
    p.add_argument("--right-dir",      default="data/test_stereo/right")
    p.add_argument("--calibration",    default="config/stereo_calibration.yaml")
    p.add_argument("--debug-rectify",  action="store_true", help="Show rectified images")
    p.add_argument("--no-display",     action="store_true", help="Headless mode")
    p.add_argument("--save-dir",       default=None)
    p.add_argument("--frames",         type=int, default=0, help="0 = run forever")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import config as _cfg_module
    # Override config with command-line args
    import types
    cfg = types.SimpleNamespace(
        CAMERA_TYPE     = args.camera_type,
        LEFT_CAMERA_ID  = args.left_id,
        RIGHT_CAMERA_ID = args.right_id,
        CAMERA_WIDTH    = args.width,
        CAMERA_HEIGHT   = args.height,
        CAMERA_FPS      = args.fps,
        LEFT_IMAGE_DIR  = args.left_dir,
        RIGHT_IMAGE_DIR = args.right_dir,
        LOOP_FILE_SOURCE = True,
    )

    from camera import create_camera_source
    source = create_camera_source(cfg)
    source.open()

    rectifier = None
    if args.debug_rectify:
        from calibration import load_calibration, make_dummy_calibration
        from rectification import StereoRectifier
        calib = load_calibration(args.calibration)
        if calib is None:
            print(f"WARNING: No calibration found at {args.calibration!r}; using dummy calibration.")
            calib = make_dummy_calibration(args.width, args.height)
        rectifier = StereoRectifier(calib)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print(f"Camera type: {args.camera_type}")
    print(f"Press Q to quit.\n")

    frame_count = 0
    t_start = time.time()
    try:
        while True:
            pair = source.read_pair()
            if pair is None:
                print("Camera read failure")
                time.sleep(0.1)
                continue

            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            left, right = pair.left, pair.right

            if rectifier:
                left_rect, right_rect = rectifier.rectify(left, right)
                canvas = rectifier.debug_annotate(left_rect, right_rect)
                label = f"Rectified  FPS={fps:.1f}  frame={frame_count}"
            else:
                canvas = np.hstack([left, right])
                label = f"Raw stereo  FPS={fps:.1f}  frame={frame_count}"

            cv2.putText(canvas, label, (10, canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if args.save_dir and frame_count % 5 == 0:
                path = os.path.join(args.save_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(path, canvas)

            if not args.no_display:
                cv2.imshow("Stereo Camera Test", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if frame_count % 30 == 0:
                    print(f"  frame={frame_count}  FPS={fps:.1f}")

            if args.frames > 0 and frame_count >= args.frames:
                break

    finally:
        source.close()
        cv2.destroyAllWindows()

    total = time.time() - t_start
    print(f"\nTest complete: {frame_count} frames in {total:.1f}s  avg FPS={frame_count/total:.1f}")


if __name__ == "__main__":
    main()
