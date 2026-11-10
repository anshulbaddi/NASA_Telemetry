#!/usr/bin/env python3
"""
Stereo camera calibration utility.

Usage:
    # OAK-D Lite — read factory calibration directly from device flash (run once)
    python3 scripts/calibrate_stereo.py oak-calib \
        --output config/stereo_calibration.yaml \
        --width 640 --height 400

    # Step 1: Capture calibration images (USB/CSI cameras)
    python3 scripts/calibrate_stereo.py capture \
        --output data/calibration_images \
        --camera-type usb \
        --left-id 0 --right-id 1

    # Step 2: Compute calibration from captured images
    python3 scripts/calibrate_stereo.py calibrate \
        --images data/calibration_images \
        --output config/stereo_calibration.yaml \
        --board-cols 9 --board-rows 6 --square-mm 25

    # Combined (capture + calibrate in one step)
    python3 scripts/calibrate_stereo.py auto \
        --camera-type usb --left-id 0 --right-id 1 \
        --output config/stereo_calibration.yaml

Instructions for physical calibration:
  1. Print a chessboard calibration target (recommended: 9x6 inner corners).
     Boards are available at:
       https://calib.io/pages/camera-calibration-pattern-generator
     Use a flat, rigid surface (mount on a clipboard or foam board).
  2. Ensure the calibration area is well-lit with diffuse light (avoid reflections).
  3. Hold the board at different angles, distances, and positions.
     Aim for at least 20 poses covering all corners of the field of view.
  4. Keep the board stationary when capturing (motion blur degrades calibration).
  5. After calibration, check the reprojection error (< 1.0 px is good, < 0.5 px is excellent).
  6. Visually verify rectification by running:
       python3 scripts/test_camera.py --debug-rectify
     Horizontal grid lines should pass through the same features in left/right images.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate_stereo")

# Allow running from project root or scripts/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../services/stereo-perception"))

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo camera calibration")
    sub = p.add_subparsers(dest="command", required=True)

    # ── oak-calib ──
    oak = sub.add_parser("oak-calib",
                         help="Read OAK-D Lite factory calibration from device flash")
    oak.add_argument("--output", default="config/stereo_calibration.yaml")
    oak.add_argument("--width",  type=int, default=640)
    oak.add_argument("--height", type=int, default=400)

    # ── capture ──
    cap = sub.add_parser("capture", help="Capture calibration image pairs")
    cap.add_argument("--output",      default="data/calibration_images")
    cap.add_argument("--camera-type", default="usb", choices=["csi", "usb"])
    cap.add_argument("--left-id",     type=int, default=0)
    cap.add_argument("--right-id",    type=int, default=1)
    cap.add_argument("--width",       type=int, default=640)
    cap.add_argument("--height",      type=int, default=480)
    cap.add_argument("--fps",         type=int, default=30)
    cap.add_argument("--board-cols",  type=int, default=9, help="Inner corners per row")
    cap.add_argument("--board-rows",  type=int, default=6, help="Inner corners per column")

    # ── calibrate ──
    cal = sub.add_parser("calibrate", help="Compute calibration from saved images")
    cal.add_argument("--images",      required=True, help="Directory with left/ and right/ subdirs")
    cal.add_argument("--output",      default="config/stereo_calibration.yaml")
    cal.add_argument("--board-cols",  type=int, default=9)
    cal.add_argument("--board-rows",  type=int, default=6)
    cal.add_argument("--square-mm",   type=float, default=25.0, help="Square size in millimetres")

    # ── auto ──
    aut = sub.add_parser("auto", help="Capture + calibrate in one step")
    aut.add_argument("--output",      default="config/stereo_calibration.yaml")
    aut.add_argument("--camera-type", default="usb", choices=["csi", "usb"])
    aut.add_argument("--left-id",     type=int, default=0)
    aut.add_argument("--right-id",    type=int, default=1)
    aut.add_argument("--width",       type=int, default=640)
    aut.add_argument("--height",      type=int, default=480)
    aut.add_argument("--board-cols",  type=int, default=9)
    aut.add_argument("--board-rows",  type=int, default=6)
    aut.add_argument("--square-mm",   type=float, default=25.0)

    return p.parse_args()


def open_cameras(camera_type: str, left_id: int, right_id: int,
                 width: int, height: int, fps: int):
    """Open both cameras, return (cap_left, cap_right)."""
    if camera_type == "csi":
        from camera import _csi_pipeline
        cap_l = cv2.VideoCapture(_csi_pipeline(left_id,  width, height, fps), cv2.CAP_GSTREAMER)
        cap_r = cv2.VideoCapture(_csi_pipeline(right_id, width, height, fps), cv2.CAP_GSTREAMER)
    else:
        cap_l = cv2.VideoCapture(left_id,  cv2.CAP_V4L2)
        cap_r = cv2.VideoCapture(right_id, cv2.CAP_V4L2)
        for cap in (cap_l, cap_r):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap_l.isOpened():
        raise RuntimeError(f"Cannot open left camera id={left_id}")
    if not cap_r.isOpened():
        raise RuntimeError(f"Cannot open right camera id={right_id}")

    return cap_l, cap_r


def cmd_capture(args: argparse.Namespace) -> list:
    """Interactive capture loop. Returns list of (left_img, right_img) pairs."""
    left_dir  = os.path.join(args.output, "left")
    right_dir = os.path.join(args.output, "right")
    os.makedirs(left_dir,  exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    board = (args.board_cols, args.board_rows)
    cap_l, cap_r = open_cameras(
        args.camera_type, args.left_id, args.right_id,
        args.width, args.height, args.fps,
    )

    pairs = []
    frame_idx = 0
    print(f"\nCalibration capture started.")
    print(f"  Board: {args.board_cols}x{args.board_rows} inner corners")
    print(f"  Press SPACE to capture a pose, Q to finish.\n")

    try:
        while True:
            ret_l, img_l = cap_l.read()
            ret_r, img_r = cap_r.read()
            if not ret_l or not ret_r:
                print("Camera read failed")
                continue

            # Show detection preview
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            found_l, corners_l = cv2.findChessboardCorners(gray_l, board, None)
            found_r, corners_r = cv2.findChessboardCorners(gray_r, board, None)

            preview_l = img_l.copy()
            preview_r = img_r.copy()
            cv2.drawChessboardCorners(preview_l, board, corners_l, found_l)
            cv2.drawChessboardCorners(preview_r, board, corners_r, found_r)

            status = "DETECTED" if (found_l and found_r) else "NOT FOUND"
            colour = (0, 255, 0) if (found_l and found_r) else (0, 0, 255)
            for img in (preview_l, preview_r):
                cv2.putText(img, f"{status}  ({len(pairs)} captured)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

            canvas = np.hstack([preview_l, preview_r])
            cv2.imshow("Calibration — SPACE to capture, Q to finish", canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(" ") and found_l and found_r:
                fn = f"calib_{frame_idx:04d}.png"
                cv2.imwrite(os.path.join(left_dir,  fn), img_l)
                cv2.imwrite(os.path.join(right_dir, fn), img_r)
                pairs.append((img_l.copy(), img_r.copy()))
                frame_idx += 1
                print(f"  Captured pose {frame_idx}  (target: ~20)")
            elif key == ord("q"):
                break
    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()

    print(f"\nCaptured {len(pairs)} calibration poses in {args.output}")
    return pairs


def cmd_calibrate(image_dir: str, output: str,
                  board_cols: int, board_rows: int, square_mm: float) -> None:
    from calibration import compute_stereo_calibration, save_calibration

    left_dir  = os.path.join(image_dir, "left")
    right_dir = os.path.join(image_dir, "right")

    left_files  = sorted(f for f in os.listdir(left_dir)  if f.endswith(".png"))
    right_files = sorted(f for f in os.listdir(right_dir) if f.endswith(".png"))

    if not left_files:
        print(f"ERROR: No images found in {left_dir}")
        sys.exit(1)

    # Load matched pairs
    pairs_l = [cv2.imread(os.path.join(left_dir,  f)) for f in left_files]
    pairs_r = [cv2.imread(os.path.join(right_dir, f)) for f in right_files]
    valid   = [(l, r) for l, r in zip(pairs_l, pairs_r) if l is not None and r is not None]

    print(f"Computing calibration from {len(valid)} image pairs ...")
    calib = compute_stereo_calibration(
        left_images   = [p[0] for p in valid],
        right_images  = [p[1] for p in valid],
        board_size    = (board_cols, board_rows),
        square_size_m = square_mm / 1000.0,
    )
    save_calibration(calib, output)
    print(f"\nCalibration saved: {output}")
    print(f"  Baseline:      {calib.baseline_m*100:.1f} cm")
    print(f"  Focal length:  {calib.focal_length:.1f} px")
    print(f"  Image size:    {calib.image_size[0]}×{calib.image_size[1]}")
    print(f"\nNext step: run 'python3 scripts/test_camera.py --debug-rectify' to verify.")


def cmd_oak_calib(args: argparse.Namespace) -> None:
    """Read OAK-D Lite factory calibration from device flash and save to YAML."""
    from camera import OAKCameraSource
    from calibration import load_calibration_from_oak, save_calibration

    print(f"Opening OAK-D Lite at {args.width}x{args.height} ...")
    cam = OAKCameraSource(args.width, args.height, 10)
    cam.open()
    try:
        calib = load_calibration_from_oak(cam.dai_device, args.width, args.height)
        save_calibration(calib, args.output)
        print(f"\nOAK calibration saved: {args.output}")
        print(f"  Baseline:      {calib.baseline_m * 100:.2f} cm")
        print(f"  Focal length:  {calib.focal_length:.1f} px")
        print(f"  Image size:    {calib.image_size[0]}x{calib.image_size[1]}")
        print(f"\nNext: start the rover service with CAMERA_TYPE=oak")
    finally:
        cam.close()


def main() -> None:
    args = parse_args()

    if args.command == "oak-calib":
        cmd_oak_calib(args)

    elif args.command == "capture":
        cmd_capture(args)

    elif args.command == "calibrate":
        cmd_calibrate(args.images, args.output,
                      args.board_cols, args.board_rows, args.square_mm)

    elif args.command == "auto":
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Capture
            cap_args = argparse.Namespace(
                output=tmp, camera_type=args.camera_type,
                left_id=args.left_id, right_id=args.right_id,
                width=args.width, height=args.height, fps=30,
                board_cols=args.board_cols, board_rows=args.board_rows,
            )
            pairs = cmd_capture(cap_args)
            if len(pairs) < 5:
                print("Not enough poses captured (minimum 5). Try again.")
                sys.exit(1)
            # Calibrate
            import tempfile, cv2
            img_dir = tmp
            os.makedirs(os.path.join(img_dir, "left"),  exist_ok=True)
            os.makedirs(os.path.join(img_dir, "right"), exist_ok=True)
            for i, (l, r) in enumerate(pairs):
                fn = f"calib_{i:04d}.png"
                cv2.imwrite(os.path.join(img_dir, "left",  fn), l)
                cv2.imwrite(os.path.join(img_dir, "right", fn), r)
            cmd_calibrate(img_dir, args.output,
                          args.board_cols, args.board_rows, args.square_mm)


if __name__ == "__main__":
    main()
