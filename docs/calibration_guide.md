# Stereo Camera Calibration Guide

This guide walks through calibrating a physical stereo camera for use with the rover's
stereo-perception pipeline. Calibration produces a `stereo_calibration.yaml` file that
encodes the intrinsic and extrinsic parameters needed to:

- Rectify each frame pair so corresponding points lie on the same horizontal scanline
- Compute a disparity map via StereoSGBM
- Reproject disparity to metric depth using the Q matrix

---

## Prerequisites

- Python 3.8+ with OpenCV: `pip install opencv-python PyYAML`
- A printed chessboard target (default: 9×6 inner corners, 25 mm squares)
- The stereo camera mounted and fixed in its final configuration

**Do not move the cameras relative to each other after calibration.** If the cameras
shift (e.g., after a hard bump), re-run calibration.

---

## 1. Print and prepare the calibration target

Print `docs/chessboard_9x6.pdf` (or any chessboard with known square size) on matte
paper. Mount it flat on a rigid backing — warping the target introduces error.

The `calibrate_stereo.py` script defaults to a **9 × 6 inner-corner** board with
**25 mm** squares. Override with `--board-cols`, `--board-rows`, `--square-size-mm`.

---

## 2. Capture calibration images

Connect both cameras and run the capture tool:

```bash
python3 scripts/calibrate_stereo.py capture \
    --camera-type usb \
    --left-index 0 \
    --right-index 1 \
    --output-dir data/calib_images \
    --board-cols 9 \
    --board-rows 6
```

**Tips for good calibration:**

- Capture **25–40 image pairs** from varied positions and orientations
- Cover all corners of the frame — tilt, rotate, and move the board
- Include poses where the board fills most of the frame
- Keep the board at distances from **30 cm to 150 cm**
- Ensure consistent, diffuse lighting — avoid reflections and shadows on the board
- Press **`s`** to save a frame pair, **`q`** to quit

CSI cameras on Jetson (no display):

```bash
CAMERA_TYPE=csi python3 scripts/calibrate_stereo.py capture \
    --output-dir data/calib_images \
    --headless
```

---

## 3. Run calibration

```bash
python3 scripts/calibrate_stereo.py calibrate \
    --image-dir data/calib_images \
    --output config/stereo_calibration.yaml \
    --board-cols 9 \
    --board-rows 6 \
    --square-size-mm 25
```

This runs `cv2.stereoCalibrate` followed by `cv2.stereoRectify` and saves all matrices
to `config/stereo_calibration.yaml`.

**Accept the result only if:**

- RMS reprojection error is **< 1.0 px** (< 0.5 px is excellent)
- The printed baseline matches `baseline_m` in the output file within ~5 mm

If RMS is high, discard the worst-scoring frames from `data/calib_images/` and re-run.

---

## 4. One-shot (capture + calibrate)

```bash
python3 scripts/calibrate_stereo.py auto \
    --output config/stereo_calibration.yaml
```

Captures images interactively and immediately calibrates when you press **`c`**.

---

## 5. Verify the calibration

```bash
python3 scripts/test_depth.py \
    --camera-type usb \
    --calibration config/stereo_calibration.yaml \
    --debug-save
```

This runs the full pipeline (rectify → disparity → depth → obstacle_map) and saves
debug frames to `/tmp/rover_debug/`. Check that:

- Horizontal green lines in the rectified side-by-side image pass through the same
  feature in both left and right frames (epipolar alignment)
- The depth heat-map shows close objects as warm and distant objects as cool
- `nearest_obstacle_m` printed to the console matches a ruler measurement

---

## 6. The calibration YAML format

`config/stereo_calibration.yaml` stores all parameters as nested Python-style lists
(rows × columns):

```yaml
image_size: [640, 480]
baseline_m: 0.12        # measured distance between camera centres in metres

K_left:                 # 3×3 intrinsic matrix, left camera
  - [fx, 0,  cx]
  - [0,  fy, cy]
  - [0,  0,  1 ]

D_left:  [k1, k2, p1, p2, k3]   # distortion coefficients (5-element)
K_right: ...
D_right: ...

R:  ...    # 3×3 rotation between cameras
T:  ...    # 3×1 translation between cameras (metres)

R1: ...    # 3×3 rectification transform, left
R2: ...    # 3×3 rectification transform, right
P1: ...    # 3×4 projection matrix after rectification, left
P2: ...    # 3×4 projection matrix after rectification, right
Q:  ...    # 4×4 disparity-to-depth mapping matrix
```

`load_calibration(path)` in `services/stereo-perception/calibration.py` reads this file
and returns a `CalibrationData` dataclass with all matrices as `np.ndarray`.

---

## 7. Transferring the calibration to the Jetson

After calibrating on a laptop with a display:

```bash
scp config/stereo_calibration.yaml jetson@<JETSON_IP>:/home/jetson/rover/config/
```

Then verify on the Jetson:

```bash
python3 scripts/test_depth.py \
    --camera-type csi \
    --calibration config/stereo_calibration.yaml
```

---

## 8. Re-calibration triggers

Re-calibrate if any of the following occur:

- The stereo rig is dropped or receives a strong impact
- A camera is unmounted and remounted
- Depth readings are consistently off by > 10 % vs. a ground-truth measurement
- Epipolar lines are visibly misaligned in the rectified debug images

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cv2.error` in `findChessboardCorners` | Blurry image or bad lighting | Better lighting; clean lens |
| RMS > 1.5 px | Too few images or poor poses | Add more diverse poses |
| `baseline_m` near 0 in output | T vector wrong scale | Check `square_size_mm` argument |
| Depth too large by constant factor | Wrong `square_size_mm` | Measure squares physically and rerun |
| Vertical parallax in rectified images | Cameras pitched differently | Tighten rig; recalibrate |
| `CALIBRATION_FILE` not found at runtime | Wrong path in `.env.rover` | Set `CALIBRATION_FILE=/full/path/to/stereo_calibration.yaml` |
