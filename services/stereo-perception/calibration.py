"""
Stereo camera calibration utilities.

Calibration data is stored in config/stereo_calibration.yaml.
Run scripts/calibrate_stereo.py to generate a calibration file from a
physical chessboard target.

The YAML schema stores all numpy arrays as nested lists so the file is
human-readable and does not require numpy for loading the schema itself.

Typical baseline for a 12 cm stereo camera pair: ~0.12 m.
For the Intel RealSense D435i stereo module: ~0.05 m.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

log = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """All parameters produced by stereo calibration."""
    image_size: Tuple[int, int]   # (width, height)
    K_left:     np.ndarray        # 3×3 camera matrix (intrinsics)
    D_left:     np.ndarray        # distortion coefficients (1×5)
    K_right:    np.ndarray
    D_right:    np.ndarray
    R:          np.ndarray        # 3×3 rotation from left to right camera
    T:          np.ndarray        # 3×1 translation from left to right camera (metres)
    R1:         np.ndarray        # 3×3 rectification transform for left camera
    R2:         np.ndarray        # 3×3 rectification transform for right camera
    P1:         np.ndarray        # 3×4 projection matrix for left camera (after rectification)
    P2:         np.ndarray        # 3×4 projection matrix for right camera (after rectification)
    Q:          np.ndarray        # 4×4 disparity-to-depth mapping matrix
    baseline_m: float             # camera baseline in metres (derived from T)

    @property
    def focal_length(self) -> float:
        """Focal length in pixels (from P1)."""
        return float(self.P1[0, 0])

    @property
    def cx_left(self) -> float:
        return float(self.P1[0, 2])

    @property
    def cy_left(self) -> float:
        return float(self.P1[1, 2])


def _arr_to_list(a: np.ndarray) -> list:
    return a.tolist()


def _list_to_arr(lst: list, dtype=np.float64) -> np.ndarray:
    return np.array(lst, dtype=dtype)


def save_calibration(data: CalibrationData, path: str) -> None:
    """Serialise CalibrationData to a human-readable YAML file."""
    record = {
        "image_size":  list(data.image_size),
        "K_left":      _arr_to_list(data.K_left),
        "D_left":      _arr_to_list(data.D_left),
        "K_right":     _arr_to_list(data.K_right),
        "D_right":     _arr_to_list(data.D_right),
        "R":           _arr_to_list(data.R),
        "T":           _arr_to_list(data.T),
        "R1":          _arr_to_list(data.R1),
        "R2":          _arr_to_list(data.R2),
        "P1":          _arr_to_list(data.P1),
        "P2":          _arr_to_list(data.P2),
        "Q":           _arr_to_list(data.Q),
        "baseline_m":  float(data.baseline_m),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(record, f, default_flow_style=False)
    log.info("Calibration saved to %s  baseline=%.4f m  f=%.1f px",
             path, data.baseline_m, data.focal_length)


def load_calibration(path: str) -> Optional[CalibrationData]:
    """
    Load CalibrationData from a YAML file.

    Returns None if the file does not exist.  Raises ValueError on parse
    errors so that the caller can decide whether to abort or continue with
    uncalibrated depth (development / simulation).
    """
    if not Path(path).exists():
        log.warning("Calibration file not found: %s", path)
        return None

    with open(path) as f:
        rec = yaml.safe_load(f)

    try:
        T = _list_to_arr(rec["T"])
        baseline_m = float(rec.get("baseline_m", float(np.linalg.norm(T))))
        return CalibrationData(
            image_size = tuple(rec["image_size"]),
            K_left     = _list_to_arr(rec["K_left"]),
            D_left     = _list_to_arr(rec["D_left"]),
            K_right    = _list_to_arr(rec["K_right"]),
            D_right    = _list_to_arr(rec["D_right"]),
            R          = _list_to_arr(rec["R"]),
            T          = T,
            R1         = _list_to_arr(rec["R1"]),
            R2         = _list_to_arr(rec["R2"]),
            P1         = _list_to_arr(rec["P1"]),
            P2         = _list_to_arr(rec["P2"]),
            Q          = _list_to_arr(rec["Q"]),
            baseline_m = baseline_m,
        )
    except KeyError as e:
        raise ValueError(f"Calibration file {path!r} is missing field: {e}")


def compute_stereo_calibration(
    left_images:   list[np.ndarray],
    right_images:  list[np.ndarray],
    board_size:    Tuple[int, int] = (9, 6),  # inner corners (cols, rows)
    square_size_m: float           = 0.025,   # chessboard square size in metres
    image_size:    Optional[Tuple[int, int]] = None,
) -> CalibrationData:
    """
    Compute stereo calibration from paired chessboard images.

    board_size is the number of INNER corners per row and column.
    For a standard 10×7 square chessboard that means board_size=(9, 6).

    Returns a CalibrationData ready for save_calibration().
    """
    import cv2  # only needed when running actual calibration, not in tests

    cols, rows = board_size

    # Prepare object points (Z=0 plane)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

    obj_points   = []   # 3D points in world space
    left_points  = []   # 2D points in left image
    right_points = []   # 2D points in right image

    if image_size is None:
        h, w = left_images[0].shape[:2]
        image_size = (w, h)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    found_count = 0
    for i, (img_l, img_r) in enumerate(zip(left_images, right_images)):
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (cols, rows), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (cols, rows), None)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            left_points.append(corners_l)
            right_points.append(corners_r)
            found_count += 1
        else:
            log.warning("Chessboard not found in image pair %d (left=%s right=%s)",
                        i, ret_l, ret_r)

    if found_count < 5:
        raise ValueError(
            f"Only {found_count} valid image pairs found (minimum 5 required). "
            f"Ensure the calibration target is clearly visible in both cameras."
        )
    log.info("Calibrating with %d image pairs ...", found_count)

    # Individual camera calibrations (required for stereoCalibrate)
    flags_mono = cv2.CALIB_RATIONAL_MODEL
    _, K_left, D_left, _, _ = cv2.calibrateCamera(
        obj_points, left_points, image_size, None, None, flags=flags_mono
    )
    _, K_right, D_right, _, _ = cv2.calibrateCamera(
        obj_points, right_points, image_size, None, None, flags=flags_mono
    )

    # Stereo calibration
    flags_stereo = (
        cv2.CALIB_FIX_INTRINSIC |   # use individual calibration results
        cv2.CALIB_RATIONAL_MODEL
    )
    rms, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
        obj_points, left_points, right_points,
        K_left, D_left, K_right, D_right,
        image_size,
        criteria=criteria,
        flags=flags_stereo,
    )
    log.info("Stereo calibration RMS reprojection error: %.4f px", rms)

    # Stereo rectification
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_left, D_left, K_right, D_right,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,   # alpha=0 crops to valid pixels; alpha=1 keeps all pixels
    )

    baseline_m = float(np.linalg.norm(T))
    return CalibrationData(
        image_size = image_size,
        K_left     = K_left,
        D_left     = D_left,
        K_right    = K_right,
        D_right    = D_right,
        R          = R,
        T          = T,
        R1         = R1,
        R2         = R2,
        P1         = P1,
        P2         = P2,
        Q          = Q,
        baseline_m = baseline_m,
    )


def load_calibration_from_oak(
    device,
    width:  int = 640,
    height: int = 400,
) -> CalibrationData:
    """
    Read the OAK-D Lite's factory stereo calibration from device flash and
    return a CalibrationData with a Q matrix matching the OAK's rectification.

    Call this ONCE to generate a YAML file:
        from camera import OAKCameraSource
        import depthai as dai
        cam = OAKCameraSource(640, 400, 30)
        cam.open()
        calib = load_calibration_from_oak(cam.dai_device, 640, 400)
        save_calibration(calib, "config/stereo_calibration.yaml")
        cam.close()

    Or use the helper script:
        python3 scripts/calibrate_stereo.py oak-calib --output config/stereo_calibration.yaml

    Args:
        device: an open depthai.Device instance
        width, height: the resolution you configured on the mono cameras
    """
    import cv2
    import depthai as dai

    calib_handler = device.readCalibration()

    left_socket  = dai.CameraBoardSocket.CAM_B   # left mono on OAK-D Lite
    right_socket = dai.CameraBoardSocket.CAM_C   # right mono on OAK-D Lite

    # Raw (unrectified) intrinsics from device flash
    K_left  = np.array(calib_handler.getCameraIntrinsics(left_socket,  width, height), dtype=np.float64)
    K_right = np.array(calib_handler.getCameraIntrinsics(right_socket, width, height), dtype=np.float64)

    # Distortion coefficients (OpenCV 5-element model)
    D_left  = np.array(calib_handler.getDistortionCoefficients(left_socket),  dtype=np.float64)[:5].reshape(1, 5)
    D_right = np.array(calib_handler.getDistortionCoefficients(right_socket), dtype=np.float64)[:5].reshape(1, 5)

    # Extrinsics: 4×4 [R|t; 0 0 0 1], translation in cm
    extrinsics = np.array(
        calib_handler.getCameraExtrinsics(left_socket, right_socket),
        dtype=np.float64,
    )
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3:4] / 100.0          # cm → m

    baseline_m = float(abs(calib_handler.getBaselineDistance() / 100.0))

    # Compute rectification transforms — cv2.stereoRectify matches the OAK's
    # internal rectification algorithm (alpha=0 crops to valid intersection).
    image_size = (width, height)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_left, D_left, K_right, D_right,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    log.info(
        "OAK calibration loaded from device: baseline=%.4f m  f=%.1f px  %dx%d",
        baseline_m, float(P1[0, 0]), width, height,
    )
    return CalibrationData(
        image_size = image_size,
        K_left     = K_left,
        D_left     = D_left,
        K_right    = K_right,
        D_right    = D_right,
        R          = R,
        T          = T,
        R1         = R1,
        R2         = R2,
        P1         = P1,
        P2         = P2,
        Q          = Q,
        baseline_m = baseline_m,
    )


def make_dummy_calibration(width: int = 640, height: int = 480,
                           baseline_m: float = 0.12) -> CalibrationData:
    """
    Return a synthetic calibration for development without a physical camera.

    Values approximate a 640×480 camera with 3.6 mm focal length and 12 cm baseline.
    Depth accuracy will be wrong but the pipeline structure is exercised.
    """
    f  = 3.6e-3 / (3.6e-6)   # focal length in pixels ≈ 1000 for typical webcam
    cx = width  / 2.0
    cy = height / 2.0

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1.0]], dtype=np.float64)
    D = np.zeros((1, 5), dtype=np.float64)
    R_stereo = np.eye(3, dtype=np.float64)
    T_stereo = np.array([[baseline_m], [0.0], [0.0]], dtype=np.float64)

    R1 = R2 = np.eye(3, dtype=np.float64)
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = np.hstack([K, np.array([[-f * baseline_m], [0], [0]])])

    # Build Q matrix manually.
    # OpenCV convention: Tx = -baseline_m, so Q[3,2] = -1/Tx = +1/baseline_m.
    # This makes W = (1/baseline_m)*disparity > 0 and depth = f/W > 0.
    Q = np.array([
        [1, 0,  0,                -cx],
        [0, 1,  0,                -cy],
        [0, 0,  0,                  f],
        [0, 0,  1.0 / baseline_m,  0],
    ], dtype=np.float64)

    return CalibrationData(
        image_size = (width, height),
        K_left     = K.copy(),
        D_left     = D.copy(),
        K_right    = K.copy(),
        D_right    = D.copy(),
        R          = R_stereo,
        T          = T_stereo,
        R1         = R1,
        R2         = R2,
        P1         = P1,
        P2         = P2,
        Q          = Q,
        baseline_m = baseline_m,
    )
