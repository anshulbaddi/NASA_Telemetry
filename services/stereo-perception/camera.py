"""
Stereo camera abstraction for the perception pipeline.

Supports three camera source types:
  csi  — NVIDIA Jetson CSI cameras via GStreamer / nvarguscamerasrc
  usb  — USB V4L2 cameras via OpenCV VideoCapture
  file — Pre-recorded image pairs from disk (development / CI)

Usage:
    source = create_camera_source(config)
    source.open()
    while True:
        pair = source.read_pair()
        if pair is None:
            handle_failure()
            continue
        process(pair.left, pair.right)
    source.close()
"""
from __future__ import annotations

import glob
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FramePair:
    left:      np.ndarray
    right:     np.ndarray
    timestamp: float       = field(default_factory=time.time)
    frame_id:  int         = 0


class CameraSource(ABC):
    """Abstract base for any stereo camera source."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read_pair(self) -> Optional[FramePair]: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def is_healthy(self) -> Tuple[bool, bool]:
        """Return (left_ok, right_ok)."""
        ...


# ── CSI Camera (Jetson Nano native) ──────────────────────────────────────────

def _csi_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    """
    GStreamer pipeline for Jetson CSI cameras using nvarguscamerasrc.

    This is the low-latency path on Jetson; it uses the ISP hardware directly.
    flip-method=0 keeps the image upright — change to 2 if the camera is mounted
    inverted.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"! video/x-raw(memory:NVMM),"
        f"width=(int){width},height=(int){height},framerate=(fraction){fps}/1 "
        f"! nvvidconv flip-method=0 "
        f"! video/x-raw,width=(int){width},height=(int){height},format=(string)BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=(string)BGR "
        f"! appsink drop=true max-buffers=2"
    )


class CSICameraSource(CameraSource):
    """
    Dual CSI camera source for Jetson Nano.

    Each camera is opened with its own GStreamer pipeline so that both streams
    run independently in hardware.  Frame synchronisation relies on capturing
    both at the same nominal FPS; hardware sync (via a GPIO trigger) would give
    better results but is not required for static scenes.
    """

    def __init__(self, left_id: int, right_id: int,
                 width: int, height: int, fps: int) -> None:
        self._left_id  = left_id
        self._right_id = right_id
        self._width    = width
        self._height   = height
        self._fps      = fps
        self._cap_left:  Optional[cv2.VideoCapture] = None
        self._cap_right: Optional[cv2.VideoCapture] = None
        self._frame_id  = 0

    def open(self) -> None:
        left_pipeline  = _csi_pipeline(self._left_id,  self._width, self._height, self._fps)
        right_pipeline = _csi_pipeline(self._right_id, self._width, self._height, self._fps)

        self._cap_left  = cv2.VideoCapture(left_pipeline,  cv2.CAP_GSTREAMER)
        self._cap_right = cv2.VideoCapture(right_pipeline, cv2.CAP_GSTREAMER)

        if not self._cap_left.isOpened():
            raise RuntimeError(f"CSI left camera (sensor-id={self._left_id}) failed to open")
        if not self._cap_right.isOpened():
            raise RuntimeError(f"CSI right camera (sensor-id={self._right_id}) failed to open")

        log.info("CSI cameras opened: left sensor-id=%d  right sensor-id=%d  %dx%d @ %d fps",
                 self._left_id, self._right_id, self._width, self._height, self._fps)

    def read_pair(self) -> Optional[FramePair]:
        if self._cap_left is None or self._cap_right is None:
            return None
        ts = time.time()
        ret_l, left  = self._cap_left.read()
        ret_r, right = self._cap_right.read()
        if not ret_l or not ret_r:
            log.warning("CSI camera read failure: left=%s right=%s", ret_l, ret_r)
            return None
        self._frame_id += 1
        return FramePair(left=left, right=right, timestamp=ts, frame_id=self._frame_id)

    def close(self) -> None:
        if self._cap_left:
            self._cap_left.release()
        if self._cap_right:
            self._cap_right.release()
        log.info("CSI cameras released")

    def is_healthy(self) -> Tuple[bool, bool]:
        left_ok  = self._cap_left is not None  and self._cap_left.isOpened()
        right_ok = self._cap_right is not None and self._cap_right.isOpened()
        return left_ok, right_ok


# ── USB Camera ────────────────────────────────────────────────────────────────

class USBCameraSource(CameraSource):
    """
    Dual USB camera source using V4L2 / OpenCV VideoCapture.

    Both cameras are read in sequence on the same thread; the capture-time
    difference is typically < 5 ms which is acceptable for slow-moving rovers.
    For tighter sync, a hardware trigger is needed.
    """

    def __init__(self, left_id: int, right_id: int,
                 width: int, height: int, fps: int) -> None:
        self._left_id  = left_id
        self._right_id = right_id
        self._width    = width
        self._height   = height
        self._fps      = fps
        self._cap_left:  Optional[cv2.VideoCapture] = None
        self._cap_right: Optional[cv2.VideoCapture] = None
        self._frame_id  = 0

    def _open_cap(self, device_id: int, label: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Fallback: try default backend
            cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            raise RuntimeError(f"USB {label} camera (id={device_id}) failed to open")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS,          self._fps)
        # Use MJPG codec for higher USB bandwidth efficiency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Minimize internal buffer to get fresh frames
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def open(self) -> None:
        self._cap_left  = self._open_cap(self._left_id,  "left")
        self._cap_right = self._open_cap(self._right_id, "right")
        actual_w = int(self._cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap_left.get(cv2.CAP_PROP_FPS)
        log.info("USB cameras opened: left=%d right=%d  actual=%dx%d @ %.1f fps",
                 self._left_id, self._right_id, actual_w, actual_h, actual_fps)

    def read_pair(self) -> Optional[FramePair]:
        if self._cap_left is None or self._cap_right is None:
            return None
        ts = time.time()
        ret_l, left  = self._cap_left.read()
        ret_r, right = self._cap_right.read()
        if not ret_l or not ret_r or left is None or right is None:
            log.warning("USB camera read failure: left=%s right=%s", ret_l, ret_r)
            return None
        self._frame_id += 1
        return FramePair(left=left, right=right, timestamp=ts, frame_id=self._frame_id)

    def close(self) -> None:
        if self._cap_left:
            self._cap_left.release()
        if self._cap_right:
            self._cap_right.release()
        log.info("USB cameras released")

    def is_healthy(self) -> Tuple[bool, bool]:
        left_ok  = self._cap_left is not None  and self._cap_left.isOpened()
        right_ok = self._cap_right is not None and self._cap_right.isOpened()
        return left_ok, right_ok


# ── File-based source (development / CI) ─────────────────────────────────────

class FileCameraSource(CameraSource):
    """
    Reads pre-recorded stereo image pairs from disk.

    Expects two directories with identically-named image files:
        LEFT_IMAGE_DIR/  frame_000001.png  frame_000002.png ...
        RIGHT_IMAGE_DIR/ frame_000001.png  frame_000002.png ...

    Files are consumed in sorted order.  If LOOP_FILE_SOURCE is True the
    sequence repeats indefinitely, which is useful for continuous testing.
    """

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, left_dir: str, right_dir: str, loop: bool = True) -> None:
        self._left_dir  = left_dir
        self._right_dir = right_dir
        self._loop      = loop
        self._pairs:    list[tuple[str, str]] = []
        self._idx       = 0
        self._frame_id  = 0
        self._left_ok   = False
        self._right_ok  = False

    def open(self) -> None:
        left_files  = sorted(f for f in os.listdir(self._left_dir)
                             if os.path.splitext(f)[1].lower() in self._IMAGE_EXTS)
        right_files = sorted(f for f in os.listdir(self._right_dir)
                             if os.path.splitext(f)[1].lower() in self._IMAGE_EXTS)
        if not left_files or not right_files:
            raise RuntimeError(
                f"No images found in {self._left_dir!r} or {self._right_dir!r}. "
                f"Place test stereo image pairs there to use CAMERA_TYPE=file."
            )
        # Match by filename — skip files that don't have a counterpart
        left_set  = set(left_files)
        right_set = set(right_files)
        common    = sorted(left_set & right_set)
        if not common:
            # Fall back to position-based pairing
            common_count = min(len(left_files), len(right_files))
            self._pairs = [
                (os.path.join(self._left_dir,  left_files[i]),
                 os.path.join(self._right_dir, right_files[i]))
                for i in range(common_count)
            ]
        else:
            self._pairs = [
                (os.path.join(self._left_dir,  f),
                 os.path.join(self._right_dir, f))
                for f in common
            ]
        self._left_ok  = True
        self._right_ok = True
        log.info("File camera source: %d pairs in %s / %s  loop=%s",
                 len(self._pairs), self._left_dir, self._right_dir, self._loop)

    def read_pair(self) -> Optional[FramePair]:
        if not self._pairs:
            return None
        if self._idx >= len(self._pairs):
            if self._loop:
                self._idx = 0
            else:
                log.info("File source exhausted (%d frames)", len(self._pairs))
                return None

        left_path, right_path = self._pairs[self._idx]
        left  = cv2.imread(left_path)
        right = cv2.imread(right_path)
        if left is None:
            log.error("Failed to read left image: %s", left_path)
            self._left_ok = False
            return None
        if right is None:
            log.error("Failed to read right image: %s", right_path)
            self._right_ok = False
            return None

        self._idx += 1
        self._frame_id += 1
        return FramePair(left=left, right=right, timestamp=time.time(),
                         frame_id=self._frame_id)

    def close(self) -> None:
        log.info("File camera source closed")

    def is_healthy(self) -> Tuple[bool, bool]:
        return self._left_ok, self._right_ok


# ── OAK-D Lite (Luxonis DepthAI) ─────────────────────────────────────────────

class OAKCameraSource(CameraSource):
    """
    Luxonis OAK-D Lite stereo camera source using the DepthAI SDK.

    The OAK-D Lite:
      - Connects via USB3 Type-C (NOT a V4L2 /dev/video* device)
      - Has two hardware-synchronised mono cameras (OV9282, 1280×800)
      - Has an onboard Myriad X VPU that rectifies and computes depth
      - Stores factory stereo calibration in device flash

    This source uses the OAK's StereoDepth node for rectification only,
    then returns the pre-rectified left+right frames as BGR numpy arrays.

    IMPORTANT: since frames are already rectified by the OAK hardware,
    StereoPipeline sets skip_rectify=True when CAMERA_TYPE=oak so that
    the StereoRectifier step is bypassed in the perception thread.

    Default resolution: 640×400 (THE_400_P).  Adjust CAMERA_HEIGHT to 720
    for THE_720_P (1280×720) — CAMERA_WIDTH must match the preset.
    """

    # DepthAI mono resolution presets indexed by height
    _RESOLUTIONS = {
        400: "THE_400_P",   # 640×400
        480: "THE_480_P",   # 640×480
        720: "THE_720_P",   # 1280×720
        800: "THE_800_P",   # 1280×800
    }

    def __init__(self, width: int, height: int, fps: int) -> None:
        self._width    = width
        self._height   = height
        self._fps      = fps
        self._device   = None
        self._q_left   = None
        self._q_right  = None
        self._frame_id = 0
        self._last_left_ok  = False
        self._last_right_ok = False

    def _build_pipeline(self):
        import depthai as dai

        res_name = self._RESOLUTIONS.get(self._height, "THE_400_P")
        resolution = getattr(dai.MonoCameraProperties.SensorResolution, res_name)

        pipeline = dai.Pipeline()

        left_cam = pipeline.create(dai.node.MonoCamera)
        left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        left_cam.setResolution(resolution)
        left_cam.setFps(self._fps)

        right_cam = pipeline.create(dai.node.MonoCamera)
        right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        right_cam.setResolution(resolution)
        right_cam.setFps(self._fps)

        # Use StereoDepth node for hardware rectification only.
        # We run our own SGBM disparity on the rectified frames.
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetType.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setOutputSize(self._width, self._height)

        left_cam.out.link(stereo.left)
        right_cam.out.link(stereo.right)

        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("rect_left")
        stereo.rectifiedLeft.link(xout_left.input)

        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName("rect_right")
        stereo.rectifiedRight.link(xout_right.input)

        return pipeline

    def open(self) -> None:
        import depthai as dai

        pipeline = self._build_pipeline()
        self._device  = dai.Device(pipeline)
        self._q_left  = self._device.getOutputQueue("rect_left",  maxSize=2, blocking=True)
        self._q_right = self._device.getOutputQueue("rect_right", maxSize=2, blocking=True)
        log.info("OAK-D Lite opened: %dx%d @ %d fps", self._width, self._height, self._fps)

    def read_pair(self) -> Optional[FramePair]:
        if self._q_left is None or self._q_right is None:
            return None
        try:
            in_left  = self._q_left.get()
            in_right = self._q_right.get()
        except Exception as exc:
            log.error("OAK frame read error: %s", exc)
            self._last_left_ok  = False
            self._last_right_ok = False
            return None

        left_gray  = in_left.getCvFrame()    # H×W uint8 grayscale
        right_gray = in_right.getCvFrame()

        # Convert mono → BGR so the rest of the pipeline is consistent
        # with USB/CSI camera output (DisparityComputer handles both anyway)
        import cv2
        left_bgr  = cv2.cvtColor(left_gray,  cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

        self._frame_id += 1
        self._last_left_ok  = True
        self._last_right_ok = True
        return FramePair(
            left      = left_bgr,
            right     = right_bgr,
            timestamp = time.time(),
            frame_id  = self._frame_id,
        )

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
        log.info("OAK-D Lite closed")

    def is_healthy(self) -> Tuple[bool, bool]:
        return self._last_left_ok, self._last_right_ok

    @property
    def dai_device(self):
        """Expose the raw depthai.Device for calibration reading."""
        return self._device


# ── Factory ───────────────────────────────────────────────────────────────────

def create_camera_source(config) -> CameraSource:
    """
    Instantiate the appropriate CameraSource based on config.CAMERA_TYPE.

    config must expose: CAMERA_TYPE, LEFT_CAMERA_ID, RIGHT_CAMERA_ID,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, LEFT_IMAGE_DIR, RIGHT_IMAGE_DIR,
    LOOP_FILE_SOURCE.
    """
    cam_type = config.CAMERA_TYPE.lower()
    if cam_type == "csi":
        return CSICameraSource(
            config.LEFT_CAMERA_ID, config.RIGHT_CAMERA_ID,
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS,
        )
    elif cam_type == "usb":
        return USBCameraSource(
            config.LEFT_CAMERA_ID, config.RIGHT_CAMERA_ID,
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS,
        )
    elif cam_type == "file":
        return FileCameraSource(
            config.LEFT_IMAGE_DIR, config.RIGHT_IMAGE_DIR,
            config.LOOP_FILE_SOURCE,
        )
    elif cam_type == "oak":
        return OAKCameraSource(
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS,
        )
    else:
        raise ValueError(
            f"Unknown CAMERA_TYPE={cam_type!r}. "
            f"Valid options: csi, usb, file, oak"
        )
