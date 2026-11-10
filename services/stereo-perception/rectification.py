"""
Stereo image rectification.

Rectification maps both images so that corresponding points lie on the same
horizontal scanline.  This is a prerequisite for disparity computation.

After rectification:
  - epipolar lines are horizontal
  - the disparity for a point is purely horizontal (dx, 0)
  - depth = focal_length * baseline / disparity

Debug mode writes rectified images with horizontal guide lines to disk so
you can visually verify that calibration is correct: straight horizontal lines
in both images should pass through matching scene features.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from calibration import CalibrationData

log = logging.getLogger(__name__)


class StereoRectifier:
    """
    Applies pre-computed stereo rectification maps to raw image pairs.

    The undistortion + rectification maps are computed once at construction
    time (from the calibration data) and reused for every frame.  This avoids
    the O(width × height) map rebuild on each frame.
    """

    def __init__(self, calib: CalibrationData) -> None:
        self._calib = calib
        w, h = calib.image_size

        # Compute remapping tables for left and right cameras
        self._map_left_x,  self._map_left_y  = cv2.initUndistortRectifyMap(
            calib.K_left,  calib.D_left,  calib.R1, calib.P1,
            (w, h), cv2.CV_16SC2,
        )
        self._map_right_x, self._map_right_y = cv2.initUndistortRectifyMap(
            calib.K_right, calib.D_right, calib.R2, calib.P2,
            (w, h), cv2.CV_16SC2,
        )
        log.info("Rectification maps computed: %dx%d  baseline=%.3f m  f=%.1f px",
                 w, h, calib.baseline_m, calib.focal_length)

    def rectify(
        self,
        left:  np.ndarray,
        right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undistort and rectify a stereo image pair.

        Uses cv2.INTER_LINEAR which is fast and produces good-quality results
        for the disparity algorithm.  Avoid INTER_CUBIC — it is slower and the
        quality gain is irrelevant for disparity computation.
        """
        left_rect  = cv2.remap(left,  self._map_left_x,  self._map_left_y,  cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self._map_right_x, self._map_right_y, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def debug_annotate(
        self,
        left_rect:  np.ndarray,
        right_rect: np.ndarray,
        n_lines:    int = 20,
    ) -> np.ndarray:
        """
        Draw horizontal guide lines on a side-by-side view of the rectified pair.

        Correctly rectified images will show that matching features lie exactly
        on the same horizontal scanline in both images.
        """
        h, w = left_rect.shape[:2]
        canvas = np.hstack([left_rect.copy(), right_rect.copy()])

        step = h // (n_lines + 1)
        for i in range(1, n_lines + 1):
            y = i * step
            cv2.line(canvas, (0, y), (w * 2, y), (0, 255, 0), 1)

        # Label each half
        cv2.putText(canvas, "LEFT",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(canvas, "RIGHT", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return canvas

    def save_debug_frame(
        self,
        left_rect:  np.ndarray,
        right_rect: np.ndarray,
        output_dir: str,
        frame_id:   int,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        annotated = self.debug_annotate(left_rect, right_rect)
        path = os.path.join(output_dir, f"rectified_{frame_id:06d}.jpg")
        cv2.imwrite(path, annotated)
