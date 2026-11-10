"""
Disparity map computation from a rectified stereo image pair.

Supports:
  - StereoSGBM (Semi-Global Block Matching) — better quality, recommended
  - StereoBM   (Block Matching)              — faster, coarser

StereoSGBM raw output is 16× the actual disparity in pixels.
All consumers of this module must divide by 16 before using the values.

The optional WLS (Weighted Least Squares) post-filter significantly reduces
disparity noise and fills holes, at the cost of ~30% extra CPU.  Disabled by
default for Jetson Nano throughput; enable with USE_WLS_FILTER=true.

Parameter tuning guide:
  NUM_DISPARITIES — increase for closer max depth range (wider = shorter max range)
  BLOCK_SIZE      — larger = smoother but loses fine detail
  P1, P2          — larger = smoother depth transitions
  UNIQUENESS_RATIO — raise to reject ambiguous matches (reduces noise, more holes)
  SPECKLE_WIN_SIZE — raise to remove more isolated noise blobs
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class DisparityComputer:
    """
    Computes a 16-bit (×16 scaled) disparity map from a rectified stereo pair.

    Input images should be:
      - rectified (via StereoRectifier)
      - grayscale  (for SGBM/BM)
      - uint8

    Output is int16 (SGBM native output) scaled by ×16.
    Use disparity.astype(np.float32) / 16.0 to get disparity in pixels.
    Negative values and MIN_DISPARITY×16 are invalid.
    """

    def __init__(self, config) -> None:
        bs = config.BLOCK_SIZE
        nd = config.NUM_DISPARITIES
        P1 = config.P1_MULT * 3 * bs * bs
        P2 = config.P2_MULT * 3 * bs * bs

        self._min_disp = config.MIN_DISPARITY
        self._algorithm = config.STEREO_ALGORITHM.lower()
        self._use_wls   = config.USE_WLS_FILTER
        self._wls_lambda = getattr(config, "WLS_LAMBDA", 8000.0)
        self._wls_sigma  = getattr(config, "WLS_SIGMA",  1.5)

        if self._algorithm == "sgbm":
            self._matcher = cv2.StereoSGBM_create(
                minDisparity       = config.MIN_DISPARITY,
                numDisparities     = nd,
                blockSize          = bs,
                P1                 = P1,
                P2                 = P2,
                disp12MaxDiff      = config.DISP12_MAX_DIFF,
                uniquenessRatio    = config.UNIQUENESS_RATIO,
                speckleWindowSize  = config.SPECKLE_WIN_SIZE,
                speckleRange       = config.SPECKLE_RANGE,
                preFilterCap       = config.PRE_FILTER_CAP,
                mode               = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
            if self._use_wls:
                self._matcher_right = cv2.ximgproc.createRightMatcher(self._matcher)
        elif self._algorithm == "bm":
            self._matcher = cv2.StereoBM_create(
                numDisparities = nd,
                blockSize      = bs,
            )
            if self._use_wls:
                self._matcher_right = cv2.ximgproc.createRightMatcher(self._matcher)
        else:
            raise ValueError(f"Unknown STEREO_ALGORITHM={config.STEREO_ALGORITHM!r}. "
                             f"Valid: sgbm, bm")

        if self._use_wls:
            self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._matcher)
            self._wls_filter.setLambda(self._wls_lambda)
            self._wls_filter.setSigmaColor(self._wls_sigma)

        log.info(
            "DisparityComputer: algorithm=%s  numDisparities=%d  blockSize=%d  "
            "P1=%d  P2=%d  WLS=%s",
            self._algorithm, nd, bs, P1, P2, self._use_wls,
        )

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        """
        Compute the disparity map.

        Inputs must be grayscale uint8 images (H×W).
        Returns int16 array (H×W) where values are disparity × 16.
        Invalid pixels have value = (MIN_DISPARITY - 1) × 16.
        """
        if left_gray.ndim == 3:
            left_gray  = cv2.cvtColor(left_gray,  cv2.COLOR_BGR2GRAY)
        if right_gray.ndim == 3:
            right_gray = cv2.cvtColor(right_gray, cv2.COLOR_BGR2GRAY)

        if self._use_wls:
            disp_left  = self._matcher.compute(left_gray, right_gray)
            disp_right = self._matcher_right.compute(right_gray, left_gray)
            disparity  = self._wls_filter.filter(
                disp_left, left_gray, disparity_map_right=disp_right
            )
        else:
            disparity = self._matcher.compute(left_gray, right_gray)

        return disparity  # int16, ×16 scaled
