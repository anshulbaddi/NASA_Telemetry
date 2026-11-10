"""Obstacle-detector gRPC service — LIDAR cluster analysis and threat scoring."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np
import grpclib.server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge, start_http_server

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generated.obstacledetector.v1.obstacle_detector_pb2 import (
    ObstacleDetectorBase,
    SensorInput, ObstacleAnalysis,
    DetectedObstacle, ClearZone,
)

# ── config ────────────────────────────────────────────────────────────────────
GRPC_PORT         = int(os.getenv("GRPC_PORT",    "50051"))
METRICS_PORT      = int(os.getenv("METRICS_PORT", "9093"))
OTLP_ENDPOINT     = os.getenv("OTLP_ENDPOINT",   "http://localhost:4317")

EMERGENCY_DIST_M  = float(os.getenv("EMERGENCY_DIST_M",  "1.5"))   # stop if anything closer
THREAT_DIST_M     = float(os.getenv("THREAT_DIST_M",     "3.0"))   # high-threat threshold
CLUSTER_GAP_DEG   = float(os.getenv("CLUSTER_GAP_DEG",   "15.0"))  # angular gap between clusters

# PERCEPTION_MODE=simulation — use incoming lidar_distances_m (existing path)
# PERCEPTION_MODE=stereo     — use left/center/right clearances from stereo camera
PERCEPTION_MODE   = os.getenv("PERCEPTION_MODE", "simulation")  # simulation | stereo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Cython-accelerated LIDAR clustering — built with `python setup.py build_ext --inplace`
try:
    from lidar_cluster import cluster_lidar as _cy_cluster_lidar
    _NATIVE_LIDAR = True
    log.info("lidar_cluster Cython extension loaded")
except ImportError:
    _NATIVE_LIDAR = False
    log.info("lidar_cluster not built — using pure-Python fallback")

# ── metrics ───────────────────────────────────────────────────────────────────
frames_processed  = Counter("obstacle_frames_total",          "LIDAR frames analyzed")
obstacles_found   = Counter("obstacle_detected_total",        "Obstacle detections",          ["threat"])
emergency_stops   = Counter("obstacle_emergency_stops_total", "Emergency-stop triggers")
analysis_latency  = Histogram("obstacle_analysis_seconds",    "Frame analysis latency",
                               buckets=[.0001, .0005, .001, .005, .01, .05])
clear_zones_gauge = Gauge("obstacle_clear_zones_last",        "Clear zones in last frame")

# ── OTel ──────────────────────────────────────────────────────────────────────
def setup_otel() -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("rover.obstacle-detector")


# ── LIDAR analysis ────────────────────────────────────────────────────────────
_DEG_PER_RAY = 5.0  # 72 rays × 5° = 360°

def _classify(dist_m: float, velocity_ms: float) -> str:
    braking_dist = velocity_ms ** 2 / (2 * 0.8)  # ~0.8 m/s² deceleration
    if dist_m < EMERGENCY_DIST_M:
        return "imminent"
    if dist_m < braking_dist + 0.5:
        return "braking"
    if dist_m < THREAT_DIST_M:
        return "close"
    if dist_m < 8.0:
        return "moderate"
    return "clear"


def _threat_score(dist_m: float, velocity_ms: float) -> float:
    if dist_m <= 0:
        return 1.0
    braking = max(1.0, velocity_ms ** 2 / (2 * 0.8))
    return max(0.0, min(1.0, 1.0 - (dist_m - EMERGENCY_DIST_M) / (braking + 4.0)))


def _cluster_lidar(
    distances: List[float],
    velocity_ms: float,
) -> Tuple[List[DetectedObstacle], List[ClearZone]]:
    """Simple angular clustering: consecutive rays within threshold form one obstacle."""
    N = len(distances)
    if N == 0:
        return [], []

    OBSTACLE_THRESHOLD_M = 8.0
    obstacles: List[DetectedObstacle] = []
    clear_zones: List[ClearZone] = []

    in_obstacle = False
    cluster_start_idx = 0
    cluster_dists: List[float] = []

    def _finalise_cluster(start_idx: int, dists: List[float]) -> None:
        if not dists:
            return
        min_dist    = min(dists)
        mean_dist   = sum(dists) / len(dists)
        start_deg   = start_idx * _DEG_PER_RAY
        end_deg     = (start_idx + len(dists) - 1) * _DEG_PER_RAY
        bearing_deg = (start_deg + end_deg) / 2.0
        angular_span = len(dists) * _DEG_PER_RAY
        width_est   = 2 * math.tan(math.radians(angular_span / 2)) * mean_dist

        obstacles.append(DetectedObstacle(
            bearing_degrees  = bearing_deg,
            distance_m       = min_dist,
            width_estimate_m = max(0.1, width_est),
            classification   = _classify(min_dist, velocity_ms),
            threat_level     = _threat_score(min_dist, velocity_ms),
        ))

    for i, d in enumerate(distances):
        if d < OBSTACLE_THRESHOLD_M:
            if not in_obstacle:
                in_obstacle       = True
                cluster_start_idx = i
                cluster_dists     = [d]
            else:
                cluster_dists.append(d)
        else:
            if in_obstacle:
                _finalise_cluster(cluster_start_idx, cluster_dists)
                in_obstacle   = False
                cluster_dists = []
            # accumulate clear zone
            # (merged below)

    if in_obstacle:
        _finalise_cluster(cluster_start_idx, cluster_dists)

    # find clear corridors (runs of rays > OBSTACLE_THRESHOLD_M)
    run_start: Optional[int] = None
    run_max = 0.0
    for i, d in enumerate(distances):
        if d >= OBSTACLE_THRESHOLD_M:
            if run_start is None:
                run_start = i
            run_max = max(run_max, d)
        else:
            if run_start is not None and i - run_start >= 4:  # ≥ 20° wide
                clear_zones.append(ClearZone(
                    bearing_start = run_start * _DEG_PER_RAY,
                    bearing_end   = (i - 1) * _DEG_PER_RAY,
                    max_safe_dist = min(run_max, 20.0),
                ))
            run_start = None
            run_max   = 0.0
    if run_start is not None and N - run_start >= 4:
        clear_zones.append(ClearZone(
            bearing_start = run_start * _DEG_PER_RAY,
            bearing_end   = (N - 1) * _DEG_PER_RAY,
            max_safe_dist = min(run_max, 20.0),
        ))

    return obstacles, clear_zones


def _recommend(obstacles: List[DetectedObstacle], clear_zones: List[ClearZone]) -> str:
    if not obstacles:
        return "PROCEED"
    threats = sorted(obstacles, key=lambda o: o.distance_m)
    closest = threats[0]
    if closest.distance_m < EMERGENCY_DIST_M:
        return "EMERGENCY_STOP"
    if closest.threat_level > 0.7:
        return "REDUCE_SPEED"
    if clear_zones:
        best = max(clear_zones, key=lambda z: z.max_safe_dist)
        return f"STEER_TO_{best.bearing_start:.0f}_{best.bearing_end:.0f}"
    return "SLOW_AND_ASSESS"


# ── Stereo-derived obstacle analysis ─────────────────────────────────────────

def _analyze_stereo(sensor_input: SensorInput) -> ObstacleAnalysis:
    """
    Produce an ObstacleAnalysis from stereo regional clearances.

    Called when sensor_input.stereo_input == True.  The three clearance
    distances (left/center/right) replace the 72-ray LIDAR ring.

    We map the regional depths to synthetic DetectedObstacle entries so
    the rest of the pipeline (path planner, FDIR, API gateway) can consume
    the same protobuf messages without modification.
    """
    left_m   = sensor_input.left_clearance_m   or 999.0
    center_m = sensor_input.center_clearance_m or 999.0
    right_m  = sensor_input.right_clearance_m  or 999.0

    velocity_ms = sensor_input.velocity_ms or 0.0
    obstacles: List[DetectedObstacle] = []
    clear_zones: List[ClearZone]      = []

    region_map = [
        ("left",   left_m,   315.0),   # bearing ≈ 315° (front-left)
        ("center", center_m,   0.0),   # bearing ≈ 0°   (straight ahead)
        ("right",  right_m,   45.0),   # bearing ≈ 45°  (front-right)
    ]

    for name, dist_m, bearing in region_map:
        if dist_m < 999.0 and dist_m > 0:
            threat = _threat_score(dist_m, velocity_ms)
            cls    = _classify(dist_m, velocity_ms)
            if dist_m < THREAT_DIST_M:
                obstacles.append(DetectedObstacle(
                    bearing_degrees  = bearing,
                    distance_m       = dist_m,
                    width_estimate_m = 0.5,
                    classification   = cls,
                    threat_level     = threat,
                ))
            else:
                clear_zones.append(ClearZone(
                    bearing_start = bearing - 20.0,
                    bearing_end   = bearing + 20.0,
                    max_safe_dist = min(dist_m, 20.0),
                ))

    e_stop = any(o.distance_m < EMERGENCY_DIST_M for o in obstacles)
    action = _recommend(obstacles, clear_zones)

    return ObstacleAnalysis(
        rover_id           = sensor_input.rover_id,
        timestamp_ms       = sensor_input.timestamp_ms,
        obstacles          = obstacles,
        clear_zones        = clear_zones,
        emergency_stop     = e_stop,
        recommended_action = action,
        processing_time_ms = 0.0,
    )


# ── gRPC service implementation ───────────────────────────────────────────────
_tracer: trace.Tracer


class ObstacleDetectorService(ObstacleDetectorBase):

    async def analyze(self, sensor_input: SensorInput) -> ObstacleAnalysis:
        with _tracer.start_as_current_span("obstacle.analyze") as span:
            span.set_attribute("rover.id",         sensor_input.rover_id)
            span.set_attribute("velocity.ms",      sensor_input.velocity_ms)
            span.set_attribute("perception_mode",  PERCEPTION_MODE)

            t0 = time.perf_counter()

            # ── Route to stereo or LIDAR path ─────────────────────────────────
            if sensor_input.stereo_input or PERCEPTION_MODE == "stereo":
                span.set_attribute("input.type", "stereo")
                result = _analyze_stereo(sensor_input)
                obstacles   = result.obstacles
                clear_zones = result.clear_zones
                e_stop      = result.emergency_stop
                action      = result.recommended_action
            else:
                span.set_attribute("lidar.rays",   len(sensor_input.lidar_distances_m))
                span.set_attribute("input.type",   "lidar")
                if _NATIVE_LIDAR:
                    arr = np.asarray(sensor_input.lidar_distances_m, dtype=np.float64)
                    obstacles, clear_zones = _cy_cluster_lidar(
                        arr, sensor_input.velocity_ms, DetectedObstacle, ClearZone
                    )
                else:
                    obstacles, clear_zones = _cluster_lidar(
                        sensor_input.lidar_distances_m,
                        sensor_input.velocity_ms,
                    )
                e_stop = any(o.distance_m < EMERGENCY_DIST_M for o in obstacles)
                action = _recommend(obstacles, clear_zones)

            elapsed = time.perf_counter() - t0
            analysis_latency.observe(elapsed)
            frames_processed.inc()
            for o in obstacles:
                obstacles_found.labels(threat=o.classification).inc()
            if e_stop:
                emergency_stops.inc()
            clear_zones_gauge.set(len(clear_zones))

            span.set_attribute("obstacles.count",   len(obstacles))
            span.set_attribute("clear_zones.count", len(clear_zones))
            span.set_attribute("emergency_stop",    e_stop)

            return ObstacleAnalysis(
                rover_id           = sensor_input.rover_id,
                timestamp_ms       = sensor_input.timestamp_ms,
                obstacles          = obstacles,
                clear_zones        = clear_zones,
                emergency_stop     = e_stop,
                recommended_action = action,
                processing_time_ms = elapsed * 1000,
            )

    async def stream_analysis(
        self, request: AsyncIterator[SensorInput]
    ) -> AsyncIterator[ObstacleAnalysis]:
        async for sensor_input in request:
            yield await self.analyze(sensor_input)


# ── entrypoint ────────────────────────────────────────────────────────────────
async def main() -> None:
    global _tracer
    start_http_server(METRICS_PORT)
    _tracer = setup_otel()

    server = grpclib.server.Server([ObstacleDetectorService()])
    await server.start("0.0.0.0", GRPC_PORT)
    log.info("ObstacleDetector gRPC server listening on :%d", GRPC_PORT)
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
