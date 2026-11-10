"""
Stereo-perception pipeline — main entry point for the Jetson Nano rover computer.

This service runs on the physical rover and owns the real-time perception and
control loop.  Ground-station services (Kafka, Prometheus, API gateway) are
reached over the network.

Pipeline architecture (four threads):

  Thread 1 — CAMERA:
    camera → frame_queue (bounded, drop-old-on-full)

  Thread 2 — PERCEPTION:
    frame_queue → rectify → disparity → depth → obstacle_map
    → perception_queue (bounded, drop-old-on-full)

  Thread 3 — CONTROL (critical path):
    perception_queue → safety.evaluate() → motor_client.send()
    Safety is enforced here; this thread is never blocked by Kafka.

  Thread 4 — TELEMETRY:
    shared_state → JetsonMetrics → SensorData proto → Kafka (non-blocking)

The control loop is isolated from telemetry/network operations by design.
If the telemetry thread is slow or Kafka is unreachable, the rover keeps
navigating safely.

Environment variables: see config.py for all configurable values.
Run with: python3 main.py
"""
from __future__ import annotations

import copy
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from typing import Optional, Tuple

import config  # local config.py

# ── Logging setup (before other local imports) ────────────────────────────────
logging.basicConfig(
    level   = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format  = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s — %(message)s",
)
log = logging.getLogger("stereo-perception")

# ── Local module imports ──────────────────────────────────────────────────────
from camera      import create_camera_source, FramePair
from calibration import load_calibration, make_dummy_calibration, CalibrationData
from rectification import StereoRectifier
from disparity   import DisparityComputer
from depth       import DepthEstimator, DepthResult
from obstacle_map import ObstacleAnalyzer, ObstacleMap
from safety      import SafetyMonitor, SafetyState
from telemetry   import TelemetryPublisher, RoverState, collect_jetson_metrics
from motor_client import MotorCommandClient

# ── Prometheus metrics ────────────────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, start_http_server
    )
    _PROM = True
except ImportError:
    _PROM = False
    log.warning("prometheus_client not available; metrics endpoint disabled")

if _PROM:
    cam_fps_gauge        = Gauge("stereo_camera_fps",              "Camera capture FPS")
    depth_fps_gauge      = Gauge("stereo_processing_fps",          "Depth processing FPS")
    depth_latency_hist   = Histogram("stereo_processing_latency_ms", "Depth pipeline latency ms",
                                     buckets=[10, 20, 30, 50, 75, 100, 150, 200, 500])
    valid_depth_gauge    = Gauge("stereo_valid_depth_fraction",    "Valid depth pixel fraction")
    nearest_obs_gauge    = Gauge("stereo_nearest_obstacle_m",      "Nearest obstacle metres")
    left_clear_gauge     = Gauge("stereo_left_clearance_m",        "Left clearance metres")
    center_clear_gauge   = Gauge("stereo_center_clearance_m",      "Center clearance metres")
    right_clear_gauge    = Gauge("stereo_right_clearance_m",       "Right clearance metres")
    camera_fail_counter  = Counter("stereo_camera_failures_total", "Camera read failures", ["side"])
    estop_counter        = Counter("stereo_emergency_stops_total", "Emergency stop events")
    jetson_cpu_gauge     = Gauge("jetson_cpu_usage_percent",       "Jetson CPU usage")
    jetson_mem_gauge     = Gauge("jetson_memory_usage_percent",    "Jetson memory usage")
    jetson_temp_gauge    = Gauge("jetson_cpu_temp_celsius",        "Jetson CPU temperature")
    jetson_gpu_gauge     = Gauge("jetson_gpu_usage_percent",       "Jetson GPU usage")


def _update_prom_gauges(omap: Optional[ObstacleMap], fps_cam: float,
                        fps_dep: float, latency_ms: float, vfrac: float) -> None:
    if not _PROM:
        return
    cam_fps_gauge.set(fps_cam)
    depth_fps_gauge.set(fps_dep)
    valid_depth_gauge.set(vfrac)
    if fps_dep > 0:
        depth_latency_hist.observe(latency_ms)
    if omap:
        nearest_obs_gauge.set(omap.nearest_m if math.isfinite(omap.nearest_m) else -1)
        left_clear_gauge.set(omap.left.nearest_m   if math.isfinite(omap.left.nearest_m)   else -1)
        center_clear_gauge.set(omap.center.nearest_m if math.isfinite(omap.center.nearest_m) else -1)
        right_clear_gauge.set(omap.right.nearest_m if math.isfinite(omap.right.nearest_m)   else -1)


# ── FPS counter helper ────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window: float = 2.0) -> None:
        self._window = window
        self._times: list[float] = []

    def tick(self) -> None:
        now = time.monotonic()
        self._times.append(now)
        cutoff = now - self._window
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


# ── Main pipeline class ───────────────────────────────────────────────────────
class StereoPipeline:

    def __init__(self) -> None:
        # Calibration
        calib = load_calibration(config.CALIBRATION_FILE)
        if calib is None:
            if config.REQUIRE_CALIBRATION:
                raise RuntimeError(
                    f"Calibration file not found: {config.CALIBRATION_FILE!r}. "
                    f"Run scripts/calibrate_stereo.py to generate it, or set "
                    f"REQUIRE_CALIBRATION=false to continue without calibration."
                )
            log.warning(
                "No calibration file found; using dummy calibration. "
                "Depth values will be APPROXIMATE. "
                "Run scripts/calibrate_stereo.py to calibrate."
            )
            calib = make_dummy_calibration(
                width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT
            )

        self._calib = calib

        # Pipeline components
        self._camera       = create_camera_source(config)
        self._rectifier    = StereoRectifier(calib)
        # OAK-D Lite hardware-rectifies frames before delivering them to us,
        # so we skip the StereoRectifier step in the perception thread.
        self._skip_rectify = config.CAMERA_TYPE.lower() == "oak"
        self._disparity = DisparityComputer(config)
        self._depth_est = DepthEstimator(calib, config.MIN_DEPTH_M, config.MAX_DEPTH_M)
        self._obstacle  = ObstacleAnalyzer(config)
        self._safety    = SafetyMonitor(config)
        self._telemetry = TelemetryPublisher(config)
        self._motor     = MotorCommandClient(config)

        # Bounded inter-thread queues (old frames dropped to maintain freshness)
        self._frame_queue:      queue.Queue[FramePair]                     = queue.Queue(maxsize=config.CAMERA_QUEUE_SIZE)
        self._perception_queue: queue.Queue[Tuple[ObstacleMap, DepthResult]] = queue.Queue(maxsize=config.PERCEPTION_QUEUE_SIZE)

        # Shared state (written by control/perception, read by telemetry)
        self._state_lock = threading.Lock()
        self._state      = RoverState()

        # Shutdown
        self._stop = threading.Event()

        # FPS counters
        self._cam_fps  = FPSCounter()
        self._dep_fps  = FPSCounter()

        # Prometheus metrics server
        if _PROM:
            try:
                start_http_server(config.METRICS_PORT)
                log.info("Prometheus metrics on :%d", config.METRICS_PORT)
            except Exception as exc:
                log.warning("Could not start metrics server: %s", exc)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start all pipeline threads and block until shutdown.

        Registers SIGINT/SIGTERM handlers so Ctrl+C or systemctl stop will
        perform a clean shutdown (EMERGENCY_STOP then thread join).
        """
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Open camera before starting threads
        try:
            self._camera.open()
        except Exception as exc:
            log.critical("Camera open failed: %s", exc)
            sys.exit(1)

        # Start motor client async loop
        self._motor.start()

        threads = [
            threading.Thread(target=self._camera_thread,     name="camera",     daemon=True),
            threading.Thread(target=self._perception_thread, name="perception", daemon=True),
            threading.Thread(target=self._control_thread,    name="control",    daemon=True),
            threading.Thread(target=self._telemetry_thread,  name="telemetry",  daemon=True),
        ]
        for t in threads:
            t.start()
        log.info("Stereo pipeline started (rover_id=%s camera=%s)",
                 config.ROVER_ID, config.CAMERA_TYPE)

        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        finally:
            log.info("Shutting down ...")
            self._stop.set()
            self._motor.stop()
            for t in threads:
                t.join(timeout=5.0)
            self._camera.close()
            self._telemetry.flush()
            log.info("Shutdown complete")

    def _handle_signal(self, sig, _frame) -> None:
        log.info("Signal %s received — initiating shutdown", sig)
        self._stop.set()

    # ── Thread 1: Camera capture ──────────────────────────────────────────────

    def _camera_thread(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            pair = self._camera.read_pair()
            if pair is None:
                consecutive_failures += 1
                if _PROM:
                    camera_fail_counter.labels(side="both").inc()
                log.warning("Camera read failure #%d", consecutive_failures)
                if consecutive_failures >= 30:
                    log.error("30 consecutive camera failures — signalling shutdown")
                    with self._state_lock:
                        self._state.camera_status = "BOTH_FAIL"
                        self._state.rover_status   = RoverStatus.ROVER_STATUS_FAULT
                        self._state.error_code     = "CAMERA_FAIL"
                    self._motor.send("EMERGENCY_STOP")
                    self._stop.set()
                    return
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            self._cam_fps.tick()

            # Check individual camera health
            left_ok, right_ok = self._camera.is_healthy()
            if not left_ok or not right_ok:
                status = "LEFT_FAIL" if not left_ok else "RIGHT_FAIL"
                with self._state_lock:
                    self._state.camera_status = status
            else:
                with self._state_lock:
                    self._state.camera_status = "OK"
                    self._state.camera_fps    = self._cam_fps.fps

            # Drop old frame if queue is full (prefer freshest frame)
            self._queue_put_nowait(self._frame_queue, pair)

    # ── Thread 2: Stereo processing ───────────────────────────────────────────

    def _perception_thread(self) -> None:
        frame_count = 0
        while not self._stop.is_set():
            try:
                pair = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                if self._skip_rectify:
                    left_rect, right_rect = pair.left, pair.right   # OAK already rectified
                else:
                    left_rect, right_rect = self._rectifier.rectify(pair.left, pair.right)
                disparity             = self._disparity.compute(left_rect, right_rect)
                depth_result          = self._depth_est.compute(disparity)
                obstacle_map          = self._obstacle.analyze(depth_result)
            except Exception as exc:
                log.error("Perception pipeline error: %s", exc)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._dep_fps.tick()

            with self._state_lock:
                self._state.depth_fps            = self._dep_fps.fps
                self._state.depth_latency_ms     = elapsed_ms
                self._state.valid_depth_fraction = depth_result.valid_fraction

            _update_prom_gauges(
                obstacle_map, self._cam_fps.fps, self._dep_fps.fps,
                elapsed_ms, depth_result.valid_fraction,
            )

            self._queue_put_nowait(self._perception_queue, (obstacle_map, depth_result))

            # Optional debug output
            frame_count += 1
            if config.DEBUG_SAVE and frame_count % config.DEBUG_EVERY_N_FRAMES == 0:
                try:
                    self._rectifier.save_debug_frame(
                        left_rect, right_rect, config.DEBUG_DIR, frame_count
                    )
                except Exception:
                    pass

    # ── Thread 3: Control loop (critical path) ────────────────────────────────

    def _control_thread(self) -> None:
        """
        Safety-critical control loop.

        This thread must NEVER be blocked by Kafka, network I/O, or disk I/O.
        All network operations are on separate threads.
        """
        last_map:       Optional[ObstacleMap] = None
        last_map_time:  float                 = 0.0
        control_period = 1.0 / config.CONTROL_RATE_HZ

        while not self._stop.is_set():
            t0 = time.perf_counter()

            # Non-blocking drain of perception queue; only keep latest
            latest_map: Optional[ObstacleMap] = None
            latest_depth: Optional[DepthResult] = None
            while True:
                try:
                    latest_map, latest_depth = self._perception_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_map is not None:
                last_map      = latest_map
                last_map_time = time.time()

            vision_age = (time.time() - last_map_time) if last_map_time > 0 else float("inf")

            # Desired direction from obstacle map (reactive navigation)
            desired = last_map.recommended_direction(
                stop_dist_m=config.STOP_DISTANCE_M,
                warning_dist_m=config.WARNING_DISTANCE_M,
            ) if last_map else "STOP"

            # Safety layer overrides
            safety_state = self._safety.evaluate(last_map, vision_age, desired)

            # Send motor command
            self._motor.send(safety_state.commanded_direction)

            if safety_state.emergency_stop and _PROM:
                estop_counter.inc()

            # Update shared state for telemetry thread
            with self._state_lock:
                self._state.commanded_direction  = safety_state.commanded_direction
                self._state.planner_state        = (
                    "OBSTACLE_AVOIDANCE" if safety_state.commanded_direction != "FORWARD"
                    else "NOMINAL"
                )
                if last_map is not None:
                    self._state.nearest_obstacle_m   = last_map.nearest_m
                    self._state.left_clearance_m     = last_map.left.nearest_m
                    self._state.center_clearance_m   = last_map.center.nearest_m
                    self._state.right_clearance_m    = last_map.right.nearest_m
                    self._state.obstacle_detected    = (
                        math.isfinite(last_map.nearest_m) and last_map.nearest_m < 5.0
                    )
                    self._state.rover_status = (
                        RoverStatus.ROVER_STATUS_FAULT
                        if safety_state.emergency_stop
                        else RoverStatus.ROVER_STATUS_NOMINAL
                    )

            # Maintain control loop rate
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, control_period - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Thread 4: Telemetry publishing ────────────────────────────────────────

    def _telemetry_thread(self) -> None:
        interval = 1.0 / config.TELEMETRY_RATE_HZ
        # Update system metrics Prometheus gauges periodically
        metrics_interval = 5.0
        last_metrics_time = 0.0

        while not self._stop.is_set():
            with self._state_lock:
                state_copy = copy.copy(self._state)
                self._state.sequence_number += 1

            self._telemetry.publish(state_copy)

            # Update Jetson system gauges less frequently (expensive)
            if _PROM and time.time() - last_metrics_time > metrics_interval:
                m = collect_jetson_metrics()
                jetson_cpu_gauge.set(m.cpu_usage_percent)
                jetson_mem_gauge.set(m.memory_usage_percent)
                jetson_temp_gauge.set(m.cpu_temp_celsius)
                jetson_gpu_gauge.set(m.gpu_usage_percent)
                last_metrics_time = time.time()

            time.sleep(interval)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _queue_put_nowait(q: queue.Queue, item) -> None:
        """Put item, discarding the oldest if the queue is full."""
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass


# Need RoverStatus imported for the control thread
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from generated.telemetry.v1.telemetry_pb2 import RoverStatus


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = StereoPipeline()
    pipeline.run()
