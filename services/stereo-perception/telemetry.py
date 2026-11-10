"""
Jetson hardware telemetry collection and Kafka publishing.

Collects system metrics from the Jetson Nano using:
  - psutil  — CPU usage, memory, disk
  - /sys/class/thermal/ — CPU and GPU temperatures
  - /sys/devices/gpu.0/ — GPU utilisation (Jetson-specific sysfs node)
  - tegrastats fallback — if sysfs nodes are unavailable

Publishes compact SensorData protobuf messages to Kafka.  Telemetry publishing
runs on a separate thread from the control loop so that Kafka backpressure or
network issues can never block obstacle avoidance.

The producer uses confluent-kafka with a small queue and acks=1 (leader only)
to prioritise throughput over durability.  Individual telemetry frames are
expendable; the rover must not wait on Kafka delivery.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Add project root so generated/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../.."))

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    log.warning("psutil not installed; CPU/memory metrics will be 0.0")

try:
    from confluent_kafka import Producer as _KafkaProducer, KafkaError
    _KAFKA = True
except ImportError:
    _KAFKA = False
    log.warning("confluent-kafka not installed; telemetry will not be published to Kafka")

from generated.telemetry.v1.telemetry_pb2 import SensorData, RoverStatus


# ── Jetson hardware metrics ───────────────────────────────────────────────────

@dataclass
class JetsonMetrics:
    cpu_usage_percent:    float = 0.0
    memory_usage_percent: float = 0.0
    gpu_usage_percent:    float = 0.0
    cpu_temp_celsius:     float = 0.0
    gpu_temp_celsius:     float = 0.0
    disk_usage_percent:   float = 0.0


# Jetson sysfs paths
_THERMAL_ZONES = "/sys/class/thermal"
_GPU_LOAD_PATH = "/sys/devices/platform/host1x/gpu.0/load"
_GPU_LOAD_ALT  = "/sys/devices/17000000.ga10b/load"  # Orin variant


def _read_sysfs_int(path: str, default: float = 0.0) -> float:
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return default


def _read_thermal_zones() -> dict[str, float]:
    """Read all /sys/class/thermal/thermal_zoneN/temp entries (millidegrees C)."""
    temps: dict[str, float] = {}
    if not os.path.exists(_THERMAL_ZONES):
        return temps
    for zone in os.listdir(_THERMAL_ZONES):
        if not zone.startswith("thermal_zone"):
            continue
        type_path = os.path.join(_THERMAL_ZONES, zone, "type")
        temp_path = os.path.join(_THERMAL_ZONES, zone, "temp")
        try:
            zone_type = open(type_path).read().strip()
            temp_mc   = float(open(temp_path).read().strip())
            temps[zone_type] = temp_mc / 1000.0  # millidegrees → degrees C
        except (OSError, ValueError):
            pass
    return temps


def _read_gpu_usage() -> float:
    """Read GPU utilisation from Jetson sysfs (returns 0-100 float)."""
    for path in (_GPU_LOAD_PATH, _GPU_LOAD_ALT):
        val = _read_sysfs_int(path, default=-1.0)
        if val >= 0:
            return val
    return 0.0


def collect_jetson_metrics() -> JetsonMetrics:
    m = JetsonMetrics()

    if _PSUTIL:
        m.cpu_usage_percent    = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        m.memory_usage_percent = vm.percent
        disk = psutil.disk_usage("/")
        m.disk_usage_percent   = disk.percent

    temps = _read_thermal_zones()
    # Common Jetson thermal zone names
    for name in ("CPU-therm", "CPU", "cpu-thermal", "Tboard_tegra", "AO-therm"):
        if name in temps:
            m.cpu_temp_celsius = temps[name]
            break
    for name in ("GPU-therm", "GPU", "gpu-thermal"):
        if name in temps:
            m.gpu_temp_celsius = temps[name]
            break

    m.gpu_usage_percent = _read_gpu_usage()
    return m


# ── Shared rover state (written by control loop, read by telemetry thread) ────

@dataclass
class RoverState:
    """Live rover state used to populate telemetry frames."""
    # Populated by control loop
    commanded_direction:  str   = "STOP"
    nearest_obstacle_m:   float = float("nan")
    left_clearance_m:     float = float("nan")
    center_clearance_m:   float = float("nan")
    right_clearance_m:    float = float("nan")
    obstacle_detected:    bool  = False
    planner_state:        str   = "INITIALISING"
    rover_status:         RoverStatus = RoverStatus.ROVER_STATUS_NOMINAL
    error_code:           str   = ""
    # Populated by perception thread
    camera_status:        str   = "UNKNOWN"
    camera_fps:           float = 0.0
    depth_fps:            float = 0.0
    depth_latency_ms:     float = 0.0
    valid_depth_fraction: float = 0.0
    # Populated by telemetry thread itself
    sequence_number:      int   = 0


# ── Kafka telemetry publisher ─────────────────────────────────────────────────

class TelemetryPublisher:
    """
    Publishes compact SensorData protobuf frames to a Kafka topic.

    All publish operations are non-blocking (fire-and-forget).  The Kafka
    producer internal queue absorbs bursts; if the queue is full the frame
    is silently dropped rather than blocking the caller.
    """

    def __init__(self, config) -> None:
        self._topic     = config.KAFKA_TELEMETRY_TOPIC
        self._rover_id  = config.ROVER_ID
        self._producer  = None

        if _KAFKA:
            try:
                self._producer = _KafkaProducer({
                    "bootstrap.servers":            config.KAFKA_BOOTSTRAP_SERVERS,
                    "queue.buffering.max.messages": 1_000,
                    "queue.buffering.max.kbytes":   8_192,
                    "batch.num.messages":           100,
                    "linger.ms":                    50,
                    "compression.codec":            "lz4",
                    "acks":                         "1",
                    "socket.timeout.ms":            3_000,
                    "message.timeout.ms":           5_000,
                })
                log.info("Kafka producer initialised: brokers=%s topic=%s",
                         config.KAFKA_BOOTSTRAP_SERVERS, self._topic)
            except Exception as exc:
                log.warning("Kafka producer init failed: %s — telemetry will be local only", exc)
                self._producer = None

    def publish(self, state: RoverState) -> None:
        """Build and publish one SensorData frame. Non-blocking."""
        if self._producer is None:
            return

        metrics = collect_jetson_metrics()
        import math
        frame = SensorData(
            rover_id             = self._rover_id,
            timestamp_ms         = int(time.time() * 1000),
            battery_percent      = 100.0,        # populate from hardware if available
            cpu_temp_celsius     = metrics.cpu_temp_celsius,
            motor_temp_celsius   = 0.0,          # read from motor driver if available
            motor_load_left      = 0.0,
            motor_load_right     = 0.0,
            obstacle_distances_m = [],           # empty — stereo provides regional depths
            velocity_ms          = 0.0,
            heading_degrees      = 0.0,
            status               = state.rover_status,
            error_code           = state.error_code,
            sequence_number      = state.sequence_number,
            # Stereo-specific fields
            nearest_obstacle_m   = state.nearest_obstacle_m if math.isfinite(state.nearest_obstacle_m) else 0.0,
            left_clearance_m     = state.left_clearance_m   if math.isfinite(state.left_clearance_m)   else 0.0,
            center_clearance_m   = state.center_clearance_m if math.isfinite(state.center_clearance_m) else 0.0,
            right_clearance_m    = state.right_clearance_m  if math.isfinite(state.right_clearance_m)  else 0.0,
            obstacle_detected    = state.obstacle_detected,
            commanded_direction  = state.commanded_direction,
            camera_status        = state.camera_status,
            depth_fps            = state.depth_fps,
            camera_fps           = state.camera_fps,
            gpu_usage_percent    = metrics.gpu_usage_percent,
            memory_usage_percent = metrics.memory_usage_percent,
            gpu_temp_celsius     = metrics.gpu_temp_celsius,
            planner_state        = state.planner_state,
            depth_latency_ms     = state.depth_latency_ms,
            valid_depth_fraction = state.valid_depth_fraction,
        )

        try:
            self._producer.produce(
                topic    = self._topic,
                key      = self._rover_id.encode(),
                value    = bytes(frame),
                on_delivery = self._on_delivery,
            )
            self._producer.poll(0)   # trigger delivery callbacks without blocking
        except Exception as exc:
            log.debug("Kafka produce failed (non-fatal): %s", exc)

    def flush(self, timeout: float = 5.0) -> None:
        if self._producer:
            self._producer.flush(timeout=timeout)

    @staticmethod
    def _on_delivery(err: Optional[KafkaError], _msg) -> None:
        if err:
            log.debug("Telemetry delivery failed (non-fatal): %s", err)
