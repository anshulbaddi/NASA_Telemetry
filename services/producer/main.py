"""
Rover fleet telemetry producer.

TELEMETRY_MODE=simulation (default):
  Simulates NUM_ROVERS rovers at PUBLISH_RATE_HZ with realistic random faults.
  Use this for development and ground-station testing without the physical rover.

TELEMETRY_MODE=hardware:
  Publishes one real telemetry frame for the local Jetson Nano.
  System metrics (CPU, GPU, temperature) are read from the Jetson sysfs.
  Perception data (obstacle clearances, depth FPS, etc.) must be provided
  by the stereo-perception service which writes to HARDWARE_STATE_FILE.

  In practice, the stereo-perception service (services/stereo-perception/main.py)
  handles Kafka publishing directly in hardware mode; this producer operates as
  a supplementary system-metrics publisher only.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from confluent_kafka import Producer, KafkaError
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, start_http_server

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generated.telemetry.v1.telemetry_pb2 import SensorData, RoverStatus

# ── config ────────────────────────────────────────────────────────────────────
KAFKA_BROKERS      = os.getenv("KAFKA_BROKERS",     "localhost:9092")
KAFKA_TOPIC        = os.getenv("KAFKA_TOPIC",        "rover.telemetry")
NUM_ROVERS         = int(os.getenv("NUM_ROVERS",     "10"))
PUBLISH_RATE_HZ    = float(os.getenv("PUBLISH_RATE_HZ", "200"))  # per rover
METRICS_PORT       = int(os.getenv("METRICS_PORT",   "9090"))
OTLP_ENDPOINT      = os.getenv("OTLP_ENDPOINT",     "http://localhost:4317")
TELEMETRY_MODE     = os.getenv("TELEMETRY_MODE",    "simulation")  # simulation | hardware
ROVER_ID           = os.getenv("ROVER_ID",           "PHYSICAL-ROVER-001")
HARDWARE_RATE_HZ   = float(os.getenv("HARDWARE_RATE_HZ", "2.0"))  # hz for hardware mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── metrics ───────────────────────────────────────────────────────────────────
messages_published = Counter("producer_messages_published_total", "Total protobuf frames published", ["rover_id"])
publish_errors     = Counter("producer_publish_errors_total",     "Kafka delivery errors",           ["rover_id"])
publish_latency    = Histogram("producer_publish_latency_seconds", "Kafka produce call latency",
                               buckets=[.0001, .0005, .001, .005, .01, .05, .1])

# ── OTel ──────────────────────────────────────────────────────────────────────
def setup_otel() -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("rover.producer")

# ── rover simulator ───────────────────────────────────────────────────────────
@dataclass
class RoverState:
    rover_id:         str
    latitude:         float = 0.0
    longitude:        float = 0.0
    altitude_m:       float = 0.0
    heading_degrees:  float = 0.0
    velocity_ms:      float = 0.0
    battery_percent:  float = 100.0
    cpu_temp_celsius: float = 45.0
    motor_temp_celsius: float = 30.0
    motor_load_left:  float = 0.0
    motor_load_right: float = 0.0
    sequence_number:  int   = 0
    status:           RoverStatus = RoverStatus.ROVER_STATUS_NOMINAL
    error_code:       str  = ""

    # simulation internals
    _t: float = field(default_factory=time.time, repr=False)
    _phase: float = field(default_factory=lambda: random.uniform(0, 2 * math.pi), repr=False)
    _fault_countdown: int = field(default_factory=lambda: random.randint(500, 2000), repr=False)

    def step(self) -> None:
        dt = 0.005  # 200 Hz → 5 ms per tick
        self._t += dt
        self.sequence_number += 1

        # sinusoidal path through notional terrain (deg)
        self.latitude  += math.cos(self._t * 0.1 + self._phase) * 0.000005
        self.longitude += math.sin(self._t * 0.1 + self._phase) * 0.000005
        self.altitude_m = 100.0 + 20.0 * math.sin(self._t * 0.05)
        self.heading_degrees = (self.heading_degrees + random.gauss(0, 0.3)) % 360
        self.velocity_ms = max(0.0, min(5.0, 2.0 + math.sin(self._t * 0.2 + self._phase)))

        # battery drains ~0.01% per second
        self.battery_percent = max(0.0, self.battery_percent - 0.00005)

        # temps fluctuate
        load = 0.5 + 0.4 * math.sin(self._t * 0.3 + self._phase)
        self.motor_load_left  = max(0.0, min(1.0, load + random.gauss(0, 0.02)))
        self.motor_load_right = max(0.0, min(1.0, load + random.gauss(0, 0.02)))
        self.cpu_temp_celsius   = 45.0 + 15.0 * load + random.gauss(0, 0.5)
        self.motor_temp_celsius = 30.0 + 40.0 * load + random.gauss(0, 1.0)

        # stochastic fault injection
        self._fault_countdown -= 1
        if self._fault_countdown <= 0:
            self._inject_fault()
            self._fault_countdown = random.randint(500, 2000)
        elif self.status != RoverStatus.ROVER_STATUS_NOMINAL and random.random() < 0.01:
            self.status     = RoverStatus.ROVER_STATUS_RECOVERED
            self.error_code = ""

    def _inject_fault(self) -> None:
        severity = random.choices(
            [RoverStatus.ROVER_STATUS_WARNING,
             RoverStatus.ROVER_STATUS_CRITICAL,
             RoverStatus.ROVER_STATUS_FAULT],
            weights=[0.6, 0.3, 0.1],
        )[0]
        fault_codes = {
            RoverStatus.ROVER_STATUS_WARNING:  ["BATT_LOW", "TEMP_HIGH", "LOAD_HIGH"],
            RoverStatus.ROVER_STATUS_CRITICAL: ["MOTOR_OVERLOAD", "CPU_OVERHEAT", "COMPASS_DRIFT"],
            RoverStatus.ROVER_STATUS_FAULT:    ["MOTOR_STALL", "ENCODER_FAIL", "COMMS_LOSS"],
        }
        self.status     = severity
        self.error_code = random.choice(fault_codes[severity])

    def lidar_scan(self) -> List[float]:
        """72-point LIDAR sweep (5° resolution) with random obstacles."""
        distances = []
        for i in range(72):
            angle = i * 5.0
            base  = 10.0 + 5.0 * math.sin(math.radians(angle + self._t * 20))
            noise = random.gauss(0, 0.2)
            # random near obstacle
            if random.random() < 0.03:
                distances.append(max(0.5, random.uniform(0.5, 2.0)))
            else:
                distances.append(max(0.1, base + noise))
        return distances

    def to_proto(self) -> SensorData:
        return SensorData(
            rover_id             = self.rover_id,
            timestamp_ms         = int(time.time() * 1000),
            latitude             = self.latitude,
            longitude            = self.longitude,
            altitude_m           = self.altitude_m,
            battery_percent      = self.battery_percent,
            cpu_temp_celsius     = self.cpu_temp_celsius,
            motor_temp_celsius   = self.motor_temp_celsius,
            motor_load_left      = self.motor_load_left,
            motor_load_right     = self.motor_load_right,
            obstacle_distances_m = self.lidar_scan(),
            velocity_ms          = self.velocity_ms,
            heading_degrees      = self.heading_degrees,
            status               = self.status,
            error_code           = self.error_code,
            sequence_number      = self.sequence_number,
        )


# ── Kafka producer ────────────────────────────────────────────────────────────
def make_kafka_producer() -> Producer:
    return Producer({
        "bootstrap.servers":          KAFKA_BROKERS,
        "queue.buffering.max.messages": 500_000,
        "queue.buffering.max.kbytes":  512_000,
        "batch.num.messages":          10_000,
        "linger.ms":                   5,
        "compression.codec":           "lz4",
        "acks":                        "1",
        "retries":                     3,
        "retry.backoff.ms":            100,
    })


def delivery_callback(err: Optional[KafkaError], msg: object, rover_id: str) -> None:
    if err:
        publish_errors.labels(rover_id=rover_id).inc()
        log.warning("Delivery failed rover=%s err=%s", rover_id, err)
    else:
        messages_published.labels(rover_id=rover_id).inc()


# ── main loop ─────────────────────────────────────────────────────────────────
async def run_rover(state: RoverState, producer: Producer, tracer: trace.Tracer) -> None:
    interval = 1.0 / PUBLISH_RATE_HZ
    loop     = asyncio.get_event_loop()

    while True:
        t0 = loop.time()
        state.step()
        frame = state.to_proto()
        payload = bytes(frame)

        with tracer.start_as_current_span("rover.publish") as span:
            span.set_attribute("rover.id",        state.rover_id)
            span.set_attribute("rover.seq",       state.sequence_number)
            span.set_attribute("rover.status",    str(state.status))
            span.set_attribute("payload.bytes",   len(payload))

            t_produce = loop.time()
            producer.produce(
                topic    = KAFKA_TOPIC,
                key      = state.rover_id.encode(),
                value    = payload,
                on_delivery = lambda err, msg, rid=state.rover_id: delivery_callback(err, msg, rid),
            )
            publish_latency.observe(loop.time() - t_produce)

        # poll for delivery callbacks every batch
        if state.sequence_number % 100 == 0:
            producer.poll(0)

        elapsed = loop.time() - t0
        await asyncio.sleep(max(0, interval - elapsed))


async def flush_loop(producer: Producer) -> None:
    """Periodically flush to bound in-flight buffer."""
    while True:
        await asyncio.sleep(1.0)
        producer.flush(timeout=2.0)


async def run_hardware_rover(producer: Producer, tracer: trace.Tracer) -> None:
    """
    Hardware telemetry loop: publish real Jetson system metrics.

    The stereo-perception service also publishes to Kafka directly; this
    function publishes lightweight system-only frames as a heartbeat and
    for ground-station monitoring when stereo-perception is not running.
    """
    try:
        import psutil
        _psutil = True
    except ImportError:
        _psutil = False

    def _read_sysfs_temp(zone_names: List[str]) -> float:
        base = "/sys/class/thermal"
        if not os.path.exists(base):
            return 0.0
        for zone in os.listdir(base):
            tp = os.path.join(base, zone, "type")
            vp = os.path.join(base, zone, "temp")
            try:
                if open(tp).read().strip() in zone_names:
                    return float(open(vp).read().strip()) / 1000.0
            except (OSError, ValueError):
                pass
        return 0.0

    seq = 0
    interval = 1.0 / HARDWARE_RATE_HZ
    log.info("Hardware producer: rover_id=%s @ %.1f Hz → %s/%s",
             ROVER_ID, HARDWARE_RATE_HZ, KAFKA_BROKERS, KAFKA_TOPIC)

    while True:
        cpu_temp  = _read_sysfs_temp(["CPU-therm", "CPU", "cpu-thermal"])
        cpu_pct   = psutil.cpu_percent(interval=None) if _psutil else 0.0
        mem_pct   = psutil.virtual_memory().percent   if _psutil else 0.0

        frame = SensorData(
            rover_id             = ROVER_ID,
            timestamp_ms         = int(time.time() * 1000),
            cpu_temp_celsius     = cpu_temp,
            memory_usage_percent = mem_pct,
            status               = RoverStatus.ROVER_STATUS_NOMINAL,
            sequence_number      = seq,
        )
        seq += 1
        payload = bytes(frame)

        with tracer.start_as_current_span("rover.publish.hardware") as span:
            span.set_attribute("rover.id",  ROVER_ID)
            span.set_attribute("mode",      "hardware")
            t_produce = asyncio.get_event_loop().time()
            producer.produce(
                topic    = KAFKA_TOPIC,
                key      = ROVER_ID.encode(),
                value    = payload,
                on_delivery = lambda err, msg: (
                    messages_published.labels(rover_id=ROVER_ID).inc() if not err
                    else publish_errors.labels(rover_id=ROVER_ID).inc()
                ),
            )
            publish_latency.observe(asyncio.get_event_loop().time() - t_produce)

        producer.poll(0)
        await asyncio.sleep(interval)


async def main() -> None:
    start_http_server(METRICS_PORT)
    tracer   = setup_otel()
    producer = make_kafka_producer()

    if TELEMETRY_MODE == "hardware":
        log.info("TELEMETRY_MODE=hardware: publishing real Jetson system metrics")
        log.info("Note: stereo-perception service also publishes perception data independently")
        tasks = [
            asyncio.create_task(run_hardware_rover(producer, tracer)),
            asyncio.create_task(flush_loop(producer)),
        ]
    else:
        rovers = [
            RoverState(
                rover_id  = f"ROVER-{i:03d}",
                latitude  = 18.5 + random.uniform(-0.5, 0.5),
                longitude = 77.5 + random.uniform(-0.5, 0.5),
                altitude_m = random.uniform(80, 120),
                _phase    = random.uniform(0, 2 * math.pi),
            )
            for i in range(NUM_ROVERS)
        ]
        log.info("Starting producer: %d rovers @ %.0f Hz → %s/%s",
                 NUM_ROVERS, PUBLISH_RATE_HZ, KAFKA_BROKERS, KAFKA_TOPIC)
        tasks = [asyncio.create_task(run_rover(r, producer, tracer)) for r in rovers]
        tasks.append(asyncio.create_task(flush_loop(producer)))

    try:
        await asyncio.gather(*tasks)
    finally:
        producer.flush(timeout=10.0)


if __name__ == "__main__":
    asyncio.run(main())
