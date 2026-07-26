"""Rover telemetry consumer — deserializes protobuf frames, fans out to gRPC services."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import grpclib.client
from aiokafka import AIOKafkaConsumer
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge, start_http_server

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generated.telemetry.v1.telemetry_pb2        import SensorData, RoverStatus
from generated.obstacledetector.v1.obstacle_detector_pb2 import (
    ObstacleDetectorStub, SensorInput,
)
from generated.motorcontroller.v1.motor_controller_pb2 import (
    MotorControllerStub, DriveCommand,
)
from generated.fdir.v1.fdir_pb2 import FDIRServiceStub, FDIRStatusRequest

# ── config ────────────────────────────────────────────────────────────────────
KAFKA_BROKERS         = os.getenv("KAFKA_BROKERS",           "localhost:9092")
KAFKA_TOPIC           = os.getenv("KAFKA_TOPIC",              "rover.telemetry")
KAFKA_GROUP           = os.getenv("KAFKA_GROUP",              "rover-consumer")
OBSTACLE_ADDR         = os.getenv("OBSTACLE_DETECTOR_ADDR",  "localhost:50051")
MOTOR_ADDR            = os.getenv("MOTOR_CONTROLLER_ADDR",   "localhost:50052")
FDIR_ADDR             = os.getenv("FDIR_ADDR",               "localhost:50053")
METRICS_PORT          = int(os.getenv("METRICS_PORT",         "9091"))
OTLP_ENDPOINT         = os.getenv("OTLP_ENDPOINT",            "http://localhost:4317")
CONCURRENCY           = int(os.getenv("CONSUMER_CONCURRENCY", "32"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── metrics ───────────────────────────────────────────────────────────────────
frames_consumed   = Counter("consumer_frames_consumed_total",    "Protobuf frames consumed", ["rover_id"])
frames_errors     = Counter("consumer_frame_errors_total",       "Deserialization errors")
grpc_calls        = Counter("consumer_grpc_calls_total",         "Downstream gRPC calls",    ["service", "status"])
processing_time   = Histogram("consumer_processing_seconds",     "Frame processing latency",
                               buckets=[.001, .005, .01, .025, .05, .1, .25, .5])
lag_gauge         = Gauge("consumer_kafka_lag_frames",           "Approximate consumer lag")

# ── OTel ──────────────────────────────────────────────────────────────────────
def setup_otel() -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("rover.consumer")


# ── gRPC stubs ────────────────────────────────────────────────────────────────
def _host_port(addr: str):
    host, port = addr.rsplit(":", 1)
    return host, int(port)


async def make_obstacle_stub() -> ObstacleDetectorStub:
    host, port = _host_port(OBSTACLE_ADDR)
    channel = grpclib.client.Channel(host=host, port=port)
    return ObstacleDetectorStub(channel)


async def make_motor_stub() -> MotorControllerStub:
    host, port = _host_port(MOTOR_ADDR)
    channel = grpclib.client.Channel(host=host, port=port)
    return MotorControllerStub(channel)


async def make_fdir_stub() -> FDIRServiceStub:
    host, port = _host_port(FDIR_ADDR)
    channel = grpclib.client.Channel(host=host, port=port)
    return FDIRServiceStub(channel)


# ── per-frame processing ──────────────────────────────────────────────────────
async def process_frame(
    frame:           SensorData,
    obstacle_stub:   ObstacleDetectorStub,
    motor_stub:      MotorControllerStub,
    fdir_stub:       FDIRServiceStub,
    tracer:          trace.Tracer,
    sem:             asyncio.Semaphore,
) -> None:
    async with sem:
        t0 = time.perf_counter()
        with tracer.start_as_current_span("consumer.process_frame") as span:
            span.set_attribute("rover.id",  frame.rover_id)
            span.set_attribute("rover.seq", frame.sequence_number)
            span.set_attribute("rover.status", str(frame.status))

            # ── obstacle detection ─────────────────────────────────────────
            try:
                sensor_input = SensorInput(
                    rover_id          = frame.rover_id,
                    timestamp_ms      = frame.timestamp_ms,
                    lidar_distances_m = frame.obstacle_distances_m,
                    velocity_ms       = frame.velocity_ms,
                    heading_degrees   = frame.heading_degrees,
                )
                analysis = await obstacle_stub.analyze(sensor_input)
                grpc_calls.labels(service="obstacle-detector", status="ok").inc()

                if analysis.emergency_stop:
                    log.warning("EMERGENCY STOP rover=%s action=%s",
                                frame.rover_id, analysis.recommended_action)
                    stop_cmd = DriveCommand(
                        rover_id       = frame.rover_id,
                        emergency_stop = True,
                        target_speed_ms = 0.0,
                        turn_rate       = 0.0,
                        duration_ms     = 0.0,
                    )
                    await motor_stub.emergency_stop(stop_cmd)
                    grpc_calls.labels(service="motor-controller", status="ok").inc()

            except Exception as exc:
                grpc_calls.labels(service="obstacle-detector", status="error").inc()
                log.debug("Obstacle RPC error rover=%s: %s", frame.rover_id, exc)

            # ── FDIR status check on fault frames ─────────────────────────
            if frame.status in (
                RoverStatus.ROVER_STATUS_CRITICAL,
                RoverStatus.ROVER_STATUS_FAULT,
            ):
                try:
                    req    = FDIRStatusRequest(rover_id=frame.rover_id)
                    status = await fdir_stub.get_status(req)
                    grpc_calls.labels(service="fdir", status="ok").inc()
                    span.set_attribute("fdir.severity", str(status.current_severity))
                except Exception as exc:
                    grpc_calls.labels(service="fdir", status="error").inc()
                    log.debug("FDIR RPC error rover=%s: %s", frame.rover_id, exc)

        frames_consumed.labels(rover_id=frame.rover_id).inc()
        processing_time.observe(time.perf_counter() - t0)


# ── main consumer loop ────────────────────────────────────────────────────────
async def main() -> None:
    start_http_server(METRICS_PORT)
    tracer = setup_otel()

    obstacle_stub = await make_obstacle_stub()
    motor_stub    = await make_motor_stub()
    fdir_stub     = await make_fdir_stub()
    sem           = asyncio.Semaphore(CONCURRENCY)

    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers  = KAFKA_BROKERS,
        group_id           = KAFKA_GROUP,
        auto_offset_reset  = "latest",
        enable_auto_commit = True,
        fetch_max_bytes    = 52_428_800,   # 50 MiB
        max_poll_records   = 500,
    )

    await consumer.start()
    log.info("Consumer started: topic=%s group=%s brokers=%s",
             KAFKA_TOPIC, KAFKA_GROUP, KAFKA_BROKERS)

    pending: set[asyncio.Task] = set()

    try:
        async for msg in consumer:
            try:
                frame = SensorData().parse(msg.value)
            except Exception:
                frames_errors.inc()
                continue

            task = asyncio.create_task(
                process_frame(frame, obstacle_stub, motor_stub, fdir_stub, tracer, sem)
            )
            pending.add(task)
            task.add_done_callback(pending.discard)

            # bound in-flight to avoid unbounded memory growth
            if len(pending) >= CONCURRENCY * 4:
                done, pending_set = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                pending = pending_set

    finally:
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(main())
